from pathlib import Path

from app.models.factory import LightweightEmbeddingProvider
from app.retrieval import evidence_store as evidence_module
from app.retrieval.evidence_store import InMemoryEvidenceStore, MilvusEvidenceStore
from app.retrieval.lexical import SQLiteFTSIndex
from app.schemas.exam import EvidenceChunk, RetrievalQuery


def test_sqlite_fts_index_rebuild_and_search(tmp_path):
    index = SQLiteFTSIndex(tmp_path / "fts.sqlite3")
    chunks = [
        EvidenceChunk(
            chunk_id="c1",
            doc_id="d1",
            title="高血压复查",
            text="高血压患者建议家庭血压监测和限盐。",
            linked_node_codes=["hypertension_risk"],
        ),
        EvidenceChunk(
            chunk_id="c2",
            doc_id="d1",
            title="血脂管理",
            text="低密度脂蛋白升高需要管理饮食。",
            linked_node_codes=["dyslipidemia_risk"],
        ),
    ]

    index.rebuild(chunks)
    rows = index.search("高血压 限盐", top_k=5)

    assert rows
    assert rows[0][0] == "c1"


class CapturingMMRStore(InMemoryEvidenceStore):
    def __init__(self):
        super().__init__(embedder=LightweightEmbeddingProvider(64), mmr_candidate_limit=4)
        self.mmr_input_count = 0

    def _mmr_select(self, scores, top_k, lambda_mult):
        self.mmr_input_count = len(scores)
        return super()._mmr_select(scores, top_k, lambda_mult)


def test_mmr_candidate_input_is_limited():
    store = CapturingMMRStore()
    chunks = [
        EvidenceChunk(
            chunk_id=f"c{i}",
            doc_id="d",
            title=f"高血压证据{i}",
            text="高血压 风险 复查 干预 限盐 运动",
            linked_node_codes=["hypertension_risk"],
            source_type="guideline",
        )
        for i in range(20)
    ]
    store.rebuild_index(chunks)

    store.search(
        queries=[RetrievalQuery(label="q", text="高血压 风险 复查 干预")],
        node_codes=["hypertension_risk"],
        top_k=2,
    )

    assert store.mmr_input_count <= 6


def test_milvus_search_reuses_native_distance(monkeypatch, tmp_path):
    class CountingEmbedder:
        backend_name = "counting"

        def __init__(self):
            self.embed_calls = 0

        def embed(self, text: str):
            self.embed_calls += 1
            return [1.0, 0.0, 0.0, 0.0]

        def embed_batch(self, texts):
            return [self.embed(text) for text in texts]

    class FakeMilvusClient:
        def __init__(self, uri):
            self.uri = uri

        def list_collections(self):
            return ["medical_evidence_chunks"]

        def has_collection(self, collection_name):
            return True

        def search(self, **kwargs):
            return [[
                {
                    "distance": 0.91,
                    "entity": {
                        "chunk_id": "c1",
                        "doc_id": "d1",
                        "title": "高血压证据",
                        "text": "高血压 风险 复查",
                        "source_type": "guideline",
                        "node_codes_text": "hypertension_risk",
                        "linked_node_codes_json": "[\"hypertension_risk\"]",
                    },
                }
            ]]

    settings = type(
        "Settings",
        (),
        {
            "milvus_uri": "fake",
            "milvus_collection": "medical_evidence_chunks",
            "evidence_embedding_dim": 4,
            "drop_milvus_collection_on_rebuild": False,
            "mmr_candidate_limit": 15,
            "rerank_candidate_limit": 20,
            "lexical_index_backend": "sqlite_fts",
            "lexical_index_path": tmp_path / "fts.sqlite3",
        },
    )()
    monkeypatch.setattr(evidence_module, "MilvusClient", FakeMilvusClient)
    embedder = CountingEmbedder()
    store = MilvusEvidenceStore(settings, embedder=embedder)

    chunks = store.search(
        queries=[RetrievalQuery(label="q", text="高血压 风险")],
        node_codes=["hypertension_risk"],
        top_k=1,
    )

    assert chunks
    assert chunks[0].dense_score == 0.91
    assert embedder.embed_calls == 1
