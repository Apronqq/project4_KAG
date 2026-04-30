import asyncio
from types import SimpleNamespace

from app.api.routes import medical as medical_routes
from app.graph.store import InMemoryGraphStore
from app.models.factory import LightweightEmbeddingProvider, RerankResult
from app.retrieval.evidence_store import InMemoryEvidenceStore
from app.retrieval.risk_ranker import MedicalRiskRanker
from app.schemas.exam import EvidenceChunk, RetrievalQuery
from app.services.diagnosis_formatter import DiagnosisFormatter
from app.services.evidence_query_planner import EvidenceQueryPlanner
from app.services.indicator_normalizer import IndicatorNormalizer
from app.services.input_parser import MedicalInputParser
from app.services.medical_agent import MedicalAssessmentAgent
from app.services.rules import IndicatorRuleEngine
from app.workflows.medical_kag_pipeline import MedicalKAGWorkflow


def build_workflow():
    normalizer = IndicatorNormalizer()
    return MedicalKAGWorkflow(
        parser=MedicalInputParser(normalizer),
        normalizer=normalizer,
        rule_engine=IndicatorRuleEngine(),
        graph_store=InMemoryGraphStore(),
        evidence_store=InMemoryEvidenceStore(embedder=LightweightEmbeddingProvider(256)),
        ranker=MedicalRiskRanker(),
        formatter=DiagnosisFormatter(),
        query_planner=EvidenceQueryPlanner(),
    )


def test_async_pipeline_matches_sync_pipeline():
    workflow = build_workflow()
    text = "男，52岁，血压176/108 mmHg，肌酐128 umol/L，eGFR 54，请判断健康状况。"

    sync_response = workflow.run(text)
    async_response = asyncio.run(workflow.run_async(text))

    assert async_response.primary_diagnosis == sync_response.primary_diagnosis
    assert async_response.secondary_recommendations == sync_response.secondary_recommendations


class PreferenceReranker:
    backend_name = "fake_reranker"

    def rerank(self, query_text: str, documents: list[str], top_n: int):
        # 中文注释：模拟远程 rerank 强烈偏好第二个候选，验证 rerank 前置确实影响最终排序。
        return [RerankResult(index=1, score=1.0), RerankResult(index=0, score=0.1)]


def test_rerank_score_has_primary_influence_before_fusion():
    store = InMemoryEvidenceStore(
        embedder=LightweightEmbeddingProvider(64),
        reranker=PreferenceReranker(),
        mmr_candidate_limit=5,
        rerank_candidate_limit=5,
    )
    store.rebuild_index(
        [
            EvidenceChunk(
                chunk_id="c1",
                doc_id="d1",
                title="候选一",
                text="高血压 风险 复查",
                linked_node_codes=["hypertension_risk"],
            ),
            EvidenceChunk(
                chunk_id="c2",
                doc_id="d1",
                title="候选二",
                text="高血压 风险 复查",
                linked_node_codes=["hypertension_risk"],
            ),
        ]
    )

    chunks = store.search(
        queries=[RetrievalQuery(label="q", text="高血压 风险")],
        node_codes=["hypertension_risk"],
        top_k=2,
    )

    assert chunks[0].chunk_id == "c2"
    assert chunks[0].rerank_score == 1.0


def test_runtime_status_includes_component_latency(monkeypatch):
    runtime = SimpleNamespace(
        graph_store=SimpleNamespace(
            backend_name="in_memory",
            mode="memory",
            fallback_reason="",
            ping=lambda: True,
            data_ready=lambda: True,
        ),
        evidence_store=SimpleNamespace(
            backend_name="in_memory",
            mode="memory",
            fallback_reason="",
            ping=lambda: True,
            data_ready=lambda: True,
        ),
        database_manager=SimpleNamespace(ping=lambda: True),
        model_runtime=SimpleNamespace(
            embedding_provider=SimpleNamespace(backend_name="lightweight"),
            extractor=None,
            reranker=None,
        ),
    )
    monkeypatch.setattr(medical_routes, "get_runtime", lambda: runtime)

    response = asyncio.run(medical_routes.get_runtime_status())

    assert response.components["graph"]["status"] == "healthy"
    assert "latency_ms" in response.components["postgresql"]
    assert response.components["reranker"]["status"] == "disabled"


def test_followup_stream_includes_synthesizing_event():
    normalizer = IndicatorNormalizer()
    agent = MedicalAssessmentAgent(
        parser=MedicalInputParser(normalizer),
        normalizer=normalizer,
        rule_engine=IndicatorRuleEngine(),
        graph_store=InMemoryGraphStore(),
        evidence_store=InMemoryEvidenceStore(embedder=LightweightEmbeddingProvider(256)),
        ranker=MedicalRiskRanker(),
        formatter=DiagnosisFormatter(),
        query_planner=EvidenceQueryPlanner(),
        chat_model=None,
    )

    events = list(
        agent.stream_assess(
            "我的血压风险严不严重？",
            session_history=[{"role": "system", "content": "结构化诊断记忆：主要风险=高血压风险"}],
        )
    )

    assert any(event["type"] == "agent_synthesizing" for event in events)
    assert events[-1]["type"] == "done"
