from __future__ import annotations

from abc import ABC, abstractmethod
import asyncio
from dataclasses import dataclass
import json
import logging

try:
    from pymilvus import DataType, MilvusClient
except ImportError:  # pragma: no cover - optional runtime dependency
    MilvusClient = None

from app.core.settings import Settings
from app.graph.seed_data import EVIDENCE_CHUNKS
from app.models.factory import BaseEmbeddingProvider, RemoteReranker
from app.retrieval.lexical import BM25Lite, SQLiteFTSIndex
from app.schemas.exam import EvidenceChunk, RetrievalQuery

logger = logging.getLogger(__name__)


@dataclass
class CandidateScore:
    chunk: EvidenceChunk
    dense_rank: int | None = None
    lexical_rank: int | None = None
    dense_score: float = 0.0
    lexical_score: float = 0.0
    graph_overlap_score: float = 0.0
    source_authority_score: float = 0.0
    fusion_score: float = 0.0
    rerank_score: float = 0.0
    final_score: float = 0.0
    matched_queries: list[str] | None = None


class BaseEvidenceStore(ABC):
    @property
    @abstractmethod
    def backend_name(self) -> str:
        raise NotImplementedError

    @property
    @abstractmethod
    def mode(self) -> str:
        raise NotImplementedError

    @abstractmethod
    def ping(self) -> bool:
        raise NotImplementedError

    @abstractmethod
    def ensure_schema(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def data_ready(self) -> bool:
        raise NotImplementedError

    @abstractmethod
    def search(
        self,
        queries: list[RetrievalQuery],
        node_codes: list[str],
        top_k: int = 5,
    ) -> list[EvidenceChunk]:
        raise NotImplementedError

    async def search_async(
        self,
        queries: list[RetrievalQuery],
        node_codes: list[str],
        top_k: int = 5,
    ) -> list[EvidenceChunk]:
        return await asyncio.to_thread(self.search, queries, node_codes, top_k)

    @abstractmethod
    def rebuild_index(self, chunks: list[EvidenceChunk]) -> None:
        raise NotImplementedError

    @abstractmethod
    def add_chunks(self, chunks: list[EvidenceChunk]) -> None:
        raise NotImplementedError


class InMemoryEvidenceStore(BaseEvidenceStore):
    def __init__(
        self,
        embedder: BaseEmbeddingProvider,
        reranker: RemoteReranker | None = None,
        mmr_candidate_limit: int = 15,
        fallback_reason: str = "",
        rerank_candidate_limit: int = 20,
    ):
        self._embedder = embedder
        self._reranker = reranker
        self._mmr_candidate_limit = mmr_candidate_limit
        self._rerank_candidate_limit = rerank_candidate_limit
        self.fallback_reason = fallback_reason
        self._chunks = list(EVIDENCE_CHUNKS)
        chunk_texts = [self._build_search_text(chunk) for chunk in self._chunks]
        embeddings = self._embedder.embed_batch(chunk_texts)
        self._chunk_embeddings = {
            chunk.chunk_id: embedding
            for chunk, embedding in zip(self._chunks, embeddings)
        }
        self._bm25 = BM25Lite([self._build_search_text(chunk) for chunk in self._chunks])

    @property
    def backend_name(self) -> str:
        return "in_memory"

    @property
    def mode(self) -> str:
        return "memory"

    def ping(self) -> bool:
        return True

    def ensure_schema(self) -> None:
        return None

    def data_ready(self) -> bool:
        return bool(self._chunks)

    def search(
        self,
        queries: list[RetrievalQuery],
        node_codes: list[str],
        top_k: int = 5,
    ) -> list[EvidenceChunk]:
        return self._hybrid_search(self._chunks, queries, node_codes, top_k)

    def rebuild_index(self, chunks: list[EvidenceChunk]) -> None:
        self._chunks = list(chunks)
        chunk_texts = [self._build_search_text(chunk) for chunk in self._chunks]
        embeddings = self._embedder.embed_batch(chunk_texts) if chunk_texts else []
        self._chunk_embeddings = {
            chunk.chunk_id: embedding
            for chunk, embedding in zip(self._chunks, embeddings)
        }
        self._bm25 = BM25Lite(chunk_texts)

    def add_chunks(self, chunks: list[EvidenceChunk]) -> None:
        existing = {chunk.chunk_id: chunk for chunk in self._chunks}
        for chunk in chunks:
            existing[chunk.chunk_id] = chunk
        self.rebuild_index(list(existing.values()))

    def _hybrid_search(
        self,
        chunks: list[EvidenceChunk],
        queries: list[RetrievalQuery],
        node_codes: list[str],
        top_k: int,
    ) -> list[EvidenceChunk]:
        if not queries:
            queries = [RetrievalQuery(label="fallback", text=" ".join(node_codes))]
        node_code_set = set(node_codes)
        score_map: dict[str, CandidateScore] = {
            chunk.chunk_id: CandidateScore(
                chunk=chunk,
                graph_overlap_score=self._graph_overlap_score(chunk, node_code_set),
                source_authority_score=self._source_authority_score(chunk),
                matched_queries=[],
            )
            for chunk in chunks
        }

        rrf_k = 60.0
        for query in queries:
            dense_ranks = self._dense_ranks(query.text, chunks)
            lexical_ranks = self._lexical_ranks(query.text, chunks)
            for rank, (chunk_id, dense_score) in enumerate(dense_ranks, 1):
                score = score_map[chunk_id]
                score.dense_rank = rank if score.dense_rank is None else min(score.dense_rank, rank)
                score.dense_score = max(score.dense_score, dense_score)
                score.fusion_score += 1.0 / (rrf_k + rank)
                score.matched_queries.append(query.label)
            for rank, (chunk_id, lexical_score) in enumerate(lexical_ranks, 1):
                score = score_map[chunk_id]
                score.lexical_rank = rank if score.lexical_rank is None else min(score.lexical_rank, rank)
                score.lexical_score = max(score.lexical_score, lexical_score)
                score.fusion_score += 1.0 / (rrf_k + rank)
                score.matched_queries.append(query.label)

        ordered = sorted(score_map.values(), key=self._rough_rank_value, reverse=True)
        ordered = self._apply_remote_rerank(ordered, queries, top_k)
        self._finalize_scores(score_map, queries)
        ordered = sorted(score_map.values(), key=lambda item: item.final_score, reverse=True)
        ordered = ordered[: max(top_k * 3, self._mmr_candidate_limit)]
        ordered = [item for item in ordered if self._passes_relevance_gate(item)]
        diversified = self._mmr_select(ordered, top_k=top_k, lambda_mult=0.75)
        return [self._materialize_candidate(item) for item in diversified]

    def _dense_ranks(self, query_text: str, chunks: list[EvidenceChunk]) -> list[tuple[str, float]]:
        query_vector = self._embedder.embed(query_text)
        scored = []
        for chunk in chunks:
            chunk_vector = self._chunk_embeddings[chunk.chunk_id]
            dense_score = sum(left * right for left, right in zip(query_vector, chunk_vector))
            scored.append((chunk.chunk_id, dense_score))
        scored.sort(key=lambda item: item[1], reverse=True)
        return scored

    def _lexical_ranks(self, query_text: str, chunks: list[EvidenceChunk]) -> list[tuple[str, float]]:
        scored = []
        for index, chunk in enumerate(chunks):
            scored.append((chunk.chunk_id, self._bm25.score(query_text, index)))
        scored.sort(key=lambda item: item[1], reverse=True)
        return scored

    def _finalize_scores(self, score_map: dict[str, CandidateScore], queries: list[RetrievalQuery]) -> None:
        max_fusion = max((score.fusion_score for score in score_map.values()), default=1.0)
        max_dense = max((score.dense_score for score in score_map.values()), default=1.0)
        max_lexical = max((score.lexical_score for score in score_map.values()), default=1.0)
        max_rerank = max((score.rerank_score for score in score_map.values()), default=0.0)
        has_rerank = max_rerank > 0
        for score in score_map.values():
            fusion_norm = score.fusion_score / max(max_fusion, 1e-6)
            dense_norm = score.dense_score / max(max_dense, 1e-6)
            lexical_norm = score.lexical_score / max(max_lexical, 1e-6)
            rerank_norm = score.rerank_score / max(max_rerank, 1e-6) if has_rerank else fusion_norm
            # 中文注释：有远程 rerank 时提高语义重排权重；无 rerank 时回退到粗排融合分，保证本地模式稳定。
            score.final_score = (
                0.45 * rerank_norm
                + 0.10 * dense_norm
                + 0.15 * lexical_norm
                + 0.20 * score.graph_overlap_score
                + 0.10 * score.source_authority_score
            )

    def _materialize_candidate(self, item: CandidateScore) -> EvidenceChunk:
        return item.chunk.model_copy(
            update={
                "dense_score": round(item.dense_score, 4),
                "lexical_score": round(item.lexical_score, 4),
                "graph_overlap_score": round(item.graph_overlap_score, 4),
                "source_authority_score": round(item.source_authority_score, 4),
                "fusion_score": round(item.fusion_score, 4),
                "final_score": round(item.final_score, 4),
                "rerank_score": round(item.rerank_score, 4),
                "relevance_score": round(item.final_score, 4),
                "matched_queries": sorted(set(item.matched_queries or [])),
            }
        )

    def _apply_remote_rerank(
        self,
        ordered: list[CandidateScore],
        queries: list[RetrievalQuery],
        top_k: int,
    ) -> list[CandidateScore]:
        if not self._reranker or not ordered:
            return ordered
        query_text = " ".join(query.text for query in queries[:3]).strip()
        candidate_limit = max(top_k * 3, min(self._rerank_candidate_limit, len(ordered)))
        documents = [item.chunk.text for item in ordered[:candidate_limit]]
        try:
            reranked = self._reranker.rerank(
                query_text=query_text,
                documents=documents,
                top_n=min(len(documents), max(top_k * 2, top_k)),
            )
        except Exception:
            logger.warning(
                "evidence_store.rerank_failed",
                exc_info=True,
                extra={"backend": self.backend_name, "candidate_count": len(documents)},
            )
            return ordered
        rerank_score_map = {item.index: item.score for item in reranked}
        rescored = []
        for index, item in enumerate(ordered[: len(documents)]):
            rerank_score = rerank_score_map.get(index)
            if rerank_score is not None:
                item.rerank_score = rerank_score
            rescored.append(item)
        rescored.extend(ordered[len(documents) :])
        rescored.sort(key=lambda item: item.final_score, reverse=True)
        return rescored

    @staticmethod
    def _rough_rank_value(item: CandidateScore) -> float:
        return item.fusion_score + item.dense_score + item.lexical_score + item.graph_overlap_score

    def _graph_overlap_score(self, chunk: EvidenceChunk, node_code_set: set[str]) -> float:
        if not node_code_set:
            return 0.0
        overlap = len(node_code_set.intersection(chunk.linked_node_codes))
        return overlap / max(len(set(chunk.linked_node_codes)), 1)

    @staticmethod
    def _source_authority_score(chunk: EvidenceChunk) -> float:
        source_weights = {
            "guideline": 1.0,
            "drug_label": 0.9,
            "protocol": 0.85,
            "reference": 0.75,
        }
        return source_weights.get(chunk.source_type, 0.7)

    @staticmethod
    def _build_search_text(chunk: EvidenceChunk) -> str:
        return " ".join([chunk.title, chunk.text, " ".join(chunk.linked_node_codes)])

    def _mmr_select(self, scores: list[CandidateScore], top_k: int, lambda_mult: float) -> list[CandidateScore]:
        if not scores:
            return []
        selected: list[CandidateScore] = []
        candidates = list(scores)
        while candidates and len(selected) < top_k:
            if not selected:
                selected.append(candidates.pop(0))
                continue
            best_index = 0
            best_value = float("-inf")
            for index, candidate in enumerate(candidates):
                relevance = candidate.final_score
                diversity_penalty = max(
                    self._cosine_similarity(candidate.chunk.chunk_id, selected_item.chunk.chunk_id)
                    for selected_item in selected
                )
                mmr_score = (lambda_mult * relevance) - ((1.0 - lambda_mult) * diversity_penalty)
                if mmr_score > best_value:
                    best_value = mmr_score
                    best_index = index
            selected.append(candidates.pop(best_index))
        return selected

    def _cosine_similarity(self, left_chunk_id: str, right_chunk_id: str) -> float:
        left = self._chunk_embeddings[left_chunk_id]
        right = self._chunk_embeddings[right_chunk_id]
        return sum(left_value * right_value for left_value, right_value in zip(left, right))

    @staticmethod
    def _passes_relevance_gate(score: CandidateScore) -> bool:
        if score.graph_overlap_score > 0:
            return True
        return score.final_score >= 0.75


class MilvusEvidenceStore(BaseEvidenceStore):
    def __init__(
        self,
        settings: Settings,
        embedder: BaseEmbeddingProvider,
        reranker: RemoteReranker | None = None,
    ):
        if MilvusClient is None:
            raise RuntimeError(
                "pymilvus is not installed. Install the 'pymilvus' package or enable USE_IN_MEMORY_EVIDENCE."
            )
        self._settings = settings
        self._client = MilvusClient(uri=settings.milvus_uri)
        self._collection = settings.milvus_collection
        self._embedder = embedder
        self._reranker = reranker
        self._mmr_candidate_limit = settings.mmr_candidate_limit
        self._rerank_candidate_limit = settings.rerank_candidate_limit
        self._vector_field = "embedding"
        self._chunk_cache: list[EvidenceChunk] = []
        self._chunk_embeddings: dict[str, list[float]] = {}
        self._bm25 = BM25Lite([])
        self._fts_index = (
            SQLiteFTSIndex(settings.lexical_index_path)
            if settings.lexical_index_backend == "sqlite_fts"
            else None
        )

    @property
    def backend_name(self) -> str:
        return "milvus"

    @property
    def mode(self) -> str:
        return "remote"

    def ping(self) -> bool:
        try:
            self._client.list_collections()
            return True
        except Exception:
            logger.warning("milvus_evidence_store.ping_failed", exc_info=True)
            return False

    def ensure_schema(self) -> None:
        self._ensure_collection()

    def data_ready(self) -> bool:
        try:
            if not self._client.has_collection(self._collection):
                return False
            stats = self._client.get_collection_stats(self._collection)
            row_count = stats.get("row_count") or stats.get("row_count".upper()) or stats.get("rows")
            return int(row_count or 0) > 0
        except Exception:
            logger.warning(
                "milvus_evidence_store.data_ready_failed",
                exc_info=True,
                extra={"collection": self._collection},
            )
            return False

    def _ensure_collection(self, drop_if_exists: bool = False) -> None:
        if drop_if_exists and self._client.has_collection(self._collection):
            self._client.drop_collection(self._collection)

        if self._client.has_collection(self._collection):
            return

        schema = self._client.create_schema(auto_id=False, enable_dynamic_field=True)
        schema.add_field("chunk_id", DataType.VARCHAR, max_length=128, is_primary=True)
        schema.add_field("doc_id", DataType.VARCHAR, max_length=128)
        schema.add_field("title", DataType.VARCHAR, max_length=256)
        schema.add_field("text", DataType.VARCHAR, max_length=4096)
        schema.add_field("source_type", DataType.VARCHAR, max_length=64)
        schema.add_field("risk_code", DataType.VARCHAR, max_length=128)
        schema.add_field("disease_code", DataType.VARCHAR, max_length=128)
        schema.add_field("node_codes_text", DataType.VARCHAR, max_length=512)
        schema.add_field("linked_node_codes_json", DataType.VARCHAR, max_length=2048)
        schema.add_field(self._vector_field, DataType.FLOAT_VECTOR, dim=self._settings.evidence_embedding_dim)

        index_params = self._client.prepare_index_params()
        index_params.add_index(
            field_name=self._vector_field,
            index_type="HNSW",
            metric_type="COSINE",
            params={"M": 16, "efConstruction": 128},
        )

        self._client.create_collection(
            collection_name=self._collection,
            schema=schema,
            index_params=index_params,
        )

    def search(
        self,
        queries: list[RetrievalQuery],
        node_codes: list[str],
        top_k: int = 5,
    ) -> list[EvidenceChunk]:
        if not node_codes and not queries:
            return []
        self._ensure_collection()
        candidate_map: dict[str, EvidenceChunk] = {}
        dense_ranks_by_query: list[list[tuple[str, float]]] = []
        effective_queries = queries or [RetrievalQuery(label="fallback", text=" ".join(node_codes))]
        for query in effective_queries:
            rows = self._client.search(
                collection_name=self._collection,
                data=[self._embedder.embed(query.text)],
                anns_field=self._vector_field,
                limit=max(top_k * 4, top_k),
                output_fields=[
                    "chunk_id",
                    "doc_id",
                    "title",
                    "text",
                    "source_type",
                    "risk_code",
                    "disease_code",
                    "node_codes_text",
                    "linked_node_codes_json",
                ],
                search_params={"metric_type": "COSINE", "params": {"ef": 128}},
            )
            query_dense_ranks: list[tuple[str, float]] = []
            for row in rows[0] if rows else []:
                entity = row.get("entity", row)
                chunk_id = entity.get("chunk_id", "")
                dense_score = float(row.get("distance", row.get("score", 0.0)) or 0.0)
                linked = self._parse_linked_codes(entity)
                if chunk_id not in candidate_map:
                    candidate_map[chunk_id] = EvidenceChunk(
                        chunk_id=chunk_id,
                        doc_id=entity.get("doc_id", ""),
                        title=entity.get("title", ""),
                        text=entity.get("text", ""),
                        linked_node_codes=linked,
                        source_type=entity.get("source_type", "guideline"),
                    )
                query_dense_ranks.append((chunk_id, dense_score))
            dense_ranks_by_query.append(query_dense_ranks)

        candidates = list(candidate_map.values())
        if not candidates:
            return []
        return self._hybrid_search(candidates, effective_queries, node_codes, top_k, dense_ranks_by_query)

    def rebuild_index(self, chunks: list[EvidenceChunk]) -> None:
        self._ensure_collection(drop_if_exists=self._settings.drop_milvus_collection_on_rebuild)
        self._chunk_cache = list(chunks)
        chunk_texts = [self._build_search_text(chunk) for chunk in self._chunk_cache]
        embeddings = self._embedder.embed_batch(chunk_texts) if chunk_texts else []
        self._chunk_embeddings = {
            chunk.chunk_id: embedding
            for chunk, embedding in zip(self._chunk_cache, embeddings)
        }
        self._bm25 = BM25Lite(chunk_texts)
        if self._fts_index is not None:
            self._fts_index.rebuild(self._chunk_cache)
        records = []
        for chunk, embedding in zip(chunks, embeddings):
            risk_code = ""
            disease_code = ""
            for code in chunk.linked_node_codes:
                if code.endswith("_risk"):
                    risk_code = code
                else:
                    disease_code = disease_code or code
            records.append(
                {
                    "chunk_id": chunk.chunk_id,
                    "doc_id": chunk.doc_id,
                    "title": chunk.title,
                    "text": chunk.text,
                    "source_type": chunk.source_type,
                    "risk_code": risk_code,
                    "disease_code": disease_code,
                    "node_codes_text": " ".join(chunk.linked_node_codes),
                    "linked_node_codes_json": json.dumps(chunk.linked_node_codes, ensure_ascii=False),
                    self._vector_field: embedding,
                }
            )
        if records:
            self._client.insert(collection_name=self._collection, data=records)
            self._client.load_collection(self._collection)

    def add_chunks(self, chunks: list[EvidenceChunk]) -> None:
        if not chunks:
            return
        existing = {chunk.chunk_id: chunk for chunk in self._chunk_cache}
        for chunk in chunks:
            existing[chunk.chunk_id] = chunk
        self.rebuild_index(list(existing.values()))

    def _hybrid_search(
        self,
        chunks: list[EvidenceChunk],
        queries: list[RetrievalQuery],
        node_codes: list[str],
        top_k: int,
        precomputed_dense_ranks: list[list[tuple[str, float]]] | None = None,
    ) -> list[EvidenceChunk]:
        node_code_set = set(node_codes)
        chunk_index_map = {chunk.chunk_id: index for index, chunk in enumerate(self._chunk_cache)}
        score_map: dict[str, CandidateScore] = {
            chunk.chunk_id: CandidateScore(
                chunk=chunk,
                graph_overlap_score=self._graph_overlap_score(chunk, node_code_set),
                source_authority_score=self._source_authority_score(chunk),
                matched_queries=[],
            )
            for chunk in chunks
        }

        rrf_k = 60.0
        for query_index, query in enumerate(queries):
            if precomputed_dense_ranks is not None and query_index < len(precomputed_dense_ranks):
                # 中文注释：Milvus 已经完成向量检索，这里直接复用返回的 distance，避免 Python 再做一次 embedding 点积。
                dense_ranks = precomputed_dense_ranks[query_index]
            else:
                dense_ranks = self._dense_ranks(query.text, chunks)
            lexical_ranks = self._lexical_ranks(query.text, chunks, chunk_index_map)
            for rank, (chunk_id, dense_score) in enumerate(dense_ranks, 1):
                if chunk_id not in score_map:
                    continue
                score = score_map[chunk_id]
                score.dense_rank = rank if score.dense_rank is None else min(score.dense_rank, rank)
                score.dense_score = max(score.dense_score, dense_score)
                score.fusion_score += 1.0 / (rrf_k + rank)
                score.matched_queries.append(query.label)
            for rank, (chunk_id, lexical_score) in enumerate(lexical_ranks, 1):
                score = score_map[chunk_id]
                score.lexical_rank = rank if score.lexical_rank is None else min(score.lexical_rank, rank)
                score.lexical_score = max(score.lexical_score, lexical_score)
                score.fusion_score += 1.0 / (rrf_k + rank)
                score.matched_queries.append(query.label)

        ordered = sorted(score_map.values(), key=self._rough_rank_value, reverse=True)
        ordered = self._apply_remote_rerank(ordered, queries, top_k)
        self._finalize_scores(score_map, queries)
        ordered = sorted(score_map.values(), key=lambda item: item.final_score, reverse=True)
        ordered = ordered[: max(top_k * 3, self._mmr_candidate_limit)]
        ordered = [item for item in ordered if self._passes_relevance_gate(item)]
        diversified = self._mmr_select(ordered, top_k=top_k, lambda_mult=0.75)
        return [self._materialize_candidate(item) for item in diversified]

    def _dense_ranks(self, query_text: str, chunks: list[EvidenceChunk]) -> list[tuple[str, float]]:
        query_vector = self._embedder.embed(query_text)
        scored = []
        for chunk in chunks:
            chunk_vector = self._get_chunk_embedding(chunk)
            dense_score = sum(left * right for left, right in zip(query_vector, chunk_vector))
            scored.append((chunk.chunk_id, dense_score))
        scored.sort(key=lambda item: item[1], reverse=True)
        return scored

    def _lexical_ranks(
        self,
        query_text: str,
        chunks: list[EvidenceChunk],
        chunk_index_map: dict[str, int],
    ) -> list[tuple[str, float]]:
        scored = []
        if self._fts_index is not None:
            chunk_ids = {chunk.chunk_id for chunk in chunks}
            for chunk_id, lexical in self._fts_index.search(query_text, top_k=max(len(chunks), 20)):
                if chunk_id in chunk_ids:
                    scored.append((chunk_id, lexical))
            if scored:
                return scored
        for chunk in chunks:
            index = chunk_index_map.get(chunk.chunk_id)
            lexical = self._bm25.score(query_text, index) if index is not None else 0.0
            scored.append((chunk.chunk_id, lexical))
        scored.sort(key=lambda item: item[1], reverse=True)
        return scored

    def _finalize_scores(self, score_map: dict[str, CandidateScore], queries: list[RetrievalQuery]) -> None:
        max_fusion = max((score.fusion_score for score in score_map.values()), default=1.0)
        max_dense = max((score.dense_score for score in score_map.values()), default=1.0)
        max_lexical = max((score.lexical_score for score in score_map.values()), default=1.0)
        max_rerank = max((score.rerank_score for score in score_map.values()), default=0.0)
        has_rerank = max_rerank > 0
        for score in score_map.values():
            fusion_norm = score.fusion_score / max(max_fusion, 1e-6)
            dense_norm = score.dense_score / max(max_dense, 1e-6)
            lexical_norm = score.lexical_score / max(max_lexical, 1e-6)
            rerank_norm = score.rerank_score / max(max_rerank, 1e-6) if has_rerank else fusion_norm
            # 中文注释：rerank 前置后，最终融合阶段再叠加图谱、词汇和来源权威度。
            score.final_score = (
                0.45 * rerank_norm
                + 0.10 * dense_norm
                + 0.15 * lexical_norm
                + 0.20 * score.graph_overlap_score
                + 0.10 * score.source_authority_score
            )

    def _materialize_candidate(self, item: CandidateScore) -> EvidenceChunk:
        return item.chunk.model_copy(
            update={
                "dense_score": round(item.dense_score, 4),
                "lexical_score": round(item.lexical_score, 4),
                "graph_overlap_score": round(item.graph_overlap_score, 4),
                "source_authority_score": round(item.source_authority_score, 4),
                "fusion_score": round(item.fusion_score, 4),
                "final_score": round(item.final_score, 4),
                "rerank_score": round(item.rerank_score, 4),
                "relevance_score": round(item.final_score, 4),
                "matched_queries": sorted(set(item.matched_queries or [])),
            }
        )

    def _apply_remote_rerank(
        self,
        ordered: list[CandidateScore],
        queries: list[RetrievalQuery],
        top_k: int,
    ) -> list[CandidateScore]:
        if not self._reranker or not ordered:
            return ordered
        query_text = " ".join(query.text for query in queries[:3]).strip()
        candidate_limit = max(top_k * 3, min(self._rerank_candidate_limit, len(ordered)))
        documents = [item.chunk.text for item in ordered[:candidate_limit]]
        try:
            reranked = self._reranker.rerank(
                query_text=query_text,
                documents=documents,
                top_n=min(len(documents), max(top_k * 2, top_k)),
            )
        except Exception:
            logger.warning(
                "evidence_store.rerank_failed",
                exc_info=True,
                extra={"backend": self.backend_name, "candidate_count": len(documents)},
            )
            return ordered
        rerank_score_map = {item.index: item.score for item in reranked}
        rescored = []
        for index, item in enumerate(ordered[: len(documents)]):
            rerank_score = rerank_score_map.get(index)
            if rerank_score is not None:
                item.rerank_score = rerank_score
            rescored.append(item)
        rescored.extend(ordered[len(documents) :])
        rescored.sort(key=lambda item: item.final_score, reverse=True)
        return rescored

    @staticmethod
    def _rough_rank_value(item: CandidateScore) -> float:
        return item.fusion_score + item.dense_score + item.lexical_score + item.graph_overlap_score

    def _graph_overlap_score(self, chunk: EvidenceChunk, node_code_set: set[str]) -> float:
        if not node_code_set:
            return 0.0
        overlap = len(node_code_set.intersection(chunk.linked_node_codes))
        return overlap / max(len(set(chunk.linked_node_codes)), 1)

    @staticmethod
    def _source_authority_score(chunk: EvidenceChunk) -> float:
        source_weights = {
            "guideline": 1.0,
            "drug_label": 0.9,
            "protocol": 0.85,
            "reference": 0.75,
        }
        return source_weights.get(chunk.source_type, 0.7)

    @staticmethod
    def _build_search_text(chunk: EvidenceChunk) -> str:
        return " ".join([chunk.title, chunk.text, " ".join(chunk.linked_node_codes)])

    def _mmr_select(self, scores: list[CandidateScore], top_k: int, lambda_mult: float) -> list[CandidateScore]:
        if not scores:
            return []
        selected: list[CandidateScore] = []
        candidates = list(scores)
        while candidates and len(selected) < top_k:
            if not selected:
                selected.append(candidates.pop(0))
                continue
            best_index = 0
            best_value = float("-inf")
            for index, candidate in enumerate(candidates):
                relevance = candidate.final_score
                diversity_penalty = max(
                    self._cosine_similarity(candidate.chunk.chunk_id, selected_item.chunk.chunk_id)
                    for selected_item in selected
                )
                mmr_score = (lambda_mult * relevance) - ((1.0 - lambda_mult) * diversity_penalty)
                if mmr_score > best_value:
                    best_value = mmr_score
                    best_index = index
            selected.append(candidates.pop(best_index))
        return selected

    def _cosine_similarity(self, left_chunk_id: str, right_chunk_id: str) -> float:
        left = self._chunk_embeddings.get(left_chunk_id)
        right = self._chunk_embeddings.get(right_chunk_id)
        if left is None or right is None:
            left_chunk = next((chunk for chunk in self._chunk_cache if chunk.chunk_id == left_chunk_id), None)
            right_chunk = next((chunk for chunk in self._chunk_cache if chunk.chunk_id == right_chunk_id), None)
            if left is None and left_chunk is not None:
                left = self._get_chunk_embedding(left_chunk)
            if right is None and right_chunk is not None:
                right = self._get_chunk_embedding(right_chunk)
        if left is None or right is None:
            return 0.0
        return sum(left_value * right_value for left_value, right_value in zip(left, right))

    def _get_chunk_embedding(self, chunk: EvidenceChunk) -> list[float]:
        vector = self._chunk_embeddings.get(chunk.chunk_id)
        if vector is None:
            vector = self._embedder.embed(self._build_search_text(chunk))
            self._chunk_embeddings[chunk.chunk_id] = vector
        return vector

    @staticmethod
    def _parse_linked_codes(entity: dict) -> list[str]:
        raw_json = entity.get("linked_node_codes_json")
        if raw_json:
            try:
                loaded = json.loads(raw_json)
                if isinstance(loaded, list):
                    return [str(item) for item in loaded if item]
            except json.JSONDecodeError:
                logger.debug("milvus_evidence_store.linked_codes_json_invalid", exc_info=True)
        raw_text = entity.get("node_codes_text", "")
        if raw_text:
            return [item for item in raw_text.split() if item]
        return [item for item in (entity.get("risk_code"), entity.get("disease_code")) if item]

    @staticmethod
    def _passes_relevance_gate(score: CandidateScore) -> bool:
        if score.graph_overlap_score > 0:
            return True
        return score.final_score >= 0.75


def build_evidence_store(
    settings: Settings,
    embedder: BaseEmbeddingProvider,
    reranker: RemoteReranker | None = None,
) -> BaseEvidenceStore:
    if settings.use_in_memory_evidence:
        return InMemoryEvidenceStore(
            embedder=embedder,
            reranker=reranker,
            mmr_candidate_limit=settings.mmr_candidate_limit,
            rerank_candidate_limit=settings.rerank_candidate_limit,
        )
    try:
        store = MilvusEvidenceStore(settings, embedder=embedder, reranker=reranker)
        if store.ping():
            return store
        logger.warning("milvus_evidence_store.ping_failed_fallback_to_memory")
    except Exception:
        logger.warning("milvus_evidence_store.init_failed_fallback_to_memory", exc_info=True)
    return InMemoryEvidenceStore(
        embedder=embedder,
        reranker=reranker,
        mmr_candidate_limit=settings.mmr_candidate_limit,
        rerank_candidate_limit=settings.rerank_candidate_limit,
        fallback_reason="milvus unavailable",
    )
