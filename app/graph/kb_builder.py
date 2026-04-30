from __future__ import annotations

from dataclasses import dataclass

from app.graph.seed_data import DISEASE_TO_INTERVENTIONS, EVIDENCE_CHUNKS, STATE_TO_RISK
from app.graph.store import BaseGraphStore
from app.retrieval.evidence_store import BaseEvidenceStore
from app.services.knowledge_registry import KnowledgeDocumentRegistry


@dataclass
class KnowledgeBuildResult:
    graph_backend: str
    evidence_backend: str
    states_seeded: int
    risks_seeded: int
    interventions_seeded: int
    evidence_seeded: int


class MedicalKnowledgeBuilder:
    def __init__(
        self,
        graph_store: BaseGraphStore,
        evidence_store: BaseEvidenceStore,
        registry: KnowledgeDocumentRegistry | None = None,
    ):
        self._graph_store = graph_store
        self._evidence_store = evidence_store
        self._registry = registry

    def _merged_evidence_chunks(self):
        chunks = list(EVIDENCE_CHUNKS)
        if self._registry is not None:
            chunks.extend(self._registry.list_chunks())
        deduped: dict[str, object] = {}
        for chunk in chunks:
            deduped[chunk.chunk_id] = chunk
        return list(deduped.values())

    def build_from_seed(self) -> KnowledgeBuildResult:
        evidence_chunks = self._merged_evidence_chunks()
        self._graph_store.rebuild_from_seed(
            state_to_risk=STATE_TO_RISK,
            disease_to_interventions=DISEASE_TO_INTERVENTIONS,
            evidence_chunks=evidence_chunks,
        )
        self._evidence_store.rebuild_index(evidence_chunks)
        return KnowledgeBuildResult(
            graph_backend=self._graph_store.backend_name,
            evidence_backend=self._evidence_store.backend_name,
            states_seeded=len(STATE_TO_RISK),
            risks_seeded=sum(len(items) for items in STATE_TO_RISK.values()),
            interventions_seeded=sum(len(items["interventions"]) for items in DISEASE_TO_INTERVENTIONS.values()),
            evidence_seeded=len(evidence_chunks),
        )
