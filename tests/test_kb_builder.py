from app.graph.kb_builder import MedicalKnowledgeBuilder
from app.graph.store import InMemoryGraphStore
from app.retrieval.embeddings import LightweightTextEmbedder
from app.retrieval.evidence_store import InMemoryEvidenceStore


def test_kb_builder_reports_seed_counts():
    builder = MedicalKnowledgeBuilder(InMemoryGraphStore(), InMemoryEvidenceStore(embedder=LightweightTextEmbedder(256)))
    result = builder.build_from_seed()

    assert result.graph_backend == "in_memory"
    assert result.evidence_backend == "in_memory"
    assert result.states_seeded > 0
    assert result.risks_seeded > 0
    assert result.interventions_seeded > 0
    assert result.evidence_seeded > 0


def test_in_memory_graph_store_becomes_stateful_after_rebuild():
    graph_store = InMemoryGraphStore()
    graph_store.rebuild_from_seed(
        state_to_risk={
            "CUSTOM_STATE": [
                {
                    "risk_code": "custom_risk",
                    "risk_name": "自定义风险",
                    "disease_code": "custom_disease",
                    "disease_name": "自定义疾病",
                    "risk_level": "medium",
                    "graph_score": 0.66,
                }
            ]
        },
        disease_to_interventions={
            "custom_disease": {
                "interventions": ["自定义干预"],
                "medication_directions": ["自定义用药方向"],
                "contraindications": [],
                "follow_up_tests": ["自定义复查"],
                "departments": ["自定义科室"],
            }
        },
        evidence_chunks=[],
    )

    risks = graph_store.get_risk_candidates(["CUSTOM_STATE"])
    interventions = graph_store.get_intervention_candidates(["custom_disease"])

    assert risks and risks[0].risk_code == "custom_risk"
    assert interventions and interventions[0].interventions == ["自定义干预"]
