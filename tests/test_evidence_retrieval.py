from app.graph.store import InMemoryGraphStore
from app.retrieval.embeddings import LightweightTextEmbedder
from app.retrieval.evidence_store import InMemoryEvidenceStore
from app.schemas.exam import RetrievalQuery
from app.services.evidence_query_planner import EvidenceQueryPlanner
from app.services.indicator_normalizer import IndicatorNormalizer
from app.services.input_parser import MedicalInputParser
from app.services.rules import IndicatorRuleEngine


def test_query_planner_builds_multiple_graph_aware_queries():
    parser = MedicalInputParser(IndicatorNormalizer())
    exam = parser.parse("男，52岁，血压 176/108 mmHg，肌酐 128 umol/L，eGFR 54 mL/min/1.73m2，请判断健康风险。").normalized_exam_json
    states = IndicatorRuleEngine().detect_states(exam)
    risks = InMemoryGraphStore().get_risk_candidates([item.state_code for item in states])

    queries = EvidenceQueryPlanner().build_queries(exam, states, risks)

    labels = {query.label for query in queries}
    assert "user_question" in labels
    assert "risk_guideline" in labels
    assert "abnormal_indicator" in labels
    assert len(queries) >= 4


def test_hybrid_retrieval_filters_irrelevant_low_overlap_chunks():
    store = InMemoryEvidenceStore(embedder=LightweightTextEmbedder(256))
    queries = [
        RetrievalQuery(label="risk_guideline", text="高血压 高血压风险 成人 体检 指南 风险识别 干预建议 复查"),
        RetrievalQuery(label="abnormal_indicator", text="收缩压 收缩压显著升高 风险 分层 指南"),
    ]

    chunks = store.search(queries=queries, node_codes=["hypertension_risk", "hypertension"], top_k=5)

    assert chunks
    assert all(chunk.graph_overlap_score > 0 or chunk.final_score >= 0.75 for chunk in chunks)
    assert all(
        "hypertension" in " ".join(chunk.linked_node_codes) or "hypertension_risk" in " ".join(chunk.linked_node_codes)
        for chunk in chunks
    )
