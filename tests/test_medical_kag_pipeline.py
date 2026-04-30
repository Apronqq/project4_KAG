from app.graph.store import InMemoryGraphStore
from app.retrieval.embeddings import LightweightTextEmbedder
from app.retrieval.evidence_store import InMemoryEvidenceStore
from app.retrieval.risk_ranker import MedicalRiskRanker
from app.services.diagnosis_formatter import DiagnosisFormatter
from app.services.evidence_query_planner import EvidenceQueryPlanner
from app.services.indicator_normalizer import IndicatorNormalizer
from app.services.input_parser import MedicalInputParser
from app.services.rules import IndicatorRuleEngine
from app.workflows.medical_kag_pipeline import MedicalKAGWorkflow


def build_workflow():
    normalizer = IndicatorNormalizer()
    return MedicalKAGWorkflow(
        parser=MedicalInputParser(normalizer),
        normalizer=normalizer,
        rule_engine=IndicatorRuleEngine(),
        graph_store=InMemoryGraphStore(),
        evidence_store=InMemoryEvidenceStore(embedder=LightweightTextEmbedder(256)),
        ranker=MedicalRiskRanker(),
        formatter=DiagnosisFormatter(),
        query_planner=EvidenceQueryPlanner(),
    )


class SpyGraphStore(InMemoryGraphStore):
    def __init__(self):
        super().__init__()
        self.risk_calls = 0

    def get_risk_candidates(self, state_codes):
        self.risk_calls += 1
        return super().get_risk_candidates(state_codes)


class SpyEvidenceStore(InMemoryEvidenceStore):
    def __init__(self):
        super().__init__(embedder=LightweightTextEmbedder(256))
        self.search_calls = 0

    def search(self, queries, node_codes, top_k=5):
        self.search_calls += 1
        return super().search(queries, node_codes, top_k)


def test_workflow_detects_hypertension_risk_from_text():
    workflow = build_workflow()
    text = "男，52岁，血压 176/108 mmHg，肌酐 128 umol/L，eGFR 54 mL/min/1.73m2，请判断健康风险。"

    response = workflow.run(text)

    assert response.primary_diagnosis.health_status == "high_risk"
    assert response.primary_diagnosis.potential_risks
    assert response.secondary_recommendations.recommended_departments
    assert response.evidence.chunks
    assert all(chunk.graph_overlap_score > 0 or chunk.final_score >= 0.75 for chunk in response.evidence.chunks)


def test_workflow_detects_multiple_risks_from_structured_json():
    workflow = build_workflow()
    payload = {
        "patient_profile": {"sex": "male", "age": 52},
        "exam_items": [
            {"name": "收缩压", "value": 176, "unit": "mmHg"},
            {"name": "舒张压", "value": 108, "unit": "mmHg"},
            {"name": "eGFR", "value": 54, "unit": "mL/min/1.73m2"},
        ],
        "user_question": "请判断健康风险",
    }

    response = workflow.run(payload)

    risk_codes = {risk.risk_code for risk in response.primary_diagnosis.potential_risks}
    assert "hypertension_risk" in risk_codes
    assert "ckd_risk" in risk_codes
    assert len(response.secondary_recommendations.recommendations_by_disease) >= 2
    assert any(risk.support_count >= 2 for risk in response.primary_diagnosis.potential_risks)


def test_workflow_run_state_preserves_intermediate_context():
    workflow = build_workflow()
    state = workflow.run_state("男，52岁，血压 176/108 mmHg，肌酐 128 umol/L，eGFR 54，请判断健康风险。")

    assert state.response is not None
    assert state.detected_states
    assert state.retrieval_queries
    assert state.evidence_chunks
    assert state.primary_diagnosis == state.response.primary_diagnosis


def test_workflow_skips_graph_and_evidence_when_no_abnormal_states():
    normalizer = IndicatorNormalizer()
    graph_store = SpyGraphStore()
    evidence_store = SpyEvidenceStore()
    workflow = MedicalKAGWorkflow(
        parser=MedicalInputParser(normalizer),
        normalizer=normalizer,
        rule_engine=IndicatorRuleEngine(),
        graph_store=graph_store,
        evidence_store=evidence_store,
        ranker=MedicalRiskRanker(),
        formatter=DiagnosisFormatter(),
        query_planner=EvidenceQueryPlanner(),
    )
    payload = {
        "patient_profile": {"sex": "male", "age": 35},
        "exam_items": [
            {"name": "空腹血糖", "value": 5.0, "unit": "mmol/L"},
            {"name": "收缩压", "value": 118, "unit": "mmHg"},
        ],
        "user_question": "请判断健康风险",
    }

    response = workflow.run(payload)

    assert response.primary_diagnosis.health_status == "healthy"
    assert graph_store.risk_calls == 0
    assert evidence_store.search_calls == 0
