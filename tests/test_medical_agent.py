from app.graph.store import InMemoryGraphStore
from app.models.factory import LightweightEmbeddingProvider
from app.retrieval.evidence_store import InMemoryEvidenceStore
from app.retrieval.risk_ranker import MedicalRiskRanker
from app.services.diagnosis_formatter import DiagnosisFormatter
from app.services.evidence_query_planner import EvidenceQueryPlanner
from app.services.indicator_normalizer import IndicatorNormalizer
from app.services.input_parser import MedicalInputParser
from app.services.medical_agent import MedicalAssessmentAgent
from app.services.rules import IndicatorRuleEngine


def build_agent():
    normalizer = IndicatorNormalizer()
    return MedicalAssessmentAgent(
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


def test_medical_agent_returns_natural_language_answer():
    agent = build_agent()
    answer, structured = agent.assess("男，52岁，血压176/108 mmHg，肌酐128 umol/L，eGFR 54 mL/min/1.73m2，请判断健康状况并给出建议。")

    assert isinstance(answer, str) and answer.strip()
    assert "健康状态" in answer
    assert structured.primary_diagnosis.potential_risks


def test_medical_agent_stream_contains_steps_and_final_result():
    agent = build_agent()
    events = list(agent.stream_assess("男，52岁，血压176/108 mmHg，肌酐128 umol/L，eGFR 54 mL/min/1.73m2，请判断健康状况并给出建议。"))

    event_types = [event["type"] for event in events]
    assert "step" in event_types
    assert "content" in event_types
    assert "result" in event_types
    assert event_types[-1] == "done"


def test_medical_agent_stream_result_matches_assess_result():
    agent = build_agent()
    text = "男，52岁，血压176/108 mmHg，肌酐128 umol/L，eGFR 54 mL/min/1.73m2，请判断健康状况并给出建议。"

    _, structured = agent.assess(text)
    events = list(agent.stream_assess(text))
    stream_payload = next(event["payload"] for event in events if event["type"] == "result")

    assert stream_payload["primary_diagnosis"] == structured.primary_diagnosis.model_dump()
    assert stream_payload["secondary_recommendations"] == structured.secondary_recommendations.model_dump()


def test_medical_agent_chat_assess_falls_back_when_no_chat_model():
    agent = build_agent()
    answer = agent.chat_assess(
        "基于上面的结果，我明天早餐适合吃什么？",
        session_history=[
            {"role": "assistant", "content": "你存在高血压和慢性肾病风险。"},
        ],
    )

    assert isinstance(answer, str) and answer.strip()
