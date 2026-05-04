from types import SimpleNamespace

from app.agents.medical_multi_agent import build_medical_multi_agent_supervisor
from app.graph.store import InMemoryGraphStore
from app.models.factory import LightweightEmbeddingProvider
from app.retrieval.evidence_store import InMemoryEvidenceStore
from app.retrieval.risk_ranker import MedicalRiskRanker
from app.schemas.exam import MedicalAssessmentResponse
from app.services.agent_tools import MedicalKnowledgeRetrievalTool
from app.services.diagnosis_formatter import DiagnosisFormatter
from app.services.evidence_query_planner import EvidenceQueryPlanner
from app.services.indicator_normalizer import IndicatorNormalizer
from app.services.input_parser import MedicalInputParser
from app.services.rules import IndicatorRuleEngine
from app.workflows.medical_kag_pipeline import MedicalKAGWorkflow


class CapturingTool:
    def __init__(self):
        self.queries = []

    def invoke(self, payload):
        self.queries.append(payload["query"])
        return "高血压饮食建议证据"


def build_supervisor(tool=None, followup_answer_builder=None, memory_context_builder=None):
    normalizer = IndicatorNormalizer()
    workflow = MedicalKAGWorkflow(
        parser=MedicalInputParser(normalizer),
        normalizer=normalizer,
        rule_engine=IndicatorRuleEngine(),
        graph_store=InMemoryGraphStore(),
        evidence_store=InMemoryEvidenceStore(embedder=LightweightEmbeddingProvider(256)),
        ranker=MedicalRiskRanker(),
        formatter=DiagnosisFormatter(),
        query_planner=EvidenceQueryPlanner(),
    )
    tool = tool or MedicalKnowledgeRetrievalTool(workflow._evidence_store, top_k=2)
    return build_medical_multi_agent_supervisor(
        workflow=workflow,
        knowledge_tool=tool,
        chat_model=None,
        assessment_answer_builder=lambda response: f"健康状态：{response.primary_diagnosis.health_status}",
        followup_answer_builder=followup_answer_builder
        or (lambda user_input, history, evidence: f"回答：{user_input}\n证据：{evidence or '无'}"),
        initial_assessment_detector=lambda text: "血压" in text and any(char.isdigit() for char in text),
        memory_context_builder=memory_context_builder,
    )


def test_multi_agent_routes_initial_assessment_to_assessment_agent():
    supervisor = build_supervisor()

    result = supervisor.run("男，52岁，血压176/108 mmHg，请判断健康状况。", [])

    assert isinstance(result.structured_response, MedicalAssessmentResponse)
    assert any(event.get("agent") == "triage_agent" and event.get("action") == "route_to_assessment" for event in result.events)
    assert any(event.get("agent") == "assessment_agent" for event in result.events)
    assert result.events[-1]["type"] == "final_answer"


def test_multi_agent_followup_can_use_memory_without_retrieval():
    supervisor = build_supervisor()

    result = supervisor.run(
        "我的血压风险严不严重？",
        [{"role": "system", "content": "结构化诊断记忆：主要风险=高血压风险；异常指标=收缩压=176mmHg"}],
    )

    assert any(event.get("agent") == "memory_agent" and event.get("action") == "use_memory" for event in result.events)
    assert not any(event.get("agent") == "retrieval_agent" for event in result.events)
    assert result.answer.strip()


def test_multi_agent_followup_calls_langchain_structured_tool_for_knowledge():
    supervisor = build_supervisor()

    result = supervisor.run("高血压饮食怎么吃？", [{"role": "assistant", "content": "你有高血压风险。"}])

    assert any(event["type"] == "tool_call" and event.get("agent") == "retrieval_agent" for event in result.events)
    assert "证据：" in result.answer


def test_streaming_initial_assessment_forwards_workflow_step_events():
    supervisor = build_supervisor()

    events = list(supervisor.iter_events("男，52岁，血压176/108 mmHg，请判断健康状况。", []))

    assert any(event.get("agent") == "triage_agent" for event in events)
    assert any(event["type"] == "step" and event.get("name") == "parse_raw_input" for event in events)
    assert any(event["type"] == "assessment_result" and event.get("internal") for event in events)
    assert events[-1]["type"] == "final_answer"


def test_memory_agent_uses_injected_context_builder_when_session_id_is_available():
    def build_context(session_id: str, user_input: str):
        assert session_id == "s1"
        assert "血压" in user_input
        return SimpleNamespace(
            history=[{"role": "system", "content": "结构化诊断记忆：主要风险=高血压风险；异常指标=收缩压=176mmHg"}]
        )

    supervisor = build_supervisor(memory_context_builder=build_context)

    result = supervisor.run("我的血压风险严不严重？", [], session_id="s1")

    assert any(event.get("agent") == "memory_agent" and event.get("action") == "use_memory" for event in result.events)


def test_retrieval_query_is_expanded_with_memory_disease_context():
    tool = CapturingTool()
    supervisor = build_supervisor(tool=tool)

    supervisor.run(
        "早餐怎么吃？",
        [{"role": "system", "content": "结构化诊断记忆：主要风险=高血压风险；异常指标=收缩压=176mmHg"}],
    )

    assert tool.queries
    assert "高血压" in tool.queries[0]
    assert "饮食管理" in tool.queries[0]


def test_safety_review_rewrites_specific_dosage_advice():
    supervisor = build_supervisor(
        followup_answer_builder=lambda user_input, history, evidence: "可以口服氯沙坦片50mg，每天1次，并咨询医生。"
    )

    result = supervisor.run("我的药量要怎么调？", [{"role": "assistant", "content": "你有高血压风险。"}])

    assert any(event.get("agent") == "safety_review_agent" and event.get("action") == "rewrite_required" for event in result.events)
    assert "50mg" not in result.answer
    assert "不要自行调整" in result.answer
