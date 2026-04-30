from app.graph.store import InMemoryGraphStore
from app.models.factory import LightweightEmbeddingProvider
from app.retrieval.evidence_store import InMemoryEvidenceStore
from app.retrieval.risk_ranker import MedicalRiskRanker
from app.services.agent_tools import MedicalKnowledgeRetrievalTool
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


def test_medical_knowledge_retrieval_tool_returns_evidence_text():
    agent = build_agent()
    tool = MedicalKnowledgeRetrievalTool(agent._evidence_store, top_k=2)

    payload = tool._run("高血压患者需要怎样复查")

    assert isinstance(payload, str)
    assert payload.strip()
