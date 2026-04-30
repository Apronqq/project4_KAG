from __future__ import annotations

from dataclasses import dataclass

from app.core.settings import Settings
from app.db.database import DatabaseManager
from app.graph.kb_builder import MedicalKnowledgeBuilder
from app.graph.store import build_graph_store
from app.models.factory import ModelFactory
from app.retrieval.evidence_store import build_evidence_store
from app.retrieval.risk_ranker import MedicalRiskRanker
from app.services.chat_history_service import ChatHistoryService
from app.services.diagnosis_formatter import DiagnosisFormatter
from app.services.document_ingestion import DocumentChunker
from app.services.evidence_query_planner import EvidenceQueryPlanner
from app.services.indicator_normalizer import IndicatorNormalizer
from app.services.input_parser import MedicalInputParser
from app.services.knowledge_registry import KnowledgeDocumentRegistry
from app.services.medical_agent import MedicalAssessmentAgent
from app.services.rules import IndicatorRuleEngine
from app.services.upload_job_registry import UploadJobRegistry
from app.workflows.medical_kag_pipeline import MedicalKAGWorkflow


@dataclass
class AppRuntime:
    settings: Settings
    database_manager: DatabaseManager
    model_runtime: object
    indicator_normalizer: IndicatorNormalizer
    input_parser: MedicalInputParser
    rule_engine: IndicatorRuleEngine
    graph_store: object
    evidence_store: object
    risk_ranker: MedicalRiskRanker
    diagnosis_formatter: DiagnosisFormatter
    evidence_query_planner: EvidenceQueryPlanner
    knowledge_builder: MedicalKnowledgeBuilder
    document_chunker: DocumentChunker
    knowledge_registry: KnowledgeDocumentRegistry
    upload_job_registry: UploadJobRegistry
    chat_history_service: ChatHistoryService
    medical_workflow: MedicalKAGWorkflow
    medical_agent: MedicalAssessmentAgent


_runtime: AppRuntime | None = None


def get_runtime() -> AppRuntime:
    global _runtime
    if _runtime is not None:
        return _runtime

    settings = Settings()
    database_manager = DatabaseManager(settings.database_url)
    model_runtime = ModelFactory(settings).build()

    indicator_normalizer = IndicatorNormalizer()
    input_parser = MedicalInputParser(indicator_normalizer, extractor=model_runtime.extractor)
    rule_engine = IndicatorRuleEngine()
    graph_store = build_graph_store(settings)
    evidence_store = build_evidence_store(
        settings,
        embedder=model_runtime.embedding_provider,
        reranker=model_runtime.reranker,
    )
    risk_ranker = MedicalRiskRanker()
    diagnosis_formatter = DiagnosisFormatter()
    evidence_query_planner = EvidenceQueryPlanner()
    document_chunker = DocumentChunker()
    knowledge_registry = KnowledgeDocumentRegistry(
        registry_path=settings.knowledge_registry_path,
        upload_root=settings.knowledge_upload_root,
    )
    upload_job_registry = UploadJobRegistry(settings.data_root / "knowledge_upload_jobs.json")
    chat_history_service = ChatHistoryService(
        database_manager.session_factory,
        settings,
        summary_llm=model_runtime.assistant_chat_model,
    )
    knowledge_builder = MedicalKnowledgeBuilder(graph_store, evidence_store, knowledge_registry)

    medical_workflow = MedicalKAGWorkflow(
        parser=input_parser,
        normalizer=indicator_normalizer,
        rule_engine=rule_engine,
        graph_store=graph_store,
        evidence_store=evidence_store,
        ranker=risk_ranker,
        formatter=diagnosis_formatter,
        query_planner=evidence_query_planner,
        top_k_evidence=settings.top_k_evidence,
    )
    medical_agent = MedicalAssessmentAgent(
        parser=input_parser,
        normalizer=indicator_normalizer,
        rule_engine=rule_engine,
        graph_store=graph_store,
        evidence_store=evidence_store,
        ranker=risk_ranker,
        formatter=diagnosis_formatter,
        query_planner=evidence_query_planner,
        chat_model=model_runtime.assistant_chat_model,
        top_k_evidence=settings.top_k_evidence,
        workflow=medical_workflow,
    )

    _runtime = AppRuntime(
        settings=settings,
        database_manager=database_manager,
        model_runtime=model_runtime,
        indicator_normalizer=indicator_normalizer,
        input_parser=input_parser,
        rule_engine=rule_engine,
        graph_store=graph_store,
        evidence_store=evidence_store,
        risk_ranker=risk_ranker,
        diagnosis_formatter=diagnosis_formatter,
        evidence_query_planner=evidence_query_planner,
        knowledge_builder=knowledge_builder,
        document_chunker=document_chunker,
        knowledge_registry=knowledge_registry,
        upload_job_registry=upload_job_registry,
        chat_history_service=chat_history_service,
        medical_workflow=medical_workflow,
        medical_agent=medical_agent,
    )
    return _runtime
