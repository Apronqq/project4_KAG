from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class PatientProfile(BaseModel):
    sex: str | None = None
    age: int | None = None


class ExamItem(BaseModel):
    code: str | None = None
    name: str
    value: float | None = None
    unit: str | None = None
    source_text: str | None = None


class RawPatientProfile(BaseModel):
    sex: str | None = None
    age: int | None = None


class RawExamItem(BaseModel):
    name: str
    value: float | None = None
    unit: str | None = None
    source_text: str | None = None


class ExtractedExamPayload(BaseModel):
    patient_profile: RawPatientProfile = Field(default_factory=RawPatientProfile)
    exam_items: list[RawExamItem] = Field(default_factory=list)
    medical_history: list[str] = Field(default_factory=list)
    current_medications: list[str] = Field(default_factory=list)
    allergies: list[str] = Field(default_factory=list)
    user_question: str = ""


class NormalizedMedicalExamJSON(BaseModel):
    patient_profile: PatientProfile = Field(default_factory=PatientProfile)
    exam_items: list[ExamItem] = Field(default_factory=list)
    medical_history: list[str] = Field(default_factory=list)
    current_medications: list[str] = Field(default_factory=list)
    allergies: list[str] = Field(default_factory=list)
    user_question: str = ""
    source_type: Literal["text", "json"] = "text"


class DetectedState(BaseModel):
    indicator_code: str
    indicator_name: str
    state_code: str
    label: str
    severity: Literal["low", "medium", "high"]
    value: float
    unit: str | None = None
    rule_id: str


class GraphPath(BaseModel):
    path_type: Literal["risk", "intervention", "contraindication", "follow_up", "department"]
    nodes: list[str]
    score: float


class RiskCandidate(BaseModel):
    risk_code: str
    risk_name: str
    disease_code: str
    disease_name: str
    risk_level: Literal["low", "medium", "high"]
    graph_score: float
    support_count: int = 0
    graph_support_score: float = 0.0
    evidence_support_score: float = 0.0
    final_score: float = 0.0
    supported_states: list[str] = Field(default_factory=list)
    graph_paths: list[GraphPath] = Field(default_factory=list)


class InterventionCandidate(BaseModel):
    disease_code: str
    interventions: list[str] = Field(default_factory=list)
    medication_directions: list[str] = Field(default_factory=list)
    contraindications: list[str] = Field(default_factory=list)
    follow_up_tests: list[str] = Field(default_factory=list)
    departments: list[str] = Field(default_factory=list)
    graph_paths: list[GraphPath] = Field(default_factory=list)


class EvidenceChunk(BaseModel):
    chunk_id: str
    doc_id: str
    title: str
    text: str
    linked_node_codes: list[str] = Field(default_factory=list)
    source_type: str = "guideline"
    relevance_score: float = 0.0
    dense_score: float = 0.0
    lexical_score: float = 0.0
    graph_overlap_score: float = 0.0
    source_authority_score: float = 0.0
    fusion_score: float = 0.0
    final_score: float = 0.0
    rerank_score: float = 0.0
    matched_queries: list[str] = Field(default_factory=list)


class RetrievalQuery(BaseModel):
    label: str
    text: str


class PrimaryDiagnosis(BaseModel):
    health_status: Literal["healthy", "subhealthy", "needs_follow_up", "high_risk"]
    urgency_level: Literal["low", "medium", "high", "urgent"]
    potential_risks: list[RiskCandidate] = Field(default_factory=list)
    key_abnormal_indicators: list[DetectedState] = Field(default_factory=list)


class DiseaseRecommendation(BaseModel):
    risk_code: str
    risk_name: str
    disease_code: str
    disease_name: str
    interventions: list[str] = Field(default_factory=list)
    medication_directions: list[str] = Field(default_factory=list)
    contraindications: list[str] = Field(default_factory=list)
    follow_up_tests: list[str] = Field(default_factory=list)
    departments: list[str] = Field(default_factory=list)


class SecondaryRecommendations(BaseModel):
    recommended_departments: list[str] = Field(default_factory=list)
    follow_up_tests: list[str] = Field(default_factory=list)
    lifestyle_interventions: list[str] = Field(default_factory=list)
    medication_directions: list[str] = Field(default_factory=list)
    contraindications: list[str] = Field(default_factory=list)
    recommendations_by_disease: list[DiseaseRecommendation] = Field(default_factory=list)
    human_review_required: bool = True


class AssessmentEvidence(BaseModel):
    graph_paths: list[GraphPath] = Field(default_factory=list)
    chunks: list[EvidenceChunk] = Field(default_factory=list)


class MedicalAssessmentResponse(BaseModel):
    normalized_exam_json: NormalizedMedicalExamJSON
    primary_diagnosis: PrimaryDiagnosis
    secondary_recommendations: SecondaryRecommendations
    evidence: AssessmentEvidence


class MedicalParseResponse(BaseModel):
    normalized_exam_json: NormalizedMedicalExamJSON
    missing_fields: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class KnowledgeBuildResponse(BaseModel):
    graph_backend: str
    evidence_backend: str
    states_seeded: int
    risks_seeded: int
    interventions_seeded: int
    evidence_seeded: int


class SessionInfo(BaseModel):
    session_id: str
    title: str
    summary_text: str = ""
    created_at: str
    updated_at: str


class SessionListResponse(BaseModel):
    sessions: list[SessionInfo] = Field(default_factory=list)


class SessionMessage(BaseModel):
    role: str
    content: str
    created_at: str


class SessionMessagesResponse(BaseModel):
    session_id: str
    messages: list[SessionMessage] = Field(default_factory=list)


class SessionCreateResponse(BaseModel):
    session: SessionInfo


class KnowledgeDocument(BaseModel):
    doc_id: str
    filename: str
    file_type: str
    content_hash: str
    chunk_count: int
    file_path: str = ""
    linked_node_codes: list[str] = Field(default_factory=list)


class KnowledgeDocumentUploadResponse(BaseModel):
    document: KnowledgeDocument
    message: str


class KnowledgeDocumentListResponse(BaseModel):
    documents: list[KnowledgeDocument] = Field(default_factory=list)


class KnowledgeUploadJob(BaseModel):
    job_id: str
    filename: str
    content_hash: str
    status: Literal["pending", "processing", "completed", "failed", "duplicate"]
    message: str = ""
    document_id: str | None = None


class KnowledgeUploadJobResponse(BaseModel):
    job: KnowledgeUploadJob


class KnowledgeUploadJobListResponse(BaseModel):
    jobs: list[KnowledgeUploadJob] = Field(default_factory=list)


class RuntimeStatusResponse(BaseModel):
    graph_backend: str
    evidence_backend: str
    graph_ready: bool
    evidence_ready: bool
    graph_data_ready: bool
    evidence_data_ready: bool
    graph_mode: str
    evidence_mode: str
    embedding_backend: str
    extractor_backend: str | None = None
    reranker_backend: str | None = None
    graph_degraded: bool = False
    evidence_degraded: bool = False
    graph_fallback_reason: str = ""
    evidence_fallback_reason: str = ""
    checked_at: str = ""
    components: dict[str, dict[str, Any]] = Field(default_factory=dict)


class InternalAssessmentState(BaseModel):
    raw_input: Any
    normalized_exam_json: NormalizedMedicalExamJSON | None = None
    missing_fields: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    detected_states: list[DetectedState] = Field(default_factory=list)
    risk_candidates: list[RiskCandidate] = Field(default_factory=list)
    intervention_candidates: list[InterventionCandidate] = Field(default_factory=list)
    retrieval_queries: list[RetrievalQuery] = Field(default_factory=list)
    evidence_chunks: list[EvidenceChunk] = Field(default_factory=list)
    primary_diagnosis: PrimaryDiagnosis | None = None
    secondary_recommendations: SecondaryRecommendations | None = None
    response: MedicalAssessmentResponse | None = None
