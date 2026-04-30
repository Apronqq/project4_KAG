from __future__ import annotations

from app.schemas.exam import (
    AssessmentEvidence,
    DiseaseRecommendation,
    EvidenceChunk,
    InterventionCandidate,
    MedicalAssessmentResponse,
    NormalizedMedicalExamJSON,
    PrimaryDiagnosis,
    RiskCandidate,
    SecondaryRecommendations,
)


class DiagnosisFormatter:
    def build_primary(self, exam_json: NormalizedMedicalExamJSON, risks: list[RiskCandidate], abnormal_states) -> PrimaryDiagnosis:
        if not risks and not abnormal_states:
            return PrimaryDiagnosis(
                health_status="healthy",
                urgency_level="low",
                potential_risks=[],
                key_abnormal_indicators=[],
            )

        urgency_level = "low"
        health_status = "subhealthy"
        high_risk_count = sum(1 for risk in risks if risk.risk_level == "high")
        high_severity_state_count = sum(1 for state in abnormal_states if state.severity == "high")
        top_score = risks[0].final_score if risks else 0.0

        if high_risk_count >= 2 or (high_risk_count >= 1 and high_severity_state_count >= 2) or top_score >= 0.9:
            urgency_level = "urgent"
            health_status = "high_risk"
        elif high_risk_count >= 1:
            urgency_level = "high"
            health_status = "high_risk"
        elif risks and risks[0].risk_level == "medium":
            urgency_level = "medium"
            health_status = "needs_follow_up"
        return PrimaryDiagnosis(
            health_status=health_status,
            urgency_level=urgency_level,
            potential_risks=risks,
            key_abnormal_indicators=abnormal_states,
        )

    def build_secondary(
        self,
        risks: list[RiskCandidate],
        recommendation: InterventionCandidate | None,
        recommendations_by_disease_map: dict[str, InterventionCandidate] | None = None,
    ) -> SecondaryRecommendations:
        if not risks or recommendation is None:
            return SecondaryRecommendations(human_review_required=False)
        recommendation_map = recommendations_by_disease_map or {recommendation.disease_code: recommendation}
        recommendations_by_disease = []
        for risk in risks:
            disease_recommendation = recommendation_map.get(risk.disease_code, recommendation)
            recommendations_by_disease.append(
                DiseaseRecommendation(
                    risk_code=risk.risk_code,
                    risk_name=risk.risk_name,
                    disease_code=risk.disease_code,
                    disease_name=risk.disease_name,
                    interventions=disease_recommendation.interventions,
                    medication_directions=disease_recommendation.medication_directions,
                    contraindications=disease_recommendation.contraindications,
                    follow_up_tests=disease_recommendation.follow_up_tests,
                    departments=disease_recommendation.departments,
                )
            )
        return SecondaryRecommendations(
            recommended_departments=recommendation.departments,
            follow_up_tests=recommendation.follow_up_tests,
            lifestyle_interventions=recommendation.interventions,
            medication_directions=recommendation.medication_directions,
            contraindications=recommendation.contraindications,
            recommendations_by_disease=recommendations_by_disease,
            human_review_required=any(risk.risk_level == "high" for risk in risks)
            or bool(recommendation.contraindications)
            or len(risks) > 1,
        )

    def build_response(
        self,
        exam_json: NormalizedMedicalExamJSON,
        primary: PrimaryDiagnosis,
        secondary: SecondaryRecommendations,
        graph_paths,
        chunks: list[EvidenceChunk],
    ) -> MedicalAssessmentResponse:
        return MedicalAssessmentResponse(
            normalized_exam_json=exam_json,
            primary_diagnosis=primary,
            secondary_recommendations=secondary,
            evidence=AssessmentEvidence(graph_paths=graph_paths, chunks=chunks),
        )
