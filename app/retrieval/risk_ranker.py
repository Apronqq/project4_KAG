from __future__ import annotations

from app.schemas.exam import DetectedState, EvidenceChunk, InterventionCandidate, RiskCandidate


class MedicalRiskRanker:
    def rank_risks(
        self,
        risks: list[RiskCandidate],
        evidence_chunks: list[EvidenceChunk],
        detected_states: list[DetectedState],
    ) -> list[RiskCandidate]:
        evidence_score_map: dict[str, float] = {}
        for chunk in evidence_chunks:
            for code in chunk.linked_node_codes:
                evidence_score_map[code] = max(evidence_score_map.get(code, 0.0), chunk.final_score or chunk.relevance_score)

        severity_weight = {"low": 0.4, "medium": 0.7, "high": 1.0}
        state_weight_map = {state.state_code: severity_weight[state.severity] for state in detected_states}

        rescored: list[RiskCandidate] = []
        for risk in risks:
            support_scores = [
                evidence_score_map.get(risk.risk_code, 0.0),
                evidence_score_map.get(risk.disease_code, 0.0),
            ]
            evidence_score = max(support_scores) if support_scores else 0.0
            support_count = len(set(risk.supported_states))
            support_strength = sum(state_weight_map.get(code, 0.5) for code in set(risk.supported_states)) / max(support_count, 1)
            support_count_score = min(1.0, support_count / 3.0)
            base_graph_score = risk.graph_score
            final_score = (
                0.45 * base_graph_score
                + 0.20 * evidence_score
                + 0.20 * support_strength
                + 0.15 * support_count_score
            )
            rescored.append(
                risk.model_copy(
                    update={
                        "support_count": support_count,
                        "graph_support_score": round(base_graph_score, 4),
                        "evidence_support_score": round(evidence_score, 4),
                        "final_score": round(final_score, 4),
                        "graph_score": round(final_score, 4),
                    }
                )
            )

        rescored.sort(key=lambda item: item.final_score, reverse=True)
        return rescored

    def merge_recommendations(self, recommendations: list[InterventionCandidate]) -> InterventionCandidate:
        if not recommendations:
            return InterventionCandidate(disease_code="unknown")
        recommendations = sorted(recommendations, key=lambda item: (len(item.contraindications), len(item.follow_up_tests)), reverse=True)
        base = recommendations[0]
        interventions = []
        medication_directions = []
        contraindications = []
        follow_up_tests = []
        departments = []
        graph_paths = []
        seen_sets = {
            "interventions": interventions,
            "medication_directions": medication_directions,
            "contraindications": contraindications,
            "follow_up_tests": follow_up_tests,
            "departments": departments,
        }
        for item in recommendations:
            for key, target in seen_sets.items():
                for value in getattr(item, key):
                    if value not in target:
                        target.append(value)
            graph_paths.extend(item.graph_paths)
        return InterventionCandidate(
            disease_code=base.disease_code,
            interventions=interventions,
            medication_directions=medication_directions,
            contraindications=contraindications,
            follow_up_tests=follow_up_tests,
            departments=departments,
            graph_paths=graph_paths,
        )

    def index_recommendations_by_disease(
        self,
        recommendations: list[InterventionCandidate],
    ) -> dict[str, InterventionCandidate]:
        return {item.disease_code: item for item in recommendations}
