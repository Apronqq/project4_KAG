from __future__ import annotations

from app.schemas.exam import DetectedState, NormalizedMedicalExamJSON, RetrievalQuery, RiskCandidate


class EvidenceQueryPlanner:
    def build_queries(
        self,
        exam_json: NormalizedMedicalExamJSON,
        detected_states: list[DetectedState],
        risks: list[RiskCandidate],
    ) -> list[RetrievalQuery]:
        queries: list[RetrievalQuery] = []
        seen: set[str] = set()

        def add(label: str, text: str) -> None:
            normalized = " ".join(text.split())
            if not normalized or normalized in seen:
                return
            seen.add(normalized)
            queries.append(RetrievalQuery(label=label, text=normalized))

        if exam_json.user_question:
            add("user_question", exam_json.user_question)

        top_risks = risks[:3]
        for risk in top_risks:
            add(
                "risk_guideline",
                f"{risk.disease_name} {risk.risk_name} 成人 体检 指南 风险识别 干预建议 复查",
            )
            add(
                "risk_followup",
                f"{risk.disease_name} 复查建议 随访 科室 用药禁忌",
            )

        for state in detected_states[:4]:
            add(
                "abnormal_indicator",
                f"{state.indicator_name} {state.label} 风险 分层 指南",
            )

        if exam_json.exam_items:
            leading = " ".join(item.name for item in exam_json.exam_items[:5])
            add("indicator_summary", f"{leading} 异常 体检 风险评估")

        return queries
