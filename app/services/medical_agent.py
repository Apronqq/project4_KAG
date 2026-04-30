from __future__ import annotations

import asyncio
import json
import logging
import re
from typing import Generator

from app.schemas.exam import (
    MedicalAssessmentResponse,
    PrimaryDiagnosis,
    SecondaryRecommendations,
)
from app.services.agent_tools import MedicalKnowledgeRetrievalTool
from app.services.react_agent import MedicalReActAgent
from app.workflows.medical_kag_pipeline import MedicalKAGWorkflow

logger = logging.getLogger(__name__)


class MedicalAssessmentAgent:
    def __init__(
        self,
        parser,
        normalizer,
        rule_engine,
        graph_store,
        evidence_store,
        ranker,
        formatter,
        query_planner,
        chat_model=None,
        top_k_evidence: int = 5,
        workflow: MedicalKAGWorkflow | None = None,
    ):
        self._parser = parser
        self._normalizer = normalizer
        self._rule_engine = rule_engine
        self._graph_store = graph_store
        self._evidence_store = evidence_store
        self._ranker = ranker
        self._formatter = formatter
        self._query_planner = query_planner
        self._chat_model = chat_model
        self._top_k_evidence = top_k_evidence
        self._workflow = workflow or MedicalKAGWorkflow(
            parser=parser,
            normalizer=normalizer,
            rule_engine=rule_engine,
            graph_store=graph_store,
            evidence_store=evidence_store,
            ranker=ranker,
            formatter=formatter,
            query_planner=query_planner,
            top_k_evidence=top_k_evidence,
        )
        self._medical_knowledge_tool = MedicalKnowledgeRetrievalTool(
            evidence_store=self._evidence_store,
            top_k=self._top_k_evidence,
        )
        self._react_agent = MedicalReActAgent(
            chat_model=self._chat_model,
            knowledge_tool=self._medical_knowledge_tool,
            answer_builder=self._build_followup_fallback_answer,
        )

    def assess(self, raw_text: str) -> tuple[str, MedicalAssessmentResponse]:
        state = self._workflow.run_state(raw_text)
        answer = self._compose_answer(state.response)
        return answer, state.response

    def looks_like_initial_assessment(self, user_input: str) -> bool:
        return self._looks_like_initial_assessment(user_input)

    def chat_assess(self, user_input: str, session_history: list[dict]) -> str:
        if not session_history or self._looks_like_initial_assessment(user_input):
            return self.assess(user_input)[0]
        result = self._react_agent.run(user_input, session_history)
        return result.answer

    def stream_assess(self, raw_text: str, session_history: list[dict] | None = None) -> Generator[dict, None, None]:
        if session_history and not self._looks_like_initial_assessment(raw_text):
            yield from self.stream_followup(raw_text, session_history)
            return

        state = None
        event_iter = self._workflow.iter_events(raw_text)
        while True:
            try:
                yield next(event_iter)
            except StopIteration as stop:
                state = stop.value
                break

        yield {"type": "step", "label": "生成答复", "detail": "正在组织最终中文诊断与建议"}
        # 中文注释：首次体检评估已经走完确定性流水线，这里直接根据结构化结果生成文本，避免 Agent 再次调用评估工具。
        answer = self._compose_answer(state.response)
        for chunk in self._chunk_text(answer, size=20):
            yield {"type": "content", "content": chunk}

        yield {
            "type": "result",
            "payload": state.response.model_dump(),
        }
        yield {"type": "done"}

    def stream_followup(self, user_input: str, session_history: list[dict]) -> Generator[dict, None, None]:
        answer = ""
        # 中文注释：追问不重复跑体检流水线，而是走显式 Agent 循环，逐步暴露决策与工具调用。
        for event in self._react_agent.iter_events(user_input, session_history):
            if event["type"] == "final_answer":
                answer = event["content"]
                yield {"type": "agent_synthesizing", "detail": "正在整合上下文和检索结果生成最终回答"}
                continue
            yield event
        for chunk in self._chunk_text(answer, size=20):
            yield {"type": "content", "content": chunk}
        yield {"type": "done"}

    async def stream_assess_async(self, raw_text: str, session_history: list[dict] | None = None):
        if session_history and not self._looks_like_initial_assessment(raw_text):
            for event in self.stream_followup(raw_text, session_history):
                yield event
                await asyncio.sleep(0)
            return

        state = None
        async for event in self._workflow.iter_events_async(raw_text):
            if event.get("type") == "workflow_state":
                state = event["state"]
                continue
            yield event

        yield {"type": "step", "label": "生成答复", "detail": "正在组织最终中文诊断与建议"}
        answer = self._compose_answer(state.response)
        for chunk in self._chunk_text(answer, size=20):
            yield {"type": "content", "content": chunk}
            await asyncio.sleep(0)
        yield {"type": "result", "payload": state.response.model_dump()}
        yield {"type": "done"}

    def _compose_answer(self, response: MedicalAssessmentResponse) -> str:
        if self._chat_model is not None:
            try:
                prompt = self._build_answer_prompt(response)
                message = self._chat_model.invoke(prompt)
                content = getattr(message, "content", None)
                if isinstance(content, str) and content.strip():
                    return content.strip()
            except Exception:
                logger.warning("medical_agent.compose_answer_failed", exc_info=True)
        return self._build_fallback_answer(response)

    def _build_answer_prompt(self, response: MedicalAssessmentResponse) -> str:
        payload = {
            "health_status": response.primary_diagnosis.health_status,
            "urgency_level": response.primary_diagnosis.urgency_level,
            "risks": [
                {
                    "risk_name": item.risk_name,
                    "risk_level": item.risk_level,
                    "disease_name": item.disease_name,
                    "supported_states": item.supported_states,
                }
                for item in response.primary_diagnosis.potential_risks
            ],
            "recommendations": [
                item.model_dump() for item in response.secondary_recommendations.recommendations_by_disease
            ],
            "evidence_titles": [item.title for item in response.evidence.chunks],
        }
        return (
            "You are a medical assessment agent. Respond in concise Chinese. "
            "Explain health status, key risks, urgency, and practical suggestions. "
            "Do not expose JSON. "
            f"Structured assessment: {json.dumps(payload, ensure_ascii=False)}"
        )

    @staticmethod
    def _build_followup_fallback_answer(user_input: str, session_history: list[dict], evidence_text: str) -> str:
        memory_parts = [
            str(item.get("content", ""))
            for item in session_history
            if item.get("role") == "system" and ("诊断记忆" in str(item.get("content", "")) or "用户事实记忆" in str(item.get("content", "")))
        ]
        memory_text = "\n".join(memory_parts[-2:])
        lines = ["基于当前会话中已有的体检评估和知识库资料，我给出如下辅助说明："]
        if memory_text:
            lines.append(f"\n已参考的个人化上下文：\n{memory_text}")
        if evidence_text:
            lines.append(f"\n检索到的医学资料：\n{evidence_text}")
        lines.append("\n以上内容不能替代医生面诊；如果问题涉及用药调整、症状加重或高风险指标，请结合医生意见处理。")
        return "\n".join(lines)

    @staticmethod
    def _looks_like_initial_assessment(user_input: str) -> bool:
        lowered = user_input.lower()
        keywords = ["血压", "mmhg", "egfr", "肌酐", "空腹血糖", "hba1c", "体检", "指标", "化验"]
        has_medical_keyword = any(keyword in lowered for keyword in keywords)
        has_numeric_value = bool(re.search(r"\d+(\.\d+)?", lowered))
        explicit_initial_intent = any(token in lowered for token in ["请判断", "评估一下", "体检报告", "化验单"])
        # 中文注释：仅提到“我的血压/血糖”通常是追问；同时出现数值或明确评估意图时才判定为初诊。
        return has_medical_keyword and (has_numeric_value or explicit_initial_intent)

    def _build_fallback_answer(self, response: MedicalAssessmentResponse) -> str:
        primary: PrimaryDiagnosis = response.primary_diagnosis
        secondary: SecondaryRecommendations = response.secondary_recommendations
        lines = [
            f"健康状态：{self._cn_health_status(primary.health_status)}。",
            f"紧急程度：{self._cn_urgency(primary.urgency_level)}。",
        ]
        if primary.potential_risks:
            risk_lines = []
            for risk in primary.potential_risks:
                risk_lines.append(
                    f"{risk.risk_name}（关联疾病：{risk.disease_name}，风险等级：{self._cn_level(risk.risk_level)}）"
                )
            lines.append("主要风险判断：" + "；".join(risk_lines) + "。")
        if secondary.recommended_departments:
            lines.append("建议就诊科室：" + "、".join(secondary.recommended_departments) + "。")
        if secondary.follow_up_tests:
            lines.append("建议复查项目：" + "、".join(secondary.follow_up_tests) + "。")
        if secondary.lifestyle_interventions:
            lines.append("生活方式干预建议：" + "、".join(secondary.lifestyle_interventions) + "。")
        if secondary.medication_directions:
            lines.append("用药方向建议：" + "、".join(secondary.medication_directions) + "。")
        if secondary.contraindications:
            lines.append("禁忌与警示：" + "、".join(secondary.contraindications) + "。")
        if secondary.human_review_required:
            lines.append("该结果建议结合医生进一步复核，不应替代线下诊疗。")
        return "\n".join(lines)

    @staticmethod
    def _chunk_text(text: str, size: int = 20):
        for idx in range(0, len(text), size):
            yield text[idx : idx + size]

    @staticmethod
    def _cn_health_status(value: str) -> str:
        mapping = {
            "healthy": "健康",
            "subhealthy": "亚健康",
            "needs_follow_up": "需重点复查",
            "high_risk": "高风险",
        }
        return mapping.get(value, value)

    @staticmethod
    def _cn_urgency(value: str) -> str:
        mapping = {
            "low": "低",
            "medium": "中",
            "high": "高",
            "urgent": "紧急",
        }
        return mapping.get(value, value)

    @staticmethod
    def _cn_level(value: str) -> str:
        mapping = {
            "low": "低",
            "medium": "中",
            "high": "高",
        }
        return mapping.get(value, value)
