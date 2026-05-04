from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from typing import Callable, Literal, TypedDict

from langgraph.graph import END, StateGraph

from app.schemas.exam import InternalAssessmentState, MedicalAssessmentResponse

logger = logging.getLogger(__name__)


class MedicalMultiAgentState(TypedDict, total=False):
    user_input: str
    session_id: str | None
    session_history: list[dict]
    route: Literal["assessment", "followup"]
    memory_text: str
    needs_retrieval: bool
    retrieval_query: str
    evidence_text: str
    safety_notes: list[str]
    requires_safe_rewrite: bool
    safety_revision_count: int
    answer: str
    structured_response: MedicalAssessmentResponse
    assessment_state: InternalAssessmentState
    events: list[dict]


@dataclass
class MedicalMultiAgentRunResult:
    answer: str
    events: list[dict]
    structured_response: MedicalAssessmentResponse | None = None


MEMORY_KEYWORDS = [
    "诊断记忆",
    "用户事实记忆",
    "异常指标",
    "健康状态",
    "主要风险",
    "复查项目",
    "趋势",
    "对话摘要记忆",
]

DISEASE_TERMS = [
    "高血压",
    "慢性肾脏病",
    "糖尿病",
    "糖尿病前期",
    "血脂异常",
    "高胆固醇",
    "脂肪肝",
    "高尿酸",
    "痛风",
]


def extract_memory_text_from_history(session_history: list[dict], limit: int = 3) -> str:
    memory_parts = [
        str(item.get("content", ""))
        for item in session_history
        if item.get("role") == "system" and any(keyword in str(item.get("content", "")) for keyword in MEMORY_KEYWORDS)
    ]
    return "\n".join(memory_parts[-limit:])


def build_medical_multi_agent_supervisor(
    *,
    workflow,
    knowledge_tool,
    chat_model,
    assessment_answer_builder: Callable[[MedicalAssessmentResponse], str],
    followup_answer_builder: Callable[[str, list[dict], str], str],
    initial_assessment_detector: Callable[[str], bool],
    memory_context_builder: Callable[[str, str], object] | None = None,
) -> "MedicalMultiAgentSupervisor":
    return MedicalMultiAgentSupervisor(
        workflow=workflow,
        knowledge_tool=knowledge_tool,
        chat_model=chat_model,
        assessment_answer_builder=assessment_answer_builder,
        followup_answer_builder=followup_answer_builder,
        initial_assessment_detector=initial_assessment_detector,
        memory_context_builder=memory_context_builder,
    )


class MedicalMultiAgentSupervisor:
    """Supervisor that coordinates specialist medical agents with LangGraph.

    Agent roles:
    - TriageAgent: routes initial assessment vs. follow-up.
    - AssessmentAgent: runs the deterministic KAG workflow.
    - MemoryAgent: extracts and evaluates usable session memory.
    - RetrievalAgent: calls the LangChain medical knowledge tool.
    - SynthesisAgent: generates the user-facing answer.
    - SafetyReviewAgent: applies medical boundary checks before final output.
    """

    def __init__(
        self,
        *,
        workflow,
        knowledge_tool,
        chat_model,
        assessment_answer_builder: Callable[[MedicalAssessmentResponse], str],
        followup_answer_builder: Callable[[str, list[dict], str], str],
        initial_assessment_detector: Callable[[str], bool],
        memory_context_builder: Callable[[str, str], object] | None = None,
    ):
        self._workflow = workflow
        self._knowledge_tool = knowledge_tool
        self._chat_model = chat_model
        self._assessment_answer_builder = assessment_answer_builder
        self._followup_answer_builder = followup_answer_builder
        self._initial_assessment_detector = initial_assessment_detector
        self._memory_context_builder = memory_context_builder
        self._graph = self._build_graph()

    def run(
        self,
        user_input: str,
        session_history: list[dict] | None = None,
        *,
        session_id: str | None = None,
    ) -> MedicalMultiAgentRunResult:
        final_state = self._graph.invoke(
            {
                "user_input": user_input,
                "session_id": session_id,
                "session_history": session_history or [],
                "events": [],
                "safety_notes": [],
                "requires_safe_rewrite": False,
                "safety_revision_count": 0,
            }
        )
        return MedicalMultiAgentRunResult(
            answer=str(final_state.get("answer", "")),
            events=list(final_state.get("events", [])),
            structured_response=final_state.get("structured_response"),
        )

    def iter_events(
        self,
        user_input: str,
        session_history: list[dict] | None = None,
        *,
        session_id: str | None = None,
    ):
        state: MedicalMultiAgentState = {
            "user_input": user_input,
            "session_id": session_id,
            "session_history": session_history or [],
            "events": [],
            "safety_notes": [],
            "requires_safe_rewrite": False,
            "safety_revision_count": 0,
        }

        yield from self._run_node_for_stream(state, self._triage_agent)
        if state.get("route") == "assessment":
            yield from self._stream_assessment_agent(state)
        else:
            yield from self._run_node_for_stream(state, self._memory_agent)
            if state.get("needs_retrieval", True):
                yield from self._run_node_for_stream(state, self._retrieval_agent)
            yield from self._run_node_for_stream(state, self._synthesis_agent)

        while True:
            yield from self._run_node_for_stream(state, self._safety_review_agent)
            if not state.get("requires_safe_rewrite", False):
                return
            yield from self._run_node_for_stream(state, self._synthesis_agent)

    async def aiter_events(
        self,
        user_input: str,
        session_history: list[dict] | None = None,
        *,
        session_id: str | None = None,
    ):
        state: MedicalMultiAgentState = {
            "user_input": user_input,
            "session_id": session_id,
            "session_history": session_history or [],
            "events": [],
            "safety_notes": [],
            "requires_safe_rewrite": False,
            "safety_revision_count": 0,
        }

        for event in self._run_node_for_stream(state, self._triage_agent):
            yield event
        if state.get("route") == "assessment":
            async for event in self._astream_assessment_agent(state):
                yield event
        else:
            for event in self._run_node_for_stream(state, self._memory_agent):
                yield event
            if state.get("needs_retrieval", True):
                for event in self._run_node_for_stream(state, self._retrieval_agent):
                    yield event
            for event in self._run_node_for_stream(state, self._synthesis_agent):
                yield event

        while True:
            for event in self._run_node_for_stream(state, self._safety_review_agent):
                yield event
            if not state.get("requires_safe_rewrite", False):
                return
            for event in self._run_node_for_stream(state, self._synthesis_agent):
                yield event

    def _build_graph(self):
        graph = StateGraph(MedicalMultiAgentState)
        graph.add_node("triage_agent", self._triage_agent)
        graph.add_node("assessment_agent", self._assessment_agent)
        graph.add_node("memory_agent", self._memory_agent)
        graph.add_node("retrieval_agent", self._retrieval_agent)
        graph.add_node("synthesis_agent", self._synthesis_agent)
        graph.add_node("safety_review_agent", self._safety_review_agent)

        graph.set_entry_point("triage_agent")
        graph.add_conditional_edges(
            "triage_agent",
            self._route_after_triage,
            {
                "assessment": "assessment_agent",
                "followup": "memory_agent",
            },
        )
        graph.add_edge("assessment_agent", "safety_review_agent")
        graph.add_conditional_edges(
            "memory_agent",
            self._route_after_memory,
            {
                "retrieve": "retrieval_agent",
                "synthesize": "synthesis_agent",
            },
        )
        graph.add_edge("retrieval_agent", "synthesis_agent")
        graph.add_edge("synthesis_agent", "safety_review_agent")
        graph.add_conditional_edges(
            "safety_review_agent",
            self._route_after_safety,
            {
                "rewrite": "synthesis_agent",
                "end": END,
            },
        )
        return graph.compile()

    @staticmethod
    def _append_event(state: MedicalMultiAgentState, event: dict) -> list[dict]:
        events = list(state.get("events", []))
        events.append(event)
        return events

    def _run_node_for_stream(self, state: MedicalMultiAgentState, node_fn):
        start = len(state.get("events", []))
        state.update(node_fn(state))
        for event in state.get("events", [])[start:]:
            yield event

    def _stream_assessment_agent(self, state: MedicalMultiAgentState):
        start = len(state.get("events", []))
        state["events"] = self._append_event(
            state,
            {
                "type": "agent_thinking",
                "agent": "assessment_agent",
                "detail": "正在执行确定性 KAG 体检评估流水线",
            },
        )
        for event in state.get("events", [])[start:]:
            yield event

        assessment_state = None
        event_iter = self._workflow.iter_events(state["user_input"])
        while True:
            try:
                yield next(event_iter)
            except StopIteration as stop:
                assessment_state = stop.value
                break

        response = assessment_state.response
        state.update(
            {
                "assessment_state": assessment_state,
                "structured_response": response,
                "answer": self._assessment_answer_builder(response),
            }
        )
        yield {"type": "assessment_result", "agent": "assessment_agent", "internal": True, "payload": response}
        state["events"] = self._append_event(
            state,
            {
                "type": "agent_decision",
                "agent": "assessment_agent",
                "action": "workflow_completed",
                "reason": f"完成评估，识别 {len(response.primary_diagnosis.potential_risks)} 个主要风险",
            },
        )
        yield state["events"][-1]

    async def _astream_assessment_agent(self, state: MedicalMultiAgentState):
        start = len(state.get("events", []))
        state["events"] = self._append_event(
            state,
            {
                "type": "agent_thinking",
                "agent": "assessment_agent",
                "detail": "正在执行确定性 KAG 体检评估流水线",
            },
        )
        for event in state.get("events", [])[start:]:
            yield event

        assessment_state = None
        async for event in self._workflow.iter_events_async(state["user_input"]):
            if event.get("type") == "workflow_state":
                assessment_state = event["state"]
                continue
            yield event

        response = assessment_state.response
        state.update(
            {
                "assessment_state": assessment_state,
                "structured_response": response,
                "answer": self._assessment_answer_builder(response),
            }
        )
        yield {"type": "assessment_result", "agent": "assessment_agent", "internal": True, "payload": response}
        state["events"] = self._append_event(
            state,
            {
                "type": "agent_decision",
                "agent": "assessment_agent",
                "action": "workflow_completed",
                "reason": f"完成评估，识别 {len(response.primary_diagnosis.potential_risks)} 个主要风险",
            },
        )
        yield state["events"][-1]

    def _triage_agent(self, state: MedicalMultiAgentState) -> MedicalMultiAgentState:
        user_input = state["user_input"]
        session_history = state.get("session_history", [])
        route: Literal["assessment", "followup"]
        route = (
            "assessment"
            if self._initial_assessment_detector(user_input) or (not session_history and not state.get("session_id"))
            else "followup"
        )
        events = self._append_event(
            state,
            {
                "type": "agent_decision",
                "agent": "triage_agent",
                "action": f"route_to_{route}",
                "reason": "识别为首次体检评估" if route == "assessment" else "识别为基于既有会话的追问",
            },
        )
        return {"route": route, "events": events}

    def _assessment_agent(self, state: MedicalMultiAgentState) -> MedicalMultiAgentState:
        events = self._append_event(
            state,
            {
                "type": "agent_thinking",
                "agent": "assessment_agent",
                "detail": "正在执行确定性 KAG 体检评估流水线",
            },
        )
        assessment_state = self._workflow.run_state(state["user_input"])
        response = assessment_state.response
        answer = self._assessment_answer_builder(response)
        events.append(
            {
                "type": "agent_decision",
                "agent": "assessment_agent",
                "action": "workflow_completed",
                "reason": f"完成评估，识别 {len(response.primary_diagnosis.potential_risks)} 个主要风险",
            }
        )
        return {
            "assessment_state": assessment_state,
            "structured_response": response,
            "answer": answer,
            "events": events,
        }

    def _memory_agent(self, state: MedicalMultiAgentState) -> MedicalMultiAgentState:
        user_input = state["user_input"]
        session_history = self._resolve_session_history(state)
        memory_text = extract_memory_text_from_history(session_history)
        has_relevant_memory = bool(memory_text) and self._asks_personal_followup(user_input)
        needs_retrieval = (not has_relevant_memory) or self._requires_external_knowledge(user_input)
        events = self._append_event(
            state,
            {
                "type": "agent_decision",
                "agent": "memory_agent",
                "action": "need_retrieval" if needs_retrieval else "use_memory",
                "reason": "需要补充医学知识库证据" if needs_retrieval else "会话中已有可用诊断/事实记忆",
            },
        )
        return {"session_history": session_history, "memory_text": memory_text, "needs_retrieval": needs_retrieval, "events": events}

    def _retrieval_agent(self, state: MedicalMultiAgentState) -> MedicalMultiAgentState:
        user_input = state["user_input"]
        retrieval_query = self._build_followup_retrieval_query(user_input, state.get("memory_text", ""))
        events = self._append_event(
            state,
            {
                "type": "tool_call",
                "agent": "retrieval_agent",
                "name": "lookup_medical_knowledge",
                "args_summary": json.dumps({"query": retrieval_query}, ensure_ascii=False)[:160],
            },
        )
        evidence_text = self._invoke_knowledge_tool(retrieval_query)
        events.append(
            {
                "type": "tool_result",
                "agent": "retrieval_agent",
                "name": "lookup_medical_knowledge",
                "result_len": len(evidence_text),
            }
        )
        return {"retrieval_query": retrieval_query, "evidence_text": evidence_text, "events": events}

    def _synthesis_agent(self, state: MedicalMultiAgentState) -> MedicalMultiAgentState:
        events = self._append_event(
            state,
            {
                "type": "agent_synthesizing",
                "agent": "synthesis_agent",
                "detail": "正在整合记忆、检索证据和用户问题生成回答",
            },
        )
        evidence_text = str(state.get("evidence_text", "")).strip()
        if state.get("requires_safe_rewrite", False):
            answer = self._build_safe_followup_answer(state["user_input"], evidence_text, state.get("safety_notes", []))
            return {
                "answer": answer,
                "events": events,
                "requires_safe_rewrite": False,
                "safety_revision_count": state.get("safety_revision_count", 0) + 1,
            }
        answer = self._compose_followup_answer(state["user_input"], state.get("session_history", []), evidence_text)
        return {"answer": answer, "events": events, "requires_safe_rewrite": False}

    def _safety_review_agent(self, state: MedicalMultiAgentState) -> MedicalMultiAgentState:
        answer = str(state.get("answer", ""))
        notes = list(state.get("safety_notes", []))
        user_input = state["user_input"]
        structured = state.get("structured_response")
        requires_rewrite = False
        if structured is not None and structured.secondary_recommendations.human_review_required:
            notes.append("高风险或复杂结果需要医生复核")
        medication_risk = self._mentions_medication_adjustment(user_input) or self._contains_medication_advice(answer)
        dosage_risk = self._contains_dosage_instruction(answer)
        suspected_drugs = self._extract_suspected_drug_terms(answer)
        if medication_risk:
            notes.append("涉及用药调整，需提示结合医生意见")
        if dosage_risk:
            notes.append("回答中疑似包含具体剂量或服药频次，需改写为非处方建议")
        if suspected_drugs:
            notes.append(f"回答中出现需复核的药物/治疗词：{'、'.join(suspected_drugs[:3])}")

        if (dosage_risk or suspected_drugs) and state.get("safety_revision_count", 0) == 0:
            requires_rewrite = True
        elif medication_risk and "医生" not in answer:
            answer = answer.rstrip() + "\n涉及用药调整时，请结合医生意见处理，不要自行增减药物。"

        events = self._append_event(
            state,
            {
                "type": "agent_decision",
                "agent": "safety_review_agent",
                "action": "rewrite_required" if requires_rewrite else ("approved_with_notes" if notes else "approved"),
                "reason": "；".join(notes) if notes else "未发现需要额外拦截的医疗安全风险",
            },
        )
        if not requires_rewrite:
            events.append({"type": "final_answer", "agent": "safety_review_agent", "content": answer})
        return {"answer": answer, "safety_notes": notes, "requires_safe_rewrite": requires_rewrite, "events": events}

    @staticmethod
    def _route_after_triage(state: MedicalMultiAgentState) -> str:
        return state.get("route", "followup")

    @staticmethod
    def _route_after_memory(state: MedicalMultiAgentState) -> str:
        return "retrieve" if state.get("needs_retrieval", True) else "synthesize"

    @staticmethod
    def _route_after_safety(state: MedicalMultiAgentState) -> str:
        return "rewrite" if state.get("requires_safe_rewrite", False) else "end"

    def _invoke_knowledge_tool(self, query: str) -> str:
        try:
            if hasattr(self._knowledge_tool, "invoke"):
                return str(self._knowledge_tool.invoke({"query": query}))
            if hasattr(self._knowledge_tool, "tool"):
                return str(self._knowledge_tool.tool.invoke({"query": query}))
            return str(self._knowledge_tool._run(query))
        except Exception:
            logger.warning("medical_multi_agent.retrieval_failed", exc_info=True)
            return "No relevant medical evidence was found in the knowledge base."

    def _compose_followup_answer(self, user_input: str, session_history: list[dict], evidence_text: str) -> str:
        if self._chat_model is not None:
            try:
                prompt = self._build_synthesis_prompt(user_input, session_history, evidence_text)
                message = self._chat_model.invoke(prompt)
                content = getattr(message, "content", None)
                if isinstance(content, str) and content.strip():
                    return content.strip()
            except Exception:
                logger.warning("medical_multi_agent.synthesis_failed", exc_info=True)
        return self._followup_answer_builder(user_input, session_history, evidence_text)

    @staticmethod
    def _build_safe_followup_answer(user_input: str, evidence_text: str, safety_notes: list[str]) -> str:
        lines = [
            f"关于“{user_input}”，当前只能给出体检辅助层面的通用说明。",
            "涉及药物名称、剂量、停药、换药或加减药时，应以医生面诊和处方为准，不要自行调整。",
        ]
        if evidence_text:
            lines.append("可参考已检索到的医学资料，但应优先用于理解风险和复查方向，而不是直接形成处方。")
        if safety_notes:
            lines.append("安全复核提示：" + "；".join(dict.fromkeys(safety_notes)))
        return "\n".join(lines)

    @staticmethod
    def _build_synthesis_prompt(user_input: str, session_history: list[dict], evidence_text: str) -> str:
        context_lines = []
        for item in session_history[-8:]:
            role = item.get("role", "")
            content = str(item.get("content", ""))[:500]
            context_lines.append(f"{role}: {content}")
        return (
            "你是医疗体检辅助评估系统中的回答合成 Agent。"
            "请只基于上游 Agent 提供的会话记忆和检索证据回答用户追问。"
            "不要编造诊断，不要替代医生处方；涉及严重程度、用药、复查时提示结合医生复核。\n"
            f"用户问题：{user_input}\n"
            f"会话上下文：{json.dumps(context_lines, ensure_ascii=False)}\n"
            f"检索证据：{evidence_text or '无新增检索证据'}"
        )

    def _resolve_session_history(self, state: MedicalMultiAgentState) -> list[dict]:
        session_id = state.get("session_id")
        if self._memory_context_builder is not None and session_id:
            try:
                context = self._memory_context_builder(session_id, state["user_input"])
                history = getattr(context, "history", None)
                if isinstance(history, list):
                    return history
            except Exception:
                logger.warning("medical_multi_agent.memory_context_failed", exc_info=True)
        return state.get("session_history", [])

    @staticmethod
    def _build_followup_retrieval_query(user_input: str, memory_text: str) -> str:
        diseases = [term for term in DISEASE_TERMS if term in memory_text]
        if not diseases:
            return user_input
        suffix = " ".join(diseases[:3])
        if any(token in user_input for token in ["饮食", "怎么吃", "早餐", "晚餐"]):
            suffix += " 饮食管理 指南"
        elif any(token in user_input for token in ["复查", "随访", "科室"]):
            suffix += " 复查 随访 科室 指南"
        else:
            suffix += " 风险管理 指南"
        return f"{user_input} {suffix}"

    @staticmethod
    def _asks_personal_followup(user_input: str) -> bool:
        keywords = ["我", "上次", "之前", "指标", "血压", "血糖", "肌酐", "eGFR", "风险", "严不严重"]
        return any(keyword.lower() in user_input.lower() for keyword in keywords)

    @staticmethod
    def _requires_external_knowledge(user_input: str) -> bool:
        keywords = ["标准", "正常值", "范围", "指南", "饮食", "怎么吃", "科普", "原因", "为什么", "机制"]
        return any(keyword.lower() in user_input.lower() for keyword in keywords)

    @staticmethod
    def _mentions_medication_adjustment(user_input: str) -> bool:
        keywords = ["用药", "药量", "停药", "换药", "加药", "减药", "处方"]
        return any(keyword in user_input for keyword in keywords)

    @staticmethod
    def _contains_medication_advice(text: str) -> bool:
        keywords = ["用药", "药物", "服用", "口服", "停药", "换药", "加药", "减药", "处方", "剂量"]
        return any(keyword in text for keyword in keywords)

    @staticmethod
    def _contains_dosage_instruction(text: str) -> bool:
        return bool(re.search(r"\d+(\.\d+)?\s*(mg|g|片|粒|次/日|次每天|毫克|克)", text, flags=re.IGNORECASE))

    @staticmethod
    def _extract_suspected_drug_terms(text: str) -> list[str]:
        candidates = re.findall(r"[\u4e00-\u9fa5A-Za-z]{2,12}(?:片|胶囊|颗粒|沙坦|他汀|洛尔|地平|普利|二甲双胍)", text)
        seen: list[str] = []
        for candidate in candidates:
            if candidate not in seen:
                seen.append(candidate)
        return seen
