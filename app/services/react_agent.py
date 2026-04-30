from __future__ import annotations

import hashlib
import json
import logging
import time
from dataclasses import dataclass
from typing import Callable, Generator

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class AgentDecision:
    action: str
    tool_name: str | None = None
    tool_args: dict | None = None
    reason: str = ""


@dataclass
class AgentRunResult:
    answer: str
    events: list[dict]


class MedicalReActAgent:
    MAX_ITERATIONS = 5

    def __init__(
        self,
        *,
        chat_model,
        knowledge_tool,
        answer_builder: Callable[[str, list[dict], str], str],
    ):
        self._chat_model = chat_model
        self._knowledge_tool = knowledge_tool
        self._answer_builder = answer_builder

    def run(self, user_input: str, session_history: list[dict]) -> AgentRunResult:
        events: list[dict] = []
        answer = ""
        for event in self.iter_events(user_input, session_history):
            events.append(event)
            if event.get("type") == "final_answer":
                answer = str(event.get("content", ""))
        return AgentRunResult(answer=answer, events=events)

    def iter_events(self, user_input: str, session_history: list[dict]) -> Generator[dict, None, None]:
        called_tools: dict[str, str] = {}
        observations: list[str] = []

        for iteration in range(1, self.MAX_ITERATIONS + 1):
            yield {
                "type": "agent_thinking",
                "iteration": iteration,
                "detail": "正在根据对话记忆和问题类型选择下一步动作",
            }
            decision = self._decide_next_action(user_input, session_history, observations)
            yield {
                "type": "agent_decision",
                "iteration": iteration,
                "action": decision.action,
                "tool_name": decision.tool_name,
                "reason": decision.reason,
            }

            if decision.action == "final_answer":
                answer = self._compose_answer(user_input, session_history, observations)
                yield {"type": "final_answer", "content": answer}
                return

            if decision.tool_name is None:
                answer = self._compose_answer(user_input, session_history, observations)
                yield {"type": "final_answer", "content": answer}
                return

            args = decision.tool_args or {}
            args_hash = self._args_hash(args)
            if called_tools.get(decision.tool_name) == args_hash:
                warning = f"检测到重复调用 {decision.tool_name}，已停止工具循环并基于已有信息作答。"
                yield {"type": "agent_warning", "detail": warning}
                answer = self._compose_answer(user_input, session_history, observations)
                yield {"type": "final_answer", "content": answer}
                return
            called_tools[decision.tool_name] = args_hash

            started_at = time.perf_counter()
            yield {
                "type": "tool_call",
                "iteration": iteration,
                "name": decision.tool_name,
                "args_summary": self._summarize_args(args),
            }
            observation = self._execute_tool(decision.tool_name, args)
            observations.append(observation)
            duration_ms = round((time.perf_counter() - started_at) * 1000, 2)
            logger.info(
                "medical_react_agent.tool_completed",
                extra={
                    "tool_name": decision.tool_name,
                    "iteration": iteration,
                    "duration_ms": duration_ms,
                    "result_len": len(observation),
                },
            )
            yield {
                "type": "tool_result",
                "iteration": iteration,
                "name": decision.tool_name,
                "result_len": len(observation),
                "duration_ms": duration_ms,
            }

        yield {"type": "agent_warning", "detail": f"达到最大迭代次数 {self.MAX_ITERATIONS}，已基于已有信息作答。"}
        answer = self._compose_answer(user_input, session_history, observations)
        yield {"type": "final_answer", "content": answer}

    def _decide_next_action(
        self,
        user_input: str,
        session_history: list[dict],
        observations: list[str],
    ) -> AgentDecision:
        if observations:
            return AgentDecision(
                action="final_answer",
                reason="已有工具观察结果，进入最终回答阶段",
            )
        if self._has_relevant_memory(user_input, session_history) and not self._requires_external_knowledge(user_input):
            return AgentDecision(
                action="final_answer",
                reason="会话上下文已包含诊断或事实记忆，优先使用既有记忆",
            )
        return AgentDecision(
            action="tool_call",
            tool_name="lookup_medical_knowledge",
            tool_args={"query": user_input},
            reason="需要补充医学知识库证据",
        )

    def _execute_tool(self, tool_name: str, args: dict) -> str:
        if tool_name == "lookup_medical_knowledge":
            return self._knowledge_tool._run(str(args.get("query", "")))
        raise ValueError(f"Unsupported agent tool: {tool_name}")

    def _compose_answer(self, user_input: str, session_history: list[dict], observations: list[str]) -> str:
        evidence_text = "\n\n".join(observations).strip()
        if self._chat_model is not None:
            try:
                prompt = self._build_answer_prompt(user_input, session_history, evidence_text)
                message = self._chat_model.invoke(prompt)
                content = getattr(message, "content", None)
                if isinstance(content, str) and content.strip():
                    return content.strip()
            except Exception:
                logger.warning("medical_react_agent.llm_answer_failed", exc_info=True)
        return self._answer_builder(user_input, session_history, evidence_text)

    @staticmethod
    def _build_answer_prompt(user_input: str, session_history: list[dict], evidence_text: str) -> str:
        context_lines = []
        for item in session_history[-8:]:
            role = item.get("role", "")
            content = str(item.get("content", ""))[:500]
            context_lines.append(f"{role}: {content}")
        return (
            "你是医疗健康助手。请基于会话上下文和检索证据，用中文回答用户追问。"
            "不要捏造诊断，不要替代医生处方；涉及严重程度、用药、复查时要提示结合医生复核。\n"
            f"用户问题：{user_input}\n"
            f"会话上下文：{json.dumps(context_lines, ensure_ascii=False)}\n"
            f"检索证据：{evidence_text or '无新增检索证据'}"
        )

    @staticmethod
    def _has_relevant_memory(user_input: str, session_history: list[dict]) -> bool:
        memory_keywords = ["诊断记忆", "用户事实记忆", "异常指标", "健康状态", "主要风险", "复查项目"]
        user_keywords = ["我", "上次", "之前", "指标", "血压", "血糖", "肌酐", "eGFR", "风险", "严不严重"]
        has_memory = any(any(keyword in str(item.get("content", "")) for keyword in memory_keywords) for item in session_history)
        asks_personal = any(keyword.lower() in user_input.lower() for keyword in user_keywords)
        return has_memory and asks_personal

    @staticmethod
    def _requires_external_knowledge(user_input: str) -> bool:
        keywords = ["标准", "正常值", "范围", "指南", "饮食", "怎么吃", "科普", "原因", "为什么", "机制"]
        return any(keyword.lower() in user_input.lower() for keyword in keywords)

    @staticmethod
    def _args_hash(args: dict) -> str:
        payload = json.dumps(args, sort_keys=True, ensure_ascii=False)
        return hashlib.md5(payload.encode("utf-8")).hexdigest()

    @staticmethod
    def _summarize_args(args: dict) -> str:
        text = json.dumps(args, ensure_ascii=False)
        return text[:160]
