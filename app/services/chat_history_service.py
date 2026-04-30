from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import logging
import uuid

from sqlalchemy import delete, select

from app.db.models import ChatSession, ConversationMemory, DiagnosticMemory, UserFactMemory

logger = logging.getLogger(__name__)


@dataclass
class SessionContextBundle:
    session_id: str
    history: list[dict]
    fact_conflicts: list[str] | None = None


class ChatHistoryService:
    def __init__(self, session_factory, settings, summary_llm=None):
        self._session_factory = session_factory
        self._settings = settings
        self._summary_llm = summary_llm

    def create_session(self, title: str | None = None) -> dict:
        session_id = f"session_{uuid.uuid4().hex[:12]}"
        title = (title or "新会话").strip()[:200] or "新会话"
        with self._session_factory() as db:
            session = ChatSession(session_id=session_id, title=title, summary_text="", conversation_summary="")
            db.add(session)
            db.commit()
            return self._session_to_dict(session)

    def ensure_session(self, session_id: str, title: str | None = None) -> None:
        with self._session_factory() as db:
            session = db.execute(select(ChatSession).where(ChatSession.session_id == session_id)).scalar_one_or_none()
            if session is None:
                db.add(
                    ChatSession(
                        session_id=session_id,
                        title=(title or "新会话")[:200],
                        summary_text="",
                        conversation_summary="",
                    )
                )
                db.commit()

    def list_sessions(self) -> list[dict]:
        with self._session_factory() as db:
            rows = db.execute(select(ChatSession).order_by(ChatSession.updated_at.desc())).scalars().all()
            return [self._session_to_dict(row) for row in rows]

    def delete_session(self, session_id: str) -> bool:
        with self._session_factory() as db:
            session = db.execute(select(ChatSession).where(ChatSession.session_id == session_id)).scalar_one_or_none()
            if session is None:
                return False
            db.delete(session)
            db.commit()
            return True

    def load_session_messages(self, session_id: str) -> list[dict]:
        self.ensure_session(session_id)
        with self._session_factory() as db:
            session = db.execute(select(ChatSession).where(ChatSession.session_id == session_id)).scalar_one()
            rows = (
                db.execute(
                    select(ConversationMemory)
                    .where(ConversationMemory.session_ref_id == session.id)
                    .order_by(ConversationMemory.id.asc())
                )
                .scalars()
                .all()
            )
        return [
            {
                "role": row.role,
                "content": row.content,
                "created_at": row.created_at.isoformat(),
            }
            for row in rows
        ]

    def record_user_message(self, session_id: str, content: str) -> None:
        self.ensure_session(session_id)
        with self._session_factory() as db:
            session = db.execute(select(ChatSession).where(ChatSession.session_id == session_id)).scalar_one()
            if session.title == "新会话":
                session.title = self._truncate_content(content.replace("\n", " "), 36)
            db.add(
                ConversationMemory(
                    session_ref_id=session.id,
                    role="user",
                    content=content,
                    content_summary=self._truncate_content(content, self._settings.chat_message_char_limit),
                )
            )
            session.summary_pending_chars += len(content)
            session.updated_at = datetime.utcnow()
            db.commit()

    def record_assistant_message(self, session_id: str, content: str, structured_result: dict | None = None) -> None:
        self.ensure_session(session_id)
        with self._session_factory() as db:
            session = db.execute(select(ChatSession).where(ChatSession.session_id == session_id)).scalar_one()
            db.add(
                ConversationMemory(
                    session_ref_id=session.id,
                    role="assistant",
                    content=content,
                    content_summary=self._build_conversation_memory_summary(content, structured_result),
                )
            )
            if structured_result is not None:
                self._append_diagnostic_memory(db, session.id, structured_result)
            session.summary_pending_chars += len(content)
            trigger_chars = getattr(self._settings, "summary_trigger_chars", 2000)
            allow_llm_summary = session.summary_pending_chars >= trigger_chars
            session.conversation_summary = self._build_session_summary(
                session.id,
                session.conversation_summary,
                db,
                allow_llm=allow_llm_summary,
            )
            if allow_llm_summary:
                # 中文注释：只有累计新增对话超过阈值才允许调用摘要 LLM，降低追问路径延迟和成本。
                session.summary_pending_chars = 0
            session.summary_text = session.conversation_summary
            session.updated_at = datetime.utcnow()
            db.commit()

    def upsert_user_fact_memory(self, session_id: str, normalized_exam_json) -> list[str]:
        self.ensure_session(session_id)
        with self._session_factory() as db:
            session = db.execute(select(ChatSession).where(ChatSession.session_id == session_id)).scalar_one()
            existing_rows = (
                db.execute(select(UserFactMemory).where(UserFactMemory.session_ref_id == session.id))
                .scalars()
                .all()
            )
            existing_map = {(row.fact_group, row.fact_key): row for row in existing_rows}
            db.execute(delete(UserFactMemory).where(UserFactMemory.session_ref_id == session.id))

            conflicts: list[str] = []
            facts: list[UserFactMemory] = []

            profile = normalized_exam_json.patient_profile
            if profile.sex:
                conflicts.extend(self._detect_fact_conflict(existing_map, "patient_profile", "sex", str(profile.sex)))
                facts.append(
                    UserFactMemory(
                        session_ref_id=session.id,
                        fact_group="patient_profile",
                        fact_key="sex",
                        fact_value=str(profile.sex),
                    )
                )
            if profile.age is not None:
                conflicts.extend(self._detect_fact_conflict(existing_map, "patient_profile", "age", str(profile.age)))
                facts.append(
                    UserFactMemory(
                        session_ref_id=session.id,
                        fact_group="patient_profile",
                        fact_key="age",
                        fact_value=str(profile.age),
                    )
                )

            for item in normalized_exam_json.exam_items:
                fact_key = item.code or item.name
                fact_value = "" if item.value is None else str(item.value)
                conflicts.extend(self._detect_fact_conflict(existing_map, "exam_item", fact_key, fact_value, item.unit or ""))
                facts.append(
                    UserFactMemory(
                        session_ref_id=session.id,
                        fact_group="exam_item",
                        fact_key=fact_key,
                        fact_value=fact_value,
                        fact_unit=item.unit or "",
                    )
                )

            for history in normalized_exam_json.medical_history:
                conflicts.extend(self._detect_fact_conflict(existing_map, "medical_history", history, history))
                facts.append(
                    UserFactMemory(
                        session_ref_id=session.id,
                        fact_group="medical_history",
                        fact_key=history,
                        fact_value=history,
                    )
                )
            for medication in normalized_exam_json.current_medications:
                conflicts.extend(self._detect_fact_conflict(existing_map, "current_medications", medication, medication))
                facts.append(
                    UserFactMemory(
                        session_ref_id=session.id,
                        fact_group="current_medications",
                        fact_key=medication,
                        fact_value=medication,
                    )
                )
            for allergy in normalized_exam_json.allergies:
                conflicts.extend(self._detect_fact_conflict(existing_map, "allergies", allergy, allergy))
                facts.append(
                    UserFactMemory(
                        session_ref_id=session.id,
                        fact_group="allergies",
                        fact_key=allergy,
                        fact_value=allergy,
                    )
                )

            for fact in facts:
                db.add(fact)

            if conflicts:
                db.add(
                    ConversationMemory(
                        session_ref_id=session.id,
                        role="system",
                        content="；".join(conflicts),
                        content_summary="；".join(conflicts),
                    )
                )

            session.updated_at = datetime.utcnow()
            db.commit()
            return conflicts

    def build_context(self, session_id: str, user_input: str) -> SessionContextBundle:
        self.ensure_session(session_id)
        with self._session_factory() as db:
            session = db.execute(select(ChatSession).where(ChatSession.session_id == session_id)).scalar_one()
            recent_messages = (
                db.execute(
                    select(ConversationMemory)
                    .where(ConversationMemory.session_ref_id == session.id)
                    .order_by(ConversationMemory.id.desc())
                    .limit(self._settings.chat_recent_messages_limit * 2)
                )
                .scalars()
                .all()
            )
            fact_rows = (
                db.execute(
                    select(UserFactMemory)
                    .where(UserFactMemory.session_ref_id == session.id)
                    .order_by(UserFactMemory.id.asc())
                )
                .scalars()
                .all()
            )
            diagnostics = (
                db.execute(
                    select(DiagnosticMemory)
                    .where(DiagnosticMemory.session_ref_id == session.id)
                    .order_by(DiagnosticMemory.version_no.desc())
                    .limit(2)
                )
                .scalars()
                .all()
            )

        history: list[dict] = [
            {
                "role": "system",
                "content": (
                    "上下文使用规则：历史内容仅作参考。"
                    "如果当前用户提供了新的体检指标、化验数值或修正了事实，必须优先相信当前用户输入与最新工具结果。"
                    "不要把过去助手的推断当作不可变事实。"
                ),
            }
        ]

        if fact_rows and not self._looks_like_initial_assessment(user_input):
            history.append(
                {
                    "role": "system",
                    "content": f"用户事实记忆（确定性事实，仅参考）：{self._summarize_fact_memories(fact_rows)}",
                }
            )

        if diagnostics and not self._looks_like_initial_assessment(user_input):
            diagnostic = diagnostics[0]
            history.append(
                {
                    "role": "system",
                    "content": f"结构化诊断记忆（上轮确定性评估结果）：{self._summarize_diagnostic_memory(diagnostic)}",
                }
            )
            if len(diagnostics) >= 2:
                history.append(
                    {
                        "role": "system",
                        "content": self._summarize_diagnostic_trend(diagnostics[0], diagnostics[1]),
                    }
                )

        if session.conversation_summary:
            history.append(
                {
                    "role": "system",
                    "content": f"对话摘要记忆（仅参考）：{session.conversation_summary}",
                }
            )

        for message in reversed(recent_messages):
            content = self._truncate_content(message.content_summary or message.content, self._settings.chat_message_char_limit)
            history.append({"role": message.role, "content": content})

        history = self._clip_total_history(history, self._settings.chat_total_context_char_limit)
        return SessionContextBundle(session_id=session_id, history=history, fact_conflicts=None)

    def _append_diagnostic_memory(self, db, session_ref_id: int, structured_result: dict) -> None:
        existing = (
            db.execute(
                select(DiagnosticMemory)
                .where(DiagnosticMemory.session_ref_id == session_ref_id)
                .where(DiagnosticMemory.is_current == "true")
                .order_by(DiagnosticMemory.version_no.desc())
                .limit(1)
            )
            .scalar_one_or_none()
        )
        if existing is not None:
            existing.is_current = "false"
            version_no = existing.version_no + 1
        else:
            version_no = 1

        primary = structured_result.get("primary_diagnosis", {})
        secondary = structured_result.get("secondary_recommendations", {})
        risks = primary.get("potential_risks", [])
        abnormal = primary.get("key_abnormal_indicators", [])
        evidence_chunks = structured_result.get("evidence", {}).get("chunks", [])
        db.add(
            DiagnosticMemory(
                session_ref_id=session_ref_id,
                version_no=version_no,
                is_current="true",
                health_status=primary.get("health_status", ""),
                urgency_level=primary.get("urgency_level", ""),
                risk_summary="；".join(
                    f"{item.get('risk_name')}({item.get('disease_name')},{item.get('risk_level')})"
                    for item in risks[:5]
                ),
                abnormal_indicator_summary="；".join(
                    f"{item.get('indicator_name')}={item.get('value')}{item.get('unit') or ''}"
                    for item in abnormal[:8]
                ),
                department_summary="、".join(secondary.get("recommended_departments", [])),
                follow_up_summary="、".join(secondary.get("follow_up_tests", [])),
                lifestyle_summary="、".join(secondary.get("lifestyle_interventions", [])),
                medication_summary="、".join(secondary.get("medication_directions", [])),
                contraindication_summary="、".join(secondary.get("contraindications", [])),
                evidence_summary="；".join(chunk.get("title", "") for chunk in evidence_chunks[:5]),
            )
        )

    def _build_session_summary(self, session_ref_id: int, current_summary: str, db, allow_llm: bool = True) -> str:
        recent_rows = (
            db.execute(
                select(ConversationMemory)
                .where(ConversationMemory.session_ref_id == session_ref_id)
                .order_by(ConversationMemory.id.desc())
                .limit(self._settings.chat_recent_messages_limit)
            )
            .scalars()
            .all()
        )
        snippets = [row.content_summary or row.content for row in reversed(recent_rows)]
        deterministic_summary = self._truncate_content(" | ".join(snippets), 1200)
        if self._summary_llm is None or not allow_llm:
            return deterministic_summary

        try:
            prompt = (
                "你是医疗会话摘要器。请把以下对话压缩为一个简洁摘要，只保留稳定事实、主要结论和待跟进事项。"
                "不要新增任何没有明确出现的事实，不确定的内容不要写入。"
                f"\n已有摘要：{current_summary}\n近期对话：{deterministic_summary}"
            )
            message = self._summary_llm.invoke(prompt)
            content = getattr(message, "content", None)
            if isinstance(content, str) and content.strip():
                return self._truncate_content(content.strip(), 1200)
        except Exception:
            logger.warning("chat_history.summary_llm_failed", exc_info=True)
        return deterministic_summary

    @staticmethod
    def _build_conversation_memory_summary(content: str, structured_result: dict | None) -> str:
        if structured_result:
            primary = structured_result.get("primary_diagnosis", {})
            secondary = structured_result.get("secondary_recommendations", {})
            risks = primary.get("potential_risks", [])
            risk_summary = "；".join(
                f"{item.get('risk_name')}({item.get('risk_level')})"
                for item in risks[:3]
            )
            return (
                f"健康状态={primary.get('health_status')}；"
                f"紧急程度={primary.get('urgency_level')}；"
                f"主要风险={risk_summary or '无'}；"
                f"建议科室={','.join(secondary.get('recommended_departments', [])) or '无'}"
            )
        return content[:200]

    @staticmethod
    def _summarize_fact_memories(facts: list[UserFactMemory]) -> str:
        profile = []
        exam_items = []
        histories = []
        medications = []
        allergies = []
        for fact in facts:
            if fact.fact_group == "patient_profile":
                profile.append(f"{fact.fact_key}={fact.fact_value}")
            elif fact.fact_group == "exam_item":
                exam_items.append(f"{fact.fact_key}={fact.fact_value}{fact.fact_unit}")
            elif fact.fact_group == "medical_history":
                histories.append(fact.fact_value)
            elif fact.fact_group == "current_medications":
                medications.append(fact.fact_value)
            elif fact.fact_group == "allergies":
                allergies.append(fact.fact_value)

        parts = []
        if profile:
            parts.append("基本信息：" + "，".join(profile))
        if exam_items:
            parts.append("关键指标：" + "；".join(exam_items[:10]))
        if histories:
            parts.append("病史：" + "、".join(histories[:10]))
        if medications:
            parts.append("用药史：" + "、".join(medications[:10]))
        if allergies:
            parts.append("过敏史：" + "、".join(allergies[:10]))
        return " | ".join(parts)

    @staticmethod
    def _summarize_diagnostic_memory(memory: DiagnosticMemory) -> str:
        return (
            f"健康状态={memory.health_status}；"
            f"紧急程度={memory.urgency_level}；"
            f"主要风险={memory.risk_summary or '无'}；"
            f"异常指标={memory.abnormal_indicator_summary or '无'}；"
            f"建议科室={memory.department_summary or '无'}；"
            f"复查项目={memory.follow_up_summary or '无'}"
        )

    @staticmethod
    def _summarize_diagnostic_trend(current: DiagnosticMemory, previous: DiagnosticMemory) -> str:
        return (
            "诊断趋势记忆（最近两次确定性评估对比）："
            f"最新v{current.version_no} 健康状态={current.health_status}，紧急程度={current.urgency_level}，"
            f"异常指标={current.abnormal_indicator_summary or '无'}；"
            f"上次v{previous.version_no} 健康状态={previous.health_status}，紧急程度={previous.urgency_level}，"
            f"异常指标={previous.abnormal_indicator_summary or '无'}"
        )

    @staticmethod
    def _detect_fact_conflict(existing_map, fact_group: str, fact_key: str, fact_value: str, fact_unit: str = "") -> list[str]:
        row = existing_map.get((fact_group, fact_key))
        if row is None:
            return []
        old = f"{row.fact_value}{row.fact_unit or ''}"
        new = f"{fact_value}{fact_unit or ''}"
        if old == new:
            return []
        label = fact_key
        return [f"已更新事实：{label} 从 {old} 更新为 {new}"]

    @staticmethod
    def _truncate_content(content: str, limit: int) -> str:
        if len(content) <= limit:
            return content
        return content[: limit - 3] + "..."

    def _clip_total_history(self, history: list[dict], total_limit: int) -> list[dict]:
        clipped: list[dict] = []
        total = 0
        for item in reversed(history):
            content = item.get("content", "")
            if total + len(content) > total_limit and clipped:
                continue
            clipped.append(item)
            total += len(content)
        return list(reversed(clipped))

    def _session_to_dict(self, session: ChatSession) -> dict:
        return {
            "session_id": session.session_id,
            "title": session.title,
            "summary_text": session.conversation_summary,
            "created_at": session.created_at.isoformat(),
            "updated_at": session.updated_at.isoformat(),
        }

    @staticmethod
    def _looks_like_initial_assessment(user_input: str) -> bool:
        keywords = ["血压", "mmHg", "eGFR", "肌酐", "空腹血糖", "HbA1c", "体检", "指标"]
        return any(keyword.lower() in user_input.lower() for keyword in keywords)
