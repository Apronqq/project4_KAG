# -*- coding: utf-8 -*-
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import requests
import streamlit as st


DEFAULT_API_BASE = "http://127.0.0.1:8000"
SAMPLE_TEXT_PATH = Path(__file__).resolve().parent / "fixtures" / "exam_cases" / "high_risk_hypertension.txt"

HEALTH_STATUS_LABELS = {
    "healthy": "健康",
    "subhealthy": "亚健康",
    "needs_follow_up": "需重点复查",
    "high_risk": "高风险",
}

URGENCY_LABELS = {
    "low": "低",
    "medium": "中",
    "high": "高",
    "urgent": "紧急",
}

RISK_LEVEL_LABELS = {
    "low": "低",
    "medium": "中",
    "high": "高",
}

BACKEND_LABELS = {
    "in_memory": "内存模式",
    "neo4j": "Neo4j",
    "milvus": "Milvus",
}


def _inject_gpt_style() -> None:
    st.markdown(
        """
        <style>
        .block-container {
            padding-top: 1.2rem;
            padding-bottom: 7rem;
            max-width: 1400px;
        }
        [data-testid="stSidebar"] {
            border-right: 1px solid #e5e7eb;
        }
        [data-testid="stSidebar"] .block-container {
            padding-top: 1rem;
        }
        .memory-banner {
            background: #fff8e6;
            border: 1px solid #f3d58f;
            color: #8a5b00;
            border-radius: 12px;
            padding: 0.7rem 0.9rem;
            margin-bottom: 0.8rem;
            font-size: 0.95rem;
        }
        div[data-testid="stChatInput"] {
            position: fixed;
            bottom: 0.75rem;
            left: calc(33.333% + 2rem);
            right: 1.5rem;
            background: var(--background-color);
            padding-top: 0.5rem;
            z-index: 999;
        }
        div[data-testid="stChatInput"] > div {
            border: 1px solid #d1d5db;
            border-radius: 16px;
            box-shadow: 0 8px 24px rgba(0, 0, 0, 0.08);
            padding: 0.35rem 0.5rem;
        }
        .chat-history-spacer {
            min-height: 58vh;
        }
        @media (max-width: 900px) {
            div[data-testid="stChatInput"] {
                left: 1rem;
                right: 1rem;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _label(mapping: dict[str, str], value: str | None) -> str:
    if not value:
        return "-"
    return mapping.get(value, value)


def _build_url(base_url: str, path: str) -> str:
    return f"{base_url.rstrip('/')}{path}"


def _http_get(base_url: str, path: str) -> tuple[bool, Any]:
    try:
        response = requests.get(_build_url(base_url, path), timeout=30)
        response.raise_for_status()
        return True, response.json()
    except requests.RequestException as exc:
        return False, str(exc)


def _http_post(base_url: str, path: str, payload: Any, *, as_json: bool = True, params: dict | None = None) -> tuple[bool, Any]:
    try:
        if as_json:
            response = requests.post(_build_url(base_url, path), json=payload, params=params, timeout=90)
        else:
            response = requests.post(
                _build_url(base_url, path),
                data=payload.encode("utf-8"),
                headers={"Content-Type": "text/plain; charset=utf-8"},
                params=params,
                timeout=90,
            )
        response.raise_for_status()
        return True, response.json()
    except requests.RequestException as exc:
        return False, str(exc)


def _http_delete(base_url: str, path: str) -> tuple[bool, Any]:
    try:
        response = requests.delete(_build_url(base_url, path), timeout=30)
        response.raise_for_status()
        return True, response.json()
    except requests.RequestException as exc:
        return False, str(exc)


def _http_upload_file(base_url: str, path: str, filename: str, content: bytes, mime_type: str) -> tuple[bool, Any]:
    try:
        files = {"file": (filename, content, mime_type)}
        response = requests.post(_build_url(base_url, path), files=files, timeout=120)
        response.raise_for_status()
        return True, response.json()
    except requests.RequestException as exc:
        return False, str(exc)


def _stream_agent_chat(base_url: str, text: str):
    session_id = st.session_state.get("current_session_id")
    response = requests.post(
        _build_url(base_url, "/medical/agent/chat/stream"),
        params={"session_id": session_id} if session_id else None,
        data=text.encode("utf-8"),
        headers={"Content-Type": "text/plain; charset=utf-8"},
        stream=True,
        timeout=180,
    )
    response.raise_for_status()
    for raw_line in response.iter_lines(decode_unicode=True):
        if not raw_line:
            continue
        line = raw_line.strip()
        if not line.startswith("data: "):
            continue
        yield json.loads(line[6:])


def _init_state() -> None:
    st.session_state.setdefault("api_base", DEFAULT_API_BASE)
    st.session_state.setdefault("sample_text", SAMPLE_TEXT_PATH.read_text(encoding="utf-8") if SAMPLE_TEXT_PATH.exists() else "")
    st.session_state.setdefault("current_session_id", None)
    st.session_state.setdefault("current_session_title", "新会话")
    st.session_state.setdefault("messages", [])
    st.session_state.setdefault("sessions_cache", [])


def _refresh_sessions(api_base: str) -> None:
    ok, payload = _http_get(api_base, "/medical/sessions")
    if ok:
        st.session_state["sessions_cache"] = payload.get("sessions", [])
    else:
        st.session_state["sessions_error"] = payload


def _create_session(api_base: str, title: str | None = None) -> None:
    ok, payload = _http_post(api_base, "/medical/sessions", payload={}, params={"title": title} if title else None)
    if ok:
        session = payload["session"]
        st.session_state["current_session_id"] = session["session_id"]
        st.session_state["current_session_title"] = session["title"]
        st.session_state["messages"] = []
        _refresh_sessions(api_base)
    else:
        st.session_state["sessions_error"] = payload


def _load_session(api_base: str, session_id: str) -> None:
    ok, payload = _http_get(api_base, f"/medical/sessions/{session_id}")
    if ok:
        st.session_state["current_session_id"] = session_id
        matching = next((item for item in st.session_state.get("sessions_cache", []) if item["session_id"] == session_id), None)
        st.session_state["current_session_title"] = matching["title"] if matching else session_id
        st.session_state["messages"] = [
            {"role": row["role"], "content": row["content"], "structured_result": None}
            for row in payload.get("messages", [])
        ]
    else:
        st.session_state["sessions_error"] = payload


def _delete_session(api_base: str, session_id: str) -> None:
    ok, payload = _http_delete(api_base, f"/medical/sessions/{session_id}")
    if ok:
        if st.session_state.get("current_session_id") == session_id:
            st.session_state["current_session_id"] = None
            st.session_state["current_session_title"] = "新会话"
            st.session_state["messages"] = []
        _refresh_sessions(api_base)
    else:
        st.session_state["sessions_error"] = payload


def _render_runtime_status(payload: dict[str, Any]) -> None:
    c1, c2, c3 = st.columns(3)
    c1.metric("图谱后端", _label(BACKEND_LABELS, payload["graph_backend"]))
    c2.metric("证据后端", _label(BACKEND_LABELS, payload["evidence_backend"]))
    c3.metric("Embedding 后端", payload["embedding_backend"])
    c4, c5, c6 = st.columns(3)
    c4.metric("图谱服务", "可用" if payload["graph_ready"] else "不可用")
    c5.metric("证据服务", "可用" if payload["evidence_ready"] else "不可用")
    c6.metric("知识数据", "已就绪" if payload["graph_data_ready"] and payload["evidence_data_ready"] else "未就绪")
    st.write(f"输入抽取模型：{payload.get('extractor_backend') or '未启用'}")
    st.write(f"重排模型：{payload.get('reranker_backend') or '未启用'}")


def _render_documents(documents: list[dict[str, Any]]) -> None:
    if not documents:
        st.info("当前知识库还没有录入文档。")
        return
    for document in documents:
        with st.container(border=True):
            st.markdown(f"**{document['filename']}**")
            c1, c2, c3 = st.columns(3)
            c1.metric("文件类型", document["file_type"])
            c2.metric("Chunk 数", document["chunk_count"])
            c3.metric("关联节点数", len(document.get("linked_node_codes", [])))
            st.write(f"文档 ID：{document['doc_id']}")
            st.write(f"节点编码：{'、'.join(document.get('linked_node_codes', [])) or '-'}")


def _render_jobs(jobs: list[dict[str, Any]]) -> None:
    if not jobs:
        st.info("当前没有上传任务记录。")
        return
    for job in jobs:
        with st.container(border=True):
            st.markdown(f"**{job['filename']}**")
            c1, c2, c3 = st.columns(3)
            c1.metric("任务状态", job["status"])
            c2.metric("文档 ID", job.get("document_id") or "-")
            c3.metric("哈希", job["content_hash"][:12])
            st.write(job.get("message") or "-")


def _render_structured_result(result: dict[str, Any]) -> None:
    primary = result["primary_diagnosis"]
    secondary = result["secondary_recommendations"]
    evidence = result["evidence"]

    st.subheader("诊断摘要")
    c1, c2, c3 = st.columns(3)
    c1.metric("健康状态", _label(HEALTH_STATUS_LABELS, primary["health_status"]))
    c2.metric("紧急程度", _label(URGENCY_LABELS, primary["urgency_level"]))
    c3.metric("风险数量", len(primary["potential_risks"]))

    st.subheader("潜在疾病风险")
    for risk in primary["potential_risks"]:
        with st.container(border=True):
            st.markdown(f"**{risk['risk_name']}**（{risk['disease_name']}）")
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("风险等级", _label(RISK_LEVEL_LABELS, risk["risk_level"]))
            c2.metric("最终分数", risk.get("final_score", 0.0))
            c3.metric("支持状态数", risk.get("support_count", 0))
            c4.metric("证据支持分", risk.get("evidence_support_score", 0.0))
            st.write(f"支持状态：{'、'.join(risk.get('supported_states', [])) or '-'}")

    st.subheader("二级建议")
    st.write(f"建议科室：{'、'.join(secondary.get('recommended_departments', [])) or '-'}")
    st.write(f"建议复查项目：{'、'.join(secondary.get('follow_up_tests', [])) or '-'}")
    st.write(f"生活方式干预：{'、'.join(secondary.get('lifestyle_interventions', [])) or '-'}")
    st.write(f"用药方向建议：{'、'.join(secondary.get('medication_directions', [])) or '-'}")
    st.write(f"禁忌与警示：{'、'.join(secondary.get('contraindications', [])) or '-'}")
    st.write(f"是否需要人工复核：{'是' if secondary.get('human_review_required') else '否'}")

    if secondary.get("recommendations_by_disease"):
        st.subheader("按疾病拆分的建议")
        for item in secondary["recommendations_by_disease"]:
            with st.expander(f"{item['disease_name']}（{item['risk_name']}）", expanded=False):
                st.write(f"建议科室：{'、'.join(item.get('departments', [])) or '-'}")
                st.write(f"干预建议：{'、'.join(item.get('interventions', [])) or '-'}")
                st.write(f"用药方向：{'、'.join(item.get('medication_directions', [])) or '-'}")
                st.write(f"复查项目：{'、'.join(item.get('follow_up_tests', [])) or '-'}")
                st.write(f"禁忌提示：{'、'.join(item.get('contraindications', [])) or '-'}")

    if evidence.get("chunks"):
        st.subheader("知识库证据")
        for chunk in evidence["chunks"]:
            with st.expander(f"{chunk['title']}（{chunk['chunk_id']}）", expanded=False):
                st.write(chunk["text"])
                c1, c2, c3 = st.columns(3)
                c1.metric("最终分数", chunk.get("final_score", 0.0))
                c2.metric("图谱重合度", chunk.get("graph_overlap_score", 0.0))
                c3.metric("来源权威分", chunk.get("source_authority_score", 0.0))


def _render_left_panel(api_base: str) -> None:
    st.subheader("后端配置")
    st.session_state["api_base"] = st.text_input("后端接口地址", value=api_base)
    api_base = st.session_state["api_base"]
    if st.button("刷新全部状态", use_container_width=True):
        _refresh_sessions(api_base)

    with st.container(border=True):
        st.subheader("会话管理")
        title = st.text_input("新会话标题", value="")
        c1, c2 = st.columns(2)
        if c1.button("新建会话", use_container_width=True):
            _create_session(api_base, title=title or None)
        if c2.button("刷新会话列表", use_container_width=True):
            _refresh_sessions(api_base)

        if st.session_state.get("sessions_cache"):
            for session in st.session_state["sessions_cache"]:
                with st.container(border=True):
                    st.markdown(f"**{session['title']}**")
                    st.caption(session["updated_at"])
                    b1, b2 = st.columns(2)
                    if b1.button("切换", key=f"switch_{session['session_id']}", use_container_width=True):
                        _load_session(api_base, session["session_id"])
                    if b2.button("删除", key=f"delete_{session['session_id']}", use_container_width=True):
                        _delete_session(api_base, session["session_id"])
        else:
            st.info("当前没有历史会话。")

    with st.container(border=True):
        st.subheader("知识库管理")
        upload_file = st.file_uploader(
            "上传医疗资料文件",
            type=["txt", "md", "pdf", "json", "html", "htm"],
            help="当前支持 txt / md / pdf / json / html。",
        )
        if st.button("上传到知识库", use_container_width=True, disabled=upload_file is None):
            if upload_file is not None:
                ok, result = _http_upload_file(
                    api_base,
                    "/medical/kb/upload",
                    filename=upload_file.name,
                    content=upload_file.getvalue(),
                    mime_type=upload_file.type or "application/octet-stream",
                )
                st.session_state["upload_result"] = {"ok": ok, "payload": result}
        if st.button("刷新知识库文档", use_container_width=True):
            ok, payload = _http_get(api_base, "/medical/kb/documents")
            st.session_state["document_list_result"] = {"ok": ok, "payload": payload}
        if st.button("刷新上传任务", use_container_width=True):
            ok, payload = _http_get(api_base, "/medical/kb/jobs")
            st.session_state["job_list_result"] = {"ok": ok, "payload": payload}
        if st.button("重建知识库", use_container_width=True):
            ok, payload = _http_post(api_base, "/medical/kb/rebuild", payload={})
            st.session_state["kb_result"] = {"ok": ok, "payload": payload}

        if upload_result := st.session_state.get("upload_result"):
            if upload_result["ok"]:
                st.success(upload_result["payload"]["job"]["message"])
            else:
                st.error(upload_result["payload"])

        if kb_result := st.session_state.get("kb_result"):
            if kb_result["ok"]:
                st.success("知识库重建完成。")
            else:
                st.error(kb_result["payload"])

        if document_list_result := st.session_state.get("document_list_result"):
            if document_list_result["ok"]:
                _render_documents(document_list_result["payload"].get("documents", []))
            else:
                st.error(document_list_result["payload"])

        if job_list_result := st.session_state.get("job_list_result"):
            if job_list_result["ok"]:
                _render_jobs(job_list_result["payload"].get("jobs", []))
            else:
                st.error(job_list_result["payload"])

    with st.container(border=True):
        st.subheader("系统状态")
        if st.button("检查运行时状态", use_container_width=True):
            ok, payload = _http_get(api_base, "/medical/runtime/status")
            st.session_state["runtime_result"] = {"ok": ok, "payload": payload}
        if st.button("检查健康状态", use_container_width=True):
            ok, payload = _http_get(api_base, "/health")
            st.session_state["health_result"] = {"ok": ok, "payload": payload}

        if runtime_result := st.session_state.get("runtime_result"):
            if runtime_result["ok"]:
                _render_runtime_status(runtime_result["payload"])
            else:
                st.error(runtime_result["payload"])

        if health_result := st.session_state.get("health_result"):
            if health_result["ok"]:
                st.success("健康检查接口可用。")
            else:
                st.error(health_result["payload"])


def _render_chat_area(api_base: str) -> None:
    title = st.session_state.get("current_session_title") or "新会话"
    st.subheader(f"当前会话：{title}")
    st.caption("请直接输入体检文本或继续基于历史报告追问。系统会结合会话记忆进行受控回答。")

    user_text = st.chat_input("请输入体检文本，例如：男，52岁，血压176/108 mmHg，肌酐128 umol/L，eGFR 54...")

    if not user_text:
        _render_message_history(st.session_state["messages"])
        return

    assistant_content = ""
    assistant_steps: list[str] = []
    structured_result: dict[str, Any] | None = None

    # 中文注释：先渲染本轮最新对话，再渲染历史消息，保证右侧从上到下是最新到历史。
    with st.chat_message("user"):
        st.markdown(user_text)

    with st.chat_message("assistant"):
        step_status = st.status("处理过程", expanded=True)
        answer_placeholder = st.empty()
        try:
            for event in _stream_agent_chat(api_base, user_text):
                if event["type"] == "meta":
                    st.session_state["current_session_id"] = event["session_id"]
                    _refresh_sessions(api_base)
                    current = next((item for item in st.session_state["sessions_cache"] if item["session_id"] == event["session_id"]), None)
                    if current:
                        st.session_state["current_session_title"] = current["title"]
                elif event["type"] == "step":
                    assistant_steps.append(f"- {event['label']}：{event['detail']}")
                    step_status.write(f"{event['label']}：{event['detail']}")
                elif event["type"] in {"agent_thinking", "agent_decision", "tool_call", "tool_result", "agent_synthesizing"}:
                    detail = event.get("detail") or event.get("reason") or event.get("name") or event["type"]
                    assistant_steps.append(f"- {detail}")
                    step_status.write(str(detail))
                elif event["type"] == "content":
                    assistant_content += event["content"]
                    answer_placeholder.markdown(assistant_content)
                    if assistant_content.strip():
                        step_status.update(label="处理过程已完成", state="complete", expanded=False)
                elif event["type"] == "result":
                    structured_result = event["payload"]
                elif event["type"] == "memory_notice":
                    st.session_state["messages"].append(
                        {"role": "system", "content": event["content"], "structured_result": None}
                    )
                    st.markdown(f'<div class="memory-banner">{event["content"]}</div>', unsafe_allow_html=True)
        except requests.RequestException as exc:
            st.error(str(exc))
            return

        if structured_result:
            with st.expander("查看结构化诊断依据", expanded=False):
                _render_structured_result(structured_result)

    st.session_state["messages"].append({"role": "user", "content": user_text, "structured_result": None})
    st.session_state["messages"].append(
        {
            "role": "assistant",
            "content": assistant_content,
            "structured_result": structured_result,
        }
    )
    historical_messages = st.session_state["messages"][:-2]
    _render_message_history(historical_messages)


def _render_message_history(messages: list[dict[str, Any]]) -> None:
    if not messages:
        st.markdown('<div class="chat-history-spacer"></div>', unsafe_allow_html=True)
        return
    # 中文注释：历史消息按创建时间倒序展示，最新消息在上方，越往下越早。
    for message in reversed(messages):
        if message["role"] == "system":
            st.markdown(f'<div class="memory-banner">{message["content"]}</div>', unsafe_allow_html=True)
            continue
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message["role"] == "assistant" and message.get("structured_result"):
                with st.expander("查看结构化诊断依据", expanded=False):
                    _render_structured_result(message["structured_result"])


def main() -> None:
    st.set_page_config(
        page_title="医疗 KAG Agent 工作台",
        page_icon="🩺",
        layout="wide",
        initial_sidebar_state="collapsed",
    )
    _inject_gpt_style()

    st.session_state.setdefault("api_base", DEFAULT_API_BASE)
    st.session_state.setdefault("current_session_id", None)
    st.session_state.setdefault("current_session_title", "新会话")
    st.session_state.setdefault("messages", [])
    st.session_state.setdefault("sessions_cache", [])

    st.title("医疗 KAG Agent 工作台")
    api_base = st.session_state["api_base"]
    if not st.session_state["sessions_cache"]:
        _refresh_sessions(api_base)

    left, right = st.columns([1, 2], gap="large")
    with left:
        _render_left_panel(api_base)
    with right:
        _render_chat_area(api_base)


if __name__ == "__main__":
    main()
