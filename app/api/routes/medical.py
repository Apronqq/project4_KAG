from __future__ import annotations

import hashlib
from datetime import datetime
import uuid
import json
import logging
import time

from fastapi import APIRouter, BackgroundTasks, File, HTTPException, Query, Request, UploadFile
from fastapi.responses import StreamingResponse

from app.schemas.exam import (
    KnowledgeBuildResponse,
    KnowledgeDocumentListResponse,
    KnowledgeUploadJob,
    KnowledgeUploadJobListResponse,
    KnowledgeUploadJobResponse,
    MedicalAssessmentResponse,
    MedicalParseResponse,
    RuntimeStatusResponse,
    SessionCreateResponse,
    SessionInfo,
    SessionListResponse,
    SessionMessagesResponse,
    SessionMessage,
)
from app.services.container import get_runtime

router = APIRouter(prefix="/medical", tags=["medical"])
logger = logging.getLogger(__name__)


async def _extract_payload(request: Request):
    body = await request.body()
    if not body:
        raise HTTPException(status_code=400, detail="Request body is required.")
    text = body.decode("utf-8").strip()
    content_type = request.headers.get("content-type", "")
    if "application/json" in content_type:
        try:
            payload = json.loads(text)
            if isinstance(payload, dict) and "input_data" in payload:
                return payload["input_data"]
            return payload
        except json.JSONDecodeError:
            return text
    return text


@router.post("/exam/parse", response_model=MedicalParseResponse)
async def parse_exam_input(request: Request):
    payload = await _extract_payload(request)
    runtime = get_runtime()
    return runtime.medical_workflow.parse_only(payload)


@router.post("/exam/assess", response_model=MedicalAssessmentResponse)
async def assess_exam_input(request: Request):
    payload = await _extract_payload(request)
    runtime = get_runtime()
    return await runtime.medical_workflow.run_async(payload)


@router.post("/agent/chat")
async def agent_chat(request: Request, session_id: str | None = Query(None)):
    payload = await _extract_payload(request)
    if not isinstance(payload, str):
        raise HTTPException(status_code=400, detail="Agent chat only accepts text input.")
    runtime = get_runtime()
    session_id = session_id or f"session_{uuid.uuid4().hex[:12]}"
    is_initial = runtime.medical_agent.looks_like_initial_assessment(payload)
    if is_initial:
        answer, structured = runtime.medical_agent.assess(payload)
        runtime.chat_history_service.record_user_message(session_id, payload)
        conflicts = runtime.chat_history_service.upsert_user_fact_memory(session_id, structured.normalized_exam_json)
        if conflicts:
            answer = "\n".join(conflicts) + "\n" + answer
        runtime.chat_history_service.record_assistant_message(session_id, answer, structured.model_dump())
        return {
            "session_id": session_id,
            "answer": answer,
            "structured_result": structured.model_dump(),
            "timestamp": datetime.utcnow().isoformat(),
        }

    context = runtime.chat_history_service.build_context(session_id, payload)
    answer = runtime.medical_agent.chat_assess(payload, context.history)
    runtime.chat_history_service.record_user_message(session_id, payload)
    runtime.chat_history_service.record_assistant_message(session_id, answer)
    return {"session_id": session_id, "answer": answer, "timestamp": datetime.utcnow().isoformat()}


@router.post("/agent/chat/stream")
async def agent_chat_stream(request: Request, session_id: str | None = Query(None)):
    payload = await _extract_payload(request)
    if not isinstance(payload, str):
        raise HTTPException(status_code=400, detail="Agent chat streaming only accepts text input.")
    runtime = get_runtime()
    session_id = session_id or f"session_{uuid.uuid4().hex[:12]}"
    context = runtime.chat_history_service.build_context(session_id, payload)

    async def event_generator():
        runtime.chat_history_service.record_user_message(session_id, payload)
        assistant_content = ""
        structured_result = None
        normalized_exam_json = None
        done_event = None
        yield f"data: {json.dumps({'type': 'meta', 'session_id': session_id}, ensure_ascii=False)}\n\n"
        async for event in runtime.medical_agent.stream_assess_async(payload, session_history=context.history):
            if event["type"] == "content":
                assistant_content += event["content"]
            elif event["type"] == "result":
                structured_result = event["payload"]
                normalized_exam_json = structured_result.get("normalized_exam_json")
            elif event["type"] == "done":
                done_event = event
                continue
            yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
        if assistant_content:
            if normalized_exam_json is not None:
                from app.schemas.exam import NormalizedMedicalExamJSON

                conflicts = runtime.chat_history_service.upsert_user_fact_memory(
                    session_id,
                    NormalizedMedicalExamJSON(**normalized_exam_json),
                )
                if conflicts:
                    yield f"data: {json.dumps({'type': 'memory_notice', 'content': '；'.join(conflicts)}, ensure_ascii=False)}\n\n"
                    yield f"data: {json.dumps({'type': 'step', 'label': '事实记忆更新', 'detail': '；'.join(conflicts)}, ensure_ascii=False)}\n\n"
                    assistant_content = "\n".join(conflicts) + "\n" + assistant_content
            runtime.chat_history_service.record_assistant_message(session_id, assistant_content, structured_result)
        yield f"data: {json.dumps(done_event or {'type': 'done'}, ensure_ascii=False)}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache, no-store, must-revalidate",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@router.post("/kb/rebuild", response_model=KnowledgeBuildResponse)
async def rebuild_medical_knowledge_base():
    runtime = get_runtime()
    result = runtime.knowledge_builder.build_from_seed()
    return result.__dict__


@router.get("/runtime/status", response_model=RuntimeStatusResponse)
async def get_runtime_status():
    runtime = get_runtime()
    components = {
        "graph": _component_status(runtime.graph_store.ping),
        "evidence": _component_status(runtime.evidence_store.ping),
        "postgresql": _component_status(runtime.database_manager.ping),
        "embedding": {"status": "configured", "latency_ms": 0.0, "backend": runtime.model_runtime.embedding_provider.backend_name},
        "extractor": {
            "status": "configured" if runtime.model_runtime.extractor is not None else "disabled",
            "latency_ms": 0.0,
            "backend": getattr(runtime.model_runtime.extractor, "backend_name", None),
        },
        "reranker": {
            "status": "configured" if runtime.model_runtime.reranker is not None else "disabled",
            "latency_ms": 0.0,
            "backend": getattr(runtime.model_runtime.reranker, "backend_name", None),
        },
    }
    return RuntimeStatusResponse(
        graph_backend=runtime.graph_store.backend_name,
        evidence_backend=runtime.evidence_store.backend_name,
        graph_ready=components["graph"]["status"] == "healthy",
        evidence_ready=components["evidence"]["status"] == "healthy",
        graph_data_ready=runtime.graph_store.data_ready(),
        evidence_data_ready=runtime.evidence_store.data_ready(),
        graph_mode=runtime.graph_store.mode,
        evidence_mode=runtime.evidence_store.mode,
        embedding_backend=runtime.model_runtime.embedding_provider.backend_name,
        extractor_backend=getattr(runtime.model_runtime.extractor, "backend_name", None),
        reranker_backend=getattr(runtime.model_runtime.reranker, "backend_name", None),
        graph_degraded=bool(getattr(runtime.graph_store, "fallback_reason", "")),
        evidence_degraded=bool(getattr(runtime.evidence_store, "fallback_reason", "")),
        graph_fallback_reason=getattr(runtime.graph_store, "fallback_reason", ""),
        evidence_fallback_reason=getattr(runtime.evidence_store, "fallback_reason", ""),
        checked_at=datetime.utcnow().isoformat(),
        components=components,
    )


def _component_status(ping_fn) -> dict:
    started_at = time.perf_counter()
    try:
        ok = bool(ping_fn())
        latency_ms = round((time.perf_counter() - started_at) * 1000, 2)
        return {"status": "healthy" if ok else "unhealthy", "latency_ms": latency_ms}
    except Exception as exc:
        latency_ms = round((time.perf_counter() - started_at) * 1000, 2)
        logger.warning("runtime_status.component_check_failed", exc_info=True)
        return {"status": "error", "latency_ms": latency_ms, "error": str(exc)}


@router.get("/kb/documents", response_model=KnowledgeDocumentListResponse)
async def list_knowledge_documents():
    runtime = get_runtime()
    return KnowledgeDocumentListResponse(documents=runtime.knowledge_registry.list_documents())


@router.get("/sessions", response_model=SessionListResponse)
async def list_chat_sessions():
    runtime = get_runtime()
    sessions = [SessionInfo(**item) for item in runtime.chat_history_service.list_sessions()]
    return SessionListResponse(sessions=sessions)


@router.post("/sessions", response_model=SessionCreateResponse)
async def create_chat_session(title: str | None = Query(None)):
    runtime = get_runtime()
    session = SessionInfo(**runtime.chat_history_service.create_session(title=title))
    return SessionCreateResponse(session=session)


@router.get("/sessions/{session_id}", response_model=SessionMessagesResponse)
async def get_chat_session_messages(session_id: str):
    runtime = get_runtime()
    messages = [SessionMessage(**item) for item in runtime.chat_history_service.load_session_messages(session_id)]
    return SessionMessagesResponse(session_id=session_id, messages=messages)


@router.delete("/sessions/{session_id}")
async def delete_chat_session(session_id: str):
    runtime = get_runtime()
    deleted = runtime.chat_history_service.delete_session(session_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Session not found.")
    return {"session_id": session_id, "deleted": True}


def _process_upload_job(job_id: str, filename: str, content: bytes) -> None:
    runtime = get_runtime()
    job = runtime.upload_job_registry.get(job_id)
    if job is None:
        return
    runtime.upload_job_registry.upsert(job.model_copy(update={"status": "processing", "message": "正在解析并入库"}))
    try:
        result = runtime.document_chunker.chunk_document(filename=filename, content=content)
        runtime.knowledge_registry.upsert(result.document, result.chunks, raw_content=content)
        if not runtime.graph_store.data_ready() or not runtime.evidence_store.data_ready():
            runtime.knowledge_builder.build_from_seed()
        else:
            runtime.evidence_store.add_chunks(result.chunks)
            runtime.graph_store.add_evidence_chunks(result.chunks)
        runtime.upload_job_registry.upsert(
            job.model_copy(
                update={
                    "status": "completed",
                    "message": f"文档 {filename} 已完成入库，共 {result.document.chunk_count} 个 chunks。",
                    "document_id": result.document.doc_id,
                }
            )
        )
    except Exception as exc:
        logger.exception(
            "knowledge_upload_job.failed",
            extra={"job_id": job_id, "filename": filename},
        )
        runtime.upload_job_registry.upsert(
            job.model_copy(update={"status": "failed", "message": str(exc)})
        )


@router.post("/kb/upload", response_model=KnowledgeUploadJobResponse)
async def upload_knowledge_document(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    runtime = get_runtime()
    filename = file.filename or ""
    if not filename:
        raise HTTPException(status_code=400, detail="File name is required.")

    supported_suffixes = {".txt", ".md", ".pdf", ".json", ".html", ".htm"}
    suffix = filename.lower().rsplit(".", 1)
    suffix = f".{suffix[-1]}" if len(suffix) > 1 else ""
    if suffix not in supported_suffixes:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {suffix}. Supported types: {', '.join(sorted(supported_suffixes))}",
        )

    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    content_hash = hashlib.sha256(content).hexdigest()
    existing = runtime.knowledge_registry.find_by_hash(content_hash)
    if existing is not None:
        job = KnowledgeUploadJob(
            job_id=f"dup_{content_hash[:12]}",
            filename=filename,
            content_hash=content_hash,
            status="duplicate",
            message=f"检测到重复文件，已保留现有文档 {existing.filename}，跳过重复上传。",
            document_id=existing.doc_id,
        )
        runtime.upload_job_registry.upsert(job)
        return KnowledgeUploadJobResponse(job=job)

    job = KnowledgeUploadJob(
        job_id=f"job_{uuid.uuid4().hex[:12]}",
        filename=filename,
        content_hash=content_hash,
        status="pending",
        message="文件已接收，等待后台处理。",
    )
    runtime.upload_job_registry.upsert(job)
    background_tasks.add_task(_process_upload_job, job.job_id, filename, content)
    return KnowledgeUploadJobResponse(job=job)


@router.get("/kb/jobs", response_model=KnowledgeUploadJobListResponse)
async def list_knowledge_upload_jobs():
    runtime = get_runtime()
    return KnowledgeUploadJobListResponse(jobs=runtime.upload_job_registry.list_jobs())
