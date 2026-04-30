from __future__ import annotations

from contextlib import asynccontextmanager
from datetime import datetime

from fastapi import FastAPI

from app.api.routes.medical import router as medical_router
from app.services.container import get_runtime


@asynccontextmanager
async def lifespan(_: FastAPI):
    runtime = get_runtime()
    runtime.database_manager.create_tables()
    runtime.graph_store.ensure_schema()
    runtime.evidence_store.ensure_schema()
    if runtime.settings.bootstrap_kb_on_startup:
        runtime.knowledge_builder.build_from_seed()
    yield


app = FastAPI(
    title="Medical KAG-lite",
    version="0.1.0",
    description="Physical exam assessment with graph-first retrieval and evidence recall.",
    lifespan=lifespan,
)

app.include_router(medical_router)


@app.get("/health")
async def healthcheck():
    runtime = get_runtime()
    return {
        "status": "ok",
        "graph_backend": runtime.graph_store.backend_name,
        "evidence_backend": runtime.evidence_store.backend_name,
        "graph_degraded": bool(getattr(runtime.graph_store, "fallback_reason", "")),
        "evidence_degraded": bool(getattr(runtime.evidence_store, "fallback_reason", "")),
        "runtime_status_path": "/medical/runtime/status",
        "bootstrap_kb_on_startup": runtime.settings.bootstrap_kb_on_startup,
        "checked_at": datetime.utcnow().isoformat(),
    }
