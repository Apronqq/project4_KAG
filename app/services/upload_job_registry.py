from __future__ import annotations

import json
from pathlib import Path

from app.schemas.exam import KnowledgeUploadJob


class UploadJobRegistry:
    def __init__(self, path: Path):
        self._path = path
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._jobs: dict[str, KnowledgeUploadJob] = {}
        self._load()

    def _load(self) -> None:
        if not self._path.exists():
            return
        try:
            payload = json.loads(self._path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return
        for item in payload.get("jobs", []):
            job = KnowledgeUploadJob(**item)
            self._jobs[job.job_id] = job

    def _persist(self) -> None:
        payload = {"jobs": [job.model_dump() for job in self.list_jobs()]}
        self._path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    def upsert(self, job: KnowledgeUploadJob) -> None:
        self._jobs[job.job_id] = job
        self._persist()

    def get(self, job_id: str) -> KnowledgeUploadJob | None:
        return self._jobs.get(job_id)

    def list_jobs(self) -> list[KnowledgeUploadJob]:
        return sorted(self._jobs.values(), key=lambda item: item.job_id, reverse=True)
