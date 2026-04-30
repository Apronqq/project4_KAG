from __future__ import annotations

import json
from pathlib import Path

from app.schemas.exam import EvidenceChunk, KnowledgeDocument


class KnowledgeDocumentRegistry:
    def __init__(self, registry_path: Path, upload_root: Path):
        self._registry_path = registry_path
        self._upload_root = upload_root
        self._upload_root.mkdir(parents=True, exist_ok=True)
        self._registry_path.parent.mkdir(parents=True, exist_ok=True)
        self._documents: dict[str, KnowledgeDocument] = {}
        self._chunks_by_doc_id: dict[str, list[EvidenceChunk]] = {}
        self._load()

    def _load(self) -> None:
        if not self._registry_path.exists():
            return
        try:
            payload = json.loads(self._registry_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return
        for item in payload.get("documents", []):
            document = KnowledgeDocument(**item)
            self._documents[document.doc_id] = document
        for doc_id, items in payload.get("chunks_by_doc_id", {}).items():
            self._chunks_by_doc_id[doc_id] = [EvidenceChunk(**item) for item in items]

    def _persist(self) -> None:
        payload = {
            "documents": [document.model_dump() for document in self._documents.values()],
            "chunks_by_doc_id": {
                doc_id: [chunk.model_dump() for chunk in chunks]
                for doc_id, chunks in self._chunks_by_doc_id.items()
            },
        }
        self._registry_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    def find_by_hash(self, content_hash: str) -> KnowledgeDocument | None:
        for document in self._documents.values():
            if document.content_hash == content_hash:
                return document
        return None

    def upsert(self, document: KnowledgeDocument, chunks: list[EvidenceChunk], raw_content: bytes | None = None) -> None:
        self._documents[document.doc_id] = document
        self._chunks_by_doc_id[document.doc_id] = list(chunks)
        if raw_content is not None:
            target = self._upload_root / f"{document.doc_id}.{document.file_type}"
            target.write_bytes(raw_content)
            self._documents[document.doc_id] = document.model_copy(update={"file_path": str(target)})
        self._persist()

    def list_documents(self) -> list[KnowledgeDocument]:
        return sorted(self._documents.values(), key=lambda item: item.filename)

    def list_chunks(self) -> list[EvidenceChunk]:
        out: list[EvidenceChunk] = []
        for chunks in self._chunks_by_doc_id.values():
            out.extend(chunks)
        return out
