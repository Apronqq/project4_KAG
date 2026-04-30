from app.services.document_ingestion import DocumentChunker
from app.services.knowledge_registry import KnowledgeDocumentRegistry
from app.graph.kb_builder import MedicalKnowledgeBuilder
from app.graph.store import InMemoryGraphStore
from app.retrieval.embeddings import LightweightTextEmbedder
from app.retrieval.evidence_store import InMemoryEvidenceStore
from pathlib import Path


def test_document_chunker_supports_txt_upload():
    chunker = DocumentChunker(chunk_size=80, chunk_overlap=10)
    content = "成人高血压患者建议居家血压监测，并进行限盐和规律运动。".encode("utf-8")

    result = chunker.chunk_document("hypertension_note.txt", content)

    assert result.document.filename == "hypertension_note.txt"
    assert result.document.chunk_count >= 1
    assert result.document.linked_node_codes
    assert result.chunks


def test_document_chunker_supports_html_upload():
    chunker = DocumentChunker(chunk_size=120, chunk_overlap=10)
    html = "<html><body><h1>高血压</h1><p>成人高血压患者建议居家血压监测，并进行限盐和规律运动。</p></body></html>".encode("utf-8")

    result = chunker.chunk_document("hypertension_page.html", html)

    assert result.document.file_type == "html"
    assert result.document.chunk_count >= 1
    assert "hypertension_risk" in result.document.linked_node_codes


def test_uploaded_chunks_are_kept_after_kb_rebuild():
    graph_store = InMemoryGraphStore()
    evidence_store = InMemoryEvidenceStore(embedder=LightweightTextEmbedder(256))
    base = Path(".tmp_registry_test")
    registry = KnowledgeDocumentRegistry(
        registry_path=base / "knowledge_registry.json",
        upload_root=base / "uploads",
    )
    builder = MedicalKnowledgeBuilder(graph_store, evidence_store, registry)
    chunker = DocumentChunker(chunk_size=80, chunk_overlap=10)

    content = "成人高血压患者建议居家血压监测，并进行限盐和规律运动。".encode("utf-8")
    result = chunker.chunk_document("hypertension_note.txt", content)
    registry.upsert(result.document, result.chunks)

    build_result = builder.build_from_seed()
    chunks = evidence_store.search([], ["hypertension_risk", "hypertension"], top_k=20)
    chunk_ids = {chunk.chunk_id for chunk in chunks}

    assert build_result.evidence_seeded >= len(result.chunks)
    assert any(chunk_id.startswith(result.document.doc_id) for chunk_id in chunk_ids)

    if base.exists():
        for child in sorted(base.rglob("*"), reverse=True):
            if child.is_file():
                child.unlink()
            elif child.is_dir():
                child.rmdir()


def test_knowledge_registry_persists_and_finds_duplicate_hash():
    base = Path(".tmp_registry_dup_test")
    registry = KnowledgeDocumentRegistry(
        registry_path=base / "knowledge_registry.json",
        upload_root=base / "uploads",
    )
    chunker = DocumentChunker(chunk_size=80, chunk_overlap=10)
    content = "成人高血压患者建议居家血压监测，并进行限盐和规律运动。".encode("utf-8")
    result = chunker.chunk_document("hypertension_note.txt", content)

    registry.upsert(result.document, result.chunks, raw_content=content)
    loaded_registry = KnowledgeDocumentRegistry(
        registry_path=base / "knowledge_registry.json",
        upload_root=base / "uploads",
    )
    duplicated = loaded_registry.find_by_hash(result.document.content_hash)

    assert duplicated is not None
    assert duplicated.doc_id == result.document.doc_id

    if base.exists():
        for child in sorted(base.rglob("*"), reverse=True):
            if child.is_file():
                child.unlink()
            elif child.is_dir():
                child.rmdir()
