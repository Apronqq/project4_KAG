from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


def _as_bool(value: str | None, default: bool) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class Settings:
    app_env: str = os.getenv("APP_ENV", "dev")
    app_host: str = os.getenv("APP_HOST", "0.0.0.0")
    app_port: int = int(os.getenv("APP_PORT", "8000"))
    app_root: Path = Path(__file__).resolve().parents[2]
    data_root: Path = Path(os.getenv("DATA_ROOT", str(Path(__file__).resolve().parents[2] / "data")))
    knowledge_upload_root: Path = Path(os.getenv("KNOWLEDGE_UPLOAD_ROOT", str(Path(__file__).resolve().parents[2] / "data" / "knowledge_uploads")))
    knowledge_registry_path: Path = Path(os.getenv("KNOWLEDGE_REGISTRY_PATH", str(Path(__file__).resolve().parents[2] / "data" / "knowledge_registry.json")))
    database_url: str = os.getenv(
        "DATABASE_URL",
        "postgresql+psycopg2://postgres:postgres@localhost:5432/medical_agent",
    )

    model_name: str = os.getenv("MODEL", "qwen3-max")
    rerank_model: str = os.getenv("RERANK_MODEL", "qwen3-vl-rerank")
    rerank_binding_host: str = os.getenv(
        "RERANK_BINDING_HOST",
        "https://dashscope.aliyuncs.com/api/v1/services/rerank/text-rerank/text-rerank",
    )
    embedding_model: str = os.getenv("EMBEDDING_MODEL", "text-embedding-v4")
    embedding_device: str = os.getenv("EMBEDDING_DEVICE", "cpu")
    dense_embedding_dim: int = int(os.getenv("DENSE_EMBEDDING_DIM", "1024"))
    dashscope_api_key: str = os.getenv("DASHSCOPE_API_KEY", "")
    rerank_api_key: str = os.getenv("RERANK_API_KEY", os.getenv("DASHSCOPE_API_KEY", ""))
    enable_llm_input_parsing: bool = _as_bool(os.getenv("ENABLE_LLM_INPUT_PARSING"), True)
    enable_remote_rerank: bool = _as_bool(os.getenv("ENABLE_REMOTE_RERANK"), True)

    neo4j_uri: str = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    neo4j_user: str = os.getenv("NEO4J_USER", "neo4j")
    neo4j_password: str = os.getenv("NEO4J_PASSWORD", "")
    neo4j_database: str = os.getenv("NEO4J_DATABASE", "neo4j")
    use_in_memory_graph: bool = _as_bool(os.getenv("USE_IN_MEMORY_GRAPH"), False)

    milvus_uri: str = os.getenv("MILVUS_URI", "http://localhost:19530")
    milvus_collection: str = os.getenv("MILVUS_COLLECTION", "medical_evidence_chunks")
    use_in_memory_evidence: bool = _as_bool(os.getenv("USE_IN_MEMORY_EVIDENCE"), False)
    evidence_embedding_dim: int = int(os.getenv("EVIDENCE_EMBEDDING_DIM", "256"))

    top_k_evidence: int = int(os.getenv("TOP_K_EVIDENCE", "5"))
    lexical_index_backend: str = os.getenv("LEXICAL_INDEX_BACKEND", "sqlite_fts")
    lexical_index_path: Path = Path(os.getenv("LEXICAL_INDEX_PATH", str(Path(__file__).resolve().parents[2] / "data" / "evidence_fts.sqlite3")))
    mmr_candidate_limit: int = int(os.getenv("MMR_CANDIDATE_LIMIT", "15"))
    rerank_candidate_limit: int = int(os.getenv("RERANK_CANDIDATE_LIMIT", "20"))
    summary_trigger_chars: int = int(os.getenv("SUMMARY_TRIGGER_CHARS", "2000"))
    bootstrap_kb_on_startup: bool = _as_bool(os.getenv("BOOTSTRAP_KB_ON_STARTUP"), False)
    drop_milvus_collection_on_rebuild: bool = _as_bool(os.getenv("DROP_MILVUS_COLLECTION_ON_REBUILD"), False)
    chat_recent_messages_limit: int = int(os.getenv("CHAT_RECENT_MESSAGES_LIMIT", "6"))
    chat_message_char_limit: int = int(os.getenv("CHAT_MESSAGE_CHAR_LIMIT", "500"))
    chat_total_context_char_limit: int = int(os.getenv("CHAT_TOTAL_CONTEXT_CHAR_LIMIT", "3200"))

    def __post_init__(self) -> None:
        if self.evidence_embedding_dim != self.dense_embedding_dim:
            object.__setattr__(self, "evidence_embedding_dim", self.dense_embedding_dim)
