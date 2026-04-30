import os

from app.core.settings import Settings
from app.models.factory import ModelFactory


def test_settings_aligns_evidence_dimension_with_dense_dimension(monkeypatch):
    monkeypatch.setenv("DENSE_EMBEDDING_DIM", "1024")
    monkeypatch.setenv("EVIDENCE_EMBEDDING_DIM", "256")

    settings = Settings()

    assert settings.dense_embedding_dim == 1024
    assert settings.evidence_embedding_dim == 1024


def test_model_factory_builds_reranker_from_dashscope_key_when_rerank_key_missing(monkeypatch):
    monkeypatch.setenv("DASHSCOPE_API_KEY", "x" * 35)
    monkeypatch.setenv("RERANK_API_KEY", "")
    monkeypatch.setenv("ENABLE_REMOTE_RERANK", "true")
    monkeypatch.setenv("RERANK_MODEL", "qwen3-vl-rerank")
    monkeypatch.setenv(
        "RERANK_BINDING_HOST",
        "https://dashscope.aliyuncs.com/api/v1/services/rerank/text-rerank/text-rerank",
    )

    settings = Settings()
    runtime = ModelFactory(settings).build()

    assert runtime.reranker is not None
    assert runtime.reranker.backend_name == "remote_rerank:qwen3-vl-rerank"
