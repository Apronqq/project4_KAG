from __future__ import annotations

from dataclasses import dataclass
import logging
from typing import Any

import requests
from langchain_community.chat_models.tongyi import ChatTongyi
from langchain_community.embeddings import DashScopeEmbeddings

from app.core.settings import Settings
from app.retrieval.embeddings import LightweightTextEmbedder

logger = logging.getLogger(__name__)


class BaseEmbeddingProvider:
    def embed(self, text: str) -> list[float]:
        raise NotImplementedError

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [self.embed(text) for text in texts]

    @property
    def backend_name(self) -> str:
        raise NotImplementedError


class LightweightEmbeddingProvider(BaseEmbeddingProvider):
    def __init__(self, dimension: int):
        self._embedder = LightweightTextEmbedder(dimension)
        self._dimension = dimension

    def embed(self, text: str) -> list[float]:
        return self._embedder.embed(text)

    @property
    def backend_name(self) -> str:
        return f"lightweight_hash_{self._dimension}"


class DashScopeEmbeddingProvider(BaseEmbeddingProvider):
    def __init__(self, model: str, api_key: str):
        self._model_name = model
        self._embedder = DashScopeEmbeddings(model=model, dashscope_api_key=api_key)

    def embed(self, text: str) -> list[float]:
        return self._embedder.embed_documents([text])[0]

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return self._embedder.embed_documents(texts)

    @property
    def backend_name(self) -> str:
        return f"dashscope:{self._model_name}"


class LLMInputExtractor:
    def __init__(self, model_name: str, api_key: str):
        self._model_name = model_name
        self._chat = ChatTongyi(model=model_name, api_key=api_key)

    @property
    def backend_name(self) -> str:
        return f"tongyi:{self._model_name}"

    def extract_structured(self, text: str, schema: type[Any]) -> Any:
        prompt = (
            "你是体检文本结构化抽取器。请从用户输入里抽取体检相关字段，"
            "包括年龄、性别、体检指标、病史、用药史、过敏史和用户问题。"
            "不要补造不存在的数据，无法确定时返回 null 或空列表。"
        )
        chain = self._chat.with_structured_output(schema)
        return chain.invoke(
            [
                {"role": "system", "content": prompt},
                {"role": "user", "content": text},
            ]
        )


@dataclass
class RerankResult:
    index: int
    score: float


class RemoteReranker:
    def __init__(self, model: str, endpoint: str, api_key: str):
        self._model = model
        self._endpoint = endpoint.rstrip("/")
        self._api_key = api_key

    @property
    def backend_name(self) -> str:
        return f"remote_rerank:{self._model}"

    def rerank(self, query_text: str, documents: list[str], top_n: int) -> list[RerankResult]:
        if not documents:
            return []
        payload = {
            "model": self._model,
            "input": {
                "query": {"text": query_text},
                "documents": [{"text": text} for text in documents],
            },
            "parameters": {
                "return_documents": False,
                "top_n": min(top_n, len(documents)),
                "fps": 1.0,
            },
        }
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self._api_key}",
        }
        response = requests.post(self._endpoint, headers=headers, json=payload, timeout=20)
        response.raise_for_status()
        body = response.json()
        rows = body.get("results") or body.get("output", {}).get("results", [])
        out: list[RerankResult] = []
        for row in rows:
            index = row.get("index")
            score = row.get("relevance_score")
            if isinstance(index, int) and score is not None:
                out.append(RerankResult(index=index, score=float(score)))
        return out


@dataclass
class ModelRuntime:
    embedding_provider: BaseEmbeddingProvider
    extractor: LLMInputExtractor | None
    reranker: RemoteReranker | None
    assistant_chat_model: ChatTongyi | None


class ModelFactory:
    def __init__(self, settings: Settings):
        self._settings = settings

    def build(self) -> ModelRuntime:
        return ModelRuntime(
            embedding_provider=self._build_embedding_provider(),
            extractor=self._build_extractor(),
            reranker=self._build_reranker(),
            assistant_chat_model=self._build_chat_model(),
        )

    def _build_embedding_provider(self) -> BaseEmbeddingProvider:
        if self._settings.dashscope_api_key and self._settings.embedding_model:
            try:
                return DashScopeEmbeddingProvider(
                    model=self._settings.embedding_model,
                    api_key=self._settings.dashscope_api_key,
                )
            except Exception:
                logger.warning(
                    "model_factory.embedding_provider_failed",
                    exc_info=True,
                    extra={"model": self._settings.embedding_model},
                )
        return LightweightEmbeddingProvider(self._settings.evidence_embedding_dim)

    def _build_extractor(self) -> LLMInputExtractor | None:
        if not self._settings.enable_llm_input_parsing or not self._settings.dashscope_api_key:
            return None
        try:
            return LLMInputExtractor(
                model_name=self._settings.model_name,
                api_key=self._settings.dashscope_api_key,
            )
        except Exception:
            logger.warning(
                "model_factory.extractor_failed",
                exc_info=True,
                extra={"model": self._settings.model_name},
            )
            return None

    def _build_reranker(self) -> RemoteReranker | None:
        if not self._settings.enable_remote_rerank:
            return None
        api_key = self._settings.rerank_api_key or self._settings.dashscope_api_key
        if not (self._settings.rerank_binding_host and self._settings.rerank_model and api_key):
            return None
        try:
            return RemoteReranker(
                model=self._settings.rerank_model,
                endpoint=self._settings.rerank_binding_host,
                api_key=api_key,
            )
        except Exception:
            logger.warning(
                "model_factory.reranker_failed",
                exc_info=True,
                extra={"model": self._settings.rerank_model},
            )
            return None

    def _build_chat_model(self) -> ChatTongyi | None:
        if not self._settings.dashscope_api_key:
            return None
        try:
            return ChatTongyi(
                model=self._settings.model_name,
                api_key=self._settings.dashscope_api_key,
                streaming=False,
            )
        except Exception:
            logger.warning(
                "model_factory.chat_model_failed",
                exc_info=True,
                extra={"model": self._settings.model_name},
            )
            return None
