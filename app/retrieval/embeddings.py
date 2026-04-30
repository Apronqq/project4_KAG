from __future__ import annotations

import hashlib
import math
import re


class LightweightTextEmbedder:
    """Deterministic hashing embedder for bootstrapping Milvus search without external model dependencies."""

    def __init__(self, dimension: int = 256):
        self.dimension = dimension

    def embed(self, text: str) -> list[float]:
        vector = [0.0] * self.dimension
        for token in self._tokenize(text):
            digest = hashlib.md5(token.encode("utf-8")).hexdigest()
            idx = int(digest[:8], 16) % self.dimension
            sign = 1.0 if int(digest[8:10], 16) % 2 == 0 else -1.0
            weight = 1.0 + (len(token) / 10.0)
            vector[idx] += sign * weight

        norm = math.sqrt(sum(value * value for value in vector))
        if norm == 0:
            return vector
        return [value / norm for value in vector]

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [self.embed(text) for text in texts]

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        lowered = text.lower()
        english = re.findall(r"[a-z0-9\.\-/]+", lowered)
        chinese = re.findall(r"[\u4e00-\u9fff]+", lowered)
        tokens: list[str] = []
        for token in chinese:
            tokens.extend(token)
        tokens.extend(english)
        return tokens
