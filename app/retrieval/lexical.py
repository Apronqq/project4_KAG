from __future__ import annotations

import math
import re
import sqlite3
from collections import Counter
from pathlib import Path

from app.schemas.exam import EvidenceChunk


class BM25Lite:
    def __init__(self, texts: list[str], k1: float = 1.5, b: float = 0.75):
        self._k1 = k1
        self._b = b
        self._texts = texts
        self._tokens = [self.tokenize(text) for text in texts]
        self._doc_freq: Counter[str] = Counter()
        self._doc_lengths = [len(tokens) for tokens in self._tokens]
        self._avg_doc_length = sum(self._doc_lengths) / max(len(self._doc_lengths), 1)
        for tokens in self._tokens:
            self._doc_freq.update(set(tokens))

    @staticmethod
    def tokenize(text: str) -> list[str]:
        lowered = text.lower()
        english = re.findall(r"[a-z0-9\.\-/]+", lowered)
        chinese = re.findall(r"[\u4e00-\u9fff]+", lowered)
        tokens: list[str] = []
        for token in chinese:
            tokens.extend(token)
        tokens.extend(english)
        return tokens

    def score(self, query_text: str, doc_index: int) -> float:
        query_tokens = self.tokenize(query_text)
        if not query_tokens or doc_index >= len(self._tokens):
            return 0.0
        doc_tokens = self._tokens[doc_index]
        if not doc_tokens:
            return 0.0

        doc_len = len(doc_tokens)
        term_freq = Counter(doc_tokens)
        total_docs = max(len(self._tokens), 1)
        avg_len = max(self._avg_doc_length, 1.0)
        score = 0.0

        for token in query_tokens:
            tf = term_freq.get(token, 0)
            if tf == 0:
                continue
            df = self._doc_freq.get(token, 0)
            idf = math.log((total_docs - df + 0.5) / (df + 0.5) + 1)
            numerator = tf * (self._k1 + 1)
            denominator = tf + self._k1 * (1 - self._b + self._b * doc_len / avg_len)
            score += idf * numerator / denominator
        return score


class SQLiteFTSIndex:
    def __init__(self, db_path: str | Path):
        self._db_path = Path(db_path)
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self._db_path), check_same_thread=False)
        self._conn.execute(
            "CREATE VIRTUAL TABLE IF NOT EXISTS docs_fts "
            "USING fts5(chunk_id UNINDEXED, title, text, linked_codes)"
        )
        self._conn.commit()

    def rebuild(self, chunks: list[EvidenceChunk]) -> None:
        # 中文注释：FTS5 索引随证据库重建而重建，避免 Python 进程内全量 BM25 扫描。
        with self._conn:
            self._conn.execute("DELETE FROM docs_fts")
            self._conn.executemany(
                "INSERT INTO docs_fts(chunk_id, title, text, linked_codes) VALUES (?, ?, ?, ?)",
                [
                    (
                        chunk.chunk_id,
                        chunk.title,
                        chunk.text,
                        " ".join(chunk.linked_node_codes),
                    )
                    for chunk in chunks
                ],
            )

    def add(self, chunks: list[EvidenceChunk]) -> None:
        if not chunks:
            return
        with self._conn:
            self._conn.executemany(
                "DELETE FROM docs_fts WHERE chunk_id = ?",
                [(chunk.chunk_id,) for chunk in chunks],
            )
            self._conn.executemany(
                "INSERT INTO docs_fts(chunk_id, title, text, linked_codes) VALUES (?, ?, ?, ?)",
                [
                    (
                        chunk.chunk_id,
                        chunk.title,
                        chunk.text,
                        " ".join(chunk.linked_node_codes),
                    )
                    for chunk in chunks
                ],
            )

    def search(self, query_text: str, top_k: int) -> list[tuple[str, float]]:
        match_query = self._build_match_query(query_text)
        if not match_query:
            return []
        try:
            rows = self._conn.execute(
                "SELECT chunk_id, bm25(docs_fts) AS score "
                "FROM docs_fts WHERE docs_fts MATCH ? "
                "ORDER BY score LIMIT ?",
                (match_query, top_k),
            ).fetchall()
        except sqlite3.OperationalError:
            rows = []
        if not rows:
            rows = self._fallback_like_search(query_text, top_k)
        # SQLite bm25 分值越小越相关，这里转成越大越相关，便于和现有融合逻辑一致。
        return [(str(row[0]), 1.0 / (1.0 + abs(float(row[1])))) for row in rows]

    @staticmethod
    def _build_match_query(query_text: str) -> str:
        tokens = BM25Lite.tokenize(query_text)
        if not tokens:
            return ""
        deduped: list[str] = []
        for token in tokens:
            cleaned = token.replace('"', " ").strip()
            if cleaned and cleaned not in deduped:
                deduped.append(cleaned)
        return " OR ".join(f'"{token}"' for token in deduped[:12])

    def _fallback_like_search(self, query_text: str, top_k: int) -> list[tuple[str, float]]:
        tokens = [token for token in BM25Lite.tokenize(query_text) if token.strip()]
        if not tokens:
            return []
        rows = self._conn.execute("SELECT chunk_id, title, text, linked_codes FROM docs_fts").fetchall()
        scored: list[tuple[str, float]] = []
        for chunk_id, title, text, linked_codes in rows:
            haystack = f"{title} {text} {linked_codes}".lower()
            score = sum(1.0 for token in tokens if token.lower() in haystack)
            if score > 0:
                scored.append((str(chunk_id), score))
        scored.sort(key=lambda item: item[1], reverse=True)
        return scored[:top_k]
