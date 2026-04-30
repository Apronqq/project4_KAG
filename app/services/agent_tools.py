from __future__ import annotations

from app.schemas.exam import RetrievalQuery


class MedicalKnowledgeRetrievalTool:
    """Supplementary retrieval tool for follow-up medical knowledge questions."""

    def __init__(self, evidence_store, top_k: int = 5):
        self._evidence_store = evidence_store
        self._top_k = top_k

    def _run(self, query: str) -> str:
        chunks = self._evidence_store.search(
            queries=[RetrievalQuery(label="agent_followup", text=query)],
            node_codes=[],
            top_k=self._top_k,
        )
        if not chunks:
            return "No relevant medical evidence was found in the knowledge base."

        lines: list[str] = []
        for index, chunk in enumerate(chunks, 1):
            lines.append(
                f"[{index}] {chunk.title}\n"
                f"来源类型: {chunk.source_type}\n"
                f"内容: {chunk.text}"
            )
        return "\n\n---\n\n".join(lines)
