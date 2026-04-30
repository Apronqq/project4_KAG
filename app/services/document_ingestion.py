from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from html import unescape
from pathlib import Path

from pypdf import PdfReader

from app.graph.seed_data import DISEASE_TO_INTERVENTIONS, INDICATOR_ALIASES, STATE_TO_RISK
from app.schemas.exam import EvidenceChunk, KnowledgeDocument


@dataclass
class ChunkingResult:
    document: KnowledgeDocument
    chunks: list[EvidenceChunk]


class DocumentChunker:
    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 80):
        self._chunk_size = chunk_size
        self._chunk_overlap = chunk_overlap

    def chunk_document(self, filename: str, content: bytes) -> ChunkingResult:
        suffix = Path(filename).suffix.lower()
        text = self._extract_text(filename, content, suffix)
        content_hash = hashlib.sha256(content).hexdigest()
        doc_id = f"doc_{content_hash[:16]}"
        chunks = self._split_to_chunks(text)
        linked_node_codes = self._infer_linked_node_codes(text)

        evidence_chunks: list[EvidenceChunk] = []
        for index, chunk_text in enumerate(chunks, 1):
            evidence_chunks.append(
                EvidenceChunk(
                    chunk_id=f"{doc_id}_chunk_{index}",
                    doc_id=doc_id,
                    title=f"{filename} - chunk {index}",
                    text=chunk_text,
                    linked_node_codes=linked_node_codes,
                    source_type="guideline",
                )
            )

        document = KnowledgeDocument(
            doc_id=doc_id,
            filename=filename,
            file_type=suffix.lstrip(".") or "txt",
            content_hash=content_hash,
            chunk_count=len(evidence_chunks),
            linked_node_codes=linked_node_codes,
        )
        return ChunkingResult(document=document, chunks=evidence_chunks)

    def _extract_text(self, filename: str, content: bytes, suffix: str) -> str:
        if suffix in {".txt", ".md"}:
            return content.decode("utf-8", errors="ignore")
        if suffix in {".html", ".htm"}:
            return self._html_to_text(content.decode("utf-8", errors="ignore"))
        if suffix == ".json":
            data = json.loads(content.decode("utf-8", errors="ignore"))
            return json.dumps(data, ensure_ascii=False, indent=2)
        if suffix == ".pdf":
            tmp_path = Path.cwd() / f".tmp_{filename}"
            tmp_path.write_bytes(content)
            try:
                reader = PdfReader(str(tmp_path))
                texts = []
                for page in reader.pages:
                    texts.append(page.extract_text() or "")
                return "\n".join(texts)
            finally:
                if tmp_path.exists():
                    tmp_path.unlink()
        raise ValueError(f"Unsupported file type: {suffix}")

    def _split_to_chunks(self, text: str) -> list[str]:
        normalized = " ".join(text.split())
        if not normalized:
            return []
        chunks: list[str] = []
        start = 0
        while start < len(normalized):
            end = min(len(normalized), start + self._chunk_size)
            chunks.append(normalized[start:end])
            if end >= len(normalized):
                break
            start = max(0, end - self._chunk_overlap)
        return chunks

    def _infer_linked_node_codes(self, text: str) -> list[str]:
        lowered = text.lower()
        linked: list[str] = []

        for alias in INDICATOR_ALIASES:
            if alias.lower() in lowered:
                linked.extend(self._map_alias_to_related_codes(alias))

        for state_code, items in STATE_TO_RISK.items():
            if state_code.lower() in lowered:
                linked.append(state_code)
                for item in items:
                    linked.extend([str(item["risk_code"]), str(item["disease_code"])])

        for disease_code, payload in DISEASE_TO_INTERVENTIONS.items():
            if disease_code.lower() in lowered:
                linked.append(disease_code)
            for text_list in payload.values():
                if any(fragment in text for fragment in text_list):
                    linked.append(disease_code)

        deduped: list[str] = []
        for code in linked:
            if code not in deduped:
                deduped.append(code)
        return deduped

    @staticmethod
    def _html_to_text(html: str) -> str:
        html = re.sub(r"(?is)<script.*?>.*?</script>", " ", html)
        html = re.sub(r"(?is)<style.*?>.*?</style>", " ", html)
        html = re.sub(r"(?i)</(p|div|h1|h2|h3|li|tr|section|article|br)>", "\n", html)
        html = re.sub(r"(?is)<.*?>", " ", html)
        html = unescape(html)
        html = re.sub(r"\r", "", html)
        html = re.sub(r"\n\s*\n+", "\n\n", html)
        html = re.sub(r"[ \t]+", " ", html)
        return html.strip()

    @staticmethod
    def _map_alias_to_related_codes(alias: str) -> list[str]:
        alias_lower = alias.lower()
        mapping = {
            "收缩压": ["hypertension_risk", "hypertension"],
            "舒张压": ["hypertension_risk", "hypertension"],
            "血压": ["hypertension_risk", "hypertension"],
            "空腹血糖": ["prediabetes_risk", "diabetes_risk", "prediabetes", "type2_diabetes"],
            "fbg": ["prediabetes_risk", "diabetes_risk", "prediabetes", "type2_diabetes"],
            "hba1c": ["prediabetes_risk", "diabetes_risk", "prediabetes", "type2_diabetes"],
            "糖化血红蛋白": ["prediabetes_risk", "diabetes_risk", "prediabetes", "type2_diabetes"],
            "ldl": ["dyslipidemia_risk", "dyslipidemia"],
            "ldl-c": ["dyslipidemia_risk", "dyslipidemia"],
            "低密度脂蛋白": ["dyslipidemia_risk", "dyslipidemia"],
            "甘油三酯": ["dyslipidemia_risk", "dyslipidemia"],
            "tg": ["dyslipidemia_risk", "dyslipidemia"],
            "肌酐": ["ckd_risk", "ckd"],
            "cr": ["ckd_risk", "ckd"],
            "egfr": ["ckd_risk", "ckd"],
            "alt": ["liver_function_abnormal_risk", "liver_function_abnormality"],
            "谷丙转氨酶": ["liver_function_abnormal_risk", "liver_function_abnormality"],
            "ast": ["liver_function_abnormal_risk", "liver_function_abnormality"],
        }
        return mapping.get(alias_lower, mapping.get(alias, []))
