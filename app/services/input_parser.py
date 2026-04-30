from __future__ import annotations

import json
import logging
import re
from typing import Any

from app.schemas.exam import ExtractedExamPayload, MedicalParseResponse, NormalizedMedicalExamJSON, PatientProfile
from app.services.indicator_normalizer import IndicatorNormalizer

logger = logging.getLogger(__name__)


class MedicalInputParser:
    def __init__(self, normalizer: IndicatorNormalizer, extractor=None):
        self._normalizer = normalizer
        self._extractor = extractor

    def parse(self, raw_input: Any) -> MedicalParseResponse:
        if isinstance(raw_input, NormalizedMedicalExamJSON):
            normalized = self._normalizer.normalize(raw_input)
            return MedicalParseResponse(normalized_exam_json=normalized, missing_fields=self._missing_fields(normalized))

        if isinstance(raw_input, dict):
            payload = self._parse_dict_payload(raw_input)
            normalized = self._normalizer.normalize(payload)
            return MedicalParseResponse(normalized_exam_json=normalized, missing_fields=self._missing_fields(normalized))

        if isinstance(raw_input, str):
            text = raw_input.strip()
            if not text:
                empty = NormalizedMedicalExamJSON()
                return MedicalParseResponse(normalized_exam_json=empty, missing_fields=["user_question", "exam_items"])
            try:
                loaded = json.loads(text)
                if isinstance(loaded, dict):
                    return self.parse(loaded)
            except json.JSONDecodeError:
                logger.debug("input_parser.raw_text_not_json", exc_info=True)
            parsed = self._parse_text_payload(text)
            normalized = self._normalizer.normalize(parsed)
            return MedicalParseResponse(
                normalized_exam_json=normalized,
                missing_fields=self._missing_fields(normalized),
                warnings=self._warnings(normalized),
            )

        raise TypeError(f"Unsupported raw_input type: {type(raw_input)!r}")

    def _parse_dict_payload(self, raw_input: dict[str, Any]) -> NormalizedMedicalExamJSON:
        if "input_data" in raw_input:
            return self._parse_dict_payload(raw_input["input_data"])

        if "normalized_exam_json" in raw_input and isinstance(raw_input["normalized_exam_json"], dict):
            raw_input = raw_input["normalized_exam_json"]

        patient_profile = raw_input.get("patient_profile", {})
        return NormalizedMedicalExamJSON(
            patient_profile=PatientProfile(
                sex=patient_profile.get("sex"),
                age=patient_profile.get("age"),
            ),
            exam_items=raw_input.get("exam_items", []),
            medical_history=raw_input.get("medical_history", []),
            current_medications=raw_input.get("current_medications", []),
            allergies=raw_input.get("allergies", []),
            user_question=raw_input.get("user_question", ""),
            source_type="json",
        )

    def _parse_text_payload(self, text: str) -> NormalizedMedicalExamJSON:
        if self._extractor is not None:
            try:
                extracted = self._extractor.extract_structured(text, ExtractedExamPayload)
                return NormalizedMedicalExamJSON(
                    patient_profile=PatientProfile(
                        sex=extracted.patient_profile.sex,
                        age=extracted.patient_profile.age,
                    ),
                    exam_items=extracted.exam_items,
                    medical_history=extracted.medical_history,
                    current_medications=extracted.current_medications,
                    allergies=extracted.allergies,
                    user_question=extracted.user_question or text,
                    source_type="text",
                )
            except Exception:
                logger.warning("input_parser.llm_extractor_failed", exc_info=True)

        exam_items = self._normalizer.extract_from_text(text)
        sex = None
        if "男" in text.lower() or re.search(r"\bmale\b", text, re.IGNORECASE):
            sex = "male"
        elif "女" in text.lower() or re.search(r"\bfemale\b", text, re.IGNORECASE):
            sex = "female"

        age_match = re.search(r"(\d{1,3})\s*岁", text)
        age = int(age_match.group(1)) if age_match else None

        history = self._extract_list(text, ["既往史", "病史"])
        medications = self._extract_list(text, ["用药史", "用药"])
        allergies = self._extract_list(text, ["过敏史", "过敏"])

        return NormalizedMedicalExamJSON(
            patient_profile=PatientProfile(sex=sex, age=age),
            exam_items=exam_items,
            medical_history=history,
            current_medications=medications,
            allergies=allergies,
            user_question=text,
            source_type="text",
        )

    @staticmethod
    def _extract_list(text: str, prefixes: list[str]) -> list[str]:
        extracted: list[str] = []
        stop_tokens = [
            "既往史",
            "病史",
            "用药史",
            "用药",
            "过敏史",
            "过敏",
            "请判断",
            "并给出",
        ]
        for prefix in prefixes:
            match = re.search(rf"{prefix}\s*[:：]\s*(.+)", text, re.IGNORECASE)
            if not match:
                continue
            segment = match.group(1).strip()
            end = len(segment)
            for token in stop_tokens:
                if token == prefix:
                    continue
                idx = segment.find(token)
                if idx != -1:
                    end = min(end, idx)
            segment = re.split(r"[。；;\n]", segment[:end], maxsplit=1)[0]
            values = re.split(r"[、,，/ ]+", segment.strip().strip("。"))
            extracted.extend([value for value in values if value and value != "无"])
        deduped: list[str] = []
        for value in extracted:
            if value not in deduped:
                deduped.append(value)
        return deduped

    @staticmethod
    def _missing_fields(normalized: NormalizedMedicalExamJSON) -> list[str]:
        missing: list[str] = []
        if not normalized.user_question:
            missing.append("user_question")
        if not normalized.exam_items:
            missing.append("exam_items")
        if normalized.patient_profile.age is None:
            missing.append("patient_profile.age")
        if normalized.patient_profile.sex is None:
            missing.append("patient_profile.sex")
        return missing

    @staticmethod
    def _warnings(normalized: NormalizedMedicalExamJSON) -> list[str]:
        warnings: list[str] = []
        if len(normalized.exam_items) < 2:
            warnings.append("Too few exam items were extracted from the raw text.")
        return warnings
