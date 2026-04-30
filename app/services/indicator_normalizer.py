from __future__ import annotations

import re

from app.graph.seed_data import INDICATOR_ALIASES
from app.schemas.exam import ExamItem, NormalizedMedicalExamJSON


class IndicatorNormalizer:
    def normalize(self, exam_json: NormalizedMedicalExamJSON) -> NormalizedMedicalExamJSON:
        normalized_items = [self.normalize_item(item) for item in exam_json.exam_items]
        return exam_json.model_copy(update={"exam_items": normalized_items})

    def normalize_item(self, item: ExamItem) -> ExamItem:
        raw_name = item.name.strip()
        alias_key = raw_name.lower()
        code = item.code
        canonical_name = raw_name
        default_unit = item.unit

        if raw_name in INDICATOR_ALIASES:
            code, canonical_name, default_unit = INDICATOR_ALIASES[raw_name]
        elif alias_key in INDICATOR_ALIASES:
            code, canonical_name, default_unit = INDICATOR_ALIASES[alias_key]

        normalized_unit = self.normalize_unit(item.unit or default_unit)
        normalized_value, normalized_unit = self.convert_value_and_unit(code, item.value, normalized_unit, default_unit)

        return item.model_copy(
            update={
                "code": code,
                "name": canonical_name,
                "unit": normalized_unit,
                "value": normalized_value,
            }
        )

    @staticmethod
    def normalize_unit(unit: str | None) -> str | None:
        if unit is None:
            return None
        compact = unit.strip()
        replacements = {
            "mg/dl": "mg/dL",
            "mmol/l": "mmol/L",
            "μmol/l": "umol/L",
            "µmol/l": "umol/L",
            "umol/l": "umol/L",
            "u/l": "U/L",
        }
        return replacements.get(compact.lower(), compact)

    def convert_value_and_unit(
        self,
        code: str | None,
        value: float | None,
        unit: str | None,
        default_unit: str | None,
    ) -> tuple[float | None, str | None]:
        if value is None or code is None or unit is None:
            return value, unit
        if code == "fasting_blood_glucose" and unit == "mg/dL":
            return round(value / 18.0, 2), default_unit or "mmol/L"
        if code == "creatinine" and unit == "mg/dL":
            return round(value * 88.4, 2), default_unit or "umol/L"
        return value, unit

    def extract_from_text(self, text: str) -> list[ExamItem]:
        items: list[ExamItem] = []
        blood_pressure = re.search(r"(血压|BP)\s*[:：]?\s*(\d{2,3})\s*/\s*(\d{2,3})\s*(mmhg)?", text, re.IGNORECASE)
        if blood_pressure:
            systolic = float(blood_pressure.group(2))
            diastolic = float(blood_pressure.group(3))
            items.append(
                ExamItem(
                    name="收缩压",
                    value=systolic,
                    unit="mmHg",
                    source_text=blood_pressure.group(0),
                )
            )
            items.append(
                ExamItem(
                    name="舒张压",
                    value=diastolic,
                    unit="mmHg",
                    source_text=blood_pressure.group(0),
                )
            )

        for alias, (code, canonical_name, default_unit) in INDICATOR_ALIASES.items():
            if alias in {"收缩压", "舒张压", "血压"}:
                continue
            pattern = re.compile(
                rf"({re.escape(alias)})\s*[:：]?\s*([0-9]+(?:\.[0-9]+)?)\s*([A-Za-z/%0-9\.\-μµ/]+)?",
                re.IGNORECASE,
            )
            for match in pattern.finditer(text):
                value = float(match.group(2))
                unit = match.group(3) or default_unit
                items.append(
                    ExamItem(
                        code=code,
                        name=canonical_name,
                        value=value,
                        unit=unit,
                        source_text=match.group(0),
                    )
                )
        deduped: list[ExamItem] = []
        seen: set[tuple[str, float | None, str | None]] = set()
        for item in items:
            key = (item.name, item.value, item.unit)
            if key in seen:
                continue
            seen.add(key)
            deduped.append(item)
        return deduped
