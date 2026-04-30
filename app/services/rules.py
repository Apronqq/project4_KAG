from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from app.schemas.exam import DetectedState, ExamItem, NormalizedMedicalExamJSON


DEFAULT_RULE_CONFIG_PATH = Path(__file__).resolve().parents[1] / "config" / "medical_rules.json"


class IndicatorRuleEngine:
    def __init__(self, rule_config_path: str | Path | None = None, rule_config: dict[str, Any] | None = None):
        self._rule_config = rule_config or self._load_rule_config(rule_config_path or DEFAULT_RULE_CONFIG_PATH)
        self._single_rules = list(self._rule_config.get("single", []))
        self._composite_rules = list(self._rule_config.get("composite", []))

    def detect_states(self, exam_json: NormalizedMedicalExamJSON) -> list[DetectedState]:
        values_by_code = {
            item.code: item.value
            for item in exam_json.exam_items
            if item.code is not None and item.value is not None
        }
        detected: list[DetectedState] = []

        for item in exam_json.exam_items:
            if item.code is None or item.value is None:
                continue
            for rule in self._single_rules:
                if item.code != rule.get("indicator_code"):
                    continue
                if self._single_rule_matches(rule, item, exam_json, values_by_code):
                    detected.append(self._build_single_state(rule, item))

        for rule in self._composite_rules:
            if self._conditions_match(rule.get("conditions", []), exam_json, values_by_code):
                detected.append(self._build_composite_state(rule, values_by_code))

        return detected

    @staticmethod
    def _load_rule_config(path: str | Path) -> dict[str, Any]:
        # 中文注释：规则阈值放在 JSON 配置中，新增指标时优先改配置，不再改 if/elif 代码。
        with Path(path).open("r", encoding="utf-8") as file:
            config = json.load(file)
        if not isinstance(config, dict):
            raise ValueError("Medical rule config must be a JSON object.")
        return config

    def _single_rule_matches(
        self,
        rule: dict[str, Any],
        item: ExamItem,
        exam_json: NormalizedMedicalExamJSON,
        values_by_code: dict[str, float],
    ) -> bool:
        if "conditions" in rule:
            return self._conditions_match(rule["conditions"], exam_json, values_by_code, item)
        condition = {
            "field": "value",
            "operator": rule.get("operator"),
            "value": rule.get("threshold"),
        }
        return self._condition_matches(condition, exam_json, values_by_code, item)

    def _conditions_match(
        self,
        conditions: list[dict[str, Any]],
        exam_json: NormalizedMedicalExamJSON,
        values_by_code: dict[str, float],
        current_item: ExamItem | None = None,
        logic: str = "AND",
    ) -> bool:
        checks = [
            self._condition_matches(condition, exam_json, values_by_code, current_item)
            for condition in conditions
        ]
        if logic.upper() == "OR":
            return any(checks)
        return all(checks)

    def _condition_matches(
        self,
        condition: dict[str, Any],
        exam_json: NormalizedMedicalExamJSON,
        values_by_code: dict[str, float],
        current_item: ExamItem | None = None,
    ) -> bool:
        if "conditions" in condition:
            return self._conditions_match(
                condition.get("conditions", []),
                exam_json,
                values_by_code,
                current_item,
                logic=condition.get("logic", "AND"),
            )

        actual = self._resolve_condition_value(condition, exam_json, values_by_code, current_item)
        operator = condition.get("operator")
        expected = condition.get("value", condition.get("threshold"))
        return self._compare(actual, operator, expected)

    @staticmethod
    def _resolve_condition_value(
        condition: dict[str, Any],
        exam_json: NormalizedMedicalExamJSON,
        values_by_code: dict[str, float],
        current_item: ExamItem | None,
    ) -> Any:
        if "indicator_code" in condition:
            return values_by_code.get(condition["indicator_code"])
        field = condition.get("field", "value")
        if field == "value":
            return current_item.value if current_item is not None else None
        if field == "patient_profile.age":
            return exam_json.patient_profile.age
        if field == "patient_profile.sex":
            sex = exam_json.patient_profile.sex
            return sex.lower() if isinstance(sex, str) else sex
        return None

    @staticmethod
    def _compare(actual: Any, operator: str, expected: Any) -> bool:
        if operator in {"gte", "gt", "lte", "lt"}:
            if actual is None or expected is None:
                return False
            actual_value = float(actual)
            expected_value = float(expected)
            if operator == "gte":
                return actual_value >= expected_value
            if operator == "gt":
                return actual_value > expected_value
            if operator == "lte":
                return actual_value <= expected_value
            return actual_value < expected_value
        if operator == "eq":
            return actual == expected
        if operator == "in":
            return actual in expected
        if operator == "not_in":
            return actual not in expected
        raise ValueError(f"Unsupported rule operator: {operator}")

    @staticmethod
    def _build_single_state(rule: dict[str, Any], item: ExamItem) -> DetectedState:
        return DetectedState(
            indicator_code=item.code or "",
            indicator_name=item.name,
            state_code=rule["state_code"],
            label=rule["label"],
            severity=rule["severity"],
            value=item.value or 0,
            unit=item.unit,
            rule_id=rule["id"],
        )

    @staticmethod
    def _build_composite_state(rule: dict[str, Any], values_by_code: dict[str, float]) -> DetectedState:
        value_codes = rule.get("value_indicator_codes", [])
        values = [values_by_code[code] for code in value_codes if code in values_by_code]
        return DetectedState(
            indicator_code=rule["indicator_code"],
            indicator_name=rule["indicator_name"],
            state_code=rule["state_code"],
            label=rule["label"],
            severity=rule["severity"],
            value=max(values) if values else 0,
            unit=rule.get("unit"),
            rule_id=rule["id"],
        )
