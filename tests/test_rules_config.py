from app.schemas.exam import ExamItem, NormalizedMedicalExamJSON
from app.services.rules import IndicatorRuleEngine


def test_rule_engine_accepts_injected_rule_config():
    rule_engine = IndicatorRuleEngine(
        rule_config={
            "single": [
                {
                    "id": "custom_uric_acid_high",
                    "indicator_code": "uric_acid",
                    "operator": "gt",
                    "threshold": 420,
                    "state_code": "URIC_ACID_high",
                    "label": "尿酸升高",
                    "severity": "medium",
                }
            ],
            "composite": [],
        }
    )
    exam = NormalizedMedicalExamJSON(
        exam_items=[ExamItem(code="uric_acid", name="尿酸", value=480, unit="umol/L")]
    )

    states = rule_engine.detect_states(exam)

    assert [state.state_code for state in states] == ["URIC_ACID_high"]
    assert states[0].rule_id == "custom_uric_acid_high"
