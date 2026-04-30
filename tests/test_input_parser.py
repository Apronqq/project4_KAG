from app.services.indicator_normalizer import IndicatorNormalizer
from app.services.input_parser import MedicalInputParser


def test_input_parser_extracts_blood_pressure_and_age():
    parser = MedicalInputParser(IndicatorNormalizer())
    text = "男，52岁，血压 176/108 mmHg，空腹血糖 7.2 mmol/L，请判断健康状况和风险。"

    parsed = parser.parse(text)

    assert parsed.normalized_exam_json.patient_profile.sex == "male"
    assert parsed.normalized_exam_json.patient_profile.age == 52
    assert len(parsed.normalized_exam_json.exam_items) >= 3
    assert "patient_profile.age" not in parsed.missing_fields
    assert "patient_profile.sex" not in parsed.missing_fields

