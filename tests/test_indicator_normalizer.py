from app.schemas.exam import ExamItem, NormalizedMedicalExamJSON
from app.services.indicator_normalizer import IndicatorNormalizer


def test_indicator_normalizer_converts_fbg_from_mgdl():
    normalizer = IndicatorNormalizer()
    exam = NormalizedMedicalExamJSON(
        exam_items=[
            ExamItem(name="FBG", value=126, unit="mg/dL"),
        ]
    )

    normalized = normalizer.normalize(exam)
    item = normalized.exam_items[0]

    assert item.code == "fasting_blood_glucose"
    assert item.name == "空腹血糖"
    assert item.unit == "mmol/L"
    assert round(item.value, 2) == 7.00
