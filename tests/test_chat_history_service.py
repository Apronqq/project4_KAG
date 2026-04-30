from pathlib import Path
from types import SimpleNamespace

from sqlalchemy import select

from app.db.models import DiagnosticMemory
from app.db.database import DatabaseManager
from app.services.chat_history_service import ChatHistoryService


def test_chat_history_service_persists_and_builds_context():
    db_path = Path(".tmp_chat_history.db").resolve()
    settings = SimpleNamespace(
        database_url=f"sqlite:///{db_path}",
        chat_recent_messages_limit=4,
        chat_message_char_limit=120,
        chat_total_context_char_limit=500,
    )
    db = DatabaseManager(settings.database_url)
    db.create_tables()

    service = ChatHistoryService(db.session_factory, settings)
    session_id = "session_test"
    service.record_user_message(session_id, "男，52岁，血压176/108 mmHg。")
    service.record_assistant_message(
        session_id,
        "你存在高血压风险。",
        structured_result={
            "primary_diagnosis": {
                "health_status": "high_risk",
                "urgency_level": "high",
                "potential_risks": [
                    {"risk_name": "高血压风险", "disease_name": "高血压", "risk_level": "high"}
                ],
            },
            "secondary_recommendations": {
                "recommended_departments": ["心内科"],
                "follow_up_tests": ["动态血压监测"],
            },
        },
    )

    bundle = service.build_context(session_id, "基于上面的结果，我早餐适合吃什么？")

    assert bundle.session_id == session_id
    assert bundle.history
    assert any(item["role"] == "system" for item in bundle.history)
    assert any("高血压风险" in item["content"] for item in bundle.history)

    db.dispose()
    if db_path.exists():
        db_path.unlink()


def test_chat_history_service_session_crud_and_fact_memory():
    db_path = Path(".tmp_chat_history_session.db").resolve()
    settings = SimpleNamespace(
        database_url=f"sqlite:///{db_path}",
        chat_recent_messages_limit=4,
        chat_message_char_limit=120,
        chat_total_context_char_limit=500,
    )
    db = DatabaseManager(settings.database_url)
    db.create_tables()
    service = ChatHistoryService(db.session_factory, settings)

    created = service.create_session("测试会话")
    assert created["title"] == "测试会话"
    session_id = created["session_id"]

    service.record_user_message(session_id, "男，52岁，血压176/108 mmHg。")
    bundle = service.build_context(session_id, "基于上面的结果，我早餐适合吃什么？")
    assert bundle.history

    sessions = service.list_sessions()
    assert any(item["session_id"] == session_id for item in sessions)

    deleted = service.delete_session(session_id)
    assert deleted is True

    db.dispose()
    if db_path.exists():
        db_path.unlink()


def test_fact_memory_conflict_detection_and_diagnostic_memory_versioning():
    db_path = Path(".tmp_chat_history_version.db").resolve()
    settings = SimpleNamespace(
        database_url=f"sqlite:///{db_path}",
        chat_recent_messages_limit=4,
        chat_message_char_limit=120,
        chat_total_context_char_limit=500,
    )
    db = DatabaseManager(settings.database_url)
    db.create_tables()
    service = ChatHistoryService(db.session_factory, settings)
    created = service.create_session("版本测试")
    session_id = created["session_id"]

    from app.schemas.exam import ExamItem, NormalizedMedicalExamJSON, PatientProfile

    exam_v1 = NormalizedMedicalExamJSON(
        patient_profile=PatientProfile(sex="male", age=52),
        exam_items=[ExamItem(code="blood_pressure_systolic", name="收缩压", value=176, unit="mmHg")],
    )
    conflicts_v1 = service.upsert_user_fact_memory(session_id, exam_v1)
    assert conflicts_v1 == []

    exam_v2 = NormalizedMedicalExamJSON(
        patient_profile=PatientProfile(sex="male", age=52),
        exam_items=[ExamItem(code="blood_pressure_systolic", name="收缩压", value=166, unit="mmHg")],
    )
    conflicts_v2 = service.upsert_user_fact_memory(session_id, exam_v2)
    assert any("已更新事实" in item for item in conflicts_v2)

    structured_v1 = {
        "primary_diagnosis": {
            "health_status": "high_risk",
            "urgency_level": "high",
            "potential_risks": [{"risk_name": "高血压风险", "disease_name": "高血压", "risk_level": "high"}],
            "key_abnormal_indicators": [{"indicator_name": "收缩压", "value": 176, "unit": "mmHg"}],
        },
        "secondary_recommendations": {
            "recommended_departments": ["心内科"],
            "follow_up_tests": ["动态血压监测"],
            "lifestyle_interventions": [],
            "medication_directions": [],
            "contraindications": [],
        },
        "evidence": {"chunks": [{"title": "证据1"}]},
    }
    structured_v2 = {
        "primary_diagnosis": {
            "health_status": "needs_follow_up",
            "urgency_level": "medium",
            "potential_risks": [{"risk_name": "高血压风险", "disease_name": "高血压", "risk_level": "medium"}],
            "key_abnormal_indicators": [{"indicator_name": "收缩压", "value": 166, "unit": "mmHg"}],
        },
        "secondary_recommendations": {
            "recommended_departments": ["全科医学科"],
            "follow_up_tests": ["家庭血压监测"],
            "lifestyle_interventions": [],
            "medication_directions": [],
            "contraindications": [],
        },
        "evidence": {"chunks": [{"title": "证据2"}]},
    }

    service.record_assistant_message(session_id, "第一次诊断", structured_result=structured_v1)
    service.record_assistant_message(session_id, "第二次诊断", structured_result=structured_v2)

    with db.session_factory() as session:
        rows = session.execute(select(DiagnosticMemory).order_by(DiagnosticMemory.version_no.asc())).scalars().all()
        assert len(rows) == 2
        assert rows[0].is_current == "false"
        assert rows[1].is_current == "true"
        assert rows[1].version_no == 2

    db.dispose()
    if db_path.exists():
        db_path.unlink()


def test_context_includes_diagnostic_trend_for_followup():
    db_path = Path(".tmp_chat_history_trend.db").resolve()
    settings = SimpleNamespace(
        database_url=f"sqlite:///{db_path}",
        chat_recent_messages_limit=4,
        chat_message_char_limit=200,
        chat_total_context_char_limit=1200,
        summary_trigger_chars=2000,
    )
    db = DatabaseManager(settings.database_url)
    db.create_tables()
    service = ChatHistoryService(db.session_factory, settings)
    session_id = "session_trend"

    first = {
        "primary_diagnosis": {
            "health_status": "high_risk",
            "urgency_level": "high",
            "potential_risks": [],
            "key_abnormal_indicators": [{"indicator_name": "收缩压", "value": 176, "unit": "mmHg"}],
        },
        "secondary_recommendations": {},
        "evidence": {"chunks": []},
    }
    second = {
        "primary_diagnosis": {
            "health_status": "needs_follow_up",
            "urgency_level": "medium",
            "potential_risks": [],
            "key_abnormal_indicators": [{"indicator_name": "收缩压", "value": 150, "unit": "mmHg"}],
        },
        "secondary_recommendations": {},
        "evidence": {"chunks": []},
    }
    service.record_assistant_message(session_id, "第一次", structured_result=first)
    service.record_assistant_message(session_id, "第二次", structured_result=second)

    bundle = service.build_context(session_id, "和上次相比怎么样？")

    assert any("诊断趋势记忆" in item["content"] for item in bundle.history)

    db.dispose()
    if db_path.exists():
        db_path.unlink()


def test_summary_llm_triggered_only_after_threshold():
    class FakeSummaryLLM:
        def __init__(self):
            self.calls = 0

        def invoke(self, prompt):
            self.calls += 1
            return SimpleNamespace(content="压缩摘要")

    db_path = Path(".tmp_chat_history_summary.db").resolve()
    settings = SimpleNamespace(
        database_url=f"sqlite:///{db_path}",
        chat_recent_messages_limit=4,
        chat_message_char_limit=1000,
        chat_total_context_char_limit=2000,
        summary_trigger_chars=80,
    )
    db = DatabaseManager(settings.database_url)
    db.create_tables()
    llm = FakeSummaryLLM()
    service = ChatHistoryService(db.session_factory, settings, summary_llm=llm)
    session_id = "session_summary"

    service.record_user_message(session_id, "短问题")
    service.record_assistant_message(session_id, "短回答")
    assert llm.calls == 0

    service.record_user_message(session_id, "较长问题" * 20)
    service.record_assistant_message(session_id, "较长回答" * 20)
    assert llm.calls == 1

    db.dispose()
    if db_path.exists():
        db_path.unlink()
