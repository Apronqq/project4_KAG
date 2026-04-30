from app.services.react_agent import MedicalReActAgent


class FakeKnowledgeTool:
    def __init__(self):
        self.calls = []

    def _run(self, query: str) -> str:
        self.calls.append(query)
        return "高血压患者应限盐、规律运动，并按医嘱复查。"


def fallback_answer(user_input: str, session_history: list[dict], evidence_text: str) -> str:
    return f"回答：{user_input}\n证据：{evidence_text or '无'}"


def test_react_agent_uses_memory_without_external_tool():
    tool = FakeKnowledgeTool()
    agent = MedicalReActAgent(chat_model=None, knowledge_tool=tool, answer_builder=fallback_answer)

    events = list(
        agent.iter_events(
            "我的血压风险严不严重？",
            [{"role": "system", "content": "结构化诊断记忆：主要风险=高血压风险；异常指标=收缩压=176mmHg"}],
        )
    )

    assert not tool.calls
    assert any(event["type"] == "agent_decision" and event["action"] == "final_answer" for event in events)
    assert events[-1]["type"] == "final_answer"


def test_react_agent_calls_knowledge_tool_for_general_question():
    tool = FakeKnowledgeTool()
    agent = MedicalReActAgent(chat_model=None, knowledge_tool=tool, answer_builder=fallback_answer)

    events = list(agent.iter_events("高血压饮食怎么吃？", []))

    assert tool.calls == ["高血压饮食怎么吃？"]
    assert any(event["type"] == "tool_call" and event["name"] == "lookup_medical_knowledge" for event in events)
    assert events[-1]["type"] == "final_answer"
