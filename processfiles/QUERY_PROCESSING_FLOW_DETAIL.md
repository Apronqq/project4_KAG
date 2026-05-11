# 用户 Query 处理流程详解：初诊与追问

## 1. 文档目标

本文档专门解释本项目中“用户 query 是如何被处理的”。它聚焦两个入口场景：

- 初诊：用户提交一段包含体检指标、病史、用药史的文本，希望系统完成体检评估。
- 追问：用户已经有历史评估结果，在同一会话中继续询问风险、饮食、复查、趋势或用药边界问题。

本文档的目标是让读者不看代码，也能掌握：

- 用户请求从 API 到 Agent 的完整调用链。
- 初诊和追问如何被区分。
- 每一步内部做了什么。
- 各模块之间传递什么数据。
- 流式接口如何推送事件。
- 最终回答和记忆如何写回数据库。

涉及的核心代码文件：

```text
app/api/routes/medical.py
app/services/medical_agent.py
app/agents/medical_multi_agent.py
app/workflows/medical_kag_pipeline.py
app/services/chat_history_service.py
app/services/agent_tools.py
app/schemas/exam.py
```

## 2. 用户 Query 的入口类型

项目中与用户 query 直接相关的 API 有四个：

| API | 用途 | 是否走 Multi-Agent |
|-----|------|-------------------|
| `POST /medical/exam/parse` | 只解析体检文本，不评估风险 | 否 |
| `POST /medical/exam/assess` | 直接执行确定性体检评估 Workflow | 否 |
| `POST /medical/agent/chat` | 用户对话，自动区分初诊和追问 | 是 |
| `POST /medical/agent/chat/stream` | 用户对话流式版，自动区分初诊和追问 | 是 |

本说明文档重点解释：

```text
/medical/agent/chat
/medical/agent/chat/stream
```

因为这两个接口才是完整的“用户 query -> Multi-Agent -> Workflow / Memory / Retrieval / Safety -> Response”链路。

## 3. 总体调用链

不论初诊还是追问，用户 query 都会先进入 FastAPI 路由：

```text
app/api/routes/medical.py
```

非流式接口调用链：

```text
agent_chat()
  -> _extract_payload()
  -> get_runtime()
  -> session_id 创建或复用
  -> medical_agent.looks_like_initial_assessment()
  -> 初诊分支 或 追问分支
```

流式接口调用链：

```text
agent_chat_stream()
  -> _extract_payload()
  -> get_runtime()
  -> session_id 创建或复用
  -> chat_history_service.build_context()
  -> medical_agent.stream_assess_async()
  -> MultiAgentSupervisor.aiter_events()
  -> SSE 持续 yield
```

核心分层如下：

```text
FastAPI 路由层
  负责 HTTP 请求解析、session_id、记忆写入、SSE 包装

MedicalAssessmentAgent facade
  负责把同步、异步、流式接口统一转给 Multi-Agent Supervisor

MedicalMultiAgentSupervisor
  负责 Triage / Assessment / Memory / Retrieval / Synthesis / SafetyReview

MedicalKAGWorkflow
  负责初诊确定性体检评估

ChatHistoryService
  负责事实记忆、诊断记忆、趋势记忆、摘要记忆
```

## 4. 请求体如何被解析

入口函数：

```text
app/api/routes/medical.py::_extract_payload()
```

处理规则：

1. 读取 HTTP body。
2. 如果 body 为空，返回 400。
3. 将 body 按 UTF-8 解码并去掉首尾空白。
4. 如果 `content-type` 包含 `application/json`：
   - 尝试 `json.loads()`。
   - 如果解析后是 dict 且含有 `input_data` 字段，返回 `payload["input_data"]`。
   - 否则直接返回解析后的 JSON 对象。
   - 如果 JSON 解析失败，回退返回原始文本。
5. 非 JSON 请求直接返回文本。

因此接口支持两种常见输入：

```json
{"input_data": "男，52岁，血压176/108 mmHg，请判断健康状况"}
```

或纯文本：

```text
男，52岁，血压176/108 mmHg，请判断健康状况
```

`/medical/agent/chat` 和 `/medical/agent/chat/stream` 要求最终 payload 必须是字符串。如果不是字符串，会返回 400。

## 5. 初诊与追问如何区分

初诊判断方法位于：

```text
app/services/medical_agent.py::_looks_like_initial_assessment()
```

判断逻辑：

```text
has_medical_keyword = 是否出现体检关键词
has_numeric_value = 是否出现数字
explicit_initial_intent = 是否出现明确评估意图

return has_medical_keyword and (has_numeric_value or explicit_initial_intent)
```

体检关键词包括：

```text
血压 / mmhg / egfr / 肌酐 / 空腹血糖 / hba1c / 体检 / 指标 / 化验
```

明确评估意图包括：

```text
请判断 / 评估一下 / 体检报告 / 化验单
```

示例：

| 用户输入 | 判断 | 原因 |
|----------|------|------|
| `血压176/108 mmHg，请判断健康状况` | 初诊 | 有血压关键词、有数值 |
| `男，52岁，肌酐128，eGFR 54` | 初诊 | 有肌酐/eGFR、有数值 |
| `我的血压风险严不严重？` | 追问 | 有血压但没有新数值，也没有明确评估意图 |
| `高血压早餐怎么吃？` | 追问 | 是知识/建议类追问 |
| `我现在药量要不要调？` | 追问 | 用药边界问题 |

注意：Multi-Agent 内部还有一次 `TriageAgent` 路由。API 层先判断是为了决定记忆写入方式；Supervisor 内部再判断是为了统一图结构控制流。

## 6. 非流式初诊流程

入口：

```text
POST /medical/agent/chat
```

当 `is_initial = True` 时，代码走初诊分支。

### 6.1 API 层处理

调用链：

```text
agent_chat()
  -> runtime.medical_agent.looks_like_initial_assessment(payload)
  -> runtime.medical_agent.assess(payload)
  -> runtime.chat_history_service.record_user_message()
  -> runtime.chat_history_service.upsert_user_fact_memory()
  -> runtime.chat_history_service.record_assistant_message(..., structured_result)
  -> return JSON
```

API 层做四件事：

1. 生成或复用 `session_id`。
2. 调用 `MedicalAssessmentAgent.assess()` 完成评估。
3. 将用户输入写入对话记忆。
4. 将体检事实和诊断结果写入记忆系统。

返回结构：

```json
{
  "session_id": "...",
  "answer": "...",
  "structured_result": {...},
  "timestamp": "..."
}
```

### 6.2 MedicalAssessmentAgent.assess()

位置：

```text
app/services/medical_agent.py::assess()
```

处理逻辑：

```text
result = self._multi_agent_supervisor.run(raw_text, [])
if result.structured_response is not None:
    return result.answer, result.structured_response
fallback: 直接运行 self._workflow.run_state(raw_text)
```

正常情况下，初诊会通过 Multi-Agent Supervisor 跑完：

```text
TriageAgent -> AssessmentAgent -> SafetyReviewAgent
```

`result.structured_response` 是 `MedicalAssessmentResponse`。

### 6.3 Supervisor 初始状态

`MedicalMultiAgentSupervisor.run()` 会构造初始状态：

```python
{
    "user_input": raw_text,
    "session_id": None,
    "session_history": [],
    "events": [],
    "safety_notes": [],
    "requires_safe_rewrite": False,
    "safety_revision_count": 0,
}
```

然后调用 LangGraph：

```text
self._graph.invoke(initial_state)
```

### 6.4 TriageAgent 路由

`TriageAgent` 读取：

```text
user_input
session_history
session_id
```

初诊条件：

```text
_initial_assessment_detector(user_input) == True
或没有 session_history 且没有 session_id
```

初诊时写入：

```python
route = "assessment"
```

并追加事件：

```json
{
  "type": "agent_decision",
  "agent": "triage_agent",
  "action": "route_to_assessment",
  "reason": "识别为首次体检评估"
}
```

LangGraph 根据条件边进入：

```text
assessment_agent
```

### 6.5 AssessmentAgent 执行 KAG Workflow

非流式时，AssessmentAgent 调用：

```text
self._workflow.run_state(state["user_input"])
```

这会运行 `MedicalKAGWorkflow` 的 12 个节点，返回最终 `InternalAssessmentState`。

AssessmentAgent 从返回 state 中取：

```text
assessment_state.response
```

然后调用：

```text
assessment_answer_builder(response)
```

在生产代码中，这个 builder 是：

```text
MedicalAssessmentAgent._compose_answer()
```

`_compose_answer()` 优先尝试 LLM 生成自然语言解释；如果 LLM 不可用或失败，则调用 `_build_fallback_answer()` 生成确定性中文回答。

AssessmentAgent 最终写入：

```python
{
    "assessment_state": assessment_state,
    "structured_response": response,
    "answer": answer,
    "events": events
}
```

### 6.6 SafetyReviewAgent 初诊复核

初诊路径中，SafetyReviewAgent 主要检查：

1. `structured_response.secondary_recommendations.human_review_required`
2. 回答是否涉及用药调整。
3. 回答是否含具体剂量。
4. 回答是否含疑似药名或治疗词。

如果 `human_review_required = true`，会追加：

```text
高风险或复杂结果需要医生复核
```

如果发现具体剂量或疑似药名，且还没有改写过，会设置：

```python
requires_safe_rewrite = True
```

LangGraph 条件边会进入：

```text
synthesis_agent
```

生成安全改写版回答。否则，SafetyReviewAgent 追加最终事件：

```json
{
  "type": "final_answer",
  "agent": "safety_review_agent",
  "content": "..."
}
```

### 6.7 初诊结束后的记忆写入

API 层拿到：

```text
answer
structured
```

然后写记忆：

#### 6.7.1 用户消息

```text
record_user_message(session_id, payload)
```

写入 `ConversationMemory`：

- role = `user`
- content = 原始输入
- content_summary = 截断后的摘要

如果会话标题仍是“新会话”，会把用户输入前 36 个字符作为标题。

#### 6.7.2 事实记忆

```text
upsert_user_fact_memory(session_id, structured.normalized_exam_json)
```

写入 `UserFactMemory`：

- 性别、年龄
- 每个体检指标
- 病史
- 当前用药
- 过敏史

同时检测旧事实冲突。例如上次年龄、指标值、用药史与本次不同，会返回冲突提示。若有冲突，API 会把冲突提示拼到回答前面。

#### 6.7.3 助手消息和诊断记忆

```text
record_assistant_message(session_id, answer, structured.model_dump())
```

写入：

1. `ConversationMemory`：助手回答。
2. `DiagnosticMemory`：结构化诊断版本。

诊断记忆字段包括：

- `version_no`
- `is_current`
- `health_status`
- `urgency_level`
- `risk_summary`
- `abnormal_indicator_summary`
- `department_summary`
- `follow_up_summary`
- `lifestyle_summary`
- `medication_summary`
- `contraindication_summary`
- `evidence_summary`

旧的 current 诊断会被标记为：

```text
is_current = false
```

新诊断成为 current 版本。

## 7. 初诊 KAG Workflow 内部 12 步详解

核心文件：

```text
app/workflows/medical_kag_pipeline.py
```

初诊真正的医学评估由 `MedicalKAGWorkflow` 完成。它使用一个贯穿全链路的状态：

```text
InternalAssessmentState
```

初始状态：

```python
InternalAssessmentState(raw_input=user_input)
```

后续每个节点都读取并修改这个 state。

### 7.1 Step 1：parse_raw_input

处理函数：

```text
_parse_raw_input_node()
```

输入：

```text
state.raw_input
```

调用：

```text
self._parser.parse(state.raw_input)
```

输出写入：

```text
state.normalized_exam_json
state.missing_fields
state.warnings
```

作用：

- 把用户自然语言转成结构化体检 JSON。
- 识别患者年龄、性别、指标、病史、用药史、过敏史和用户问题。
- LLM 可用时用 LLM 抽取；不可用时走正则回退。

### 7.2 Step 2：validate_exam_json

处理函数：

```text
_validate_exam_json_node()
```

作用：

- 确认 `state.normalized_exam_json` 已存在。
- 如果解析后没有结构化结果，抛出错误。

这是 Workflow 的基础保护，避免后续节点处理空对象。

### 7.3 Step 3：normalize_exam_items

处理函数：

```text
_normalize_exam_items_node()
```

调用：

```text
self._normalizer.normalize(state.normalized_exam_json)
```

作用：

- 指标别名映射。
- 单位统一。
- 指标 code 补全。

示例：

```text
血压 / 收缩压 / SBP -> systolic_blood_pressure
肌酐 -> creatinine
eGFR -> egfr
```

输出仍写回：

```text
state.normalized_exam_json
```

### 7.4 Step 4：detect_indicator_states

处理函数：

```text
_detect_indicator_states_node()
```

调用：

```text
self._rule_engine.detect_states(state.normalized_exam_json)
```

输出：

```text
state.detected_states
```

每个 `DetectedState` 包含：

- `indicator_code`
- `indicator_name`
- `state_code`
- `label`
- `severity`
- `value`
- `unit`
- `rule_id`

这一阶段负责医学阈值判定，是系统确定性的核心。

### 7.5 条件短路：健康路径跳过检索

Workflow 中有 `_should_skip_step()`。

如果：

```text
state.normalized_exam_json is not None
并且 state.detected_states 为空
```

则跳过：

```text
retrieve_graph_candidates
expand_intervention_paths
plan_evidence_queries
retrieve_evidence_chunks
rank_medical_evidence
```

原因：

没有异常状态时，不需要访问图谱和证据库，可以直接进入诊断生成。

### 7.6 Step 5：retrieve_graph_candidates

处理函数：

```text
_retrieve_graph_candidates_node()
```

输入：

```text
state.detected_states[*].state_code
```

调用：

```text
self._graph_store.get_risk_candidates(state_codes)
```

输出：

```text
state.risk_candidates
```

这一阶段将异常状态映射到疾病风险。

示例：

```text
SBP_high_stage2 -> hypertension_risk -> Hypertension
eGFR_low -> chronic_kidney_disease_risk -> Chronic Kidney Disease
```

### 7.7 Step 6：expand_intervention_paths

处理函数：

```text
_expand_intervention_paths_node()
```

输入：

```text
state.risk_candidates[*].disease_code
```

调用：

```text
self._graph_store.get_intervention_candidates(disease_codes)
```

输出：

```text
state.intervention_candidates
```

内容包括：

- 干预建议
- 用药方向
- 禁忌
- 复查项目
- 科室
- 图谱路径

如果有异常但图谱没有命中风险候选，该步骤会被跳过，后续证据检索仍可兜底。

### 7.8 Step 7：plan_evidence_queries

处理函数：

```text
_plan_evidence_queries_node()
```

调用：

```text
self._query_planner.build_queries(
    state.normalized_exam_json,
    state.detected_states,
    state.risk_candidates,
)
```

输出：

```text
state.retrieval_queries
```

查询规划会综合：

- 用户原问题。
- Top 风险疾病。
- 异常指标。
- 指标摘要。

目标是让检索 query 比用户原文更贴近医学资料表达。

### 7.9 Step 8：retrieve_evidence_chunks

处理函数：

```text
_retrieve_evidence_chunks_node()
```

输入：

```text
state.retrieval_queries
node_codes = risk_code + disease_code
```

调用：

```text
self._evidence_store.search(
    queries=state.retrieval_queries,
    node_codes=node_codes,
    top_k=self._top_k_evidence,
)
```

输出：

```text
state.evidence_chunks
```

这一阶段内部会走：

```text
Milvus dense retrieval
SQLite FTS5 lexical retrieval
RRF 融合
Rerank
图谱重叠度加权
来源权威度加权
MMR 去重
```

### 7.10 Step 9：rank_medical_evidence

处理函数：

```text
_rank_medical_evidence_node()
```

调用：

```text
self._ranker.rank_risks(
    state.risk_candidates,
    state.evidence_chunks,
    state.detected_states,
)
```

输出覆盖：

```text
state.risk_candidates
```

这一阶段会为疾病风险计算最终排序分数，融合：

- 图谱支持度。
- 异常状态支持数量。
- 证据支持度。

### 7.11 Step 10：generate_primary_diagnosis

处理函数：

```text
_generate_primary_diagnosis_node()
```

调用：

```text
self._formatter.build_primary(
    state.normalized_exam_json,
    state.risk_candidates,
    state.detected_states,
)
```

输出：

```text
state.primary_diagnosis
```

包含：

- 健康状态：`healthy / subhealthy / needs_follow_up / high_risk`
- 紧急程度：`low / medium / high / urgent`
- 潜在风险列表
- 关键异常指标

### 7.12 Step 11：generate_secondary_recommendation

处理函数：

```text
_generate_secondary_recommendation_node()
```

如果存在 `state.intervention_candidates`，先调用：

```text
self._ranker.merge_recommendations()
self._ranker.index_recommendations_by_disease()
```

再调用：

```text
self._formatter.build_secondary()
```

输出：

```text
state.secondary_recommendations
```

包含：

- 推荐科室
- 复查项目
- 生活方式干预
- 用药方向
- 禁忌
- 按疾病聚合的建议
- 是否需要人工复核

### 7.13 Step 12：format_medical_response

处理函数：

```text
_format_medical_response_node()
```

汇总：

```text
normalized_exam_json
primary_diagnosis
secondary_recommendations
graph_paths
evidence_chunks
```

调用：

```text
self._formatter.build_response(...)
```

输出：

```text
state.response = MedicalAssessmentResponse(...)
```

这就是 API 返回和诊断记忆写入所使用的结构化结果。

## 8. 流式初诊流程

流式初诊入口：

```text
POST /medical/agent/chat/stream
```

### 8.1 API 层先做什么

`agent_chat_stream()` 会：

1. 解析 payload。
2. 生成或复用 `session_id`。
3. 调用 `chat_history_service.build_context(session_id, payload)`。
4. 进入内部异步 generator。
5. 先写入用户消息：

```text
record_user_message(session_id, payload)
```

6. 先 yield meta：

```json
{"type": "meta", "session_id": "..."}
```

### 8.2 MedicalAssessmentAgent.stream_assess_async()

调用：

```text
runtime.medical_agent.stream_assess_async(
    payload,
    session_history=context.history,
    session_id=session_id,
)
```

内部会调用：

```text
self._multi_agent_supervisor.aiter_events(...)
```

### 8.3 Supervisor.aiter_events()

异步流式流程：

```text
1. 构建 MedicalMultiAgentState
2. 执行 TriageAgent，yield route 事件
3. 如果 route == assessment：
   async for event in _astream_assessment_agent()
4. 执行 SafetyReviewAgent
5. 如果 requires_safe_rewrite：
   回到 SynthesisAgent 改写
6. 直到 final_answer
```

### 8.4 _astream_assessment_agent()

该方法用于初诊流式评估。

它会先 yield：

```json
{
  "type": "agent_thinking",
  "agent": "assessment_agent",
  "detail": "正在执行确定性 KAG 体检评估流水线"
}
```

然后调用：

```text
self._workflow.iter_events_async(state["user_input"])
```

Workflow 每个节点会 yield 两类 step：

```json
{"type": "step", "name": "...", "status": "running"}
{"type": "step", "name": "...", "status": "completed"}
```

当 Workflow 最终 yield：

```json
{"type": "workflow_state", "state": payload["state"], "internal": true}
```

`_astream_assessment_agent()` 会截获这个内部状态，不直接发给前端，而是转换成：

```json
{
  "type": "assessment_result",
  "agent": "assessment_agent",
  "internal": true,
  "payload": response
}
```

这个事件仍然是 internal，`MedicalAssessmentAgent.stream_assess_async()` 会保存它用于后续发出公开 `result` 事件。

### 8.5 MedicalAssessmentAgent 如何处理 internal 事件

`stream_assess_async()` 对事件做三类处理：

1. `internal == true`：
   - 如果是 `assessment_result`，保存 `structured_response`。
   - 不直接 yield 给前端。
2. `final_answer`：
   - 保存最终 answer。
   - 不直接 yield 给前端。
3. 其他事件：
   - 直接 yield 给 API 层转成 SSE。

Supervisor 结束后，`stream_assess_async()` 再统一输出：

```text
step: 生成答复
content: 分块回答文本
result: 结构化评估结果
done
```

### 8.6 API 层如何写入记忆

API 层在 SSE 过程中收集：

```text
assistant_content
structured_result
normalized_exam_json
```

当流结束后：

1. 如果存在 `normalized_exam_json`：
   - 调用 `upsert_user_fact_memory()`。
   - 如有冲突，额外 yield `memory_notice` 和 `事实记忆更新` step。
2. 调用：

```text
record_assistant_message(session_id, assistant_content, structured_result)
```

这样初诊流式和非流式最终都会写入事实记忆、诊断记忆和对话记忆。

## 9. 非流式追问流程

入口：

```text
POST /medical/agent/chat
```

当 `is_initial = False` 时，进入追问分支。

### 9.1 API 层处理

调用链：

```text
context = chat_history_service.build_context(session_id, payload)
answer = medical_agent.chat_assess(payload, context.history, session_id=session_id)
record_user_message(session_id, payload)
record_assistant_message(session_id, answer)
return {"session_id": ..., "answer": ..., "timestamp": ...}
```

注意顺序：

1. 先 build_context。
2. 再调用 Agent。
3. 再写入当前用户消息。

这样做的含义是：当前用户输入通过 `user_input` 显式传入 Agent，不需要先写入历史；历史上下文只代表当前 query 之前的会话。

### 9.2 build_context() 构建追问上下文

位置：

```text
app/services/chat_history_service.py::build_context()
```

它会读取：

- 当前会话。
- 最近对话消息。
- 用户事实记忆。
- 最近两版诊断记忆。
- 会话摘要。

生成 history，顺序大致如下：

```text
1. 系统规则：历史仅供参考，当前输入优先
2. 用户事实记忆（如果不是初诊）
3. 最新诊断记忆（如果不是初诊）
4. 趋势记忆（如果有两版诊断）
5. 对话摘要记忆
6. 最近若干条 user/assistant 对话
7. 总长度裁剪
```

其中，事实记忆示例：

```text
用户事实记忆（确定性事实，仅参考）：exam_item: systolic_blood_pressure=176mmHg ...
```

诊断记忆示例：

```text
结构化诊断记忆（上轮确定性评估结果）：健康状态=high_risk；主要风险=高血压风险...
```

趋势记忆示例：

```text
诊断趋势：上一版 high_risk -> 当前 high_risk；风险变化...
```

这些 system message 会传入 Multi-Agent。

### 9.3 MedicalAssessmentAgent.chat_assess()

调用：

```text
self._multi_agent_supervisor.run(
    user_input,
    session_history,
    session_id=session_id,
)
```

返回：

```text
result.answer
```

追问通常不会返回 `structured_response`，因为它不是一次新的体检评估。

### 9.4 Supervisor 追问初始状态

```python
{
    "user_input": payload,
    "session_id": session_id,
    "session_history": context.history,
    "events": [],
    "safety_notes": [],
    "requires_safe_rewrite": False,
    "safety_revision_count": 0,
}
```

### 9.5 TriageAgent 判定追问

如果当前输入不是新的体检评估，且存在 session_id 或历史上下文，则：

```python
route = "followup"
```

事件：

```json
{
  "type": "agent_decision",
  "agent": "triage_agent",
  "action": "route_to_followup",
  "reason": "识别为基于既有会话的追问"
}
```

LangGraph 进入：

```text
memory_agent
```

## 10. MemoryAgent 内部实现

处理函数：

```text
MedicalMultiAgentSupervisor._memory_agent()
```

### 10.1 获取 session_history

MemoryAgent 调用：

```text
self._resolve_session_history(state)
```

逻辑：

1. 如果注入了 `memory_context_builder` 且存在 `session_id`：
   - 调用 `memory_context_builder(session_id, user_input)`。
   - 生产环境中这个函数就是 `chat_history_service.build_context`。
   - 如果返回对象有 `history` 且是 list，就使用它。
2. 如果没有注入或调用失败：
   - 回退使用 `state["session_history"]`。

这意味着 MemoryAgent 能主动刷新四层记忆，不完全依赖 API 层传入的旧 history。

### 10.2 提取 memory_text

调用：

```text
extract_memory_text_from_history(session_history)
```

它只提取 role 为 `system` 且包含以下关键词的消息：

```text
诊断记忆
用户事实记忆
异常指标
健康状态
主要风险
复查项目
趋势
对话摘要记忆
```

默认取最后 3 条相关记忆，拼接为 `memory_text`。

### 10.3 判断是否需要检索

MemoryAgent 有两个判断：

```text
has_relevant_memory = bool(memory_text) and _asks_personal_followup(user_input)
needs_retrieval = (not has_relevant_memory) or _requires_external_knowledge(user_input)
```

`_asks_personal_followup()` 关键词：

```text
我 / 上次 / 之前 / 指标 / 血压 / 血糖 / 肌酐 / eGFR / 风险 / 严不严重
```

`_requires_external_knowledge()` 关键词：

```text
标准 / 正常值 / 范围 / 指南 / 饮食 / 怎么吃 / 科普 / 原因 / 为什么 / 机制
```

因此：

| 用户追问 | 有诊断记忆 | 是否检索 | 原因 |
|----------|------------|----------|------|
| `我的血压风险严不严重？` | 有 | 否 | 个人风险问题，记忆足够 |
| `高血压早餐怎么吃？` | 有 | 是 | 饮食类，需要知识库证据 |
| `eGFR 正常范围是多少？` | 有或无 | 是 | 正常范围类，需要外部知识 |
| `为什么肌酐高？` | 有或无 | 是 | 原因/机制类，需要外部知识 |

MemoryAgent 输出：

```python
{
    "session_history": session_history,
    "memory_text": memory_text,
    "needs_retrieval": needs_retrieval,
    "events": events
}
```

事件：

```json
{
  "type": "agent_decision",
  "agent": "memory_agent",
  "action": "use_memory" 或 "need_retrieval",
  "reason": "..."
}
```

LangGraph 条件边：

```text
needs_retrieval == true  -> retrieval_agent
needs_retrieval == false -> synthesis_agent
```

## 11. RetrievalAgent 内部实现

处理函数：

```text
MedicalMultiAgentSupervisor._retrieval_agent()
```

### 11.1 构建追问检索 query

调用：

```text
_build_followup_retrieval_query(user_input, memory_text)
```

先从 `memory_text` 中提取疾病词：

```text
高血压
慢性肾脏病
糖尿病
糖尿病前期
血脂异常
高胆固醇
脂肪肝
高尿酸
痛风
```

如果没有疾病词：

```text
retrieval_query = user_input
```

如果有疾病词，则拼接疾病上下文和任务导向词：

| 用户输入包含 | 追加 |
|--------------|------|
| 饮食 / 怎么吃 / 早餐 / 晚餐 | `饮食管理 指南` |
| 复查 / 随访 / 科室 | `复查 随访 科室 指南` |
| 其他 | `风险管理 指南` |

示例：

```text
用户输入：早餐怎么吃？
memory_text：结构化诊断记忆：主要风险=高血压风险
retrieval_query：早餐怎么吃？ 高血压 饮食管理 指南
```

### 11.2 调用知识库工具

RetrievalAgent 调用：

```text
self._invoke_knowledge_tool(retrieval_query)
```

工具是：

```text
MedicalKnowledgeRetrievalTool
```

位置：

```text
app/services/agent_tools.py
```

它使用 LangChain `StructuredTool` 包装：

```text
lookup_medical_knowledge
```

底层调用：

```text
evidence_store.search(
    queries=[RetrievalQuery(label="agent_followup", text=query)],
    node_codes=[],
    top_k=top_k
)
```

返回格式是拼接后的证据文本：

```text
[1] 标题
来源类型: guideline
内容: ...

---

[2] 标题
来源类型: guideline
内容: ...
```

如果没有证据：

```text
No relevant medical evidence was found in the knowledge base.
```

### 11.3 RetrievalAgent 输出

写入 state：

```python
{
    "retrieval_query": retrieval_query,
    "evidence_text": evidence_text,
    "events": events
}
```

事件：

```json
{
  "type": "tool_call",
  "agent": "retrieval_agent",
  "name": "lookup_medical_knowledge",
  "args_summary": "{\"query\": \"...\"}"
}
```

以及：

```json
{
  "type": "tool_result",
  "agent": "retrieval_agent",
  "name": "lookup_medical_knowledge",
  "result_len": 1234
}
```

然后进入：

```text
synthesis_agent
```

## 12. SynthesisAgent 内部实现

处理函数：

```text
MedicalMultiAgentSupervisor._synthesis_agent()
```

### 12.1 普通合成

如果当前不是安全改写流程：

```text
answer = _compose_followup_answer(user_input, session_history, evidence_text)
```

`_compose_followup_answer()` 逻辑：

1. 如果 `chat_model` 存在：
   - 构建 prompt。
   - 调用 `chat_model.invoke(prompt)`。
   - 如果返回非空文本，使用 LLM 回答。
2. 如果 LLM 不可用或失败：
   - 调用 fallback：

```text
MedicalAssessmentAgent._build_followup_fallback_answer()
```

### 12.2 Synthesis prompt 内容

Prompt 包括：

```text
系统角色：医疗体检辅助评估系统中的回答合成 Agent
约束：只基于会话记忆和检索证据回答，不编造诊断，不替代医生处方
用户问题：user_input
会话上下文：最近 8 条 history，每条最多 500 字
检索证据：evidence_text
```

这让 LLM 只负责语言合成，不负责核心诊断判断。

### 12.3 fallback 回答

如果没有 LLM，fallback 会生成：

```text
基于当前会话中已有的体检评估和知识库资料，我给出如下辅助说明：

已参考的个人化上下文：
...

检索到的医学资料：
...

以上内容不能替代医生面诊...
```

fallback 使用同一个 `extract_memory_text_from_history()` 提取记忆，避免与 MemoryAgent 记忆规则不一致。

### 12.4 安全改写合成

如果 SafetyReviewAgent 设置过：

```text
requires_safe_rewrite = True
```

SynthesisAgent 不再调用 LLM，而是调用：

```text
_build_safe_followup_answer()
```

生成安全回答：

```text
关于“用户问题”，当前只能给出体检辅助层面的通用说明。
涉及药物名称、剂量、停药、换药或加减药时，应以医生面诊和处方为准，不要自行调整。
...
```

然后设置：

```python
requires_safe_rewrite = False
safety_revision_count += 1
```

避免无限改写。

## 13. SafetyReviewAgent 内部实现

处理函数：

```text
MedicalMultiAgentSupervisor._safety_review_agent()
```

输入：

```text
answer
user_input
structured_response
safety_notes
safety_revision_count
```

### 13.1 初诊高风险复核

如果初诊有结构化结果，且：

```text
structured_response.secondary_recommendations.human_review_required == True
```

则追加：

```text
高风险或复杂结果需要医生复核
```

### 13.2 用药风险检测

检测两部分：

```text
用户是否询问用药调整
回答是否包含用药建议
```

用户关键词：

```text
用药 / 药量 / 停药 / 换药 / 加药 / 减药 / 处方
```

回答关键词：

```text
用药 / 药物 / 服用 / 口服 / 停药 / 换药 / 加药 / 减药 / 处方 / 剂量
```

如果命中，会追加：

```text
涉及用药调整，需提示结合医生意见
```

### 13.3 剂量检测

正则检测：

```text
\d+(\.\d+)?\s*(mg|g|片|粒|次/日|次每天|毫克|克)
```

例如：

```text
50mg
1片
2次/日
```

如果命中，会追加：

```text
回答中疑似包含具体剂量或服药频次，需改写为非处方建议
```

### 13.4 疑似药名检测

正则检测疑似药名或治疗词：

```text
...片 / ...胶囊 / ...颗粒 / ...沙坦 / ...他汀 / ...洛尔 / ...地平 / ...普利 / 二甲双胍
```

如果命中，会追加：

```text
回答中出现需复核的药物/治疗词：...
```

### 13.5 是否触发安全改写

如果：

```text
(dosage_risk or suspected_drugs) and safety_revision_count == 0
```

则：

```text
requires_safe_rewrite = True
```

SafetyReviewAgent 不输出 final_answer，LangGraph 条件边进入：

```text
synthesis_agent
```

如果不需要改写，则输出：

```json
{
  "type": "final_answer",
  "agent": "safety_review_agent",
  "content": answer
}
```

## 14. 流式追问流程

追问流式与初诊流式使用同一个入口：

```text
POST /medical/agent/chat/stream
```

不同点在于 TriageAgent 会路由到 `followup`。

事件顺序通常是：

```text
meta
agent_decision: triage_agent route_to_followup
agent_decision: memory_agent use_memory / need_retrieval
tool_call: retrieval_agent              # 如果需要检索
tool_result: retrieval_agent            # 如果需要检索
agent_synthesizing: synthesis_agent
agent_decision: safety_review_agent
content chunks
done
```

如果 SafetyReviewAgent 发现剂量或疑似药名：

```text
agent_decision: safety_review_agent rewrite_required
agent_synthesizing: synthesis_agent      # 安全改写
agent_decision: safety_review_agent approved_with_notes
content chunks
done
```

API 层会收集最终 `assistant_content`，但追问没有 `structured_result`，因此只写入对话记忆，不写入新的诊断记忆。

## 15. 初诊和追问的数据传递对比

| 阶段 | 初诊 | 追问 |
|------|------|------|
| API 判断 | `looks_like_initial_assessment=True` | `False` |
| Supervisor route | `assessment` | `followup` |
| 主处理模块 | `MedicalKAGWorkflow` | `MemoryAgent/RetrievalAgent/SynthesisAgent` |
| 是否解析指标 | 是 | 通常否 |
| 是否执行规则引擎 | 是 | 否 |
| 是否查图谱风险 | 是 | 否，除非未来扩展 |
| 是否检索知识库 | 异常时检索证据 | 视问题是否需要外部知识 |
| 是否生成结构化结果 | 是 | 否 |
| 是否写事实记忆 | 是 | 否 |
| 是否写诊断记忆 | 是 | 否 |
| 是否写对话记忆 | 是 | 是 |
| 是否安全复核 | 是 | 是 |

## 16. 关键状态对象如何流转

### 16.1 初诊状态流转

```text
raw_text
  -> MedicalMultiAgentState.user_input
  -> AssessmentAgent
  -> InternalAssessmentState.raw_input
  -> normalized_exam_json
  -> detected_states
  -> risk_candidates
  -> intervention_candidates
  -> retrieval_queries
  -> evidence_chunks
  -> primary_diagnosis
  -> secondary_recommendations
  -> MedicalAssessmentResponse
  -> MedicalMultiAgentState.structured_response
  -> answer
  -> API response + memory write
```

### 16.2 追问状态流转

```text
user_input + session_id
  -> ChatHistoryService.build_context()
  -> session_history
  -> MedicalMultiAgentState
  -> MemoryAgent extracts memory_text
  -> needs_retrieval?
  -> RetrievalAgent builds retrieval_query
  -> evidence_text
  -> SynthesisAgent builds answer
  -> SafetyReviewAgent validates answer
  -> final_answer
  -> API response + conversation memory
```

## 17. 四个具体 query 示例

### 17.1 示例一：初诊 query

输入：

```text
男，52岁，血压176/108 mmHg，肌酐128 umol/L，eGFR 54，请判断健康状况。
```

处理过程：

```text
1. API 判断为初诊
2. TriageAgent route_to_assessment
3. Workflow 解析出性别、年龄、血压、肌酐、eGFR
4. Normalizer 补全指标 code
5. RuleEngine 命中血压和肾功能异常状态
6. GraphStore 查到高血压风险、慢性肾脏病风险
7. QueryPlanner 生成高血压、慢性肾病、异常指标相关 query
8. EvidenceStore 检索指南和证据片段
9. RiskRanker 计算风险 final_score
10. DiagnosisFormatter 生成主诊断和二级建议
11. SafetyReviewAgent 标记高风险需医生复核
12. API 写入事实记忆和诊断记忆
```

输出包括：

```text
自然语言 answer
structured_result: MedicalAssessmentResponse
```

### 17.2 示例二：个人风险追问

输入：

```text
我的血压风险严不严重？
```

处理过程：

```text
1. API 判断不是初诊
2. ChatHistoryService 构建上下文，注入诊断记忆
3. TriageAgent route_to_followup
4. MemoryAgent 发现有“高血压风险”诊断记忆
5. 用户问题是个人风险问题，不含外部知识关键词
6. needs_retrieval = false
7. 跳过 RetrievalAgent
8. SynthesisAgent 根据记忆回答
9. SafetyReviewAgent 审核通过
10. API 写入对话记忆
```

### 17.3 示例三：饮食追问

输入：

```text
早餐怎么吃？
```

处理过程：

```text
1. API 判断不是初诊
2. MemoryAgent 提取诊断记忆中的“高血压”
3. 问题包含“怎么吃”，requires_external_knowledge = true
4. RetrievalAgent 构造 query：
   早餐怎么吃？ 高血压 饮食管理 指南
5. MedicalKnowledgeRetrievalTool 检索知识库
6. SynthesisAgent 结合证据回答
7. SafetyReviewAgent 审核通过
8. API 写入对话记忆
```

### 17.4 示例四：用药追问

输入：

```text
我的药量要不要调？
```

处理过程：

```text
1. TriageAgent route_to_followup
2. MemoryAgent 判断需要检索或使用记忆
3. SynthesisAgent 生成回答
4. SafetyReviewAgent 检测“药量”
5. 如果回答中出现具体剂量或疑似药名：
   requires_safe_rewrite = true
6. LangGraph 回到 SynthesisAgent
7. SynthesisAgent 生成安全改写：
   不给具体剂量，不建议自行增减药，提示医生面诊
8. SafetyReviewAgent 再次审核并输出 final_answer
```

## 18. 为什么这样设计

### 18.1 初诊必须确定性

初诊涉及真实体检数值和风险判断，不能让 LLM 直接根据自然语言自由判断。因此使用：

```text
Parser -> Normalizer -> RuleEngine -> Graph -> Retrieval -> Ranker -> Formatter
```

保证每个判断都有可追踪依据。

### 18.2 追问必须灵活

追问形式不固定，可能是：

- 个人风险解释
- 饮食建议
- 复查建议
- 指标含义
- 趋势对比
- 用药边界

因此使用 Multi-Agent，让 MemoryAgent、RetrievalAgent、SynthesisAgent、SafetyReviewAgent 按需协作。

### 18.3 安全复核必须能改变流程

医疗问答中，单纯在回答末尾加免责声明不够。当前设计让 SafetyReviewAgent 可以触发控制流回退：

```text
SafetyReviewAgent -> SynthesisAgent -> SafetyReviewAgent
```

这比“只标记风险”更符合医疗场景的安全要求。

## 19. 阅读代码时的建议顺序

如果后续需要看代码，建议按下面顺序：

```text
1. app/api/routes/medical.py
2. app/services/medical_agent.py
3. app/agents/medical_multi_agent.py
4. app/workflows/medical_kag_pipeline.py
5. app/schemas/exam.py
6. app/services/chat_history_service.py
7. app/services/input_parser.py
8. app/services/rules.py
9. app/graph/store.py
10. app/retrieval/evidence_store.py
```

其中 1-4 足以理解 query 主链路，5-10 用于理解具体数据和底层能力。

## 20. 总结

用户 query 的处理可以概括为：

```text
API 接收 query
  -> 判断初诊/追问
  -> Multi-Agent Supervisor 路由
  -> 初诊走确定性 KAG Workflow
  -> 追问走记忆/检索/合成链路
  -> SafetyReviewAgent 做医疗边界控制
  -> 返回回答
  -> 写入对话、事实和诊断记忆
```

初诊和追问的最大区别是：

```text
初诊：从体检数值生成结构化诊断结果
追问：从历史诊断记忆和知识库证据生成上下文回答
```

整个实现的核心是把用户 query 处理拆成了可解释、可测试、可观察的工程链路，而不是把所有逻辑交给一个黑盒 LLM。
