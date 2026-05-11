# 初诊与追问流程 — 实现细节全解

> 目标：不看代码即可完全掌握用户 query 从进入系统到返回回答的全链路实现细节。本文档聚焦于两个核心流程的每一步内部实现、数据传递与状态变化。

---

## 第一部分：系统入口 — 请求如何到达 Agent

### 1.1 两个对外入口

```
入口 A（同步）: POST /medical/agent/chat         → MedicalAssessmentAgent.chat_assess()
入口 B（流式）: POST /medical/agent/chat/stream  → MedicalAssessmentAgent.stream_assess_async()
```

两个入口的请求处理逻辑在 `app/api/routes/medical.py` 中，共同模式为：

```text
1. 提取请求体 → 纯文本字符串
2. 获取或生成 session_id
3. 查找或创建 ChatSession（PostgreSQL）
4. 调用 ChatHistoryService.build_context(session_id, user_input) → 构建会话上下文
5. 调用 MedicalAssessmentAgent 的对应方法
6. 写入用户消息 → 写入助手消息 → 更新事实记忆 → 追加诊断版本
```

**传入 Agent 的关键参数**：

| 参数 | 初诊时 | 追问时 |
|------|--------|--------|
| `raw_text` / `user_input` | "男，52岁，血压176/108..." | "我的血压严重吗" |
| `session_history` | [] (空列表) | dict 列表，由 ChatHistoryService 构建 |
| `session_id` | 新生成的 UUID | 当前会话的 session_id |

---

## 第二部分：初诊流程 — 逐步骤实现细节

### 2.1 判断分流：如何确定这是初诊

不论是同步还是流式，第一步都是判断 user_input 属于初诊还是追问。入口在 `MedicalAssessmentAgent._looks_like_initial_assessment()`。

**判断逻辑**（`medical_agent.py:207-214`）：

```python
def _looks_like_initial_assessment(user_input: str) -> bool:
    lowered = user_input.lower()
    keywords = ["血压", "mmhg", "egfr", "肌酐", "空腹血糖", "hba1c", "体检", "指标", "化验"]
    has_medical_keyword = any(keyword in lowered for keyword in keywords)
    has_numeric_value = bool(re.search(r"\d+(\.\d+)?", lowered))
    explicit_initial_intent = any(token in lowered
        for token in ["请判断", "评估一下", "体检报告", "化验单"])
    return has_medical_keyword and (has_numeric_value or explicit_initial_intent)
```

**判定规则**：
- 必须同时满足两个条件才判为初诊：
  1. 包含体检指标关键词（血压/eGFR/肌酐/空腹血糖/HbA1c 等）
  2. 包含数值 或 包含明确评估意图词（请判断/评估一下/体检报告/化验单）

**举例**：
- "男，52岁，血压 176/108" → 有关键词 + 有数值 → **初诊**
- "请判断我的体检报告" → 有关键词（体检）+ 有意图词 → **初诊**
- "我的血压怎么样了" → 有关键词但无数值且无意图词 → **追问**（这是设计的关键：避免个人对话被误判为初诊）

### 2.2 初诊同步路径：MedicalMultiAgentSupervisor.run()

```text
路由层 → MedicalAssessmentAgent.assess(raw_text)
       → _multi_agent_supervisor.run(raw_text, session_history=[])
```

#### 2.2.1 run() 方法（medical_multi_agent.py:127-149）

```text
输入: user_input="男，52岁，血压176/108 mmHg，肌酐128..."
     session_history=[]
     session_id=None

1. 初始化 MedicalMultiAgentState:
   {
     "user_input": "男，52岁，血压176/108 mmHg，肌酐128...",
     "session_id": None,
     "session_history": [],
     "events": [],
     "safety_notes": [],
     "requires_safe_rewrite": False,
     "safety_revision_count": 0,
   }

2. 调用 self._graph.invoke(initial_state)
   → LangGraph 根据图拓扑依次执行:
     triage_agent → assessment_agent → safety_review_agent → END

3. 提取 final_state 中的 answer、events、structured_response
   返回 MedicalMultiAgentRunResult
```

**关键细节**：`run()` 走 LangGraph 的 `invoke()` 方法，这走的是图构建时定义的同步执行路径，6 个 Agent 按条件边依次执行、共享同一个 state dict。

#### 2.2.2 TriageAgent 节点执行

```text
_triage_agent(state) → 更新 state 的 route 字段
```

内部逻辑：

```text
1. 从 state 取 user_input 和 session_history
2. 调用 self._initial_assessment_detector(user_input)
   → 也就是 MedicalAssessmentAgent._looks_like_initial_assessment()
3. 判定条件: detector 返回 True
             OR (session_history 为空 AND session_id 为空)
   → route = "assessment"
4. 否则 route = "followup"
5. 追加事件: agent_decision(triage_agent, route_to_assessment)
6. 返回 {"route": "assessment", "events": [新事件]}
   → state 被更新
```

**数据传递**：triage_agent 返回的 dict 被 LangGraph 合并入共享 state。关键变更：`state["route"] = "assessment"`。

#### 2.2.3 路由决策：_route_after_triage()

```text
读取 state["route"]
  → "assessment" → 下一个节点是 assessment_agent
  → "followup"   → 下一个节点是 memory_agent
```

#### 2.2.4 AssessmentAgent 节点执行

```text
_assessment_agent(state) → 执行 KAG Workflow，更新 state
```

内部逻辑：

```text
1. 追加事件: agent_thinking(assessment_agent, "正在执行确定性 KAG 体检评估流水线")

2. 调用 self._workflow.run_state(state["user_input"])
   → 这启动了 12 节点确定性流水线（见下文 2.3 节）
   → 返回 InternalAssessmentState（包含完整的评估链路中间产物）

3. 从 assessment_state.response 获取 MedicalAssessmentResponse
   → 这是流水线的最终结构化产出

4. 调用 self._assessment_answer_builder(response)
   → 这实际上是 MedicalAssessmentAgent._compose_answer()
   → 流式路径会调用 LLM 生成自然语言，同步路径先用 LLM 兜底再用 fallback
   → 生成中文回答文本

5. 追加事件: agent_decision(assessment_agent, workflow_completed,
                           "完成评估，识别 N 个主要风险")

6. 返回 {
     "assessment_state": assessment_state,     # InternalAssessmentState
     "structured_response": response,          # MedicalAssessmentResponse
     "answer": "健康状态：高风险。紧急程度：紧急...",
     "events": [新事件],
   }
```

**数据传递的关键**：`assessment_state` 是 `InternalAssessmentState`，包含了流水线 12 步的全部中间产物（normalized_exam_json、detected_states、risk_candidates、evidence_chunks 等）。这些在追问阶段不会被用到，但被保留在 state 中供外部使用。

#### 2.2.5 SafetyReviewAgent 节点执行

```text
_safety_review_agent(state) → 医疗安全复核，可能触发安全改写
```

内部逻辑：

```text
1. 从 state 取 answer 和 structured_response

2. 第一优先级检查：structured_response 是否存在
   → 如果 structured_response.secondary_recommendations.human_review_required == True
   → notes 追加 "高风险或复杂结果需要医生复核"

3. 第二优先级检查：用药风险
   medication_risk = _mentions_medication_adjustment(user_input)   # 用户问了什么
                     OR _contains_medication_advice(answer)          # 回答说了什么
   → 检查关键词: "用药" "服药" "停药" "换药" "处方" "剂量" 等

4. 第三优先级检查：剂量风险
   dosage_risk = _contains_dosage_instruction(answer)
   → 正则: \d+(\.\d+)?\s*(mg|g|片|粒|次/日|毫克|克)
   → 检测 "50mg" "1片" "次/日" 等

5. 第四优先级检查：疑似药名
   suspected_drugs = _extract_suspected_drug_terms(answer)
   → 正则: 中英文{2,12}字+(片|胶囊|颗粒|沙坦|他汀|洛尔|地平|普利|二甲双胍)
   → 检测 "氨氯地平片" "阿托伐他汀" 等

6. 安全决策:
   IF (dosage_risk OR suspected_drugs) AND safety_revision_count == 0:
       requires_rewrite = True   → 需要安全改写
   ELIF medication_risk AND "医生" not in answer:
       answer += "\n涉及用药调整时，请结合医生意见处理..."
       requires_rewrite = False

7. 追加事件: agent_decision(safety_review_agent,
                           rewrite_required | approved_with_notes | approved)
   → 仅当 requires_rewrite=False 时追加 final_answer 事件

8. 返回 {
     "answer": answer,
     "safety_notes": notes,
     "requires_safe_rewrite": requires_rewrite,
     "events": [新事件],
   }
```

**关键设计**：安全复核在初诊路径上通常只会追加人工复核提示（human_review_required=true），不会触发安全改写。安全改写主要针对追问中 LLM 生成的回答。

#### 2.2.6 安全复核后的路由决策

```text
_route_after_safety(state):
  → requires_safe_rewrite == True → "rewrite" → synthesis_agent
  → requires_safe_rewrite == False → "end" → END
```

#### 2.2.7 同步 run() 返回后的处理

回到 `MedicalAssessmentAgent.assess()`（medical_agent.py:72-77）：

```text
result = self._multi_agent_supervisor.run(raw_text, [])
  → result.answer: "健康状态：高风险。紧急程度：紧急。主要风险判断..."
  → result.structured_response: MedicalAssessmentResponse 实例
     (如果初诊路径正常执行，这里一定有值)

路由层（routes/medical.py agent_chat 方法）继续处理：
  1. chat_history_service.record_user_message(session_id, raw_text)
  2. chat_history_service.upsert_user_fact_memory(session_id, structured.normalized_exam_json)
     → 从 structured_result 中取出 normalized_exam_json
     → 写入/更新 UserFactMemory 表（性别/年龄/每项指标值/病史/用药/过敏）
     → 冲突检测：如果旧值存在且不同，生成 "已更新事实" 提示
  3. chat_history_service.record_assistant_message(session_id, answer, structured.model_dump())
     → 写入 ConversationMemory（assistant 角色）
     → 追加 DiagnosticMemory 新版本
     → 更新 conversation_summary
     → 更新 session.updated_at
```

### 2.3 KAG Workflow 12 节点详细实现

这是初诊流程的核心引擎。由 `MedicalKAGWorkflow.run_state()` 启动。

#### 2.3.1 执行入口

```text
run_state(raw_input) → _execute_step_sequence(raw_input) → 12 个 PipelineStep 依次执行
```

`_execute_step_sequence()` 的逻辑（pipeline.py:172-178）：

```text
1. 创建初始 WorkflowState: {"state": InternalAssessmentState(raw_input=raw_input)}
2. 对每个 step in _step_definitions():
   a. 检查 _should_skip_step(step.name, state) → 是否跳过
   b. 如果不跳过: step.handler(payload) → 执行节点
   c. 返回的 dict 合并入 payload
3. 返回最终 payload
```

**关键设计**：每个 step.handler 的签名是 `(WorkflowState) -> WorkflowState`，即输入和输出都是 `{"state": InternalAssessmentState}`。节点之间通过 payload 中的 state 字段传递所有中间数据。

#### 2.3.2 节点 1: parse_raw_input — 输入解析

```text
处理器: _parse_raw_input_node(payload)
文件: app/services/input_parser.py
```

内部实现逐步展开：

**第一步：类型判断**
```text
MedicalInputParser.parse(raw_input)
  ├── 如果是 NormalizedMedicalExamJSON 实例 → 直接归一化返回
  ├── 如果是 dict → _parse_dict_payload()
  │    ├── 有 "input_data" 键 → 递归解析 input_data
  │    ├── 有 "normalized_exam_json" 键 → 递归解析
  │    └── 直接提取 patient_profile / exam_items / medical_history 等字段
  └── 如果是 str → 文本解析
```

**第二步：文本解析策略 — 双通道**
```text
_parse_text_payload(text)
  → 第一通道: LLM 结构化抽取
    如果 self._extractor is not None:
      try:
        extracted = self._extractor.extract_structured(text, ExtractedExamPayload)
        → 这调用 ChatTongyi.with_structured_output(ExtractedExamPayload)
        → LLM 从自然语言中抽取 patient_profile(sex/age)、exam_items、病史、用药、过敏
        → 如果成功，直接返回 NormalizedMedicalExamJSON
      except Exception:
        logger.warning("llm_extractor_failed")
        → 回退第二通道

  → 第二通道: 正则解析
    1. IndicatorNormalizer.extract_from_text(text) → 逐指标正则匹配
       例如: 血压模式: r"(血压|BP)\s*[:：]?\s*(\d{2,3})\s*/\s*(\d{2,3})"
            血糖模式: r"(空腹血糖|FBG)\s*[:：]?\s*([0-9]+(?:\.[0-9]+)?)"
       每个匹配生成一个 ExamItem(name, value, unit, source_text)

    2. 提取性别: "男" / "male" → sex="male"
                "女" / "female" → sex="female"

    3. 提取年龄: r"(\d{1,3})\s*岁"

    4. 提取病史: _extract_list(text, ["既往史", "病史"])
       通过正则从 "既往史：高血压" 中提取 "高血压"

    5. 提取用药: _extract_list(text, ["用药史", "用药"])
    6. 提取过敏: _extract_list(text, ["过敏史", "过敏"])
```

**第三步：去重**
```text
IndicatorNormalizer.extract_from_text() 的最后一步:
  → 按 (name, value, unit) 三元组去重
  → 同一指标在同一文本中多次出现只保留第一次
```

**输出到 state**：
```text
state.normalized_exam_json → NormalizedMedicalExamJSON 实例
  包含: patient_profile(sex="male", age=52)
        exam_items=[ExamItem(name="收缩压", value=176, unit="mmHg"), ...]
        medical_history=["高血压"]
        current_medications=["缬沙坦"]
        allergies=[]
        user_question="男，52岁，血压176/108 mmHg..."
        source_type="text"

state.missing_fields → ["user_question"] 或其他缺失项
state.warnings → ["Too few exam items..."] 或其他警告
```

#### 2.3.3 节点 2: validate_exam_json — 结构校验

```text
处理器: _validate_exam_json_node(payload)
```

逻辑非常简单：
```python
if state.normalized_exam_json is None:
    raise ValueError("normalized_exam_json is required after parsing")
```

唯一目的是在后续节点使用 `state.normalized_exam_json` 之前保证它不为空。

#### 2.3.4 节点 3: normalize_exam_items — 指标归一化

```text
处理器: _normalize_exam_items_node(payload)
文件: app/services/indicator_normalizer.py
```

对 `exam_json.exam_items` 中的每个 ExamItem 执行三层转换：

**第一层：别名映射**
```text
raw_name = "收缩压"
  → INDICATOR_ALIASES["收缩压"] = ("blood_pressure_systolic", "收缩压", "mmHg")
  → code = "blood_pressure_systolic"
  → canonical_name = "收缩压"
  → default_unit = "mmHg"
```

**第二层：单位标准化**
```text
normalize_unit("mmol/l") → "mmol/L"
normalize_unit("μmol/l") → "umol/L"
normalize_unit("mg/dl") → "mg/dL"
```

**第三层：单位换算**
```text
convert_value_and_unit(code, value, unit, default_unit):
  → code == "fasting_blood_glucose" AND unit == "mg/dL"
    → value = value / 18.0, unit = "mmol/L"
  → code == "creatinine" AND unit == "mg/dL"
    → value = value * 88.4, unit = "umol/L"
```

**输出到 state**：
```text
state.normalized_exam_json → 更新后的版本
  exam_items 中每一项的 code/name/unit 都已校正
```

#### 2.3.5 节点 4: detect_indicator_states — 规则判定

```text
处理器: _detect_indicator_states_node(payload)
文件: app/services/rules.py + app/config/medical_rules.json
```

内部实现：

```text
IndicatorRuleEngine.detect_states(exam_json):

第一步: 构建指标值索引
  values_by_code = {
    "blood_pressure_systolic": 176,
    "blood_pressure_diastolic": 108,
    "creatinine": 128,
    "egfr": 54,
    ...
  }

第二步: 遍历 exam_items，对每个 item 匹配单指标规则
  遍历 self._single_rules (从 JSON 加载的规则列表):

  item.code == "blood_pressure_systolic", value=176
    → 匹配规则 bp_sbp_stage2: indicator_code="blood_pressure_systolic", operator="gte", threshold=160
    → _compare(176, "gte", 160) → True
    → _build_single_state(rule, item) → DetectedState(
        indicator_code="blood_pressure_systolic",
        indicator_name="收缩压",
        state_code="SBP_high_stage2",
        label="收缩压显著升高",
        severity="high",
        value=176,
        unit="mmHg",
        rule_id="bp_sbp_stage2"
      )

  item.code == "creatinine", value=128, patient_profile.sex="male"
    → 匹配规则 creatinine_high:
      conditions: [{logic: OR, conditions: [
        {AND: [sex="male", value>=110]},
        {AND: [sex="female", value>=90]},
        {AND: [sex not in [male,female], value>=100]}
      ]}]
    → 进入 _conditions_match() 递归:
      第一层 conditions (AND): 只有一个子条件 (OR)
      第二层 sub-conditions (OR):
        sub-condition[0] (AND): sex=="male" → True, value=128 >= 110 → True
        → AND 全部 True → OR 返回 True
    → 匹配成功 → DetectedState(state_code="CREATININE_high", severity="high")

第三步: 遍历组合规则
  遍历 self._composite_rules:
  rule BP_stage2_combined:
    conditions: [
      {indicator_code="blood_pressure_systolic", operator="gte", value=160},
      {indicator_code="blood_pressure_diastolic", operator="gte", value=100}
    ]
    → values_by_code["blood_pressure_systolic"]=176 >= 160 → True
    → values_by_code["blood_pressure_diastolic"]=108 >= 100 → True
    → AND → True
    → DetectedState(state_code="BP_stage2_combined", severity="high")

  rule CKD_strong_combined:
    conditions: [
      {indicator_code="egfr", operator="lt", value=60},
      {logic: OR, conditions: [...性别分层肌酐阈值...]}
    ]
    → egfr=54 < 60 → True
    → creatinine 性别分层 → sex="male", 128>=110 → True
    → AND → True
    → DetectedState(state_code="CKD_strong_combined", severity="high")
```

**输出到 state**：
```text
state.detected_states = [
  DetectedState(SBP_high_stage2, severity="high"),
  DetectedState(DBP_high_stage2, severity="high"),
  DetectedState(CREATININE_high, severity="high"),
  DetectedState(eGFR_moderately_low, severity="high"),
  DetectedState(BP_stage2_combined, severity="high"),
  DetectedState(CKD_strong_combined, severity="high"),
]
```

#### 2.3.6 节点 5: retrieve_graph_candidates — 图谱检索

```text
处理器: _retrieve_graph_candidates_node(payload)
文件: app/graph/store.py
```

```text
1. 提取 state_codes = [s.state_code for s in state.detected_states]
   → ["SBP_high_stage2", "DBP_high_stage2", "CREATININE_high",
      "eGFR_moderately_low", "BP_stage2_combined", "CKD_strong_combined"]

2. 调用 graph_store.get_risk_candidates(state_codes)

3. Neo4j 执行 Cypher:
   MATCH (s:IndicatorState)-[:STATE_IMPLIES_RISK]->(r:DiseaseRisk)
         -[:RISK_RELATED_DISEASE]->(d:Disease)
   WHERE s.state_code IN $state_codes
   RETURN s.state_code, r.risk_code, r.name, r.risk_level,
          d.disease_code, d.name

4. 结果行:
   SBP_high_stage2 → hypertension_risk → hypertension
   DBP_high_stage2 → hypertension_risk → hypertension   (汇聚到同一条)
   CREATININE_high → ckd_risk → ckd
   eGFR_moderately_low → ckd_risk → ckd               (汇聚到同一条)
   BP_stage2_combined → hypertension_risk → hypertension
   CKD_strong_combined → ckd_risk → ckd

5. 聚合去重:
   用 (risk_code, disease_code) 作为 key 合并相同风险
   → RiskCandidate(risk_code="hypertension_risk", risk_name="高血压风险",
                    disease_code="hypertension", disease_name="高血压",
                    risk_level="high", graph_score=0.98,
                    supported_states=["SBP_high_stage2","DBP_high_stage2","BP_stage2_combined"])
   → RiskCandidate(risk_code="ckd_risk", risk_name="慢性肾病风险",
                    disease_code="ckd", disease_name="慢性肾病",
                    risk_level="high", graph_score=0.96,
                    supported_states=["CREATININE_high","eGFR_moderately_low","CKD_strong_combined"])

6. 按 graph_score 降序排列

7. 每个 RiskCandidate 包含 graph_paths:
   GraphPath(path_type="risk", nodes=["SBP_high_stage2","hypertension_risk","hypertension"], score=0.95)
   GraphPath(path_type="risk", nodes=["DBP_high_stage2","hypertension_risk","hypertension"], score=0.90)
   ...
```

**输出到 state**：
```text
state.risk_candidates = [RiskCandidate(hypertension), RiskCandidate(ckd)]
```

#### 2.3.7 节点 6: expand_intervention_paths — 路径扩展

```text
处理器: _expand_intervention_paths_node(payload)
```

```text
1. 提取 disease_codes = [r.disease_code for r in state.risk_candidates]
   → ["hypertension", "ckd"]

2. 调用 graph_store.get_intervention_candidates(disease_codes)

3. Neo4j 执行:
   MATCH (d:Disease) WHERE d.disease_code IN $disease_codes
   OPTIONAL MATCH (d)-[:DISEASE_RECOMMENDS_INTERVENTION]->(i:Intervention)
   OPTIONAL MATCH (d)-[:DISEASE_RECOMMENDS_DEPARTMENT]->(dep:Department)
   OPTIONAL MATCH (d)-[:DISEASE_REQUIRES_FOLLOWUP_TEST]->(f:FollowUpTest)
   OPTIONAL MATCH (d)-[:DISEASE_HAS_CONTRAINDICATION]->(c:Contraindication)
   OPTIONAL MATCH (d)-[:DISEASE_RECOMMENDS_MEDICATION_DIRECTION]->(m:MedicationDirection)
   RETURN d.disease_code,
          collect(DISTINCT i.name) AS interventions,
          collect(DISTINCT m.name) AS medication_directions,
          collect(DISTINCT c.name) AS contraindications,
          collect(DISTINCT f.name) AS follow_up_tests,
          collect(DISTINCT dep.name) AS departments

4. 结果:
   hypertension → interventions: ["限盐","减重","规律运动","居家血压监测"]
                  medication: ["评估是否需要启动降压治疗","优先结合心内科评估 ACEI/ARB"]
                  contraindications: []
                  follow_up: ["动态血压监测","肾功能复查","尿常规"]
                  departments: ["心内科","全科医学科"]

   ckd → interventions: ["控制血压","避免肾毒性药物","低盐饮食"]
          medication: ["合并高血压时可由医生评估 ACEI/ARB","肾功能下降时按 eGFR 调整"]
          contraindications: ["肾功能不全时部分药物需要减量或禁用"]
          follow_up: ["肌酐复查","eGFR 复查","尿蛋白评估","肾脏超声"]
          departments: ["肾内科"]

5. 生成 InterventionCandidate 对象，每条包含:
   - 上述干预、用药、禁忌、复查、科室列表
   - graph_paths: 每个关系对应一条 GraphPath (path_type="intervention"/"department"等)
```

**输出到 state**：
```text
state.intervention_candidates = [
  InterventionCandidate(disease_code="hypertension", interventions=[...], ...),
  InterventionCandidate(disease_code="ckd", interventions=[...], ...),
]
```

#### 2.3.8 节点 7: plan_evidence_queries — 查询规划

```text
处理器: _plan_evidence_queries_node(payload)
文件: app/services/evidence_query_planner.py
```

```text
EvidenceQueryPlanner.build_queries(exam_json, detected_states, risks):

生成查询（去重）:
  1. user_question → 用户原始问题文本

  2. 对前 3 个风险生成:
     "高血压 高血压风险 成人 体检 指南 风险识别 干预建议 复查"
     "慢性肾病 慢性肾病风险 成人 体检 指南 风险识别 干预建议 复查"

  3. 对前 3 个风险生成随访查询:
     "高血压 复查建议 随访 科室 用药禁忌"
     "慢性肾病 复查建议 随访 科室 用药禁忌"

  4. 对前 4 个异常状态生成:
     "收缩压 收缩压显著升高 风险 分层 指南"
     "舒张压 舒张压显著升高 风险 分层 指南"
     ...

  5. 指标聚合:
     "收缩压 舒张压 肌酐 估算肾小球滤过率 异常 体检 风险评估"

→ 输出: [RetrievalQuery(label="user_question", text="..."),
          RetrievalQuery(label="risk_guideline", text="..."),
          ...]
  最多 8 个 RetrievalQuery
```

**输出到 state**：
```text
state.retrieval_queries = [RetrievalQuery×8]
```

#### 2.3.9 节点 8: retrieve_evidence_chunks — 证据检索

```text
处理器: _retrieve_evidence_chunks_node(payload)
文件: app/retrieval/evidence_store.py + lexical.py
```

这是整个流水线中最复杂的节点，展开五步：

**第一步：构建 node_codes 集合**
```text
node_codes = []
for risk in state.risk_candidates:
    node_codes.extend([risk.risk_code, risk.disease_code])
→ ["hypertension_risk", "hypertension", "ckd_risk", "ckd"]
```

**第二步：Milvus 向量检索（粗召回）**
```text
对每个 retrieval_query:
  1. self._embedder.embed(query.text) → 生成 query vector
  2. client.search(collection, data=[query_vector], top_k=top_k*4, output_fields=[...])
     → HNSW 索引检索
     → 返回: [{chunk_id, doc_id, title, text, linked_node_codes_json, ..., distance}]
  3. 所有 query 的召回结果合并 → candidate_map (按 chunk_id 去重)
```

**第三步：SQLite FTS5 词汇检索**
```text
对每个 chunk 在 FTS5 索引中做词汇检索:
  1. _build_match_query(query_text) → 将 query 拆为 tokens, 构建 FTS5 MATCH 表达式
  2. conn.execute("SELECT chunk_id, bm25(docs_fts) FROM docs_fts WHERE MATCH ?", ...)
  3. 将 bm25 分数转为正向分: 1.0 / (1.0 + abs(bm25_score))
  4. FTS5 匹配失败时 → _fallback_like_search() → LIKE 字符级匹配
```

**第四步：RRF 融合 + Rerank 前置排序**
```text
对每个候选 CandidateScore(chunk, graph_overlap_score, source_authority_score):

  # RRF 融合: 对 dense_ranks 和 lexical_ranks 按 RRF(k=60) 公式累加
  for query in queries:
    for rank, (chunk_id, dense_score) in enumerate(dense_ranks, 1):
      score.fusion_score += 1.0 / (60 + rank)     # 稠密路径贡献
    for rank, (chunk_id, lexical_score) in enumerate(lexical_ranks, 1):
      score.fusion_score += 1.0 / (60 + rank)     # 词汇路径贡献

  # Rerank 前置精排 (在 fusion 加权之前):
  ordered = sorted_by_rough_rank    # 粗排
  ordered = _apply_remote_rerank(ordered, queries, top_k)
    → 取 top 20 候选送入 Rerank API
    → Rerank 返回的 score 以 75% 权重融入 final_score
    → 无 Rerank 时跳过此步

  # 多信号融合 (在 Rerank 之后):
  final_score = 0.45 * rerank_norm        # Rerank 语义分
              + 0.20 * graph_overlap      # 图谱节点重叠度
              + 0.15 * lexical_norm       # 词汇匹配分
              + 0.10 * dense_norm         # 稠密向量分
              + 0.10 * source_authority   # 来源权威度
```

**第五步：质量门控 + MMR 去重**
```text
# 候选裁剪
ordered = ordered[:max(top_k * 3, mmr_candidate_limit)]
  → 默认 mmr_candidate_limit=15

# 质量门控
ordered = [item for item in ordered if _passes_relevance_gate(item)]
  通过条件: graph_overlap_score > 0 OR final_score >= 0.75

# MMR 多样性 (λ=0.75)
selected = _mmr_select(ordered, top_k=top_k, lambda_mult=0.75)
  算法: 每次从候选中选择 mmr_score 最大的
  mmr_score = 0.75 * relevance - 0.25 * max(similarity_to_each_selected)
  (相关性和多样性分别占 75% 和 25%)
```

**输出到 state**：
```text
state.evidence_chunks = [EvidenceChunk×5]  (top_k=5)
  每条 EvidenceChunk 包含:
    chunk_id, doc_id, title, text, linked_node_codes, source_type,
    dense_score, lexical_score, graph_overlap_score, source_authority_score,
    fusion_score, final_score, rerank_score, relevance_score, matched_queries
```

#### 2.3.10 节点 9: rank_medical_evidence — 风险排序

```text
处理器: _rank_medical_evidence_node(payload)
文件: app/retrieval/risk_ranker.py
```

```text
MedicalRiskRanker.rank_risks(risks, evidence_chunks, detected_states):

第一步: 构建证据分映射
  evidence_score_map = {}
  for chunk in evidence_chunks:
    for code in chunk.linked_node_codes:
      evidence_score_map[code] = max(existing, chunk.final_score)
  → {"hypertension_risk": 0.85, "hypertension": 0.78, "ckd_risk": 0.82, ...}

第二步: 构建状态严重度权重
  severity_weight = {"low": 0.4, "medium": 0.7, "high": 1.0}
  state_weight_map = {state.state_code: severity_weight[state.severity]
                      for state in detected_states}

第三步: 对每个 RiskCandidate 重新算分
  例: hypertension_risk/hypertension
    → evidence_score = max(evidence_score_map["hypertension_risk"],
                           evidence_score_map["hypertension"]) = 0.85
    → support_count = len(set(supported_states)) = 3 (SBP/DBP/BP_combined)
    → support_strength = sum(weight for each state) / support_count
      = (1.0+1.0+1.0) / 3 = 1.0
    → support_count_score = min(1.0, 3/3.0) = 1.0
    → base_graph_score = 0.98

    final_score = 0.45 * 0.98      # 图谱置信度
                + 0.20 * 0.85      # 证据支持度
                + 0.20 * 1.0       # 状态严重度
                + 0.15 * 1.0       # 支持状态数量
                = 0.441 + 0.17 + 0.20 + 0.15 = 0.961

第四步: 更新 RiskCandidate 的评分字段
  support_count, graph_support_score, evidence_support_score, final_score, graph_score

第五步: 按 final_score 降序排列
```

**输出到 state**：
```text
state.risk_candidates → 已更新 final_score 的排序后列表
```

#### 2.3.11 节点 10: generate_primary_diagnosis — 主诊断

```text
处理器: _generate_primary_diagnosis_node(payload)
文件: app/services/diagnosis_formatter.py
```

```text
DiagnosisFormatter.build_primary(exam_json, risks, abnormal_states):

计算 health_status 和 urgency_level:

  high_risk_count = 2 (两个风险都是 high)
  high_severity_state_count = 6 (四个单指标 + 两个组合 = 6 个 high 状态)
  top_score = 0.961

  IF high_risk_count >= 2 OR (high_risk_count >= 1 AND high_severity_state_count >= 2) OR top_score >= 0.9:
    urgency_level = "urgent"
    health_status = "high_risk"

  输出:
  PrimaryDiagnosis(
    health_status="high_risk",
    urgency_level="urgent",
    potential_risks=[RiskCandidate(hypertension, 0.961), RiskCandidate(ckd, 0.945)],
    key_abnormal_indicators=[DetectedState×6]
  )
```

**输出到 state**：
```text
state.primary_diagnosis = PrimaryDiagnosis(...)
```

#### 2.3.12 节点 11: generate_secondary_recommendation — 建议生成

```text
处理器: _generate_secondary_recommendation_node(payload)
```

```text
1. 调用 ranker.merge_recommendations(state.intervention_candidates)
   → 合并两个 InterventionCandidate (hypertension + ckd)
   → 去重: 各列表中重复项只保留一次
   → 得到合并后的 InterventionCandidate:
     interventions: ["限盐","减重","规律运动","居家血压监测","控制血压","避免肾毒性药物","低盐饮食"]
     departments: ["心内科","全科医学科","肾内科"]
     follow_up_tests: ["动态血压监测","肾功能复查","尿常规","肌酐复查","eGFR 复查","尿蛋白评估","肾脏超声"]
     ...

2. 调用 ranker.index_recommendations_by_disease(recommendations)
   → {"hypertension": InterventionCandidate(hypertension),
      "ckd": InterventionCandidate(ckd)}

3. 调用 formatter.build_secondary(risks, merged_reco, disease_map):
   → 遍历每个 RiskCandidate，匹配对应的 InterventionCandidate
   → 生成 DiseaseRecommendation 列表 (每个疾病一个)
   → SecondaryRecommendations(
       recommended_departments: merged 的 departments,
       follow_up_tests: merged 的 follow_up_tests,
       lifestyle_interventions: merged 的 interventions,
       medication_directions: merged 的 medication_directions,
       contraindications: merged 的 contraindications,
       recommendations_by_disease: [DiseaseRecommendation×2],
       human_review_required: True (因为存在 high 风险 > 1 且存在禁忌)
     )
```

**输出到 state**：
```text
state.secondary_recommendations = SecondaryRecommendations(...)
```

#### 2.3.13 节点 12: format_medical_response — 响应封装

```text
处理器: _format_medical_response_node(payload)
```

```text
1. 整理 graph_paths:
   遍历 state.risk_candidates 和 state.intervention_candidates
   → 提取所有 GraphPath

2. formatter.build_response(exam_json, primary, secondary, graph_paths, evidence_chunks):
   → MedicalAssessmentResponse(
       normalized_exam_json: 归一化后的体检数据,
       primary_diagnosis: 主诊断结果,
       secondary_recommendations: 二级建议,
       evidence: AssessmentEvidence(graph_paths, evidence_chunks)
     )
```

**输出到 state**：
```text
state.response = MedicalAssessmentResponse → 这是流水线的最终产出
```

#### 2.3.14 条件短路机制

在 `_execute_step_sequence()` 和 `iter_events()` 中，每个节点执行前都经过 `_should_skip_step()` 检查：

```text
条件一: 无异常状态 → 跳过图谱检索和证据检索
  step_name in {"retrieve_graph_candidates", "expand_intervention_paths",
                "plan_evidence_queries", "retrieve_evidence_chunks",
                "rank_medical_evidence"}
  AND state.detected_states 为空
  → return True (跳过)

条件二: 有异常但图谱未命中 → 跳过干预扩展
  step_name == "expand_intervention_paths"
  AND state.risk_candidates 为空
  → return True (跳过)

条件三: 有异常且图谱命中 → 继续执行全部节点
  所有 step 都不跳过
```

跳过的节点意味着 state 中对应的字段保持初始空值。后续 diagnosis_formatter 处理空值时能正常工作（输出 "healthy"）。

---

## 第三部分：追问流程 — 逐步骤实现细节

### 3.1 判断分流：如何确定这是追问

`_looks_like_initial_assessment("我的血压风险严重吗")` → 有关键词（血压）但无数值且无意图词 → 返回 False。

### 3.2 追问流式路径：MedicalMultiAgentSupervisor.iter_events()

```text
路由层 → MedicalAssessmentAgent.stream_assess_async(raw_text, session_history, session_id)
       → _multi_agent_supervisor.aiter_events(raw_text, session_history, session_id)
```

`aiter_events()` 是逐节点执行的 async generator（不是 `graph.invoke()` 完再产出事件）。

#### 3.2.1 初始化共享状态

```text
state = {
  "user_input": "我的血压风险严不严重？",
  "session_id": "session_abc123...",
  "session_history": [
    {"role": "system", "content": "上下文使用规则：历史内容仅作参考..."},
    {"role": "system", "content": "用户事实记忆：基本信息：sex=male，age=52；关键指标：blood_pressure_systolic=176mmHg..."},
    {"role": "system", "content": "结构化诊断记忆：健康状态=high_risk；紧急程度=urgent..."},
    {"role": "user", "content": "男，52岁，血压176/108..."},
    {"role": "assistant", "content": "健康状态：高风险。紧急程度：紧急..."},
    ...  # 由 ChatHistoryService.build_context() 提前构建
  ],
  "events": [],
  "safety_notes": [],
  "requires_safe_rewrite": False,
  "safety_revision_count": 0,
}
```

**关键**：`session_history` 不是空的，而是由 `ChatHistoryService.build_context()` 在路由层提前构建好。它包含了四层记忆的全部文本化注入。

#### 3.2.2 节点 1：TriageAgent

```text
_run_node_for_stream(state, self._triage_agent)
  → 执行 _triage_agent(state)
  → 产出新增的 events
```

与初诊相同逻辑，但这次 `_initial_assessment_detector("我的血压风险严不严重？")` 返回 False：
- 有关键词（血压、风险）但无数值且无意图词
- 而且 session_history 非空
→ route = "followup"

产出事件：
```json
{"type": "agent_decision", "agent": "triage_agent", "action": "route_to_followup",
 "reason": "识别为基于既有会话的追问"}
```

#### 3.2.3 追问路径选择

```text
state["route"] == "followup" → 进入追问路径:
  memory_agent → (条件) retrieval_agent → synthesis_agent → safety_review_agent
```

#### 3.2.4 节点 2：MemoryAgent

```text
_run_node_for_stream(state, self._memory_agent)
```

**第一步：获取会话上下文**

```text
_resolve_session_history(state):
  1. 取 session_id = state["session_id"]
  2. 如果 self._memory_context_builder 不为空 AND session_id 不为空:
     → 调用 self._memory_context_builder(session_id, user_input)
     → 这是 ChatHistoryService.build_context()
     → 返回 SessionContextBundle(history=[...], ...)
     → 提取 history 字段 → 更新 session_history
  3. 如果无 session_id 或 builder 失败:
     → 使用 state 中原有的 session_history

结果: session_history = [
  "上下文使用规则...",
  "用户事实记忆：sex=male, age=52, blood_pressure_systolic=176mmHg...",
  "结构化诊断记忆：健康状态=high_risk；紧急程度=urgent；主要风险=高血压风险(高血压,high)...",
  "对话摘要记忆：...",
  ...最近 12 条消息...
]
```

**第二步：提取记忆文本**

```text
memory_text = extract_memory_text_from_history(session_history)
  → 遍历 session_history，筛选 role="system" 且内容包含记忆关键词的消息
  → 关键词: "诊断记忆" "用户事实记忆" "异常指标" "健康状态" "主要风险" "复查项目" "趋势" "对话摘要记忆"
  → 取最后 3 条
  → memory_text = "用户事实记忆：...\n结构化诊断记忆：...\n对话摘要记忆：..."
```

**第三步：判断是否需要检索**

```text
1. has_relevant_memory:
   bool(memory_text) → True (有诊断记忆)
   AND _asks_personal_followup("我的血压风险严不严重？") → True
     (关键词匹配: "我" "血压" "风险" "严不严重")
   → has_relevant_memory = True

2. needs_retrieval:
   (not has_relevant_memory) → False
   OR _requires_external_knowledge("我的血压风险严不严重？") → False
     (不匹配 "标准" "正常值" "指南" "饮食" "科普" "原因" "机制")
   → needs_retrieval = False
```

决策结果：**不需要检索** — 记忆中有诊断结果，可以直接基于记忆回答。

产出事件：
```json
{"type": "agent_decision", "agent": "memory_agent", "action": "use_memory",
 "reason": "会话中已有可用诊断/事实记忆"}
```

**数据传递**：
```text
state["session_history"] → 可能被 _resolve_session_history 更新
state["memory_text"] → "用户事实记忆：...\n结构化诊断记忆：..."
state["needs_retrieval"] → False
```

#### 3.2.5 跳过 RetrievalAgent

```text
_route_after_memory(state):
  → needs_retrieval = False → "synthesize"
  → 直接跳到 synthesis_agent
```

**关键设计**：追问路径中是否调用检索工具不是 LLM 决定的，而是 MemoryAgent 的规则化判断。这避免了 LLM 自己决定「要不要查资料」的不可靠性。

#### 3.2.6 节点 3：SynthesisAgent

```text
_run_node_for_stream(state, self._synthesis_agent)
```

**第一步：检查是否需要安全改写**

```text
requires_safe_rewrite = state.get("requires_safe_rewrite", False)
→ 首次执行时为 False → 走普通路径
```

**第二步：生成回答**

```text
_compose_followup_answer(user_input, session_history, evidence_text="")

1. 如果 self._chat_model is not None:
   → 构建 prompt:
     - 从 session_history 取最后 8 条消息，每条截断到 500 字符
     - evidence_text 为空时填充 "无新增检索证据"
     - 组装: "你是医疗体检辅助评估系统中的回答合成 Agent。请只基于上游 Agent 提供的会话记忆和检索证据回答用户追问..."

   → self._chat_model.invoke(prompt)
   → 提取 message.content
   → 返回中文回答

   例:
   "根据您最近的体检评估结果，您的血压风险较高。收缩压为 176 mmHg，舒张压为 108 mmHg，
    已达到 2 级高血压标准，属于高风险等级。评估结果显示您的健康状态为高风险，
    紧急程度为紧急。建议您前往心内科进一步诊治...以上内容不替代线下诊疗。"

2. 如果 chat_model 为 None (无 API Key):
   → fallback: _followup_answer_builder(user_input, session_history, evidence_text)
   → 确定性模板拼接: "基于当前会话中已有的体检评估和知识库资料..."
```

产出事件：
```json
{"type": "agent_synthesizing", "agent": "synthesis_agent",
 "detail": "正在整合记忆、检索证据和用户问题生成回答"}
```

**数据传递**：
```text
state["answer"] → "根据您最近的体检评估结果..."
state["requires_safe_rewrite"] → False (普通路径)
```

#### 3.2.7 节点 4：SafetyReviewAgent

与初诊的安全复核逻辑相同，但追问的回答是 LLM 生成的，安全复核更有意义：

```text
1. answer = "根据您最近的体检评估结果...心内科进一步诊治..."
2. structured_response = None (追问没有跑 Workflow)
   → human_review_required 不触发

3. medication_risk:
   _mentions_medication_adjustment("我的血压风险严不严重？") → False
   _contains_medication_advice(answer):
     检查: "用药" "药物" "服用" "口服" "停药" "换药" "处方" "剂量"
     → 回答中无这些词 → False

4. dosage_risk:
   _contains_dosage_instruction(answer):
     正则: \d+(\.\d+)?\s*(mg|g|片|粒|次/日)
     → 未匹配 → False

5. suspected_drugs:
   _extract_suspected_drug_terms(answer):
     正则: 中英文{2,12}字+(片|胶囊|沙坦|他汀|...)
     → 未匹配 → False

6. 结论: 无风险 → approved
   notes → 可能有 "未发现需要额外拦截的医疗安全风险" 但 notes 为空字符串
```

产出事件：
```json
{"type": "agent_decision", "agent": "safety_review_agent", "action": "approved",
 "reason": "未发现需要额外拦截的医疗安全风险"}
{"type": "final_answer", "agent": "safety_review_agent",
 "content": "根据您最近的体检评估结果..."}
```

#### 3.2.8 安全改写场景（如果发生）

假设用户问："缬沙坦要不要加量到 80mg？"

```text
MemoryAgent: memory_text 中有用药史 → 但需要外部知识 → needs_retrieval=True

RetrievalAgent: 检索后返回证据

SynthesisAgent: LLM 生成回答
  answer = "根据您的情况，缬沙坦可以考虑增加到 80mg，每天一次..."

SafetyReviewAgent:
  1. medication_risk: _contains_medication_advice → True (含 "用药" "服用")
  2. dosage_risk: _contains_dosage_instruction → True (含 "80mg")
  3. suspected_drugs: _extract_suspected_drug_terms → ["缬沙坦"]
  4. 判定:
     (dosage_risk=True OR suspected_drugs=["缬沙坦"])
     AND safety_revision_count == 0
     → requires_safe_rewrite = True

  → 路由: "rewrite" → synthesis_agent (第二次执行)

SynthesisAgent (安全改写):
  requires_safe_rewrite = True
  → _build_safe_followup_answer(user_input, evidence_text, safety_notes)
  → "关于'缬沙坦要不要加量到 80mg？'，当前只能给出体检辅助层面的通用说明。
     涉及药物名称、剂量、停药、换药或加减药时，应以医生面诊和处方为准，不要自行调整。..."
  → 返回安全版本回答

SafetyReviewAgent (第二次):
  safety_revision_count = 1
  → dosage_risk 或 suspected_drugs 仍然检测到，但 safety_revision_count 不是 0
  → requires_rewrite = False (不再循环)
  → approved_with_notes
```

---

## 第四部分：流式事件 vs 同步执行的差异

### 4.1 同步路径 (run/invoke)

```text
MedicalMultiAgentSupervisor.run()
  → self._graph.invoke(initial_state)
  → LangGraph 同步执行全部节点
  → 返回 final_state
  → 所有 events 在 invoke 完成后一次性获取

优势: 简单，适合非流式 API
劣势: 前端看不到过程
```

### 4.2 流式路径 (iter_events/aiter_events)

```text
MedicalMultiAgentSupervisor.iter_events()
  → 不调用 invoke()
  → 手动逐节点执行 _run_node_for_stream(state, node_fn)
  → 每执行完一个节点，立即 yield 该节点新增的 events
  → 初诊路径中透传 Workflow 的全部 12 节点 step 事件

优势: 前端实时看到处理进度
实现要点:
  - _run_node_for_stream 通过 start 索引记录 node 执行前的 events 数量
  - node 执行后从 events[start:] 切片获取新增事件
  - 逐条 yield 而不是批量返回
```

### 4.3 异步路径 (aiter_events)

```text
与 iter_events 的区别:
  1. 初诊路径: iter_events 走 workflow.iter_events() (同步 generator)
               aiter_events 走 workflow.iter_events_async() (async generator)
               异步版本中 graph + evidence 并行执行

  2. 追问路径: 两者相同

  3. 异步版本中 yield 后有 await asyncio.sleep(0)
     → 确保每个事件被送入事件循环后再继续
```

---

## 第五部分：数据流转全景

### 5.1 初诊路径完整数据流

```text
HTTP 请求体 (文本)
  │
  ▼
ChatHistoryService.ensure_session() → session_id
ChatHistoryService.build_context()   → session_history (通常为空)
  │
  ▼
MedicalAssessmentAgent.stream_assess_async()
  │
  ├── _looks_like_initial_assessment → True
  │
  ▼
MedicalMultiAgentSupervisor.aiter_events()
  │
  ├── TriageAgent:
  │     user_input → detector → route="assessment"
  │
  ├── AssessmentAgent:
  │     user_input → workflow.iter_events_async()
  │       │
  │       ├── parse_raw_input
  │       │     user_input (str) → LLM+正则 → NormalizedMedicalExamJSON
  │       │
  │       ├── normalize_exam_items
  │       │     NormalizedMedicalExamJSON → 别名映射+单位归一化 → 更新后的 NormalizedMedicalExamJSON
  │       │
  │       ├── detect_indicator_states
  │       │     NormalizedMedicalExamJSON → JSON规则匹配 → [DetectedState]
  │       │
  │       ├── retrieve_graph_candidates
  │       │     [DetectedState.state_code] → Neo4j Cypher → [RiskCandidate]
  │       │
  │       ├── expand_intervention_paths
  │       │     [RiskCandidate.disease_code] → Neo4j Cypher → [InterventionCandidate]
  │       │
  │       ├── plan_evidence_queries
  │       │     NormalizedMedicalExamJSON + [DetectedState] + [RiskCandidate] → [RetrievalQuery]
  │       │
  │       ├── retrieve_evidence_chunks
  │       │     [RetrievalQuery] + node_codes → Milvus+FTS5+RRF+Rerank+MMR → [EvidenceChunk]
  │       │
  │       ├── rank_medical_evidence
  │       │     [RiskCandidate] + [EvidenceChunk] + [DetectedState] → 更新 final_score 的 [RiskCandidate]
  │       │
  │       ├── generate_primary_diagnosis
  │       │     NormalizedMedicalExamJSON + [RiskCandidate] + [DetectedState] → PrimaryDiagnosis
  │       │
  │       ├── generate_secondary_recommendation
  │       │     [RiskCandidate] + [InterventionCandidate] → SecondaryRecommendations
  │       │
  │       └── format_medical_response
  │             ... → MedicalAssessmentResponse
  │
  └── SafetyReviewAgent:
        answer + structured_response → 安全复核 → final_answer 事件
  │
  ▼
路由层后处理:
  ├── record_user_message(session_id, raw_text)
  ├── upsert_user_fact_memory(session_id, normalized_exam_json)
  └── record_assistant_message(session_id, answer, structured_result)
```

### 5.2 追问路径完整数据流

```text
HTTP 请求体 (文本) + session_id
  │
  ▼
ChatHistoryService.build_context(session_id, user_input)
  → 从 PostgreSQL 读取:
    - UserFactMemory (事实记忆)
    - DiagnosticMemory (诊断记忆, 最近 2 版)
    - ConversationMemory (最近 12 条对话)
    - conversation_summary (摘要记忆)
  → 组装为 session_history list[dict]
  │
  ▼
MedicalAssessmentAgent.stream_assess_async()
  │
  ├── _looks_like_initial_assessment → False
  │
  ▼
MedicalMultiAgentSupervisor.aiter_events()
  │
  ├── TriageAgent:
  │     user_input + session_history → route="followup"
  │
  ├── MemoryAgent:
  │     session_id → resolve_session_history → 可能刷新 session_history
  │     session_history → extract_memory_text → memory_text (str)
  │     memory_text + user_input → has_relevant_memory + needs_retrieval
  │     → needs_retrieval=False → skip retrieval
  │
  ├── SynthesisAgent:
  │     user_input + session_history[-8:] + evidence_text="" → LLM/fallback → answer
  │
  └── SafetyReviewAgent:
        answer → 用药/剂量/药名检测 → approved/final_answer 事件
  │
  ▼
路由层后处理:
  ├── record_user_message(session_id, user_input)
  └── record_assistant_message(session_id, answer) (无 structured_result)
```
