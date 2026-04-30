# 医疗 KAG Agent 代码优化说明

本次优化基于 `OPTIMIZATION_PLAN.md`，优先处理 P0/P1 中收益高、风险可控的部分：流水线收敛、错误可观测、规则配置化，以及对应回归测试。

## 1. 流水线唯一化

改动文件：

- `app/workflows/medical_kag_pipeline.py`
- `app/services/medical_agent.py`
- `app/services/agent_tools.py`
- `app/services/container.py`

优化前，确定性评估流程分别存在于：

- `MedicalAssessmentAgent._execute_pipeline()`
- `MedicalAssessmentAgent.stream_assess()`
- `MedicalKAGWorkflow` 的 LangGraph 节点

这三处都包含 `parse -> normalize -> rules -> graph -> evidence -> rank -> format` 的核心编排，后续维护容易出现行为分叉。

优化后：

- `MedicalKAGWorkflow` 成为唯一确定性执行引擎。
- 新增 `run_state(raw_input)`，返回完整 `InternalAssessmentState`，保留中间状态，便于 Agent、测试和后续可观测使用。
- 新增 `iter_events(raw_input)`，由 Workflow 逐节点产出 `step` 事件，`stream_assess()` 不再手写业务流水线。
- `MedicalAssessmentAgent._execute_pipeline()` 保留为兼容 shim，内部只委托 `workflow.run_state()`。
- `StandardAssessmentTool` 改为可接收 `workflow.run()` 或旧版 state runner，降低外部调用耦合。
- `container.py` 将同一个 `medical_workflow` 注入 `MedicalAssessmentAgent`，避免 Agent 内部重新拼出另一套组件。

## 2. 流式事件可观测增强

改动文件：

- `app/workflows/medical_kag_pipeline.py`

`iter_events()` 对每个流水线步骤产出两个事件：

- `status="running"`：步骤开始。
- `status="completed"`：步骤完成，并附带 `duration_ms`。

同时在日志中记录关键指标：

- `exam_items_count`
- `detected_states_count`
- `risk_candidates_count`
- `evidence_chunks_count`

这些字段都是新增字段，保留了原有 `type/label/detail` 格式，兼容现有 Streamlit 和测试。

## 3. 规则配置化

改动文件：

- `app/services/rules.py`
- `app/config/medical_rules.json`

优化前，规则判断集中在 `if/elif` 中，新增指标或调整阈值必须改代码。

优化后：

- 单指标规则、组合规则都进入 `medical_rules.json`。
- `IndicatorRuleEngine` 只负责加载配置、通用条件匹配、构造 `DetectedState`。
- 支持 `gte/gt/lte/lt/eq/in/not_in` 操作符。
- 支持嵌套 `AND/OR` 条件，可表达性别差异阈值、组合风险等逻辑。
- 保留原有 `state_code`、`rule_id`、中文标签和严重程度，避免影响图谱映射和既有测试。

代码中保留了中文注释，说明配置驱动的维护入口。

## 4. 错误处理与日志

改动文件：

- `app/models/factory.py`
- `app/services/medical_agent.py`
- `app/retrieval/evidence_store.py`
- `app/services/input_parser.py`
- `app/services/chat_history_service.py`
- `app/api/routes/medical.py`

优化原则：不改变原降级行为，只补可追踪日志。

已增加日志的位置：

- 远程 embedding、extractor、reranker、chat model 初始化失败。
- LangChain Agent 调用失败并回退。
- LLM 生成自然语言答复失败并回退。
- rerank 失败并回退原排序。
- Milvus `ping()` / `data_ready()` 检查失败。
- LLM 输入抽取失败并回退正则解析。
- 会话摘要 LLM 失败并回退确定性摘要。
- 知识库上传后台任务失败。

这样线上或演示时出现降级，不再只能从前端结果反推原因。

## 5. 测试补强

新增或调整测试：

- `tests/test_agent_tools.py`
  - `StandardAssessmentTool` 改为绑定 `workflow.run()`，不再依赖 `agent._execute_pipeline` 私有方法。
- `tests/test_medical_kag_pipeline.py`
  - 新增 `run_state()` 中间状态保留测试。
- `tests/test_medical_agent.py`
  - 新增 `stream_assess()` 最终结构化结果与 `assess()` 一致性测试。
- `tests/test_rules_config.py`
  - 新增规则配置注入测试，确认无需修改代码即可新增规则。

验证结果：

```text
25 passed
```

## 6. 暂未实施的优化

以下内容暂未在本轮落地，原因是改动面更大，建议作为下一批迭代：

- 显式 ReAct / Plan-Execute Agent loop。
- Milvus 检索完全回归原生 distance，并拆出独立全文索引。
- FastAPI 全链路异步化。
- 用户事实记忆向量化检索。
- `/medical/runtime/status` 的完整组件级健康检查。

当前代码已经先消除了维护风险最高的双流水线问题，并把关键降级路径变得可观测，为后续继续做 Agent loop 和检索层性能优化打下基础。
