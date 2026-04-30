# 医疗 KAG Agent — 最终轮优化计划 (Round 3)

> 基于 Round 1 & Round 2 完成后的代码现状。两轮优化后项目已具备：唯一流水线引擎、配置化规则、Agent 显式 ReAct 循环、SQLite FTS5 词汇检索、Milvus distance 复用、条件短路、诊断趋势记忆、摘要阈值触发、依赖自动降级、结构化日志全覆盖。33 项回归测试通过。

## 两轮前后对比

| 维度 | 原始代码 | Round 1 后 | Round 2 后 |
|------|----------|-----------|-----------|
| 流水线 | 三处重复实现 | 唯一 `MedicalKAGWorkflow` | 唯一引擎 + 条件短路 |
| 规则引擎 | `if/elif` 硬编码 | JSON 配置驱动 | 同左 |
| Agent | 黑盒 `create_agent` | 同左 | 显式 ReAct 循环 + 结构化事件 |
| 词汇检索 | 内存 BM25 | 同左 | SQLite FTS5 持久化 |
| 向量检索 | Python 重新算 dense | 同左 | 复用 Milvus distance |
| MMR | 全量 O(n²) | 同左 | 裁剪到 15 候选 |
| 记忆 | 事实+对话+诊断 | 同左 | +诊断趋势对比、摘要阈值触发 |
| 降级 | 无，裸 `except: pass` | 结构化日志 | +Neo4j/Milvus 自动 fallback |
| 测试 | 25 passed | 25 passed | 33 passed |

---

## 最终轮：目标与范围

Round 3 定位为**收尾轮**，不做大架构改动，聚焦三项：**性能（API 异步化）**、**排序质量（Rerank 前置）**、**完成度（流式追问、健康检查、清理）**。

---

## 一、API 异步化 — 并行图检索与证据检索

### 现状

`_execute_step_sequence()` 中各步骤串行执行，其中 `retrieve_graph_candidates`（Neo4j 查询 ~80ms）和 `retrieve_evidence_chunks`（Milvus 检索 ~120ms）可以并行——它们的数据依赖在 `detect_indicator_states` 完成后就都满足了。

### 方案

```python
# 在 iter_events / _execute_step_sequence 中
async def _execute_step_sequence_async(self, raw_input):
    state = InternalAssessmentState(raw_input=raw_input)
    
    # 串行阶段：parse → normalize → rules
    state = await self._parse_and_normalize(state)
    state = await self._detect_states(state)
    
    # 并行阶段：graph + evidence
    state_codes = [s.state_code for s in state.detected_states]
    risk_task = asyncio.create_task(self._graph_store.get_risk_candidates_async(state_codes))
    evidence_task = asyncio.create_task(self._graph_store.get_intervention_candidates_async(
        []))  # placeholder — evidence 的 node_codes 依赖图结果，但可预计算
    
    # 实际并行的是：graph 检索 + evidence 检索的 embedding 编码
    state.risk_candidates = await asyncio.to_thread(self._graph_store.get_risk_candidates, state_codes)
    queries = self._query_planner.build_queries(...)
    state.evidence_chunks = await asyncio.to_thread(self._evidence_store.search, queries, node_codes, top_k)
```

更实际的并行策略：

```
Step A: detect_states → 产出 state_codes
Step B (并行): 
  - graph_store.get_risk_candidates(state_codes)     # Neo4j 查询
  - query_planner.build_queries() + embed queries      # Embedding 编码（可以在知道 queries 内容后就开始）
Step C: wait B → 得到 risk_candidates 和 pre-embedded queries
Step D (并行):
  - evidence_store.search(queries, node_codes)         # Milvus 检索
  - graph_store.get_intervention_candidates(disease_codes)  # Neo4j 二次查询
```

**改动面**：

| 文件 | 改动内容 |
|------|---------|
| `app/workflows/medical_kag_pipeline.py` | 新增 `iter_events_async()`，并行编排 |
| `app/graph/store.py` | `get_risk_candidates_async()` / `get_intervention_candidates_async()` |
| `app/retrieval/evidence_store.py` | `search_async()` |
| `app/api/routes/medical.py` | SSE 路由改为 `async for` |
| `app/models/factory.py` | `RemoteReranker.rerank()` 改为 `httpx.AsyncClient` |

预期收益：wall-clock 时间从 ~350ms 降到 ~220ms。

---

## 二、Rerank 前置 — 排序链路优化

### 现状

当前排序链路：

```
dense + lexical → fusion(0.35) + graph(0.20) + authority(0.10) → rerank(0.25) → MMR
                     ↑ 融合权重                  ↑ rerank 加权的空间只有 25%
```

Rerank 只占 final_score 的 25%，且作用在已经经过 fusion 的候选项上，调序能力有限。

### 方案

```python
# 改造后
candidates = self._dense_search(query)  # Milvus native distance → top 30
candidates = self._lexical_search(query, candidates)  # FTS5 score → 叠加

# Rerank 前置：在未融合结构化偏置前做语义重排
candidates = self._rerank(query, candidates[:20])  # 只取粗排 top 20 送入 reranker

# Fusion 后置：对 reranker 重排后的结果加入结构化偏置
for candidate in candidates:
    candidate.final_score = (
        0.45 * candidate.rerank_score       # rerank 语义分数（权重提高）
        + 0.20 * candidate.graph_overlap     # 图谱节点重叠
        + 0.15 * candidate.lexical_score     # 词汇匹配
        + 0.10 * candidate.source_authority  # 来源权威度
        + 0.10 * candidate.dense_score       # 向量相似度
    )

# MMR 去重（top 15）
candidates = self._mmr_select(candidates, top_k=15)
```

**注意**：`_finalize_scores()` 中需要调整权重分配，将语义得分权重从 35% fusion 提升到 45% rerank。

改动面：
- `app/retrieval/evidence_store.py` — `_hybrid_search()` 搜索链路顺序调整
- 权重在 `_finalize_scores()` 中调整
- 新增参数 `RERANK_CANDIDATE_LIMIT` 控制送入 reranker 的候选数

---

## 三、流式追问体验

### 现状

`stream_assess()` 的追问路径（`stream_followup()`）目前在工作：Agent 的 thinking/decision/tool_call/tool_result/final_answer 事件已经产出。但 final_answer 之后的文本 chunk 是逐个发送的，前端看到的流程是：

```
agent_thinking → ... → final_answer → 文本 chunk × N → done
```

处理步骤完成了，但文本推送期间前端感知不到进度。

### 方案

追问路径增强：
1. `agent_thinking` 事件携带预估步骤信息（"正在评估是否需要查询知识库"）
2. 如果走了 `tool_call`，在 `tool_result` 之后增加一个 `agent_synthesizing` 事件，告知前端"正在整合检索结果生成回答"
3. 文本 chunk 从 20 字符改为逐 token 推送（如果用了 streaming LLM call），或者至少增加周期性心跳

```python
def stream_followup(self, user_input, session_history):
    for event in self._react_agent.iter_events(user_input, session_history):
        if event["type"] == "final_answer":
            answer = event["content"]
            yield {"type": "agent_synthesizing", "detail": "正在生成最终回答"}
            # 流式推送 answer 文本
            for chunk in self._chunk_text(answer, size=20):
                yield {"type": "content", "content": chunk}
            yield {"type": "done"}
            return
        yield event
```

改动面很小，仅 `medical_agent.py` 的 `stream_followup()`。

---

## 四、组件级健康检查

### 现状

`/health` 已经返回 `graph_backend`、`evidence_backend`、`graph_degraded`、`evidence_degraded`。

`/medical/runtime/status` 已经包含 `graph_degraded`、`evidence_degraded`、`graph_fallback_reason`、`evidence_fallback_reason`。

### 方案

新增各组件延迟检测：

```python
@router.get("/runtime/status")
async def get_runtime_status():
    runtime = get_runtime()
    components = {}
    
    # 每个组件测活 + 计时
    for name, ping_fn in [
        ("neo4j", lambda: runtime.graph_store.ping()),
        ("milvus", lambda: runtime.evidence_store.ping()),
        ("postgresql", lambda: runtime.database_manager.ping()),
    ]:
        started = time.perf_counter()
        try:
            ok = ping_fn()
            latency_ms = round((time.perf_counter() - started) * 1000, 2)
            components[name] = {"status": "healthy" if ok else "unhealthy", "latency_ms": latency_ms}
        except Exception:
            components[name] = {"status": "error", "latency_ms": None}
    
    return RuntimeStatusResponse(
        # ... 已有字段 ...
        components=components,
    )
```

在 `DatabaseManager` 中增加 `ping()` 方法（`SELECT 1`）。

改动面很小，仅 `routes/medical.py` + `db/database.py` + `schemas/exam.py`。

---

## 五、最终清理

### 5.1 移除 LangChain Agent 遗留代码

Round 2 引入了 `MedicalReActAgent`，`MedicalAssessmentAgent` 中仍然保留了 `_build_langchain_agent()` 方法和 `self._langchain_agent` 字段，但不再被主链路使用（`chat_assess` 已改为走 `_react_agent.run()`，`stream_assess` 走 `_react_agent.iter_events()`）。

- 删除 `_build_langchain_agent()`
- 删除 `_convert_history_to_messages()`（不再需要 LangChain 消息转换）
- 删除 `_extract_agent_text()`（不再解析 `create_agent` 返回值）
- 删除 `_fallback_chat_assess()`（ReAct agent 自带 fallback 链路）
- 删除 `MedicalAssessmentAgent.__init__` 中不再需要的 `_langchain_agent` 构建
- `_compose_answer` 的 `_build_answer_prompt` 仍然有用（初诊路径格式化），保留

### 5.2 移除 `agent_tools.py` 中的 LangChain 依赖

`StandardAssessmentTool` 和 `MedicalKnowledgeRetrievalTool` 继承自 `StructuredTool`（依赖 `langchain_core.tools`）。Round 2 中 `MedicalReActAgent` 不再使用 LangChain 的工具抽象，直接调用 `knowledge_tool._run()`。但 `_build_langchain_agent` 删掉后，`as_tool()` 方法就没有调用方了。

- 保留 `MedicalKnowledgeRetrievalTool._run()`（被 `MedicalReActAgent` 调用）
- 删除 `as_tool()` 方法
- 删除 `StandardAssessmentTool` 类（初诊路径直接调 workflow，不再需要 Agent 工具封装）

### 5.3 删除未使用的 LangChain `create_agent` import

`medical_agent.py` 中 `from langchain.agents import create_agent` 不再需要。如有其他未使用 import 一并清理。

---

## 六、测试补充

新增：

- `tests/test_pipeline_async.py` — 验证并行执行结果与串行一致
- `tests/test_rerank_reorder.py` — 验证 rerank 前置后排序结果
- `tests/test_health_components.py` — 验证组件级健康检查字段

目标：35+ passed。

---

## 最终轮优先级

| 优先级 | 优化项 | 工时 | 理由 |
|:------:|--------|:----:|------|
| **P1** | API 异步化 | 2-3 天 | 最大性能收益，wall-clock 延迟降 35% |
| **P2** | Rerank 前置 | 0.5 天 | 排序质量提升，改动集中在一个文件 |
| **P2** | 清理遗留代码 | 0.5 天 | 减少死代码、降低后续维护者困惑 |
| **P3** | 组件级健康检查 | 0.5 天 | 生产级完成度 |
| **P3** | 流式追问增强 | 0.5 天 | 前端体验优化 |
| **P3** | 测试补充 | 0.5 天 | 覆盖新改动的关键路径 |

### 建议实施顺序

```
Day 1: P1 API 异步化（核心改动）
Day 2: P1 API 异步化（测试 + edge cases）
Day 3: P2 Rerank 前置 + P2 清理遗留代码
Day 4: P3 健康检查 + 流式追问 + 测试补充
```

### Round 3 完成后预期状态

| 指标 | Round 2 后 | Round 3 后 |
|------|-----------|-----------|
| 初诊延迟（正常路径） | ~50ms（短路） | ~50ms |
| 初诊延迟（异常路径） | ~350ms | ~220ms |
| 追问延迟 | ~200ms | ~150ms |
| Agent 事件类型 | 7 种 | 8 种 (+agent_synthesizing) |
| 健康检查粒度 | 组件名 + 降级标记 | +组件延迟 |
| 代码死量 | LangChain Agent 遗留 | 全清理 |
| 测试 | 33 passed | 35+ passed |

### 三轮优化总览

```
Round 1: 消除技术债 — 流水线唯一化、规则配置化、日志全覆盖
Round 2: 建核心能力 — Agent 显式循环、检索性能、条件短路、趋势记忆、自动降级
Round 3: 做最后收敛 — API 异步化、Rerank 前置、遗留清理、健康检查完善
```
