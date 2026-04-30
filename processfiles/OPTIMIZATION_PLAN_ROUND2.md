# 医疗 KAG Agent — 第二轮优化计划

> 基于 2026-04-29 Round 1 完成后的代码现状产出。Round 1 已完成：流水线唯一化、规则配置化、流式事件可观测、错误日志全覆盖、测试 25 passed。

---

## 现状总览

| 首轮计划项 | 状态 |
|-----------|------|
| 流水线唯一化 (P0) | ✅ 完成 — `MedicalKAGWorkflow` 成为唯一引擎，`iter_events()` 统一产事件 |
| 规则配置化 (P0) | ✅ 完成 — `medical_rules.json` 驱动，支持 AND/OR 嵌套 + 性别差异阈值 |
| 错误日志 (P0) | ✅ 完成 — 全链路 12 个降级点均有 `logger.warning` |
| 流水线指标记录 (P0) | ✅ 完成 — 每步 `duration_ms` + 关键计数 |

**第二轮聚焦三个方向**：Agent 层可控性（面试核心亮点）、检索层性能（数据量上去后的瓶颈）、API 异步化（吞吐优化）。

---

## 一、Agent 显式循环 — 优先级最高

### 为什么第二轮必须做

当前 `langchain.agents.create_agent` 仍然是黑盒。Agent 面试的核心考察点就是 agent loop 的设计能力（think/act/observe/reflect 循环、工具选择策略、退出条件、上下文管理）。自实现一个可控的 agent loop 是简历上最有分量的改动。

### 设计方案

#### 1. 自实现 ReAct Agent Loop

```python
class MedicalReActAgent:
    MAX_ITERATIONS = 10
    
    def run(self, user_input: str, context: SessionContextBundle) -> Generator[dict, None, str]:
        # 首次评估判断 — 短路优化
        if self._looks_like_initial_assessment(user_input):
            yield {"type": "agent_action", "action": "route_to_pipeline"}
            result = self._workflow.run_state(user_input)
            return self._format_final_answer(result)
        
        messages = [
            SystemMessage(content=self._build_system_prompt()),
            *self._context_to_messages(context),
            HumanMessage(content=user_input),
        ]
        
        iteration = 0
        called_tools = {}  # tool_name -> last_args_hash，用于循环检测
        
        while iteration < self.MAX_ITERATIONS:
            iteration += 1
            
            # 1. Think — LLM 决定下一步
            yield {"type": "agent_thinking", "iteration": iteration}
            response = self._chat_model.invoke(
                messages + [self._tool_choice_hint()]
            )
            
            # 2. 判断是 tool_call 还是 final_answer
            if response.tool_calls:
                for tool_call in response.tool_calls:
                    tool_name = tool_call["name"]
                    args_hash = hashlib.md5(json.dumps(tool_call["args"]).encode()).hexdigest()
                    
                    # 循环检测
                    prev_hash = called_tools.get(tool_name)
                    if prev_hash == args_hash:
                        yield {"type": "agent_warning", "detail": f"检测到重复调用 {tool_name}，强制终止"}
                        return self._compose_fallback(context)
                    called_tools[tool_name] = args_hash
                    
                    # 3. Act — 执行工具
                    yield {"type": "tool_call", "name": tool_name, "args_summary": str(tool_call["args"])[:120]}
                    result = self._execute_tool(tool_name, tool_call["args"])
                    
                    # 4. Observe — 结果入上下文
                    yield {"type": "tool_result", "name": tool_name, "result_len": len(result)}
                    messages.append(AIMessage(content="", tool_calls=[tool_call]))
                    messages.append(ToolMessage(content=result, tool_call_id=tool_call["id"]))
            else:
                # 5. Final answer
                yield {"type": "agent_final_answer"}
                return response.content
        
        yield {"type": "agent_warning", "detail": f"达到最大迭代次数 {self.MAX_ITERATIONS}"}
        return self._compose_fallback(context)
```

#### 2. 工具实现细化

```python
# 当前只有 2 个工具，第二轮拆分为 5 个：
tools = [
    "run_medical_assessment",    # 确定性流水线（仅首次）
    "lookup_medical_knowledge",  # 知识库证据检索
    "check_user_facts",          # 查用户事实记忆（只读）
    "check_last_diagnosis",      # 查最近一次诊断快照
    "calculate_trend",           # 对比最近两次诊断的变化
]
```

关键约束：
- `run_medical_assessment` 在一次会话中只能调用一次（guard state）
- `check_user_facts` / `check_last_diagnosis` 优先于 `lookup_medical_knowledge`，避免 LLM 偏好外呼而忽略既有记忆

#### 3. 与当前 Agent 的兼容

- 保留现有 `MedicalAssessmentAgent` 作为 "legacy agent"
- 新增 `MedicalReActAgent` 并存
- `container.py` 通过 `Settings.use_react_agent` 开关控制使用哪套
- 所有现有测试不受影响

---

## 二、证据检索性能优化

### 2.1 Milvus 原生 distance 复用

**当前问题**：`MilvusEvidenceStore.search()` 先用 `self._client.search()` 做一次向量检索获取候选项（Milvus 已经返回了 distance），然后在 `_dense_ranks()` 中重新在 Python 侧对所有 chunk 做 embedding 点积。第一次 search 的结果被丢弃了。

**改动**：

```python
# 改造前（Milvus search 结果只用于取候选项）:
rows = self._client.search(...)  # 返回 distance，但没用
for row in rows:
    candidate_map[chunk_id] = ...
dense_ranks = self._dense_ranks(query.text, chunks)  # Python 侧重新算

# 改造后（直接复用 Milvus distance）:
dense_ranks = []
rows = self._client.search(...)
for row in rows:
    entity = row.get("entity", row)
    chunk_id = entity.get("chunk_id")
    distance = row.get("distance", 0.0)
    if chunk_id not in candidate_map:
        candidate_map[chunk_id] = self._build_chunk(entity)
    dense_ranks.append((chunk_id, distance))
```

收益：消除一次 Python 侧全量 embedding 点积，chunk 数量越大收益越明显。

### 2.2 BM25 迁移到 SQLite FTS5

**当前问题**：`BM25Lite` 将所有 chunk text + embedding 存在 Python 内存（`_chunk_cache`、`_chunk_embeddings`），chunk 上万时内存占用显著，且重启后丢失索引。

**改动**：

```python
class SQLiteFTSIndex:
    def __init__(self, db_path: Path):
        self._conn = sqlite3.connect(str(db_path))
        self._conn.execute("CREATE VIRTUAL TABLE IF NOT EXISTS docs_fts USING fts5(chunk_id, title, text, linked_codes)")
    
    def index(self, chunks: list[EvidenceChunk]) -> None:
        self._conn.execute("DELETE FROM docs_fts")
        for chunk in chunks:
            self._conn.execute(
                "INSERT INTO docs_fts(chunk_id, title, text, linked_codes) VALUES (?, ?, ?, ?)",
                (chunk.chunk_id, chunk.title, chunk.text, " ".join(chunk.linked_node_codes)),
            )
    
    def search(self, query: str, top_k: int) -> list[tuple[str, float]]:
        # SQLite FTS5 的 bm25() 函数
        rows = self._conn.execute(
            "SELECT chunk_id, bm25(docs_fts) AS score FROM docs_fts WHERE docs_fts MATCH ? ORDER BY score LIMIT ?",
            (query, top_k),
        ).fetchall()
        return [(row[0], 1.0 / (1.0 + abs(row[1]))) for row in rows]
```

收益：词汇检索不再依赖内存缓存，支持持久化，chunk 上万后性能远优于 Python 侧遍历。

### 2.3 MMR 裁剪优化

**改动**：只对 rerank 后的 top 15 做 MMR，不遍历全量候选。15 个候选项的两两计算（105 次 cosine）远小于 50+ 的计算。

---

## 三、API 异步化

### 3.1 当前瓶颈

`_execute_step_sequence()` 中各步骤是串行同步的：

```
graph_query (80ms) → 等待
evidence_search (120ms) → 等待
rerank (200ms) → 等待
总耗时 ≈ 400ms+
```

其中 `graph_query` 和 `evidence_search` 之间没有数据依赖（evidence 可以用已解析的 state_codes 做查询，不需要等 graph 结果），可以并行。

### 3.2 并行化方案

```python
# 改造后 (async)
async def _retrieve_graph_candidates_node(self, payload):
    state = payload["state"]
    state_codes = [item.state_code for item in state.detected_states]
    state.risk_candidates = await self._graph_store.get_risk_candidates_async(state_codes)
    return {"state": state}

# 在 pipeline 中并行执行
risk_task = asyncio.create_task(graph_search(state))
evidence_task = asyncio.create_task(evidence_search(queries, node_codes))
state.risk_candidates, state.evidence_chunks = await asyncio.gather(risk_task, evidence_task)
```

**改动面**：

| 组件 | 改动 |
|------|------|
| `Neo4jGraphStore` | 新增 `get_risk_candidates_async()`，使用 `neo4j.AsyncGraphDatabase` |
| `MilvusEvidenceStore` | search 本身是 IO，`asyncio.to_thread` 包装即可 |
| `RemoteReranker` | `requests.post` → `httpx.AsyncClient.post` |
| `MedicalKAGWorkflow` | `iter_events()` 改为 `async def` |
| FastAPI routes | 无改动（FastAPI 原生支持 async generator 作为 StreamingResponse） |

预期收益：wall-clock 时间从 ~400ms 降到 ~250ms（graph 和 evidence 并行，取 max）。

---

## 四、LangGraph 条件分支

### 4.1 现状

所有 12 个节点线性串联，即使没有检测到任何异常状态，也会走完整的图谱→证据→排序→格式化链路。

### 4.2 新增分支逻辑

```python
def _build_graph(self):
    graph = StateGraph(WorkflowState)
    # ... add nodes ...
    
    # 条件分支 1：无异常 → 跳过图谱和证据检索
    graph.add_conditional_edges(
        "detect_indicator_states",
        self._should_continue_assessment,
        {
            "healthy": "generate_primary_diagnosis",  # 直接跳到格式化
            "abnormal": "retrieve_graph_candidates",   # 继续正常链路
        },
    )
    
    # 条件分支 2：图谱未命中 → 跳过干预扩展
    graph.add_conditional_edges(
        "retrieve_graph_candidates",
        self._has_graph_hits,
        {
            "has_hits": "expand_intervention_paths",
            "no_hits": "plan_evidence_queries",
        },
    )

def _should_continue_assessment(self, payload):
    state = payload["state"]
    if not state.detected_states:
        return "healthy"
    return "abnormal"

def _has_graph_hits(self, payload):
    state = payload["state"]
    if not state.risk_candidates:
        return "no_hits"
    return "has_hits"
```

收益：健康体检输入（无异常指标）直接返回"指标均在正常范围"，响应时间从 ~400ms 降到 ~50ms。

---

## 五、记忆层增强

### 5.1 诊断记忆历史追溯

**当前**：`build_context()` 只注入最新诊断（`is_current="true"`）。

**改动**：同时注入最近 2 个诊断版本（当前 + 上一版）做 diff：

```python
recent_diagnostics = (
    db.execute(
        select(DiagnosticMemory)
        .where(DiagnosticMemory.session_ref_id == session.id)
        .order_by(DiagnosticMemory.version_no.desc())
        .limit(2)
    )
    .scalars()
    .all()
)

if len(recent_diagnostics) >= 2:
    current, previous = recent_diagnostics[0], recent_diagnostics[1]
    # 标注变化
    history.append({
        "role": "system",
        "content": (
            f"最新诊断(v{current.version_no})：健康状态={current.health_status}，紧急程度={current.urgency_level}\n"
            f"上次诊断(v{previous.version_no})：健康状态={previous.health_status}，紧急程度={previous.urgency_level}\n"
            f"对比变化：当前异常指标={current.abnormal_indicator_summary}；上次异常指标={previous.abnormal_indicator_summary}"
        ),
    })
```

收益：支持"和上次比怎么样"的趋势追问。

### 5.2 摘要触发阈值

**当前**：每次 `record_assistant_message` 都触发生成摘要（如果 LLM 可用）。

**改动**：

```python
# 配置
SUMMARY_TRIGGER_CHARS = 2000  # 累计新增对话字符数超过此阈值才生成

# 在 record_assistant_message 中
session._pending_chars = getattr(session, '_pending_chars', 0) + len(content)
if session._pending_chars >= SUMMARY_TRIGGER_CHARS:
    session.conversation_summary = self._build_session_summary(...)
    session._pending_chars = 0
```

收益：减少 LLM 调用频率，降低延迟和成本。

---

## 六、降级增强

### 6.1 Neo4j → InMemory 自动切换

```python
def build_graph_store(settings: Settings) -> BaseGraphStore:
    if settings.use_in_memory_graph:
        return InMemoryGraphStore()
    try:
        store = Neo4jGraphStore(settings)
        if store.ping():
            return store
        logger.warning("neo4j.ping_failed, falling back to in-memory graph store")
    except Exception:
        logger.warning("neo4j.init_failed", exc_info=True)
    return InMemoryGraphStore()
```

### 6.2 Milvus → InMemory 自动切换

同上逻辑。

收益：外部依赖不可用时系统仍然可运行（降级模式），而非直接崩溃。

---

## 第二轮实施优先级

| 优先级 | 优化项 | 预估工时 | 影响面 | 理由 |
|:------:|--------|:--------:|:------:|------|
| **P1-1** | Agent 显式 ReAct 循环 | 3-4 天 | Agent 层 + tools | 面试核心亮点，Agent 工程能力的最佳体现 |
| **P1-2** | Milvus native distance 复用 | 0.5 天 | evidence_store.py | 改动很小但消除明显浪费 |
| **P1-3** | LangGraph 条件分支 | 1 天 | medical_kag_pipeline.py | 改善正常体检路径体验，实现简单 |
| **P2-1** | 诊断记忆历史追溯 | 0.5 天 | chat_history_service.py | 改动小，解锁"趋势对比"追问 |
| **P2-2** | BM25 → SQLite FTS5 | 1 天 | 新增 files | chunk 万级以上必需 |
| **P2-3** | API 异步化 | 2-3 天 | pipeline + stores | 吞吐优化，30-40% 延迟下降 |
| **P3-1** | 摘要触发阈值 | 0.5 天 | chat_history_service.py | 减少 LLM 调用 |
| **P3-2** | 降级自动切换 | 0.5 天 | graph/store.py + evidence_store.py | 提升系统韧性 |
| **P3-3** | MMR 裁剪 + Rerank 前置 | 1 天 | evidence_store.py | 排序质量优化 |

### 建议实施顺序

```
Week 1: P1-1 Agent 显式 ReAct 循环（核心改动）
Week 1: P1-2 Milvus distance 复用 + P1-3 LangGraph 条件分支（快速见效）
Week 2: P2-1 诊断记忆历史 + P2-2 SQLite FTS5 + P2-3 API 异步化
Week 3: P3-1 摘要阈值 + P3-2 降级切换 + P3-3 排序优化 + 回归测试
```

### Round 2 完成后预期状态

| 指标 | Round 1 后 | Round 2 后 |
|------|-----------|-----------|
| Agent 可控性 | 黑盒 `create_agent` | 显式 ReAct 循环，全事件流式可见 |
| 检索延迟 | ~400ms（Python 侧计算主导） | ~250ms（Milvus native + 异步并行） |
| 健康路径延迟 | ~400ms | ~50ms（条件分支短路） |
| 记忆能力 | 事实+诊断+摘要 | +趋势对比、摘要按需触发 |
| 韧性 | 依赖不可用即崩溃 | Neo4j/Milvus 不可用时自动降级 |
| 测试覆盖 | 25 passed | 目标 35+ passed |
