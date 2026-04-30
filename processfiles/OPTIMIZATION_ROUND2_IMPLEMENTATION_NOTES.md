# 医疗 KAG Agent 第二轮优化实施说明

本轮参考 `OPTIMIZATION_PLAN.md` 和 `OPTIMIZATION_PLAN_ROUND2.md`，结合现有代码状态，优先落地低风险且收益明确的 Round 2 项目。

## 1. Agent 显式循环

改动文件：

- `app/services/react_agent.py`
- `app/services/medical_agent.py`

新增 `MedicalReActAgent`，替代原先对 LangChain `create_agent` 黑盒循环的依赖。追问场景现在会产生结构化事件：

- `agent_thinking`
- `agent_decision`
- `tool_call`
- `tool_result`
- `agent_warning`
- `final_answer`

循环具备：

- 最大迭代次数限制。
- 重复工具参数 hash 检测。
- 工具 allowlist，目前先开放 `lookup_medical_knowledge`。
- 追问优先使用会话中的诊断记忆和事实记忆。
- 初次体检评估仍走确定性 Workflow，不经过 Agent 工具循环。

`stream_assess()` 现在会区分：

- 初诊：走 `MedicalKAGWorkflow.iter_events()`。
- 追问：走 `MedicalReActAgent.iter_events()`。

## 2. 检索层性能优化

改动文件：

- `app/retrieval/evidence_store.py`
- `app/retrieval/lexical.py`
- `app/core/settings.py`

### 2.1 Milvus distance 复用

`MilvusEvidenceStore.search()` 现在直接复用 Milvus search 返回的 `distance/score` 作为 dense score，不再对 Milvus 候选做第二次 Python 侧 embedding 点积。

新增测试确认：同一次 Milvus 检索中 query embedding 只调用一次，返回的 `dense_score` 等于 Milvus distance。

### 2.2 词汇检索方案选择

本轮选择 **SQLite FTS5**，没有选择 PostgreSQL `tsvector`。

原因：

- 当前知识库证据块主存储是 JSON registry + Milvus，不在 PostgreSQL。
- 使用 PostgreSQL `tsvector` 会引入新的证据块双写链路和迁移复杂度。
- SQLite FTS5 使用 Python 标准库 `sqlite3`，零额外服务依赖，适合当前原型和本地部署。
- FTS 索引可以随 `rebuild_index()` 重建，和现有知识库构建链路贴合。

实现细节：

- 新增 `SQLiteFTSIndex`。
- `MilvusEvidenceStore` 在 `lexical_index_backend=sqlite_fts` 时启用。
- 中文检索使用 FTS5 优先，必要时回退到轻量 LIKE 匹配，避免中文 tokenizer 差异导致空召回。
- `InMemoryEvidenceStore` 保持原 BM25Lite，减少已有测试和本地路径风险。

### 2.3 MMR 裁剪

在 rerank 后、MMR 前裁剪候选：

```text
limit = max(top_k * 3, mmr_candidate_limit)
```

默认 `MMR_CANDIDATE_LIMIT=15`。这样保留 rerank 前置，同时避免 MMR 对大量候选做两两 cosine。

## 3. Workflow 条件短路

改动文件：

- `app/workflows/medical_kag_pipeline.py`

新增执行层短路：

- 无异常状态时，跳过图谱检索、证据检索、排序，直接生成健康评估。
- 有异常但图谱未命中时，跳过干预路径扩展，仍允许证据检索兜底。

这样健康路径不会继续访问 Neo4j/Milvus，降低延迟，也减少外部依赖波动影响。

## 4. 记忆层增强

改动文件：

- `app/services/chat_history_service.py`
- `app/db/models.py`
- `app/db/database.py`
- `app/core/settings.py`

### 4.1 诊断趋势记忆

`build_context()` 现在在非初诊追问中读取最近两版 `DiagnosticMemory`，注入：

- 最新诊断版本。
- 上次诊断版本。
- 两次异常指标、健康状态、紧急程度对比。

用于支持“和上次相比怎么样”这类趋势追问。

### 4.2 摘要触发阈值

新增 `summary_pending_chars` 字段，持久化累计新增对话字符数。

只有累计字符数达到 `SUMMARY_TRIGGER_CHARS`，才允许调用摘要 LLM。未达到阈值时仍使用确定性摘要，避免每轮追问都触发 LLM。

## 5. 依赖自动降级和健康状态

改动文件：

- `app/graph/store.py`
- `app/retrieval/evidence_store.py`
- `app/main.py`
- `app/api/routes/medical.py`
- `app/schemas/exam.py`

新增降级策略：

- Neo4j 初始化失败或 ping 失败，自动回落 `InMemoryGraphStore`。
- Milvus 初始化失败或 ping 失败，自动回落 `InMemoryEvidenceStore`。
- fallback 原因记录到 `fallback_reason`，并通过健康接口展示。

`/health` 现在返回实际运行后端，而不是只看配置项。

`/medical/runtime/status` 新增：

- `graph_degraded`
- `evidence_degraded`
- `graph_fallback_reason`
- `evidence_fallback_reason`
- `checked_at`

## 6. 新增测试

新增或扩展：

- `tests/test_react_agent.py`
- `tests/test_retrieval_round2.py`
- `tests/test_medical_kag_pipeline.py`
- `tests/test_chat_history_service.py`

覆盖：

- Agent 显式循环使用记忆直接回答。
- Agent 必要时调用知识库工具。
- 健康路径跳过图谱和证据检索。
- SQLite FTS5 rebuild/search。
- MMR 输入裁剪。
- Milvus distance 复用。
- 诊断趋势记忆注入。
- 摘要 LLM 阈值触发。

验证结果：

```text
33 passed
```

