# 医疗 KAG Agent 工作台 — 后端优化方案

> 基于 2026-04-29 代码 review 产出，涵盖 Agent 层、流水线层、检索层、规则引擎、记忆层、API 层、错误处理、可观测性八个方向。

## 实施状态

| 轮次 | 文档 | 状态 |
|------|------|------|
| Round 1 | [OPTIMIZATION_IMPLEMENTATION_NOTES.md](OPTIMIZATION_IMPLEMENTATION_NOTES.md) | ✅ 已完成 — 流水线唯一化、规则配置化、错误日志、流式事件 |
| Round 2 | [OPTIMIZATION_ROUND2_IMPLEMENTATION_NOTES.md](OPTIMIZATION_ROUND2_IMPLEMENTATION_NOTES.md) | ✅ 已完成 — Agent 显式循环、检索性能优化、条件短路、趋势记忆、自动降级 |
| Round 3 | [OPTIMIZATION_ROUND3_FINAL_NOTES.md](OPTIMIZATION_ROUND3_FINAL_NOTES.md) | ✅ 已完成 — API 异步化、Rerank 前置、遗留清理、组件健康检查 |

> 以下为原始全量方案。其中 **已实施** 标记的条目已在 Round 1 完成，**待实施** 的条目已纳入 Round 2 计划。

---

## 一、Agent 层：从黑盒到显式可控 `[✅ Round 2 已完成]`

### 现状问题

[app/services/medical_agent.py](app/services/medical_agent.py) 使用 `langchain.agents.create_agent` 创建 Agent，该接口将工具调用循环完全封装为黑盒。无法观测 Agent 的真实推理路径、工具选择理由、何时切换工具，出了错只能靠 catch Exception 兜底。

### 优化方案

#### 1. 替换为显式 ReAct/Plan-Execute Agent 循环

自实现轻量 agent loop：

```
while not finished:
    think → select_tool → execute → observe → reflect
```

每一步产生结构化事件：

| 事件类型 | 内容 |
|----------|------|
| `reasoning` | LLM 思考链路 |
| `tool_call` | 调用的工具名 + 参数 |
| `tool_result` | 工具返回摘要 |
| `final_answer` | 最终回答 |

全部事件对前端流式可见。

核心约束：
- 加入最大迭代次数限制（建议 ≤10 轮）
- 加入循环检测（连续两次调用同一工具且参数相同 → 强制终止）
- 不再依赖特定 LangChain 版本的 API 兼容性

#### 2. 工具调用策略优化

- **首次评估**：不走 Agent loop，直接调用 `_execute_pipeline`，节省一次 LLM 判断调用的延迟和 token 消耗
- **追问场景**：Agent 先检查记忆中是否已有相关诊断对象，命中则直接引用，不需要主动调用工具
- **置信度门控**：追问问题若关键词命中知识库高置信度指标（如"标准范围""正常值"），直接走检索工具；若涉及个体化判断（"严不严重""要不要吃药"），必须结合诊断记忆作答

---

## 二、流水线层：消除代码重复 `[✅ Round 1 已完成]`

### 现状问题

`MedicalAssessmentAgent._execute_pipeline()` 和 `MedicalKAGWorkflow` 的 13 个 node 方法是两套几乎完全相同的流水线实现：

```
input → parse → normalize → rules → graph → evidence → rank → format
```

存在维护风险 — 修改一处必须同步另一处，很容易出现两边行为不一致。

### 优化方案

#### 1. 收归唯一执行引擎

- 删除 `MedicalAssessmentAgent._execute_pipeline()`
- Agent 的 `StandardAssessmentTool._run` 改为调用 `workflow.run(raw_text)`
- `stream_assess` 中逐步调用 Workflow 各节点，产生 step 事件
- 流水线的所有编排逻辑只存在于 `medical_kag_pipeline.py`

改造前后对比：

```
改造前:
  MedicalAssessmentAgent._execute_pipeline()  ← 流水线副本 A
  MedicalKAGWorkflow (13 nodes)               ← 流水线副本 B

改造后:
  MedicalKAGWorkflow (13 nodes)               ← 唯一流水线
  MedicalAssessmentAgent 调用 Workflow        ← 仅做编排调度
  StandardAssessmentTool 调用 Workflow        ← 仅做封装
```

#### 2. LangGraph 支持真正的条件分支

当前 13 个节点全部串联，无分支。建议增加条件边：

- `detect_indicator_states` →（无异常状态时）→ 直接跳 `format_medical_response`，输出"指标均在正常范围，无需干预"
- `retrieve_graph_candidates` →（图谱未命中）→ 跳过 `expand_intervention_paths`，直接走证据检索兜底
- `rank_medical_evidence` →（证据为空且图谱分数低）→ 标记 `human_review_required=True`，前端提示"建议线下就医复核"

---

## 三、证据检索层：让 Milvus 做它该做的事 `[✅ Round 2 已完成]`

### 现状问题

[app/retrieval/evidence_store.py](app/retrieval/evidence_store.py) 的 `MilvusEvidenceStore` 将所有 chunk 和 embedding 加载到 Python 内存（`_chunk_cache`、`_chunk_embeddings`），然后在 Python 侧计算 dense_ranks、lexical_ranks、MMR。这与 `InMemoryEvidenceStore` 的做法完全一致，Milvus 的核心优势（原生向量检索、属性过滤、混合搜索（Milvus 2.4+ Hybrid Search））完全没有被利用。

### 优化方案

#### 1. 向量检索回归 Milvus

- 现有 `self._client.search()` 已经在做一次向量检索取候选项，但后续又重新在 Python 里算一次 dense_ranks
- 去掉 `_dense_ranks()` 中的 Python 侧计算，直接复用 Milvus search 返回的 `distance` 作为 `dense_score`
- 配合 Milvus 2.4+ 的 `search()` 返回 distance 值完成 RRF 融合

#### 2. 词汇检索独立索引

- BM25 当前依赖 `_chunk_cache` 的全量内存索引，chunk 数量上万时内存和计算开销不可忽略
- 推荐方案：SQLite FTS5（零外部依赖，轻量级）或 PostgreSQL `tsvector` + GIN 索引做词汇倒排
- 备选方案：Elasticsearch（如果后续有全文搜索集群需求）

#### 3. MMR 重排序优化

- 当前 MMR 需对候选项两两计算 cosine similarity（O(n²)），50+ candidate 时性能劣化明显
- 优化：Milvus 侧 search 时设置 `search_params={"ef": 128}` 增大召回池，在 Python 侧只对 top 20 做 MMR 去重，其余直接截断
- 长期方案：将 embedding 和 MMR 计算逻辑迁移到 Milvus 侧或专用排序服务

#### 4. Rerank 前置

当前流程：

```
dense → lexical → graph → authority → fusion → rerank
```

问题：rerank 放在最后，前面大量的分数已经被 fusion 锁定，rerank 的微调空间很小。

改为：

```
dense/lexical 粗排 → top 15 → rerank 精排 → fusion 融入 graph/authority 偏置 → MMR
```

让 rerank 在更大的语义空间中发挥作用，fusion 只是加入结构化偏置。

---

## 四、规则引擎：从硬编码到配置驱动 `[✅ Round 1 已完成]`

### 现状问题

[app/services/rules.py](app/services/rules.py) 的 `detect_states()` 是逐指标 if-elif 链，每个新增指标都需要修改代码逻辑。组合规则同样硬编码在 `_detect_composite_states()` 中。

### 优化方案

#### 1. 规则配置化

将规则定义为 YAML 配置文件（或数据库表），例如：

```yaml
rules:
  single:
    - id: fbg_diabetes
      indicator_code: fasting_blood_glucose
      operator: gte
      threshold: 7.0
      state_code: FBG_diabetes
      label: "空腹血糖达到糖尿病风险区间"
      severity: high

    - id: egfr_low
      indicator_code: egfr
      operator: lt
      threshold: 60
      state_code: eGFR_moderately_low
      label: "eGFR 中度下降"
      severity: high

  composite:
    - id: diabetes_strong_combined
      conditions:
        - indicator_code: fasting_blood_glucose
          operator: gte
          threshold: 7.0
        - indicator_code: hba1c
          operator: gte
          threshold: 6.5
      logic: AND
      state_code: DIABETES_strong_combined
      label: "空腹血糖与糖化血红蛋白共同支持糖尿病高风险"
      severity: high

    - id: elderly_hypertension_combined
      conditions:
        - field: age
          operator: gte
          threshold: 60
        - indicator_code: blood_pressure_systolic
          operator: gte
          threshold: 160
      logic: AND
      state_code: ELDERLY_HYPERTENSION_combined
      label: "老年患者收缩压显著升高"
      severity: high
```

#### 2. 引擎改造

- `IndicatorRuleEngine` 初始化时加载 YAML → 解析为 `Rule` 对象列表 → 缓存
- `detect_states()` 改为遍历规则列表 + 通用条件匹配器（支持 `gte / lte / gt / lt / eq` 操作符）
- 新增指标只需修改 YAML，无需改代码、无需重跑 CI

---

## 五、记忆层：增强召回精度 `[✅ Round 2 已完成]`

### 现状问题

[app/services/chat_history_service.py](app/services/chat_history_service.py) 的记忆系统已经结构清晰，但 `build_context()` 将所有记忆全量注入到 system prompt 中，依赖 LLM 自行判断哪些相关。当事实记忆条目较多时（反复体检的场景），prompt 膨胀迅速，触发上下文窗口限制。

### 优化方案

#### 1. 事实记忆的向量化检索

- 用户事实记忆做 embedding，存入专用 Milvus collection（`user_facts`）
- 上下文构建时，用当前用户输入做相似度检索，只注入最相关的 5 条事实
- 例如用户问"我的血糖怎么样了"，只召回 `fasting_blood_glucose=7.2` 和 `hba1c=6.8`，而不是全量 15 条指标

#### 2. 诊断记忆的时效衰减与历史追溯

- 当前诊断记忆只保存最新版本（`is_current="true"`），旧版本标记为 false 但不可被引用
- 建议保留最近 N 个诊断版本（建议 N=5），对 Agent 可见，每条标注版本号和创建时间
- Agent 可以回答趋势类问题："和上次相比我的血压有什么变化"
- 在 `build_context()` 中将最近两次诊断做 diff，如果有变化，显式标注"以下指标与上次相比有变化：..."

#### 3. 摘要记忆的触发策略

- 当前每次对话都触发生成摘要，频繁调用 LLM 增加延迟
- 改为按 token 累计阈值触发：只有当新增对话量超过一定长度（如 2000 字符）时才重新生成摘要

---

## 六、错误处理与韧性 `[✅ Round 1 + Round 2 已完成]`

### 现状问题

代码中大量 `except Exception: pass` 静默吞错：

| 位置 | 代码 | 风险 |
|------|------|------|
| `models/factory.py` 各 `_build_*` 方法 | `except Exception: return None` | 组件静默降级，前端无感知 |
| `medical_agent.py:_compose_answer` | `except Exception: pass` | LLM 格式化失败无日志 |
| `medical_agent.py:chat_assess` | `except Exception: return fallback` | Agent 异常无记录 |
| `chat_history_service.py:summary` | `except Exception: pass` | 摘要生成失败无感知 |
| `evidence_store.py:rerank` | `except Exception: return ordered` | Rerank 失败无告警 |

### 优化方案

#### 1. 分级降级策略

| 级别 | 场景 | 策略 |
|------|------|------|
| L1 | LLM 提取失败 | 回退到正则解析（已实现） |
| L1 | Rerank 失败 | 跳过 rerank 步骤（已实现） |
| L2 | Neo4j 不可达 | 自动切换 InMemoryGraphStore（需新增健康检查 + 自动 fallback） |
| L3 | Milvus 不可达 | 自动切换 InMemoryEvidenceStore |
| L3 | PostgreSQL 不可达 | 会话降级为前端 session_state 模式 |

#### 2. 结构化日志替换裸 except

引入 `logging` / `structlog`：

```python
import logging
logger = logging.getLogger(__name__)

# 替换前
except Exception:
    return None

# 替换后
except Exception:
    logger.warning("component.failed", exc_info=True, extra={"component": "reranker"})
    return None
```

- 可恢复错误（连接超时、服务暂不可用）→ WARNING 级别
- 不可恢复错误（配置错误、维度不匹配）→ ERROR 级别 + 启动时快速失败，而非运行时才暴露

---

## 七、API 层优化 `[✅ Round 3 已完成]`

### 现状问题

FastAPI 路由只做同步阻塞调用。`_execute_pipeline` 中 Neo4j 查询、Milvus 检索、Rerank HTTP 调用都是同步的，阻塞了事件循环，无法利用 FastAPI 的异步并发能力。

### 优化方案

#### 1. 异步化关键路径

- Neo4j Python driver 原生支持 async（`neo4j.AsyncGraphDatabase`），迁移成本低
- `requests.post`（rerank 调用）→ 改为 `httpx.AsyncClient` 或 `aiohttp`
- 流水线改为 `async def`，最大收益在图检索和证据检索的并行执行 — 它们之间没有依赖关系，可以 `asyncio.gather` 并发：

```python
# 改造前（串行，总耗时 = T_graph + T_evidence）
state.risk_candidates = self._graph_store.get_risk_candidates(state_codes)
state.evidence_chunks = self._evidence_store.search(queries, node_codes, top_k)

# 改造后（并行，总耗时 = max(T_graph, T_evidence)）
state.risk_candidates, state.evidence_chunks = await asyncio.gather(
    self._graph_store.get_risk_candidates_async(state_codes),
    self._evidence_store.search_async(queries, node_codes, top_k),
)
```

#### 2. 真正的流式改造

当前 `stream_assess` 实际是先完整执行完 `_execute_pipeline`，再对生成的 answer 文本做 chunk 推送。前端感知到的"流式"其实是文本级别的假流式。

改造为真正流式：流水线每完成一个阶段就立即 yield 该阶段的中间结果：

```
yield step: 解析完成 → 前端立即显示"已识别 8 个体检指标"
yield step: 规则判定完成 → 前端立即显示"发现 3 个异常状态"
yield step: 图谱检索完成 → 前端立即显示"匹配到 2 个疾病风险"
...
yield content: 最终回答逐 token 流式输出
```

前端可以在看到中间步骤时就开始渲染，而不是等全部完成后才显示。

---

## 八、可观测性 `[✅ 三轮全部完成]`

### 现状问题

除了 `stream_assess` 的 step 事件外，系统几乎没有任何内部状态可观察：

- Rerank 有没有被调用？
- Milvus 返回了多少条候选？
- LLM 提取用了多少 token？
- Agent 循环了几轮？
- 每步耗时多少？

全都不可知。

### 优化方案

#### 1. 流水线链路追踪

在每个流水线步骤记录耗时和关键指标：

| 步骤 | 指标 |
|------|------|
| 输入解析 | `parse_duration_ms`, `exam_items_count` |
| 规则判定 | `rules_duration_ms`, `detected_states_count` |
| 图谱检索 | `graph_duration_ms`, `risk_candidates_count` |
| 证据检索 | `evidence_duration_ms`, `chunks_recalled_count`, `rerank_candidates_count` |
| 排序融合 | `rank_duration_ms`, `final_risks_count` |

实现方式：用 OpenTelemetry 或简单的 `time.perf_counter()` 装饰器。

#### 2. Agent 决策日志

记录每次 Agent 迭代的关键信息：

```
[Agent Trace] iteration=1, tool=standard_assessment_tool, args_len=342, result_len=2105, duration=1.2s
[Agent Trace] iteration=2, tool=None, action=answer, tokens=156
```

出问题时可以完整回放 Agent 的决策轨迹。

#### 3. 健康检查增强

当前 `/health` 端点只返回静态配置信息。建议增强：

```json
{
  "status": "ok",
  "components": {
    "neo4j": {"status": "healthy", "latency_ms": 12, "node_count": 156},
    "milvus": {"status": "healthy", "latency_ms": 8, "chunk_count": 423},
    "postgresql": {"status": "healthy", "latency_ms": 3},
    "dashscope_embedding": {"status": "healthy"},
    "dashscope_rerank": {"status": "healthy"},
    "dashscope_llm": {"status": "healthy"}
  }
}
```

---

## 优化优先级总览

| 优先级 | 优化项 | 状态 |
|:------:|--------|:----:|
| **P0** | 消除双流水线重复 | ✅ Round 1 完成 |
| **P0** | 结构化日志替换裸 except | ✅ Round 1 完成 |
| **P1** | Agent 显式循环 | ✅ Round 2 完成 |
| **P1** | 规则配置化 | ✅ Round 1 完成 |
| **P1** | 证据检索回归 Milvus | ✅ Round 2 完成 |
| **P2** | API 异步化 | ✅ Round 3 完成 |
| **P2** | 记忆向量化检索 | ✅ Round 2 完成（趋势记忆+阈值触发）/ 完整向量化归入生产化 |
| **P2** | Rerank 前置 | ✅ Round 3 完成 |
| **P3** | 可观测性增强（流水线指标） | ✅ Round 1 完成 |
| **P3** | 健康检查增强 | ✅ Round 3 完成（组件延迟检测） |

### 实施进度

```
Round 1 (已完成): P0 双流水线 + P0 日志 + P1 规则配置化 + P3 流水线指标            → 25 tests
Round 2 (已完成): P1 Agent 显式循环 + P1 检索优化 + 条件短路 + 趋势记忆 + 自动降级 → 33 tests
Round 3 (已完成): P2 API 异步化 + P2 Rerank 前置 + 遗留清理 + 组件健康检查          → 36 tests
```

✅ 三轮优化全部完成。详细审查报告见 [PROJECT_FINAL_REVIEW_AND_SUMMARY.md](PROJECT_FINAL_REVIEW_AND_SUMMARY.md)。
