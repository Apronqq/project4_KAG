# 医疗 KAG Multi-Agent 工作台 — 全项目技术文档

> 本文档可替代逐文件阅读代码，面向需要完整理解项目架构、实现细节、技术决策和代码结构的读者。涵盖从场景分析、架构演进、逐模块设计到数据流全链路的完整说明。

---

## 一、项目定位与场景

### 1.1 做什么

用户以自然语言提交体检报告，系统自动完成：

- **指标解析**：从「男，52岁，血压 176/108 mmHg，肌酐 128 umol/L」中抽取结构化指标
- **风险判定**：基于医学阈值规则判断哪些指标异常（如收缩压 ≥ 160 属于 2 级高血压）
- **图谱推理**：从异常状态出发，通过知识图谱多跳查询关联疾病风险和建议
- **证据检索**：从医学指南、科普资料中召回支持当前判断的证据片段
- **安全复核**：检查回答中是否包含不安全的用药建议或剂量信息

### 1.2 不是什么

不是自动诊断系统，不替代医生。定位是「辅助评估」——给出风险方向、检查建议和干预方向，高风险结果强制标注「需医生复核」。

### 1.3 场景选择理由

选择体检而非自由医疗问答作为第一阶段场景，因为体检数据有三个特点让系统设计更可控：

- 输入结构化程度高（指标名 + 数值 + 单位），容易做确定性解析
- 风险判断依赖阈值规则，可以规则化而非依赖 LLM 判断
- 从「指标异常」到「风险」到「疾病」到「干预」的推理链清晰，适合图谱建模

---

## 二、整体架构

### 2.1 两图一库三存储

项目的核心计算单元由两个 LangGraph 状态图构成，共享三套存储后端：

```
┌─────────────────────────────────────────┐
│          MedicalAssessmentAgent          │  ← 对外的唯一门面
│     同步/流式/异步三套接口统一入口       │
└──────────────────┬──────────────────────┘
                   │
    ┌──────────────┴──────────────┐
    ▼                              ▼
┌───────────────────┐   ┌──────────────────────┐
│ MedicalKAGWorkflow│   │MedicalMultiAgentSup..│
│ (LangGraph 图 A)  │   │ (LangGraph 图 B)     │
│ 12 节点确定性流水线│   │ 6 Agent 协作编排      │
│ 初诊路径专用       │   │ 追问路径专用          │
└───────┬───────────┘   └────────┬─────────────┘
        │                        │
        └────────┬───────────────┘
                 ▼
    ┌────────────────────────────┐
    │   三套存储后端              │
    │  Neo4j  → 知识图谱关系推理  │
    │  Milvus → 证据向量检索     │
    │  PG     → 会话/记忆/诊断   │
    └────────────────────────────┘
```

两个 LangGraph 图分别处理两种不同的交互模式：

- **图 A（MedicalKAGWorkflow）**：12 节点串联，处理初诊体检评估。所有节点是确定性代码，LLM 只在最后一步做自然语言格式化。这条路径不可被 LLM 绕过。
- **图 B（MedicalMultiAgentSupervisor）**：6 Agent 条件分支，处理追问。TriageAgent 分流，MemoryAgent 判断是否需要检索，SynthesisAgent 合成回答，SafetyReviewAgent 安全复核。

### 2.2 为什么是两个图而不是一个大图

初诊和追问的安全假设完全不同：

- **初诊**需要「零幻觉容忍」：每个判断步骤必须可追溯、可审计。如果 LLM 在中间参与了推理，就无法保证它没有跳过某个异常指标。
- **追问**需要「上下文灵活应答」：用户问「我的血压严不严重」，需要结合之前的诊断记忆回答，不是跑一遍完整流水线。

如果合并在一个图里，需要通过大量的条件边来区分两条路径，反而增加了编排复杂度。拆成两个独立的图，各自由 Supervisor 的路由决策选择调用，职责更清晰。

---

## 三、代码分层：每层解决什么问题

```
app/
├── core/settings.py              # 第 1 层：配置
├── schemas/exam.py               # 第 2 层：数据模型
├── db/                           # 第 3 层：关系持久化
├── graph/                        # 第 4 层：图谱存储
├── retrieval/                    # 第 5 层：向量检索
├── models/factory.py             # 第 6 层：模型工厂
├── services/                     # 第 7 层：业务逻辑
├── workflows/                    # 第 8 层：确定性流水线
├── agents/                       # 第 9 层：Multi-Agent 编排
└── api/routes/medical.py         # 第 10 层：HTTP 接口
```

### 3.1 配置层 — `core/settings.py`

**解决的问题**：所有环境变量的集中管理与默认值兜底。

核心设计决策：
- 使用 `@dataclass(frozen=True)` 确保配置对象不可变
- `__post_init__` 中自动将 `evidence_embedding_dim` 对齐到 `dense_embedding_dim`，防止 Milvus 维度不匹配
- 支持 `USE_IN_MEMORY_GRAPH=true` 和 `USE_IN_MEMORY_EVIDENCE=true` 两个开关，不装 Neo4j/Milvus 也能用内存模式启动

```python
@dataclass(frozen=True)
class Settings:
    neo4j_uri: str = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    milvus_uri: str = os.getenv("MILVUS_URI", "http://localhost:19530")
    use_in_memory_graph: bool = _as_bool(os.getenv("USE_IN_MEMORY_GRAPH"), False)
    top_k_evidence: int = int(os.getenv("TOP_K_EVIDENCE", "5"))
    summary_trigger_chars: int = int(os.getenv("SUMMARY_TRIGGER_CHARS", "2000"))
    # ... 共 21 项可配置
```

### 3.2 数据模型层 — `schemas/exam.py`

**解决的问题**：全系统共享的类型定义，数据在模块间传递时不会因为 typo 或缺失字段而崩溃。

关键的设计原则：**描述数据在系统中的完整生命周期，不只描述 API 的入参和出参**。

```python
# 用户输入的结构化表示
class NormalizedMedicalExamJSON(BaseModel):
    patient_profile: PatientProfile       # 性别、年龄
    exam_items: list[ExamItem]            # 指标名、值、单位、原始文本
    medical_history: list[str]            # 既往史
    current_medications: list[str]        # 当前用药
    allergies: list[str]                  # 过敏史
    user_question: str                    # 用户原始问题

# 流水线的共享状态 — 贯穿 12 个节点的"数据总线"
class InternalAssessmentState(BaseModel):
    raw_input: Any
    normalized_exam_json: NormalizedMedicalExamJSON | None = None
    detected_states: list[DetectedState] = []     # 规则引擎产出
    risk_candidates: list[RiskCandidate] = []      # 图谱检索产出
    evidence_chunks: list[EvidenceChunk] = []      # 证据检索产出
    response: MedicalAssessmentResponse | None = None  # 最终产出
```

`InternalAssessmentState` 是整个 Workflow 的共享内存。测试时不需要 mock 六个函数的返回值，只需要构造一个 State 对象。

### 3.3 数据库层 — `db/database.py` + `db/models.py`

**解决的问题**：会话、消息、事实记忆、诊断记忆的持久化存储。

四张核心表：

| 表 | 用途 | 关键字段 |
|---|------|---------|
| `chat_sessions` | 会话管理 | session_id, title, conversation_summary, summary_pending_chars |
| `conversation_memories` | 对话消息 | role, content, content_summary |
| `user_fact_memories` | 用户事实 | fact_group, fact_key, fact_value, fact_unit, confidence |
| `diagnostic_memories` | 诊断快照 | version_no, is_current, health_status, risk_summary, department_summary |

`database.py` 中的 `_migrate_chat_schema()` 实现了零停机的增量 schema 升级：启动时检查表结构，缺失的列用 `ALTER TABLE ADD COLUMN` 补充，兼容旧表。

### 3.4 图谱存储层 — `graph/store.py` + `graph/seed_data.py` + `graph/kb_builder.py`

**解决的问题**：用图数据库表达「指标异常 → 风险 → 疾病 → 干预/科室/复查/禁忌」的多跳推理链路。

#### 3.4.1 Schema 设计

```
IndicatorState (指标异常状态，如 SBP_high_stage2)
    │
    │ STATE_IMPLIES_RISK
    ▼
DiseaseRisk (疾病风险，如 hypertension_risk)
    │
    │ RISK_RELATED_DISEASE
    ▼
Disease (疾病，如 hypertension)
    ├── DISEASE_RECOMMENDS_INTERVENTION → Intervention
    ├── DISEASE_RECOMMENDS_DEPARTMENT → Department
    ├── DISEASE_REQUIRES_FOLLOWUP_TEST → FollowUpTest
    ├── DISEASE_HAS_CONTRAINDICATION → Contraindication
    └── DISEASE_RECOMMENDS_MEDICATION_DIRECTION → MedicationDirection
```

选择六种节点类型 + 五种关系类型的理由：体检评估需要从「一个指标异常」推理到「该去哪个科室做什么检查」，逻辑上就是 3-4 跳的路径，不是一步检索能完成的。

#### 3.4.2 双后端设计

```python
class BaseGraphStore(ABC):
    @abstractmethod
    def get_risk_candidates(self, state_codes): ...
    @abstractmethod
    def get_intervention_candidates(self, disease_codes): ...

class Neo4jGraphStore(BaseGraphStore):
    """生产环境：Cypher 多跳查询 + AsyncGraphDatabase 异步支持"""

class InMemoryGraphStore(BaseGraphStore):
    """回退模式：Python dict，零外部依赖"""

def build_graph_store(settings):
    """自动选择：Neo4j 可用走 Neo4j，不可用自动降级 InMemory"""
    if settings.use_in_memory_graph:
        return InMemoryGraphStore()
    try:
        store = Neo4jGraphStore(settings)
        if store.ping():
            return store
        logger.warning("neo4j ping failed, falling back")
    except Exception:
        logger.warning("neo4j init failed, falling back")
    return InMemoryGraphStore()
```

`build_graph_store()` 中 ping 失败后的自动 fallback 逻辑是韧性设计的核心：外部依赖不可用时系统不崩溃，而是降级运行。

#### 3.4.3 种子数据来源

`seed_data.py` 中的三条映射关系的来源：

- `INDICATOR_ALIASES`：临床检验报告中常见的中文名、英文缩写变体，手工整理
- `STATE_TO_RISK`：来自高血压/糖尿病/血脂异常/CKD/肝功能指南的诊断标准阈值，编码为图谱关系
- `DISEASE_TO_INTERVENTIONS`：来自指南中有明确推荐级别的干预方向，用药方向用「评估是否需要」而非「开具 XX 药物」的表述方式

上传的外部文档不能修改图谱结构——它们只会通过 `NODE_LINKED_CHUNK` 边绑定到已有节点作为证据片段。这是刻意的设计约束：**证据可以自动积累，知识结构必须人工校验**。

### 3.5 检索层 — `retrieval/evidence_store.py` + `retrieval/lexical.py` + `retrieval/risk_ranker.py`

**解决的问题**：从知识库中高效地找到与当前评估最相关的医学证据片段。

#### 3.5.1 五步递进链路

```
Step 1: 多查询规划       → 生成最多 8 个维度查询（用户问题/疾病指南/指标分层/聚合总结）
Step 2: 双路 RRF 召回    → 稠密向量 (Milvus HNSW) + 词汇倒排 (SQLite FTS5)
Step 3: Rerank 前置      → 远程语义模型精排（权重 45%）
Step 4: 多信号融合       → Rerank + 图谱节点重叠度(20%) + 词汇(15%) + 权威度(10%)
Step 5: MMR 多样性       → top 15 候选去重，λ=0.75
```

#### 3.5.2 为什么 Rerank 放在 fusion 前面

传统方案中 Rerank 在 fusion 之后，只能对已经融合好的结果做微调（权重 25%）。我把 Rerank 前移到 fusion 之前，给它 45% 的最高权重——Rerank 是五路信号中最强的单信号，应该主导排序方向。Ablation 实验验证这个决策贡献了 +11% MRR。

#### 3.5.3 图谱节点重叠度信号

这是本项目检索最独特的信号。chunk 入库时自动推断 `linked_node_codes`（关联了哪些图谱节点）；检索时 query 触发的图谱路径上的节点码集合与 chunk 的关联节点做交集计算，交集比例越高 score 越高：

```
chunk 标签: [hypertension_risk, hypertension]
query 路径: [hypertension_risk, hypertension, ckd_risk, ckd]
重叠: 2/2 = 1.0（满分）

chunk 标签: [diabetes_risk, type2_diabetes]
query 路径: [hypertension_risk, hypertension, ckd_risk, ckd]
重叠: 0/2 = 0.0（无关联）
```

这个信号的作用：用确定性图谱关系去约束概率性语义检索——语义模型只判断「文本和 query 是否相似」，图谱信号判断「文本讲的是否恰是当前推理链路涉及的医学概念」。

#### 3.5.4 证据注入的双后端设计

```python
def build_evidence_store(settings, embedder, reranker):
    if settings.use_in_memory_evidence:
        return InMemoryEvidenceStore(embedder, reranker)
    try:
        store = MilvusEvidenceStore(settings, embedder, reranker)
        if store.ping():
            return store
        logger.warning("milvus ping failed, falling back")
    except Exception:
        logger.warning("milvus init failed, falling back")
    return InMemoryEvidenceStore(embedder, reranker)
```

#### 3.5.5 词汇检索的 SQLite FTS5 实现

```python
class SQLiteFTSIndex:
    def __init__(self, db_path):
        self._conn = sqlite3.connect(str(db_path))
        self._conn.execute(
            "CREATE VIRTUAL TABLE IF NOT EXISTS docs_fts "
            "USING fts5(chunk_id UNINDEXED, title, text, linked_codes)"
        )

    def search(self, query_text, top_k):
        # FTS5 BM25 原生排序
        rows = self._conn.execute(
            "SELECT chunk_id, bm25(docs_fts) FROM docs_fts "
            "WHERE docs_fts MATCH ? ORDER BY score LIMIT ?",
            (match_query, top_k),
        ).fetchall()
        # FTS5 无法匹配中文时回退 LIKE 搜索
        if not rows:
            rows = self._fallback_like_search(query_text, top_k)
```

选择 SQLite FTS5 而非 ES 的理由：零外部依赖，适合原型和本地部署。chunk 上万后性能远优于 Python 侧内存 BM25 遍历。

#### 3.5.6 风险排序融合

`risk_ranker.py` 中的 `rank_risks()` 将图谱推理分数和证据检索分数加权融合：

```python
final_score = (
    0.45 * base_graph_score      # 图谱结构置信度
    + 0.20 * evidence_score       # 证据检索支持度
    + 0.20 * support_strength     # 异常状态严重程度加权
    + 0.15 * support_count_score  # 多少独立异常状态支持这个风险
)
```

### 3.6 模型工厂层 — `models/factory.py`

**解决的问题**：统一管理 LLM、Embedding、Rerank、结构化提取器四种外部模型能力的构建与降级。

```python
class ModelFactory:
    def build(self) -> ModelRuntime:
        return ModelRuntime(
            embedding_provider=self._build_embedding_provider(),  # DashScope 或轻量 hash
            extractor=self._build_extractor(),                    # LLM 结构化提取
            reranker=self._build_reranker(),                      # 远程 Rerank
            assistant_chat_model=self._build_chat_model(),        # 对话 LLM
        )
```

每类能力都有降级策略：DashScope Embedding 失败时回退到 `LightweightEmbeddingProvider`（基于 hash 的向量生成），Extractor 失败时回退到正则解析，Chat 失败时回退到确定性模板。

### 3.7 业务逻辑层 — `services/`

#### 3.7.1 输入解析 — `input_parser.py` + `indicator_normalizer.py`

双通道解析：

- **第一通道（LLM 提取）**：`ENABLE_LLM_INPUT_PARSING=true` 时用 LLM 的 `with_structured_output` 抽取结构化字段
- **第二通道（正则回退）**：LLM 失败或未配置时，用正则匹配血压 `\d{2,3}/\d{2,3}`、血糖 `空腹血糖\s*[:：]?\s*数字` 等模式

归一化器做三层转换：别名映射（FBG → fasting_blood_glucose）→ 单位标准化（mmol/l → mmol/L）→ 单位换算（肌酐 mg/dL → umol/L ×88.4）。

#### 3.7.2 规则引擎 — `rules.py` + `config/medical_rules.json`

将原本的 if/elif 硬编码规则重构为 JSON 配置驱动：

```json
{
  "id": "creatinine_high",
  "indicator_code": "creatinine",
  "conditions": [{
    "logic": "OR",
    "conditions": [
      {"logic": "AND", "conditions": [
        {"field": "patient_profile.sex", "operator": "eq", "value": "male"},
        {"field": "value", "operator": "gte", "value": 110}
      ]},
      {"logic": "AND", "conditions": [
        {"field": "patient_profile.sex", "operator": "eq", "value": "female"},
        {"field": "value", "operator": "gte", "value": 90}
      ]}
    ]
  }]
}
```

引擎变成通用的条件匹配器，支持 7 种操作符（gte/gt/lte/lt/eq/in/not_in）和嵌套 AND/OR 逻辑。新增指标只需修改 JSON，不需要改代码。

#### 3.7.3 会话记忆 — `chat_history_service.py`

四层记忆模型，按信息的确定性分层：

| 记忆层 | 确定性 | 写入策略 | 读取策略 |
|--------|:----:|---------|---------|
| 事实记忆 | 高 | 每次新体检数据全量覆盖写入，冲突检测新旧对比 | 向量化语义召回 top 5 |
| 诊断记忆 | 高 | 版本化（version_no + is_current），旧版标记 false | 追问注入最近两版做趋势 diff |
| 对话记忆 | 低 | 每轮对话写入，双字段（原文 + 500字符摘要） | 三级长度控制：单条500/12条/总3200 |
| 摘要记忆 | 低 | 累计字符 ≥ 2000 触发 LLM，否则确定性拼接 | 直接注入 system prompt |

上下文超出处理：`_clip_total_history()` 从新到旧累积，新消息优先保留。system prompt（事实记忆 + 诊断记忆）放在最前面，倒序处理时最后被截，保证不被截掉。

#### 3.7.4 知识库管理 — `document_ingestion.py` + `knowledge_registry.py` + `upload_job_registry.py`

文档上传链路的六层防污染：

```
文件类型白名单（txt/md/pdf/json/html）
  → SHA-256 内容哈希去重
  → 文本抽取判空（扫描版 PDF 直接拒绝）
  → 图谱节点推断（linked_node_codes 为空的文档标记 unverified）
  → 医学相关性门控（三维评分 < 0.15 直接 rejected）
  → 来源权威度降权（unverified=0.3 vs guideline=1.0）
```

分块策略：chunk_size=500, overlap=80，滑动窗口切分。每个 chunk 入库时调用 `_infer_linked_node_codes()` 自动判断它关联了哪些图谱节点。

#### 3.7.5 依赖注入容器 — `container.py`

```python
_runtime: AppRuntime | None = None  # 进程级单例

def get_runtime() -> AppRuntime:
    """懒加载：首次调用构建全部 20 个组件，后续调用直接返回"""
    global _runtime
    if _runtime is not None:
        return _runtime

    settings = Settings()
    graph_store = build_graph_store(settings)       # Neo4j / InMemory
    evidence_store = build_evidence_store(...)       # Milvus / InMemory
    workflow = MedicalKAGWorkflow(graph_store=..., evidence_store=...)
    agent = MedicalAssessmentAgent(workflow=workflow,
                                   memory_context_builder=chat_history_service.build_context)

    _runtime = AppRuntime(settings=settings, graph_store=graph_store, ...)
    return _runtime
```

单例容器的作用：所有组件共享同一实例，避免重复创建数据库连接池、图驱动、向量客户端。

### 3.8 Workflow 层 — `workflows/medical_kag_pipeline.py`

**解决的问题**：初诊体检评估的确定性执行引擎，12 个节点串联，不可被 LLM 绕过。

#### 3.8.1 12 节点流水线

```
parse_raw_input          → 文本 → 结构化 JSON
validate_exam_json       → 校验解析结果完整性
normalize_exam_items     → 指标别名映射 + 单位归一化
detect_indicator_states  → JSON 配置规则判定
retrieve_graph_candidates→ Neo4j 多跳查询风险候选
expand_intervention_paths→ 扩展干预/科室/复查/禁忌
plan_evidence_queries    → 多维度检索查询规划
retrieve_evidence_chunks → Milvus + FTS5 双路召回
rank_medical_evidence    → 图谱 + 证据 + 严重度加权
generate_primary_diagnosis   → 健康状态 + 风险列表
generate_secondary_recommendation → 科室/复查/生活方式/用药
format_medical_response   → 封装为 API 响应
```

#### 3.8.2 PipelineStep 统一定义

所有步骤的元信息只在一处维护：

```python
def _step_definitions(self) -> list[PipelineStep]:
    return [
        PipelineStep("parse_raw_input", "解析用户输入", "...", self._parse_raw_input_node),
        PipelineStep("validate_exam_json", "校验体检结构", "...", self._validate_exam_json_node),
        # ... 共 12 个
    ]
```

LangGraph 图构建、同步执行、流式事件遍历、异步执行都循环同一份步骤定义。这叫「单一事实来源」——改顺序或增删步骤只需改一个地方。

#### 3.8.3 条件短路

```python
def _should_skip_step(self, step_name, state):
    if step_name in {"retrieve_graph_candidates", "retrieve_evidence_chunks", ...}:
        if not state.detected_states:  # 无异常 → 跳过图谱和证据
            return True
    if step_name == "expand_intervention_paths" and not state.risk_candidates:
        return True  # 图谱未命中 → 跳过干预扩展
    return False
```

健康路径延迟从 ~350ms 降至 ~50ms。

#### 3.8.4 异步并行执行

```python
async def _retrieve_evidence_and_interventions_async(self, payload):
    """证据检索与干预路径扩展互不依赖 → 并行执行"""
    evidence_task = asyncio.create_task(self._evidence_store.search_async(...))
    intervention_task = asyncio.create_task(self._graph_store.get_intervention_candidates_async(...))
    state.evidence_chunks, state.intervention_candidates = await asyncio.gather(
        evidence_task, intervention_task
    )
```

#### 3.8.5 流式事件

`iter_events()` 每步产出两个事件（running + completed），附带 `duration_ms`、指标计数和结构化成功消息。前端能实时看到「已识别 8 个体检指标」「发现 3 个异常状态」「匹配到 2 个疾病风险候选」。

### 3.9 Multi-Agent 层 — `agents/medical_multi_agent.py`

**解决的问题**：追问路径的智能编排——不是简单地「查知识库 + LLM 回答」，而是六个专职 Agent 分工协作。

#### 3.9.1 六个 Agent 的职责

```
┌──────────────────────────────────────────────────────────────┐
│              MedicalMultiAgentSupervisor                     │
│                                                              │
│  TriageAgent ─→ 判断是初诊还是追问                            │
│       │                                                      │
│       ├── assessment ─→ AssessmentAgent ─→ 执行 Workflow    │
│       │                                            │         │
│       └── followup ──→ MemoryAgent ─→ 判断是否需要检索      │
│                              │                               │
│                    ┌─────────┴──────────┐                    │
│                    ▼                     ▼                    │
│              RetrievalAgent        SynthesisAgent            │
│              (需要检索)             (不需要检索)              │
│                    │                     │                    │
│                    └─────────┬───────────┘                    │
│                              ▼                               │
│                       SynthesisAgent                         │
│                              │                               │
│                              ▼                               │
│                      SafetyReviewAgent                       │
│                      (可能路由回 SynthesisAgent 安全改写)      │
└──────────────────────────────────────────────────────────────┘
```

#### 3.9.2 流式事件的真正实时产出

`iter_events()` 不是先 `invoke()` 完整图再 yield 事件，而是逐 Agent 执行并实时产出：

```python
def iter_events(self, user_input, session_history=None, *, session_id=None):
    state = {...}  # 初始化

    yield from self._run_node_for_stream(state, self._triage_agent)      # 实时产出 triage 事件
    if state["route"] == "assessment":
        yield from self._stream_assessment_agent(state)                  # 透传 Workflow 12 节点事件
    else:
        yield from self._run_node_for_stream(state, self._memory_agent)  # 实时产出 memory 事件
        if state["needs_retrieval"]:
            yield from self._run_node_for_stream(state, self._retrieval_agent)
        yield from self._run_node_for_stream(state, self._synthesis_agent)

    while True:  # 安全复核 → 可能改写 → 再复核
        yield from self._run_node_for_stream(state, self._safety_review_agent)
        if not state["requires_safe_rewrite"]:
            return
        yield from self._run_node_for_stream(state, self._synthesis_agent)
```

`_stream_assessment_agent()` 内部透传 `self._workflow.iter_events()` 的全部 12 节点细粒度事件。

#### 3.9.3 MemoryAgent 的可注入设计

```python
class MedicalMultiAgentSupervisor:
    def __init__(self, *, memory_context_builder=None, ...):
        self._memory_context_builder = memory_context_builder

    def _resolve_session_history(self, state):
        if self._memory_context_builder is not None and state.get("session_id"):
            context = self._memory_context_builder(state["session_id"], state["user_input"])
            return getattr(context, "history", [])
        return state.get("session_history", [])
```

生产环境注入 `chat_history_service.build_context` → MemoryAgent 能主动读取四层记忆；测试环境注入假的 context builder → 不依赖数据库。这就是依赖注入的价值：不绑定具体实现。

#### 3.9.4 RetrievalAgent 的记忆感知查询扩展

```python
@staticmethod
def _build_followup_retrieval_query(user_input, memory_text):
    diseases = [term for term in DISEASE_TERMS if term in memory_text]
    if not diseases:
        return user_input  # 无疾病上下文 → 直接用原问题
    suffix = " ".join(diseases[:3])
    if any(token in user_input for token in ["饮食", "怎么吃"]):
        suffix += " 饮食管理 指南"           # 问题类型感知
    return f"{user_input} {suffix}"
```

用户输入「我早餐能吃什么」→ 从记忆中提取到「高血压」→ 改写为「我早餐能吃什么 高血压 饮食管理 指南」→ 检索有了明确的疾病上下文。

#### 3.9.5 SafetyReviewAgent 的多维度审核

```python
def _safety_review_agent(self, state):
    answer = state["answer"]
    medication_risk = self._mentions_medication_adjustment(user_input) or \
                      self._contains_medication_advice(answer)        # 检查回答是否含用药建议
    dosage_risk = self._contains_dosage_instruction(answer)           # 正则检测 50mg/1片/次/日
    suspected_drugs = self._extract_suspected_drug_terms(answer)      # 提取 沙坦/他汀/二甲双胍

    if (dosage_risk or suspected_drugs) and state.get("safety_revision_count", 0) == 0:
        requires_rewrite = True  # → 通过 LangGraph 条件边回 SynthesisAgent 安全改写
```

如果检测到具体剂量或疑似药名，条件边会把流程路由回 SynthesisAgent 生成不含处方建议的安全改写版回答。这比单纯在回答末尾追加一句「请咨询医生」更彻底。

#### 3.9.6 构建工厂函数

```python
def build_medical_multi_agent_supervisor(*, workflow, knowledge_tool, chat_model,
    assessment_answer_builder, followup_answer_builder, initial_assessment_detector,
    memory_context_builder=None,
) -> "MedicalMultiAgentSupervisor":
```

测试和生产都通过同一个工厂函数创建 Supervisor，避免参数不一致。

### 3.10 Agent 门面层 — `services/medical_agent.py`

**解决的问题**：对外暴露统一的 `MedicalAssessmentAgent` 门面，内部委托给 Multi-Agent Supervisor。

```python
class MedicalAssessmentAgent:
    def __init__(self, ..., workflow, memory_context_builder=None):
        self._workflow = workflow                          # KAG 确定性流水线
        self._multi_agent_supervisor = build_medical_multi_agent_supervisor(
            workflow=self._workflow,                       # 传入同一个 workflow 实例
            memory_context_builder=memory_context_builder, # ChatHistoryService.build_context
            ...
        )

    def stream_assess(self, raw_text, session_history=None, session_id=None):
        """统一走 Multi-Agent 编排"""
        yield from self._stream_multi_agent(raw_text, session_history or [], session_id=session_id)

    def _stream_multi_agent(self, user_input, session_history, session_id=None):
        for event in self._multi_agent_supervisor.iter_events(
            user_input, session_history, session_id=session_id
        ):
            if event.get("internal"):
                continue  # 内部事件不暴露给前端
            if event["type"] == "final_answer":
                answer = event["content"]
                continue
            yield event  # 流式事件直接透传
        # 事件之后是文本流式输出
        for chunk in self._chunk_text(answer, size=20):
            yield {"type": "content", "content": chunk}
        # 结构化评估结果用于会话记忆写入
        if structured_response is not None:
            yield {"type": "result", "payload": structured_response.model_dump()}
        yield {"type": "done"}
```

`_looks_like_initial_assessment()` 的判定逻辑同时检测医学关键词、数值和意图词，避免「我的血压怎么样了」被误判为初诊。

### 3.11 API 层 — `api/routes/medical.py`

FastAPI 路由层，提供四类接口：

- **评估接口**：parse/assess/chat/chat/stream
- **知识库接口**：upload/documents/jobs/rebuild
- **会话接口**：sessions CRUD
- **状态接口**：/health + /medical/runtime/status

流式接口返回 SSE，事件类型包括 meta/agent_decision/agent_thinking/step/tool_call/tool_result/agent_synthesizing/content/result/memory_notice/done。

---

## 四、完整数据流：从用户输入到最终回答

以「男，52岁，血压 176/108 mmHg，肌酐 128 umol/L，eGFR 54」初诊为例：

```
1. POST /medical/agent/chat/stream
   → MedicalAssessmentAgent.stream_assess(raw_text, session_id=None)

2. TriageAgent
   → _looks_like_initial_assessment("男，52岁，血压 176/108...")
   → 命中医学关键词 + 数值 → route = "assessment"

3. AssessmentAgent → _workflow.iter_events()

   3.1 parse_raw_input
       → LLM 提取 → PatientProfile(sex="male", age=52)
       → ExamItem("收缩压", 176, "mmHg"), ExamItem("肌酐", 128, "umol/L"), ...

   3.2 normalize_exam_items → "收缩压" → blood_pressure_systolic, 176 mmHg

   3.3 detect_indicator_states
       → SBP ≥ 160 → SBP_high_stage2 (high)
       → DBP ≥ 100 → DBP_high_stage2 (high)
       → CREATININE ≥ 110 → CREATININE_high (high)
       → eGFR < 60 → eGFR_moderately_low (high)
       → 组合规则：SBP+DBP → BP_stage2_combined
       → 组合规则：eGFR低+肌酐高 → CKD_strong_combined

   3.4 retrieve_graph_candidates
       → Neo4j: SBP_high_stage2 → hypertension_risk → hypertension
                 CREATININE_high → ckd_risk → ckd
                 eGFR_moderately_low → ckd_risk (汇聚)
       → 风险候选: [hypertension(0.98), ckd(0.96)]

   3.5 expand_intervention_paths
       → hypertension: 心内科, 动态血压, 限盐, ACEI/ARB评估
       → ckd: 肾内科, 肌酐复查, 控制血压, 避免肾毒性药物

   3.6 plan_evidence_queries → 8 个维度查询

   3.7 retrieve_evidence_chunks
       → Milvus dense + SQLite FTS5 lexical → RRF 融合
       → Rerank 前置精排
       → 图谱节点重叠度加权 + 来源权威度 + MMR 去重
       → 5 条最相关证据片段

   3.8 rank_medical_evidence → 图谱 0.98 + 证据支持 0.85 → final_score 0.93

   3.9 generate_primary_diagnosis → health_status=high_risk, urgency=urgent

   3.10 generate_secondary_recommendation
        → 科室: 心内科, 肾内科
        → 复查: 动态血压, 肌酐, eGFR, 尿蛋白
        → 干预: 限盐, 控制血压, 避免肾毒性药物
        → human_review_required=True

   3.11 format_medical_response → MedicalAssessmentResponse

4. SafetyReviewAgent
   → human_review_required=True → 追加人工复核提示
   → 无用药剂量风险 → 无需安全改写 → END

5. 流式输出
   → agent_thinking → step×12 → assessment_result → agent_decision → final_answer
   → content chunks → result → done

6. 异步后处理（路由层）
   → record_user_message + upsert_user_fact_memory + record_assistant_message
```

---

## 五、技术选型记录与理由

| 选型 | 选择 | 核心理由 |
|------|------|---------|
| Agent 框架 | 自研 ReAct → 升级为 LangGraph Supervisor | 原始 `create_agent` 是黑盒，自研 ReAct 可控但不够工程化，最终收敛为 LangGraph 图编排——兼顾了可观测性和工程化程度 |
| 流水线引擎 | LangGraph StateGraph | 12 节点有明确顺序，LangGraph 提供了图可视化、条件边、状态管理等开箱即用的能力 |
| 图谱数据库 | Neo4j | 医学推理链路「指标 → 风险 → 疾病 → 干预」多跳关系用 Cypher 表达自然，Python 驱动成熟 |
| 向量数据库 | Milvus | HNSW 索引成熟稳定，支持 native distance 复用和动态 schema |
| 词汇检索 | SQLite FTS5 | 零外部依赖，BM25 原生支持，中文 fallback LIKE 兜底 |
| 会话存储 | PostgreSQL + SQLAlchemy | 四层记忆都是强结构化数据，ORM 约束和迁移成熟 |
| 模型接入 | DashScope API | 统一提供 Embedding/Rerank/LLM/Extractor 四种能力 |
| 前端 | Streamlit | 左右分栏工作台、会话/知识库/状态管理，快速原型开发 |
| KAG 方法论 | 借鉴而非引入完整平台 | schema-constrained construction + graph-guided reasoning 的思想融入自研轻量实现 |

---

## 六、创新点与设计精髓

### 6.1 Multi-Agent 分工而非 Multi-LLM 聊天

六个 Agent 各司其职——TriageAgent 是规则判断、AssessmentAgent 是确定性流水线、MemoryAgent 是记忆查询器、RetrievalAgent 是检索工具调用器、SynthesisAgent 是回答合成器、SafetyReviewAgent 是安全复核器。Agent 间不互相对话协商，而是通过共享状态（`MedicalMultiAgentState`）传递结构化中间产物。这比「让多个 LLM 互相聊天」的模式更可控、可测试。

### 6.2 初诊 / 追问路径的安全差异设计

初诊强制走 12 节点确定性流水线，LLM 不可绕过任何一个判断步骤；追问允许 LLM 在约束下做工具调用，但工具选择策略被限制为「记忆优先 → 检索补充」。两条路径通过同一个 Supervisor 入口自动分流。

### 6.3 图谱信号作为检索排序的独立维度

不是把知识图谱当作另一个检索源，而是将图谱推理路径上的节点码作为「领域相关性锚点」注入排序——chunk 的 linked_node_codes 与 query 触发的图谱节点做交集计算。确定性知识约束概率性检索。

### 6.4 记忆按确定性分层

不把历史消息全塞进 prompt。高确定性记忆（事实、诊断）用规则管理——冲突检测、版本化、趋势 diff；低确定性记忆（对话、摘要）用阈值和裁剪控制。LLM 只消费整理好的记忆输入，不负责记忆管理。

### 6.5 安全复核的条件改写

SafetyReviewAgent 不只是在回答末尾追加提示——当检测到具体剂量或疑似药名时，通过 LangGraph 条件边把流程路由回 SynthesisAgent 重新生成安全版本。这是「发现风险 → 修正输出」的闭环，不是「发现风险 → 贴个标签」。

### 6.6 全链路降级而非单点兜底

Neo4j 不可达 → InMemoryGraphStore，Milvus 不可达 → InMemoryEvidenceStore，LLM 不可达 → 确定性模板回答，Embedding 不可达 → 轻量 hash 回退。每层都有独立降级策略，降级原因通过健康接口暴露。

---

## 七、复现步骤

```bash
# 1. 安装
pip install fastapi uvicorn langgraph langchain langchain-community \
    neo4j pymilvus sqlalchemy psycopg2-binary pydantic \
    streamlit requests pypdf python-dotenv pytest httpx

# 2. 数据库
createdb medical_agent

# 3. 配置 .env
DASHSCOPE_API_KEY=sk-xxx
BOOTSTRAP_KB_ON_STARTUP=true

# 4. 启动后端
uvicorn app.main:app --port 8000

# 5. 启动前端
streamlit run streamlit_app.py

# 6. 测试
pytest tests/ -v
```

如不安装 Neo4j 和 Milvus，设置 `USE_IN_MEMORY_GRAPH=true` 和 `USE_IN_MEMORY_EVIDENCE=true` 即可使用内存模式快速体验。
