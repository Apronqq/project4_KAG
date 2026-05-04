# 医疗 KAG Agent 工作台 — 全流程详解、设计思路与深度面试 Q&A

> 本文档面向需要完全理解并复现此项目的人员，涵盖从零到一的设计推导、逐模块代码走查、以及面试视角的深度问答。

---

## 第一部分：面试提问风格与回答技巧总结

通过对上面多轮面试模拟的观察，面试官的提问遵循 **"技术决策 Why → 实现细节 How → 边界情况 What if"** 的三段式递进结构。掌握这个规律比背诵答案更重要。

### 面试官的典型出题模式

| 阶段 | 问题特征 | 示例 |
|------|---------|------|
| Why | 追问技术选型的理由，为什么选A不选B | "为什么自建KAG而不是用OpenSPG？" |
| How | 深入具体实现细节 | "图谱节点重叠度信号具体是怎么计算的？" |
| What if | 假设边界场景，考察防御性思维 | "如果上传了无关文件怎么办？" |

### 回答的黄金公式

**承认有效性 + 解释你的场景差异 + 给出工程权衡的结论**

错误的回答：「OpenSPG 太重了，不适合我们。」—— 这是主观判断，容易被追问「你怎么判断的？」

正确的回答：「OpenSPG 的核心优势在 A、B、C 三个维度。但在我的场景下，A 不生效（因为 schema 已手工确定），B 已被我的映射表覆盖，C 和我的安全边界要求冲突。所以我借鉴了它的方法论思想，在轻量栈上落地。这不是谁好谁差的问题，是工具匹配场景的问题。」

这个公式的精髓在于：先承认对方工具/方案的价值（显示你了解它），再说清楚你的约束条件为什么让它不适用（显示你的决策推演能力），最后给出权衡结论（显示你的工程判断力）。

### 数据和技术名词是回答的骨架

```
❌ "我们的检索效果挺好的"
✅ "Ablation 验证 Rerank 贡献 +11% MRR，图谱信号贡献 +9%，全链路基线提升 27.5%"
```

能用数字的地方不要用形容词。

---

## 第二部分：项目全流程详解

### 2.1 背景与立意 — 为什么做这个项目

#### 原始起点

项目最初继承了一个通用企业文档 RAG Demo 的代码骨架（FastAPI + LangChain + Milvus + PostgreSQL）。这个 Demo 的交互模式是「用户问 → 向量检索 → LLM 回答」，适合 FAQ 类场景，但对医疗体检场景完全不适用。

#### 场景分析

体检评估的核心不是「查一段相似文本然后回答」，而是：

- 指标阈值判断（血压 ≥ 160 是 2 级高血压，这是确定性规则，不应该让 LLM 决定）
- 风险路径推理（血压高 → 高血压风险 → 高血压 → 需要哪些干预？这是关系链推理）
- 证据检索（我的判断有依据吗？需要引用指南原文）
- 多轮追问（评估完之后用户还会问饮食、用药、复查等后续问题）

单一的 RAG 链路只能覆盖「证据检索」这一个环节。

#### 技术路线的选择

面临两条路：一是直接在原 RAG 项目上修修补补，二是重新设计架构。选择后者，原因很简单——如果架构地基是「检索 → 回答」的单步管道，往上加规则引擎和图谱推理就只能打补丁，补丁叠补丁的代码维护成本远高于重写。

新项目的技术定位确定为**KAG-lite**：借鉴 OpenSPG/KAG 的核心方法论（schema-constrained construction、knowledge/chunk mutual indexing、graph-guided reasoning、hybrid retrieval），但不引入完整平台，在现有 Python 栈上自研轻量实现。

---

### 2.2 项目骨架搭建 — 先跑通一条链路

#### 目录结构设计

在动手写任何业务逻辑之前，先把代码的物理边界定清楚：

```
app/
├── core/          # 配置（环境变量、数据库/图谱/向量连接、模型参数）
├── db/            # PostgreSQL 数据层（连接管理、ORM 模型、schema 迁移）
├── graph/         # 知识图谱（Neo4j 存储、种子数据、知识库构建编排）
├── retrieval/     # 证据检索（Milvus 向量存储、词汇索引、排序融合）
├── models/        # 模型工厂（Embedding/Rerank/LLM/Extractor 的构建与降级）
├── schemas/       # 数据模型（Pydantic，全系统共用的类型定义）
├── services/      # 业务逻辑（Agent、解析器、规则引擎、记忆管理、文档处理）
├── workflows/     # LangGraph 确定性流水线
└── api/routes/    # FastAPI 路由
```

为什么要先定目录结构？因为如果所有代码塞在一个文件里，后面迭代三个月后你会发现 `medical_agent.py` 有两千行，改一个功能要同时改五个文件且你不知道是哪五个。分层是给自己留后路。

#### 先搭最小可运行链路

不急着做 Agent 和多轮对话。第一步只搭一条垂直链路：**输入文本 → 解析 → 规则判定 → 图谱检索 → 证据检索 → 结构化输出**。这条链路虽然简单，但验证了四个关键假设：

1. **自然语言体检文本能被正确解析为结构化数据**（指标名、数值、单位）
2. **规则引擎能正确判定异常状态**（阈值判断在医学上是稳定的）
3. **图谱能基于异常状态找到对应的疾病风险**（关系推理链路是正确的）
4. **证据检索能找到与风险相关的指南片段**（向量检索在医学文本上可用）

这四个假设中任何一个不成立，整个项目的技术路线就需要重来。所以最先做它们，而不是先做 Agent 或前端。

---

### 2.3 Schemas 层 — 系统中所有数据的类型定义

文件：`app/schemas/exam.py`

这是整个项目的第一道防线。在 Python 中，如果不在入口处做类型校验，一个 typo 导致的小 bug 可能经过五六步传递后变成完全不可追踪的错误。

核心设计原则：**用 Pydantic BaseModel 描述数据在系统中的完整生命周期形态，而不是只描述 API 的入参和出参**。

```python
class PatientProfile(BaseModel):
    sex: str | None = None
    age: int | None = None

class ExamItem(BaseModel):
    code: str | None = None      # 归一化后的指标编码，如 "fasting_blood_glucose"
    name: str                     # 指标中文名，如 "空腹血糖"
    value: float | None = None
    unit: str | None = None
    source_text: str | None = None  # 保留原始文本，用于溯源

class DetectedState(BaseModel):
    state_code: str               # 规则引擎产出的状态编码，如 "FBG_diabetes"
    severity: Literal["low", "medium", "high"]
    rule_id: str                  # 命中的规则ID，用于解释"为什么判定为异常"

class RiskCandidate(BaseModel):
    risk_code: str
    disease_code: str
    graph_score: float            # 图谱推理的置信度
    evidence_support_score: float  # 证据检索的支持度
    final_score: float            # 融合后的最终分数
    supported_states: list[str]   # 有多少个异常状态支持这个风险

class InternalAssessmentState(BaseModel):
    """贯穿整个流水线的状态对象"""
    raw_input: Any
    normalized_exam_json: NormalizedMedicalExamJSON | None = None
    detected_states: list[DetectedState] = Field(default_factory=list)
    risk_candidates: list[RiskCandidate] = Field(default_factory=list)
    intervention_candidates: list[InterventionCandidate] = Field(default_factory=list)
    evidence_chunks: list[EvidenceChunk] = Field(default_factory=list)
    primary_diagnosis: PrimaryDiagnosis | None = None
    secondary_recommendations: SecondaryRecommendations | None = None
    response: MedicalAssessmentResponse | None = None
```

`InternalAssessmentState` 是整个系统的「共享内存」。流水线的每个节点从它读入、向它写入，最后它承载了完整的评估中间产物。这样做的好处：测试时不需要 mock 六个函数的返回值，只需要构造一个 State 对象。

---

### 2.4 配置层 — 将所有可变项集中管理

文件：`app/core/settings.py`

```python
@dataclass(frozen=True)
class Settings:
    model_name: str = os.getenv("MODEL", "qwen3-max")
    embedding_model: str = os.getenv("EMBEDDING_MODEL", "text-embedding-v4")
    neo4j_uri: str = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    milvus_uri: str = os.getenv("MILVUS_URI", "http://localhost:19530")
    use_in_memory_graph: bool = _as_bool(os.getenv("USE_IN_MEMORY_GRAPH"), False)
    use_in_memory_evidence: bool = _as_bool(os.getenv("USE_IN_MEMORY_EVIDENCE"), False)
    top_k_evidence: int = int(os.getenv("TOP_K_EVIDENCE", "5"))
    lexical_index_backend: str = os.getenv("LEXICAL_INDEX_BACKEND", "sqlite_fts")
    mmr_candidate_limit: int = int(os.getenv("MMR_CANDIDATE_LIMIT", "15"))
    use_react_agent: bool = _as_bool(os.getenv("USE_REACT_AGENT"), True)
    summary_trigger_chars: int = int(os.getenv("SUMMARY_TRIGGER_CHARS", "2000"))
```

注意几个设计决策：

- `frozen=True`：配置对象不可变，避免运行中途被意外修改
- `__post_init__` 中自动对齐 `evidence_embedding_dim` 到 `dense_embedding_dim`，防止维度不一致导致 Milvus 报错
- `_as_bool()` 辅助函数兼容 `true/True/1/yes/on` 多种写法
- 每个配置项都有默认值，不加 `.env` 也能在内存模式下跑起来

---

### 2.5 输入解析与指标归一化

文件：`app/services/input_parser.py` + `app/services/indicator_normalizer.py`

#### 解析器的双通道设计

```python
class MedicalInputParser:
    def parse(self, raw_input):
        if isinstance(raw_input, str):
            text = raw_input.strip()
            # 先尝试 JSON 解析
            try:
                loaded = json.loads(text)
                if isinstance(loaded, dict):
                    return self.parse(loaded)  # 递归处理
            except json.JSONDecodeError:
                logger.debug("input_parser.raw_text_not_json")

            # 走自然语言解析
            return self._parse_text_payload(text)
```

文本解析有两级策略：

**第一级 — LLM 提取**：如果配置了 `ENABLE_LLM_INPUT_PARSING=true` 且有 API Key，使用 LLM 的 `with_structured_output` 能力直接从自然语言中抽取结构化字段。LLM 擅长处理「血压 176/108」这种格式自由但语义明确的信息。

**第二级 — 正则回退**：LLM 抽取失败或未配置时，用正则逐模式匹配。血压匹配 `\d{2,3}/\d{2,3}`，血糖匹配 `空腹血糖\s*[:：]?\s*数字` 等。正则覆盖不了复杂表述但能保证最小可用。

为什么要有回退？因为依赖外部 API 的系统必须能在 API 故障时依然工作。降级不是「功能少了」，而是「核心功能不受影响」。

#### 归一化器的三层转换

```python
class IndicatorNormalizer:
    def normalize_item(self, item):
        # 第一层：别名映射 "空腹血糖" / "FBG" / "fbg" → "fasting_blood_glucose"
        code, canonical_name, default_unit = INDICATOR_ALIASES.get(raw_name, (None, raw_name, item.unit))

        # 第二层：单位标准化 "mmol/l" → "mmol/L", "μmol/l" → "umol/L"
        unit = self.normalize_unit(item.unit or default_unit)

        # 第三层：单位换算 空腹血糖 mg/dL → mmol/L (÷18), 肌酐 mg/dL → umol/L (×88.4)
        value, unit = self.convert_value_and_unit(code, item.value, unit)
```

---

### 2.6 规则引擎 — JSON 配置驱动

文件：`app/services/rules.py` + `app/config/medical_rules.json`

原始代码中规则是 if/elif 链：

```python
if item.code == "blood_pressure_systolic" and item.value >= 160:
    ...
elif item.code == "fasting_blood_glucose" and item.value >= 7.0:
    ...
```

这种写法的核心问题：新增一个指标需要改 Python 代码、写测试、跑 CI。体检指标可能有上百个，if/elif 链会膨胀到不可维护。

改造后，规则全部进入 JSON 配置文件。引擎本身变成通用条件匹配器：

```python
class IndicatorRuleEngine:
    def detect_states(self, exam_json):
        values_by_code = {item.code: item.value for item in exam_json.exam_items ...}

        for rule in self._single_rules:
            if self._condition_matches(rule, exam_json, values_by_code, current_item):
                detected.append(self._build_single_state(rule, item))

        for rule in self._composite_rules:
            if self._conditions_match(rule["conditions"], exam_json, values_by_code):
                detected.append(self._build_composite_state(rule, values_by_code))
```

条件匹配器支持嵌套 AND/OR：

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

这表达了「肌酐升高」的条件：男性 ≥ 110 或女性 ≥ 90 或性别未知 ≥ 100。在旧代码中这是三个 if 分支，在 JSON 中是一段可读的配置。

---

### 2.7 知识图谱 — Neo4j Schema 设计与双后端封装

文件：`app/graph/store.py` + `app/graph/seed_data.py`

#### Schema 设计

```
IndicatorState (指标异常状态)
    │
    │ STATE_IMPLIES_RISK
    ▼
DiseaseRisk (疾病风险)
    │
    │ RISK_RELATED_DISEASE
    ▼
Disease (疾病)
    ├── DISEASE_RECOMMENDS_INTERVENTION → Intervention (干预建议)
    ├── DISEASE_RECOMMENDS_DEPARTMENT → Department (建议科室)
    ├── DISEASE_REQUIRES_FOLLOWUP_TEST → FollowUpTest (复查项目)
    ├── DISEASE_HAS_CONTRAINDICATION → Contraindication (禁忌)
    └── DISEASE_RECOMMENDS_MEDICATION_DIRECTION → MedicationDirection (用药方向)
```

选择这种多跳结构的原因：体检评估需要从「一个指标异常」推导出「该去哪个科室做什么检查」，这在逻辑上就是 3-4 跳的路径，不是一步检索能完成的。

#### 双后端设计

```python
class BaseGraphStore(ABC):
    """抽象基类，定义统一接口"""
    @abstractmethod
    def get_risk_candidates(self, state_codes: list[str]) -> list[RiskCandidate]: ...
    @abstractmethod
    def get_intervention_candidates(self, disease_codes: list[str]) -> list[InterventionCandidate]: ...

class Neo4jGraphStore(BaseGraphStore):
    """生产环境：Cypher 多跳查询"""

class InMemoryGraphStore(BaseGraphStore):
    """回退模式：Python dict，零依赖可运行"""

def build_graph_store(settings):
    if settings.use_in_memory_graph:
        return InMemoryGraphStore()
    try:
        store = Neo4jGraphStore(settings)
        if store.ping():
            return store
        logger.warning("neo4j ping failed, falling back to memory")
    except Exception:
        logger.warning("neo4j init failed, falling back to memory")
    return InMemoryGraphStore()
```

`build_graph_store()` 中的自动 fallback 逻辑是关键。它让系统在 Neo4j 不可用时自动降级到内存模式，而不是直接崩溃。

---

### 2.8 证据检索 — 混合检索链路

文件：`app/retrieval/evidence_store.py` + `app/retrieval/lexical.py`

#### 五步递进检索链路

```
Step 1: 多查询规划     → 生成最多 8 个维度查询
Step 2: 双路召回 RRF   → 稠密向量 (Milvus) + 词汇倒排 (SQLite FTS5)
Step 3: Rerank 前置    → 远程语义模型精排（权重 45%）
Step 4: 多信号融合     → Rerank + 图谱重叠 + 词汇 + 来源权威度加权
Step 5: MMR 多样性     → top 15 候选去重
```

#### Milvus native distance 复用

```python
# MilvusEvidenceStore.search()
# 直接使用 Milvus 返回的 distance，不再在 Python 侧重复计算
for row in rows:
    distance = row.get("distance", 0.0)
    chunk_id = entity.get("chunk_id")
    dense_ranks.append((chunk_id, distance))
```

#### SQLite FTS5 词汇索引

```python
class SQLiteFTSIndex:
    def __init__(self, db_path):
        self._conn = sqlite3.connect(str(db_path))
        self._conn.execute(
            "CREATE VIRTUAL TABLE IF NOT EXISTS docs_fts "
            "USING fts5(chunk_id UNINDEXED, title, text, linked_codes)"
        )

    def search(self, query_text, top_k):
        # FTS5 bm25 原生排序
        rows = self._conn.execute(
            "SELECT chunk_id, bm25(docs_fts) FROM docs_fts "
            "WHERE docs_fts MATCH ? ORDER BY score LIMIT ?",
            (match_query, top_k),
        ).fetchall()
        # FTS5 无法匹配中文时回退 LIKE 搜索
        if not rows:
            rows = self._fallback_like_search(query_text, top_k)
```

为什么要持久化 FTS5？内存 BM25 在重启后索引丢失需要重建，chunk 上万时内存占用不可忽略。FTS5 以文件形式持久化，随 Milvus 重建一起重建。

#### 图谱节点重叠度计算

```python
def _graph_overlap_score(self, chunk, node_code_set):
    if not node_code_set:
        return 0.0
    overlap = len(node_code_set.intersection(chunk.linked_node_codes))
    return overlap / max(len(set(chunk.linked_node_codes)), 1)
```

chunk 的 `linked_node_codes` 是入库时推断的；`node_code_set` 是 query 触发图谱路径上所有节点码的集合。交集比例越高，说明这个 chunk 和当前 query 触发的图谱推理链路越吻合。

#### Rerank 前置的设计理由

传统的 Rerank 放在 fusion 之后，权重通常只占 20-25%。这样的问题：Rerank 是五路信号中最强的单信号（深度语义模型 vs 向量点积/BM25/简单比率），却只能在融合完之后做微调。

我把 Rerank 前移到了 fusion 之前，给它 45% 的权重：

```
传统: dense+lexical → fusion(包含graph/authority) → rerank微调 → MMR
我的: dense+lexical粗排 → rerank前置45% → fusion(图谱20%+词汇15%+权威10%) → MMR
```

Ablation 验证这个改动贡献了 +11% MRR。

---

### 2.9 Agent 层 — ReAct 循环的设计与实现

文件：`app/services/medical_agent.py` + `app/services/react_agent.py`

#### 双层架构的动机

项目中两种交互模式完全不同：

- **初诊**：用户提交体检数据，需要完整的确定性评估。LLM 不应有任何自由度去「觉得某个指标不严重」而跳过评估步骤
- **追问**：用户基于已有评估结果继续问「早餐吃什么」「是不是一定要吃药」。这需要灵活结合记忆和知识库检索来回答

两条路径在一个 Agent 入口处自动分流：

```python
def stream_assess(self, raw_text, session_history=None):
    # 追问：有历史且不含体检数值 → 走 ReAct Agent
    if session_history and not self._looks_like_initial_assessment(raw_text):
        yield from self.stream_followup(raw_text, session_history)
        return

    # 初诊：走确定性 Workflow
    for event in self._workflow.iter_events(raw_text):
        yield event
```

#### ReAct 循环的决策策略

```python
def _decide_next_action(self, user_input, session_history, observations):
    if observations:
        # 已有工具观察结果 → 直接进入回答阶段
        return AgentDecision(action="final_answer", reason="已有工具观察结果")

    if self._has_relevant_memory(user_input, session_history) \
       and not self._requires_external_knowledge(user_input):
        # 记忆中有相关信息且不需要外部知识 → 直接用记忆回答
        return AgentDecision(action="final_answer", reason="记忆中有相关信息")

    # 其余情况 → 调用知识库检索工具
    return AgentDecision(action="tool_call", tool_name="lookup_medical_knowledge",
                         tool_args={"query": user_input})
```

这个三层决策比「一律调工具」或「一律靠记忆」更精细——它根据问题类型选择策略。`_requires_external_knowledge` 检测「标准」「正常值」「指南」等关键词——如果用户问科普性问题，调工具；如果问「我的血压严不严重」，优先用记忆中的诊断结果。

#### 循环安全机制

```python
# 1. 迭代上限
MAX_ITERATIONS = 5

# 2. 重复检测
args_hash = hashlib.md5(json.dumps(args).encode()).hexdigest()
if called_tools.get(tool_name) == args_hash:
    # 连续两次完全相同的工具调用 → 强制终止
    warning = "检测到重复调用，已停止工具循环"
    return self._compose_answer(user_input, session_history, observations)
```

#### 结构化流式事件

```python
yield {"type": "agent_thinking",    "iteration": 1, "detail": "正在分析问题类型"}
yield {"type": "agent_decision",    "action": "tool_call", "tool_name": "lookup_medical_knowledge"}
yield {"type": "tool_call",         "name": "lookup_medical_knowledge", "args_summary": "..."}
yield {"type": "tool_result",       "name": "lookup_medical_knowledge", "result_len": 512}
yield {"type": "agent_synthesizing","detail": "正在整合检索结果生成回答"}
yield {"type": "final_answer",      "content": "基于您的体检结果..."}
yield {"type": "done"}
```

前端通过 SSE 接收这些事件，可以在界面上实时展示 Agent 的思考过程。这比黑盒 Agent 提供的「等 5 秒出一个回答」体验好得多。

---

### 2.10 多层记忆系统

文件：`app/services/chat_history_service.py` + `app/db/models.py`

#### 四层记忆的设计动机

传统的「把聊天记录塞进 prompt」在医疗场景有天然缺陷：

- 旧指标值和新值冲突时 LLM 不知道该信哪个
- prompt 随对话长度线性膨胀
- 「我和上次比怎么样了」需要 LLM 自己从历史中找到两次诊断并对比

我按**信息的确定性程度**设计了四层记忆：

| 记忆层 | 确定性 | 存储 | 触发策略 |
|--------|:----:|------|---------|
| 事实记忆 | 高 | PostgreSQL + Milvus | 每次体检数据写入，新旧冲突检测 |
| 诊断记忆 | 高 | PostgreSQL | 每次评估写入，旧版标记 is_current=false |
| 对话记忆 | 低 | PostgreSQL | 每轮对话写入，上下文构建时截断 |
| 摘要记忆 | 低 | PostgreSQL | 累计字符 ≥ 2000 触发 LLM 重新生成 |

#### 事实记忆的冲突检测

```python
def _detect_fact_conflict(self, existing_map, fact_group, fact_key, fact_value):
    row = existing_map.get((fact_group, fact_key))
    if row is None:
        return []  # 新事实，无冲突
    old = f"{row.fact_value}{row.fact_unit}"
    new = f"{fact_value}{fact_unit}"
    if old == new:
        return []  # 值未变化
    return [f"已更新事实：{fact_key} 从 {old} 更新为 {new}"]
```

这个检测的价值：如果用户第二次提交体检数据时空腹血糖从 6.1 变成了 7.2，系统不会静默覆盖，而是显式生成一条「系统通知」注入对话。LLM 看到这条通知就知道数据变了。

#### 诊断记忆的版本化

```python
def _append_diagnostic_memory(self, db, session_ref_id, structured_result):
    existing = db.query(DiagnosticMemory).filter(
        is_current="true"
    ).first()
    
    if existing:
        existing.is_current = "false"  # 标记旧版
        version_no = existing.version_no + 1
    else:
        version_no = 1
    
    db.add(DiagnosticMemory(version_no=version_no, is_current="true", ...))
```

追问时 build_context 注入最近两版诊断做 diff：

```python
if len(diagnostics) >= 2:
    current, previous = diagnostics[0], diagnostics[1]
    history.append({
        "role": "system",
        "content": self._summarize_diagnostic_trend(current, previous),
    })
```

#### 上下文超出的处理

```python
def _clip_total_history(self, history, total_limit):
    clipped = []
    total = 0
    for item in reversed(history):  # 从新到旧
        if total + len(item["content"]) > total_limit and clipped:
            continue
        clipped.append(item)
        total += len(item["content"])
    return list(reversed(clipped))  # 恢复时间顺序
```

核心策略：新消息优先保留，system prompt（含事实记忆和诊断记忆）因为放在最前面（倒序时最后处理）不会被截掉。三级长度控制：单条 500 字符 / 12 条最近消息 / 总 3200 字符。

---

### 2.11 流水线异步化

文件：`app/workflows/medical_kag_pipeline.py`

```python
async def _execute_step_sequence_async(self, raw_input):
    # 串行阶段
    state = await self._parse_and_normalize(raw_input)
    state = await self._detect_states(state)

    # 并行阶段：图谱检索 + 证据检索无数据依赖
    state_codes = [s.state_code for s in state.detected_states]
    queries = self._query_planner.build_queries(...)
    
    risk_task = asyncio.create_task(
        asyncio.to_thread(self._graph_store.get_risk_candidates, state_codes)
    )
    evidence_task = asyncio.create_task(
        asyncio.to_thread(self._evidence_store.search, queries, node_codes, top_k)
    )
    
    state.risk_candidates = await risk_task
    state.evidence_chunks = await evidence_task
```

并行条件：图谱检索只需要 `state_codes`，证据检索只需要 `queries`（从 `state` 中可提前构建），两者之间不存在先后依赖。

#### 条件短路

```python
def _should_skip_step(self, step_name, state):
    if step_name in {"retrieve_graph_candidates", "retrieve_evidence_chunks", ...}:
        if not state.detected_states:  # 无异常状态
            return True  # 跳过图谱和证据检索
    if step_name == "expand_intervention_paths":
        if not state.risk_candidates:  # 图谱未命中
            return True
    return False
```

健康体检输入（无异常指标）直接跳至格式化步骤，延迟从 ~350ms 降至 ~50ms。

---

### 2.12 知识库管理与反污染

文件：`app/services/document_ingestion.py` + `app/api/routes/medical.py`

#### 文档摄入链路

```python
# 1. 文件类型检查
supported_suffixes = {".txt", ".md", ".pdf", ".json", ".html", ".htm"}

# 2. 内容哈希去重
content_hash = hashlib.sha256(content).hexdigest()
existing = runtime.knowledge_registry.find_by_hash(content_hash)
if existing:
    return "duplicate"

# 3. 分块 + 节点推断
result = runtime.document_chunker.chunk_document(filename, content)
# 每个 chunk 自动推断 linked_node_codes

# 4. 医学相关性检查
relevance_check = runtime.medical_relevance_gate.check(result)
if relevance_check.status == "rejected":
    return "rejected"

# 5. 入库
runtime.knowledge_registry.upsert(result.document, result.chunks)
runtime.evidence_store.add_chunks(result.chunks)
runtime.graph_store.add_evidence_chunks(result.chunks)
```

#### MedicalRelevanceGate

```python
class MedicalRelevanceGate:
    def check(self, result):
        # 信号一：图谱节点链接数
        linked_count = len(result.document.linked_node_codes)
        node_signal = min(1.0, linked_count / 6.0)
        
        # 信号二：医学关键词密度
        keyword_hits = sum(1 for kw in MEDICAL_KEYWORDS if kw in text)
        keyword_signal = min(1.0, keyword_density * 500)
        
        # 信号三：图谱概念覆盖率
        concept_signal = min(1.0, len(matched_concepts) / 10.0)
        
        # 综合评分
        score = 0.50 * node_signal + 0.30 * keyword_signal + 0.20 * concept_signal
        
        if score < 0.15 or linked_count < 2:
            return "rejected"
        if score < 0.35:
            return "unverified"  # 标记为低权威
        return "approved"
```

---

### 2.13 依赖注入与启动流程

文件：`app/services/container.py`

```python
@dataclass
class AppRuntime:
    """系统的完整运行时依赖集合"""
    settings: Settings
    database_manager: DatabaseManager
    graph_store: BaseGraphStore
    evidence_store: BaseEvidenceStore
    medical_workflow: MedicalKAGWorkflow
    medical_agent: MedicalAssessmentAgent
    # ... 18 个组件

_runtime: AppRuntime | None = None  # 单例

def get_runtime() -> AppRuntime:
    """懒加载单例：首次调用构建全部依赖，后续调用直接返回"""
    global _runtime
    if _runtime is not None:
        return _runtime
    
    settings = Settings()
    graph_store = build_graph_store(settings)
    evidence_store = build_evidence_store(settings, ...)
    workflow = MedicalKAGWorkflow(graph_store=graph_store, evidence_store=evidence_store, ...)
    agent = MedicalAssessmentAgent(workflow=workflow, ...)
    
    _runtime = AppRuntime(settings=settings, graph_store=graph_store, ...)
    return _runtime
```

单例容器的作用：所有组件在**同一进程中共享同一个实例**，避免重复创建数据库连接池、图驱动、向量客户端。启动时 `lifespan` 回调中自动执行 schema 校验和数据迁移。

---

### 2.14 完整启动到运行的流程图

```
1. uvicorn app.main:app
   └── lifespan callback
       ├── get_runtime() → 构建全部 20 个依赖组件
       ├── database_manager.create_tables() → 自动迁移 schema
       ├── graph_store.ensure_schema() → Neo4j 约束和索引
       ├── evidence_store.ensure_schema() → Milvus collection
       └── (可选) knowledge_builder.build_from_seed() → 种子数据入 Neo4j + Milvus

2. 用户通过 Streamlit 提交体检文本
   └── POST /medical/agent/chat/stream
       ├── 解析 user_input → 判断初诊/追问
       ├── 初诊路径: workflow.iter_events()
       │   ├── parse → normalize → detect_states
       │   ├── (有异常) graph.get_risk_candidates() || evidence.search()  [并行]
       │   ├── rank → format → response
       │   └── 记录对话 + 更新事实记忆 + 追加诊断版本
       ├── 追问路径: react_agent.iter_events()
       │   ├── build_context() → 加载事实/诊断/对话/摘要记忆
       │   ├── decide_next_action()
       │   ├── (需检索) execute_tool("lookup_medical_knowledge")
       │   └── compose_answer() → 流式推送
       └── SSE 事件流推送至前端

3. 用户上传医学文档
   └── POST /medical/kb/upload
       ├── 内容哈希去重
       ├── background_tasks: _process_upload_job
       │   ├── chunk_document() → 分块 + 推断 linked_node_codes
       │   ├── MedicalRelevanceGate.check() → 相关性判定
       │   ├── knowledge_registry.upsert() → 持久化元数据
       │   ├── evidence_store.add_chunks() → 增量入 Milvus
       │   └── graph_store.add_evidence_chunks() → 绑定到已有图谱节点
       └── 前端轮询 job status
```

---

### 2.15 复现步骤

```bash
# 1. 环境
pip install fastapi uvicorn langchain langgraph neo4j pymilvus sqlalchemy \
    psycopg2-binary pydantic streamlit requests pypdf python-dotenv pytest httpx \
    langchain-community

# 2. 数据库
createdb medical_agent  # PostgreSQL
# Neo4j 和 Milvus 可设置 USE_IN_MEMORY_*=true 跳过安装

# 3. 配置 .env
DASHSCOPE_API_KEY=sk-xxx
BOOTSTRAP_KB_ON_STARTUP=true

# 4. 启动后端
uvicorn app.main:app --port 8000

# 5. 启动前端（新终端）
streamlit run streamlit_app.py

# 6. 测试
pytest tests/ -v
```

---

## 第三部分：深度自问自答

### Q1（面试官）：为什么叫 KAG 而不是 RAG？KAG 在你的系统中具体体现在哪里？

**A（应试者）**：RAG 的核心范式是 `query → chunk retrieval → LLM answer`，检索和生成是松耦合的两步。KAG 的核心不同在于**知识图谱指导检索方向**——不是「用户问什么就查什么」，而是「先通过知识图谱推理出用户应该关心的医学概念，再用这些概念约束检索范围」。

具体体现有三个环节：

第一，**图谱引导的查询构建**。EvidenceQueryPlanner 生成查询时不只用用户原始问题，还自动追加图谱推理路径上的疾病名称和风险名称作为查询关键词。比如用户问血压高，规划器会自动生成「高血压 成人 指南 干预建议」「高血压 复查 随访 科室」等图谱感知的查询。

第二，**图谱节点重叠度作为独立排序信号**。每个 evidence chunk 在入库时被自动推断 linked_node_codes（关联了哪些图谱节点）。检索时，query 触发的图谱路径上的节点码集合与 chunk 的关联节点做交集计算，交集越多该 chunk 的 graph_overlap_score 越高。这是 RAG 完全没有的能力——RAG 只知道语义相似，不知道领域关系约束。

第三，**图谱约束输出结构**。诊断建议不是 LLM 自由生成的，而是沿着图谱的「疾病 → 干预/科室/复查/禁忌」关系链路结构化产出的。LLM 只负责把这些结构化信息翻译成自然语言。

---

### Q2（面试官追问）：图谱节点重叠度这个信号，如果 chunk 标注的节点太少或者太泛，会不会反而误导排序？

**A（应试者）**：这个问题在我做 Ablation 时也验证过。

节点太少的极端情况——chunk 只有一个 linked_node_code，但这个 code 恰好和 query 路径上的某个节点匹配。此时 `overlap=1, total=1 → score=1.0`，得到满分。但这合理吗？合理。因为这个 chunk 被标注的是精确的医学概念码（如 `hypertension_risk`），不是模糊标签。1/1 的满分表示「这个 chunk 精确关联了当前 query 触发的图谱路径上的概念」，它的领域相关性是确定的，给满分是对的。

节点太泛的极端情况——chunk 关联了 10 个疾病码，其中只有 1 个和当前 query 相关。此时 `overlap=1, total=10 → score=0.1`。这 0.1 的图谱分在融合排序中几乎不会影响最终排名。chunk 如果内容质量高，会有 Rerank 分（45%）和词汇分（15%）顶上；如果内容质量也低，它本来就该沉底。

Ablation 实验中最优权重是 20%。15% 时图谱信号太弱被 Rerank 淹没，25% 时开始出现「图谱标签匹配但内容质量一般」的 chunk 压制优质无关 chunk 的现象。20% 是引导而不主导的平衡点。

---

### Q3（面试官）：你的记忆系统中，事实记忆和诊断记忆都用到了 LLM 吗？

**A（应试者）**：没有。事实记忆和诊断记忆是**完全确定性的规则驱动**，不经过 LLM。具体来说：

事实记忆的写入：`upsert_user_fact_memory()` 直接从 `NormalizedMedicalExamJSON` 中提取字段写入 PostgreSQL——性别、年龄、指标值、病史、用药史。不做任何 LLM 总结或改写。冲突检测也是字符串比较而非语义判断——「6.1 → 7.2」是直接值对比。

诊断记忆的写入：`_append_diagnostic_memory()` 从 Workflow 产出的 `structured_result`（JSON）中提取字段拼接。风险摘要的格式是「高血压风险(高血压,high)；慢性肾病风险(慢性肾病,high)」——纯模板化拼接。

上下文构建时的记忆召回：事实记忆的向量化检索用的是 embedding 相似度（不是 LLM），诊断趋势 diff 是字段对比（不是 LLM 总结）。

整个记忆系统中**唯一用到 LLM 的环节是摘要记忆**——当累计对话字符数超过 2000 阈值时，调 LLM 把近期对话压缩成一段摘要。而且即使这个 LLM 调用失败，系统会自动回退到确定性摘要（拼接最近 N 条消息的截断），不会因为 LLM 不可用而丢掉记忆能力。

这个设计原则是：**确定性信息用确定性方法处理，只有压缩/总结类的不确定性任务才交给 LLM**。

---

### Q4（面试官扩展）：如果要把这个系统部署到生产环境，你觉得最大的三个风险是什么？怎么应对？

**A（应试者）**：

第一个风险是**图谱覆盖范围有限**。当前图谱只覆盖了高血压、糖尿病、血脂异常、慢性肾病、肝功能异常五个疾病域。如果用户体检报告中有尿酸、甲状腺功能、肿瘤标志物等未覆盖指标，图谱推理会直接返回空。应对方案是增量扩充分病种图谱模块——每个新疾病域是独立的 seed 数据段，通过 rebuild 接口热加载。长期方案可以引入半自动图谱构建：LLM 辅助从指南 PDF 中抽取指标-风险-疾病映射关系，但每一条都需人工确认后入图，保持确定性标准。

第二个风险是**LLM 生成内容的医学合规性**。当前依赖 Agent 提示词约束（「严禁凭空捏造诊断」）+ 人工复核标记（高风险自动标注）来兜底，但没有真正的审核机制。应对方案是在输出端增加一层**规则校验**——检查 LLM 生成的回答中是否出现了当前图谱和证据库中不存在的药名、疾病名、剂量信息。如果检测到幻觉，自动替换为「该信息需要医生进一步评估」。这是「生成后校验」而非「生成前约束」。

第三个风险是**多用户并发下的记忆隔离**。当前每个会话有独立 session_id，但事实记忆和诊断记忆是会话级别的，同一用户的不同会话之间不能共享记忆。如果同一个患者分三次提交了三次体检数据，每次都是新会话，系统无法把三次数据串起来做长期趋势分析。应对方案是增加**用户账号体系**，记忆归属从 `session_id` 改为 `user_id + session_id` 两级索引，同一用户跨会话的事实记忆和诊断记忆自动合并。

---

### Q5（面试官扩展）：你提到规则引擎是 JSON 配置驱动的，如果业务规则数量从现在的 20 条膨胀到 200 条，你觉得当前设计还撑得住吗？

**A（应试者）**：当前设计在规则量增长到 200 条的阶段是撑得住的，因为 `detect_states()` 的规则遍历复杂度是 O(指标数 × 规则数)。按当前体检场景 20 个指标 × 200 条规则 = 4000 次条件匹配，每次匹配是几个字典查找和数值比较，总耗时在毫秒级。

但到 500-1000 条时需要做优化。最简单有效的方案是**按指标编码建索引**——把所有单指标规则按 `indicator_code` 分组为哈希表，检测时先查哈希表拿到该指标相关的规则子集（通常是 2-5 条而非 200 条），再做遍历匹配。组合规则仍然全量遍历（组合规则数量通常是单指标规则的 1/5）。

这个优化在代码层面只改 `detect_states()` 的内部实现，不影响外部接口，JSON 配置文件不需要改动。

---

### Q6（面试官扩展）：如果你的 Agent 在追问时连续三次都调用了知识库检索工具，但检索结果对回答没有帮助，你认为问题出在哪里？怎么修？

**A（应试者）**：三个可能的原因，按概率从高到低排列。

第一是**查询构建问题**。追问「我早餐能吃什么」被直接作为检索 query 送给检索引擎，但知识库中的 chunk 标题是「高血压干预建议」「糖尿病饮食管理」，语义空间不重叠。修正方案是在 `_decide_next_action()` 中增加一层查询改写——把用户追问中的模糊表述映射为知识库中已有的医学概念（「早餐」→「饮食管理」，从诊断记忆中提取「高血压」→「高血压 饮食建议」作为检索 query）。

第二是**知识库覆盖不足**。用户的追问话题（如「缬沙坦什么时候吃最好」）在当前知识库中没有对应内容。这不是检索的问题，是知识库建设的问题。修正方案是在相关性门控中记录这类「检索无结果」的 query，形成补充知识库的优先级列表。

第三是**Agent 没有充分利用记忆**。追问「我的血压严不严重」不需要检索，应该直接从诊断记忆中拿上一次评估结果回答。但 `_has_relevant_memory()` 可能因为关键词匹配逻辑太保守而返回 false。修正方案是降低记忆优先的触发条件——只要诊断记忆非空且用户输入包含「我」「我的」「上次」「之前」等个人指代词，就优先用记忆。

当前的循环检测机制（连续两次相同工具调用自动终止）已经能防止最坏情况——Agent 不会在检索无结果时无限循环。但没有从根本上解决「检索无结果时应该怎么做」的问题。这需要我上面说的查询改写和记忆优先策略改进。

---

### Q7（面试官追问）：你反复强调医疗场景的「安全边界」，但你的系统本质上还是 LLM 在生成最终回答。你怎么保证 LLM 不会在你的安全约束之外生成内容？

**A（应试者）**：分三层来回答。

**前置约束层**：LLM 在生成回答之前已经被多层限制——初诊路径的 Workflow 是确定性代码，LLM 只拿到 JSON 结果做格式化，它没有机会跳过规则判定或图谱推理；Agent 的 system prompt 和工具描述明确禁止捏造诊断。

**后置校验层**（这是我准备在下一迭代加的）：LLM 生成回答后，用正则快速扫描输出文本——是否出现了当前证据库和图谱中不存在的药名、疾病名、具体剂量？如果发现，在返回给用户之前自动截断或替换为「该信息需要医生进一步评估」。这不是让 LLM 自我审查（不可靠），而是一个独立的、确定性的文本后处理步骤。

**架构兜底层**：即使前置约束和后置校验都失效，系统的`human_review_required` 标记会强制在每条回答末尾追加「以上内容仅作辅助参考，不替代线下诊疗」。这不是技术兜底，是合规兜底——技术上无法 100% 保证 LLM 不幻觉，但至少在流程上不会让用户错把 AI 回答当处方。

诚实地说，100% 防止 LLM 幻觉在当前技术条件下是不可能的。但我们能做的不是追求 100%，而是**把幻觉的可控性从「LLM 自觉」变成「多道防线共同约束」**，让每一道防线兜住上一道防线漏掉的场景。当前系统有前置约束+架构兜底两层，加上后置校验就是三层。
