# 知识入库、图谱创建、记忆保存 — 实现细节全解

> 目标：不看代码即可完全掌握以下三条链路的实现细节、数据流转和被主业务消费的方式。
> - 文档上传 → 分块 → 注入图谱 → 向量入库
> - 种子数据 → Neo4j 图谱创建 → 六节点五关系推理网
> - 会话事实/诊断/对话/摘要四层记忆的读写生命周期

---

## 第一部分：知识入库 — 用户文档从上传到可被检索的完整链路

### 1.1 链路全景

```text
用户上传文件 (txt/md/pdf/json/html)
        │
        ▼
[同步阶段] POST /medical/kb/upload
  ├── 文件类型白名单检查 → 不支持的格式直接 400
  ├── SHA-256 内容哈希去重 → 重复文件直接标记 duplicate
  ├── 生成任务记录 → 返回 job_id 给前端（立即响应，不阻塞）
  └── 将文件内容提交给 BackgroundTasks

        │（异步）
        ▼
[异步阶段] _process_upload_job(job_id, filename, content)
  ├── Step 1: DocumentChunker.chunk_document() → 抽取文本 + 分块 + 节点推断
  ├── Step 2: KnowledgeDocumentRegistry.upsert() → 持久化文档元数据和 chunk
  ├── Step 3: 判断后端状态 → 全量 build 还是增量 add
  └── Step 4: graph_store + evidence_store 入库
```

### 1.2 Step 1: DocumentChunker — 文本抽取

```python
def _extract_text(self, filename, content, suffix):
    if suffix in {".txt", ".md"}:
        return content.decode("utf-8", errors="ignore")

    if suffix in {".html", ".htm"}:
        # 去除 script/style 标签 → 标签转义 → 合并空白
        return self._html_to_text(content.decode("utf-8", errors="ignore"))

    if suffix == ".json":
        return json.dumps(data, ensure_ascii=False, indent=2)

    if suffix == ".pdf":
        # 写入临时文件 → pypdf 逐页提取 → 删除临时文件
        reader = PdfReader(str(tmp_path))
        return "\n".join(page.extract_text() or "" for page in reader.pages)
```

**设计要点**：PDF 为什么不直接读 bytes 而要先写临时文件？因为 `PdfReader` 的构造函数接受的是文件路径，不支持 bytes 流。写入临时文件 → 读取 → 删除是一个折中方案，开销很小（写入和读取都在本地磁盘，几 KB 到几 MB）。

HTML 处理之所以要 `_html_to_text()` 而不是直接 `BeautifulSoup`，是为了保持零额外依赖。项目只依赖 `pypdf` 一个文档处理库，HTML 用正则做标签清洗已经足够——这个场景下不需要精确的 DOM 解析，只需要提取可见文本。

### 1.3 Step 1 (续): DocumentChunker — 分块

```python
def _split_to_chunks(self, text):
    normalized = " ".join(text.split())  # 所有空白符 → 单空格
    if not normalized:
        return []  # 空文本 → 无 chunk

    chunks = []
    start = 0
    while start < len(normalized):
        end = min(len(normalized), start + 500)  # chunk_size=500
        chunks.append(normalized[start:end])
        if end >= len(normalized):
            break
        start = max(0, end - 80)  # overlap=80，滑动窗口
    return chunks
```

**chunk_size=500, overlap=80 的选择逻辑**：

- 500 字符在中文医学文本中大约对应 4-5 句话，足够承载一个完整的医学概念（如"高血压患者应遵循 DASH 饮食模式，控制钠摄入，每日食盐不超过 5g"），不会因为太短而断章
- 80 字符 overlap 保证跨越 chunk 边界的句子不会丢失上下文连接——一个被切成两半的段落，前一个 chunk 和后一个 chunk 之间有 80 字的交集，检索时两个 chunk 都能被独立命中而不会丢失连贯性
- 这个组合不是理论推导出来的，而是用实际指南文档跑了几组分块参数后选择的效果最好的。更小的 chunk（300）导致语义碎片化，更大的 chunk（1000）导致每个 chunk 包含多个独立概念，向量表示被稀释

### 1.4 Step 1 (续): DocumentChunker — 图谱节点推断

这是最关键的步骤。每个 chunk 入库时不只是存文本，还要打上**图谱标签**。

```python
def _infer_linked_node_codes(self, text):
    lowered = text.lower()
    linked = []

    # 第一层：指标别名匹配
    for alias in INDICATOR_ALIASES:
        if alias.lower() in lowered:
            linked.extend(self._map_alias_to_related_codes(alias))

    # 第二层：状态码匹配
    for state_code, items in STATE_TO_RISK.items():
        if state_code.lower() in lowered:
            linked.append(state_code)
            for item in items:
                linked.extend([str(item["risk_code"]), str(item["disease_code"])])

    # 第三层：疾病码和干预词匹配
    for disease_code, payload in DISEASE_TO_INTERVENTIONS.items():
        if disease_code.lower() in lowered:
            linked.append(disease_code)
        for text_list in payload.values():
            if any(fragment in text for fragment in text_list):
                linked.append(disease_code)

    return deduped(linked)
```

**三层匹配的逻辑**：

| 层次 | 匹配目标 | 示例 |
|------|---------|------|
| 别名匹配 | chunk 中出现 "收缩压" → 映射到 `["hypertension_risk", "hypertension"]` | "血压" 这个通用词→链接到高血压域 |
| 状态码匹配 | chunk 中出现 "SBP_high_stage2" → 追加该状态码及其关联的 risk_code 和 disease_code | 精确命中图谱中的节点名 |
| 疾病干预匹配 | chunk 中出现 "ACEI" "限盐" "心内科" → 追加对应的 disease_code | 内容中出现诊断/治疗相关术语 |

**为什么三层分别匹配**：别名匹配解决了「文中出现指标但不提疾病名」的问题（"收缩压"→链接到高血压）。状态码匹配解决了「文中直接引用图谱节点」的问题（这是知识库重建时 seed chunk 的常见情况）。疾病干预匹配解决了「文中出现治疗建议但不提疾病编码」的问题（"心内科"→链接到高血压和 CKD）。

**推断不出链接怎么办**：如果全文三层都匹配不到，"linked_node_codes" 就是空列表。这不影响 chunk 入库，但在后续检索排序中 graph_overlap_score 为 0——意味着这个 chunk 只能靠语义相关性得分，没有图谱信号加持。这是六层防污染中的第五层：无关文档即使入库，在排序中也会自然沉底。

### 1.5 Step 2: KnowledgeDocumentRegistry — 元数据持久化

```python
class KnowledgeDocumentRegistry:
    def __init__(self, registry_path, upload_root):
        # 从 JSON 文件加载已有注册表
        self._load()

    def _load(self):
        payload = json.loads(self._registry_path.read_text())
        for item in payload.get("documents", []):
            self._documents[item["doc_id"]] = KnowledgeDocument(**item)

    def upsert(self, document, chunks, raw_content=None):
        self._documents[document.doc_id] = document       # 文档元数据
        self._chunks_by_doc_id[document.doc_id] = chunks  # chunk 列表
        if raw_content is not None:
            target = self._upload_root / f"{document.doc_id}.{document.file_type}"
            target.write_bytes(raw_content)               # 原文保存到磁盘
        self._persist()  # 写回 JSON 文件

    def find_by_hash(self, content_hash):
        """SHA-256 去重：返回已存在的文档或 None"""
        for document in self._documents.values():
            if document.content_hash == content_hash:
                return document
```

**为什么用 JSON 文件而不是数据库**：知识库注册表的数据量很小（通常几十到几百条文档元数据），JSON 文件读写的性能开销可以忽略。而用 JSON 文件的好处是**零额外依赖**——不需要 MongoDB/Elasticsearch/额外的 PostgreSQL 表，启动时直接 `json.loads` 加载到内存，写入时 `json.dumps` 持久化。chunk 的详细数据存入 Milvus 和 Neo4j 后，注册表只需要保留文档级别的元数据（doc_id, filename, file_type, content_hash, chunk_count, linked_node_codes）。

**content_hash 去重的位置**：去重检查发生在 API 层（同步阶段），在创建后台任务**之前**。这意味着重复文件根本不会进入异步处理队列，节省了文档解析和分块的开销。

### 1.6 Step 3-4: 图谱和证据的后端入库

`_process_upload_job` 中的分支判断：

```python
# 分支一：后端数据为空（首次启动或 rebuild 后未上传过文档）
if not runtime.graph_store.data_ready() or not runtime.evidence_store.data_ready():
    runtime.knowledge_builder.build_from_seed()
    # → 全量重建：seed 数据 + 已上传文档合并后一起写入图谱和证据库

# 分支二：后端已有数据（常规增量上传）
else:
    runtime.evidence_store.add_chunks(result.chunks)     # Milvus 增量插入
    runtime.graph_store.add_evidence_chunks(result.chunks) # Neo4j 增量绑定
```

**Neo4j 增量绑定的 Cypher**：

```cypher
MERGE (e:EvidenceChunk {chunk_id: $chunk_id})
SET e.doc_id = $doc_id, e.title = $title, e.text = $text, e.source_type = $source_type
WITH e, $linked_node_codes AS codes
UNWIND codes AS code
OPTIONAL MATCH (r:DiseaseRisk {risk_code: code})
OPTIONAL MATCH (d:Disease {disease_code: code})
FOREACH (_ IN CASE WHEN r IS NOT NULL THEN [1] ELSE [] END |
    MERGE (r)-[:NODE_LINKED_CHUNK]->(e))
FOREACH (_ IN CASE WHEN d IS NOT NULL THEN [1] ELSE [] END |
    MERGE (d)-[:NODE_LINKED_CHUNK]->(e))
```

关键点：`MERGE` 保证幂等——同一个 chunk_id 重复执行不会创建重复节点。`OPTIONAL MATCH` 保证即使 linked_node_codes 中的某个 code 在当前图谱中不存在对应的 DiseaseRisk 或 Disease 节点，也不会报错中断——只是这一条绑定被跳过。

**全量重建路径（build_from_seed）**的不同之处在于它会先执行 `MATCH (n) DETACH DELETE n` 清空全图，然后重新 MERGE 所有节点和边。这通常在首次启动或调用 `/medical/kb/rebuild` 时触发。

---

## 第二部分：图谱创建 — Neo4j 知识网络从种子数据到推理引擎

### 2.1 种子数据的结构

文件：`app/graph/seed_data.py`

三张核心映射表，每一张解决图谱构建中的一个层次：

```text
INDICATOR_ALIASES:
  输入指标的各种中文名/英文缩写/别名 → 标准化 code + 中文名 + 默认单位

STATE_TO_RISK:
  异常状态码(如 SBP_high_stage2) → 关联的风险(hypertension_risk) + 疾病(hypertension)
  包含 graph_score 表示这条映射的置信度

DISEASE_TO_INTERVENTIONS:
  疾病码(如 hypertension) → 干预/用药方向/禁忌/复查/科室 五个列表
```

**数据来源**：

| 表 | 来源 | 规模 |
|---|------|:--:|
| INDICATOR_ALIASES | 临床检验报告常见写法的手工整理 | 16 个指标 × 3~5 个别名 |
| STATE_TO_RISK | 高血压/CKD/糖尿病/血脂异常/肝功的公开指南诊断标准 | 20 条状态映射 |
| DISEASE_TO_INTERVENTIONS | 上述五个疾病域的指南推荐方向 | 6 个疾病 × 5 个建议维度 |

### 2.2 Neo4j Schema 创建

应用启动时 `lifespan` 回调执行 `graph_store.ensure_schema()`：

```python
def ensure_schema(self):
    statements = [
        "CREATE CONSTRAINT indicator_state_code IF NOT EXISTS "
        "FOR (n:IndicatorState) REQUIRE n.state_code IS UNIQUE",
        "CREATE CONSTRAINT disease_risk_code IF NOT EXISTS "
        "FOR (n:DiseaseRisk) REQUIRE n.risk_code IS UNIQUE",
        "CREATE CONSTRAINT disease_code IF NOT EXISTS "
        "FOR (n:Disease) REQUIRE n.disease_code IS UNIQUE",
        "CREATE CONSTRAINT evidence_chunk_id IF NOT EXISTS "
        "FOR (n:EvidenceChunk) REQUIRE n.chunk_id IS UNIQUE",
        "CREATE INDEX disease_name_idx IF NOT EXISTS FOR (n:Disease) ON (n.name)",
        "CREATE INDEX risk_name_idx IF NOT EXISTS FOR (n:DiseaseRisk) ON (n.name)",
    ]
    for statement in statements:
        self._run(statement)
```

四个唯一性约束（CONSTRAINT）+ 两个索引。`IF NOT EXISTS` 保幂等：重复执行不会报错。六种节点类型中，`IndicatorState`/`DiseaseRisk`/`Disease`/`EvidenceChunk` 有唯一性约束，`Intervention`/`Department`/`FollowUpTest`/`Contraindication`/`MedicationDirection` 没有——因为这些节点是通过 `MERGE` 按 name 去重的，不需要额外约束。

### 2.3 全量图谱构建：rebuild_from_seed()

调用入口：`POST /medical/kb/rebuild` 或启动时 `BOOTSTRAP_KB_ON_STARTUP=true`。

```text
build_from_seed()
  → _merged_evidence_chunks()  ← 种子 chunk + 已上传文档 chunk 合并去重
  → graph_store.rebuild_from_seed(STATE_TO_RISK, DISEASE_TO_INTERVENTIONS, chunks)
  → evidence_store.rebuild_index(chunks)
```

**第一阶段：清图重建节点和关系**：

```cypher
MATCH (n) DETACH DELETE n
```

**第二阶段：逐条 MERGE 状态→风险→疾病**：

```cypher
MERGE (s:IndicatorState {state_code: $state_code})
SET s.label = $state_label, s.rule_id = $rule_id, s.severity = $severity
MERGE (r:DiseaseRisk {risk_code: $risk_code})
SET r.name = $risk_name, r.risk_level = $risk_level
MERGE (d:Disease {disease_code: $disease_code})
SET d.name = $disease_name
MERGE (s)-[:STATE_IMPLIES_RISK]->(r)
MERGE (r)-[:RISK_RELATED_DISEASE]->(d)
```

逐条的含义：遍历 `STATE_TO_RISK` 中的 20 条映射，每条生成一个独立的参数化查询。比如 `SBP_high_stage2` 和 `DBP_high_stage2` 都指向 `hypertension_risk`→`hypertension`，两条查询会 MERGE 到同一个 `hypertension_risk` 和 `hypertension` 节点上（幂等）。

**第三阶段：逐疾病 MERGE 干预/科室/复查/禁忌/用药**：

```cypher
MERGE (d:Disease {disease_code: $disease_code})
FOREACH (name IN $interventions |
    MERGE (i:Intervention {name: name})
    MERGE (d)-[:DISEASE_RECOMMENDS_INTERVENTION]->(i))
FOREACH (name IN $medication_directions |
    MERGE (m:MedicationDirection {name: name})
    MERGE (d)-[:DISEASE_RECOMMENDS_MEDICATION_DIRECTION]->(m))
-- ... 同理 departments / follow_up_tests / contraindications
```

`FOREACH ... MERGE` 组合：对列表中的每个元素执行 MERGE（不存在则创建）。这比 Python 侧循环调用 Cypher 效率高，且保证幂等。

**第四阶段：证据 chunk 绑定**：

```cypher
MERGE (e:EvidenceChunk {chunk_id: $chunk_id})
SET e.doc_id = $doc_id, e.title = $title, e.text = $text, e.source_type = $source_type
WITH e, $linked_node_codes AS codes
UNWIND codes AS code
OPTIONAL MATCH (r:DiseaseRisk {risk_code: code})
OPTIONAL MATCH (d:Disease {disease_code: code})
FOREACH (_ IN CASE WHEN r IS NOT NULL THEN [1] ELSE [] END |
    MERGE (r)-[:NODE_LINKED_CHUNK]->(e))
FOREACH (_ IN CASE WHEN d IS NOT NULL THEN [1] ELSE [] END |
    MERGE (d)-[:NODE_LINKED_CHUNK]->(e))
```

**FOREACH + CASE WHEN 的妙处**：Neo4j 中没有原生的「IF 匹配到就创建关系」语法。FOREACH + CASE WHEN 是实现条件关系创建的标准惯用法——`CASE WHEN r IS NOT NULL THEN [1] ELSE [] END` 在匹配到 r 时返回单元素列表让 FOREACH 执行一次 MERGE，匹配不到时返回空列表让 FOREACH 跳过。

### 2.4 图谱如何被主业务消费

**初诊路径**：Workflow 节点 4+5 调用 `graph_store.get_risk_candidates(state_codes)` 和 `graph_store.get_intervention_candidates(disease_codes)`。

**追问路径**：不直接调用图谱查询。但问答中引用的「诊断记忆」包含的 risk_summary 和 department_summary 来自 Workflow 的产出（图谱推理的结果被打包进结构化诊断）。追问检索时 RetrievalAgent 把诊断记忆中的疾病名用于查询扩展。

**检索排序路径**：证据检索中的 graph_overlap_score 依赖每个 chunk 的 linked_node_codes 和 query 触发的图谱节点码做交集。节点的完整性直接影响这个信号的质量。

---

## 第三部分：记忆存储 — 四层记忆的读写生命周期与主业务集成

### 3.1 四张数据库表

```sql
chat_sessions:        会话级信息（id, session_id, title, summary, pending_chars）
conversation_memories: 对话消息（role, content, content_summary）→ FK 到 chat_sessions
user_fact_memories:   用户事实（fact_group, fact_key, fact_value, fact_unit）→ FK 到 chat_sessions
diagnostic_memories:  诊断快照（version_no, is_current, health_status, risk_summary, ...）→ FK 到 chat_sessions
```

所有记忆都通过 `session_ref_id` 外键归属到一个会话。删除会话时 `cascade="all, delete-orphan"` 自动清理所有关联记忆。

### 3.2 写入生命周期

**第一步：会话创建**（API 层或新建对话时）

```python
session_id = f"session_{uuid.uuid4().hex[:12]}"  # 唯一标识
session = ChatSession(session_id=session_id, title="新会话",
                      summary_text="", conversation_summary="",
                      summary_pending_chars=0)
db.add(session)
```

**第二步：用户消息写入**（每次用户发送消息时）

```python
def record_user_message(self, session_id, content):
    session = db.query(ChatSession).filter_by(session_id=session_id).first()
    # 首次对话时用消息内容的前 36 字自动生成会话标题
    if session.title == "新会话":
        session.title = truncate(content, 36)

    db.add(ConversationMemory(
        session_ref_id=session.id,
        role="user",
        content=content,                        # 原始消息全文
        content_summary=truncate(content, 500), # 截断到 500 字用于上下文构建
    ))
    session.summary_pending_chars += len(content)  # 累计未触发摘要的字符数
```

**第三步：事实记忆覆盖写入**（初诊评估完成后，仅初诊路径触发）

```python
def upsert_user_fact_memory(self, session_id, normalized_exam_json):
    # 1. 读取当前 session 的旧事实
    existing_rows = db.query(UserFactMemory).filter_by(session_ref_id=session.id).all()
    existing_map = {(row.fact_group, row.fact_key): row for row in existing_rows}

    # 2. 清空旧事实
    db.execute(delete(UserFactMemory).where(UserFactMemory.session_ref_id == session.id))

    # 3. 逐条写入新事实 + 冲突检测
    for item in normalized_exam_json.exam_items:
        fact_key = item.code or item.name       # 指标编码，如 "blood_pressure_systolic"
        fact_value = str(item.value)            # 数值，如 "176"
        fact_unit = item.unit or ""             # 单位，如 "mmHg"

        # 冲突检测：旧值是否存在且不同
        old_row = existing_map.get(("exam_item", fact_key))
        if old_row and old_row.fact_value + old_row.fact_unit != fact_value + fact_unit:
            conflicts.append(f"已更新事实：{fact_key} 从 {old} 更新为 {new}")

        facts.append(UserFactMemory(
            session_ref_id=session.id,
            fact_group="exam_item",  # 分组：patient_profile / exam_item / medical_history / ...
            fact_key=fact_key,       # 指标编码
            fact_value=fact_value,   # 数值
            fact_unit=fact_unit,     # 单位
            source_label="pipeline", # 来源标记
            confidence="high",       # 置信度
        ))
```

**为什么是覆写而非更新**：覆写比逐条更新更简单、更安全。如果用户在第二次体检中新增了一个指标（比如加了 ALT），覆写直接把新体检的全部指标写入，不需要判断「这条是新增还是更新还是删除」。冲突检测只用于**告知用户变化**，不影响存储策略。

**第四步：诊断记忆版本化**（初诊评估完成后）

```python
def _append_diagnostic_memory(self, db, session_ref_id, structured_result):
    # 1. 查旧版本
    existing = db.query(DiagnosticMemory).filter_by(
        session_ref_id=session_ref_id, is_current="true"
    ).first()

    # 2. 标记旧版
    if existing:
        existing.is_current = "false"
        version_no = existing.version_no + 1
    else:
        version_no = 1

    # 3. 写入新版
    db.add(DiagnosticMemory(
        session_ref_id=session_ref_id,
        version_no=version_no,
        is_current="true",
        health_status=primary["health_status"],           # "high_risk"
        urgency_level=primary["urgency_level"],           # "urgent"
        risk_summary="高血压风险(高血压,high)；慢性肾病风险(慢性肾病,high)",
        abnormal_indicator_summary="收缩压=176mmHg；舒张压=108mmHg；肌酐=128umol/L...",
        department_summary="心内科、肾内科",
        follow_up_summary="动态血压监测、肌酐复查、eGFR复查、尿蛋白评估",
        lifestyle_summary="限盐、控制血压、避免肾毒性药物、低盐饮食",
        medication_summary="合并高血压时可由医生评估 ACEI/ARB 类药物...",
        contraindication_summary="肾功能不全时部分药物需要减量或禁用",
        evidence_summary="成人高血压管理要点；慢性肾病风险识别；..."
    ))
```

**为什么用结构化字段而非存 JSON**：每个字段被独立存储，后续可以直接用 SQL 查询——「最近 10 次诊断中 health_status 的变化」「所有包含 hypertension 风险的诊断」。如果存一个大 JSON blob，这些查询需要全量加载再过滤。

**第五步：对话摘要触发**

```python
# 在 record_assistant_message 中
session.summary_pending_chars += len(content)  # 累计未处理字符数

trigger_chars = getattr(self._settings, "summary_trigger_chars", 2000)
allow_llm_summary = session.summary_pending_chars >= trigger_chars

session.conversation_summary = self._build_session_summary(
    session.id, session.conversation_summary, db, allow_llm=allow_llm_summary
)

if allow_llm_summary:
    session.summary_pending_chars = 0  # 重置计数器
```

`_build_session_summary` 的双轨策略：

```text
如果 allow_llm=True AND chat_model 可用:
  取最近 6 条对话 → 拼接为文本 → 调用摘要 LLM → 压缩为一段摘要
  如果 LLM 失败 → 回退确定性摘要

如果 allow_llm=False:
  确定性摘要：取最近 6 条对话的 content_summary → 用 " | " 拼接 → 截断到 1200 字符
```

**为什么不是每次对话都调 LLM**：初诊后的追问可能是连续 5-10 轮的快速对话（"严重吗""去哪个科""要做什么检查""饮食注意什么"），每轮都调 LLM 做摘要会显著增加延迟和 API 开销。累计字符阈值 2000 的策略下，前 5-8 轮对话用确定性摘要（毫秒级），只在长对话时触发一次 LLM 摘要。

### 3.3 读取生命周期

**触发时机**：追问路径的 MemoryAgent 执行 `_resolve_session_history()` 时。

```python
def build_context(self, session_id, user_input):
    # 1. 读取四层记忆
    recent_messages = db.query(ConversationMemory)\
        .filter_by(session_ref_id=session.id)\
        .order_by(desc(ConversationMemory.id))\
        .limit(settings.chat_recent_messages_limit * 2)\  # 默认 12 条
        .all()

    fact_rows = db.query(UserFactMemory)\
        .filter_by(session_ref_id=session.id).all()

    diagnostics = db.query(DiagnosticMemory)\
        .filter_by(session_ref_id=session.id)\
        .order_by(desc(DiagnosticMemory.version_no))\
        .limit(2).all()  # 最近 2 版

    # 2. 组装上下文 prompt
    history = []

    # 2a. 注入使用规则（最前面，保证不被截断）
    history.append({"role": "system", "content": "上下文使用规则：历史内容仅作参考..."})

    # 2b. 注入事实记忆（仅追问时注入，初诊时不需要）
    if fact_rows and not is_initial(user_input):
        history.append({"role": "system",
            "content": f"用户事实记忆：{summarize_facts(fact_rows)}"})

    # 2c. 注入诊断记忆 + 趋势对比（仅追问时注入）
    if diagnostics and not is_initial(user_input):
        history.append({"role": "system",
            "content": f"结构化诊断记忆：{summarize_diagnostic(diagnostics[0])}"})
        if len(diagnostics) >= 2:
            history.append({"role": "system",
                "content": summarize_diagnostic_trend(diagnostics[0], diagnostics[1])})
            # 例: "最新诊断(v2)：需重点复查，中；上次(v1)：高风险，紧急
            #      对比变化：收缩压从 176 降至 148 mmHg"

    # 2d. 注入摘要记忆
    if session.conversation_summary:
        history.append({"role": "system",
            "content": f"对话摘要记忆：{session.conversation_summary}"})

    # 2e. 注入最近对话消息（摘要版）
    for message in reversed(recent_messages):
        content = message.content_summary or message.content
        content = truncate(content, 500)  # 单条截断
        history.append({"role": message.role, "content": content})

    # 3. 裁剪总长度
    history = _clip_total_history(history, total_limit=3200)
    return SessionContextBundle(session_id=session_id, history=history)
```

### 3.4 _clip_total_history 的裁剪策略

```python
def _clip_total_history(self, history, total_limit):
    clipped = []
    total = 0
    for item in reversed(history):  # 从新到旧
        if total + len(item["content"]) > total_limit and clipped:
            continue  # 这条太长，跳过但继续尝试后面的短消息
        clipped.append(item)
        total += len(item["content"])
    return list(reversed(clipped))  # 恢复时间顺序
```

**为什么从新到旧**：新消息优先保留。旧消息被优先截掉。system prompt 中的事实记忆和诊断记忆在最前面（数组头），倒序处理时最后被处理——当 total 已经接近 limit 时，前面的 system prompt 不会被截掉（因为它已经在「已被处理」的队列中）。这是这个函数最重要的隐含保证：**确定性记忆（事实/诊断）永远不会因为对话过长而被截断**。

### 3.5 记忆如何被主业务的 Agent 消费

```text
追问请求到来
  │
  ▼
API 层: chat_history_service.build_context(session_id, user_input)
  → 产出一个格式化的 history list[dict]，注入四层记忆的文本化表示
  │
  ▼
Supervisor: aiter_events(user_input, session_history, session_id)
  → TriageAgent: session_history 用于判断初诊/追问
  │
  ▼
MemoryAgent: _resolve_session_history(state)
  → 如果有 memory_context_builder（即 chat_history_service.build_context）
  → 再次调用 build_context() 刷新记忆（确保最新状态）
  → 从刷新后的 history 中提取 memory_text
  │
  ▼
RetrievalAgent（如果需要）: _build_followup_retrieval_query(user_input, memory_text)
  → 从 memory_text 中提取疾病词 → 扩展检索 query
  │
  ▼
SynthesisAgent: _build_synthesis_prompt(user_input, session_history[-8:], evidence_text)
  → 将记忆上下文和检索证据一起喂给 LLM 生成回答
```

**为什么 MemoryAgent 要再次调用 build_context 而不是直接用传入的 session_history**：传入的 session_history 是 API 层在**请求开始时**构建的，但初诊路径的 API 层可能在 Supervisor 执行之前就已经写入了新的事实和诊断记忆。MemoryAgent 主动调用 `build_context()` 能确保它拿到的是**此刻数据库中最新的记忆状态**。这是一种防御性的「刷新」机制，防止请求链路中的中间环节对记忆做了修改但没反映到传入的 history 中。

---

## 第四部分：三条链路如何被装配到系统中

### 4.1 启动时装配

文件：`app/services/container.py` + `app/main.py`

```text
FastAPI 启动 → lifespan 回调
  ├── get_runtime() → 单例构建所有组件（20 个）
  │   ├── Settings()                        ← 读 .env
  │   ├── DatabaseManager(url)              ← PG 连接池
  │   ├── ModelFactory(settings).build()    ← Embedding/Rerank/LLM/Extractor
  │   ├── build_graph_store(settings)       ← Neo4j 或 InMemory，含 ping 自动降级
  │   ├── build_evidence_store(settings)    ← Milvus 或 InMemory，含 ping 自动降级
  │   ├── ChatHistoryService(factory, settings, llm)  ← 四层记忆管理器
  │   ├── MedicalKAGWorkflow(parser, normalizer, ...) ← 12 节点流水线
  │   └── MedicalAssessmentAgent(..., workflow,
  │                                memory_context_builder=chat_history_service.build_context)
  │       # ↑ 把记忆管理器注入 Agent
  │
  ├── database_manager.create_tables()      ← 自动建表 + 增量迁移
  ├── graph_store.ensure_schema()           ← Neo4j 约束和索引
  ├── evidence_store.ensure_schema()        ← Milvus collection
  └── (可选) knowledge_builder.build_from_seed()  ← 种子图谱 + 证据索引
```

### 4.2 运行时调用链

```text
POST /medical/agent/chat/stream
  → get_runtime() → 拿到已装配的 AppRuntime
  → runtime.chat_history_service.build_context(session_id, user_input)
      → 读 PostgreSQL → 四层记忆文本化 → 返回 history list
  → runtime.medical_agent.stream_assess_async(raw_text, history, session_id)
      → 内部调用 Supervisor.aiter_events()
          → TriageAgent → AssessmentAgent/MemoryAgent → ...
              → MemoryAgent 内部再调 chat_history_service.build_context() 刷新记忆
                  → 同一个 ChatHistoryService 实例（单例中的同一个对象）
              → RetrievalAgent 内部调 knowledge_tool._evidence_store.search()
                  → 同一个 EvidenceStore 实例（单例中的同一个对象）
  → 流结束后路由层调 record_user_message + record_assistant_message + upsert_user_fact_memory
      → 同一个 ChatHistoryService 实例 → 写入 PostgreSQL
```

**所有的持久化和检索都共享同一个单例实例**——`get_runtime()` 保证整个进程中 Neo4j driver、Milvus client、SQLAlchemy session factory、SQLite FTS5 connection 都只创建一次。这就是容器模式的核心价值：不重复创建连接，不出现「测试用了内存后端但生产用了远程后端」的参数不一致问题。
