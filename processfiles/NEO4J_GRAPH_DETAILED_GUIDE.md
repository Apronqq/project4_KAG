# Neo4j 知识图谱 — 结构、入库、查询与种子数据

> 聚焦三个问题：图谱中存了什么、怎么存进去的、种子数据承担什么角色。

---

## 1. 图谱中存了什么 — 六种节点 + 五类关系

### 完整 Schema

```
┌─────────────────┐      ┌─────────────────┐      ┌─────────────────┐
│  IndicatorState │      │   DiseaseRisk   │      │     Disease     │
│─────────────────│      │─────────────────│      │─────────────────│
│ state_code (PK) │      │ risk_code  (PK) │      │ disease_code(PK)│
│ label           │      │ name             │      │ name            │
│ rule_id         │      │ risk_level       │      │                 │
│ severity        │      │                  │      │                 │
└────────┬────────┘      └────────┬────────┘      └────────┬────────┘
         │                        │                        │
         │ STATE_IMPLIES_RISK     │ RISK_RELATED_DISEASE   │
         ├────────────────────────┤                        │
         │                        │                        │
         │                        ├────────────────────────┤
         │                        │                        │
         │                        │    ┌───────────────────┤
         │                        │    │                   │
         │                        │    │     DISEASE_      │
         │                        │    │   RECOMMENDS_     │
         │                        │    │  INTERVENTION     │
         │                        │    ├───────────────────┤
         │                        │    │   Intervention    │
         │                        │    │  ─────────────   │
         │                        │    │   name            │
         │                        │    └───────────────────┘
         │                        │
         │                        │    ┌───────────────────┐
         │                        │    │   Department      │
         │                        ├────│  ─────────────   │
         │                              │   name            │
         │                              └───────────────────┘
         │
         │                        ┌───────────────────┐
         │                        │  MedicationDir..  │
         │                        │  ───────────────  │
         │                        │  name              │
         │                        └───────────────────┘
         │
         │                        ┌───────────────────┐
         │                        │  FollowUpTest     │
         │                        │  ───────────────  │
         │                        │  name              │
         │                        └───────────────────┘
         │
         │                        ┌───────────────────┐
         │                        │ Contraindication  │
         │                        │  ───────────────  │
         │                        │  name              │
         │                        └───────────────────┘
         │
         │
         │  ┌─────────────────┐
         │  │  EvidenceChunk  │  ← 存在图谱中，但通过独立的关系连入
         │  │─────────────────│
         │  │ chunk_id   (PK) │
         └──│ doc_id          │
            │ title           │
            │ text            │
            │ source_type     │
            └─────────────────┘
                    ▲
                    │ NODE_LINKED_CHUNK
                    │ (关系方向：Risk/Disease → Chunk)
         ┌──────────┴──────────┐
         │                     │
    DiseaseRisk            Disease
```

**设计上 InMemoryGraphStore 和 Neo4jGraphStore 遵循完全相同的抽象接口**（`BaseGraphStore`），因此上面这个 schema 对两种后端都成立。区别在于 Neo4j 用 Cypher 操作真实的图数据库，InMemory 用 Python dict 模拟同等语义。

### 每种节点存什么

| 节点类型 | 唯一键 | 其他属性 | 示例 |
|---------|--------|---------|------|
| `IndicatorState` | `state_code` | label, rule_id, severity | `state_code="SBP_high_stage2"`, `severity="high"` |
| `DiseaseRisk` | `risk_code` | name, risk_level | `risk_code="hypertension_risk"`, `risk_level="high"` |
| `Disease` | `disease_code` | name | `disease_code="hypertension"`, `name="高血压"` |
| `Intervention` | name | — | `name="限盐"` |
| `Department` | name | — | `name="心内科"` |
| `FollowUpTest` | name | — | `name="动态血压监测"` |
| `Contraindication` | name | — | `name="肾功能不全时部分药物需减量"` |
| `MedicationDirection` | name | — | `name="由医生评估 ACEI/ARB 方案"` |
| `EvidenceChunk` | `chunk_id` | doc_id, title, text, source_type | `chunk_id="doc_abc123_chunk_1"`, `text="高血压患者应遵循..."` |

为什么前三类节点用**独立编码**（state_code / risk_code / disease_code）做主键，后面五类用**中文 name**？因为 IndicatorState、DiseaseRisk、Disease 是整个推理链路的骨架——它们的 code 是系统内部统一语言，贯穿规则引擎的 state_code、图谱查询、风险排序、证据链接。使用确定性编码而非自然语言，确保所有模块对同一个医学概念的引用不会有歧义。后五类节点只是骨架上的「叶子」——它们被 MERGE 到 Disease 上，系统中没有其他地方需要跨模块引用它们，用中文 name 更直观。

### 关系类型

| 关系 | 方向 | 语义 |
|------|------|------|
| `STATE_IMPLIES_RISK` | IndicatorState → DiseaseRisk | 指标异常状态隐含了某个疾病风险 |
| `RISK_RELATED_DISEASE` | DiseaseRisk → Disease | 风险指向具体疾病 |
| `DISEASE_RECOMMENDS_INTERVENTION` | Disease → Intervention | 疾病推荐的干预方向 |
| `DISEASE_RECOMMENDS_DEPARTMENT` | Disease → Department | 疾病对应的建议科室 |
| `DISEASE_REQUIRES_FOLLOWUP_TEST` | Disease → FollowUpTest | 疾病需要的复查项目 |
| `DISEASE_HAS_CONTRAINDICATION` | Disease → Contraindication | 疾病的禁忌事项 |
| `DISEASE_RECOMMENDS_MEDICATION_DIRECTION` | Disease → MedicationDirection | 疾病的用药方向建议 |
| `NODE_LINKED_CHUNK` | DiseaseRisk/Disease → EvidenceChunk | 证据片段与该节点关联 |

**NODE_LINKED_CHUNK 为什么是跨界关系**：前七种关系存在于「医学推理层」内部。NODE_LINKED_CHUNK 是唯一跨越「推理层」和「证据层」的关系——它把外部上传的文档 chunk 绑定到图谱中的 DiseaseRisk 或 Disease 节点上，使证据在检索阶段能通过图谱路径被定位。这个关系是「知识图谱指导检索」的物理基础。

---

## 2. 图谱是怎么建起来的

### 2.1 Schema 初始化

应用启动时，`lifespan` 回调执行 `graph_store.ensure_schema()`：

```python
def ensure_schema(self):
    statements = [
        # 四个唯一性约束 — 防止重复节点
        "CREATE CONSTRAINT indicator_state_code IF NOT EXISTS
         FOR (n:IndicatorState) REQUIRE n.state_code IS UNIQUE",
        "CREATE CONSTRAINT disease_risk_code IF NOT EXISTS
         FOR (n:DiseaseRisk) REQUIRE n.risk_code IS UNIQUE",
        "CREATE CONSTRAINT disease_code IF NOT EXISTS
         FOR (n:Disease) REQUIRE n.disease_code IS UNIQUE",
        "CREATE CONSTRAINT evidence_chunk_id IF NOT EXISTS
         FOR (n:EvidenceChunk) REQUIRE n.chunk_id IS UNIQUE",
        # 两个索引 — 加速按名称查询
        "CREATE INDEX disease_name_idx IF NOT EXISTS FOR (n:Disease) ON (n.name)",
        "CREATE INDEX risk_name_idx IF NOT EXISTS FOR (n:DiseaseRisk) ON (n.name)",
    ]
```

`IF NOT EXISTS` 保证幂等。这四个约束决定了一个核心行为：**当你试图用 MERGE 创建节点时，Neo4j 会先按约束字段查找是否已存在同名节点，存在则复用，不存在则新建**。这是整个数据导入链路依赖的基础机制。

### 2.2 种子数据驱动图谱构建

调用入口：`POST /medical/kb/rebuild` 或启动时 `BOOTSTRAP_KB_ON_STARTUP=true`。

执行者：`MedicalKnowledgeBuilder.build_from_seed()` → `Neo4jGraphStore.rebuild_from_seed()`。

分为三个连续的 Cypher 阶段。

**阶段一：清空全图**

```cypher
MATCH (n) DETACH DELETE n
```

**阶段二：构建推理骨架** — 遍历 STATE_TO_RISK 的 20 条映射，逐条执行：

原始数据长这样：

```python
STATE_TO_RISK = {
    "SBP_high_stage2": [{                     # 异常状态码
        "risk_code": "hypertension_risk",     # 关联的风险
        "risk_name": "高血压风险",
        "disease_code": "hypertension",        # 关联的疾病
        "disease_name": "高血压",
        "risk_level": "high",
        "graph_score": 0.95,                   # 图谱置信度
    }],
    "DBP_high_stage2": [{
        "risk_code": "hypertension_risk",     # ← 和上面同一个 risk_code
        "risk_name": "高血压风险",
        "disease_code": "hypertension",        # ← 和上面同一个 disease_code
        ...
    }],
    ...
}
```

注意 `SBP_high_stage2` 和 `DBP_high_stage2` 的 `risk_code` 和 `disease_code` 相同。对应的 Cypher：

```cypher
-- SBP_high_stage2 的映射
MERGE (s:IndicatorState {state_code: "SBP_high_stage2"})
SET s.label = "SBP high stage2", s.rule_id = "sbp_high_stage2", s.severity = "high"
MERGE (r:DiseaseRisk {risk_code: "hypertension_risk"})
SET r.name = "高血压风险", r.risk_level = "high"
MERGE (d:Disease {disease_code: "hypertension"})
SET d.name = "高血压"
MERGE (s)-[:STATE_IMPLIES_RISK]->(r)
MERGE (r)-[:RISK_RELATED_DISEASE]->(d)

-- DBP_high_stage2 的映射
MERGE (s2:IndicatorState {state_code: "DBP_high_stage2"})
SET s2.label = "DBP high stage2", ...
MERGE (r:DiseaseRisk {risk_code: "hypertension_risk"})  -- MERGE 到已存在的同一个 r
MERGE (d:Disease {disease_code: "hypertension"})        -- MERGE 到已存在的同一个 d
MERGE (s2)-[:STATE_IMPLIES_RISK]->(r)
MERGE (r)-[:RISK_RELATED_DISEASE]->(d)
```

结果：两个 `IndicatorState` 节点（`SBP_high_stage2` 和 `DBP_high_stage2`）通过各自的 `STATE_IMPLIES_RISK` 边指向**同一个** `DiseaseRisk` 节点（`hypertension_risk`），再指向同一个 `Disease` 节点（`hypertension`）。

这就是图谱的**汇聚特性**——多个异常状态可以支持同一个风险，多个风险可以指向同一个疾病。没有冗余节点，关系网自然形成。

**阶段三：挂载干预/科室/复查/禁忌/用药** — 遍历 DISEASE_TO_INTERVENTIONS 的 6 个疾病：

原始数据：

```python
DISEASE_TO_INTERVENTIONS = {
    "hypertension": {
        "interventions": ["限盐", "减重", "规律运动", "居家血压监测"],
        "medication_directions": ["评估是否需要启动降压治疗", "优先结合心内科评估 ACEI/ARB 等方案"],
        "contraindications": [],
        "follow_up_tests": ["动态血压监测", "肾功能复查", "尿常规"],
        "departments": ["心内科", "全科医学科"],
    },
    "ckd": {
        "interventions": ["控制血压", "避免肾毒性药物", "低盐饮食"],
        ...
    },
    ...
}
```

对应 Cypher：

```cypher
MERGE (d:Disease {disease_code: "hypertension"})  -- MERGE 到已存在的 d

FOREACH (name IN ["限盐", "减重", "规律运动", "居家血压监测"] |
    MERGE (i:Intervention {name: name})
    MERGE (d)-[:DISEASE_RECOMMENDS_INTERVENTION]->(i))

FOREACH (name IN ["心内科", "全科医学科"] |
    MERGE (dep:Department {name: name})
    MERGE (d)-[:DISEASE_RECOMMENDS_DEPARTMENT]->(dep))

-- 其余四类同理
```

FOREACH + MERGE：对列表中的每个元素执行一次 MERGE。不存在则创建，存在则复用。保证即使多次重建，同名的 Intervention/Department 节点不会重复。

**阶段四：证据 chunk 挂载**：

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

逐行解释：

```text
MERGE → 幂等创建 chunk 节点
SET → 更新 chunk 节点的属性（重复执行时覆盖旧值）
UNWIND codes AS code → 把 linked_node_codes 数组展开为逐行
OPTIONAL MATCH (r:DiseaseRisk {risk_code: code}) → 根据 code 查找匹配的 Risk 节点
OPTIONAL MATCH (d:Disease {disease_code: code})   → 根据 code 查找匹配的 Disease 节点
FOREACH + CASE WHEN → 如果找到了 r，创建 NODE_LINKED_CHUNK 关系；没找到则跳过
```

**为什么用 OPTIONAL MATCH + FOREACH + CASE WHEN 而不是直接用 MATCH**：因为 linked_node_codes 是文档分块时**自动推断**出来的，推断算法可能产出当前图谱中不存在的 code（比如推断算法把 "ACEI" 映射到了 "hypertension"，但 rebuild 时 hypertension 节点确实存在）。如果用 MATCH 而非 OPTIONAL MATCH，遇到不存在的节点 Cypher 会返回空行导致后续 FOREACH 不执行。OPTIONAL MATCH 保证了「能关联就关联，关联不上就跳过」的容错语义。

### 2.3 增量上传 vs 全量重建

**全量重建路径**（rebuild / 首次启动）：

```text
MedicalKnowledgeBuilder.build_from_seed()
  → _merged_evidence_chunks()
    → 种子 chunk (EVIDENCE_CHUNKS) + registry 中所有已上传文档的 chunk
    → 按 chunk_id 去重
  → graph_store.rebuild_from_seed() → 上面的四阶段全流程
  → evidence_store.rebuild_index()  → Milvus 全量重建
```

**增量上传路径**（用户上传新文档）：

```text
_process_upload_job() 检测 graph_store.data_ready() == True
  → evidence_store.add_chunks(new_chunks)     → Milvus 增量插入向量
  → graph_store.add_evidence_chunks(new_chunks) → Neo4j 增量绑定
     → 只执行阶段四的 Cypher（创建 EvidenceChunk 节点 + NODE_LINKED_CHUNK 关系）
     → 不修改 IndicatorState / DiseaseRisk / Disease 节点
```

**增量上传不会修改图谱的推理结构**——外部文档只能通过 NODE_LINKED_CHUNK 挂载到已有的 Risk/Disease 节点上，不能创建新的节点类型或关系类型。这是刻意的安全设计：证据可以自动积累，知识结构必须人工校验。

---

## 3. 种子数据承担什么角色

seed_data.py 中的三张表是整个图谱系统的**唯一数据源**。

### INDICATOR_ALIASES

```python
"收缩压": ("blood_pressure_systolic", "收缩压", "mmHg")
"fbg": ("fasting_blood_glucose", "空腹血糖", "mmol/L")
"egfr": ("egfr", "估算肾小球滤过率", "mL/min/1.73m2")
```

| 角色 | 说明 |
|------|------|
| 统一指标语言 | 无论用户输入"血压""收缩压""SBP"，最终都落到 `blood_pressure_systolic` |
| 文档分块推断 | chunk 中出现 "收缩压" → 通过别名映射到 `["hypertension_risk", "hypertension"]` |
| 归一化参考 | IndicatorNormalizer 用这张表把用户输入的指标名映射到标准 code |

### STATE_TO_RISK

```python
"SBP_high_stage2": [{"risk_code": "hypertension_risk", "disease_code": "hypertension",
                      "risk_level": "high", "graph_score": 0.95}]
"CKD_strong_combined": [{"risk_code": "ckd_risk", "disease_code": "ckd",
                          "risk_level": "high", "graph_score": 0.96}]
```

| 角色 | 说明 |
|------|------|
| 图谱推理骨架 | 每条映射在 Neo4j 中产生一个 IndicatorState 节点、对应的 DiseaseRisk 和 Disease 节点、以及 STATE_IMPLIES_RISK → RISK_RELATED_DISEASE 两跳关系 |
| 置信度量化 | `graph_score` 不是随便填的——组合规则（双指标阳性，如 CKD_strong_combined=0.96）高于单指标规则（如单独的 CREATININE_high=0.74），反映了医学证据等级的差异 |
| 文档推断 | chunk 文本中出现 `SBP_high_stage2` 字样 → 自动链接到 `hypertension_risk` + `hypertension` |

### DISEASE_TO_INTERVENTIONS

```python
"hypertension": {
    "interventions": ["限盐", "减重", "规律运动", "居家血压监测"],
    "medication_directions": ["评估是否需要启动降压治疗", "优先结合心内科评估 ACEI/ARB 等方案"],
    "departments": ["心内科", "全科医学科"],
    ...
}
```

| 角色 | 说明 |
|------|------|
| 建议生成模板 | Workflow 节点 5（expand_intervention_paths）查询后拿到的 InterventionCandidate 直接来自这张表 |
| 安全措辞约束 | 用药方向用"评估是否需要..."而非"开具 XX 药物"——这不是 LLM 的输出，是种子数据里人工撰写的、经过安全审核的文本 |
| 文档推断 | chunk 中出现"限盐""心内科""ACEI" → 推断算法将其链接到 `hypertension` |

**种子数据的维护约束**：不通过 LLM 自动生成、不通过文档上传自动扩展。新增疾病域时，人工对照指南编写 STATE_TO_RISK 映射和 DISEASE_TO_INTERVENTIONS 干预方案，确认每条映射的医学正确性和措辞安全性后再入图。这是「确定性知识 vs 概率性生成」的分界线。

---

## 4. 图谱如何被主业务查询

### 查询一：风险候选（初诊 Workflow 节点 4）

```text
输入: state_codes = ["SBP_high_stage2", "DBP_high_stage2", "CREATININE_high", "eGFR_moderately_low", ...]

Cypher:
MATCH (s:IndicatorState)-[:STATE_IMPLIES_RISK]->(r:DiseaseRisk)-[:RISK_RELATED_DISEASE]->(d:Disease)
WHERE s.state_code IN $state_codes
RETURN s.state_code, r.risk_code, r.name, r.risk_level, d.disease_code, d.name

结果:
  SBP_high_stage2 → hypertension_risk (high) → hypertension
  DBP_high_stage2 → hypertension_risk (high) → hypertension   ← 汇聚
  CREATININE_high → ckd_risk (high) → ckd
  eGFR_moderately_low → ckd_risk (high) → ckd                 ← 汇聚

聚合后: 2 个 RiskCandidate
  hypertension: supported_states=[SBP_high_stage2, DBP_high_stage2, BP_stage2_combined]
  ckd: supported_states=[CREATININE_high, eGFR_moderately_low, CKD_strong_combined]
```

### 查询二：干预建议（初诊 Workflow 节点 5）

```text
输入: disease_codes = ["hypertension", "ckd"]

Cypher:
MATCH (d:Disease) WHERE d.disease_code IN $disease_codes
OPTIONAL MATCH (d)-[:DISEASE_RECOMMENDS_INTERVENTION]->(i)
OPTIONAL MATCH (d)-[:DISEASE_RECOMMENDS_DEPARTMENT]->(dep)
OPTIONAL MATCH (d)-[:DISEASE_REQUIRES_FOLLOWUP_TEST]->(f)
OPTIONAL MATCH (d)-[:DISEASE_HAS_CONTRAINDICATION]->(c)
RETURN d.disease_code,
       collect(DISTINCT i.name) AS interventions,
       collect(DISTINCT dep.name) AS departments,
       collect(DISTINCT f.name) AS follow_up_tests,
       collect(DISTINCT c.name) AS contraindications

结果: 2 个 InterventionCandidate（hypertension 和 ckd 各一个），
      每个包含三个维度的建议列表
```

### 查询三：证据 chunk 在图谱中的角色

证据检索不直接查询图谱——chunk 在 Milvus/InMemory 中做向量检索。但图谱中的 NODE_LINKED_CHUNK 关系产出了两个间接价值：

1. **graph_overlap_score**：chunk 入库时被打上了 linked_node_codes。检索排序时，chunk 的 linked_node_codes 和当前 query 触发的图谱路径节点码做交集计算，交集越多则该 chunk 的领域相关性分数越高
2. **图谱证据可视化**：前端可通过 Neo4j 直接查询某个 DiseaseRisk 节点关联了哪些 EvidenceChunk，展示评估依据的来源链
