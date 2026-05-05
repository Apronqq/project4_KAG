# 医疗 KAG Multi-Agent 工作台

> 基于体检报告构建用户健康画像的智能评估系统 — 能读懂报告、记住数据、关联知识、给出个性化建议的 Multi-Agent 协作体。

[![Python](https://img.shields.io/badge/Python-3.10+-blue)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.111+-green)](https://fastapi.tiangolo.com/)
[![LangGraph](https://img.shields.io/badge/LangGraph-Multi--Agent-purple)](https://www.langchain.com/langgraph)
[![pytest](https://img.shields.io/badge/pytest-41%20passed-brightgreen)](tests/)
[![MRR](https://img.shields.io/badge/MRR-0.80-orange)]()
[![P@5](https://img.shields.io/badge/P@5-0.79-orange)]()

## 项目定位

**这不是一个”问一句、查一段、回一句”的简单问答系统。**

用户提交体检报告后，系统会像医生一样逐项解读指标、对照医学标准做风险判定、借助知识图谱推导指标之间的关联关系，并生成结构化的评估报告。但核心能力不在于单次评估——而在于**记住了你**。

系统将每次体检数据沉淀为用户的**个体健康画像**：你的年龄性别、每次的血压血糖肌酐 eGFR 数据、每次评估诊断出的风险等级和建议方向，都以结构化记忆的形式被持久化管理。当你下次回来追问”我最近血压控制得怎么样”时，系统不是去漫无目的地检索——它直接调取你的健康画像，对比最近两次诊断结果，告诉你收缩压从 176 降到了 152，风险等级从高风险变为需重点复查，并基于你的具体数据而非泛泛的医学常识来回答问题。

**一句话概括**：系统读得懂体检报告、记得住你的身体数据、关联得了医学知识、给得出个性化的健康建议——而且是基于权威指南来源、每一步推理都可追溯的建议。

在技术层面，系统由 LangGraph Multi-Agent Supervisor 编排六个专职 Agent 协作完成：分诊 Agent（判断初诊还是追问）、评估 Agent（确定性 Workflow 执行指标解析、规则判定、图谱推理和证据检索）、记忆 Agent（管理四层结构化记忆）、检索 Agent（从知识库中找回与当前问题最相关的证据）、合成 Agent（整合记忆与证据生成回答）、安全复核 Agent（检测并修正不安全输出）。

## 架构概览

```text
用户输入
  |
  v
MedicalAssessmentAgent
  |
  v
MedicalMultiAgentSupervisor (LangGraph StateGraph)
  |
  +-- TriageAgent
  |     判断首次体检评估 / 历史追问
  |
  +-- AssessmentAgent
  |     调用确定性 KAG Workflow
  |     解析 -> 归一化 -> 规则 -> 图谱 -> 证据 -> 排序 -> 诊断
  |
  +-- MemoryAgent
  |     接入 ChatHistoryService 四层记忆
  |     事实记忆 / 诊断记忆 / 趋势记忆 / 摘要记忆
  |
  +-- RetrievalAgent
  |     LangChain StructuredTool
  |     记忆感知查询扩展 -> 医学知识库检索
  |
  +-- SynthesisAgent
  |     整合记忆、证据和用户问题生成自然语言回答
  |
  +-- SafetyReviewAgent
        用药、剂量、疑似药名、高风险结果复核
        必要时通过条件边回到 SynthesisAgent 安全改写
```

底层知识与数据后端：

```text
Neo4j / InMemory Graph
  指标异常状态 -> 疾病风险 -> 疾病 -> 干预 / 科室 / 复查 / 禁忌

Milvus / InMemory Evidence
  指南、科普、上传文档证据块向量检索

SQLite FTS5
  词汇检索和 BM25 召回

PostgreSQL
  会话、事实记忆、诊断版本、对话摘要
```

## 核心能力

### 1. 能读懂报告 — 确定性医学评估

系统接到体检文本后，不是直接把问题发给 LLM，而是进入一条 **12 步确定性流水线**：自然语言文本被解析为结构化指标（”收缩压 176” → `blood_pressure_systolic=176 mmHg`），经规则引擎对照医学指南标准逐项判断（≥160 判定为 2 级高血压），再由 Neo4j 知识图谱沿着”异常状态→疾病风险→疾病→干预/科室/复查/禁忌”多跳链路推理，同时从 Milvus 知识库中召回支持该判断的指南片段作为证据。最终产出一份包含健康状态分级、紧急程度、风险列表、建议科室、复查项目、生活方式干预方向和人工复核提示的**结构化评估报告**——每个判断都有据可查，不是 LLM 的凭空推理。

如果体检指标全部在正常范围内，系统自动跳过图谱检索和证据查询，直接给出”各项指标正常”的结论，响应时间 < 50ms。

### 2. 能记住你 — 个体健康画像

系统将每次体检数据沉淀为结构化的**个体健康画像**，而不是模糊的聊天记录：

- **事实记忆**：你的年龄、性别、每次体检中每项指标的具体数值和单位。如果你第二次提交的数据中空腹血糖从 6.1 变为 7.2，系统会主动标注”空腹血糖已更新”，而不是悄悄覆盖
- **诊断记忆**：每次评估的结构化结果（健康状态、风险等级、建议方向）以版本化方式存储。追问时自动对比最近两版诊断，告诉你”收缩压从 176 降至 152 mmHg，风险从高风险变为需重点复查”
- **趋势感知**：当你问”我的血压控制得怎么样”，系统不是去检索”血压”相关的通用知识，而是直接调取你的健康画像中最近两次血压数据和诊断变化趋势

### 3. 能个性化回答 — 六 Agent 协作

追问不走简单的”检索+LLM 回答”路线，而是由 LangGraph Supervisor 编排六个专职 Agent 分工协作：

- **分诊 Agent** 判断这是新体检报告还是基于历史数据的追问
- **记忆 Agent** 检查你的健康画像中是否有相关信息——如果有且足够回答问题，直接用画像数据回答，不需要去知识库检索
- **检索 Agent** 当问题超出画像覆盖范围时（如”高血压患者该怎么吃”），在检索前先从记忆提取你的疾病标签（”高血压”），将问题改写为”高血压患者该怎么吃 高血压 饮食管理 指南”，用画像信息增强检索精准度
- **合成 Agent** 将画像中的个人数据和检索到的医学证据整合为自然语言回答
- **安全复核 Agent** 在回答返回前检查是否出现了具体药名、剂量建议（如”50mg”、”1片/次”），一旦检测到就触发安全改写

### 4. 能保护你 — 多层安全防线

- 指标阈值判定由规则引擎完成，不交给 LLM 决定（收缩压 ≥ 160 就是 2 级高血压，LLM 不能”觉得不太严重”而忽略）
- Workflow 强制初诊必须走完整评估链路，LLM 不可绕过任何判断步骤
- 知识图谱限定风险推理方向，不会凭空关联不存在的疾病关系
- 上传文档经过六层防污染检查，无关文件（如公司财报、养宠指南）自动被拒绝或降权
- 回答中检测到疑似药名、剂量表达时触发安全改写，而非简单追加一句”请咨询医生”

### 4. 检索与知识库

证据检索链路：

```text
查询规划 / 追问查询扩展
  -> Milvus dense retrieval
  -> SQLite FTS5 lexical retrieval
  -> RRF 融合
  -> Rerank 语义重排
  -> 图谱节点重叠度 + 来源权威度 + 词汇匹配分
  -> MMR 多样性选择
```

评测指标：

| 指标 | 得分 | 说明 |
|------|------|------|
| MRR | 0.80 | 相对纯向量基线提升 27.5% |
| Precision@5 | 0.79 | Top 5 证据准确率 |
| nDCG@5 | 0.75 | 排序质量 |

知识库上传支持 `txt/md/pdf/json/html`。文档入库前会经过文件类型白名单、SHA-256 去重、文本抽取判空、医学相关性门控和来源权威度降权。

### 5. 医疗安全边界

系统采用多层安全策略：

- 规则引擎负责确定性阈值判断，避免 LLM 自行判定核心指标。
- Workflow 强制初诊必须走完整医学评估链路。
- 图谱关系限定风险推理方向。
- RetrievalAgent 提供证据补充，降低无依据回答。
- SafetyReviewAgent 检查高风险、用药调整、剂量表达和疑似药名。
- 高风险或复杂结果自动标记 `human_review_required=true`。

如果回答中出现类似 `50mg`、`1片`、`次/日` 的具体剂量，或疑似药名/治疗词，SafetyReviewAgent 会通过 LangGraph 条件边路由回 SynthesisAgent，生成不含具体处方建议的安全改写版回答。

## 快速开始

### 环境要求

- Python 3.10+
- PostgreSQL 12+，用于会话和记忆
- Neo4j 4.4+，可选，支持 InMemory 回退
- Milvus 2.2+，可选，支持 InMemory 回退
- DashScope API Key，可选，用于 LLM、Embedding、Rerank；未配置时部分能力会降级

### 安装

```bash
cd project_4

python -m venv venv
venv\Scripts\activate

pip install -e .
pip install pytest
```

也可以手动安装核心依赖：

```bash
pip install fastapi uvicorn pydantic python-dotenv langgraph langchain langchain-community ^
    neo4j pymilvus sqlalchemy psycopg2-binary requests streamlit pypdf pytest
```

### 配置

创建或编辑 `.env`：

```env
# 模型配置
DASHSCOPE_API_KEY=your_api_key
MODEL=qwen3-max
EMBEDDING_MODEL=text-embedding-v4
RERANK_MODEL=qwen3-vl-rerank

# PostgreSQL
DATABASE_URL=postgresql+psycopg2://postgres:password@localhost:5432/medical_agent

# Neo4j，可选
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=your_password
USE_IN_MEMORY_GRAPH=false

# Milvus，可选
MILVUS_URI=http://localhost:19530
USE_IN_MEMORY_EVIDENCE=false

# 知识库
BOOTSTRAP_KB_ON_STARTUP=true
TOP_K_EVIDENCE=5
```

本地快速体验时可以开启内存回退：

```env
USE_IN_MEMORY_GRAPH=true
USE_IN_MEMORY_EVIDENCE=true
```

### 启动

```bash
# FastAPI 后端
uvicorn app.main:app --host 0.0.0.0 --port 8000

# Streamlit 前端
streamlit run streamlit_app.py
```

访问：

- 前端工作台：`http://localhost:8501`
- API 文档：`http://localhost:8000/docs`
- 健康检查：`http://localhost:8000/health`

## API

### 医疗评估

| 方法 | 路径 | 说明 |
|------|------|------|
| `POST` | `/medical/exam/parse` | 仅解析体检文本，返回结构化指标 |
| `POST` | `/medical/exam/assess` | 同步体检评估，返回结构化结果 |
| `POST` | `/medical/agent/chat` | Multi-Agent 对话，自动区分初诊和追问 |
| `POST` | `/medical/agent/chat/stream` | SSE 流式对话，实时推送 Agent 和 Workflow 事件 |

### 知识库

| 方法 | 路径 | 说明 |
|------|------|------|
| `POST` | `/medical/kb/upload` | 上传文档并增量入库 |
| `GET` | `/medical/kb/documents` | 查询已入库文档 |
| `GET` | `/medical/kb/jobs` | 查询上传任务状态 |
| `POST` | `/medical/kb/rebuild` | 重建 Seed 图谱和证据索引 |

### 会话

| 方法 | 路径 | 说明 |
|------|------|------|
| `GET` | `/medical/sessions` | 列出会话 |
| `POST` | `/medical/sessions` | 创建会话 |
| `GET` | `/medical/sessions/{session_id}` | 获取会话消息 |
| `DELETE` | `/medical/sessions/{session_id}` | 删除会话 |

### 系统状态

| 方法 | 路径 | 说明 |
|------|------|------|
| `GET` | `/health` | 应用健康检查 |
| `GET` | `/medical/runtime/status` | 图谱、证据库、模型、数据库组件状态和降级原因 |

## 流式事件

`/medical/agent/chat/stream` 返回 SSE，常见事件包括：

| 事件类型 | 说明 |
|----------|------|
| `meta` | 会话 ID |
| `agent_decision` | Agent 路由或决策 |
| `agent_thinking` | Agent 开始处理 |
| `step` | Workflow 细粒度步骤状态 |
| `tool_call` | RetrievalAgent 调用知识库工具 |
| `tool_result` | 工具返回结果摘要 |
| `agent_synthesizing` | SynthesisAgent 正在合成回答 |
| `content` | 最终回答文本片段 |
| `result` | 初诊结构化评估结果 |
| `memory_notice` | 事实记忆冲突或更新提示 |
| `done` | 流结束 |

## 项目结构

```text
project_4/
├── app/
│   ├── agents/
│   │   └── medical_multi_agent.py      # LangGraph Multi-Agent Supervisor
│   ├── api/routes/
│   │   └── medical.py                  # FastAPI 路由
│   ├── core/
│   │   └── settings.py                 # 环境变量配置
│   ├── db/
│   │   ├── database.py                 # PostgreSQL 连接和迁移
│   │   └── models.py                   # 会话、事实、诊断、消息 ORM
│   ├── graph/
│   │   ├── store.py                    # Neo4j / InMemory 图谱后端
│   │   ├── kb_builder.py               # 知识库构建
│   │   └── seed_data.py                # 医学图谱种子数据
│   ├── retrieval/
│   │   ├── evidence_store.py           # Milvus / InMemory 证据检索
│   │   ├── lexical.py                  # SQLite FTS5 / BM25 词汇检索
│   │   ├── risk_ranker.py              # 风险和证据融合排序
│   │   └── embeddings.py               # 轻量 embedding 回退
│   ├── services/
│   │   ├── medical_agent.py            # Multi-Agent facade
│   │   ├── agent_tools.py              # LangChain StructuredTool
│   │   ├── chat_history_service.py     # 四层记忆
│   │   ├── document_ingestion.py       # 文档解析、分块、相关性判断
│   │   ├── evidence_query_planner.py   # 初诊多查询规划
│   │   ├── input_parser.py             # 输入解析
│   │   ├── indicator_normalizer.py     # 指标归一化
│   │   ├── rules.py                    # 医学规则引擎
│   │   └── container.py                # 运行时依赖注入
│   ├── workflows/
│   │   └── medical_kag_pipeline.py     # 12 节点 KAG Workflow
│   └── config/
│       └── medical_rules.json          # 医学阈值规则
├── tests/                              # 41 项回归测试
├── fixtures/exam_cases/                # 体检样例
├── knowledge_sources/                  # 可入库医学资料
├── streamlit_app.py                    # 前端工作台
└── data/                               # 运行时索引、注册表、上传文件
```

## 测试

```bash
pytest tests/ -q
# 预期: 41 passed
```

测试覆盖：

- 输入解析和指标归一化
- 规则引擎，包含组合规则和性别分层
- KAG Workflow 同步/异步一致性
- 图谱检索、证据检索、Rerank、MMR
- LangGraph Multi-Agent 路由、真流式事件、记忆接入、检索查询扩展、安全改写
- 会话记忆、诊断版本、趋势 diff、摘要阈值
- 文档上传、知识库构建、运行状态 API

## 示例

### 初诊输入

```text
男，52岁，血压 176/108 mmHg，肌酐 128 umol/L，eGFR 54 mL/min/1.73m2，
既往史：高血压，用药史：缬沙坦。请判断健康状况、潜在疾病风险，并给出干预建议。
```

### 结构化输出节选

```json
{
  "primary_diagnosis": {
    "health_status": "high_risk",
    "urgency_level": "urgent",
    "potential_risks": [
      {
        "risk_name": "高血压风险",
        "risk_level": "high",
        "disease_name": "高血压",
        "final_score": 0.93
      },
      {
        "risk_name": "慢性肾病风险",
        "risk_level": "high",
        "disease_name": "慢性肾脏病",
        "final_score": 0.88
      }
    ]
  },
  "secondary_recommendations": {
    "recommended_departments": ["心内科", "肾内科"],
    "follow_up_tests": ["24h动态血压监测", "肾脏超声", "尿微量白蛋白"],
    "lifestyle_interventions": ["低盐饮食", "限制蛋白质摄入", "戒酒"],
    "human_review_required": true
  }
}
```

### 追问示例

```text
我的血压风险严不严重？
高血压早餐怎么吃？
我现在的药量需要调整吗？
```

追问会优先使用诊断记忆和事实记忆；如果问题需要指南或科普证据，RetrievalAgent 会自动加入记忆中的疾病上下文扩展查询。涉及药量、停药、换药、剂量等问题时，SafetyReviewAgent 会阻止直接处方化回答，并提示结合医生意见。

## 重要说明

本系统输出仅用于体检报告辅助理解、风险提示和复查方向参考，不能替代医生面诊、诊断或处方。高风险指标、症状加重、用药调整、孕产妇、儿童、老年人或合并多病情况，应及时咨询专业医生。

