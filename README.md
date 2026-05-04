# 医疗 KAG Multi-Agent 工作台

> 面向体检报告解读的 KAG 辅助评估系统：LangGraph Multi-Agent Supervisor + 确定性医学 Workflow + 知识图谱 + 混合检索 + 四层会话记忆 + 安全复核。

[![Python](https://img.shields.io/badge/Python-3.10+-blue)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.111+-green)](https://fastapi.tiangolo.com/)
[![LangGraph](https://img.shields.io/badge/LangGraph-Multi--Agent-purple)](https://www.langchain.com/langgraph)
[![pytest](https://img.shields.io/badge/pytest-41%20passed-brightgreen)](tests/)
[![MRR](https://img.shields.io/badge/MRR-0.80-orange)]()
[![P@5](https://img.shields.io/badge/P@5-0.79-orange)]()

## 项目定位

本项目不是自动诊断系统，而是面向体检数据的医疗辅助评估原型。用户提交自然语言体检报告后，系统会解析体检指标，执行规则判定，基于医学知识图谱做风险路径推理，再通过混合检索补充指南和证据片段，最终输出结构化风险评估、复查建议、科室建议、生活方式干预和人工复核提示。

最新版本已从“单 ReAct Agent + Workflow”升级为真实的 LangGraph Multi-Agent 架构。`MedicalMultiAgentSupervisor` 统一协调分诊、体检评估、记忆判断、知识检索、回答合成和安全复核；同步接口和流式 SSE 接口都走同一套多 Agent 编排。

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

### 1. KAG 体检评估 Workflow

`app/workflows/medical_kag_pipeline.py` 使用 LangGraph `StateGraph` 编排 12 个确定性节点：

| 阶段 | 说明 |
|------|------|
| 输入解析 | 自然语言体检文本 -> LLM 抽取，失败时正则回退 |
| 结构校验 | 校验患者信息、指标列表、病史、用药史 |
| 指标归一化 | 指标别名映射、单位统一、标准编码补全 |
| 规则判定 | JSON 配置规则，支持单指标、组合规则、AND/OR、性别分层 |
| 图谱检索 | 异常状态 -> 疾病风险 -> 疾病 -> 建议路径 |
| 路径扩展 | 干预、科室、复查、禁忌等图谱关系扩展 |
| 查询规划 | 基于异常指标和风险候选生成多维检索 query |
| 证据检索 | Milvus 向量 + SQLite FTS5 词汇双路召回 |
| 结果排序 | RRF、Rerank、图谱信号、权威度、MMR 多样性 |
| 主诊断生成 | 健康状态、紧急程度、潜在风险 |
| 二级建议生成 | 科室、复查、生活方式、用药方向、禁忌 |
| 响应封装 | 生成 `MedicalAssessmentResponse` |

健康路径支持条件短路：没有异常指标时跳过图谱检索、证据检索和排序，降低不必要的后端访问。

### 2. Multi-Agent 协作

`app/agents/medical_multi_agent.py` 是多 Agent 核心实现。

| Agent | 职责 | 关键实现 |
|-------|------|----------|
| `TriageAgent` | 判断初诊或追问 | 结合体检指标、数值和 `session_id` 路由 |
| `AssessmentAgent` | 运行确定性 KAG Workflow | 透传 Workflow 细粒度 step 事件 |
| `MemoryAgent` | 判断是否可直接基于记忆回答 | 可注入 `ChatHistoryService.build_context` |
| `RetrievalAgent` | 补充医学知识证据 | 基于诊断记忆扩展查询，调用 LangChain `StructuredTool` |
| `SynthesisAgent` | 生成自然语言回答 | LLM 可用时调用模型，否则使用确定性 fallback |
| `SafetyReviewAgent` | 医疗安全复核 | 检测用药、剂量、疑似药名，必要时触发安全改写 |

同步路径使用 LangGraph `invoke()` 执行完整图。流式路径使用逐节点 generator：

- `iter_events()`：同步流式事件
- `aiter_events()`：异步 SSE 流式事件，初诊仍复用 `MedicalKAGWorkflow.iter_events_async()`

这保证前端能看到真实过程事件，而不是图执行完后批量返回。

### 3. 四层记忆

`app/services/chat_history_service.py` 管理会话上下文：

| 记忆层 | 存储 | 用途 |
|--------|------|------|
| 事实记忆 | PostgreSQL + 可向量召回 | 指标值、病史、用药史，新旧事实冲突检测 |
| 诊断记忆 | PostgreSQL | 结构化评估结果版本化，保留最近诊断 |
| 趋势记忆 | PostgreSQL 派生 | 比较最近两版诊断，支持“比上次是否严重”类追问 |
| 摘要记忆 | PostgreSQL | 长对话超过阈值后压缩，控制 prompt 膨胀 |

生产环境中 `container.py` 将 `chat_history_service.build_context` 注入 `MedicalMultiAgentSupervisor`，MemoryAgent 可以主动刷新四层记忆上下文；没有 `session_id` 时会回退使用调用方传入的 `session_history`。

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

