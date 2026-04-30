# 医疗 KAG Agent 工作台

> 基于知识图谱与混合检索的体检辅助评估系统 — 规则引擎 + Neo4j + Milvus + ReAct Agent + 多层记忆

[![Python](https://img.shields.io/badge/Python-3.10+-blue)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-green)](https://fastapi.tiangolo.com/)
[![pytest](https://img.shields.io/badge/pytest-36%20passed-brightgreen)](tests/)
[![MRR](https://img.shields.io/badge/MRR-0.80-orange)]()
[![P@5](https://img.shields.io/badge/P@5-0.79-orange)]()

---

## 概述

本项目构建面向体检场景的智能辅助评估 Agent 系统。用户以自然语言提交体检报告后，系统自动完成指标解析、医学规则判定、知识图谱风险推理、混合证据检索与排序融合，产出结构化诊断评估与干预建议，并支持多轮追问。

### 核心架构

```
用户输入（自然语言体检文本）
        │
        ▼
┌──────────────────────────┐
│  MedicalAssessmentAgent  │
│  初诊 → 确定性 Workflow   │
│  追问 → ReAct Agent 循环  │
└──────────┬───────────────┘
           │
    ┌──────┴──────┐
    ▼              ▼
┌────────┐   ┌──────────┐
│ Neo4j  │   │  Milvus  │
│ 知识图谱│   │ 证据向量库│
└────┬───┘   └────┬─────┘
     │            │
     └─────┬──────┘
           ▼
┌──────────────────┐
│   混合检索排序    │
│ Dense + Lexical  │
│ + Rerank + MMR   │
└────────┬─────────┘
         ▼
┌──────────────────┐
│  PostgreSQL 会话  │
│  四层记忆模型     │
└──────────────────┘
```

---

## 快速开始

### 环境要求

- Python 3.10+
- Neo4j 4.4+（可选，支持内存模式回退）
- Milvus 2.2+（可选，支持内存模式回退）
- PostgreSQL 12+（必须）
- [DashScope API Key](https://dashscope.console.aliyun.com/)（LLM / Embedding / Rerank）

### 安装

```bash
cd project_4

# 创建虚拟环境
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 安装依赖
pip install fastapi uvicorn langchain langgraph langchain-community \
    neo4j pymilvus sqlalchemy psycopg2-binary pydantic \
    streamlit requests pypdf python-dotenv pytest httpx
```

### 配置

编辑 `.env` 文件：

```env
# 模型配置
DASHSCOPE_API_KEY=your_api_key
MODEL=qwen3-max

# Neo4j（可选：设置 USE_IN_MEMORY_GRAPH=true 使用内存模式）
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=your_password

# Milvus（可选：设置 USE_IN_MEMORY_EVIDENCE=true 使用内存模式）
MILVUS_URI=http://localhost:19530

# PostgreSQL
DATABASE_URL=postgresql+psycopg2://postgres:password@localhost:5432/medical_agent

# 启动时自动构建知识库
BOOTSTRAP_KB_ON_STARTUP=true
```

> **提示**：如果不想安装 Neo4j 和 Milvus，设置 `USE_IN_MEMORY_GRAPH=true` 和 `USE_IN_MEMORY_EVIDENCE=true` 即可使用内存模式快速体验。

### 启动

```bash
# 1. 启动 FastAPI 后端
uvicorn app.main:app --host 0.0.0.0 --port 8000

# 2. 启动 Streamlit 前端（新终端）
streamlit run streamlit_app.py
```

前端访问 `http://localhost:8501`，API 文档访问 `http://localhost:8000/docs`。

---

## API 端点

### 医疗评估

| 方法 | 路径 | 说明 |
|------|------|------|
| `POST` | `/medical/exam/parse` | 仅解析体检文本，返回结构化指标 |
| `POST` | `/medical/exam/assess` | 同步评估（返回完整结构化结果） |
| `POST` | `/medical/agent/chat` | Agent 对话（初诊/追问自动分流） |
| `POST` | `/medical/agent/chat/stream` | Agent 流式对话（SSE，含处理步骤推送） |

### 知识库管理

| 方法 | 路径 | 说明 |
|------|------|------|
| `POST` | `/medical/kb/upload` | 上传文档（txt/md/pdf/json/html） |
| `GET` | `/medical/kb/documents` | 列出已入库文档 |
| `GET` | `/medical/kb/jobs` | 查询上传任务状态 |
| `POST` | `/medical/kb/rebuild` | 重建知识库（图谱 + 向量索引） |

### 会话管理

| 方法 | 路径 | 说明 |
|------|------|------|
| `GET` | `/medical/sessions` | 列出全部会话 |
| `POST` | `/medical/sessions` | 创建新会话 |
| `GET` | `/medical/sessions/{id}` | 获取会话历史消息 |
| `DELETE` | `/medical/sessions/{id}` | 删除会话 |

### 系统状态

| 方法 | 路径 | 说明 |
|------|------|------|
| `GET` | `/health` | 健康检查（后端 + 降级状态） |
| `GET` | `/medical/runtime/status` | 运行时状态（组件延迟、降级原因） |

---

## 项目结构

```
project_4/
├── app/
│   ├── main.py                    # FastAPI 入口 + lifespan
│   ├── core/
│   │   └── settings.py            # 环境变量配置（21 项可配置）
│   ├── db/
│   │   ├── database.py            # PostgreSQL 连接管理 + 自动迁移
│   │   └── models.py              # ORM 模型（会话/事实/诊断/对话）
│   ├── graph/
│   │   ├── store.py               # Neo4j / InMemory 双后端图谱存储
│   │   ├── kb_builder.py          # 知识库构建编排
│   │   └── seed_data.py           # Seed 医学图谱 + 指标别名表
│   ├── retrieval/
│   │   ├── evidence_store.py      # Milvus / InMemory 双后端证据检索
│   │   ├── lexical.py             # BM25Lite + SQLite FTS5 词汇检索
│   │   ├── risk_ranker.py         # 风险排序融合
│   │   └── embeddings.py          # 轻量 hash embedder（回退方案）
│   ├── models/
│   │   └── factory.py             # 模型工厂（Embedding/Rerank/LLM/Extractor）
│   ├── schemas/
│   │   └── exam.py                # Pydantic 数据模型（27 个类）
│   ├── services/
│   │   ├── medical_agent.py       # Agent 编排入口（初诊/追问分流）
│   │   ├── react_agent.py         # 自研 ReAct Agent 循环
│   │   ├── agent_tools.py         # 知识库检索工具
│   │   ├── input_parser.py        # 输入解析（LLM 提取 + 正则回退）
│   │   ├── indicator_normalizer.py # 指标别名映射 + 单位归一化
│   │   ├── rules.py               # JSON 配置驱动的规则引擎
│   │   ├── chat_history_service.py # 会话记忆管理（四层记忆）
│   │   ├── diagnosis_formatter.py  # 诊断结果格式化
│   │   ├── evidence_query_planner.py # 多维度检索查询规划
│   │   ├── document_ingestion.py   # 文档解析 + 分块 + 节点推断
│   │   ├── knowledge_registry.py   # 知识库文档元数据持久化
│   │   ├── upload_job_registry.py  # 上传任务状态管理
│   │   └── container.py            # DI 容器（AppRuntime 单例）
│   ├── workflows/
│   │   └── medical_kag_pipeline.py # LangGraph KAG 确定性流水线（12 节点）
│   ├── api/routes/
│   │   └── medical.py             # FastAPI 路由（评估/知识库/会话/状态）
│   └── config/
│       └── medical_rules.json     # 医学阈值规则配置
├── tests/                          # 36 项 pytest 回归测试
├── fixtures/exam_cases/           # 22 个体检用例样本
├── knowledge_sources/             # 公开医学资料（可上传到知识库）
├── streamlit_app.py               # Streamlit 前端工作台
├── .env                           # 环境变量模板
└── data/                          # 运行时数据（知识库注册表/FTS 索引/上传文件）
```

---

## 核心能力

### 体检评估流水线

| 步骤 | 描述 |
|:----:|------|
| 输入解析 | 自然语言文本 → LLM 结构化抽取（回退正则）→ 指标别名映射 → 单位归一化 |
| 规则判定 | JSON 配置驱动，支持单指标/组合规则，AND/OR 嵌套，性别分层，年龄加权 |
| 图谱检索 | Neo4j 多跳查询：异常状态 → 疾病风险 → 疾病 → 干预/科室/复查/禁忌 |
| 证据检索 | 多查询规划 → 稠密向量（Milvus）+ 词汇倒排（SQLite FTS5）RRF 融合 → Rerank 语义重排 → 图谱/权威度多信号加权 → MMR 多样性选择 |
| 诊断生成 | 健康状态分级 + 紧急程度 + 风险列表 + 干预建议 + 人工复核标记 |

### Agent 多轮问答

- **初诊路径**：强制执行确定性 Workflow，LLM 不可绕过。结果用于格式化自然语言回答
- **追问路径**：自研 ReAct Agent 循环接管，分层决策 — 诊断记忆优先 → 事实记忆参考 → 知识库检索补充 → LLM 综合回答
- **流式事件**：7 种结构化事件（thinking → decision → tool_call → tool_result → warning → synthesizing → final_answer），全链路前端可见

### 多层记忆

| 记忆层 | 存储 | 特性 |
|--------|------|------|
| 事实记忆 | PostgreSQL + Milvus | 指标值/病史/用药；新旧冲突检测与显式更新提示；向量化语义检索 top 5 防 prompt 膨胀 |
| 诊断记忆 | PostgreSQL | 版本化（version_no + is_current）；追问注入最近两版做趋势对比 |
| 对话记忆 | PostgreSQL | 原文 + 摘要双字段；三级长度控制（单条 500 字符 / 12 条 / 总 3200 字符） |
| 摘要记忆 | PostgreSQL | 累计字符阈值 2000 触发 LLM 重新生成，未达阈值用确定性摘要 |

### 检索评测

| 指标 | 得分 | 相对纯向量基线提升 |
|------|:----:|:------------------:|
| MRR | **0.80** | +27.5% |
| Precision@5 | **0.79** | — |
| nDCG@5 | **0.75** | — |

各模块 Ablation 贡献：Rerank 前置贡献 +11% MRR，图谱信号贡献 +9% MRR。

### 知识库防污染

六层纵深防御：文件类型白名单 → SHA-256 哈希去重 → 文本抽取判空 → 医学相关性门控（三维评分 < 0.15 直接拒绝） → 来源权威度降权（未验证 0.3 vs 指南 1.0） → 前端标签告警。

### 系统韧性

- Neo4j / Milvus 不可达 → 自动降级 InMemory 后端，降级原因通过 API 暴露
- 健康路径条件短路（无异常指标时跳过检索/排序步骤，延迟 ~50ms）
- 全链路 12+ 降级点结构化日志
- 6 个后端组件级健康检查含延迟检测

---

## 运行测试

```bash
pytest tests/ -v
# 预期输出: 36 passed
```

测试覆盖：输入解析、指标归一化、规则判定（含 AND/OR 嵌套 + 性别分层）、证据检索（Milvus distance 复用 / FTS5 / Rerank 前置 / MMR）、ReAct Agent 决策分支、会话记忆（诊断版本化 / 趋势 diff / 摘要阈值）、同步异步一致性。

---

## 示例

### 输入

```
男，52岁，血压 176/108 mmHg，肌酐 128 umol/L，eGFR 54 mL/min/1.73m2，
既往史：高血压，用药史：缬沙坦。请判断健康状况、潜在疾病风险，并给出干预建议。
```

### 输出（结构化，部分）

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
