# 医疗 KAG Agent 项目实施全过程总结

## 1. 文档定位

本文档从一个实际开发工程师的视角，对 `project_4` 项目的完整实施过程进行总结。

目标不是简单罗列代码文件，而是系统性说明：

- 项目最初的业务场景是如何被定义的
- 需求是如何逐步收敛和明确的
- 技术路线为什么这么选，而不是选别的方案
- 核心架构是如何一步步演进出来的
- 模块设计与职责如何划分
- 代码实现过程中遇到了哪些问题、踩了哪些坑
- 问题是如何定位和解决的
- 当前项目已经具备哪些能力、还处于什么阶段

这份文档是对整个会话上下文和 `project_4` 当前代码的最终汇总。

---

## 2. 项目背景与场景设计

### 2.1 原始项目背景

项目最初的基础代码来自一个通用企业文档问答 / RAG Demo。  
原项目已经具备：

- FastAPI 接口层
- LangChain / LangGraph 问答流程
- Milvus 向量库
- PostgreSQL / Redis
- 会话历史管理

但这个基础项目是围绕“企业文档检索问答”搭建的，不适合直接用于医疗体检诊断场景。

### 2.2 业务场景重新定义

在后续迭代中，场景逐步收敛为：

**面向体检场景的医疗辅助判断 Agent**

用户侧目标是：

1. 用户输入一段自然语言体检文本
2. 系统识别体检指标与病史信息
3. 系统判断：
   - 身体健康状况
   - 潜在疾病风险
4. 系统给出：
   - 建议科室
   - 复查项目
   - 生活方式干预建议
   - 用药方向建议
   - 风险提示 / 禁忌信息
5. 用户可以在评估结果基础上继续追问

### 2.3 业务边界

为了降低项目风险、确保可控性，项目从一开始就不是“自动诊断系统”，而是：

**医疗辅助评估系统**

也就是说：

- 可以给风险判断和干预方向
- 可以给检查和就医建议
- 不应直接替代医生下处方
- 需要保留“人工复核”提示能力

### 2.4 为什么先做体检场景

体检场景比自由医疗问答更适合做第一阶段落地，原因包括：

1. 输入结构化程度较高
2. 高价值信息集中在有限指标集合
3. 风险判断可以较好地规则化
4. 与指南、随访、饮食、用药方向的关联较强
5. 更适合构建“规则 + 图谱 + 检索 + Agent”的混合架构

---

## 3. 需求分析与收敛过程

### 3.1 第一阶段需求

第一阶段的目标是把原始 RAG Demo 迁移到医疗问答方向，主要聚焦：

- 当前项目在医疗场景下会出现什么问题
- 什么技术优化是必须的
- 应该如何输出一份设计 baseline

这一阶段的主要产出是：

- 场景风险分析
- 技术基线文档
- 后续实施路线

### 3.2 第二阶段需求

第二阶段对场景进行了进一步收敛：

- 输入从自由问答收敛到 **体检数据 JSON**
- 一级输出聚焦“健康状态 + 潜在疾病风险”
- 二级输出聚焦“干预 / 治疗建议方向”

同时明确了两个重要方向：

1. 技术重点放在 **检索层和知识库层**
2. 用 **KAG / 知识图谱方法论** 替代传统 RAG 主检索链

### 3.3 第三阶段需求

第三阶段从“设计”走向“编码”，新增了明确要求：

- 新项目代码单独放到 `project_4`
- 尽量保留原技术栈
- 但对于关键能力，如果原栈不适合，就允许引入新技术
- 特别是图数据库不要求强行兼容原实现

于是技术路线从“原 RAG 项目改改看”变成：

**构建一个独立的 Medical KAG-lite 项目**

### 3.4 第四阶段需求

随着实现推进，又陆续出现新要求：

- 项目不只是 KAG workflow，要最终形成 **Agent 形态**
- 要有知识库上传、知识库管理、文件解析和增量入库
- 要有 Streamlit 前端
- 要有多轮问答
- 要有会话管理、会话切换、历史记忆
- 要有 PostgreSQL 会话数据库
- 要有事实记忆和诊断记忆

需求已经从：

- “做一个场景 demo”

升级成：

- **做一个项目级、具备产品形态的医疗 Agent 原型系统**

---

## 4. 技术选型与决策理由

### 4.1 为什么不是直接继续做纯 RAG

原始 RAG 方案核心是：

- query -> chunk retrieval -> LLM answer

但体检场景的核心不是“查一段相似文本然后回答”，而是：

- 指标解释
- 阈值判断
- 风险路径推理
- 疾病 / 干预建议映射

这些更适合：

- 规则引擎
- 图谱关系
- 检索增强
- Agent 交互

因此纯 RAG 不足以支撑该场景。

### 4.2 为什么采用 KAG 方法论，而不是全盘接入完整 KAG 平台

我们借鉴了 KAG / OpenSPG 的核心方法论：

- schema-constrained construction
- knowledge/chunk mutual indexing
- graph-guided reasoning
- hybrid retrieval

但没有直接把完整 KAG 平台作为项目主框架。

原因：

1. 当前项目已有大量可复用应用层代码
2. 当前目标是项目级原型，而不是知识服务平台
3. 直接全量引入 OpenSPG/KAG 产品形态会显著提高复杂度
4. 我们可以把 KAG 的思想融入当前架构，而不必替换整个系统底座

因此最终走的是：

**KAG 方法论 + 自研轻量实现**

### 4.3 为什么选择 Neo4j

项目中图谱层最终选择 Neo4j，原因是：

1. 关系型结构非常适合医疗指标 -> 风险 -> 疾病 -> 建议的路径表达
2. Cypher 对多跳关系查询非常自然
3. Python 驱动成熟
4. 可视化调试和人工排查容易
5. 适合作为项目级图谱原型系统的落地选择

### 4.4 为什么保留 Milvus

虽然主检索从纯向量检索升级到了图谱主导，但 Milvus 仍然保留，并承担了新的职责：

- 保存证据块向量
- 保存指南、资料、补充文档的证据片段
- 在图谱检索后做证据补召回
- 支撑 hybrid retrieval

换句话说：

- **图谱层负责关系**
- **向量层负责证据**

### 4.5 为什么使用 FastAPI

FastAPI 被保留，是因为它非常适合当前项目：

1. 便于快速暴露 REST / SSE 接口
2. 便于上传文件、知识库管理、聊天接口统一管理
3. 对后续异步后台任务、流式接口都友好
4. 与 Pydantic 结合，适合复杂结构化响应

### 4.6 为什么使用 SQLAlchemy + PostgreSQL

后续会话记忆和会话管理引入 PostgreSQL，是因为：

1. 会话、消息、事实记忆、诊断记忆都属于强结构化数据
2. 与 Neo4j 和 Milvus 的职责边界清晰
3. SQLAlchemy 对模型定义、关系约束、测试支持都成熟
4. 适合后续做 session CRUD、历史回放、会话列表、标题、诊断快照

### 4.7 为什么继续使用 LangGraph

LangGraph 保留下来不是为了“做复杂 agent 流程图炫技”，而是因为它适合当前项目的底层安全执行流：

- 解析输入
- 标准化
- 规则判定
- 图谱检索
- 证据检索
- 排序融合
- 诊断生成

这些步骤都需要：

- 明确顺序
- 可观察
- 可插桩
- 可在流式场景里逐步输出

LangGraph 很适合作为这条底层确定性流水线。

### 4.8 为什么最后还要接入 LangChain Agent

LangGraph 适合底层保障，但不擅长用户体验层的多轮问答和灵活工具使用。

因此我们最终采用了：

- **LangGraph 负责底层强制规则与检索流**
- **LangChain Agent 负责对外问答与 Tool Calling**

这样可以同时满足：

1. 底层安全边界
2. 多轮自然语言交互体验

---

## 5. 核心架构路线

### 5.1 演进路线概览

项目大体经历了以下技术演进：

1. 原始企业文档 RAG Demo
2. 面向医疗场景的设计与风险分析
3. Medical KAG-lite 架构成型
4. 加入知识库录入和证据层
5. 升级为 Agent 形态
6. 增加会话数据库与多层记忆
7. 增加前端工作台与会话管理

### 5.2 最终混合架构

当前项目不是纯 Agent、也不是纯 Workflow，而是**混合架构**：

- **底层**：MedicalAssessmentWorkflow / `_execute_pipeline`
- **中间层**：规则、图谱、证据检索、排序
- **上层**：LangChain Agent + Tool Calling
- **交互层**：Streamlit 工作台

### 5.3 当前主链路

主链路可概括为：

1. 用户输入体检文本
2. MedicalAssessmentAgent 判断是否为初次体检评估
3. 若为初次评估：
   - 强制执行 `_execute_pipeline`
   - 得到结构化评估结果
   - 生成自然语言答复
4. 若为后续追问：
   - 从多层记忆构建上下文
   - Agent 结合历史与工具作答
5. 最终：
   - 流式返回过程
   - 记录会话
   - 更新用户事实记忆
   - 更新诊断记忆对象

---

## 6. 模块设计

### 6.1 核心目录结构

当前 `app/` 目录已经分成明确的层次：

- `api/`
- `core/`
- `db/`
- `graph/`
- `models/`
- `retrieval/`
- `schemas/`
- `services/`
- `workflows/`

这样的分层，能让系统保持工程可维护性，而不是所有逻辑塞在一个文件里。

### 6.2 `core`

文件：
- [app/core/settings.py](/d:/Jim1/AgentLearn/project_4/app/core/settings.py:1)

职责：
- 统一环境变量
- 数据库、图谱、向量、模型配置
- 启动参数

### 6.3 `db`

文件：
- [app/db/database.py](/d:/Jim1/AgentLearn/project_4/app/db/database.py:1)
- [app/db/models.py](/d:/Jim1/AgentLearn/project_4/app/db/models.py:1)

职责：
- PostgreSQL 数据层
- 会话、消息、事实、诊断快照表

### 6.4 `graph`

文件：
- [app/graph/store.py](/d:/Jim1/AgentLearn/project_4/app/graph/store.py:1)
- [app/graph/kb_builder.py](/d:/Jim1/AgentLearn/project_4/app/graph/kb_builder.py:1)
- [app/graph/seed_data.py](/d:/Jim1/AgentLearn/project_4/app/graph/seed_data.py:1)

职责：
- 图谱主检索
- Neo4j 约束、结构管理
- seed 图谱构建

### 6.5 `retrieval`

文件：
- [app/retrieval/evidence_store.py](/d:/Jim1/AgentLearn/project_4/app/retrieval/evidence_store.py:1)
- [app/retrieval/embeddings.py](/d:/Jim1/AgentLearn/project_4/app/retrieval/embeddings.py:1)
- [app/retrieval/lexical.py](/d:/Jim1/AgentLearn/project_4/app/retrieval/lexical.py:1)
- [app/retrieval/risk_ranker.py](/d:/Jim1/AgentLearn/project_4/app/retrieval/risk_ranker.py:1)

职责：
- 证据检索
- embedding
- lexical retrieval
- hybrid rerank

### 6.6 `models`

文件：
- [app/models/factory.py](/d:/Jim1/AgentLearn/project_4/app/models/factory.py:1)

职责：
- 统一模型工厂
- embedding provider
- LLM 输入抽取器
- rerank client
- assistant chat model

### 6.7 `services`

这里是当前项目最核心的业务层，包含：

- `medical_agent.py`
- `agent_tools.py`
- `input_parser.py`
- `indicator_normalizer.py`
- `rules.py`
- `chat_history_service.py`
- `document_ingestion.py`
- `knowledge_registry.py`
- `upload_job_registry.py`
- `evidence_query_planner.py`
- `diagnosis_formatter.py`
- `container.py`

其中关键服务分别负责：

#### `medical_agent.py`
- Agent 层
- 多轮问答
- 工具编排

#### `agent_tools.py`
- `StandardAssessmentTool`
- `MedicalKnowledgeRetrievalTool`

#### `chat_history_service.py`
- 会话生命周期
- 用户事实记忆
- 对话记忆
- 诊断记忆对象
- 上下文构建

#### `document_ingestion.py`
- 文件解析
- chunk 切分
- `linked_node_codes` 推断

### 6.8 `workflows`

文件：
- [app/workflows/medical_kag_pipeline.py](/d:/Jim1/AgentLearn/project_4/app/workflows/medical_kag_pipeline.py:1)

职责：
- 底层确定性 KAG-lite 流水线

### 6.9 `streamlit_app.py`

职责：
- 前端工作台
- 会话管理
- 知识库管理
- 聊天区
- 系统状态

---

## 7. 代码实现过程的主要阶段

### 7.1 阶段一：项目骨架搭建

在 `project_4` 刚创建时，首先搭了基础目录结构：

- `app/`
- `tests/`
- `fixtures/`
- `knowledge_sources/`

并先构建了最小可运行 vertical slice：

- 结构化体检输入
- 规则判定
- 图谱检索
- 证据检索
- 结果输出

### 7.2 阶段二：知识库录入链补全

早期项目最大的缺口之一是：

**没有“外部文档进入知识库”的完整闭环**

于是补上了：

- 文件上传接口
- 文档解析
- chunk 切分
- 证据块注册
- 知识库重建

并支持：

- `txt`
- `md`
- `pdf`
- `json`
- `html`

### 7.3 阶段三：Agent 形态改造

一开始系统虽然有完整的 KAG-lite 流水线，但更像：

- `workflow -> result`

不是对外的 Agent 形态。

于是引入：

- LangChain `create_agent`
- `StandardAssessmentTool`
- `MedicalKnowledgeRetrievalTool`

实现：

- 首次评估必须走底层安全流水线
- 后续追问可结合上下文与知识工具

### 7.4 阶段四：运行时工程问题收敛

这一步主要解决：

- embedding 维度契约
- rerank 未真正接入
- 上传知识库太慢
- Neo4j / Milvus 后端使用状态不透明

结果是：

- 维度自动对齐
- rerank 构建修复
- 后台任务化上传
- runtime status 更完整

### 7.5 阶段五：会话数据库与多层记忆

为了真正支持多轮问答和会话切换，接入了 PostgreSQL，并把记忆设计成：

- 用户事实记忆
- 对话记忆
- 诊断记忆对象

同时增加：

- 会话列表
- 会话创建 / 删除 / 切换
- 记忆上下文构造

### 7.6 阶段六：前端工作台升级

最后把 Streamlit 前端从：

- demo 页面

重构成：

- 左侧功能区
- 右侧聊天主区
- 会话管理
- 知识库管理
- 流式聊天体验

---

## 8. 核心技术选择的业务理由与优势

### 8.1 规则引擎的业务价值

体检场景里，大量判断依赖阈值：

- 血压高不高
- FBG / HbA1c 是否达糖前期或糖尿病
- LDL / TG 是否达风险阈值
- eGFR / 肌酐是否提示 CKD

这类规则如果完全交给 LLM，自由度太高、稳定性太差。

因此使用规则引擎的优势是：

1. 可解释
2. 稳定
3. 易调试
4. 符合医疗场景对边界值敏感的特点

### 8.2 图谱层的业务价值

图谱层非常适合表达：

- 指标异常 -> 风险
- 风险 -> 疾病
- 疾病 -> 科室 / 干预 / 复查

它比单纯向量召回更适合做“关系驱动”的医疗决策支持。

优势：

1. 结构关系清晰
2. 可控多跳路径
3. 易于补规则与节点
4. 适合后续扩展

### 8.3 证据检索层的业务价值

图谱只能表达结构，不能替代真实医学文本证据。

所以需要 Milvus 负责：

- 文本证据召回
- 指南片段支撑
- 生活方式 / 用药建议支撑

优势：

1. 可以提供可审计依据
2. 可以支持后续解释性输出
3. 可与图谱主路径形成互补

### 8.4 Agent 层的业务价值

仅靠 workflow 只能生成单次评估报告。

但实际用户还会继续问：

- 早餐怎么吃？
- 肌酐偏高是否严重？
- 这是不是一定要吃药？

Agent 的价值在于：

1. 多轮问答
2. 根据上下文做灵活工具调用
3. 把底层评估能力包装成用户可交互的智能助手

### 8.5 多层记忆的业务价值

医疗会话里，历史非常重要，但不能一股脑把所有对话喂给模型。

多层记忆的价值在于：

- 用户事实记忆：保存稳定事实
- 诊断记忆对象：保存确定性结果
- 对话记忆：保存语义上下文
- 摘要记忆：控制长度

这比传统“直接拼接全部历史消息”更适合医疗场景。

---

## 9. 代码过程中遇到的问题与踩坑总结

下面列的是整个实现过程中真实遇到的关键问题和处理方式。

### 9.1 问题：KAG 方案和现有栈之间怎么取舍

一开始就面临一个核心问题：

- 直接重做成 KAG 平台？
- 还是在现有项目上做 KAG-lite？

最终选择了：

- **KAG 方法论 + 现有应用栈**

原因是：

- 当前目标是项目原型，不是平台产品
- 全盘引入 OpenSPG/KAG 成本过高
- 保留现有代码更高效

### 9.2 问题：图数据库没看到实体

用户在前端看到了图谱输出，但 Neo4j 里没实体。

根因：

- 实际运行时仍然走 `in-memory` backend
- `.env` 未正确设置
- 或者启动流程中没有强制使用真实后端

解决：

- 默认配置改为真实后端优先
- 增加 runtime status
- 启动时做 schema 检查
- 补文档说明真实运行方式

### 9.3 问题：Rerank 明明开了但没生效

根因：

- `RERANK_API_KEY` 为空时，没有正确回退到 `DASHSCOPE_API_KEY`

解决：

- 工厂逻辑改成：
  - `rerank_api_key = RERANK_API_KEY or DASHSCOPE_API_KEY`

### 9.4 问题：Embedding 维度不一致

根因：

- `DENSE_EMBEDDING_DIM`
- `EVIDENCE_EMBEDDING_DIM`

如果不一致，而 evidence 仍使用同一 embedding 模型，就会导致 Milvus 维度错误。

解决：

- 配置层自动对齐 evidence 维度到 dense 维度

### 9.5 问题：知识库上传太慢

根因：

- 上传时同步执行：
  - 解析
  - chunk
  - embedding
  - 入库
  - rebuild

解决：

- BackgroundTasks 后台任务化
- 上传立即返回 job_id
- 前端轮询任务状态
- embedding 改为 batch
- 已就绪后端走增量写入，不做全量 rebuild

### 9.6 问题：知识文件会被重复上传

根因：

- 早期没有持久化查重

解决：

- 基于 `content_hash`
- 持久化 registry
- 重复文件直接跳过

### 9.7 问题：上传文档后 rebuild 会覆盖掉上传内容

根因：

- rebuild 只重放 seed 数据

解决：

- `knowledge_registry` 存文档和 chunks
- `build_from_seed()` 时把上传文档一起合并

### 9.8 问题：LangChain 方案与当前版本 API 不兼容

Gemini 建议的：

- `create_openai_tools_agent`
- `AgentExecutor`

在当前环境 `langchain==1.2.10` 下并不存在。

解决：

- 改为 `langchain.agents.create_agent`
- 用当前版本支持的 Tool Calling 能力重构

### 9.9 问题：旧版数据库 schema 与新版模型冲突

例如：

- `chat_sessions.summary_text` 旧库存在且 `NOT NULL`
- 新模型主要使用 `conversation_summary`

导致创建会话时报：

- `summary_text null value violates not-null constraint`

解决：

- 保留兼容字段
- 启动 migration 补默认值
- 会话创建时显式写入旧字段和值

### 9.10 问题：Streamlit API 用法错误

曾出现：

- `st.write()` 多参数导致 `StreamlitAPIException`

解决：

- 改成单字符串格式输出

### 9.11 问题：前端乱码 / 文案编码问题

在不同 shell 编码环境下，中文文案容易出现 mojibake 现象。

解决：

- Streamlit 页面改成 UTF-8 源码 + Unicode 转义方案
- 页面级文案统一中文

### 9.12 问题：事实更新应该如何提示

如果用户第二次输入和第一次指标不同，不应静默覆盖。

解决：

- 显式生成 “已更新事实”
- 存入 `conversation_memories`
- 流式事件单独输出
- 追加到 assistant 最终文本里

### 9.13 问题：会话系统不能只靠前端 local state

如果只依赖前端 `session_state`：

- 刷新就没了
- 无法真正切换历史会话

解决：

- 接入 PostgreSQL 会话库
- 前后端共同管理 session_id

---

## 10. 当前系统能力总览

到当前为止，项目已经具备：

### 输入处理能力

- 自然语言体检文本输入
- 可选 JSON 输入
- 结构化解析
- 单位归一
- 指标别名映射

### 风险评估能力

- 单指标规则判定
- 组合规则判定
- 图谱主检索
- 证据补召回
- 风险排序融合

### 知识库能力

- 知识文件上传
- 文档持久化
- 去重
- chunk 生成
- 节点映射
- 任务化上传
- 文档列表查看
- 知识库重建

### Agent 能力

- 首次评估 Tool Calling
- 后续追问 Tool Calling
- 多轮自然语言问答
- 流式过程输出

### 会话能力

- 会话新建
- 会话切换
- 会话删除
- 历史消息回放
- 用户事实记忆
- 对话记忆
- 诊断记忆对象
- 会话摘要记忆

### 前端能力

- 左侧功能工作台
- 右侧主聊天区
- 流式问答体验
- 会话列表
- 知识库管理
- 系统状态查看

---

## 11. 当前测试与验证结果

当前项目已经有较完整的回归测试体系，覆盖：

- 输入解析
- 指标归一化
- 规则引擎
- 检索链
- 知识库构建
- 文档上传
- Agent Tool
- 会话数据库
- 记忆上下文
- 事实冲突检测
- 诊断记忆版本化

当前测试结果：

- `pytest`: **22 passed**

这说明当前项目已经不是一次性拼装的 demo，而是开始具备持续迭代的工程基础。

---

## 12. 当前仍然存在的边界

虽然项目已经具备完整主链路，但仍有一些边界：

1. 图谱本体仍然是轻量版，未自动从文档抽取全新 schema
2. 知识库删除 / 覆盖更新 / docx 支持等管理能力还可以继续补
3. 对话记忆虽然已经很完整，但“版本回溯查看”与“记忆调试面板”还可以继续做
4. 目前是项目级原型系统，还未进入真正生产级权限、安全、审计体系

---

## 13. 下一步建议

如果继续推进，最值得做的是：

1. 会话标题编辑与搜索
2. 诊断记忆对象版本回溯页
3. 用户事实记忆可视化与人工修正
4. 文档删除 / 覆盖更新 / docx 支持
5. 更完整的生产级审计和权限隔离

---

## 14. 最终总结

整个项目从一个企业文档问答 Demo，逐步演进为一个具有以下特点的医疗 Agent 原型系统：

- 业务上聚焦体检场景
- 架构上采用规则 + 图谱 + 检索 + Agent 的混合方案
- 工程上已经拆出清晰模块层
- 交互上已经具备知识库优先、会话管理、多轮问答、流式过程反馈
- 记忆上已经具备用户事实记忆、对话记忆、诊断记忆对象和摘要记忆

换句话说，这个项目已经不是“把几个技术拼一起”的试验，而是一个**具备项目级实现思路和可持续迭代结构的医疗 Agent 工作台原型**。

