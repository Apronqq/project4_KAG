# 医疗 KAG Multi-Agent 体检辅助评估系统详解

## 1. 文档目标

本文档用于完整解释 `project_4` 项目的业务背景、设计思路、技术选型、实现流程、代码结构、关键模块、创新点和运行方式。目标是让没有读过源码的人，只通过这一份文档就能理解项目的整体逻辑和主要代码细节。

项目一句话概括：

> 这是一个面向体检报告解读场景的 KAG Multi-Agent 辅助评估系统，使用 LangGraph 编排多个医学 Agent，结合确定性规则、知识图谱、混合检索、四层记忆和安全复核，完成体检风险判断与多轮追问。

## 2. 项目解决什么问题

普通 RAG 问答系统通常是：

```text
用户问题 -> 向量检索 -> LLM 生成回答
```

这种方式不适合体检场景。体检评估不是简单问答，它包含很多确定性判断和结构化推理：

- 血压、血糖、肌酐、eGFR 等指标需要按医学阈值判断。
- 单个指标异常需要映射到异常状态，例如 `SBP_high_stage2`。
- 多个异常状态需要合并成疾病风险，例如高血压风险、慢性肾脏病风险。
- 风险需要关联疾病、科室、复查项目、干预建议和禁忌。
- 用户后续会追问“严不严重”“早餐怎么吃”“是否要调药”等上下文问题。
- 医疗场景不能让 LLM 自由发挥，必须限制诊断边界和用药建议。

因此项目采用的是：

```text
确定性医学 Workflow + KAG 知识图谱 + 混合证据检索 + Multi-Agent 协作 + 安全复核
```

项目定位是“辅助评估”，不是自动诊断，也不替代医生。

## 3. 总体设计目标

项目设计围绕六个目标展开。

### 3.1 医学判断可控

关键体检指标的异常判断不能交给 LLM。项目把阈值规则写入 `app/config/medical_rules.json`，由 `IndicatorRuleEngine` 执行，保证核心判断可解释、可测试、可复现。

### 3.2 风险路径可解释

项目用图谱表达医学关系：

```text
IndicatorState -> DiseaseRisk -> Disease -> Intervention / Department / FollowUp / Contraindication
```

这样系统能说明“为什么该指标异常会关联某个疾病风险”，而不是只给出模糊结论。

### 3.3 检索证据有质量控制

项目不是单纯向量检索，而是混合检索：

```text
Milvus 向量召回 + SQLite FTS5 词汇召回 + RRF 融合 + Rerank + 图谱信号 + MMR 去重
```

这让证据召回既能覆盖语义相似，也能覆盖关键词精确匹配，并通过图谱关系提升医学相关性。

### 3.4 多轮追问要利用记忆

用户做完一次评估后，会继续问：

- “我的血压风险严不严重？”
- “我早餐怎么吃？”
- “比上次严重了吗？”
- “需要去哪个科？”

这些问题必须结合上一次诊断结果和用户事实，而不是每次都从零开始。项目通过 PostgreSQL 保存事实记忆、诊断记忆、趋势记忆和摘要记忆。

### 3.5 Agent 协作要真实存在

项目不是只把类名改成 Agent，而是用 LangGraph `StateGraph` 编排多个有职责分工的 Agent：

- `TriageAgent`
- `AssessmentAgent`
- `MemoryAgent`
- `RetrievalAgent`
- `SynthesisAgent`
- `SafetyReviewAgent`

每个 Agent 读取和更新共享状态，Supervisor 负责路由和控制流。

### 3.6 医疗安全边界必须前置

系统对高风险结果、用药调整、剂量表达、疑似药名做额外检查。涉及处方或剂量时，`SafetyReviewAgent` 会阻止直接输出处方化回答，并触发安全改写。

## 4. 总体架构

整体架构分为五层：

```text
前端 / API 层
  Streamlit + FastAPI

Agent 编排层
  MedicalAssessmentAgent
  MedicalMultiAgentSupervisor

确定性评估层
  MedicalKAGWorkflow
  Parser / Normalizer / RuleEngine / Graph / Retrieval / Ranker / Formatter

知识与记忆层
  Neo4j / Milvus / SQLite FTS5 / PostgreSQL

模型与降级层
  DashScope LLM / Embedding / Rerank
  Lightweight fallback / InMemory fallback
```

简化流程：

```text
用户输入
  |
  v
FastAPI / Streamlit
  |
  v
MedicalAssessmentAgent
  |
  v
MedicalMultiAgentSupervisor
  |
  +-- 初诊 -> AssessmentAgent -> MedicalKAGWorkflow -> SafetyReviewAgent
  |
  +-- 追问 -> MemoryAgent -> RetrievalAgent? -> SynthesisAgent -> SafetyReviewAgent
```

## 5. 核心请求流程

### 5.1 初诊评估流程

用户输入一段体检文本，例如：

```text
男，52岁，血压176/108 mmHg，肌酐128 umol/L，eGFR 54，请判断健康状况。
```

流程如下：

```text
1. FastAPI 接收 /medical/agent/chat 或 /medical/agent/chat/stream
2. ChatHistoryService 构建当前会话上下文
3. MedicalAssessmentAgent 调用 MedicalMultiAgentSupervisor
4. TriageAgent 判断这是初诊
5. AssessmentAgent 启动 MedicalKAGWorkflow
6. Workflow 执行 12 个确定性节点
7. SafetyReviewAgent 做高风险和用药安全检查
8. 返回自然语言回答和结构化 MedicalAssessmentResponse
9. ChatHistoryService 写入用户消息、助手消息、事实记忆和诊断记忆
```

初诊的关键点是：LLM 不能绕过 Workflow。指标解析可以用 LLM 辅助，但风险判定必须经过规则、图谱和排序链路。

### 5.2 追问流程

用户在初诊后继续问：

```text
我的血压风险严不严重？
```

流程如下：

```text
1. TriageAgent 判断这是追问
2. MemoryAgent 读取四层记忆上下文
3. 如果记忆足够回答，则跳过检索
4. 如果需要医学知识补充，RetrievalAgent 调用知识库工具
5. SynthesisAgent 整合记忆、证据、用户问题生成回答
6. SafetyReviewAgent 做医疗边界复核
7. 返回最终回答
```

对于饮食、复查、机制类问题，MemoryAgent 会判断需要补充外部知识，RetrievalAgent 会自动把记忆中的疾病上下文加入检索 query。

例如用户问：

```text
早餐怎么吃？
```

如果诊断记忆中存在“高血压风险”，检索 query 会扩展为类似：

```text
早餐怎么吃？ 高血压 饮食管理 指南
```

### 5.3 流式 SSE 流程

流式接口是：

```text
POST /medical/agent/chat/stream
```

流式事件不是等整张图执行完再批量返回，而是逐 Agent / 逐 Workflow 节点推送：

```text
meta
agent_decision        # TriageAgent 路由
agent_thinking        # AssessmentAgent 或其他 Agent 开始工作
step                  # Workflow 节点开始/完成
tool_call             # RetrievalAgent 调用知识库
tool_result           # 工具返回
agent_synthesizing    # 合成回答
content               # 最终回答分块
result                # 初诊结构化结果
memory_notice         # 事实记忆冲突提示
done
```

同步流式使用 `MedicalMultiAgentSupervisor.iter_events()`。

异步 SSE 使用 `MedicalMultiAgentSupervisor.aiter_events()`，初诊路径会继续复用 `MedicalKAGWorkflow.iter_events_async()`，避免退化为阻塞式同步执行。

## 6. Multi-Agent 设计详解

核心文件：

```text
app/agents/medical_multi_agent.py
```

### 6.1 共享状态

Multi-Agent 使用 `MedicalMultiAgentState` 作为共享状态，核心字段包括：

| 字段 | 作用 |
|------|------|
| `user_input` | 当前用户输入 |
| `session_id` | 会话 ID，用于 MemoryAgent 主动构建上下文 |
| `session_history` | 当前上下文消息 |
| `route` | `assessment` 或 `followup` |
| `memory_text` | 从四层记忆提取出的文本 |
| `needs_retrieval` | 是否需要知识库检索 |
| `retrieval_query` | RetrievalAgent 实际使用的检索 query |
| `evidence_text` | 检索返回的证据文本 |
| `answer` | 当前回答 |
| `safety_notes` | 安全复核提示 |
| `requires_safe_rewrite` | 是否需要安全改写 |
| `structured_response` | 初诊结构化结果 |
| `events` | 已产生的结构化事件 |

### 6.2 Supervisor 图结构

`MedicalMultiAgentSupervisor._build_graph()` 构建 LangGraph 状态图：

```text
triage_agent
  |
  +-- assessment -> assessment_agent -> safety_review_agent -> END
  |
  +-- followup -> memory_agent
                   |
                   +-- retrieve -> retrieval_agent -> synthesis_agent
                   |
                   +-- synthesize -> synthesis_agent
                                      |
                                      v
                                safety_review_agent
                                      |
                                      +-- rewrite -> synthesis_agent
                                      +-- end -> END
```

这个图体现两个设计点：

1. 初诊和追问是两条不同路径。
2. 安全复核不是末尾打标签，而是可以改变控制流，要求 SynthesisAgent 重新生成安全回答。

### 6.3 TriageAgent

职责：判断当前输入是初诊还是追问。

判断依据：

- 输入是否包含体检指标关键词，例如血压、eGFR、肌酐、HbA1c。
- 输入是否包含数值。
- 是否存在 `session_id` 或历史上下文。

结果：

```text
route = "assessment" 或 "followup"
```

### 6.4 AssessmentAgent

职责：运行确定性 KAG Workflow。

初诊时，AssessmentAgent 不直接让 LLM 回答，而是调用：

```text
MedicalKAGWorkflow.run_state()
```

流式时调用：

```text
MedicalKAGWorkflow.iter_events()
MedicalKAGWorkflow.iter_events_async()
```

它会产生完整的 Workflow step 事件，前端可以看到“解析输入、规则判定、图谱检索、证据检索、排序、生成诊断”等过程。

### 6.5 MemoryAgent

职责：判断现有会话记忆是否足够回答。

生产环境中，`container.py` 注入：

```text
chat_history_service.build_context
```

因此 MemoryAgent 可以主动基于 `session_id` 刷新四层记忆。如果没有 `session_id`，则回退使用调用方传入的 `session_history`。

MemoryAgent 的关键决策：

```text
有相关个人记忆，且问题不需要外部知识 -> use_memory
否则 -> need_retrieval
```

### 6.6 RetrievalAgent

职责：补充医学知识库证据。

工具封装在：

```text
app/services/agent_tools.py
```

`MedicalKnowledgeRetrievalTool` 使用 LangChain `StructuredTool` 包装，统一工具调用协议：

```python
StructuredTool.from_function(
    func=self._run,
    name="lookup_medical_knowledge",
    description="检索医学知识库中的指南、科普和证据片段..."
)
```

RetrievalAgent 不只是直接检索用户原文，还会基于 `memory_text` 做疾病上下文扩展，例如把“早餐怎么吃”扩展为“早餐怎么吃 高血压 饮食管理 指南”。

### 6.7 SynthesisAgent

职责：生成最终自然语言回答。

如果配置了模型，使用 `ChatTongyi` 调用 LLM；如果没有配置模型，则使用确定性 fallback，保证本地测试和无 API Key 环境也能运行。

SynthesisAgent 输入：

- 用户问题
- 会话记忆
- 检索证据
- 安全复核要求

输出：

- `answer`

### 6.8 SafetyReviewAgent

职责：医疗安全复核。

检查内容：

- 用户是否询问用药调整。
- 回答中是否出现用药建议。
- 回答中是否出现剂量或频次，例如 `50mg`、`1片`、`次/日`。
- 回答中是否出现疑似药名或治疗词。
- 初诊结果是否要求人工复核。

如果发现具体剂量或疑似药名，且还没有改写过：

```text
requires_safe_rewrite = True
```

LangGraph 条件边会把流程路由回 SynthesisAgent，生成安全改写回答。

## 7. KAG Workflow 设计详解

核心文件：

```text
app/workflows/medical_kag_pipeline.py
```

Workflow 的状态对象是：

```text
InternalAssessmentState
```

定义在：

```text
app/schemas/exam.py
```

它贯穿整个评估链路，主要字段包括：

| 字段 | 含义 |
|------|------|
| `raw_input` | 原始用户输入 |
| `normalized_exam_json` | 归一化后的体检数据 |
| `missing_fields` | 缺失字段 |
| `warnings` | 解析或评估警告 |
| `detected_states` | 规则引擎识别出的异常状态 |
| `risk_candidates` | 图谱召回和排序后的疾病风险 |
| `intervention_candidates` | 干预、科室、复查、禁忌候选 |
| `retrieval_queries` | 证据检索 query |
| `evidence_chunks` | 证据片段 |
| `primary_diagnosis` | 主诊断 |
| `secondary_recommendations` | 二级建议 |
| `response` | 最终 API 响应 |

12 个节点由 `_step_definitions()` 统一维护，同步执行、异步执行和流式事件都复用同一份步骤定义。

### 7.1 输入解析

文件：

```text
app/services/input_parser.py
```

职责：

- 支持自然语言文本和 JSON 输入。
- 优先使用 LLM 结构化抽取。
- LLM 不可用或失败时，用正则回退。
- 输出 `MedicalParseResponse` 和 `NormalizedMedicalExamJSON`。

### 7.2 指标归一化

文件：

```text
app/services/indicator_normalizer.py
```

职责：

- 将“血压”“收缩压”“SBP”等别名映射到统一 code。
- 统一单位。
- 保留原始文本 `source_text`，方便溯源。

### 7.3 规则判定

文件：

```text
app/services/rules.py
app/config/medical_rules.json
```

职责：

- 加载 JSON 医学规则。
- 支持单指标规则。
- 支持组合规则。
- 支持 AND/OR 嵌套。
- 支持性别、年龄等条件。
- 输出 `DetectedState`。

例如血压 176/108 会被判定为高风险血压异常状态。

### 7.4 图谱检索

文件：

```text
app/graph/store.py
app/graph/seed_data.py
```

职责：

- 根据异常状态查询疾病风险。
- 根据疾病风险查询疾病。
- 根据疾病查询干预、科室、复查、禁忌。
- 支持 Neo4j 后端和 InMemory 后端。

Neo4j 不可用时，系统可以自动降级到内存图谱，方便本地开发和测试。

### 7.5 证据查询规划

文件：

```text
app/services/evidence_query_planner.py
```

根据结构化评估状态构造多维 query：

- 用户原问题。
- Top 风险疾病的指南 query。
- 异常指标的风险分层 query。
- 指标聚合 query。

这样比直接用用户原文检索更稳定。

### 7.6 证据检索

文件：

```text
app/retrieval/evidence_store.py
app/retrieval/lexical.py
```

职责：

- Milvus 向量检索。
- SQLite FTS5 词汇检索。
- RRF 融合。
- Rerank。
- 图谱节点重叠度加权。
- 来源权威度加权。
- MMR 去重。

证据片段模型是 `EvidenceChunk`，包含：

- `title`
- `text`
- `linked_node_codes`
- `dense_score`
- `lexical_score`
- `rerank_score`
- `graph_overlap_score`
- `source_authority_score`
- `final_score`

### 7.7 风险排序

文件：

```text
app/retrieval/risk_ranker.py
```

职责：

- 融合图谱支持度和证据支持度。
- 计算 `final_score`。
- 让最终风险排序既考虑医学关系，也考虑证据强度。

### 7.8 诊断格式化

文件：

```text
app/services/diagnosis_formatter.py
```

职责：

- 根据风险候选生成 `PrimaryDiagnosis`。
- 根据干预路径生成 `SecondaryRecommendations`。
- 汇总科室、复查、生活方式、用药方向、禁忌。
- 设置 `human_review_required`。

## 8. 数据模型说明

核心 Pydantic schema 位于：

```text
app/schemas/exam.py
```

### 8.1 输入相关

| 模型 | 作用 |
|------|------|
| `RawPatientProfile` | LLM 或原始解析阶段的患者信息 |
| `RawExamItem` | 原始体检指标 |
| `ExtractedExamPayload` | LLM 抽取结果 |
| `NormalizedMedicalExamJSON` | 系统内部统一体检 JSON |
| `ExamItem` | 归一化后的单个指标 |

### 8.2 评估中间状态

| 模型 | 作用 |
|------|------|
| `DetectedState` | 规则命中的异常状态 |
| `GraphPath` | 图谱路径 |
| `RiskCandidate` | 疾病风险候选 |
| `InterventionCandidate` | 干预建议候选 |
| `RetrievalQuery` | 检索 query |
| `EvidenceChunk` | 证据片段 |
| `InternalAssessmentState` | Workflow 贯穿状态 |

### 8.3 输出相关

| 模型 | 作用 |
|------|------|
| `PrimaryDiagnosis` | 健康状态、紧急程度、主要风险 |
| `DiseaseRecommendation` | 单疾病建议 |
| `SecondaryRecommendations` | 科室、复查、生活方式、用药方向、禁忌 |
| `AssessmentEvidence` | 图谱路径和证据片段 |
| `MedicalAssessmentResponse` | 最终结构化评估响应 |

### 8.4 会话和知识库 API

| 模型 | 作用 |
|------|------|
| `SessionInfo` | 会话信息 |
| `SessionMessage` | 会话消息 |
| `KnowledgeDocument` | 知识库文档元数据 |
| `KnowledgeUploadJob` | 上传任务状态 |
| `RuntimeStatusResponse` | 后端组件状态 |

## 9. 数据库存储设计

ORM 模型位于：

```text
app/db/models.py
```

### 9.1 ChatSession

保存会话级信息：

- `session_id`
- `title`
- `summary_text`
- `conversation_summary`
- `summary_pending_chars`
- `created_at`
- `updated_at`

### 9.2 ConversationMemory

保存原始对话消息：

- `role`
- `content`
- `content_summary`
- `created_at`

对话过长时会被摘要压缩。

### 9.3 UserFactMemory

保存确定性事实：

- 指标值
- 单位
- 病史
- 用药史
- 事实来源
- 更新时间

当新体检数据与旧事实冲突时，系统会显式提示用户。

### 9.4 DiagnosticMemory

保存结构化诊断版本：

- `version_no`
- `is_current`
- `health_status`
- `urgency_level`
- `risk_summary`
- `abnormal_indicator_summary`
- `department_summary`
- `follow_up_summary`
- `lifestyle_summary`
- `medication_summary`
- `evidence_summary`

这支持“比上次严重吗”这类趋势追问。

## 10. API 层说明

核心文件：

```text
app/api/routes/medical.py
```

主要端点：

| 端点 | 作用 |
|------|------|
| `POST /medical/exam/parse` | 只解析输入，不做完整评估 |
| `POST /medical/exam/assess` | 直接运行 KAG Workflow，返回结构化结果 |
| `POST /medical/agent/chat` | Multi-Agent 非流式对话 |
| `POST /medical/agent/chat/stream` | Multi-Agent SSE 流式对话 |
| `POST /medical/kb/upload` | 上传知识库文档 |
| `POST /medical/kb/rebuild` | 重建图谱和证据索引 |
| `GET /medical/runtime/status` | 查看组件状态和降级原因 |
| `GET/POST/DELETE /medical/sessions` | 会话管理 |

`/medical/agent/chat` 的核心行为：

```text
1. 提取请求体
2. 获取或创建 session_id
3. 判断是否初诊
4. 初诊：运行评估，写入事实记忆和诊断记忆
5. 追问：构建上下文，调用 Multi-Agent Supervisor
6. 保存对话消息
```

`/medical/agent/chat/stream` 的核心行为：

```text
1. 先返回 meta 事件
2. 逐步转发 Agent / Workflow 事件
3. 收集最终 content
4. 如有结构化 result，写入事实和诊断记忆
5. 返回 done
```

## 11. 依赖注入与运行时构建

核心文件：

```text
app/services/container.py
```

`get_runtime()` 构建全局单例 `AppRuntime`，包含：

- Settings
- DatabaseManager
- ModelRuntime
- IndicatorNormalizer
- MedicalInputParser
- IndicatorRuleEngine
- GraphStore
- EvidenceStore
- MedicalRiskRanker
- DiagnosisFormatter
- EvidenceQueryPlanner
- MedicalKnowledgeBuilder
- ChatHistoryService
- MedicalKAGWorkflow
- MedicalAssessmentAgent

这样 API 层只需要调用：

```python
runtime = get_runtime()
```

就能拿到所有依赖。

生产环境中，`MedicalAssessmentAgent` 初始化时会注入：

```text
memory_context_builder = chat_history_service.build_context
```

这让 Multi-Agent 中的 MemoryAgent 能主动读取四层记忆。

## 12. 模型与降级策略

核心文件：

```text
app/models/factory.py
```

项目支持三类模型能力：

| 能力 | 首选实现 | 回退 |
|------|----------|------|
| LLM 结构化抽取 | DashScope / ChatTongyi | 正则解析 |
| Embedding | DashScopeEmbeddings | Lightweight hash embedding |
| Rerank | 远程 Rerank API | 跳过远程 rerank |
| Chat 回答 | ChatTongyi | 确定性 fallback 文本 |

图谱和证据库也有降级：

| 组件 | 首选 | 回退 |
|------|------|------|
| 图谱 | Neo4j | InMemoryGraphStore |
| 证据库 | Milvus | InMemoryEvidenceStore |
| 词汇索引 | SQLite FTS5 | fallback lexical logic |

降级原因会通过：

```text
GET /medical/runtime/status
```

暴露给前端或运维人员。

## 13. 知识库构建与上传

相关文件：

```text
app/graph/kb_builder.py
app/services/document_ingestion.py
app/services/knowledge_registry.py
app/services/upload_job_registry.py
```

知识库来源包括：

- `knowledge_sources/` 中的公开医学资料。
- 用户通过 `/medical/kb/upload` 上传的 txt、md、pdf、json、html。

上传处理流程：

```text
1. 接收文件
2. 计算 SHA-256，避免重复入库
3. 根据文件类型抽取文本
4. 文本判空
5. 医学相关性门控
6. 文档分块
7. 推断 linked_node_codes
8. 写入 knowledge_registry
9. 写入 evidence_store
10. 写入 graph_store 的证据节点关联
11. 更新 upload_job_registry
```

医学相关性低的文档会被拒绝或降权，避免知识库污染。

## 14. 技术选型理由

### 14.1 FastAPI

适合构建结构化 API，天然支持异步接口和 OpenAPI 文档。项目同时需要普通 JSON 接口和 SSE 流式接口，FastAPI 能较好覆盖。

### 14.2 Streamlit

用于快速搭建医疗评估工作台，不需要复杂前端工程即可展示对话、流式步骤、知识库管理和会话管理。

### 14.3 LangGraph

项目有两个图：

1. `MedicalKAGWorkflow`：确定性 12 节点评估图。
2. `MedicalMultiAgentSupervisor`：多 Agent 协作图。

LangGraph 适合表达有状态、多节点、带条件边的工作流，尤其适合安全复核触发回路这种控制流。

### 14.4 LangChain

项目没有把所有逻辑都交给 LangChain Agent，而是使用其中稳定的工具抽象：

```text
StructuredTool
```

RetrievalAgent 通过 StructuredTool 调用知识库，保留标准工具协议，又避免黑盒 Agent 循环。

### 14.5 Neo4j

体检评估有天然关系结构：

```text
指标异常 -> 疾病风险 -> 疾病 -> 建议
```

图数据库适合表达多跳路径，也便于解释“为什么推荐这些科室和复查”。

### 14.6 Milvus

医学指南和科普资料适合向量检索。Milvus 用于存储证据块向量，支持语义召回。

### 14.7 SQLite FTS5

医学文本中关键词很重要，例如“eGFR”“尿微量白蛋白”“高血压”。单纯向量召回可能漏掉精确术语，因此引入 FTS5 做词汇召回。

### 14.8 PostgreSQL

会话、事实记忆、诊断版本和对话摘要都是结构化数据，适合关系型数据库持久化。

### 14.9 Pydantic

项目中间状态复杂，Pydantic 用于统一定义输入、状态、输出和 API schema，降低字段不一致风险。

## 15. 创新点设计

### 15.1 KAG-lite 而不是纯 RAG

项目借鉴 KAG 思路，但没有引入完整 KAG 平台，而是在现有 Python 技术栈中实现轻量版：

- schema 约束的医学图谱。
- 图谱路径推理。
- 图谱节点与 chunk 互相关联。
- 图谱信号参与证据排序。

### 15.2 确定性 Workflow + Multi-Agent

初诊由确定性 Workflow 负责，保证医学判断可控；追问由 Multi-Agent 负责，保证交互灵活。两者不是互相替代，而是分工：

```text
Workflow 管医学判断底线
Agent 管上下文决策和交互编排
```

### 15.3 真流式 Agent 事件

早期实现容易出现“先执行完整图，再把事件批量吐出”的假流式。当前实现中，`iter_events()` 和 `aiter_events()` 会逐节点产出事件，前端能真实看到处理过程。

### 15.4 MemoryAgent 接入四层记忆

MemoryAgent 不是简单关键词判断，而是可以通过 `ChatHistoryService.build_context` 主动获取：

- 事实记忆
- 诊断记忆
- 趋势记忆
- 摘要记忆

这让追问能真正基于历史评估结果工作。

### 15.5 记忆感知检索

追问中的短问题往往缺少医学实体，例如“早餐怎么吃”。RetrievalAgent 会从诊断记忆中提取疾病上下文，扩展检索 query，提高召回质量。

### 15.6 安全复核改变控制流

SafetyReviewAgent 不只是附加一句免责声明。它可以通过 LangGraph 条件边触发 SynthesisAgent 重新生成回答。具体剂量、药名和处方化表达会被改写为安全建议。

### 15.7 全链路降级

Neo4j、Milvus、Embedding、Rerank、LLM 都有降级路径。项目可以在缺少外部服务的环境中继续运行主要测试和本地体验。

## 16. 代码结构总览

```text
project_4/
├── app/
│   ├── agents/
│   │   └── medical_multi_agent.py
│   ├── api/routes/
│   │   └── medical.py
│   ├── core/
│   │   └── settings.py
│   ├── db/
│   │   ├── database.py
│   │   └── models.py
│   ├── graph/
│   │   ├── store.py
│   │   ├── seed_data.py
│   │   └── kb_builder.py
│   ├── models/
│   │   └── factory.py
│   ├── retrieval/
│   │   ├── evidence_store.py
│   │   ├── lexical.py
│   │   ├── risk_ranker.py
│   │   └── embeddings.py
│   ├── schemas/
│   │   └── exam.py
│   ├── services/
│   │   ├── agent_tools.py
│   │   ├── chat_history_service.py
│   │   ├── container.py
│   │   ├── diagnosis_formatter.py
│   │   ├── document_ingestion.py
│   │   ├── evidence_query_planner.py
│   │   ├── indicator_normalizer.py
│   │   ├── input_parser.py
│   │   ├── knowledge_registry.py
│   │   ├── medical_agent.py
│   │   ├── rules.py
│   │   └── upload_job_registry.py
│   └── workflows/
│       └── medical_kag_pipeline.py
├── tests/
├── fixtures/
├── knowledge_sources/
├── streamlit_app.py
└── data/
```

## 17. 逐目录说明

### 17.1 `app/agents`

多 Agent 编排层。

核心文件：

```text
medical_multi_agent.py
```

包含：

- `MedicalMultiAgentState`
- `MedicalMultiAgentRunResult`
- `MedicalMultiAgentSupervisor`
- `build_medical_multi_agent_supervisor`
- `extract_memory_text_from_history`

### 17.2 `app/workflows`

确定性 KAG Workflow。

核心文件：

```text
medical_kag_pipeline.py
```

负责从原始输入到结构化评估响应的完整医学评估。

### 17.3 `app/services`

业务服务层。大部分业务逻辑都在这里。

| 文件 | 职责 |
|------|------|
| `medical_agent.py` | Multi-Agent facade，统一同步、异步、流式接口 |
| `agent_tools.py` | LangChain 工具包装 |
| `input_parser.py` | 输入解析 |
| `indicator_normalizer.py` | 指标归一化 |
| `rules.py` | 规则引擎 |
| `diagnosis_formatter.py` | 诊断和建议格式化 |
| `evidence_query_planner.py` | 初诊证据查询规划 |
| `chat_history_service.py` | 会话和四层记忆 |
| `document_ingestion.py` | 文档解析和防污染 |
| `knowledge_registry.py` | 知识库文档元数据 |
| `upload_job_registry.py` | 上传任务状态 |
| `container.py` | 依赖注入 |

### 17.4 `app/retrieval`

检索与排序层。

| 文件 | 职责 |
|------|------|
| `evidence_store.py` | 向量证据库和混合召回 |
| `lexical.py` | SQLite FTS5 词汇检索 |
| `risk_ranker.py` | 风险和证据融合排序 |
| `embeddings.py` | 轻量 embedding fallback |

### 17.5 `app/graph`

医学图谱层。

| 文件 | 职责 |
|------|------|
| `store.py` | Neo4j / InMemory 双后端 |
| `seed_data.py` | 医学种子图谱 |
| `kb_builder.py` | 图谱和证据库构建 |

### 17.6 `app/db`

数据库层。

| 文件 | 职责 |
|------|------|
| `database.py` | SQLAlchemy engine、session、自动建表 |
| `models.py` | 会话、对话、事实、诊断 ORM |

### 17.7 `app/api`

接口层。

核心文件：

```text
app/api/routes/medical.py
```

负责评估、对话、流式事件、知识库和会话 API。

### 17.8 `app/models`

模型工厂层。

核心文件：

```text
app/models/factory.py
```

负责构建：

- Embedding provider
- LLM extractor
- Reranker
- Assistant chat model

并提供 fallback。

### 17.9 `tests`

测试层，当前 41 项通过。

覆盖：

- 解析
- 归一化
- 规则
- KAG Workflow
- 检索
- Multi-Agent
- 记忆
- 知识库构建
- API 状态

## 18. 典型代码调用链

### 18.1 非流式 Agent 对话

```text
app/api/routes/medical.py
  agent_chat()
    -> get_runtime()
    -> chat_history_service.build_context()
    -> medical_agent.chat_assess()
    -> MedicalMultiAgentSupervisor.run()
    -> LangGraph invoke()
    -> final answer
```

### 18.2 流式 Agent 对话

```text
app/api/routes/medical.py
  agent_chat_stream()
    -> medical_agent.stream_assess_async()
    -> MedicalMultiAgentSupervisor.aiter_events()
    -> yield agent/workflow events
    -> collect final content
    -> write memory
```

### 18.3 初诊结构化评估

```text
POST /medical/exam/assess
  -> medical_workflow.run_async()
  -> parse
  -> normalize
  -> rules
  -> graph
  -> evidence
  -> rank
  -> format
  -> MedicalAssessmentResponse
```

### 18.4 知识库上传

```text
POST /medical/kb/upload
  -> save upload job
  -> background task
  -> DocumentChunker
  -> KnowledgeDocumentRegistry
  -> EvidenceStore.add_chunks()
  -> GraphStore.add_evidence_chunks()
```

## 19. 测试与质量保证

运行：

```bash
pytest tests/ -q
```

当前预期：

```text
41 passed
```

重要测试文件：

| 测试文件 | 覆盖内容 |
|----------|----------|
| `test_input_parser.py` | 文本解析 |
| `test_indicator_normalizer.py` | 指标归一化 |
| `test_rules_config.py` | 配置化规则 |
| `test_medical_kag_pipeline.py` | Workflow 主链路 |
| `test_medical_multi_agent.py` | Multi-Agent 路由、流式、记忆、检索扩展、安全改写 |
| `test_evidence_retrieval.py` | 证据检索 |
| `test_retrieval_round2.py` | FTS5、Rerank、MMR 等检索优化 |
| `test_chat_history_service.py` | 会话记忆 |
| `test_document_ingestion.py` | 文档解析和防污染 |
| `test_kb_builder.py` | 知识库构建 |
| `test_round3_final.py` | 最终轮优化回归 |

## 20. 当前实现边界

项目已经具备完整原型能力，但仍有边界：

- 医学图谱是项目级 seed 数据，不是完整医学知识库。
- 远程 LLM、Embedding、Rerank 依赖外部服务，离线时会降级。
- SafetyReviewAgent 使用关键词和正则做安全检测，适合原型，但不是合规级医疗审核。
- 追问检索使用记忆感知 query 扩展，还没有单独抽象成完整 `FollowupQueryPlanner`。
- 前端是 Streamlit 工作台，适合演示和调试，不是生产级 UI。

## 21. 后续可优化方向

建议优先级：

1. 新增 `FollowupQueryPlanner`，让追问检索规划从 Supervisor 中独立出来。
2. 引入更结构化的安全策略，例如药名白名单、剂量规则、禁忌匹配。
3. 增强图谱构建，从文档中自动抽取疾病、指标和干预关系。
4. 增加 token-level streaming，让最终回答也是真 LLM token 流。
5. 增加评测集，持续评估检索、排序和回答质量。
6. 将 Streamlit 前端升级为更完整的产品界面。

## 22. 如何向别人介绍这个项目

可以按下面这段话介绍：

> 我做的是一个面向体检报告场景的 KAG Multi-Agent 辅助评估系统。它不是简单 RAG，而是把体检指标解析、医学阈值规则、Neo4j 知识图谱、多路证据检索和 LangGraph Multi-Agent 编排结合起来。首次评估必须经过确定性 Workflow，保证医学判断可控；多轮追问由 Triage、Memory、Retrieval、Synthesis、SafetyReview 等 Agent 协作完成，能利用诊断记忆和事实记忆，并在涉及用药或剂量时触发安全改写。系统支持 FastAPI、SSE 流式事件、Streamlit 工作台、PostgreSQL 会话记忆、Neo4j/Milvus 降级，以及 41 项自动化测试。

## 23. 总结

这个项目的核心价值不是“能回答医疗问题”，而是把医疗体检评估拆成可控、可解释、可测试的工程链路：

```text
确定性规则保证底线
知识图谱提供推理路径
混合检索提供证据支撑
四层记忆支持多轮追问
Multi-Agent 负责决策编排
SafetyReviewAgent 负责医疗边界
```

因此，它更适合作为医疗 Agent / KAG / RAG 工程能力展示项目，而不是普通聊天机器人 demo。

