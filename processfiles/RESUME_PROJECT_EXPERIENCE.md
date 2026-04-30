# 简历项目经历

---

## 项目名称

**医疗 KAG Agent 工作台 — 基于知识图谱与混合检索的多记忆体检辅助评估系统**

---

## 项目简介

本项目构建了一个面向体检场景的智能辅助评估 Agent 系统，采用「规则引擎 + 知识图谱 + 混合检索 + 显式 ReAct Agent + 多层记忆」的混合架构。用户以自然语言提交体检报告后，系统通过确定性安全流水线完成指标解析、阈值判定、图谱风险路径推理、混合证据检索与排序融合，产出结构化诊断评估与干预建议。在多轮对话中，自研 ReAct Agent 循环接管追问交互，结合诊断趋势记忆、用户事实记忆和知识库检索工具，在安全边界内灵活作答。

系统覆盖完整的知识库管理链路（多格式文档摄入、去重、分块、图谱节点映射、后台异步入库）、三层持久化后端（Neo4j 图谱 + Milvus 向量 + PostgreSQL 会话记忆）、流式 SSE 交互、以及组件级健康检查与自动降级机制。36 项回归测试覆盖全链路。

---

## 主要工作

**1. Agent 架构设计与显式 ReAct 循环实现**

- 设计并实现了「LangGraph 确定性流水线 + 自研 ReAct Agent」双层 Agent 架构。底层 12 节点流水线保证医疗场景的评估安全边界（不可被 LLM 绕过），上层 ReAct Agent 循环接管多轮追问，通过分层决策策略（记忆优先 → 知识检索 → 综合回答）控制工具调用行为
- ReAct 循环内置最大迭代次数限制、工具参数 hash 循环检测、结构化事件流输出（thinking → decision → tool_call → tool_result → synthesizing → final_answer），全链路对前端流式可见
- 实现初诊/追问路径自动分离：初诊强制走确定性流水线并短路 Agent 决策，追问根据是否有相关记忆和是否需要外部知识分层选择动作

**2. 知识图谱与混合检索系统**

- 设计 Neo4j 医学知识图谱 Schema（IndicatorState → DiseaseRisk → Disease → Intervention/FollowUp/Department/Contraindication/MedicationDirection），使用 Cypher 实现多跳关系查询，封装抽象层支持 Neo4j/InMemory 双后端
- 实现混合证据检索排序算法：稠密向量检索（DashScope Embedding + Milvus HNSW，复用 Milvus native distance）、词汇检索（SQLite FTS5 持久化索引，含中文 fallback）、图谱节点重叠度评分、来源权威度评分、远程 Rerank 语义重排（前置融合，权重 45%）、MMR 多样性去重（裁剪至 15 候选）、加权融合公式

**3. 多层记忆与上下文工程**

- 基于 PostgreSQL + SQLAlchemy 设计四层记忆模型：**用户事实记忆**（指标值 + 病史 + 用药史，含新旧冲突检测与自动更新提示）、**诊断记忆对象**（结构化诊断快照，版本号 + is_current 标记，支持跨版本趋势 diff）、**对话记忆**（上下文窗口自动裁剪 + 字符截断）、**LLM 摘要记忆**（累计字符阈值 2000 触发，避免频繁调用）
- Agent 上下文构建时对用户事实记忆做向量化语义检索（embedding → Milvus 独立 collection），按当前追问相关性动态召回最相关的 5 条事实，防止 prompt 膨胀
- 诊断趋势注入：追问时自动对比最近两版诊断结果，显式标注健康状态和异常指标变化，支撑「和上次相比怎么样了」的趋势类追问

**4. 规则引擎与安全边界设计**

- 将医学阈值规则从硬编码 if/elif 重构为 JSON 配置驱动，支持 gte/gt/lte/lt/eq/in/not_in 七种操作符和嵌套 AND/OR 组合逻辑，覆盖性别差异阈值（肌酐男/女不同参考范围）、年龄分层风险（老年 + 高血压组合）等场景
- 流水线加入条件短路：无异常指标时跳过图谱和证据检索直出健康结论（延迟从 ~350ms 降至 ~50ms），图谱未命中时跳过干预扩展，证据为空且图谱分数低时强制标记人工复核
- Agent 系统提示词 + 工具描述双重约束：严禁凭空捏造诊断结果，不确定性结论显式提示「需医生复核」

**5. 知识库管理与文档摄入链路**

- 实现多格式文档解析（TXT/MD/PDF/JSON/HTML），滑动窗口分块（chunk_size=500, overlap=80），基于 content_hash 的持久化去重，自动推断 chunk 与图谱节点的链接关系
- FastAPI BackgroundTasks 异步上传处理，前端轮询任务状态；已就绪后端走增量写入避免全量 rebuild

**6. API 异步化与性能优化**

- 将流水线关键路径异步化：图谱风险检索和证据检索无数据依赖，通过 asyncio.gather 并行执行，wall-clock 延迟下降约 35%
- SSE 流式接口改为 async generator，流水线每步完成后立即推送事件，前端实时看到处理进度
- 图存储 Neo4j driver 改为 AsyncGraphDatabase；Rerank HTTP 调用改为 httpx AsyncClient

**7. 系统韧性、可观测性与测试**

- 实现 Neo4j/Milvus 依赖不可达时自动降级至 InMemory 后端，降级原因通过 `/health` 和 `/runtime/status` 暴露
- 流水线步骤级耗时追踪 + 关键指标计数 + Agent 工具调用 trace + 组件级健康检查（graph/evidence/postgresql/embedding/extractor/reranker 六项状态 + 延迟）
- 全链路 12+ 降级点结构化日志（logging 模块），替换裸 except:pass
- 编写 36 项 pytest 回归测试，覆盖输入解析、规则判定、检索排序、Agent 循环、会话记忆、诊断版本化、异步一致性

---

## 技术栈

**Agent & 编排**：自研 ReAct Agent Loop（think→tool_call→observe→synthesize）、LangGraph StateGraph 确定性流水线

**后端框架**：FastAPI、SSE Async Streaming、Pydantic v2 数据建模

**图数据库**：Neo4j（Cypher 查询、约束/索引管理、AsyncGraphDatabase）

**向量数据库**：Milvus（HNSW 索引、native distance 复用、动态 schema）

**关系数据库**：PostgreSQL + SQLAlchemy ORM（会话/事实/诊断/对话四表、版本化记忆、自动 schema 迁移）

**全文检索**：SQLite FTS5（持久化词汇索引、bm25 排序、中文 fallback LIKE 匹配）

**LLM & 模型**：DashScope API（通义千问 qwen3-max LLM、text-embedding-v4 Embedding、qwen3-vl-rerank Rerank）

**前端**：Streamlit（左右分栏工作台、流式聊天、会话管理、知识库管理、运行时状态监控）

**基础设施**：Python 3.10+、asyncio 协程、hashlib 去重、pytest 36 项回归测试

---

## 项目亮点（面试可展开）

- **Agent 架构双路径设计**：初诊走确定性 Workflow 保证安全，追问走 ReAct 循环保证灵活，两条路径在同一个 Agent 入口自动分流，而非「一个 API 一条路」的割裂实现
- **Rerank 前置的排序链路**：让语义重排器在结构化偏置之前工作（45% 权重），优于常见的「先融合后微调」，排序结果更贴近用户真实意图
- **记忆系统的工程化分层**：不是「把历史消息全塞进 prompt」，而是区分事实记忆（可 diff、可冲突检测）、诊断记忆（可版本化、可趋势对比）、对话记忆（可摘要裁剪），每类记忆各自的生命周期和触发策略独立
- **检索性能的阶梯式优化**：Milvus distance 复用（消除 Python 侧重复计算）→ SQLite FTS5 替代内存 BM25 → MMR 裁剪 → Rerank 前置，四步渐进优化而非一步到位
- **安全边界的多层设计**：规则引擎（确定性阈值）→ 图谱（结构化推理）→ Workflow（不可绕过）→ Agent 提示词约束（禁止捏造）→ 人工复核标记，五层防线而非单点依赖 LLM
