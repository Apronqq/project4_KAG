基于 KAG 的 Multi-Agent 体检辅助评估系统

项目简介
面向体检数据评估场景，构建基于 LangGraph Supervisor 的 KAG Multi-Agent 智能体系统。Supervisor 协调 TriageAgent、AssessmentAgent、MemoryAgent、RetrievalAgent、SynthesisAgent、SafetyReviewAgent 六类角色：首次评估强制进入确定性 KAG Workflow，完成指标解析、规则判定、Neo4j 图谱风险推理与结构化诊断生成；多轮追问优先复用诊断版本记忆和事实记忆，不足时通过 LangChain StructuredTool 调用知识库检索，再由合成与安全复核 Agent 输出回答。系统强调医疗场景下的安全边界、可解释推理链路与多 Agent 协作能力。

技术栈
FastAPI、LangGraph、ReAct、Neo4j、Milvus、PostgreSQL、SQLite FTS5、Rerank、SSE、Streamlit

核心技术实现
1. 设计 LangGraph Multi-Agent 编排：TriageAgent 负责路由，AssessmentAgent 强制执行确定性 Workflow 保障安全，Memory/Retrieval/Synthesis/SafetyReview Agent 协作处理追问，避免 LLM 直接绕过医学规则生成诊断；
2. 将追问链路从单 Agent 循环升级为 Supervisor 状态图，按「记忆判断→必要时工具检索→综合回答→安全复核」进行分层决策，结构化流式事件实现路由、工具调用、检索到回答全过程可观测；
3. 构建 Neo4j 医学 KAG 图谱，建模「异常状态→疾病风险→疾病→干预/科室/复查/禁忌」多跳链路，以图谱节点重叠度作为检索排序独立信号；
4. 实现图谱感知混合检索：Milvus 向量 + SQLite FTS5 词汇双路 RRF 召回 → Rerank 前置精排 → 图谱信号加权 → MMR 去重，Ablation 验证全链路相对纯向量基线 MRR 提升 27.5%；
5. 四层记忆机制：事实记忆冲突检测与语义向量召回防 prompt 膨胀、诊断结果版本化支持趋势对比、对话上下文自动裁剪、摘要 LLM 阈值触发（2000 字符）；
6. 医学场景安全纵深：JSON 配置化阈值规则（性别/年龄分层 + AND/OR 组合）、图谱关系约束检索方向、Workflow 强制编排防绕过、Agent 提示词双重约束、高风险自动标注人工复核。
