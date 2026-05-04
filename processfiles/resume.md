基于 KAG 的 Multi-Agent 体检辅助评估系统

项目简介
面向体检数据评估场景，构建基于 LangGraph Supervisor 的 KAG Multi-Agent 智能体系统。Supervisor 协调 TriageAgent、AssessmentAgent、MemoryAgent、RetrievalAgent、SynthesisAgent、SafetyReviewAgent 六类角色：首次评估强制进入确定性 KAG Workflow，完成体检指标解析、规则判定、Neo4j 图谱风险推理与结构化诊断生成；多轮追问优先复用诊断版本记忆和事实记忆，不足时通过 LangChain StructuredTool 调用知识库检索，再由合成与安全复核 Agent 输出回答。系统强调医疗场景下的安全边界、可解释推理链路与多 Agent 协作能力。

技术栈
FastAPI、LangGraph、ReAct、Neo4j、Milvus、PostgreSQL、SQLite FTS5、Rerank、SSE、Streamlit

核心技术实现
1. 设计 LangGraph Multi-Agent 编排：TriageAgent 负责路由，AssessmentAgent 强制执行「解析指标→规则判定→图谱推理→证据补充→诊断生成」确定性 Workflow，Memory/Retrieval/Synthesis/SafetyReview Agent 协作处理追问，避免 LLM 直接绕过医学规则输出诊断；
2. 将追问链路从单 Agent 循环升级为 Supervisor 状态图，按「记忆判断→必要时工具检索→综合回答→安全复核」进行分层决策，并通过结构化流式事件暴露路由、工具调用、证据整合和最终回答全过程；
3. 构建 Neo4j 医学 KAG 图谱，建模「异常状态→疾病风险→疾病→干预/科室/复查/禁忌」多跳链路，将体检异常从单点指标判断扩展为关系推理和路径可解释；
4. 实现图谱感知检索增强：基于图谱节点生成检索约束，结合 Milvus 向量召回、SQLite FTS5 词汇索引、Rerank 精排、图谱信号加权与 MMR 去重，提升证据与当前风险路径的一致性；
5. 设计四层记忆机制：事实记忆做新旧指标冲突检测，诊断记忆做版本化趋势对比，对话记忆做上下文裁剪，摘要记忆按字符阈值触发压缩，支撑多轮追问中的上下文复用；
6. 落地医疗场景五层安全防线：配置化阈值规则兜底、图谱关系约束推理方向、Workflow 强制编排防绕过、Agent 提示词限制诊断边界、高风险结果自动标注人工复核。
