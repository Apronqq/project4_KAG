# 医疗 KAG Multi-Agent 工作台 — 面试自问自答全套

> 覆盖项目全貌，按面试官的追问逻辑组织：项目概述 → 架构决策 → Agent 设计 → 图谱与检索 → 记忆系统 → 
> 安全边界 → 性能韧性 → 扩展优化。每个问题独立可答，串联起来是完整叙事。

---

## 一、项目概述与动机

### Q1：一句话介绍你的项目

**A**：一个面向体检报告解读的 KAG Multi-Agent 辅助评估系统。用户提交体检文本后，系统通过确定性规则引擎判断指标异常、Neo4j 知识图谱推理疾病风险、混合检索补充医学证据，产出结构化评估报告。多轮追问由 LangGraph 编排的六个专职 Agent 协作处理，结合四层个体化记忆作答，并在检出用药剂量风险时自动拦截改写。

### Q2：和普通的 RAG 医疗问答有什么区别？

**A**：三个本质区别。第一，**判断不由 LLM 做**。体检指标的阈值判定是确定性规则引擎执行的——收缩压 ≥ 160 就是 2 级高血压，LLM 不能「觉得不严重」而跳过。纯 RAG 把判断权交给了 LLM。

第二，**推理有知识图谱约束**。异常状态到疾病风险到干预建议是 Neo4j 多跳路径推理的，不是 LLM 自由联想。用户可以看到「收缩压显著升高 → 高血压风险 → 高血压 → 心内科/ACEI 药物评估/低盐饮食」的完整推理链。

第三，**系统记得住用户的健康数据**。不是把历史聊天记录塞进 prompt，而是用四层结构化记忆管理——事实记忆（指标值）做新旧冲突检测、诊断记忆做版本化趋势对比。同一用户第二次提交体检报告，系统主动告知「收缩压从 176 降至 148」。

### Q3：为什么选体检场景而不是更泛用的医疗问答？

**A**：体检数据有三个特点让技术方案更可控——输入结构化为指标名+数值+单位，适合做确定性解析；风险判断依赖公开指南的阈值标准，可以规则化；推理链路是「指标异常→风险→疾病→干预」的固定路径，适合图谱建模。泛用医疗问答这三个条件都不满足，不确定性太高，不适合作为原型系统的第一个验证场景。

---

## 二、架构与设计决策

### Q4：系统整体架构是怎样的？

**A**：两个 LangGraph 图 + 三套存储。

**图 A** 是 MedicalKAGWorkflow，12 节点确定性流水线，处理初诊体检评估。LLM 只在最后一步把结构化 JSON 翻译成自然语言，中间的规则判定、图谱检索、证据排序全部是代码逻辑。

**图 B** 是 MedicalMultiAgentSupervisor，6 Agent 条件分支协作，处理多轮追问。TriageAgent（分诊）→ AssessmentAgent 或 MemoryAgent（记忆判断）→ RetrievalAgent（按需检索）→ SynthesisAgent（回答合成）→ SafetyReviewAgent（安全复核）。

三套存储：Neo4j 存知识图谱关系，Milvus 存证据块向量和 SQLite FTS5 做词汇索引，PostgreSQL 存会话和四层记忆。

### Q5：为什么是两个独立的 LangGraph 图，而不是一个大图？

**A**：初诊和追问的**安全假设不同**。初诊要求零幻觉容忍——12 个节点的每一步都必须可追溯、可审计。追问要求上下文灵活应答——不能每次都重跑一遍完整流水线。如果合并在一个图里，需要大量条件边区分路径，编排复杂度超过拆分两个图。两个图各自独立，由 Supervisor 在入口处做一次路由决策选择执行哪条路径，各自的安全边界和优化策略也独立设计。

### Q6：你提到借鉴了 KAG 方法论但没有引入完整平台。KAG 在你的系统里具体体现在哪？

**A**：三个环节。第一，**schema-constrained 图谱构建**——六种节点类型和五种关系类型的 schema 是在设计阶段就确定的，不是让系统从文档中自动归纳的。第二，**knowledge/chunk 互索引**——每个证据 chunk 入库时被自动打上 linked_node_codes（关联了哪些图谱节点），检索排序时 chunk 的 linked_node_codes 和 query 触发的图谱节点做交集计算，产生 graph_overlap_score，用确定性知识约束概率性检索。第三，**graph-guided reasoning**——初诊的风险推理不靠检索，而是沿着 Cypher 的多跳路径（STATE_IMPLIES_RISK → RISK_RELATED_DISEASE → DISEASE_RECOMMENDS_*）做确定性的链式推导。

### Q7：你们的 embedding 和 rerank 都是调用外部 API，如果某天不能访问怎么办？

**A**：全链路降级。DashScope Embedding 不可用 → 回退到轻量 hash embedding（本地计算，无需 API）。Rerank API 不可用 → `_apply_remote_rerank` 中 try-catch 后跳过 Rerank 步骤，`_finalize_scores` 中检测到 `has_rerank=False` 后自动用 `fusion_norm` 替代 `rerank_norm`，排序降级到四信号融合。LLM 不可用 → `_compose_followup_answer` 和 `_compose_answer` 回退到确定性模板拼接。Neo4j 不可达 → `build_graph_store` 中 ping 失败后自动降级 InMemoryGraphStore。Milvus 同理。每层都有独立的降级策略，降级原因通过 /medical/runtime/status 暴露。系统在没有外部服务的情况下也能在内存模式下完成核心功能。

---

## 三、Multi-Agent 设计

### Q8：你用了六个 Agent，它们是怎么协作的？

**A**：不是让六个 LLM 互相聊天。而是六个**职责明确的执行单元**，通过 LangGraph 的共享状态传递结构化中间产物。TriageAgent 是规则判断（关键词+数值+意图词），不是 LLM。AssessmentAgent 是 Workflow 调度器，不是 LLM。MemoryAgent 是记忆查询器，不是 LLM——它从 PostgreSQL 拉数据、做关键词匹配、输出 needs_retrieval 布尔值。RetrievalAgent 是工具调用器。只有 SynthesisAgent 在需要复杂文本整合时调用 LLM。SafetyReviewAgent 是正则+规则。

Agent 之间没有对话协商。MemoryAgent 产出的 needs_retrieval 是布尔值，RetrievalAgent 读到它决定要不要调工具。SynthesisAgent 读 memory_text 和 evidence_text 做回答合成。全程用 state dict 传数据，不走 LLM 推理。

### Q9：初诊和追问的分流逻辑是什么？

**A**：调用同一个判定函数 `_looks_like_initial_assessment`。三个条件必须同时成立才判为初诊：输入包含体检指标关键词（血压/eGFR/肌酐等）、包含数值或明确评估意图词（请判断/评估一下/体检报告）。「我的血压怎么样了」有关键词但无数值且无意图词 → 判定为追问。这个设计很关键——如果只有关键词匹配，大量个人追问会被误判为初诊。

另外 API 层和 TriageAgent 都调用了同一个判定函数。API 层需要它来决定写不写事实记忆和诊断记忆（初诊才写），TriageAgent 需要它来决定走 assessment_agent 还是 memory_agent。两处判断用的是同一个函数，但各自为不同的下游行为做决策。

### Q10：你提到 MemoryAgent 可以注入 ChatHistoryService。为什么要用注入而不是直接在 Agent 里 import？

**A**：两个原因。第一，测试不需要启动 PostgreSQL。测试传入一个假的 context builder 返回预设的 session_history，Agent 逻辑完全不受影响。第二，Agent 的单元测试不应该依赖数据库。如果把 ChatHistoryService 硬编码在 Agent 里，测试就必须起一个真实的 PostgreSQL 实例。注入让 MemoryAgent 的逻辑和「数据从哪来」解耦。

### Q11：为什么不直接用 LangChain 的 create_agent 或者 CrewAI/AutoGen 这种多 Agent 框架？

**A**：`create_agent` 的最初版本尝试过——它是黑盒，看不到 Agent 为什么选了工具 A 而不是 B，出了错无法追溯。自研循环的每步都产出结构化事件（agent_thinking → agent_decision → tool_call → tool_result），全链路前端可见。

CrewAI/AutoGen 这类框架让多个 LLM Agent 互相对话协商。在医疗场景有两个问题：第一是确定性管道的产出被 LLM Agent 重新解释了一次，引入不确定性——Workflow 已经判断了 hypertension，ReAct Agent 又去「质疑」这个判断怎么办。第二是每个 Agent 间的对话需要 prompt+LLM 推理+解析，三个 Agent 一轮对话就是 3 次 LLM 调用，体检追问的延迟要求在 1-2 秒内，这个开销不可接受。我的 Agent 间走的是共享 state dict 传结构化中间产物，不走 LLM。

---

## 四、知识图谱（Neo4j）

### Q12：Neo4j 里存了什么？图谱结构是怎样的？

**A**：六种节点类型，五类关系。

节点：IndicatorState（异常状态，如 SBP_high_stage2）→ DiseaseRisk（疾病风险，如 hypertension_risk）→ Disease（疾病，如 hypertension）→ 五类建议叶子节点（Intervention/Department/FollowUpTest/Contraindication/MedicationDirection）。

关系：STATE_IMPLIES_RISK（状态→风险）、RISK_RELATED_DISEASE（风险→疾病）、DISEASE_RECOMMENDS_* 五条（疾病→各类建议）。外加一条 NODE_LINKED_CHUNK（Risk/Disease→EvidenceChunk），是唯一跨推理层和证据层的关系，把上传文档的 chunk 挂载到图谱节点上。

Seed 数据约 20 条异常状态映射、6 个疾病、每个疾病 5 个维度的建议。全部来自公开医学指南的手工整理。上传的外部文档不能修改图谱结构——只能通过 NODE_LINKED_CHUNK 绑定到已有节点。证据可以自动积累，知识结构必须人工校验。

### Q13：文档上传时 linked_node_codes 是怎么打上去的？Neo4j 参与了吗？

**A**：不参与。linked_node_codes 是在 `DocumentChunker._infer_linked_node_codes()` 中推断的，纯 Python 字符串匹配。三层匹配：chunk 文本中出现 "收缩压" → 映射到 hypertension_risk+hypertension（别名匹配）；chunk 中出现 "SBP_high_stage2" → 直接追加上该状态码及其关联的 risk_code 和 disease_code（状态码匹配）；chunk 中出现 "心内科""限盐""ACEI" → 追加对应的 disease_code（干预词匹配）。

Neo4j 在这之后的角色是把**已经推断好**的 linked_node_codes 落地为 NODE_LINKED_CHUNK 关系——用 Cypher 的 UNWIND+OPTIONAL MATCH+FOREACH+CASE WHEN 尝试把 chunk 绑定到已有的 Risk/Disease 节点上。linked_node_codes 中的某个 code 如果在当前图谱中找不到对应节点，这一条绑定被跳过，不影响其他绑定。

### Q14：图谱信号在你的检索排序中怎么用的？

**A**：图谱的 Risk/Disease 节点码作为**排序信号**而非**检索源**。检索源仍然是 EvidenceChunk（向量+词汇），但每个 chunk 被召回后，用它的 linked_node_codes 和当前 query 触发的图谱路径节点码做交集计算——交集比例越高，graph_overlap_score 越高，占融合权重 20%。

举例：chunk A 的 linked_node_codes 是 `[hypertension_risk, hypertension]`，当前 query 的图谱路径节点码集合是 `{hypertension_risk, hypertension, ckd_risk, ckd}`，交集 2/总数 2 = 1.0 满分。chunk B 的 linked_node_codes 是 `[diabetes_risk, type2_diabetes]`，交集 0/总数 2 = 0.0。前者在排序中得到图谱信号加持。

这个信号解决了纯语义检索的一个盲区——语义模型只判断「文本和 query 在向量空间的距离」，但判断不了「文本讲的是否恰是当前推理链路涉及的医学概念」。一段糖尿病用药的文字在语义上可能和高血压用药产生相似的向量表示，但图谱信号会给它 0 分。

### Q15：为什么自己建而不是直接用 OpenSPG？

**A**：目标不同。OpenSPG 解决的是「从非结构化文档中自动归纳 schema、做实体链接、构建大规模知识图谱」的问题。我的场景是「schema 已经手工设计好、seed 数据量小而精、图谱的每条边都需要确定性保证」。Neo4j 直接建模十几条关系链路，比引入完整平台做适配的成本低得多。另外 OpenSPG 的自动实体链接和关系抽取能力在医疗场景下有安全风险——如果它从文档中自动抽取出「肌酐升高 → 建议服用某保健品」并加入图谱，后续所有评估都会沿着这条未经校验的边推理。确定性比覆盖率更重要。

---

## 五、混合检索与排序

### Q16：检索链路是怎么设计的？

**A**：初诊五步，逐层递进。

**Step 1 — 多查询规划**：不止用用户原话检索，而是从评估上下文生成 8 个维度查询——用户原问题、前 3 个高风险疾病的指南查询、随访查询、前 4 个异常指标的风险分层查询、指标聚合查询。每个查询独立召回，多查询的 RRF 累加效应让真正相关的 chunk 在不同检索视角下都能获得排名贡献。

**Step 2 — 双路 RRF 粗排**：稠密向量（Milvus HNSW 复用 native distance）+ 词汇倒排（SQLite FTS5 BM25）。RRF 不关心各路分数的绝对大小，只看排名——`Σ 1/(60+rank_in_each_path)`。9 个查询 × 2 条路径 = 18 次贡献，排名稳定的 chunk 自然浮到前面。

**Step 3 — Rerank 前置精排**：取粗排 top 20 送 DashScope Rerank API。Rerank 是五信号中最强的单信号（深度语义模型对 query-document 匹配的理解远优于向量点积和 BM25），给它 45% 的最高权重。

**Step 4 — 多信号融合**：Rerank(45%) + 图谱节点重叠度(20%) + 词汇匹配(15%) + 稠密向量(10%) + 来源权威度(10%)。

**Step 5 — 质量门控 + MMR 去重**：graph_overlap_score > 0 直接放行（有图谱锚定），或 final_score ≥ 0.75（无锚定但质量高）。然后对 top 15 候选项做 MMR(λ=0.75)，保证返回的 5 条结果不是同一篇文章的不同分块。

### Q17：Rerank 为什么放在 fusion 之前而不是之后？

**A**：传统做法 Rerank 放在 fusion 之后做微调，权重只有 20-25%。问题在于——前置的 dense/lexical/graph/authority 四个信号已经把排名基本锁定了，Rerank 的语义深度理解在最后一步能起的作用很小。

我的做法是 dense+lexical 粗排 → Rerank 精排 → fusion 融入 graph+authority 偏置。Rerank 的分数直接参与 weighted sum 并拿 45% 的最高权重。**语义模型主导排序，结构化信号修正偏差**——这就是逻辑。Ablation 验证 Rerank 贡献了 +11% MRR。

### Q18：追问检索和初诊检索有差距吗？

**A**：检索引擎是同一个 `evidence_store.search()`。两个差距：第一，初诊用 8 个查询做 RRF，追问只用一个查询（即使做了记忆感知查询扩展）——RRF 多查询的累积效应丢失。第二，初诊的 node_codes 来自 Workflow 节点 4 的 Cypher 查询结果，四五个节点码全部精确；追问路径的 node_codes 目前传的是空列表——graph_overlap_score 永远为 0。这两个差距不在检索引擎本身，在于查询规划和图谱信号的输入不足。优化方向：追问路径用多查询规划 + 从记忆文本反推 node_codes。

---

## 六、记忆系统

### Q19：你的记忆系统为什么分了四层？

**A**：按信息的**确定性**分层管理，而不是把所有历史聊天记录塞进 prompt。确定性高的记忆用规则管理，确定性低的用阈值和裁剪控制。

**事实记忆**（确定性高）：指标值和病史用药，每次新体检数据全量覆盖写入，新旧对比做冲突检测。追问时用语义向量做相关召回 top 5，防止 prompt 膨胀。**诊断记忆**（确定性高）：每次初诊评估的完整结构化结果做版本化存储。追问时注入最近两版做趋势对比——「收缩压从 176 降至 148，风险从高风险变为需重点复查」。**对话记忆**（确定性低）：双字段原文+摘要，追问时用截断版注入上下文。**摘要记忆**（确定性低）：累计字符阈值 2000 触发 LLM 压缩，否则确定性拼接。

本质区别：传统方案把 LLM 当作记忆管理器，让它自己从消息流中提取相关信息；我的方案把记忆管理作为独立工程子系统，LLM 只消费整理好的记忆输入。

### Q20：上下文过长时怎么处理？保证系统 prompt 不被截掉？

**A**：`_clip_total_history` 从新到旧累积，新消息优先保留。system prompt（事实记忆+诊断记忆）被放在数组最前面，倒序遍历时最后处理——当 total 已经接近上限时，前面的 system prompt 因为长度较短（100-300 字符），通常能在最后几轮挤进去。这是隐含保证：确定性记忆永远不会因为对话太长而被优先截掉。

三级长度控制：单条 500 字符截断 + 只取最近 12 条 + 总上限 3200 字符。每级在 `build_context` 中独立控制。

---

## 七、安全边界

### Q21：你怎么保证 LLM 不会在回答中给出危险的用药建议？

**A**：分了轻度和严重两级处理。

SafetyReviewAgent 在每次回答返回到用户之前检查五个维度：高风险标记、用户是否在问用药、回答中是否出现了用药建议关键词、是否出现了具体剂量（`\d+mg`、`\d+片`、`次/日`）、是否出现了疑似药名（X沙坦/X他汀/X洛尔+片/胶囊）。

轻度（只有用药关键词但没有剂量药名）：回答末尾追加提示「涉及用药调整时，请结合医生意见处理」。严重（检测到具体剂量或药名）**直接拦截原回答**——设置 requires_safe_rewrite=True，通过 LangGraph 条件边把流程路由回 SynthesisAgent 用模板替换原回答。替换不调 LLM，用纯确定性模板拼接：「涉及药物名称、剂量、停药、换药或加减药时，应以医生面诊和处方为准」。

安全改写只触发一次（safety_revision_count 防无限循环），因为模板回答本身不含剂量。

### Q22：为什么要在 prompt 里已经有安全约束的情况下再做后处理检查？

**A**：prompt 约束和正则检查是互补的。prompt 中写了「不要替代医生处方」「提示结合医生复核」。LLM 可能遵守了——在回答末尾追加了一句「以上仅供参考，请咨询医生」。但同时在前文里写了「可以考虑减至 40mg，每日一次」。用户看到前半句就行动了，后面的免责声明完全被忽略。

正则检查是**不依赖 LLM 自觉的硬性校验**。它不读语义，不判断意图，只做纯模式匹配——有 `\d+mg` 就触发。这和 LLM 的 self-moderation 有本质区别：LLM 可以说服自己「我加了免责声明我是合规的」，正则不会。

### Q23：上传了无关文档怎么办？

**A**：六层纵深防御。文件类型白名单（只收 txt/md/pdf/json/html）→ SHA-256 去重 → 文本抽取判空（扫描版 PDF 无文本直接结束）→ 医学相关性门控（linked_node_codes 数量+关键词密度+概念覆盖率三维评分，<0.15 直接拒绝入库）→ 来源权威度降权（未验证文档在排序中 authority 分仅为 guideline 的 30%）→ 质量门控+图谱信号压制（无 linked_node_codes 的 chunk 在排序中 graph_overlap_score=0，很难进入 top 5）。

---

## 八、性能与韧性

### Q24：做了哪些性能优化？

**A**：流水线层面——条件短路（无异常指标时跳过图谱+证据+排序，延迟从 ~350ms 降至 ~50ms）+ 异步并行（证据检索和干预路径扩展无依赖，`asyncio.gather` 并行，墙钟时间下降约 37%）。检索层面——Milvus native distance 复用（消除 Python 侧重复 embedding 点积）+ SQLite FTS5 替代内存 BM25（持久化+大数量级）+ MMR 候选截断到 15 + Rerank 前置权重 45%。记忆层面——摘要 LLM 阈值触发（2000 字符才调用，不是每轮调）+ 事实记忆向量化 top 5 召回（防 prompt 膨胀）。

### Q25：你的混合检索 MRR 0.80，这个分数怎么来的？各模块贡献多少？

**A**：Ablation study 在 50 个场景的标注集上跑出来的。纯向量基线 MRR 0.58 → +词汇检索 RRF → 0.65 → +图谱信号 → 0.71 → +Rerank 前置 → 0.78 → +MMR → 0.80。全链路提升 27.5%。各模块的贡献是叠加关系，验证了每一层优化都有独立的增量收益。

---

## 九、扩展与优化

### Q26：如果图谱没有覆盖某个指标（比如尿酸），系统怎么办？

**A**：当前会判为健康——规则引擎不识别、图谱无映射 → detected_states 空 → health_status="healthy"，但尿酸 520 显然不正常。优化方向：在规则引擎中增加未知指标检测——indicator_code 不在 INDICATOR_ALIASES 中的指标标记为 unknown_indicator。在评估报告中单独列出未知指标及其参考范围，并附上通用说明（该指标偏高通常与什么有关、建议咨询什么科室），严格标注「非个体诊断，请咨询医生」。LLM 可以补充科普信息但不能做疾病判断。

### Q27：如果让你继续迭代这个项目，优先级最高的是什么？

**A**：三个方向。第一，追问检索质量对齐初诊——用多查询规划 + 从诊断记忆反推 node_codes + 独立的 FollowupQueryPlanner，把追问的单查询 RRF 提升到多查询 RRF。第二，未知指标的优雅降级——规则引擎不能只返回空，要在图谱盲区给出有意义的信息。第三，LLM token-level streaming——当前回答是生成完后 chunk 推送，改为真正的 token streaming 对用户体验是质变。

### Q28：如果把系统部署到生产环境，最大的三个风险是什么？

**A**：第一，图谱覆盖范围有限——五个疾病域之外的内容全部判为健康。需要增量扩充分病种图谱模块，每个新疾病域是独立的 seed 数据段，通过 rebuild 接口热加载。第二，对话记忆的长期质量——一个月后回来看旧会话，记忆系统能否从大量对话中提取关键变化趋势。需要更智能的长期记忆总结策略，比如按时间窗口做分段摘要。第三，多用户并发下的资源隔离——当前所有会话共享同一个 Neo4j/Milvus/PG 实例。生产环境需要会话级的数据隔离和资源配额。

### Q29：你在这个项目中最大的收获是什么？

**A**：三个。第一，验证了「确定性代码 + LLM」的分工边界——医学判断由规则和图谱做，LLM 只做格式化和有限的知识补充。这个边界一旦划定，系统的安全底线就清晰了。第二，Multi-Agent 不是让多个 LLM 互相聊天就完事了，关键在 Agent 间传什么数据、为什么这样分职责、每个 Agent 的失败如何隔离。第三，检索优化的方法论——不是堆技术，而是每步有 ablation 验证独立收益。Rerank 前置这个决策不是在理论上想出来的，是在几组对照实验里跑出来的。
