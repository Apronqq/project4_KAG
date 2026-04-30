# 检索评测报告

## 评测范围

- Corpus: `data/knowledge_registry.json`，共 1848 个 EvidenceChunk。
- Cases: 12 条离线医学检索查询，覆盖高血压、慢性肾病、血脂异常、脂肪肝、糖尿病前期等主题。
- Relevance: 使用文档主题弱标注，2=强相关主文档，1=交叉/背景相关文档，0=不相关。
- Metrics: MRR、Precision@5、nDCG@5；nDCG 使用 chunk 级理想排序归一化。
- Mode: `query_only` 模拟 ReAct 工具追问检索；`graph_aware` 模拟主流水线带图谱节点约束的证据检索。
- Runtime: 离线评测不调用外部 embedding、rerank 或 Milvus 服务，使用项目内置轻量 embedding，保证本地可复现。

## 汇总结果

| Mode | MRR | Precision@5 | nDCG@5 | Avg Latency ms | P95 Latency ms |
| --- | ---: | ---: | ---: | ---: | ---: |
| query_only | 0.7500 | 0.3500 | 0.3964 | 60.12 | 65.49 |
| graph_aware | 0.7917 | 0.7667 | 0.6543 | 65.13 | 94.89 |

## 分查询明细

### query_only

| Case | RR | P@5 | nDCG@5 | Count | Grades | Doc IDs |
| --- | ---: | ---: | ---: | ---: | --- | --- |
| hypertension_definition | 1.0000 | 0.4000 | 0.5531 | 2 | [2, 2] | `doc_32b2e11aec9b51bb, doc_32b2e11aec9b51bb` |
| hypertension_lifestyle | 1.0000 | 0.4000 | 0.5531 | 2 | [2, 2] | `doc_32b2e11aec9b51bb, doc_32b2e11aec9b51bb` |
| blood_pressure_medication | 1.0000 | 0.4000 | 0.5531 | 2 | [2, 2] | `doc_32b2e11aec9b51bb, doc_32b2e11aec9b51bb` |
| ckd_egfr_management | 1.0000 | 0.8000 | 0.8688 | 4 | [2, 2, 2, 2] | `doc_e06d0f0a5aba6a23, doc_e06d0f0a5aba6a23, doc_e06d0f0a5aba6a23, doc_e06d0f0a5aba6a23` |
| ckd_diet | 1.0000 | 0.8000 | 0.8688 | 4 | [2, 2, 2, 2] | `doc_e06d0f0a5aba6a23, doc_e06d0f0a5aba6a23, doc_e06d0f0a5aba6a23, doc_e06d0f0a5aba6a23` |
| dyslipidemia_ldl | 0.0000 | 0.0000 | 0.0000 | 1 | [0] | `doc_32b2e11aec9b51bb` |
| dyslipidemia_statin | 1.0000 | 0.4000 | 0.5531 | 2 | [2, 2] | `doc_a2f5059ac2686b20, doc_a2f5059ac2686b20` |
| fatty_liver_diagnosis | 0.0000 | 0.0000 | 0.0000 | 1 | [0] | `doc_a2f5059ac2686b20` |
| fatty_liver_weight | 0.0000 | 0.0000 | 0.0000 | 1 | [0] | `doc_32b2e11aec9b51bb` |
| prediabetes | 1.0000 | 0.4000 | 0.1844 | 2 | [1, 1] | `doc_32b2e11aec9b51bb, doc_32b2e11aec9b51bb` |
| cholesterol_patient_edu | 1.0000 | 0.4000 | 0.5087 | 3 | [2, 0, 2] | `doc_a2f5059ac2686b20, doc_e06d0f0a5aba6a23, doc_a2f5059ac2686b20` |
| high_blood_pressure_patient_edu | 1.0000 | 0.2000 | 0.1131 | 1 | [1] | `doc_32b2e11aec9b51bb` |

### graph_aware

| Case | RR | P@5 | nDCG@5 | Count | Grades | Doc IDs |
| --- | ---: | ---: | ---: | ---: | --- | --- |
| hypertension_definition | 1.0000 | 1.0000 | 1.0000 | 5 | [2, 2, 2, 2, 2] | `doc_32b2e11aec9b51bb, doc_32b2e11aec9b51bb, doc_32b2e11aec9b51bb, doc_32b2e11aec9b51bb, doc_32b2e11aec9b51bb` |
| hypertension_lifestyle | 1.0000 | 1.0000 | 1.0000 | 5 | [2, 2, 2, 2, 2] | `doc_32b2e11aec9b51bb, doc_32b2e11aec9b51bb, doc_32b2e11aec9b51bb, doc_32b2e11aec9b51bb, doc_32b2e11aec9b51bb` |
| blood_pressure_medication | 1.0000 | 1.0000 | 1.0000 | 5 | [2, 2, 2, 2, 2] | `doc_32b2e11aec9b51bb, doc_32b2e11aec9b51bb, doc_32b2e11aec9b51bb, doc_32b2e11aec9b51bb, doc_32b2e11aec9b51bb` |
| ckd_egfr_management | 1.0000 | 1.0000 | 1.0000 | 5 | [2, 2, 2, 2, 2] | `doc_e06d0f0a5aba6a23, doc_e06d0f0a5aba6a23, doc_e06d0f0a5aba6a23, doc_e06d0f0a5aba6a23, doc_e06d0f0a5aba6a23` |
| ckd_diet | 1.0000 | 0.8000 | 0.8688 | 5 | [2, 2, 2, 2, 0] | `doc_e06d0f0a5aba6a23, doc_e06d0f0a5aba6a23, doc_e06d0f0a5aba6a23, doc_e06d0f0a5aba6a23, doc_32b2e11aec9b51bb` |
| dyslipidemia_ldl | 0.5000 | 0.8000 | 0.6608 | 5 | [0, 2, 2, 2, 2] | `doc_32b2e11aec9b51bb, doc_a2f5059ac2686b20, doc_a2f5059ac2686b20, doc_a2f5059ac2686b20, doc_a2f5059ac2686b20` |
| dyslipidemia_statin | 1.0000 | 1.0000 | 1.0000 | 5 | [2, 2, 2, 2, 2] | `doc_a2f5059ac2686b20, doc_a2f5059ac2686b20, doc_a2f5059ac2686b20, doc_a2f5059ac2686b20, doc_a2f5059ac2686b20` |
| fatty_liver_diagnosis | 0.0000 | 0.0000 | 0.0000 | 5 | [0, 0, 0, 0, 0] | `doc_a2f5059ac2686b20, doc_a2f5059ac2686b20, doc_e06d0f0a5aba6a23, doc_32b2e11aec9b51bb, doc_a2f5059ac2686b20` |
| fatty_liver_weight | 0.0000 | 0.0000 | 0.0000 | 5 | [0, 0, 0, 0, 0] | `doc_32b2e11aec9b51bb, doc_a2f5059ac2686b20, doc_a2f5059ac2686b20, doc_32b2e11aec9b51bb, doc_32b2e11aec9b51bb` |
| prediabetes | 1.0000 | 1.0000 | 0.3333 | 5 | [1, 1, 1, 1, 1] | `doc_32b2e11aec9b51bb, doc_32b2e11aec9b51bb, doc_32b2e11aec9b51bb, doc_32b2e11aec9b51bb, doc_32b2e11aec9b51bb` |
| cholesterol_patient_edu | 1.0000 | 0.6000 | 0.6548 | 5 | [2, 0, 2, 2, 0] | `doc_a2f5059ac2686b20, doc_e06d0f0a5aba6a23, doc_a2f5059ac2686b20, doc_a2f5059ac2686b20, doc_e06d0f0a5aba6a23` |
| high_blood_pressure_patient_edu | 1.0000 | 1.0000 | 0.3333 | 5 | [1, 1, 1, 1, 1] | `doc_32b2e11aec9b51bb, doc_32b2e11aec9b51bb, doc_32b2e11aec9b51bb, doc_32b2e11aec9b51bb, doc_32b2e11aec9b51bb` |
