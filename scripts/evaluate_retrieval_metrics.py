from __future__ import annotations

import json
import math
import statistics
import sys
import time
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from app.models.factory import LightweightEmbeddingProvider
from app.retrieval.evidence_store import InMemoryEvidenceStore
from app.schemas.exam import EvidenceChunk, RetrievalQuery


REGISTRY_PATH = PROJECT_ROOT / "data" / "knowledge_registry.json"
REPORT_PATH = PROJECT_ROOT / "RETRIEVAL_EVALUATION_REPORT.md"


@dataclass(frozen=True)
class EvaluationCase:
    case_id: str
    query: str
    node_codes: list[str]
    grade_by_doc_id: dict[str, int]


@dataclass
class EvaluationRow:
    case_id: str
    reciprocal_rank: float
    precision_at_5: float
    ndcg_at_5: float
    result_count: int
    doc_ids: list[str]
    grades: list[int]


EVALUATION_CASES = [
    EvaluationCase(
        case_id="hypertension_definition",
        query="高血压 诊断标准 收缩压 舒张压 分级 成人 指南",
        node_codes=["hypertension", "hypertension_risk"],
        grade_by_doc_id={"doc_32b2e11aec9b51bb": 2, "doc_e2b44292a27a254a": 1},
    ),
    EvaluationCase(
        case_id="hypertension_lifestyle",
        query="高血压 生活方式干预 限盐 DASH 运动 减重",
        node_codes=["hypertension", "hypertension_risk"],
        grade_by_doc_id={"doc_32b2e11aec9b51bb": 2, "doc_e2b44292a27a254a": 1},
    ),
    EvaluationCase(
        case_id="blood_pressure_medication",
        query="高血压 降压药物 治疗 启动时机 心血管风险",
        node_codes=["hypertension", "hypertension_risk"],
        grade_by_doc_id={"doc_32b2e11aec9b51bb": 2, "doc_e2b44292a27a254a": 1},
    ),
    EvaluationCase(
        case_id="ckd_egfr_management",
        query="慢性肾脏病 eGFR 下降 管理 延缓进展 临床指南",
        node_codes=["ckd", "ckd_risk"],
        grade_by_doc_id={"doc_e06d0f0a5aba6a23": 2, "doc_3c112ec228685813": 1},
    ),
    EvaluationCase(
        case_id="ckd_diet",
        query="慢性肾脏病 饮食 管理 蛋白 摄入 盐 控制",
        node_codes=["ckd", "ckd_risk"],
        grade_by_doc_id={"doc_e06d0f0a5aba6a23": 2, "doc_3c112ec228685813": 1},
    ),
    EvaluationCase(
        case_id="dyslipidemia_ldl",
        query="血脂异常 LDL-C 胆固醇 管理 目标值 指南",
        node_codes=["dyslipidemia", "dyslipidemia_risk"],
        grade_by_doc_id={"doc_a2f5059ac2686b20": 2},
    ),
    EvaluationCase(
        case_id="dyslipidemia_statin",
        query="他汀 降脂治疗 血脂管理 ASCVD 风险",
        node_codes=["dyslipidemia", "dyslipidemia_risk"],
        grade_by_doc_id={"doc_a2f5059ac2686b20": 2},
    ),
    EvaluationCase(
        case_id="fatty_liver_diagnosis",
        query="脂肪肝 诊断 治疗 指南 肝功能异常",
        node_codes=["liver_function_abnormality", "liver_function_abnormal_risk"],
        grade_by_doc_id={"doc_93329b2548dd1165": 2},
    ),
    EvaluationCase(
        case_id="fatty_liver_weight",
        query="非酒精性脂肪性肝病 减重 生活方式 治疗",
        node_codes=["liver_function_abnormality", "liver_function_abnormal_risk"],
        grade_by_doc_id={"doc_93329b2548dd1165": 2},
    ),
    EvaluationCase(
        case_id="prediabetes",
        query="糖尿病前期 血糖 健康风险 预防 生活方式",
        node_codes=["prediabetes", "prediabetes_risk"],
        grade_by_doc_id={"doc_f39b83fed61e5819": 2, "doc_32b2e11aec9b51bb": 1},
    ),
    EvaluationCase(
        case_id="cholesterol_patient_edu",
        query="胆固醇 高胆固醇 患者教育 风险 因素",
        node_codes=["dyslipidemia", "dyslipidemia_risk"],
        grade_by_doc_id={"doc_a2f5059ac2686b20": 2},
    ),
    EvaluationCase(
        case_id="high_blood_pressure_patient_edu",
        query="高血压 患者教育 症状 原因 风险 MedlinePlus",
        node_codes=["hypertension", "hypertension_risk"],
        grade_by_doc_id={"doc_e2b44292a27a254a": 2, "doc_32b2e11aec9b51bb": 1},
    ),
]


def load_chunks() -> list[EvidenceChunk]:
    payload = json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))
    chunks: list[EvidenceChunk] = []
    for items in payload.get("chunks_by_doc_id", {}).values():
        chunks.extend(EvidenceChunk(**item) for item in items)
    return chunks


def dcg(grades: list[int]) -> float:
    return sum((2**grade - 1) / math.log2(index + 2) for index, grade in enumerate(grades))


def evaluate_case(
    store: InMemoryEvidenceStore,
    all_chunks: list[EvidenceChunk],
    case: EvaluationCase,
    *,
    graph_aware: bool,
) -> EvaluationRow:
    # 中文注释：query_only 模拟 ReAct 工具追问检索；graph_aware 模拟主流水线带图谱节点约束的证据检索。
    node_codes = case.node_codes if graph_aware else []
    results = store.search(
        queries=[RetrievalQuery(label="eval", text=case.query)],
        node_codes=node_codes,
        top_k=5,
    )
    grades = [case.grade_by_doc_id.get(chunk.doc_id, 0) for chunk in results[:5]]
    binary_hits = [1 if grade > 0 else 0 for grade in grades]
    reciprocal_rank = next((1 / rank for rank, hit in enumerate(binary_hits, 1) if hit), 0.0)
    precision_at_5 = sum(binary_hits) / 5

    # 中文注释：nDCG 的理想排序从全量 chunk 相关性生成，避免同一相关文档多个 chunk 时被低估或高估。
    ideal_grades = sorted(
        (case.grade_by_doc_id.get(chunk.doc_id, 0) for chunk in all_chunks),
        reverse=True,
    )[:5]
    ndcg_at_5 = dcg(grades + [0] * (5 - len(grades))) / (dcg(ideal_grades) or 1.0)

    return EvaluationRow(
        case_id=case.case_id,
        reciprocal_rank=reciprocal_rank,
        precision_at_5=precision_at_5,
        ndcg_at_5=ndcg_at_5,
        result_count=len(results),
        doc_ids=[chunk.doc_id for chunk in results[:5]],
        grades=grades,
    )


def summarize(rows: list[EvaluationRow], latencies_ms: list[float]) -> dict[str, float]:
    return {
        "MRR": sum(row.reciprocal_rank for row in rows) / len(rows),
        "Precision@5": sum(row.precision_at_5 for row in rows) / len(rows),
        "nDCG@5": sum(row.ndcg_at_5 for row in rows) / len(rows),
        "AvgLatencyMs": statistics.mean(latencies_ms),
        "P95LatencyMs": sorted(latencies_ms)[math.ceil(len(latencies_ms) * 0.95) - 1],
    }


def run_mode(store: InMemoryEvidenceStore, chunks: list[EvidenceChunk], *, graph_aware: bool) -> tuple[list[EvaluationRow], list[float]]:
    rows: list[EvaluationRow] = []
    latencies_ms: list[float] = []
    for case in EVALUATION_CASES:
        start = time.perf_counter()
        rows.append(evaluate_case(store, chunks, case, graph_aware=graph_aware))
        latencies_ms.append((time.perf_counter() - start) * 1000)
    return rows, latencies_ms


def format_float(value: float) -> str:
    return f"{value:.4f}"


def write_report(chunks: list[EvidenceChunk], results: dict[str, tuple[list[EvaluationRow], dict[str, float]]]) -> None:
    lines = [
        "# 检索评测报告",
        "",
        "## 评测范围",
        "",
        f"- Corpus: `data/knowledge_registry.json`，共 {len(chunks)} 个 EvidenceChunk。",
        f"- Cases: {len(EVALUATION_CASES)} 条离线医学检索查询，覆盖高血压、慢性肾病、血脂异常、脂肪肝、糖尿病前期等主题。",
        "- Relevance: 使用文档主题弱标注，2=强相关主文档，1=交叉/背景相关文档，0=不相关。",
        "- Metrics: MRR、Precision@5、nDCG@5；nDCG 使用 chunk 级理想排序归一化。",
        "- Mode: `query_only` 模拟 ReAct 工具追问检索；`graph_aware` 模拟主流水线带图谱节点约束的证据检索。",
        "- Runtime: 离线评测不调用外部 embedding、rerank 或 Milvus 服务，使用项目内置轻量 embedding，保证本地可复现。",
        "",
        "## 汇总结果",
        "",
        "| Mode | MRR | Precision@5 | nDCG@5 | Avg Latency ms | P95 Latency ms |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for mode, (_, summary) in results.items():
        lines.append(
            f"| {mode} | {format_float(summary['MRR'])} | {format_float(summary['Precision@5'])} | "
            f"{format_float(summary['nDCG@5'])} | {summary['AvgLatencyMs']:.2f} | {summary['P95LatencyMs']:.2f} |"
        )
    lines.extend(["", "## 分查询明细", ""])
    for mode, (rows, _) in results.items():
        lines.extend(
            [
                f"### {mode}",
                "",
                "| Case | RR | P@5 | nDCG@5 | Count | Grades | Doc IDs |",
                "| --- | ---: | ---: | ---: | ---: | --- | --- |",
            ]
        )
        for row in rows:
            lines.append(
                f"| {row.case_id} | {format_float(row.reciprocal_rank)} | {format_float(row.precision_at_5)} | "
                f"{format_float(row.ndcg_at_5)} | {row.result_count} | {row.grades} | `{', '.join(row.doc_ids)}` |"
            )
        lines.append("")
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    chunks = load_chunks()
    store = InMemoryEvidenceStore(embedder=LightweightEmbeddingProvider(256), reranker=None)
    store.rebuild_index(chunks)

    results: dict[str, tuple[list[EvaluationRow], dict[str, float]]] = {}
    for mode, graph_aware in (("query_only", False), ("graph_aware", True)):
        rows, latencies = run_mode(store, chunks, graph_aware=graph_aware)
        results[mode] = (rows, summarize(rows, latencies))

    write_report(chunks, results)
    print(f"Corpus chunks: {len(chunks)}")
    print(f"Evaluation cases: {len(EVALUATION_CASES)}")
    for mode, (_, summary) in results.items():
        print(
            f"{mode}: "
            f"MRR={format_float(summary['MRR'])}, "
            f"Precision@5={format_float(summary['Precision@5'])}, "
            f"nDCG@5={format_float(summary['nDCG@5'])}, "
            f"AvgLatencyMs={summary['AvgLatencyMs']:.2f}, "
            f"P95LatencyMs={summary['P95LatencyMs']:.2f}"
        )
    print(f"Report written: {REPORT_PATH}")


if __name__ == "__main__":
    main()
