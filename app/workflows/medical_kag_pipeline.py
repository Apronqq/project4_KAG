from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass
from typing import Any, Callable, Generator, TypedDict

from langgraph.graph import END, StateGraph

from app.graph.store import BaseGraphStore
from app.retrieval.evidence_store import BaseEvidenceStore
from app.retrieval.risk_ranker import MedicalRiskRanker
from app.schemas.exam import InternalAssessmentState
from app.services.diagnosis_formatter import DiagnosisFormatter
from app.services.evidence_query_planner import EvidenceQueryPlanner
from app.services.indicator_normalizer import IndicatorNormalizer
from app.services.input_parser import MedicalInputParser
from app.services.rules import IndicatorRuleEngine

logger = logging.getLogger(__name__)


class WorkflowState(TypedDict):
    state: InternalAssessmentState


@dataclass(frozen=True)
class PipelineStep:
    name: str
    label: str
    start_detail: str
    handler: Callable[[WorkflowState], WorkflowState]


class MedicalKAGWorkflow:
    def __init__(
        self,
        parser: MedicalInputParser,
        normalizer: IndicatorNormalizer,
        rule_engine: IndicatorRuleEngine,
        graph_store: BaseGraphStore,
        evidence_store: BaseEvidenceStore,
        ranker: MedicalRiskRanker,
        formatter: DiagnosisFormatter,
        query_planner: EvidenceQueryPlanner,
        top_k_evidence: int = 5,
    ):
        self._parser = parser
        self._normalizer = normalizer
        self._rule_engine = rule_engine
        self._graph_store = graph_store
        self._evidence_store = evidence_store
        self._ranker = ranker
        self._formatter = formatter
        self._query_planner = query_planner
        self._top_k_evidence = top_k_evidence
        self._graph = self._build_graph()

    def _build_graph(self):
        graph = StateGraph(WorkflowState)
        steps = self._step_definitions()
        for step in steps:
            graph.add_node(step.name, step.handler)

        graph.set_entry_point(steps[0].name)
        for current_step, next_step in zip(steps, steps[1:]):
            graph.add_edge(current_step.name, next_step.name)
        graph.add_edge(steps[-1].name, END)
        return graph.compile()

    def parse_only(self, raw_input: Any):
        return self._parser.parse(raw_input)

    def run(self, raw_input: Any):
        return self.run_state(raw_input).response

    async def run_async(self, raw_input: Any):
        return (await self.run_state_async(raw_input)).response

    def run_state(self, raw_input: Any) -> InternalAssessmentState:
        payload = self._execute_step_sequence(raw_input)
        return payload["state"]

    async def run_state_async(self, raw_input: Any) -> InternalAssessmentState:
        payload = await self._execute_step_sequence_async(raw_input)
        return payload["state"]

    def iter_events(self, raw_input: Any) -> Generator[dict, None, InternalAssessmentState]:
        payload: WorkflowState = {"state": InternalAssessmentState(raw_input=raw_input)}
        for step in self._step_definitions():
            if self._should_skip_step(step.name, payload["state"]):
                continue
            yield {
                "type": "step",
                "name": step.name,
                "label": step.label,
                "detail": step.start_detail,
                "status": "running",
            }
            started_at = time.perf_counter()
            payload = step.handler(payload)
            duration_ms = round((time.perf_counter() - started_at) * 1000, 2)
            state = payload["state"]
            logger.info(
                "medical_pipeline.step_completed",
                extra={
                    "step": step.name,
                    "duration_ms": duration_ms,
                    "exam_items_count": len(state.normalized_exam_json.exam_items)
                    if state.normalized_exam_json is not None
                    else 0,
                    "detected_states_count": len(state.detected_states),
                    "risk_candidates_count": len(state.risk_candidates),
                    "evidence_chunks_count": len(state.evidence_chunks),
                },
            )
            yield {
                "type": "step",
                "name": step.name,
                "label": step.label,
                "detail": self._step_success_detail(step.name, state),
                "status": "completed",
                "duration_ms": duration_ms,
            }
        return payload["state"]

    async def iter_events_async(self, raw_input: Any):
        payload: WorkflowState = {"state": InternalAssessmentState(raw_input=raw_input)}
        for step in self._step_definitions():
            if self._should_skip_step(step.name, payload["state"]):
                continue
            if step.name == "expand_intervention_paths":
                continue
            yield {
                "type": "step",
                "name": step.name,
                "label": step.label,
                "detail": step.start_detail,
                "status": "running",
            }
            started_at = time.perf_counter()
            if step.name == "retrieve_evidence_chunks":
                payload = await self._retrieve_evidence_and_interventions_async(payload)
            else:
                payload = await asyncio.to_thread(step.handler, payload)
            duration_ms = round((time.perf_counter() - started_at) * 1000, 2)
            state = payload["state"]
            logger.info(
                "medical_pipeline.async_step_completed",
                extra={
                    "step": step.name,
                    "duration_ms": duration_ms,
                    "exam_items_count": len(state.normalized_exam_json.exam_items)
                    if state.normalized_exam_json is not None
                    else 0,
                    "detected_states_count": len(state.detected_states),
                    "risk_candidates_count": len(state.risk_candidates),
                    "evidence_chunks_count": len(state.evidence_chunks),
                },
            )
            yield {
                "type": "step",
                "name": step.name,
                "label": step.label,
                "detail": self._step_success_detail(step.name, state),
                "status": "completed",
                "duration_ms": duration_ms,
            }
        yield {"type": "workflow_state", "state": payload["state"], "internal": True}

    def _execute_step_sequence(self, raw_input: Any) -> WorkflowState:
        payload: WorkflowState = {"state": InternalAssessmentState(raw_input=raw_input)}
        for step in self._step_definitions():
            if self._should_skip_step(step.name, payload["state"]):
                continue
            payload = step.handler(payload)
        return payload

    async def _execute_step_sequence_async(self, raw_input: Any) -> WorkflowState:
        payload: WorkflowState = {"state": InternalAssessmentState(raw_input=raw_input)}
        for step in self._step_definitions():
            if self._should_skip_step(step.name, payload["state"]):
                continue
            if step.name == "retrieve_evidence_chunks":
                payload = await self._retrieve_evidence_and_interventions_async(payload)
            elif step.name == "expand_intervention_paths":
                continue
            else:
                payload = await asyncio.to_thread(step.handler, payload)
        return payload

    async def _retrieve_evidence_and_interventions_async(self, payload: WorkflowState) -> WorkflowState:
        state = payload["state"]
        node_codes = []
        for risk in state.risk_candidates:
            node_codes.extend([risk.risk_code, risk.disease_code])
        disease_codes = [item.disease_code for item in state.risk_candidates]
        # 中文注释：证据检索与干预路径扩展互不依赖，异步入口并行执行以降低异常体检路径耗时。
        evidence_task = asyncio.create_task(
            self._evidence_store.search_async(
                queries=state.retrieval_queries,
                node_codes=node_codes,
                top_k=self._top_k_evidence,
            )
        )
        intervention_task = asyncio.create_task(
            self._graph_store.get_intervention_candidates_async(disease_codes)
        )
        state.evidence_chunks, state.intervention_candidates = await asyncio.gather(evidence_task, intervention_task)
        return {"state": state}

    def _should_skip_step(self, step_name: str, state: InternalAssessmentState) -> bool:
        if step_name in {
            "retrieve_graph_candidates",
            "expand_intervention_paths",
            "plan_evidence_queries",
            "retrieve_evidence_chunks",
            "rank_medical_evidence",
        }:
            # 中文注释：没有异常状态时直接进入诊断格式化，避免健康路径仍然访问图谱和证据库。
            return state.normalized_exam_json is not None and not state.detected_states
        if step_name == "expand_intervention_paths" and not state.risk_candidates:
            # 中文注释：有异常但图谱未命中时跳过干预路径扩展，后续证据检索仍可兜底。
            return True
        return False

    def _step_definitions(self) -> list[PipelineStep]:
        # 中文注释：流水线步骤只在这里维护，LangGraph、同步执行和流式事件共用这一份定义。
        return [
            PipelineStep("parse_raw_input", "解析用户输入", "正在将体检文本解析为结构化数据", self._parse_raw_input_node),
            PipelineStep("validate_exam_json", "校验体检结构", "正在校验解析后的体检数据", self._validate_exam_json_node),
            PipelineStep("normalize_exam_items", "标准化指标", "正在统一指标别名与单位", self._normalize_exam_items_node),
            PipelineStep("detect_indicator_states", "规则判定", "正在识别异常状态与组合风险", self._detect_indicator_states_node),
            PipelineStep("retrieve_graph_candidates", "图谱检索", "正在从知识图谱中查找疾病风险候选", self._retrieve_graph_candidates_node),
            PipelineStep("expand_intervention_paths", "路径扩展", "正在扩展干预、复查和科室建议", self._expand_intervention_paths_node),
            PipelineStep("plan_evidence_queries", "证据规划", "正在构建图感知检索查询", self._plan_evidence_queries_node),
            PipelineStep("retrieve_evidence_chunks", "证据检索", "正在从知识库召回相关证据片段", self._retrieve_evidence_chunks_node),
            PipelineStep("rank_medical_evidence", "结果排序", "正在融合图谱支持度和证据支持度", self._rank_medical_evidence_node),
            PipelineStep("generate_primary_diagnosis", "生成主诊断", "正在生成健康状态和主要风险判断", self._generate_primary_diagnosis_node),
            PipelineStep("generate_secondary_recommendation", "生成建议", "正在生成科室、复查和干预建议", self._generate_secondary_recommendation_node),
            PipelineStep("format_medical_response", "封装响应", "正在封装结构化评估结果", self._format_medical_response_node),
        ]

    def _step_success_detail(self, step_name: str, state: InternalAssessmentState) -> str:
        if step_name == "parse_raw_input":
            count = len(state.normalized_exam_json.exam_items) if state.normalized_exam_json else 0
            return f"已识别 {count} 个体检指标"
        if step_name == "detect_indicator_states":
            return f"发现 {len(state.detected_states)} 个异常或组合状态"
        if step_name == "retrieve_graph_candidates":
            return f"匹配到 {len(state.risk_candidates)} 个疾病风险候选"
        if step_name == "retrieve_evidence_chunks":
            return f"召回 {len(state.evidence_chunks)} 条证据片段"
        if step_name == "rank_medical_evidence":
            return f"完成 {len(state.risk_candidates)} 个风险候选排序"
        if step_name == "format_medical_response":
            return "结构化评估结果已生成"
        return "已完成"

    def _parse_raw_input_node(self, payload: WorkflowState) -> WorkflowState:
        state = payload["state"]
        parsed = self._parser.parse(state.raw_input)
        state.normalized_exam_json = parsed.normalized_exam_json
        state.missing_fields = parsed.missing_fields
        state.warnings = parsed.warnings
        return {"state": state}

    def _validate_exam_json_node(self, payload: WorkflowState) -> WorkflowState:
        state = payload["state"]
        if state.normalized_exam_json is None:
            raise ValueError("normalized_exam_json is required after parsing")
        return {"state": state}

    def _normalize_exam_items_node(self, payload: WorkflowState) -> WorkflowState:
        state = payload["state"]
        state.normalized_exam_json = self._normalizer.normalize(state.normalized_exam_json)
        return {"state": state}

    def _detect_indicator_states_node(self, payload: WorkflowState) -> WorkflowState:
        state = payload["state"]
        state.detected_states = self._rule_engine.detect_states(state.normalized_exam_json)
        return {"state": state}

    def _retrieve_graph_candidates_node(self, payload: WorkflowState) -> WorkflowState:
        state = payload["state"]
        state_codes = [item.state_code for item in state.detected_states]
        state.risk_candidates = self._graph_store.get_risk_candidates(state_codes)
        return {"state": state}

    def _expand_intervention_paths_node(self, payload: WorkflowState) -> WorkflowState:
        state = payload["state"]
        disease_codes = [item.disease_code for item in state.risk_candidates]
        state.intervention_candidates = self._graph_store.get_intervention_candidates(disease_codes)
        return {"state": state}

    def _plan_evidence_queries_node(self, payload: WorkflowState) -> WorkflowState:
        state = payload["state"]
        state.retrieval_queries = self._query_planner.build_queries(
            state.normalized_exam_json,
            state.detected_states,
            state.risk_candidates,
        )
        return {"state": state}

    def _retrieve_evidence_chunks_node(self, payload: WorkflowState) -> WorkflowState:
        state = payload["state"]
        node_codes = []
        for risk in state.risk_candidates:
            node_codes.extend([risk.risk_code, risk.disease_code])
        state.evidence_chunks = self._evidence_store.search(
            queries=state.retrieval_queries,
            node_codes=node_codes,
            top_k=self._top_k_evidence,
        )
        return {"state": state}

    def _rank_medical_evidence_node(self, payload: WorkflowState) -> WorkflowState:
        state = payload["state"]
        state.risk_candidates = self._ranker.rank_risks(
            state.risk_candidates,
            state.evidence_chunks,
            state.detected_states,
        )
        return {"state": state}

    def _generate_primary_diagnosis_node(self, payload: WorkflowState) -> WorkflowState:
        state = payload["state"]
        state.primary_diagnosis = self._formatter.build_primary(
            state.normalized_exam_json,
            state.risk_candidates,
            state.detected_states,
        )
        return {"state": state}

    def _generate_secondary_recommendation_node(self, payload: WorkflowState) -> WorkflowState:
        state = payload["state"]
        recommendation = None
        recommendation_map = None
        if state.intervention_candidates:
            recommendation = self._ranker.merge_recommendations(state.intervention_candidates)
            recommendation_map = self._ranker.index_recommendations_by_disease(state.intervention_candidates)
        state.secondary_recommendations = self._formatter.build_secondary(
            state.risk_candidates,
            recommendation,
            recommendation_map,
        )
        return {"state": state}

    def _format_medical_response_node(self, payload: WorkflowState) -> WorkflowState:
        state = payload["state"]
        graph_paths = []
        for risk in state.risk_candidates:
            graph_paths.extend(risk.graph_paths)
        for recommendation in state.intervention_candidates:
            graph_paths.extend(recommendation.graph_paths)
        state.response = self._formatter.build_response(
            state.normalized_exam_json,
            state.primary_diagnosis,
            state.secondary_recommendations,
            graph_paths,
            state.evidence_chunks,
        )
        return {"state": state}
