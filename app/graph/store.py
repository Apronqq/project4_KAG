from __future__ import annotations

from abc import ABC, abstractmethod
import asyncio
import logging
from typing import Iterable

try:
    from neo4j import GraphDatabase
except ImportError:  # pragma: no cover - optional runtime dependency
    GraphDatabase = None

from app.core.settings import Settings
from app.schemas.exam import GraphPath, InterventionCandidate, RiskCandidate

logger = logging.getLogger(__name__)


class BaseGraphStore(ABC):
    @property
    @abstractmethod
    def backend_name(self) -> str:
        raise NotImplementedError

    @property
    @abstractmethod
    def mode(self) -> str:
        raise NotImplementedError

    @abstractmethod
    def ping(self) -> bool:
        raise NotImplementedError

    @abstractmethod
    def ensure_schema(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def data_ready(self) -> bool:
        raise NotImplementedError

    @abstractmethod
    def get_risk_candidates(self, state_codes: list[str]) -> list[RiskCandidate]:
        raise NotImplementedError

    async def get_risk_candidates_async(self, state_codes: list[str]) -> list[RiskCandidate]:
        return await asyncio.to_thread(self.get_risk_candidates, state_codes)

    @abstractmethod
    def get_intervention_candidates(self, disease_codes: list[str]) -> list[InterventionCandidate]:
        raise NotImplementedError

    async def get_intervention_candidates_async(self, disease_codes: list[str]) -> list[InterventionCandidate]:
        return await asyncio.to_thread(self.get_intervention_candidates, disease_codes)

    @abstractmethod
    def rebuild_from_seed(
        self,
        state_to_risk: dict,
        disease_to_interventions: dict,
        evidence_chunks: list,
    ) -> None:
        raise NotImplementedError

    @abstractmethod
    def add_evidence_chunks(self, evidence_chunks: list) -> None:
        raise NotImplementedError


class InMemoryGraphStore(BaseGraphStore):
    def __init__(self, fallback_reason: str = ""):
        from app.graph.seed_data import DISEASE_TO_INTERVENTIONS, STATE_TO_RISK

        self._state_to_risk = {
            state_code: [dict(item) for item in items]
            for state_code, items in STATE_TO_RISK.items()
        }
        self._disease_to_interventions = {
            disease_code: {key: list(values) for key, values in payload.items()}
            for disease_code, payload in DISEASE_TO_INTERVENTIONS.items()
        }
        self.fallback_reason = fallback_reason

    @property
    def backend_name(self) -> str:
        return "in_memory"

    @property
    def mode(self) -> str:
        return "memory"

    def ping(self) -> bool:
        return True

    def ensure_schema(self) -> None:
        return None

    def data_ready(self) -> bool:
        return bool(self._state_to_risk and self._disease_to_interventions)

    def get_risk_candidates(self, state_codes: list[str]) -> list[RiskCandidate]:
        aggregated: dict[tuple[str, str], RiskCandidate] = {}
        for state_code in state_codes:
            for item in self._state_to_risk.get(state_code, []):
                key = (str(item["risk_code"]), str(item["disease_code"]))
                candidate = aggregated.get(key)
                path = GraphPath(
                    path_type="risk",
                    nodes=[state_code, str(item["risk_code"]), str(item["disease_code"])],
                    score=float(item["graph_score"]),
                )
                if candidate is None:
                    aggregated[key] = RiskCandidate(
                        risk_code=str(item["risk_code"]),
                        risk_name=str(item["risk_name"]),
                        disease_code=str(item["disease_code"]),
                        disease_name=str(item["disease_name"]),
                        risk_level=str(item["risk_level"]),
                        graph_score=float(item["graph_score"]),
                        supported_states=[state_code],
                        graph_paths=[path],
                    )
                else:
                    candidate.supported_states.append(state_code)
                    candidate.graph_paths.append(path)
                    candidate.graph_score = max(candidate.graph_score, float(item["graph_score"]))
        return sorted(aggregated.values(), key=lambda item: item.graph_score, reverse=True)

    def get_intervention_candidates(self, disease_codes: list[str]) -> list[InterventionCandidate]:
        out: list[InterventionCandidate] = []
        for disease_code in disease_codes:
            data = self._disease_to_interventions.get(disease_code)
            if not data:
                continue
            graph_paths: list[GraphPath] = []
            graph_paths.extend(
                GraphPath(path_type="intervention", nodes=[disease_code, name], score=0.80)
                for name in data["interventions"]
            )
            graph_paths.extend(
                GraphPath(path_type="follow_up", nodes=[disease_code, name], score=0.75)
                for name in data["follow_up_tests"]
            )
            graph_paths.extend(
                GraphPath(path_type="department", nodes=[disease_code, name], score=0.70)
                for name in data["departments"]
            )
            if data["contraindications"]:
                graph_paths.extend(
                    GraphPath(path_type="contraindication", nodes=[disease_code, name], score=0.78)
                    for name in data["contraindications"]
                )
            out.append(
                InterventionCandidate(
                    disease_code=disease_code,
                    interventions=list(data["interventions"]),
                    medication_directions=list(data["medication_directions"]),
                    contraindications=list(data["contraindications"]),
                    follow_up_tests=list(data["follow_up_tests"]),
                    departments=list(data["departments"]),
                    graph_paths=graph_paths,
                )
            )
        return out

    def rebuild_from_seed(self, state_to_risk: dict, disease_to_interventions: dict, evidence_chunks: list) -> None:
        self._state_to_risk = {
            state_code: [dict(item) for item in items]
            for state_code, items in state_to_risk.items()
        }
        self._disease_to_interventions = {
            disease_code: {key: list(values) for key, values in payload.items()}
            for disease_code, payload in disease_to_interventions.items()
        }

    def add_evidence_chunks(self, evidence_chunks: list) -> None:
        return None


class Neo4jGraphStore(BaseGraphStore):
    def __init__(self, settings: Settings):
        if GraphDatabase is None:
            raise RuntimeError(
                "Neo4j driver is not installed. Install the 'neo4j' package or enable USE_IN_MEMORY_GRAPH."
            )
        self._driver = GraphDatabase.driver(
            settings.neo4j_uri,
            auth=(settings.neo4j_user, settings.neo4j_password),
        )
        self._database = settings.neo4j_database
        self._verify_connectivity()

    @property
    def backend_name(self) -> str:
        return "neo4j"

    @property
    def mode(self) -> str:
        return "remote"

    def close(self) -> None:
        self._driver.close()

    def _verify_connectivity(self) -> None:
        self._driver.verify_connectivity()

    def _run(self, query: str, **params):
        with self._driver.session(database=self._database) as session:
            return list(session.run(query, **params))

    def ping(self) -> bool:
        try:
            rows = self._run("RETURN 1 AS ok")
            return bool(rows and rows[0]["ok"] == 1)
        except Exception:
            return False

    def ensure_schema(self) -> None:
        statements = [
            "CREATE CONSTRAINT indicator_state_code IF NOT EXISTS FOR (n:IndicatorState) REQUIRE n.state_code IS UNIQUE",
            "CREATE CONSTRAINT disease_risk_code IF NOT EXISTS FOR (n:DiseaseRisk) REQUIRE n.risk_code IS UNIQUE",
            "CREATE CONSTRAINT disease_code IF NOT EXISTS FOR (n:Disease) REQUIRE n.disease_code IS UNIQUE",
            "CREATE CONSTRAINT evidence_chunk_id IF NOT EXISTS FOR (n:EvidenceChunk) REQUIRE n.chunk_id IS UNIQUE",
            "CREATE INDEX disease_name_idx IF NOT EXISTS FOR (n:Disease) ON (n.name)",
            "CREATE INDEX risk_name_idx IF NOT EXISTS FOR (n:DiseaseRisk) ON (n.name)",
        ]
        for statement in statements:
            self._run(statement)

    def data_ready(self) -> bool:
        try:
            rows = self._run("MATCH (n) RETURN count(n) AS total")
            return bool(rows and int(rows[0]["total"]) > 0)
        except Exception:
            return False

    def get_risk_candidates(self, state_codes: list[str]) -> list[RiskCandidate]:
        if not state_codes:
            return []
        query = """
        MATCH (s:IndicatorState)-[:STATE_IMPLIES_RISK]->(r:DiseaseRisk)-[:RISK_RELATED_DISEASE]->(d:Disease)
        WHERE s.state_code IN $state_codes
        RETURN s.state_code AS state_code,
               r.risk_code AS risk_code,
               r.name AS risk_name,
               r.risk_level AS risk_level,
               d.disease_code AS disease_code,
               d.name AS disease_name
        """
        rows = self._run(query, state_codes=state_codes)
        aggregated: dict[tuple[str, str], RiskCandidate] = {}
        for row in rows:
            key = (row["risk_code"], row["disease_code"])
            path = GraphPath(
                path_type="risk",
                nodes=[row["state_code"], row["risk_code"], row["disease_code"]],
                score=0.90,
            )
            if key not in aggregated:
                aggregated[key] = RiskCandidate(
                    risk_code=row["risk_code"],
                    risk_name=row["risk_name"] or row["risk_code"],
                    disease_code=row["disease_code"],
                    disease_name=row["disease_name"] or row["disease_code"],
                    risk_level=row["risk_level"] or "medium",
                    graph_score=0.90,
                    supported_states=[row["state_code"]],
                    graph_paths=[path],
                )
            else:
                aggregated[key].supported_states.append(row["state_code"])
                aggregated[key].graph_paths.append(path)
        return sorted(aggregated.values(), key=lambda item: item.graph_score, reverse=True)

    def get_intervention_candidates(self, disease_codes: list[str]) -> list[InterventionCandidate]:
        if not disease_codes:
            return []
        query = """
        MATCH (d:Disease)
        WHERE d.disease_code IN $disease_codes
        OPTIONAL MATCH (d)-[:DISEASE_RECOMMENDS_INTERVENTION]->(i:Intervention)
        OPTIONAL MATCH (d)-[:DISEASE_RECOMMENDS_MEDICATION_DIRECTION]->(m:MedicationDirection)
        OPTIONAL MATCH (d)-[:DISEASE_HAS_CONTRAINDICATION]->(c:Contraindication)
        OPTIONAL MATCH (d)-[:DISEASE_REQUIRES_FOLLOWUP_TEST]->(f:FollowUpTest)
        OPTIONAL MATCH (d)-[:DISEASE_RECOMMENDS_DEPARTMENT]->(dep:Department)
        RETURN d.disease_code AS disease_code,
               collect(DISTINCT i.name) AS interventions,
               collect(DISTINCT m.name) AS medication_directions,
               collect(DISTINCT c.name) AS contraindications,
               collect(DISTINCT f.name) AS follow_up_tests,
               collect(DISTINCT dep.name) AS departments
        """
        rows = self._run(query, disease_codes=disease_codes)
        out: list[InterventionCandidate] = []
        for row in rows:
            disease_code = row["disease_code"]
            nodes = [disease_code]
            graph_paths = []
            for name in row["interventions"] or []:
                if name:
                    graph_paths.append(GraphPath(path_type="intervention", nodes=nodes + [name], score=0.80))
            for name in row["follow_up_tests"] or []:
                if name:
                    graph_paths.append(GraphPath(path_type="follow_up", nodes=nodes + [name], score=0.75))
            for name in row["departments"] or []:
                if name:
                    graph_paths.append(GraphPath(path_type="department", nodes=nodes + [name], score=0.70))
            for name in row["contraindications"] or []:
                if name:
                    graph_paths.append(GraphPath(path_type="contraindication", nodes=nodes + [name], score=0.78))
            out.append(
                InterventionCandidate(
                    disease_code=disease_code,
                    interventions=[item for item in row["interventions"] or [] if item],
                    medication_directions=[item for item in row["medication_directions"] or [] if item],
                    contraindications=[item for item in row["contraindications"] or [] if item],
                    follow_up_tests=[item for item in row["follow_up_tests"] or [] if item],
                    departments=[item for item in row["departments"] or [] if item],
                    graph_paths=graph_paths,
                )
            )
        return out

    def rebuild_from_seed(self, state_to_risk: dict, disease_to_interventions: dict, evidence_chunks: list) -> None:
        self._run("MATCH (n) DETACH DELETE n")

        risk_query = """
        MERGE (s:IndicatorState {state_code: $state_code})
        SET s.label = $state_label, s.rule_id = $rule_id, s.severity = $severity
        MERGE (r:DiseaseRisk {risk_code: $risk_code})
        SET r.name = $risk_name, r.risk_level = $risk_level
        MERGE (d:Disease {disease_code: $disease_code})
        SET d.name = $disease_name
        MERGE (s)-[:STATE_IMPLIES_RISK]->(r)
        MERGE (r)-[:RISK_RELATED_DISEASE]->(d)
        """
        for state_code, risks in state_to_risk.items():
            state_label = state_code.replace("_", " ")
            severity = "high" if "high" in state_code.lower() or "low" in state_code.lower() else "medium"
            for risk in risks:
                self._run(
                    risk_query,
                    state_code=state_code,
                    state_label=state_label,
                    rule_id=state_code.lower(),
                    severity=severity,
                    risk_code=risk["risk_code"],
                    risk_name=risk["risk_name"],
                    risk_level=risk["risk_level"],
                    disease_code=risk["disease_code"],
                    disease_name=risk["disease_name"],
                )

        intervention_query = """
        MERGE (d:Disease {disease_code: $disease_code})
        FOREACH (name IN $interventions |
            MERGE (i:Intervention {name: name})
            MERGE (d)-[:DISEASE_RECOMMENDS_INTERVENTION]->(i)
        )
        FOREACH (name IN $medication_directions |
            MERGE (m:MedicationDirection {name: name})
            MERGE (d)-[:DISEASE_RECOMMENDS_MEDICATION_DIRECTION]->(m)
        )
        FOREACH (name IN $contraindications |
            MERGE (c:Contraindication {name: name})
            MERGE (d)-[:DISEASE_HAS_CONTRAINDICATION]->(c)
        )
        FOREACH (name IN $follow_up_tests |
            MERGE (f:FollowUpTest {name: name})
            MERGE (d)-[:DISEASE_REQUIRES_FOLLOWUP_TEST]->(f)
        )
        FOREACH (name IN $departments |
            MERGE (dep:Department {name: name})
            MERGE (d)-[:DISEASE_RECOMMENDS_DEPARTMENT]->(dep)
        )
        """
        for disease_code, payload in disease_to_interventions.items():
            self._run(
                intervention_query,
                disease_code=disease_code,
                interventions=payload["interventions"],
                medication_directions=payload["medication_directions"],
                contraindications=payload["contraindications"],
                follow_up_tests=payload["follow_up_tests"],
                departments=payload["departments"],
            )

        evidence_query = """
        MERGE (e:EvidenceChunk {chunk_id: $chunk_id})
        SET e.doc_id = $doc_id,
            e.title = $title,
            e.text = $text,
            e.source_type = $source_type
        WITH e, $linked_node_codes AS codes
        UNWIND codes AS code
        OPTIONAL MATCH (r:DiseaseRisk {risk_code: code})
        OPTIONAL MATCH (d:Disease {disease_code: code})
        FOREACH (_ IN CASE WHEN r IS NOT NULL THEN [1] ELSE [] END | MERGE (r)-[:NODE_LINKED_CHUNK]->(e))
        FOREACH (_ IN CASE WHEN d IS NOT NULL THEN [1] ELSE [] END | MERGE (d)-[:NODE_LINKED_CHUNK]->(e))
        """
        for chunk in evidence_chunks:
            self._run(
                evidence_query,
                chunk_id=chunk.chunk_id,
                doc_id=chunk.doc_id,
                title=chunk.title,
                text=chunk.text,
                source_type=chunk.source_type,
                linked_node_codes=chunk.linked_node_codes,
            )

    def add_evidence_chunks(self, evidence_chunks: list) -> None:
        evidence_query = """
        MERGE (e:EvidenceChunk {chunk_id: $chunk_id})
        SET e.doc_id = $doc_id,
            e.title = $title,
            e.text = $text,
            e.source_type = $source_type
        WITH e, $linked_node_codes AS codes
        UNWIND codes AS code
        OPTIONAL MATCH (r:DiseaseRisk {risk_code: code})
        OPTIONAL MATCH (d:Disease {disease_code: code})
        FOREACH (_ IN CASE WHEN r IS NOT NULL THEN [1] ELSE [] END | MERGE (r)-[:NODE_LINKED_CHUNK]->(e))
        FOREACH (_ IN CASE WHEN d IS NOT NULL THEN [1] ELSE [] END | MERGE (d)-[:NODE_LINKED_CHUNK]->(e))
        """
        for chunk in evidence_chunks:
            self._run(
                evidence_query,
                chunk_id=chunk.chunk_id,
                doc_id=chunk.doc_id,
                title=chunk.title,
                text=chunk.text,
                source_type=chunk.source_type,
                linked_node_codes=chunk.linked_node_codes,
            )


def build_graph_store(settings: Settings) -> BaseGraphStore:
    if settings.use_in_memory_graph:
        return InMemoryGraphStore()
    try:
        store = Neo4jGraphStore(settings)
        if store.ping():
            return store
        logger.warning("neo4j_graph_store.ping_failed_fallback_to_memory")
        return InMemoryGraphStore(fallback_reason="neo4j ping failed")
    except Exception:
        logger.warning("neo4j_graph_store.init_failed_fallback_to_memory", exc_info=True)
        return InMemoryGraphStore(fallback_reason="neo4j init failed")


def dedupe_keep_order(values: Iterable[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        out.append(value)
    return out
