"""NOVA Execution Plan Graph Contracts (Layer 4/6 Boundary — ADR-003 / ADR-010 / Phase 6).

Implements the directed execution graph domain model from NOVA_ADR.md:
- ADR-003: Models task execution as a directed graph of nodes and edges supporting parallel
  fan-out/fan-in, conditional routing, bounded reflection/retry loops, and static permission pre-flighting.
- Phase 1 Final Recommendation: Automatically flags high-stakes / irreversible-action nodes for
  Reflexion self-critique (`requires_reflection=True`).
- Phase 6: Supports topological wave scheduling, hardware-bounded concurrency (`NFR-11`),
  and critical-path calculation.
"""

from __future__ import annotations

import uuid
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple

from pydantic import BaseModel, Field

try:
    from pydantic import ConfigDict

    PYDANTIC_V2 = True
except ImportError:  # pragma: no cover
    PYDANTIC_V2 = False

from nova.core.domain.primitives import (
    CURRENT_SCHEMA_VERSION,
    AgentRole,
    SensitivityTier,
    validate_schema_version,
)


class PlanGraphValidationError(ValueError):
    """Raised when an execution plan graph violates ADR-003 structural or safety invariants."""


class PlanNodeStatus(str, Enum):
    """Lifecycle states of an individual node within an ExecutionPlanGraph."""

    PENDING = "pending"
    READY = "ready"
    RUNNING = "running"
    AWAITING_REFLECTION = "awaiting_reflection"
    AWAITING_APPROVAL = "awaiting_approval"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


class EdgeCondition(str, Enum):
    """Control-flow transition conditions between nodes in an ExecutionPlanGraph (ADR-003)."""

    ALWAYS = "always"
    ON_SUCCESS = "on_success"
    ON_FAILURE = "on_failure"
    ON_REVISE = "on_revise"
    ON_APPROVAL = "on_approval"


class PlanEdge(BaseModel):
    """Immutable directed edge connecting two nodes in an ExecutionPlanGraph."""

    if PYDANTIC_V2:
        model_config = ConfigDict(frozen=True, extra="forbid")
    else:

        class Config:
            allow_mutation = False
            extra = "forbid"

    source_node_id: str = Field(..., min_length=1, description="Upstream node ID")
    target_node_id: str = Field(..., min_length=1, description="Downstream node ID")
    condition: EdgeCondition = Field(
        default=EdgeCondition.ON_SUCCESS,
        description="Transition condition triggering this edge",
    )

    def __init__(self, **data: Any) -> None:
        src = data.get("source_node_id")
        dst = data.get("target_node_id")
        cond = data.get("condition", EdgeCondition.ON_SUCCESS)
        if not isinstance(src, str) or not src.strip():
            raise PlanGraphValidationError("PlanEdge 'source_node_id' must be non-empty.")
        if not isinstance(dst, str) or not dst.strip():
            raise PlanGraphValidationError("PlanEdge 'target_node_id' must be non-empty.")
        if src == dst and cond != EdgeCondition.ON_REVISE:
            raise PlanGraphValidationError(
                f"ADR-003 violation: Self-edge on node {src!r} is only allowed with condition='on_revise'."
            )
        super().__init__(**data)


class PlanNode(BaseModel):
    """Immutable execution graph node assigned to a single specialized agent (ADR-003 / ADR-008)."""

    if PYDANTIC_V2:
        model_config = ConfigDict(frozen=True, extra="forbid")
    else:

        class Config:
            allow_mutation = False
            extra = "forbid"

    node_id: str = Field(..., min_length=1, description="Unique node identifier in the graph")
    target_agent: AgentRole = Field(
        ..., description="Specialized agent responsible for executing this node"
    )
    objective: str = Field(..., min_length=1, description="Scoped subtask objective for the agent")
    depends_on: Tuple[str, ...] = Field(
        default_factory=tuple,
        description="Upstream node IDs that must succeed before this node can run",
    )
    estimated_sensitivity: SensitivityTier = Field(
        default=SensitivityTier.READ_ONLY,
        description="Highest anticipated sensitivity tier of actions in this node",
    )
    requires_reflection: bool = Field(
        default=False,
        description="Phase 1: True if node must undergo Reflexion self-critique before execution",
    )
    requires_human_approval: bool = Field(
        default=False,
        description="ADR-010: True if node requires explicit human approval at the Permission Gate",
    )
    max_react_iterations: int = Field(
        default=5,
        description="ADR-004: Maximum Thought-Action-Observation cycles allowed inside this node",
    )
    max_retries: int = Field(
        default=3,
        description="Phase 6: Maximum bounded retries on transient failure",
    )

    def __init__(self, **data: Any) -> None:
        node_id = data.get("node_id")
        if not isinstance(node_id, str) or not node_id.strip():
            raise PlanGraphValidationError("PlanNode 'node_id' must be a non-empty string.")
        objective = data.get("objective")
        if not isinstance(objective, str) or not objective.strip():
            raise PlanGraphValidationError("PlanNode 'objective' must be a non-empty string.")

        depends_on = tuple(data.get("depends_on", ()))
        if node_id in depends_on:
            raise PlanGraphValidationError(
                f"ADR-003 violation: PlanNode {node_id!r} cannot depend on itself."
            )

        raw_tier = data.get("estimated_sensitivity", SensitivityTier.READ_ONLY)
        tier = raw_tier if isinstance(raw_tier, SensitivityTier) else SensitivityTier(raw_tier)

        # Enforce Phase 1 Final Recommendation & ADR-010:
        # High-stakes / irreversible or approval-gated nodes MUST enable reflection & approval flags
        req_reflection = bool(data.get("requires_reflection", False)) or (
            tier.is_irreversible or tier.requires_human_approval_by_default
        )
        req_approval = bool(data.get("requires_human_approval", False)) or (
            tier.requires_human_approval_by_default
        )

        max_react = int(data.get("max_react_iterations", 5))
        if not (1 <= max_react <= 15):
            raise PlanGraphValidationError(
                f"ADR-004 violation: 'max_react_iterations' must be in [1, 15], got {max_react}."
            )
        max_retries = int(data.get("max_retries", 3))
        if not (0 <= max_retries <= 5):
            raise PlanGraphValidationError(
                f"Phase 6 violation: 'max_retries' must be in [0, 5], got {max_retries}."
            )

        updated = dict(data)
        updated["estimated_sensitivity"] = tier
        updated["requires_reflection"] = req_reflection
        updated["requires_human_approval"] = req_approval
        super().__init__(**updated)


class ExecutionPlanGraph(BaseModel):
    """Validated directed execution graph owned by the Layer 6 Orchestrator (ADR-003)."""

    if PYDANTIC_V2:
        model_config = ConfigDict(frozen=True, extra="forbid")
    else:

        class Config:
            allow_mutation = False
            extra = "forbid"

    schema_version: str = Field(default=CURRENT_SCHEMA_VERSION)
    plan_id: str = Field(default_factory=lambda: f"plan_{uuid.uuid4().hex}")
    task_id: str = Field(..., min_length=1)
    goal_summary: str = Field(..., min_length=1)
    nodes: Tuple[PlanNode, ...] = Field(..., description="Graph nodes in execution order")
    edges: Tuple[PlanEdge, ...] = Field(
        default_factory=tuple,
        description="Explicit control/data flow edges connecting nodes",
    )
    max_graph_steps: int = Field(
        default=25,
        description="ADR-003 upper bound on total graph nodes",
    )

    def __init__(self, **data: Any) -> None:
        version = data.get("schema_version", CURRENT_SCHEMA_VERSION)
        if not isinstance(version, str):
            raise PlanGraphValidationError("schema_version must be a string.")
        try:
            validate_schema_version(version)
        except ValueError as exc:
            raise PlanGraphValidationError(str(exc)) from exc

        raw_nodes = data.get("nodes")
        if not raw_nodes or not isinstance(raw_nodes, (list, tuple)):
            raise PlanGraphValidationError("ExecutionPlanGraph requires at least one PlanNode.")

        max_steps = int(data.get("max_graph_steps", 25))
        if not (1 <= max_steps <= 100):
            raise PlanGraphValidationError(
                f"ADR-003 violation: max_graph_steps must be between 1 and 100, got {max_steps}."
            )

        parsed_nodes = tuple(
            n if isinstance(n, PlanNode) else PlanNode(**n) for n in raw_nodes
        )
        if len(parsed_nodes) > max_steps:
            raise PlanGraphValidationError(
                f"ADR-003 violation: Plan contains {len(parsed_nodes)} nodes, exceeding "
                f"max_graph_steps={max_steps}."
            )

        node_map: Dict[str, PlanNode] = {}
        for node in parsed_nodes:
            if node.node_id in node_map:
                raise PlanGraphValidationError(f"Duplicate node_id {node.node_id!r} in plan graph.")
            node_map[node.node_id] = node

        for node in parsed_nodes:
            for dep_id in node.depends_on:
                if dep_id not in node_map:
                    raise PlanGraphValidationError(
                        f"Node {node.node_id!r} depends on non-existent node {dep_id!r}."
                    )

        raw_edges = data.get("edges")
        if raw_edges is None:
            # Automatically synthesize ON_SUCCESS edges from depends_on declarations
            synthesized: List[PlanEdge] = []
            for node in parsed_nodes:
                for dep_id in node.depends_on:
                    synthesized.append(
                        PlanEdge(
                            source_node_id=dep_id,
                            target_node_id=node.node_id,
                            condition=EdgeCondition.ON_SUCCESS,
                        )
                    )
            parsed_edges = tuple(synthesized)
        else:
            parsed_edges = tuple(
                e if isinstance(e, PlanEdge) else PlanEdge(**e) for e in raw_edges
            )

        for edge in parsed_edges:
            if edge.source_node_id not in node_map:
                raise PlanGraphValidationError(
                    f"Edge source {edge.source_node_id!r} does not exist in graph nodes."
                )
            if edge.target_node_id not in node_map:
                raise PlanGraphValidationError(
                    f"Edge target {edge.target_node_id!r} does not exist in graph nodes."
                )

        # Verify dependency structure is acyclic (Kahn's algorithm)
        cls_waves = ExecutionPlanGraph._compute_topological_waves(parsed_nodes, parsed_edges)
        if sum(len(w) for w in cls_waves) != len(parsed_nodes):
            raise PlanGraphValidationError(
                "ADR-003 violation: Unbounded dependency cycle detected in ExecutionPlanGraph."
            )

        updated = dict(data)
        updated["nodes"] = parsed_nodes
        updated["edges"] = parsed_edges
        updated["max_graph_steps"] = max_steps
        super().__init__(**updated)

    @staticmethod
    def _compute_topological_waves(
        nodes: Tuple[PlanNode, ...],
        edges: Tuple[PlanEdge, ...],
    ) -> Tuple[Tuple[str, ...], ...]:
        """Partition nodes into concurrent execution waves based on dependencies."""
        deps: Dict[str, Set[str]] = {n.node_id: set(n.depends_on) for n in nodes}
        for edge in edges:
            if edge.condition in (EdgeCondition.ON_SUCCESS, EdgeCondition.ALWAYS):
                if edge.source_node_id != edge.target_node_id:
                    deps[edge.target_node_id].add(edge.source_node_id)

        remaining = set(deps.keys())
        completed: Set[str] = set()
        waves: List[Tuple[str, ...]] = []

        # Preserve original node declaration order within each wave for determinism
        ordered_ids = [n.node_id for n in nodes]

        while remaining:
            ready = tuple(
                nid for nid in ordered_ids if nid in remaining and deps[nid].issubset(completed)
            )
            if not ready:
                break  # Cycle detected
            waves.append(ready)
            completed.update(ready)
            remaining.difference_update(ready)

        return tuple(waves)

    def topological_waves(self) -> Tuple[Tuple[str, ...], ...]:
        """Return parallelizable execution waves in topological dependency order (Phase 6)."""
        return self._compute_topological_waves(self.nodes, self.edges)

    def ready_nodes(
        self,
        completed_node_ids: Set[str],
        running_node_ids: Optional[Set[str]] = None,
        max_concurrency: int = 4,
    ) -> Tuple[PlanNode, ...]:
        """Return the next runnable nodes bounded by local hardware concurrency (NFR-11)."""
        if max_concurrency < 1:
            raise ValueError("max_concurrency must be >= 1.")
        active = running_node_ids or set()
        available_slots = max(0, max_concurrency - len(active))
        if available_slots == 0:
            return ()

        deps: Dict[str, Set[str]] = {n.node_id: set(n.depends_on) for n in self.nodes}
        for edge in self.edges:
            if edge.condition in (EdgeCondition.ON_SUCCESS, EdgeCondition.ALWAYS):
                if edge.source_node_id != edge.target_node_id:
                    deps[edge.target_node_id].add(edge.source_node_id)

        candidates: List[PlanNode] = []
        for node in self.nodes:
            if node.node_id in completed_node_ids or node.node_id in active:
                continue
            if deps[node.node_id].issubset(completed_node_ids):
                candidates.append(node)
                if len(candidates) >= available_slots:
                    break

        return tuple(candidates)

    def critical_path(self) -> Tuple[str, ...]:
        """Compute the longest dependency chain (critical path) across the graph (Phase 6, p. 35)."""
        deps: Dict[str, Set[str]] = {n.node_id: set(n.depends_on) for n in self.nodes}
        for edge in self.edges:
            if edge.condition in (EdgeCondition.ON_SUCCESS, EdgeCondition.ALWAYS):
                if edge.source_node_id != edge.target_node_id:
                    deps[edge.target_node_id].add(edge.source_node_id)

        longest_path_to: Dict[str, Tuple[str, ...]] = {}
        for wave in self.topological_waves():
            for nid in wave:
                parent_ids = deps[nid]
                if not parent_ids:
                    longest_path_to[nid] = (nid,)
                else:
                    best_parent = max(
                        parent_ids,
                        key=lambda pid: len(longest_path_to.get(pid, ())),
                    )
                    longest_path_to[nid] = longest_path_to[best_parent] + (nid,)

        if not longest_path_to:
            return ()
        return max(longest_path_to.values(), key=len)

    def preflight_approval_nodes(self) -> Tuple[PlanNode, ...]:
        """Statically identify all nodes requiring human permission approval (ADR-003 / ADR-010)."""
        return tuple(n for n in self.nodes if n.requires_human_approval)