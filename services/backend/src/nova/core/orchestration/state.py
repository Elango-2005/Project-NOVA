"""NOVA Orchestration Task & Node State Machine (Layer 6 — ADR-003 / Phase 2 / Phase 4 / Phase 6).

Real-Life Analogy:
Think of `TaskExecutionState` as the **Master Project Clipboard** carried by the Office Manager
(the Orchestrator). It tracks:
1. Where the overall task is in its lifecycle (`queued` -> `planning` -> `running` ->
   `awaiting_approval` -> `completed` / `failed` / `cancelled`).
2. The status of every single step (`PlanNode`) in the project flowchart (`NodeExecutionRecord`),
   including each agent's ReAct notepad (`ReActNodeState`), how many retries it has used,
   any tool request currently paused waiting for human approval, and the final output of each step.
3. A hard step counter (`total_transitions`) so the task can never get stuck in an infinite loop
   (enforcing ADR-003's `max_graph_steps` limit).
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Dict, Mapping, Optional, Set, Tuple

from pydantic import BaseModel, Field, PrivateAttr

try:
    from pydantic import ConfigDict, model_validator  # noqa: F401

    PYDANTIC_V2 = True
except ImportError:  # pragma: no cover
    PYDANTIC_V2 = False

from nova.core.agents.react import ReActNodeState
from nova.core.domain.plan import ExecutionPlanGraph, PlanNodeStatus
from nova.core.domain.primitives import (
    CURRENT_SCHEMA_VERSION,
    AgentRole,
    TaskStatus,
    validate_schema_version,
)
from nova.core.domain.proposals import FinalAnswerProposal, ToolCallProposal


class InvalidStateTransitionError(RuntimeError):
    """Raised when the orchestrator attempts an illegal task or node state transition."""


class GraphStepBudgetExceededError(RuntimeError):
    """Raised when a task exceeds its ADR-003 `max_graph_steps` transition ceiling."""


# Coarse task state machine allowed transitions (Phase 4, p. 33)
_ALLOWED_TASK_TRANSITIONS: Dict[TaskStatus, Set[TaskStatus]] = {
    TaskStatus.QUEUED: {
        TaskStatus.PLANNING,
        TaskStatus.RUNNING,
        TaskStatus.FAILED,
        TaskStatus.CANCELLED,
    },
    TaskStatus.PLANNING: {
        TaskStatus.RUNNING,
        TaskStatus.AWAITING_APPROVAL,
        TaskStatus.FAILED,
        TaskStatus.CANCELLED,
    },
    TaskStatus.RUNNING: {
        TaskStatus.AWAITING_APPROVAL,
        TaskStatus.PLANNING,  # Allowed when incremental re-planning is triggered (Phase 6)
        TaskStatus.COMPLETED,
        TaskStatus.FAILED,
        TaskStatus.CANCELLED,
    },
    TaskStatus.AWAITING_APPROVAL: {
        TaskStatus.RUNNING,
        TaskStatus.FAILED,
        TaskStatus.CANCELLED,
    },
    # Terminal states: cannot transition out once reached
    TaskStatus.COMPLETED: set(),
    TaskStatus.FAILED: set(),
    TaskStatus.CANCELLED: set(),
}


def validate_task_status_transition(current: TaskStatus, target: TaskStatus) -> None:
    """Verify that transitioning a task from `current` to `target` is legal (Phase 4)."""
    if current == target:
        return
    allowed = _ALLOWED_TASK_TRANSITIONS.get(current, set())
    if target not in allowed:
        raise InvalidStateTransitionError(
            f"Phase 4 state machine violation: Cannot transition task from "
            f"{current.value!r} to {target.value!r}. Allowed targets: "
            f"{sorted(s.value for s in allowed)}."
        )


class NodeExecutionRecord(BaseModel):
    """Immutable runtime status tracker for a single step (`PlanNode`) on the Project Clipboard."""

    if PYDANTIC_V2:
        model_config = ConfigDict(frozen=True, extra="forbid")
    else:

        class Config:
            allow_mutation = False
            extra = "forbid"

    node_id: str = Field(..., min_length=1)
    agent_role: AgentRole = Field(..., description="Specialized agent assigned to this step")
    status: PlanNodeStatus = Field(default=PlanNodeStatus.PENDING)
    attempt_count: int = Field(
        default=0,
        ge=0,
        description="Number of execution attempts / retries used so far (Phase 6)",
    )
    react_state: ReActNodeState = Field(
        ..., description="Bounded Thought-Action-Observation notepad for this node"
    )
    direct_action: bool = False
    approval_expires_at: Optional[float] = None
    pending_tool_proposal: Optional[ToolCallProposal] = Field(
        default=None,
        description="ToolCallProposal currently paused awaiting Reflection or Human Approval",
    )
    final_answer: Optional[FinalAnswerProposal] = Field(
        default=None,
        description="Validated final output once the node reaches COMPLETED",
    )
    last_error: Optional[str] = Field(
        default=None,
        description="Most recent error or critique revision message if retry/failure occurred",
    )


class TaskExecutionState(BaseModel):
    """Immutable snapshot of a task's entire orchestration state (saved at every checkpoint)."""

    if PYDANTIC_V2:
        model_config = ConfigDict(frozen=True, extra="forbid")
    else:

        class Config:
            allow_mutation = False
            extra = "forbid"

    schema_version: str = Field(default=CURRENT_SCHEMA_VERSION)
    task_id: str = Field(..., min_length=1)
    session_id: str = Field(..., min_length=1)
    user_goal: str = Field(..., min_length=1)
    status: TaskStatus = Field(default=TaskStatus.QUEUED)
    plan_graph: Optional[ExecutionPlanGraph] = Field(
        default=None,
        description="Compiled execution graph once planning completes",
    )
    node_records: Dict[str, NodeExecutionRecord] = Field(
        default_factory=dict,
        description="Per-node runtime state keyed by node_id",
    )
    total_transitions: int = Field(
        default=0,
        ge=0,
        description="Total node state transitions executed so far",
    )
    max_graph_steps: int = Field(
        default=25,
        ge=1,
        le=100,
        description="ADR-003 hard ceiling on total graph transitions for this task",
    )
    checkpoint_seq: int = Field(
        default=0,
        ge=0,
        description="Monotonically increasing save sequence number (0 = initial state)",
    )
    updated_at_utc: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(),
    )
    _checkpoint_origin: Optional[Tuple[int, Optional[str]]] = PrivateAttr(default=None)

    approval_attempt_id: Optional[str] = Field(default=None)
    failure_reason: Optional[str] = Field(default=None)

    def __init__(self, **data: object) -> None:
        version = data.get("schema_version", CURRENT_SCHEMA_VERSION)
        if isinstance(version, str):
            validate_schema_version(version)
        super().__init__(**data)

    @classmethod
    def create_initial(
        cls,
        task_id: str,
        session_id: str,
        user_goal: str,
        max_graph_steps: int = 25,
    ) -> "TaskExecutionState":
        """Create a fresh QUEUED task state before planning begins."""
        return cls(
            task_id=task_id,
            session_id=session_id,
            user_goal=user_goal,
            status=TaskStatus.QUEUED,
            max_graph_steps=max_graph_steps,
        )

    @property
    def completed_node_ids(self) -> Set[str]:
        """Return the set of node IDs that have finished with status COMPLETED."""
        return {
            nid
            for nid, rec in self.node_records.items()
            if rec.status == PlanNodeStatus.COMPLETED
        }

    @property
    def running_node_ids(self) -> Set[str]:
        """Return the set of node IDs currently in RUNNING state."""
        return {
            nid
            for nid, rec in self.node_records.items()
            if rec.status == PlanNodeStatus.RUNNING
        }

    @property
    def awaiting_approval_node_ids(self) -> Tuple[str, ...]:
        """Return node IDs currently paused at the Permission Gate (`AWAITING_APPROVAL`)."""
        return tuple(
            nid
            for nid, rec in self.node_records.items()
            if rec.status == PlanNodeStatus.AWAITING_APPROVAL
        )

    def attach_plan_graph(self, plan_graph: ExecutionPlanGraph) -> "TaskExecutionState":
        """Attach a compiled ExecutionPlanGraph and initialize NodeExecutionRecords for all nodes."""
        if plan_graph.task_id != self.task_id:
            raise ValueError(
                f"PlanGraph task_id {plan_graph.task_id!r} does not match "
                f"TaskExecutionState task_id {self.task_id!r}."
            )
        validate_task_status_transition(self.status, TaskStatus.RUNNING)

        records: Dict[str, NodeExecutionRecord] = {}
        for node in plan_graph.nodes:
            existing = self.node_records.get(node.node_id)
            if existing is not None and existing.status == PlanNodeStatus.COMPLETED:
                # Preserve already-completed nodes during incremental re-planning (Phase 6 / NFR-09)
                records[node.node_id] = existing
            else:
                records[node.node_id] = NodeExecutionRecord(
                    node_id=node.node_id,
                    agent_role=node.target_agent,
                    status=PlanNodeStatus.PENDING,
                    attempt_count=0,
                    react_state=ReActNodeState(
                        node_id=node.node_id,
                        max_iterations=node.max_react_iterations,
                    ),
                )

        return self._copy_with(
            status=TaskStatus.RUNNING,
            plan_graph=plan_graph,
            node_records=records,
            increment_transition=True,
        )

    def transition_task_status(
        self,
        new_status: TaskStatus,
        failure_reason: Optional[str] = None,
    ) -> "TaskExecutionState":
        """Transition coarse task status after validating Phase 4 state-machine rules."""
        validate_task_status_transition(self.status, new_status)
        return self._copy_with(
            status=new_status,
            failure_reason=failure_reason if failure_reason is not None else self.failure_reason,
            increment_transition=True,
        )

    def update_node_record(
        self,
        record: NodeExecutionRecord,
        new_task_status: Optional[TaskStatus] = None,
        failure_reason: Optional[str] = None,
    ) -> "TaskExecutionState":
        """Update a single node's execution record and enforce ADR-003 step budget."""
        if record.node_id not in self.node_records:
            raise KeyError(f"Unknown node_id {record.node_id!r} in task {self.task_id!r}.")

        target_task_status = new_task_status if new_task_status is not None else self.status
        validate_task_status_transition(self.status, target_task_status)

        updated_records = dict(self.node_records)
        updated_records[record.node_id] = record

        # If all nodes in the plan graph are now COMPLETED, automatically mark task COMPLETED
        if (
            self.plan_graph is not None
            and new_task_status is None
            and all(
                updated_records[n.node_id].status == PlanNodeStatus.COMPLETED
                for n in self.plan_graph.nodes
            )
        ):
            validate_task_status_transition(self.status, TaskStatus.COMPLETED)
            target_task_status = TaskStatus.COMPLETED

        return self._copy_with(
            status=target_task_status,
            node_records=updated_records,
            failure_reason=failure_reason if failure_reason is not None else self.failure_reason,
            increment_transition=True,
        )

    def build_upstream_outputs_summary(self, node_id: str) -> Optional[str]:
        """Gather outputs from completed upstream dependency nodes to pass as working memory."""
        if self.plan_graph is None:
            return None
        target_node = next((n for n in self.plan_graph.nodes if n.node_id == node_id), None)
        if target_node is None or not target_node.depends_on:
            return None

        sections = []
        for dep_id in target_node.depends_on:
            dep_rec = self.node_records.get(dep_id)
            if dep_rec is not None and dep_rec.final_answer is not None:
                sections.append(
                    f"[Upstream Step '{dep_id}' ({dep_rec.agent_role.value}) Output]:\n"
                    f"{dep_rec.final_answer.content}"
                )
        return "\n\n".join(sections) if sections else None

    def _copy_with(
        self,
        *,
        status: Optional[TaskStatus] = None,
        plan_graph: Optional[ExecutionPlanGraph] = None,
        node_records: Optional[Mapping[str, NodeExecutionRecord]] = None,
        failure_reason: Optional[str] = None,
        increment_transition: bool = False,
    ) -> "TaskExecutionState":
        next_transitions = (
            self.total_transitions + 1 if increment_transition else self.total_transitions
        )
        if next_transitions > self.max_graph_steps:
            raise GraphStepBudgetExceededError(
                f"ADR-003 violation: Task {self.task_id!r} exceeded max_graph_steps="
                f"{self.max_graph_steps} (attempted transition #{next_transitions})."
            )

        updated = TaskExecutionState(
            schema_version=self.schema_version,
            task_id=self.task_id,
            session_id=self.session_id,
            user_goal=self.user_goal,
            status=status if status is not None else self.status,
            plan_graph=plan_graph if plan_graph is not None else self.plan_graph,
            node_records=(
                dict(node_records) if node_records is not None else dict(self.node_records)
            ),
            total_transitions=next_transitions,
            max_graph_steps=self.max_graph_steps,
            checkpoint_seq=self.checkpoint_seq + 1,
            updated_at_utc=datetime.now(timezone.utc).isoformat(),
            approval_attempt_id=self.approval_attempt_id,
            failure_reason=failure_reason,
        )
        updated._checkpoint_origin = self._checkpoint_origin
        return updated
