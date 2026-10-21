"""NOVA Layer 4 Planning Engine Service (Phase 1 / Phase 6 / ADR-003).

Implements the Plan-and-Execute compilation and mode-selection logic from NOVA_ADR.md:
- Phase 1 Final Recommendation (p. 29): Plan-and-Execute at the orchestrator level with
  ReAct reasoning inside each node and Reflexion reserved for high-stakes/irreversible nodes.
- Phase 6 Final Recommendation (p. 36): Default to upfront Plan-and-Execute for well-understood
  tasks, and switch to incremental re-planning with per-step reflection for open-ended/exploratory
  tasks or when a prior plan step fails structurally.
"""

from __future__ import annotations

from enum import Enum

from nova.core.domain.plan import (
    EdgeCondition,
    ExecutionPlanGraph,
    PlanEdge,
    PlanNode,
)
from nova.core.domain.proposals import PlanProposal


class PlanningMode(str, Enum):
    """Execution planning strategies defined in NOVA_ADR.md Phase 6 (p. 36)."""

    UPFRONT_PLAN_AND_EXECUTE = "upfront_plan_and_execute"
    INCREMENTAL_REPLAN = "incremental_replan"


def select_planning_mode(
    is_exploratory_query: bool = False,
    has_structural_failure: bool = False,
) -> PlanningMode:
    """Select the planning strategy per Phase 6's Final Recommendation (p. 36).

    - Routine / well-structured tasks (file ops, scheduled comms) use UPFRONT_PLAN_AND_EXECUTE
      to avoid unnecessary re-planning latency.
    - Open-ended exploratory tasks or tasks recovering from a structural plan failure switch to
      INCREMENTAL_REPLAN.
    """
    if is_exploratory_query or has_structural_failure:
        return PlanningMode.INCREMENTAL_REPLAN
    return PlanningMode.UPFRONT_PLAN_AND_EXECUTE


def compile_plan_proposal(
    task_id: str,
    proposal: PlanProposal,
    max_graph_steps: int = 25,
    default_max_react_iterations: int = 5,
    default_max_retries: int = 3,
    force_reflection_all_nodes: bool = False,
) -> ExecutionPlanGraph:
    """Compile a PlannerAgent's PlanProposal into a validated ExecutionPlanGraph (ADR-003).

    Automatically:
    - Converts each SubtaskPlanStep into a bounded PlanNode assigned to its target_agent.
    - Synthesizes ON_SUCCESS control-flow edges from step dependencies.
    - Enforces requires_reflection=True on irreversible or human-approval-gated steps
      (or all steps when force_reflection_all_nodes=True in exploratory mode).
    """
    nodes = []
    edges = []

    for step in proposal.steps:
        requires_reflection = (
            force_reflection_all_nodes
            or step.estimated_sensitivity.is_irreversible
            or step.estimated_sensitivity.requires_human_approval_by_default
        )
        node = PlanNode(
            node_id=step.step_id,
            target_agent=step.target_agent,
            objective=step.objective,
            depends_on=step.depends_on,
            estimated_sensitivity=step.estimated_sensitivity,
            requires_reflection=requires_reflection,
            requires_human_approval=step.estimated_sensitivity.requires_human_approval_by_default,
            max_react_iterations=default_max_react_iterations,
            max_retries=default_max_retries,
        )
        nodes.append(node)

        for dep_id in step.depends_on:
            edges.append(
                PlanEdge(
                    source_node_id=dep_id,
                    target_node_id=step.step_id,
                    condition=EdgeCondition.ON_SUCCESS,
                )
            )

    return ExecutionPlanGraph(
        task_id=task_id,
        goal_summary=proposal.goal_summary,
        nodes=tuple(nodes),
        edges=tuple(edges),
        max_graph_steps=max_graph_steps,
    )