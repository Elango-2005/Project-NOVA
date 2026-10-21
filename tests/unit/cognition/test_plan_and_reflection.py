"""Unit tests for NOVA Task 1.4: Plan-and-Execute & Conditional Reflexion Contracts."""

from __future__ import annotations

import unittest

from nova.core.cognition.planning import (
    PlanningMode,
    compile_plan_proposal,
    select_planning_mode,
)
from nova.core.cognition.reflection import (
    ReflectionTriggerReason,
    evaluate_reflection_trigger,
)
from nova.core.domain.plan import (
    EdgeCondition,
    ExecutionPlanGraph,
    PlanEdge,
    PlanGraphValidationError,
    PlanNode,
)
from nova.core.domain.primitives import AgentRole, SensitivityTier
from nova.core.domain.proposals import (
    FinalAnswerProposal,
    PlanProposal,
    SubtaskPlanStep,
    ToolCallProposal,
)


class TestPlanAndReflectionContracts(unittest.TestCase):
    """Deterministic unit tests for plan.py, cognition/planning.py, and cognition/reflection.py."""

    def test_compile_plan_proposal_and_topological_waves(self) -> None:
        """ADR-003 / Phase 6: Parallel branches group into waves; critical path is computed accurately."""
        proposal = PlanProposal(
            agent_role=AgentRole.PLANNER,
            node_id="planner_root",
            thought="Fan out to summarize two files concurrently, then reconcile and email.",
            goal_summary="Reconcile two invoices and email accounting.",
            steps=(
                SubtaskPlanStep(
                    step_id="read_inv_1",
                    target_agent=AgentRole.RESEARCH,
                    objective="Summarize invoice 1",
                    depends_on=(),
                    estimated_sensitivity=SensitivityTier.READ_ONLY,
                ),
                SubtaskPlanStep(
                    step_id="read_inv_2",
                    target_agent=AgentRole.RESEARCH,
                    objective="Summarize invoice 2",
                    depends_on=(),
                    estimated_sensitivity=SensitivityTier.READ_ONLY,
                ),
                SubtaskPlanStep(
                    step_id="reconcile_code",
                    target_agent=AgentRole.CODING,
                    objective="Compare totals and update local tracker",
                    depends_on=("read_inv_1", "read_inv_2"),
                    estimated_sensitivity=SensitivityTier.REVERSIBLE_WRITE,
                ),
                SubtaskPlanStep(
                    step_id="send_summary_email",
                    target_agent=AgentRole.COMMS,
                    objective="Email discrepancy summary to accounting",
                    depends_on=("reconcile_code",),
                    estimated_sensitivity=SensitivityTier.IRREVERSIBLE_WRITE,
                ),
            ),
        )

        graph = compile_plan_proposal(task_id="task_invoice_01", proposal=proposal)

        # Wave 0 runs read_inv_1 and read_inv_2 in parallel; Wave 1 runs reconcile; Wave 2 runs email
        self.assertEqual(
            graph.topological_waves(),
            (
                ("read_inv_1", "read_inv_2"),
                ("reconcile_code",),
                ("send_summary_email",),
            ),
        )

        # Critical path has length 3
        c_path = graph.critical_path()
        self.assertEqual(len(c_path), 3)
        self.assertEqual(c_path[-2:], ("reconcile_code", "send_summary_email"))

        # Static permission pre-flighting (ADR-003 / ADR-010) catches only send_summary_email
        approval_nodes = graph.preflight_approval_nodes()
        self.assertEqual(len(approval_nodes), 1)
        self.assertEqual(approval_nodes[0].node_id, "send_summary_email")
        self.assertTrue(approval_nodes[0].requires_reflection)
        self.assertTrue(approval_nodes[0].requires_human_approval)

    def test_ready_nodes_respects_hardware_concurrency_cap(self) -> None:
        """NFR-11: ready_nodes caps concurrent runnable nodes to max_concurrency."""
        nodes = tuple(
            PlanNode(
                node_id=f"doc_{i}",
                target_agent=AgentRole.RESEARCH,
                objective=f"Summarize doc {i}",
            )
            for i in range(5)
        )
        graph = ExecutionPlanGraph(
            task_id="task_parallel",
            goal_summary="Summarize 5 docs",
            nodes=nodes,
        )
        # With max_concurrency=2 and 0 running, only 2 nodes are returned
        batch = graph.ready_nodes(completed_node_ids=set(), running_node_ids=set(), max_concurrency=2)
        self.assertEqual([n.node_id for n in batch], ["doc_0", "doc_1"])

        # When 2 are already running and max_concurrency=2, 0 new nodes are returned
        batch_full = graph.ready_nodes(
            completed_node_ids=set(),
            running_node_ids={"doc_0", "doc_1"},
            max_concurrency=2,
        )
        self.assertEqual(batch_full, ())

    def test_detects_and_rejects_dependency_cycles_and_step_cap(self) -> None:
        """ADR-003: Circular ON_SUCCESS edges and graphs exceeding max_graph_steps are rejected."""
        n1 = PlanNode(node_id="n1", target_agent=AgentRole.RESEARCH, objective="Step 1")
        n2 = PlanNode(node_id="n2", target_agent=AgentRole.RESEARCH, objective="Step 2")

        with self.assertRaises(PlanGraphValidationError):
            ExecutionPlanGraph(
                task_id="task_cycle",
                goal_summary="Cyclic graph",
                nodes=(n1, n2),
                edges=(
                    PlanEdge(source_node_id="n1", target_node_id="n2", condition=EdgeCondition.ON_SUCCESS),
                    PlanEdge(source_node_id="n2", target_node_id="n1", condition=EdgeCondition.ON_SUCCESS),
                ),
            )

        with self.assertRaises(PlanGraphValidationError):
            ExecutionPlanGraph(
                task_id="task_too_large",
                goal_summary="Exceeds cap",
                nodes=(n1, n2),
                max_graph_steps=1,
            )

    def test_select_planning_mode_follows_phase_6_recommendation(self) -> None:
        """Phase 6 Final Recommendation: upfront Plan-and-Execute by default; incremental for exploratory/failures."""
        self.assertEqual(
            select_planning_mode(is_exploratory_query=False, has_structural_failure=False),
            PlanningMode.UPFRONT_PLAN_AND_EXECUTE,
        )
        self.assertEqual(
            select_planning_mode(is_exploratory_query=True, has_structural_failure=False),
            PlanningMode.INCREMENTAL_REPLAN,
        )
        self.assertEqual(
            select_planning_mode(is_exploratory_query=False, has_structural_failure=True),
            PlanningMode.INCREMENTAL_REPLAN,
        )

    def test_reflection_gate_bypasses_routine_reads_and_triggers_on_irreversible_actions(self) -> None:
        """Phase 1 Final Recommendation: Reflexion is skipped on routine read-only tools and enforced on irreversible tools."""
        read_proposal = ToolCallProposal(
            agent_role=AgentRole.RESEARCH,
            node_id="node_read",
            thought="Read the local invoice file.",
            tool_name="filesystem.read_file",
            arguments={"path": "invoice.pdf"},
            declared_sensitivity=SensitivityTier.READ_ONLY,
            rationale="Read invoice total",
        )
        read_decision = evaluate_reflection_trigger(read_proposal)
        self.assertFalse(read_decision.should_reflect)
        self.assertEqual(read_decision.reason, ReflectionTriggerReason.NOT_REQUIRED)
        self.assertEqual(read_decision.verification_questions, ())

        send_email_proposal = ToolCallProposal(
            agent_role=AgentRole.COMMS,
            node_id="node_send",
            thought="Send the summary email to accounting.",
            tool_name="gmail.send_email",
            arguments={"to": "sam@example.com", "body": "Total is $1,250"},
            declared_sensitivity=SensitivityTier.IRREVERSIBLE_WRITE,
            rationale="Send invoice summary email",
        )
        send_decision = evaluate_reflection_trigger(send_email_proposal)
        self.assertTrue(send_decision.should_reflect)
        self.assertEqual(
            send_decision.reason,
            ReflectionTriggerReason.IRREVERSIBLE_OR_HIGH_RISK_ACTION,
        )
        self.assertGreaterEqual(len(send_decision.verification_questions), 2)

    def test_reflection_gate_catches_missing_citations_and_low_confidence(self) -> None:
        """Phase 9 & Phase 1: Un-cited grounded answers or low-confidence answers trigger reflection."""
        uncited_answer = FinalAnswerProposal(
            agent_role=AgentRole.RESEARCH,
            node_id="node_ans",
            thought="Synthesized answer from PDF.",
            content="The invoice total is $1,250.00.",
            cited_source_ids=(),
            confidence=0.95,
        )
        citation_decision = evaluate_reflection_trigger(uncited_answer, require_citations=True)
        self.assertTrue(citation_decision.should_reflect)
        self.assertEqual(
            citation_decision.reason,
            ReflectionTriggerReason.MISSING_REQUIRED_CITATIONS,
        )

        cited_low_conf = FinalAnswerProposal(
            agent_role=AgentRole.RESEARCH,
            node_id="node_ans",
            thought="OCR was blurry on line 4.",
            content="The invoice total appears to be $1,250.00.",
            cited_source_ids=("doc_1",),
            confidence=0.55,
        )
        conf_decision = evaluate_reflection_trigger(
            cited_low_conf,
            require_citations=True,
            confidence_threshold=0.75,
        )
        self.assertTrue(conf_decision.should_reflect)
        self.assertEqual(conf_decision.reason, ReflectionTriggerReason.LOW_CONFIDENCE)

        cited_high_conf = FinalAnswerProposal(
            agent_role=AgentRole.RESEARCH,
            node_id="node_ans",
            thought="Clear extraction from doc_1.",
            content="The invoice total is $1,250.00.",
            cited_source_ids=("doc_1",),
            confidence=0.95,
        )
        pass_decision = evaluate_reflection_trigger(cited_high_conf, require_citations=True)
        self.assertFalse(pass_decision.should_reflect)


if __name__ == "__main__":
    unittest.main()