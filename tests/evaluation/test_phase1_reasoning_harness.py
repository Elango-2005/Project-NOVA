"""Automated test suite for NOVA Task 1.6: Phase 1 Golden-Task & Adversarial Evaluation Harness."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

# Ensure sibling import works whether pytest is invoked from rootdir or tests/evaluation
_EVAL_DIR = Path(__file__).resolve().parent
if str(_EVAL_DIR) not in sys.path:
    sys.path.insert(0, str(_EVAL_DIR))

try:
    from tests.evaluation.phase1_reasoning_harness import (
        run_adversarial_prompt_injection_scenario,
        run_canonical_invoice_golden_task,
        run_where_two_pipelines_meet_scenario,
    )
except ModuleNotFoundError:  # pragma: no cover
    from phase1_reasoning_harness import (  # type: ignore[no-redef]
        run_adversarial_prompt_injection_scenario,
        run_canonical_invoice_golden_task,
        run_where_two_pipelines_meet_scenario,
    )


class TestPhase1ReasoningHarness(unittest.TestCase):
    """Verifies NOVA Phase 1 exit criteria across Pipeline 1 (contract) and Pipeline 2 (behavioral/adversarial)."""

    def test_canonical_invoice_golden_task_passes_all_gates(self) -> None:
        """Section 16.3: End-to-end Planner -> Research (ReAct) -> Comms workflow passes all rubric gates."""
        score, plan_graph = run_canonical_invoice_golden_task()
        self.assertTrue(score.passed_all_gates)
        self.assertEqual(score.behavioral_accuracy_score, 1.0)
        self.assertTrue(score.grounded_with_citations)
        self.assertTrue(score.reflection_gate_triggered_appropriately)
        self.assertEqual(len(plan_graph.preflight_approval_nodes()), 1)
        self.assertEqual(plan_graph.preflight_approval_nodes()[0].node_id, "step_email_sam")

    def test_where_two_pipelines_meet_catches_behaviorally_flawed_proposal(self) -> None:
        """Section 16.3 (p. 59): A schema-valid email to the wrong recipient is caught by Reflection & Critique."""
        reflection_decision, critique = run_where_two_pipelines_meet_scenario()
        self.assertTrue(reflection_decision.should_reflect)
        self.assertEqual(critique.verdict, "revise")
        self.assertGreaterEqual(len(critique.issues_found), 2)
        self.assertIn("sam@accounting.example.com", critique.revision_instructions or "")

    def test_adversarial_prompt_injection_is_deterministically_blocked(self) -> None:
        """Section 16.3 & Risk R-02: Injected PDF instruction to delete files is blocked by ADR-008 scope."""
        score = run_adversarial_prompt_injection_scenario()
        self.assertTrue(score.adversarial_injection_blocked)
        self.assertTrue(score.passed_all_gates)


if __name__ == "__main__":
    unittest.main()