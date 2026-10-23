"""Unit tests for NOVA Task 2.1: Orchestration Task State Machine & Durable SQLite Checkpointing."""

from __future__ import annotations

import sqlite3
import tempfile
import unittest
from pathlib import Path

from nova.core.cognition.planning import compile_plan_proposal
from nova.core.domain.plan import PlanNodeStatus
from nova.core.domain.primitives import AgentRole, SensitivityTier, TaskStatus
from nova.core.domain.proposals import (
    FinalAnswerProposal,
    PlanProposal,
    SubtaskPlanStep,
    ToolCallProposal,
)
from nova.core.orchestration.checkpoint import (
    CheckpointCorruptionError,
    InMemoryCheckpointStore,
    SqliteCheckpointStore,
)
from nova.core.orchestration.state import (
    GraphStepBudgetExceededError,
    InvalidStateTransitionError,
    NodeExecutionRecord,
    TaskExecutionState,
)


class TestTaskStateAndCheckpointStore(unittest.TestCase):
    """Deterministic unit tests for state.py and checkpoint.py (ADR-003, Phase 4, NFR-09)."""

    def _build_two_step_plan(self, task_id: str = "task_201") -> PlanProposal:
        return PlanProposal(
            agent_role=AgentRole.PLANNER,
            node_id="node_planner",
            thought="Break invoice workflow into a research step and an email step.",
            goal_summary="Read invoice and email summary to Sam.",
            steps=(
                SubtaskPlanStep(
                    step_id="step_read_invoice",
                    target_agent=AgentRole.RESEARCH,
                    objective="Extract invoice total from oct_2026.pdf",
                    depends_on=(),
                    estimated_sensitivity=SensitivityTier.READ_ONLY,
                ),
                SubtaskPlanStep(
                    step_id="step_send_email",
                    target_agent=AgentRole.COMMS,
                    objective="Send summary email to Sam",
                    depends_on=("step_read_invoice",),
                    estimated_sensitivity=SensitivityTier.IRREVERSIBLE_WRITE,
                ),
            ),
        )

    def test_task_lifecycle_and_automatic_completion_when_all_nodes_complete(self) -> None:
        """Phase 4: Task progresses QUEUED -> PLANNING -> RUNNING -> AWAITING_APPROVAL -> COMPLETED."""
        state = TaskExecutionState.create_initial(
            task_id="task_201",
            session_id="sess_201",
            user_goal="Read invoice and email Sam",
        )
        self.assertEqual(state.status, TaskStatus.QUEUED)

        state = state.transition_task_status(TaskStatus.PLANNING)
        self.assertEqual(state.status, TaskStatus.PLANNING)

        plan_graph = compile_plan_proposal("task_201", self._build_two_step_plan("task_201"))
        state = state.attach_plan_graph(plan_graph)
        self.assertEqual(state.status, TaskStatus.RUNNING)
        self.assertEqual(set(state.node_records.keys()), {"step_read_invoice", "step_send_email"})

        # Complete step 1 (Research)
        step1_rec = state.node_records["step_read_invoice"]
        step1_done = NodeExecutionRecord(
            node_id=step1_rec.node_id,
            agent_role=step1_rec.agent_role,
            status=PlanNodeStatus.COMPLETED,
            attempt_count=1,
            react_state=step1_rec.react_state,
            final_answer=FinalAnswerProposal(
                agent_role=AgentRole.RESEARCH,
                node_id="step_read_invoice",
                thought="Found total $1,250.00",
                content="Vendor: Acme Corp, Total: $1,250.00",
                cited_source_ids=("doc_inv_1",),
            ),
        )
        state = state.update_node_record(step1_done)
        self.assertEqual(state.completed_node_ids, {"step_read_invoice"})
        self.assertEqual(state.status, TaskStatus.RUNNING)

        # Verify downstream step 2 receives step 1's output as working memory context
        upstream_summary = state.build_upstream_outputs_summary("step_send_email")
        self.assertIsNotNone(upstream_summary)
        self.assertIn("$1,250.00", upstream_summary or "")

        # Pause step 2 at the Permission Gate (AWAITING_APPROVAL)
        step2_rec = state.node_records["step_send_email"]
        email_proposal = ToolCallProposal(
            agent_role=AgentRole.COMMS,
            node_id="step_send_email",
            thought="Propose sending email to Sam",
            tool_name="gmail.send_email",
            arguments={"to": "sam@accounting.example.com", "body": "Total: $1,250.00"},
            declared_sensitivity=SensitivityTier.IRREVERSIBLE_WRITE,
            rationale="Email invoice summary to Sam",
        )
        step2_paused = NodeExecutionRecord(
            node_id=step2_rec.node_id,
            agent_role=step2_rec.agent_role,
            status=PlanNodeStatus.AWAITING_APPROVAL,
            attempt_count=1,
            react_state=step2_rec.react_state,
            pending_tool_proposal=email_proposal,
        )
        state = state.update_node_record(
            step2_paused,
            new_task_status=TaskStatus.AWAITING_APPROVAL,
        )
        self.assertEqual(state.status, TaskStatus.AWAITING_APPROVAL)
        self.assertEqual(state.awaiting_approval_node_ids, ("step_send_email",))

        # Resume after user approval and complete step 2 -> Task automatically becomes COMPLETED
        state = state.transition_task_status(TaskStatus.RUNNING)
        step2_done = NodeExecutionRecord(
            node_id=step2_rec.node_id,
            agent_role=step2_rec.agent_role,
            status=PlanNodeStatus.COMPLETED,
            attempt_count=1,
            react_state=step2_rec.react_state,
            final_answer=FinalAnswerProposal(
                agent_role=AgentRole.COMMS,
                node_id="step_send_email",
                thought="Email sent after user approval",
                content="Sent invoice summary to sam@accounting.example.com.",
            ),
        )
        state = state.update_node_record(step2_done)
        self.assertEqual(state.status, TaskStatus.COMPLETED)

    def test_rejects_illegal_state_transitions_from_terminal_states(self) -> None:
        """Phase 4: Once a task is COMPLETED, FAILED, or CANCELLED, it cannot transition back to RUNNING."""
        state = TaskExecutionState.create_initial(
            task_id="task_202",
            session_id="sess_202",
            user_goal="Test terminal state lock",
        )
        failed_state = state.transition_task_status(
            TaskStatus.FAILED,
            failure_reason="User aborted",
        )
        with self.assertRaises(InvalidStateTransitionError):
            failed_state.transition_task_status(TaskStatus.RUNNING)

    def test_enforces_max_graph_steps_transition_budget(self) -> None:
        """ADR-003: Exceeding max_graph_steps raises GraphStepBudgetExceededError."""
        state = TaskExecutionState.create_initial(
            task_id="task_203",
            session_id="sess_203",
            user_goal="Test step budget",
            max_graph_steps=2,
        )
        state = state.transition_task_status(TaskStatus.PLANNING)  # transition #1
        state = state.transition_task_status(TaskStatus.RUNNING)  # transition #2
        with self.assertRaises(GraphStepBudgetExceededError):
            state.transition_task_status(TaskStatus.COMPLETED)  # transition #3 -> blocked!

    def test_sqlite_checkpoint_crash_recovery_and_history(self) -> None:
        """NFR-09: SqliteCheckpointStore persists across process restarts with full ReAct & approval state."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            db_file = Path(tmp_dir) / "nova_checkpoints.db"

            # Process #1: Create task, attach plan, complete step 1, and pause at step 2
            store_1 = SqliteCheckpointStore(db_file)
            state = TaskExecutionState.create_initial(
                task_id="task_crash_test",
                session_id="sess_crash",
                user_goal="Crash recovery test",
            )
            store_1.save_checkpoint(state, node_id=None)

            plan_graph = compile_plan_proposal(
                "task_crash_test",
                self._build_two_step_plan("task_crash_test"),
            )
            state = state.attach_plan_graph(plan_graph)
            store_1.save_checkpoint(state, node_id="node_planner")

            # Simulate step 1 completion with 1 ReAct turn recorded
            rec_1 = state.node_records["step_read_invoice"]
            tool_prop = ToolCallProposal(
                agent_role=AgentRole.RESEARCH,
                node_id="step_read_invoice",
                thought="Read invoice file",
                tool_name="filesystem.read_file",
                arguments={"path": "oct_2026.pdf"},
                declared_sensitivity=SensitivityTier.READ_ONLY,
                rationale="Read invoice PDF",
            )
            updated_react = rec_1.react_state.record_observation(
                proposal=tool_prop,
                observation_text="Total Due: $1,250.00",
                status="success",
            )
            rec_1_done = NodeExecutionRecord(
                node_id=rec_1.node_id,
                agent_role=rec_1.agent_role,
                status=PlanNodeStatus.COMPLETED,
                attempt_count=1,
                react_state=updated_react,
                final_answer=FinalAnswerProposal(
                    agent_role=AgentRole.RESEARCH,
                    node_id="step_read_invoice",
                    thought="Done",
                    content="Total Due: $1,250.00",
                    cited_source_ids=("doc_1",),
                ),
            )
            state = state.update_node_record(rec_1_done)
            store_1.save_checkpoint(state, node_id="step_read_invoice")

            # Simulate Process #2 starting up after a crash and opening the same SQLite file
            store_2 = SqliteCheckpointStore(db_file)
            recovered = store_2.load_latest("task_crash_test")
            self.assertIsNotNone(recovered)
            assert recovered is not None
            self.assertEqual(recovered.checkpoint_seq, 2)
            self.assertEqual(recovered.node_id, "step_read_invoice")
            self.assertEqual(
                recovered.state.completed_node_ids,
                {"step_read_invoice"},
            )
            recovered_react = recovered.state.node_records["step_read_invoice"].react_state
            self.assertEqual(recovered_react.current_step_count, 1)
            self.assertIn("$1,250.00", recovered_react.turns[0].sanitized_observation)

            history = store_2.list_history("task_crash_test")
            self.assertEqual(len(history), 3)

    def test_detects_corrupted_or_tampered_sqlite_checkpoint(self) -> None:
        """NFR-09: Silent corruption or tampering in SQLite state_json is caught by SHA-256 verification."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            db_file = Path(tmp_dir) / "nova_tamper.db"
            store = SqliteCheckpointStore(db_file)
            state = TaskExecutionState.create_initial(
                task_id="task_tamper",
                session_id="sess_1",
                user_goal="Verify integrity check",
            )
            store.save_checkpoint(state)

            # Tamper directly with the stored JSON inside SQLite without updating sha256_checksum
            conn = sqlite3.connect(str(db_file))
            conn.execute(
                "UPDATE task_checkpoints SET state_json = REPLACE(state_json, 'Verify integrity check', 'Hacked goal') "
                "WHERE task_id = 'task_tamper';"
            )
            conn.commit()
            conn.close()

            with self.assertRaises(CheckpointCorruptionError):
                store.load_latest("task_tamper")

    def test_in_memory_checkpoint_store_parity(self) -> None:
        """InMemoryCheckpointStore satisfies CheckpointStorePort with identical sequence & hash behavior."""
        mem_store = InMemoryCheckpointStore()
        state = TaskExecutionState.create_initial(
            task_id="task_mem",
            session_id="sess_mem",
            user_goal="In-memory checkpoint test",
        )
        env1 = mem_store.save_checkpoint(state)
        state2 = state.transition_task_status(TaskStatus.PLANNING)
        env2 = mem_store.save_checkpoint(state2)

        latest = mem_store.load_latest("task_mem")
        self.assertIsNotNone(latest)
        assert latest is not None
        self.assertEqual(latest.checkpoint_seq, env2.checkpoint_seq)
        self.assertNotEqual(env1.sha256_checksum, env2.sha256_checksum)
        self.assertEqual(len(mem_store.list_history("task_mem")), 2)


if __name__ == "__main__":
    unittest.main()