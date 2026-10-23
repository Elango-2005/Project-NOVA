"""NOVA Task 2.4: Phase 2 Crash-Recovery, Pause/Resume & Diamond Fan-Out/Fan-In Evaluation Suite.

Real-Life Analogy:
This test file is the **Final Exam for Phase 2 (The Office Manager & Auto-Save System)**.
It proves three critical real-world guarantees required by NOVA_ADR.md (ADR-003, ADR-010, NFR-09):

1. The "Laptop Battery Dies While Waiting for Your Click" Test:
   - Process #1 plans a task, reads a file, and pauses at `AWAITING_APPROVAL` before sending an email.
   - Then Process #1 is completely destroyed (simulating a PC restart or power loss — all RAM is wiped).
   - Process #2 boots up fresh from the SQLite database file (`nova.db`) on disk.
   - When you click "Approve", Process #2 resumes from the exact paused step WITHOUT re-running
     the Planner or re-reading the file, sends the email, and finishes!

2. The "Diamond Flowchart (Parallel Fan-Out & Fan-In)" Test:
   - Tests a 3-step diamond graph:
       Step A (Read Q1)  \
                          +--> Step C (Coding: Combine Q1 + Q2 into summary.md)
       Step B (Read Q2)  /
   - Proves that Step C automatically receives the outputs of BOTH Step A and Step B in its
     working memory clipboard (`build_upstream_outputs_summary`).

3. The "Mid-Task Power Cut Recovery" Test:
   - Simulates a crash after Step 1 finishes (while the task is still `RUNNING`).
   - When Process #2 calls `run_until_blocked_or_complete()`, it sees Step 1 is already `COMPLETED`
     in SQLite, skips Step 1 (0 duplicate tool calls), and runs only Step 2 to finish the task.
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from nova.core.agents.base import BaseSpecializedAgent
from nova.core.domain.events import EventType
from nova.core.domain.primitives import AgentRole, TaskStatus
from nova.core.model_serving.interface import DeterministicMockModelAdapter
from nova.core.orchestration.checkpoint import SqliteCheckpointStore
from nova.core.orchestration.engine import NovaGraphOrchestrator
from nova.core.orchestration.ports import (
    DefaultPolicyPermissionGate,
    InMemoryMockToolExecutor,
)
from nova.core.platform.event_bus import InMemoryEventBus


class TestPhase2CrashRecoveryAndGraphFlows(unittest.TestCase):
    """End-to-end Phase 2 evaluation tests for SQLite crash recovery and multi-branch DAGs."""

    def test_full_process_crash_and_restart_during_approval_pause(self) -> None:
        """NFR-09 & ADR-010: Task pauses in SQLite, survives full process restart, and resumes on Approve."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            db_path = Path(tmp_dir) / "nova_production.db"

            # =========================================================================
            # PROCESS #1 (Before PC Restart / Power Loss)
            # =========================================================================
            bus_1 = InMemoryEventBus()
            store_1 = SqliteCheckpointStore(db_path)
            gate_1 = DefaultPolicyPermissionGate()
            executor_1 = InMemoryMockToolExecutor(
                tool_handlers={
                    "filesystem.read_file": lambda args: f"Read {args['path']}: Total $1,250.00",
                    "gmail.send_email": lambda args: f"Sent to {args['to']}",
                }
            )

            planner_model_1 = DeterministicMockModelAdapter(
                scripted_outputs=[
                    {
                        "kind": "plan",
                        "agent_role": "planner",
                        "node_id": "node_planner",
                        "thought": "Step 1: Read invoice. Step 2: Email Sam.",
                        "goal_summary": "Read invoice and email Sam.",
                        "steps": [
                            {
                                "step_id": "step_read",
                                "target_agent": "research",
                                "objective": "Read oct_2026.pdf",
                                "depends_on": [],
                                "estimated_sensitivity": "read_only",
                            },
                            {
                                "step_id": "step_email",
                                "target_agent": "comms",
                                "objective": "Email summary to sam@accounting.example.com",
                                "depends_on": ["step_read"],
                                "estimated_sensitivity": "irreversible_write",
                            },
                        ],
                    }
                ]
            )
            research_model_1 = DeterministicMockModelAdapter(
                scripted_outputs=[
                    {
                        "kind": "tool_call",
                        "agent_role": "research",
                        "node_id": "step_read",
                        "thought": "Read oct_2026.pdf from disk.",
                        "tool_name": "filesystem.read_file",
                        "arguments": {"path": "oct_2026.pdf"},
                        "declared_sensitivity": "read_only",
                        "rationale": "Read invoice total",
                    },
                    {
                        "kind": "final_answer",
                        "agent_role": "research",
                        "node_id": "step_read",
                        "thought": "Invoice total is $1,250.00.",
                        "content": "Acme Corp Invoice Total: $1,250.00",
                        "cited_source_ids": ["oct_2026.pdf"],
                        "confidence": 0.99,
                    },
                ]
            )
            comms_model_1 = DeterministicMockModelAdapter(
                scripted_outputs=[
                    {
                        "kind": "tool_call",
                        "agent_role": "comms",
                        "node_id": "step_email",
                        "thought": "Propose sending email to Sam.",
                        "tool_name": "gmail.send_email",
                        "arguments": {
                            "to": "sam@accounting.example.com",
                            "body": "Acme Corp Invoice Total: $1,250.00",
                        },
                        "declared_sensitivity": "irreversible_write",
                        "rationale": "Email invoice summary to Sam",
                    }
                ]
            )

            orchestrator_1 = NovaGraphOrchestrator(
                agents={
                    AgentRole.PLANNER: BaseSpecializedAgent(AgentRole.PLANNER, planner_model_1, bus_1),
                    AgentRole.RESEARCH: BaseSpecializedAgent(AgentRole.RESEARCH, research_model_1, bus_1),
                    AgentRole.COMMS: BaseSpecializedAgent(AgentRole.COMMS, comms_model_1, bus_1),
                },
                checkpoint_store=store_1,
                permission_gate=gate_1,
                tool_executor=executor_1,
                event_bus=bus_1,
            )

            paused_state = orchestrator_1.start_task(
                task_id="task_survive_reboot",
                session_id="sess_reboot_01",
                user_goal="Read oct_2026.pdf and email Sam.",
            )
            self.assertEqual(paused_state.status, TaskStatus.AWAITING_APPROVAL)
            self.assertEqual(paused_state.completed_node_ids, {"step_read"})
            pending_prop = paused_state.node_records["step_email"].pending_tool_proposal
            self.assertIsNotNone(pending_prop)
            assert pending_prop is not None
            saved_proposal_id = pending_prop.proposal_id

            # Destroy Process #1 completely (simulate power loss / reboot)
            del orchestrator_1, store_1, gate_1, executor_1, bus_1

            # =========================================================================
            # PROCESS #2 (Fresh Startup After Reboot — Reconnects to SQLite `db_path`)
            # =========================================================================
            bus_2 = InMemoryEventBus()
            store_2 = SqliteCheckpointStore(db_path)
            gate_2 = DefaultPolicyPermissionGate()
            executor_2 = InMemoryMockToolExecutor(
                tool_handlers={
                    "filesystem.read_file": lambda args: "ERROR: Should NOT re-read file after reboot!",
                    "gmail.send_email": lambda args: f"Successfully sent email to {args['to']}",
                }
            )

            # Notice: Planner and Research models have ZERO scripted outputs in Process #2!
            # If Process #2 accidentally re-runs Planner or Research, it will crash immediately.
            planner_model_2 = DeterministicMockModelAdapter(scripted_outputs=[])
            research_model_2 = DeterministicMockModelAdapter(scripted_outputs=[])
            comms_model_2 = DeterministicMockModelAdapter(
                scripted_outputs=[
                    {
                        "kind": "final_answer",
                        "agent_role": "comms",
                        "node_id": "step_email",
                        "thought": "Email was approved and sent after reboot.",
                        "content": "Email sent to sam@accounting.example.com.",
                        "cited_source_ids": ["oct_2026.pdf"],
                        "confidence": 1.0,
                    }
                ]
            )

            orchestrator_2 = NovaGraphOrchestrator(
                agents={
                    AgentRole.PLANNER: BaseSpecializedAgent(AgentRole.PLANNER, planner_model_2, bus_2),
                    AgentRole.RESEARCH: BaseSpecializedAgent(AgentRole.RESEARCH, research_model_2, bus_2),
                    AgentRole.COMMS: BaseSpecializedAgent(AgentRole.COMMS, comms_model_2, bus_2),
                },
                checkpoint_store=store_2,
                permission_gate=gate_2,
                tool_executor=executor_2,
                event_bus=bus_2,
            )

            # User clicks "Approve" in the newly restarted app!
            final_state = orchestrator_2.submit_user_approval(
                task_id="task_survive_reboot",
                node_id="step_email",
                proposal_id=saved_proposal_id,
                approved=True,
                reason="Approved after reboot",
            )

            self.assertEqual(final_state.status, TaskStatus.COMPLETED)
            self.assertEqual(final_state.completed_node_ids, {"step_read", "step_email"})
            # Process #2 ONLY executed gmail.send_email (never re-executed filesystem.read_file!)
            self.assertEqual(
                [p.tool_name for p in executor_2.executed_proposals],
                ["gmail.send_email"],
            )
            self.assertEqual(len(planner_model_2.recorded_requests), 0)
            self.assertEqual(len(research_model_2.recorded_requests), 0)

    def test_parallel_fan_out_and_fan_in_diamond_workflow(self) -> None:
        """ADR-003 & Phase 6: Two parallel research branches fan-in to a coding node."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            db_path = Path(tmp_dir) / "nova_diamond.db"
            bus = InMemoryEventBus()
            store = SqliteCheckpointStore(db_path)
            gate = DefaultPolicyPermissionGate(auto_approve_reversible_local_writes=True)
            executor = InMemoryMockToolExecutor(
                tool_handlers={
                    "filesystem.read_file": lambda args: (
                        "Q1 Revenue: $10,000" if "q1" in str(args["path"]) else "Q2 Revenue: $15,000"
                    ),
                    "filesystem.write_file": lambda args: f"Wrote {args['path']}",
                }
            )

            # Diamond graph:
            #   step_read_q1 \
            #                 +--> step_combine_report (Coding)
            #   step_read_q2 /
            planner_model = DeterministicMockModelAdapter(
                scripted_outputs=[
                    {
                        "kind": "plan",
                        "agent_role": "planner",
                        "node_id": "node_planner",
                        "thought": "Read Q1 and Q2 in parallel, then combine into a local report file.",
                        "goal_summary": "Combine Q1 and Q2 revenue into a report.",
                        "steps": [
                            {
                                "step_id": "step_read_q1",
                                "target_agent": "research",
                                "objective": "Read q1.txt",
                                "depends_on": [],
                                "estimated_sensitivity": "read_only",
                            },
                            {
                                "step_id": "step_read_q2",
                                "target_agent": "research",
                                "objective": "Read q2.txt",
                                "depends_on": [],
                                "estimated_sensitivity": "read_only",
                            },
                            {
                                "step_id": "step_combine_report",
                                "target_agent": "coding",
                                "objective": "Combine Q1 and Q2 totals and write summary.md",
                                "depends_on": ["step_read_q1", "step_read_q2"],
                                "estimated_sensitivity": "reversible_write",
                            },
                        ],
                    }
                ]
            )

            research_model = DeterministicMockModelAdapter(
                scripted_outputs=[
                    # Branch 1: step_read_q1
                    {
                        "kind": "tool_call",
                        "agent_role": "research",
                        "node_id": "step_read_q1",
                        "thought": "Read q1.txt",
                        "tool_name": "filesystem.read_file",
                        "arguments": {"path": "q1.txt"},
                        "declared_sensitivity": "read_only",
                        "rationale": "Read Q1 revenue",
                    },
                    {
                        "kind": "final_answer",
                        "agent_role": "research",
                        "node_id": "step_read_q1",
                        "thought": "Got Q1 revenue.",
                        "content": "Q1 Revenue = $10,000",
                        "cited_source_ids": ["q1.txt"],
                        "confidence": 0.99,
                    },
                    # Branch 2: step_read_q2
                    {
                        "kind": "tool_call",
                        "agent_role": "research",
                        "node_id": "step_read_q2",
                        "thought": "Read q2.txt",
                        "tool_name": "filesystem.read_file",
                        "arguments": {"path": "q2.txt"},
                        "declared_sensitivity": "read_only",
                        "rationale": "Read Q2 revenue",
                    },
                    {
                        "kind": "final_answer",
                        "agent_role": "research",
                        "node_id": "step_read_q2",
                        "thought": "Got Q2 revenue.",
                        "content": "Q2 Revenue = $15,000",
                        "cited_source_ids": ["q2.txt"],
                        "confidence": 0.99,
                    },
                ]
            )

            coding_model = DeterministicMockModelAdapter(
                scripted_outputs=[
                    {
                        "kind": "tool_call",
                        "agent_role": "coding",
                        "node_id": "step_combine_report",
                        "thought": "Write combined H1 total ($25,000) to summary.md.",
                        "tool_name": "filesystem.write_file",
                        "arguments": {
                            "path": "summary.md",
                            "content": "H1 Total Revenue: $25,000 ($10,000 + $15,000)",
                        },
                        "declared_sensitivity": "reversible_write",
                        "rationale": "Save combined revenue report locally",
                    },
                    {
                        "kind": "final_answer",
                        "agent_role": "coding",
                        "node_id": "step_combine_report",
                        "thought": "Saved summary.md.",
                        "content": "Combined report written: H1 Total = $25,000.",
                        "cited_source_ids": ["q1.txt", "q2.txt"],
                        "confidence": 1.0,
                    },
                ]
            )

            orchestrator = NovaGraphOrchestrator(
                agents={
                    AgentRole.PLANNER: BaseSpecializedAgent(AgentRole.PLANNER, planner_model, bus),
                    AgentRole.RESEARCH: BaseSpecializedAgent(AgentRole.RESEARCH, research_model, bus),
                    AgentRole.CODING: BaseSpecializedAgent(AgentRole.CODING, coding_model, bus),
                },
                checkpoint_store=store,
                permission_gate=gate,
                tool_executor=executor,
                event_bus=bus,
            )

            final_state = orchestrator.start_task(
                task_id="task_diamond_01",
                session_id="sess_diamond",
                user_goal="Combine Q1 and Q2 revenue into summary.md",
            )

            self.assertEqual(final_state.status, TaskStatus.COMPLETED)
            self.assertEqual(
                final_state.completed_node_ids,
                {"step_read_q1", "step_read_q2", "step_combine_report"},
            )
            # Verify the Coding Agent received BOTH upstream outputs in its working memory message!
            coding_first_request = coding_model.recorded_requests[0]
            user_msg = coding_first_request.messages[1]["content"]
            self.assertIn("Q1 Revenue = $10,000", user_msg)
            self.assertIn("Q2 Revenue = $15,000", user_msg)

    def test_crash_mid_graph_resumes_from_last_completed_node_without_repeating_work(self) -> None:
        """NFR-09: If a RUNNING task crashes after Step 1, calling run_until_blocked_or_complete runs only Step 2."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            db_path = Path(tmp_dir) / "nova_mid_crash.db"
            store = SqliteCheckpointStore(db_path)
            bus = InMemoryEventBus()
            gate = DefaultPolicyPermissionGate()
            executor = InMemoryMockToolExecutor()

            # Simulate a task that crashed right after `step_1` completed (while `step_2` is still PENDING)
            planner_model = DeterministicMockModelAdapter(
                scripted_outputs=[
                    {
                        "kind": "plan",
                        "agent_role": "planner",
                        "node_id": "node_planner",
                        "thought": "Two-step research workflow.",
                        "goal_summary": "Step 1 then Step 2.",
                        "steps": [
                            {
                                "step_id": "step_1",
                                "target_agent": "research",
                                "objective": "First research step",
                                "depends_on": [],
                                "estimated_sensitivity": "read_only",
                            },
                            {
                                "step_id": "step_2",
                                "target_agent": "research",
                                "objective": "Second research step",
                                "depends_on": ["step_1"],
                                "estimated_sensitivity": "read_only",
                            },
                        ],
                    }
                ]
            )
            # In Process #1, research_model only has 1 output (for step_1), and then "crashes" on step_2!
            research_model_crash = DeterministicMockModelAdapter(
                scripted_outputs=[
                    {
                        "kind": "final_answer",
                        "agent_role": "research",
                        "node_id": "step_1",
                        "thought": "Completed step 1.",
                        "content": "Step 1 Result Data",
                        "cited_source_ids": [],
                        "confidence": 0.95,
                    }
                    # Queue is empty for step_2 -> simulates an unexpected runtime failure on step_2
                ]
            )

            orchestrator_1 = NovaGraphOrchestrator(
                agents={
                    AgentRole.PLANNER: BaseSpecializedAgent(AgentRole.PLANNER, planner_model, bus),
                    AgentRole.RESEARCH: BaseSpecializedAgent(AgentRole.RESEARCH, research_model_crash, bus),
                },
                checkpoint_store=store,
                permission_gate=gate,
                tool_executor=executor,
                event_bus=bus,
            )

            orchestrator_1.start_task(
                task_id="task_mid_crash",
                session_id="sess_mid",
                user_goal="Run two steps",
            )

            # Inspect checkpoint history in SQLite:
            # Checkpoint #0: QUEUED
            # Checkpoint #1: PLANNING
            # Checkpoint #2: RUNNING (plan attached)
            # Checkpoint #3: RUNNING (step_1 COMPLETED, step_2 still PENDING!)
            history = store.list_history("task_mid_crash")
            checkpoint_after_step_1 = next(
                env
                for env in history
                if "step_1" in env.state.completed_node_ids
                and env.state.status == TaskStatus.RUNNING
            )
            # Roll SQLite back to the exact checkpoint right after step_1 finished (before step_2 crashed)
            # by saving it as the latest recovery point
            recovered_state = checkpoint_after_step_1.state
            store_2 = SqliteCheckpointStore(db_path)
            store_2.save_checkpoint(
                recovered_state.transition_task_status(TaskStatus.RUNNING),
                node_id="step_1",
            )

            # Now Process #2 starts up with ONLY step_2's scripted response!
            research_model_resume = DeterministicMockModelAdapter(
                scripted_outputs=[
                    {
                        "kind": "final_answer",
                        "agent_role": "research",
                        "node_id": "step_2",
                        "thought": "Resumed after crash and finished step 2 using step 1's output.",
                        "content": "Step 2 Final Output",
                        "cited_source_ids": [],
                        "confidence": 0.99,
                    }
                ]
            )
            orchestrator_2 = NovaGraphOrchestrator(
                agents={
                    AgentRole.PLANNER: BaseSpecializedAgent(AgentRole.PLANNER, planner_model, bus),
                    AgentRole.RESEARCH: BaseSpecializedAgent(AgentRole.RESEARCH, research_model_resume, bus),
                },
                checkpoint_store=store_2,
                permission_gate=gate,
                tool_executor=executor,
                event_bus=bus,
            )

            resumed_final = orchestrator_2.run_until_blocked_or_complete("task_mid_crash")
            self.assertEqual(resumed_final.status, TaskStatus.COMPLETED)
            self.assertEqual(resumed_final.completed_node_ids, {"step_1", "step_2"})
            # Verify step_1 was NOT re-executed (only 1 model call was made in Process #2, for step_2!)
            self.assertEqual(len(research_model_resume.recorded_requests), 1)
            self.assertEqual(research_model_resume.recorded_requests[0].node_id, "step_2")
            self.assertEqual(len(bus.events_by_type(EventType.TASK_COMPLETED)), 1)


if __name__ == "__main__":
    unittest.main()