"""Unit tests for NOVA Task 2.3: NovaGraphOrchestrator Engine (engine.py)."""

from __future__ import annotations

import unittest

from nova.core.agents.base import BaseSpecializedAgent
from nova.core.domain.context import UntrustedDataItem
from nova.core.domain.events import EventType
from nova.core.domain.plan import PlanNodeStatus
from nova.core.domain.primitives import (
    AgentRole,
    DataClassification,
    TaskStatus,
)
from nova.core.model_serving.interface import DeterministicMockModelAdapter
from nova.core.orchestration.checkpoint import InMemoryCheckpointStore
from nova.core.orchestration.engine import NovaGraphOrchestrator
from nova.core.orchestration.ports import (
    DefaultPolicyPermissionGate,
    InMemoryMockToolExecutor,
)
from nova.core.platform.event_bus import InMemoryEventBus


class TestNovaGraphOrchestrator(unittest.TestCase):
    """Deterministic unit & integration tests for NovaGraphOrchestrator (ADR-003, ADR-004, ADR-010)."""

    def _build_two_step_plan_dict(self) -> dict[str, object]:
        return {
            "kind": "plan",
            "agent_role": "planner",
            "node_id": "node_planner",
            "thought": "Decompose invoice workflow into local read followed by email to Sam.",
            "goal_summary": "Read October invoice and email summary to Sam.",
            "steps": [
                {
                    "step_id": "step_read_invoice",
                    "target_agent": "research",
                    "objective": "Read invoices/oct_2026.pdf and extract the total.",
                    "depends_on": [],
                    "estimated_sensitivity": "read_only",
                },
                {
                    "step_id": "step_send_email",
                    "target_agent": "comms",
                    "objective": "Send invoice summary email to sam@accounting.example.com.",
                    "depends_on": ["step_read_invoice"],
                    "estimated_sensitivity": "irreversible_write",
                },
            ],
        }

    def test_end_to_end_plan_execute_pause_approve_and_complete(self) -> None:
        """Orchestrator runs Planner -> Research (auto-approved read) -> Comms (pauses) -> User Approves -> Completes."""
        bus = InMemoryEventBus()
        checkpoint_store = InMemoryCheckpointStore()
        gate = DefaultPolicyPermissionGate()
        executor = InMemoryMockToolExecutor(
            tool_handlers={
                "filesystem.read_file": lambda args: f"File {args['path']}: Vendor Acme Corp, Total $1,250.00",
                "gmail.send_email": lambda args: f"Email sent to {args['to']} with body '{args['body']}'",
            }
        )

        planner_model = DeterministicMockModelAdapter(
            scripted_outputs=[self._build_two_step_plan_dict()]
        )
        research_model = DeterministicMockModelAdapter(
            scripted_outputs=[
                {
                    "kind": "tool_call",
                    "agent_role": "research",
                    "node_id": "step_read_invoice",
                    "thought": "Read the invoice file from disk.",
                    "tool_name": "filesystem.read_file",
                    "arguments": {"path": "invoices/oct_2026.pdf"},
                    "declared_sensitivity": "read_only",
                    "rationale": "Read local invoice PDF",
                },
                {
                    "kind": "final_answer",
                    "agent_role": "research",
                    "node_id": "step_read_invoice",
                    "thought": "Extracted total $1,250.00 from tool output.",
                    "content": "Vendor: Acme Corp | Total: $1,250.00",
                    "cited_source_ids": ["doc_inv_1"],
                    "confidence": 0.99,
                },
            ]
        )
        comms_model = DeterministicMockModelAdapter(
            scripted_outputs=[
                {
                    "kind": "tool_call",
                    "agent_role": "comms",
                    "node_id": "step_send_email",
                    "thought": "Send the extracted summary to Sam.",
                    "tool_name": "gmail.send_email",
                    "arguments": {
                        "to": "sam@accounting.example.com",
                        "body": "Vendor: Acme Corp | Total: $1,250.00",
                    },
                    "declared_sensitivity": "irreversible_write",
                    "rationale": "Send October invoice summary to Sam",
                },
                {
                    "kind": "final_answer",
                    "agent_role": "comms",
                    "node_id": "step_send_email",
                    "thought": "Email was approved and sent.",
                    "content": "Dispatched invoice summary email to sam@accounting.example.com.",
                    "cited_source_ids": ["doc_inv_1"],
                    "confidence": 1.0,
                },
            ]
        )

        agents = {
            AgentRole.PLANNER: BaseSpecializedAgent(AgentRole.PLANNER, planner_model, bus),
            AgentRole.RESEARCH: BaseSpecializedAgent(AgentRole.RESEARCH, research_model, bus),
            AgentRole.COMMS: BaseSpecializedAgent(AgentRole.COMMS, comms_model, bus),
        }

        orchestrator = NovaGraphOrchestrator(
            agents=agents,
            checkpoint_store=checkpoint_store,
            permission_gate=gate,
            tool_executor=executor,
            event_bus=bus,
        )

        doc = UntrustedDataItem(
            item_id="doc_inv_1",
            source_type="local_file",
            source_uri="file:///invoices/oct_2026.pdf",
            data_classification=DataClassification.SENSITIVE_CATEGORY,
            content="Vendor: Acme Corp\nTotal: $1,250.00",
        )

        # Phase A: Start task -> Research finishes -> Comms pauses at Permission Gate!
        paused_state = orchestrator.start_task(
            task_id="task_orch_01",
            session_id="sess_orch_01",
            user_goal="Read invoices/oct_2026.pdf and email summary to Sam.",
            untrusted_data=(doc,),
        )

        self.assertEqual(paused_state.status, TaskStatus.AWAITING_APPROVAL)
        self.assertEqual(paused_state.completed_node_ids, {"step_read_invoice"})
        self.assertEqual(paused_state.awaiting_approval_node_ids, ("step_send_email",))

        pending_prop = paused_state.node_records["step_send_email"].pending_tool_proposal
        self.assertIsNotNone(pending_prop)
        assert pending_prop is not None
        self.assertEqual(pending_prop.tool_name, "gmail.send_email")
        # Only filesystem.read_file has executed so far; gmail.send_email has NOT executed yet!
        self.assertEqual([p.tool_name for p in executor.executed_proposals], ["filesystem.read_file"])

        # Phase B: User clicks "Approve" -> Comms executes gmail.send_email and completes!
        final_state = orchestrator.submit_user_approval(
            task_id="task_orch_01",
            node_id="step_send_email",
            proposal_id=pending_prop.proposal_id,
            approved=True,
            reason="User confirmed recipient Sam and amount $1,250.00",
            untrusted_data=(doc,),
        )

        self.assertEqual(final_state.status, TaskStatus.COMPLETED)
        self.assertEqual(
            final_state.completed_node_ids,
            {"step_read_invoice", "step_send_email"},
        )
        self.assertEqual(
            [p.tool_name for p in executor.executed_proposals],
            ["filesystem.read_file", "gmail.send_email"],
        )
        self.assertEqual(len(bus.events_by_type(EventType.TASK_COMPLETED)), 1)

    def test_reflection_critic_catches_wrong_recipient_and_agent_revises_before_permission_gate(self) -> None:
        """Phase 1 & Section 16.3: Critic catches wrong email recipient on Attempt 1; Comms fixes it on Attempt 2."""
        bus = InMemoryEventBus()
        checkpoint_store = InMemoryCheckpointStore()
        gate = DefaultPolicyPermissionGate()
        executor = InMemoryMockToolExecutor()

        planner_model = DeterministicMockModelAdapter(
            scripted_outputs=[
                {
                    "kind": "plan",
                    "agent_role": "planner",
                    "node_id": "node_planner",
                    "thought": "Single email step to Sam.",
                    "goal_summary": "Email summary to Sam.",
                    "steps": [
                        {
                            "step_id": "step_send_email",
                            "target_agent": "comms",
                            "objective": "Send summary to sam@accounting.example.com.",
                            "depends_on": [],
                            "estimated_sensitivity": "irreversible_write",
                        }
                    ],
                }
            ]
        )

        # Comms Agent Attempt 1 proposes wrong recipient (bob@external.example.com);
        # Attempt 2 (after critic feedback) proposes the fixed recipient (sam@accounting.example.com)!
        comms_model = DeterministicMockModelAdapter(
            scripted_outputs=[
                {
                    "kind": "tool_call",
                    "agent_role": "comms",
                    "node_id": "step_send_email",
                    "thought": "Send email to Bob.",
                    "tool_name": "gmail.send_email",
                    "arguments": {"to": "bob@external.example.com", "body": "Total $1,250"},
                    "declared_sensitivity": "irreversible_write",
                    "rationale": "Send invoice summary",
                },
                {
                    "kind": "tool_call",
                    "agent_role": "comms",
                    "node_id": "step_send_email",
                    "thought": "Fixed recipient to sam@accounting.example.com per Inspector feedback.",
                    "tool_name": "gmail.send_email",
                    "arguments": {"to": "sam@accounting.example.com", "body": "Total $1,250"},
                    "declared_sensitivity": "irreversible_write",
                    "rationale": "Send invoice summary to Sam",
                },
            ]
        )

        # Critic Agent rejects Attempt 1 ("revise") and approves Attempt 2 ("approve")
        critic_model = DeterministicMockModelAdapter(
            scripted_outputs=[
                {
                    "kind": "critique",
                    "agent_role": "planner",
                    "node_id": "step_send_email_reflect",
                    "thought": "Recipient bob@external.example.com is wrong; must be sam@accounting.example.com.",
                    "target_proposal_id": "prop_attempt_1",
                    "verdict": "revise",
                    "issues_found": ["Wrong recipient bob@external.example.com"],
                    "verification_questions_checked": ["Does recipient match Sam?"],
                    "revision_instructions": "Change 'to' argument to 'sam@accounting.example.com'.",
                },
                {
                    "kind": "critique",
                    "agent_role": "planner",
                    "node_id": "step_send_email_reflect",
                    "thought": "Recipient is now sam@accounting.example.com.",
                    "target_proposal_id": "prop_attempt_2",
                    "verdict": "approve",
                    "issues_found": [],
                    "verification_questions_checked": ["Does recipient match Sam?"],
                },
            ]
        )

        agents = {
            AgentRole.PLANNER: BaseSpecializedAgent(AgentRole.PLANNER, planner_model, bus),
            AgentRole.COMMS: BaseSpecializedAgent(AgentRole.COMMS, comms_model, bus),
        }
        critic_agent = BaseSpecializedAgent(AgentRole.PLANNER, critic_model, bus)

        orchestrator = NovaGraphOrchestrator(
            agents=agents,
            checkpoint_store=checkpoint_store,
            permission_gate=gate,
            tool_executor=executor,
            event_bus=bus,
            critic_agent=critic_agent,
        )

        state = orchestrator.start_task(
            task_id="task_reflect_fix",
            session_id="sess_reflect",
            user_goal="Email invoice summary to Sam.",
        )

        # Verify task paused at Permission Gate ONLY on the corrected proposal!
        self.assertEqual(state.status, TaskStatus.AWAITING_APPROVAL)
        rec = state.node_records["step_send_email"]
        self.assertIsNotNone(rec.pending_tool_proposal)
        assert rec.pending_tool_proposal is not None
        self.assertEqual(
            rec.pending_tool_proposal.arguments["to"],
            "sam@accounting.example.com",
        )
        # Verify reflection was triggered twice (Attempt 1 revised, Attempt 2 approved)
        self.assertEqual(len(bus.events_by_type(EventType.AGENT_REFLECTION_TRIGGERED)), 2)

    def test_user_denial_transitions_task_to_failed_without_executing_tool(self) -> None:
        """ADR-010: Clicking Deny aborts the paused task cleanly and never calls the tool executor."""
        bus = InMemoryEventBus()
        checkpoint_store = InMemoryCheckpointStore()
        gate = DefaultPolicyPermissionGate()
        executor = InMemoryMockToolExecutor()

        planner_model = DeterministicMockModelAdapter(
            scripted_outputs=[
                {
                    "kind": "plan",
                    "agent_role": "planner",
                    "node_id": "node_planner",
                    "thought": "Single email step.",
                    "goal_summary": "Email Sam.",
                    "steps": [
                        {
                            "step_id": "step_send_email",
                            "target_agent": "comms",
                            "objective": "Send email to Sam.",
                            "depends_on": [],
                            "estimated_sensitivity": "irreversible_write",
                        }
                    ],
                }
            ]
        )
        comms_model = DeterministicMockModelAdapter(
            scripted_outputs=[
                {
                    "kind": "tool_call",
                    "agent_role": "comms",
                    "node_id": "step_send_email",
                    "thought": "Propose sending email.",
                    "tool_name": "gmail.send_email",
                    "arguments": {"to": "sam@accounting.example.com", "body": "Hi"},
                    "declared_sensitivity": "irreversible_write",
                    "rationale": "Send email",
                }
            ]
        )

        orchestrator = NovaGraphOrchestrator(
            agents={
                AgentRole.PLANNER: BaseSpecializedAgent(AgentRole.PLANNER, planner_model, bus),
                AgentRole.COMMS: BaseSpecializedAgent(AgentRole.COMMS, comms_model, bus),
            },
            checkpoint_store=checkpoint_store,
            permission_gate=gate,
            tool_executor=executor,
            event_bus=bus,
        )

        paused = orchestrator.start_task(
            task_id="task_deny_01",
            session_id="sess_deny",
            user_goal="Email Sam.",
        )
        pending_prop = paused.node_records["step_send_email"].pending_tool_proposal
        assert pending_prop is not None

        denied_state = orchestrator.submit_user_approval(
            task_id="task_deny_01",
            node_id="step_send_email",
            proposal_id=pending_prop.proposal_id,
            approved=False,
            reason="Do not send this email yet",
        )

        self.assertEqual(denied_state.status, TaskStatus.FAILED)
        self.assertEqual(
            denied_state.node_records["step_send_email"].status,
            PlanNodeStatus.FAILED,
        )
        self.assertEqual(len(executor.executed_proposals), 0)
        self.assertEqual(len(bus.events_by_type(EventType.PERMISSION_DENIED)), 1)


if __name__ == "__main__":
    unittest.main()