"""Unit tests for NOVA Task 1.3: Structured Action Proposals & Specialized Agent Roster Contracts."""

from __future__ import annotations

import json
import unittest

from nova.core.domain.agent_spec import (
    AgentScopeViolationError,
    INITIAL_AGENT_ROSTER,
    get_agent_spec,
)
from nova.core.domain.primitives import (
    AgentRole,
    MemoryPartition,
    SensitivityTier,
)
from nova.core.domain.proposals import (
    ClarificationProposal,
    CritiqueProposal,
    FinalAnswerProposal,
    PlanProposal,
    ProposalValidationError,
    SubtaskPlanStep,
    ToolCallProposal,
    parse_agent_proposal,
)


class TestProposalsAndAgentRoster(unittest.TestCase):
    """Deterministic unit tests for proposals.py and agent_spec.py."""

    def test_rejects_raw_free_text_model_output(self) -> None:
        """ADR-004: Free-form unstructured model text must never be accepted as an action proposal."""
        with self.assertRaises(ProposalValidationError):
            parse_agent_proposal("I will now read the file invoice.pdf for you.")

    def test_valid_tool_call_proposal_and_default_fail_closed_sensitivity(self) -> None:
        """ADR-004 / ADR-010: ToolCallProposal requires namespace.action format and defaults to IRREVERSIBLE_WRITE."""
        raw = json.dumps(
            {
                "kind": "tool_call",
                "agent_role": "research",
                "node_id": "node_1",
                "thought": "Need to read the invoice PDF from local disk.",
                "tool_name": "filesystem.read_file",
                "arguments": {"path": "invoices/oct_2026.pdf"},
                "rationale": "Read the user's October 2026 invoice to extract the total.",
            }
        )
        proposal = parse_agent_proposal(raw)
        self.assertIsInstance(proposal, ToolCallProposal)
        assert isinstance(proposal, ToolCallProposal)
        self.assertEqual(proposal.tool_name, "filesystem.read_file")
        self.assertEqual(proposal.declared_sensitivity, SensitivityTier.IRREVERSIBLE_WRITE)

    def test_rejects_malformed_tool_name_or_empty_thought(self) -> None:
        """Tool names without a namespace or proposals without a reasoning trace fail closed."""
        with self.assertRaises(ProposalValidationError):
            ToolCallProposal(
                agent_role=AgentRole.CODING,
                node_id="n1",
                thought="Run shell command",
                tool_name="rm -rf /",  # Invalid tool name format
                arguments={},
                rationale="Clean temp files",
            )

        with self.assertRaises(ProposalValidationError):
            FinalAnswerProposal(
                agent_role=AgentRole.RESEARCH,
                node_id="n1",
                thought="   ",  # Blank thought trace forbidden
                content="The total is $420.",
            )

    def test_plan_proposal_validates_unique_steps_and_dependencies(self) -> None:
        """ADR-003: PlanProposal rejects self-dependencies and unknown step references."""
        valid_plan = PlanProposal(
            agent_role=AgentRole.PLANNER,
            node_id="plan_root",
            thought="Break invoice reconciliation into research and email steps.",
            goal_summary="Summarize invoice and draft email to accounting.",
            steps=(
                SubtaskPlanStep(
                    step_id="step_1",
                    target_agent=AgentRole.RESEARCH,
                    objective="Extract total from invoice.pdf",
                    depends_on=(),
                    estimated_sensitivity=SensitivityTier.READ_ONLY,
                ),
                SubtaskPlanStep(
                    step_id="step_2",
                    target_agent=AgentRole.COMMS,
                    objective="Send summary email to accounting",
                    depends_on=("step_1",),
                    estimated_sensitivity=SensitivityTier.IRREVERSIBLE_WRITE,
                ),
            ),
        )
        self.assertEqual(len(valid_plan.steps), 2)

        # Unknown dependency must fail closed
        with self.assertRaises(ProposalValidationError):
            PlanProposal(
                agent_role=AgentRole.PLANNER,
                node_id="plan_root",
                thought="Invalid dependency test",
                goal_summary="Test",
                steps=(
                    SubtaskPlanStep(
                        step_id="step_1",
                        target_agent=AgentRole.RESEARCH,
                        objective="Extract total",
                        depends_on=("non_existent_step",),
                    ),
                ),
            )

    def test_critique_proposal_requires_revision_instructions_when_verdict_is_revise(self) -> None:
        """Phase 1 Reflexion: verdict='revise' requires explicit revision_instructions."""
        with self.assertRaises(ProposalValidationError):
            CritiqueProposal(
                agent_role=AgentRole.PLANNER,
                node_id="reflect_1",
                thought="The proposed email recipient does not match the user request.",
                target_proposal_id="prop_123",
                verdict="revise",
                issues_found=("Wrong recipient email address",),
                revision_instructions=None,
            )

        valid_critique = CritiqueProposal(
            agent_role=AgentRole.PLANNER,
            node_id="reflect_1",
            thought="The proposed email recipient does not match the user request.",
            target_proposal_id="prop_123",
            verdict="revise",
            issues_found=("Wrong recipient email address",),
            verification_questions_checked=("Does recipient match Sam's address in memory?",),
            revision_instructions="Change recipient to sam@accounting.example.com.",
        )
        self.assertEqual(valid_critique.verdict, "revise")

    def test_initial_agent_roster_covers_all_seven_adr_roles(self) -> None:
        """Section 12.3: All 7 specialized agents are registered with immutable specs."""
        self.assertEqual(set(INITIAL_AGENT_ROSTER.keys()), set(AgentRole))

    def test_planner_agent_cannot_propose_any_tool_call_or_write_memory(self) -> None:
        """Section 12.3 Design Note: Planner is reasoning-only (0 tools) and has read-only memory."""
        planner_spec = get_agent_spec(AgentRole.PLANNER)
        self.assertEqual(planner_spec.allowed_tools, ())

        tool_proposal = ToolCallProposal(
            agent_role=AgentRole.PLANNER,
            node_id="n1",
            thought="Try to read a file directly from Planner",
            tool_name="filesystem.read_file",
            arguments={"path": "notes.txt"},
            rationale="Shortcut",
        )
        with self.assertRaises(AgentScopeViolationError):
            planner_spec.validate_proposal_scope(tool_proposal)

        # Planner can read semantic/episodic memory, but cannot write to any long-term partition
        planner_spec.validate_memory_access(MemoryPartition.SEMANTIC, "read")
        planner_spec.validate_memory_access(MemoryPartition.EPISODIC, "read")
        with self.assertRaises(AgentScopeViolationError):
            planner_spec.validate_memory_access(MemoryPartition.SEMANTIC, "write")

    def test_research_agent_blocks_destructive_or_out_of_scope_tools(self) -> None:
        """ADR-008 / Section 16.3: Research agent proposing delete_file or terminal.execute is blocked."""
        research_spec = get_agent_spec(AgentRole.RESEARCH)

        allowed = ToolCallProposal(
            agent_role=AgentRole.RESEARCH,
            node_id="n_res",
            thought="Read invoice PDF",
            tool_name="filesystem.read_file",
            arguments={"path": "invoice.pdf"},
            declared_sensitivity=SensitivityTier.READ_ONLY,
            rationale="Read local invoice",
        )
        research_spec.validate_proposal_scope(allowed)

        forbidden = ToolCallProposal(
            agent_role=AgentRole.RESEARCH,
            node_id="n_res",
            thought="Injected instruction trying to run shell command",
            tool_name="terminal.execute",
            arguments={"command": "whoami"},
            rationale="Malicious injection",
        )
        with self.assertRaises(AgentScopeViolationError):
            research_spec.validate_proposal_scope(forbidden)

    def test_desktop_and_voice_agent_memory_isolation(self) -> None:
        """Section 12.3: Desktop has working-memory read only; Voice has zero independent memory access."""
        desktop_spec = get_agent_spec(AgentRole.DESKTOP)
        desktop_spec.validate_memory_access(MemoryPartition.WORKING, "read")
        with self.assertRaises(AgentScopeViolationError):
            desktop_spec.validate_memory_access(MemoryPartition.SEMANTIC, "read")
        with self.assertRaises(AgentScopeViolationError):
            desktop_spec.validate_memory_access(MemoryPartition.EPISODIC, "write")

        voice_spec = get_agent_spec(AgentRole.VOICE)
        for partition in MemoryPartition:
            with self.assertRaises(AgentScopeViolationError):
                voice_spec.validate_memory_access(partition, "read")
            with self.assertRaises(AgentScopeViolationError):
                voice_spec.validate_memory_access(partition, "write")


if __name__ == "__main__":
    unittest.main()