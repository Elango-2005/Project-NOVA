"""NOVA Specialized Agent Specifications & Initial Roster (Layer 5 — ADR-008 / Section 12.3).

Codifies the exact least-privilege agent specifications from NOVA_ADR.md Section 12.3 (pp. 24-26):
- Every specialized agent has an explicit mandate, a minimal allowed tool subset,
  explicit read/write memory partition permissions (ADR-007), and a designated model tier.
- Deterministic boundary validation ensures an agent can never propose an out-of-scope tool
  or write to an unauthorized memory partition (ADR-008 / ADR-010).
"""

from __future__ import annotations

from typing import Dict, FrozenSet, Literal, Tuple

from pydantic import BaseModel, Field

try:
    from pydantic import ConfigDict

    PYDANTIC_V2 = True
except ImportError:  # pragma: no cover
    PYDANTIC_V2 = False

from nova.core.domain.primitives import AgentRole, MemoryPartition
from nova.core.domain.proposals import (
    AgentProposal,
    ProposalKind,
    ToolCallProposal,
)

ModelTier = Literal[
    "large_local_or_cloud_opt_in",
    "mid_tier_local",
    "mid_tier_local_with_vision_fallback",
    "small_latency_optimized_local",
]


class AgentScopeViolationError(PermissionError):
    """Raised when an agent attempts a tool call, proposal kind, or memory op outside its ADR-008 scope."""


class MemoryAccessScope(BaseModel):
    """Defines which memory partitions an agent may read from and write to (ADR-007 / Section 12.3)."""

    if PYDANTIC_V2:
        model_config = ConfigDict(frozen=True, extra="forbid")
    else:

        class Config:
            allow_mutation = False
            extra = "forbid"

    read_partitions: Tuple[MemoryPartition, ...] = Field(default_factory=tuple)
    write_partitions: Tuple[MemoryPartition, ...] = Field(default_factory=tuple)

    def can_read(self, partition: MemoryPartition) -> bool:
        return partition in self.read_partitions

    def can_write(self, partition: MemoryPartition) -> bool:
        return partition in self.write_partitions


class AgentSpecification(BaseModel):
    """Immutable specification and least-privilege security boundary for a specialized NOVA agent."""

    if PYDANTIC_V2:
        model_config = ConfigDict(frozen=True, extra="forbid")
    else:

        class Config:
            allow_mutation = False
            extra = "forbid"

    role: AgentRole = Field(..., description="Specialized agent identifier (ADR-008)")
    mandate: str = Field(..., min_length=1, description="Narrow domain responsibility (Section 12.3)")
    allowed_tools: Tuple[str, ...] = Field(
        default_factory=tuple,
        description="Exhaustive whitelist of MCP/native tools this agent may propose",
    )
    allowed_proposal_kinds: Tuple[ProposalKind, ...] = Field(
        default=(
            ProposalKind.TOOL_CALL,
            ProposalKind.FINAL_ANSWER,
            ProposalKind.CRITIQUE,
            ProposalKind.ASK_USER,
        ),
        description="Proposal types this agent is permitted to emit",
    )
    memory_access: MemoryAccessScope = Field(
        ..., description="Scoped read/write access across working and long-term memory partitions"
    )
    typical_model_tier: ModelTier = Field(
        ..., description="Recommended model tier per NOVA_ADR.md Section 12.3"
    )

    @property
    def allowed_tools_set(self) -> FrozenSet[str]:
        return frozenset(self.allowed_tools)

    def validate_proposal_scope(self, proposal: AgentProposal) -> None:
        """Verify that a structured proposal strictly respects this agent's least-privilege boundary."""
        if proposal.agent_role != self.role:
            raise AgentScopeViolationError(
                f"ADR-008 violation: Proposal agent_role {proposal.agent_role.value!r} "
                f"does not match executing agent {self.role.value!r}."
            )

        proposal_kind = ProposalKind(proposal.kind)
        if proposal_kind not in self.allowed_proposal_kinds:
            raise AgentScopeViolationError(
                f"ADR-008 violation: Agent {self.role.value!r} is not permitted to emit "
                f"proposal kind {proposal_kind.value!r}."
            )

        if isinstance(proposal, ToolCallProposal):
            if proposal.tool_name not in self.allowed_tools_set:
                raise AgentScopeViolationError(
                    f"ADR-008 / ADR-010 least-privilege violation: Agent {self.role.value!r} "
                    f"proposed unauthorized tool {proposal.tool_name!r}. "
                    f"Allowed tools: {list(self.allowed_tools)}."
                )

    def validate_memory_access(
        self,
        partition: MemoryPartition,
        mode: Literal["read", "write"],
    ) -> None:
        """Verify that a memory operation is within this agent's declared Section 12.3 scope."""
        if mode == "read" and not self.memory_access.can_read(partition):
            raise AgentScopeViolationError(
                f"ADR-007 / Section 12.3 violation: Agent {self.role.value!r} is not permitted "
                f"to READ from memory partition {partition.value!r}."
            )
        if mode == "write" and not self.memory_access.can_write(partition):
            raise AgentScopeViolationError(
                f"ADR-007 / Section 12.3 violation: Agent {self.role.value!r} is not permitted "
                f"to WRITE to memory partition {partition.value!r}."
            )


# ============================================================================
# Canonical Initial Agent Roster (NOVA_ADR.md Section 12.3, pp. 24-26)
# ============================================================================

PLANNER_AGENT_SPEC = AgentSpecification(
    role=AgentRole.PLANNER,
    mandate=(
        "Decompose a user goal into an execution graph; select which downstream agents "
        "handle which subgraphs; own re-planning when execution deviates from expectation."
    ),
    allowed_tools=(),  # Section 12.3 Design Note: None directly (reasoning-only)
    allowed_proposal_kinds=(
        ProposalKind.PLAN,
        ProposalKind.ASK_USER,
        ProposalKind.FINAL_ANSWER,
        ProposalKind.CRITIQUE,
    ),
    memory_access=MemoryAccessScope(
        read_partitions=(
            MemoryPartition.WORKING,
            MemoryPartition.SEMANTIC,
            MemoryPartition.EPISODIC,
        ),
        write_partitions=(),  # No direct write access
    ),
    typical_model_tier="large_local_or_cloud_opt_in",
)

RESEARCH_AGENT_SPEC = AgentSpecification(
    role=AgentRole.RESEARCH,
    mandate=(
        "Answer information-seeking subtasks using local RAG and, where permitted, web search."
    ),
    allowed_tools=(
        "filesystem.read_file",
        "filesystem.list_directory",
        "filesystem.search_files",
        "rag.retrieve",
        "search.web_search",
    ),
    memory_access=MemoryAccessScope(
        read_partitions=(
            MemoryPartition.WORKING,
            MemoryPartition.SEMANTIC,
            MemoryPartition.EPISODIC,
        ),
        write_partitions=(
            MemoryPartition.WORKING,
            MemoryPartition.SEMANTIC,
        ),
    ),
    typical_model_tier="mid_tier_local",
)

BROWSER_AGENT_SPEC = AgentSpecification(
    role=AgentRole.BROWSER,
    mandate="Execute web-based subtasks (navigate, extract, form-fill, submit) via Playwright.",
    allowed_tools=(
        "playwright.navigate",
        "playwright.extract",
        "playwright.fill_form",
        "playwright.click",
        "playwright.submit",
        "browser.fetch_readonly",
    ),
    memory_access=MemoryAccessScope(
        read_partitions=(MemoryPartition.WORKING,),
        write_partitions=(),  # Does not persist long-term memory directly
    ),
    typical_model_tier="mid_tier_local_with_vision_fallback",
)

DESKTOP_AGENT_SPEC = AgentSpecification(
    role=AgentRole.DESKTOP,
    mandate=(
        "Execute desktop application automation subtasks via accessibility APIs "
        "(and vision/coordinate fallback)."
    ),
    allowed_tools=(
        "accessibility.inspect_tree",
        "accessibility.invoke_element",
        "pyautogui.fallback_action",
        "screen.capture",
    ),
    memory_access=MemoryAccessScope(
        read_partitions=(MemoryPartition.WORKING,),
        write_partitions=(),  # No long-term memory access by default given desktop sensitivity
    ),
    typical_model_tier="mid_tier_local_with_vision_fallback",
)

CODING_AGENT_SPEC = AgentSpecification(
    role=AgentRole.CODING,
    mandate=(
        "Handle software development subtasks: read/write code, run tests, "
        "interact with version control and CI."
    ),
    allowed_tools=(
        "notes.create_note",  # Narrow local note creation; no additional path access.
        "filesystem.read_file",
        "filesystem.write_file",
        "filesystem.list_directory",
        "git.status",
        "git.diff",
        "git.commit",
        "git.branch",
        "github.read_issue",
        "github.read_pr",
        "github.create_pr",
        "docker.run_container",
        "terminal.execute",
        "vscode.inspect_workspace",
        "vscode.run_task",
    ),
    memory_access=MemoryAccessScope(
        read_partitions=(
            MemoryPartition.WORKING,
            MemoryPartition.PROCEDURAL,
            MemoryPartition.SEMANTIC,
        ),
        write_partitions=(
            MemoryPartition.WORKING,
            MemoryPartition.PROCEDURAL,
        ),
    ),
    typical_model_tier="large_local_or_cloud_opt_in",
)

COMMS_AGENT_SPEC = AgentSpecification(
    role=AgentRole.COMMS,
    mandate="Handle communication subtasks: read and draft/send email and messages.",
    allowed_tools=(
        "gmail.read_email",
        "gmail.draft_email",
        "gmail.send_email",
        "slack.read_messages",
        "slack.send_message",
        "discord.read_messages",
        "discord.send_message",
        "calendar.list_events",
        "calendar.create_event",
        "calendar.update_event",
    ),
    memory_access=MemoryAccessScope(
        read_partitions=(
            MemoryPartition.WORKING,
            MemoryPartition.EPISODIC,
        ),
        write_partitions=(
            MemoryPartition.WORKING,
            MemoryPartition.EPISODIC,
        ),
    ),
    typical_model_tier="mid_tier_local",
)

VOICE_AGENT_SPEC = AgentSpecification(
    role=AgentRole.VOICE,
    mandate=(
        "Manage the streaming STT/TTS pipeline and interrupt handling; translates between "
        "audio and the orchestrator's text-based internal representation."
    ),
    allowed_tools=(
        "voice.stt_transcribe",
        "voice.tts_synthesize",
        "voice.interrupt_stream",
    ),
    memory_access=MemoryAccessScope(
        read_partitions=(),  # No independent memory access (modality adapter)
        write_partitions=(),
    ),
    typical_model_tier="small_latency_optimized_local",
)

INITIAL_AGENT_ROSTER: Dict[AgentRole, AgentSpecification] = {
    AgentRole.PLANNER: PLANNER_AGENT_SPEC,
    AgentRole.RESEARCH: RESEARCH_AGENT_SPEC,
    AgentRole.BROWSER: BROWSER_AGENT_SPEC,
    AgentRole.DESKTOP: DESKTOP_AGENT_SPEC,
    AgentRole.CODING: CODING_AGENT_SPEC,
    AgentRole.COMMS: COMMS_AGENT_SPEC,
    AgentRole.VOICE: VOICE_AGENT_SPEC,
}


def get_agent_spec(role: AgentRole) -> AgentSpecification:
    """Return the canonical AgentSpecification for a given AgentRole."""
    return INITIAL_AGENT_ROSTER[role]