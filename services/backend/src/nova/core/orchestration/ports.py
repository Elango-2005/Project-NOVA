"""NOVA Orchestration Permission Gate & Tool Execution Ports (Layer 6 / Layer 3 — ADR-005 / ADR-006 / ADR-010).

Real-Life Analogy:
Imagine the Office Manager (the Orchestrator) sitting at the central desk. When an employee
(an Agent) fills out a Tool Request Slip (`ToolCallProposal`), the Manager NEVER runs the tool
directly. Instead, the Manager uses two walkie-talkies:

1. Walkie-Talkie #1 — `PermissionGatePort` (The Security Guard at the Vault Door):
   - Checks the employee's ID badge (`AgentSpecification`) and the danger level (`SensitivityTier`).
   - Safe actions (`read_only`) get stamped `AUTO_APPROVED` immediately.
   - Risky or permanent actions (`irreversible_write`, like sending an email or deleting a file)
     get stamped `requires_interactive_prompt = True`, which tells the Office Manager to PAUSE
     the task (`AWAITING_APPROVAL`), save a checkpoint to SQLite, and ask YOU for permission.

2. Walkie-Talkie #2 — `ToolExecutorPort` (The Tool Room Operator):
   - Refuses to touch any tool unless you hand it a valid, matching approval stamp
     (`PermissionCheckResult` with `is_approved = True`). If someone tries to run a tool without
     an approval stamp, it raises `UnapprovedToolExecutionError` immediately.
"""

from __future__ import annotations

from typing import Any, TypeVar, Callable, Dict, List, Literal, Mapping, Optional, Protocol, Tuple, runtime_checkable

from pydantic import BaseModel, Field

try:
    from pydantic import ConfigDict, model_validator  # noqa: F401

    PYDANTIC_V2 = True
except ImportError:  # pragma: no cover
    PYDANTIC_V2 = False

from nova.core.domain.agent_spec import AgentScopeViolationError, get_agent_spec
from nova.core.domain.context import sanitize_untrusted_text
from nova.core.domain.primitives import (
    AgentRole,
    PermissionDecisionKind,
    SensitivityTier,
)
from nova.core.domain.proposals import ToolCallProposal


class UnapprovedToolExecutionError(PermissionError):
    """Raised when code attempts to execute a tool without a verified approval stamp (ADR-010)."""


class AuthorizationAlreadyClaimedError(UnapprovedToolExecutionError):
    """An execution winner exists; its outcome must not be replaced."""


ModelT = TypeVar("ModelT", bound=BaseModel)


def copy_model(model: ModelT, **updates: Any) -> ModelT:
    """Preserve the repository's Pydantic 1/2 compatibility for immutable updates."""
    if hasattr(model, "model_copy"):
        return model.model_copy(update=updates)
    return model.copy(update=updates)


class ToolActionContext(BaseModel):
    """Identity supplied by the runtime, never accepted from a model or HTTP body."""
    if PYDANTIC_V2:
        model_config = ConfigDict(frozen=True, extra="forbid")
    else:
        class Config:
            allow_mutation = False
            extra = "forbid"
    task_id: str = Field(..., min_length=1)
    session_id: str = Field(..., min_length=1)
    node_id: str = Field(..., min_length=1)
    agent_role: AgentRole


class PermissionCheckResult(BaseModel):
    """Official decision stamp returned by the Permission Gate (ADR-010 / NFR-08)."""

    if PYDANTIC_V2:
        model_config = ConfigDict(frozen=True, extra="forbid")
    else:

        class Config:
            allow_mutation = False
            extra = "forbid"

    proposal_id: str = Field(..., min_length=1, description="ID of the evaluated ToolCallProposal")
    tool_name: str = Field(..., min_length=1)
    agent_role: AgentRole = Field(...)
    effective_sensitivity: SensitivityTier = Field(
        ..., description="Sensitivity tier enforced by the gate"
    )
    decision: Optional[PermissionDecisionKind] = Field(
        default=None,
        description="Final decision kind, or None when paused waiting for human input",
    )
    requires_interactive_prompt: bool = Field(
        default=False,
        description="True when the orchestrator must pause in AWAITING_APPROVAL and ask the user",
    )
    reason: str = Field(..., min_length=1, description="Audit explanation for this decision")
    context: Optional[ToolActionContext] = None
    action_digest: Optional[str] = None
    authorization_id: Optional[str] = None
    expires_at: Optional[float] = None

    @property
    def is_approved(self) -> bool:
        """Return True ONLY if the tool call is explicitly approved to execute right now."""
        return (
            not self.requires_interactive_prompt
            and self.decision
            in (
                PermissionDecisionKind.AUTO_APPROVED,
                PermissionDecisionKind.USER_APPROVED,
            )
        )

    @property
    def is_denied(self) -> bool:
        """Return True if the tool call was rejected by policy, scope, or the user."""
        return (
            not self.requires_interactive_prompt
            and self.decision
            in (
                PermissionDecisionKind.DENIED_BY_POLICY,
                PermissionDecisionKind.DENIED_BY_USER,
                PermissionDecisionKind.DENIED_OUT_OF_SCOPE,
            )
        )


class ToolExecutionResult(BaseModel):
    """Sanitized result envelope returned by a tool execution (ADR-005 / Phase 15 Layer 1)."""

    if PYDANTIC_V2:
        model_config = ConfigDict(frozen=True, extra="forbid")
    else:

        class Config:
            allow_mutation = False
            extra = "forbid"

    proposal_id: str = Field(..., min_length=1)
    tool_name: str = Field(..., min_length=1)
    status: Literal["success", "error"] = Field(default="success")
    sanitized_output: str = Field(
        ...,
        description="Delimiter-escaped tool output safe to place inside <untrusted_data>",
    )
    execution_latency_ms: int = Field(default=5, ge=0)

    def __init__(self, **data: object) -> None:
        raw_out = data.get("sanitized_output", "")
        if isinstance(raw_out, str):
            data = dict(data)
            data["sanitized_output"] = sanitize_untrusted_text(raw_out)
        super().__init__(**data)


@runtime_checkable
class PermissionGatePort(Protocol):
    """Abstract interface for NOVA's non-bypassable Permission Gate (ADR-006 / ADR-010)."""

    def evaluate_tool_proposal(
        self,
        proposal: ToolCallProposal,
        *, context: Optional[ToolActionContext] = None,
    ) -> PermissionCheckResult:
        """Evaluate whether `proposal` is auto-approved, denied, or requires user confirmation."""

    def record_user_decision(
        self,
        proposal_id: str,
        approved: bool,
        reason: str = "User interactive decision",
        *, proposal: Optional[ToolCallProposal] = None, context: Optional[ToolActionContext] = None,
    ) -> None:
        """Record the human user's Approve/Deny click for a paused `proposal_id`."""


@runtime_checkable
class ToolExecutorPort(Protocol):
    """Abstract interface for executing approved MCP or native tools (ADR-005 / ADR-006)."""

    def execute_approved_tool(
        self,
        proposal: ToolCallProposal,
        permission_stamp: PermissionCheckResult,
        *, context: Optional[ToolActionContext] = None,
        on_started: Optional[Callable[[], None]] = None,
    ) -> ToolExecutionResult:
        """Execute `proposal` ONLY after verifying `permission_stamp.is_approved` is True."""


class DefaultPolicyPermissionGate:
    """Deterministic, fail-closed Permission Gate enforcing ADR-008 & ADR-010 rules.

    How it works:
    1. Checks the agent's ID badge (`get_agent_spec`) to make sure the tool is allowed for that agent.
    2. Checks if the human user already clicked Approve or Deny for this `proposal_id`.
    3. Looks up the tool's true danger level (`effective_sensitivity`) from a trusted catalog
       (falling back to the proposal's declared sensitivity, or `IRREVERSIBLE_WRITE` if unknown).
    4. Auto-approves safe local `read_only` tools, and pauses to ask the user for risky/irreversible tools.
    """

    def __init__(
        self,
        tool_sensitivity_catalog: Optional[Mapping[str, SensitivityTier]] = None,
        sensitive_action_policy: Literal["ask_every_time", "deny"] = "ask_every_time",
        auto_approve_reversible_local_writes: bool = False,
    ) -> None:
        self._catalog: Dict[str, SensitivityTier] = dict(tool_sensitivity_catalog or {})
        self._sensitive_policy = sensitive_action_policy
        self._auto_approve_local_writes = auto_approve_reversible_local_writes
        self._user_decisions: Dict[str, Tuple[bool, str]] = {}

    def record_user_decision(
        self,
        proposal_id: str,
        approved: bool,
        reason: str = "User interactive decision",
        *, proposal: Optional[ToolCallProposal] = None, context: Optional[ToolActionContext] = None,
    ) -> None:
        """Store a human Approve (`True`) or Deny (`False`) decision for `proposal_id`."""
        self._user_decisions[proposal_id] = (approved, reason)

    def evaluate_tool_proposal(
        self,
        proposal: ToolCallProposal,
        *, context: Optional[ToolActionContext] = None,
    ) -> PermissionCheckResult:
        """Evaluate a ToolCallProposal against ADR-008 scope and ADR-010 sensitivity rules."""
        spec = get_agent_spec(proposal.agent_role)

        # Step 1: Verify agent badge scope (ADR-008)
        try:
            spec.validate_proposal_scope(proposal)
        except AgentScopeViolationError as exc:
            return PermissionCheckResult(
                proposal_id=proposal.proposal_id,
                tool_name=proposal.tool_name,
                agent_role=proposal.agent_role,
                effective_sensitivity=SensitivityTier.IRREVERSIBLE_WRITE,
                decision=PermissionDecisionKind.DENIED_OUT_OF_SCOPE,
                requires_interactive_prompt=False,
                reason=str(exc),
            )

        # Step 2: Determine authoritative sensitivity tier (take the stricter of catalog vs declared)
        catalog_tier = self._catalog.get(proposal.tool_name)
        effective_tier = catalog_tier if catalog_tier is not None else proposal.declared_sensitivity

        # Step 3: Check if the human user already responded to an interactive prompt for this proposal
        if proposal.proposal_id in self._user_decisions:
            approved, user_reason = self._user_decisions[proposal.proposal_id]
            return PermissionCheckResult(
                proposal_id=proposal.proposal_id,
                tool_name=proposal.tool_name,
                agent_role=proposal.agent_role,
                effective_sensitivity=effective_tier,
                decision=(
                    PermissionDecisionKind.USER_APPROVED
                    if approved
                    else PermissionDecisionKind.DENIED_BY_USER
                ),
                requires_interactive_prompt=False,
                reason=user_reason,
            )

        # Step 4: Safe read-only tools are auto-approved
        if effective_tier == SensitivityTier.READ_ONLY:
            return PermissionCheckResult(
                proposal_id=proposal.proposal_id,
                tool_name=proposal.tool_name,
                agent_role=proposal.agent_role,
                effective_sensitivity=effective_tier,
                decision=PermissionDecisionKind.AUTO_APPROVED,
                requires_interactive_prompt=False,
                reason="Auto-approved safe local read_only tool.",
            )

        # Step 5: Optional auto-approval for reversible local writes if configured
        if (
            effective_tier == SensitivityTier.REVERSIBLE_WRITE
            and self._auto_approve_local_writes
        ):
            return PermissionCheckResult(
                proposal_id=proposal.proposal_id,
                tool_name=proposal.tool_name,
                agent_role=proposal.agent_role,
                effective_sensitivity=effective_tier,
                decision=PermissionDecisionKind.AUTO_APPROVED,
                requires_interactive_prompt=False,
                reason="Auto-approved reversible local write per policy.",
            )

        # Step 6: Sensitive / networked / irreversible actions follow sensitive_action_policy
        if self._sensitive_policy == "deny":
            return PermissionCheckResult(
                proposal_id=proposal.proposal_id,
                tool_name=proposal.tool_name,
                agent_role=proposal.agent_role,
                effective_sensitivity=effective_tier,
                decision=PermissionDecisionKind.DENIED_BY_POLICY,
                requires_interactive_prompt=False,
                reason=f"Denied by policy ('deny') for sensitivity tier {effective_tier.value!r}.",
            )

        # Default fail-closed behavior: pause and ask the human user!
        return PermissionCheckResult(
            proposal_id=proposal.proposal_id,
            tool_name=proposal.tool_name,
            agent_role=proposal.agent_role,
            effective_sensitivity=effective_tier,
            decision=None,
            requires_interactive_prompt=True,
            reason=(
                f"ADR-010: Tool {proposal.tool_name!r} ({effective_tier.value}) requires "
                "explicit human approval before execution."
            ),
        )


def verify_approval_stamp(
    proposal: ToolCallProposal,
    permission_stamp: PermissionCheckResult,
) -> None:
    """Enforce that `permission_stamp` is a genuine, matching approval for `proposal` (ADR-010)."""
    if not isinstance(permission_stamp, PermissionCheckResult):
        raise UnapprovedToolExecutionError(
            "ADR-010 violation: Missing PermissionCheckResult stamp."
        )
    if permission_stamp.agent_role != proposal.agent_role:
        raise UnapprovedToolExecutionError("Approval agent identity mismatch.")
    if permission_stamp.context is not None:
        from nova.core.permissions.authorization import action_digest
        if action_digest(proposal, permission_stamp.context) != permission_stamp.action_digest:
            raise UnapprovedToolExecutionError("Approval arguments/context mismatch.")
    if permission_stamp.proposal_id != proposal.proposal_id:
        raise UnapprovedToolExecutionError(
            f"ADR-010 violation: Permission stamp proposal_id {permission_stamp.proposal_id!r} "
            f"does not match tool proposal_id {proposal.proposal_id!r}."
        )
    if permission_stamp.tool_name != proposal.tool_name:
        raise UnapprovedToolExecutionError(
            f"ADR-010 violation: Permission stamp tool_name {permission_stamp.tool_name!r} "
            f"does not match proposed tool {proposal.tool_name!r}."
        )
    if not permission_stamp.is_approved:
        raise UnapprovedToolExecutionError(
            f"ADR-010 violation: Tool {proposal.tool_name!r} (proposal {proposal.proposal_id!r}) "
            f"is NOT approved for execution (decision={permission_stamp.decision}, "
            f"requires_interactive_prompt={permission_stamp.requires_interactive_prompt})."
        )


ToolHandlerFn = Callable[[Dict[str, object]], str]


class InMemoryMockToolExecutor:
    """Deterministic tool executor for testing orchestration flows before real MCP servers attach."""

    def __init__(
        self,
        tool_handlers: Optional[Mapping[str, ToolHandlerFn]] = None,
    ) -> None:
        self._handlers: Dict[str, ToolHandlerFn] = dict(tool_handlers or {})
        self.executed_proposals: List[ToolCallProposal] = []

    def register_tool(self, tool_name: str, handler: ToolHandlerFn) -> None:
        """Register a deterministic handler function for `tool_name`."""
        self._handlers[tool_name] = handler

    def execute_approved_tool(
        self,
        proposal: ToolCallProposal,
        permission_stamp: PermissionCheckResult,
        *, context: Optional[ToolActionContext] = None,
        on_started: Optional[Callable[[], None]] = None,
    ) -> ToolExecutionResult:
        """Verify the approval stamp and run the registered handler."""
        verify_approval_stamp(proposal, permission_stamp)
        self.executed_proposals.append(proposal)

        handler = self._handlers.get(proposal.tool_name)
        if handler is None:
            return ToolExecutionResult(
                proposal_id=proposal.proposal_id,
                tool_name=proposal.tool_name,
                status="error",
                sanitized_output=f"Tool '{proposal.tool_name}' is not registered in executor.",
            )

        if on_started is not None:
            on_started()
        try:
            raw_output = handler(dict(proposal.arguments))
            return ToolExecutionResult(
                proposal_id=proposal.proposal_id,
                tool_name=proposal.tool_name,
                status="success",
                sanitized_output=str(raw_output),
            )
        except Exception as exc:
            return ToolExecutionResult(
                proposal_id=proposal.proposal_id,
                tool_name=proposal.tool_name,
                status="error",
                sanitized_output=f"Tool execution error: {exc}",
            )
