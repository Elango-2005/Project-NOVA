"""NOVA Structured Action Proposals (Layer 5/6 Boundary — ADR-004 / ADR-010).

Implements the bounded reasoning output contracts from NOVA_ADR.md:
- ADR-004: "The LLM proposes, the runtime disposes." Agents never execute tools directly;
  they return a strictly typed, schema-validated proposal.
- Engineering Principle 4: Fail closed on malformed, free-text, or unknown proposal shapes.
- Phase 1: Supports ReAct reasoning traces (`thought`), Plan-and-Execute decompositions (`PlanProposal`),
  and Reflexion / Chain-of-Verification critiques (`CritiqueProposal`).
- Phase 9: First-class source citation tracking (`cited_source_ids`) on final answers.
"""

from __future__ import annotations

import json
import re
import uuid
from enum import Enum
from typing import Any, Dict, Literal, Mapping, Optional, Tuple, Union

from pydantic import BaseModel, Field

try:
    from pydantic import ConfigDict

    PYDANTIC_V2 = True
except ImportError:  # pragma: no cover
    PYDANTIC_V2 = False

from nova.core.domain.primitives import (
    CURRENT_SCHEMA_VERSION,
    AgentRole,
    SensitivityTier,
    validate_schema_version,
)

# Canonical MCP / Native tool identifier format: "<server_or_namespace>.<tool_name>"
_TOOL_NAME_PATTERN = re.compile(r"^[a-z][a-z0-9_]*\.[a-z][a-z0-9_]*$")


class ProposalValidationError(ValueError):
    """Raised when a model output fails NOVA's deterministic proposal schema validation (ADR-004)."""


class ProposalKind(str, Enum):
    """Discriminated union tags for all structured agent proposals."""

    TOOL_CALL = "tool_call"
    FINAL_ANSWER = "final_answer"
    PLAN = "plan"
    CRITIQUE = "critique"
    ASK_USER = "ask_user"


class _BaseProposal(BaseModel):
    """Common immutable metadata across all agent proposals."""

    if PYDANTIC_V2:
        model_config = ConfigDict(frozen=True, extra="forbid")
    else:

        class Config:
            allow_mutation = False
            extra = "forbid"

    schema_version: str = Field(
        default=CURRENT_SCHEMA_VERSION,
        description="Versioned contract tag (Engineering Principle 7)",
    )
    proposal_id: str = Field(
        default_factory=lambda: f"prop_{uuid.uuid4().hex}",
        description="Unique proposal identifier for audit and permission gating",
    )
    agent_role: AgentRole = Field(..., description="Specialized agent role emitting the proposal")
    node_id: str = Field(..., min_length=1, description="Execution graph node ID (ADR-003)")
    thought: str = Field(
        ...,
        min_length=1,
        description="Explicit reasoning trace preceding the proposed action (Phase 1 ReAct)",
    )

    def _validate_base_fields(self, data: Mapping[str, Any]) -> None:
        version = data.get("schema_version", CURRENT_SCHEMA_VERSION)
        if not isinstance(version, str):
            raise ProposalValidationError("schema_version must be a string.")
        validate_schema_version(version)

        thought = data.get("thought")
        if not isinstance(thought, str) or not thought.strip():
            raise ProposalValidationError(
                "ADR-004 / Phase 1 violation: 'thought' reasoning trace must be a non-empty string."
            )
        node_id = data.get("node_id")
        if not isinstance(node_id, str) or not node_id.strip():
            raise ProposalValidationError("'node_id' must be a non-empty string.")


class ToolCallProposal(_BaseProposal):
    """Structured proposal to invoke a single MCP or native tool through the Permission Gate."""

    kind: Literal["tool_call"] = Field(default="tool_call")
    tool_name: str = Field(
        ...,
        description="Canonical '<namespace>.<action>' tool identifier (e.g., 'filesystem.read_file')",
    )
    arguments: Dict[str, Any] = Field(
        default_factory=dict,
        description="JSON-serializable arguments validated against the tool's JSON Schema",
    )
    declared_sensitivity: SensitivityTier = Field(
        default=SensitivityTier.IRREVERSIBLE_WRITE,
        description="Fail-closed sensitivity classification (defaults to IRREVERSIBLE_WRITE if omitted)",
    )
    rationale: str = Field(
        ...,
        min_length=1,
        description="Human-inspectable explanation shown in Permission Gate prompts (ADR-010)",
    )

    def __init__(self, **data: Any) -> None:
        self._validate_base_fields(data)
        tool_name = data.get("tool_name")
        if not isinstance(tool_name, str) or not _TOOL_NAME_PATTERN.match(tool_name):
            raise ProposalValidationError(
                f"ADR-005 / Phase 7 violation: tool_name must match '<namespace>.<action>' "
                f"(lowercase alphanumeric/underscore), got {tool_name!r}."
            )
        rationale = data.get("rationale")
        if not isinstance(rationale, str) or not rationale.strip():
            raise ProposalValidationError(
                "ADR-010 violation: ToolCallProposal requires a non-empty 'rationale'."
            )
        arguments = data.get("arguments", {})
        if not isinstance(arguments, Mapping):
            raise ProposalValidationError("ToolCallProposal 'arguments' must be a JSON object/dict.")
        try:
            json.dumps(dict(arguments))
        except (TypeError, ValueError) as exc:
            raise ProposalValidationError(
                f"ToolCallProposal 'arguments' must be strictly JSON-serializable: {exc}"
            ) from exc
        super().__init__(**data)


class FinalAnswerProposal(_BaseProposal):
    """Structured proposal completing an agent node with a grounded response."""

    kind: Literal["final_answer"] = Field(default="final_answer")
    content: str = Field(..., min_length=1, description="Final response or subtask output")
    cited_source_ids: Tuple[str, ...] = Field(
        default_factory=tuple,
        description="Phase 9: Identifiers of source documents/chunks grounding this response",
    )
    confidence: float = Field(
        default=1.0,
        description="Self-assessed confidence score in [0.0, 1.0]",
    )

    def __init__(self, **data: Any) -> None:
        self._validate_base_fields(data)
        content = data.get("content")
        if not isinstance(content, str) or not content.strip():
            raise ProposalValidationError("FinalAnswerProposal 'content' must be a non-empty string.")
        confidence = data.get("confidence", 1.0)
        if not isinstance(confidence, (int, float)) or not (0.0 <= float(confidence) <= 1.0):
            raise ProposalValidationError(
                f"FinalAnswerProposal 'confidence' must be between 0.0 and 1.0, got {confidence!r}."
            )
        super().__init__(**data)


class SubtaskPlanStep(BaseModel):
    """Single decomposed step inside a PlannerAgent's PlanProposal (ADR-003 / Phase 6)."""

    if PYDANTIC_V2:
        model_config = ConfigDict(frozen=True, extra="forbid")
    else:

        class Config:
            allow_mutation = False
            extra = "forbid"

    step_id: str = Field(..., min_length=1, description="Unique node identifier in the plan graph")
    target_agent: AgentRole = Field(
        ..., description="Specialized agent assigned to execute this step (ADR-008)"
    )
    objective: str = Field(..., min_length=1, description="Concrete goal for the target agent")
    depends_on: Tuple[str, ...] = Field(
        default_factory=tuple,
        description="Upstream step_ids that must complete before this step runs",
    )
    estimated_sensitivity: SensitivityTier = Field(
        default=SensitivityTier.READ_ONLY,
        description="Estimated highest sensitivity tier for permission pre-flighting (ADR-003)",
    )

    def __init__(self, **data: Any) -> None:
        step_id = data.get("step_id")
        if not isinstance(step_id, str) or not step_id.strip():
            raise ProposalValidationError("SubtaskPlanStep 'step_id' must be a non-empty string.")
        objective = data.get("objective")
        if not isinstance(objective, str) or not objective.strip():
            raise ProposalValidationError("SubtaskPlanStep 'objective' must be a non-empty string.")
        depends_on = tuple(data.get("depends_on", ()))
        if step_id in depends_on:
            raise ProposalValidationError(
                f"ADR-003 violation: step {step_id!r} cannot depend on itself."
            )
        super().__init__(**data)


class PlanProposal(_BaseProposal):
    """Structured task decomposition emitted by the PlannerAgent (Phase 1 Plan-and-Execute)."""

    kind: Literal["plan"] = Field(default="plan")
    goal_summary: str = Field(..., min_length=1, description="Normalized summary of the user goal")
    steps: Tuple[SubtaskPlanStep, ...] = Field(
        ..., description="Ordered/DAG steps to be compiled into the orchestrator execution graph"
    )

    def __init__(self, **data: Any) -> None:
        self._validate_base_fields(data)
        raw_steps = data.get("steps")
        if not raw_steps or not isinstance(raw_steps, (list, tuple)):
            raise ProposalValidationError("PlanProposal must contain at least one SubtaskPlanStep.")

        parsed_steps = tuple(
            s if isinstance(s, SubtaskPlanStep) else SubtaskPlanStep(**s) for s in raw_steps
        )
        seen_ids = set()
        for step in parsed_steps:
            if step.step_id in seen_ids:
                raise ProposalValidationError(
                    f"Duplicate step_id {step.step_id!r} in PlanProposal."
                )
            seen_ids.add(step.step_id)

        for step in parsed_steps:
            for dep in step.depends_on:
                if dep not in seen_ids:
                    raise ProposalValidationError(
                        f"Step {step.step_id!r} references unknown dependency {dep!r}."
                    )

        data = dict(data)
        data["steps"] = parsed_steps
        super().__init__(**data)


class CritiqueProposal(_BaseProposal):
    """Structured self-critique or verification result (Phase 1 Reflexion & Chain-of-Verification)."""

    kind: Literal["critique"] = Field(default="critique")
    target_proposal_id: str = Field(
        ..., min_length=1, description="ID of the candidate proposal being evaluated"
    )
    verdict: Literal["approve", "revise", "abort"] = Field(
        ..., description="Deterministic routing decision for the orchestrator"
    )
    issues_found: Tuple[str, ...] = Field(
        default_factory=tuple,
        description="Specific correctness, safety, or grounding defects identified",
    )
    verification_questions_checked: Tuple[str, ...] = Field(
        default_factory=tuple,
        description="Phase 1 Chain-of-Verification questions evaluated against the candidate output",
    )
    revision_instructions: Optional[str] = Field(
        default=None,
        description="Required actionable guidance when verdict is 'revise'",
    )

    def __init__(self, **data: Any) -> None:
        self._validate_base_fields(data)
        verdict = data.get("verdict")
        if verdict not in ("approve", "revise", "abort"):
            raise ProposalValidationError(
                f"CritiqueProposal 'verdict' must be 'approve', 'revise', or 'abort', got {verdict!r}."
            )
        revision = data.get("revision_instructions")
        if verdict == "revise" and (not isinstance(revision, str) or not revision.strip()):
            raise ProposalValidationError(
                "CritiqueProposal with verdict='revise' must provide non-empty 'revision_instructions'."
            )
        super().__init__(**data)


class ClarificationProposal(_BaseProposal):
    """Structured request for user clarification when task-critical inputs are missing (ADR-003)."""

    kind: Literal["ask_user"] = Field(default="ask_user")
    question: str = Field(..., min_length=1, description="Direct question to surface to the user")
    missing_fields: Tuple[str, ...] = Field(
        default_factory=tuple,
        description="Names of missing parameters or ambiguous entities blocking execution",
    )
    options: Tuple[str, ...] = Field(
        default_factory=tuple,
        description="Concrete choices offered to the user",
    )

    def __init__(self, **data: Any) -> None:
        self._validate_base_fields(data)
        question = data.get("question")
        if not isinstance(question, str) or not question.strip():
            raise ProposalValidationError("ClarificationProposal 'question' must be non-empty.")
        super().__init__(**data)


AgentProposal = Union[
    ToolCallProposal,
    FinalAnswerProposal,
    PlanProposal,
    CritiqueProposal,
    ClarificationProposal,
]

_PROPOSAL_MAP = {
    ProposalKind.TOOL_CALL.value: ToolCallProposal,
    ProposalKind.FINAL_ANSWER.value: FinalAnswerProposal,
    ProposalKind.PLAN.value: PlanProposal,
    ProposalKind.CRITIQUE.value: CritiqueProposal,
    ProposalKind.ASK_USER.value: ClarificationProposal,
}


def parse_agent_proposal(raw_input: Union[str, Mapping[str, Any]]) -> AgentProposal:
    """Deterministically parse and validate raw model output into a typed AgentProposal.

    Fails closed (Engineering Principle 4) if raw_input is unstructured text, invalid JSON,
    or does not conform to one of the five canonical proposal schemas.
    """
    if isinstance(raw_input, str):
        stripped = raw_input.strip()
        if not stripped.startswith("{"):
            raise ProposalValidationError(
                "ADR-004 violation: Raw free-text model output is forbidden; expected JSON object."
            )
        try:
            payload = json.loads(stripped)
        except json.JSONDecodeError as exc:
            raise ProposalValidationError(f"Malformed JSON in agent proposal: {exc}") from exc
    elif isinstance(raw_input, Mapping):
        payload = dict(raw_input)
    else:
        raise ProposalValidationError(
            f"Expected JSON string or mapping for agent proposal, got {type(raw_input).__name__}."
        )

    if not isinstance(payload, dict):
        raise ProposalValidationError("Agent proposal root must be a JSON object.")

    kind = payload.get("kind")
    if kind not in _PROPOSAL_MAP:
        raise ProposalValidationError(
            f"Unknown or missing proposal 'kind': {kind!r}. "
            f"Allowed kinds: {sorted(_PROPOSAL_MAP.keys())}."
        )

    model_cls = _PROPOSAL_MAP[kind]
    try:
        return model_cls(**payload)  # type: ignore[return-value]
    except Exception as exc:
        if isinstance(exc, ProposalValidationError):
            raise
        raise ProposalValidationError(f"Invalid {kind} proposal schema: {exc}") from exc