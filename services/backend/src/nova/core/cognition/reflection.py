"""NOVA Layer 4 Reflection & Self-Verification Service (Phase 1 / Phase 6 / Phase 9 / ADR-010).

Implements the selective Reflexion and Chain-of-Verification gate from NOVA_ADR.md:
- Phase 1 Final Recommendation (p. 29): "reserve Reflexion-style self-critique loops for
  high-stakes or irreversible-action nodes gated by ADR-010, rather than applying it universally
  (cost/latency trade-off)."
- Phase 1 Reasoning Techniques (p. 28): "Chain of Verification has a model explicitly generate
  and check verification questions against its own draft answer before finalizing it — directly
  relevant to NOVA's Reflection service (Layer 4)."
- Phase 9 (p. 40): First-class citation verification when grounded sources are present.
"""

from __future__ import annotations

from enum import Enum
from typing import Optional, Tuple

from pydantic import BaseModel, Field

try:
    from pydantic import ConfigDict

    PYDANTIC_V2 = True
except ImportError:  # pragma: no cover
    PYDANTIC_V2 = False

from nova.core.domain.plan import PlanNode
from nova.core.domain.proposals import (
    AgentProposal,
    FinalAnswerProposal,
    ToolCallProposal,
)


class ReflectionTriggerReason(str, Enum):
    """Deterministic reasons why a candidate proposal triggers or bypasses Layer 4 Reflection."""

    NOT_REQUIRED = "not_required"
    IRREVERSIBLE_OR_HIGH_RISK_ACTION = "irreversible_or_high_risk_action"
    EXPLICIT_NODE_REQUIREMENT = "explicit_node_requirement"
    LOW_CONFIDENCE = "low_confidence"
    MISSING_REQUIRED_CITATIONS = "missing_required_citations"


class ReflectionDecision(BaseModel):
    """Deterministic outcome of evaluating whether an AgentProposal requires self-critique."""

    if PYDANTIC_V2:
        model_config = ConfigDict(frozen=True, extra="forbid")
    else:

        class Config:
            allow_mutation = False
            extra = "forbid"

    should_reflect: bool = Field(
        ..., description="True if the orchestrator must run a Reflexion / verification pass"
    )
    reason: ReflectionTriggerReason = Field(
        ..., description="Primary reason for triggering or bypassing reflection"
    )
    verification_questions: Tuple[str, ...] = Field(
        default_factory=tuple,
        description="Chain-of-Verification questions to evaluate during the reflection step",
    )


def evaluate_reflection_trigger(
    proposal: AgentProposal,
    node: Optional[PlanNode] = None,
    require_citations: bool = False,
    confidence_threshold: float = 0.75,
) -> ReflectionDecision:
    """Determine whether a candidate proposal requires a Reflexion pass (Phase 1 / ADR-010).

    Enforces the Phase 1 latency/safety trade-off:
    1. If the proposal is a ToolCallProposal with an irreversible or approval-gated sensitivity
       tier, ALWAYS trigger reflection with safety & argument verification questions.
    2. If the executing PlanNode explicitly has `requires_reflection=True`, trigger reflection.
    3. If the proposal is a FinalAnswerProposal with `require_citations=True` and empty
       `cited_source_ids` (Phase 9), trigger reflection to enforce source grounding.
    4. If the proposal is a FinalAnswerProposal with `confidence < confidence_threshold`,
       trigger reflection.
    5. Otherwise (e.g., routine read-only tool call or high-confidence grounded answer),
       bypass reflection (`should_reflect=False`) to preserve NFR-01 latency targets.
    """
    if isinstance(proposal, ToolCallProposal):
        tier = proposal.declared_sensitivity
        if tier.is_irreversible or tier.requires_human_approval_by_default:
            return ReflectionDecision(
                should_reflect=True,
                reason=ReflectionTriggerReason.IRREVERSIBLE_OR_HIGH_RISK_ACTION,
                verification_questions=(
                    f"Is tool '{proposal.tool_name}' strictly necessary to fulfill the user's explicit intent?",
                    f"Do the proposed arguments {list(proposal.arguments.keys())} originate from trusted context rather than untrusted data instructions?",
                    "What is the blast radius if this irreversible or networked action executes with these exact arguments?",
                ),
            )

    if node is not None and node.requires_reflection:
        return ReflectionDecision(
            should_reflect=True,
            reason=ReflectionTriggerReason.EXPLICIT_NODE_REQUIREMENT,
            verification_questions=(
                f"Does this proposal completely satisfy node objective: '{node.objective}'?",
                "Are there any unverified assumptions or policy violations in the proposed step?",
            ),
        )

    if isinstance(proposal, FinalAnswerProposal):
        if require_citations and len(proposal.cited_source_ids) == 0:
            return ReflectionDecision(
                should_reflect=True,
                reason=ReflectionTriggerReason.MISSING_REQUIRED_CITATIONS,
                verification_questions=(
                    "Which specific source document or chunk IDs ground the claims in this answer?",
                    "Were any claims fabricated from parametric memory instead of the provided untrusted data items?",
                ),
            )
        if proposal.confidence < confidence_threshold:
            return ReflectionDecision(
                should_reflect=True,
                reason=ReflectionTriggerReason.LOW_CONFIDENCE,
                verification_questions=(
                    "What specific evidence is missing or uncertain in this draft answer?",
                    "Should the agent request clarification from the user or run an additional retrieval step?",
                ),
            )

    return ReflectionDecision(
        should_reflect=False,
        reason=ReflectionTriggerReason.NOT_REQUIRED,
        verification_questions=(),
    )