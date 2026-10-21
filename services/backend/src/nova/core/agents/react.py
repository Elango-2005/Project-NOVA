"""NOVA Bounded ReAct Step State & Observation Accumulator (Layer 5 — Phase 1 / ADR-004).

Implements the per-node ReAct (Reason + Act) state machine from NOVA_ADR.md:
- Phase 1 Final Recommendation (p. 29): "Use Plan-and-Execute at the orchestrator level
  (matches the graph model of ADR-003) with ReAct-style reasoning within each graph node
  where an agent needs to interleave observation and action."
- ADR-004: The model never runs an autonomous while-loop. Instead, `ReActNodeState`
  records bounded `(Thought -> Proposed Action -> Observation)` turns up to `max_iterations`,
  and the deterministic orchestrator advances each turn.
- Phase 15 Layer 1: Every tool/environment observation is delimiter-sanitized and wrapped
  inside `<untrusted_data source_type="tool_output">` so tool results can never inject
  system instructions into subsequent ReAct steps.
"""

from __future__ import annotations

from typing import Dict, List, Literal, Optional, Tuple

from pydantic import BaseModel, Field

try:
    from pydantic import ConfigDict

    PYDANTIC_V2 = True
except ImportError:  # pragma: no cover
    PYDANTIC_V2 = False

from nova.core.domain.context import sanitize_untrusted_text
from nova.core.domain.proposals import ToolCallProposal


class ReActStepLimitExceededError(RuntimeError):
    """Raised when an agent node exceeds its bounded ReAct step budget (ADR-003 / ADR-004)."""


class ReActTurnRecord(BaseModel):
    """Immutable record of a single completed Thought -> Action -> Observation turn."""

    if PYDANTIC_V2:
        model_config = ConfigDict(frozen=True, extra="forbid")
    else:

        class Config:
            allow_mutation = False
            extra = "forbid"

    step_index: int = Field(..., ge=1, description="1-based ReAct iteration number")
    thought: str = Field(..., min_length=1, description="Agent's reasoning trace for this turn")
    tool_name: str = Field(..., min_length=1, description="Proposed tool identifier")
    tool_arguments: Dict[str, object] = Field(default_factory=dict)
    observation_status: Literal["success", "error", "permission_denied", "validation_failed"] = (
        Field(..., description="Deterministic outcome category of the action")
    )
    sanitized_observation: str = Field(
        ..., description="Delimiter-escaped observation output treated strictly as passive data"
    )

    def __init__(self, **data: object) -> None:
        raw_obs = data.get("sanitized_observation", "")
        if isinstance(raw_obs, str):
            data = dict(data)
            data["sanitized_observation"] = sanitize_untrusted_text(raw_obs)
        super().__init__(**data)

    def render_delimited_turn(self) -> str:
        """Render the turn with strict separation between the action trace and untrusted observation."""
        return (
            f"<react_turn step=\"{self.step_index}\">\n"
            f"  <prior_thought>{sanitize_untrusted_text(self.thought)}</prior_thought>\n"
            f"  <proposed_tool name=\"{self.tool_name}\" status=\"{self.observation_status}\" />\n"
            f"  <untrusted_data id=\"obs_step_{self.step_index}\" "
            f"source_type=\"tool_output\" status=\"{self.observation_status}\">\n"
            f"{self.sanitized_observation}\n"
            f"  </untrusted_data>\n"
            f"</react_turn>"
        )


class ReActNodeState(BaseModel):
    """Immutable accumulator tracking bounded ReAct turns inside a single execution graph node."""

    if PYDANTIC_V2:
        model_config = ConfigDict(frozen=True, extra="forbid")
    else:

        class Config:
            allow_mutation = False
            extra = "forbid"

    node_id: str = Field(..., min_length=1)
    max_iterations: int = Field(
        default=5,
        description="ADR-004 hard upper bound on ReAct iterations inside a node (1..15)",
    )
    turns: Tuple[ReActTurnRecord, ...] = Field(
        default_factory=tuple,
        description="Ordered history of completed Thought-Action-Observation turns",
    )

    def __init__(self, **data: object) -> None:
        max_iter = int(data.get("max_iterations", 5))  # type: ignore[arg-type]
        if not (1 <= max_iter <= 15):
            raise ValueError(
                f"ADR-004 violation: ReActNodeState 'max_iterations' must be in [1, 15], got {max_iter}."
            )
        super().__init__(**data)

    @property
    def current_step_count(self) -> int:
        """Return the number of completed ReAct turns so far."""
        return len(self.turns)

    @property
    def remaining_steps(self) -> int:
        """Return the number of remaining reasoning steps allowed in this node."""
        return max(0, self.max_iterations - len(self.turns))

    def can_continue(self) -> bool:
        """Return True if the node has remaining ReAct step budget."""
        return len(self.turns) < self.max_iterations

    def require_step_budget(self) -> None:
        """Fail closed if the node has exhausted its bounded ReAct iterations (ADR-004)."""
        if not self.can_continue():
            raise ReActStepLimitExceededError(
                f"ADR-004 violation: Node {self.node_id!r} exhausted its bounded ReAct budget "
                f"({self.max_iterations} iterations)."
            )

    def record_observation(
        self,
        proposal: ToolCallProposal,
        observation_text: str,
        status: Literal["success", "error", "permission_denied", "validation_failed"] = "success",
    ) -> "ReActNodeState":
        """Return a new ReActNodeState with the latest tool/environment observation appended."""
        self.require_step_budget()
        next_turn = ReActTurnRecord(
            step_index=len(self.turns) + 1,
            thought=proposal.thought,
            tool_name=proposal.tool_name,
            tool_arguments=dict(proposal.arguments),
            observation_status=status,
            sanitized_observation=observation_text,
        )
        return ReActNodeState(
            node_id=self.node_id,
            max_iterations=self.max_iterations,
            turns=self.turns + (next_turn,),
        )

    def render_scratchpad_message(self) -> Optional[Dict[str, str]]:
        """Render accumulated ReAct turns as a delimited user-role data message for the next step."""
        if not self.turns:
            return None
        rendered_turns: List[str] = [t.render_delimited_turn() for t in self.turns]
        content = (
            f"<react_observation_history node_id=\"{self.node_id}\" "
            f"completed_steps=\"{self.current_step_count}\" "
            f"remaining_steps=\"{self.remaining_steps}\">\n"
            + "\n".join(rendered_turns)
            + "\n</react_observation_history>"
        )
        return {"role": "user", "content": content}