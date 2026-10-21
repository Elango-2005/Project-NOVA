"""NOVA Base Specialized Agent Implementation (Layer 5 — ADR-002 / ADR-004 / ADR-008 / ADR-010).

Implements the bounded specialized agent execution step from NOVA_ADR.md:
- ADR-004: "The LLM is invoked as a bounded reasoning function: given a well-defined context
  and a well-defined output schema, it returns a structured result."
- ADR-008: Enforces per-agent least-privilege tool and proposal scopes (`AgentSpecification`).
- Phase 15 Layer 1: Sends strictly separated system instructions and delimited untrusted data
  to the `ModelReasoningPort`.
- ADR-001 & NFR-04: Verifies that no cloud inference occurred unless explicitly authorized
  by both runtime settings and the context's data-classification governance check.
- ADR-002 & NFR-08: Emits structured lifecycle and security events (`agent.started`,
  `model.generation.started`, `model.generation.completed`, `tool.proposed`, `agent.completed`,
  `security.violation`, `task.failed`).
"""

from __future__ import annotations

from typing import Optional

from nova.core.agents.react import ReActNodeState, ReActStepLimitExceededError
from nova.core.config.settings import NovaSettings, load_settings
from nova.core.domain.agent_spec import (
    AgentScopeViolationError,
    AgentSpecification,
    get_agent_spec,
)
from nova.core.domain.context import AgentContext, DataGovernanceError
from nova.core.domain.events import EventType, NovaEvent
from nova.core.domain.primitives import AgentRole
from nova.core.domain.proposals import (
    AgentProposal,
    ProposalValidationError,
    ToolCallProposal,
    parse_agent_proposal,
)
from nova.core.model_serving.interface import ModelInferenceRequest, ModelReasoningPort
from nova.core.platform.event_bus import EventPublisherPort


class BaseSpecializedAgent:
    """Bounded, observable specialized agent executing a single reasoning step per invocation."""

    def __init__(
        self,
        role_or_spec: AgentRole | AgentSpecification,
        model_port: ModelReasoningPort,
        event_bus: EventPublisherPort,
        settings: Optional[NovaSettings] = None,
    ) -> None:
        self._spec = (
            role_or_spec
            if isinstance(role_or_spec, AgentSpecification)
            else get_agent_spec(role_or_spec)
        )
        self._model_port = model_port
        self._event_bus = event_bus
        self._settings = settings if settings is not None else load_settings(environ={})

    @property
    def spec(self) -> AgentSpecification:
        """Return this agent's immutable ADR-008 specification."""
        return self._spec

    @property
    def role(self) -> AgentRole:
        """Return this agent's canonical role."""
        return self._spec.role

    def propose_next_action(
        self,
        context: AgentContext,
        react_state: Optional[ReActNodeState] = None,
    ) -> AgentProposal:
        """Execute a single bounded reasoning step and return a validated AgentProposal (ADR-004).

        Never executes tools directly and never runs an unbounded loop.
        """
        # 1. Verify context instruction role matches this specialized agent
        if context.instruction.agent_role != self._spec.role:
            self._emit_event(
                context=context,
                event_type=EventType.SECURITY_VIOLATION,
                payload={
                    "reason": "context_role_mismatch",
                    "expected_role": self._spec.role.value,
                    "received_role": context.instruction.agent_role.value,
                },
            )
            raise AgentScopeViolationError(
                f"ADR-008 violation: Context assigned to {context.instruction.agent_role.value!r} "
                f"cannot be executed by {self._spec.role.value!r} agent."
            )

        # 2. Enforce bounded ReAct step budget before invoking the model (ADR-004)
        if react_state is not None:
            if react_state.node_id != context.node_id:
                raise ValueError(
                    f"ReActNodeState.node_id ({react_state.node_id!r}) does not match "
                    f"AgentContext.node_id ({context.node_id!r})."
                )
            try:
                react_state.require_step_budget()
            except ReActStepLimitExceededError as exc:
                self._emit_event(
                    context=context,
                    event_type=EventType.TASK_FAILED,
                    payload={
                        "reason": "react_step_limit_exceeded",
                        "max_iterations": react_state.max_iterations,
                    },
                )
                raise

        # 3. Emit agent.started lifecycle event (ADR-002 / NFR-08)
        step_number = (react_state.current_step_count + 1) if react_state is not None else 1
        self._emit_event(
            context=context,
            event_type=EventType.AGENT_STARTED,
            payload={
                "step_number": step_number,
                "allowed_tools": list(self._spec.allowed_tools),
            },
        )

        # 4. Assemble Phase 15 Layer 1 separated messages + delimited ReAct observation history
        messages = list(context.render_separated_messages())
        if react_state is not None:
            scratchpad_msg = react_state.render_scratchpad_message()
            if scratchpad_msg is not None:
                messages.append(scratchpad_msg)

        allow_cloud = context.can_route_to_cloud(
            allow_cloud_inference=self._settings.allow_cloud_inference,
            allow_cloud_sensitive_data=self._settings.allow_cloud_sensitive_data,
        )

        inference_request = ModelInferenceRequest(
            task_id=context.task_id,
            session_id=context.session_id,
            node_id=context.node_id,
            agent_role=self._spec.role,
            model_tier=self._spec.typical_model_tier,
            messages=tuple(messages),
            output_schema_name=context.instruction.expected_output_schema,
            allow_cloud_routing=allow_cloud,
        )

        self._emit_event(
            context=context,
            event_type=EventType.MODEL_GENERATION_STARTED,
            payload={
                "model_tier": self._spec.typical_model_tier,
                "allow_cloud_routing": allow_cloud,
            },
        )

        # 5. Invoke Layer 2 Model Serving port
        response = self._model_port.generate_structured(inference_request)

        # Verify ADR-001 / NFR-04 privacy guarantee: fail closed if a model adapter used cloud without consent
        if response.is_cloud and not allow_cloud:
            self._emit_event(
                context=context,
                event_type=EventType.SECURITY_VIOLATION,
                payload={
                    "reason": "unconsented_cloud_inference",
                    "provider": response.provider,
                    "model_id": response.model_id,
                },
            )
            raise DataGovernanceError(
                f"ADR-001 / NFR-04 violation: Model adapter {response.provider!r} executed "
                "cloud inference when allow_cloud_routing=False."
            )

        self._emit_event(
            context=context,
            event_type=EventType.MODEL_GENERATION_COMPLETED,
            payload={
                "model_id": response.model_id,
                "provider": response.provider,
                "is_cloud": response.is_cloud,
                "prompt_tokens": response.prompt_tokens,
                "completion_tokens": response.completion_tokens,
                "first_token_latency_ms": response.first_token_latency_ms,
            },
        )

        # 6. Deterministically parse & validate structured proposal (ADR-004)
        try:
            proposal = parse_agent_proposal(response.raw_output)
        except ProposalValidationError as exc:
            self._emit_event(
                context=context,
                event_type=EventType.SECURITY_VIOLATION,
                payload={
                    "reason": "invalid_proposal_schema",
                    "error": str(exc),
                },
            )
            raise

        # 7. Enforce ADR-008 / ADR-010 per-agent least-privilege scope
        try:
            self._spec.validate_proposal_scope(proposal)
        except AgentScopeViolationError as exc:
            self._emit_event(
                context=context,
                event_type=EventType.SECURITY_VIOLATION,
                correlation_id=proposal.proposal_id,
                payload={
                    "reason": "agent_scope_violation",
                    "proposal_kind": proposal.kind,
                    "proposed_tool": (
                        proposal.tool_name if isinstance(proposal, ToolCallProposal) else None
                    ),
                    "error": str(exc),
                },
            )
            raise

        # 8. Emit tool.proposed (if ToolCallProposal) and agent.completed events
        if isinstance(proposal, ToolCallProposal):
            self._emit_event(
                context=context,
                event_type=EventType.TOOL_PROPOSED,
                correlation_id=proposal.proposal_id,
                payload={
                    "tool_name": proposal.tool_name,
                    "declared_sensitivity": proposal.declared_sensitivity.value,
                    "rationale": proposal.rationale,
                },
            )

        self._emit_event(
            context=context,
            event_type=EventType.AGENT_COMPLETED,
            correlation_id=proposal.proposal_id,
            payload={
                "proposal_id": proposal.proposal_id,
                "proposal_kind": proposal.kind,
                "step_number": step_number,
            },
        )

        return proposal

    def _emit_event(
        self,
        context: AgentContext,
        event_type: EventType,
        payload: dict[str, object],
        correlation_id: Optional[str] = None,
    ) -> None:
        self._event_bus.publish(
            NovaEvent(
                event_type=event_type,
                task_id=context.task_id,
                session_id=context.session_id,
                node_id=context.node_id,
                agent_role=self._spec.role,
                correlation_id=correlation_id,
                payload=payload,
            )
        )