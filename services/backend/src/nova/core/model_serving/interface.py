"""NOVA Layer 2 Model Serving Interface (ADR-001 / ADR-004 / ADR-006).

Defines the framework-agnostic, runtime-replaceable contract between Layer 5 Agents
and Layer 2 Model Serving:
- ADR-006: Application and agent code depend strictly on `ModelReasoningPort`, never on
  concrete `llama.cpp`, `vLLM`, `Ollama`, or cloud SDK objects.
- ADR-004: Models are invoked as bounded structured-output functions returning a discrete
  payload to be validated by `parse_agent_proposal()`.
- Engineering Principle 9: Provides `DeterministicMockModelAdapter` for deterministic
  unit and integration testing of agent and orchestration control flow.
"""

from __future__ import annotations

from collections import deque
from typing import Any, Deque, Dict, List, Mapping, Protocol, Sequence, Tuple, Union, runtime_checkable

from pydantic import BaseModel, Field

try:
    from pydantic import ConfigDict

    PYDANTIC_V2 = True
except ImportError:  # pragma: no cover
    PYDANTIC_V2 = False

from nova.core.domain.primitives import AgentRole


class ModelInferenceRequest(BaseModel):
    """Immutable request envelope sent from a bounded agent node to the Model Router."""

    if PYDANTIC_V2:
        model_config = ConfigDict(frozen=True, extra="forbid")
    else:

        class Config:
            allow_mutation = False
            extra = "forbid"

    task_id: str = Field(..., min_length=1)
    session_id: str = Field(..., min_length=1)
    node_id: str = Field(..., min_length=1)
    agent_role: AgentRole = Field(..., description="Requesting specialized agent role")
    model_tier: str = Field(
        default="mid_tier_local",
        description="Requested capability tier from AgentSpecification (Section 12.3)",
    )
    messages: Tuple[Dict[str, str], ...] = Field(
        ..., description="Strictly separated system (instruction) and user (delimited data) messages"
    )
    output_schema_name: str = Field(
        default="AgentProposal/v1",
        description="Target JSON Schema contract name",
    )
    allow_cloud_routing: bool = Field(
        default=False,
        description="True ONLY if ADR-001 and Section 16.2 data-governance checks passed",
    )


class ModelInferenceResponse(BaseModel):
    """Immutable response envelope returned by a local or opt-in cloud model adapter."""

    if PYDANTIC_V2:
        model_config = ConfigDict(frozen=True, extra="forbid")
    else:

        class Config:
            allow_mutation = False
            extra = "forbid"

    raw_output: Union[str, Dict[str, Any]] = Field(
        ..., description="Raw JSON string or structured dict produced by the model runtime"
    )
    model_id: str = Field(..., min_length=1, description="Executed model identifier")
    provider: str = Field(
        default="llamacpp",
        description="Serving backend identifier (e.g., 'llamacpp', 'vllm', 'cloud', 'mock')",
    )
    is_cloud: bool = Field(
        default=False,
        description="True if inference crossed a network boundary to a cloud provider",
    )
    prompt_tokens: int = Field(default=0, ge=0)
    completion_tokens: int = Field(default=0, ge=0)
    first_token_latency_ms: int = Field(default=0, ge=0)


@runtime_checkable
class ModelReasoningPort(Protocol):
    """Abstract interface for Layer 2 Model Serving (ADR-006)."""

    def generate_structured(self, request: ModelInferenceRequest) -> ModelInferenceResponse:
        """Execute a single bounded reasoning call and return a structured response."""


class DeterministicMockModelAdapter:
    """Deterministic in-memory ModelReasoningPort implementation for testing (Principle 9)."""

    def __init__(
        self,
        scripted_outputs: Sequence[Union[str, Mapping[str, Any], ModelInferenceResponse]] = (),
        default_model_id: str = "qwen2.5-7b-instruct-q4_k_m.gguf",
    ) -> None:
        self._queue: Deque[Union[str, Mapping[str, Any], ModelInferenceResponse]] = deque(
            scripted_outputs
        )
        self._default_model_id = default_model_id
        self.recorded_requests: List[ModelInferenceRequest] = []

    def enqueue(
        self, output: Union[str, Mapping[str, Any], ModelInferenceResponse]
    ) -> None:
        """Append a scripted model response to the FIFO queue."""
        self._queue.append(output)

    def generate_structured(self, request: ModelInferenceRequest) -> ModelInferenceResponse:
        """Record the request and pop the next scripted response."""
        self.recorded_requests.append(request)
        if not self._queue:
            raise RuntimeError(
                "DeterministicMockModelAdapter has no scripted responses remaining in queue."
            )
        next_item = self._queue.popleft()
        if isinstance(next_item, ModelInferenceResponse):
            return next_item
        raw_payload: Union[str, Dict[str, Any]] = (
            dict(next_item) if isinstance(next_item, Mapping) else str(next_item)
        )
        return ModelInferenceResponse(
            raw_output=raw_payload,
            model_id=self._default_model_id,
            provider="mock_local",
            is_cloud=False,
            prompt_tokens=128,
            completion_tokens=64,
            first_token_latency_ms=110,
        )