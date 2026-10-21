"""NOVA Structured Event Contracts (Layer 1 / ADR-002 / ADR-009 / NFR-08).

Implements the observable event model from NOVA_ADR.md:
- ADR-002: Typed event schemas per domain (task, agent, model, tool, permission, memory, security).
- Engineering Principle 3 & NFR-08: Every agent step, tool call, memory op, and model call emits a structured event.
- Section 16.2: Credentials/Secrets must never be logged in plaintext; NovaEvent deterministically redacts secret keys.
"""

from __future__ import annotations

import re
import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, Mapping, Optional

from pydantic import BaseModel, Field

try:
    from pydantic import ConfigDict, field_validator, model_validator

    PYDANTIC_V2 = True
except ImportError:  # pragma: no cover
    from pydantic import root_validator, validator

    PYDANTIC_V2 = False

from nova.core.domain.primitives import (
    CURRENT_SCHEMA_VERSION,
    AgentRole,
    validate_schema_version,
)

REDACTED_VALUE = "[REDACTED_SECRET]"

# Matches sensitive credential keys while avoiding benign telemetry keys like token_count or max_tokens
_SECRET_KEY_PATTERN = re.compile(
    r"(^|_)(api_?key|secret|password|passwd|access_?token|refresh_?token|auth_?token|"
    r"client_?secret|private_?key|credential|authorization|bearer)($|_)",
    re.IGNORECASE,
)
_SAFE_TOKEN_KEYS = frozenset(
    {
        "token_count",
        "prompt_tokens",
        "completion_tokens",
        "total_tokens",
        "max_tokens",
        "first_token_latency_ms",
        "token_text",
        "token_index",
    }
)


def redact_secrets(data: Any) -> Any:
    """Recursively redact secret or credential fields from event payloads (Section 16.2 / NFR-08)."""
    if isinstance(data, Mapping):
        sanitized: Dict[str, Any] = {}
        for k, v in data.items():
            key_str = str(k)
            key_lower = key_str.lower()
            if key_lower not in _SAFE_TOKEN_KEYS and (
                _SECRET_KEY_PATTERN.search(key_lower) or key_lower == "token"
            ):
                sanitized[key_str] = REDACTED_VALUE
            else:
                sanitized[key_str] = redact_secrets(v)
        return sanitized
    if isinstance(data, list):
        return [redact_secrets(item) for item in data]
    if isinstance(data, tuple):
        return tuple(redact_secrets(item) for item in data)
    return data


class EventType(str, Enum):
    """Canonical observable event types emitted across NOVA layers (ADR-002)."""

    # Task lifecycle
    TASK_STARTED = "task.started"
    TASK_PLANNING = "task.planning"
    TASK_COMPLETED = "task.completed"
    TASK_FAILED = "task.failed"

    # Agent lifecycle & cognition
    AGENT_STARTED = "agent.started"
    AGENT_COMPLETED = "agent.completed"
    AGENT_REFLECTION_TRIGGERED = "agent.reflection_triggered"

    # Model serving & streaming (ADR-009)
    MODEL_GENERATION_STARTED = "model.generation.started"
    MODEL_TOKEN_STREAMED = "model.token.streamed"
    MODEL_GENERATION_COMPLETED = "model.generation.completed"

    # Tool & Permission Gate lifecycle (ADR-005 / ADR-010)
    TOOL_PROPOSED = "tool.proposed"
    PERMISSION_REQUESTED = "permission.requested"
    PERMISSION_APPROVED = "permission.approved"
    PERMISSION_DENIED = "permission.denied"
    TOOL_STARTED = "tool.started"
    TOOL_COMPLETED = "tool.completed"
    TOOL_FAILED = "tool.failed"

    # Memory lifecycle (ADR-007)
    MEMORY_READ = "memory.read"
    MEMORY_WRITE = "memory.write"
    MEMORY_PROMOTED = "memory.promoted"

    # Security & policy enforcement (Phase 15)
    SECURITY_VIOLATION = "security.violation"


class NovaEvent(BaseModel):
    """Immutable structured event envelope published on NOVA's event bus."""

    if PYDANTIC_V2:
        model_config = ConfigDict(frozen=True, extra="forbid")
    else:

        class Config:
            allow_mutation = False
            extra = "forbid"

    schema_version: str = Field(
        default=CURRENT_SCHEMA_VERSION,
        description="Versioned event contract schema tag (Engineering Principle 7)",
    )
    event_id: str = Field(
        default_factory=lambda: f"evt_{uuid.uuid4().hex}",
        description="Unique event identifier",
    )
    event_type: EventType = Field(..., description="Domain event type")
    timestamp_utc: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(),
        description="ISO-8601 UTC timestamp when the event was created",
    )
    task_id: str = Field(..., min_length=1, description="Associated task identifier")
    session_id: str = Field(..., min_length=1, description="Active user session identifier")
    node_id: Optional[str] = Field(
        default=None,
        description="Execution graph node ID if emitted inside a graph node (ADR-003)",
    )
    agent_role: Optional[AgentRole] = Field(
        default=None,
        description="Acting specialized agent role if applicable (ADR-008)",
    )
    correlation_id: Optional[str] = Field(
        default=None,
        description="Optional parent event or proposal ID for causal tracing",
    )
    payload: Dict[str, Any] = Field(
        default_factory=dict,
        description="Redacted domain-specific event payload",
    )

    if PYDANTIC_V2:

        @field_validator("schema_version")
        @classmethod
        def _validate_version(cls, value: str) -> str:
            return validate_schema_version(value)

        @field_validator("payload", mode="before")
        @classmethod
        def _sanitize_payload(cls, value: Any) -> Dict[str, Any]:
            if value is None:
                return {}
            if not isinstance(value, Mapping):
                raise ValueError("Event payload must be a mapping.")
            return dict(redact_secrets(value))

    else:

        @validator("schema_version")
        def _validate_version(cls, value: str) -> str:
            return validate_schema_version(value)

        @validator("payload", pre=True, always=True)
        def _sanitize_payload(cls, value: Any) -> Dict[str, Any]:
            if value is None:
                return {}
            if not isinstance(value, Mapping):
                raise ValueError("Event payload must be a mapping.")
            return dict(redact_secrets(value))