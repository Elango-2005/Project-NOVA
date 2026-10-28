"""NOVA Layer 7 Backend API Surface: FastAPI Local Controller, Typed Schemas & SSE Streaming (Section 12.1 / Phase 16 / ADR-002 / ADR-009 / ADR-010 / FR-12 / FR-15).

Plain-English Purpose (The Reception Desk & Live PA Loudspeaker):
All 6 internal layers of NOVA (Platform, Model Serving, MCP Tools/Sandbox, Memory/RAG,
Specialized Agents, and the Graph Orchestrator) now work together.
`app.py` is the **Front Reception Desk** (`NovaLocalApiService` + `create_nova_fastapi_app`)
that connects your Desktop UI (Electron / Tauri + React) to the Python backend over localhost:

1. **`GET /health` (System Dashboard Check)**:
   Reports NOVA's offline-first status, active local model engine (`llama.cpp` or `vLLM`),
   hardware profile, and number of registered MCP servers and tools.
2. **`POST /tasks` & `GET /tasks/{task_id}` (Start or Inspect a Task)**:
   Submits a user goal to `NovaGraphOrchestrator` and returns a clean `TaskStatusResponse`
   showing completed steps, final answers, or any tool currently paused at `awaiting_approval`.
3. **`POST /tasks/{task_id}/approval` (Click "Approve" or "Deny" in the UI — ADR-010)**:
   Sends your human approval or denial click to the paused task so NOVA either runs the
   sandboxed tool and finishes, or aborts safely.
4. **`GET /tasks/{task_id}/events/sse` (Live Server-Sent Events Stream — ADR-002 & ADR-009)**:
   Streams every task, agent, tool, and token event in standard `text/event-stream` SSE format
   (`id: ...\nevent: ...\ndata: {...}\n\n`) so the UI never freezes.
5. **`GET /permissions/policies`, `POST /permissions/tighten`, `POST /permissions/revoke`, `GET /audit/logs` (FR-12 & FR-15)**:
   Lets you inspect, tighten, or revoke tool permissions and query the 30-day SQLite Black-Box Audit Log.
"""

from __future__ import annotations

import json
from typing import Any, Dict, Iterator, List, Mapping, Optional, Sequence, Tuple

from pydantic import BaseModel, Field

try:
    from pydantic import ConfigDict

    PYDANTIC_V2 = True
except ImportError:  # pragma: no cover
    PYDANTIC_V2 = False

try:
    from fastapi import FastAPI, HTTPException
    from fastapi.responses import StreamingResponse

    FASTAPI_AVAILABLE = True
except ImportError:  # pragma: no cover
    FastAPI = None  # type: ignore[assignment,misc]
    HTTPException = Exception  # type: ignore[assignment,misc]
    StreamingResponse = None  # type: ignore[assignment,misc]
    FASTAPI_AVAILABLE = False

from nova.core.config.settings import NovaSettings, load_settings
from nova.core.domain.context import UntrustedDataItem
from nova.core.domain.events import NovaEvent
from nova.core.domain.primitives import (
    AgentRole,
    SensitivityTier,
    TaskStatus,
)
from nova.core.orchestration.checkpoint import CheckpointStorePort
from nova.core.orchestration.engine import NovaGraphOrchestrator
from nova.core.orchestration.state import TaskExecutionState
from nova.core.permissions.sandbox_and_audit import (
    SqliteToolAuditLog,
    ToolAuditEntry,
    UserPermissionPolicyStore,
)
from nova.core.platform.event_bus import InMemoryEventBus
from nova.core.tools.registry import ToolRegistry, build_default_tool_registry


# ============================================================================
# Typed Request & Response Schemas (Layer 7 API Contract — Section 12.1)
# ============================================================================


class HealthStatusResponse(BaseModel):
    """Health and capability summary returned by `GET /health`."""

    if PYDANTIC_V2:
        model_config = ConfigDict(frozen=True, extra="forbid")
    else:

        class Config:
            allow_mutation = False
            extra = "forbid"

    app_name: str = Field(...)
    version: str = Field(...)
    offline_by_default: bool = Field(...)
    allow_cloud_inference: bool = Field(...)
    registered_mcp_servers: int = Field(..., ge=0)
    registered_tools: int = Field(..., ge=0)


class StartTaskRequest(BaseModel):
    """Payload for `POST /tasks` to launch a new orchestrated task."""

    if PYDANTIC_V2:
        model_config = ConfigDict(frozen=True, extra="forbid")
    else:

        class Config:
            allow_mutation = False
            extra = "forbid"

    task_id: str = Field(..., min_length=1)
    session_id: str = Field(..., min_length=1)
    user_goal: str = Field(..., min_length=1)
    require_citations: bool = Field(default=False)


class PendingApprovalCard(BaseModel):
    """UI card payload shown when a task pauses in `AWAITING_APPROVAL` (ADR-010)."""

    if PYDANTIC_V2:
        model_config = ConfigDict(frozen=True, extra="forbid")
    else:

        class Config:
            allow_mutation = False
            extra = "forbid"

    node_id: str = Field(..., min_length=1)
    proposal_id: str = Field(..., min_length=1)
    agent_role: AgentRole = Field(...)
    tool_name: str = Field(..., min_length=1)
    arguments: Dict[str, Any] = Field(default_factory=dict)
    declared_sensitivity: SensitivityTier = Field(...)
    rationale: str = Field(..., min_length=1)


class TaskStatusResponse(BaseModel):
    """Structured task status returned by `POST /tasks`, `GET /tasks/{task_id}`, and `/approval`."""

    if PYDANTIC_V2:
        model_config = ConfigDict(frozen=True, extra="forbid")
    else:

        class Config:
            allow_mutation = False
            extra = "forbid"

    task_id: str = Field(..., min_length=1)
    session_id: str = Field(..., min_length=1)
    user_goal: str = Field(..., min_length=1)
    status: TaskStatus = Field(...)
    completed_nodes: Tuple[str, ...] = Field(default_factory=tuple)
    pending_approvals: Tuple[PendingApprovalCard, ...] = Field(default_factory=tuple)
    node_outputs: Dict[str, str] = Field(default_factory=dict)
    cited_sources: Dict[str, Tuple[str, ...]] = Field(default_factory=dict)
    failure_reason: Optional[str] = Field(default=None)


class SubmitApprovalRequest(BaseModel):
    """Payload for `POST /tasks/{task_id}/approval` when the user clicks Approve or Deny (ADR-010)."""

    if PYDANTIC_V2:
        model_config = ConfigDict(frozen=True, extra="forbid")
    else:

        class Config:
            allow_mutation = False
            extra = "forbid"

    node_id: str = Field(..., min_length=1)
    proposal_id: str = Field(..., min_length=1)
    approved: bool = Field(...)
    reason: str = Field(default="User decision via NOVA UI")
    require_citations: bool = Field(default=False)


class PolicyRevokeRequest(BaseModel):
    """Payload for `POST /permissions/revoke` or `/permissions/restore` (FR-15)."""

    if PYDANTIC_V2:
        model_config = ConfigDict(frozen=True, extra="forbid")
    else:

        class Config:
            allow_mutation = False
            extra = "forbid"

    tool_name: str = Field(..., min_length=1)
    agent_role: Optional[AgentRole] = Field(default=None)


class PolicyTightenRequest(BaseModel):
    """Payload for `POST /permissions/tighten` (FR-15 / Section 14)."""

    if PYDANTIC_V2:
        model_config = ConfigDict(frozen=True, extra="forbid")
    else:

        class Config:
            allow_mutation = False
            extra = "forbid"

    tool_name: str = Field(..., min_length=1)
    new_tier: SensitivityTier = Field(...)


# ============================================================================
# SSE Formatting Utility (ADR-002 / ADR-009 Streaming-First Pipeline)
# ============================================================================


def format_nova_event_as_sse(event: NovaEvent) -> str:
    """Format a structured `NovaEvent` into a standard W3C Server-Sent Events (`text/event-stream`) frame."""
    data_dict = {
        "event_id": event.event_id,
        "event_type": event.event_type.value,
        "timestamp_utc": event.timestamp_utc,
        "task_id": event.task_id,
        "session_id": event.session_id,
        "node_id": event.node_id,
        "agent_role": event.agent_role.value if event.agent_role else None,
        "correlation_id": event.correlation_id,
        "payload": event.payload,
    }
    json_line = json.dumps(data_dict, sort_keys=True, separators=(",", ":"))
    return f"id: {event.event_id}\nevent: {event.event_type.value}\ndata: {json_line}\n\n"


# ============================================================================
# Local API Service Controller & FastAPI Factory
# ============================================================================


class NovaLocalApiService:
    """Deterministic Layer 7 Backend API Controller uniting Orchestrator, Policy Store, Audit Log & Event Stream."""

    def __init__(
        self,
        orchestrator: NovaGraphOrchestrator,
        checkpoint_store: CheckpointStorePort,
        event_bus: InMemoryEventBus,
        tool_registry: Optional[ToolRegistry] = None,
        policy_store: Optional[UserPermissionPolicyStore] = None,
        audit_log: Optional[SqliteToolAuditLog] = None,
        settings: Optional[NovaSettings] = None,
    ) -> None:
        self._orchestrator = orchestrator
        self._checkpoint_store = checkpoint_store
        self._event_bus = event_bus
        self._registry = tool_registry or build_default_tool_registry()
        self._policy_store = policy_store or UserPermissionPolicyStore(self._registry)
        self._audit_log = audit_log
        self._settings = settings or load_settings(environ={})

    def get_health(self) -> HealthStatusResponse:
        """Return system health, offline-first posture, and registered MCP catalog counts."""
        return HealthStatusResponse(
            app_name=self._settings.app_name,
            version=self._settings.version,
            offline_by_default=self._settings.offline_by_default,
            allow_cloud_inference=self._settings.allow_cloud_inference,
            registered_mcp_servers=len(self._registry.registered_server_ids),
            registered_tools=len(self._registry.registered_tool_names),
        )

    def start_task(
        self,
        request: StartTaskRequest,
        *,
        untrusted_data: Sequence[UntrustedDataItem] = (),
    ) -> TaskStatusResponse:
        """Launch a task through `NovaGraphOrchestrator` and return its structured status."""
        state = self._orchestrator.start_task(
            task_id=request.task_id,
            session_id=request.session_id,
            user_goal=request.user_goal,
            untrusted_data=untrusted_data,
            require_citations=request.require_citations,
        )
        return self._state_to_response(state)

    def get_task_status(self, task_id: str) -> TaskStatusResponse:
        """Load the latest checkpoint for `task_id` and return its structured status."""
        envelope = self._checkpoint_store.load_latest(task_id)
        if envelope is None:
            raise KeyError(f"Task {task_id!r} not found.")
        return self._state_to_response(envelope.state)

    def submit_approval(
        self,
        task_id: str,
        request: SubmitApprovalRequest,
        *,
        untrusted_data: Sequence[UntrustedDataItem] = (),
    ) -> TaskStatusResponse:
        """Apply the user's Approve/Deny decision to a paused task and return the updated status."""
        state = self._orchestrator.submit_user_approval(
            task_id=task_id,
            node_id=request.node_id,
            proposal_id=request.proposal_id,
            approved=request.approved,
            reason=request.reason,
            untrusted_data=untrusted_data,
            require_citations=request.require_citations,
        )
        return self._state_to_response(state)

    def stream_task_events_sse(self, task_id: str) -> Iterator[str]:
        """Yield all published events for `task_id` formatted as W3C Server-Sent Events (`text/event-stream`)."""
        for event in self._event_bus.published_events:
            if event.task_id == task_id:
                yield format_nova_event_as_sse(event)

    def inspect_permission_policies(self) -> Dict[str, Any]:
        """Return active user permission overrides (`FR-15`)."""
        return self._policy_store.inspect_policies()

    def tighten_permission(self, request: PolicyTightenRequest) -> Dict[str, Any]:
        """Tighten a tool's sensitivity tier (`FR-15` / Section 14)."""
        updated_tier = self._policy_store.tighten_tool_sensitivity(
            request.tool_name,
            request.new_tier,
        )
        return {
            "tool_name": request.tool_name,
            "effective_tier": updated_tier.value,
            "policies": self._policy_store.inspect_policies(),
        }

    def revoke_permission(self, request: PolicyRevokeRequest) -> Dict[str, Any]:
        """Revoke a tool globally or for a specific specialized agent (`FR-15`)."""
        self._policy_store.revoke_tool(
            request.tool_name,
            agent_role=request.agent_role,
        )
        return self._policy_store.inspect_policies()

    def restore_permission(self, request: PolicyRevokeRequest) -> Dict[str, Any]:
        """Restore a previously revoked tool (`FR-15`)."""
        self._policy_store.restore_tool(
            request.tool_name,
            agent_role=request.agent_role,
        )
        return self._policy_store.inspect_policies()

    def query_audit_logs(
        self,
        *,
        task_id: Optional[str] = None,
        agent_role: Optional[AgentRole] = None,
        tool_name: Optional[str] = None,
    ) -> Tuple[ToolAuditEntry, ...]:
        """Query the SQLite tool and permission audit log (`FR-12` / `NFR-08`)."""
        if self._audit_log is None:
            return ()
        return self._audit_log.query_logs(
            task_id=task_id,
            agent_role=agent_role,
            tool_name=tool_name,
        )

    @staticmethod
    def _state_to_response(state: TaskExecutionState) -> TaskStatusResponse:
        pending_cards: List[PendingApprovalCard] = []
        outputs: Dict[str, str] = {}
        citations: Dict[str, Tuple[str, ...]] = {}

        for node_id, rec in state.node_records.items():
            if rec.pending_tool_proposal is not None:
                prop = rec.pending_tool_proposal
                pending_cards.append(
                    PendingApprovalCard(
                        node_id=node_id,
                        proposal_id=prop.proposal_id,
                        agent_role=prop.agent_role,
                        tool_name=prop.tool_name,
                        arguments=dict(prop.arguments),
                        declared_sensitivity=prop.declared_sensitivity,
                        rationale=prop.rationale,
                    )
                )
            if rec.final_answer is not None:
                outputs[node_id] = rec.final_answer.content
                citations[node_id] = rec.final_answer.cited_source_ids

        return TaskStatusResponse(
            task_id=state.task_id,
            session_id=state.session_id,
            user_goal=state.user_goal,
            status=state.status,
            completed_nodes=tuple(sorted(state.completed_node_ids)),
            pending_approvals=tuple(pending_cards),
            node_outputs=outputs,
            cited_sources=citations,
            failure_reason=state.failure_reason,
        )


def create_nova_fastapi_app(api_service: NovaLocalApiService) -> Any:
    """Create the FastAPI ASGI application wired to `api_service` (Section 12.1 / Phase 16)."""
    if not FASTAPI_AVAILABLE:  # pragma: no cover
        raise RuntimeError("FastAPI is not installed in the current Python environment.")

    app = FastAPI(
        title="NOVA Local Personal AI Operating Layer API",
        version="0.1.0",
        description="Single-user, offline-first agentic backend API (ADR-001, ADR-009, ADR-010).",
    )

    @app.get("/health", response_model=HealthStatusResponse)
    def get_health() -> HealthStatusResponse:
        return api_service.get_health()

    @app.post("/tasks", response_model=TaskStatusResponse)
    def start_task(request: StartTaskRequest) -> TaskStatusResponse:
        return api_service.start_task(request)

    @app.get("/tasks/{task_id}", response_model=TaskStatusResponse)
    def get_task_status(task_id: str) -> TaskStatusResponse:
        try:
            return api_service.get_task_status(task_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.post("/tasks/{task_id}/approval", response_model=TaskStatusResponse)
    def submit_approval(task_id: str, request: SubmitApprovalRequest) -> TaskStatusResponse:
        try:
            return api_service.submit_approval(task_id, request)
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.get("/tasks/{task_id}/events/sse")
    def stream_task_events(task_id: str) -> Any:
        return StreamingResponse(
            api_service.stream_task_events_sse(task_id),
            media_type="text/event-stream",
        )

    @app.get("/permissions/policies")
    def inspect_policies() -> Dict[str, Any]:
        return api_service.inspect_permission_policies()

    @app.post("/permissions/tighten")
    def tighten_permission(request: PolicyTightenRequest) -> Dict[str, Any]:
        try:
            return api_service.tighten_permission(request)
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/permissions/revoke")
    def revoke_permission(request: PolicyRevokeRequest) -> Dict[str, Any]:
        return api_service.revoke_permission(request)

    @app.post("/permissions/restore")
    def restore_permission(request: PolicyRevokeRequest) -> Dict[str, Any]:
        return api_service.restore_permission(request)

    @app.get("/audit/logs")
    def get_audit_logs(task_id: Optional[str] = None) -> List[Dict[str, Any]]:
        entries = api_service.query_audit_logs(task_id=task_id)
        out: List[Dict[str, Any]] = []
        for entry in entries:
            if hasattr(entry, "model_dump"):
                out.append(entry.model_dump(mode="json"))  # type: ignore[attr-defined]
            else:  # pragma: no cover
                out.append(json.loads(entry.json()))
        return out

    return app


__all__ = [
    "HealthStatusResponse",
    "NovaLocalApiService",
    "PendingApprovalCard",
    "PolicyRevokeRequest",
    "PolicyTightenRequest",
    "StartTaskRequest",
    "SubmitApprovalRequest",
    "TaskStatusResponse",
    "create_nova_fastapi_app",
    "format_nova_event_as_sse",
]