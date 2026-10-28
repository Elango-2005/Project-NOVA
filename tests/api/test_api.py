"""Unit tests for NOVA Layer 7 Backend API Service, Pause/Approve Endpoints & SSE Streaming (Task 7.1).

Verifies:
1. `GET /health` (`get_health`) reports offline-by-default posture and registered MCP servers/tools.
2. `POST /tasks` (`start_task`) and `POST /tasks/{task_id}/approval` (`submit_approval`) drive the
   full task lifecycle: start -> pause with a `PendingApprovalCard` -> user approves -> complete!
3. `GET /tasks/{task_id}/events/sse` (`stream_task_events_sse`) formats real-time `NovaEvent`s into
   W3C `text/event-stream` SSE frames (`id:`, `event:`, `data:`).
4. Permission Control Panel endpoints (`tighten_permission`, `revoke_permission`, `restore_permission`,
   `query_audit_logs`) enforce `FR-12` and `FR-15`.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
import pytest

from nova.api.app import (
    NovaLocalApiService,
    PolicyRevokeRequest,
    PolicyTightenRequest,
    StartTaskRequest,
    SubmitApprovalRequest,
)
from nova.core.agents.base import BaseSpecializedAgent
from nova.core.domain.primitives import (
    AgentRole,
    SensitivityTier,
    TaskStatus,
)
from nova.core.model_serving.interface import DeterministicMockModelAdapter
from nova.core.orchestration.checkpoint import SqliteCheckpointStore
from nova.core.orchestration.engine import NovaGraphOrchestrator
from nova.core.permissions.sandbox_and_audit import (
    CatalogBackedPermissionGate,
    FilesystemSandbox,
     PolicyModificationError,
    SandboxedMcpToolExecutor,
    SqliteToolAuditLog,
    UserPermissionPolicyStore,
)
from nova.core.platform.event_bus import InMemoryEventBus
from nova.core.tools.registry import build_default_tool_registry


class TestNovaLocalApiServiceAndSse:
    """Verification suite for Task 7.1 (`app.py`)."""

    def _build_api_fixture(self, tmp_path: Path) -> NovaLocalApiService:
        workspace = tmp_path / "workspace"
        workspace.mkdir(parents=True, exist_ok=True)
        (workspace / "oct_2026.pdf").write_text(
            "Vendor: Acme Corp | Total: $1,250.00",
            encoding="utf-8",
        )

        bus = InMemoryEventBus()
        registry = build_default_tool_registry()
        policy_store = UserPermissionPolicyStore(registry=registry)
        audit_log = SqliteToolAuditLog(tmp_path / "api_audit.db", retention_days=30)
        checkpoint_store = SqliteCheckpointStore(tmp_path / "api_checkpoints.db")

        gate = CatalogBackedPermissionGate(
            registry=registry,
            policy_store=policy_store,
            audit_log=audit_log,
            default_task_id="task_api_01",
            default_session_id="sess_api_01",
        )
        executor = SandboxedMcpToolExecutor(
            registry=registry,
            filesystem_sandbox=FilesystemSandbox(allowed_roots=(workspace,)),
            audit_log=audit_log,
            event_bus=bus,
            tool_handlers={
                "filesystem.read_file": lambda args: {
                    "content": Path(str(args["path"])).read_text(encoding="utf-8")
                },
                "gmail.send_email": lambda args: {
                    "message_id": "msg_9001",
                    "status": f"sent_to_{args['to']}",
                },
            },
            default_task_id="task_api_01",
            default_session_id="sess_api_01",
        )

        planner_model = DeterministicMockModelAdapter(
            scripted_outputs=[
                {
                    "kind": "plan",
                    "agent_role": "planner",
                    "node_id": "node_planner",
                    "thought": "Read oct_2026.pdf then email Sam.",
                    "goal_summary": "Read October invoice and email Sam.",
                    "steps": [
                        {
                            "step_id": "step_read",
                            "target_agent": "research",
                            "objective": "Read oct_2026.pdf.",
                            "depends_on": [],
                            "estimated_sensitivity": "read_only",
                        },
                        {
                            "step_id": "step_email",
                            "target_agent": "comms",
                            "objective": "Email summary to sam@accounting.example.com.",
                            "depends_on": ["step_read"],
                            "estimated_sensitivity": "irreversible_write",
                        },
                    ],
                }
            ]
        )
        research_model = DeterministicMockModelAdapter(
            scripted_outputs=[
                {
                    "kind": "tool_call",
                    "agent_role": "research",
                    "node_id": "step_read",
                    "thought": "Read oct_2026.pdf.",
                    "tool_name": "filesystem.read_file",
                    "arguments": {"path": "oct_2026.pdf"},
                    "declared_sensitivity": "read_only",
                    "rationale": "Read invoice total",
                },
                {
                    "kind": "final_answer",
                    "agent_role": "research",
                    "node_id": "step_read",
                    "thought": "Extracted total $1,250.00.",
                    "content": "Vendor: Acme Corp | Total: $1,250.00",
                    "cited_source_ids": ["oct_2026.pdf"],
                    "confidence": 0.99,
                },
            ]
        )
        comms_model = DeterministicMockModelAdapter(
            scripted_outputs=[
                {
                    "kind": "tool_call",
                    "agent_role": "comms",
                    "node_id": "step_email",
                    "thought": "Send summary email to Sam.",
                    "tool_name": "gmail.send_email",
                    "arguments": {
                        "to": "sam@accounting.example.com",
                        "subject": "October Invoice Total",
                        "body": "Vendor: Acme Corp | Total: $1,250.00",
                    },
                    "declared_sensitivity": "irreversible_write",
                    "rationale": "Send October invoice summary to Sam",
                },
                {
                    "kind": "final_answer",
                    "agent_role": "comms",
                    "node_id": "step_email",
                    "thought": "Email sent.",
                    "content": "Email sent to sam@accounting.example.com (msg_9001).",
                    "cited_source_ids": ["oct_2026.pdf"],
                    "confidence": 1.0,
                },
            ]
        )

        orchestrator = NovaGraphOrchestrator(
            agents={
                AgentRole.PLANNER: BaseSpecializedAgent(AgentRole.PLANNER, planner_model, bus),
                AgentRole.RESEARCH: BaseSpecializedAgent(AgentRole.RESEARCH, research_model, bus),
                AgentRole.COMMS: BaseSpecializedAgent(AgentRole.COMMS, comms_model, bus),
            },
            checkpoint_store=checkpoint_store,
            permission_gate=gate,
            tool_executor=executor,
            event_bus=bus,
        )

        return NovaLocalApiService(
            orchestrator=orchestrator,
            checkpoint_store=checkpoint_store,
            event_bus=bus,
            tool_registry=registry,
            policy_store=policy_store,
            audit_log=audit_log,
        )

    def test_health_endpoint_reports_offline_posture_and_catalog_counts(self) -> None:
        """GET /health returns offline_by_default=True and MCP server/tool counts."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            api = self._build_api_fixture(Path(tmp_dir))
            health = api.get_health()
            assert health.app_name == "NOVA"
            assert health.offline_by_default is True
            assert health.allow_cloud_inference is False
            assert health.registered_mcp_servers >= 15
            assert health.registered_tools >= 30

    def test_start_task_pause_approval_card_and_resume_to_completion(self) -> None:
        """POST /tasks pauses with PendingApprovalCard; POST /approval resumes and completes."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            api = self._build_api_fixture(Path(tmp_dir))

            # 1. Start task -> Research finishes, Comms pauses with PendingApprovalCard
            paused_resp = api.start_task(
                StartTaskRequest(
                    task_id="task_api_01",
                    session_id="sess_api_01",
                    user_goal="Read oct_2026.pdf and email Sam.",
                    require_citations=True,
                )
            )
            assert paused_resp.status == TaskStatus.AWAITING_APPROVAL
            assert paused_resp.completed_nodes == ("step_read",)
            assert len(paused_resp.pending_approvals) == 1

            card = paused_resp.pending_approvals[0]
            assert card.node_id == "step_email"
            assert card.tool_name == "gmail.send_email"
            assert card.arguments["to"] == "sam@accounting.example.com"

            # 2. GET /tasks/{task_id} returns the exact same paused checkpoint state
            fetched = api.get_task_status("task_api_01")
            assert fetched.status == TaskStatus.AWAITING_APPROVAL
            assert fetched.pending_approvals[0].proposal_id == card.proposal_id

            # 3. POST /tasks/{task_id}/approval with approved=True completes the task!
            completed_resp = api.submit_approval(
                "task_api_01",
                SubmitApprovalRequest(
                    node_id="step_email",
                    proposal_id=card.proposal_id,
                    approved=True,
                    reason="Approved from Desktop UI",
                    require_citations=True,
                ),
            )
            assert completed_resp.status == TaskStatus.COMPLETED
            assert completed_resp.completed_nodes == ("step_email", "step_read")
            assert len(completed_resp.pending_approvals) == 0
            assert "msg_9001" in completed_resp.node_outputs["step_email"]
            assert completed_resp.cited_sources["step_read"] == ("oct_2026.pdf",)

            # 4. Audit log endpoint returns both executed tools (`filesystem.read_file` & `gmail.send_email`)
            audit_entries = api.query_audit_logs(task_id="task_api_01")
            assert len(audit_entries) == 2

    def test_sse_event_stream_formats_valid_w3c_event_frames(self) -> None:
        """GET /tasks/{task_id}/events/sse yields W3C `id:`, `event:`, `data:` frames."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            api = self._build_api_fixture(Path(tmp_dir))
            api.start_task(
                StartTaskRequest(
                    task_id="task_api_01",
                    session_id="sess_api_01",
                    user_goal="Read oct_2026.pdf and email Sam.",
                )
            )

            sse_frames = list(api.stream_task_events_sse("task_api_01"))
            assert len(sse_frames) >= 6
            first_frame = sse_frames[0]
            assert first_frame.startswith("id: evt_")
            assert "\nevent: task.started\n" in first_frame
            assert "\ndata: {" in first_frame
            assert first_frame.endswith("\n\n")

            # Parse the JSON payload from the `data:` line
            data_line = [
                line for line in first_frame.splitlines() if line.startswith("data: ")
            ][0]
            parsed = json.loads(data_line[len("data: ") :])
            assert parsed["task_id"] == "task_api_01"
            assert parsed["event_type"] == "task.started"

    def test_permission_control_panel_endpoints_tighten_revoke_and_restore(self) -> None:
        """FR-15: API endpoints tighten, revoke, and restore permissions while blocking loosening."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            api = self._build_api_fixture(Path(tmp_dir))

            # 1. Tighten `filesystem.read_file` to `IRREVERSIBLE_WRITE`
            tightened = api.tighten_permission(
                PolicyTightenRequest(
                    tool_name="filesystem.read_file",
                    new_tier=SensitivityTier.IRREVERSIBLE_WRITE,
                )
            )
            assert tightened["effective_tier"] == "irreversible_write"

            # 2. Attempting to loosen `gmail.send_email` to `READ_ONLY` fails closed
            with pytest.raises(PolicyModificationError):
                api.tighten_permission(
                    PolicyTightenRequest(
                        tool_name="gmail.send_email",
                        new_tier=SensitivityTier.READ_ONLY,
                    )
                )

            # 3. Revoke and restore `search.web_search` for Research agent
            after_revoke = api.revoke_permission(
                PolicyRevokeRequest(
                    tool_name="search.web_search",
                    agent_role=AgentRole.RESEARCH,
                )
            )
            assert "search.web_search" in after_revoke["revoked_agent_tools"]["research"]

            after_restore = api.restore_permission(
                PolicyRevokeRequest(
                    tool_name="search.web_search",
                    agent_role=AgentRole.RESEARCH,
                )
            )
            assert "research" not in after_restore["revoked_agent_tools"]