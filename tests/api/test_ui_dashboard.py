"""Unit and integration tests for NOVA Phase 9: Mission-Control Web UI & Telemetry Dashboard (`test_ui_dashboard.py`).

Verifies:
1. `render_mission_control_html()` generates the complete Jarvis-style dark-mode Mission Control UI
   with all 4 interactive panels (Command Center, Audit Trail, Hybrid RAG Ingest, MCP Tool Catalog)
   and client-side fetch wiring (`/tasks`, `/approve`, `/rag/ingest`, `/ui/overview`).
2. `get_dashboard_overview_snapshot(container)` returns live telemetry (`offline_by_default=True`,
   `registered_tools_count == 43`, and real-time `recent_audit_logs` after a task runs).
3. `get_dashboard_overview_snapshot` tracks multi-step paused email tasks (`AWAITING_APPROVAL`) and
   subsequent user approval cleanly.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

from nova.api.app import StartTaskRequest, SubmitApprovalRequest
from nova.api.dashboard import (
    get_dashboard_overview_snapshot,
    render_mission_control_html,
)
from nova.core.domain.primitives import TaskStatus
from nova.main import NovaRuntimeContainer


class TestNovaMissionControlDashboard:
    """Verification suite for Phase 9 (`services/backend/src/nova/api/dashboard.py`)."""

    def test_render_mission_control_html_includes_all_four_interactive_panels(
        self,
    ) -> None:
        """Ensures the Mission Control HTML contains all 4 operational panels and API endpoints."""
        html = render_mission_control_html(api_base_url="http://127.0.0.1:8000")

        assert "<!DOCTYPE html>" in html
        assert "NOVA MISSION CONTROL" in html
        assert "1. Agentic Command Center (State-Graph Orchestrator)" in html
        assert "2. 30-Day SQLite Black-Box Tool Audit Trail" in html
        assert "3. Local Hybrid RAG Knowledge Ingest" in html
        assert "4. MCP Tool Catalog &amp; Permission Tiers" in html

        # Verify client-side API wiring
        assert "http://127.0.0.1:8000" in html
        assert "/ui/overview" in html
        assert "/tasks" in html
        assert "/approve" in html
        assert "/rag/ingest" in html

    def test_dashboard_overview_snapshot_reflects_live_tools_and_audit_trail(
        self,
    ) -> None:
        """Verifies get_dashboard_overview_snapshot updates live after a file-writing task runs."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            container = NovaRuntimeContainer(
                data_dir=tmp_path / "data",
                workspace_dir=tmp_path / "workspace",
            )

            initial_snap = get_dashboard_overview_snapshot(container)
            assert initial_snap["status"] == "ok"
            assert initial_snap["offline_by_default"] is True
            assert initial_snap["registered_mcp_servers"] == 21
            assert initial_snap["registered_tools_count"] == 43
            assert len(initial_snap["tools"]) == 43
            assert initial_snap["recent_audit_logs"] == []

            # Run a real task and confirm the dashboard snapshot includes the new audit log entry
            resp = container.api_service.start_task(
                StartTaskRequest(
                    task_id="task_ui_test",
                    session_id="sess_ui_01",
                    user_goal="Write UI verification summary to ui_check.txt",
                )
            )
            assert resp.status == TaskStatus.COMPLETED

            updated_snap = get_dashboard_overview_snapshot(container)
            assert len(updated_snap["recent_audit_logs"]) == 1
            entry = updated_snap["recent_audit_logs"][0]
            assert entry["tool_name"] == "filesystem.write_file"
            assert entry["decision"] == "auto_approved"
            assert entry["execution_status"] == "success"

    def test_dashboard_snapshot_tracks_paused_email_approval_and_resolution(
        self,
    ) -> None:
        """Verifies the UI workflow from paused email approval to audit log recording."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            container = NovaRuntimeContainer(
                data_dir=tmp_path / "data",
                workspace_dir=tmp_path / "workspace",
            )
            (container.workspace_dir / "summary.txt").write_text(
                "All 105 tests green.",
                encoding="utf-8",
            )

            paused = container.api_service.start_task(
                StartTaskRequest(
                    task_id="task_ui_email",
                    session_id="sess_ui_02",
                    user_goal="Read summary.txt and email it to team@nova.local",
                )
            )
            assert paused.status == TaskStatus.AWAITING_APPROVAL
            snap_while_paused = get_dashboard_overview_snapshot(container)
            assert [e["tool_name"] for e in snap_while_paused["recent_audit_logs"]] == [
                "filesystem.read_file"
            ]

            card = paused.pending_approvals[0]
            resumed = container.api_service.submit_approval(
                "task_ui_email",
                SubmitApprovalRequest(
                    node_id=card.node_id,
                    proposal_id=card.proposal_id,
                    approved=True,
                    reason="Approved from Mission Control UI test",
                ),
            )
            assert resumed.status == TaskStatus.COMPLETED
            snap_after_approve = get_dashboard_overview_snapshot(container)
            assert [e["tool_name"] for e in snap_after_approve["recent_audit_logs"]] == [
                "filesystem.read_file",
                "gmail.send_email",
            ]