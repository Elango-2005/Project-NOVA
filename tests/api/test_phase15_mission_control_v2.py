"""Unit and integration tests for NOVA Phase 15: Mission Control v2 — Live Brain Switcher & SHA-256 Audit Seal UI (`test_phase15_mission_control_v2.py`).

Verifies:
1. `render_mission_control_html()` includes Panel 5 (`5. Live Brain Switcher & 6GB VRAM GPU Offload`),
   header pills for `Brain`, `GPU Split`, and `Audit Chain: SHA-256 VERIFIED`, and client-side
   `switchBrain('gguf_qwen35')` wiring (`/brain/switch`).
2. `get_dashboard_overview_snapshot(container)` includes `brain_and_security` with the 6.0 GB VRAM
   `30/36` GPU layer offload plan and cryptographic SHA-256 audit chain status.
3. Switching the brain via `container.brain_controller.switch_brain_mode(BrainEngineMode.GGUF_QWEN35)`
   updates the Mission Control overview snapshot and seals subsequent task executions into the
   verified SHA-256 audit chain.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

from nova.api.app import StartTaskRequest
from nova.api.dashboard import (
    BrainEngineMode,
    get_dashboard_overview_snapshot,
    render_mission_control_html,
)
from nova.core.domain.primitives import TaskStatus
from nova.main import NovaRuntimeContainer


class TestPhase15MissionControlV2:
    """Verification suite for Phase 15 (`services/backend/src/nova/api/dashboard.py` v2)."""

    def test_mission_control_v2_html_contains_brain_switcher_and_sha256_seal_pills(
        self,
    ) -> None:
        """Verifies Panel 5 and the new GPU Split / SHA-256 Audit Seal status pills in the HTML."""
        html = render_mission_control_html(api_base_url="http://127.0.0.1:8765")

        assert "5. Live Brain Switcher &amp; 6GB VRAM GPU Offload" in html
        assert "Use Qwen 3.5 9B GGUF (30/36 GPU Split)" in html
        assert "Audit Chain: SHA-256 VERIFIED" in html
        assert "/brain/switch" in html

    def test_overview_snapshot_embeds_6gb_gpu_split_and_sha256_audit_chain(
        self,
    ) -> None:
        """Verifies get_dashboard_overview_snapshot returns brain_and_security telemetry."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            container = NovaRuntimeContainer(
                data_dir=tmp_path / "data",
                workspace_dir=tmp_path / "workspace",
            )

            snap = get_dashboard_overview_snapshot(container)
            assert "brain_and_security" in snap
            bs = snap["brain_and_security"]
            assert bs["active_brain_mode"] == "bootstrap"
            assert bs["gpu_offload_plan"]["offload_mode"] == "hybrid_gpu_cpu_split"
            assert bs["gpu_offload_plan"]["recommended_gpu_layers"] == 30
            assert bs["gpu_offload_plan"]["total_layers"] == 36
            assert bs["security_telemetry"]["audit_chain_valid"] is True

    def test_switching_to_gguf_qwen35_updates_dashboard_snapshot_and_seals_audit_log(
        self,
    ) -> None:
        """Switches active brain to GGUF_QWEN35, runs a task, and verifies the dashboard snapshot."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            container = NovaRuntimeContainer(
                data_dir=tmp_path / "data",
                workspace_dir=tmp_path / "workspace",
            )

            # Initialize brain_controller via get_dashboard_overview_snapshot
            get_dashboard_overview_snapshot(container)
            container.brain_controller.switch_brain_mode(BrainEngineMode.GGUF_QWEN35)

            resp = container.api_service.start_task(
                StartTaskRequest(
                    task_id="task_p15_ui",
                    session_id="sess_p15_ui",
                    user_goal="Save Mission Control v2 verification to v2_ok.txt",
                )
            )
            assert resp.status == TaskStatus.COMPLETED

            snap_after = get_dashboard_overview_snapshot(container)
            bs_after = snap_after["brain_and_security"]
            assert bs_after["active_brain_mode"] == "gguf_qwen35"
            assert bs_after["security_telemetry"]["audit_chain_valid"] is True
            assert bs_after["security_telemetry"]["sealed_audit_entries"] == 1
            assert snap_after["recent_audit_logs"][0]["tool_name"] == "filesystem.write_file"