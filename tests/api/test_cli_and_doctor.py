"""Unit and integration tests for NOVA Phase 10: Interactive Terminal CLI & 7-Layer System Doctor (`test_cli_and_doctor.py`).

Verifies:
1. `run_system_doctor(container)` probes all 7 layers of NOVA and returns `all_healthy=True` with
   7 passing `LayerHealthCheck` records.
2. `run_cli_args(["ingest", ...])`, `run_cli_args(["ask", ...])`, and `run_cli_args(["audit"])`
   index a local file into Hybrid RAG, execute a file-creation task in the workspace, and inspect
   the SQLite audit trail via the CLI.
3. `run_cli_args(["ask", ...], interactive_approval_fn=...)` interactively approves a paused
   `gmail.send_email` action (`AWAITING_APPROVAL` -> `COMPLETED`).
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any, Dict, List

from nova.cli import run_cli_args, run_system_doctor
from nova.main import NovaRuntimeContainer


class TestNovaCliAndSystemDoctor:
    """Verification suite for Phase 10 (`services/backend/src/nova/cli.py`)."""

    def test_system_doctor_verifies_all_seven_architectural_layers(self) -> None:
        """Runs the 7-layer NOVA System Doctor and confirms every layer reports healthy."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            container = NovaRuntimeContainer(
                data_dir=tmp_path / "data",
                workspace_dir=tmp_path / "workspace",
            )

            report = run_system_doctor(container)
            assert report.app_name == "NOVA"
            assert report.all_healthy is True
            assert report.offline_by_default is True
            assert len(report.checks) == 7
            assert [c.layer_number for c in report.checks] == [1, 2, 3, 4, 5, 6, 7]
            assert all(c.healthy for c in report.checks)

            cli_doc = run_cli_args(["doctor"], container=container)
            assert cli_doc["command"] == "doctor"
            assert cli_doc["all_healthy"] is True
            assert len(cli_doc["checks"]) == 7

    def test_cli_ingest_ask_and_audit_commands_end_to_end(self) -> None:
        """Tests `nova ingest`, `nova ask`, and `nova audit` commands against a live container."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            container = NovaRuntimeContainer(
                data_dir=tmp_path / "data",
                workspace_dir=tmp_path / "workspace",
            )

            # 1. Create a local markdown file and ingest it via `nova ingest`
            notes_md = tmp_path / "architecture_notes.md"
            notes_md.write_text(
                "NOVA enforces a 7-layer local-first security boundary.",
                encoding="utf-8",
            )
            ingest_out = run_cli_args(
                ["ingest", str(notes_md), "--doc-id", "doc_arch_notes"],
                container=container,
            )
            assert ingest_out["command"] == "ingest"
            assert ingest_out["document_id"] == "doc_arch_notes"
            assert ingest_out["chunks_indexed"] == 1

            # 2. Run a file-writing goal via `nova ask`
            ask_out = run_cli_args(
                [
                    "ask",
                    "Write deployment checklist to checklist.txt",
                    "--task-id",
                    "task_cli_write",
                ],
                container=container,
            )
            assert ask_out["status"] == "completed"
            assert (container.workspace_dir / "checklist.txt").exists()

            # 3. Inspect the SQLite audit trail via `nova audit`
            audit_out = run_cli_args(["audit", "--limit", "5"], container=container)
            assert audit_out["command"] == "audit"
            assert audit_out["count"] == 1
            assert audit_out["entries"][0]["tool_name"] == "filesystem.write_file"

    def test_cli_ask_interactive_approval_callback_resolves_email_gate(self) -> None:
        """Verifies interactive CLI approval resolves a paused `gmail.send_email` action."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            container = NovaRuntimeContainer(
                data_dir=tmp_path / "data",
                workspace_dir=tmp_path / "workspace",
            )
            (container.workspace_dir / "status.txt").write_text(
                "Phase 10 CLI ready.",
                encoding="utf-8",
            )

            seen_prompts: List[Dict[str, Any]] = []

            def _approve_callback(card: Dict[str, Any]) -> bool:
                seen_prompts.append(card)
                return True

            res = run_cli_args(
                [
                    "ask",
                    "Read status.txt and email it to ops@nova.local",
                    "--task-id",
                    "task_cli_email",
                ],
                container=container,
                interactive_approval_fn=_approve_callback,
            )
            assert res["status"] == "completed"
            assert len(seen_prompts) == 1
            assert seen_prompts[0]["tool_name"] == "gmail.send_email"
            assert seen_prompts[0]["arguments"]["to"] == "ops@nova.local"