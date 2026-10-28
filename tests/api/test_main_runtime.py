"""Unit and integration tests for NOVA Phase 8: Live Runnable Runtime Container & Real Local Tools (`test_main_runtime.py`).

Verifies:
1. `NovaRuntimeContainer` boots all 7 layers with zero external configuration and executes a real
   local file creation task (`filesystem.write_file`) inside `workspace_dir` using the built-in
   `RuleGuidedLocalBootstrapAdapter`.
2. `NovaRuntimeContainer` ingests a document into the live Hybrid RAG engine (`ingest_rag_document`),
   runs a multi-step `"Read invoice.txt and email it to sam@accounting.example.com"` task, pauses at
   `AWAITING_APPROVAL` on `gmail.send_email`, and completes when `submit_approval` is called!
"""

from __future__ import annotations

import tempfile
from pathlib import Path

from nova.api.app import StartTaskRequest, SubmitApprovalRequest
from nova.core.domain.primitives import TaskStatus
from nova.main import NovaRuntimeContainer, RagIngestRequest


class TestNovaMainRuntimeContainer:
    """Verification suite for Phase 8 (`services/backend/src/nova/main.py`)."""

    def test_zero_config_runtime_container_writes_real_file_in_sandboxed_workspace(
        self,
    ) -> None:
        """Boots NovaRuntimeContainer and writes a real file (`notes.txt`) inside workspace_dir."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            container = NovaRuntimeContainer(
                data_dir=tmp_path / "data",
                workspace_dir=tmp_path / "workspace",
            )

            health = container.api_service.get_health()
            assert health.offline_by_default is True
            assert health.registered_tools >= 30

            # Send a natural-language file-writing goal to the live bootstrap brain!
            resp = container.api_service.start_task(
                StartTaskRequest(
                    task_id="task_live_write",
                    session_id="sess_live_01",
                    user_goal="Save project kickoff summary to notes.txt",
                )
            )
            assert resp.status == TaskStatus.COMPLETED
            created_file = container.workspace_dir / "notes.txt"
            assert created_file.exists()
            assert "Save project kickoff summary to notes.txt" in created_file.read_text(
                encoding="utf-8"
            )

    def test_zero_config_runtime_container_rag_ingest_read_and_email_approval_flow(
        self,
    ) -> None:
        """Ingests a RAG doc, reads a workspace file, pauses on email to Sam, and finishes on Approve."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            container = NovaRuntimeContainer(
                data_dir=tmp_path / "data",
                workspace_dir=tmp_path / "workspace",
            )

            # 1. Ingest a document into the live Hybrid RAG engine
            ingest_res = container.ingest_rag_document(
                RagIngestRequest(
                    document_id="doc_oct_budget",
                    source_uri="file:///workspace/budget.md",
                    content="October AI server budget approved at $1,250.00.",
                )
            )
            assert ingest_res["chunks_indexed"] == 1

            # 2. Create a local file `invoice.txt` in the workspace
            (container.workspace_dir / "invoice.txt").write_text(
                "Vendor: Acme Corp | Total: $1,250.00",
                encoding="utf-8",
            )

            # 3. Start a 2-step goal ("Read invoice.txt and email it to sam@accounting.example.com")
            paused_resp = container.api_service.start_task(
                StartTaskRequest(
                    task_id="task_live_email",
                    session_id="sess_live_02",
                    user_goal="Read invoice.txt and email it to sam@accounting.example.com",
                )
            )
            assert paused_resp.status == TaskStatus.AWAITING_APPROVAL
            assert "step_research_info" in paused_resp.completed_nodes
            assert len(paused_resp.pending_approvals) == 1

            card = paused_resp.pending_approvals[0]
            assert card.tool_name == "gmail.send_email"
            assert card.arguments["to"] == "sam@accounting.example.com"

            # 4. Approve the paused email action -> task completes and logs all steps!
            final_resp = container.api_service.submit_approval(
                "task_live_email",
                SubmitApprovalRequest(
                    node_id=card.node_id,
                    proposal_id=card.proposal_id,
                    approved=True,
                    reason="Approved live email in Phase 8 test",
                ),
            )
            assert final_resp.status == TaskStatus.COMPLETED
            logs = container.api_service.query_audit_logs()
            assert [entry.tool_name for entry in logs] == [
                "filesystem.read_file",
                "gmail.send_email",
            ]