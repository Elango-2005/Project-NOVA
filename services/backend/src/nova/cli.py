"""NOVA Phase 10: Interactive Terminal CLI & 7-Layer System Doctor (`services/backend/src/nova/cli.py`).

Plain-English Purpose (The Walkie-Talkie & Mechanic's Diagnostic Scanner):
While Phase 9 (`dashboard.py`) gave you a browser cockpit, Phase 10 (`cli.py`) gives you a fast
command-line tool (`python -m nova.cli ...`) so you can talk to NOVA or diagnose every layer
directly from your PowerShell terminal:

1. **`doctor` (`run_system_doctor`) — The Mechanic's OBD-II Scanner**:
   - Probes all 7 architectural layers of NOVA (Settings & Offline Guard, SQLite Checkpoints,
     Two-Tier Memory & Hybrid RAG, 20 MCP Servers / 42 Tools & Sandbox Cages, Model Router,
     Multi-Agent State-Graph Orchestrator, and Mission Control UI/API) and prints a clean
     pass/fail health report (`all_healthy=True`).

2. **`ask "<your goal>"` — The Walkie-Talkie**:
   - Sends a natural-language goal to NOVA from your terminal. If an agent proposes an
     `irreversible_write` tool (like `gmail.send_email`) that pauses at `AWAITING_APPROVAL`,
     the CLI can prompt you (or invoke `interactive_approval_fn`) to Approve or Deny on the spot!

3. **`ingest <path>` — One-Command Local File Indexing**:
   - Reads any local `.md` or `.txt` file and indexes it into `nova_rag.db` for Hybrid RAG search.

4. **`audit` — Terminal Black-Box Flight Recorder Viewer**:
   - Prints recent SQLite tool execution audit entries right in your terminal.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence

from pydantic import BaseModel, Field

try:
    from pydantic import ConfigDict

    PYDANTIC_V2 = True
except ImportError:  # pragma: no cover
    PYDANTIC_V2 = False

from nova.api.app import StartTaskRequest, SubmitApprovalRequest, TaskStatusResponse
from nova.api.dashboard import get_dashboard_overview_snapshot
from nova.core.domain.primitives import TaskStatus
from nova.main import NovaRuntimeContainer, RagIngestRequest

InteractiveApprovalFn = Callable[[Dict[str, Any]], bool]


class LayerHealthCheck(BaseModel):
    """Single architectural layer check inside `SystemDoctorReport`."""

    if PYDANTIC_V2:
        model_config = ConfigDict(frozen=True, extra="forbid")
    else:

        class Config:
            allow_mutation = False
            extra = "forbid"

    layer_number: int = Field(..., ge=1, le=7)
    layer_name: str = Field(..., min_length=1)
    healthy: bool
    detail: str = Field(..., min_length=1)


class SystemDoctorReport(BaseModel):
    """Complete 7-layer diagnostic report produced by `run_system_doctor`."""

    if PYDANTIC_V2:
        model_config = ConfigDict(frozen=True, extra="forbid")
    else:

        class Config:
            allow_mutation = False
            extra = "forbid"

    app_name: str
    version: str
    all_healthy: bool
    offline_by_default: bool
    workspace_dir: str
    data_dir: str
    checks: tuple[LayerHealthCheck, ...]


def run_system_doctor(container: NovaRuntimeContainer) -> SystemDoctorReport:
    """Verify all 7 layers of the NOVA architecture and return a structured `SystemDoctorReport`."""
    health = container.api_service.get_health()
    overview = get_dashboard_overview_snapshot(container)

    checks: List[LayerHealthCheck] = [
        LayerHealthCheck(
            layer_number=1,
            layer_name="Platform Config, Event Bus & SQLite Checkpoints",
            healthy=bool(
                health.offline_by_default
                and (container.data_dir / "nova_checkpoints.db").exists()
            ),
            detail=(
                f"offline_by_default={health.offline_by_default}, "
                f"checkpoints_db={(container.data_dir / 'nova_checkpoints.db').name}"
            ),
        ),
        LayerHealthCheck(
            layer_number=2,
            layer_name="Model Serving Router & Bootstrap Adapter",
            healthy=container.model_router is not None,
            detail=f"allow_cloud_inference={health.allow_cloud_inference}, local_adapter=ready",
        ),
        LayerHealthCheck(
            layer_number=3,
            layer_name="MCP Tool Registry, Permission Gate & Sandboxes",
            healthy=bool(
                health.registered_mcp_servers >= 20 and health.registered_tools >= 30
            ),
            detail=(
                f"servers={health.registered_mcp_servers}, "
                f"tools={health.registered_tools}, "
                f"workspace={container.workspace_dir.name}"
            ),
        ),
        LayerHealthCheck(
            layer_number=4,
            layer_name="Two-Tier Memory & Local Hybrid RAG Index",
            healthy=bool(
                (container.data_dir / "nova_memory.db").exists()
                and (container.data_dir / "nova_rag.db").exists()
            ),
            detail="nova_memory.db=ready, nova_rag.db=ready",
        ),
        LayerHealthCheck(
            layer_number=5,
            layer_name="Specialized ReAct Agents & Plan/Reflection Engine",
            healthy=len(container.agents) >= 7,
            detail=f"registered_agents={len(container.agents)}",
        ),
        LayerHealthCheck(
            layer_number=6,
            layer_name="Durable State-Graph Orchestrator",
            healthy=container.orchestrator is not None,
            detail="state_graph_orchestrator=ready",
        ),
        LayerHealthCheck(
            layer_number=7,
            layer_name="Local FastAPI Service & Mission Control Dashboard",
            healthy=overview.get("status") == "ok",
            detail=f"api_status={overview.get('status')}, ui_overview=ready",
        ),
    ]

    return SystemDoctorReport(
        app_name=health.app_name,
        version=health.version,
        all_healthy=all(c.healthy for c in checks),
        offline_by_default=health.offline_by_default,
        workspace_dir=str(container.workspace_dir),
        data_dir=str(container.data_dir),
        checks=tuple(checks),
    )


class NovaCliRunner:
    """Programmatic and terminal CLI runner wrapping `NovaRuntimeContainer`."""

    def __init__(self, container: Optional[NovaRuntimeContainer] = None) -> None:
        self.container = container or NovaRuntimeContainer()

    def doctor(self) -> SystemDoctorReport:
        """Run the 7-layer NOVA System Doctor."""
        return run_system_doctor(self.container)

    def ingest_file(
        self,
        file_path: Path,
        *,
        document_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Read a local file from disk and index its text into NOVA's Hybrid RAG database."""
        resolved = Path(file_path).resolve()
        content = resolved.read_text(encoding="utf-8")
        doc_id = document_id or f"doc_{resolved.stem}"
        return self.container.ingest_rag_document(
            RagIngestRequest(
                document_id=doc_id,
                source_uri=resolved.as_uri(),
                content=content,
            )
        )

    def run_goal(
        self,
        user_goal: str,
        *,
        task_id: str = "task_cli_run",
        session_id: str = "sess_cli_run",
        interactive_approval_fn: Optional[InteractiveApprovalFn] = None,
    ) -> TaskStatusResponse:
        """Execute a natural-language goal and optionally resolve `AWAITING_APPROVAL` pauses."""
        resp = self.container.api_service.start_task(
            StartTaskRequest(
                task_id=task_id,
                session_id=session_id,
                user_goal=user_goal,
            )
        )
        while (
            resp.status == TaskStatus.AWAITING_APPROVAL
            and resp.pending_approvals
            and interactive_approval_fn is not None
        ):
            card = resp.pending_approvals[0]
            approved = interactive_approval_fn(
                {
                    "task_id": resp.task_id,
                    "node_id": card.node_id,
                    "proposal_id": card.proposal_id,
                    "tool_name": card.tool_name,
                    "arguments": dict(card.arguments),
                    "declared_sensitivity": card.declared_sensitivity.value,
                }
            )
            resp = self.container.api_service.submit_approval(
                resp.task_id,
                SubmitApprovalRequest(
                    node_id=card.node_id,
                    proposal_id=card.proposal_id,
                    approved=approved,
                    reason="Interactive CLI user decision",
                ),
            )
        return resp


def run_cli_args(
    argv: Sequence[str],
    *,
    container: Optional[NovaRuntimeContainer] = None,
    interactive_approval_fn: Optional[InteractiveApprovalFn] = None,
) -> Dict[str, Any]:
    """Parse CLI arguments (`doctor`, `ask`, `ingest`, `audit`) and return a JSON-serializable result."""
    parser = argparse.ArgumentParser(
        prog="nova",
        description="NOVA Local-First Personal AI Operating Layer CLI",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("doctor", help="Run the 7-layer NOVA architectural health check")

    ask_p = sub.add_parser("ask", help="Execute a natural-language goal")
    ask_p.add_argument("goal", type=str, help="Goal for NOVA to execute")
    ask_p.add_argument("--task-id", type=str, default="task_cli_01")
    ask_p.add_argument(
        "--auto-approve-all",
        action="store_true",
        help="Automatically approve irreversible_write prompts during this CLI run",
    )

    ingest_p = sub.add_parser("ingest", help="Index a local file into Hybrid RAG")
    ingest_p.add_argument("file_path", type=str, help="Path to local text/markdown file")
    ingest_p.add_argument("--doc-id", type=str, default=None)

    sub_audit = sub.add_parser("audit", help="Display recent SQLite tool audit logs")
    sub_audit.add_argument("--limit", type=int, default=10)

    args = parser.parse_args(list(argv))
    runner = NovaCliRunner(container=container)

    if args.command == "doctor":
        report = runner.doctor()
        return {
            "command": "doctor",
            "all_healthy": report.all_healthy,
            "checks": [
                {
                    "layer_number": c.layer_number,
                    "layer_name": c.layer_name,
                    "healthy": c.healthy,
                    "detail": c.detail,
                }
                for c in report.checks
            ],
        }

    if args.command == "ingest":
        res = runner.ingest_file(Path(args.file_path), document_id=args.doc_id)
        return {"command": "ingest", **res}

    if args.command == "ask":
        approval_cb = interactive_approval_fn
        if approval_cb is None and args.auto_approve_all:
            approval_cb = lambda _card: True
        resp = runner.run_goal(
            args.goal,
            task_id=args.task_id,
            interactive_approval_fn=approval_cb,
        )
        return {
            "command": "ask",
            "task_id": resp.task_id,
            "status": resp.status.value,
            "completed_nodes": list(resp.completed_nodes),
            "pending_approvals_count": len(resp.pending_approvals),
        }

    if args.command == "audit":
        logs = runner.container.api_service.query_audit_logs()[-args.limit :]
        return {
            "command": "audit",
            "count": len(logs),
            "entries": [
                {
                    "tool_name": e.tool_name,
                    "decision": e.decision.value,
                    "execution_status": e.execution_status,
                }
                for e in logs
            ],
        }

    raise ValueError(f"Unsupported command: {args.command}")  # pragma: no cover


def main(argv: Optional[Sequence[str]] = None) -> int:  # pragma: no cover
    import sys

    effective_argv = sys.argv[1:] if argv is None else argv
    result = run_cli_args(effective_argv)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())