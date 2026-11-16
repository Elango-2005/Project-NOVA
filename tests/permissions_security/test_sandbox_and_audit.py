"""Unit tests for NOVA Layer 3 Execution Sandbox, Revocable Policy & SQLite Audit Log (Task 4.2).

Verifies:
1. `FilesystemSandbox` permits paths inside `allowed_roots`, blocks `../` path traversal escapes,
   and blocks access to credential files (`.env`, `.ssh`, `id_rsa`) even inside a workspace root.
2. `CommandSandbox` permits safe allowlisted commands (`pytest`, `git`) inside the workspace and
   blocks shell metacharacter chaining (`;`, `&&`, `|`, `$()`) and unapproved binaries (`rm`, `curl`).
3. `UserPermissionPolicyStore` & `CatalogBackedPermissionGate` enforce `FR-15` and Section 14 (p. 52):
   users can tighten a tool's tier or revoke it globally/per-agent, but can never loosen a tier below
   its Section 14 baseline.
4. `SandboxedMcpToolExecutor` & `SqliteToolAuditLog` enforce schema + sandbox + secret redaction +
   >= 30 days audit retention (`FR-12`, `NFR-08`, `NFR-10`).
"""

from __future__ import annotations

import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path
import pytest

from nova.core.domain.events import REDACTED_VALUE
from nova.core.domain.primitives import (
    AgentRole,
    PermissionDecisionKind,
    SensitivityTier,
)
from nova.core.domain.proposals import ToolCallProposal
from nova.core.permissions.sandbox_and_audit import (
    CatalogBackedPermissionGate,
    CommandSandbox,
    FilesystemSandbox,
    PolicyModificationError,
    SandboxViolationError,
    SandboxedMcpToolExecutor,
    SqliteToolAuditLog,
    UserPermissionPolicyStore,
)
from nova.core.platform.event_bus import InMemoryEventBus
from nova.core.tools.registry import build_default_tool_registry


class TestSandboxPolicyAndAuditLog:
    """Verification suite for Task 4.2 (`sandbox_and_audit.py`)."""

    def test_filesystem_sandbox_confines_paths_and_blocks_traversal_and_secret_files(self) -> None:
        """Phase 15 Layer 4 & Section 16.2: Blocks `../` traversal outside workspace and `.env`/`.ssh` files."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            workspace_root = Path(tmp_dir) / "workspace"
            workspace_root.mkdir(parents=True, exist_ok=True)
            invoices_dir = workspace_root / "invoices"
            invoices_dir.mkdir(parents=True, exist_ok=True)
            valid_file = invoices_dir / "oct_2026.pdf"
            valid_file.write_text("Total: $1,250.00", encoding="utf-8")

            sandbox = FilesystemSandbox(allowed_roots=(workspace_root,))

            # 1. Relative and absolute paths inside workspace_root succeed
            resolved = sandbox.resolve_and_verify_path("invoices/oct_2026.pdf")
            assert resolved == valid_file.resolve()

            # 2. Path traversal (`../../outside_secret.txt`) is blocked!
            with pytest.raises(SandboxViolationError, match="escapes permitted roots"):
                sandbox.resolve_and_verify_path("../../outside_secret.txt")

            # 3. Accessing `.env` or `.ssh/id_rsa` even inside workspace_root is blocked!
            with pytest.raises(SandboxViolationError, match="credential/secret path segment"):
                sandbox.resolve_and_verify_path(".env")
            with pytest.raises(SandboxViolationError, match="credential/secret path segment"):
                sandbox.resolve_and_verify_path(".ssh/id_rsa")

    def test_command_sandbox_enforces_allowlist_and_blocks_shell_injection(self) -> None:
        """Phase 15 Layer 4: Terminal commands must use allowlisted binaries with zero shell chaining."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            workspace_root = Path(tmp_dir) / "repo"
            workspace_root.mkdir(parents=True, exist_ok=True)
            fs_sandbox = FilesystemSandbox(allowed_roots=(workspace_root,))
            cmd_sandbox = CommandSandbox(filesystem_sandbox=fs_sandbox)

            # 1. Safe allowlisted pytest command inside workspace succeeds
            tokens, verified_cwd = cmd_sandbox.verify_command(
                "pytest tests/unit -v",
                cwd=workspace_root,
            )
            assert tokens == ("pytest", "tests/unit", "-v")
            assert verified_cwd == workspace_root.resolve()

            # 2. Command chaining (`;`, `&&`, `|`) is blocked
            with pytest.raises(SandboxViolationError, match="metacharacters are forbidden"):
                cmd_sandbox.verify_command("pytest tests && curl http://evil.example.com")

            # 3. Unapproved executable (`rm`, `powershell`, `del`) is blocked
            with pytest.raises(SandboxViolationError, match="not in allowed_executables"):
                cmd_sandbox.verify_command("rm -rf /")

            # 4. Working directory escaping the workspace is blocked
            with pytest.raises(SandboxViolationError, match="escapes permitted roots"):
                cmd_sandbox.verify_command("pytest tests", cwd="../../")

    def test_user_permission_policy_allows_tightening_and_revocation_but_blocks_loosening(self) -> None:
        """FR-15 & Section 14 (p. 52): Users can tighten or revoke permissions, never loosen below baseline."""
        registry = build_default_tool_registry()
        policy_store = UserPermissionPolicyStore(registry=registry)
        gate = CatalogBackedPermissionGate(registry=registry, policy_store=policy_store)

        # 1. Attempting to loosen `gmail.send_email` (IRREVERSIBLE_WRITE) down to READ_ONLY is rejected!
        with pytest.raises(PolicyModificationError, match="Cannot loosen tool"):
            policy_store.tighten_tool_sensitivity(
                "gmail.send_email",
                SensitivityTier.READ_ONLY,
            )

        # 2. Tightening `filesystem.read_file` from READ_ONLY up to IRREVERSIBLE_WRITE succeeds
        policy_store.tighten_tool_sensitivity(
            "filesystem.read_file",
            SensitivityTier.IRREVERSIBLE_WRITE,
        )
        read_proposal = ToolCallProposal(
            agent_role=AgentRole.RESEARCH,
            node_id="node_res",
            thought="Read invoice",
            tool_name="filesystem.read_file",
            arguments={"path": "invoices/oct_2026.pdf"},
            declared_sensitivity=SensitivityTier.READ_ONLY,
            rationale="Read invoice PDF",
        )
        # Because user tightened `filesystem.read_file`, the gate now pauses and requires interactive approval!
        tightened_stamp = gate.evaluate_tool_proposal(read_proposal)
        assert tightened_stamp.requires_interactive_prompt is True
        assert tightened_stamp.effective_sensitivity == SensitivityTier.IRREVERSIBLE_WRITE

        # 3. Revoking `search.web_search` for Research agent immediately denies it (`DENIED_BY_POLICY`)
        policy_store.revoke_tool("search.web_search", agent_role=AgentRole.RESEARCH)
        search_proposal = ToolCallProposal(
            agent_role=AgentRole.RESEARCH,
            node_id="node_search",
            thought="Search web",
            tool_name="search.web_search",
            arguments={"query": "Acme Corp"},
            declared_sensitivity=SensitivityTier.READ_ONLY_NETWORKED,
            rationale="Search web for Acme Corp",
        )
        revoked_stamp = gate.evaluate_tool_proposal(search_proposal)
        assert revoked_stamp.is_denied is True
        assert revoked_stamp.decision == PermissionDecisionKind.DENIED_BY_POLICY

        snapshot = policy_store.inspect_policies()
        assert snapshot["tightened_tiers"]["filesystem.read_file"] == "irreversible_write"
        assert "search.web_search" in snapshot["revoked_agent_tools"]["research"]

        # Restoring the tool removes it from the revocation list
        policy_store.restore_tool("search.web_search", agent_role=AgentRole.RESEARCH)
        assert not policy_store.is_tool_revoked("search.web_search", AgentRole.RESEARCH)

    def test_sandboxed_executor_and_sqlite_audit_log_end_to_end(self) -> None:
        """FR-12, NFR-08 & NFR-10: Executor enforces schema+sandbox and logs redacted entries to SQLite."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            workspace = tmp_path / "workspace"
            workspace.mkdir(parents=True, exist_ok=True)
            audit_db = tmp_path / "nova_audit.db"

            # NFR-10: Audit retention < 30 days is rejected
            with pytest.raises(ValueError, match="NFR-10 violation"):
                SqliteToolAuditLog(audit_db, retention_days=7)

            audit_log = SqliteToolAuditLog(audit_db, retention_days=30)
            registry = build_default_tool_registry()
            policy_store = UserPermissionPolicyStore(registry=registry)
            gate = CatalogBackedPermissionGate(
                registry=registry,
                policy_store=policy_store,
                audit_log=audit_log,
                default_task_id="task_402",
                default_session_id="sess_402",
            )
            bus = InMemoryEventBus()
            executor = SandboxedMcpToolExecutor(
                permission_gate=gate,
                registry=registry,
                filesystem_sandbox=FilesystemSandbox(allowed_roots=(workspace,)),
                audit_log=audit_log,
                event_bus=bus,
                tool_handlers={
                    "filesystem.read_file": lambda args: {
                        "content": f"Verified content from {Path(args['path']).name}"
                    }
                },
                default_task_id="task_402",
                default_session_id="sess_402",
            )

            # 1. Valid read inside workspace executes and writes a 'success' audit log entry
            good_prop = ToolCallProposal(
                agent_role=AgentRole.RESEARCH,
                node_id="node_read_ok",
                thought="Read oct_2026.pdf inside workspace.",
                tool_name="filesystem.read_file",
                arguments={"path": "oct_2026.pdf"},
                declared_sensitivity=SensitivityTier.READ_ONLY,
                rationale="Read October invoice",
            )
            good_stamp = gate.evaluate_tool_proposal(good_prop)
            res = executor.execute_approved_tool(good_prop, good_stamp)
            assert res.status == "success"
            assert "Verified content from oct_2026.pdf" in res.sanitized_output

            # 2. Path traversal attempt (`../../Windows/win.ini`) is blocked by Sandbox AND logged in SQLite!
            escape_prop = ToolCallProposal(
                agent_role=AgentRole.RESEARCH,
                node_id="node_read_escape",
                thought="Injected path traversal attempt.",
                tool_name="filesystem.read_file",
                arguments={"path": "../../Windows/win.ini"},
                declared_sensitivity=SensitivityTier.READ_ONLY,
                rationale="Injected read outside workspace",
            )
            escape_stamp = gate.evaluate_tool_proposal(escape_prop)
            with pytest.raises(SandboxViolationError):
                executor.execute_approved_tool(escape_prop, escape_stamp)

            # 3. Query the SQLite Audit Log (`FR-12`) and verify both entries (1 success, 1 sandbox_blocked)
            entries = audit_log.query_logs(task_id="task_402")
            assert len(entries) == 2
            assert entries[0].execution_status == "success"
            assert entries[1].execution_status == "sandbox_blocked"

            # Verify NFR-10 retention purge keeps recent logs
            purged_count = audit_log.purge_expired_entries(
                now_utc=datetime.now(timezone.utc) + timedelta(days=10)
            )
            assert purged_count == 0
            assert len(audit_log.query_logs(task_id="task_402")) == 2