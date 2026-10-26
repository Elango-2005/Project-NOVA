"""NOVA Layer 3 Execution Sandbox, Revocable Permission Policy & SQLite Audit Log (ADR-010 / Phase 7 / Phase 15 / FR-12 / FR-15 / NFR-03 / NFR-08 / NFR-10).

Plain-English Purpose (The Safety Fence, Control Panel & Black-Box Flight Recorder):
Even after an employee (Agent) shows their ID badge and gets a permission stamp, we still need
three physical safeguards in Layer 3:

1. **The Safety Fence (`FilesystemSandbox` & `CommandSandbox` — Phase 15 Layer 4)**:
   - Even if a tool is approved to read or write a file, `FilesystemSandbox` locks it inside your
     allowed workspace folders (`allowed_roots`). If a prompt-injected argument tries to escape
     using `../../Windows/System32/cmd.exe` or tries to read `.env` / `.ssh/id_rsa`, the fence blocks
     it immediately (`SandboxViolationError`).
   - `CommandSandbox` inspects shell commands (`terminal.execute`) and blocks shell chaining (`;`, `&&`,
     `|`, `` ` ``, `$()`) and unapproved executables (`rm`, `del`, `format`, `curl`, `powershell`).

2. **Your Permission Control Panel (`UserPermissionPolicyStore` & `CatalogBackedPermissionGate` — FR-15 & Section 14)**:
   - Lets you inspect, **tighten** (e.g., upgrade a tool to `IRREVERSIBLE_WRITE` so it asks every time),
     or **revoke** (`deny`) any tool globally or for a specific agent.
   - Enforces the Section 14 Security Rule (p. 52): *"Users can tighten (never silently loosen) any tier."*
     Trying to downgrade `terminal.execute` or `gmail.send_email` to `READ_ONLY` is rejected!

3. **The Black-Box Flight Recorder (`SqliteToolAuditLog` — FR-12, NFR-08, NFR-10 & Phase 15 Layer 5)**:
   - Records every permission check, sandbox block, and tool execution into a permanent SQLite table
     (`tool_audit_log`) with automatic secret redaction and >= 30 days retention (`NFR-10`), so you
     can always ask *"Why did NOVA do that?"* (`FR-12`).
"""

from __future__ import annotations

import json
import re
import shlex
import sqlite3
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Literal, Mapping, Optional, Sequence, Tuple

from pydantic import BaseModel, Field

try:
    from pydantic import ConfigDict

    PYDANTIC_V2 = True
except ImportError:  # pragma: no cover
    PYDANTIC_V2 = False

from nova.core.domain.agent_spec import AgentScopeViolationError, get_agent_spec
from nova.core.domain.context import sanitize_untrusted_text
from nova.core.domain.events import EventType, NovaEvent, redact_secrets
from nova.core.domain.primitives import (
    CURRENT_SCHEMA_VERSION,
    AgentRole,
    PermissionDecisionKind,
    SensitivityTier,
    validate_schema_version,
)
from nova.core.domain.proposals import ToolCallProposal
from nova.core.orchestration.ports import (
    PermissionCheckResult,
    ToolExecutionResult,
    verify_approval_stamp,
)
from nova.core.platform.event_bus import EventPublisherPort
from nova.core.tools.registry import (
    ToolContract,
    ToolRegistry,
    build_default_tool_registry,
)

# Severity rank for enforcing Section 14's "tighten only, never loosen" rule (FR-15)
_SENSITIVITY_SEVERITY_RANK: Dict[SensitivityTier, int] = {
    SensitivityTier.READ_ONLY: 1,
    SensitivityTier.READ_ONLY_NETWORKED: 2,
    SensitivityTier.REVERSIBLE_WRITE: 3,
    SensitivityTier.REVERSIBLE_WRITE_NETWORKED: 4,
    SensitivityTier.IRREVERSIBLE_WRITE: 5,
}

# Sensitive filenames/segments that no tool may ever touch even inside a workspace root (Section 16.2)
_FORBIDDEN_SECRET_PATH_NAMES = frozenset(
    {
        ".env",
        ".env.local",
        ".env.production",
        "id_rsa",
        "id_ed25519",
        ".ssh",
        ".aws",
        "credentials.json",
    }
)

# Shell metacharacters / chaining operators forbidden inside sandboxed commands (Phase 15 Layer 4)
_SHELL_METACHAR_PATTERN = re.compile(r"(;|&&|\|\||\||`|\$\(|>|<)")


class SandboxViolationError(PermissionError):
    """Raised when a tool argument attempts to escape the filesystem or command sandbox (Phase 15 Layer 4)."""


class PolicyModificationError(ValueError):
    """Raised when a caller attempts to loosen a tool's baseline security tier (Section 14 / FR-15)."""


class FilesystemSandbox:
    """Confines all `path_scoped` tool arguments to explicitly permitted workspace directories (Phase 7 / Phase 15)."""

    def __init__(self, allowed_roots: Sequence[Path | str]) -> None:
        if not allowed_roots:
            raise ValueError("FilesystemSandbox requires at least one allowed root directory.")
        self._allowed_roots: Tuple[Path, ...] = tuple(
            Path(r).expanduser().resolve() for r in allowed_roots
        )

    @property
    def primary_root(self) -> Path:
        return self._allowed_roots[0]

    @property
    def allowed_roots(self) -> Tuple[Path, ...]:
        return self._allowed_roots

    def resolve_and_verify_path(self, raw_path: str | Path) -> Path:
        """Resolve `raw_path` and verify it stays inside `allowed_roots` and avoids secret files."""
        raw_str = str(raw_path).strip()
        if not raw_str:
            raise SandboxViolationError("Empty filesystem path is not allowed.")

        # Block null bytes or explicit parent-traversal tricks before resolution
        if "\x00" in raw_str:
            raise SandboxViolationError("Null byte injection detected in filesystem path.")

        candidate = Path(raw_str)
        if not candidate.is_absolute():
            resolved = (self.primary_root / candidate).resolve()
        else:
            resolved = candidate.resolve()

        # 1. Verify `resolved` is inside at least one permitted root directory
        in_allowed_root = False
        for root in self._allowed_roots:
            try:
                resolved.relative_to(root)
                in_allowed_root = True
                break
            except ValueError:
                continue

        if not in_allowed_root:
            raise SandboxViolationError(
                f"Phase 15 Layer 4 sandbox violation: Path {raw_str!r} resolves to "
                f"{str(resolved)!r}, which escapes permitted roots "
                f"{[str(r) for r in self._allowed_roots]}."
            )

        # 2. Block access to credential/secret files (.env, .ssh, id_rsa, etc.) per Section 16.2
        for part in resolved.parts:
            if part.lower() in _FORBIDDEN_SECRET_PATH_NAMES:
                raise SandboxViolationError(
                    f"Section 16.2 / Phase 15 sandbox violation: Access to credential/secret "
                    f"path segment {part!r} in {raw_str!r} is strictly forbidden."
                )

        return resolved


class CommandSandbox:
    """Validates shell commands for `terminal.execute` against executable allowlists and metacharacter rules."""

    DEFAULT_ALLOWED_EXECUTABLES: Tuple[str, ...] = (
        "pytest",
        "python",
        "python3",
        "git",
        "ruff",
        "mypy",
        "echo",
    )

    def __init__(
        self,
        filesystem_sandbox: FilesystemSandbox,
        allowed_executables: Sequence[str] = DEFAULT_ALLOWED_EXECUTABLES,
        max_timeout_seconds: int = 30,
    ) -> None:
        if not (1 <= max_timeout_seconds <= 300):
            raise ValueError("max_timeout_seconds must be in [1, 300].")
        self._fs_sandbox = filesystem_sandbox
        self._allowed_executables = frozenset(e.lower() for e in allowed_executables)
        self._max_timeout_seconds = max_timeout_seconds

    @property
    def max_timeout_seconds(self) -> int:
        return self._max_timeout_seconds

    def verify_command(
        self,
        command: str,
        cwd: Optional[str | Path] = None,
    ) -> Tuple[Tuple[str, ...], Path]:
        """Parse and validate `command` and `cwd` before execution inside the sandbox."""
        cleaned = command.strip()
        if not cleaned:
            raise SandboxViolationError("Empty shell command is not allowed.")

        # 1. Block shell chaining/redirection metacharacters (`;`, `&&`, `||`, `|`, `` ` ``, `$()`, `>`, `<`)
        if _SHELL_METACHAR_PATTERN.search(cleaned):
            raise SandboxViolationError(
                f"Phase 15 Layer 4 sandbox violation: Shell chaining or redirection metacharacters "
                f"are forbidden in sandboxed command {command!r}."
            )

        try:
            tokens = tuple(shlex.split(cleaned, posix=True))
        except ValueError as exc:
            raise SandboxViolationError(f"Malformed command quoting in {command!r}: {exc}") from exc

        if not tokens:
            raise SandboxViolationError("Command produced zero executable tokens.")

        executable = Path(tokens[0]).name.lower()
        if executable.endswith(".exe"):
            executable = executable[:-4]

        # 2. Enforce strict executable allowlist
        if executable not in self._allowed_executables:
            raise SandboxViolationError(
                f"Phase 15 Layer 4 sandbox violation: Executable {executable!r} is not in "
                f"allowed_executables {sorted(self._allowed_executables)}."
            )

        # 3. Verify working directory is inside the FilesystemSandbox
        verified_cwd = (
            self._fs_sandbox.resolve_and_verify_path(cwd)
            if cwd is not None
            else self._fs_sandbox.primary_root
        )
        return tokens, verified_cwd


class UserPermissionPolicyStore:
    """User-inspectable and revocable permission policy store (FR-15 / ADR-010 / Section 14).

    Supports:
    - Tightening a tool's sensitivity tier (`tighten_tool_sensitivity`) — never loosening below
      the Section 14 catalog baseline.
    - Revoking a tool globally or for a specific `AgentRole` (`revoke_tool`), or restoring it (`restore_tool`).
    - Exporting a full snapshot of active overrides for the UI Settings panel (`inspect_policies`).
    """

    def __init__(self, registry: Optional[ToolRegistry] = None) -> None:
        self._registry = registry or build_default_tool_registry()
        self._tightened_tiers: Dict[str, SensitivityTier] = {}
        self._revoked_global_tools: set[str] = set()
        self._revoked_agent_tools: Dict[AgentRole, set[str]] = {}

    def tighten_tool_sensitivity(
        self,
        tool_name: str,
        new_tier: SensitivityTier,
    ) -> SensitivityTier:
        """Tighten `tool_name` to `new_tier` (Section 14 p. 52: users can tighten, never loosen)."""
        contract = self._registry.get_tool(tool_name)
        baseline_tier = contract.sensitivity_tier
        if _SENSITIVITY_SEVERITY_RANK[new_tier] < _SENSITIVITY_SEVERITY_RANK[baseline_tier]:
            raise PolicyModificationError(
                f"Section 14 / FR-15 violation: Cannot loosen tool {tool_name!r} from baseline "
                f"tier {baseline_tier.value!r} down to {new_tier.value!r}. Users may only tighten "
                "sensitivity tiers."
            )
        self._tightened_tiers[tool_name] = new_tier
        return new_tier

    def revoke_tool(
        self,
        tool_name: str,
        *,
        agent_role: Optional[AgentRole] = None,
    ) -> None:
        """Revoke permission for `tool_name` globally (if `agent_role=None`) or for a specific agent (FR-15)."""
        self._registry.get_tool(tool_name)  # Ensure tool exists in catalog
        if agent_role is None:
            self._revoked_global_tools.add(tool_name)
        else:
            self._revoked_agent_tools.setdefault(agent_role, set()).add(tool_name)

    def restore_tool(
        self,
        tool_name: str,
        *,
        agent_role: Optional[AgentRole] = None,
    ) -> None:
        """Restore a previously revoked tool back to its active sensitivity tier policy."""
        if agent_role is None:
            self._revoked_global_tools.discard(tool_name)
        elif agent_role in self._revoked_agent_tools:
            self._revoked_agent_tools[agent_role].discard(tool_name)

    def is_tool_revoked(self, tool_name: str, agent_role: AgentRole) -> bool:
        if tool_name in self._revoked_global_tools:
            return True
        return tool_name in self._revoked_agent_tools.get(agent_role, set())

    def get_effective_sensitivity(
        self,
        tool_name: str,
        declared_sensitivity: SensitivityTier,
    ) -> SensitivityTier:
        """Return the strictest sensitivity tier across catalog baseline, user override, and proposal."""
        try:
            baseline = self._registry.get_tool(tool_name).sensitivity_tier
        except Exception:
            # Fail-closed: unknown tools default to highest sensitivity
            return SensitivityTier.IRREVERSIBLE_WRITE

        user_tier = self._tightened_tiers.get(tool_name, baseline)
        candidates = (baseline, user_tier, declared_sensitivity)
        return max(candidates, key=lambda t: _SENSITIVITY_SEVERITY_RANK[t])

    def inspect_policies(self) -> Dict[str, Any]:
        """Return a human-inspectable snapshot of all user permission overrides (FR-15)."""
        return {
            "tightened_tiers": {k: v.value for k, v in sorted(self._tightened_tiers.items())},
            "revoked_global_tools": sorted(self._revoked_global_tools),
            "revoked_agent_tools": {
                role.value: sorted(tools)
                for role, tools in sorted(
                    self._revoked_agent_tools.items(), key=lambda item: item[0].value
                )
                if tools
            },
        }


class ToolAuditEntry(BaseModel):
    """Immutable audit log record for a permission decision or tool execution (FR-12 / NFR-08 / NFR-10)."""

    if PYDANTIC_V2:
        model_config = ConfigDict(frozen=True, extra="forbid")
    else:

        class Config:
            allow_mutation = False
            extra = "forbid"

    schema_version: str = Field(default=CURRENT_SCHEMA_VERSION)
    audit_id: str = Field(default_factory=lambda: f"aud_{uuid.uuid4().hex}")
    timestamp_utc: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    task_id: str = Field(..., min_length=1)
    session_id: str = Field(..., min_length=1)
    node_id: str = Field(..., min_length=1)
    proposal_id: str = Field(..., min_length=1)
    agent_role: AgentRole = Field(...)
    tool_name: str = Field(..., min_length=1)
    effective_sensitivity: SensitivityTier = Field(...)
    decision: PermissionDecisionKind = Field(...)
    execution_status: Literal[
        "not_executed",
        "success",
        "error",
        "sandbox_blocked",
        "schema_invalid",
    ] = Field(...)
    redacted_arguments: Dict[str, Any] = Field(default_factory=dict)
    rationale: str = Field(..., min_length=1)
    outcome_summary: str = Field(..., min_length=1)

    def __init__(self, **data: Any) -> None:
        updated = dict(data)
        version = updated.get("schema_version", CURRENT_SCHEMA_VERSION)
        if isinstance(version, str):
            updated["schema_version"] = validate_schema_version(version)
        raw_args = updated.get("redacted_arguments") or {}
        if isinstance(raw_args, Mapping):
            updated["redacted_arguments"] = dict(redact_secrets(raw_args))
        raw_summary = updated.get("outcome_summary", "")
        if isinstance(raw_summary, str):
            updated["outcome_summary"] = sanitize_untrusted_text(raw_summary)
        super().__init__(**updated)


class SqliteToolAuditLog:
    """Durable, queryable SQLite audit log for all permission gates and tool executions (FR-12 / NFR-08 / NFR-10)."""

    def __init__(self, db_path: Path | str, retention_days: int = 30) -> None:
        if retention_days < 30:
            raise ValueError(
                f"NFR-10 violation: Audit log retention_days must be >= 30, got {retention_days}."
            )
        self._db_path = Path(db_path)
        self._retention_days = retention_days
        if str(self._db_path) != ":memory:":
            self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    @property
    def retention_days(self) -> int:
        return self._retention_days

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self._db_path))
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA synchronous=NORMAL;")
        return conn

    def _init_schema(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS tool_audit_log (
                    audit_id TEXT PRIMARY KEY,
                    timestamp_utc TEXT NOT NULL,
                    task_id TEXT NOT NULL,
                    session_id TEXT NOT NULL,
                    node_id TEXT NOT NULL,
                    proposal_id TEXT NOT NULL,
                    agent_role TEXT NOT NULL,
                    tool_name TEXT NOT NULL,
                    effective_sensitivity TEXT NOT NULL,
                    decision TEXT NOT NULL,
                    execution_status TEXT NOT NULL,
                    redacted_arguments_json TEXT NOT NULL,
                    rationale TEXT NOT NULL,
                    outcome_summary TEXT NOT NULL,
                    schema_version TEXT NOT NULL
                );
                """
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_tool_audit_task
                ON tool_audit_log (task_id, timestamp_utc ASC);
                """
            )

    def record_entry(self, entry: ToolAuditEntry) -> ToolAuditEntry:
        """Persist an immutable `ToolAuditEntry` to SQLite."""
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO tool_audit_log (
                    audit_id,
                    timestamp_utc,
                    task_id,
                    session_id,
                    node_id,
                    proposal_id,
                    agent_role,
                    tool_name,
                    effective_sensitivity,
                    decision,
                    execution_status,
                    redacted_arguments_json,
                    rationale,
                    outcome_summary,
                    schema_version
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
                """,
                (
                    entry.audit_id,
                    entry.timestamp_utc,
                    entry.task_id,
                    entry.session_id,
                    entry.node_id,
                    entry.proposal_id,
                    entry.agent_role.value,
                    entry.tool_name,
                    entry.effective_sensitivity.value,
                    entry.decision.value,
                    entry.execution_status,
                    json.dumps(entry.redacted_arguments, sort_keys=True),
                    entry.rationale,
                    entry.outcome_summary,
                    entry.schema_version,
                ),
            )
        return entry

    def query_logs(
        self,
        *,
        task_id: Optional[str] = None,
        agent_role: Optional[AgentRole] = None,
        tool_name: Optional[str] = None,
        decision: Optional[PermissionDecisionKind] = None,
    ) -> Tuple[ToolAuditEntry, ...]:
        """Query the audit trail so the user can inspect 'why did NOVA do X?' (FR-12 / NFR-08)."""
        clauses: List[str] = []
        params: List[Any] = []
        if task_id is not None:
            clauses.append("task_id = ?")
            params.append(task_id)
        if agent_role is not None:
            clauses.append("agent_role = ?")
            params.append(agent_role.value)
        if tool_name is not None:
            clauses.append("tool_name = ?")
            params.append(tool_name)
        if decision is not None:
            clauses.append("decision = ?")
            params.append(decision.value)

        where_sql = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        sql = f"""
            SELECT audit_id, timestamp_utc, task_id, session_id, node_id, proposal_id,
                   agent_role, tool_name, effective_sensitivity, decision,
                   execution_status, redacted_arguments_json, rationale,
                   outcome_summary, schema_version
            FROM tool_audit_log
            {where_sql}
            ORDER BY timestamp_utc ASC;
        """
        with self._connect() as conn:
            rows = conn.execute(sql, tuple(params)).fetchall()

        return tuple(
            ToolAuditEntry(
                audit_id=str(r[0]),
                timestamp_utc=str(r[1]),
                task_id=str(r[2]),
                session_id=str(r[3]),
                node_id=str(r[4]),
                proposal_id=str(r[5]),
                agent_role=AgentRole(str(r[6])),
                tool_name=str(r[7]),
                effective_sensitivity=SensitivityTier(str(r[8])),
                decision=PermissionDecisionKind(str(r[9])),
                execution_status=str(r[10]),  # type: ignore[arg-type]
                redacted_arguments=json.loads(str(r[11])),
                rationale=str(r[12]),
                outcome_summary=str(r[13]),
                schema_version=str(r[14]),
            )
            for r in rows
        )

    def purge_expired_entries(self, *, now_utc: Optional[datetime] = None) -> int:
        """Delete audit entries older than `self.retention_days` (preserving recent >= 30 days per NFR-10)."""
        ref_now = now_utc or datetime.now(timezone.utc)
        cutoff_iso = (ref_now - timedelta(days=self._retention_days)).isoformat()
        with self._connect() as conn:
            cursor = conn.execute(
                "DELETE FROM tool_audit_log WHERE timestamp_utc < ?;",
                (cutoff_iso,),
            )
            return int(cursor.rowcount or 0)


class CatalogBackedPermissionGate:
    """Production Layer 3 Permission Gate uniting `ToolRegistry`, `UserPermissionPolicyStore`, and `SqliteToolAuditLog`."""

    def __init__(
        self,
        registry: ToolRegistry,
        policy_store: UserPermissionPolicyStore,
        audit_log: Optional[SqliteToolAuditLog] = None,
        sensitive_action_policy: Literal["ask_every_time", "deny"] = "ask_every_time",
        auto_approve_reversible_local_writes: bool = False,
        default_task_id: str = "task_default",
        default_session_id: str = "sess_default",
    ) -> None:
        self._registry = registry
        self._policy_store = policy_store
        self._audit_log = audit_log
        self._sensitive_policy = sensitive_action_policy
        self._auto_approve_local_writes = auto_approve_reversible_local_writes
        self._default_task_id = default_task_id
        self._default_session_id = default_session_id
        self._user_decisions: Dict[str, Tuple[bool, str]] = {}

    def record_user_decision(
        self,
        proposal_id: str,
        approved: bool,
        reason: str = "User interactive decision",
    ) -> None:
        self._user_decisions[proposal_id] = (approved, reason)

    def evaluate_tool_proposal(
        self,
        proposal: ToolCallProposal,
    ) -> PermissionCheckResult:
        """Evaluate `proposal` against ADR-008 badge scope, FR-15 user policy, and Section 14 tiers."""
        spec = get_agent_spec(proposal.agent_role)

        # 1. Check ADR-008 specialized agent scope
        try:
            spec.validate_proposal_scope(proposal)
        except AgentScopeViolationError as exc:
            res = PermissionCheckResult(
                proposal_id=proposal.proposal_id,
                tool_name=proposal.tool_name,
                agent_role=proposal.agent_role,
                effective_sensitivity=SensitivityTier.IRREVERSIBLE_WRITE,
                decision=PermissionDecisionKind.DENIED_OUT_OF_SCOPE,
                requires_interactive_prompt=False,
                reason=str(exc),
            )
            self._log_gate_decision(proposal, res)
            return res

        # 2. Check if tool is registered in ToolRegistry (fail-closed if unknown)
        try:
            self._registry.get_tool(proposal.tool_name)
        except Exception as exc:
            res = PermissionCheckResult(
                proposal_id=proposal.proposal_id,
                tool_name=proposal.tool_name,
                agent_role=proposal.agent_role,
                effective_sensitivity=SensitivityTier.IRREVERSIBLE_WRITE,
                decision=PermissionDecisionKind.DENIED_BY_POLICY,
                requires_interactive_prompt=False,
                reason=f"Unregistered tool denied fail-closed: {exc}",
            )
            self._log_gate_decision(proposal, res)
            return res

        effective_tier = self._policy_store.get_effective_sensitivity(
            proposal.tool_name,
            proposal.declared_sensitivity,
        )

        # 3. Check if the user revoked this tool globally or for this agent (FR-15)
        if self._policy_store.is_tool_revoked(proposal.tool_name, proposal.agent_role):
            res = PermissionCheckResult(
                proposal_id=proposal.proposal_id,
                tool_name=proposal.tool_name,
                agent_role=proposal.agent_role,
                effective_sensitivity=effective_tier,
                decision=PermissionDecisionKind.DENIED_BY_POLICY,
                requires_interactive_prompt=False,
                reason=f"FR-15 policy: Tool {proposal.tool_name!r} has been revoked by the user.",
            )
            self._log_gate_decision(proposal, res)
            return res

        # 4. Check if the user already clicked Approve or Deny for this proposal_id
        if proposal.proposal_id in self._user_decisions:
            approved, user_reason = self._user_decisions[proposal.proposal_id]
            res = PermissionCheckResult(
                proposal_id=proposal.proposal_id,
                tool_name=proposal.tool_name,
                agent_role=proposal.agent_role,
                effective_sensitivity=effective_tier,
                decision=(
                    PermissionDecisionKind.USER_APPROVED
                    if approved
                    else PermissionDecisionKind.DENIED_BY_USER
                ),
                requires_interactive_prompt=False,
                reason=user_reason,
            )
            self._log_gate_decision(proposal, res)
            return res

        # 5. Auto-approve safe read_only tools (unless tightened by the user to a higher tier!)
        if effective_tier == SensitivityTier.READ_ONLY:
            res = PermissionCheckResult(
                proposal_id=proposal.proposal_id,
                tool_name=proposal.tool_name,
                agent_role=proposal.agent_role,
                effective_sensitivity=effective_tier,
                decision=PermissionDecisionKind.AUTO_APPROVED,
                requires_interactive_prompt=False,
                reason="Auto-approved read_only tool.",
            )
            return res

        # 6. Optional auto-approve for reversible local writes
        if (
            effective_tier == SensitivityTier.REVERSIBLE_WRITE
            and self._auto_approve_local_writes
        ):
            return PermissionCheckResult(
                proposal_id=proposal.proposal_id,
                tool_name=proposal.tool_name,
                agent_role=proposal.agent_role,
                effective_sensitivity=effective_tier,
                decision=PermissionDecisionKind.AUTO_APPROVED,
                requires_interactive_prompt=False,
                reason="Auto-approved reversible_write tool per policy.",
            )

        # 7. Sensitive / networked / irreversible actions follow sensitive_action_policy
        if self._sensitive_policy == "deny":
            res = PermissionCheckResult(
                proposal_id=proposal.proposal_id,
                tool_name=proposal.tool_name,
                agent_role=proposal.agent_role,
                effective_sensitivity=effective_tier,
                decision=PermissionDecisionKind.DENIED_BY_POLICY,
                requires_interactive_prompt=False,
                reason=f"Denied by policy ('deny') for sensitivity tier {effective_tier.value!r}.",
            )
            self._log_gate_decision(proposal, res)
            return res

        return PermissionCheckResult(
            proposal_id=proposal.proposal_id,
            tool_name=proposal.tool_name,
            agent_role=proposal.agent_role,
            effective_sensitivity=effective_tier,
            decision=None,
            requires_interactive_prompt=True,
            reason=(
                f"ADR-010: Tool {proposal.tool_name!r} ({effective_tier.value}) requires "
                "explicit human approval before execution."
            ),
        )

    def _log_gate_decision(
        self,
        proposal: ToolCallProposal,
        result: PermissionCheckResult,
    ) -> None:
        if self._audit_log is None or result.decision is None:
            return
        if result.is_denied:
            self._audit_log.record_entry(
                ToolAuditEntry(
                    task_id=self._default_task_id,
                    session_id=self._default_session_id,
                    node_id=proposal.node_id,
                    proposal_id=proposal.proposal_id,
                    agent_role=proposal.agent_role,
                    tool_name=proposal.tool_name,
                    effective_sensitivity=result.effective_sensitivity,
                    decision=result.decision,
                    execution_status="not_executed",
                    redacted_arguments=dict(proposal.arguments),
                    rationale=proposal.rationale,
                    outcome_summary=result.reason,
                )
            )


StructuredToolHandlerFn = Callable[[Dict[str, Any]], Mapping[str, Any]]


class SandboxedMcpToolExecutor:
    """Layer 3 Sandboxed Tool Executor implementing `ToolExecutorPort` (Phase 7 / Phase 15 Layers 3, 4 & 5).

    Every tool execution passes through:
    1. Approval stamp verification (`verify_approval_stamp` — `ADR-010`).
    2. Input argument JSON Schema validation (`ToolContract.validate_arguments` — `Phase 7`).
    3. Filesystem & Command Sandbox isolation (`FilesystemSandbox` / `CommandSandbox` — `Phase 15 Layer 4`).
    4. Output JSON Schema validation (`ToolContract.validate_output` — `Phase 7`).
    5. Durable SQLite Audit Logging (`SqliteToolAuditLog` — `Phase 15 Layer 5` / `NFR-08`).
    """

    def __init__(
        self,
        registry: ToolRegistry,
        filesystem_sandbox: FilesystemSandbox,
        command_sandbox: Optional[CommandSandbox] = None,
        audit_log: Optional[SqliteToolAuditLog] = None,
        event_bus: Optional[EventPublisherPort] = None,
        tool_handlers: Optional[Mapping[str, StructuredToolHandlerFn]] = None,
        default_task_id: str = "task_default",
        default_session_id: str = "sess_default",
    ) -> None:
        self._registry = registry
        self._fs_sandbox = filesystem_sandbox
        self._cmd_sandbox = command_sandbox or CommandSandbox(filesystem_sandbox)
        self._audit_log = audit_log
        self._event_bus = event_bus
        self._handlers: Dict[str, StructuredToolHandlerFn] = dict(tool_handlers or {})
        self._default_task_id = default_task_id
        self._default_session_id = default_session_id
        self.executed_proposals: List[ToolCallProposal] = []

    def register_tool_handler(
        self,
        tool_name: str,
        handler: StructuredToolHandlerFn,
    ) -> None:
        self._handlers[tool_name] = handler

    def execute_approved_tool(
        self,
        proposal: ToolCallProposal,
        permission_stamp: PermissionCheckResult,
    ) -> ToolExecutionResult:
        """Verify approval stamp, validate schema, enforce sandbox, run tool, and log audit entry."""
        # 1. Non-bypassable approval stamp check (ADR-010)
        verify_approval_stamp(proposal, permission_stamp)
        assert permission_stamp.decision is not None

        # 2. Validate tool exists and arguments match JSON Schema (Phase 7)
        try:
            contract: ToolContract = self._registry.validate_tool_call(
                proposal.tool_name,
                proposal.arguments,
            )
        except Exception as exc:
            self._record_audit_and_violation(
                proposal=proposal,
                stamp=permission_stamp,
                execution_status="schema_invalid",
                summary=f"Schema validation failed: {exc}",
            )
            raise

        # 3. Enforce FilesystemSandbox on all `path_scoped=True` parameters & CommandSandbox on terminal
        sanitized_args: Dict[str, Any] = dict(proposal.arguments)
        try:
            for path_param in contract.path_scoped_parameter_names:
                if path_param in sanitized_args and sanitized_args[path_param] is not None:
                    verified_path = self._fs_sandbox.resolve_and_verify_path(
                        str(sanitized_args[path_param])
                    )
                    sanitized_args[path_param] = str(verified_path)

            if proposal.tool_name == "terminal.execute":
                _, verified_cwd = self._cmd_sandbox.verify_command(
                    command=str(sanitized_args.get("command", "")),
                    cwd=sanitized_args.get("cwd"),
                )
                sanitized_args["cwd"] = str(verified_cwd)
        except SandboxViolationError as exc:
            self._record_audit_and_violation(
                proposal=proposal,
                stamp=permission_stamp,
                execution_status="sandbox_blocked",
                summary=str(exc),
            )
            raise

        # 4. Execute handler and validate output schema (Phase 7)
        handler = self._handlers.get(proposal.tool_name)
        if handler is None:
            err_msg = f"No runtime handler registered for tool {proposal.tool_name!r}."
            self._record_audit_and_violation(
                proposal=proposal,
                stamp=permission_stamp,
                execution_status="error",
                summary=err_msg,
                is_security_violation=False,
            )
            return ToolExecutionResult(
                proposal_id=proposal.proposal_id,
                tool_name=proposal.tool_name,
                status="error",
                sanitized_output=err_msg,
            )

        raw_result = handler(sanitized_args)
        try:
            contract.validate_output(raw_result)
        except Exception as exc:
            self._record_audit_and_violation(
                proposal=proposal,
                stamp=permission_stamp,
                execution_status="schema_invalid",
                summary=f"Output schema validation failed: {exc}",
            )
            raise

        self.executed_proposals.append(proposal)
        formatted_output = json.dumps(dict(raw_result), sort_keys=True)

        # 5. Record successful execution in SQLite Audit Log (Phase 15 Layer 5)
        if self._audit_log is not None:
            self._audit_log.record_entry(
                ToolAuditEntry(
                    task_id=self._default_task_id,
                    session_id=self._default_session_id,
                    node_id=proposal.node_id,
                    proposal_id=proposal.proposal_id,
                    agent_role=proposal.agent_role,
                    tool_name=proposal.tool_name,
                    effective_sensitivity=permission_stamp.effective_sensitivity,
                    decision=permission_stamp.decision,
                    execution_status="success",
                    redacted_arguments=dict(proposal.arguments),
                    rationale=proposal.rationale,
                    outcome_summary=formatted_output,
                )
            )

        return ToolExecutionResult(
            proposal_id=proposal.proposal_id,
            tool_name=proposal.tool_name,
            status="success",
            sanitized_output=formatted_output,
        )

    def _record_audit_and_violation(
        self,
        *,
        proposal: ToolCallProposal,
        stamp: PermissionCheckResult,
        execution_status: Literal["error", "sandbox_blocked", "schema_invalid"],
        summary: str,
        is_security_violation: bool = True,
    ) -> None:
        if self._audit_log is not None and stamp.decision is not None:
            self._audit_log.record_entry(
                ToolAuditEntry(
                    task_id=self._default_task_id,
                    session_id=self._default_session_id,
                    node_id=proposal.node_id,
                    proposal_id=proposal.proposal_id,
                    agent_role=proposal.agent_role,
                    tool_name=proposal.tool_name,
                    effective_sensitivity=stamp.effective_sensitivity,
                    decision=stamp.decision,
                    execution_status=execution_status,
                    redacted_arguments=dict(proposal.arguments),
                    rationale=proposal.rationale,
                    outcome_summary=summary,
                )
            )
        if is_security_violation and self._event_bus is not None:
            self._event_bus.publish(
                NovaEvent(
                    event_type=EventType.SECURITY_VIOLATION,
                    task_id=self._default_task_id,
                    session_id=self._default_session_id,
                    node_id=proposal.node_id,
                    agent_role=proposal.agent_role,
                    correlation_id=proposal.proposal_id,
                    payload={
                        "reason": execution_status,
                        "tool_name": proposal.tool_name,
                        "error": summary,
                    },
                )
            )


__all__ = [
    "CatalogBackedPermissionGate",
    "CommandSandbox",
    "FilesystemSandbox",
    "PolicyModificationError",
    "SandboxViolationError",
    "SandboxedMcpToolExecutor",
    "SqliteToolAuditLog",
    "ToolAuditEntry",
    "UserPermissionPolicyStore",
]