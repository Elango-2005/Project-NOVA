"""R01 authorization ledger: exact-action binding and durable, single-use claims.

This is runtime-owned authorization state, not a model output contract. A claimed
action is never reissued, even after a crash. SQLite cannot make a filesystem write
atomic with the ledger; an interrupted claim therefore has an uncertain outcome.
"""
from __future__ import annotations

from contextlib import contextmanager
import hashlib
import json
import sqlite3
import threading
import time
import uuid
from typing import Any, Callable, Iterator, Mapping, Optional

from pydantic import BaseModel

from nova.core.domain.primitives import PermissionDecisionKind
from nova.core.domain.proposals import ToolCallProposal
from nova.core.orchestration.ports import (
    PermissionCheckResult, ToolActionContext, UnapprovedToolExecutionError, copy_model,
    verify_approval_stamp, AuthorizationAlreadyClaimedError, AuthorizationOutcomeMismatch,
)


def _json(model: BaseModel) -> str:
    return model.model_dump_json() if hasattr(model, "model_dump_json") else model.json()


def _stamp(raw: str) -> PermissionCheckResult:
    if hasattr(PermissionCheckResult, "model_validate_json"):
        return PermissionCheckResult.model_validate_json(raw)
    return PermissionCheckResult.parse_raw(raw)


def action_digest(proposal: ToolCallProposal, context: ToolActionContext) -> str:
    if proposal.agent_role != context.agent_role or proposal.node_id != context.node_id:
        raise UnapprovedToolExecutionError("Proposal does not match trusted runtime identity.")
    payload = {
        "context": json.loads(_json(context)),
        "proposal_id": proposal.proposal_id,
        "tool": proposal.tool_name,
        "arguments": proposal.arguments,
        "sensitivity": proposal.declared_sensitivity.value,
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


class AuthorizationLedger:
    def __init__(self, db_path: str = ":memory:", *, ttl: int = 300,
                 clock: Callable[[], float] = time.time) -> None:
        self.lock = threading.RLock()
        self.clock = clock
        self.ttl = ttl
        self.db_path = db_path
        self.connection = sqlite3.connect(db_path, check_same_thread=False)
        self.connection.execute("PRAGMA synchronous=FULL")
        self.connection.execute("""CREATE TABLE IF NOT EXISTS tool_authorizations (
            proposal_id TEXT PRIMARY KEY, digest TEXT NOT NULL,
            expires REAL NOT NULL, status TEXT NOT NULL, stamp TEXT NOT NULL
        )""")
        self.connection.execute("""CREATE TABLE IF NOT EXISTS authorization_attempts (
            attempt_id TEXT PRIMARY KEY, timestamp REAL NOT NULL, outcome TEXT NOT NULL,
            reason_code TEXT NOT NULL, trusted_context TEXT, submitted_claims TEXT NOT NULL
        )""")
        self.connection.commit()
        if db_path != ":memory:":
            self.connection.close()

    @contextmanager
    def _transaction(self) -> Iterator[None]:
        with self.lock:
            if self.db_path != ":memory:":
                self.connection = sqlite3.connect(self.db_path)
            try:
                with self.connection:
                    yield
            finally:
                if self.db_path != ":memory:":
                    self.connection.close()

    def _row(self, proposal_id: str) -> Optional[tuple[str, float, str, str]]:
        return self.connection.execute(
            "SELECT digest, expires, status, stamp FROM tool_authorizations WHERE proposal_id=?",
            (proposal_id,),
        ).fetchone()

    def _check(self, row: Optional[tuple[str, float, str, str]], digest: str) -> None:
        if row is None or row[0] != digest:
            raise UnapprovedToolExecutionError("Unknown or modified authorization context/arguments.")
        if row[2] in ("claimed", "success", "error"):
            raise AuthorizationAlreadyClaimedError("Authorization already claimed; replay forbidden (outcome may be uncertain).")

        if self.clock() >= row[1]:
            raise UnapprovedToolExecutionError("Approval expired; submit a new action.")

    def issue(self, proposal: ToolCallProposal, context: ToolActionContext,
              policy: PermissionCheckResult) -> PermissionCheckResult:
        digest = action_digest(proposal, context)
        with self._transaction():
            self.connection.execute("BEGIN IMMEDIATE")
            row = self._row(proposal.proposal_id)
            if row is not None:
                self._check(row, digest)
                old = _stamp(row[3])
                if not policy.is_denied and old.decision in (
                    PermissionDecisionKind.USER_APPROVED, PermissionDecisionKind.DENIED_BY_USER,
                ):
                    policy = old
                expires = row[1]
                token = old.authorization_id
            else:
                expires = self.clock() + self.ttl
                token = uuid.uuid4().hex
            stamp = copy_model(policy, **{
                "context": context, "action_digest": digest,
                "authorization_id": token, "expires_at": expires,
            })
            status = "approved" if stamp.is_approved else "denied" if stamp.is_denied else "pending"
            self.connection.execute(
                "INSERT OR REPLACE INTO tool_authorizations VALUES (?, ?, ?, ?, ?)",
                (proposal.proposal_id, digest, expires, status, _json(stamp)),
            )
            return stamp

    def decide(self, proposal: ToolCallProposal, context: ToolActionContext,
               approved: bool, reason: str) -> None:
        with self._transaction():
            self.connection.execute("BEGIN IMMEDIATE")
            row = self._row(proposal.proposal_id)
            self._check(row, action_digest(proposal, context))
            if row[2] != "pending":
                raise UnapprovedToolExecutionError("Action is not awaiting a decision.")
            stamp = copy_model(_stamp(row[3]), **{
                "decision": PermissionDecisionKind.USER_APPROVED if approved else PermissionDecisionKind.DENIED_BY_USER,
                "requires_interactive_prompt": False, "reason": reason,
            })
            self.connection.execute(
                "UPDATE tool_authorizations SET status=?, stamp=? WHERE proposal_id=?",
                ("approved" if approved else "denied", _json(stamp), proposal.proposal_id),
            )

    def claim(self, proposal: ToolCallProposal, stamp: PermissionCheckResult,
              context: ToolActionContext, *, policy_check: Optional[Callable[[sqlite3.Connection], None]] = None) -> None:
        with self._transaction():
            self.connection.execute("BEGIN IMMEDIATE")
            verify_approval_stamp(proposal, stamp)
            row = self._row(proposal.proposal_id)
            self._check(row, action_digest(proposal, context))
            if policy_check is not None:
                policy_check(self.connection)
            if row[2] != "approved" or json.loads(row[3]) != json.loads(_json(stamp)):
                raise UnapprovedToolExecutionError("Forged, stale, or unapproved authorization.")
            self.connection.execute(
                "UPDATE tool_authorizations SET status='claimed' WHERE proposal_id=?",
                (proposal.proposal_id,),
            )
            self._insert_attempt("claimed", "authorization_claimed", context, {
                "proposal_id": proposal.proposal_id, "tool_name": proposal.tool_name})

    def _insert_attempt(self, outcome: str, reason_code: str, context: Optional[ToolActionContext],
                        submitted: Mapping[str, Any]) -> None:
        # No tokens, arguments, rationale or arbitrary exception messages in this trail.
        allowed = ("task_id", "session_id", "node_id", "proposal_id", "tool_name", "agent_role")
        claims = {k: str(v)[:256] for k, v in submitted.items() if k in allowed and v is not None}
        self.connection.execute("INSERT INTO authorization_attempts VALUES (?, ?, ?, ?, ?, ?)",
            (uuid.uuid4().hex, self.clock(), outcome, reason_code,
             _json(context) if context is not None else None, json.dumps(claims)))

    def record_rejection(self, context: Optional[ToolActionContext], submitted: Mapping[str, Any],
                         reason_code: str) -> None:
        with self._transaction():
            self._insert_attempt("rejected", reason_code, context, submitted)

    def query_attempts(self) -> list[dict[str, Any]]:
        with self._transaction():
            rows = self.connection.execute("SELECT * FROM authorization_attempts ORDER BY timestamp, rowid").fetchall()
        return [{"attempt_id": r[0], "timestamp": r[1], "outcome": r[2], "reason_code": r[3],
                 "trusted_context": json.loads(r[4]) if r[4] else None,
                 "submitted_claims": json.loads(r[5])} for r in rows]

    def execution_outcome(self, proposal: ToolCallProposal, context: ToolActionContext) -> Optional[str]:
        """Read the exact bound action without treating a claimed/expired record as a new request."""
        try:
            digest = action_digest(proposal, context)
        except UnapprovedToolExecutionError as exc:
            raise AuthorizationOutcomeMismatch("Execution outcome identity mismatch") from exc
        with self._transaction():
            row = self._row(proposal.proposal_id)
            if row is None:
                return None
            if row[0] != digest:
                raise AuthorizationOutcomeMismatch("Execution outcome identity mismatch")
            return row[2]

    def unconfirmed_actions(self) -> list[dict[str, Any]]:
        with self._transaction():
            rows = self.connection.execute("SELECT stamp FROM tool_authorizations WHERE status='claimed'").fetchall()
        return [{"proposal_id": s.proposal_id, "tool_name": s.tool_name,
                 "context": json.loads(_json(s.context)) if s.context else None}
                for s in (_stamp(row[0]) for row in rows)]

    def finish(self, proposal_id: str, status: str) -> None:
        with self._transaction():
            self.connection.execute(
                "UPDATE tool_authorizations SET status=? WHERE proposal_id=? AND status='claimed'",
                (status, proposal_id),
            )
