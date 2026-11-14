"""Unit and integration tests for NOVA Phase 13: Local API Security Hardening (`test_security_hardening.py`).

Verifies:
1. `LocalApiAuthGuard` enforces constant-time `X-Nova-Token` validation and blocks foreign website
   `Origin` headers (CSRF protection) while allowing `http://127.0.0.1:8765` and `http://localhost:8765`.
2. `LocalRateAndPayloadGuard` blocks oversized request payloads (`PayloadTooLargeError`) and blocks
   request floods exceeding the sliding-window rate limit (`RateLimitExceededError`).
3. `TamperEvidentAuditChain` seals real `ToolAuditEntry` records from `NovaRuntimeContainer` with an
   HMAC-SHA256 hash chain and mathematically detects both field modification (e.g., changing `decision`
   or `outcome_summary`) and row deletion in the middle of the audit trail (`AuditTamperDetectedError`).
"""

from __future__ import annotations

import tempfile
from pathlib import Path

from nova.api.app import StartTaskRequest, SubmitApprovalRequest
from nova.core.domain.primitives import PermissionDecisionKind
from nova.core.permissions.sandbox_and_audit import ToolAuditEntry
from nova.core.permissions.security_hardening import (
    AuditTamperDetectedError,
    LocalApiAuthGuard,
    LocalRateAndPayloadGuard,
    PayloadTooLargeError,
    RateLimitExceededError,
    SealedAuditRecord,
    TamperEvidentAuditChain,
    UnauthorizedApiTokenError,
    UntrustedOriginCsrfError,
)
from nova.main import NovaRuntimeContainer


class TestPhase13SecurityHardening:
    """Verification suite for Phase 13 (`services/backend/src/nova/core/permissions/security_hardening.py`)."""

    def test_local_api_auth_guard_enforces_token_and_blocks_foreign_csrf_origins(
        self,
    ) -> None:
        """Verifies X-Nova-Token constant-time check and localhost Origin CSRF protection."""
        guard = LocalApiAuthGuard(api_token="nova_secret_key_2026", require_token=True)
        assert guard.verify_token("nova_secret_key_2026") is True

        for bad_token in (None, "", "wrong_token"):
            try:
                guard.verify_token(bad_token)
                raise AssertionError(f"Expected UnauthorizedApiTokenError for {bad_token!r}")
            except UnauthorizedApiTokenError:
                pass

        # Allowed local / CLI origins
        assert guard.verify_origin(None) is True
        assert guard.verify_origin("http://127.0.0.1:8765") is True
        assert guard.verify_origin("http://localhost:5173") is True

        # Blocked foreign origins (CSRF attempt from a malicious browser tab)
        for evil_origin in ("https://evil-attacker.example.com", "http://192.168.1.99:8000"):
            try:
                guard.verify_origin(evil_origin)
                raise AssertionError(f"Expected UntrustedOriginCsrfError for {evil_origin!r}")
            except UntrustedOriginCsrfError:
                pass

    def test_rate_and_payload_guard_blocks_oversized_bodies_and_request_floods(
        self,
    ) -> None:
        """Verifies payload size cap and sliding-window rate limiter."""
        fake_now = {"t": 100.0}
        limiter = LocalRateAndPayloadGuard(
            max_payload_bytes=1024,
            max_requests_per_window=3,
            window_seconds=10.0,
            clock_fn=lambda: fake_now["t"],
        )

        # 1. Payload size check
        assert limiter.check_payload_size({"goal": "short text"}) < 1024
        try:
            limiter.check_payload_size("X" * 2048)
            raise AssertionError("Expected PayloadTooLargeError for 2KB string")
        except PayloadTooLargeError:
            pass

        # 2. Sliding-window rate limit (3 requests allowed in 10s; 4th is blocked)
        assert limiter.check_rate_limit("client_1") == 1
        assert limiter.check_rate_limit("client_1") == 2
        assert limiter.check_rate_limit("client_1") == 3
        try:
            limiter.check_rate_limit("client_1")
            raise AssertionError("Expected RateLimitExceededError on 4th request")
        except RateLimitExceededError:
            pass

        # Advance clock past the 10-second window -> requests succeed again!
        fake_now["t"] = 111.0
        assert limiter.check_rate_limit("client_1") == 1

    def test_tamper_evident_audit_chain_verifies_real_container_logs_and_catches_tampering(
        self,
    ) -> None:
        """Runs a 2-step task in NovaRuntimeContainer, seals audit logs, and catches tampering."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            container = NovaRuntimeContainer(
                data_dir=tmp_path / "data",
                workspace_dir=tmp_path / "workspace",
            )
            (container.workspace_dir / "report.txt").write_text(
                "Q4 security audit passed.",
                encoding="utf-8",
            )

            paused = container.api_service.start_task(
                StartTaskRequest(
                    task_id="task_sec_13",
                    session_id="sess_sec_13",
                    user_goal="Read report.txt and email it to security@nova.local",
                )
            )
            card = paused.pending_approvals[0]
            container.api_service.submit_approval(
                "task_sec_13",
                SubmitApprovalRequest(
                    node_id=card.node_id,
                    proposal_id=card.proposal_id,
                    approved=True,
                    reason="Approved in Phase 13 test",
                ),
            )

            logs = container.api_service.query_audit_logs()
            assert len(logs) == 2

            chain = TamperEvidentAuditChain(secret_key="phase13_audit_hmac_secret")
            sealed_records = chain.seal_existing_logs(logs)
            assert len(sealed_records) == 2

            # 1. Untampered chain verifies cleanly
            clean_report = chain.verify_chain()
            assert clean_report.valid is True
            assert clean_report.checked_count == 2
            assert clean_report.tampered_audit_id is None

            # 2. Simulate an attacker editing the 2nd audit log entry's outcome_summary on disk
            orig_rec = sealed_records[1]
            forged_entry = ToolAuditEntry(
                schema_version=orig_rec.entry.schema_version,
                audit_id=orig_rec.entry.audit_id,
                timestamp_utc=orig_rec.entry.timestamp_utc,
                task_id=orig_rec.entry.task_id,
                session_id=orig_rec.entry.session_id,
                node_id=orig_rec.entry.node_id,
                proposal_id=orig_rec.entry.proposal_id,
                agent_role=orig_rec.entry.agent_role,
                tool_name=orig_rec.entry.tool_name,
                effective_sensitivity=orig_rec.entry.effective_sensitivity,
                decision=PermissionDecisionKind.AUTO_APPROVED,  # Forged decision!
                execution_status=orig_rec.entry.execution_status,
                redacted_arguments=orig_rec.entry.redacted_arguments,
                rationale=orig_rec.entry.rationale,
                outcome_summary=orig_rec.entry.outcome_summary,
            )
            forged_chain = [
                sealed_records[0],
                SealedAuditRecord(
                    sequence_index=orig_rec.sequence_index,
                    audit_id=orig_rec.audit_id,
                    prev_hash=orig_rec.prev_hash,
                    seal_hash=orig_rec.seal_hash,
                    entry=forged_entry,
                ),
            ]

            tamper_report = chain.verify_chain(forged_chain)
            assert tamper_report.valid is False
            assert tamper_report.tampered_audit_id == orig_rec.audit_id
            assert "Tampered audit record detected" in str(tamper_report.failure_reason)

            # 3. Verify raise_on_tamper=True raises AuditTamperDetectedError
            try:
                chain.verify_chain(forged_chain, raise_on_tamper=True)
                raise AssertionError("Expected AuditTamperDetectedError on forged chain")
            except AuditTamperDetectedError:
                pass

    def test_tamper_evident_audit_chain_detects_deleted_first_row(self) -> None:
        """Verifies that deleting the first row from a 2-row audit log breaks the hash chain."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            container = NovaRuntimeContainer(
                data_dir=tmp_path / "data",
                workspace_dir=tmp_path / "workspace",
            )
            container.api_service.start_task(
                StartTaskRequest(
                    task_id="t_del_1",
                    session_id="s_del_1",
                    user_goal="Write first note to one.txt",
                )
            )
            container.api_service.start_task(
                StartTaskRequest(
                    task_id="t_del_2",
                    session_id="s_del_2",
                    user_goal="Write second note to two.txt",
                )
            )
            logs = container.api_service.query_audit_logs()
            chain = TamperEvidentAuditChain(secret_key="row_delete_test_secret")
            sealed = chain.seal_existing_logs(logs)
            assert len(sealed) == 2

            # Attacker deletes row 0 and re-indexes row 1 as sequence_index=0
            reindexed_survivor = SealedAuditRecord(
                sequence_index=0,
                audit_id=sealed[1].audit_id,
                prev_hash=sealed[1].prev_hash,
                seal_hash=sealed[1].seal_hash,
                entry=sealed[1].entry,
            )
            report = chain.verify_chain([reindexed_survivor])
            assert report.valid is False
            assert report.tampered_audit_id == sealed[1].audit_id
            assert "Broken prev_hash link" in str(report.failure_reason)