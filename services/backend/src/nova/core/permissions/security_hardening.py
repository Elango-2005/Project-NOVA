"""NOVA Phase 13: Local API Security Hardening — Token Guard, Rate/Payload Limiter & Tamper-Evident Audit Chain (`services/backend/src/nova/core/permissions/security_hardening.py`).

Plain-English Purpose (Locking the Outer Server Doors & Sealing the Flight Recorder):
In Phases 1–12, we armored the AI agents and sandboxes inside NOVA. Now, `security_hardening.py`
closes the 3 outer-perimeter attack surfaces on your local server (`localhost:8765`) and SQLite logs:

1. **VIP Keycard & Caller-ID Bouncer (`LocalApiAuthGuard`)**:
   - Protects sensitive API actions (`/tasks`, `/approve`, `/permissions/*`) with a 256-bit cryptographic
     token (`X-Nova-Token`) checked via `hmac.compare_digest` (immune to timing attacks).
   - Inspects browser `Origin` headers so a malicious website tab (`https://evil.example.com`) can never
     send Cross-Site Request Forgery (CSRF) commands to your `127.0.0.1` server.

2. **Circuit Breaker & Weight Scale (`LocalRateAndPayloadGuard`)**:
   - Blocks oversized payloads (e.g., someone pasting a 500 MB string into `/rag/ingest`) before they
     can exhaust your RAM or disk.
   - Enforces a sliding-window rate limit (e.g., max `60` requests per minute) so runaway loops or
     local scripts cannot freeze your CPU.

3. **Tamper-Evident SHA-256 Blockchain Seal (`TamperEvidentAuditChain`)**:
   - Links every `ToolAuditEntry` to the previous entry using an HMAC-SHA256 cryptographic hash chain
     (`prev_hash -> current_hash`).
   - If anyone opens `nova_audit.db` in an external SQLite program and edits or deletes a single row,
     `verify_chain()` immediately detects the broken seal and reports the exact tampered `audit_id`!
"""

from __future__ import annotations

from collections import deque
import hashlib
import hmac
import json
import secrets
import time
import urllib.parse
from typing import Any, Deque, Dict, List, Optional, Sequence, Tuple

from pydantic import BaseModel, Field

try:
    from pydantic import ConfigDict

    PYDANTIC_V2 = True
except ImportError:  # pragma: no cover
    PYDANTIC_V2 = False

from nova.core.permissions.sandbox_and_audit import ToolAuditEntry

_TRUSTED_LOOPBACK_HOSTS = frozenset({"127.0.0.1", "localhost", "::1"})


class SecurityHardeningError(Exception):
    """Base exception raised when a Phase 13 perimeter security check fails."""


class UnauthorizedApiTokenError(SecurityHardeningError):
    """Raised when `X-Nova-Token` is missing or invalid."""


class UntrustedOriginCsrfError(SecurityHardeningError):
    """Raised when a request comes from a non-local browser `Origin` (CSRF protection)."""


class PayloadTooLargeError(SecurityHardeningError):
    """Raised when a request body exceeds `max_payload_bytes`."""


class RateLimitExceededError(SecurityHardeningError):
    """Raised when a client exceeds `max_requests_per_window`."""


class AuditTamperDetectedError(SecurityHardeningError):
    """Raised when `TamperEvidentAuditChain` detects an edited, reordered, or deleted audit log row."""


class LocalApiAuthGuard:
    """Constant-time `X-Nova-Token` validator and localhost `Origin` CSRF guard."""

    def __init__(
        self,
        *,
        api_token: Optional[str] = None,
        require_token: bool = True,
    ) -> None:
        self._api_token = api_token or secrets.token_urlsafe(32)
        self.require_token = require_token

    @property
    def api_token(self) -> str:
        """Return the active 256-bit local API token."""
        return self._api_token

    def verify_token(self, provided_token: Optional[str]) -> bool:
        """Verify `X-Nova-Token` using constant-time `hmac.compare_digest`."""
        if not self.require_token:
            return True
        if not provided_token or not isinstance(provided_token, str):
            raise UnauthorizedApiTokenError(
                "Missing X-Nova-Token header for protected NOVA endpoint."
            )
        if not hmac.compare_digest(
            provided_token.encode("utf-8"),
            self._api_token.encode("utf-8"),
        ):
            raise UnauthorizedApiTokenError("Invalid X-Nova-Token provided.")
        return True

    @staticmethod
    def verify_origin(origin_header: Optional[str]) -> bool:
        """Allow empty origin (CLI/same-origin) or loopback origins (`127.0.0.1` / `localhost`); block foreign websites."""
        if not origin_header or not origin_header.strip():
            return True
        parsed = urllib.parse.urlparse(origin_header.strip())
        hostname = (parsed.hostname or "").lower()
        if parsed.scheme in ("http", "https") and hostname in _TRUSTED_LOOPBACK_HOSTS:
            return True
        raise UntrustedOriginCsrfError(
            f"CSRF protection blocked untrusted Origin header: {origin_header!r}."
        )


class LocalRateAndPayloadGuard:
    """Sliding-window rate limiter and request payload size cap."""

    def __init__(
        self,
        *,
        max_payload_bytes: int = 262_144,
        max_requests_per_window: int = 60,
        window_seconds: float = 60.0,
        clock_fn: Any = time.monotonic,
    ) -> None:
        self.max_payload_bytes = int(max_payload_bytes)
        self.max_requests_per_window = int(max_requests_per_window)
        self.window_seconds = float(window_seconds)
        self._clock_fn = clock_fn
        self._client_timestamps: Dict[str, Deque[float]] = {}

    def check_payload_size(self, payload: Any) -> int:
        """Verify that raw bytes, string, or JSON-serializable payload does not exceed `max_payload_bytes`."""
        if isinstance(payload, bytes):
            size = len(payload)
        elif isinstance(payload, str):
            size = len(payload.encode("utf-8"))
        else:
            size = len(json.dumps(payload, default=str).encode("utf-8"))

        if size > self.max_payload_bytes:
            raise PayloadTooLargeError(
                f"Request payload ({size} bytes) exceeds maximum allowed limit "
                f"({self.max_payload_bytes} bytes)."
            )
        return size

    def check_rate_limit(self, client_id: str = "local_client") -> int:
        """Record a request for `client_id` and raise `RateLimitExceededError` if window limit is exceeded."""
        now = float(self._clock_fn())
        history = self._client_timestamps.setdefault(client_id, deque())
        cutoff = now - self.window_seconds
        while history and history[0] <= cutoff:
            history.popleft()

        if len(history) >= self.max_requests_per_window:
            raise RateLimitExceededError(
                f"Rate limit exceeded for {client_id!r}: max {self.max_requests_per_window} "
                f"requests per {self.window_seconds}s window."
            )
        history.append(now)
        return len(history)


class SealedAuditRecord(BaseModel):
    """Wraps a `ToolAuditEntry` with its cryptographic hash-chain links (`prev_hash` and `seal_hash`)."""

    if PYDANTIC_V2:
        model_config = ConfigDict(frozen=True, extra="forbid")
    else:

        class Config:
            allow_mutation = False
            extra = "forbid"

    sequence_index: int = Field(..., ge=0)
    audit_id: str = Field(..., min_length=1)
    prev_hash: str = Field(..., min_length=64, max_length=64)
    seal_hash: str = Field(..., min_length=64, max_length=64)
    entry: ToolAuditEntry


class AuditChainVerificationReport(BaseModel):
    """Result of verifying a sequence of `SealedAuditRecord` items."""

    if PYDANTIC_V2:
        model_config = ConfigDict(frozen=True, extra="forbid")
    else:

        class Config:
            allow_mutation = False
            extra = "forbid"

    valid: bool
    checked_count: int = Field(..., ge=0)
    head_seal_hash: str
    tampered_audit_id: Optional[str] = None
    failure_reason: Optional[str] = None


_GENESIS_PREV_HASH = "0" * 64


class TamperEvidentAuditChain:
    """HMAC-SHA256 hash-chain sealer and verifier for `ToolAuditEntry` records."""

    def __init__(self, *, secret_key: Optional[str] = None) -> None:
        raw_secret = secret_key or secrets.token_hex(32)
        self._secret_bytes = raw_secret.encode("utf-8")
        self._records: List[SealedAuditRecord] = []

    @property
    def records(self) -> Tuple[SealedAuditRecord, ...]:
        return tuple(self._records)

    def compute_entry_seal(
        self,
        entry: ToolAuditEntry,
        *,
        prev_hash: str,
        sequence_index: int,
    ) -> str:
        """Compute deterministic HMAC-SHA256 over `prev_hash`, `sequence_index`, and canonical entry fields."""
        canonical_payload = {
            "sequence_index": sequence_index,
            "prev_hash": prev_hash,
            "audit_id": entry.audit_id,
            "timestamp_utc": entry.timestamp_utc,
            "task_id": entry.task_id,
            "session_id": entry.session_id,
            "node_id": entry.node_id,
            "proposal_id": entry.proposal_id,
            "agent_role": entry.agent_role.value,
            "tool_name": entry.tool_name,
            "effective_sensitivity": entry.effective_sensitivity.value,
            "decision": entry.decision.value,
            "execution_status": entry.execution_status,
            "redacted_arguments": dict(sorted(entry.redacted_arguments.items())),
            "outcome_summary": entry.outcome_summary,
        }
        canonical_bytes = json.dumps(
            canonical_payload,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return hmac.new(self._secret_bytes, canonical_bytes, hashlib.sha256).hexdigest()

    def append_and_seal(self, entry: ToolAuditEntry) -> SealedAuditRecord:
        """Seal a new `ToolAuditEntry` onto the end of the hash chain."""
        seq_idx = len(self._records)
        prev_hash = self._records[-1].seal_hash if self._records else _GENESIS_PREV_HASH
        seal = self.compute_entry_seal(entry, prev_hash=prev_hash, sequence_index=seq_idx)
        record = SealedAuditRecord(
            sequence_index=seq_idx,
            audit_id=entry.audit_id,
            prev_hash=prev_hash,
            seal_hash=seal,
            entry=entry,
        )
        self._records.append(record)
        return record

    def seal_existing_logs(
        self,
        entries: Sequence[ToolAuditEntry],
    ) -> Tuple[SealedAuditRecord, ...]:
        """Reset and seal a chronological sequence of `ToolAuditEntry` records from `nova_audit.db`."""
        self._records.clear()
        for entry in entries:
            self.append_and_seal(entry)
        return self.records

    def verify_chain(
        self,
        records: Optional[Sequence[SealedAuditRecord]] = None,
        *,
        raise_on_tamper: bool = False,
    ) -> AuditChainVerificationReport:
        """Mathematically verify every link and entry in the sealed audit chain."""
        target = self._records if records is None else list(records)
        expected_prev = _GENESIS_PREV_HASH

        for idx, rec in enumerate(target):
            if rec.sequence_index != idx:
                msg = (
                    f"Audit sequence index mismatch at position {idx} "
                    f"(got sequence_index={rec.sequence_index} on {rec.audit_id!r})."
                )
                if raise_on_tamper:
                    raise AuditTamperDetectedError(msg)
                return AuditChainVerificationReport(
                    valid=False,
                    checked_count=idx,
                    head_seal_hash=expected_prev,
                    tampered_audit_id=rec.audit_id,
                    failure_reason=msg,
                )

            if not hmac.compare_digest(rec.prev_hash, expected_prev):
                msg = (
                    f"Broken prev_hash link at audit_id={rec.audit_id!r} "
                    f"(expected {expected_prev[:12]}..., got {rec.prev_hash[:12]}...)."
                )
                if raise_on_tamper:
                    raise AuditTamperDetectedError(msg)
                return AuditChainVerificationReport(
                    valid=False,
                    checked_count=idx,
                    head_seal_hash=expected_prev,
                    tampered_audit_id=rec.audit_id,
                    failure_reason=msg,
                )

            recomputed = self.compute_entry_seal(
                rec.entry,
                prev_hash=rec.prev_hash,
                sequence_index=idx,
            )
            if not hmac.compare_digest(rec.seal_hash, recomputed):
                msg = (
                    f"Tampered audit record detected at audit_id={rec.audit_id!r}: "
                    "entry fields do not match cryptographic HMAC-SHA256 seal."
                )
                if raise_on_tamper:
                    raise AuditTamperDetectedError(msg)
                return AuditChainVerificationReport(
                    valid=False,
                    checked_count=idx,
                    head_seal_hash=expected_prev,
                    tampered_audit_id=rec.audit_id,
                    failure_reason=msg,
                )

            expected_prev = rec.seal_hash

        return AuditChainVerificationReport(
            valid=True,
            checked_count=len(target),
            head_seal_hash=expected_prev,
            tampered_audit_id=None,
            failure_reason=None,
        )


__all__ = [
    "AuditChainVerificationReport",
    "AuditTamperDetectedError",
    "LocalApiAuthGuard",
    "LocalRateAndPayloadGuard",
    "PayloadTooLargeError",
    "RateLimitExceededError",
    "SealedAuditRecord",
    "SecurityHardeningError",
    "TamperEvidentAuditChain",
    "UnauthorizedApiTokenError",
    "UntrustedOriginCsrfError",
]