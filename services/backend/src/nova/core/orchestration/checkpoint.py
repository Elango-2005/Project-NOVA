"""NOVA Durable Task Checkpointing Substrate (Layer 6 / Layer 1 — ADR-003 / ADR-006 / NFR-09).

Real-Life Analogy:
Think of `SqliteCheckpointStore` as a **Video Game Auto-Save System**.
Every time an agent finishes a step or pauses to ask your permission (`AWAITING_APPROVAL`),
the Orchestrator saves a complete snapshot of the `TaskExecutionState` (the Project Clipboard)
to your local SQLite database (`nova.db`).
- If your laptop loses power or the backend process crashes mid-task, NOVA reads the latest
  checkpoint on startup, verifies its SHA-256 tamper/corruption checksum, and resumes from the
  exact node where it left off without redoing completed steps (satisfying NFR-09: 100% crash
  recovery with zero silent data loss).
- Per ADR-006, the Orchestrator talks to the `CheckpointStorePort` interface so we can use
  `InMemoryCheckpointStore` for fast unit tests or `SqliteCheckpointStore` (or LangGraph's
  SQLite checkpointer) in production.
"""

from __future__ import annotations

from nova.core.domain.events import NovaEvent
from contextlib import closing

import hashlib
import json
import sqlite3
import threading
from pathlib import Path
from typing import Literal, Dict, List, Optional, Protocol, Tuple, runtime_checkable

from pydantic import BaseModel, Field

try:
    from pydantic import ConfigDict, model_validator  # noqa: F401

    PYDANTIC_V2 = True
except ImportError:  # pragma: no cover
    PYDANTIC_V2 = False

from nova.core.orchestration.state import TaskExecutionState


_OWNER_UNSET = object()

OwnerTransition = Literal["retain", "acquire", "release"]


class CheckpointConflict(ValueError):
    """The observed checkpoint revision or requested owner transition is invalid."""


def _validate_write(latest: Optional["CheckpointEnvelope"], state: TaskExecutionState,
                    expected_seq: Optional[int], expected_owner: object,
                    owner_transition: OwnerTransition) -> bool:
    """Validate under the store's write lock. True means an exact idempotent save."""
    origin = state._checkpoint_origin
    if origin is not None and expected_seq is not None and expected_seq != origin[0]:
        raise CheckpointConflict("Explicit sequence differs from the observed source revision")
    if origin is not None and expected_owner is not _OWNER_UNSET and expected_owner != origin[1]:
        raise CheckpointConflict("Explicit owner differs from the observed source revision")
    if expected_seq is None and origin is not None:
        expected_seq = origin[0]
    if expected_owner is _OWNER_UNSET:
        expected_owner = origin[1] if origin else None
    if latest is None:
        if expected_seq is not None or state.approval_attempt_id is not None or owner_transition != "retain":
            raise CheckpointConflict("Initial checkpoint must be unowned and expect no existing revision")
        return False
    current = latest.state
    if expected_seq != latest.checkpoint_seq or expected_owner != current.approval_attempt_id:
        raise CheckpointConflict("Stale checkpoint sequence or owner; terminal outcome cannot be overwritten")
    if _serialize_state_json(current) == _serialize_state_json(state) and owner_transition == "retain":
        return True
    if current.status.value in ("completed", "failed", "cancelled"):
        raise CheckpointConflict("Task already has a terminal outcome")
    if state.checkpoint_seq <= latest.checkpoint_seq:
        raise CheckpointConflict("Checkpoint sequence must advance without replacing history")
    old_owner, new_owner = current.approval_attempt_id, state.approval_attempt_id
    if owner_transition == "retain":
        if old_owner != new_owner:
            raise CheckpointConflict("Ownership replacement requires an explicit legitimate transition")
    elif owner_transition == "acquire":
        pending = {k: r.pending_tool_proposal for k, r in current.node_records.items() if r.pending_tool_proposal}
        if (old_owner is not None or new_owner is None or current.status.value != "awaiting_approval"
                or state.status.value != "running" or not pending
                or any(state.node_records.get(k) != current.node_records[k] for k in pending)):
            raise CheckpointConflict("Invalid approval ownership acquisition")
    elif owner_transition == "release":
        if (old_owner is None or new_owner is not None or state.status.value != "running"
                or any(r.pending_tool_proposal is not None for r in state.node_records.values())):
            raise CheckpointConflict("Invalid approval ownership release")
    else:
        raise CheckpointConflict("Unknown ownership transition")
    return False


class CheckpointCorruptionError(RuntimeError):
    """Raised when a stored checkpoint fails SHA-256 integrity verification or schema parsing (NFR-09)."""


def _serialize_state_json(state: TaskExecutionState) -> str:
    """Serialize a TaskExecutionState deterministically to JSON across Pydantic v1 and v2."""
    if hasattr(state, "model_dump"):
        raw_dict = state.model_dump(mode="json")  # type: ignore[attr-defined]
    else:  # pragma: no cover
        raw_dict = json.loads(state.json())
    return json.dumps(raw_dict, sort_keys=True, separators=(",", ":"))


def _compute_sha256(payload_json: str) -> str:
    """Compute a hex SHA-256 digest over the serialized checkpoint JSON."""
    return hashlib.sha256(payload_json.encode("utf-8")).hexdigest()


class CheckpointEnvelope(BaseModel):
    """Immutable metadata + state wrapper stored at each node boundary (NFR-09)."""

    if PYDANTIC_V2:
        model_config = ConfigDict(frozen=True, extra="forbid")
    else:

        class Config:
            allow_mutation = False
            extra = "forbid"

    task_id: str = Field(..., min_length=1)
    checkpoint_seq: int = Field(..., ge=0)
    node_id: Optional[str] = Field(
        default=None,
        description="Graph node boundary that triggered this save (or None for task creation)",
    )
    sha256_checksum: str = Field(
        ...,
        min_length=64,
        max_length=64,
        description="SHA-256 integrity hash preventing silent data corruption (NFR-09)",
    )
    created_at_utc: str = Field(..., min_length=1)
    state: TaskExecutionState = Field(..., description="Deserialized task execution state")


@runtime_checkable
class CheckpointStorePort(Protocol):
    """Abstract interface for saving and recovering task execution checkpoints (ADR-006 / NFR-09)."""

    def save_checkpoint(
        self,
        state: TaskExecutionState,
        node_id: Optional[str] = None,
        *, protect_terminal: bool = False, expected_seq: Optional[int] = None,
        terminal_event: Optional[NovaEvent] = None,
        expected_owner: object = _OWNER_UNSET, owner_transition: OwnerTransition = "retain",
    ) -> CheckpointEnvelope:
        """Persist a task state snapshot and return its verified CheckpointEnvelope."""

    def load_latest(self, task_id: str) -> Optional[CheckpointEnvelope]:
        """Load and integrity-check the most recent checkpoint for `task_id`."""

    def terminal_events(self, task_id: str) -> Tuple[NovaEvent, ...]:
        """Return durable task terminal events."""

    def claim_terminal_event(self, task_id: str, key: str) -> bool:
        """Reserve one terminal emission; crash-safe delivery is not guaranteed."""

    def list_history(self, task_id: str) -> Tuple[CheckpointEnvelope, ...]:
        """Return all saved checkpoints for `task_id` in chronological sequence order."""


class InMemoryCheckpointStore:
    """Fast in-memory CheckpointStorePort implementation for isolated unit tests."""

    def __init__(self) -> None:
        self._terminal_events = set()
        self._durable_events = {}
        self._event_lock = threading.RLock()
        self._history: Dict[str, List[Tuple[int, Optional[str], str, str, str]]] = {}

    def terminal_events(self, task_id: str) -> Tuple[NovaEvent, ...]:
        event = self._durable_events.get(task_id)
        return (event,) if event else ()

    def claim_terminal_event(self, task_id: str, key: str) -> bool:
        with self._event_lock:
            identity = (task_id, key)
            if identity in self._terminal_events:
                return False
            self._terminal_events.add(identity)
            return True

    def save_checkpoint(
        self,
        state: TaskExecutionState,
        node_id: Optional[str] = None,
        *, protect_terminal: bool = False, expected_seq: Optional[int] = None,
        terminal_event: Optional[NovaEvent] = None,
        expected_owner: object = _OWNER_UNSET, owner_transition: OwnerTransition = "retain",
    ) -> CheckpointEnvelope:
        with self._event_lock:
            latest = self.load_latest(state.task_id)
            if _validate_write(latest, state, expected_seq, expected_owner, owner_transition):
                return latest
            payload_json = _serialize_state_json(state)
            checksum = _compute_sha256(payload_json)
            seq = state.checkpoint_seq
            created_at = state.updated_at_utc

            bucket = self._history.setdefault(state.task_id, [])
            bucket.append((seq, node_id, checksum, created_at, payload_json))
            bucket.sort(key=lambda r: r[0])
            self._history[state.task_id] = bucket
            state._checkpoint_origin = (seq, state.approval_attempt_id)
            if terminal_event is not None:
                self._terminal_events.add((state.task_id, terminal_event.event_type.value))
                self._durable_events.setdefault(state.task_id, terminal_event)

            return CheckpointEnvelope(
                task_id=state.task_id,
                checkpoint_seq=seq,
                node_id=node_id,
                sha256_checksum=checksum,
                created_at_utc=created_at,
                state=state,
            )

    def load_latest(self, task_id: str) -> Optional[CheckpointEnvelope]:
        bucket = self._history.get(task_id, [])
        if not bucket:
            return None
        seq, node_id, checksum, created_at, payload_json = bucket[-1]
        return self._verify_and_hydrate(task_id, seq, node_id, checksum, created_at, payload_json)

    def list_history(self, task_id: str) -> Tuple[CheckpointEnvelope, ...]:
        bucket = self._history.get(task_id, [])
        return tuple(
            self._verify_and_hydrate(task_id, seq, node_id, checksum, created_at, payload_json)
            for seq, node_id, checksum, created_at, payload_json in bucket
        )

    @staticmethod
    def _verify_and_hydrate(
        task_id: str,
        seq: int,
        node_id: Optional[str],
        checksum: str,
        created_at: str,
        payload_json: str,
    ) -> CheckpointEnvelope:
        computed = _compute_sha256(payload_json)
        if computed != checksum:
            raise CheckpointCorruptionError(
                f"NFR-09 integrity violation: Checkpoint {task_id}#{seq} SHA-256 mismatch "
                f"(expected {checksum}, got {computed})."
            )
        try:
            raw_dict = json.loads(payload_json)
            state = TaskExecutionState(**raw_dict)
            state._checkpoint_origin = (seq, state.approval_attempt_id)
        except Exception as exc:
            raise CheckpointCorruptionError(
                f"NFR-09 schema violation: Failed to deserialize checkpoint {task_id}#{seq}: {exc}"
            ) from exc

        return CheckpointEnvelope(
            task_id=task_id,
            checkpoint_seq=seq,
            node_id=node_id,
            sha256_checksum=checksum,
            created_at_utc=created_at,
            state=state,
        )


class SqliteCheckpointStore:
    """Durable, embedded SQLite checkpoint store with WAL mode and SHA-256 verification (NFR-09)."""

    def __init__(self, db_path: Path | str) -> None:
        self._db_path = Path(db_path)
        if str(self._db_path) != ":memory:":
            self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    @property
    def db_path(self) -> Path:
        return self._db_path

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self._db_path))
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA synchronous=NORMAL;")
        conn.execute("PRAGMA foreign_keys=ON;")
        return conn

    def _init_schema(self) -> None:
        with closing(self._connect()) as conn, conn:
            conn.execute("CREATE TABLE IF NOT EXISTS task_terminal_events (task_id TEXT PRIMARY KEY, event_json TEXT NOT NULL)")
            conn.execute("CREATE TABLE IF NOT EXISTS terminal_event_claims (task_id TEXT NOT NULL, event_key TEXT NOT NULL, PRIMARY KEY(task_id,event_key))")
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS task_checkpoints (
                    task_id TEXT NOT NULL,
                    checkpoint_seq INTEGER NOT NULL,
                    node_id TEXT,
                    task_status TEXT NOT NULL,
                    sha256_checksum TEXT NOT NULL,
                    created_at_utc TEXT NOT NULL,
                    state_json TEXT NOT NULL,
                    PRIMARY KEY (task_id, checkpoint_seq)
                );
                """
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_task_checkpoints_latest
                ON task_checkpoints (task_id, checkpoint_seq DESC);
                """
            )

    def terminal_events(self, task_id: str) -> Tuple[NovaEvent, ...]:
        with closing(self._connect()) as conn, conn:
            rows = conn.execute("SELECT event_json FROM task_terminal_events WHERE task_id=?", (task_id,)).fetchall()
        return tuple(NovaEvent(**json.loads(row[0])) for row in rows)

    def claim_terminal_event(self, task_id: str, key: str) -> bool:
        conn = self._connect()
        try:
            with conn:
                changed = conn.execute("INSERT OR IGNORE INTO terminal_event_claims VALUES (?, ?)",
                                       (task_id, key)).rowcount
                return changed == 1
        finally:
            conn.close()

    def save_checkpoint(
        self,
        state: TaskExecutionState,
        node_id: Optional[str] = None,
        *, protect_terminal: bool = False, expected_seq: Optional[int] = None,
        terminal_event: Optional[NovaEvent] = None,
        expected_owner: object = _OWNER_UNSET, owner_transition: OwnerTransition = "retain",
    ) -> CheckpointEnvelope:
        """Atomically persist a checkpoint snapshot to SQLite."""
        payload_json = _serialize_state_json(state)
        checksum = _compute_sha256(payload_json)
        seq = state.checkpoint_seq
        created_at = state.updated_at_utc

        with closing(self._connect()) as conn, conn:
            conn.execute("BEGIN IMMEDIATE")
            latest = conn.execute("SELECT checkpoint_seq, node_id, sha256_checksum, created_at_utc, state_json, task_status FROM task_checkpoints WHERE task_id=? ORDER BY checkpoint_seq DESC LIMIT 1", (state.task_id,)).fetchone()
            envelope = None if latest is None else InMemoryCheckpointStore._verify_and_hydrate(
                task_id=state.task_id, seq=latest[0], node_id=latest[1], checksum=latest[2],
                created_at=latest[3], payload_json=latest[4])
            if _validate_write(envelope, state, expected_seq, expected_owner, owner_transition):
                return envelope
            if terminal_event is not None:
                raw = terminal_event.model_dump_json() if hasattr(terminal_event, "model_dump_json") else terminal_event.json()
                conn.execute("INSERT INTO task_terminal_events VALUES (?, ?)", (state.task_id, raw))
                conn.execute("INSERT OR IGNORE INTO terminal_event_claims VALUES (?, ?)", (state.task_id, terminal_event.event_type.value))
            conn.execute(
                """
                INSERT INTO task_checkpoints (
                    task_id,
                    checkpoint_seq,
                    node_id,
                    task_status,
                    sha256_checksum,
                    created_at_utc,
                    state_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?);
                """,
                (
                    state.task_id,
                    seq,
                    node_id,
                    state.status.value,
                    checksum,
                    created_at,
                    payload_json,
                ),
            )

        state._checkpoint_origin = (seq, state.approval_attempt_id)
        return CheckpointEnvelope(
            task_id=state.task_id,
            checkpoint_seq=seq,
            node_id=node_id,
            sha256_checksum=checksum,
            created_at_utc=created_at,
            state=state,
        )

    def load_latest(self, task_id: str) -> Optional[CheckpointEnvelope]:
        """Load and verify the highest-sequence checkpoint for `task_id`."""
        with closing(self._connect()) as conn, conn:
            cursor = conn.execute(
                """
                SELECT checkpoint_seq, node_id, sha256_checksum, created_at_utc, state_json
                FROM task_checkpoints
                WHERE task_id = ?
                ORDER BY checkpoint_seq DESC
                LIMIT 1;
                """,
                (task_id,),
            )
            row = cursor.fetchone()

        if row is None:
            return None
        seq, node_id, checksum, created_at, payload_json = row
        return InMemoryCheckpointStore._verify_and_hydrate(
            task_id=task_id,
            seq=int(seq),
            node_id=node_id,
            checksum=str(checksum),
            created_at=str(created_at),
            payload_json=str(payload_json),
        )

    def list_history(self, task_id: str) -> Tuple[CheckpointEnvelope, ...]:
        """Return the complete ordered checkpoint history for `task_id`."""
        with closing(self._connect()) as conn, conn:
            cursor = conn.execute(
                """
                SELECT checkpoint_seq, node_id, sha256_checksum, created_at_utc, state_json
                FROM task_checkpoints
                WHERE task_id = ?
                ORDER BY checkpoint_seq ASC;
                """,
                (task_id,),
            )
            rows = cursor.fetchall()

        return tuple(
            InMemoryCheckpointStore._verify_and_hydrate(
                task_id=task_id,
                seq=int(seq),
                node_id=node_id,
                checksum=str(checksum),
                created_at=str(created_at),
                payload_json=str(payload_json),
            )
            for seq, node_id, checksum, created_at, payload_json in rows
        )