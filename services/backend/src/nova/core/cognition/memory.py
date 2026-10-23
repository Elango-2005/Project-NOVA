"""NOVA Layer 4 Two-Tier Memory Architecture (ADR-006 / ADR-007 / ADR-008 / Phase 5 / Section 16.2).

Real-Life Analogy:
Imagine NOVA's memory system as two things on an Office Manager's desk:
1. A **Sticky-Note Pad (`WorkingMemoryStore`)** — Short-Term Memory:
   - Holds quick scratch notes for the current conversation or active task in fast RAM.
   - Cleared when the session ends unless an important note is promoted to the filing cabinet.

2. A **3-Drawer Locked Filing Cabinet (`SqliteLongTermMemoryStore`)** — Long-Term Memory on Disk:
   - **Drawer 1 (`SEMANTIC` — Facts & Preferences)**: Durable facts about you, like
     *"Sam's email is sam@accounting.example.com"* or *"Explain code in simple English with analogies."*
   - **Drawer 2 (`EPISODIC` — Past Events Diary)**: Records of completed tasks, like
     *"On Oct 23, NOVA summarized Invoice #INV-2026-10 ($1,250.00) and emailed Sam."*
   - **Drawer 3 (`PROCEDURAL` — How-To Recipes)**: Learned step-by-step habits, like
     *"Before submitting an expense report, always convert amounts to USD."*

3. A **Security Guard at the Filing Cabinet (`MemoryManager`)**:
   - **Checks ID Badges (`ADR-008` / `Section 12.3`)**: Before any agent can open a drawer,
     `MemoryManager` checks `get_agent_spec(agent_role).validate_memory_access(partition, mode)`.
     For example, `Desktop` and `Voice` agents are blocked from opening any long-term drawer,
     and `Planner` can read `SEMANTIC`/`EPISODIC` but can never write to them directly!
   - **Blocks Passwords/Secrets (`Section 16.2`)**: Refuses to store `CREDENTIALS_SECRETS` in memory.
   - **Smart Search & Conflict Resolution (`Phase 5`)**: Ranks memories by combining keyword/meaning
     match, importance score, and recency, and automatically supersedes outdated facts when a
     `memory_key` is updated (most-recent-wins, while keeping old versions for audit history).
   - **Broadcasts Events (`ADR-002` / `NFR-08`)**: Emits `memory.read`, `memory.write`, and
     `memory.promoted` events on the Event Bus for 100% auditability.
"""

from __future__ import annotations

import json
import math
import re
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Protocol, Sequence, Tuple, runtime_checkable

from pydantic import BaseModel, Field

try:
    from pydantic import ConfigDict

    PYDANTIC_V2 = True
except ImportError:  # pragma: no cover
    PYDANTIC_V2 = False

from nova.core.domain.agent_spec import get_agent_spec
from nova.core.domain.context import DataGovernanceError, sanitize_untrusted_text
from nova.core.domain.events import EventType, NovaEvent
from nova.core.domain.primitives import (
    CURRENT_SCHEMA_VERSION,
    AgentRole,
    DataClassification,
    MemoryPartition,
    validate_schema_version,
)
from nova.core.platform.event_bus import EventPublisherPort

_WORD_TOKEN_RE = re.compile(r"[a-z0-9_@.]+")


def _tokenize(text: str) -> Tuple[str, ...]:
    """Extract lowercase alphanumeric/email tokens for lightweight local similarity scoring."""
    return tuple(_WORD_TOKEN_RE.findall(text.lower()))


def _cosine_similarity(vec_a: Sequence[float], vec_b: Sequence[float]) -> float:
    """Compute cosine similarity between two embedding vectors in [-1.0, 1.0]."""
    if not vec_a or not vec_b or len(vec_a) != len(vec_b):
        return 0.0
    dot = sum(a * b for a, b in zip(vec_a, vec_b))
    norm_a = math.sqrt(sum(a * a for a in vec_a))
    norm_b = math.sqrt(sum(b * b for b in vec_b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return max(0.0, min(1.0, dot / (norm_a * norm_b)))


def _lexical_overlap_score(
    query: str,
    content: str,
    tags: Sequence[str],
    memory_key: Optional[str] = None,
) -> float:
    """Compute normalized token overlap in [0.0, 1.0] between a search query and memory text."""
    q_tokens = set(_tokenize(query))
    if not q_tokens:
        return 1.0
    doc_tokens = set(_tokenize(content)) | {t.lower() for t in tags}
    if memory_key:
        doc_tokens |= set(_tokenize(memory_key)) | set(memory_key.lower().replace(".", " ").split())
    if not doc_tokens:
        return 0.0
    matched = len(q_tokens & doc_tokens)
    return matched / len(q_tokens)


class MemoryRecord(BaseModel):
    """Immutable record stored in either Short-Term Working Memory or Long-Term SQLite Memory."""

    if PYDANTIC_V2:
        model_config = ConfigDict(frozen=True, extra="forbid")
    else:

        class Config:
            allow_mutation = False
            extra = "forbid"

    schema_version: str = Field(default=CURRENT_SCHEMA_VERSION)
    memory_id: str = Field(default_factory=lambda: f"mem_{uuid.uuid4().hex}")
    partition: MemoryPartition = Field(
        ..., description="WORKING, SEMANTIC, EPISODIC, or PROCEDURAL (ADR-007)"
    )
    memory_key: Optional[str] = Field(
        default=None,
        description=(
            "Optional canonical key (e.g., 'contact.sam.email' or 'pref.units') used by "
            "Phase 5 conflict resolution to supersede outdated facts cleanly."
        ),
    )
    content: str = Field(..., min_length=1, description="Sanitized memory text")
    tags: Tuple[str, ...] = Field(default_factory=tuple)
    importance: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Phase 5 importance weight in [0.0, 1.0]",
    )
    data_classification: DataClassification = Field(
        default=DataClassification.USER_CONTENT_LOCAL,
        description="Section 16.2 data classification (CREDENTIALS_SECRETS is forbidden)",
    )
    source_task_id: str = Field(..., min_length=1)
    source_session_id: str = Field(..., min_length=1)
    created_by_agent: AgentRole = Field(...)
    created_at_utc: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    embedding: Tuple[float, ...] = Field(
        default_factory=tuple,
        description="Optional local dense vector embedding for semantic similarity ranking",
    )
    superseded_by_id: Optional[str] = Field(
        default=None,
        description="If non-None, this record was superseded by a newer fact with the same memory_key",
    )

    def __init__(self, **data: object) -> None:
        version = data.get("schema_version", CURRENT_SCHEMA_VERSION)
        if isinstance(version, str):
            validate_schema_version(version)

        raw_cls = data.get("data_classification", DataClassification.USER_CONTENT_LOCAL)
        if (
            raw_cls == DataClassification.CREDENTIALS_SECRETS
            or raw_cls == "credentials_secrets"
        ):
            raise DataGovernanceError(
                "Section 16.2 violation: CREDENTIALS_SECRETS must never be stored in NOVA memory."
            )

        raw_content = data.get("content")
        if isinstance(raw_content, str):
            data = dict(data)
            data["content"] = sanitize_untrusted_text(raw_content)

        super().__init__(**data)


class ScoredMemoryMatch(BaseModel):
    """Ranked memory retrieval result combining similarity, importance, and recency (Phase 5)."""

    if PYDANTIC_V2:
        model_config = ConfigDict(frozen=True, extra="forbid")
    else:

        class Config:
            allow_mutation = False
            extra = "forbid"

    record: MemoryRecord = Field(...)
    score: float = Field(..., ge=0.0, le=1.0)
    similarity_score: float = Field(..., ge=0.0, le=1.0)
    importance_score: float = Field(..., ge=0.0, le=1.0)
    recency_score: float = Field(..., ge=0.0, le=1.0)


@runtime_checkable
class LongTermMemoryStorePort(Protocol):
    """Abstract interface for persistent Long-Term Memory storage (ADR-006 / ADR-007)."""

    def store_record(self, record: MemoryRecord) -> MemoryRecord:
        """Persist `record` and supersede any older active record sharing the same `(partition, memory_key)`."""

    def get_record(self, memory_id: str) -> Optional[MemoryRecord]:
        """Look up a single MemoryRecord by ID."""

    def search_partition(
        self,
        partition: MemoryPartition,
        query: str = "",
        query_embedding: Sequence[float] = (),
        top_k: int = 5,
        include_superseded: bool = False,
    ) -> Tuple[ScoredMemoryMatch, ...]:
        """Retrieve and rank records in `partition` using similarity + importance + recency."""


class WorkingMemoryStore:
    """Fast in-process Short-Term Working Memory scoped to a session or task (ADR-007 / Phase 5)."""

    def __init__(self) -> None:
        self._by_session: Dict[str, List[MemoryRecord]] = {}

    def write(self, record: MemoryRecord) -> MemoryRecord:
        if record.partition != MemoryPartition.WORKING:
            raise ValueError(
                f"WorkingMemoryStore only accepts MemoryPartition.WORKING, got {record.partition.value!r}."
            )
        bucket = self._by_session.setdefault(record.source_session_id, [])
        if record.memory_key:
            # Mark any prior active working note with the same memory_key as superseded
            updated_bucket: List[MemoryRecord] = []
            for existing in bucket:
                if (
                    existing.memory_key == record.memory_key
                    and existing.superseded_by_id is None
                ):
                    updated_bucket.append(
                        MemoryRecord(
                            schema_version=existing.schema_version,
                            memory_id=existing.memory_id,
                            partition=existing.partition,
                            memory_key=existing.memory_key,
                            content=existing.content,
                            tags=existing.tags,
                            importance=existing.importance,
                            data_classification=existing.data_classification,
                            source_task_id=existing.source_task_id,
                            source_session_id=existing.source_session_id,
                            created_by_agent=existing.created_by_agent,
                            created_at_utc=existing.created_at_utc,
                            embedding=existing.embedding,
                            superseded_by_id=record.memory_id,
                        )
                    )
                else:
                    updated_bucket.append(existing)
            bucket = updated_bucket

        bucket.append(record)
        self._by_session[record.source_session_id] = bucket
        return record

    def read_session(
        self,
        session_id: str,
        task_id: Optional[str] = None,
        include_superseded: bool = False,
    ) -> Tuple[MemoryRecord, ...]:
        bucket = self._by_session.get(session_id, [])
        results: List[MemoryRecord] = []
        for rec in bucket:
            if not include_superseded and rec.superseded_by_id is not None:
                continue
            if task_id is not None and rec.source_task_id != task_id:
                continue
            results.append(rec)
        return tuple(results)

    def get_by_id(self, session_id: str, memory_id: str) -> Optional[MemoryRecord]:
        for rec in self._by_session.get(session_id, ()):
            if rec.memory_id == memory_id:
                return rec
        return None

    def clear_session(self, session_id: str) -> None:
        self._by_session.pop(session_id, None)


class SqliteLongTermMemoryStore:
    """Embedded SQLite Long-Term Memory store supporting SEMANTIC, EPISODIC, and PROCEDURAL partitions."""

    def __init__(self, db_path: Path | str) -> None:
        self._db_path = Path(db_path)
        if str(self._db_path) != ":memory:":
            self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self._db_path))
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA synchronous=NORMAL;")
        return conn

    def _init_schema(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS long_term_memories (
                    memory_id TEXT PRIMARY KEY,
                    partition TEXT NOT NULL,
                    memory_key TEXT,
                    content TEXT NOT NULL,
                    tags_json TEXT NOT NULL,
                    importance REAL NOT NULL,
                    data_classification TEXT NOT NULL,
                    source_task_id TEXT NOT NULL,
                    source_session_id TEXT NOT NULL,
                    created_by_agent TEXT NOT NULL,
                    created_at_utc TEXT NOT NULL,
                    embedding_json TEXT NOT NULL,
                    superseded_by_id TEXT,
                    schema_version TEXT NOT NULL
                );
                """
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_ltm_partition_active
                ON long_term_memories (partition, superseded_by_id, created_at_utc DESC);
                """
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_ltm_partition_key
                ON long_term_memories (partition, memory_key);
                """
            )

    def store_record(self, record: MemoryRecord) -> MemoryRecord:
        if record.partition == MemoryPartition.WORKING:
            raise ValueError(
                "SqliteLongTermMemoryStore only stores long-term partitions "
                "(SEMANTIC, EPISODIC, PROCEDURAL), not WORKING."
            )

        with self._connect() as conn:
            # Phase 5 Conflict Resolution (most-recent-wins):
            # If `memory_key` is set, mark any older active record in the same partition as superseded!
            if record.memory_key:
                conn.execute(
                    """
                    UPDATE long_term_memories
                    SET superseded_by_id = ?
                    WHERE partition = ?
                      AND memory_key = ?
                      AND superseded_by_id IS NULL
                      AND memory_id != ?;
                    """,
                    (
                        record.memory_id,
                        record.partition.value,
                        record.memory_key,
                        record.memory_id,
                    ),
                )

            conn.execute(
                """
                INSERT OR REPLACE INTO long_term_memories (
                    memory_id,
                    partition,
                    memory_key,
                    content,
                    tags_json,
                    importance,
                    data_classification,
                    source_task_id,
                    source_session_id,
                    created_by_agent,
                    created_at_utc,
                    embedding_json,
                    superseded_by_id,
                    schema_version
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
                """,
                (
                    record.memory_id,
                    record.partition.value,
                    record.memory_key,
                    record.content,
                    json.dumps(list(record.tags)),
                    record.importance,
                    record.data_classification.value,
                    record.source_task_id,
                    record.source_session_id,
                    record.created_by_agent.value,
                    record.created_at_utc,
                    json.dumps(list(record.embedding)),
                    record.superseded_by_id,
                    record.schema_version,
                ),
            )

        return record

    def get_record(self, memory_id: str) -> Optional[MemoryRecord]:
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT memory_id, partition, memory_key, content, tags_json, importance,
                       data_classification, source_task_id, source_session_id,
                       created_by_agent, created_at_utc, embedding_json,
                       superseded_by_id, schema_version
                FROM long_term_memories
                WHERE memory_id = ?;
                """,
                (memory_id,),
            ).fetchone()
        if row is None:
            return None
        return self._row_to_record(row)

    def search_partition(
        self,
        partition: MemoryPartition,
        query: str = "",
        query_embedding: Sequence[float] = (),
        top_k: int = 5,
        include_superseded: bool = False,
    ) -> Tuple[ScoredMemoryMatch, ...]:
        if partition == MemoryPartition.WORKING:
            raise ValueError("Use WorkingMemoryStore for WORKING partition queries.")

        sql = """
            SELECT memory_id, partition, memory_key, content, tags_json, importance,
                   data_classification, source_task_id, source_session_id,
                   created_by_agent, created_at_utc, embedding_json,
                   superseded_by_id, schema_version
            FROM long_term_memories
            WHERE partition = ?
        """
        params: List[object] = [partition.value]
        if not include_superseded:
            sql += " AND superseded_by_id IS NULL"
        sql += " ORDER BY created_at_utc DESC;"

        with self._connect() as conn:
            rows = conn.execute(sql, tuple(params)).fetchall()

        records = [self._row_to_record(r) for r in rows]
        if not records:
            return ()

        total = len(records)
        scored: List[ScoredMemoryMatch] = []
        for rank_idx, rec in enumerate(records):
            # Recency score in [0.2, 1.0] based on chronological order (newest = 1.0)
            recency_score = 1.0 if total == 1 else max(0.2, 1.0 - (0.8 * rank_idx / (total - 1)))

            # Similarity score combines lexical overlap and optional vector cosine similarity
            lex_sim = _lexical_overlap_score(query, rec.content, rec.tags, rec.memory_key)
            vec_sim = (
                _cosine_similarity(query_embedding, rec.embedding)
                if (query_embedding and rec.embedding)
                else 0.0
            )
            if query.strip() and query_embedding and rec.embedding:
                sim_score = 0.5 * lex_sim + 0.5 * vec_sim
            elif query_embedding and rec.embedding:
                sim_score = vec_sim
            else:
                sim_score = lex_sim

            # Filter out completely unrelated records when a specific text query was provided
            if query.strip() and sim_score == 0.0:
                continue

            # Phase 5 weighted ranking: 55% similarity + 25% importance + 20% recency
            combined = round(
                (0.55 * sim_score) + (0.25 * rec.importance) + (0.20 * recency_score),
                4,
            )
            scored.append(
                ScoredMemoryMatch(
                    record=rec,
                    score=min(1.0, max(0.0, combined)),
                    similarity_score=round(sim_score, 4),
                    importance_score=round(rec.importance, 4),
                    recency_score=round(recency_score, 4),
                )
            )

        scored.sort(key=lambda m: (m.score, m.record.created_at_utc), reverse=True)
        return tuple(scored[: max(1, top_k)])

    @staticmethod
    def _row_to_record(row: Tuple[object, ...]) -> MemoryRecord:
        return MemoryRecord(
            memory_id=str(row[0]),
            partition=MemoryPartition(str(row[1])),
            memory_key=str(row[2]) if row[2] is not None else None,
            content=str(row[3]),
            tags=tuple(json.loads(str(row[4]))),
            importance=float(row[5]),  # type: ignore[arg-type]
            data_classification=DataClassification(str(row[6])),
            source_task_id=str(row[7]),
            source_session_id=str(row[8]),
            created_by_agent=AgentRole(str(row[9])),
            created_at_utc=str(row[10]),
            embedding=tuple(float(x) for x in json.loads(str(row[11]))),
            superseded_by_id=str(row[12]) if row[12] is not None else None,
            schema_version=str(row[13]),
        )


class MemoryManager:
    """Layer 4 Memory Manager enforcing ADR-007 two-tier separation and ADR-008 agent badge scopes."""

    def __init__(
        self,
        working_store: WorkingMemoryStore,
        long_term_store: LongTermMemoryStorePort,
        event_bus: EventPublisherPort,
    ) -> None:
        self._working = working_store
        self._long_term = long_term_store
        self._event_bus = event_bus

    def write_memory(
        self,
        *,
        agent_role: AgentRole,
        partition: MemoryPartition,
        content: str,
        task_id: str,
        session_id: str,
        memory_key: Optional[str] = None,
        tags: Sequence[str] = (),
        importance: float = 0.5,
        data_classification: DataClassification = DataClassification.USER_CONTENT_LOCAL,
        embedding: Sequence[float] = (),
    ) -> MemoryRecord:
        """Validate agent write permission for `partition`, store the record, and emit `memory.write`."""
        spec = get_agent_spec(agent_role)
        try:
            spec.validate_memory_access(partition, mode="write")
        except Exception as exc:
            self._event_bus.publish(
                NovaEvent(
                    event_type=EventType.SECURITY_VIOLATION,
                    task_id=task_id,
                    session_id=session_id,
                    agent_role=agent_role,
                    payload={
                        "reason": "unauthorized_memory_write",
                        "partition": partition.value,
                        "error": str(exc),
                    },
                )
            )
            raise

        record = MemoryRecord(
            partition=partition,
            memory_key=memory_key,
            content=content,
            tags=tuple(tags),
            importance=importance,
            data_classification=data_classification,
            source_task_id=task_id,
            source_session_id=session_id,
            created_by_agent=agent_role,
            embedding=tuple(embedding),
        )

        if partition == MemoryPartition.WORKING:
            saved = self._working.write(record)
        else:
            saved = self._long_term.store_record(record)

        self._event_bus.publish(
            NovaEvent(
                event_type=EventType.MEMORY_WRITE,
                task_id=task_id,
                session_id=session_id,
                agent_role=agent_role,
                correlation_id=saved.memory_id,
                payload={
                    "memory_id": saved.memory_id,
                    "partition": partition.value,
                    "memory_key": memory_key,
                    "importance": importance,
                },
            )
        )
        return saved

    def query_memory(
        self,
        *,
        agent_role: AgentRole,
        partition: MemoryPartition,
        task_id: str,
        session_id: str,
        query: str = "",
        query_embedding: Sequence[float] = (),
        top_k: int = 5,
        include_superseded: bool = False,
    ) -> Tuple[ScoredMemoryMatch, ...]:
        """Validate agent read permission for `partition`, search memories, and emit `memory.read`."""
        spec = get_agent_spec(agent_role)
        try:
            spec.validate_memory_access(partition, mode="read")
        except Exception as exc:
            self._event_bus.publish(
                NovaEvent(
                    event_type=EventType.SECURITY_VIOLATION,
                    task_id=task_id,
                    session_id=session_id,
                    agent_role=agent_role,
                    payload={
                        "reason": "unauthorized_memory_read",
                        "partition": partition.value,
                        "error": str(exc),
                    },
                )
            )
            raise

        if partition == MemoryPartition.WORKING:
            working_recs = self._working.read_session(
                session_id=session_id,
                include_superseded=include_superseded,
            )
            matches: List[ScoredMemoryMatch] = []
            for rec in working_recs:
                sim = _lexical_overlap_score(query, rec.content, rec.tags, rec.memory_key)
                if query.strip() and sim == 0.0:
                    continue
                matches.append(
                    ScoredMemoryMatch(
                        record=rec,
                        score=sim,
                        similarity_score=sim,
                        importance_score=rec.importance,
                        recency_score=1.0,
                    )
                )
            results = tuple(matches[: max(1, top_k)])
        else:
            results = self._long_term.search_partition(
                partition=partition,
                query=query,
                query_embedding=query_embedding,
                top_k=top_k,
                include_superseded=include_superseded,
            )

        self._event_bus.publish(
            NovaEvent(
                event_type=EventType.MEMORY_READ,
                task_id=task_id,
                session_id=session_id,
                agent_role=agent_role,
                payload={
                    "partition": partition.value,
                    "query": query,
                    "result_count": len(results),
                },
            )
        )
        return results

    def promote_working_to_long_term(
        self,
        *,
        session_id: str,
        task_id: str,
        working_memory_id: str,
        target_partition: MemoryPartition,
        promoting_agent: AgentRole,
        memory_key: Optional[str] = None,
        importance: Optional[float] = None,
    ) -> MemoryRecord:
        """Promote a short-term working memory note into a durable long-term SQLite partition (ADR-007)."""
        if target_partition == MemoryPartition.WORKING:
            raise ValueError("Promotion target_partition must be SEMANTIC, EPISODIC, or PROCEDURAL.")

        spec = get_agent_spec(promoting_agent)
        spec.validate_memory_access(MemoryPartition.WORKING, mode="read")
        spec.validate_memory_access(target_partition, mode="write")

        source_rec = self._working.get_by_id(session_id=session_id, memory_id=working_memory_id)
        if source_rec is None:
            raise KeyError(
                f"Working memory {working_memory_id!r} not found in session {session_id!r}."
            )

        promoted = MemoryRecord(
            partition=target_partition,
            memory_key=memory_key if memory_key is not None else source_rec.memory_key,
            content=source_rec.content,
            tags=source_rec.tags,
            importance=importance if importance is not None else source_rec.importance,
            data_classification=source_rec.data_classification,
            source_task_id=task_id,
            source_session_id=session_id,
            created_by_agent=promoting_agent,
            embedding=source_rec.embedding,
        )
        saved = self._long_term.store_record(promoted)

        self._event_bus.publish(
            NovaEvent(
                event_type=EventType.MEMORY_PROMOTED,
                task_id=task_id,
                session_id=session_id,
                agent_role=promoting_agent,
                correlation_id=saved.memory_id,
                payload={
                    "from_working_memory_id": working_memory_id,
                    "promoted_memory_id": saved.memory_id,
                    "target_partition": target_partition.value,
                    "memory_key": saved.memory_key,
                },
            )
        )
        return saved