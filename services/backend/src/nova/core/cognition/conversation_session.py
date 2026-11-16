"""NOVA Phase 17: Human-Readable Conversational Chat & Personal Assistant Session (`services/backend/src/nova/core/cognition/conversation_session.py`).

Plain-English Purpose (The Friendly Translator Between You and NOVA's Engine):
Earlier, NOVA's API returned raw developer JSON (`{"task_id": "...", "completed_nodes": [...]}`),
which made the dashboard look technical and hard to read.

`conversation_session.py` acts as a **warm, plain-English translator and conversation memory**:

1. **Natural Chat Input (`send_message`)**:
   - You type normal sentences like:
     - *"Note: Deep Learning Exam — Study CNNs, RNNs, and backpropagation"*
     - *"Schedule Python Practice from 2026-11-16T18:00:00 to 2026-11-16T19:30:00"*
     - *"What should I study for the Deep Learning Exam?"*
     - *"Save project kickoff summary to notes.txt"*
     - *"Read invoice.txt and email it to sam@accounting.example.com"*
   - NOVA runs the real tools behind the scenes and replies in **clear, everyday English** with
     helpful bullet-point highlights (`highlights`) instead of raw JSON dumps!

2. **Friendly Approval Cards (`resolve_approval`)**:
   - When an action needs your permission (like sending an email), NOVA explains *why* in plain
     English and attaches a simple `pending_approval` card. When you click **Approve** or **Deny**,
     NOVA replies in chat confirming what happened.

3. **Persistent SQLite Chat History (`nova_chat_history.db`)**:
   - Saves every user and assistant message in SQLite so your chat history stays intact even when
     you close the browser or restart the server.
"""

from __future__ import annotations

from datetime import datetime, timezone
import json
import re
import sqlite3
import uuid
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field

try:
    from pydantic import ConfigDict

    PYDANTIC_V2 = True
except ImportError:  # pragma: no cover
    PYDANTIC_V2 = False

from nova.api.app import StartTaskRequest, SubmitApprovalRequest, TaskStatusResponse
from nova.core.domain.primitives import TaskStatus
from nova.core.tools.personal_assistant_tools import (
    LocalCalendarStore,
    LocalNotesManager,
    LocalSystemMetricsProvider,
)

if TYPE_CHECKING:
    from nova.main import NovaRuntimeContainer

_ISO_RANGE_REGEX = re.compile(
    r"(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}(?::\d{2})?)\s*(?:to|-|until|->)\s*(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}(?::\d{2})?)",
    re.IGNORECASE,
)


class ChatMessage(BaseModel):
    """Single human-friendly message in NOVA's conversational chat stream."""

    if PYDANTIC_V2:
        model_config = ConfigDict(frozen=True, extra="forbid")
    else:

        class Config:
            allow_mutation = False
            extra = "forbid"

    message_id: str = Field(..., min_length=1)
    role: str = Field(..., min_length=1)
    text: str = Field(..., min_length=1)
    action_category: str = Field(default="general")
    highlights: Tuple[str, ...] = Field(default_factory=tuple)
    pending_approval: Optional[Dict[str, Any]] = None
    timestamp_utc: str = Field(..., min_length=1)


class PersonalAssistantChatSession:
    """Translates natural user messages into real NOVA actions and plain-English replies."""

    def __init__(self, container: "NovaRuntimeContainer") -> None:
        self.container = container
        self.calendar_store = container.calendar_store
        self.notes_manager = container.notes_manager
        self.metrics_provider = container.metrics_provider
        self._chat_db_path = self.container.data_dir / "nova_chat_history.db"
        self._init_chat_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self._chat_db_path))
        conn.row_factory = sqlite3.Row
        return conn

    def _init_chat_db(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS chat_messages (
                    seq_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    message_id TEXT NOT NULL,
                    role TEXT NOT NULL,
                    text TEXT NOT NULL,
                    action_category TEXT NOT NULL,
                    highlights_json TEXT NOT NULL,
                    pending_approval_json TEXT,
                    timestamp_utc TEXT NOT NULL
                )
                """
            )
            conn.commit()

    def _save_message(
        self,
        *,
        role: str,
        text: str,
        action_category: str,
        highlights: Tuple[str, ...] = (),
        pending_approval: Optional[Dict[str, Any]] = None,
    ) -> ChatMessage:
        msg = ChatMessage(
            message_id=f"msg_{uuid.uuid4().hex[:10]}",
            role=role,
            text=text,
            action_category=action_category,
            highlights=highlights,
            pending_approval=pending_approval,
            timestamp_utc=datetime.now(timezone.utc).isoformat(),
        )
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO chat_messages (
                    message_id, role, text, action_category,
                    highlights_json, pending_approval_json, timestamp_utc
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    msg.message_id,
                    msg.role,
                    msg.text,
                    msg.action_category,
                    json.dumps(list(msg.highlights)),
                    json.dumps(msg.pending_approval) if msg.pending_approval else None,
                    msg.timestamp_utc,
                ),
            )
            conn.commit()
        return msg

    def list_messages(self, limit: int = 50) -> List[ChatMessage]:
        """Return chronological chat messages stored in `nova_chat_history.db`."""
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT * FROM (
                    SELECT * FROM chat_messages ORDER BY seq_id DESC LIMIT ?
                ) ORDER BY seq_id ASC
                """,
                (int(limit),),
            ).fetchall()
        out: List[ChatMessage] = []
        for r in rows:
            pending_raw = r["pending_approval_json"]
            out.append(
                ChatMessage(
                    message_id=str(r["message_id"]),
                    role=str(r["role"]),
                    text=str(r["text"]),
                    action_category=str(r["action_category"]),
                    highlights=tuple(json.loads(str(r["highlights_json"]))),
                    pending_approval=json.loads(pending_raw) if pending_raw else None,
                    timestamp_utc=str(r["timestamp_utc"]),
                )
            )
        return out

    def send_message(self, user_text: str) -> ChatMessage:
        """Process a natural-language user message and return NOVA's plain-English response."""
        cleaned = user_text.strip()
        if not cleaned:
            raise ValueError("Message cannot be empty.")

        self._save_message(role="user", text=cleaned, action_category="user_input")
        lower = cleaned.lower()

        # 1. Smart Note Creation ("Note: Title — Body" or "Save note ...")
        if lower.startswith("note:") or lower.startswith("save note"):
            raw_note = cleaned.split(":", 1)[1].strip() if ":" in cleaned else cleaned
            if "—" in raw_note:
                title_part, body_part = raw_note.split("—", 1)
            elif "-" in raw_note:
                title_part, body_part = raw_note.split("-", 1)
            else:
                title_part, body_part = raw_note[:32], raw_note
            return self.submit_action("notes.create_note", {
                "title": title_part.strip() or "Quick Note", "body": body_part.strip() or cleaned})

        # 2. Calendar Event Scheduling ("Schedule <Title> from <ISO> to <ISO>")
        iso_match = _ISO_RANGE_REGEX.search(cleaned)
        if iso_match or lower.startswith("schedule "):
            if iso_match:
                start_iso, end_iso = iso_match.group(1), iso_match.group(2)
                title_prefix = cleaned[: iso_match.start()].strip()
                title = re.sub(
                    r"^(?:schedule|book|add event)\s+",
                    "",
                    title_prefix,
                    flags=re.IGNORECASE,
                ).removesuffix("from").strip() or "Scheduled Session"
            else:
                return self._save_message(role="assistant", text="Please provide a start and end time in local ISO format.",
                    action_category="clarification_required")
            return self.submit_action("calendar.create_event", {
                "title": title, "start_iso": start_iso, "end_iso": end_iso})

        # 3. Question / Local Knowledge Search ("What...", "How...", "Search...", or ends with "?")
        if (
            cleaned.endswith("?")
            or lower.startswith("what ")
            or lower.startswith("search ")
            or lower.startswith("find in notes")
        ) and "email" not in lower:
            rag_res = self.container.rag_engine.retrieve(cleaned, top_k=2)
            if rag_res.hits:
                top_chunk = rag_res.hits[0].chunk
                raw_text = getattr(
                    top_chunk,
                    "content",
                    getattr(top_chunk, "text", ""),
                ) or rag_res.format_observation_summary()
                snippet = str(raw_text).strip()
                return self._save_message(
                    role="assistant",
                    text=f"Here is what I found in your local notes and documents:\n\n{snippet}",
                    action_category="knowledge_search",
                    highlights=(
                        f"Source Document: {top_chunk.document_id}",
                        f"Matching Chunks Found: {len(rag_res.hits)}",
                    ),
                )
            return self._save_message(
                role="assistant",
                text=(
                    "I searched your local notes and knowledge base, but I didn't find any matching "
                    "notes yet. You can add one anytime by typing 'Note: Title — Your details'."
                ),
                action_category="knowledge_search",
                highlights=("Search Result: 0 matching local notes",),
            )

        # 4. Multi-Agent Task Execution (File writing, reading + emailing, etc.)
        task_id = f"task_chat_{uuid.uuid4().hex[:8]}"
        task_resp = self.container.api_service.start_task(
            StartTaskRequest(
                task_id=task_id,
                session_id="sess_personal_chat",
                user_goal=cleaned,
            )
        )

        return self._render_task_result(task_resp)

    def submit_action(self, tool_name: str, arguments: Dict[str, Any]) -> ChatMessage:
        response = self.container.api_service.start_assistant_action(tool_name, arguments)
        return self._render_task_result(response, tool_name)

    def _render_task_result(self, response: TaskStatusResponse, tool_name: Optional[str] = None) -> ChatMessage:
        if response.status == TaskStatus.AWAITING_APPROVAL and response.pending_approvals:
            card = response.pending_approvals[0]
            pending = {
                "task_id": response.task_id, "node_id": card.node_id,
                "proposal_id": card.proposal_id, "tool_name": card.tool_name,
                "agent_role": card.agent_role.value, "arguments": card.arguments,
                "expires_at": card.expires_at, "summary_label": card.tool_name,
                # Existing email clients can continue reading these optional fields.
                "recipient": card.arguments.get("to", ""), "subject": card.arguments.get("subject", ""),
                "body": card.arguments.get("body", ""),
            }
            return self._save_message(role="assistant", text="Please review the exact action and click Approve or Deny.",
                action_category="awaiting_approval", pending_approval=pending,
                highlights=(f"Task: {response.task_id}",))
        if response.status != TaskStatus.COMPLETED:
            return self._save_message(role="assistant",
                text=f"Action not completed: {response.failure_reason or response.status.value}",
                action_category="action_failed", highlights=(f"Task: {response.task_id}",))
        output = next(iter(response.node_outputs.values()), "")
        if tool_name in ("notes.create_note", "calendar.create_event"):
            result = json.loads(output)
            if tool_name == "notes.create_note":
                text = f"I've saved your note '{result['title']}'. Indexed chunks: {result['chunks_indexed']}."
                category = "note_saved"
            else:
                conflicts = result.get("conflicting_titles", [])
                suffix = "overlaps with " + ", ".join(repr(t) for t in conflicts) if conflicts else "no schedule conflicts"
                text = f"I've scheduled '{result['title']}': {suffix}."
                category = "event_scheduled"
        else:
            text, category = "Task completed. " + output, "task_completed"
        return self._save_message(role="assistant", text=text, action_category=category,
            highlights=(f"Task: {response.task_id}", "Status: Completed"))

    def resolve_approval(
        self,
        *,
        task_id: str,
        node_id: str,
        proposal_id: str,
        approved: bool,
    ) -> ChatMessage:
        """Apply the user's Approve or Deny click and return a plain-English confirmation message."""
        # Look up action type from persisted server state, not client-supplied arguments.
        before = self.container.api_service.get_task_status(task_id)
        matching = [c for c in before.pending_approvals if c.proposal_id == proposal_id and c.node_id == node_id]
        tool_name = matching[0].tool_name if matching else None
        res = self.container.api_service.submit_approval(task_id,
            SubmitApprovalRequest(node_id=node_id, proposal_id=proposal_id, approved=approved,
                                  reason="User decision from Personal Assistant Chat"))
        return self._render_task_result(res, tool_name)

    def get_assistant_workspace_summary(self) -> Dict[str, Any]:
        """Return a clean, human-friendly summary of chat messages, notes, calendar events, and health."""
        events = self.calendar_store.list_events()
        notes = self.notes_manager.list_notes()
        metrics = self.metrics_provider.get_metrics()
        messages = self.list_messages(limit=30)

        return {
            "messages": [
                {
                    "message_id": m.message_id,
                    "role": m.role,
                    "text": m.text,
                    "action_category": m.action_category,
                    "highlights": list(m.highlights),
                    "pending_approval": m.pending_approval,
                    "timestamp_utc": m.timestamp_utc,
                }
                for m in messages
            ],
            "calendar_events": [
                {
                    "event_id": e.event_id,
                    "title": e.title,
                    "start_iso": e.start_iso,
                    "end_iso": e.end_iso,
                    "description": e.description,
                }
                for e in events
            ],
            "notes": notes,
            "system_summary": {
                "disk_free_gb": metrics["disk_free_gb"],
                "workspace_files_count": metrics["workspace_files_count"],
                "notes_count": len(notes),
                "upcoming_events_count": len(events),
            },
        }


__all__ = [
    "ChatMessage",
    "PersonalAssistantChatSession",
]