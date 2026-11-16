"""Unit and integration tests for NOVA Phase 17: Human-Readable Conversational Chat & Auto-Memory Loop (`test_conversation_session.py`).

Verifies:
1. `PersonalAssistantChatSession.send_message("Note: ...")` saves a Markdown note, auto-indexes it
   into Hybrid RAG, and answers a natural-language question (`"What should I study...?"`) in plain
   English using the saved note!
2. `PersonalAssistantChatSession.send_message("Schedule ...")` creates real calendar events in
   `nova_calendar.db`, detects overlapping schedule conflicts, and explains them in plain English.
3. `PersonalAssistantChatSession.send_message("Read invoice.txt and email it to ...")` returns a
   plain-English explanation with a `pending_approval` card, and `resolve_approval(approved=True)`
   completes the task with a friendly confirmation message.
4. `get_assistant_workspace_summary()` persists chat messages in `nova_chat_history.db` and returns
   a unified summary of messages, calendar events, notes, and disk/workspace stats.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

from nova.core.cognition.conversation_session import PersonalAssistantChatSession
from nova.main import NovaRuntimeContainer


class TestPhase17ConversationSession:
    """Verification suite for Phase 17 (`services/backend/src/nova/core/cognition/conversation_session.py`)."""

    def test_chat_session_saves_smart_note_and_answers_question_in_plain_english(
        self,
    ) -> None:
        """Saves a note via natural chat and immediately answers a question about it."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            container = NovaRuntimeContainer(
                data_dir=tmp_path / "data",
                workspace_dir=tmp_path / "workspace",
            )
            session = PersonalAssistantChatSession(container)

            note_reply = session.send_message(
                "Note: Deep Learning Exam — Study CNNs, attention mechanisms, and backpropagation."
            )
            assert note_reply.role == "assistant"
            assert note_reply.action_category == "note_saved"
            assert "Deep Learning Exam" in note_reply.text

            # Now ask a natural question in chat!
            qa_reply = session.send_message("What should I study for backpropagation?")
            assert qa_reply.action_category == "knowledge_search"
            assert "attention mechanisms" in qa_reply.text

    def test_chat_session_schedules_events_and_warns_about_conflicts_in_plain_english(
        self,
    ) -> None:
        """Schedules two overlapping sessions via natural chat and verifies plain-English warning."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            container = NovaRuntimeContainer(
                data_dir=tmp_path / "data",
                workspace_dir=tmp_path / "workspace",
            )
            session = PersonalAssistantChatSession(container)

            first = session.send_message(
                "Schedule Python Practice from 2026-11-16T18:00:00 to 2026-11-16T19:30:00"
            )
            assert first.action_category == "event_scheduled"
            assert "no schedule conflicts" in first.text

            second = session.send_message(
                "Schedule AI Lab Review from 2026-11-16T19:00:00 to 2026-11-16T20:00:00"
            )
            assert second.action_category == "event_scheduled"
            assert "overlaps with 'Python Practice'" in second.text

    def test_chat_session_handles_email_approval_flow_in_plain_english(self) -> None:
        """Runs a file-read + email request via chat, pauses with a friendly card, and approves."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            container = NovaRuntimeContainer(
                data_dir=tmp_path / "data",
                workspace_dir=tmp_path / "workspace",
            )
            (container.workspace_dir / "invoice.txt").write_text(
                "Total due: $1,250.00",
                encoding="utf-8",
            )
            session = PersonalAssistantChatSession(container)

            paused_msg = session.send_message(
                "Read invoice.txt and email it to sam@accounting.example.com"
            )
            assert paused_msg.action_category == "awaiting_approval"
            assert paused_msg.pending_approval is not None
            assert paused_msg.pending_approval["recipient"] == "sam@accounting.example.com"
            assert "click Approve or Deny" in paused_msg.text

            approved_msg = session.resolve_approval(
                task_id=paused_msg.pending_approval["task_id"],
                node_id=paused_msg.pending_approval["node_id"],
                proposal_id=paused_msg.pending_approval["proposal_id"],
                approved=True,
            )
            assert approved_msg.action_category == "task_completed"
            assert "Task completed." in approved_msg.text

    def test_workspace_summary_persists_chat_history_and_counts(self) -> None:
        """Verifies get_assistant_workspace_summary returns persisted messages, notes, and events."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            container = NovaRuntimeContainer(
                data_dir=tmp_path / "data",
                workspace_dir=tmp_path / "workspace",
            )
            session = PersonalAssistantChatSession(container)
            session.send_message("Note: Project Idea — Build offline AI assistant")
            session.send_message(
                "Schedule Team Demo from 2026-11-18T14:00:00 to 2026-11-18T15:00:00"
            )

            summary = session.get_assistant_workspace_summary()
            assert len(summary["messages"]) == 4  # 2 user messages + 2 assistant replies
            assert summary["system_summary"]["notes_count"] == 1
            assert summary["system_summary"]["upcoming_events_count"] == 1