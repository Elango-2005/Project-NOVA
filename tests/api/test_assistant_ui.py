"""Unit and integration tests for NOVA Phase 18: Clean, User-Friendly Personal AI Assistant Web App (`test_assistant_ui.py`).

Verifies:
1. `render_user_friendly_assistant_html()` renders a clean, human-understandable interface (`Chat with NOVA`,
   `My Upcoming Schedule`, `My Saved Notes`, `Recent Actions NOVA Took for You`, `Your Workspace at a Glance`)
   with zero raw JSON `<pre>` dumps.
2. `get_friendly_assistant_state(container)` translates technical tool names (`filesystem.write_file`,
   `gmail.send_email`) and decisions (`auto_approved`, `user_approved`) into plain-English activity
   descriptions (`"Created a file in your workspace"`, `"Safe action (Automatic)"`).
3. Saving a note, scheduling an event, and asking a question update `get_friendly_assistant_state`
   counts (`notes_count`, `upcoming_events_count`) and chat bubbles cleanly.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

from nova.api.assistant_ui import (
    get_friendly_assistant_state,
    render_user_friendly_assistant_html,
)
from nova.core.cognition.conversation_session import PersonalAssistantChatSession
from nova.main import NovaRuntimeContainer


class TestPhase18UserFriendlyAssistantUi:
    """Verification suite for Phase 18 (`services/backend/src/nova/api/assistant_ui.py`)."""

    def test_assistant_html_contains_friendly_sections_and_zero_raw_pre_json_dumps(
        self,
    ) -> None:
        """Ensures the new Personal Assistant HTML has clear everyday sections and no <pre> JSON dumps."""
        html = render_user_friendly_assistant_html(api_base_url="http://127.0.0.1:8765")

        assert "NOVA — Your Personal AI Assistant" in html
        assert "Chat with NOVA" in html
        assert "My Upcoming Schedule" in html
        assert "My Saved Notes" in html
        assert "Recent Actions NOVA Took for You" in html
        assert "Your Workspace at a Glance" in html
        assert "<pre" not in html  # Zero raw JSON dump blocks!

    def test_friendly_assistant_state_translates_technical_audit_logs_into_plain_english(
        self,
    ) -> None:
        """Verifies technical tool executions appear as human-readable activity items."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            container = NovaRuntimeContainer(
                data_dir=tmp_path / "data",
                workspace_dir=tmp_path / "workspace",
            )
            session = PersonalAssistantChatSession(container)
            session.send_message("Save project kickoff summary to notes.txt")

            state = get_friendly_assistant_state(container, chat_session=session)
            assert state["privacy_mode"] == "100% Offline & Private on Your Laptop"
            assert state["security_log_verified"] is True
            assert len(state["recent_activity"]) == 1
            act = state["recent_activity"][0]
            assert act["what_happened"] == "Created a file in your workspace"
            assert act["permission_note"] == "Safe action (Automatic)"
            assert act["status"] == "Success"

    def test_friendly_assistant_state_reflects_notes_calendar_and_chat_workflow(
        self,
    ) -> None:
        """Verifies notes, calendar events, and Q&A appear cleanly in get_friendly_assistant_state."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            container = NovaRuntimeContainer(
                data_dir=tmp_path / "data",
                workspace_dir=tmp_path / "workspace",
            )
            session = PersonalAssistantChatSession(container)

            session.send_message(
                "Note: Exam Syllabus — Neural networks, backpropagation, and Qwen 3.5 GGUF."
            )
            session.send_message(
                "Schedule Deep Learning Study from 2026-11-17T18:00:00 to 2026-11-17T19:30:00"
            )
            session.send_message("What should I study for backpropagation?")

            state = get_friendly_assistant_state(container, chat_session=session)
            assert state["system_summary"]["notes_count"] == 1
            assert state["system_summary"]["upcoming_events_count"] == 1
            assert len(state["messages"]) == 6  # 3 user prompts + 3 assistant replies
            assert state["notes"][0]["title"] == "Exam Syllabus"
            assert state["calendar_events"][0]["title"] == "Deep Learning Study"