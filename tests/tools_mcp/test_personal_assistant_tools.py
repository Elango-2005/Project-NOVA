"""Unit and integration tests for NOVA Phase 16: Real Everyday Personal Assistant Tools (`test_personal_assistant_tools.py`).

Verifies:
1. `LocalCalendarStore` persists events in SQLite (`nova_calendar.db`), lists them in chronological
   order, and automatically detects overlapping schedule conflicts (`find_conflicts`).
2. `LocalNotesManager` writes Markdown notes to `<workspace_dir>/notes/*.md` AND automatically
   indexes them into `HybridRetrievalEngine` (`nova_rag.db`) so they are immediately searchable!
3. `LocalSystemMetricsProvider` returns real laptop disk free GB, CPU logical cores, and workspace
   file counts.
4. `build_personal_assistant_tool_handlers` executes `calendar.create_event`, `calendar.list_events`,
   `notes.create_note`, and `system.get_metrics` and satisfies `ToolRegistry` output contracts!
"""

from __future__ import annotations

import tempfile
from pathlib import Path

from nova.core.cognition.retrieval import (
    HybridRetrievalEngine,
    SqliteHybridChunkIndex,
)
from nova.core.tools.personal_assistant_tools import (
    LocalCalendarStore,
    LocalNotesManager,
    LocalSystemMetricsProvider,
    build_personal_assistant_tool_handlers,
)
from nova.main import NovaRuntimeContainer


class TestPhase16PersonalAssistantTools:
    """Verification suite for Phase 16 (`services/backend/src/nova/core/tools/personal_assistant_tools.py`)."""

    def test_local_calendar_store_persists_events_and_detects_overlapping_conflicts(
        self,
    ) -> None:
        """Creates events in nova_calendar.db and verifies automatic overlap conflict detection."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            cal = LocalCalendarStore(Path(tmp_dir) / "nova_calendar.db")

            ev1, conflicts1 = cal.create_event(
                title="Deep Learning Lecture",
                start_iso="2026-11-15T10:00:00",
                end_iso="2026-11-15T11:30:00",
                description="CNN and RNN architectures",
            )
            assert ev1.event_id.startswith("evt_")
            assert conflicts1 == []

            # Schedule a second event that overlaps (11:00 to 12:00 overlaps 10:00 to 11:30!)
            ev2, conflicts2 = cal.create_event(
                title="Project Team Sync",
                start_iso="2026-11-15T11:00:00",
                end_iso="2026-11-15T12:00:00",
            )
            assert len(conflicts2) == 1
            assert conflicts2[0].title == "Deep Learning Lecture"

            all_events = cal.list_events()
            assert [e.title for e in all_events] == [
                "Deep Learning Lecture",
                "Project Team Sync",
            ]

    def test_local_notes_manager_saves_markdown_and_auto_indexes_into_hybrid_rag(
        self,
    ) -> None:
        """Creates a personal note and confirms it is immediately searchable via Hybrid RAG."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            rag = HybridRetrievalEngine(
                index=SqliteHybridChunkIndex(tmp_path / "nova_rag.db")
            )
            notes = LocalNotesManager(workspace_dir=tmp_path / "workspace", rag_engine=rag)

            res = notes.create_note(
                title="Exam Preparation Checklist",
                body="Study backpropagation, attention mechanisms, and Qwen 3.5 GGUF quantization.",
            )
            assert res["note_id"] == "note_exam_preparation_checklist"
            assert res["chunks_indexed"] == 1
            assert Path(res["file_path"]).exists()

            # Verify the note is immediately retrievable via Hybrid RAG!
            rag_res = rag.retrieve("What should I study for Qwen 3.5 quantization?", top_k=2)
            assert len(rag_res.hits) >= 1
            assert "backpropagation" in rag_res.format_observation_summary()

    def test_system_metrics_provider_reports_real_disk_and_cpu_stats(self) -> None:
        """Verifies LocalSystemMetricsProvider returns positive disk GB and CPU core counts."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            ws = Path(tmp_dir) / "workspace"
            ws.mkdir(parents=True, exist_ok=True)
            (ws / "sample.txt").write_text("hello", encoding="utf-8")

            provider = LocalSystemMetricsProvider(workspace_dir=ws)
            metrics = provider.get_metrics()
            assert metrics["disk_total_gb"] > 0
            assert metrics["disk_free_gb"] >= 0
            assert metrics["cpu_logical_cores"] >= 1
            assert metrics["workspace_files_count"] == 1

    def test_personal_assistant_handlers_satisfy_mcp_registry_output_contracts(
        self,
    ) -> None:
        """Verifies all 4 personal assistant handlers execute cleanly and return expected keys."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            container = NovaRuntimeContainer(
                data_dir=tmp_path / "data",
                workspace_dir=tmp_path / "workspace",
            )
            handlers, _, _, _ = build_personal_assistant_tool_handlers(
                data_dir=container.data_dir,
                workspace_dir=container.workspace_dir,
                rag_engine=container.rag_engine,
            )

            # 1. `calendar.create_event`
            create_out = handlers["calendar.create_event"](
                {
                    "title": "Python Algorithms Practice",
                    "start_iso": "2026-11-16T18:00:00",
                    "end_iso": "2026-11-16T19:30:00",
                }
            )
            cal_create_contract = container.tool_registry.get_tool("calendar.create_event")
            for req_key in cal_create_contract.required_output_keys:
                assert req_key in create_out

            # 2. `calendar.list_events`
            list_out = handlers["calendar.list_events"]({})
            cal_list_contract = container.tool_registry.get_tool("calendar.list_events")
            for req_key in cal_list_contract.required_output_keys:
                assert req_key in list_out
            assert len(list_out["events"]) == 1

            # 3. `notes.create_note`
            note_out = handlers["notes.create_note"](
                {
                    "title": "Dynamic Programming Tips",
                    "body": "Always define the subproblem state and base cases clearly.",
                }
            )
            assert note_out["note_id"] == "note_dynamic_programming_tips"
            assert note_out["chunks_indexed"] == 1

            # 4. `system.get_metrics`
            metrics_out = handlers["system.get_metrics"]({})
            assert "cpu_pct" in metrics_out
            assert "ram_pct" in metrics_out
            assert "disk_free_gb" in metrics_out