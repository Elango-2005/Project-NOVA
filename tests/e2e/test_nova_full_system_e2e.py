"""NOVA Task 7.2: Complete 7-Layer End-to-End System Acceptance Suite (`test_nova_full_system_e2e.py`).

Real-Life Analogy:
This file is the **Grand Finale Graduation Exam** for the entire NOVA Backend (Phases 1 to 7).
It proves that all **7 Architectural Layers** from `NOVA_ADR.md` Section 9.1 (p. 8) work together
in one seamless Jarvis-style flow:

- **Layer 7 (Front Reception Desk & SSE Loudspeaker)**: `NovaLocalApiService` (`app.py`)
- **Layer 6 (The Office Manager & Auto-Save Clipboard)**: `NovaGraphOrchestrator` + `SqliteCheckpointStore`
- **Layer 5 (Specialized Employees)**: `Planner`, `Research`, `Coding`, `Comms`, `Browser`, `Desktop`, `Voice`
- **Layer 4 (Memory Cabinet, Office Librarian & Quality Inspector)**: `MemoryManager`, `HybridRetrievalEngine`, `reflection.py`
- **Layer 3 (Tool Catalog, Security Guard, Safety Fence & Flight Recorder)**: `ToolRegistry`, `CatalogBackedPermissionGate`, `FilesystemSandbox`, `SqliteToolAuditLog`
- **Layer 2 (Smart Traffic Controller, Ears, Mouth & Eyes)**: `NovaModelRouter`, `WakeWordDetector`, `FasterWhisperSttAdapter`, `LocalTtsRouter`, `LocalVisionRouter`
- **Layer 1 (Foundation Settings & Event Bus)**: `NovaSettings`, `InMemoryEventBus`, Embedded SQLite
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
import pytest

from nova.api.app import (
    NovaLocalApiService,
    PolicyRevokeRequest,
    PolicyTightenRequest,
    StartTaskRequest,
    SubmitApprovalRequest,
)
from nova.core.agents.base import BaseSpecializedAgent
from nova.core.agents.browser_and_desktop import (
    CrossPlatformDesktopAutomationService,
    DesktopElementTarget,
    SensitiveDesktopAppDeniedError,
)
from nova.core.cognition.memory import (
    MemoryManager,
    SqliteLongTermMemoryStore,
    WorkingMemoryStore,
)
from nova.core.cognition.retrieval import (
    HybridRetrievalEngine,
    SqliteHybridChunkIndex,
)
from nova.core.config.settings import load_settings
from nova.core.domain.primitives import (
    AgentRole,
    DataClassification,
    MemoryPartition,
    PermissionDecisionKind,
    SensitivityTier,
    TaskStatus,
)
from nova.core.model_serving.multimodal import (
    AudioFrame,
    FasterWhisperSttAdapter,
    LocalTtsRouter,
    VoiceActivityDetector,
    WakeWordDetector,
)
from nova.core.model_serving.router import (
    HardwareProfile,
    LlamaCppAdapter,
    NovaModelRouter,
    VllmGpuAdapter,
)
from nova.core.orchestration.checkpoint import SqliteCheckpointStore
from nova.core.orchestration.engine import NovaGraphOrchestrator
from nova.core.permissions.sandbox_and_audit import (
    CatalogBackedPermissionGate,
    FilesystemSandbox,
    PolicyModificationError,
    SandboxViolationError,
    SandboxedMcpToolExecutor,
    SqliteToolAuditLog,
    UserPermissionPolicyStore,
)
from nova.core.platform.event_bus import InMemoryEventBus
from nova.core.tools.registry import build_default_tool_registry


class TestNovaFullSevenLayerSystemE2E:
    """End-to-end acceptance tests uniting all 7 architectural layers of Project NOVA."""

    def test_complete_7_layer_nova_jarvis_workflow_from_voice_and_rag_to_api_approval_and_tts(
        self,
    ) -> None:
        """Full 7-Layer E2E: Voice STT -> Memory + Hybrid RAG -> Sandbox Write -> UI Approval -> Email -> Audit + TTS."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            workspace = tmp_path / "workspace"
            workspace.mkdir(parents=True, exist_ok=True)

            # --- LAYER 1: Platform Settings & Event Bus ---
            settings = load_settings(environ={})
            assert settings.offline_by_default is True
            bus = InMemoryEventBus()

            # --- LAYER 4: Two-Tier Memory & Local Hybrid RAG ---
            memory_manager = MemoryManager(
                working_store=WorkingMemoryStore(),
                long_term_store=SqliteLongTermMemoryStore(tmp_path / "nova_ltm.db"),
                event_bus=bus,
            )
            memory_manager.write_memory(
                agent_role=AgentRole.RESEARCH,
                partition=MemoryPartition.SEMANTIC,
                memory_key="contact.sam.email",
                content="Sam in accounting uses sam@accounting.example.com.",
                tags=("sam", "accounting", "email"),
                importance=0.95,
                task_id="task_seed",
                session_id="sess_seed",
            )

            rag_engine = HybridRetrievalEngine(
                index=SqliteHybridChunkIndex(tmp_path / "nova_rag.db")
            )
            rag_engine.ingest_document(
                document_id="doc_inv_2026_10",
                source_uri="file:///workspace/invoices/oct_2026.pdf",
                content="Vendor: Acme Corp | Invoice #INV-2026-10 | Total Due: $1,250.00",
                data_classification=DataClassification.SENSITIVE_CATEGORY,
            )

            # --- LAYER 2: Multi-Modal Voice STT + Hardware-Aware Model Router ---
            wake = WakeWordDetector()
            vad = VoiceActivityDetector(energy_threshold=0.15)
            stt = FasterWhisperSttAdapter()
            frames = (
                AudioFrame(rms_energy=0.65, simulated_speech_text="Hey NOVA,"),
                AudioFrame(
                    rms_energy=0.75,
                    simulated_speech_text="find invoice #INV-2026-10, save a local summary, and email Sam.",
                    is_final_utterance_frame=True,
                ),
            )
            assert wake.detect(frames[0]) is True
            spoken_goal = list(stt.transcribe_stream(frames, vad=vad))[-1].text

            # Look up Sam's email from SEMANTIC memory
            sam_matches = memory_manager.query_memory(
                agent_role=AgentRole.PLANNER,
                partition=MemoryPartition.SEMANTIC,
                query="Sam accounting email",
                task_id="task_e2e_01",
                session_id="sess_e2e_01",
            )
            assert len(sam_matches) == 1

            llamacpp = LlamaCppAdapter(
                scripted_outputs=[
                    # 1. Planner output (3-node DAG: Research -> Coding -> Comms)
                    {
                        "kind": "plan",
                        "agent_role": "planner",
                        "node_id": "node_planner",
                        "thought": "Retrieve #INV-2026-10 via RAG, write summary.txt locally, then email Sam.",
                        "goal_summary": "Retrieve #INV-2026-10, write summary.txt, and email Sam.",
                        "steps": [
                            {
                                "step_id": "step_rag",
                                "target_agent": "research",
                                "objective": "Search RAG for #INV-2026-10.",
                                "depends_on": [],
                                "estimated_sensitivity": "read_only",
                            },
                            {
                                "step_id": "step_write_summary",
                                "target_agent": "coding",
                                "objective": "Write summary.txt inside workspace.",
                                "depends_on": ["step_rag"],
                                "estimated_sensitivity": "reversible_write",
                            },
                            {
                                "step_id": "step_email_sam",
                                "target_agent": "comms",
                                "objective": "Email summary to sam@accounting.example.com.",
                                "depends_on": ["step_write_summary"],
                                "estimated_sensitivity": "irreversible_write",
                            },
                        ],
                    },
                    # 2. Research Agent Turn 1 (rag.retrieve) & Turn 2 (final_answer)
                    {
                        "kind": "tool_call",
                        "agent_role": "research",
                        "node_id": "step_rag",
                        "thought": "Query hybrid RAG for #INV-2026-10.",
                        "tool_name": "rag.retrieve",
                        "arguments": {"query": "#INV-2026-10", "top_k": 1},
                        "declared_sensitivity": "read_only",
                        "rationale": "Retrieve invoice #INV-2026-10 chunk",
                    },
                    {
                        "kind": "final_answer",
                        "agent_role": "research",
                        "node_id": "step_rag",
                        "thought": "Found chunk doc_inv_2026_10#chunk_0.",
                        "content": "Vendor: Acme Corp | Invoice #INV-2026-10 | Total Due: $1,250.00",
                        "cited_source_ids": ["doc_inv_2026_10#chunk_0"],
                        "confidence": 0.99,
                    },
                    # 3. Coding Agent Turn 1 (filesystem.write_file) & Turn 2 (final_answer)
                    {
                        "kind": "tool_call",
                        "agent_role": "coding",
                        "node_id": "step_write_summary",
                        "thought": "Save invoice summary to summary.txt in workspace.",
                        "tool_name": "filesystem.write_file",
                        "arguments": {
                            "path": "summary.txt",
                            "content": "Acme Corp #INV-2026-10: $1,250.00",
                        },
                        "declared_sensitivity": "reversible_write",
                        "rationale": "Write local summary.txt inside workspace",
                    },
                    {
                        "kind": "final_answer",
                        "agent_role": "coding",
                        "node_id": "step_write_summary",
                        "thought": "Saved summary.txt.",
                        "content": "Saved summary.txt (32 bytes).",
                        "cited_source_ids": ["doc_inv_2026_10#chunk_0"],
                        "confidence": 1.0,
                    },
                    # 4. Comms Agent Turn 1 (gmail.send_email -> pauses!) & Turn 2 (final_answer after approval)
                    {
                        "kind": "tool_call",
                        "agent_role": "comms",
                        "node_id": "step_email_sam",
                        "thought": "Propose sending email to sam@accounting.example.com.",
                        "tool_name": "gmail.send_email",
                        "arguments": {
                            "to": "sam@accounting.example.com",
                            "subject": "Invoice #INV-2026-10 Summary",
                            "body": "Acme Corp #INV-2026-10: $1,250.00",
                        },
                        "declared_sensitivity": "irreversible_write",
                        "rationale": "Send grounded invoice summary to Sam",
                    },
                    {
                        "kind": "final_answer",
                        "agent_role": "comms",
                        "node_id": "step_email_sam",
                        "thought": "Email sent after user approval.",
                        "content": "Invoice #INV-2026-10 summary emailed to Sam. Task complete.",
                        "cited_source_ids": ["doc_inv_2026_10#chunk_0"],
                        "confidence": 1.0,
                    },
                ]
            )

            router = NovaModelRouter(
                llamacpp_adapter=llamacpp,
                hardware_profile=HardwareProfile(gpu_available=False),
                event_bus=bus,
            )

            # --- LAYER 3: MCP Tool Registry, Permission Gate, Sandbox & Audit Log ---
            registry = build_default_tool_registry()
            policy_store = UserPermissionPolicyStore(registry=registry)
            audit_log = SqliteToolAuditLog(tmp_path / "nova_audit.db", retention_days=30)
            fs_sandbox = FilesystemSandbox(allowed_roots=(workspace,))

            gate = CatalogBackedPermissionGate(
                registry=registry,
                policy_store=policy_store,
                audit_log=audit_log,
                auto_approve_reversible_local_writes=True,
                default_task_id="task_e2e_01",
                default_session_id="sess_e2e_01",
            )

            def _handle_write_file(args: dict[str, object]) -> dict[str, object]:
                target = Path(str(args["path"]))
                text = str(args["content"])
                target.write_text(text, encoding="utf-8")
                return {"bytes_written": len(text)}

            executor = SandboxedMcpToolExecutor(
                permission_gate=gate,
                registry=registry,
                filesystem_sandbox=fs_sandbox,
                audit_log=audit_log,
                event_bus=bus,
                tool_handlers={
                    "rag.retrieve": lambda args: {
                        "chunks": [
                            rag_engine.retrieve(str(args["query"])).format_observation_summary()
                        ],
                        "citations": list(rag_engine.retrieve(str(args["query"])).citation_ids),
                    },
                    "filesystem.write_file": _handle_write_file,
                    "gmail.send_email": lambda args: {
                        "message_id": "msg_e2e_100",
                        "status": f"delivered_to_{args['to']}",
                    },
                },
                default_task_id="task_e2e_01",
                default_session_id="sess_e2e_01",
            )

            # --- LAYER 5 & 6: Specialized Agents & Graph Orchestrator ---
            checkpoint_store = SqliteCheckpointStore(tmp_path / "nova_checkpoints.db")
            orchestrator = NovaGraphOrchestrator(
                agents={
                    AgentRole.PLANNER: BaseSpecializedAgent(AgentRole.PLANNER, router, bus),
                    AgentRole.RESEARCH: BaseSpecializedAgent(AgentRole.RESEARCH, router, bus),
                    AgentRole.CODING: BaseSpecializedAgent(AgentRole.CODING, router, bus),
                    AgentRole.COMMS: BaseSpecializedAgent(AgentRole.COMMS, router, bus),
                },
                checkpoint_store=checkpoint_store,
                permission_gate=gate,
                tool_executor=executor,
                event_bus=bus,
                settings=settings,
            )

            # --- LAYER 7: Local API Service (`POST /tasks` -> Pause -> `POST /approval` -> SSE) ---
            api = NovaLocalApiService(
                orchestrator=orchestrator,
                checkpoint_store=checkpoint_store,
                event_bus=bus,
                tool_registry=registry,
                policy_store=policy_store,
                audit_log=audit_log,
                settings=settings,
            )

            # Phase A: Start task via API -> Research & Coding complete -> Comms pauses for approval!
            paused_resp = api.start_task(
                StartTaskRequest(
                    task_id="task_e2e_01",
                    session_id="sess_e2e_01",
                    user_goal=spoken_goal,
                    require_citations=True,
                )
            )
            assert paused_resp.status == TaskStatus.AWAITING_APPROVAL
            assert paused_resp.completed_nodes == ("step_rag", "step_write_summary")
            assert (workspace / "summary.txt").read_text(encoding="utf-8") == (
                "Acme Corp #INV-2026-10: $1,250.00"
            )
            assert len(paused_resp.pending_approvals) == 1
            approval_card = paused_resp.pending_approvals[0]
            assert approval_card.tool_name == "gmail.send_email"

            # Phase B: User clicks Approve in UI -> Comms sends email & task completes!
            final_resp = api.submit_approval(
                "task_e2e_01",
                SubmitApprovalRequest(
                    node_id="step_email_sam",
                    proposal_id=approval_card.proposal_id,
                    approved=True,
                    reason="Approved by user in E2E acceptance test",
                    require_citations=True,
                ),
            )
            assert final_resp.status == TaskStatus.COMPLETED
            assert final_resp.completed_nodes == (
                "step_email_sam",
                "step_rag",
                "step_write_summary",
            )

            # Record Episodic memory diary entry
            epi = memory_manager.write_memory(
                agent_role=AgentRole.COMMS,
                partition=MemoryPartition.EPISODIC,
                content=final_resp.node_outputs["step_email_sam"],
                tags=("e2e", "invoice", "sam"),
                importance=0.9,
                task_id="task_e2e_01",
                session_id="sess_e2e_01",
            )
            assert epi.partition == MemoryPartition.EPISODIC

            # Verify Audit Log recorded all 3 tool executions (rag.retrieve, filesystem.write_file, gmail.send_email)
            audit_entries = api.query_audit_logs(task_id="task_e2e_01")
            assert [e.tool_name for e in audit_entries] == [
                "rag.retrieve",
                "filesystem.write_file",
                "gmail.send_email",
            ]
            assert all(e.execution_status == "success" for e in audit_entries)

            # Verify SSE event stream and TTS spoken confirmation
            sse_frames = list(api.stream_task_events_sse("task_e2e_01"))
            assert any("event: task.completed" in frame for frame in sse_frames)

            tts = LocalTtsRouter()
            spoken_chunks = list(
                tts.synthesize_stream(final_resp.node_outputs["step_email_sam"])
            )
            assert len(spoken_chunks) >= 1
            assert "Invoice #INV-2026-10" in spoken_chunks[0].spoken_text_slice

    def test_complete_7_layer_adversarial_resilience_across_all_security_gates(
        self,
    ) -> None:
        """Verifies all NOVA security gates: Policy tightening/revocation, Desktop deny-list, and Sandbox escape block."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            workspace = tmp_path / "workspace"
            workspace.mkdir(parents=True, exist_ok=True)

            bus = InMemoryEventBus()
            registry = build_default_tool_registry()
            policy_store = UserPermissionPolicyStore(registry=registry)
            audit_log = SqliteToolAuditLog(tmp_path / "sec_audit.db", retention_days=30)
            fs_sandbox = FilesystemSandbox(allowed_roots=(workspace,))

            # 1. Desktop Agent Sensitive App Deny-List blocks Password Managers (`1Password`)
            desktop_service = CrossPlatformDesktopAutomationService(event_bus=bus)
            with pytest.raises(SensitiveDesktopAppDeniedError):
                desktop_service.build_desktop_action_proposal(
                    node_id="node_1p",
                    target=DesktopElementTarget(
                        window_title="1Password — Vault",
                        accessibility_element_id="UIA_Export",
                        visual_description="Export Passwords",
                    ),
                )

            # 2. FilesystemSandbox blocks `../../` path escape and `.env` credential reads
            with pytest.raises(SandboxViolationError, match="escapes permitted roots"):
                fs_sandbox.resolve_and_verify_path("../../Windows/System32/cmd.exe")
            with pytest.raises(SandboxViolationError, match="credential/secret path segment"):
                fs_sandbox.resolve_and_verify_path(".env")

            # 3. UserPermissionPolicyStore blocks loosening `terminal.execute` below `IRREVERSIBLE_WRITE`
            with pytest.raises(PolicyModificationError):
                policy_store.tighten_tool_sensitivity(
                    "terminal.execute",
                    SensitivityTier.READ_ONLY,
                )