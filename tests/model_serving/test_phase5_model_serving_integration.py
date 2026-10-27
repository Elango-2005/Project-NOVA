"""NOVA Task 5.3: Phase 5 End-to-End Model Router + Multi-Modal (Voice & Vision) + Orchestrator Integration.

Real-Life Analogy:
This test file is the **Final Exam for Phase 5 (Layer 2 — The Brain, Ears, Mouth & Eyes)**.
It connects everything we built in Phase 5 directly into our **Office Manager (`NovaGraphOrchestrator`)**:

1. `test_end_to_end_voice_to_vision_ocr_to_orchestrator_with_mid_task_hardware_switch`:
   - **Ears (`WakeWordDetector` + `VoiceActivityDetector` + `FasterWhisperSttAdapter`)**:
     You speak into the microphone: *"Hey NOVA, summarize the scanned October invoice."*
   - **Eyes (`LocalVisionRouter` — Qwen-VL)**:
     Reads the scanned invoice image (`oct_2026_scan.png`) via OCR and wraps it in a safe
     `<untrusted_data>` envelope (`UntrustedDataItem`).
   - **Brain (`NovaModelRouter`) with Mid-Task Laptop Unplugging (`Phase 10`, p. 42)**:
     * While your laptop is plugged in with a 12GB GPU, the `Planner` agent runs on **`vLLM`**.
     * Halfway through the task, you unplug your laptop (`on_battery_power=True`).
     * Without dropping a single step, `NovaModelRouter` automatically switches the `Research`
       agent to **`llama.cpp`** (CPU mode) and finishes the task!
   - **Mouth (`LocalTtsRouter`)**:
     Streams the final answer out loud using **`Piper`** (battery-friendly local TTS).

2. `test_privacy_floor_keeps_sensitive_financial_task_local_even_when_cloud_enabled`:
   - Proves `ADR-001`, `Section 12.2`, and `Section 16.2`: when a task touches a `SENSITIVE_CATEGORY`
     financial document without explicit sensitive-cloud consent, `BaseSpecializedAgent` and
     `NovaModelRouter` keep 100% of inference on your local device (`0` calls to `OptInCloudAdapter`).
"""

from __future__ import annotations

import tempfile
from pathlib import Path

from nova.core.agents.base import BaseSpecializedAgent
from nova.core.config.settings import load_settings
from nova.core.domain.primitives import (
    AgentRole,
    DataClassification,
    TaskStatus,
)
from nova.core.model_serving.multimodal import (
    AudioFrame,
    FasterWhisperSttAdapter,
    LocalTtsRouter,
    LocalVisionRouter,
    TtsEngineKind,
    VisionEngineKind,
    VoiceActivityDetector,
    WakeWordDetector,
)
from nova.core.model_serving.router import (
    HardwareProfile,
    LlamaCppAdapter,
    NovaModelRouter,
    OptInCloudAdapter,
    ServingEngineKind,
    VllmGpuAdapter,
)
from nova.core.orchestration.checkpoint import SqliteCheckpointStore
from nova.core.orchestration.engine import NovaGraphOrchestrator
from nova.core.orchestration.ports import (
    DefaultPolicyPermissionGate,
    InMemoryMockToolExecutor,
)
from nova.core.platform.event_bus import InMemoryEventBus


class TestPhase5ModelServingAndMultimodalIntegration:
    """End-to-end integration tests uniting Voice, Vision, Model Router, and NovaGraphOrchestrator."""

    def test_end_to_end_voice_to_vision_ocr_to_orchestrator_with_mid_task_hardware_switch(
        self,
    ) -> None:
        """Full multi-modal flow: Voice STT -> Qwen-VL OCR -> vLLM Plan -> Laptop Unplugged -> llama.cpp Research -> Piper TTS."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            bus = InMemoryEventBus()

            # 1. Start on a plugged-in workstation with 12GB GPU VRAM and 16GB RAM
            initial_hw = HardwareProfile(
                gpu_available=True,
                gpu_vram_gb=12.0,
                system_ram_gb=16.0,
                is_network_available=False,
                on_battery_power=False,
            )

            # 2. NOVA's Ears: Wake-Word + VAD + Faster-Whisper Streaming STT
            wake = WakeWordDetector()
            vad = VoiceActivityDetector(energy_threshold=0.15)
            stt = FasterWhisperSttAdapter()

            mic_frames = (
                AudioFrame(rms_energy=0.02, simulated_speech_text="[silence]"),
                AudioFrame(rms_energy=0.60, simulated_speech_text="Hey NOVA,"),
                AudioFrame(
                    rms_energy=0.72,
                    simulated_speech_text="summarize the scanned October invoice.",
                    is_final_utterance_frame=True,
                ),
            )
            assert wake.detect(mic_frames[1]) is True
            stt_segments = list(stt.transcribe_stream(mic_frames, vad=vad))
            spoken_goal = stt_segments[-1].text
            assert spoken_goal == "Hey NOVA, summarize the scanned October invoice."

            # 3. NOVA's Eyes: Qwen-VL OCR extracts text from the scanned invoice image
            vision_router = LocalVisionRouter(hardware_profile=initial_hw)
            assert vision_router.select_vision_engine() == VisionEngineKind.QWEN_VL
            vision_router.register_mock_ocr(
                "file:///scans/oct_2026_scan.png",
                "Vendor: Acme Corp | Invoice #INV-2026-10 | Total Due: $1,250.00",
            )
            ocr_data_item = vision_router.extract_ocr_as_untrusted_data(
                item_id="scan_oct_2026",
                image_uri="file:///scans/oct_2026_scan.png",
                data_classification=DataClassification.SENSITIVE_CATEGORY,
            )

            # 4. NOVA's Brain: Model Router with vLLM (GPU) and llama.cpp (CPU universal floor)
            vllm_adapter = VllmGpuAdapter(
                scripted_outputs=[
                    {
                        "kind": "plan",
                        "agent_role": "planner",
                        "node_id": "node_planner",
                        "thought": "Single research step to summarize the OCR-extracted invoice.",
                        "goal_summary": "Summarize scan_oct_2026.",
                        "steps": [
                            {
                                "step_id": "step_summarize_scan",
                                "target_agent": "research",
                                "objective": "Summarize Vendor, Invoice #, and Total Due from scan_oct_2026.",
                                "depends_on": [],
                                "estimated_sensitivity": "read_only",
                            }
                        ],
                    }
                ]
            )

            llamacpp_adapter = LlamaCppAdapter(
                scripted_outputs=[
                    {
                        "kind": "final_answer",
                        "agent_role": "research",
                        "node_id": "step_summarize_scan",
                        "thought": "Extracted total from scan_oct_2026 on CPU after battery switch.",
                        "content": "Acme Corp Invoice #INV-2026-10 Total Due is $1,250.00. Speaking complete.",
                        "cited_source_ids": ["scan_oct_2026"],
                        "confidence": 0.99,
                    }
                ]
            )

            model_router = NovaModelRouter(
                llamacpp_adapter=llamacpp_adapter,
                vllm_adapter=vllm_adapter,
                hardware_profile=initial_hw,
                event_bus=bus,
            )

            # Simulate unplugging the laptop onto battery power right after the Planner runs on vLLM!
            # We do this by wrapping `vllm_adapter.generate_structured` or updating hardware profile
            # via an event subscriber when `TASK_PLANNING` / `AGENT_COMPLETED` fires for the Planner!
            tts_router = LocalTtsRouter(hardware_profile=initial_hw)

            def _on_planner_done(event: object) -> None:
                from nova.core.domain.events import NovaEvent

                if isinstance(event, NovaEvent) and event.agent_role == AgentRole.PLANNER:
                    battery_hw = HardwareProfile(
                        gpu_available=True,
                        gpu_vram_gb=12.0,
                        system_ram_gb=16.0,
                        is_network_available=False,
                        on_battery_power=True,  # Unplugged!
                    )
                    model_router.update_hardware_profile(battery_hw)
                    tts_router.update_hardware_profile(battery_hw)

            from nova.core.domain.events import EventType

            bus.subscribe(_on_planner_done, EventType.AGENT_COMPLETED)

            orchestrator = NovaGraphOrchestrator(
                agents={
                    AgentRole.PLANNER: BaseSpecializedAgent(AgentRole.PLANNER, model_router, bus),
                    AgentRole.RESEARCH: BaseSpecializedAgent(AgentRole.RESEARCH, model_router, bus),
                },
                checkpoint_store=SqliteCheckpointStore(tmp_path / "p5_checkpoints.db"),
                permission_gate=DefaultPolicyPermissionGate(),
                tool_executor=InMemoryMockToolExecutor(),
                event_bus=bus,
            )

            final_state = orchestrator.start_task(
                task_id="task_p5_multimodal",
                session_id="sess_p5_01",
                user_goal=spoken_goal,
                untrusted_data=(ocr_data_item,),
                require_citations=True,
            )

            assert final_state.status == TaskStatus.COMPLETED

            # Verify Routing History:
            # Call #0 (Planner) ran on VLLM (plugged in GPU)
            # Call #1 (Research) automatically switched to LLAMACPP (battery power) mid-task!
            assert len(model_router.routing_history) == 2
            assert model_router.routing_history[0].selected_engine == ServingEngineKind.VLLM
            assert model_router.routing_history[1].selected_engine == ServingEngineKind.LLAMACPP

            # 5. NOVA's Mouth: Synthesize the final answer into streaming TTS chunks (using Piper on battery)
            ans = final_state.node_records["step_summarize_scan"].final_answer
            assert ans is not None
            tts_chunks = list(tts_router.synthesize_stream(ans.content))
            assert len(tts_chunks) == 2
            assert tts_chunks[0].engine == TtsEngineKind.PIPER
            assert "$1,250.00" in tts_chunks[0].spoken_text_slice

    def test_privacy_floor_keeps_sensitive_financial_task_local_even_when_cloud_enabled(
        self,
    ) -> None:
        """ADR-001 & Section 16.2: Sensitive financial data stays 100% local unless sensitive-cloud consent is granted."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            bus = InMemoryEventBus()

            # User enabled general cloud inference (`allow_cloud_inference=True`),
            # but did NOT enable `allow_cloud_sensitive_data`!
            settings = load_settings(
                environ={},
                runtime_overrides={
                    "allow_cloud_inference": True,
                    "allow_cloud_sensitive_data": False,
                },
            )

            cloud_adapter = OptInCloudAdapter(
                scripted_outputs=[{"ERROR": "Cloud should NEVER be called for sensitive financial data!"}]
            )
            llamacpp_adapter = LlamaCppAdapter(
                scripted_outputs=[
                    {
                        "kind": "plan",
                        "agent_role": "planner",
                        "node_id": "node_planner",
                        "thought": "Plan local coding analysis of financial tax record.",
                        "goal_summary": "Analyze tax record locally.",
                        "steps": [
                            {
                                "step_id": "step_code_tax",
                                "target_agent": "coding",
                                "objective": "Verify tax total.",
                                "depends_on": [],
                                "estimated_sensitivity": "read_only",
                            }
                        ],
                    },
                    {
                        "kind": "final_answer",
                        "agent_role": "coding",
                        "node_id": "step_code_tax",
                        "thought": "Verified locally.",
                        "content": "Tax total verified locally.",
                        "cited_source_ids": ["fin_tax_2026"],
                        "confidence": 1.0,
                    },
                ]
            )

            router = NovaModelRouter(
                llamacpp_adapter=llamacpp_adapter,
                cloud_adapter=cloud_adapter,
                hardware_profile=HardwareProfile(is_network_available=True),
                event_bus=bus,
            )

            orchestrator = NovaGraphOrchestrator(
                agents={
                    AgentRole.PLANNER: BaseSpecializedAgent(
                        AgentRole.PLANNER, router, bus, settings=settings
                    ),
                    AgentRole.CODING: BaseSpecializedAgent(
                        AgentRole.CODING, router, bus, settings=settings
                    ),
                },
                checkpoint_store=SqliteCheckpointStore(tmp_path / "privacy_checkpoints.db"),
                permission_gate=DefaultPolicyPermissionGate(),
                tool_executor=InMemoryMockToolExecutor(),
                event_bus=bus,
                settings=settings,
            )

            vision = LocalVisionRouter()
            vision.register_mock_ocr("file:///tax/form_2026.png", "Taxable Income: $85,000")
            sensitive_tax_doc = vision.extract_ocr_as_untrusted_data(
                item_id="fin_tax_2026",
                image_uri="file:///tax/form_2026.png",
                data_classification=DataClassification.SENSITIVE_CATEGORY,
            )

            final_state = orchestrator.start_task(
                task_id="task_p5_privacy",
                session_id="sess_p5_privacy",
                user_goal="Verify taxable income from form_2026.png.",
                untrusted_data=(sensitive_tax_doc,),
            )

            assert final_state.status == TaskStatus.COMPLETED
            # Zero requests were ever sent to the cloud adapter!
            assert len(cloud_adapter.recorded_requests) == 0
            assert len(llamacpp_adapter.recorded_requests) == 2
            for decision in router.routing_history:
                assert decision.selected_engine == ServingEngineKind.LLAMACPP