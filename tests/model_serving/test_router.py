"""Unit tests for NOVA Layer 2 Model Serving Router & Runtime Adapters (Task 5.1).

Verifies:
1. Hardware-aware local routing (Phase 10, p. 42): selects `llama.cpp` on CPU-only or battery-powered
   laptops, and auto-selects `vLLM` when a GPU with >= 8GB VRAM is plugged in.
2. Section 12.2 Local vs. Cloud Decision Matrix (p. 24):
   - Routes `large_local_or_cloud_opt_in` to Cloud ONLY when `allow_cloud_routing=True` AND network is up.
   - Automatically falls back to local (`vLLM` or `llama.cpp`) when offline (`is_network_available=False`)
     or when the cloud API fails mid-task!
   - Never routes `small_latency_optimized_local` (e.g., Voice / intent classification) to the cloud.
3. Phase 15 Supply-Chain Security (p. 47): `ModelWeightManifest` verifies `.gguf` SHA-256 checksums
   and blocks tampered model files (`ModelWeightIntegrityError`).
4. ADR-009 Streaming-First Pipeline: `generate_stream()` yields token chunks and publishes
   `EventType.MODEL_TOKEN_STREAMED` events on the Event Bus.
"""

from __future__ import annotations

import hashlib
import pytest

from nova.core.domain.events import EventType
from nova.core.domain.primitives import AgentRole
from nova.core.model_serving.interface import ModelInferenceRequest
from nova.core.model_serving.router import (
    HardwareProfile,
    LlamaCppAdapter,
    ModelServingError,
    ModelWeightIntegrityError,
    ModelWeightManifest,
    NovaModelRouter,
    OptInCloudAdapter,
    QuantizationLevel,
    ServingEngineKind,
    VllmGpuAdapter,
)
from nova.core.platform.event_bus import InMemoryEventBus


def _sample_request(
    *,
    agent_role: AgentRole = AgentRole.CODING,
    model_tier: str = "large_local_or_cloud_opt_in",
    allow_cloud_routing: bool = False,
) -> ModelInferenceRequest:
    return ModelInferenceRequest(
        task_id="task_501",
        session_id="sess_501",
        node_id="node_501",
        agent_role=agent_role,
        model_tier=model_tier,
        messages=(
            {"role": "system", "content": "You are a NOVA agent."},
            {"role": "user", "content": "<untrusted_data> Sample </untrusted_data>"},
        ),
        allow_cloud_routing=allow_cloud_routing,
    )


class TestNovaModelRouterAndAdapters:
    """Verification suite for Task 5.1 (`router.py`)."""

    def test_hardware_aware_local_routing_between_llamacpp_and_vllm(self) -> None:
        """Phase 10 (p. 42): CPU/battery uses llama.cpp; plugged-in GPU (>=8GB VRAM) uses vLLM."""
        llamacpp = LlamaCppAdapter(
            scripted_outputs=[{"kind": "final_answer"}, {"kind": "final_answer"}]
        )
        vllm = VllmGpuAdapter(scripted_outputs=[{"kind": "final_answer"}])

        # 1. CPU-only laptop -> selects llama.cpp
        router = NovaModelRouter(
            llamacpp_adapter=llamacpp,
            vllm_adapter=vllm,
            hardware_profile=HardwareProfile(gpu_available=False, gpu_vram_gb=0.0),
        )
        resp_cpu = router.generate_structured(_sample_request(allow_cloud_routing=False))
        assert resp_cpu.provider == ServingEngineKind.LLAMACPP.value
        assert router.routing_history[-1].selected_engine == ServingEngineKind.LLAMACPP

        # 2. Plugged-in desktop/laptop with 12GB VRAM GPU -> auto-selects vLLM
        router.update_hardware_profile(
            HardwareProfile(gpu_available=True, gpu_vram_gb=12.0, on_battery_power=False)
        )
        resp_gpu = router.generate_structured(_sample_request(allow_cloud_routing=False))
        assert resp_gpu.provider == ServingEngineKind.VLLM.value
        assert router.routing_history[-1].selected_engine == ServingEngineKind.VLLM

        # 3. Laptop unplugged onto battery power -> switches back to llama.cpp gracefully
        router.update_hardware_profile(
            HardwareProfile(gpu_available=True, gpu_vram_gb=12.0, on_battery_power=True)
        )
        resp_battery = router.generate_structured(_sample_request(allow_cloud_routing=False))
        assert resp_battery.provider == ServingEngineKind.LLAMACPP.value

    def test_cloud_opt_in_routing_and_automatic_offline_fallback(self) -> None:
        """Section 12.2 (p. 24): Cloud is used only when consented + online, and falls back locally if offline."""
        llamacpp = LlamaCppAdapter(
            scripted_outputs=[
                {"source": "local_offline_fallback"},
                {"source": "local_error_fallback"},
                {"source": "local_voice"},
            ]
        )
        cloud = OptInCloudAdapter(
            scripted_outputs=[
                {"source": "cloud_frontier"},
                RuntimeError("Cloud API 503 Service Unavailable"),
            ]
        )

        router = NovaModelRouter(
            llamacpp_adapter=llamacpp,
            cloud_adapter=cloud,
            hardware_profile=HardwareProfile(is_network_available=True),
        )

        # 1. Consented + online + frontier tier -> routes to Cloud
        resp_cloud = router.generate_structured(
            _sample_request(
                agent_role=AgentRole.CODING,
                model_tier="large_local_or_cloud_opt_in",
                allow_cloud_routing=True,
            )
        )
        assert resp_cloud.is_cloud is True
        assert resp_cloud.provider == ServingEngineKind.CLOUD.value

        # 2. Network drops (`is_network_available=False`) -> automatically falls back to local llama.cpp!
        router.update_hardware_profile(HardwareProfile(is_network_available=False))
        resp_offline = router.generate_structured(
            _sample_request(
                agent_role=AgentRole.CODING,
                model_tier="large_local_or_cloud_opt_in",
                allow_cloud_routing=True,
            )
        )
        assert resp_offline.is_cloud is False
        assert resp_offline.provider == ServingEngineKind.LLAMACPP.value
        assert router.routing_history[-1].used_fallback is True

        # 3. Network returns, but Cloud throws 503 error -> automatically falls back to local llama.cpp!
        router.update_hardware_profile(HardwareProfile(is_network_available=True))
        resp_err_fallback = router.generate_structured(
            _sample_request(
                agent_role=AgentRole.CODING,
                model_tier="large_local_or_cloud_opt_in",
                allow_cloud_routing=True,
            )
        )
        assert resp_err_fallback.is_cloud is False
        assert router.routing_history[-1].used_fallback is True

        # 4. Small latency-optimized tier (Voice) NEVER routes to cloud even if allow_cloud_routing=True
        resp_voice = router.generate_structured(
            _sample_request(
                agent_role=AgentRole.VOICE,
                model_tier="small_latency_optimized_local",
                allow_cloud_routing=True,
            )
        )
        assert resp_voice.is_cloud is False
        assert resp_voice.provider == ServingEngineKind.LLAMACPP.value

    def test_model_weight_manifest_verifies_sha256_supply_chain_integrity(self) -> None:
        """Phase 15 (p. 47): Tampered local .gguf model weights are blocked before loading."""
        trusted_gguf_bytes = b"GGUF-v3-qwen2.5-7b-q4_k_m-weights"
        trusted_sha = hashlib.sha256(trusted_gguf_bytes).hexdigest()

        manifest = ModelWeightManifest(
            model_id="qwen2.5-7b-instruct-q4_k_m.gguf",
            quantization=QuantizationLevel.Q4_K_M,
            expected_sha256=trusted_sha,
        )

        # Matching SHA-256 succeeds
        adapter = LlamaCppAdapter(
            weight_manifest=manifest,
            observed_weight_sha256=trusted_sha,
        )
        assert adapter.default_model_id == "qwen2.5-7b-instruct-q4_k_m.gguf"

        # Tampered SHA-256 raises ModelWeightIntegrityError
        tampered_sha = hashlib.sha256(b"backdoored-gguf-weights").hexdigest()
        with pytest.raises(ModelWeightIntegrityError, match="checksum mismatch"):
            LlamaCppAdapter(
                weight_manifest=manifest,
                observed_weight_sha256=tampered_sha,
            )

    def test_token_streaming_emits_bus_events_and_reconstructs_payload(self) -> None:
        """ADR-009 & NFR-01: generate_stream yields chunks and emits MODEL_TOKEN_STREAMED events."""
        bus = InMemoryEventBus()
        llamacpp = LlamaCppAdapter(
            scripted_outputs=['{"kind":"final_answer","content":"Streaming works!"}'],
            first_token_latency_ms=180,
        )
        router = NovaModelRouter(llamacpp_adapter=llamacpp, event_bus=bus)

        req = _sample_request(allow_cloud_routing=False)
        chunks = list(router.generate_stream(req, chunk_size=12))
        assert "".join(chunks) == '{"kind":"final_answer","content":"Streaming works!"}'
        assert len(chunks) >= 4

        stream_events = bus.events_by_type(EventType.MODEL_TOKEN_STREAMED)
        assert len(stream_events) == len(chunks)
        assert stream_events[0].payload["first_token_latency_ms"] == 180
        assert stream_events[0].payload["provider"] == "llamacpp"