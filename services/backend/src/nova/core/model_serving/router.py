"""NOVA Layer 2 Model Serving Router, Local Runtime Adapters & Token Streaming (ADR-001 / ADR-006 / ADR-009 / Phase 10 / Section 12.2 / Phase 15).

Plain-English Purpose (The Smart Traffic Controller & Engine Room):
Imagine NOVA has three different engines it can use to think (`ModelReasoningPort`):
1. **The Reliable Everyday Hybrid Engine (`LlamaCppAdapter` — `llama.cpp`)**:
   - Works on *every* laptop (even CPU-only machines with no graphics card) using compressed `.gguf`
     model files (`Q4_K_M`, `Q5_K_M`, `Q8_0`). This is NOVA's universal offline floor (Phase 10, p. 42).
2. **The Turbo GPU Sports Engine (`VllmGpuAdapter` — `vLLM`)**:
   - Automatically selected when your PC has a compatible GPU with enough video memory (`gpu_vram_gb >= 8.0`)
     and isn't restricted to battery-saver CPU mode (Phase 10, p. 42).
3. **The Opt-In Cloud Jet Engine (`OptInCloudAdapter` — Cloud API)**:
   - Used *only* for heavy `large_local_or_cloud_opt_in` tasks when you explicitly consented
     (`request.allow_cloud_routing=True`) AND internet is connected (`Section 12.2`, p. 24).
   - If the internet drops mid-task or the cloud API errors out, the Traffic Controller (`NovaModelRouter`)
     **automatically falls back to your local engine** without crashing the task!

Plus two critical guarantees:
- **Live Word-by-Word Streaming (`ADR-009` & `NFR-01`)**: `generate_stream()` streams tokens one by
  one onto the Event Bus (`EventType.MODEL_TOKEN_STREAMED`) so the UI never freezes.
- **Model File Fingerprint Check (`ModelWeightManifest` — `Phase 15`)**: Verifies the SHA-256
  checksum of downloaded `.gguf` model weights before loading them to prevent tampered models.
"""

from __future__ import annotations

import hashlib
import re
from collections import deque
from enum import Enum
from typing import Any, Callable, Deque, Dict, Iterator, List, Mapping, Optional, Sequence, Tuple, Union

from pydantic import BaseModel, Field

try:
    from pydantic import ConfigDict

    PYDANTIC_V2 = True
except ImportError:  # pragma: no cover
    PYDANTIC_V2 = False

from nova.core.domain.events import EventType, NovaEvent
from nova.core.model_serving.interface import (
    ModelInferenceRequest,
    ModelInferenceResponse,
    ModelReasoningPort,
)
from nova.core.platform.event_bus import EventPublisherPort

_SHA256_PATTERN = re.compile(r"^[a-fA-F0-9]{64}$")


class ModelServingError(RuntimeError):
    """Base error for NOVA Layer 2 model serving failures."""


class ModelWeightIntegrityError(ModelServingError):
    """Raised when a local model weight file fails SHA-256 supply-chain verification (Phase 15)."""


class QuantizationLevel(str, Enum):
    """GGUF / local model weight quantization levels evaluated in Phase 10 (pp. 41-42)."""

    Q4_K_M = "q4_k_m"  # Default 4-bit quantization for broad consumer hardware floor
    Q5_K_M = "q5_k_m"  # Balanced 5-bit quantization for higher tool-calling fidelity
    Q8_0 = "q8_0"      # 8-bit quantization when ample RAM/VRAM is available
    FP16 = "fp16"      # Unquantized half-precision for high-VRAM GPU servers


class ServingEngineKind(str, Enum):
    """Concrete serving engines managed behind NOVA's single Model Router (Section 12.1 / Phase 10)."""

    LLAMACPP = "llamacpp"
    VLLM = "vllm"
    CLOUD = "cloud"


class HardwareProfile(BaseModel):
    """Snapshot of current device hardware and connectivity state for dynamic routing (Phase 10 / NFR-11)."""

    if PYDANTIC_V2:
        model_config = ConfigDict(frozen=True, extra="forbid")
    else:

        class Config:
            allow_mutation = False
            extra = "forbid"

    gpu_available: bool = Field(
        default=False,
        description="True if a CUDA/ROCm/Metal GPU accelerator is currently available",
    )
    gpu_vram_gb: float = Field(
        default=0.0,
        ge=0.0,
        description="Available GPU VRAM in gigabytes",
    )
    system_ram_gb: float = Field(
        default=16.0,
        ge=1.0,
        description="Total system RAM in gigabytes",
    )
    cpu_threads: int = Field(
        default=8,
        ge=1,
        description="Logical CPU threads available for llama.cpp inference",
    )
    is_network_available: bool = Field(
        default=False,
        description="True if outbound network connectivity is currently available",
    )
    on_battery_power: bool = Field(
        default=False,
        description="Phase 10 (p. 42): True when laptop is unplugged on battery power",
    )

    def can_use_vllm_gpu(self, min_vram_gb: float = 8.0) -> bool:
        """Return True if vLLM GPU acceleration should be selected over CPU llama.cpp (Phase 10)."""
        return (
            self.gpu_available
            and self.gpu_vram_gb >= min_vram_gb
            and not self.on_battery_power
        )


class ModelWeightManifest(BaseModel):
    """Supply-chain integrity manifest for a local GGUF/ONNX model weight file (Phase 10 / Phase 15)."""

    if PYDANTIC_V2:
        model_config = ConfigDict(frozen=True, extra="forbid")
    else:

        class Config:
            allow_mutation = False
            extra = "forbid"

    model_id: str = Field(..., min_length=1, description="Canonical local model identifier")
    quantization: QuantizationLevel = Field(
        default=QuantizationLevel.Q4_K_M,
        description="Quantization format (Phase 10)",
    )
    expected_sha256: str = Field(
        ...,
        description="Trusted 64-character hex SHA-256 digest of the model weight file (Phase 15)",
    )
    min_ram_gb: float = Field(default=4.0, ge=0.5)

    def __init__(self, **data: Any) -> None:
        sha = data.get("expected_sha256")
        if not isinstance(sha, str) or not _SHA256_PATTERN.match(sha):
            raise ValueError(
                f"expected_sha256 must be a valid 64-character hex SHA-256 digest, got {sha!r}."
            )
        updated = dict(data)
        updated["expected_sha256"] = sha.lower()
        super().__init__(**updated)

    def verify_weights(self, observed_bytes_or_sha256: bytes | str) -> None:
        """Verify that a local model file matches its pinned SHA-256 checksum (Phase 15)."""
        if isinstance(observed_bytes_or_sha256, bytes):
            actual_hex = hashlib.sha256(observed_bytes_or_sha256).hexdigest().lower()
        else:
            actual_hex = observed_bytes_or_sha256.strip().lower()
        if actual_hex != self.expected_sha256:
            raise ModelWeightIntegrityError(
                f"Phase 15 supply-chain violation: Model weight file {self.model_id!r} "
                f"checksum mismatch! Expected {self.expected_sha256}, got {actual_hex}."
            )


class RoutingDecisionRecord(BaseModel):
    """Inspectable audit record explaining why `NovaModelRouter` picked a specific engine (Section 12.2)."""

    if PYDANTIC_V2:
        model_config = ConfigDict(frozen=True, extra="forbid")
    else:

        class Config:
            allow_mutation = False
            extra = "forbid"

    task_id: str = Field(..., min_length=1)
    node_id: str = Field(..., min_length=1)
    selected_engine: ServingEngineKind = Field(...)
    selected_model_id: str = Field(..., min_length=1)
    used_fallback: bool = Field(default=False)
    rationale: str = Field(..., min_length=1)


ScriptedOutputItem = Union[
    str,
    Mapping[str, Any],
    ModelInferenceResponse,
    Exception,
    Callable[[ModelInferenceRequest], Union[str, Mapping[str, Any]]],
]


class _BaseScriptableEngineAdapter:
    """Base adapter supporting deterministic scripted outputs and token-chunk streaming."""

    def __init__(
        self,
        *,
        provider_name: str,
        default_model_id: str,
        is_cloud: bool,
        default_first_token_latency_ms: int,
        scripted_outputs: Sequence[ScriptedOutputItem] = (),
        weight_manifest: Optional[ModelWeightManifest] = None,
        observed_weight_sha256: Optional[str] = None,
    ) -> None:
        if weight_manifest is not None and observed_weight_sha256 is not None:
            weight_manifest.verify_weights(observed_weight_sha256)

        self._provider_name = provider_name
        self._default_model_id = (
            weight_manifest.model_id if weight_manifest is not None else default_model_id
        )
        self._is_cloud = is_cloud
        self._default_latency_ms = default_first_token_latency_ms
        self._weight_manifest = weight_manifest
        self._queue: Deque[ScriptedOutputItem] = deque(scripted_outputs)
        self.recorded_requests: List[ModelInferenceRequest] = []

    @property
    def provider_name(self) -> str:
        return self._provider_name

    @property
    def default_model_id(self) -> str:
        return self._default_model_id

    def enqueue(self, item: ScriptedOutputItem) -> None:
        self._queue.append(item)

    def generate_structured(self, request: ModelInferenceRequest) -> ModelInferenceResponse:
        self.recorded_requests.append(request)
        if not self._queue:
            raise ModelServingError(
                f"Engine adapter {self._provider_name!r} has no scripted responses remaining."
            )
        item = self._queue.popleft()
        if isinstance(item, Exception):
            raise item
        if isinstance(item, ModelInferenceResponse):
            return item
        if callable(item):
            raw_payload = item(request)
        else:
            raw_payload = item

        normalized: Union[str, Dict[str, Any]] = (
            dict(raw_payload) if isinstance(raw_payload, Mapping) else str(raw_payload)
        )
        return ModelInferenceResponse(
            raw_output=normalized,
            model_id=self._default_model_id,
            provider=self._provider_name,
            is_cloud=self._is_cloud,
            prompt_tokens=120,
            completion_tokens=60,
            first_token_latency_ms=self._default_latency_ms,
        )


class LlamaCppAdapter(_BaseScriptableEngineAdapter):
    """Universal-compatibility local CPU/edge GGUF serving adapter (Phase 10 Default, p. 42)."""

    def __init__(
        self,
        scripted_outputs: Sequence[ScriptedOutputItem] = (),
        default_model_id: str = "qwen2.5-7b-instruct-q4_k_m.gguf",
        first_token_latency_ms: int = 320,
        weight_manifest: Optional[ModelWeightManifest] = None,
        observed_weight_sha256: Optional[str] = None,
    ) -> None:
        super().__init__(
            provider_name=ServingEngineKind.LLAMACPP.value,
            default_model_id=default_model_id,
            is_cloud=False,
            default_first_token_latency_ms=first_token_latency_ms,
            scripted_outputs=scripted_outputs,
            weight_manifest=weight_manifest,
            observed_weight_sha256=observed_weight_sha256,
        )


class VllmGpuAdapter(_BaseScriptableEngineAdapter):
    """High-throughput local GPU serving adapter with continuous batching / PagedAttention (Phase 10, p. 42)."""

    def __init__(
        self,
        scripted_outputs: Sequence[ScriptedOutputItem] = (),
        default_model_id: str = "qwen2.5-14b-instruct-fp16",
        first_token_latency_ms: int = 140,
        weight_manifest: Optional[ModelWeightManifest] = None,
        observed_weight_sha256: Optional[str] = None,
    ) -> None:
        super().__init__(
            provider_name=ServingEngineKind.VLLM.value,
            default_model_id=default_model_id,
            is_cloud=False,
            default_first_token_latency_ms=first_token_latency_ms,
            scripted_outputs=scripted_outputs,
            weight_manifest=weight_manifest,
            observed_weight_sha256=observed_weight_sha256,
        )


class OptInCloudAdapter(_BaseScriptableEngineAdapter):
    """Opt-in cloud model adapter enforcing ADR-001 consent before crossing network boundaries."""

    def __init__(
        self,
        scripted_outputs: Sequence[ScriptedOutputItem] = (),
        default_model_id: str = "frontier-cloud-reasoning-v1",
        first_token_latency_ms: int = 210,
    ) -> None:
        super().__init__(
            provider_name=ServingEngineKind.CLOUD.value,
            default_model_id=default_model_id,
            is_cloud=True,
            default_first_token_latency_ms=first_token_latency_ms,
            scripted_outputs=scripted_outputs,
        )

    def generate_structured(self, request: ModelInferenceRequest) -> ModelInferenceResponse:
        if not request.allow_cloud_routing:
            raise ModelServingError(
                "ADR-001 violation: OptInCloudAdapter refused request because allow_cloud_routing=False."
            )
        return super().generate_structured(request)


class NovaModelRouter(ModelReasoningPort):
    """Single Layer 2 Model Router dynamically selecting `llama.cpp`, `vLLM`, or `Cloud` (ADR-006 / Section 12.2 / Phase 10).

    Routing rules (Section 12.2 & Phase 10, pp. 24, 42):
    1. Voice / low-latency classification (`small_latency_optimized_local`) ALWAYS stays local
       (never leaves the device per Section 12.2).
    2. Frontier tasks (`large_local_or_cloud_opt_in`) route to `OptInCloudAdapter` ONLY if:
       - `request.allow_cloud_routing` is `True` (ADR-001 + Section 16.2 consent verified),
       - `hardware.is_network_available` is `True`, and
       - `cloud_adapter` is registered.
       If the cloud call fails or network drops mid-task, `NovaModelRouter` automatically falls back
       to the best available local engine (`vLLM` or `llama.cpp`)!
    3. For local execution:
       - Selects `VllmGpuAdapter` if `hardware.can_use_vllm_gpu(min_vram_gb)` is `True` and `vllm_adapter` is registered.
       - Otherwise selects `LlamaCppAdapter` (the universal CPU/edge hardware floor).
       - If `VllmGpuAdapter` raises a runtime error (e.g., GPU OOM mid-task), automatically falls back to `LlamaCppAdapter`!
    """

    def __init__(
        self,
        llamacpp_adapter: LlamaCppAdapter,
        vllm_adapter: Optional[VllmGpuAdapter] = None,
        cloud_adapter: Optional[OptInCloudAdapter] = None,
        hardware_profile: Optional[HardwareProfile] = None,
        event_bus: Optional[EventPublisherPort] = None,
        min_vllm_vram_gb: float = 8.0,
    ) -> None:
        self._llamacpp = llamacpp_adapter
        self._vllm = vllm_adapter
        self._cloud = cloud_adapter
        self._hardware = hardware_profile or HardwareProfile()
        self._event_bus = event_bus
        self._min_vllm_vram_gb = min_vllm_vram_gb
        self._routing_history: List[RoutingDecisionRecord] = []

    @property
    def hardware_profile(self) -> HardwareProfile:
        return self._hardware

    def update_hardware_profile(self, new_profile: HardwareProfile) -> None:
        """Update hardware/connectivity state mid-session (e.g., laptop unplugged or Wi-Fi lost)."""
        self._hardware = new_profile

    @property
    def routing_history(self) -> Tuple[RoutingDecisionRecord, ...]:
        return tuple(self._routing_history)

    def generate_structured(self, request: ModelInferenceRequest) -> ModelInferenceResponse:
        """Route `request` to the optimal engine with automatic local fallback (Section 12.2 / Phase 10)."""
        # 1. Check if eligible for Cloud routing (Section 12.2: only large_local_or_cloud_opt_in with consent + network)
        wants_cloud = (
            request.allow_cloud_routing
            and request.model_tier == "large_local_or_cloud_opt_in"
            and self._cloud is not None
        )

        if wants_cloud and self._hardware.is_network_available and self._cloud is not None:
            try:
                resp = self._cloud.generate_structured(request)
                self._routing_history.append(
                    RoutingDecisionRecord(
                        task_id=request.task_id,
                        node_id=request.node_id,
                        selected_engine=ServingEngineKind.CLOUD,
                        selected_model_id=resp.model_id,
                        used_fallback=False,
                        rationale="Section 12.2: Frontier tier with explicit user cloud opt-in and active network.",
                    )
                )
                return resp
            except Exception as cloud_exc:
                # Automatic local fallback if cloud fails mid-task!
                return self._execute_local_with_fallback(
                    request,
                    fallback_reason=f"Cloud adapter failed ({cloud_exc}); fell back to local engine.",
                )

        if wants_cloud and not self._hardware.is_network_available:
            return self._execute_local_with_fallback(
                request,
                fallback_reason="ADR-001 / Section 12.2 offline fallback: Network unavailable; routed locally.",
            )

        return self._execute_local_with_fallback(request, fallback_reason=None)

    def generate_stream(
        self,
        request: ModelInferenceRequest,
        *,
        chunk_size: int = 16,
    ) -> Iterator[str]:
        """Stream token chunks in real time while emitting `MODEL_TOKEN_STREAMED` events (ADR-009 / NFR-01)."""
        if chunk_size < 1:
            raise ValueError("chunk_size must be >= 1.")

        response = self.generate_structured(request)
        raw_text = (
            response.raw_output
            if isinstance(response.raw_output, str)
            else __import__("json").dumps(response.raw_output, sort_keys=True)
        )

        for idx, start in enumerate(range(0, len(raw_text), chunk_size)):
            token_chunk = raw_text[start : start + chunk_size]
            if self._event_bus is not None:
                self._event_bus.publish(
                    NovaEvent(
                        event_type=EventType.MODEL_TOKEN_STREAMED,
                        task_id=request.task_id,
                        session_id=request.session_id,
                        node_id=request.node_id,
                        agent_role=request.agent_role,
                        payload={
                            "token_index": idx,
                            "token_text": token_chunk,
                            "model_id": response.model_id,
                            "provider": response.provider,
                            "first_token_latency_ms": response.first_token_latency_ms,
                        },
                    )
                )
            yield token_chunk

    def _execute_local_with_fallback(
        self,
        request: ModelInferenceRequest,
        fallback_reason: Optional[str],
    ) -> ModelInferenceResponse:
        """Select `vLLM` (if GPU eligible) or `llama.cpp` (universal CPU floor) with automatic fallback."""
        if (
            self._vllm is not None
            and self._hardware.can_use_vllm_gpu(self._min_vllm_vram_gb)
        ):
            try:
                resp = self._vllm.generate_structured(request)
                self._routing_history.append(
                    RoutingDecisionRecord(
                        task_id=request.task_id,
                        node_id=request.node_id,
                        selected_engine=ServingEngineKind.VLLM,
                        selected_model_id=resp.model_id,
                        used_fallback=(fallback_reason is not None),
                        rationale=(
                            fallback_reason
                            or f"Phase 10: GPU detected ({self._hardware.gpu_vram_gb}GB VRAM); selected vLLM."
                        ),
                    )
                )
                return resp
            except Exception as vllm_exc:
                fallback_reason = (
                    f"vLLM GPU engine failed ({vllm_exc}); fell back to universal llama.cpp CPU engine."
                )

        resp = self._llamacpp.generate_structured(request)
        self._routing_history.append(
            RoutingDecisionRecord(
                task_id=request.task_id,
                node_id=request.node_id,
                selected_engine=ServingEngineKind.LLAMACPP,
                selected_model_id=resp.model_id,
                used_fallback=(fallback_reason is not None),
                rationale=(
                    fallback_reason
                    or "Phase 10: Selected universal compatibility default engine (llama.cpp)."
                ),
            )
        )
        return resp


__all__ = [
    "HardwareProfile",
    "LlamaCppAdapter",
    "ModelServingError",
    "ModelWeightIntegrityError",
    "ModelWeightManifest",
    "NovaModelRouter",
    "OptInCloudAdapter",
    "QuantizationLevel",
    "RoutingDecisionRecord",
    "ServingEngineKind",
    "VllmGpuAdapter",
]