"""NOVA Phase 14: VRAM-Aware GGUF GPU Layer Calculator, Live Brain Switcher & Security Telemetry (`services/backend/src/nova/core/model_serving/brain_switcher.py`).

Plain-English Purpose (The Gear Shift Lever & 6GB VRAM Load Balancer):
Your `qwen3.5-9b-Q4_K_M.gguf` model is **5.88 GB (`5,883.54 MB`)**. On a laptop GPU with **6 GB VRAM**
(where Windows already uses ~0.8–1.0 GB for the display), trying to cram 100% of a 5.88 GB model
plus its conversation memory (KV cache) onto the GPU all at once causes a CUDA Out-Of-Memory crash.

`brain_switcher.py` solves that and connects Phases 11, 12, and 13 into a single live controller:

1. **VRAM-Aware GPU Layer Split Calculator (`compute_optimal_gguf_gpu_plan`)**:
   - Calculates exactly how many transformer layers of `qwen3.5-9b-Q4_K_M.gguf` fit safely inside
     your GPU's available VRAM (`hybrid_gpu_cpu_split` — e.g., putting `28/36` layers on your 6 GB
     GPU for high speed and keeping `8` layers in system RAM so it never overflows VRAM!).
   - Generates a tuned Ollama `Modelfile` with `PARAMETER num_gpu <layers>`.

2. **Live Runtime Brain Switcher (`NovaBrainAndSecurityController`)**:
   - Lets you switch NOVA's active reasoning engine on the fly (`switch_brain_mode`) between:
     - `"bootstrap"`: Instant zero-config `RuleGuidedLocalBootstrapAdapter`
     - `"ollama_live"`: Live Ollama (`127.0.0.1:11434`) with automatic fallback
     - `"gguf_qwen35"`: Your local `qwen3.5-9b-Q4_K_M.gguf` with VRAM-tuned layer offload
   - Automatically seals every SQLite audit log row into Phase 13's `TamperEvidentAuditChain` and
     reports live cryptographic verification (`audit_chain_valid: True`)!
"""

from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Optional

from pydantic import BaseModel, Field

try:
    from pydantic import ConfigDict

    PYDANTIC_V2 = True
except ImportError:  # pragma: no cover
    PYDANTIC_V2 = False

from nova.core.model_serving.gguf_live import (
    DiscoveredGgufBundle,
    LiveGgufModelAdapter,
    discover_local_gguf_models,
)
from nova.core.model_serving.ollama_live import (
    HybridLocalModelAdapter,
    LiveOllamaHttpClient,
)
from nova.core.permissions.security_hardening import (
    LocalApiAuthGuard,
    LocalRateAndPayloadGuard,
    TamperEvidentAuditChain,
)
from nova.main import RuleGuidedLocalBootstrapAdapter

if TYPE_CHECKING:
    from nova.main import NovaRuntimeContainer


class BrainEngineMode(str, Enum):
    """Supported runtime brain modes selectable in NOVA."""

    BOOTSTRAP = "bootstrap"
    OLLAMA_LIVE = "ollama_live"
    GGUF_QWEN35 = "gguf_qwen35"


class GpuOffloadPlan(BaseModel):
    """Calculated GPU/CPU layer split for running a `.gguf` model without VRAM overflow."""

    if PYDANTIC_V2:
        model_config = ConfigDict(frozen=True, extra="forbid")
    else:

        class Config:
            allow_mutation = False
            extra = "forbid"

    offload_mode: str = Field(..., min_length=1)
    gpu_vram_gb: float = Field(..., ge=0.0)
    usable_vram_mb: float = Field(..., ge=0.0)
    model_size_mb: float = Field(..., ge=0.0)
    total_layers: int = Field(..., ge=1)
    recommended_gpu_layers: int = Field(..., ge=0)
    cpu_fallback_layers: int = Field(..., ge=0)
    recommended_ctx_tokens: int = Field(..., ge=512)
    ollama_modelfile: str = Field(..., min_length=1)


def compute_optimal_gguf_gpu_plan(
    *,
    model_size_mb: float,
    gpu_vram_gb: float = 6.0,
    total_layers: int = 36,
    os_and_kv_reserve_mb: float = 1200.0,
    gguf_path: Optional[Path] = None,
) -> GpuOffloadPlan:
    """Calculate the safe number of GPU layers (`num_gpu` / `n_gpu_layers`) for a given VRAM budget."""
    total_vram_mb = max(0.0, float(gpu_vram_gb) * 1024.0)
    usable_vram_mb = max(0.0, round(total_vram_mb - float(os_and_kv_reserve_mb), 2))

    if gpu_vram_gb <= 0.0 or usable_vram_mb <= 0.0:
        gpu_layers = 0
        offload_mode = "cpu_only"
        ctx_tokens = 2048
    elif usable_vram_mb >= model_size_mb:
        gpu_layers = total_layers
        offload_mode = "full_gpu"
        ctx_tokens = 8192 if (usable_vram_mb - model_size_mb) >= 1500.0 else 4096
    else:
        fraction = usable_vram_mb / max(1.0, float(model_size_mb))
        gpu_layers = max(1, min(total_layers - 1, int(fraction * total_layers)))
        offload_mode = "hybrid_gpu_cpu_split"
        ctx_tokens = 4096

    cpu_layers = max(0, total_layers - gpu_layers)
    model_from = str(gguf_path) if gguf_path is not None else "qwen3.5-9b-Q4_K_M.gguf"
    modelfile_lines = [
        f"# NOVA VRAM-Tuned Modelfile ({offload_mode}: {gpu_layers}/{total_layers} layers on GPU)",
        f'FROM "{model_from}"',
        "PARAMETER temperature 0",
        f"PARAMETER num_ctx {ctx_tokens}",
        f"PARAMETER num_gpu {gpu_layers}",
    ]

    return GpuOffloadPlan(
        offload_mode=offload_mode,
        gpu_vram_gb=float(gpu_vram_gb),
        usable_vram_mb=usable_vram_mb,
        model_size_mb=round(float(model_size_mb), 2),
        total_layers=total_layers,
        recommended_gpu_layers=gpu_layers,
        cpu_fallback_layers=cpu_layers,
        recommended_ctx_tokens=ctx_tokens,
        ollama_modelfile="\n".join(modelfile_lines) + "\n",
    )


class BrainSwitchRequest(BaseModel):
    """Payload for `POST /brain/switch` to change NOVA's active reasoning engine at runtime."""

    if PYDANTIC_V2:
        model_config = ConfigDict(frozen=True, extra="forbid")
    else:

        class Config:
            allow_mutation = False
            extra = "forbid"

    mode: BrainEngineMode
    ollama_model: str = Field(default="qwen3.5-9b", min_length=1)
    gpu_vram_gb: float = Field(default=6.0, ge=0.0)


class NovaBrainAndSecurityController:
    """Coordinates live brain switching (`bootstrap` / `ollama_live` / `gguf_qwen35`) and Phase 13 security."""

    def __init__(
        self,
        container: "NovaRuntimeContainer",
        *,
        gpu_vram_gb: float = 6.0,
        search_root: Optional[Path] = None,
        api_token: Optional[str] = None,
        require_token: bool = False,
    ) -> None:
        self.container = container
        self.gpu_vram_gb = float(gpu_vram_gb)
        self.search_root = search_root
        self.active_mode: BrainEngineMode = BrainEngineMode.BOOTSTRAP

        self.auth_guard = LocalApiAuthGuard(
            api_token=api_token,
            require_token=require_token,
        )
        self.rate_guard = LocalRateAndPayloadGuard()
        self.audit_chain = TamperEvidentAuditChain()

        bundles = discover_local_gguf_models(search_root=search_root)
        self.primary_gguf: Optional[DiscoveredGgufBundle] = bundles[0] if bundles else None

    def compute_current_gpu_plan(self, gpu_vram_gb: Optional[float] = None) -> GpuOffloadPlan:
        """Compute the GPU layer offload plan for the discovered GGUF model (or 5,883.54 MB default)."""
        vram = self.gpu_vram_gb if gpu_vram_gb is None else float(gpu_vram_gb)
        size_mb = self.primary_gguf.size_mb if self.primary_gguf is not None else 5883.54
        gguf_path = self.primary_gguf.llm_gguf_path if self.primary_gguf is not None else None
        return compute_optimal_gguf_gpu_plan(
            model_size_mb=size_mb,
            gpu_vram_gb=vram,
            gguf_path=gguf_path,
        )

    def switch_brain_mode(
        self,
        mode: BrainEngineMode,
        *,
        ollama_model: str = "qwen3.5-9b",
        gpu_vram_gb: Optional[float] = None,
        ollama_client: Optional[LiveOllamaHttpClient] = None,
        gguf_adapter: Optional[LiveGgufModelAdapter] = None,
    ) -> Dict[str, Any]:
        """Switch NOVA's active model adapter at runtime without restarting the FastAPI server."""
        if gpu_vram_gb is not None:
            self.gpu_vram_gb = float(gpu_vram_gb)

        plan = self.compute_current_gpu_plan()

        if mode == BrainEngineMode.BOOTSTRAP:
            adapter: RuleGuidedLocalBootstrapAdapter = RuleGuidedLocalBootstrapAdapter()
        elif mode == BrainEngineMode.OLLAMA_LIVE:
            client = ollama_client or LiveOllamaHttpClient(default_model=ollama_model)
            adapter = HybridLocalModelAdapter(ollama_client=client)
        elif mode == BrainEngineMode.GGUF_QWEN35:
            adapter = gguf_adapter or LiveGgufModelAdapter(
                bundle=self.primary_gguf,
                n_ctx=plan.recommended_ctx_tokens,
                n_gpu_layers=plan.recommended_gpu_layers,
            )
        else:  # pragma: no cover
            raise ValueError(f"Unsupported BrainEngineMode: {mode!r}")

        self.container.bootstrap_adapter = adapter
        self.container.model_router._llamacpp = adapter
        self.active_mode = mode
        return self.get_security_and_brain_status()

    def get_security_and_brain_status(self) -> Dict[str, Any]:
        """Return a unified status dictionary covering active brain, GPU split, and audit chain integrity."""
        logs = self.container.api_service.query_audit_logs()
        self.audit_chain.seal_existing_logs(logs)
        chain_report = self.audit_chain.verify_chain()
        gpu_plan = self.compute_current_gpu_plan()

        return {
            "active_brain_mode": self.active_mode.value,
            "gguf_discovered": self.primary_gguf is not None,
            "gguf_model_name": (
                self.primary_gguf.model_name
                if self.primary_gguf is not None
                else "qwen3.5-9b-Q4_K_M (default profile)"
            ),
            "gguf_size_mb": gpu_plan.model_size_mb,
            "gpu_offload_plan": {
                "offload_mode": gpu_plan.offload_mode,
                "gpu_vram_gb": gpu_plan.gpu_vram_gb,
                "usable_vram_mb": gpu_plan.usable_vram_mb,
                "recommended_gpu_layers": gpu_plan.recommended_gpu_layers,
                "cpu_fallback_layers": gpu_plan.cpu_fallback_layers,
                "total_layers": gpu_plan.total_layers,
                "recommended_ctx_tokens": gpu_plan.recommended_ctx_tokens,
            },
            "security_telemetry": {
                "token_auth_enabled": self.auth_guard.require_token,
                "max_payload_bytes": self.rate_guard.max_payload_bytes,
                "max_requests_per_window": self.rate_guard.max_requests_per_window,
                "audit_chain_valid": chain_report.valid,
                "sealed_audit_entries": chain_report.checked_count,
                "head_seal_hash": chain_report.head_seal_hash,
            },
        }


def register_brain_and_security_routes(
    fastapi_app: Any,
    controller: NovaBrainAndSecurityController,
) -> None:
    """Attach `GET /brain/status` and `POST /brain/switch` to the NOVA FastAPI app."""

    @fastapi_app.get("/brain/status")
    def get_brain_status() -> Dict[str, Any]:
        return controller.get_security_and_brain_status()

    @fastapi_app.post("/brain/switch")
    def switch_brain(request: BrainSwitchRequest) -> Dict[str, Any]:
        return controller.switch_brain_mode(
            request.mode,
            ollama_model=request.ollama_model,
            gpu_vram_gb=request.gpu_vram_gb,
        )


__all__ = [
    "BrainEngineMode",
    "BrainSwitchRequest",
    "GpuOffloadPlan",
    "NovaBrainAndSecurityController",
    "compute_optimal_gguf_gpu_plan",
    "register_brain_and_security_routes",
]