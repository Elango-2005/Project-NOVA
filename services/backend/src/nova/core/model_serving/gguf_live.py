"""NOVA Phase 12: Direct HuggingFace GGUF Model Discovery & Live Qwen 3.5 9B Brain Adapter (`services/backend/src/nova/core/model_serving/gguf_live.py`).

Plain-English Purpose (Plugging Your Qwen 3.5 9B `.gguf` File Directly into NOVA):
You already have two `.gguf` files downloaded in your HuggingFace cache:
- `qwen3.5-9b-Q4_K_M.gguf` (your main 9B reasoning brain, quantized to 4-bit `Q4_K_M`)
- `mmproj-qwen3.5-9b-BF16.gguf` (your multimodal vision projector for reading images/screenshots)

This module (`gguf_live.py`) connects those exact `.gguf` files on your hard drive to NOVA:

1. **Automatic HuggingFace Cache Scanner (`discover_local_gguf_models`)**:
   - Scans `C:\\Users\\<you>\\.cache\\huggingface\\hub` (or `NOVA_GGUF_MODEL_PATH`) and automatically
     separates main LLM weights (`qwen3.5-9b-Q4_K_M.gguf`) from vision projectors (`mmproj-*.gguf`).

2. **Three Ways to Execute Your `.gguf` File Locally (`LiveGgufModelAdapter`)**:
   - **Engine 1 — In-Process `llama-cpp-python` (`from llama_cpp import Llama`)**:
     If `llama-cpp-python` is installed in your `.venv`, NOVA loads `qwen3.5-9b-Q4_K_M.gguf`
     directly into RAM/VRAM and calls `create_chat_completion(..., response_format={"type": "json_object"})`.
   - **Engine 2 — Local `llama-server` (`http://127.0.0.1:8080`) or Ollama (`http://127.0.0.1:11434`)**:
     If you run `llama-server -m <path-to-qwen3.5.gguf>` or import your `.gguf` into Ollama via
     `generate_ollama_modelfile()`, NOVA talks to it over local loopback HTTP (`127.0.0.1`).
   - **Engine 3 — Automatic Offline Bootstrap Fallback**:
     If neither runtime is started yet, NOVA still detects and verifies your `.gguf` file on disk
     and falls back cleanly so nothing ever crashes.
"""

from __future__ import annotations

import json
import os
import urllib.request
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Tuple

from pydantic import BaseModel, Field

try:
    from pydantic import ConfigDict

    PYDANTIC_V2 = True
except ImportError:  # pragma: no cover
    PYDANTIC_V2 = False

from nova.core.model_serving.interface import ModelInferenceRequest
from nova.core.model_serving.ollama_live import (
    _build_json_schema_system_hint,
    normalize_llm_proposal_dict,
    validate_loopback_base_url,
)
from nova.main import RuleGuidedLocalBootstrapAdapter


class DiscoveredGgufBundle(BaseModel):
    """Holds the discovered paths for a local `.gguf` LLM and its optional `mmproj` vision file."""

    if PYDANTIC_V2:
        model_config = ConfigDict(frozen=True, extra="forbid")
    else:

        class Config:
            allow_mutation = False
            extra = "forbid"

    model_name: str = Field(..., min_length=1)
    llm_gguf_path: Path
    mmproj_gguf_path: Optional[Path] = None
    size_mb: float = Field(..., ge=0.0)


def _default_hf_cache_dir() -> Path:
    env_override = os.environ.get("NOVA_HF_CACHE_DIR", "").strip()
    if env_override:
        return Path(env_override).expanduser().resolve()
    return (Path.home() / ".cache" / "huggingface" / "hub").resolve()


def discover_local_gguf_models(
    search_root: Optional[Path] = None,
    explicit_gguf_path: Optional[Path] = None,
) -> List[DiscoveredGgufBundle]:
    """Find `.gguf` models on disk, pairing primary LLM weights (`qwen3.5-9b-Q4_K_M.gguf`) with `mmproj`."""
    env_path = os.environ.get("NOVA_GGUF_MODEL_PATH", "").strip()
    target_explicit = explicit_gguf_path or (Path(env_path) if env_path else None)

    if target_explicit is not None and target_explicit.exists() and target_explicit.is_file():
        parent = target_explicit.parent
        mmproj_candidates = sorted(parent.glob("*mmproj*.gguf"))
        size_mb = round(target_explicit.stat().st_size / (1024 * 1024), 2)
        return [
            DiscoveredGgufBundle(
                model_name=target_explicit.stem,
                llm_gguf_path=target_explicit.resolve(),
                mmproj_gguf_path=mmproj_candidates[0].resolve() if mmproj_candidates else None,
                size_mb=size_mb,
            )
        ]

    root = (search_root or _default_hf_cache_dir()).resolve()
    if not root.exists():
        return []

    all_ggufs = sorted(root.rglob("*.gguf"))
    mmproj_by_dir: Dict[Path, Path] = {}
    llm_files: List[Path] = []

    for gguf in all_ggufs:
        lower_name = gguf.name.lower()
        if "mmproj" in lower_name:
            mmproj_by_dir[gguf.parent] = gguf.resolve()
        else:
            llm_files.append(gguf.resolve())

    bundles: List[DiscoveredGgufBundle] = []
    for llm_path in llm_files:
        size_mb = round(llm_path.stat().st_size / (1024 * 1024), 2)
        bundles.append(
            DiscoveredGgufBundle(
                model_name=llm_path.stem,
                llm_gguf_path=llm_path,
                mmproj_gguf_path=mmproj_by_dir.get(llm_path.parent),
                size_mb=size_mb,
            )
        )
    return bundles


def generate_ollama_modelfile(bundle: DiscoveredGgufBundle) -> str:
    """Generate an Ollama `Modelfile` pointing directly to your existing `.gguf` file (zero re-download)."""
    lines = [
        f"# Auto-generated NOVA Modelfile for {bundle.model_name}",
        f'FROM "{bundle.llm_gguf_path}"',
        "PARAMETER temperature 0",
        "PARAMETER num_ctx 4096",
    ]
    return "\n".join(lines) + "\n"


LlamaCompletionFn = Callable[[List[Dict[str, str]]], str]


class LiveGgufModelAdapter(RuleGuidedLocalBootstrapAdapter):
    """Connects NOVA to your local `qwen3.5-9b-Q4_K_M.gguf` via `llama-cpp-python` or `llama-server`."""

    def __init__(
        self,
        *,
        bundle: Optional[DiscoveredGgufBundle] = None,
        llama_server_url: str = "http://127.0.0.1:8080",
        completion_fn: Optional[LlamaCompletionFn] = None,
        n_ctx: int = 4096,
        n_gpu_layers: int = -1,
    ) -> None:
        super().__init__()
        discovered = discover_local_gguf_models() if bundle is None else [bundle]
        self.bundle: Optional[DiscoveredGgufBundle] = discovered[0] if discovered else None
        self.llama_server_url = validate_loopback_base_url(llama_server_url)
        self._custom_completion_fn = completion_fn
        self._llama_instance: Any = None
        self._n_ctx = n_ctx
        self._n_gpu_layers = n_gpu_layers
        self.gguf_calls_succeeded: int = 0
        self.fallback_calls_used: int = 0

    def _get_or_load_inprocess_llama(self) -> Any:
        """Lazily load `llama_cpp.Llama` in-process if `llama-cpp-python` is installed."""
        if self._llama_instance is not None:
            return self._llama_instance
        if self.bundle is None or not self.bundle.llm_gguf_path.exists():
            return None
        try:
            from llama_cpp import Llama  # type: ignore[import-not-found]

            self._llama_instance = Llama(
                model_path=str(self.bundle.llm_gguf_path),
                n_ctx=self._n_ctx,
                n_gpu_layers=self._n_gpu_layers,
                verbose=False,
            )
            return self._llama_instance
        except Exception:
            return None

    def _call_llama_server_http(self, messages: List[Dict[str, str]]) -> str:
        """Call a local `llama-server.exe` OpenAI-compatible `/v1/chat/completions` endpoint."""
        payload = {
            "messages": messages,
            "temperature": 0.0,
            "response_format": {"type": "json_object"},
        }
        req = urllib.request.Request(
            url=f"{self.llama_server_url}/v1/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=30.0) as resp:
            body = json.loads(resp.read().decode("utf-8"))
            return str(body["choices"][0]["message"]["content"])

    def _generate_raw_json_from_gguf(self, messages: List[Dict[str, str]]) -> str:
        if self._custom_completion_fn is not None:
            return self._custom_completion_fn(messages)

        llama_obj = self._get_or_load_inprocess_llama()
        if llama_obj is not None:
            out = llama_obj.create_chat_completion(
                messages=messages,
                temperature=0.0,
                response_format={"type": "json_object"},
            )
            return str(out["choices"][0]["message"]["content"])

        return self._call_llama_server_http(messages)

    def generate_structured(self, request: ModelInferenceRequest) -> Any:
        if self._queue:
            return super().generate_structured(request)

        messages = [dict(m) for m in request.messages]
        schema_hint = _build_json_schema_system_hint(request)
        if messages and messages[0].get("role") == "system":
            messages[0] = {
                "role": "system",
                "content": f"{messages[0]['content']}\n\n{schema_hint}",
            }
        else:
            messages.insert(0, {"role": "system", "content": schema_hint})

        try:
            raw_json_text = self._generate_raw_json_from_gguf(messages)
            parsed = json.loads(raw_json_text)
            if isinstance(parsed, Mapping):
                normalized = normalize_llm_proposal_dict(parsed, request)
                self.gguf_calls_succeeded += 1
                self.enqueue(normalized)
                return super().generate_structured(request)
        except Exception:
            pass

        self.fallback_calls_used += 1
        return super().generate_structured(request)


def inspect_local_gguf_brain_status(search_root: Optional[Path] = None) -> Dict[str, Any]:
    """Scan your machine for `.gguf` models (like `qwen3.5-9b-Q4_K_M.gguf`) and report ready commands."""
    bundles = discover_local_gguf_models(search_root=search_root)
    try:
        import llama_cpp  # type: ignore[import-not-found]  # noqa: F401

        llama_cpp_installed = True
    except ImportError:
        llama_cpp_installed = False

    if not bundles:
        return {
            "gguf_found": False,
            "llama_cpp_python_installed": llama_cpp_installed,
            "models": [],
        }

    primary = bundles[0]
    return {
        "gguf_found": True,
        "primary_model_name": primary.model_name,
        "primary_llm_gguf_path": str(primary.llm_gguf_path),
        "mmproj_vision_gguf_path": str(primary.mmproj_gguf_path) if primary.mmproj_gguf_path else None,
        "size_mb": primary.size_mb,
        "llama_cpp_python_installed": llama_cpp_installed,
        "ready_to_load_inprocess": llama_cpp_installed,
        "ollama_import_modelfile_preview": generate_ollama_modelfile(primary).strip(),
    }


if __name__ == "__main__":  # pragma: no cover
    print(json.dumps(inspect_local_gguf_brain_status(), indent=2))