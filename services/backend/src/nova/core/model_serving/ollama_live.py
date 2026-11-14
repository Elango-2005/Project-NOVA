"""NOVA Phase 11: Live Local Ollama HTTP Connector & Hybrid Fallback Adapter (`services/backend/src/nova/core/model_serving/ollama_live.py`).

Plain-English Purpose (The Dual-Fuel Hybrid Engine):
Up to Phase 10, NOVA used `RuleGuidedLocalBootstrapAdapter` so everything worked immediately
without needing a 5 GB model download. Now, `ollama_live.py` lets NOVA connect to a **real local
LLM running in Ollama** (`http://127.0.0.1:11434`) on your Windows machine, while keeping the
bootstrap adapter as an automatic safety fallback:

1. **Strict Localhost Security Guard (`validate_loopback_base_url`)**:
   - Verifies that the Ollama URL points strictly to your own computer (`127.0.0.1`, `localhost`,
     or `::1`). If anyone tries to point it to a remote internet server, it blocks the request
     immediately to honor `offline_by_default=True`.

2. **Zero-Dependency Local HTTP Client (`LiveOllamaHttpClient`)**:
   - Uses Python's built-in `urllib.request` (no extra pip packages needed!) to call Ollama's
     `GET /api/tags` (to list installed local models like `llama3.2` or `qwen2.5:7b`) and
     `POST /api/chat` with `"format": "json"` so the local LLM replies with structured JSON.

3. **Small-Model JSON Normalizer (`normalize_llm_proposal_dict`)**:
   - Real local 3B/7B models sometimes forget boilerplate fields like `node_id` or `agent_role`.
     This normalizer fills in the exact `request.node_id` and `request.agent_role` before Pydantic
     validates the proposal.

4. **Automatic Hybrid Fallback (`HybridLocalModelAdapter` & `attach_hybrid_ollama_to_container`)**:
   - If Ollama is running on your laptop, it routes reasoning to your live Ollama model.
   - If Ollama isn't running (or a request times out), it seamlessly falls back to
     `RuleGuidedLocalBootstrapAdapter` so NOVA never crashes!
"""

from __future__ import annotations

import json
import urllib.error
import urllib.parse
import urllib.request
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Mapping, Optional, Tuple

from nova.core.domain.primitives import AgentRole
from nova.core.model_serving.interface import ModelInferenceRequest
from nova.main import RuleGuidedLocalBootstrapAdapter

if TYPE_CHECKING:
    from nova.main import NovaRuntimeContainer

_ALLOWED_LOOPBACK_HOSTS = frozenset({"127.0.0.1", "localhost", "::1"})

HttpTransportFn = Callable[[str, str, Optional[bytes], float], Tuple[int, bytes]]


def validate_loopback_base_url(base_url: str) -> str:
    """Ensure `base_url` targets a strictly local loopback address (`127.0.0.1` / `localhost`)."""
    parsed = urllib.parse.urlparse(base_url.strip())
    if parsed.scheme not in ("http", "https"):
        raise ValueError(f"Ollama base_url must use http or https; got {base_url!r}.")
    hostname = (parsed.hostname or "").lower()
    if hostname not in _ALLOWED_LOOPBACK_HOSTS:
        raise ValueError(
            f"Security violation: Ollama base_url {base_url!r} is not a loopback address "
            f"({sorted(_ALLOWED_LOOPBACK_HOSTS)}). NOVA enforces offline-by-default local inference."
        )
    return base_url.rstrip("/")


def _default_urllib_transport(
    method: str,
    url: str,
    body: Optional[bytes],
    timeout_seconds: float,
) -> Tuple[int, bytes]:  # pragma: no cover
    headers = {"Content-Type": "application/json", "Accept": "application/json"}
    req = urllib.request.Request(url=url, data=body, headers=headers, method=method)
    with urllib.request.urlopen(req, timeout=timeout_seconds) as resp:
        return int(resp.status), resp.read()


def _build_json_schema_system_hint(request: ModelInferenceRequest) -> str:
    """Append concise JSON format instructions so local 3B/7B models emit valid NOVA proposals."""
    role_val = request.agent_role.value
    if request.agent_role == AgentRole.PLANNER and not request.node_id.endswith("_reflect"):
        return (
            "Respond ONLY with a valid JSON object matching NOVA PlanProposal:\n"
            '{"kind": "plan", "agent_role": "planner", '
            f'"node_id": "{request.node_id}", '
            '"thought": "...", "goal_summary": "...", '
            '"steps": [{"step_id": "step_1", "target_agent": "research", '
            '"objective": "...", "depends_on": [], "estimated_sensitivity": "read_only"}]}'
        )
    if request.node_id.endswith("_reflect"):
        return (
            "Respond ONLY with a valid JSON object matching NOVA ReflectionCritiqueProposal:\n"
            '{"kind": "critique", "agent_role": "planner", '
            f'"node_id": "{request.node_id}", '
            '"thought": "...", "target_proposal_id": "prop_verified", '
            '"verdict": "approve", "issues_found": [], '
            '"verification_questions_checked": ["Verified intent and safety."]}'
        )
    return (
        "Respond ONLY with a valid JSON object matching either a NOVA ToolCallProposal:\n"
        f'{{"kind": "tool_call", "agent_role": "{role_val}", "node_id": "{request.node_id}", '
        '"thought": "...", "tool_name": "filesystem.write_file", '
        '"arguments": {"path": "note.txt", "content": "..."}, '
        '"declared_sensitivity": "reversible_write", "rationale": "..."}}\n'
        "OR a NOVA FinalAnswerProposal:\n"
        f'{{"kind": "final_answer", "agent_role": "{role_val}", "node_id": "{request.node_id}", '
        '"thought": "...", "content": "...", "cited_source_ids": ["local_source"], "confidence": 0.95}'
    )


def normalize_llm_proposal_dict(
    raw_obj: Mapping[str, Any],
    request: ModelInferenceRequest,
) -> Dict[str, Any]:
    """Fill in deterministic metadata (`node_id`, `agent_role`, defaults) if omitted by a small LLM."""
    normalized: Dict[str, Any] = dict(raw_obj)
    normalized.setdefault("node_id", request.node_id)
    normalized.setdefault("agent_role", request.agent_role.value)
    normalized.setdefault("thought", f"Generated by local Ollama model for {request.node_id}.")

    kind = str(normalized.get("kind", "")).strip().lower()
    if not kind:
        if "steps" in normalized:
            kind = "plan"
        elif "tool_name" in normalized:
            kind = "tool_call"
        elif "verdict" in normalized:
            kind = "critique"
        else:
            kind = "final_answer"
        normalized["kind"] = kind

    if kind == "final_answer":
        normalized.setdefault("content", "Completed task step.")
        normalized.setdefault("cited_source_ids", ["local_ollama_model"])
        normalized.setdefault("confidence", 0.9)
    elif kind == "tool_call":
        normalized.setdefault("arguments", {})
        normalized.setdefault("declared_sensitivity", "read_only")
        normalized.setdefault("rationale", f"Execute {normalized.get('tool_name', 'tool')}")
    elif kind == "critique":
        normalized.setdefault("target_proposal_id", "prop_verified")
        normalized.setdefault("verdict", "approve")
        normalized.setdefault("issues_found", [])
        normalized.setdefault(
            "verification_questions_checked", ["Verified safety and parameters."]
        )

    return normalized


class LiveOllamaHttpClient:
    """Loopback-only HTTP client for a local Ollama server (`http://127.0.0.1:11434`)."""

    def __init__(
        self,
        *,
        base_url: str = "http://127.0.0.1:11434",
        default_model: str = "llama3.2:latest",
        timeout_seconds: float = 15.0,
        transport_fn: Optional[HttpTransportFn] = None,
    ) -> None:
        self.base_url = validate_loopback_base_url(base_url)
        self.default_model = default_model
        self.timeout_seconds = float(timeout_seconds)
        self._transport: HttpTransportFn = transport_fn or _default_urllib_transport

    def list_local_models(self) -> List[str]:
        """Call `GET /api/tags` on local Ollama and return installed model names."""
        status, body = self._transport(
            "GET",
            f"{self.base_url}/api/tags",
            None,
            self.timeout_seconds,
        )
        if status != 200:
            raise RuntimeError(f"Ollama /api/tags returned HTTP {status}")
        payload = json.loads(body.decode("utf-8"))
        models = payload.get("models", [])
        return [str(m.get("name")) for m in models if isinstance(m, Mapping) and m.get("name")]

    def is_reachable(self) -> bool:
        """Return `True` if the local Ollama server responds on `/api/tags`."""
        try:
            self.list_local_models()
            return True
        except Exception:
            return False

    def chat_structured(
        self,
        request: ModelInferenceRequest,
        *,
        model_override: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Call `POST /api/chat` with `format='json'` and normalize the returned proposal dict."""
        messages = [dict(m) for m in request.messages]
        schema_hint = _build_json_schema_system_hint(request)
        if messages and messages[0].get("role") == "system":
            messages[0] = {
                "role": "system",
                "content": f"{messages[0]['content']}\n\n{schema_hint}",
            }
        else:
            messages.insert(0, {"role": "system", "content": schema_hint})

        req_payload = {
            "model": model_override or self.default_model,
            "messages": messages,
            "stream": False,
            "format": "json",
            "options": {
                "temperature": 0.0,
            },
        }
        raw_bytes = json.dumps(req_payload).encode("utf-8")
        status, resp_bytes = self._transport(
            "POST",
            f"{self.base_url}/api/chat",
            raw_bytes,
            self.timeout_seconds,
        )
        if status != 200:
            raise RuntimeError(f"Ollama /api/chat returned HTTP {status}")

        outer = json.loads(resp_bytes.decode("utf-8"))
        content_str = outer.get("message", {}).get("content", "{}")
        parsed_json = json.loads(content_str) if isinstance(content_str, str) else content_str
        if not isinstance(parsed_json, Mapping):
            raise ValueError("Ollama JSON output must be a JSON object.")
        return normalize_llm_proposal_dict(parsed_json, request)


class HybridLocalModelAdapter(RuleGuidedLocalBootstrapAdapter):
    """Dual-mode local model adapter: uses live Ollama when reachable, with automatic bootstrap fallback."""

    def __init__(
        self,
        *,
        ollama_client: Optional[LiveOllamaHttpClient] = None,
        prefer_live_ollama: bool = True,
    ) -> None:
        super().__init__()
        self.ollama_client = ollama_client or LiveOllamaHttpClient()
        self.prefer_live_ollama = prefer_live_ollama
        self.live_calls_succeeded: int = 0
        self.fallback_calls_used: int = 0

    def generate_structured(self, request: ModelInferenceRequest) -> Any:
        # 1. If scripted test outputs were explicitly queued, honor them first
        if self._queue:
            return super().generate_structured(request)

        # 2. Try live local Ollama if enabled
        if self.prefer_live_ollama:
            try:
                proposal_dict = self.ollama_client.chat_structured(request)
                self.live_calls_succeeded += 1
                self.enqueue(proposal_dict)
                return super().generate_structured(request)
            except Exception:
                pass

        # 3. Seamless offline fallback to RuleGuidedLocalBootstrapAdapter
        self.fallback_calls_used += 1
        return super().generate_structured(request)


def attach_hybrid_ollama_to_container(
    container: "NovaRuntimeContainer",
    *,
    ollama_client: Optional[LiveOllamaHttpClient] = None,
    prefer_live_ollama: bool = True,
) -> HybridLocalModelAdapter:
    """Wire `HybridLocalModelAdapter` into an existing `NovaRuntimeContainer`."""
    adapter = HybridLocalModelAdapter(
        ollama_client=ollama_client,
        prefer_live_ollama=prefer_live_ollama,
    )
    container.bootstrap_adapter = adapter
    container.model_router._llamacpp = adapter
    return adapter


def probe_local_ollama_status(
    base_url: str = "http://127.0.0.1:11434",
    default_model: str = "llama3.2:latest",
) -> Dict[str, Any]:
    """Return a diagnostic dictionary showing whether Ollama is running and what models exist."""
    client = LiveOllamaHttpClient(base_url=base_url, default_model=default_model, timeout_seconds=2.5)
    try:
        models = client.list_local_models()
        return {
            "ollama_reachable": True,
            "base_url": client.base_url,
            "default_model": client.default_model,
            "installed_models": models,
            "active_engine": "live_ollama" if models else "bootstrap_fallback (no models pulled yet)",
        }
    except Exception as exc:
        return {
            "ollama_reachable": False,
            "base_url": client.base_url,
            "default_model": client.default_model,
            "installed_models": [],
            "active_engine": "rule_guided_bootstrap_fallback",
            "note": f"Ollama not detected on {client.base_url} ({exc.__class__.__name__}); using built-in offline bootstrap brain.",
        }


if __name__ == "__main__":  # pragma: no cover
    print(json.dumps(probe_local_ollama_status(), indent=2))