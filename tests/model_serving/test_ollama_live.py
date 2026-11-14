"""Unit and integration tests for NOVA Phase 11: Live Local Ollama Connector & Hybrid Fallback (`test_ollama_live.py`).

Verifies:
1. `validate_loopback_base_url` allows `http://127.0.0.1:11434` and `http://localhost:11434`, and
   strictly blocks remote/cloud URLs (e.g., `https://api.openai.com`) to enforce `offline_by_default`.
2. `LiveOllamaHttpClient` calls `/api/tags` and `/api/chat` (`format="json"`) and normalizes partial
   small-model JSON outputs into schema-valid NOVA proposals (`PlanProposal`, `ToolCallProposal`,
   `FinalAnswerProposal`).
3. `HybridLocalModelAdapter` routes inference to `LiveOllamaHttpClient` when Ollama responds, and
   automatically falls back to `RuleGuidedLocalBootstrapAdapter` when Ollama is offline.
4. End-to-end task execution with `NovaRuntimeContainer` wired via `attach_hybrid_ollama_to_container`.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Optional, Tuple

from nova.api.app import StartTaskRequest
from nova.core.domain.primitives import AgentRole, TaskStatus
from nova.core.model_serving.interface import ModelInferenceRequest
from nova.core.model_serving.ollama_live import (
    HybridLocalModelAdapter,
    LiveOllamaHttpClient,
    attach_hybrid_ollama_to_container,
    validate_loopback_base_url,
)
from nova.main import NovaRuntimeContainer


def _make_sample_request(
    role: AgentRole = AgentRole.CODING,
    node_id: str = "step_write_file",
    objective: str = "Write summary to hello.txt",
) -> ModelInferenceRequest:
    return ModelInferenceRequest(
        task_id="task_ollama_01",
        session_id="sess_ollama_01",
        agent_role=role,
        node_id=node_id,
        messages=(
            {
                "role": "system",
                "content": f"[NOVA TRUSTED SYSTEM INSTRUCTION]\nTask Objective: {objective}",
            },
        ),
        output_schema_name="AgentProposal",
    )


class TestLiveOllamaConnectorAndHybridFallback:
    """Verification suite for Phase 11 (`services/backend/src/nova/core/model_serving/ollama_live.py`)."""

    def test_loopback_security_guard_blocks_non_local_urls(self) -> None:
        """Ensures NOVA never sends prompts to an external internet host via LiveOllamaHttpClient."""
        assert validate_loopback_base_url("http://127.0.0.1:11434/") == "http://127.0.0.1:11434"
        assert validate_loopback_base_url("http://localhost:11434") == "http://localhost:11434"

        for bad_url in ("https://api.openai.com/v1", "http://192.168.1.50:11434"):
            try:
                LiveOllamaHttpClient(base_url=bad_url)
                raise AssertionError(f"Expected ValueError for non-loopback URL: {bad_url}")
            except ValueError as exc:
                assert "Security violation" in str(exc)

    def test_live_ollama_http_client_parses_tags_and_normalizes_chat_json(self) -> None:
        """Simulates Ollama `/api/tags` and `/api/chat` HTTP responses and verifies normalization."""

        def fake_ollama_transport(
            method: str,
            url: str,
            body: Optional[bytes],
            timeout: float,
        ) -> Tuple[int, bytes]:
            if method == "GET" and url.endswith("/api/tags"):
                return 200, json.dumps(
                    {"models": [{"name": "llama3.2:latest"}, {"name": "qwen2.5:7b"}]}
                ).encode("utf-8")

            if method == "POST" and url.endswith("/api/chat"):
                assert body is not None
                req_json = json.loads(body.decode("utf-8"))
                assert req_json["format"] == "json"
                assert req_json["stream"] is False
                # Simulate a small 3B model returning a tool_call without `node_id` or `agent_role`
                llm_reply = {
                    "kind": "tool_call",
                    "tool_name": "filesystem.write_file",
                    "arguments": {"path": "hello.txt", "content": "Hello from local Ollama!"},
                    "declared_sensitivity": "reversible_write",
                }
                return 200, json.dumps(
                    {"message": {"role": "assistant", "content": json.dumps(llm_reply)}}
                ).encode("utf-8")

            raise AssertionError(f"Unexpected call: {method} {url}")

        client = LiveOllamaHttpClient(transport_fn=fake_ollama_transport)
        assert client.is_reachable() is True
        assert client.list_local_models() == ["llama3.2:latest", "qwen2.5:7b"]

        req = _make_sample_request()
        normalized = client.chat_structured(req)
        assert normalized["kind"] == "tool_call"
        assert normalized["node_id"] == "step_write_file"
        assert normalized["agent_role"] == "coding"
        assert normalized["arguments"]["path"] == "hello.txt"

    def test_hybrid_local_model_adapter_uses_ollama_when_up_and_falls_back_when_down(
        self,
    ) -> None:
        """Verifies HybridLocalModelAdapter uses Ollama when reachable and falls back cleanly when offline."""
        ollama_online = {"up": True}

        def toggle_transport(
            method: str,
            url: str,
            body: Optional[bytes],
            timeout: float,
        ) -> Tuple[int, bytes]:
            if not ollama_online["up"]:
                raise ConnectionRefusedError("Ollama server not running on 127.0.0.1:11434")
            llm_reply = {
                "kind": "final_answer",
                "content": "Answered by live local Ollama model.",
            }
            return 200, json.dumps(
                {"message": {"role": "assistant", "content": json.dumps(llm_reply)}}
            ).encode("utf-8")

        client = LiveOllamaHttpClient(transport_fn=toggle_transport)
        hybrid = HybridLocalModelAdapter(ollama_client=client)

        # 1. When Ollama is online -> uses live Ollama
        req = _make_sample_request(role=AgentRole.RESEARCH, node_id="step_research_info")
        resp_online = hybrid.generate_structured(req)
        assert hybrid.live_calls_succeeded == 1
        assert hybrid.fallback_calls_used == 0
        assert "Answered by live local Ollama model." in str(resp_online.raw_output)

        # 2. When Ollama is stopped -> automatically falls back to RuleGuidedLocalBootstrapAdapter!
        ollama_online["up"] = False
        resp_fallback = hybrid.generate_structured(req)
        assert hybrid.live_calls_succeeded == 1
        assert hybrid.fallback_calls_used == 1
        assert "tool_call" in str(resp_fallback.raw_output)

    def test_runtime_container_executes_full_task_with_hybrid_ollama_adapter(
        self,
    ) -> None:
        """Swaps HybridLocalModelAdapter into NovaRuntimeContainer and completes a live file task."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            container = NovaRuntimeContainer(
                data_dir=tmp_path / "data",
                workspace_dir=tmp_path / "workspace",
            )

            # Configure HybridLocalModelAdapter with an offline Ollama port so it exercises fallback
            offline_client = LiveOllamaHttpClient(
                transport_fn=lambda m, u, b, t: (_ for _ in ()).throw(
                    ConnectionRefusedError("Simulated offline Ollama")
                )
            )
            hybrid_adapter = attach_hybrid_ollama_to_container(
                container,
                ollama_client=offline_client,
            )

            resp = container.api_service.start_task(
                StartTaskRequest(
                    task_id="task_hybrid_ollama",
                    session_id="sess_hybrid_01",
                    user_goal="Save hybrid adapter test output to hybrid_note.txt",
                )
            )
            assert resp.status == TaskStatus.COMPLETED
            assert (container.workspace_dir / "hybrid_note.txt").exists()
            assert hybrid_adapter.fallback_calls_used >= 2