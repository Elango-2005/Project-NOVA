"""Unit and integration tests for NOVA Phase 14: VRAM-Aware GPU Layer Calculator & Live Brain Switcher (`test_brain_switcher.py`).

Verifies:
1. `compute_optimal_gguf_gpu_plan` calculates a safe `hybrid_gpu_cpu_split` (`29/36` layers on GPU,
   `7` layers on CPU) for a `5,883.54 MB` (`qwen3.5-9b-Q4_K_M.gguf`) model on a `6.0 GB` VRAM GPU
   so it never triggers a CUDA Out-Of-Memory crash.
2. `compute_optimal_gguf_gpu_plan` selects `full_gpu` (`36/36` layers) when given a `12.0 GB` GPU
   and `cpu_only` (`0/36` layers) when `gpu_vram_gb=0.0`.
3. `NovaBrainAndSecurityController` switches live between `BOOTSTRAP`, `OLLAMA_LIVE`, and
   `GGUF_QWEN35` modes and executes tasks through `NovaRuntimeContainer`.
4. `get_security_and_brain_status()` seals live `ToolAuditEntry` records into Phase 13's
   `TamperEvidentAuditChain` and reports `audit_chain_valid=True`.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

from nova.api.app import StartTaskRequest
from nova.core.domain.primitives import TaskStatus
from nova.core.model_serving.brain_switcher import (
    BrainEngineMode,
    NovaBrainAndSecurityController,
    compute_optimal_gguf_gpu_plan,
)
from nova.main import NovaRuntimeContainer


class TestPhase14BrainSwitcherAndGpuPlanner:
    """Verification suite for Phase 14 (`services/backend/src/nova/core/model_serving/brain_switcher.py`)."""

    def test_6gb_vram_gpu_plan_splits_5883mb_qwen35_layers_safely(self) -> None:
        """Verifies a 6.0 GB VRAM GPU offloads ~29/36 layers of 5.88 GB Qwen 3.5 9B without overflowing VRAM."""
        plan_6gb = compute_optimal_gguf_gpu_plan(
            model_size_mb=5883.54,
            gpu_vram_gb=6.0,
            total_layers=36,
        )
        assert plan_6gb.offload_mode == "hybrid_gpu_cpu_split"
        assert 25 <= plan_6gb.recommended_gpu_layers < 36
        assert plan_6gb.recommended_gpu_layers + plan_6gb.cpu_fallback_layers == 36
        assert plan_6gb.recommended_ctx_tokens == 4096
        assert f"PARAMETER num_gpu {plan_6gb.recommended_gpu_layers}" in plan_6gb.ollama_modelfile

    def test_gpu_plan_handles_12gb_full_gpu_and_0gb_cpu_only(self) -> None:
        """Verifies 12 GB VRAM gets full 36-layer GPU offload and 0 GB VRAM gets CPU-only mode."""
        plan_12gb = compute_optimal_gguf_gpu_plan(
            model_size_mb=5883.54,
            gpu_vram_gb=12.0,
            total_layers=36,
        )
        assert plan_12gb.offload_mode == "full_gpu"
        assert plan_12gb.recommended_gpu_layers == 36
        assert plan_12gb.cpu_fallback_layers == 0

        plan_cpu = compute_optimal_gguf_gpu_plan(
            model_size_mb=5883.54,
            gpu_vram_gb=0.0,
            total_layers=36,
        )
        assert plan_cpu.offload_mode == "cpu_only"
        assert plan_cpu.recommended_gpu_layers == 0
        assert plan_cpu.cpu_fallback_layers == 36

    def test_brain_switcher_toggles_between_bootstrap_ollama_and_gguf_modes(
        self,
    ) -> None:
        """Verifies live brain switching across BOOTSTRAP, OLLAMA_LIVE, and GGUF_QWEN35 modes."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            fake_hf = tmp_path / "hf_cache"
            fake_hf.mkdir(parents=True, exist_ok=True)
            fake_gguf = fake_hf / "qwen3.5-9b-Q4_K_M.gguf"
            fake_gguf.write_bytes(b"GGUF_WEIGHTS" * 200)

            container = NovaRuntimeContainer(
                data_dir=tmp_path / "data",
                workspace_dir=tmp_path / "workspace",
            )
            controller = NovaBrainAndSecurityController(
                container,
                gpu_vram_gb=6.0,
                search_root=fake_hf,
            )

            # 1. Initial mode is BOOTSTRAP and GGUF is discovered
            status_init = controller.get_security_and_brain_status()
            assert status_init["active_brain_mode"] == "bootstrap"
            assert status_init["gguf_discovered"] is True
            assert status_init["gguf_model_name"] == "qwen3.5-9b-Q4_K_M"

            # 2. Switch to OLLAMA_LIVE
            status_ollama = controller.switch_brain_mode(
                BrainEngineMode.OLLAMA_LIVE,
                ollama_model="qwen3.5-9b",
            )
            assert status_ollama["active_brain_mode"] == "ollama_live"

            # 3. Switch to GGUF_QWEN35 and run a real task (falls back cleanly if llama-server isn't up)
            status_gguf = controller.switch_brain_mode(BrainEngineMode.GGUF_QWEN35)
            assert status_gguf["active_brain_mode"] == "gguf_qwen35"

            resp = container.api_service.start_task(
                StartTaskRequest(
                    task_id="task_phase14_brain",
                    session_id="sess_phase14",
                    user_goal="Write Phase 14 brain test output to brain_switched.txt",
                )
            )
            assert resp.status == TaskStatus.COMPLETED
            assert (container.workspace_dir / "brain_switched.txt").exists()

    def test_security_telemetry_seals_and_verifies_audit_chain_after_task(
        self,
    ) -> None:
        """Confirms get_security_and_brain_status seals audit entries and reports audit_chain_valid=True."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            container = NovaRuntimeContainer(
                data_dir=tmp_path / "data",
                workspace_dir=tmp_path / "workspace",
            )
            controller = NovaBrainAndSecurityController(
                container,
                gpu_vram_gb=6.0,
                require_token=True,
            )

            container.api_service.start_task(
                StartTaskRequest(
                    task_id="task_phase14_sec",
                    session_id="sess_phase14_sec",
                    user_goal="Save security telemetry check to sec_check.txt",
                )
            )

            telemetry = controller.get_security_and_brain_status()["security_telemetry"]
            assert telemetry["token_auth_enabled"] is True
            assert telemetry["audit_chain_valid"] is True
            assert telemetry["sealed_audit_entries"] == 1
            assert len(telemetry["head_seal_hash"]) == 64