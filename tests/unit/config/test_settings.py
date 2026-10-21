"""Unit tests for NOVA Layered Configuration System (Task 1.1).

Verifies:
- ADR-001: Offline-by-default and opt-in cloud constraints
- ADR-003 / ADR-004: Bounded execution graph and ReAct step limits
- ADR-010 & Principle 4: Fail-closed permissions and sensitive-action policy bounds
- Section 12.1: 4-layer precedence (defaults -> file -> env -> runtime override)
- Section 16.2: Sensitive data cloud routing requires general cloud inference opt-in
- NFR-04, NFR-06, NFR-10, NFR-11: Telemetry, memory, audit retention, and concurrency invariants
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from nova.core.config.settings import ConfigurationError, NovaSettings, load_settings


class TestNovaSettings(unittest.TestCase):
    """Deterministic unit test suite for NovaSettings and load_settings()."""

    def test_default_settings_enforce_offline_and_security_floors(self) -> None:
        """Default configuration must satisfy ADR-001, ADR-010, and NFR floors."""
        settings = load_settings(environ={})
        self.assertTrue(settings.offline_by_default)
        self.assertFalse(settings.allow_cloud_inference)
        self.assertFalse(settings.allow_cloud_sensitive_data)
        self.assertFalse(settings.telemetry_enabled)
        self.assertTrue(settings.fail_closed_permissions)
        self.assertEqual(settings.default_sensitive_action_policy, "ask_every_time")
        self.assertEqual(settings.max_graph_steps, 25)
        self.assertEqual(settings.max_react_iterations_per_node, 5)
        self.assertEqual(settings.max_idle_memory_mb, 500)
        self.assertEqual(settings.audit_log_retention_days, 30)
        self.assertEqual(settings.local_model_provider, "llamacpp")
        self.assertEqual(settings.vector_db_provider, "lancedb")

    def test_layered_precedence_defaults_file_env_runtime(self) -> None:
        """Section 12.1: defaults < file < env < runtime override."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            cfg_path = Path(tmp_dir) / "nova_config.json"
            cfg_path.write_text(
                json.dumps(
                    {
                        "max_graph_steps": 40,
                        "max_react_iterations_per_node": 6,
                        "audit_log_retention_days": 45,
                        "local_model_provider": "vllm",
                    }
                ),
                encoding="utf-8",
            )

            env = {
                "NOVA_MAX_REACT_ITERATIONS_PER_NODE": "8",
                "NOVA_AUDIT_LOG_RETENTION_DAYS": "60",
            }
            runtime = {
                "audit_log_retention_days": 90,
            }

            settings = load_settings(
                config_file=cfg_path,
                environ=env,
                runtime_overrides=runtime,
            )

            # From file (overrides default 25 and llamacpp)
            self.assertEqual(settings.max_graph_steps, 40)
            self.assertEqual(settings.local_model_provider, "vllm")
            # From env (overrides file 6)
            self.assertEqual(settings.max_react_iterations_per_node, 8)
            # From runtime override (overrides env 60 and file 45)
            self.assertEqual(settings.audit_log_retention_days, 90)

    def test_toml_config_file_loading(self) -> None:
        """TOML configuration files must load cleanly in layer 2."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            cfg_path = Path(tmp_dir) / "nova_config.toml"
            cfg_path.write_text(
                'environment = "production"\nmax_concurrent_agents = 8\n',
                encoding="utf-8",
            )
            settings = load_settings(config_file=cfg_path, environ={})
            self.assertEqual(settings.environment, "production")
            self.assertEqual(settings.max_concurrent_agents, 8)

    def test_rejects_disabling_fail_closed_permissions(self) -> None:
        """ADR-010 / Engineering Principle 4: fail_closed_permissions cannot be False."""
        with self.assertRaises(ConfigurationError):
            load_settings(
                environ={},
                runtime_overrides={"fail_closed_permissions": False},
            )

    def test_rejects_disabling_offline_by_default(self) -> None:
        """ADR-001: offline_by_default cannot be False."""
        with self.assertRaises(ConfigurationError):
            load_settings(
                environ={},
                runtime_overrides={"offline_by_default": False},
            )

    def test_rejects_auto_approve_for_sensitive_actions(self) -> None:
        """ADR-010: Sensitive/irreversible actions can never default to auto_approve."""
        with self.assertRaises(ConfigurationError):
            load_settings(
                environ={},
                runtime_overrides={"default_sensitive_action_policy": "auto_approve"},
            )

    def test_rejects_sensitive_cloud_data_without_general_cloud_opt_in(self) -> None:
        """Section 16.2 Governance Principle: sensitive cloud opt-in requires cloud inference enabled."""
        with self.assertRaises(ConfigurationError):
            load_settings(
                environ={},
                runtime_overrides={
                    "allow_cloud_inference": False,
                    "allow_cloud_sensitive_data": True,
                },
            )

        # Valid when both are explicitly enabled
        valid = load_settings(
            environ={},
            runtime_overrides={
                "allow_cloud_inference": True,
                "allow_cloud_sensitive_data": True,
            },
        )
        self.assertTrue(valid.allow_cloud_inference)
        self.assertTrue(valid.allow_cloud_sensitive_data)

    def test_enforces_nfr_bounds_at_startup(self) -> None:
        """NFR-06 (<500MB idle RAM), NFR-10 (>=30 days audit retention), ADR-003/004 bounds."""
        with self.assertRaises(ConfigurationError):
            load_settings(environ={}, runtime_overrides={"max_idle_memory_mb": 600})

        with self.assertRaises(ConfigurationError):
            load_settings(environ={}, runtime_overrides={"audit_log_retention_days": 14})

        with self.assertRaises(ConfigurationError):
            load_settings(environ={}, runtime_overrides={"max_graph_steps": 0})

        with self.assertRaises(ConfigurationError):
            load_settings(environ={}, runtime_overrides={"max_react_iterations_per_node": 50})

    def test_settings_instance_is_immutable(self) -> None:
        """Loaded settings must be frozen so no agent or tool can mutate security policy at runtime."""
        settings = load_settings(environ={})
        with self.assertRaises(Exception):
            settings.allow_cloud_inference = True  # type: ignore[misc]

    def test_resolved_storage_paths(self) -> None:
        """Default SQLite and vector DB paths resolve inside data_dir."""
        custom_dir = Path("/tmp/nova_test_data")
        settings = load_settings(environ={}, runtime_overrides={"data_dir": custom_dir})
        self.assertEqual(settings.resolved_sqlite_db_path, custom_dir / "nova.db")
        self.assertEqual(settings.resolved_vector_db_dir, custom_dir / "vectors")


if __name__ == "__main__":
    unittest.main()