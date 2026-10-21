"""NOVA Layered Configuration System (Layer 1 — Platform).

Implements the configuration specification from NOVA_ADR.md:
- Section 12.1: Typed config via Pydantic Settings, layered:
  (1) Built-in defaults -> (2) Config file (JSON/TOML) -> (3) Environment variables (NOVA_*) -> (4) Runtime overrides
- ADR-001: Offline-first by default; cloud inference and sensitive-data cloud routing require distinct opt-in.
- ADR-003 / ADR-004: Bounded graph execution steps and per-node ReAct reasoning loops.
- ADR-010 & Engineering Principle 4: Fail-closed permission enforcement and non-bypassable approval policy for sensitive actions.
- NFR-01, NFR-04, NFR-06, NFR-10, NFR-11: Latency targets, zero unconsented telemetry, <500MB idle memory cap, >=30 days audit retention, and hardware-bounded concurrency.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, Literal, Mapping, Optional

try:
    import tomllib  # Python 3.11+
except ImportError:  # pragma: no cover
    try:
        import tomli as tomllib  # type: ignore[no-redef]
    except ImportError:
        tomllib = None  # type: ignore[assignment]

from pydantic import BaseModel, Field

try:
    # Pydantic v2
    from pydantic import ConfigDict, field_validator, model_validator

    PYDANTIC_V2 = True
except ImportError:  # pragma: no cover
    # Pydantic v1 fallback for older test environments
    from pydantic import root_validator, validator

    PYDANTIC_V2 = False


EnvironmentType = Literal["development", "testing", "production"]
LocalModelProvider = Literal["llamacpp", "vllm", "ollama"]
VectorDBProvider = Literal["lancedb", "chroma"]
SensitiveActionPolicy = Literal["ask_every_time", "deny"]
LogLevel = Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]


class ConfigurationError(ValueError):
    """Raised when NOVA configuration violates an architectural or security invariant."""


class NovaSettings(BaseModel):
    """Canonical immutable runtime configuration for a single-user NOVA instance."""

    if PYDANTIC_V2:
        model_config = ConfigDict(frozen=True, extra="forbid", validate_default=True)
    else:

        class Config:
            allow_mutation = False
            extra = "forbid"
            validate_all = True

    # --- General / Environment ---
    app_name: str = Field(default="NOVA", description="Application identifier")
    version: str = Field(default="0.1.0", description="NOVA runtime version")
    environment: EnvironmentType = Field(
        default="development",
        description="Runtime execution profile (development, testing, production)",
    )
    log_level: LogLevel = Field(
        default="INFO",
        description="Structured logging verbosity level",
    )

    # --- ADR-001 & Section 16.2: Offline-First & Privacy Floor ---
    offline_by_default: bool = Field(
        default=True,
        description="ADR-001: Core reasoning, tools, and memory operate offline by default",
    )
    allow_cloud_inference: bool = Field(
        default=False,
        description="ADR-001: Explicit user opt-in required before routing any task to cloud models",
    )
    allow_cloud_sensitive_data: bool = Field(
        default=False,
        description=(
            "Section 16.2 Governance Principle: Distinct explicit consent required before "
            "routing Financial/Health/Legal data to any cloud model"
        ),
    )
    telemetry_enabled: bool = Field(
        default=False,
        description="NFR-04: Diagnostic export is strictly off by default (0 unconsented instances)",
    )

    # --- ADR-010 & Principle 4: Security-First Permission Model ---
    fail_closed_permissions: bool = Field(
        default=True,
        description="Engineering Principle 4: Ambiguous or unrecognized actions default to deny-and-ask",
    )
    default_sensitive_action_policy: SensitiveActionPolicy = Field(
        default="ask_every_time",
        description="ADR-010: Sensitive/irreversible actions must never default to auto_approve",
    )

    # --- ADR-003 & ADR-004: Bounded Orchestration & Reasoning ---
    max_graph_steps: int = Field(
        default=25,
        description="ADR-003: Hard upper bound on execution graph node transitions per task",
    )
    max_react_iterations_per_node: int = Field(
        default=5,
        description="ADR-004 / Phase 1: Maximum Thought-Action-Observation cycles inside a single agent node",
    )
    max_retries_per_node: int = Field(
        default=3,
        description="Phase 6: Maximum bounded retries for transient tool or model validation failures",
    )

    # --- NFRs: Resource, Latency, Concurrency & Audit Guarantees ---
    max_idle_memory_mb: int = Field(
        default=500,
        description="NFR-06: Idle memory footprint ceiling in MB (< 500MB RAM)",
    )
    audit_log_retention_days: int = Field(
        default=30,
        description="NFR-10: Minimum local retention window for decision and tool audit logs (>= 30 days)",
    )
    max_concurrent_agents: int = Field(
        default=4,
        description="NFR-11: Hardware-bounded local agent concurrency cap",
    )
    target_first_token_latency_ms_local: int = Field(
        default=800,
        description="NFR-01: Perceived first-token streaming latency target (local inference)",
    )
    target_first_token_latency_ms_cloud: int = Field(
        default=400,
        description="NFR-01: Perceived first-token streaming latency target (cloud-assisted)",
    )

    # --- Layer 1 & Layer 2: Local Storage & Model Serving Defaults ---
    data_dir: Path = Field(
        default=Path.home() / ".nova" / "data",
        description="Local root directory for SQLite, vector indexes, checkpoints, and audit logs",
    )
    sqlite_db_path: Optional[Path] = Field(
        default=None,
        description="Path to primary embedded SQLite database (defaults to <data_dir>/nova.db)",
    )
    vector_db_dir: Optional[Path] = Field(
        default=None,
        description="Path to embedded local vector database directory (defaults to <data_dir>/vectors)",
    )
    vector_db_provider: VectorDBProvider = Field(
        default="lancedb",
        description="Section 12.1: Embedded local vector store implementation (lancedb or chroma)",
    )
    local_model_provider: LocalModelProvider = Field(
        default="llamacpp",
        description="Section 12.1 / Phase 10: Local inference runtime adapter behind Model Router",
    )
    default_local_model_id: str = Field(
        default="qwen2.5-7b-instruct-q4_k_m.gguf",
        description="Default local quantized model identifier for bounded reasoning",
    )

    @property
    def resolved_sqlite_db_path(self) -> Path:
        """Return explicit sqlite_db_path or default inside data_dir."""
        return self.sqlite_db_path if self.sqlite_db_path is not None else (self.data_dir / "nova.db")

    @property
    def resolved_vector_db_dir(self) -> Path:
        """Return explicit vector_db_dir or default inside data_dir."""
        return self.vector_db_dir if self.vector_db_dir is not None else (self.data_dir / "vectors")

    @staticmethod
    def _validate_invariants(values: Dict[str, Any]) -> Dict[str, Any]:
        """Enforce non-negotiable ADR architectural and security constraints."""
        if not values.get("offline_by_default", True):
            raise ConfigurationError(
                "ADR-001 violation: 'offline_by_default' must remain True. "
                "Cloud capabilities are opt-in only."
            )

        if not values.get("fail_closed_permissions", True):
            raise ConfigurationError(
                "ADR-010 / Principle 4 violation: 'fail_closed_permissions' cannot be disabled."
            )

        policy = values.get("default_sensitive_action_policy", "ask_every_time")
        if policy not in ("ask_every_time", "deny"):
            raise ConfigurationError(
                f"ADR-010 violation: 'default_sensitive_action_policy' must be "
                f"'ask_every_time' or 'deny', got {policy!r}."
            )

        if values.get("allow_cloud_sensitive_data", False) and not values.get(
            "allow_cloud_inference", False
        ):
            raise ConfigurationError(
                "Section 16.2 Data Governance violation: 'allow_cloud_sensitive_data' "
                "cannot be True when 'allow_cloud_inference' is False."
            )

        max_graph_steps = values.get("max_graph_steps", 25)
        if not (1 <= int(max_graph_steps) <= 100):
            raise ConfigurationError(
                f"ADR-003 violation: 'max_graph_steps' must be bounded between 1 and 100 "
                f"(got {max_graph_steps})."
            )

        max_react = values.get("max_react_iterations_per_node", 5)
        if not (1 <= int(max_react) <= 15):
            raise ConfigurationError(
                f"ADR-004 violation: 'max_react_iterations_per_node' must be bounded between 1 and 15 "
                f"(got {max_react})."
            )

        max_retries = values.get("max_retries_per_node", 3)
        if not (0 <= int(max_retries) <= 5):
            raise ConfigurationError(
                f"Phase 6 violation: 'max_retries_per_node' must be between 0 and 5 "
                f"(got {max_retries})."
            )

        idle_mem = values.get("max_idle_memory_mb", 500)
        if not (64 <= int(idle_mem) <= 500):
            raise ConfigurationError(
                f"NFR-06 violation: 'max_idle_memory_mb' must be between 64 and 500 MB "
                f"(got {idle_mem})."
            )

        retention_days = values.get("audit_log_retention_days", 30)
        if int(retention_days) < 30:
            raise ConfigurationError(
                f"NFR-10 violation: 'audit_log_retention_days' must be at least 30 days "
                f"(got {retention_days})."
            )

        concurrency = values.get("max_concurrent_agents", 4)
        if not (1 <= int(concurrency) <= 16):
            raise ConfigurationError(
                f"NFR-11 violation: 'max_concurrent_agents' must be between 1 and 16 "
                f"(got {concurrency})."
            )

        return values

    if PYDANTIC_V2:

        @model_validator(mode="after")
        def _check_all_invariants(self) -> "NovaSettings":
            self._validate_invariants(self.__dict__)
            return self

    else:

        @root_validator(pre=False)
        def _check_all_invariants(cls, values: Dict[str, Any]) -> Dict[str, Any]:
            return cls._validate_invariants(values)


def _load_config_file(config_file: Optional[Path]) -> Dict[str, Any]:
    """Load configuration dictionary from a JSON or TOML file if provided."""
    if config_file is None:
        return {}
    path = Path(config_file)
    if not path.exists():
        raise ConfigurationError(f"Specified configuration file does not exist: {path}")

    suffix = path.suffix.lower()
    raw_text = path.read_text(encoding="utf-8")
    if suffix == ".json":
        data = json.loads(raw_text)
    elif suffix == ".toml":
        if tomllib is None:  # pragma: no cover
            raise ConfigurationError("TOML config parsing requires tomllib (3.11+) or tomli.")
        data = tomllib.loads(raw_text)
    else:
        raise ConfigurationError(
            f"Unsupported config file extension {suffix!r}. Use .json or .toml."
        )

    if not isinstance(data, dict):
        raise ConfigurationError("Configuration file root must be a key-value mapping.")
    return data


def _parse_env_value(raw: str) -> Any:
    """Coerce environment variable strings into boolean, integer, or string values."""
    normalized = raw.strip()
    lower = normalized.lower()
    if lower in ("true", "yes", "1", "on"):
        return True
    if lower in ("false", "no", "0", "off"):
        return False
    try:
        return int(normalized)
    except ValueError:
        return normalized


def _load_env_variables(
    environ: Optional[Mapping[str, str]] = None,
    prefix: str = "NOVA_",
) -> Dict[str, Any]:
    """Extract NOVA_* environment variables matching NovaSettings fields."""
    env_source = os.environ if environ is None else environ
    known_fields = (
        set(NovaSettings.model_fields.keys())
        if PYDANTIC_V2
        else set(NovaSettings.__fields__.keys())
    )
    extracted: Dict[str, Any] = {}
    for key, raw_val in env_source.items():
        if not key.startswith(prefix):
            continue
        field_name = key[len(prefix) :].lower()
        if field_name in known_fields:
            extracted[field_name] = _parse_env_value(raw_val)
    return extracted


def load_settings(
    config_file: Optional[Path] = None,
    environ: Optional[Mapping[str, str]] = None,
    runtime_overrides: Optional[Mapping[str, Any]] = None,
) -> NovaSettings:
    """Load NovaSettings using the ADR Section 12.1 4-layer hierarchy.

    Precedence (lowest to highest):
    1. Built-in field defaults
    2. Config file values (JSON or TOML)
    3. Environment variables (prefixed with NOVA_)
    4. Explicit runtime_overrides dictionary
    """
    merged: Dict[str, Any] = {}

    # Layer 2: Config file
    merged.update(_load_config_file(config_file))

    # Layer 3: Environment variables (NOVA_*)
    merged.update(_load_env_variables(environ=environ))

    # Layer 4: Runtime overrides
    if runtime_overrides:
        merged.update(dict(runtime_overrides))

    try:
        return NovaSettings(**merged)
    except Exception as exc:
        if isinstance(exc, ConfigurationError):
            raise
        raise ConfigurationError(f"Invalid NOVA configuration: {exc}") from exc