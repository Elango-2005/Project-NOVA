"""NOVA Core Domain Primitives (Layer 5/6 Domain Foundation).

Defines canonical enumerations, value types, and architectural invariants from NOVA_ADR.md:
- Engineering Principle 7: Versioned schema contracts (SemVer)
- ADR-007 / Phase 5: Two-tier memory partitions (working, semantic, episodic, procedural)
- ADR-008 / Section 12.3: Specialized Agent Roster roles
- ADR-010 / Phase 7 / Section 14: Action Sensitivity Tiers & Permission Decisions
- Phase 4: Coarse Task Lifecycle States
- Section 16.2: Data Classification & Governance tiers
"""

from __future__ import annotations

import re
from enum import Enum

SEMVER_PATTERN = re.compile(r"^(0|[1-9]\d*)\.(0|[1-9]\d*)\.(0|[1-9]\d*)$")
CURRENT_SCHEMA_VERSION = "1.0.0"


def validate_schema_version(version: str) -> str:
    """Validate that a schema version string follows strict major.minor.patch SemVer."""
    cleaned = version.strip()
    if not SEMVER_PATTERN.match(cleaned):
        raise ValueError(
            f"Engineering Principle 7 violation: schema_version must be SemVer 'X.Y.Z', got {version!r}."
        )
    return cleaned


class AgentRole(str, Enum):
    """Canonical specialized agent roles defined in NOVA_ADR.md Section 12.3 (ADR-008)."""

    PLANNER = "planner"
    RESEARCH = "research"
    BROWSER = "browser"
    DESKTOP = "desktop"
    CODING = "coding"
    COMMS = "comms"
    VOICE = "voice"


class MemoryPartition(str, Enum):
    """Canonical memory partitions defined in NOVA_ADR.md ADR-007 and Phase 5."""

    WORKING = "working"
    SEMANTIC = "semantic"
    EPISODIC = "episodic"
    PROCEDURAL = "procedural"


class TaskStatus(str, Enum):
    """Coarse task lifecycle state machine states defined in NOVA_ADR.md Phase 4."""

    QUEUED = "queued"
    PLANNING = "planning"
    RUNNING = "running"
    AWAITING_APPROVAL = "awaiting_approval"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class SensitivityTier(str, Enum):
    """Tool and action sensitivity tiers defined in NOVA_ADR.md ADR-010, Phase 7, and Section 14."""

    READ_ONLY = "read_only"
    READ_ONLY_NETWORKED = "read_only_networked"
    REVERSIBLE_WRITE = "reversible_write"
    REVERSIBLE_WRITE_NETWORKED = "reversible_write_networked"
    IRREVERSIBLE_WRITE = "irreversible_write"

    @property
    def is_write(self) -> bool:
        """Return True if the action modifies local or remote state."""
        return self in (
            SensitivityTier.REVERSIBLE_WRITE,
            SensitivityTier.REVERSIBLE_WRITE_NETWORKED,
            SensitivityTier.IRREVERSIBLE_WRITE,
        )

    @property
    def is_irreversible(self) -> bool:
        """Return True if the action cannot be automatically rolled back (e.g., send email, delete, shell)."""
        return self is SensitivityTier.IRREVERSIBLE_WRITE

    @property
    def is_networked(self) -> bool:
        """Return True if the action communicates over a network boundary."""
        return self in (
            SensitivityTier.READ_ONLY_NETWORKED,
            SensitivityTier.REVERSIBLE_WRITE_NETWORKED,
        )

    @property
    def requires_human_approval_by_default(self) -> bool:
        """ADR-010: Sensitive/irreversible actions and networked writes require confirmation by default."""
        return self in (
            SensitivityTier.REVERSIBLE_WRITE_NETWORKED,
            SensitivityTier.IRREVERSIBLE_WRITE,
        )


class DataClassification(str, Enum):
    """Explicit data governance classes defined in NOVA_ADR.md Section 16.2."""

    PUBLIC = "public"
    USER_CONTENT_LOCAL = "user_content_local"
    USER_CONTENT_CONNECTED = "user_content_connected"
    CREDENTIALS_SECRETS = "credentials_secrets"
    SENSITIVE_CATEGORY = "sensitive_category"
    SYSTEM_DIAGNOSTIC = "system_diagnostic"


class PermissionDecisionKind(str, Enum):
    """Outcome of a Permission Gate evaluation (ADR-010 / NFR-08)."""

    AUTO_APPROVED = "auto_approved"
    USER_APPROVED = "user_approved"
    DENIED_BY_POLICY = "denied_by_policy"
    DENIED_BY_USER = "denied_by_user"
    DENIED_OUT_OF_SCOPE = "denied_out_of_scope"