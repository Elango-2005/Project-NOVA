"""Unit tests for NOVA Layer 3 MCP-First Tool Registry & Schema Validator (Task 4.1).

Verifies:
1. Every tool across all 7 specialized agents in `INITIAL_AGENT_ROSTER` (Section 12.3) is registered
   in the default Section 14 MCP/native catalog with an explicit sensitivity tier (ADR-005 / ADR-010).
2. JSON Schema argument and output validation deterministically rejects missing required parameters,
   undeclared extra arguments, and wrong primitive types (including `bool` masquerading as `int`).
3. Offline-first MCP transport security blocks remote `HTTP_SSE` MCP servers unless both global and
   per-server opt-in are enabled (ADR-001 / Phase 3), and blocks tampered server packages via SHA-256
   checksum verification (Phase 15 / Risk R-05).
4. Agent-scoped tool discovery (`list_tools_for_agent` and `discover_tools`) respects ADR-008
   least-privilege boundaries.
"""

from __future__ import annotations

import hashlib
import pytest

from nova.core.domain.agent_spec import INITIAL_AGENT_ROSTER
from nova.core.domain.primitives import AgentRole, SensitivityTier
from nova.core.tools.registry import (
    JsonSchemaPrimitiveType,
    McpServerManifest,
    McpSupplyChainIntegrityError,
    McpTransportKind,
    McpTransportSecurityError,
    ToolContract,
    ToolNotFoundError,
    ToolParameterSpec,
    ToolRegistry,
    ToolSchemaValidationError,
    ToolSourceKind,
    build_default_tool_registry,
)


class TestMcpToolRegistryAndValidator:
    """Verification suite for Task 4.1 (`registry.py`)."""

    def test_default_catalog_covers_entire_agent_roster_and_section14_tiers(self) -> None:
        """Every tool allowed for every agent in Section 12.3 must exist in the default catalog."""
        registry = build_default_tool_registry()

        for role, spec in INITIAL_AGENT_ROSTER.items():
            agent_tools = registry.list_tools_for_agent(role)
            registered_names = {t.tool_name for t in agent_tools}
            assert registered_names == set(spec.allowed_tools), (
                f"Agent {role.value} has tools missing from default catalog!"
            )

        # Spot-check Section 14 sensitivity tiers (pp. 50-52)
        assert (
            registry.get_tool("filesystem.read_file").sensitivity_tier
            == SensitivityTier.READ_ONLY
        )
        assert (
            registry.get_tool("filesystem.write_file").sensitivity_tier
            == SensitivityTier.REVERSIBLE_WRITE
        )
        assert (
            registry.get_tool("gmail.send_email").sensitivity_tier
            == SensitivityTier.IRREVERSIBLE_WRITE
        )
        assert (
            registry.get_tool("terminal.execute").sensitivity_tier
            == SensitivityTier.IRREVERSIBLE_WRITE
        )

    def test_json_schema_argument_and_output_validation_is_strict(self) -> None:
        """Blueprint Inspector must reject missing args, extra args, and wrong types (Phase 7)."""
        registry = build_default_tool_registry()

        # 1. Valid call passes and returns the contract
        contract = registry.validate_tool_call(
            "github.read_issue",
            {"repo": "nova/core", "issue_number": 42},
        )
        assert contract.tool_name == "github.read_issue"
        schema = contract.to_json_schema()
        assert schema["inputSchema"]["additionalProperties"] is False

        # 2. Missing required parameter -> blocked
        with pytest.raises(ToolSchemaValidationError, match="missing required parameter"):
            registry.validate_tool_call("github.read_issue", {"repo": "nova/core"})

        # 3. Undeclared extra parameter -> blocked
        with pytest.raises(ToolSchemaValidationError, match="undeclared argument"):
            registry.validate_tool_call(
                "github.read_issue",
                {"repo": "nova/core", "issue_number": 42, "inject_flag": True},
            )

        # 4. Wrong type (passing a boolean `True` where an integer is required) -> blocked
        with pytest.raises(ToolSchemaValidationError, match="must be of type 'integer'"):
            registry.validate_tool_call(
                "github.read_issue",
                {"repo": "nova/core", "issue_number": True},
            )

        # 5. Output schema validation enforces required output keys
        contract.validate_output({"title": "Bug report", "body": "Steps to reproduce"})
        with pytest.raises(ToolSchemaValidationError, match="missing required key"):
            contract.validate_output({"title": "Bug report"})

    def test_blocks_unconsented_remote_http_sse_server_and_tampered_package(self) -> None:
        """Enforces ADR-001/Phase 3 remote MCP opt-in and Phase 15 SHA-256 supply-chain check."""
        trusted_bytes = b"mcp-remote-postgres-v1.0.0-binary"
        expected_sha = hashlib.sha256(trusted_bytes).hexdigest()

        remote_manifest = McpServerManifest(
            server_id="postgresql",
            display_name="PostgreSQL Remote MCP Server",
            purpose="Query self-hosted PostgreSQL databases over HTTP+SSE.",
            source_kind=ToolSourceKind.MCP,
            transport=McpTransportKind.HTTP_SSE,
            remote_opt_in_enabled=False,
            package_sha256=expected_sha,
            tools=(
                ToolContract(
                    tool_name="postgresql.query",
                    server_id="postgresql",
                    description="Run a SQL query against a remote PostgreSQL database.",
                    sensitivity_tier=SensitivityTier.REVERSIBLE_WRITE_NETWORKED,
                    parameters=(
                        ToolParameterSpec(
                            name="sql",
                            json_type=JsonSchemaPrimitiveType.STRING,
                            description="SQL statement",
                        ),
                    ),
                    required_output_keys=("rows",),
                ),
            ),
        )

        # 1. Default offline-first registry rejects HTTP_SSE server without opt-in
        offline_registry = ToolRegistry(allow_remote_mcp_globally=False)
        with pytest.raises(McpTransportSecurityError, match="Remote HTTP\\+SSE MCP server"):
            offline_registry.register_server(remote_manifest)

        # 2. Even when opted in, a tampered package binary (wrong SHA-256) is blocked (Phase 15 / Risk R-05)
        opted_in_manifest = McpServerManifest(
            server_id="postgresql",
            display_name="PostgreSQL Remote MCP Server",
            purpose="Query self-hosted PostgreSQL databases over HTTP+SSE.",
            source_kind=ToolSourceKind.MCP,
            transport=McpTransportKind.HTTP_SSE,
            remote_opt_in_enabled=True,
            package_sha256=expected_sha,
            tools=remote_manifest.tools,
        )
        online_registry = ToolRegistry(allow_remote_mcp_globally=True)
        tampered_sha = hashlib.sha256(b"malicious-backdoored-mcp-package").hexdigest()
        with pytest.raises(McpSupplyChainIntegrityError, match="checksum mismatch"):
            online_registry.register_server(
                opted_in_manifest, observed_package_sha256=tampered_sha
            )

        # 3. With valid opt-in AND matching SHA-256, registration succeeds
        online_registry.register_server(
            opted_in_manifest, observed_package_sha256=expected_sha
        )
        assert online_registry.get_tool("postgresql.query").server_id == "postgresql"

    def test_agent_scoped_capability_discovery_respects_least_privilege(self) -> None:
        """Research Agent can discover read/search tools but never Comms or Terminal tools (ADR-008)."""
        registry = build_default_tool_registry()

        # Research agent searching for "email" or "terminal" only searches within its own allowed toolset
        research_matches = registry.discover_tools(
            "search local files and web documents",
            agent_role=AgentRole.RESEARCH,
        )
        assert len(research_matches) >= 1
        for tool in research_matches:
            assert tool.tool_name in INITIAL_AGENT_ROSTER[AgentRole.RESEARCH].allowed_tools_set

        # Planner agent has zero direct tools (Section 12.3 Design Note)
        assert registry.list_tools_for_agent(AgentRole.PLANNER) == ()

        # Unknown tool raises ToolNotFoundError (fail-closed)
        with pytest.raises(ToolNotFoundError):
            registry.get_tool("nonexistent.magic_tool")