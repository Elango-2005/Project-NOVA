"""NOVA MCP-First Tool Registry, Server Manifests & Schema Validator (Layer 3 — ADR-005 / Phase 3 / Phase 7 / Section 14).

Plain-English Purpose (The Tool Catalog & Blueprint Inspector):
Imagine NOVA is a construction company. Before any worker (Specialized Agent) can borrow a power tool:
1. The tool must be listed in the official **Company Tool Catalog** (`ToolRegistry`), which records
   whether it comes from a standard plug-and-play **MCP Server** (`source_kind="mcp"`) or a built-in
   **Native Fallback Adapter** (`source_kind="native"`, ADR-005).
2. Every MCP server has a **Delivery Manifest** (`McpServerManifest`) that checks:
   - Is it running locally on the user's machine (`stdio` transport — default, offline-safe)?
   - Or is it trying to connect over the internet (`http_sse` transport — blocked unless the user
     explicitly opted in, ADR-001 & Phase 3)?
   - Does its SHA-256 package fingerprint match the trusted checksum (Phase 15 supply-chain security)?
3. Every tool has a **Blueprint Inspector** (`ToolContract.validate_arguments` and `validate_output`)
   that verifies the AI's proposed arguments and the tool's returned data match the exact JSON Schema
   types before anything is executed or trusted downstream (Phase 7).
"""

from __future__ import annotations

import hashlib
import re
from enum import Enum
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple

from pydantic import BaseModel, Field

try:
    from pydantic import ConfigDict

    PYDANTIC_V2 = True
except ImportError:  # pragma: no cover
    PYDANTIC_V2 = False

from nova.core.domain.agent_spec import get_agent_spec
from nova.core.domain.primitives import (
    CURRENT_SCHEMA_VERSION,
    AgentRole,
    DataClassification,
    SensitivityTier,
    validate_schema_version,
)

_SHA256_PATTERN = re.compile(r"^[a-fA-F0-9]{64}$")
_TOOL_NAME_PATTERN = re.compile(r"^[a-z][a-z0-9_]*(\.[a-z][a-z0-9_]*)+$")


class ToolRegistryError(Exception):
    """Base error for NOVA Layer 3 tool registry violations."""


class ToolNotFoundError(ToolRegistryError):
    """Raised when an agent or caller requests a tool that is not registered."""


class ToolSchemaValidationError(ToolRegistryError):
    """Raised when tool arguments or tool outputs violate their declared JSON Schema contract."""


class McpTransportSecurityError(ToolRegistryError):
    """Raised when a remote HTTP/SSE MCP server is registered or invoked without explicit opt-in (ADR-001 / Phase 3)."""


class McpSupplyChainIntegrityError(ToolRegistryError):
    """Raised when an MCP server package fails SHA-256 checksum verification (Phase 15 / Risk R-05)."""


class ToolSourceKind(str, Enum):
    """Whether a tool is provided by an MCP server (primary) or NOVA's native fallback registry (ADR-005)."""

    MCP = "mcp"
    NATIVE = "native"


class McpTransportKind(str, Enum):
    """MCP transport mechanism (Phase 3, p. 32).

    - STDIO: Local subprocess communicating over stdin/stdout (default, offline-safe).
    - HTTP_SSE: Remote server over HTTP + Server-Sent Events (requires explicit per-server user opt-in).
    - IN_PROCESS: Native in-process Python adapter for fallback capabilities.
    """

    STDIO = "stdio"
    HTTP_SSE = "http_sse"
    IN_PROCESS = "in_process"


class JsonSchemaPrimitiveType(str, Enum):
    """Supported JSON Schema parameter types for deterministic validation (Phase 7)."""

    STRING = "string"
    INTEGER = "integer"
    NUMBER = "number"
    BOOLEAN = "boolean"
    ARRAY = "array"
    OBJECT = "object"


class ToolParameterSpec(BaseModel):
    """Blueprint for a single tool input parameter (Phase 7 JSON Schema validation)."""

    if PYDANTIC_V2:
        model_config = ConfigDict(frozen=True, extra="forbid")
    else:

        class Config:
            allow_mutation = False
            extra = "forbid"

    name: str = Field(..., min_length=1, description="Parameter name")
    json_type: JsonSchemaPrimitiveType = Field(..., description="Expected JSON Schema primitive type")
    required: bool = Field(default=True, description="Whether the parameter must be provided")
    description: str = Field(..., min_length=1, description="Human/LLM-readable parameter description")
    path_scoped: bool = Field(
        default=False,
        description="True if this parameter represents a filesystem path that must be sandbox-checked",
    )

    def validate_value(self, value: Any) -> None:
        """Deterministically verify that `value` conforms to `self.json_type`."""
        t = self.json_type
        if t == JsonSchemaPrimitiveType.STRING:
            if not isinstance(value, str):
                raise ToolSchemaValidationError(
                    f"Parameter {self.name!r} must be of type 'string', got {type(value).__name__}."
                )
        elif t == JsonSchemaPrimitiveType.BOOLEAN:
            if not isinstance(value, bool):
                raise ToolSchemaValidationError(
                    f"Parameter {self.name!r} must be of type 'boolean', got {type(value).__name__}."
                )
        elif t == JsonSchemaPrimitiveType.INTEGER:
            # Note: In Python, isinstance(True, int) is True, so explicitly exclude bool!
            if isinstance(value, bool) or not isinstance(value, int):
                raise ToolSchemaValidationError(
                    f"Parameter {self.name!r} must be of type 'integer', got {type(value).__name__}."
                )
        elif t == JsonSchemaPrimitiveType.NUMBER:
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ToolSchemaValidationError(
                    f"Parameter {self.name!r} must be of type 'number', got {type(value).__name__}."
                )
        elif t == JsonSchemaPrimitiveType.ARRAY:
            if not isinstance(value, (list, tuple)):
                raise ToolSchemaValidationError(
                    f"Parameter {self.name!r} must be of type 'array', got {type(value).__name__}."
                )
        elif t == JsonSchemaPrimitiveType.OBJECT:
            if not isinstance(value, Mapping):
                raise ToolSchemaValidationError(
                    f"Parameter {self.name!r} must be of type 'object', got {type(value).__name__}."
                )


class ToolContract(BaseModel):
    """Versioned contract and security classification for a single tool (ADR-005 / ADR-010 / Phase 7)."""

    if PYDANTIC_V2:
        model_config = ConfigDict(frozen=True, extra="forbid")
    else:

        class Config:
            allow_mutation = False
            extra = "forbid"

    tool_name: str = Field(..., description="Fully-qualified `<server>.<action>` tool identifier")
    server_id: str = Field(..., min_length=1, description="Owning MCP server or native adapter ID")
    version: str = Field(
        default=CURRENT_SCHEMA_VERSION,
        description="SemVer contract version (Engineering Principle 7 & 8)",
    )
    description: str = Field(..., min_length=5, description="Clear description of what the tool does")
    sensitivity_tier: SensitivityTier = Field(
        ..., description="Default ADR-010 / Section 14 action sensitivity tier"
    )
    data_classification: DataClassification = Field(
        default=DataClassification.PUBLIC,
        description="Default data classification touched by this tool (Section 16.2)",
    )
    parameters: Tuple[ToolParameterSpec, ...] = Field(
        default_factory=tuple,
        description="Typed JSON-Schema parameter specifications",
    )
    required_output_keys: Tuple[str, ...] = Field(
        default_factory=tuple,
        description="Keys that must appear in the tool's structured dictionary result",
    )
    reliability_score: float = Field(
        default=0.98,
        ge=0.0,
        le=1.0,
        description="Historical reliability score for Phase 7 capability ranking",
    )

    def __init__(self, **data: Any) -> None:
        tool_name = data.get("tool_name")
        if not isinstance(tool_name, str) or not _TOOL_NAME_PATTERN.match(tool_name):
            raise ValueError(
                f"Tool name {tool_name!r} must follow `<namespace>.<action>` format."
            )
        version = data.get("version", CURRENT_SCHEMA_VERSION)
        if isinstance(version, str):
            data = dict(data)
            data["version"] = validate_schema_version(version)
        super().__init__(**data)

    @property
    def path_scoped_parameter_names(self) -> Tuple[str, ...]:
        """Return parameter names that hold filesystem paths requiring sandbox confinement."""
        return tuple(p.name for p in self.parameters if p.path_scoped)

    def to_json_schema(self) -> Dict[str, Any]:
        """Export standard MCP / JSON Schema representation of this tool's input contract."""
        properties: Dict[str, Any] = {}
        required: List[str] = []
        for param in self.parameters:
            properties[param.name] = {
                "type": param.json_type.value,
                "description": param.description,
            }
            if param.path_scoped:
                properties[param.name]["x-nova-path-scoped"] = True
            if param.required:
                required.append(param.name)
        return {
            "name": self.tool_name,
            "version": self.version,
            "description": self.description,
            "inputSchema": {
                "type": "object",
                "properties": properties,
                "required": required,
                "additionalProperties": False,
            },
        }

    def validate_arguments(self, arguments: Mapping[str, Any]) -> None:
        """Validate model-proposed arguments against this tool's JSON Schema contract (Phase 7)."""
        if not isinstance(arguments, Mapping):
            raise ToolSchemaValidationError(
                f"Arguments for {self.tool_name!r} must be a JSON object/dict."
            )

        param_map: Dict[str, ToolParameterSpec] = {p.name: p for p in self.parameters}

        # 1. Reject undeclared extra arguments (strict schema enforcement)
        unknown_keys = [k for k in arguments.keys() if k not in param_map]
        if unknown_keys:
            raise ToolSchemaValidationError(
                f"Tool {self.tool_name!r} received undeclared argument(s): {unknown_keys}. "
                f"Allowed parameters: {list(param_map.keys())}."
            )

        # 2. Check all required parameters are present and non-null
        for name, spec in param_map.items():
            if spec.required and (name not in arguments or arguments[name] is None):
                raise ToolSchemaValidationError(
                    f"Tool {self.tool_name!r} is missing required parameter {name!r} "
                    f"(expected type {spec.json_type.value!r})."
                )

        # 3. Validate types for all provided parameters
        for name, value in arguments.items():
            if value is not None:
                param_map[name].validate_value(value)

    def validate_output(self, output: Mapping[str, Any]) -> None:
        """Validate that the tool's returned output conforms to its declared output schema (Phase 7)."""
        if not isinstance(output, Mapping):
            raise ToolSchemaValidationError(
                f"Output from tool {self.tool_name!r} must be a structured dictionary/object."
            )
        missing_keys = [k for k in self.required_output_keys if k not in output]
        if missing_keys:
            raise ToolSchemaValidationError(
                f"Output from tool {self.tool_name!r} is missing required key(s): {missing_keys}."
            )


class McpServerManifest(BaseModel):
    """Manifest describing an MCP server or native tool provider (ADR-005 / Phase 3 / Phase 15)."""

    if PYDANTIC_V2:
        model_config = ConfigDict(frozen=True, extra="forbid")
    else:

        class Config:
            allow_mutation = False
            extra = "forbid"

    server_id: str = Field(..., min_length=1, description="Canonical server namespace (e.g., 'filesystem')")
    display_name: str = Field(..., min_length=1, description="Human-readable server name")
    purpose: str = Field(..., min_length=5, description="Why this server is useful to NOVA (Section 14)")
    version: str = Field(
        default=CURRENT_SCHEMA_VERSION,
        description="Server manifest SemVer version",
    )
    source_kind: ToolSourceKind = Field(
        default=ToolSourceKind.MCP,
        description="Primary MCP server vs. native fallback registry (ADR-005)",
    )
    transport: McpTransportKind = Field(
        default=McpTransportKind.STDIO,
        description="Transport mechanism (STDIO local subprocess by default per Phase 3)",
    )
    remote_opt_in_enabled: bool = Field(
        default=False,
        description="Must be True before an HTTP_SSE remote MCP server is permitted (ADR-001 / Phase 3)",
    )
    package_sha256: str = Field(
        ...,
        description="Expected 64-hex-character SHA-256 digest for supply-chain integrity (Phase 15 / Risk R-05)",
    )
    tools: Tuple[ToolContract, ...] = Field(
        default_factory=tuple,
        description="Tools exposed by this MCP server",
    )

    def __init__(self, **data: Any) -> None:
        updated = dict(data)
        version = updated.get("version", CURRENT_SCHEMA_VERSION)
        if isinstance(version, str):
            updated["version"] = validate_schema_version(version)

        sha = updated.get("package_sha256")
        if not isinstance(sha, str) or not _SHA256_PATTERN.match(sha):
            raise ValueError(
                f"package_sha256 must be a valid 64-character hex SHA-256 digest, got {sha!r}."
            )
        updated["package_sha256"] = sha.lower()

        server_id = updated.get("server_id")
        tools: Tuple[ToolContract, ...] = tuple(updated.get("tools") or ())
        for tool in tools:
            if tool.server_id != server_id:
                raise ValueError(
                    f"Tool {tool.tool_name!r} declares server_id={tool.server_id!r}, "
                    f"which does not match manifest server_id={server_id!r}."
                )
        super().__init__(**updated)

    def verify_package_checksum(self, actual_package_bytes_or_hex: bytes | str) -> None:
        """Verify that an MCP server package matches its pinned SHA-256 digest (Phase 15 / Risk R-05)."""
        if isinstance(actual_package_bytes_or_hex, bytes):
            actual_hex = hashlib.sha256(actual_package_bytes_or_hex).hexdigest().lower()
        else:
            actual_hex = actual_package_bytes_or_hex.strip().lower()
        if actual_hex != self.package_sha256:
            raise McpSupplyChainIntegrityError(
                f"Supply-chain integrity failure (Phase 15 / Risk R-05): MCP server {self.server_id!r} "
                f"checksum mismatch! Expected {self.package_sha256}, got {actual_hex}."
            )


def _deterministic_sha256(label: str) -> str:
    """Generate a deterministic 64-char SHA-256 hex digest for built-in catalog manifests."""
    return hashlib.sha256(f"nova-mcp-catalog-v1:{label}".encode("utf-8")).hexdigest()


class ToolRegistry:
    """Canonical, versioned catalog of MCP and native tools available to NOVA (Layer 3 — Phase 7).

    Responsibilities:
    1. Enforce offline-first MCP transport rules (block `HTTP_SSE` servers unless opted in).
    2. Verify supply-chain package SHA-256 checksums on registration when package bytes/digests are supplied.
    3. Provide fast lookup of `ToolContract` metadata and sensitivity tiers for the Permission Gate.
    4. Provide agent-scoped tool discovery (`ADR-008`) and fuzzy keyword capability search (`Phase 7`).
    """

    def __init__(self, *, allow_remote_mcp_globally: bool = False) -> None:
        self._allow_remote_mcp_globally = allow_remote_mcp_globally
        self._manifests: Dict[str, McpServerManifest] = {}
        self._tools: Dict[str, ToolContract] = {}

    @property
    def registered_server_ids(self) -> Tuple[str, ...]:
        return tuple(sorted(self._manifests.keys()))

    @property
    def registered_tool_names(self) -> Tuple[str, ...]:
        return tuple(sorted(self._tools.keys()))

    def register_server(
        self,
        manifest: McpServerManifest,
        *,
        observed_package_sha256: Optional[str] = None,
    ) -> None:
        """Register an MCP server manifest and all of its tools after transport and checksum checks."""
        # 1. Enforce Phase 3 / ADR-001 transport security: HTTP+SSE requires explicit opt-in
        if manifest.transport == McpTransportKind.HTTP_SSE:
            if not (self._allow_remote_mcp_globally and manifest.remote_opt_in_enabled):
                raise McpTransportSecurityError(
                    f"ADR-001 / Phase 3 violation: Remote HTTP+SSE MCP server {manifest.server_id!r} "
                    "cannot be registered unless both global remote MCP support and per-server "
                    "remote_opt_in_enabled=True are explicitly enabled by the user."
                )

        # 2. Enforce Phase 15 / Risk R-05 supply-chain checksum verification if observed digest provided
        if observed_package_sha256 is not None:
            manifest.verify_package_checksum(observed_package_sha256)

        self._manifests[manifest.server_id] = manifest
        for tool in manifest.tools:
            self._tools[tool.tool_name] = tool

    def get_server_manifest(self, server_id: str) -> McpServerManifest:
        if server_id not in self._manifests:
            raise ToolNotFoundError(f"MCP server {server_id!r} is not registered.")
        return self._manifests[server_id]

    def get_tool(self, tool_name: str) -> ToolContract:
        """Return the `ToolContract` for `tool_name` or raise `ToolNotFoundError` (fail-closed)."""
        if tool_name not in self._tools:
            raise ToolNotFoundError(
                f"Tool {tool_name!r} is not registered in NOVA's ToolRegistry."
            )
        return self._tools[tool_name]

    def validate_tool_call(self, tool_name: str, arguments: Mapping[str, Any]) -> ToolContract:
        """Look up `tool_name`, validate `arguments` against its JSON Schema, and return its contract."""
        contract = self.get_tool(tool_name)
        contract.validate_arguments(arguments)
        return contract

    def list_tools_for_agent(self, agent_role: AgentRole) -> Tuple[ToolContract, ...]:
        """Return only the registered `ToolContract`s permitted for `agent_role` per ADR-008."""
        spec = get_agent_spec(agent_role)
        contracts: List[ToolContract] = []
        for tool_name in spec.allowed_tools:
            if tool_name in self._tools:
                contracts.append(self._tools[tool_name])
        return tuple(contracts)

    def discover_tools(
        self,
        query: str,
        *,
        agent_role: Optional[AgentRole] = None,
        top_k: int = 5,
    ) -> Tuple[ToolContract, ...]:
        """Fuzzy capability discovery over tool names and descriptions (Phase 7, p. 37).

        If `agent_role` is provided, discovery is strictly filtered to that agent's ADR-008 allowed tools.
        Candidates are ranked by lexical token overlap + historical `reliability_score`.
        """
        candidates: Iterable[ToolContract]
        if agent_role is not None:
            candidates = self.list_tools_for_agent(agent_role)
        else:
            candidates = self._tools.values()

        query_tokens = {
            tok for tok in re.findall(r"[a-z0-9]+", query.lower()) if len(tok) >= 2
        }
        if not query_tokens:
            sorted_all = sorted(
                candidates, key=lambda c: (c.reliability_score, c.tool_name), reverse=True
            )
            return tuple(sorted_all[:top_k])

        scored: List[Tuple[float, ToolContract]] = []
        for contract in candidates:
            haystack_tokens = set(
                re.findall(
                    r"[a-z0-9]+",
                    f"{contract.tool_name} {contract.description} {contract.server_id}".lower(),
                )
            )
            overlap = len(query_tokens & haystack_tokens)
            if overlap > 0:
                score = (overlap / float(len(query_tokens))) * 0.85 + (
                    contract.reliability_score * 0.15
                )
                scored.append((score, contract))

        scored.sort(key=lambda item: (item[0], item[1].reliability_score), reverse=True)
        return tuple(contract for _, contract in scored[:top_k])


# ============================================================================
# Section 14 Canonical MCP Server Catalog & Native Fallbacks (pp. 49-52)
# ============================================================================


def _param(
    name: str,
    json_type: JsonSchemaPrimitiveType,
    description: str,
    *,
    required: bool = True,
    path_scoped: bool = False,
) -> ToolParameterSpec:
    return ToolParameterSpec(
        name=name,
        json_type=json_type,
        required=required,
        description=description,
        path_scoped=path_scoped,
    )


def build_default_mcp_catalog() -> Tuple[McpServerManifest, ...]:
    """Construct the canonical Section 14 MCP server manifests + ADR-005 native fallback adapters."""
    S = JsonSchemaPrimitiveType
    T = SensitivityTier
    D = DataClassification

    return (
        # 1. Filesystem MCP Server (Section 14, p. 50)
        McpServerManifest(
            server_id="filesystem",
            display_name="Filesystem MCP Server",
            purpose="Read, write, and organize local documents within explicitly permitted directories.",
            source_kind=ToolSourceKind.MCP,
            transport=McpTransportKind.STDIO,
            package_sha256=_deterministic_sha256("filesystem"),
            tools=(
                ToolContract(
                    tool_name="filesystem.read_file",
                    server_id="filesystem",
                    description="Read text content from a file inside a permitted workspace directory.",
                    sensitivity_tier=T.READ_ONLY,
                    data_classification=D.USER_CONTENT_LOCAL,
                    parameters=(
                        _param("path", S.STRING, "File path to read", path_scoped=True),
                    ),
                    required_output_keys=("content",),
                ),
                ToolContract(
                    tool_name="filesystem.write_file",
                    server_id="filesystem",
                    description="Write or update a file inside a permitted workspace directory.",
                    sensitivity_tier=T.REVERSIBLE_WRITE,
                    data_classification=D.USER_CONTENT_LOCAL,
                    parameters=(
                        _param("path", S.STRING, "Target file path to write", path_scoped=True),
                        _param("content", S.STRING, "Text content to write to the file"),
                    ),
                    required_output_keys=("bytes_written",),
                ),
                ToolContract(
                    tool_name="filesystem.list_directory",
                    server_id="filesystem",
                    description="List files and subdirectories inside a permitted directory.",
                    sensitivity_tier=T.READ_ONLY,
                    data_classification=D.USER_CONTENT_LOCAL,
                    parameters=(
                        _param("path", S.STRING, "Directory path to list", path_scoped=True),
                    ),
                    required_output_keys=("entries",),
                ),
                ToolContract(
                    tool_name="filesystem.search_files",
                    server_id="filesystem",
                    description="Search for files matching a pattern inside a permitted directory.",
                    sensitivity_tier=T.READ_ONLY,
                    data_classification=D.USER_CONTENT_LOCAL,
                    parameters=(
                        _param("path", S.STRING, "Root directory path to search", path_scoped=True),
                        _param("pattern", S.STRING, "Filename or glob pattern to match"),
                    ),
                    required_output_keys=("matches",),
                ),
            ),
        ),
        # 2. Git MCP Server (Section 14, p. 50)
        McpServerManifest(
            server_id="git",
            display_name="Git Local Version Control MCP Server",
            purpose="Local version control operations (status, diff, commit, branch) independent of a remote host.",
            source_kind=ToolSourceKind.MCP,
            transport=McpTransportKind.STDIO,
            package_sha256=_deterministic_sha256("git"),
            tools=(
                ToolContract(
                    tool_name="git.status",
                    server_id="git",
                    description="Inspect working tree status of a local Git repository.",
                    sensitivity_tier=T.READ_ONLY,
                    parameters=(
                        _param("repo_path", S.STRING, "Local repository path", path_scoped=True),
                    ),
                    required_output_keys=("status",),
                ),
                ToolContract(
                    tool_name="git.diff",
                    server_id="git",
                    description="Show unstaged or staged line diffs in a local Git repository.",
                    sensitivity_tier=T.READ_ONLY,
                    parameters=(
                        _param("repo_path", S.STRING, "Local repository path", path_scoped=True),
                    ),
                    required_output_keys=("diff",),
                ),
                ToolContract(
                    tool_name="git.commit",
                    server_id="git",
                    description="Create a local Git commit with a descriptive commit message.",
                    sensitivity_tier=T.REVERSIBLE_WRITE,
                    parameters=(
                        _param("repo_path", S.STRING, "Local repository path", path_scoped=True),
                        _param("message", S.STRING, "Commit message"),
                    ),
                    required_output_keys=("commit_sha",),
                ),
                ToolContract(
                    tool_name="git.branch",
                    server_id="git",
                    description="Create or switch branches in a local Git repository.",
                    sensitivity_tier=T.REVERSIBLE_WRITE,
                    parameters=(
                        _param("repo_path", S.STRING, "Local repository path", path_scoped=True),
                        _param("branch_name", S.STRING, "Target branch name"),
                    ),
                    required_output_keys=("branch",),
                ),
            ),
        ),
        # 3. GitHub MCP Server (Section 14, p. 50)
        McpServerManifest(
            server_id="github",
            display_name="GitHub MCP Server",
            purpose="Read issues/PRs and open pull requests for software development workflows.",
            source_kind=ToolSourceKind.MCP,
            transport=McpTransportKind.STDIO,
            package_sha256=_deterministic_sha256("github"),
            tools=(
                ToolContract(
                    tool_name="github.read_issue",
                    server_id="github",
                    description="Read a GitHub issue by repository and issue number.",
                    sensitivity_tier=T.READ_ONLY_NETWORKED,
                    parameters=(
                        _param("repo", S.STRING, "Repository in owner/name format"),
                        _param("issue_number", S.INTEGER, "Issue number"),
                    ),
                    required_output_keys=("title", "body"),
                ),
                ToolContract(
                    tool_name="github.read_pr",
                    server_id="github",
                    description="Read a GitHub pull request by repository and PR number.",
                    sensitivity_tier=T.READ_ONLY_NETWORKED,
                    parameters=(
                        _param("repo", S.STRING, "Repository in owner/name format"),
                        _param("pr_number", S.INTEGER, "Pull request number"),
                    ),
                    required_output_keys=("title", "diff_summary"),
                ),
                ToolContract(
                    tool_name="github.create_pr",
                    server_id="github",
                    description="Open a new pull request on a remote GitHub repository.",
                    sensitivity_tier=T.REVERSIBLE_WRITE_NETWORKED,
                    parameters=(
                        _param("repo", S.STRING, "Repository in owner/name format"),
                        _param("title", S.STRING, "Pull request title"),
                        _param("head", S.STRING, "Source branch name"),
                        _param("base", S.STRING, "Target base branch name"),
                    ),
                    required_output_keys=("pr_url",),
                ),
            ),
        ),
        # 4. SQLite MCP Server (Section 14, p. 50)
        McpServerManifest(
            server_id="sqlite",
            display_name="SQLite Local Database MCP Server",
            purpose="Query and manage local structured SQLite databases.",
            source_kind=ToolSourceKind.MCP,
            transport=McpTransportKind.STDIO,
            package_sha256=_deterministic_sha256("sqlite"),
            tools=(
                ToolContract(
                    tool_name="sqlite.query",
                    server_id="sqlite",
                    description="Execute a read-only SELECT query against a local SQLite database.",
                    sensitivity_tier=T.READ_ONLY,
                    parameters=(
                        _param("db_path", S.STRING, "SQLite file path", path_scoped=True),
                        _param("sql", S.STRING, "SELECT SQL query"),
                    ),
                    required_output_keys=("rows",),
                ),
            ),
        ),
        # 5. Docker MCP Server (Section 14, p. 50)
        McpServerManifest(
            server_id="docker",
            display_name="Docker Container MCP Server",
            purpose="Build, run, and test containerized projects without unrestricted host shell access.",
            source_kind=ToolSourceKind.MCP,
            transport=McpTransportKind.STDIO,
            package_sha256=_deterministic_sha256("docker"),
            tools=(
                ToolContract(
                    tool_name="docker.run_container",
                    server_id="docker",
                    description="Run an isolated container image for build or test execution.",
                    sensitivity_tier=T.IRREVERSIBLE_WRITE,
                    parameters=(
                        _param("image", S.STRING, "Container image tag"),
                        _param("command", S.STRING, "Command to run inside container"),
                    ),
                    required_output_keys=("exit_code", "stdout"),
                ),
            ),
        ),
        # 6. Playwright MCP Server (Section 14, p. 50 / Phase 13)
        McpServerManifest(
            server_id="playwright",
            display_name="Playwright Browser Automation MCP Server",
            purpose="Primary Browser Agent automation backend behind permission and audit controls.",
            source_kind=ToolSourceKind.MCP,
            transport=McpTransportKind.STDIO,
            package_sha256=_deterministic_sha256("playwright"),
            tools=(
                ToolContract(
                    tool_name="playwright.navigate",
                    server_id="playwright",
                    description="Navigate the isolated automation browser profile to a URL.",
                    sensitivity_tier=T.REVERSIBLE_WRITE,
                    parameters=(_param("url", S.STRING, "Target https URL"),),
                    required_output_keys=("title", "status_code"),
                ),
                ToolContract(
                    tool_name="playwright.extract",
                    server_id="playwright",
                    description="Extract structured DOM text from the active browser page as untrusted data.",
                    sensitivity_tier=T.READ_ONLY_NETWORKED,
                    parameters=(_param("selector", S.STRING, "CSS or accessibility selector"),),
                    required_output_keys=("extracted_text",),
                ),
                ToolContract(
                    tool_name="playwright.fill_form",
                    server_id="playwright",
                    description="Fill form fields on a web page without submitting.",
                    sensitivity_tier=T.REVERSIBLE_WRITE,
                    parameters=(
                        _param("selector", S.STRING, "Input field selector"),
                        _param("value", S.STRING, "Value to fill"),
                    ),
                    required_output_keys=("filled",),
                ),
                ToolContract(
                    tool_name="playwright.click",
                    server_id="playwright",
                    description="Click an interactive element on a web page.",
                    sensitivity_tier=T.REVERSIBLE_WRITE,
                    parameters=(_param("selector", S.STRING, "Element selector"),),
                    required_output_keys=("clicked",),
                ),
                ToolContract(
                    tool_name="playwright.submit",
                    server_id="playwright",
                    description="Submit a web form or trigger an external web transaction.",
                    sensitivity_tier=T.IRREVERSIBLE_WRITE,
                    parameters=(_param("form_selector", S.STRING, "Form selector to submit"),),
                    required_output_keys=("submitted",),
                ),
            ),
        ),
        # 7. Browser Read-Only MCP Server (Section 14, p. 50)
        McpServerManifest(
            server_id="browser",
            display_name="Generic Read-Only Browser MCP Server",
            purpose="Fetch and extract web pages for read-only tasks with minimal permission footprint.",
            source_kind=ToolSourceKind.MCP,
            transport=McpTransportKind.STDIO,
            package_sha256=_deterministic_sha256("browser"),
            tools=(
                ToolContract(
                    tool_name="browser.fetch_readonly",
                    server_id="browser",
                    description="Fetch a web page in read-only mode and extract clean text.",
                    sensitivity_tier=T.READ_ONLY_NETWORKED,
                    parameters=(_param("url", S.STRING, "Web page URL to fetch"),),
                    required_output_keys=("content",),
                ),
            ),
        ),
        # 8. Gmail MCP Server (Section 14, p. 51)
        McpServerManifest(
            server_id="gmail",
            display_name="Gmail Communication MCP Server",
            purpose="Powers the Comms Agent's email reading, drafting, and sending capabilities.",
            source_kind=ToolSourceKind.MCP,
            transport=McpTransportKind.STDIO,
            package_sha256=_deterministic_sha256("gmail"),
            tools=(
                ToolContract(
                    tool_name="gmail.read_email",
                    server_id="gmail",
                    description="Read an email thread or message by ID.",
                    sensitivity_tier=T.READ_ONLY_NETWORKED,
                    data_classification=D.USER_CONTENT_CONNECTED,
                    parameters=(_param("message_id", S.STRING, "Email message or thread ID"),),
                    required_output_keys=("sender", "subject", "body"),
                ),
                ToolContract(
                    tool_name="gmail.draft_email",
                    server_id="gmail",
                    description="Create a reversible draft email without sending it.",
                    sensitivity_tier=T.REVERSIBLE_WRITE,
                    data_classification=D.USER_CONTENT_CONNECTED,
                    parameters=(
                        _param("to", S.STRING, "Recipient email address"),
                        _param("subject", S.STRING, "Email subject line"),
                        _param("body", S.STRING, "Email body text"),
                    ),
                    required_output_keys=("draft_id",),
                ),
                ToolContract(
                    tool_name="gmail.send_email",
                    server_id="gmail",
                    description="Send an email to an external recipient (irreversible action).",
                    sensitivity_tier=T.IRREVERSIBLE_WRITE,
                    data_classification=D.USER_CONTENT_CONNECTED,
                    parameters=(
                        _param("to", S.STRING, "Recipient email address"),
                        _param("subject", S.STRING, "Email subject line"),
                        _param("body", S.STRING, "Email body text"),
                    ),
                    required_output_keys=("message_id", "status"),
                ),
            ),
        ),
        # 9. Calendar MCP Server (Section 14, p. 51)
        McpServerManifest(
            server_id="calendar",
            display_name="Calendar Scheduling MCP Server",
            purpose="List, create, and update calendar events for personal scheduling.",
            source_kind=ToolSourceKind.MCP,
            transport=McpTransportKind.STDIO,
            package_sha256=_deterministic_sha256("calendar"),
            tools=(
                ToolContract(
                    tool_name="calendar.list_events",
                    server_id="calendar",
                    description="List upcoming calendar events in a date window.",
                    sensitivity_tier=T.READ_ONLY,
                    data_classification=D.USER_CONTENT_CONNECTED,
                    parameters=(
                        _param("start_iso", S.STRING, "Start timestamp ISO-8601"),
                        _param("end_iso", S.STRING, "End timestamp ISO-8601"),
                    ),
                    required_output_keys=("events",),
                ),
                ToolContract(
                    tool_name="calendar.create_event",
                    server_id="calendar",
                    description="Create a new calendar event.",
                    sensitivity_tier=T.REVERSIBLE_WRITE,
                    data_classification=D.USER_CONTENT_CONNECTED,
                    parameters=(
                        _param("title", S.STRING, "Event title"),
                        _param("start_iso", S.STRING, "Start timestamp ISO-8601"),
                        _param("end_iso", S.STRING, "End timestamp ISO-8601"),
                    ),
                    required_output_keys=("event_id",),
                ),
                ToolContract(
                    tool_name="calendar.update_event",
                    server_id="calendar",
                    description="Update an existing calendar event.",
                    sensitivity_tier=T.REVERSIBLE_WRITE,
                    data_classification=D.USER_CONTENT_CONNECTED,
                    parameters=(
                        _param("event_id", S.STRING, "Event ID to modify"),
                        _param("title", S.STRING, "Updated event title"),
                    ),
                    required_output_keys=("updated",),
                ),
            ),
        ),
        # 10. Slack & Discord MCP Servers (Section 14, p. 51)
        McpServerManifest(
            server_id="slack",
            display_name="Slack Workplace Messaging MCP Server",
            purpose="Read and send workplace Slack messages via the Comms Agent.",
            source_kind=ToolSourceKind.MCP,
            transport=McpTransportKind.STDIO,
            package_sha256=_deterministic_sha256("slack"),
            tools=(
                ToolContract(
                    tool_name="slack.read_messages",
                    server_id="slack",
                    description="Read recent messages from a Slack channel.",
                    sensitivity_tier=T.READ_ONLY_NETWORKED,
                    parameters=(_param("channel_id", S.STRING, "Slack channel ID"),),
                    required_output_keys=("messages",),
                ),
                ToolContract(
                    tool_name="slack.send_message",
                    server_id="slack",
                    description="Post a message to a Slack channel (irreversible external message).",
                    sensitivity_tier=T.IRREVERSIBLE_WRITE,
                    parameters=(
                        _param("channel_id", S.STRING, "Slack channel ID"),
                        _param("text", S.STRING, "Message content"),
                    ),
                    required_output_keys=("message_ts",),
                ),
            ),
        ),
        McpServerManifest(
            server_id="discord",
            display_name="Discord Messaging MCP Server",
            purpose="Read and send Discord channel messages via the Comms Agent.",
            source_kind=ToolSourceKind.MCP,
            transport=McpTransportKind.STDIO,
            package_sha256=_deterministic_sha256("discord"),
            tools=(
                ToolContract(
                    tool_name="discord.read_messages",
                    server_id="discord",
                    description="Read recent messages from a Discord channel.",
                    sensitivity_tier=T.READ_ONLY_NETWORKED,
                    parameters=(_param("channel_id", S.STRING, "Discord channel ID"),),
                    required_output_keys=("messages",),
                ),
                ToolContract(
                    tool_name="discord.send_message",
                    server_id="discord",
                    description="Send a message to a Discord channel.",
                    sensitivity_tier=T.IRREVERSIBLE_WRITE,
                    parameters=(
                        _param("channel_id", S.STRING, "Discord channel ID"),
                        _param("content", S.STRING, "Message text"),
                    ),
                    required_output_keys=("message_id",),
                ),
            ),
        ),
        # 11. VS Code & Terminal MCP Servers (Section 14, p. 51)
        McpServerManifest(
            server_id="vscode",
            display_name="VS Code Editor Context MCP Server",
            purpose="Operate within the user's editor context (diagnostics, workspace tasks).",
            source_kind=ToolSourceKind.MCP,
            transport=McpTransportKind.STDIO,
            package_sha256=_deterministic_sha256("vscode"),
            tools=(
                ToolContract(
                    tool_name="vscode.inspect_workspace",
                    server_id="vscode",
                    description="Inspect open files and diagnostics in the active VS Code workspace.",
                    sensitivity_tier=T.READ_ONLY,
                    parameters=(
                        _param("workspace_path", S.STRING, "Workspace folder path", path_scoped=True),
                    ),
                    required_output_keys=("diagnostics",),
                ),
                ToolContract(
                    tool_name="vscode.run_task",
                    server_id="vscode",
                    description="Run a configured VS Code build or test task.",
                    sensitivity_tier=T.REVERSIBLE_WRITE,
                    parameters=(
                        _param("task_name", S.STRING, "Configured VS Code task name"),
                    ),
                    required_output_keys=("task_status",),
                ),
            ),
        ),
        McpServerManifest(
            server_id="terminal",
            display_name="Sandboxed Terminal MCP Server",
            purpose="Execute shell commands for coding and automation under strict sandbox & approval gates.",
            source_kind=ToolSourceKind.MCP,
            transport=McpTransportKind.STDIO,
            package_sha256=_deterministic_sha256("terminal"),
            tools=(
                ToolContract(
                    tool_name="terminal.execute",
                    server_id="terminal",
                    description="Execute a shell command inside a sandboxed working directory.",
                    sensitivity_tier=T.IRREVERSIBLE_WRITE,
                    parameters=(
                        _param("command", S.STRING, "Shell command string to execute"),
                        _param("cwd", S.STRING, "Working directory path", path_scoped=True),
                    ),
                    required_output_keys=("exit_code", "stdout", "stderr"),
                ),
            ),
        ),
        # 12. Weather & Search MCP Servers (Section 14, p. 51)
        McpServerManifest(
            server_id="weather",
            display_name="Weather Utility MCP Server",
            purpose="Low-stakes, purely read-only weather forecast query.",
            source_kind=ToolSourceKind.MCP,
            transport=McpTransportKind.STDIO,
            package_sha256=_deterministic_sha256("weather"),
            tools=(
                ToolContract(
                    tool_name="weather.get_forecast",
                    server_id="weather",
                    description="Retrieve current weather and forecast for a city.",
                    sensitivity_tier=T.READ_ONLY,
                    parameters=(_param("location", S.STRING, "City or location name"),),
                    required_output_keys=("temperature_c", "condition"),
                ),
            ),
        ),
        McpServerManifest(
            server_id="search",
            display_name="Web Search MCP Server",
            purpose="Ground the Research Agent in current web information beyond local documents.",
            source_kind=ToolSourceKind.MCP,
            transport=McpTransportKind.STDIO,
            package_sha256=_deterministic_sha256("search"),
            tools=(
                ToolContract(
                    tool_name="search.web_search",
                    server_id="search",
                    description="Run a read-only web search query and return top snippets with URLs.",
                    sensitivity_tier=T.READ_ONLY_NETWORKED,
                    parameters=(_param("query", S.STRING, "Search query string"),),
                    required_output_keys=("results",),
                ),
            ),
        ),
        # 13. Native Fallback Providers (ADR-005: Local RAG, Desktop Accessibility, Screen & Voice)
        McpServerManifest(
            server_id="rag",
            display_name="NOVA Native Hybrid RAG Adapter",
            purpose="Retrieve grounded chunks and mandatory citations from NOVA's local hybrid RAG engine.",
            source_kind=ToolSourceKind.NATIVE,
            transport=McpTransportKind.IN_PROCESS,
            package_sha256=_deterministic_sha256("rag"),
            tools=(
                ToolContract(
                    tool_name="rag.retrieve",
                    server_id="rag",
                    description="Search local indexed documents using BM25 + vector hybrid retrieval.",
                    sensitivity_tier=T.READ_ONLY,
                    data_classification=D.USER_CONTENT_LOCAL,
                    parameters=(
                        _param("query", S.STRING, "Natural language or keyword query"),
                        _param("top_k", S.INTEGER, "Maximum chunks to retrieve", required=False),
                    ),
                    required_output_keys=("chunks", "citations"),
                ),
            ),
        ),
        McpServerManifest(
            server_id="accessibility",
            display_name="NOVA Native OS Accessibility Automation Adapter",
            purpose="Inspect and invoke desktop UI elements via native OS accessibility trees (Phase 14).",
            source_kind=ToolSourceKind.NATIVE,
            transport=McpTransportKind.IN_PROCESS,
            package_sha256=_deterministic_sha256("accessibility"),
            tools=(
                ToolContract(
                    tool_name="accessibility.inspect_tree",
                    server_id="accessibility",
                    description="Read the structured accessibility tree of the active desktop window.",
                    sensitivity_tier=T.READ_ONLY,
                    parameters=(_param("window_title", S.STRING, "Target window title"),),
                    required_output_keys=("elements",),
                ),
                ToolContract(
                    tool_name="accessibility.invoke_element",
                    server_id="accessibility",
                    description="Invoke a button or control via its OS accessibility element ID.",
                    sensitivity_tier=T.REVERSIBLE_WRITE,
                    parameters=(
                        _param("window_title", S.STRING, "Target window title"),
                        _param("element_id", S.STRING, "Accessibility element identifier"),
                    ),
                    required_output_keys=("invoked",),
                ),
            ),
        ),
        McpServerManifest(
            server_id="pyautogui",
            display_name="NOVA Coordinate Fallback Desktop Adapter",
            purpose="Fallback pixel/coordinate automation when OS accessibility trees are unavailable (Phase 14).",
            source_kind=ToolSourceKind.NATIVE,
            transport=McpTransportKind.IN_PROCESS,
            package_sha256=_deterministic_sha256("pyautogui"),
            tools=(
                ToolContract(
                    tool_name="pyautogui.fallback_action",
                    server_id="pyautogui",
                    description="Execute a coordinate-based click or keystroke (high sensitivity fallback).",
                    sensitivity_tier=T.IRREVERSIBLE_WRITE,
                    parameters=(
                        _param("action", S.STRING, "Action type (click or type)"),
                        _param("x", S.INTEGER, "Screen X coordinate"),
                        _param("y", S.INTEGER, "Screen Y coordinate"),
                    ),
                    required_output_keys=("executed",),
                ),
            ),
        ),
        McpServerManifest(
            server_id="screen",
            display_name="NOVA Native Screen Capture Adapter",
            purpose="Capture screenshots for VLM screen understanding (Phase 12 / Phase 14).",
            source_kind=ToolSourceKind.NATIVE,
            transport=McpTransportKind.IN_PROCESS,
            package_sha256=_deterministic_sha256("screen"),
            tools=(
                ToolContract(
                    tool_name="screen.capture",
                    server_id="screen",
                    description="Capture a read-only screenshot of the active display or window.",
                    sensitivity_tier=T.READ_ONLY,
                    parameters=(_param("region", S.STRING, "Display or window region", required=False),),
                    required_output_keys=("image_ref",),
                ),
            ),
        ),
        McpServerManifest(
            server_id="voice",
            display_name="NOVA Native Streaming Voice Adapter",
            purpose="Local Faster-Whisper STT and Piper/Kokoro TTS streaming modality tools (Phase 11).",
            source_kind=ToolSourceKind.NATIVE,
            transport=McpTransportKind.IN_PROCESS,
            package_sha256=_deterministic_sha256("voice"),
            tools=(
                ToolContract(
                    tool_name="voice.stt_transcribe",
                    server_id="voice",
                    description="Transcribe a local audio stream segment to text via Faster-Whisper.",
                    sensitivity_tier=T.READ_ONLY,
                    parameters=(_param("audio_stream_id", S.STRING, "Audio stream identifier"),),
                    required_output_keys=("transcript",),
                ),
                ToolContract(
                    tool_name="voice.tts_synthesize",
                    server_id="voice",
                    description="Synthesize speech audio chunks from text via local Piper/Kokoro TTS.",
                    sensitivity_tier=T.READ_ONLY,
                    parameters=(_param("text", S.STRING, "Text to speak"),),
                    required_output_keys=("audio_chunk_id",),
                ),
                ToolContract(
                    tool_name="voice.interrupt_stream",
                    server_id="voice",
                    description="Cancel active TTS playback when the user speaks over NOVA (barge-in).",
                    sensitivity_tier=T.REVERSIBLE_WRITE,
                    parameters=(_param("stream_id", S.STRING, "Active stream ID to interrupt"),),
                    required_output_keys=("interrupted",),
                ),
            ),
        ),
    )


def build_default_tool_registry(*, allow_remote_mcp_globally: bool = False) -> ToolRegistry:
    """Create a `ToolRegistry` pre-populated with NOVA's Section 14 MCP catalog and native adapters."""
    registry = ToolRegistry(allow_remote_mcp_globally=allow_remote_mcp_globally)
    for manifest in build_default_mcp_catalog():
        registry.register_server(manifest)
    return registry


__all__ = [
    "JsonSchemaPrimitiveType",
    "McpServerManifest",
    "McpSupplyChainIntegrityError",
    "McpTransportKind",
    "McpTransportSecurityError",
    "ToolContract",
    "ToolNotFoundError",
    "ToolParameterSpec",
    "ToolRegistry",
    "ToolRegistryError",
    "ToolSchemaValidationError",
    "ToolSourceKind",
    "build_default_mcp_catalog",
    "build_default_tool_registry",
]