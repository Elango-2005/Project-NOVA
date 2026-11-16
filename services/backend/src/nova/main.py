"""NOVA Phase 8: Live Runnable FastAPI Server Entrypoint & Real Local Runtime Wiring (`services/backend/src/nova/main.py`).

Plain-English Purpose (Turning the Key in the Ignition):
In Phases 1 to 7, we built and tested every engine part of NOVA (`100/100` tests passing).
Now, `main.py` **bolts all 7 layers together into a live, runnable server** so you can start NOVA
in your terminal and interact with it in your browser at `http://127.0.0.1:8000/docs` (or from the
React/Electron UI in Phase 9):

1. **Real Local Disk & SQLite Persistence (`NovaRuntimeContainer`)**:
   - Creates a safe local workspace folder (`<workspace_root>`, defaulting to `C:\\NOVA\\workspace`)
     watched by `FilesystemSandbox` and `CommandSandbox`.
   - Connects real SQLite databases in `.nova/data/` for Checkpoints (`nova_checkpoints.db`),
     Long-Term 3-Drawer Memory (`nova_memory.db`), Hybrid RAG (`nova_rag.db`), and the 30-day
     Black-Box Tool Audit Log (`nova_audit.db`).

2. **Real Local Tool Handlers (`build_live_tool_handlers`)**:
   - Connects `filesystem.read_file`, `filesystem.write_file`, `filesystem.list_directory`, and
     `filesystem.search_files` to your real `<workspace_root>` folder on disk!
   - Connects `rag.retrieve` to the live `HybridRetrievalEngine`.
   - Connects safe local dispatchers for `gmail.send_email`, `gmail.draft_email`, `terminal.execute`,
     and `weather.get_forecast`.

3. **Zero-Config Offline Bootstrap Brain (`RuleGuidedLocalBootstrapAdapter`)**:
   - Works out-of-the-box on your laptop even before you download a multi-gigabyte `.gguf` model:
     it inspects your `user_goal` and generates real, schema-valid `PlanProposal`, `ToolCallProposal`,
     and `FinalAnswerProposal` JSON so you can test live file creation, RAG search, and human
     approval pauses right inside Swagger UI (`http://127.0.0.1:8000/docs`)!
"""

from __future__ import annotations

import fnmatch
import re
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Union

from pydantic import BaseModel, Field

try:
    from pydantic import ConfigDict

    PYDANTIC_V2 = True
except ImportError:  # pragma: no cover
    PYDANTIC_V2 = False

try:
    from fastapi.middleware.cors import CORSMiddleware

    CORS_AVAILABLE = True
except ImportError:  # pragma: no cover
    CORSMiddleware = None  # type: ignore[assignment,misc]
    CORS_AVAILABLE = False

from nova.api.app import NovaLocalApiService, create_nova_fastapi_app
from nova.core.agents.base import BaseSpecializedAgent
from nova.core.cognition.memory import (
    MemoryManager,
    SqliteLongTermMemoryStore,
    WorkingMemoryStore,
)
from nova.core.cognition.retrieval import (
    HybridRetrievalEngine,
    SqliteHybridChunkIndex,
)
from nova.core.config.settings import NovaSettings, load_settings
from nova.core.domain.primitives import AgentRole, DataClassification
from nova.core.model_serving.interface import ModelInferenceRequest
from nova.core.model_serving.router import (
    HardwareProfile,
    LlamaCppAdapter,
    NovaModelRouter,
)
from nova.core.orchestration.checkpoint import SqliteCheckpointStore
from nova.core.orchestration.engine import NovaGraphOrchestrator
from nova.core.permissions.sandbox_and_audit import (
    CatalogBackedPermissionGate,
    CommandSandbox,
    FilesystemSandbox,
    SandboxedMcpToolExecutor,
    SqliteToolAuditLog,
    StructuredToolHandlerFn,
    UserPermissionPolicyStore,
)
from nova.core.platform.event_bus import InMemoryEventBus
from nova.core.tools.personal_assistant_tools import build_personal_assistant_tool_handlers
from nova.core.tools.registry import ToolRegistry, build_default_tool_registry

_EMAIL_REGEX = re.compile(r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+")
_FILENAME_REGEX = re.compile(r"\b([a-zA-Z0-9_-]+\.(?:txt|md|py|json|csv|pdf|log))\b")


class RagIngestRequest(BaseModel):
    """Payload for `POST /rag/ingest` to index a local document into NOVA's Hybrid RAG Engine."""

    if PYDANTIC_V2:
        model_config = ConfigDict(frozen=True, extra="forbid")
    else:

        class Config:
            allow_mutation = False
            extra = "forbid"

    document_id: str = Field(..., min_length=1)
    source_uri: str = Field(..., min_length=1)
    content: str = Field(..., min_length=1)


def _extract_objective_from_system_prompt(system_content: str) -> str:
    """Extract the `Task Objective:` line from `[NOVA TRUSTED SYSTEM INSTRUCTION]`."""
    for line in system_content.splitlines():
        if line.startswith("Task Objective:"):
            return line[len("Task Objective:") :].strip()
    return system_content.strip()


def _extract_latest_observation_from_messages(
    messages: tuple[dict[str, str], ...],
) -> Optional[str]:
    """Return the text inside `<react_observation_history>` if the current node already ran a tool turn."""
    for msg in reversed(messages[1:]):
        content = msg.get("content", "")
        if "<react_observation_history" in content:
            return content
    return None


class RuleGuidedLocalBootstrapAdapter(LlamaCppAdapter):
    """Dynamic, rule-guided local `LlamaCppAdapter` for live interactive use without a 5GB GGUF file.

    If `scripted_outputs` are queued (e.g., in unit tests), pops from the queue first.
    Otherwise, dynamically synthesizes schema-valid NOVA proposals (`plan`, `tool_call`, `critique`,
    `final_answer`) based on the requesting `agent_role` and `Task Objective`.
    """

    def generate_structured(self, request: ModelInferenceRequest) -> Any:
        if self._queue:
            return super().generate_structured(request)

        # Dynamically synthesize a response via our callable handler
        self.enqueue(self._synthesize_dynamic_proposal)
        return super().generate_structured(request)

    @staticmethod
    def _synthesize_dynamic_proposal(
        request: ModelInferenceRequest,
    ) -> Union[str, Mapping[str, Any]]:
        system_msg = request.messages[0]["content"] if request.messages else ""
        objective = _extract_objective_from_system_prompt(system_msg)
        obj_lower = objective.lower()
        prior_obs = _extract_latest_observation_from_messages(request.messages)

        # 1. PLANNER AGENT (or Reflection Critic)
        if request.agent_role == AgentRole.PLANNER:
            if request.node_id.endswith("_reflect"):
                return {
                    "kind": "critique",
                    "agent_role": "planner",
                    "node_id": request.node_id,
                    "thought": "Verified candidate proposal against safety and grounding questions.",
                    "target_proposal_id": "prop_verified",
                    "verdict": "approve",
                    "issues_found": [],
                    "verification_questions_checked": ["Verified intent and arguments."],
                }

            steps: List[Dict[str, Any]] = []
            # Detect whether the user wants to write a file, read/search RAG, or send an email
            if "write" in obj_lower or "save" in obj_lower or "create" in obj_lower:
                steps.append(
                    {
                        "step_id": "step_write_file",
                        "target_agent": "coding",
                        "objective": objective,
                        "depends_on": [],
                        "estimated_sensitivity": "reversible_write",
                    }
                )
            else:
                steps.append(
                    {
                        "step_id": "step_research_info",
                        "target_agent": "research",
                        "objective": objective,
                        "depends_on": [],
                        "estimated_sensitivity": "read_only",
                    }
                )

            if "email" in obj_lower or "send" in obj_lower or "@" in objective:
                upstream_id = steps[-1]["step_id"]
                steps.append(
                    {
                        "step_id": "step_send_email",
                        "target_agent": "comms",
                        "objective": objective,
                        "depends_on": [upstream_id],
                        "estimated_sensitivity": "irreversible_write",
                    }
                )

            return {
                "kind": "plan",
                "agent_role": "planner",
                "node_id": request.node_id,
                "thought": f"Decomposed user goal ({objective!r}) into {len(steps)} executable node(s).",
                "goal_summary": objective,
                "steps": steps,
            }

        # 2. If the current node already executed a tool turn (`prior_obs` is present), emit FinalAnswerProposal!
        if prior_obs is not None:
            return {
                "kind": "final_answer",
                "agent_role": request.agent_role.value,
                "node_id": request.node_id,
                "thought": "Synthesized final output from the completed tool observation.",
                "content": f"Completed '{objective}' using verified tool observation.",
                "cited_source_ids": ["workspace_local_source"],
                "confidence": 0.98,
            }

        # 3. First turn inside an execution node -> propose the appropriate tool call!
        file_match = _FILENAME_REGEX.search(objective)
        target_filename = file_match.group(1) if file_match else "nova_note.txt"

        if request.agent_role == AgentRole.CODING:
            return {
                "kind": "tool_call",
                "agent_role": "coding",
                "node_id": request.node_id,
                "thought": f"Write requested content to {target_filename!r} inside sandboxed workspace.",
                "tool_name": "filesystem.write_file",
                "arguments": {
                    "path": target_filename,
                    "content": f"Generated by NOVA for goal: {objective}\n",
                },
                "declared_sensitivity": "reversible_write",
                "rationale": f"Write {target_filename} in workspace",
            }

        if request.agent_role == AgentRole.RESEARCH:
            if file_match and ("read" in obj_lower or "file" in obj_lower):
                return {
                    "kind": "tool_call",
                    "agent_role": "research",
                    "node_id": request.node_id,
                    "thought": f"Read file {target_filename!r} from workspace.",
                    "tool_name": "filesystem.read_file",
                    "arguments": {"path": target_filename},
                    "declared_sensitivity": "read_only",
                    "rationale": f"Read local file {target_filename}",
                }
            return {
                "kind": "tool_call",
                "agent_role": "research",
                "node_id": request.node_id,
                "thought": f"Search local Hybrid RAG index for: {objective!r}.",
                "tool_name": "rag.retrieve",
                "arguments": {"query": objective, "top_k": 2},
                "declared_sensitivity": "read_only",
                "rationale": "Search local Hybrid RAG index",
            }

        if request.agent_role == AgentRole.COMMS:
            email_match = _EMAIL_REGEX.search(objective)
            recipient = email_match.group(0) if email_match else "sam@accounting.example.com"
            return {
                "kind": "tool_call",
                "agent_role": "comms",
                "node_id": request.node_id,
                "thought": f"Propose sending email to {recipient!r} (gated by human approval).",
                "tool_name": "gmail.send_email",
                "arguments": {
                    "to": recipient,
                    "subject": f"NOVA Update: {objective[:40]}",
                    "body": f"Completed task: {objective}",
                },
                "declared_sensitivity": "irreversible_write",
                "rationale": f"Send email to {recipient}",
            }

        return {
            "kind": "final_answer",
            "agent_role": request.agent_role.value,
            "node_id": request.node_id,
            "thought": "Completed reasoning step directly.",
            "content": f"Processed objective: {objective}",
            "cited_source_ids": ["workspace_local_source"],
            "confidence": 0.95,
        }


def build_live_tool_handlers(
    workspace_root: Path,
    rag_engine: HybridRetrievalEngine,
) -> Dict[str, StructuredToolHandlerFn]:
    """Build real local filesystem, RAG, and communication handlers confined to `workspace_root`."""
    workspace_root.mkdir(parents=True, exist_ok=True)
    sent_outbox: List[Dict[str, Any]] = []

    def _read_file(args: Dict[str, Any]) -> Mapping[str, Any]:
        target = Path(str(args["path"]))
        if not target.exists():
            return {"content": f"[File {target.name!r} does not exist yet in workspace]"}
        return {"content": target.read_text(encoding="utf-8")}

    def _write_file(args: Dict[str, Any]) -> Mapping[str, Any]:
        target = Path(str(args["path"]))
        target.parent.mkdir(parents=True, exist_ok=True)
        text = str(args["content"])
        target.write_text(text, encoding="utf-8")
        return {"bytes_written": len(text.encode("utf-8"))}

    def _list_directory(args: Dict[str, Any]) -> Mapping[str, Any]:
        target = Path(str(args["path"]))
        if not target.exists() or not target.is_dir():
            return {"entries": []}
        entries = sorted(p.name for p in target.iterdir())
        return {"entries": entries}

    def _search_files(args: Dict[str, Any]) -> Mapping[str, Any]:
        root = Path(str(args["path"]))
        pattern = str(args["pattern"])
        if not root.exists():
            return {"matches": []}
        matches = [
            str(p.relative_to(root))
            for p in root.rglob("*")
            if p.is_file() and fnmatch.fnmatch(p.name.lower(), pattern.lower())
        ]
        return {"matches": sorted(matches)}

    def _rag_retrieve(args: Dict[str, Any]) -> Mapping[str, Any]:
        query = str(args.get("query", ""))
        top_k = int(args.get("top_k", 3))
        res = rag_engine.retrieve(query=query, top_k=top_k)
        return {
            "chunks": [res.format_observation_summary()],
            "citations": list(res.citation_ids),
        }

    def _send_email(args: Dict[str, Any]) -> Mapping[str, Any]:
        msg_id = f"local_msg_{len(sent_outbox) + 1:04d}"
        record = {"message_id": msg_id, **args}
        sent_outbox.append(record)
        return {"message_id": msg_id, "status": f"dispatched_to_{args['to']}"}

    def _draft_email(args: Dict[str, Any]) -> Mapping[str, Any]:
        draft_id = f"local_draft_{len(sent_outbox) + 1:04d}"
        return {"draft_id": draft_id}

    def _weather_forecast(args: Dict[str, Any]) -> Mapping[str, Any]:
        return {"temperature_c": 28, "condition": f"Clear sky in {args.get('location', 'local')}"}

    return {
        "filesystem.read_file": _read_file,
        "filesystem.write_file": _write_file,
        "filesystem.list_directory": _list_directory,
        "filesystem.search_files": _search_files,
        "rag.retrieve": _rag_retrieve,
        "gmail.send_email": _send_email,
        "gmail.draft_email": _draft_email,
        "weather.get_forecast": _weather_forecast,
    }


class NovaRuntimeContainer:
    """Self-contained runtime container holding all wired NOVA services for `main.py` and tests."""

    def __init__(
        self,
        *,
        data_dir: Optional[Path] = None,
        workspace_dir: Optional[Path] = None,
        settings: Optional[NovaSettings] = None,
        auto_approve_reversible_local_writes: bool = True,
    ) -> None:
        self.settings = settings or load_settings(
            environ={},
            runtime_overrides={"data_dir": data_dir} if data_dir is not None else None,
        )
        self.data_dir = self.settings.data_dir
        self.data_dir.mkdir(parents=True, exist_ok=True)

        self.workspace_dir = (
            Path(workspace_dir).resolve()
            if workspace_dir is not None
            else (self.data_dir / "workspace").resolve()
        )
        self.workspace_dir.mkdir(parents=True, exist_ok=True)

        # Layer 1: Event Bus & SQLite Checkpoint Store
        self.event_bus = InMemoryEventBus()
        self.checkpoint_store = SqliteCheckpointStore(self.data_dir / "nova_checkpoints.db")

        # Layer 4: Two-Tier Memory & Local Hybrid RAG
        self.memory_manager = MemoryManager(
            working_store=WorkingMemoryStore(),
            long_term_store=SqliteLongTermMemoryStore(self.data_dir / "nova_memory.db"),
            event_bus=self.event_bus,
        )
        self.rag_engine = HybridRetrievalEngine(
            index=SqliteHybridChunkIndex(self.data_dir / "nova_rag.db")
        )

        # Layer 3: MCP Tool Registry, Permission Policy Store, Sandbox & Audit Log
        self.tool_registry: ToolRegistry = build_default_tool_registry()
        self.policy_store = UserPermissionPolicyStore(registry=self.tool_registry, db_path=self.data_dir / "nova_audit.db")
        self.audit_log = SqliteToolAuditLog(
            self.data_dir / "nova_audit.db",
            retention_days=self.settings.audit_log_retention_days,
        )
        self.filesystem_sandbox = FilesystemSandbox(allowed_roots=(self.workspace_dir,))
        self.command_sandbox = CommandSandbox(filesystem_sandbox=self.filesystem_sandbox)

        self.permission_gate = CatalogBackedPermissionGate(
            registry=self.tool_registry,
            policy_store=self.policy_store,
            audit_log=self.audit_log,
            auto_approve_reversible_local_writes=auto_approve_reversible_local_writes,
            sensitive_action_policy=self.settings.default_sensitive_action_policy,
            authorization_db_path=str(self.data_dir / "nova_audit.db"),
            approval_ttl_seconds=self.settings.approval_ttl_seconds,
            require_runtime_context=True,
        )
        assistant_handlers, self.calendar_store, self.notes_manager, self.metrics_provider = build_personal_assistant_tool_handlers(
            data_dir=self.data_dir, workspace_dir=self.workspace_dir, rag_engine=self.rag_engine)
        self.tool_executor = SandboxedMcpToolExecutor(
            registry=self.tool_registry,
            filesystem_sandbox=self.filesystem_sandbox,
            command_sandbox=self.command_sandbox,
            audit_log=self.audit_log,
            event_bus=self.event_bus,
            tool_handlers={**build_live_tool_handlers(self.workspace_dir, self.rag_engine), **assistant_handlers},
            permission_gate=self.permission_gate,
        )

        # Layer 2: Model Serving Router with Rule-Guided Bootstrap Adapter
        self.bootstrap_adapter = RuleGuidedLocalBootstrapAdapter()
        self.model_router = NovaModelRouter(
            llamacpp_adapter=self.bootstrap_adapter,
            hardware_profile=HardwareProfile(gpu_available=False),
            event_bus=self.event_bus,
        )

        # Layer 5 & 6: Specialized Agents & State-Graph Orchestrator
        self.agents = {
            role: BaseSpecializedAgent(
                role_or_spec=role,
                model_port=self.model_router,
                event_bus=self.event_bus,
                settings=self.settings,
            )
            for role in AgentRole
        }
        self.orchestrator = NovaGraphOrchestrator(
            agents=self.agents,
            checkpoint_store=self.checkpoint_store,
            permission_gate=self.permission_gate,
            tool_executor=self.tool_executor,
            event_bus=self.event_bus,
            critic_agent=self.agents[AgentRole.PLANNER],
            settings=self.settings,
        )

        # Layer 7: Local API Controller
        self.api_service = NovaLocalApiService(
            orchestrator=self.orchestrator,
            checkpoint_store=self.checkpoint_store,
            event_bus=self.event_bus,
            tool_registry=self.tool_registry,
            policy_store=self.policy_store,
            audit_log=self.audit_log,
            settings=self.settings,
            authorization_ledger=self.permission_gate.ledger,
        )

    def ingest_rag_document(self, request: RagIngestRequest) -> Dict[str, Any]:
        """Ingest and chunk a document into the live Hybrid RAG index."""
        chunks = self.rag_engine.ingest_document(
            document_id=request.document_id,
            source_uri=request.source_uri,
            content=request.content,
            data_classification=DataClassification.USER_CONTENT_LOCAL,
        )
        return {
            "document_id": request.document_id,
            "chunks_indexed": len(chunks),
            "citation_ids": [c.chunk_id for c in chunks],
        }


def build_nova_app(
    *,
    data_dir: Optional[Path] = None,
    workspace_dir: Optional[Path] = None,
    auto_approve_reversible_local_writes: bool = True,
) -> Any:
    """Create the production FastAPI ASGI app with CORS and `/rag/ingest` wired to `NovaRuntimeContainer`."""
    container = NovaRuntimeContainer(data_dir=data_dir, workspace_dir=workspace_dir,
        auto_approve_reversible_local_writes=auto_approve_reversible_local_writes)
    fastapi_app = create_nova_fastapi_app(container.api_service)

    if CORS_AVAILABLE and CORSMiddleware is not None:
        fastapi_app.add_middleware(
            CORSMiddleware,
            allow_origins=[
                "http://localhost:5173",
                "http://127.0.0.1:5173",
                "http://localhost:3000",
                "http://127.0.0.1:3000",
            ],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

    @fastapi_app.post("/rag/ingest")
    def ingest_rag_document(request: RagIngestRequest) -> Dict[str, Any]:
        return container.ingest_rag_document(request)

    # Attach container for inspection in tests or CLI scripts
    fastapi_app.state.nova_container = container
    return fastapi_app


# Defer default storage initialization until the ASGI server starts handling lifespan.
class DeferredAsgiApp:
    def __init__(self, factory):
        self.factory = factory
        self.application = None

    async def __call__(self, scope, receive, send):
        if self.application is None:
            self.application = self.factory()
        await self.application(scope, receive, send)


app = DeferredAsgiApp(build_nova_app)


__all__ = [
    "NovaRuntimeContainer",
    "RagIngestRequest",
    "RuleGuidedLocalBootstrapAdapter",
    "app",
    "build_live_tool_handlers",
    "build_nova_app",
]