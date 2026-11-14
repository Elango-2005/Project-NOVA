# NOVA — Single-User, Local-First Personal AI Operating Layer

NOVA is a privacy-first, offline-capable, multi-agent personal AI operating layer built across **7 architectural layers** and verified by **108 automated tests**.

## Architectural Layers (7-Layer Stack)

| Layer | Module Path | Responsibility |
| :--- | :--- | :--- |
| **Layer 1: Platform & Schemas** | `nova.core.config`, `nova.core.domain`, `nova.core.platform` | Immutable settings (`offline_by_default=True`), strict domain primitives, untrusted data envelopes (`[NOVA UNTRUSTED EXTERNAL CONTENT]`), and pub/sub `InMemoryEventBus`. |
| **Layer 2: Model Serving Router** | `nova.core.model_serving` | Hardware-aware `NovaModelRouter`, local `LlamaCppAdapter` / `OllamaAdapter`, zero-config `RuleGuidedLocalBootstrapAdapter`, and offline voice/vision pipeline (`multimodal.py`). |
| **Layer 3: MCP Tools & Security** | `nova.core.tools`, `nova.core.permissions` | 20 MCP server manifests (42 typed tool contracts), `CatalogBackedPermissionGate` (cannot be bypassed by agent prompts), `FilesystemSandbox`, `CommandSandbox`, and 30-day SQLite Black-Box Audit Log (`nova_audit.db`). |
| **Layer 4: Two-Tier Memory & RAG** | `nova.core.cognition.memory`, `nova.core.cognition.retrieval` | Session Working Memory + 3-Drawer SQLite Long-Term Memory (`episodic`, `semantic`, `procedural` in `nova_memory.db`) and local BM25 + Vector Reciprocal Rank Fusion Hybrid RAG (`nova_rag.db`). |
| **Layer 5: Specialized ReAct Agents** | `nova.core.agents`, `nova.core.cognition.planning`, `reflection.py` | 7 specialized agents (`planner`, `research`, `coding`, `browser`, `desktop`, `comms`, `memory`), DAG `PlanGraph` validation, and pre-execution Reflection Critic. |
| **Layer 6: Durable Orchestrator** | `nova.core.orchestration` | `NovaGraphOrchestrator` with SQLite checkpointing (`nova_checkpoints.db`), crash recovery, and human-in-the-loop pause/resume (`AWAITING_APPROVAL`). |
| **Layer 7: FastAPI, UI & CLI** | `nova.api`, `nova.main`, `nova.cli` | Local FastAPI service (`app.py`), dark-mode **NOVA Mission Control** Web UI (`dashboard.py`), live runtime container (`main.py`), and terminal CLI & 7-Layer System Doctor (`cli.py`). |

## Quickstart Commands (PowerShell from `C:\NOVA`)

1. **Run the Full 108-Test Verification Suite**:
   ```powershell
   pytest
   ```
   *(With `pyproject.toml` saved in `C:\NOVA`, `pytest` automatically loads `services/backend/src`!)*

2. **Run the 7-Layer System Doctor**:
   ```powershell
   python -m nova.cli doctor
   ```

3. **Run a Goal or Ingest Knowledge from Terminal**:
   ```powershell
   python -m nova.cli ask "Save project kickoff summary to notes.txt"
   python -m nova.cli audit --limit 5
   ```

4. **Launch the Live NOVA Mission Control Web UI (`http://127.0.0.1:8000/`)**:
   ```powershell
   uvicorn nova.api.dashboard:app --reload --port 8000
   ```
   - **Mission Control Dashboard**: `http://127.0.0.1:8000/`
   - **Interactive OpenAPI / Swagger Docs**: `http://127.0.0.1:8000/docs`
