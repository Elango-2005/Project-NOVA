"""NOVA Phase 9 & Phase 15: Interactive Mission-Control v2 Web UI, Live Brain Switcher & SHA-256 Audit Telemetry (`services/backend/src/nova/api/dashboard.py`).

Plain-English Purpose (The Upgraded Jarvis Cockpit with Brain Gear-Shift & Blockchain Seal):
This module serves NOVA's dark-mode web dashboard at `http://127.0.0.1:8765/` and integrates
everything built across Phases 1–14:

1. **Agentic Command Center (`POST /tasks` & `POST /tasks/{id}/approve`)**:
   - Launch multi-agent goals and approve/deny paused `irreversible_write` actions (`gmail.send_email`).
2. **30-Day SQLite Black-Box Audit Trail + SHA-256 Tamper Seal**:
   - Displays every tool execution from `nova_audit.db` alongside Phase 13's cryptographic
     `TamperEvidentAuditChain` status (`Audit Chain: VERIFIED` + head SHA-256 seal hash).
3. **Local Hybrid RAG Knowledge Ingest (`POST /rag/ingest`)**:
   - Index local markdown/text documents directly into `nova_rag.db`.
4. **MCP Tool Catalog & Permission Tiers**:
   - Inspect all 42 registered MCP tools across 20 MCP servers.
5. **NEW in Phase 15 — Live Brain Switcher & 6GB VRAM GPU Split Panel (`/brain/status` & `/brain/switch`)**:
   - Switch NOVA's active brain with one click between `bootstrap`, `ollama_live`, and `gguf_qwen35`
     (`qwen3.5-9b-Q4_K_M.gguf`), and inspect the live `30/36` GPU layer offload plan!
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from nova.core.model_serving.brain_switcher import (
    BrainEngineMode,
    BrainSwitchRequest,
    NovaBrainAndSecurityController,
    register_brain_and_security_routes,
)

if TYPE_CHECKING:
    from nova.main import NovaRuntimeContainer


def get_dashboard_overview_snapshot(
    container: "NovaRuntimeContainer",
    brain_controller: Optional[NovaBrainAndSecurityController] = None,
) -> Dict[str, Any]:
    """Return a single JSON snapshot combining health, MCP tools, audit logs, brain mode, and SHA-256 seal."""
    health = container.api_service.get_health()
    policies = container.api_service.inspect_permission_policies()
    revoked_tools = set(policies.get("revoked_global_tools", []))

    controller = brain_controller or getattr(container, "brain_controller", None)
    if controller is None:
        controller = NovaBrainAndSecurityController(container, gpu_vram_gb=6.0)
        setattr(container, "brain_controller", controller)

    brain_and_sec = controller.get_security_and_brain_status()

    tool_rows: List[Dict[str, Any]] = []
    for name in container.tool_registry.registered_tool_names:
        contract = container.tool_registry.get_tool(name)
        effective_tier = container.policy_store.get_effective_sensitivity(
            name, contract.sensitivity_tier
        )
        is_revoked = name in revoked_tools
        tool_rows.append(
            {
                "tool_name": contract.tool_name,
                "mcp_server": contract.server_id,
                "sensitivity": contract.sensitivity_tier.value,
                "effective_policy": "revoked" if is_revoked else effective_tier.value,
                "reliability_score": contract.reliability_score,
            }
        )

    audit_logs = container.api_service.query_audit_logs()[-25:]
    return {
        "status": "ok",
        "app_name": health.app_name,
        "version": health.version,
        "offline_by_default": health.offline_by_default,
        "allow_cloud_inference": health.allow_cloud_inference,
        "registered_mcp_servers": health.registered_mcp_servers,
        "registered_tools_count": health.registered_tools,
        "workspace_dir": str(container.workspace_dir),
        "data_dir": str(container.data_dir),
        "brain_and_security": brain_and_sec,
        "tools": tool_rows,
        "recent_audit_logs": [
            {
                "audit_id": entry.audit_id,
                "tool_name": entry.tool_name,
                "agent_role": entry.agent_role.value,
                "sensitivity": entry.effective_sensitivity.value,
                "decision": entry.decision.value,
                "execution_status": entry.execution_status,
                "redacted_arguments": dict(entry.redacted_arguments),
            }
            for entry in audit_logs
        ],
    }


def render_mission_control_html(api_base_url: str = "") -> str:
    """Return the self-contained HTML5/CSS/JS NOVA Mission Control v2 Web UI."""
    base = api_base_url.rstrip("/")
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>NOVA Mission Control — Local-First Personal AI Operating Layer</title>
  <style>
    :root {{
      --bg-primary: #0b0f19;
      --bg-panel: #111827;
      --bg-card: #1f2937;
      --border-subtle: #374151;
      --text-main: #f9fafb;
      --text-muted: #9ca3af;
      --accent-cyan: #06b6d4;
      --accent-emerald: #10b981;
      --accent-amber: #f59e0b;
      --accent-rose: #f43f5e;
    }}
    * {{ box-sizing: border-box; margin: 0; padding: 0; }}
    body {{
      font-family: 'Segoe UI', system-ui, -apple-system, sans-serif;
      background: var(--bg-primary);
      color: var(--text-main);
      line-height: 1.5;
      padding: 20px;
    }}
    header {{
      display: flex;
      justify-content: space-between;
      align-items: center;
      padding: 16px 22px;
      background: var(--bg-panel);
      border: 1px solid var(--border-subtle);
      border-radius: 12px;
      margin-bottom: 20px;
      flex-wrap: wrap;
      gap: 12px;
    }}
    .brand {{
      display: flex;
      align-items: center;
      gap: 12px;
    }}
    .brand-orb {{
      width: 14px;
      height: 14px;
      border-radius: 50%;
      background: var(--accent-cyan);
      box-shadow: 0 0 12px var(--accent-cyan);
    }}
    .brand h1 {{ font-size: 1.25rem; letter-spacing: 0.04em; }}
    .status-pills {{ display: flex; gap: 10px; flex-wrap: wrap; }}
    .pill {{
      font-size: 0.8rem;
      padding: 5px 12px;
      border-radius: 999px;
      background: var(--bg-card);
      border: 1px solid var(--border-subtle);
      color: var(--text-muted);
    }}
    .pill.ok {{ color: var(--accent-emerald); border-color: rgba(16, 185, 129, 0.4); }}
    .pill.cyan {{ color: var(--accent-cyan); border-color: rgba(6, 182, 212, 0.4); }}
    .grid {{
      display: grid;
      grid-template-columns: 1.35fr 1fr;
      gap: 20px;
    }}
    @media (max-width: 960px) {{ .grid {{ grid-template-columns: 1fr; }} }}
    .panel {{
      background: var(--bg-panel);
      border: 1px solid var(--border-subtle);
      border-radius: 12px;
      padding: 20px;
      margin-bottom: 20px;
    }}
    .panel h2 {{
      font-size: 1.05rem;
      margin-bottom: 12px;
      color: var(--accent-cyan);
      display: flex;
      justify-content: space-between;
      align-items: center;
    }}
    label {{ display: block; font-size: 0.85rem; color: var(--text-muted); margin-bottom: 6px; }}
    input, textarea {{
      width: 100%;
      padding: 10px 12px;
      background: var(--bg-primary);
      border: 1px solid var(--border-subtle);
      border-radius: 8px;
      color: var(--text-main);
      font-family: inherit;
      font-size: 0.92rem;
      margin-bottom: 12px;
    }}
    .btn-row {{ display: flex; gap: 10px; flex-wrap: wrap; margin-bottom: 14px; }}
    button {{
      cursor: pointer;
      padding: 9px 16px;
      border-radius: 8px;
      border: 1px solid transparent;
      font-weight: 600;
      font-size: 0.88rem;
      transition: opacity 0.15s ease;
    }}
    button:hover {{ opacity: 0.9; }}
    .btn-primary {{ background: var(--accent-cyan); color: #04131a; }}
    .btn-secondary {{ background: var(--bg-card); color: var(--text-main); border-color: var(--border-subtle); }}
    .btn-approve {{ background: var(--accent-emerald); color: #041f14; }}
    .btn-deny {{ background: var(--accent-rose); color: #fff; }}
    .approval-card {{
      background: rgba(245, 158, 11, 0.1);
      border: 1px solid var(--accent-amber);
      border-radius: 10px;
      padding: 16px;
      margin-top: 14px;
    }}
    .approval-card h3 {{ color: var(--accent-amber); font-size: 0.98rem; margin-bottom: 8px; }}
    pre {{
      background: var(--bg-primary);
      padding: 12px;
      border-radius: 8px;
      border: 1px solid var(--border-subtle);
      overflow-x: auto;
      font-size: 0.82rem;
      color: #d1d5db;
      margin-top: 8px;
    }}
    table {{ width: 100%; border-collapse: collapse; font-size: 0.84rem; margin-top: 8px; }}
    th, td {{ text-align: left; padding: 8px 10px; border-bottom: 1px solid var(--border-subtle); }}
    th {{ color: var(--text-muted); font-weight: 600; }}
    .badge {{
      padding: 2px 8px;
      border-radius: 6px;
      font-size: 0.75rem;
      font-weight: 600;
    }}
    .badge-completed {{ background: rgba(16, 185, 129, 0.2); color: var(--accent-emerald); }}
    .badge-awaiting {{ background: rgba(245, 158, 11, 0.2); color: var(--accent-amber); }}
    .badge-failed {{ background: rgba(244, 63, 94, 0.2); color: var(--accent-rose); }}
  </style>
</head>
<body>
  <header>
    <div class="brand">
      <div class="brand-orb"></div>
      <div>
        <h1>NOVA MISSION CONTROL</h1>
        <div style="font-size: 0.8rem; color: var(--text-muted);">Single-User Local-First Agentic Operating Layer</div>
      </div>
    </div>
    <div class="status-pills">
      <span id="pill-offline" class="pill ok">Offline-by-Default: TRUE</span>
      <span id="pill-brain" class="pill cyan">Brain: BOOTSTRAP</span>
      <span id="pill-gpu" class="pill">GPU Split: 30/36 Layers</span>
      <span id="pill-seal" class="pill ok">Audit Chain: SHA-256 VERIFIED</span>
      <span id="pill-servers" class="pill">MCP Servers: --</span>
      <span id="pill-tools" class="pill">MCP Tools: --</span>
    </div>
  </header>

  <div class="grid">
    <div>
      <!-- 1. Task Command Center -->
      <section class="panel" id="command-center-panel">
        <h2>
          <span>1. Agentic Command Center (State-Graph Orchestrator)</span>
          <span id="task-status-badge" class="badge badge-completed">IDLE</span>
        </h2>
        <label for="goal-input">Enter a natural-language goal for NOVA:</label>
        <input id="goal-input" type="text" value="Save project kickoff summary to notes.txt" />
        <div class="btn-row">
          <button class="btn-primary" onclick="launchTask()">Execute Goal</button>
          <button class="btn-secondary" onclick="setPreset('Save project kickoff summary to notes.txt')">Preset 1: Write Local File</button>
          <button class="btn-secondary" onclick="setPreset('Read invoice.txt and email it to sam@accounting.example.com')">Preset 2: Read File + Email (Triggers Approval Gate)</button>
        </div>
        <div id="approval-container"></div>
        <label>Latest Task Execution State &amp; Plan Graph:</label>
        <pre id="task-output">Ready. Click "Execute Goal" to run a multi-agent task.</pre>
      </section>

      <!-- 2. Black-Box Audit Trail -->
      <section class="panel" id="audit-log-panel">
        <h2>
          <span>2. 30-Day SQLite Black-Box Tool Audit Trail</span>
          <button class="btn-secondary" onclick="refreshDashboard()">Refresh</button>
        </h2>
        <table>
          <thead>
            <tr>
              <th>Tool Name</th>
              <th>Sensitivity</th>
              <th>Gate Decision</th>
              <th>Status</th>
            </tr>
          </thead>
          <tbody id="audit-table-body">
            <tr><td colspan="4" style="color: var(--text-muted);">No tool executions recorded yet.</td></tr>
          </tbody>
        </table>
      </section>
    </div>

    <div>
      <!-- 5. Live Brain Switcher & 6GB VRAM GPU Split -->
      <section class="panel" id="brain-switcher-panel">
        <h2>
          <span>5. Live Brain Switcher &amp; 6GB VRAM GPU Offload</span>
          <span id="active-brain-badge" class="badge badge-completed">BOOTSTRAP</span>
        </h2>
        <div class="btn-row">
          <button class="btn-primary" onclick="switchBrain('gguf_qwen35')">Use Qwen 3.5 9B GGUF (30/36 GPU Split)</button>
          <button class="btn-secondary" onclick="switchBrain('ollama_live')">Use Live Ollama</button>
          <button class="btn-secondary" onclick="switchBrain('bootstrap')">Use Fast Bootstrap</button>
        </div>
        <pre id="brain-status-output">Loading GPU layer split &amp; SHA-256 audit chain status...</pre>
      </section>

      <!-- 3. Hybrid RAG Local Document Ingest -->
      <section class="panel" id="rag-ingest-panel">
        <h2><span>3. Local Hybrid RAG Knowledge Ingest</span></h2>
        <label for="rag-doc-id">Document ID:</label>
        <input id="rag-doc-id" type="text" value="doc_q4_roadmap" />
        <label for="rag-uri">Local Source URI:</label>
        <input id="rag-uri" type="text" value="file:///workspace/q4_roadmap.md" />
        <label for="rag-content">Document Content (Indexed into BM25 + Vector SQLite Store):</label>
        <textarea id="rag-content" rows="3">NOVA Q4 budget is $1,250.00 and prioritizes local-first privacy.</textarea>
        <button class="btn-primary" onclick="ingestRagDoc()">Index into Local RAG</button>
        <pre id="rag-output">No documents ingested in this session yet.</pre>
      </section>

      <!-- 4. Registered MCP Tools & Permission Catalog -->
      <section class="panel" id="mcp-tools-panel">
        <h2><span>4. MCP Tool Catalog &amp; Permission Tiers</span></h2>
        <table>
          <thead>
            <tr>
              <th>MCP Tool</th>
              <th>Server</th>
              <th>Tier</th>
              <th>Policy</th>
            </tr>
          </thead>
          <tbody id="tools-table-body">
            <tr><td colspan="4" style="color: var(--text-muted);">Loading tool registry...</td></tr>
          </tbody>
        </table>
      </section>
    </div>
  </div>

  <script>
    const API_BASE = "{base}";
    let currentTaskId = null;

    function setPreset(text) {{
      document.getElementById('goal-input').value = text;
    }}

    async function switchBrain(mode) {{
      const res = await fetch(API_BASE + '/brain/switch', {{
        method: 'POST',
        headers: {{ 'Content-Type': 'application/json' }},
        body: JSON.stringify({{ mode: mode, ollama_model: 'qwen3.5-9b', gpu_vram_gb: 6.0 }})
      }});
      await res.json();
      refreshDashboard();
    }}

    async function refreshDashboard() {{
      try {{
        const res = await fetch(API_BASE + '/ui/overview');
        const data = await res.json();
        document.getElementById('pill-offline').textContent = 'Offline-by-Default: ' + (data.offline_by_default ? 'TRUE' : 'FALSE');
        document.getElementById('pill-servers').textContent = 'MCP Servers: ' + data.registered_mcp_servers;
        document.getElementById('pill-tools').textContent = 'MCP Tools: ' + data.registered_tools_count;

        if (data.brain_and_security) {{
          const bs = data.brain_and_security;
          const modeUpper = bs.active_brain_mode.toUpperCase();
          document.getElementById('pill-brain').textContent = 'Brain: ' + modeUpper;
          document.getElementById('active-brain-badge').textContent = modeUpper;
          const plan = bs.gpu_offload_plan;
          document.getElementById('pill-gpu').textContent = `GPU Split: ${{plan.recommended_gpu_layers}}/${{plan.total_layers}} Layers (${{plan.gpu_vram_gb}}GB)`;
          const sec = bs.security_telemetry;
          document.getElementById('pill-seal').textContent = 'Audit Seal: ' + (sec.audit_chain_valid ? 'VERIFIED (' + sec.head_seal_hash.slice(0, 8) + '...)' : 'TAMPERED!');
          document.getElementById('brain-status-output').textContent = JSON.stringify(bs, null, 2);
        }}

        const auditBody = document.getElementById('audit-table-body');
        if (data.recent_audit_logs && data.recent_audit_logs.length > 0) {{
          auditBody.innerHTML = data.recent_audit_logs.map(row => `
            <tr>
              <td><code>${{row.tool_name}}</code></td>
              <td>${{row.sensitivity}}</td>
              <td><span class="badge ${{row.decision.includes('approved') ? 'badge-completed' : 'badge-failed'}}">${{row.decision}}</span></td>
              <td>${{row.execution_status}}</td>
            </tr>
          `).join('');
        }}

        const toolsBody = document.getElementById('tools-table-body');
        if (data.tools && data.tools.length > 0) {{
          toolsBody.innerHTML = data.tools.slice(0, 12).map(t => `
            <tr>
              <td><code>${{t.tool_name}}</code></td>
              <td>${{t.mcp_server}}</td>
              <td>${{t.sensitivity}}</td>
              <td>${{t.effective_policy}}</td>
            </tr>
          `).join('');
        }}
      }} catch (err) {{
        console.error('Dashboard refresh error:', err);
      }}
    }}

    function renderTaskResponse(task) {{
      currentTaskId = task.task_id;
      const badge = document.getElementById('task-status-badge');
      badge.textContent = task.status.toUpperCase();
      badge.className = 'badge ' + (
        task.status === 'completed' ? 'badge-completed' :
        task.status === 'awaiting_approval' ? 'badge-awaiting' : 'badge-failed'
      );
      document.getElementById('task-output').textContent = JSON.stringify(task, null, 2);

      const approvalBox = document.getElementById('approval-container');
      if (task.pending_approvals && task.pending_approvals.length > 0) {{
        const card = task.pending_approvals[0];
        approvalBox.innerHTML = `
          <div class="approval-card">
            <h3>Human Permission Gate Paused Execution (${{card.declared_sensitivity}})</h3>
            <p style="font-size: 0.88rem; margin-bottom: 8px;">
              Agent <b>${{card.agent_role}}</b> on node <code>${{card.node_id}}</code> requests permission to execute
              <code>${{card.tool_name}}</code>:
            </p>
            <pre>${{JSON.stringify(card.arguments, null, 2)}}</pre>
            <div class="btn-row" style="margin-top: 12px; margin-bottom: 0;">
              <button class="btn-approve" onclick="submitApproval('${{card.node_id}}', '${{card.proposal_id}}', true)">Approve &amp; Resume</button>
              <button class="btn-deny" onclick="submitApproval('${{card.node_id}}', '${{card.proposal_id}}', false)">Deny &amp; Abort</button>
            </div>
          </div>
        `;
      }} else {{
        approvalBox.innerHTML = '';
      }}
      refreshDashboard();
    }}

    async function launchTask() {{
      const goal = document.getElementById('goal-input').value.trim();
      if (!goal) return;
      const taskId = 'task_' + Date.now();
      const res = await fetch(API_BASE + '/tasks', {{
        method: 'POST',
        headers: {{ 'Content-Type': 'application/json' }},
        body: JSON.stringify({{
          task_id: taskId,
          session_id: 'sess_ui_live',
          user_goal: goal
        }})
      }});
      const data = await res.json();
      renderTaskResponse(data);
    }}

    async function submitApproval(nodeId, proposalId, approved) {{
      if (!currentTaskId) return;
      const res = await fetch(API_BASE + '/tasks/' + currentTaskId + '/approve', {{
        method: 'POST',
        headers: {{ 'Content-Type': 'application/json' }},
        body: JSON.stringify({{
          node_id: nodeId,
          proposal_id: proposalId,
          approved: approved,
          reason: approved ? 'Approved via NOVA Mission Control UI' : 'Denied via NOVA Mission Control UI'
        }})
      }});
      const data = await res.json();
      renderTaskResponse(data);
    }}

    async function ingestRagDoc() {{
      const docId = document.getElementById('rag-doc-id').value.trim();
      const uri = document.getElementById('rag-uri').value.trim();
      const content = document.getElementById('rag-content').value.trim();
      const res = await fetch(API_BASE + '/rag/ingest', {{
        method: 'POST',
        headers: {{ 'Content-Type': 'application/json' }},
        body: JSON.stringify({{ document_id: docId, source_uri: uri, content: content }})
      }});
      const data = await res.json();
      document.getElementById('rag-output').textContent = JSON.stringify(data, null, 2);
      refreshDashboard();
    }}

    refreshDashboard();
  </script>
</body>
</html>
"""


def register_mission_control_routes(
    fastapi_app: Any,
    container: "NovaRuntimeContainer",
) -> None:
    """Attach `GET /`, `GET /ui/overview`, `GET /brain/status`, and `POST /brain/switch` to FastAPI."""
    controller = getattr(container, "brain_controller", None)
    if controller is None:
        controller = NovaBrainAndSecurityController(container, gpu_vram_gb=6.0)
        setattr(container, "brain_controller", controller)

    register_brain_and_security_routes(fastapi_app, controller)

    try:
        from fastapi.responses import HTMLResponse
    except ImportError:  # pragma: no cover
        return

    @fastapi_app.get("/", response_class=HTMLResponse, include_in_schema=False)
    @fastapi_app.get("/dashboard", response_class=HTMLResponse, include_in_schema=False)
    def serve_mission_control_ui() -> Any:
        return HTMLResponse(content=render_mission_control_html())

    @fastapi_app.get("/ui/overview")
    def get_ui_overview() -> Dict[str, Any]:
        return get_dashboard_overview_snapshot(container, brain_controller=controller)


def build_mission_control_app(
    *,
    data_dir: Optional[Path] = None,
    workspace_dir: Optional[Path] = None,
) -> Any:
    """Build the full NOVA FastAPI app with the live Mission Control v2 Web UI mounted at `/`."""
    from nova.main import build_nova_app

    fastapi_app = build_nova_app(data_dir=data_dir, workspace_dir=workspace_dir)
    container = fastapi_app.state.nova_container
    register_mission_control_routes(fastapi_app, container)
    return fastapi_app


try:
    app = build_mission_control_app()
except Exception:  # pragma: no cover
    app = None


__all__ = [
    "BrainEngineMode",
    "BrainSwitchRequest",
    "app",
    "build_mission_control_app",
    "get_dashboard_overview_snapshot",
    "register_mission_control_routes",
    "render_mission_control_html",
]