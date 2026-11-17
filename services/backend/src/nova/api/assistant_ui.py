"""NOVA Phase 18: Clean, User-Friendly Personal AI Assistant Web App (`services/backend/src/nova/api/assistant_ui.py`).

Plain-English Purpose (Replacing Technical Developer JSON with a Clean Everyday Assistant):
Earlier, the engineering dashboard showed raw JSON objects and technical labels (`BM25`, `DAG`,
`irreversible_write`). Now, `assistant_ui.py` gives you a **clean, warm, everyday Personal AI
Assistant screen** at `http://127.0.0.1:8765/` designed for real human use:

1. **Conversational Chat Bubble Stream (Left Column)**:
   - Chat with NOVA in plain English (just like ChatGPT), with 1-click starter buttons for saving
     study notes, scheduling calendar events, asking questions about your notes, or testing email
     approvals.
   - Zero raw JSON dumps—every reply is formatted as a clean chat message with summary badges and
     simple **"Yes, Approve"** / **"Deny Action"** buttons when NOVA asks for permission.

2. **My Upcoming Schedule & Quick Event Creator (Right Column, Card 1)**:
   - View your scheduled classes, meetings, and study sessions at a glance, or add a new event
     using simple Title / Start / End boxes.

3. **My Saved Notes & Quick Note Pad (Right Column, Card 2)**:
   - View all your saved Markdown notes and add new ones with one click (automatically indexed so
     you can ask NOVA questions about them in chat!).

4. **Simple Laptop Health & Activity History (Right Column, Card 3)**:
   - Shows your free disk space in GB, active AI brain selector (`Fast Offline`, `Qwen 3.5 9B`,
     or `Ollama`), and a plain-English history of what NOVA has done for you.
"""

from __future__ import annotations

from nova.core.orchestration.ports import ExecutionOutcomeUnavailable

from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from pydantic import BaseModel, Field

try:
    from pydantic import ConfigDict

    PYDANTIC_V2 = True
except ImportError:  # pragma: no cover
    PYDANTIC_V2 = False

from nova.core.cognition.conversation_session import PersonalAssistantChatSession
from nova.core.model_serving.brain_switcher import (
    BrainEngineMode,
    BrainSwitchRequest,
    NovaBrainAndSecurityController,
    register_brain_and_security_routes,
)

if TYPE_CHECKING:
    from nova.main import NovaRuntimeContainer
    from nova.core.permissions.sandbox_and_audit import ToolAuditEntry


class AssistantChatRequest(BaseModel):
    """Payload for `POST /assistant/chat`."""

    if PYDANTIC_V2:
        model_config = ConfigDict(frozen=True, extra="forbid")
    else:

        class Config:
            allow_mutation = False
            extra = "forbid"

    message: str = Field(..., min_length=1)


class AssistantApproveRequest(BaseModel):
    """Payload for `POST /assistant/approve`."""

    if PYDANTIC_V2:
        model_config = ConfigDict(frozen=True, extra="forbid")
    else:

        class Config:
            allow_mutation = False
            extra = "forbid"

    task_id: str = Field(..., min_length=1)
    node_id: str = Field(..., min_length=1)
    proposal_id: str = Field(..., min_length=1)
    approved: bool


class QuickNoteRequest(BaseModel):
    """Payload for `POST /assistant/quick-note`."""

    if PYDANTIC_V2:
        model_config = ConfigDict(frozen=True, extra="forbid")
    else:

        class Config:
            allow_mutation = False
            extra = "forbid"

    title: str = Field(..., min_length=1)
    body: str = Field(..., min_length=1)


class QuickEventRequest(BaseModel):
    """Payload for `POST /assistant/quick-event`."""

    if PYDANTIC_V2:
        model_config = ConfigDict(frozen=True, extra="forbid")
    else:

        class Config:
            allow_mutation = False
            extra = "forbid"

    title: str = Field(..., min_length=1)
    start_iso: str = Field(..., min_length=1)
    end_iso: str = Field(..., min_length=1)


_FRIENDLY_TOOL_NAMES: Dict[str, str] = {
    "filesystem.write_file": "Created a file in your workspace",
    "filesystem.read_file": "Read a file from your workspace",
    "filesystem.list_directory": "Listed your workspace folder",
    "filesystem.search_files": "Searched your workspace files",
    "rag.retrieve": "Searched your saved notes",
    "gmail.send_email": "Sent an email",
    "gmail.draft_email": "Drafted an email",
    "calendar.create_event": "Scheduled a calendar event",
    "calendar.list_events": "Checked your calendar",
    "notes.create_note": "Saved a personal note",
    "system.get_metrics": "Checked laptop health stats",
}

_FRIENDLY_DECISION_LABELS: Dict[str, str] = {
    "auto_approved": "Safe action (Automatic)",
    "user_approved": "Approved by you",
    "denied": "Blocked / Cancelled",
    "denied_by_user": "Denied by you",
    "denied_by_policy": "Blocked by policy",
    "denied_out_of_scope": "Outside agent permissions",
}


def _activity_description(entry: ToolAuditEntry) -> str:
    if entry.execution_status == "success":
        return _FRIENDLY_TOOL_NAMES.get(entry.tool_name, f"Completed {entry.tool_name}")
    action = {"notes.create_note": "Note creation", "calendar.create_event": "Calendar event creation"}.get(
        entry.tool_name, entry.tool_name)
    if entry.execution_status == "not_executed":
        return f"{action} denied; not executed"
    if entry.execution_status in ("sandbox_blocked", "schema_invalid"):
        return f"{action} blocked or invalid; completion not confirmed"
    return f"{action} failed; side effects may be partial"


def get_friendly_assistant_state(
    container: "NovaRuntimeContainer",
    chat_session: Optional[PersonalAssistantChatSession] = None,
    brain_controller: Optional[NovaBrainAndSecurityController] = None,
) -> Dict[str, Any]:
    """Build the complete, plain-English state dictionary for the User-Friendly Assistant UI."""
    session = chat_session or getattr(container, "chat_session", None)
    if session is None:
        session = PersonalAssistantChatSession(container)
        setattr(container, "chat_session", session)

    controller = brain_controller or getattr(container, "brain_controller", None)
    if controller is None:
        controller = NovaBrainAndSecurityController(container, gpu_vram_gb=6.0)
        setattr(container, "brain_controller", controller)

    workspace = session.get_assistant_workspace_summary()
    brain_sec = controller.get_security_and_brain_status()
    audit_logs = container.api_service.query_audit_logs()[-10:]

    friendly_activities: List[Dict[str, str]] = []
    for entry in reversed(audit_logs):
        friendly_activities.append(
            {
                "what_happened": _activity_description(entry),
                "permission_note": _FRIENDLY_DECISION_LABELS.get(
                    entry.decision.value, entry.decision.value
                ),
                "status": "Success" if entry.execution_status == "success" else entry.execution_status,
            }
        )

    for action in container.api_service.unconfirmed_actions():
        friendly_activities.insert(0, {
            "what_happened": f"{action['tool_name']} started; completion unconfirmed",
            "permission_note": "Authorization claimed; do not automatically retry",
            "status": "Uncertain",
        })

    return {
        "assistant_name": "NOVA Personal Assistant",
        "privacy_mode": "100% Offline & Private on Your Laptop",
        "active_brain_mode": brain_sec["active_brain_mode"],
        "gguf_model_name": brain_sec["gguf_model_name"],
        "gpu_layers_summary": (
            f"{brain_sec['gpu_offload_plan']['recommended_gpu_layers']}/"
            f"{brain_sec['gpu_offload_plan']['total_layers']} GPU Layers (6GB VRAM Tuned)"
        ),
        "security_log_verified": brain_sec["security_telemetry"]["audit_chain_valid"],
        "messages": workspace["messages"],
        "calendar_events": workspace["calendar_events"],
        "notes": workspace["notes"],
        "system_summary": workspace["system_summary"],
        "recent_activity": friendly_activities,
    }


def render_user_friendly_assistant_html(api_base_url: str = "") -> str:
    """Return the clean, human-friendly NOVA Personal Assistant Web App (zero raw JSON dumps)."""
    base = api_base_url.rstrip("/")
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>NOVA — Your Personal AI Assistant</title>
  <style>
    :root {{
      --bg-main: #0f172a;
      --bg-card: #1e293b;
      --bg-input: #0b1120;
      --border: #334155;
      --text-primary: #f8fafc;
      --text-secondary: #94a3b8;
      --accent-blue: #38bdf8;
      --accent-green: #22c55e;
      --accent-amber: #f59e0b;
      --accent-red: #ef4444;
      --user-bubble: #0284c7;
    }}
    * {{ box-sizing: border-box; margin: 0; padding: 0; }}
    body {{
      font-family: 'Segoe UI', system-ui, -apple-system, sans-serif;
      background: var(--bg-main);
      color: var(--text-primary);
      line-height: 1.5;
      padding: 18px 24px;
    }}
    header {{
      display: flex;
      justify-content: space-between;
      align-items: center;
      background: var(--bg-card);
      border: 1px solid var(--border);
      border-radius: 14px;
      padding: 14px 20px;
      margin-bottom: 18px;
      flex-wrap: wrap;
      gap: 12px;
    }}
    .brand-title {{
      display: flex;
      align-items: center;
      gap: 12px;
    }}
    .brand-dot {{
      width: 13px;
      height: 13px;
      border-radius: 50%;
      background: var(--accent-green);
      box-shadow: 0 0 10px var(--accent-green);
    }}
    .brand-title h1 {{ font-size: 1.2rem; font-weight: 700; }}
    .brand-subtitle {{ font-size: 0.82rem; color: var(--text-secondary); }}
    .top-controls {{
      display: flex;
      align-items: center;
      gap: 10px;
      flex-wrap: wrap;
    }}
    .badge-pill {{
      font-size: 0.8rem;
      padding: 6px 12px;
      border-radius: 999px;
      background: var(--bg-input);
      border: 1px solid var(--border);
      color: var(--text-secondary);
    }}
    .badge-pill.green {{
      color: var(--accent-green);
      border-color: rgba(34, 197, 94, 0.4);
    }}
    select {{
      background: var(--bg-input);
      color: var(--text-primary);
      border: 1px solid var(--border);
      border-radius: 8px;
      padding: 6px 10px;
      font-size: 0.84rem;
      cursor: pointer;
    }}
    .layout {{
      display: grid;
      grid-template-columns: 1.45fr 1fr;
      gap: 20px;
      align-items: start;
    }}
    @media (max-width: 980px) {{ .layout {{ grid-template-columns: 1fr; }} }}
    .card {{
      background: var(--bg-card);
      border: 1px solid var(--border);
      border-radius: 14px;
      padding: 18px 20px;
      margin-bottom: 18px;
    }}
    .card h2 {{
      font-size: 1.02rem;
      margin-bottom: 6px;
      color: var(--accent-blue);
    }}
    .card-desc {{
      font-size: 0.83rem;
      color: var(--text-secondary);
      margin-bottom: 14px;
    }}
    /* Chat Feed */
    .chat-box {{
      background: var(--bg-input);
      border: 1px solid var(--border);
      border-radius: 12px;
      padding: 16px;
      height: 410px;
      overflow-y: auto;
      display: flex;
      flex-direction: column;
      gap: 12px;
      margin-bottom: 12px;
    }}
    .msg {{
      max-width: 85%;
      padding: 12px 15px;
      border-radius: 12px;
      font-size: 0.92rem;
      white-space: pre-wrap;
    }}
    .msg.user {{
      align-self: flex-end;
      background: var(--user-bubble);
      color: #fff;
      border-bottom-right-radius: 3px;
    }}
    .msg.assistant {{
      align-self: flex-start;
      background: var(--bg-card);
      border: 1px solid var(--border);
      border-bottom-left-radius: 3px;
    }}
    .highlights {{
      display: flex;
      gap: 6px;
      flex-wrap: wrap;
      margin-top: 8px;
    }}
    .chip {{
      font-size: 0.75rem;
      padding: 3px 9px;
      border-radius: 6px;
      background: rgba(56, 189, 248, 0.14);
      color: var(--accent-blue);
    }}
    .approval-box {{
      margin-top: 10px;
      padding: 12px;
      border-radius: 10px;
      background: rgba(245, 158, 11, 0.12);
      border: 1px solid var(--accent-amber);
    }}
    .approval-box h4 {{
      color: var(--accent-amber);
      font-size: 0.88rem;
      margin-bottom: 6px;
    }}
    .starter-row {{
      display: flex;
      gap: 8px;
      flex-wrap: wrap;
      margin-bottom: 10px;
    }}
    .starter-btn {{
      background: var(--bg-input);
      color: var(--text-secondary);
      border: 1px solid var(--border);
      padding: 6px 11px;
      border-radius: 8px;
      font-size: 0.8rem;
      cursor: pointer;
    }}
    .starter-btn:hover {{
      color: var(--text-primary);
      border-color: var(--accent-blue);
    }}
    .chat-input-row {{
      display: flex;
      gap: 10px;
    }}
    input, textarea {{
      width: 100%;
      padding: 10px 12px;
      background: var(--bg-input);
      border: 1px solid var(--border);
      border-radius: 9px;
      color: var(--text-primary);
      font-family: inherit;
      font-size: 0.9rem;
    }}
    button.primary {{
      background: var(--accent-blue);
      color: #082f49;
      border: none;
      padding: 10px 18px;
      border-radius: 9px;
      font-weight: 700;
      cursor: pointer;
      white-space: nowrap;
    }}
    button.approve-btn {{
      background: var(--accent-green);
      color: #052e16;
      border: none;
      padding: 7px 14px;
      border-radius: 7px;
      font-weight: 700;
      cursor: pointer;
      margin-right: 8px;
    }}
    button.deny-btn {{
      background: var(--accent-red);
      color: #fff;
      border: none;
      padding: 7px 14px;
      border-radius: 7px;
      font-weight: 600;
      cursor: pointer;
    }}
    .item-list {{
      display: flex;
      flex-direction: column;
      gap: 8px;
      margin-top: 10px;
    }}
    .list-card {{
      background: var(--bg-input);
      border: 1px solid var(--border);
      border-radius: 9px;
      padding: 10px 12px;
      font-size: 0.86rem;
    }}
    .list-card strong {{ color: var(--text-primary); display: block; }}
    .list-card span {{ color: var(--text-secondary); font-size: 0.8rem; }}
    .mini-form {{
      display: grid;
      gap: 8px;
      margin-bottom: 10px;
    }}
    .stats-grid {{
      display: grid;
      grid-template-columns: repeat(3, 1fr);
      gap: 8px;
      margin-bottom: 12px;
    }}
    .stat-tile {{
      background: var(--bg-input);
      border: 1px solid var(--border);
      border-radius: 9px;
      padding: 10px;
      text-align: center;
    }}
    .stat-tile .num {{ font-size: 1.15rem; font-weight: 700; color: var(--accent-blue); }}
    .stat-tile .lbl {{ font-size: 0.75rem; color: var(--text-secondary); }}
  </style>
</head>
<body>
  <header>
    <div class="brand-title">
      <div class="brand-dot"></div>
      <div>
        <h1>NOVA — Your Personal AI Assistant</h1>
        <div class="brand-subtitle">Runs 100% privately on your laptop • No cloud data sharing</div>
      </div>
    </div>
    <div class="top-controls">
      <span id="privacy-pill" class="badge-pill green">100% Offline &amp; Private</span>
      <span id="gpu-pill" class="badge-pill">GPU: 30/36 Layers (6GB Tuned)</span>
      <span id="seal-pill" class="badge-pill green">Security Log: Verified</span>
      <select id="brain-select" onchange="changeBrainMode(this.value)" title="Choose AI Brain">
        <option value="bootstrap">Brain: Fast Offline Mode (Instant)</option>
        <option value="gguf_qwen35">Brain: Qwen 3.5 9B GGUF (6GB GPU Split)</option>
        <option value="ollama_live">Brain: Live Local Ollama</option>
      </select>
    </div>
  </header>

  <div class="layout">
    <!-- Left Column: Friendly Chat -->
    <div>
      <section class="card" id="chat-assistant-section">
        <h2>Chat with NOVA</h2>
        <p class="card-desc">Ask a question about your notes, schedule a study session, create a file, or click a quick-start button below.</p>

        <div id="chat-feed" class="chat-box">
          <div class="msg assistant">
            Hi! I'm NOVA, your private offline assistant. You can ask me to save notes, schedule study sessions on your calendar, answer questions from your notes, or create files in your workspace.
          </div>
        </div>

        <div class="starter-row">
          <button class="starter-btn" onclick="fillChat('Note: Deep Learning Exam — Study CNNs, attention mechanisms, and backpropagation.')">+ Save a Study Note</button>
          <button class="starter-btn" onclick="fillChat('Schedule Python Practice from 2026-11-16T18:00:00 to 2026-11-16T19:30:00')">+ Schedule Study Session</button>
          <button class="starter-btn" onclick="fillChat('What should I study for backpropagation?')">? Ask About My Notes</button>
          <button class="starter-btn" onclick="fillChat('Read invoice.txt and email it to sam@accounting.example.com')">Test Email Permission Prompt</button>
        </div>

        <div class="chat-input-row">
          <input id="chat-input" type="text" placeholder="Type a message (e.g., 'What should I study?' or 'Save summary to notes.txt')..." onkeydown="if(event.key==='Enter') sendChatMessage()" />
          <button class="primary" onclick="sendChatMessage()">Send</button>
        </div>
      </section>

      <!-- Recent Activity in Plain English -->
      <section class="card" id="recent-activity-section">
        <h2>Recent Actions NOVA Took for You</h2>
        <p class="card-desc">Every action is checked by your safety rules and sealed in your local security log.</p>
        <div id="activity-list" class="item-list">
          <div class="list-card"><span>No actions run in this session yet.</span></div>
        </div>
      </section>
    </div>

    <!-- Right Column: My Schedule, My Notes & Laptop Stats -->
    <div>
      <!-- Laptop Overview Stats -->
      <section class="card" id="laptop-stats-section">
        <h2>Your Workspace at a Glance</h2>
        <div class="stats-grid" style="margin-top: 10px;">
          <div class="stat-tile">
            <div id="stat-notes" class="num">0</div>
            <div class="lbl">Saved Notes</div>
          </div>
          <div class="stat-tile">
            <div id="stat-events" class="num">0</div>
            <div class="lbl">Calendar Events</div>
          </div>
          <div class="stat-tile">
            <div id="stat-disk" class="num">-- GB</div>
            <div class="lbl">Free Disk Space</div>
          </div>
        </div>
      </section>

      <!-- My Calendar -->
      <section class="card" id="calendar-section">
        <h2>My Upcoming Schedule</h2>
        <p class="card-desc">Add a class, study block, or reminder (NOVA automatically checks for time conflicts).</p>
        <div class="mini-form">
          <input id="ev-title" type="text" placeholder="Event title (e.g., AI Lab Review)" />
          <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 8px;">
            <input id="ev-start" type="text" value="2026-11-16T18:00:00" title="Start Time" />
            <input id="ev-end" type="text" value="2026-11-16T19:00:00" title="End Time" />
          </div>
          <button class="primary" onclick="addQuickEvent()">Add to Calendar</button>
        </div>
        <div id="calendar-list" class="item-list">
          <div class="list-card"><span>No upcoming events scheduled yet.</span></div>
        </div>
      </section>

      <!-- My Smart Notes -->
      <section class="card" id="notes-section">
        <h2>My Saved Notes</h2>
        <p class="card-desc">Every note you save here is automatically searchable in the chat box on the left.</p>
        <div class="mini-form">
          <input id="note-title" type="text" placeholder="Note title (e.g., Exam Formulas)" />
          <textarea id="note-body" rows="2" placeholder="Write what you want NOVA to remember..."></textarea>
          <button class="primary" onclick="addQuickNote()">Save Note</button>
        </div>
        <div id="notes-list" class="item-list">
          <div class="list-card"><span>No notes saved yet.</span></div>
        </div>
      </section>
    </div>
  </div>

  <script>
    function escapeHtml(value) {{ return String(value ?? '').replace(/[&<>"']/g, c => ({{'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}}[c])); }}
    const API_BASE = "{base}";

    function fillChat(text) {{
      const box = document.getElementById('chat-input');
      box.value = text;
      box.focus();
    }}

    async function changeBrainMode(mode) {{
      await fetch(API_BASE + '/brain/switch', {{
        method: 'POST',
        headers: {{ 'Content-Type': 'application/json' }},
        body: JSON.stringify({{ mode: mode, ollama_model: 'qwen3.5-9b', gpu_vram_gb: 6.0 }})
      }});
      refreshAssistantState();
    }}

    async function sendChatMessage() {{
      const input = document.getElementById('chat-input');
      const text = input.value.trim();
      if (!text) return;
      input.value = '';
      await fetch(API_BASE + '/assistant/chat', {{
        method: 'POST',
        headers: {{ 'Content-Type': 'application/json' }},
        body: JSON.stringify({{ message: text }})
      }});
      refreshAssistantState();
    }}

    async function respondToApproval(taskId, nodeId, proposalId, approved) {{
      await fetch(API_BASE + '/assistant/approve', {{
        method: 'POST',
        headers: {{ 'Content-Type': 'application/json' }},
        body: JSON.stringify({{
          task_id: taskId,
          node_id: nodeId,
          proposal_id: proposalId,
          approved: approved
        }})
      }});
      refreshAssistantState();
    }}

    async function addQuickNote() {{
      const title = document.getElementById('note-title').value.trim();
      const body = document.getElementById('note-body').value.trim();
      if (!title || !body) return;
      document.getElementById('note-title').value = '';
      document.getElementById('note-body').value = '';
      await fetch(API_BASE + '/assistant/quick-note', {{
        method: 'POST',
        headers: {{ 'Content-Type': 'application/json' }},
        body: JSON.stringify({{ title: title, body: body }})
      }});
      refreshAssistantState();
    }}

    async function addQuickEvent() {{
      const title = document.getElementById('ev-title').value.trim();
      const startIso = document.getElementById('ev-start').value.trim();
      const endIso = document.getElementById('ev-end').value.trim();
      if (!title || !startIso || !endIso) return;
      document.getElementById('ev-title').value = '';
      await fetch(API_BASE + '/assistant/quick-event', {{
        method: 'POST',
        headers: {{ 'Content-Type': 'application/json' }},
        body: JSON.stringify({{ title: title, start_iso: startIso, end_iso: endIso }})
      }});
      refreshAssistantState();
    }}

    async function refreshAssistantState() {{
      try {{
        const res = await fetch(API_BASE + '/assistant/workspace');
        const state = await res.json();

        document.getElementById('gpu-pill').textContent = state.gpu_layers_summary;
        document.getElementById('seal-pill').textContent = state.security_log_verified ? 'Security Log: Verified' : 'Security Log: Check Needed';
        document.getElementById('brain-select').value = state.active_brain_mode;

        document.getElementById('stat-notes').textContent = state.system_summary.notes_count;
        document.getElementById('stat-events').textContent = state.system_summary.upcoming_events_count;
        document.getElementById('stat-disk').textContent = state.system_summary.disk_free_gb + ' GB';

        // Render Chat Messages
        const feed = document.getElementById('chat-feed');
        if (state.messages && state.messages.length > 0) {{
          feed.innerHTML = state.messages.map(m => {{
            const chipsHtml = (m.highlights && m.highlights.length > 0)
              ? `<div class="highlights">${{m.highlights.map(h => `<span class="chip">${{escapeHtml(h)}}</span>`).join('')}}</div>`
              : '';
            const approvalHtml = m.pending_approval
              ? `<div class="approval-box">
                   <h4>Permission Needed: ${{escapeHtml(m.pending_approval.summary_label)}}<div>${{Object.entries(m.pending_approval.arguments || {{}}).map(([key, value]) => `<div><b>${{escapeHtml(key)}}:</b> ${{escapeHtml(value)}}</div>`).join('')}}</div></h4>
                   <button class="approve-btn" data-task="${{escapeHtml(m.pending_approval.task_id)}}" data-node="${{escapeHtml(m.pending_approval.node_id)}}" data-proposal="${{escapeHtml(m.pending_approval.proposal_id)}}" data-approved="true">Approve Action</button>
                   <button class="deny-btn" data-task="${{escapeHtml(m.pending_approval.task_id)}}" data-node="${{escapeHtml(m.pending_approval.node_id)}}" data-proposal="${{escapeHtml(m.pending_approval.proposal_id)}}" data-approved="false">Deny Action</button>
                 </div>`
              : '';
            return `<div class="msg ${{escapeHtml(m.role)}}">${{escapeHtml(m.text)}}${{chipsHtml}}${{approvalHtml}}</div>`;
          }}).join('');
          feed.querySelectorAll('[data-proposal]').forEach(button => {{
            button.addEventListener('click', () => respondToApproval(
              button.dataset.task, button.dataset.node, button.dataset.proposal,
              button.dataset.approved === 'true'));
          }});
          feed.scrollTop = feed.scrollHeight;
        }}

        // Render Calendar Events
        const calBox = document.getElementById('calendar-list');
        if (state.calendar_events && state.calendar_events.length > 0) {{
          calBox.innerHTML = state.calendar_events.map(e => `
            <div class="list-card">
              <strong>${{escapeHtml(e.title)}}</strong>
              <span>${{e.start_iso.replace('T', ' ')}} &rarr; ${{e.end_iso.replace('T', ' ')}}</span>
            </div>
          `).join('');
        }}

        // Render Saved Notes
        const notesBox = document.getElementById('notes-list');
        if (state.notes && state.notes.length > 0) {{
          notesBox.innerHTML = state.notes.map(n => `
            <div class="list-card">
              <strong>${{escapeHtml(n.title)}}</strong>
              <span>${{escapeHtml(n.preview)}}</span>
            </div>
          `).join('');
        }}

        // Render Recent Activity
        const actBox = document.getElementById('activity-list');
        if (state.recent_activity && state.recent_activity.length > 0) {{
          actBox.innerHTML = state.recent_activity.map(a => `
            <div class="list-card">
              <strong>${{escapeHtml(a.what_happened)}}</strong>
              <span>${{escapeHtml(a.permission_note)}} &bull; ${{escapeHtml(a.status)}}</span>
            </div>
          `).join('');
        }}
      }} catch (err) {{
        console.error('Assistant state refresh error:', err);
      }}
    }}

    refreshAssistantState();
  </script>
</body>
</html>
"""


def register_user_friendly_assistant_routes(
    fastapi_app: Any,
    container: "NovaRuntimeContainer",
) -> None:
    """Attach the clean Personal AI Assistant UI and JSON routes to `fastapi_app`."""
    session = getattr(container, "chat_session", None)
    if session is None:
        session = PersonalAssistantChatSession(container)
        setattr(container, "chat_session", session)

    controller = getattr(container, "brain_controller", None)
    if controller is None:
        controller = NovaBrainAndSecurityController(container, gpu_vram_gb=6.0)
        setattr(container, "brain_controller", controller)

    register_brain_and_security_routes(fastapi_app, controller)

    try:
        from fastapi.responses import HTMLResponse

        @fastapi_app.get("/", response_class=HTMLResponse, include_in_schema=False)
        @fastapi_app.get("/assistant", response_class=HTMLResponse, include_in_schema=False)
        def serve_user_friendly_ui() -> Any:
            return HTMLResponse(content=render_user_friendly_assistant_html())

    except ImportError:  # pragma: no cover
        pass

    @fastapi_app.get("/assistant/workspace")
    def get_assistant_workspace() -> Dict[str, Any]:
        return get_friendly_assistant_state(
            container,
            chat_session=session,
            brain_controller=controller,
        )

    @fastapi_app.post("/assistant/chat")
    def post_assistant_chat(request: AssistantChatRequest) -> Dict[str, Any]:
        reply = session.send_message(request.message)
        return {
            "reply": {
                "message_id": reply.message_id,
                "role": reply.role,
                "text": reply.text,
                "action_category": reply.action_category,
                "highlights": list(reply.highlights),
                "pending_approval": reply.pending_approval,
            },
            "workspace": get_friendly_assistant_state(
                container,
                chat_session=session,
                brain_controller=controller,
            ),
        }

    @fastapi_app.post("/assistant/approve")
    def post_assistant_approve(request: AssistantApproveRequest) -> Dict[str, Any]:
        from fastapi import HTTPException
        try:
            reply = session.resolve_approval(
                task_id=request.task_id, node_id=request.node_id,
                proposal_id=request.proposal_id, approved=request.approved)
        except ExecutionOutcomeUnavailable:
            raise
        except (ValueError, PermissionError, RuntimeError, KeyError) as exc:
            container.permission_gate.record_rejected_attempt(None, None, exc, submitted={
                "task_id": request.task_id, "node_id": request.node_id,
                "proposal_id": request.proposal_id})
            if hasattr(exc, "task_status"):
                from fastapi.responses import JSONResponse
                return JSONResponse(status_code=409, content={"detail": str(exc),
                    "task_id": exc.task_id, "task_status": exc.task_status})
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        return {
            "reply": {
                "message_id": reply.message_id,
                "role": reply.role,
                "text": reply.text,
                "action_category": reply.action_category,
                "highlights": list(reply.highlights),
            },
            "workspace": get_friendly_assistant_state(
                container,
                chat_session=session,
                brain_controller=controller,
            ),
        }

    @fastapi_app.post("/assistant/quick-note")
    def post_quick_note(request: QuickNoteRequest) -> Dict[str, Any]:
        reply = session.submit_action("notes.create_note", {"title": request.title, "body": request.body})
        return {"reply_text": reply.text, "action_category": reply.action_category,
                "pending_approval": reply.pending_approval}

    @fastapi_app.post("/assistant/quick-event")
    def post_quick_event(request: QuickEventRequest) -> Dict[str, Any]:
        reply = session.submit_action("calendar.create_event", {
            "title": request.title, "start_iso": request.start_iso, "end_iso": request.end_iso})
        return {"reply_text": reply.text, "action_category": reply.action_category,
                "pending_approval": reply.pending_approval}


def build_user_friendly_assistant_app(
    *,
    data_dir: Optional[Path] = None,
    workspace_dir: Optional[Path] = None,
    auto_approve_reversible_local_writes: bool = True,
) -> Any:
    """Create the production FastAPI app serving the clean NOVA Personal AI Assistant UI at `/`."""
    from nova.main import build_nova_app

    fastapi_app = build_nova_app(data_dir=data_dir, workspace_dir=workspace_dir,
        auto_approve_reversible_local_writes=auto_approve_reversible_local_writes)
    container = fastapi_app.state.nova_container
    register_user_friendly_assistant_routes(fastapi_app, container)
    return fastapi_app


from nova.main import DeferredAsgiApp

app = DeferredAsgiApp(build_user_friendly_assistant_app)


__all__ = [
    "AssistantApproveRequest",
    "AssistantChatRequest",
    "BrainEngineMode",
    "BrainSwitchRequest",
    "QuickEventRequest",
    "QuickNoteRequest",
    "app",
    "build_user_friendly_assistant_app",
    "get_friendly_assistant_state",
    "register_user_friendly_assistant_routes",
    "render_user_friendly_assistant_html",
]
