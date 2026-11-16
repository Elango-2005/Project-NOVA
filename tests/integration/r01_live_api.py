"""Standalone real HTTP/process verification. Run with the repository venv Python.

Only synthetic data in a newly allocated temp directory is used. Evidence is
written to docs/NOVA-R01-evidence.json. No external service or model is invoked.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import socket
import sqlite3
import subprocess
import sys
import tempfile
import time
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "services/backend/src"))


def serve(data: Path, port: int, crash: bool) -> None:
    import uvicorn
    from nova.api.assistant_ui import build_user_friendly_assistant_app
    app = build_user_friendly_assistant_app(data_dir=data / "data", workspace_dir=data / "workspace",
                                          auto_approve_reversible_local_writes=False)
    if crash:
        # Explicit fault injection AFTER the durable claim, BEFORE the local write.
        def interrupted_handler(arguments):
            (data / "crash-marker.txt").write_text("Handler entered after durable claim", encoding="utf-8")
            os._exit(73)
        app.state.nova_container.tool_executor.register_tool_handler("notes.create_note", interrupted_handler)
    uvicorn.run(app, host="127.0.0.1", port=port, log_level="warning")


def verify() -> None:
    root = Path(tempfile.mkdtemp(prefix="nova-r01-live-"))
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
    evidence = {"data_root": str(root), "transport": "real HTTP to loopback Uvicorn child process",
                "checks": [], "processes": []}
    process = None
    log = None

    def request(path, payload=None):
        req = Request(f"http://127.0.0.1:{port}{path}",
                      data=json.dumps(payload).encode() if payload is not None else None,
                      headers={"Content-Type": "application/json"})
        try:
            with urlopen(req, timeout=10) as response:
                return response.status, json.load(response)
        except HTTPError as exc:
            return exc.code, json.load(exc)

    def start(crash=False):
        nonlocal process, log
        log_path = root / f"server-{len(evidence['processes'])}.log"
        log = log_path.open("w", encoding="utf-8")
        command = [sys.executable, "-B", str(Path(__file__).resolve()), "--serve", str(root), "--port", str(port)]
        if crash:
            command.append("--crash")
        process = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                                   creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        evidence["processes"].append({"pid": process.pid, "command": command, "log": str(log_path)})
        for _ in range(100):
            if process.poll() is not None:
                raise RuntimeError(log_path.read_text())
            try:
                if request("/assistant/workspace")[0] == 200:
                    return
            except (URLError, ConnectionError):
                pass
            time.sleep(0.05)
        raise RuntimeError("API startup timed out")

    def stop():
        if process is not None and process.poll() is None:
            process.terminate()
            process.wait(timeout=10)
        if log:
            log.close()

    def pending(path, payload):
        code, result = request(path, payload)
        assert code == 200, result
        assert result["action_category"] == "awaiting_approval", result
        return {k: result["pending_approval"][k] for k in ("task_id", "node_id", "proposal_id")}

    def check(name, **details):
        evidence["checks"].append({"name": name, "passed": True, **details})

    try:
        start()
        denied = pending("/assistant/quick-note", {"title": "Denied note", "body": "Must not exist"})
        code, result = request("/assistant/approve", {**denied, "approved": False})
        assert code == 200 and result["reply"]["action_category"] == "action_failed"
        assert not list((root / "workspace/notes").glob("*.md"))
        check("DENY produces no file", **denied)

        note = pending("/assistant/quick-note", {"title": "Exact - title", "body": "Body with — delimiter\nSecond line <script>untrusted</script>"})
        code, result = request("/assistant/approve", {**note, "approved": True})
        assert code == 200 and result["reply"]["action_category"] == "note_saved", result
        text = (root / "workspace/notes/exact_title.md").read_text(encoding="utf-8")
        assert "Second line <script>untrusted</script>" in text
        check("APPROVE writes exact structured note", **note, file_text=text)
        assert request("/assistant/approve", {**note, "approved": True})[0] == 409
        check("HTTP replay rejected", http_status=409)

        event_args = {"title": "Study", "start_iso": "2026-11-17T10:00:00", "end_iso": "2026-11-17T11:00:00"}
        event = pending("/assistant/quick-event", event_args)
        with ThreadPoolExecutor(max_workers=6) as pool:
            results = list(pool.map(lambda _: request("/assistant/approve", {**event, "approved": True}), range(6)))
        assert sum(code == 200 for code, _ in results) == 1, results
        check("Concurrent HTTP approvals allow one calendar write", statuses=[code for code, _ in results], **event)

        revoked = pending("/assistant/quick-event", {**event_args, "title": "Revoked"})
        assert request("/permissions/revoke", {"tool_name": "calendar.create_event"})[0] == 200
        code, result = request("/assistant/approve", {**revoked, "approved": True})
        assert code == 200 and result["reply"]["action_category"] == "action_failed", result
        check("REVOKE blocks pending approval", **revoked)

        stop()
        start()
        _, workspace = request("/assistant/workspace")
        assert workspace["system_summary"]["notes_count"] == 1
        assert workspace["system_summary"]["upcoming_events_count"] == 1
        assert request("/assistant/approve", {**note, "approved": True})[0] == 409
        _, result = request("/assistant/quick-event", {**event_args, "title": "Still revoked"})
        assert result["action_category"] == "action_failed"
        check("Process restart preserves data, revocation and replay rejection")

        uncertain = pending("/assistant/quick-note", {"title": "Uncertain write", "body": "Do not replay"})
        stop()
        start(crash=True)
        try:
            request("/assistant/approve", {**uncertain, "approved": True})
            raise AssertionError("Expected injected process exit")
        except (URLError, ConnectionError):
            pass
        process.wait(timeout=10)
        assert process.returncode == 73
        stop()
        start()
        assert request("/assistant/approve", {**uncertain, "approved": True})[0] == 409
        assert not (root / "workspace/notes/uncertain_write.md").exists()
        check("Injected crash after claim rejects replay after restart", exit_code=73, **uncertain)
        stop()

        with sqlite3.connect(root / "data/nova_calendar.db") as conn:
            events = conn.execute("SELECT title, start_iso, end_iso FROM calendar_events").fetchall()
        assert events == [("Study", event_args["start_iso"], event_args["end_iso"])], events
        with sqlite3.connect(root / "data/nova_audit.db") as conn:
            authorizations = conn.execute("SELECT proposal_id, status FROM tool_authorizations").fetchall()
            audits = conn.execute("SELECT task_id, session_id, agent_role, tool_name, decision, execution_status FROM tool_audit_log").fetchall()
        assert sum(status == "success" for _, status in authorizations) == 2
        assert (uncertain["proposal_id"], "claimed") in authorizations
        assert all(row[0].startswith("task_assistant_") and row[1] == "sess_personal_chat" for row in audits)
        check("Direct SQLite inspection confirms writes, claim state and audit identities", events=events,
              authorizations=authorizations, audits=audits)
        evidence["result"] = "PASS"
    finally:
        stop()
        output = ROOT / "docs/NOVA-R01-evidence.json"
        output.parent.mkdir(exist_ok=True)
        output.write_text(json.dumps(evidence, indent=2), encoding="utf-8")
    print(json.dumps({"result": evidence["result"], "checks": len(evidence["checks"]), "evidence": str(output)}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--serve", type=Path)
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--crash", action="store_true")
    args = parser.parse_args()
    if args.serve:
        serve(args.serve, args.port, args.crash)
    else:
        verify()
