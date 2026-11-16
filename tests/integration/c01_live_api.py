"""Standalone real HTTP/process verification. Run with the repository venv Python.

Only synthetic data in a newly allocated temp directory is used. Evidence is
written to a fresh C01 evidence file. No external service or model is invoked.
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


def serve(data: Path, port: int, crash: bool, pause: bool = False, ttl: int = 300) -> None:
    import uvicorn
    from nova.api.assistant_ui import build_user_friendly_assistant_app
    app = build_user_friendly_assistant_app(data_dir=data / "data", workspace_dir=data / "workspace",
                                          auto_approve_reversible_local_writes=False)
    app.state.nova_container.permission_gate.ledger.ttl = ttl
    if crash or pause:
        # Test instrumentation wraps the real handler; it never fabricates a result.
        actual = app.state.nova_container.tool_executor._handlers["notes.create_note"]
        def interrupted_handler(arguments):
            if pause:
                (data / "claimed-marker.txt").write_text("claim committed", encoding="utf-8")
                deadline = time.monotonic() + 15
                while not (data / "release-marker.txt").exists():
                    if time.monotonic() > deadline:
                        raise TimeoutError("Verification barrier timed out")
                    time.sleep(0.01)
            result = actual(arguments)
            if crash:
                os._exit(73)
            return result
        app.state.nova_container.tool_executor.register_tool_handler("notes.create_note", interrupted_handler)
    uvicorn.run(app, host="127.0.0.1", port=port, log_level="warning")


def verify() -> None:
    root = Path(tempfile.mkdtemp(prefix="nova-c01-live-"))
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
    evidence = {"data_root": str(root), "transport": "real HTTP to loopback Uvicorn child process",
                "checks": [], "processes": [], "http": [], "snapshots": []}
    process = None
    log = None
    second = None
    second_log = None

    def request(path, payload=None, *, target_port=None):
        req = Request(f"http://127.0.0.1:{target_port or port}{path}",
                      data=json.dumps(payload).encode() if payload is not None else None,
                      headers={"Content-Type": "application/json"})
        try:
            with urlopen(req, timeout=20) as response:
                raw = response.read().decode()
                body = json.loads(raw) if "application/json" in response.headers.get("Content-Type", "") else raw
                status = response.status
        except HTTPError as exc:
            status, body = exc.code, json.load(exc)
        evidence["http"].append({"port": target_port or port, "path": path, "request": payload,
                                  "status": status, "response": body})
        return status, body

    def events(card):
        code, stream = request(f"/tasks/{card['task_id']}/events/sse")
        assert code == 200
        return [json.loads(line[6:]) for line in stream.splitlines() if line.startswith("data: ")]

    def start_second():
        nonlocal second, second_log
        with socket.socket() as listener:
            listener.bind(("127.0.0.1", 0))
            other_port = listener.getsockname()[1]
        log_path = root / "second-server.log"
        second_log = log_path.open("x", encoding="utf-8")
        command = [sys.executable, "-B", str(Path(__file__).resolve()), "--serve", str(root), "--port", str(other_port)]
        second = subprocess.Popen(command, cwd=ROOT, stdout=second_log, stderr=subprocess.STDOUT,
                                  creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        evidence["processes"].append({"pid": second.pid, "command": command, "log": str(log_path)})
        for _ in range(100):
            try:
                if request("/health", target_port=other_port)[0] == 200:
                    return other_port
            except (URLError, ConnectionError):
                pass
            time.sleep(0.05)
        raise RuntimeError("Second runtime did not start")

    def snapshot(label):
        import hashlib
        entry = {"label": label, "files": {}, "authorizations": [], "attempts": []}
        for file in (root / "workspace/notes").glob("*.md"):
            entry["files"][file.name] = {"sha256": hashlib.sha256(file.read_bytes()).hexdigest(),
                                          "text": file.read_text(encoding="utf-8")}
        with sqlite3.connect(root / "data/nova_audit.db") as conn:
            entry["authorizations"] = conn.execute("SELECT proposal_id, status FROM tool_authorizations").fetchall()
            entry["attempts"] = conn.execute("SELECT attempt_id, outcome, reason_code FROM authorization_attempts").fetchall()
        evidence["snapshots"].append(entry)

    def start(crash=False, pause=False, ttl=300):
        nonlocal process, log
        log_path = root / f"server-{len(evidence['processes'])}.log"
        log = log_path.open("w", encoding="utf-8")
        command = [sys.executable, "-B", str(Path(__file__).resolve()), "--serve", str(root), "--port", str(port)]
        command.extend(["--ttl", str(ttl)])
        if crash:
            command.append("--crash")
        if pause:
            command.append("--pause")
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
        if process is not None:
            for item in evidence["processes"]:
                if item["pid"] == process.pid:
                    item["exit_code"] = process.returncode
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
        snapshot("initial_empty_workspace")
        denied = pending("/assistant/quick-note", {"title": "Denied note", "body": "Must not exist"})
        code, result = request("/assistant/approve", {**denied, "approved": False})
        assert code == 200 and result["reply"]["action_category"] == "action_failed"
        assert not list((root / "workspace/notes").glob("*.md"))
        assert "denied" in result["workspace"]["recent_activity"][0]["what_happened"]
        check("F01 denied activity is truthful and produces no file", **denied)

        note = pending("/assistant/quick-note", {"title": "Exact - title", "body": "Body with — delimiter\nSecond line <script>untrusted</script>"})
        code, result = request("/assistant/approve", {**note, "approved": True})
        assert code == 200 and result["reply"]["action_category"] == "note_saved", result
        assert request("/assistant/approve", {**note, "approved": True, "arguments": {"body": "SYNTHETIC_SECRET_NOT_FOR_AUDIT"}})[0] == 422
        assert request("/assistant/approve", {**note, "task_id": "unknown_task", "approved": True})[0] == 409
        text = (root / "workspace/notes/exact_title.md").read_text(encoding="utf-8")
        assert "Second line <script>untrusted</script>" in text
        check("APPROVE writes exact structured note", **note, file_text=text)
        assert request("/assistant/approve", {**note, "approved": True})[0] == 409
        note_events = events(note)
        assert [e["event_type"] for e in note_events] == [
            "permission.requested", "permission.approved", "tool.started", "tool.completed", "task.completed"]
        assert all(e["correlation_id"] == note["proposal_id"] for e in note_events)
        check("F02 ordered correlated completion events without replay duplicates", events=note_events)
        duplicate = pending("/assistant/quick-note", {"title": "Exact - title", "body": "Must not overwrite"})
        code, failure = request("/assistant/approve", {**duplicate, "approved": True})
        assert code == 200 and failure["reply"]["action_category"] == "action_failed"
        assert "failed" in failure["workspace"]["recent_activity"][0]["what_happened"]
        failed_events = events(duplicate)
        assert [e["event_type"] for e in failed_events][-2:] == ["tool.failed", "task.failed"]
        check("F01 F02 actual duplicate note failure is truthful and emits failure events", events=failed_events)

        event_args = {"title": "Study", "start_iso": "2026-11-17T10:00:00", "end_iso": "2026-11-17T11:00:00"}
        event = pending("/assistant/quick-event", event_args)
        with ThreadPoolExecutor(max_workers=6) as pool:
            results = list(pool.map(lambda _: request("/assistant/approve", {**event, "approved": True}), range(6)))
        assert sum(code == 200 for code, _ in results) == 1, results
        check("Concurrent HTTP approvals allow one calendar write", statuses=[code for code, _ in results], **event)

        revoked = pending("/assistant/quick-event", {**event_args, "title": "Revoked"})
        other_port = start_second()
        assert request("/permissions/revoke", {"tool_name": "calendar.create_event"}, target_port=other_port)[0] == 200
        code, result = request("/assistant/approve", {**revoked, "approved": True})
        assert code == 200 and result["reply"]["action_category"] == "action_failed", result
        check("F03 runtime B committed revoke blocks runtime A pending approval", **revoked)

        stop()
        start()
        _, workspace = request("/assistant/workspace")
        assert workspace["system_summary"]["notes_count"] == 1
        assert workspace["system_summary"]["upcoming_events_count"] == 1
        assert request("/assistant/approve", {**note, "approved": True})[0] == 409
        _, result = request("/assistant/quick-event", {**event_args, "title": "Still revoked"})
        assert result["action_category"] == "action_failed"
        check("Process restart preserves data, revocation and replay rejection")
        snapshot("after_process_restart")

        # Claim-first ordering: pause the REAL handler after the production claim commits.
        claimed_first = pending("/assistant/quick-note", {"title": "Claim first", "body": "Allowed claimed operation"})
        stop()
        start(pause=True)
        with ThreadPoolExecutor(1) as pool:
            completion = pool.submit(request, "/assistant/approve", {**claimed_first, "approved": True})
            deadline = time.monotonic() + 10
            while not (root / "claimed-marker.txt").exists():
                assert time.monotonic() < deadline
                time.sleep(0.01)
            assert request("/permissions/revoke", {"tool_name": "notes.create_note"}, target_port=other_port)[0] == 200
            (root / "release-marker.txt").write_text("release", encoding="utf-8")
            code, body = completion.result()
        assert code == 200 and body["reply"]["action_category"] == "note_saved"
        check("F03 claim committed first may finish after runtime B revokes", **claimed_first)
        assert request("/permissions/restore", {"tool_name": "notes.create_note"}, target_port=other_port)[0] == 200
        stop()
        start()
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
        snapshot("after_crash")
        start()
        assert request("/assistant/approve", {**uncertain, "approved": True})[0] == 409
        assert (root / "workspace/notes/uncertain_write.md").read_text(encoding="utf-8").endswith("Do not replay\n")
        _, workspace = request("/assistant/workspace")
        assert any(a["status"] == "Uncertain" for a in workspace["recent_activity"])
        _, attempts = request("/audit/authorization-attempts")
        assert any(a["outcome"] == "rejected" for a in attempts)
        assert all("arguments" not in a["submitted_claims"] for a in attempts)
        check("F04 rejected requests auditable; unresolved claim visibly uncertain", attempts=attempts)
        check("Injected crash after claim rejects replay after restart", exit_code=73, **uncertain)
        stop()
        start(ttl=1)
        expired = pending("/assistant/quick-note", {"title": "Expired", "body": "No write allowed"})
        time.sleep(1.1)
        code, result = request("/assistant/approve", {**expired, "approved": True})
        assert code == 409 and "expired" in str(result)
        _, attempts = request("/audit/authorization-attempts")
        assert any(a["reason_code"] == "expired" for a in attempts)
        assert any(a["reason_code"] == "invalid_approval_payload" for a in attempts)
        assert "SYNTHETIC_SECRET_NOT_FOR_AUDIT" not in json.dumps(attempts)
        check("F04 real expiry and malformed approval rejection audited without submitted body", http_status=code)
        snapshot("final")
        stop()

        with sqlite3.connect(root / "data/nova_calendar.db") as conn:
            events = conn.execute("SELECT title, start_iso, end_iso FROM calendar_events").fetchall()
        assert events == [("Study", event_args["start_iso"], event_args["end_iso"])], events
        with sqlite3.connect(root / "data/nova_audit.db") as conn:
            authorizations = conn.execute("SELECT proposal_id, status FROM tool_authorizations").fetchall()
            audits = conn.execute("SELECT task_id, session_id, agent_role, tool_name, decision, execution_status FROM tool_audit_log").fetchall()
        assert sum(status == "success" for _, status in authorizations) == 3
        assert (uncertain["proposal_id"], "claimed") in authorizations
        assert all(row[0].startswith("task_assistant_") and row[1] == "sess_personal_chat" for row in audits)
        check("Direct SQLite inspection confirms writes, claim state and audit identities", events=events,
              authorizations=authorizations, audits=audits)
        evidence["result"] = "PASS"
    finally:
        stop()
        if second is not None:
            second.terminate()
            second.wait(timeout=10)
            for item in evidence["processes"]:
                if item["pid"] == second.pid:
                    item["exit_code"] = second.returncode
        if second_log:
            second_log.close()
        output = ROOT / "docs" / ("C01-live-" + root.name + ".json")
        output.parent.mkdir(exist_ok=True)
        output.write_text(json.dumps(evidence, indent=2), encoding="utf-8")
    print(json.dumps({"result": evidence["result"], "checks": len(evidence["checks"]), "evidence": str(output)}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--serve", type=Path)
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--crash", action="store_true")
    parser.add_argument("--pause", action="store_true")
    parser.add_argument("--ttl", type=int, default=300)
    args = parser.parse_args()
    if args.serve:
        serve(args.serve, args.port, args.crash, args.pause, args.ttl)
    else:
        verify()
