"""C01 regressions using real SQLite and note handlers in isolated directories."""
import json
import sqlite3
from concurrent.futures import ThreadPoolExecutor
from threading import Event
import time

import pytest

from nova.main import NovaRuntimeContainer
from nova.api.assistant_ui import get_friendly_assistant_state
from nova.core.cognition.conversation_session import PersonalAssistantChatSession
from nova.core.domain.primitives import AgentRole, TaskStatus
from nova.core.domain.proposals import ToolCallProposal
from nova.core.orchestration.ports import ToolActionContext, UnapprovedToolExecutionError, ExecutionOutcomeUnavailable
from nova.core.orchestration.state import TaskExecutionState
from nova.core.orchestration.engine import OrchestrationExecutionError


def runtime(root, auto=False):
    return NovaRuntimeContainer(data_dir=root / "data", workspace_dir=root / "workspace",
                                auto_approve_reversible_local_writes=auto)


def authorized(c):
    p = ToolCallProposal(agent_role="coding", node_id="node_c01", thought="Synthetic verification",
        tool_name="notes.create_note", arguments={"title": "C01", "body": "Private synthetic body"},
        declared_sensitivity="reversible_write", rationale="C01 verification")
    ctx = ToolActionContext(task_id="task_c01", session_id="session_c01", node_id="node_c01", agent_role="coding")
    c.permission_gate.evaluate_tool_proposal(p, context=ctx)
    c.permission_gate.record_user_decision(p.proposal_id, True, proposal=p, context=ctx)
    return p, ctx, c.permission_gate.evaluate_tool_proposal(p, context=ctx)


def test_other_runtime_revocation_prevents_held_stamp_and_records_rejection(tmp_path):
    a, b = runtime(tmp_path), runtime(tmp_path)
    p, ctx, stamp = authorized(a)
    b.policy_store.revoke_tool(p.tool_name)
    with pytest.raises(UnapprovedToolExecutionError, match="revoked"):
        a.tool_executor.execute_approved_tool(p, stamp, context=ctx)
    assert not list(a.notes_manager.notes_dir.glob("*.md"))
    rows = a.permission_gate.ledger.query_attempts()
    assert len(rows) == 1 and rows[0]["reason_code"] == "revoked"
    assert rows[0]["trusted_context"]["task_id"] == ctx.task_id
    assert a.policy_store.inspect_policies() == b.policy_store.inspect_policies()


def test_concurrent_policy_changes_do_not_erase_revocations(tmp_path):
    a, b = runtime(tmp_path), runtime(tmp_path)
    with ThreadPoolExecutor(2) as pool:
        results = [pool.submit(a.policy_store.revoke_tool, "notes.create_note"),
                   pool.submit(b.policy_store.revoke_tool, "calendar.create_event")]
        for future in results:
            future.result()
    assert set(a.policy_store.inspect_policies()["revoked_global_tools"]) == {
        "notes.create_note", "calendar.create_event"}


def test_claim_committed_first_is_allowed_to_finish_after_revocation(tmp_path):
    a, b = runtime(tmp_path), runtime(tmp_path)
    p, ctx, stamp = authorized(a)
    entered, release = Event(), Event()
    actual = a.tool_executor._handlers[p.tool_name]
    def instrumented_handler(args):
        entered.set()
        assert release.wait(5)
        return actual(args)
    a.tool_executor.register_tool_handler(p.tool_name, instrumented_handler)
    with ThreadPoolExecutor(2) as pool:
        future = pool.submit(a.tool_executor.execute_approved_tool, p, stamp, context=ctx)
        try:
            assert entered.wait(5)
            b.policy_store.revoke_tool(p.tool_name)
        finally:
            release.set()
        assert future.result().status == "success"
    assert (a.notes_manager.notes_dir / "c01.md").exists()
    with pytest.raises(UnapprovedToolExecutionError):
        a.tool_executor.execute_approved_tool(p, stamp, context=ctx)


@pytest.mark.parametrize("mutation", ["token", "arguments", "role", "task_id", "session_id", "node_id", "expired"])
def test_rejections_are_audited_once_without_arguments_or_tokens(tmp_path, mutation):
    c = runtime(tmp_path)
    p, ctx, stamp = authorized(c)
    if mutation == "token":
        stamp = stamp.model_copy(update={"authorization_id": "SECRET_TOKEN_SENTINEL"})
    elif mutation == "arguments":
        p = p.model_copy(update={"arguments": {"title": "changed", "body": "SECRET_CONTENT_SENTINEL"}})
    elif mutation == "role":
        p = p.model_copy(update={"agent_role": AgentRole.COMMS})
    elif mutation == "expired":
        c.permission_gate.ledger.clock = lambda: stamp.expires_at + 1
    else:
        ctx = ctx.model_copy(update={mutation: "wrong_context"})
    with pytest.raises(UnapprovedToolExecutionError):
        c.tool_executor.execute_approved_tool(p, stamp, context=ctx)
    rows = c.permission_gate.ledger.query_attempts()
    assert len(rows) == 1 and rows[0]["outcome"] == "rejected"
    serialized = json.dumps(rows)
    assert "SECRET_" not in serialized and "Private synthetic body" not in serialized
    if mutation == "role":
        assert rows[0]["trusted_context"]["agent_role"] == "coding"
        assert rows[0]["submitted_claims"]["agent_role"] == "comms"


def test_audit_storage_failure_rolls_back_claim_and_prevents_handler(tmp_path):
    c = runtime(tmp_path)
    p, ctx, stamp = authorized(c)
    with sqlite3.connect(c.data_dir / "nova_audit.db") as conn:
        conn.execute("CREATE TRIGGER fail_attempt BEFORE INSERT ON authorization_attempts BEGIN SELECT RAISE(ABORT, 'Injected audit failure'); END")
    with pytest.raises(sqlite3.DatabaseError):
        c.tool_executor.execute_approved_tool(p, stamp, context=ctx)
    assert not list(c.notes_manager.notes_dir.glob("*.md"))
    with sqlite3.connect(c.data_dir / "nova_audit.db") as conn:
        assert conn.execute("SELECT status FROM tool_authorizations").fetchone()[0] == "approved"
    forged = stamp.model_copy(update={"authorization_id": "forged"})
    with pytest.raises(sqlite3.DatabaseError):
        c.tool_executor.execute_approved_tool(p, forged, context=ctx)
    assert not list(c.notes_manager.notes_dir.glob("*.md"))


def test_truthful_activity_and_ordered_correlated_terminal_events(tmp_path):
    c = runtime(tmp_path)
    session = PersonalAssistantChatSession(c)
    def request(title):
        card = session.submit_action("notes.create_note", {"title": title, "body": "Synthetic"}).pending_approval
        return {k: card[k] for k in ("task_id", "node_id", "proposal_id")}
    denied = request("Denied")
    session.resolve_approval(**denied, approved=False)
    assert "denied" in get_friendly_assistant_state(c, session)["recent_activity"][0]["what_happened"]
    card = request("Created")
    session.resolve_approval(**card, approved=True)
    c.orchestrator.run_until_blocked_or_complete(card["task_id"])
    with pytest.raises(Exception):
        session.resolve_approval(**card, approved=True)
    events = [e for e in c.event_bus.published_events if e.task_id == card["task_id"]]
    names = [e.event_type.value for e in events]
    assert names == ["permission.requested", "permission.approved", "tool.started", "tool.completed", "task.completed"]
    assert all(e.correlation_id == card["proposal_id"] for e in events)
    failed = request("Created")  # Real exclusive-file-create failure, not a fake handler.
    session.resolve_approval(**failed, approved=True)
    names = [e.event_type.value for e in c.event_bus.published_events if e.task_id == failed["task_id"]]
    assert names[-2:] == ["tool.failed", "task.failed"]
    assert "failed" in get_friendly_assistant_state(c, session)["recent_activity"][0]["what_happened"]
    assert c.api_service.get_task_status(card["task_id"]).status == TaskStatus.COMPLETED
    with pytest.raises(OrchestrationExecutionError, match="terminal outcome"):
        c.orchestrator._save_checkpoint(TaskExecutionState.create_initial(
            task_id=card["task_id"], session_id="sess_personal_chat", user_goal="Stale start"))
    assert c.api_service.get_task_status(card["task_id"]).status == TaskStatus.COMPLETED


def test_unconfirmed_claim_is_visible_and_not_replayed(tmp_path):
    c = runtime(tmp_path)
    p, ctx, stamp = authorized(c)
    c.permission_gate.claim_authorization(p, stamp, context=ctx)
    restarted = runtime(tmp_path)
    state = get_friendly_assistant_state(restarted)
    assert state["recent_activity"][0]["status"] == "Uncertain"
    with pytest.raises(UnapprovedToolExecutionError):
        restarted.tool_executor.execute_approved_tool(p, stamp, context=ctx)
    assert not list(c.notes_manager.notes_dir.glob("*.md"))


@pytest.mark.parametrize("first", ["revoke", "claim"])
def test_overlapping_transactions_serialize_policy_and_claim(tmp_path, first):
    a, b = runtime(tmp_path), runtime(tmp_path)
    p, ctx, stamp = authorized(a)
    entered, release, contender_started = Event(), Event(), Event()
    # Instrument only a transaction boundary; the production policy and handler run unchanged.
    if first == "claim":
        original = a.permission_gate._evaluate_policy
        def held_policy(*args, **kwargs):
            result = original(*args, **kwargs)
            if kwargs.get("policy_store") is not None:
                entered.set()
                assert release.wait(5)
            return result
        a.permission_gate._evaluate_policy = held_policy
    else:
        original = b.policy_store._load
        def held_load(conn):
            original(conn)
            entered.set()
            assert release.wait(5)
        b.policy_store._load = held_load
    def execute():
        try:
            return a.tool_executor.execute_approved_tool(p, stamp, context=ctx).status
        except UnapprovedToolExecutionError:
            return "rejected"
    def revoke():
        b.policy_store.revoke_tool(p.tool_name)
        return "revoked"
    winner, contender = (execute, revoke) if first == "claim" else (revoke, execute)
    def compete():
        contender_started.set()
        return contender()
    with ThreadPoolExecutor(2) as pool:
        first_future = pool.submit(winner)
        try:
            assert entered.wait(5)
            second_future = pool.submit(compete)
            assert contender_started.wait(5)
            time.sleep(0.1)
            assert not second_future.done(), "Competing SQLite writer bypassed the active transaction"
        finally:
            release.set()
        results = (first_future.result(), second_future.result())
    assert results == (("success", "revoked") if first == "claim" else ("revoked", "rejected"))
    assert (a.notes_manager.notes_dir / "c01.md").exists() == (first == "claim")


def test_post_write_audit_failure_stays_uncertain(tmp_path):
    c = runtime(tmp_path, auto=True)
    with sqlite3.connect(c.data_dir / "nova_audit.db") as conn:
        conn.execute("CREATE TRIGGER fail_outcome BEFORE INSERT ON tool_audit_log BEGIN SELECT RAISE(ABORT, 'Injected outcome audit failure'); END")
    with pytest.raises(ExecutionOutcomeUnavailable):
        PersonalAssistantChatSession(c).submit_action("notes.create_note", {"title": "Written", "body": "Real write"})
    assert (c.notes_manager.notes_dir / "written.md").exists()
    assert get_friendly_assistant_state(c)["recent_activity"][0]["status"] == "Uncertain"
    with sqlite3.connect(c.data_dir / "nova_audit.db") as conn:
        assert conn.execute("SELECT status FROM tool_authorizations").fetchone()[0] == "claimed"
    names = [e.event_type.value for e in c.event_bus.published_events]
    assert "tool.completed" not in names and "task.completed" not in names
    assert "tool.failed" not in names and "task.failed" not in names
