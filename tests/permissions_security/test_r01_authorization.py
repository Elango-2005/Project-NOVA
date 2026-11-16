"""R01 real local handlers and adversarial authorization tests; no model required."""
from concurrent.futures import ThreadPoolExecutor

import pytest

from nova.main import NovaRuntimeContainer
from nova.core.cognition.conversation_session import PersonalAssistantChatSession
from nova.core.domain.primitives import AgentRole
from nova.core.domain.agent_spec import get_agent_spec
from nova.core.domain.proposals import ToolCallProposal
from nova.core.orchestration.ports import ToolActionContext, UnapprovedToolExecutionError


def runtime(root, auto=False):
    return NovaRuntimeContainer(data_dir=root / "data", workspace_dir=root / "workspace",
                                auto_approve_reversible_local_writes=auto)


def action(container, title="Bound note"):
    proposal = ToolCallProposal(agent_role=AgentRole.CODING, node_id="write",
        thought="Explicit test command", tool_name="notes.create_note",
        arguments={"title": title, "body": "Exact body"},
        declared_sensitivity="reversible_write", rationale="R01 adversarial test")
    context = ToolActionContext(task_id="task_r01", session_id="session_r01",
                                node_id="write", agent_role=AgentRole.CODING)
    stamp = container.permission_gate.evaluate_tool_proposal(proposal, context=context)
    return proposal, context, stamp


def approve(container, proposal, context):
    container.permission_gate.record_user_decision(proposal.proposal_id, True,
                                                  proposal=proposal, context=context)
    return container.permission_gate.evaluate_tool_proposal(proposal, context=context)


@pytest.mark.parametrize("mutation", ["arguments", "role", "task", "session", "node", "token", "expiry"])
def test_exact_binding_rejects_forgery(tmp_path, mutation):
    c = runtime(tmp_path)
    p, ctx, _ = action(c)
    stamp = approve(c, p, ctx)
    if mutation == "arguments":
        p = p.model_copy(update={"arguments": {"title": "Changed", "body": "Altered"}})
    elif mutation == "role":
        p = p.model_copy(update={"agent_role": AgentRole.COMMS})
    elif mutation in ("task", "session", "node"):
        ctx = ctx.model_copy(update={mutation + "_id": "forged"})
    elif mutation == "token":
        stamp = stamp.model_copy(update={"authorization_id": "forged"})
    else:
        stamp = stamp.model_copy(update={"expires_at": stamp.expires_at + 10000})
    with pytest.raises(UnapprovedToolExecutionError):
        c.tool_executor.execute_approved_tool(p, stamp, context=ctx)
    assert not list(c.notes_manager.notes_dir.glob("*.md"))


def test_runtime_identity_required_and_model_cannot_choose_role(tmp_path):
    c = runtime(tmp_path)
    p, ctx, _ = action(c)
    with pytest.raises(UnapprovedToolExecutionError):
        c.permission_gate.evaluate_tool_proposal(p)
    with pytest.raises(UnapprovedToolExecutionError):
        c.permission_gate.evaluate_tool_proposal(p, context=ctx.model_copy(update={"agent_role": AgentRole.COMMS}))


def test_expiry_and_revocation_rechecked_before_handler(tmp_path):
    c = runtime(tmp_path)
    p, ctx, _ = action(c)
    stamp = approve(c, p, ctx)
    c.policy_store.revoke_tool(p.tool_name, agent_role=AgentRole.CODING)
    with pytest.raises(UnapprovedToolExecutionError, match="revoked"):
        c.tool_executor.execute_approved_tool(p, stamp, context=ctx)
    c.policy_store.restore_tool(p.tool_name, agent_role=AgentRole.CODING)
    c.permission_gate.ledger.clock = lambda: stamp.expires_at + 1
    with pytest.raises(UnapprovedToolExecutionError, match="expired"):
        c.tool_executor.execute_approved_tool(p, stamp, context=ctx)
    assert not list(c.notes_manager.notes_dir.glob("*.md"))


def test_atomic_claim_allows_one_write_and_rejects_restart_replay(tmp_path):
    c = runtime(tmp_path)
    p, ctx, _ = action(c)
    stamp = approve(c, p, ctx)
    def run(_):
        try:
            return c.tool_executor.execute_approved_tool(p, stamp, context=ctx).status
        except UnapprovedToolExecutionError:
            return "rejected"
    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(run, range(8)))
    assert results.count("success") == 1
    assert results.count("rejected") == 7
    restarted = runtime(tmp_path)
    with pytest.raises(UnapprovedToolExecutionError, match="replay"):
        restarted.tool_executor.execute_approved_tool(p, stamp, context=ctx)
    logs = c.audit_log.query_logs(task_id=ctx.task_id)
    assert sum(log.execution_status == "success" for log in logs) == 1
    assert all(log.session_id == ctx.session_id and log.agent_role == ctx.agent_role for log in logs)


def test_interrupted_claim_is_never_reissued(tmp_path):
    c = runtime(tmp_path)
    p, ctx, _ = action(c)
    stamp = approve(c, p, ctx)
    c.permission_gate.claim_authorization(p, stamp, context=ctx)
    restarted = runtime(tmp_path)
    with pytest.raises(UnapprovedToolExecutionError, match="uncertain"):
        restarted.permission_gate.evaluate_tool_proposal(p, context=ctx)
    assert not list(c.notes_manager.notes_dir.glob("*.md"))


def test_deny_revoke_and_policy_persist(tmp_path):
    c = runtime(tmp_path)
    session = PersonalAssistantChatSession(c)
    pending = session.submit_action("notes.create_note", {"title": "Denied", "body": "No write"}).pending_approval
    denied = session.resolve_approval(**{k: pending[k] for k in ("task_id", "node_id", "proposal_id")}, approved=False)
    assert denied.action_category == "action_failed"
    c.policy_store.revoke_tool("calendar.create_event")
    restarted = runtime(tmp_path)
    assert restarted.policy_store.is_tool_revoked("calendar.create_event", AgentRole.COMMS)
    assert not list(c.notes_manager.notes_dir.glob("*.md"))


def test_actual_failure_collision_validation_and_conflicts(tmp_path):
    c = runtime(tmp_path, auto=True)
    session = PersonalAssistantChatSession(c)
    args = {"title": "One note", "body": "Original"}
    assert session.submit_action("notes.create_note", args).action_category == "note_saved"
    result = session.submit_action("notes.create_note", {**args, "body": "Replacement"})
    assert result.action_category == "action_failed"
    assert "Original" in (c.notes_manager.notes_dir / "one_note.md").read_text()
    event = {"title": "Study", "start_iso": "2026-11-17T10:00:00", "end_iso": "2026-11-17T11:00:00"}
    assert session.submit_action("calendar.create_event", event).action_category == "event_scheduled"
    assert "overlaps" in session.submit_action("calendar.create_event", {**event, "title": "Overlap"}).text
    assert session.submit_action("calendar.create_event", {**event, "start_iso": "invalid"}).action_category == "action_failed"
    assert len(c.calendar_store.list_events()) == 2
    assert session.send_message("Schedule an event tomorrow").action_category == "clarification_required"
    assert any(log.execution_status == "error" for log in c.audit_log.query_logs())


def test_note_capability_is_narrow_and_rejects_extra_path(tmp_path):
    c = runtime(tmp_path)
    assert "notes.create_note" in get_agent_spec(AgentRole.CODING).allowed_tools
    assert "notes.create_note" not in get_agent_spec(AgentRole.COMMS).allowed_tools
    p, ctx, _ = action(c)
    forged = p.model_copy(update={"proposal_id": "prop_extra_path",
        "arguments": {**p.arguments, "path": "../../outside.md"}})
    # Schema violations cannot grant arbitrary filesystem access.
    stamp = c.permission_gate.evaluate_tool_proposal(forged, context=ctx)
    assert stamp.is_denied
    assert not list(c.notes_manager.notes_dir.glob("*.md"))


def test_canonical_argument_order_does_not_change_binding(tmp_path):
    c = runtime(tmp_path)
    p, ctx, _ = action(c)
    stamp = approve(c, p, ctx)
    reordered = p.model_copy(update={"arguments": dict(reversed(list(p.arguments.items())))})
    assert c.tool_executor.execute_approved_tool(reordered, stamp, context=ctx).status == "success"
