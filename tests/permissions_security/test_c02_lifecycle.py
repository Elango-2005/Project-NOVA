"""C02 concurrency checks use real runtimes/SQLite/handlers; callbacks inject ordering and faults."""
from concurrent.futures import ThreadPoolExecutor
import sqlite3
import pytest
from nova.main import NovaRuntimeContainer
from nova.core.domain.events import EventType
from nova.core.domain.primitives import TaskStatus
from nova.core.orchestration.ports import copy_model, ExecutionOutcomeUnavailable


def runtime(root):
    return NovaRuntimeContainer(data_dir=root/'data', workspace_dir=root/'workspace', auto_approve_reversible_local_writes=False)


def pending(c):
    s=c.orchestrator.start_assistant_action(tool_name='notes.create_note', arguments={'title':'C02','body':'Synthetic'})
    r=s.node_records['assistant_action']
    return s, dict(task_id=s.task_id,node_id=r.node_id,proposal_id=r.pending_tool_proposal.proposal_id,approved=True)


def test_revoke_before_claim_terminal_and_restart(tmp_path):
    a,b=runtime(tmp_path),runtime(tmp_path)
    state,card=pending(a)
    actual=a.permission_gate.claim_authorization
    def revoked(*args,**kwargs):
        b.policy_store.revoke_tool('notes.create_note')
        return actual(*args,**kwargs)
    a.permission_gate.claim_authorization=revoked
    with pytest.raises(PermissionError) as error:
        a.orchestrator.submit_user_approval(**card)
    assert error.value.task_status=="failed"
    assert not list(a.notes_manager.notes_dir.glob('*.md'))
    events=a.event_bus.published_events
    assert not any(e.event_type==EventType.TOOL_STARTED for e in events)
    terminal=[e for e in events if e.event_type==EventType.TASK_FAILED]
    assert len(terminal)==1 and terminal[0].correlation_id==card['proposal_id']
    restarted=runtime(tmp_path)
    assert restarted.api_service.get_task_status(state.task_id).status==TaskStatus.FAILED
    assert 'task.failed' in ''.join(restarted.api_service.stream_task_events_sse(state.task_id))
    with pytest.raises(Exception): restarted.orchestrator.submit_user_approval(**card)


def test_concurrent_approval_and_terminal_winner(tmp_path):
    a,b=runtime(tmp_path),runtime(tmp_path)
    state,card=pending(a)
    def run(c):
        try:return c.orchestrator.submit_user_approval(**card).status.value
        except Exception:return 'rejected'
    with ThreadPoolExecutor(2) as pool: results=list(pool.map(run,[a,b]))
    assert sorted(results)==['completed','rejected']
    assert len(list(a.notes_manager.notes_dir.glob('*.md')))==1
    stale=state.transition_task_status(TaskStatus.FAILED)
    with pytest.raises(ValueError):
        a.checkpoint_store.save_checkpoint(stale,protect_terminal=True,expected_seq=state.checkpoint_seq)
    assert b.api_service.get_task_status(state.task_id).status==TaskStatus.COMPLETED


def test_publication_failure_retains_terminal_event(tmp_path):
    a=runtime(tmp_path)
    state,card=pending(a)
    def fail(event):
        if event.event_type==EventType.TASK_FAILED:raise RuntimeError('Injected event subscriber fault')
    a.event_bus.subscribe(fail)
    card['approved']=False
    assert a.orchestrator.submit_user_approval(**card).status==TaskStatus.FAILED
    assert 'task.failed' in ''.join(runtime(tmp_path).api_service.stream_task_events_sse(state.task_id))


def test_event_storage_failure_rolls_back_terminal_checkpoint(tmp_path):
    a=runtime(tmp_path)
    state,card=pending(a)
    with sqlite3.connect(a.checkpoint_store.db_path) as conn:
        conn.execute("CREATE TRIGGER fail_event BEFORE INSERT ON task_terminal_events BEGIN SELECT RAISE(ABORT, 'injected'); END")
    card['approved']=False
    with pytest.raises(ExecutionOutcomeUnavailable) as failure:a.orchestrator.submit_user_approval(**card)
    assert isinstance(failure.value.__cause__, sqlite3.DatabaseError)
    assert a.api_service.get_task_status(state.task_id).status==TaskStatus.RUNNING
    assert not list(a.notes_manager.notes_dir.glob('*.md'))
    assert not a.checkpoint_store.terminal_events(state.task_id)


@pytest.mark.parametrize("winner", [TaskStatus.COMPLETED, TaskStatus.FAILED])
def test_rejection_finalization_preserves_terminal_winner(tmp_path, winner):
    a,b=runtime(tmp_path),runtime(tmp_path)
    state,card=pending(a)
    actual=a.permission_gate.claim_authorization
    def competing(*args, **kwargs):
        latest=b.checkpoint_store.load_latest(state.task_id).state
        won=copy_model(latest, status=winner, checkpoint_seq=latest.checkpoint_seq+1)
        b.orchestrator._save_checkpoint(won, node_id=card['node_id'])
        b.policy_store.revoke_tool('notes.create_note')
        return actual(*args,**kwargs)
    a.permission_gate.claim_authorization=competing
    with pytest.raises(PermissionError) as error:a.orchestrator.submit_user_approval(**card)
    assert error.value.task_status==winner.value
    assert a.api_service.get_task_status(state.task_id).status==winner
    events=a.checkpoint_store.terminal_events(state.task_id)
    assert len(events)==1 and events[0].event_type.value=='task.'+winner.value
    assert not list(a.notes_manager.notes_dir.glob('*.md'))
