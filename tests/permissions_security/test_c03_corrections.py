"""C03 regressions: real runtime handlers/SQLite with explicit publication and persistence faults."""
import json
import sqlite3
from concurrent.futures import ThreadPoolExecutor

import pytest

from nova.main import NovaRuntimeContainer
from nova.core.domain.events import EventType
from nova.core.domain.primitives import TaskStatus
from nova.core.orchestration.ports import (copy_model, UnapprovedToolExecutionError,
    ExecutionOutcomeUnavailable, ExecutionPhase, ExecutionProgress)
from nova.core.orchestration.checkpoint import InMemoryCheckpointStore, SqliteCheckpointStore, CheckpointConflict
from nova.core.orchestration.state import TaskExecutionState


def runtime(root):
    return NovaRuntimeContainer(data_dir=root/'data', workspace_dir=root/'workspace',
                                auto_approve_reversible_local_writes=False)


def pending(c, tool_name='notes.create_note'):
    args={'title':'C03','body':'Synthetic'} if tool_name=='notes.create_note' else {
        'title':'C03 calendar','start_iso':'2026-12-01T10:00:00','end_iso':'2026-12-01T11:00:00'}
    s=c.orchestrator.start_assistant_action(tool_name=tool_name,arguments=args)
    r=s.node_records['assistant_action']
    return s, dict(task_id=s.task_id,node_id=r.node_id,proposal_id=r.pending_tool_proposal.proposal_id,approved=True)


@pytest.mark.parametrize('tool_name',['notes.create_note','calendar.create_event'])
@pytest.mark.parametrize('event_type',[EventType.TOOL_STARTED,EventType.TOOL_COMPLETED,EventType.TASK_COMPLETED])
@pytest.mark.parametrize('error',[UnapprovedToolExecutionError,TypeError,OSError])
def test_publisher_errors_do_not_reclassify_confirmed_write(tmp_path, caplog, event_type, error, tool_name):
    c=runtime(tmp_path);state,card=pending(c,tool_name);actual=c.event_bus.publish
    def fail(event):
        if event.event_type==event_type:raise error('synthetic_secret_do_not_log')
        actual(event)
    c.event_bus.publish=fail
    result=c.orchestrator.submit_user_approval(**card)
    assert result.status==TaskStatus.COMPLETED
    assert len(list(c.notes_manager.notes_dir.glob('*.md')))==(1 if tool_name=='notes.create_note' else 0)
    with sqlite3.connect(c.data_dir/'nova_calendar.db') as db:
        assert db.execute('SELECT count(*) FROM calendar_events').fetchone()[0]==(1 if tool_name=='calendar.create_event' else 0)
    assert 'event_publication_failed' in caplog.text and 'synthetic_secret_do_not_log' not in caplog.text
    with sqlite3.connect(c.data_dir/'nova_audit.db') as db:
        assert db.execute('SELECT status FROM tool_authorizations').fetchone()[0]=='success'
        assert db.execute("SELECT count(*) FROM authorization_attempts WHERE outcome='rejected'").fetchone()[0]==0
    assert not any(e.event_type in (EventType.TOOL_FAILED,EventType.TASK_FAILED) for e in c.event_bus.published_events)
    assert 'task.completed' in ''.join(runtime(tmp_path).api_service.stream_task_events_sse(state.task_id))


@pytest.mark.parametrize('store_kind',['sqlite','memory'])
def test_owner_and_revision_invariant(tmp_path,store_kind):
    c=runtime(tmp_path);s,_=pending(c)
    store=SqliteCheckpointStore(tmp_path/'checkpoints.db') if store_kind=='sqlite' else InMemoryCheckpointStore()
    state=TaskExecutionState(**json.loads(s.model_dump_json()))  # legacy/data-only state has no observation token
    store.save_checkpoint(state)
    owned=copy_model(state.transition_task_status(TaskStatus.RUNNING),approval_attempt_id='A')
    store.save_checkpoint(owned,expected_seq=state.checkpoint_seq,expected_owner=None,owner_transition='acquire')
    for flag in (True,False):
        stolen=copy_model(owned,approval_attempt_id='B',checkpoint_seq=owned.checkpoint_seq+1)
        with pytest.raises(CheckpointConflict):
            store.save_checkpoint(stolen,protect_terminal=flag,expected_seq=owned.checkpoint_seq,expected_owner='A')
    changed=owned.transition_task_status(TaskStatus.RUNNING)
    with pytest.raises(CheckpointConflict):store.save_checkpoint(changed,expected_seq=owned.checkpoint_seq,expected_owner='B')
    with pytest.raises(CheckpointConflict):store.save_checkpoint(changed,expected_owner='B')
    with pytest.raises(CheckpointConflict):store.save_checkpoint(changed,expected_owner=None)
    store.save_checkpoint(changed,expected_seq=owned.checkpoint_seq,expected_owner='A')
    stale_body=copy_model(owned,checkpoint_seq=changed.checkpoint_seq+1)
    with pytest.raises(CheckpointConflict):
        store.save_checkpoint(stale_body,expected_seq=changed.checkpoint_seq,expected_owner='A')
    with pytest.raises(CheckpointConflict):store.save_checkpoint(owned)
    done=changed.transition_task_status(TaskStatus.FAILED)
    store.save_checkpoint(done)
    with pytest.raises(CheckpointConflict):
        store.save_checkpoint(copy_model(done,status=TaskStatus.RUNNING,checkpoint_seq=done.checkpoint_seq+1),
                              expected_seq=done.checkpoint_seq,expected_owner='A',protect_terminal=False)
    assert store.load_latest(done.task_id).state.status==TaskStatus.FAILED
    assert len(store.list_history(done.task_id))==4


@pytest.mark.parametrize('store_kind',['sqlite','memory'])
def test_owner_release_requires_explicit_transition(tmp_path,store_kind):
    c=runtime(tmp_path);s,card=pending(c)
    store=SqliteCheckpointStore(tmp_path/'store.db') if store_kind=='sqlite' else InMemoryCheckpointStore()
    s=TaskExecutionState(**json.loads(s.model_dump_json()));store.save_checkpoint(s)
    owned=copy_model(s.transition_task_status(TaskStatus.RUNNING),approval_attempt_id='A')
    store.save_checkpoint(owned,owner_transition='acquire')
    rec=copy_model(owned.node_records[card['node_id']],pending_tool_proposal=None)
    released=copy_model(owned.update_node_record(rec),approval_attempt_id=None)
    with pytest.raises(CheckpointConflict):store.save_checkpoint(released)
    store.save_checkpoint(released,owner_transition='release')
    assert store.load_latest(s.task_id).state.approval_attempt_id is None


@pytest.mark.parametrize('boundary',['audit','ledger','checkpoint'])
def test_required_post_write_persistence_failure_is_unconfirmed(tmp_path,boundary):
    c=runtime(tmp_path);s,card=pending(c)
    if boundary in ('audit','ledger'):
        with sqlite3.connect(c.data_dir/'nova_audit.db') as db:
            if boundary=='audit':db.execute("CREATE TRIGGER fail BEFORE INSERT ON tool_audit_log WHEN NEW.execution_status='success' BEGIN SELECT RAISE(ABORT,'outcome fault'); END")
            else:db.execute("CREATE TRIGGER fail BEFORE UPDATE ON tool_authorizations WHEN NEW.status='success' BEGIN SELECT RAISE(ABORT,'outcome fault'); END")
    else:
        with sqlite3.connect(c.checkpoint_store.db_path) as db:
            db.execute("CREATE TRIGGER fail BEFORE INSERT ON task_terminal_events BEGIN SELECT RAISE(ABORT,'checkpoint fault'); END")
    with pytest.raises(ExecutionOutcomeUnavailable):c.orchestrator.submit_user_approval(**card)
    assert len(list(c.notes_manager.notes_dir.glob('*.md')))==1
    assert c.api_service.get_task_status(s.task_id).status==TaskStatus.RUNNING
    assert not c.checkpoint_store.terminal_events(s.task_id)
    restart=runtime(tmp_path)
    assert restart.orchestrator.run_until_blocked_or_complete(s.task_id).status==TaskStatus.RUNNING
    with pytest.raises(Exception):restart.orchestrator.submit_user_approval(**card)
    assert len(list(c.notes_manager.notes_dir.glob('*.md')))==1


def test_claim_committed_then_exception_never_becomes_preclaim_denial(tmp_path):
    c=runtime(tmp_path);s,card=pending(c);actual=c.permission_gate.claim_authorization
    def committed_then_lost(*args,**kwargs):
        actual(*args,**kwargs)
        raise UnapprovedToolExecutionError('Injected loss of claim return')
    c.permission_gate.claim_authorization=committed_then_lost
    with pytest.raises(ExecutionOutcomeUnavailable):c.orchestrator.submit_user_approval(**card)
    assert c.api_service.get_task_status(s.task_id).status==TaskStatus.RUNNING
    assert not list(c.notes_manager.notes_dir.glob('*.md'))
    assert not c.checkpoint_store.terminal_events(s.task_id)


def test_executor_phase_records_confirmed_outcome(tmp_path):
    c=runtime(tmp_path);s,card=pending(c);rec=s.node_records[card['node_id']];p=rec.pending_tool_proposal
    ctx=c.orchestrator._action_context(s,rec.node_id)
    c.permission_gate.record_user_decision(p.proposal_id,True,proposal=p,context=ctx)
    stamp=c.permission_gate.evaluate_tool_proposal(p,context=ctx);progress=ExecutionProgress()
    def started():
        assert progress.phase==ExecutionPhase.CLAIM_COMMITTED
        assert c.permission_gate.ledger.execution_outcome(p,ctx)=='claimed'
    result=c.tool_executor.execute_approved_tool(p,stamp,context=ctx,progress=progress,on_started=started)
    assert result.status=='success' and progress.phase==ExecutionPhase.OUTCOME_CONFIRMED
    assert progress.persisted_outcome=='success'


def test_successful_ledger_commit_followed_by_exception_preserves_outcome(tmp_path):
    c=runtime(tmp_path);s,card=pending(c);actual=c.permission_gate.ledger.finish
    def committed(proposal_id,status):
        actual(proposal_id,status)
        raise UnapprovedToolExecutionError('Injected error after persisted success')
    c.permission_gate.ledger.finish=committed
    with pytest.raises(ExecutionOutcomeUnavailable) as error:c.orchestrator.submit_user_approval(**card)
    assert error.value.persisted_outcome=='success'
    assert c.api_service.get_task_status(s.task_id).status==TaskStatus.RUNNING
    assert len(list(c.notes_manager.notes_dir.glob('*.md')))==1
    assert not c.checkpoint_store.terminal_events(s.task_id)
    assert not any(r['outcome']=='rejected' for r in c.permission_gate.ledger.query_attempts())


def test_claim_return_and_outcome_lookup_loss_does_not_audit_false_rejection(tmp_path):
    c=runtime(tmp_path);s,card=pending(c);claim=c.permission_gate.claim_authorization
    def lost(*args,**kwargs):
        claim(*args,**kwargs)
        raise UnapprovedToolExecutionError('Injected ambiguous claim return')
    def unreadable(*args,**kwargs):raise sqlite3.OperationalError('Injected unavailable outcome read')
    c.permission_gate.claim_authorization=lost
    c.permission_gate.ledger.execution_outcome=unreadable
    with pytest.raises(ExecutionOutcomeUnavailable):c.orchestrator.submit_user_approval(**card)
    assert c.api_service.get_task_status(s.task_id).status==TaskStatus.RUNNING
    assert not any(r['outcome']=='rejected' for r in c.permission_gate.ledger.query_attempts())
    assert not c.checkpoint_store.terminal_events(s.task_id)


def test_legacy_checkpoint_loads_without_persisted_runtime_metadata(tmp_path):
    import hashlib
    c=runtime(tmp_path);s,card=pending(c)
    with sqlite3.connect(c.checkpoint_store.db_path) as db:
        raw=json.loads(db.execute('SELECT state_json FROM task_checkpoints WHERE task_id=? AND checkpoint_seq=?',
                                  (s.task_id,s.checkpoint_seq)).fetchone()[0])
        assert '_checkpoint_origin' not in raw
        raw.pop('approval_attempt_id')
        payload=json.dumps(raw,sort_keys=True,separators=(',',':'))
        db.execute('UPDATE task_checkpoints SET state_json=?,sha256_checksum=? WHERE task_id=? AND checkpoint_seq=?',
                   (payload,hashlib.sha256(payload.encode()).hexdigest(),s.task_id,s.checkpoint_seq))
    restarted=runtime(tmp_path)
    assert restarted.checkpoint_store.load_latest(s.task_id).state.approval_attempt_id is None
    assert restarted.orchestrator.submit_user_approval(**card).status==TaskStatus.COMPLETED
    assert len(list(c.notes_manager.notes_dir.glob('*.md')))==1
