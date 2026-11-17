"""V04 independent assertions; uses existing HTTP launch/record utilities only."""
import sys, json, tempfile, sqlite3, importlib.util
from pathlib import Path
ROOT=Path('C:/NOVA'); OUT=Path(__file__).parent
sys.path.insert(0,str(ROOT/'services/backend/src'))
spec=importlib.util.spec_from_file_location('c03',ROOT/'tests/integration/c03_live_api.py')
h=importlib.util.module_from_spec(spec);spec.loader.exec_module(h)
from nova.main import NovaRuntimeContainer
from nova.core.orchestration.checkpoint import SqliteCheckpointStore,InMemoryCheckpointStore,CheckpointConflict
from nova.core.orchestration.ports import copy_model
from nova.core.orchestration.state import TaskExecutionState
from nova.core.domain.primitives import TaskStatus
records=[]
for calendar in (False,True):
 c=h.Case('v04-calendar' if calendar else 'v04-note')
 try:
  a=c.start('post_write_calendar_permission_exception' if calendar else 'post_write_permission_exception')
  card=c.pending(a,title='V04 independent synthetic',calendar=calendar)
  code,body=c.request(a,'/assistant/approve',{**card,'approved':True})
  s,e,f,l=c.inspect(a,card)
  assert code==200 and s['status']=='completed' and l==[(card['proposal_id'],'success')]
  assert len(f)==(0 if calendar else 1)
  assert len(c.record['snapshots'][-1]['calendar'])==(1 if calendar else 0)
  assert not any(x[0]=='rejected' for x in c.record['snapshots'][-1]['attempts'])
  first=[x for x in e if x['event_type']=='task.completed'];assert len(first)==1
  # Stop the original process before testing restart recovery.
  c.close();c.procs=[]
  b=c.start();s,e,f,l=c.inspect(b,card)
  terminal=[x for x in e if x['event_type']=='task.completed']
  assert s['status']=='completed' and terminal==first
  assert not any(x['event_type']=='task.failed' for x in e)
  c.record['verdict']='PASS'
 finally:
  c.close();records.append(c.record)
  (OUT/'independent.json').write_text(json.dumps(records,indent=2))
 print(c.record['name'],'PASS',flush=True)
for kind in ('sqlite','memory'):
 root=Path(tempfile.mkdtemp(prefix='nova-v04-owner-'))
 c=NovaRuntimeContainer(data_dir=root/'data',workspace_dir=root/'workspace',auto_approve_reversible_local_writes=False)
 s=c.orchestrator.start_assistant_action(tool_name='notes.create_note',arguments={'title':'V04','body':'synthetic'})
 s=TaskExecutionState(**json.loads(s.model_dump_json()))
 store=SqliteCheckpointStore(root/'owner.db') if kind=='sqlite' else InMemoryCheckpointStore()
 store.save_checkpoint(s)
 a=copy_model(s.transition_task_status(TaskStatus.RUNNING),approval_attempt_id='A')
 store.save_checkpoint(a,owner_transition='acquire')
 record={'name':'owner-'+kind,'root':str(root),'rejections':[]}
 def reject(label,state,**kw):
  try:store.save_checkpoint(state,**kw)
  except CheckpointConflict as ex:record['rejections'].append({'name':label,'error':str(ex)})
  else:raise AssertionError(label+' accepted')
 stolen=copy_model(a,approval_attempt_id='B',checkpoint_seq=a.checkpoint_seq+1)
 reject('original V03 current-sequence takeover',stolen,expected_seq=a.checkpoint_seq,protect_terminal=True)
 reject('takeover with correct old owner',stolen,expected_seq=a.checkpoint_seq,expected_owner='A',protect_terminal=False)
 b=a.transition_task_status(TaskStatus.RUNNING);store.save_checkpoint(b,expected_seq=a.checkpoint_seq,expected_owner='A')
 reject('stale sequence',copy_model(a,checkpoint_seq=b.checkpoint_seq+1),expected_seq=a.checkpoint_seq)
 pending_release=copy_model(b,approval_attempt_id=None,checkpoint_seq=b.checkpoint_seq+1)
 reject('release with pending proposal',pending_release,owner_transition='release')
 rec=copy_model(b.node_records['assistant_action'],pending_tool_proposal=None)
 released=copy_model(b.update_node_record(rec),approval_attempt_id=None)
 reject('implicit release',released)
 store.save_checkpoint(released,owner_transition='release')
 assert store.load_latest(s.task_id).state.approval_attempt_id is None
 done=released.transition_task_status(TaskStatus.COMPLETED);store.save_checkpoint(done)
 reject('terminal overwrite',copy_model(done,status=TaskStatus.RUNNING,checkpoint_seq=done.checkpoint_seq+1),expected_seq=done.checkpoint_seq,protect_terminal=False)
 record.update(verdict='PASS',same_owner_update='PASS',explicit_release='PASS',history=[x.state.status.value for x in store.list_history(s.task_id)])
 records.append(record);(OUT/'independent.json').write_text(json.dumps(records,indent=2));print(record['name'],'PASS',flush=True)
