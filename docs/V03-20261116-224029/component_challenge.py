import sys,json,tempfile,sqlite3,hashlib,traceback
from pathlib import Path
sys.path.insert(0,'C:/NOVA/services/backend/src')
from nova.main import NovaRuntimeContainer
from nova.core.orchestration.ports import copy_model
from nova.core.domain.primitives import TaskStatus
from nova.core.orchestration.checkpoint import SqliteCheckpointStore
out=Path(__file__).parent;root=Path(tempfile.mkdtemp(prefix='nova-v03-component-'));records=[]
def runtime():return NovaRuntimeContainer(data_dir=root/'data',workspace_dir=root/'workspace',auto_approve_reversible_local_writes=False)
a,b=runtime(),runtime()
def pending(title):return a.orchestrator.start_assistant_action(tool_name='notes.create_note',arguments={'title':title,'body':'synthetic'})
s=pending('Ownership')
owned=copy_model(s.transition_task_status(TaskStatus.RUNNING),approval_attempt_id='runtime-A')
a.checkpoint_store.save_checkpoint(owned,protect_terminal=True,expected_seq=s.checkpoint_seq)
stolen=copy_model(owned,approval_attempt_id='runtime-B',checkpoint_seq=owned.checkpoint_seq+1)
try:
 b.checkpoint_store.save_checkpoint(stolen,protect_terminal=True,expected_seq=owned.checkpoint_seq)
 records.append({'name':'owner cannot be stolen with current sequence','verdict':'FAIL','before_owner':owned.approval_attempt_id,'after_owner':b.checkpoint_store.load_latest(s.task_id).state.approval_attempt_id,'scope':'internal CheckpointStorePort, no public HTTP route demonstrated'})
except Exception as e:records.append({'name':'owner protection','verdict':'PASS','error':str(e)})
try:
 a.checkpoint_store.save_checkpoint(owned,protect_terminal=True,expected_seq=s.checkpoint_seq)
 records.append({'name':'stale sequence rejected','verdict':'FAIL'})
except ValueError as e:records.append({'name':'stale sequence rejected','verdict':'PASS','error':str(e)})
for winner in (TaskStatus.COMPLETED,TaskStatus.FAILED):
 s=pending(winner.value);terminal=s.transition_task_status(winner)
 a.orchestrator._save_checkpoint(terminal,node_id='assistant_action')
 replacement=copy_model(terminal,status=TaskStatus.RUNNING,checkpoint_seq=terminal.checkpoint_seq+1)
 result=b.checkpoint_store.save_checkpoint(replacement,protect_terminal=True,expected_seq=terminal.checkpoint_seq)
 records.append({'name':'terminal winner '+winner.value,'verdict':'PASS' if result.state.status==winner else 'FAIL','persisted':b.checkpoint_store.load_latest(s.task_id).state.status.value})
# A synthetic legacy checkpoint removes the new field and retains a correct checksum.
s=pending('legacy')
with sqlite3.connect(a.checkpoint_store.db_path) as conn:
 row=conn.execute('SELECT state_json FROM task_checkpoints WHERE task_id=? AND checkpoint_seq=?',(s.task_id,s.checkpoint_seq)).fetchone();raw=json.loads(row[0]);raw.pop('approval_attempt_id');payload=json.dumps(raw,sort_keys=True,separators=(',',':'))
 conn.execute('UPDATE task_checkpoints SET state_json=?,sha256_checksum=? WHERE task_id=? AND checkpoint_seq=?',(payload,hashlib.sha256(payload.encode()).hexdigest(),s.task_id,s.checkpoint_seq))
restarted=runtime();loaded=restarted.checkpoint_store.load_latest(s.task_id).state
records.append({'name':'legacy checkpoint without ownership field','verdict':'PASS' if loaded.approval_attempt_id is None else 'FAIL'})
r=loaded.node_records['assistant_action'];result=restarted.orchestrator.submit_user_approval(loaded.task_id,r.node_id,r.pending_tool_proposal.proposal_id,True)
records.append({'name':'legacy pending approval real handler','verdict':'PASS' if result.status==TaskStatus.COMPLETED else 'FAIL','files':[str(p) for p in restarted.notes_manager.notes_dir.glob('*.md')]})
(out/'component-challenges.json').write_text(json.dumps({'root':str(root),'checks':records},indent=2));print(json.dumps(records,indent=2))
