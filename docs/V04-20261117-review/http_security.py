import importlib.util,sys,json,sqlite3
from pathlib import Path
r=Path('C:/NOVA');o=Path(__file__).parent
s=importlib.util.spec_from_file_location('live',r/'tests/integration/c03_live_api.py');h=importlib.util.module_from_spec(s);s.loader.exec_module(h)
records=[]
c=h.Case('v04-http-security')
try:
 p=c.start();card=c.pending(p,title='V04 rejected attempts')
 for field in ('task_id','node_id','proposal_id'):
  code,body=c.request(p,'/assistant/approve',{**card,field:'v04-forged', 'approved':True})
  assert code==409
 code,body=c.request(p,'/assistant/approve',{**card,'approved':True,'arguments':{'body':'V04_SECRET_SENTINEL'}})
 assert code==422
 state,events,files,ledger=c.inspect(p,card)
 assert state['status']=='awaiting_approval' and not files and ledger[0][1]=='pending'
 code,audits=c.request(p,'/audit/authorization-attempts')
 assert code==200 and len(audits)==4 and all(x['outcome']=='rejected' for x in audits)
 assert 'V04_SECRET_SENTINEL' not in json.dumps(audits)
 # Real SQLite abort of mandatory audit insert; valid claim must roll back.
 with sqlite3.connect(c.root/'data/nova_audit.db') as db:
  db.execute("CREATE TRIGGER v04_fail_audit BEFORE INSERT ON authorization_attempts BEGIN SELECT RAISE(ABORT,'V04 injected audit failure'); END")
 code,body=c.request(p,'/assistant/approve',{**card,'approved':True})
 assert code==503 and body['recovery_required'] is True
 state,events,files,ledger=c.inspect(p,card)
 assert state['status']=='running' and not files and ledger[0][1]=='approved'
 assert not any(e['event_type'] in ('tool.started','task.completed','task.failed') for e in events)
 # Audit failure on an unauthorized request must also fail closed.
 code,body=c.request(p,'/assistant/approve',{**card,'proposal_id':'v04-forged','approved':True})
 assert code>=400
 assert not c.inspect(p,card)[2]
 c.record['verdict']='PASS'
finally:
 c.close();records.append(c.record);(o/'http-security.json').write_text(json.dumps(records,indent=2))
print('HTTP binding, sanitized audit, audit failure rollback: PASS')
