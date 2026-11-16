"""Disposable V03 verification driver; not application or repository test source."""
import sys,os,json,time,tempfile,socket,subprocess,sqlite3,hashlib,traceback
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from urllib.request import Request,urlopen
from urllib.error import HTTPError,URLError
ROOT=Path('C:/NOVA');OUT=Path(__file__).parent
sys.path.insert(0,str(ROOT/'services/backend/src'))

def serve(root,port,mode):
    import uvicorn
    from nova.api.assistant_ui import build_user_friendly_assistant_app
    from nova.core.orchestration.ports import UnapprovedToolExecutionError
    app=build_user_friendly_assistant_app(data_dir=root/'data',workspace_dir=root/'workspace',auto_approve_reversible_local_writes=False)
    c=app.state.nova_container
    if mode=='crash_owned':
        actual=c.checkpoint_store.save_checkpoint
        def save(state,*args,**kwargs):
            result=actual(state,*args,**kwargs)
            if state.approval_attempt_id:os._exit(81)
            return result
        c.checkpoint_store.save_checkpoint=save
    if mode=='crash_claim':
        actual=c.permission_gate.claim_authorization
        def claim(*args,**kwargs):
            actual(*args,**kwargs);os._exit(82)
        c.permission_gate.claim_authorization=claim
    if mode in ('publication_failure','post_write_permission_exception'):
        actual=c.event_bus.publish
        def publish(event):
            if mode=='publication_failure' and event.event_type.value=='task.failed':raise OSError('V03 injected publication failure before append')
            if mode=='post_write_permission_exception' and event.event_type.value=='tool.completed':raise UnapprovedToolExecutionError('V03 injected post-execution subscriber authorization exception')
            return actual(event)
        c.event_bus.publish=publish
    if mode=='event_storage_failure':
        with sqlite3.connect(c.checkpoint_store.db_path) as db:
            db.execute("CREATE TRIGGER v03_fail BEFORE INSERT ON task_terminal_events BEGIN SELECT RAISE(ABORT, 'V03 event storage failure'); END")
    @app.post('/__verification/recover/{task_id}')
    def recover(task_id:str):
        return {'status':c.orchestrator.run_until_blocked_or_complete(task_id).status.value}
    uvicorn.run(app,host='127.0.0.1',port=port,log_level='warning')

class Case:
    def __init__(self,name):
        self.root=Path(tempfile.mkdtemp(prefix='nova-v03-'+name+'-')); self.record={'name':name,'root':str(self.root),'http':[],'processes':[]};self.procs=[]
    def request(self,port,path,body=None):
        req=Request(f'http://127.0.0.1:{port}{path}',data=json.dumps(body).encode() if body is not None else None,headers={'Content-Type':'application/json'})
        try:
            with urlopen(req,timeout=20) as r:code,raw=r.status,r.read().decode()
        except HTTPError as e:code,raw=e.code,e.read().decode()
        except Exception as e:
            self.record['http'].append({'port':port,'path':path,'transport_error':repr(e)});return 0,repr(e)
        try:result=json.loads(raw)
        except ValueError:result=raw
        self.record['http'].append({'port':port,'path':path,'request':body,'status':code,'body':result});return code,result
    def start(self,mode='normal'):
        with socket.socket() as s:s.bind(('127.0.0.1',0));port=s.getsockname()[1]
        cmd=[sys.executable,'-B',str(Path(__file__).resolve()),'serve',str(self.root),str(port),mode]
        log=(self.root/f'server-{port}.log').open('x');p=subprocess.Popen(cmd,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0));self.procs.append((p,log));self.record['processes'].append({'command':cmd,'pid':p.pid})
        for _ in range(100):
            try:
                with urlopen(f'http://127.0.0.1:{port}/health',timeout=1) as r:
                    if r.status==200:return port
            except OSError:pass
            time.sleep(.1)
        raise RuntimeError('startup failed')
    def pending(self,port,title='V03 synthetic'):
        code,res=self.request(port,'/assistant/quick-note',{'title':title,'body':'V03 synthetic body'})
        assert code==200,res
        return {k:res['pending_approval'][k] for k in ('task_id','node_id','proposal_id')}
    def inspect(self,port,card):
        task='/tasks/'+card['task_id'];code,status=self.request(port,task);code,sse=self.request(port,task+'/events/sse')
        events=[json.loads(s[6:]) for s in sse.splitlines() if s.startswith('data: ')]
        files={str(p.relative_to(self.root)):p.read_text(encoding='utf-8') for p in (self.root/'workspace').rglob('*.md')}
        with sqlite3.connect(self.root/'data/nova_audit.db') as db:
            ledger=db.execute('SELECT proposal_id,status FROM tool_authorizations').fetchall();audits=db.execute('SELECT outcome,reason_code,trusted_context FROM authorization_attempts').fetchall()
        self.record.setdefault('snapshots',[]).append({'task':status,'events':events,'markdown':files,'authorizations':ledger,'attempts':audits})
        return status,events,files,ledger
    def close(self):
        for p,log in self.procs:
            if p.poll() is None:p.terminate()
            p.wait(timeout=10);log.close()
            for row in self.record['processes']:
                if row['pid']==p.pid:row['exit_code']=p.returncode

def main():
    records=[]
    for mode in ['concurrent_processes','failed_winner','publication_failure','event_storage_failure','crash_owned','crash_claim','post_write_permission_exception']:
        c=Case(mode)
        try:
            a=c.start(mode if mode not in ('concurrent_processes','failed_winner') else 'normal');card=c.pending(a)
            if mode=='concurrent_processes':
                b=c.start()
                with ThreadPoolExecutor(2) as pool:
                    futures=[pool.submit(c.request,p,'/assistant/approve',{**card,'approved':True}) for p in (a,b)];results=[f.result() for f in futures]
                assert sorted(code for code,_ in results)==[200,409],results
                status,events,files,ledger=c.inspect(b,card);assert status['status']=='completed' and len(files)==1
                assert c.request(b,'/assistant/approve',{**card,'approved':True})[0]==409
                assert c.inspect(a,card)[0]['status']=='completed'
            elif mode in ('crash_owned','crash_claim'):
                c.request(a,'/assistant/approve',{**card,'approved':True});c.procs[0][0].wait(timeout=10)
                assert c.procs[0][0].returncode==(81 if mode=='crash_owned' else 82)
                b=c.start();status,events,files,ledger=c.inspect(b,card)
                assert status['status']=='running' and not status['pending_approvals'] and not files
                assert ledger[0][1]==('pending' if mode=='crash_owned' else 'claimed')
                assert c.request(b,'/__verification/recover/'+card['task_id'],{})[1]['status']=='running'
                assert c.request(b,'/assistant/approve',{**card,'approved':True})[0]==409
                assert not c.inspect(b,card)[2]
            else:
                code,res=c.request(a,'/assistant/approve',{**card,'approved':mode=='post_write_permission_exception'})
                status,events,files,ledger=c.inspect(a,card)
                if mode=='event_storage_failure':
                    assert code==500 and status['status']=='running' and not files
                    assert not any(e['event_type']=='task.failed' for e in events)
                elif mode=='post_write_permission_exception':
                    assert code==409 and status['status']=='failed' and len(files)==1 and ledger[0][1]=='success'
                    terminal=[e for e in events if e['event_type']=='task.failed']
                    assert terminal[0]['payload']['execution_started'] is False
                    c.record['verdict']='FAIL';c.record['finding']='Real write and successful authorization ledger falsely finalized as authorization_rejected with execution_started=false.'
                else:
                    assert code==200 and status['status']=='failed' and not files
                    terminal=[e for e in events if e['event_type']=='task.failed'];assert len(terminal)==1
                    b=c.start();assert len([e for e in c.inspect(b,card)[1] if e['event_type']=='task.failed'])==1
                    assert c.request(b,'/assistant/approve',{**card,'approved':True})[0]==409
            c.record.setdefault('verdict','PASS')
        except Exception:
            c.record['verdict']='BLOCKED';c.record['driver_error']=traceback.format_exc()
        finally:
            c.close();records.append(c.record);(OUT/'fresh-challenges.json').write_text(json.dumps(records,indent=2));print(mode,c.record['verdict'],flush=True)

if __name__=='__main__':
    if len(sys.argv)>1:serve(Path(sys.argv[2]),int(sys.argv[3]),sys.argv[4])
    else:main()
