"""C02 real two-process HTTP race. Uses barriers around the actual claim, no handler substitution."""
import sys,os,json,time,tempfile,socket,subprocess,sqlite3
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from urllib.request import Request,urlopen
from urllib.error import HTTPError
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'services/backend/src'))

def serve(root,port,barrier):
    import uvicorn
    from nova.api.assistant_ui import build_user_friendly_assistant_app
    app=build_user_friendly_assistant_app(data_dir=root/'data',workspace_dir=root/'workspace',auto_approve_reversible_local_writes=False)
    if barrier:
        gate=app.state.nova_container.permission_gate
        actual=gate.claim_authorization
        def claim(*args,**kwargs):
            (root/'before-claim').write_text('ready')
            deadline=time.monotonic()+25
            while not (root/'release').exists():
                if time.monotonic()>deadline:raise TimeoutError('barrier')
                time.sleep(.01)
            return actual(*args,**kwargs)
        gate.claim_authorization=claim
    uvicorn.run(app,host='127.0.0.1',port=port,log_level='warning')

def verify():
    root=Path(tempfile.mkdtemp(prefix='nova-c02-http-')); evidence={'root':str(root),'http':[],'processes':[]};procs=[]
    def req(port,path,body=None):
        r=Request(f'http://127.0.0.1:{port}{path}',data=json.dumps(body).encode() if body is not None else None,headers={'Content-Type':'application/json'})
        try:
            with urlopen(r,timeout=30) as response: code,raw=response.status,response.read().decode()
        except HTTPError as exc:code,raw=exc.code,exc.read().decode()
        try: result=json.loads(raw)
        except ValueError:result=raw
        evidence['http'].append({'port':port,'path':path,'status':code,'body':result})
        return code,result
    def start(barrier=False):
        with socket.socket() as sock:sock.bind(('127.0.0.1',0));port=sock.getsockname()[1]
        cmd=[sys.executable,'-B',str(Path(__file__).resolve()),'serve',str(root),str(port),str(int(barrier))]
        log=(root/f'{port}.log').open('x');p=subprocess.Popen(cmd,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0));procs.append((p,log));evidence['processes'].append({'command':cmd,'pid':p.pid})
        for i in range(150):
            try:
                if req(port,'/health')[0]==200:return port
            except OSError:pass
            time.sleep(.1)
        raise RuntimeError('server unavailable')
    try:
        a,b=start(True),start()
        code,result=req(a,'/assistant/quick-note',{'title':'Revoked C02','body':'Synthetic'})
        assert code==200
        card={k:result['pending_approval'][k] for k in ('task_id','node_id','proposal_id')}
        with ThreadPoolExecutor(2) as pool:
            f=pool.submit(req,a,'/assistant/approve',{**card,'approved':True})
            deadline=time.monotonic()+20
            while not (root/'before-claim').exists():
                assert time.monotonic()<deadline
                time.sleep(.01)
            assert req(b,'/permissions/revoke',{'tool_name':'notes.create_note'})[0]==200
            (root/'release').write_text('go')
            code,result=f.result()
        assert code==409 and result['task_status']=='failed',result
        task=f"/tasks/{card['task_id']}"
        assert req(b,task)[1]['status']=='failed'
        assert not req(b,task)[1]['pending_approvals']
        code,sse=req(a,task+'/events/sse')
        events=[json.loads(line[6:]) for line in sse.splitlines() if line.startswith('data: ')]
        assert 'tool.started' not in [e['event_type'] for e in events]
        failed=[e for e in events if e['event_type']=='task.failed']
        assert len(failed)==1 and failed[0]['correlation_id']==card['proposal_id']
        assert req(b,'/assistant/approve',{**card,'approved':True})[0]==409
        assert not list((root/'workspace/notes').glob('*.md'))
        for p,log in procs:p.terminate();p.wait(timeout=10)
        c=start()
        assert req(c,task)[1]['status']=='failed'
        assert 'task.failed' in req(c,task+'/events/sse')[1]
        with sqlite3.connect(root/'data/nova_audit.db') as conn:
            evidence['authorization_attempts']=conn.execute('SELECT outcome,reason_code,trusted_context FROM authorization_attempts').fetchall()
            assert any(r[1]=='revoked' for r in evidence['authorization_attempts'])
        evidence['files']=[str(p.relative_to(root)) for p in (root/'workspace').rglob('*') if p.is_file()]
        evidence['result']='PASS'
    finally:
        for p,log in procs:
            if p.poll() is None:p.terminate();p.wait(timeout=10)
            log.close()
            for item in evidence['processes']:
                if item['pid']==p.pid:item['exit_code']=p.returncode
        out=ROOT/'docs'/('C02-live-'+root.name+'.json');out.write_text(json.dumps(evidence,indent=2));print(out)
    print(evidence['result'])

if __name__=='__main__':
    if len(sys.argv)>1:serve(Path(sys.argv[2]),int(sys.argv[3]),bool(int(sys.argv[4])))
    else:verify()
