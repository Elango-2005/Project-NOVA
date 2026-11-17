import subprocess,sys,json,time
from pathlib import Path
o=Path(__file__).parent;r=o.parents[1];records=[]
commands=[('independent',[sys.executable,'-B',str(o/'challenge.py')]),('pytest',[sys.executable,'-B','-m','pytest','-q','--tb=short']),('c01',[sys.executable,'-B','tests/integration/c01_live_api.py']),('c02',[sys.executable,'-B','tests/integration/c02_live_api.py']),('c03',[sys.executable,'-B','tests/integration/c03_live_api.py']),('diff-check',['git','diff','--check'])]
for label,cmd in commands:
 start=time.monotonic()
 with (o/(label+'.txt')).open('w',encoding='utf-8') as log:
  p=subprocess.run(cmd,cwd=r,stdout=log,stderr=subprocess.STDOUT)
 records.append({'name':label,'command':cmd,'exit_code':p.returncode,'seconds':time.monotonic()-start})
 (o/'commands.json').write_text(json.dumps(records,indent=2));print(label,p.returncode,flush=True)
