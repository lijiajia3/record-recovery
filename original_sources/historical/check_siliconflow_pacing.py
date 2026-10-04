from siliconflow_pacing import DEFAULT,SharedPacer,admission_decision
from pathlib import Path
import tempfile,time
c=DEFAULT
s={'entries':[{'time':100,'tokens':140000}], 'active':{},'client_tpm':150000}
assert not admission_decision(s,110,20000,c)['fits_now']
assert admission_decision(s,160,20000,c)['fits_now']
s['entries']=[];s['active']={'a':{'time':100},'b':{'time':100}}
assert not admission_decision(s,110,1,c)['fits_now']
s['active']={};s['cooldown_until']=170
assert admission_decision(s,110,1,c)['wait']==60
s['cooldown_until']=0
assert admission_decision(s,110,300000,c)['fits_now'] # oversized single request cannot deadlock
with tempfile.TemporaryDirectory() as directory:
 p=SharedPacer(Path(directory),c);a=p.acquire('model',10000);p.release(a,usage={'total_tokens':123})
 with p.locked('model') as state:
  assert state['entries'][0]['tokens']==123 and not state['active']
 b=p.acquire('model',1000);p.release(b,rate_failure=True,retry_after=70)
 with p.locked('model') as state:
  assert state['client_tpm']==75000 and state['cooldown_until']>time.time()+65
 print('PASS: rolling reservations, concurrency, cooldown, oversized admission, usage reconciliation and 429 target reduction')
