"""Process-shared request pacing; client targets are not provider quota claims.

Only scheduling and transport diagnostics change. Prompt content, model sampling,
response parsing, complete-stage requirement and downstream scoring stay frozen.
"""
from contextlib import contextmanager
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import time
import uuid

ROOT=Path(__file__).resolve().parents[1]
DEFAULT={'initial_client_tpm':150000,'minimum_client_tpm':30000,
         'maximum_client_tpm':1000000,'client_rpm':30,'max_inflight':2,
         'cooldown_seconds':65,'healthy_growth_interval_seconds':300,
         'healthy_growth_multiplier':1.25,'rate_failure_multiplier':.5}


def admission_decision(state,now,tokens,config):
    """Pure decision used by the real clock/lock implementation and offline checks."""
    entries=[x for x in state.get('entries',[]) if now-x['time']<60]
    active={k:v for k,v in state.get('active',{}).items()
            if now-v['time']<1800 and (v.get('alive',True))}
    budget=state.get('client_tpm',config['initial_client_tpm'])
    wait=max(0,state.get('cooldown_until',0)-now)
    if len(active)>=config['max_inflight']:wait=max(wait,1)
    if len(entries)>=config['client_rpm']:
        wait=max(wait,60-(now-min(x['time'] for x in entries)))
    # Permit one oversize request only in an otherwise empty rolling window.
    # This avoids deadlock while spacing it at least a complete minute apart.
    if entries and sum(x['tokens'] for x in entries)+tokens>budget:
        wait=max(wait,60-(now-min(x['time'] for x in entries)))
    return {'entries':entries,'active':active,'wait':max(0,wait),
            'client_tpm':budget,'fits_now':wait<=0}


class SharedPacer:
    def __init__(self,root=ROOT/'logs/siliconflow_pacing',config=None):
        self.root=Path(root);self.root.mkdir(parents=True,exist_ok=True)
        self.config=dict(DEFAULT if config is None else config)

    @contextmanager
    def locked(self,model):
        key=hashlib.sha256(model.encode()).hexdigest()[:20]
        with (self.root/(key+'.lock')).open('a+') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX)
            path=self.root/(key+'.json')
            state=json.loads(path.read_text()) if path.exists() else {
                'model':model,'client_tpm':self.config['initial_client_tpm'],
                'entries':[],'active':{},'cooldown_until':0,'last_rate_failure':0,
                'last_growth':time.time(),'config':self.config}
            yield state
            temp=path.with_suffix('.tmp');temp.write_text(json.dumps(state,indent=2)+'\n');temp.replace(path)
            fcntl.flock(lock,fcntl.LOCK_UN)

    def acquire(self,model,tokens):
        started=time.monotonic();ticket=uuid.uuid4().hex
        while True:
            now=time.time()
            with self.locked(model) as state:
                for value in state.get('active',{}).values():
                    try:os.kill(value['pid'],0)
                    except ProcessLookupError:value['alive']=False
                decision=admission_decision(state,now,tokens,self.config)
                state['entries']=decision['entries'];state['active']=decision['active']
                if decision['fits_now']:
                    if (now-max(state.get('last_growth',0),state.get('last_rate_failure',0))
                            >=self.config['healthy_growth_interval_seconds']
                            and sum(x['tokens'] for x in state['entries'])>.5*state['client_tpm']):
                        state['client_tpm']=min(self.config['maximum_client_tpm'],math.ceil(
                            state['client_tpm']*self.config['healthy_growth_multiplier']))
                        state['last_growth']=now
                    state['entries'].append({'ticket':ticket,'time':now,'tokens':tokens})
                    state['active'][ticket]={'time':now,'pid':os.getpid()}
                    return {'model':model,'ticket':ticket,'reserved_tokens':tokens,
                            'admission_wait_s':time.monotonic()-started,
                            'client_tpm_at_admission':state['client_tpm']}
                wait=decision['wait']
            time.sleep(min(5,max(.1,wait)))

    def release(self,ticket,*,usage=None,rate_failure=False,retry_after=0):
        now=time.time()
        with self.locked(ticket['model']) as state:
            state['active'].pop(ticket['ticket'],None)
            if usage and 'total_tokens' in usage:
                for item in state['entries']:
                    if item['ticket']==ticket['ticket']:
                        item['tokens']=max(0,int(usage['total_tokens']))
            if rate_failure:
                state['client_tpm']=max(self.config['minimum_client_tpm'],math.floor(
                    state['client_tpm']*self.config['rate_failure_multiplier']))
                state['cooldown_until']=max(state.get('cooldown_until',0),now+max(
                    self.config['cooldown_seconds'],float(retry_after)))
                state['last_rate_failure']=now
                state['rate_failure_count']=state.get('rate_failure_count',0)+1
            state['released_at_utc_seconds']=now


PACER=SharedPacer()
