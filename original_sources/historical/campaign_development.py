#!/usr/bin/env python3
"""Run a frozen external-development suite, preserving every repeat and cache."""
import argparse
from contextlib import redirect_stdout
from datetime import datetime, timezone
import hashlib
import importlib
import json
import os
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
FREEZE=ROOT/'research/external_development_v3_v2_freeze.json'


def now():
    return datetime.now(timezone.utc).isoformat()


def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix('.tmp')
    temp.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')
    temp.replace(path)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--dataset',choices=['polyie','mulms'],required=True)
    ap.add_argument('--model',required=True)
    ap.add_argument('--workers',type=int,default=8)
    ap.add_argument('--repeats',default='0,1,2')
    args=ap.parse_args()
    freeze=json.loads(FREEZE.read_text())
    for name,expected in freeze['file_sha256'].items():
        assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==expected,'Frozen file changed: '+name
    repeats=list(map(int,args.repeats.split(',')))
    assert repeats==freeze['planned_repeat_ids'],'Do not select or omit a planned repetition.'
    assert args.model in freeze['model_profiles']
    for name in ['FEEDBACK_API_TOP_P','FEEDBACK_API_TOP_K','FEEDBACK_API_MIN_P']:
        os.environ.pop(name,None)
    os.environ.update({k:str(v) for k,v in freeze['common_environment'].items()})
    os.environ.update({k:str(v) for k,v in freeze['model_profiles'][args.model].items()})
    os.environ['POLYIE_PROTOCOL_VERSION']=freeze['versions']['polyie']
    os.environ['MULMS_PROTOCOL_VERSION']=freeze['versions']['mulms']
    os.environ['SILICONFLOW_KEY_FILE']=str(Path.home()/'.siliconflow_feedback_key')
    task=importlib.import_module(args.dataset+'_experiment')
    analysis=importlib.import_module('analyze_edit_controls')
    name=args.dataset+'__'+args.model.replace('/','__')
    state_path=ROOT/'logs/development_jobs'/(name+'.json')
    log_path=ROOT/'logs/development_jobs'/(name+'.log')
    state={'dataset':args.dataset,'model':args.model,'version':task.VERSION,
           'planned_repeat_ids':repeats,'started_at_utc':now(),'pid':os.getpid(),
           'freeze_sha256':hashlib.sha256(FREEZE.read_bytes()).hexdigest(),
           'status':'running','completed_stages':[],'evaluations':[]}
    write(state_path,state)
    print(json.dumps({'started':name,'pid':os.getpid(),'state':str(state_path.relative_to(ROOT))}),flush=True)
    try:
        for repeat in repeats:
            scopes=['feasibility','full'] if repeat==0 else ['full']
            for scope in scopes:
                size=4 if scope=='feasibility' else 0
                for stage in task.GENERATED:
                    state.update(repeat=repeat,scope=scope,current_stage=stage,stage_started_at_utc=now())
                    write(state_path,state)
                    with log_path.open('a') as log,redirect_stdout(log):
                        if args.dataset=='polyie':
                            failed=task.run_stage(args.model,'dev',stage,args.workers,repeat,size)
                        else:
                            failed=task.run_stage(args.model,stage,args.workers,repeat,size)
                    item={'repeat':repeat,'scope':scope,'stage':stage,'failed':failed,'completed_at_utc':now()}
                    state['completed_stages'].append(item);write(state_path,state)
                    print(json.dumps(item),flush=True)
                    if failed:
                        raise RuntimeError('Explicit failed items retained; suite not complete, no artificial empty-baseline comparison.')
                if scope=='feasibility':
                    continue  # Format first; do not rank methods on this subset.
                state.update(current_stage='offline_evaluation');write(state_path,state)
                with log_path.open('a') as log,redirect_stdout(log):
                    if args.dataset=='polyie':task.evaluate(args.model,'dev',repeat,0)
                    else:task.evaluate(args.model,repeat,0)
                    analysis.analyze(args.dataset,args.model,repeat,0,0)
                state['evaluations'].append({'repeat':repeat,'scope':'full_development','completed_at_utc':now()})
                write(state_path,state)
                print(json.dumps({'repeat':repeat,'full_development_evaluation':'completed'}),flush=True)
        state.update(status='completed',completed_at_utc=now());write(state_path,state)
        print(json.dumps({'completed':name,'all_planned_repeats':repeats}),flush=True)
    except BaseException as error:
        state.update(status='stopped_explicit_failure',error=str(error),stopped_at_utc=now())
        write(state_path,state)
        raise


if __name__=='__main__':main()
