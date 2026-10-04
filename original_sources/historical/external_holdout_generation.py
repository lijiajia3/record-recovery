#!/usr/bin/env python3
"""Separately frozen held-out generation; disabled until protocol/selection exist.

No evaluation labels are loaded. Only a complete three-repeat development selection
with unchanged input hashes permits first held-out inference for that setting.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor,as_completed
from datetime import datetime,timezone
import hashlib,importlib,json,os
from pathlib import Path
from external_holdout_prompts import polyie_prompt,mulms_prompt
from pilot import ROOT,begin_attempt,read_json,write_json

FREEZE=ROOT/'research/external_holdout_v1_freeze.json'
VERSION='holdout_v1'
MODELS=['deepseek-ai/DeepSeek-V3.2','Qwen/Qwen3.5-122B-A10B']


def now():return datetime.now(timezone.utc).isoformat()


def cache_path(dataset,model,stage,inp,repeat):
    identifier=inp['window_id'] if dataset=='polyie' else inp['id']
    name=hashlib.sha256(identifier.encode()).hexdigest()[:20]+'.json'
    return ROOT/f'results/{dataset}/cache_{VERSION}/{model.replace("/","__")}/test/repeat{repeat}/{stage}/{name}'


def validate_setting(dataset,model):
    if not FREEZE.exists():raise RuntimeError('Held-out inference disabled: separately verified formal protocol freeze is absent.')
    freeze=read_json(FREEZE)
    if model not in freeze['models'] or dataset not in freeze['datasets']:raise ValueError('Undeclared held-out setting.')
    for name,sha in freeze['file_sha256'].items():
        if hashlib.sha256((ROOT/name).read_bytes()).hexdigest()!=sha:raise ValueError('Frozen held-out file changed: '+name)
    selection_path=ROOT/'research/development_selections'/(dataset+'__'+model.replace('/','__')+'.json')
    if not selection_path.exists():raise RuntimeError('Held-out inference disabled: complete pooled three-repeat development selection is absent.')
    selection=read_json(selection_path)
    if selection['dataset']!=dataset or selection['model']!=model or selection['development_repeat_ids']!=[0,1,2]:
        raise ValueError('Wrong/incomplete development selection.')
    if selection['selection_code_sha256']!=hashlib.sha256((ROOT/'src/pool_development_selection.py').read_bytes()).hexdigest():
        raise ValueError('Development selection algorithm changed after selection.')
    for name,sha in selection['input_sha256'].items():
        if hashlib.sha256((ROOT/name).read_bytes()).hexdigest()!=sha:raise ValueError('Development selection evidence changed: '+name)
    return freeze,selection_path,selection


def run_stage(dataset,model,stage,repeat,inputs,task,retriever,workers):
    from siliconflow_rate_call import call
    attempt=begin_attempt(dataset+'_'+VERSION+'_'+stage+f'_repeat{repeat}',model,'test')
    builder=polyie_prompt if dataset=='polyie' else mulms_prompt
    def work(inp):
        path=cache_path(dataset,model,stage,inp,repeat)
        if path.exists():return read_json(path)
        if dataset=='polyie' and not inp['call_needed']:
            out={'relations':[],'source_only_empty_role_shortcut':True,'generation_protocol':VERSION};write_json(path,out);return out
        def get(parent):
            p=cache_path(dataset,model,parent,inp,repeat)
            if not p.exists():raise RuntimeError('Missing held-out parent cache: '+str(p))
            output=read_json(p)
            if output.get('error'):raise RuntimeError('Failed parent cannot be used in an allegedly complete held-out arm.')
            return output
        prompt,old,shortcut=builder(task,inp,stage,None if stage=='baseline' else retriever,get)
        if shortcut is not None:
            shortcut['generation_protocol']=VERSION;write_json(path,shortcut);return shortcut
        out=call(model,prompt,path)
        if out.get('error') and old is not None and stage!='verify_edits':
            out['relations']=old['relations'];out['fallback_to_parent']=True
        out['generation_protocol']=VERSION;write_json(path,out)
        return out
    failed=0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for index,future in enumerate(as_completed([pool.submit(work,inp) for inp in inputs]),1):
            failed+=bool(future.result().get('error'))
            if index%20==0 or index==len(inputs):
                print(json.dumps({'dataset':dataset,'model':model,'repeat':repeat,'stage':stage,
                                  'completed_items':index,'scope_items':len(inputs),'failed_items':failed}),flush=True)
    return failed


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--dataset',choices=['polyie','mulms'],required=True)
    ap.add_argument('--model',choices=MODELS,required=True);ap.add_argument('--workers',type=int,default=2)
    ap.add_argument('--check-only',action='store_true');args=ap.parse_args()
    freeze,selection_path,selection=validate_setting(args.dataset,args.model)
    if args.check_only:
        print(json.dumps({'status':'protocol_and_selection_verified','test_inference_performed':False,
                          'threshold':selection['threshold'],'control':selection['strongest_actual_three_call_control']}));return
    for key in ['FEEDBACK_API_TOP_P','FEEDBACK_API_TOP_K','FEEDBACK_API_MIN_P']:os.environ.pop(key,None)
    os.environ.update({k:str(v) for k,v in freeze['common_environment'].items()})
    os.environ.update({k:str(v) for k,v in freeze['model_profiles'][args.model].items()})
    os.environ['SILICONFLOW_KEY_FILE']=str(Path.home()/'.siliconflow_feedback_key')
    os.environ['POLYIE_PROTOCOL_VERSION']=freeze['source_development_versions']['polyie']
    os.environ['MULMS_PROTOCOL_VERSION']=freeze['source_development_versions']['mulms']
    task=importlib.import_module(args.dataset+'_experiment')
    if task.VERSION!=freeze['source_development_versions'][args.dataset]:raise ValueError('Wrong common source prompt version.')
    source_path=task.DATA/'test_inputs.json';inputs=read_json(source_path)
    expected=freeze['source_scopes'][args.dataset]
    if len(inputs)!=expected['items'] or sorted({inp['doc_key'] for inp in inputs})!=expected['document_ids']:
        raise ValueError('Held-out source scope differs from the pre-inference freeze.')
    input_keys={'polyie':{'doc_key','window_id','start','end','text','tokens','entities','call_needed'},
                'mulms':{'id','doc_key','text'}}[args.dataset]
    if any(set(inp)!=input_keys for inp in inputs):raise ValueError('Unexpected annotation field in held-out generation input.')
    retriever=task.Retriever();name=args.dataset+'__'+args.model.replace('/','__')
    state_path=ROOT/'logs/holdout_jobs'/(name+'.json');state_path.parent.mkdir(exist_ok=True)
    frozen={'dataset':args.dataset,'model':args.model,'version':VERSION,'pid':os.getpid(),
      'started_at_utc':now(),'status':'running','completed_stages':[],
      'source_items':len(inputs),'document_ids':expected['document_ids'],
      'planned_repeat_ids':[0,1,2],'selection_sha256':hashlib.sha256(selection_path.read_bytes()).hexdigest(),
      'protocol_freeze_sha256':hashlib.sha256(FREEZE.read_bytes()).hexdigest(),
      'threshold':selection['threshold'],'three_call_control':selection['strongest_actual_three_call_control'],
      'test_targets_loaded':False}
    if state_path.exists():
        old=read_json(state_path)
        try:os.kill(old['pid'],0)
        except ProcessLookupError:pass
        else:raise RuntimeError('Held-out job already alive; do not duplicate.')
        if old['selection_sha256']!=frozen['selection_sha256']:raise ValueError('Cannot change selection after held-out inference.')
        archive=state_path.with_name(name+'__previous_'+str(old['pid'])+'.json')
        if not archive.exists():archive.write_text(state_path.read_text())
    write_json(state_path,frozen)
    try:
        for repeat in [0,1,2]:
            for stage in task.GENERATED:
                frozen.update(current_repeat=repeat,current_stage=stage,stage_started_at_utc=now());write_json(state_path,frozen)
                failed=run_stage(args.dataset,args.model,stage,repeat,inputs,task,retriever,args.workers)
                frozen['completed_stages'].append({'repeat':repeat,'stage':stage,'failed_items':failed,'ended_at_utc':now()})
                write_json(state_path,frozen)
                if failed:raise RuntimeError('Explicit held-out failure; dependent stages stop and raw evidence remains.')
        frozen.update(status='completed',completed_at_utc=now());write_json(state_path,frozen)
    except BaseException as exc:
        frozen.update(status='stopped_explicit_failure',error=str(exc),stopped_at_utc=now());write_json(state_path,frozen);raise


if __name__=='__main__':main()
