"""Capture original development prompt bytes and compare future holdout builders.

Offline source-only test, actual training retriever, invented parent predictions.
Reading any dev/test gold file or making any API request is forbidden here.
"""
from contextlib import redirect_stdout
import hashlib
import importlib
import io,json
from pathlib import Path
import tempfile
from unittest.mock import patch
from external_holdout_prompts import polyie_prompt,mulms_prompt
from pilot import ROOT,read_json,write_json

report={'api_calls':False,'gold_targets_read':False,'stage_prompt_sha256':{},'builders_identical_to_original_source_prompts':True}
for dataset,builder in [('polyie',polyie_prompt),('mulms',mulms_prompt)]:
    task=importlib.import_module(dataset+'_experiment');original_read=task.read_json
    items=original_read(task.DATA/'dev_inputs.json')
    inp=next(x for x in items if x.get('call_needed',True))
    retriever=task.Retriever()
    if dataset=='polyie':
        em=task.entity_map(inp)
        roles={role:[key for key,value in em.items() if value['type']==role] for role in task.ROLES}
        relation={role:roles[role][:1] if role!='Condition' else [] for role in task.ROLES}
        relation['evidence']='invented parent record for offline prompt-equivalence check'
    else:
        words=inp['text'].split();head=words[0];tail=words[-1]
        relation={'h':{'text':head,'type':'MAT','occurrence':0},
                  't':{'text':tail,'type':'PROPERTY','occurrence':0},'r':'usedAs',
                  'evidence':'invented parent record for offline prompt-equivalence check'}
    parents={stage:{'relations':[relation]} for stage in task.GENERATED}
    parents['targeted']={'relations':[]} # ensures verifier sees a deletion candidate
    with tempfile.TemporaryDirectory() as directory:
        base=Path(directory)
        def path_for(model,split,stage,inp,repeat):return base/(stage+'.json')
        def guarded_read(path):
            path=Path(path)
            if 'gold' in path.name:raise AssertionError('Gold-file read in prompt construction: '+str(path))
            if path==task.DATA/'dev_inputs.json':return [inp]
            return original_read(path)
        for stage in task.GENERATED:
            for name in task.GENERATED:
                p=path_for('offline','dev',name,inp,0)
                if p.exists():p.unlink()
                if name!=stage:write_json(p,parents[name])
            captured=[]
            def fake_call(model,prompt,path):captured.append(prompt);return {'relations':[]}
            with patch.object(task,'read_json',guarded_read),patch.object(task,'path_for',path_for), \
                 patch.object(task,'Retriever',lambda:retriever),patch.object(task,'call',fake_call), \
                 patch.object(task,'begin_attempt',lambda *_:0),patch.object(task,'append',lambda *_:None), \
                 redirect_stdout(io.StringIO()):
                if dataset=='polyie':task.run_stage('offline','dev',stage,1,0,0)
                else:task.run_stage('offline',stage,1,0,0)
            expected,_,shortcut=builder(task,inp,stage,None if stage=='baseline' else retriever,
                                       lambda parent:parents[parent])
            assert shortcut is None
            assert len(captured)==1 and expected==captured[0],(dataset,stage)
            report['stage_prompt_sha256'][dataset+'|'+stage]=hashlib.sha256(expected.encode()).hexdigest()
        # Both original and new builders must keep the same no-edit verifier shortcut.
        _,old,shortcut=builder(task,inp,'verify_edits',retriever,lambda _:parents['retrieval'])
        assert shortcut=={'relations':[],'no_proposed_edits':True}
(ROOT/'research/holdout_prompt_equivalence_checks.json').write_text(json.dumps(report,indent=2)+'\n')
print('PASS: all14 source/training/parent prompt strings are byte-identical; no gold read or API call; same empty-edit shortcut')
