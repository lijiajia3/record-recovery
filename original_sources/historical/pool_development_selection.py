#!/usr/bin/env python3
"""Select declared settings only after three whole-development suites complete.

This reads development scores only. It has no test label loader or model client.
Every repeat must have the full same document set and complete successful stages.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from holdout_statistics import f1_counts

ROOT=Path(__file__).resolve().parents[1]


def pool_repeat_summaries(summaries,expected_docs,*,model,dataset,version):
    if [s['repeat'] for s in summaries]!=[0,1,2]:
        raise ValueError('All three predeclared repeat IDs must appear exactly once in order.')
    expected_docs=set(expected_docs);names=set(summaries[0]['per_document_counts'])
    expected_generated={'baseline','retrieval','generic','targeted','generic_twice','source_confirmation','verify_edits'}
    required_scored={name+'_clean' for name in expected_generated-{'verify_edits'}}
    required_scored|={name+str(t) for t in [3,4,5] for name in ['semantic_clean_','quote_edits_clean_']}
    required_scored|={f'budget_random_clean_{t}_mean' for t in [3,4,5]}
    if not required_scored<=names:raise ValueError('Required method/control scores are missing.')
    source_item_counts=[]
    first_targets={doc:sum(summaries[0]['per_document_counts']['retrieval_clean'][doc][i] for i in [0,2]) for doc in expected_docs}
    for summary in summaries:
        if (summary['model'],summary['dataset'],summary['generation_protocol'])!=(model,dataset,version):
            raise ValueError('Wrong setting/version; cannot mix earlier exploratory methods.')
        if set(summary['document_ids'])!=expected_docs or summary['n_documents']!=len(expected_docs):
            raise ValueError('A repeat omits or adds source papers.')
        if set(summary['per_document_counts'])!=names:raise ValueError('Stage sets differ across repeats.')
        source_item_counts.append(summary['n_source_items'])
        if set(summary['generation_stage_accounting'])!=expected_generated:
            raise ValueError('All seven generated stages must be accounted for.')
        if any(v['failed_items'] for v in summary['generation_stage_accounting'].values()):
            raise ValueError('A full successful suite is required before selection.')
        for counts in summary['per_document_counts'].values():
            if set(counts)!=expected_docs:raise ValueError('Stage omits source papers.')
            for doc,row in counts.items():
                values=np.asarray(row,dtype=float)
                if values.shape!=(3,) or not np.isfinite(values).all() or (values<0).any():
                    raise ValueError('Invalid TP/FP/FN vector.')
                expected=summary['per_document_counts']['retrieval_clean'][doc]
                if not np.isclose(values[0]+values[2],first_targets[doc]):
                    raise ValueError('Target count changed between repetitions.')
                if not np.isclose(values[0]+values[2],expected[0]+expected[2]):
                    raise ValueError('Target count changed between methods.')
    if len(set(source_item_counts))!=1:raise ValueError('Repeat source scope changed.')
    cells={name:np.array([[sum(s['per_document_counts'][name][doc][i] for s in summaries)
                           for i in range(3)] for doc in sorted(expected_docs)],dtype=float)
           for name in names}
    stages={}
    for name,values in cells.items():
        total=values.sum(axis=0);entry={'pooled_tp_fp_fn':total.tolist(),'pooled_micro_f1':float(f1_counts(total))}
        if name.startswith('semantic_clean_'):
            edits={key:sum(s['stages'][name]['edits_vs_clean_retrieval'][key] for s in summaries)
                   for key in ['correct_added','wrong_added','correct_deleted','wrong_deleted']}
            entry['edits_vs_clean_retrieval']=edits
        stages[name]=entry
    def threshold_key(threshold):
        entry=stages[f'semantic_clean_{threshold}'];edits=entry['edits_vs_clean_retrieval']
        return (entry['pooled_micro_f1'],-edits['wrong_added']-edits['correct_deleted'],
                -sum(edits.values()),threshold)
    threshold=max([3,4,5],key=threshold_key)
    control=max(['generic_twice_clean','source_confirmation_clean'],
                key=lambda name:(stages[name]['pooled_micro_f1'],name))
    return {'dataset':dataset,'model':model,'generation_protocol':version,
        'development_repeat_ids':[0,1,2],'source_document_ids':sorted(expected_docs),
        'source_items_per_repeat':source_item_counts[0],
        'independent_units_are_source_papers':True,'repeats_sum_within_paper':True,
        'threshold':threshold,'strongest_actual_three_call_control':control,
        'selection_rule':'Pooled repeat TP/FP/FN; max F1, fewer wrong additions+correct deletions, fewer edits, higher threshold. Control max pooled F1 with lexicographically larger name tie.',
        'all_pooled_stages':stages,
        'per_source_paper_counts_summed_across_repeats':{name:{doc:values[i].tolist()
              for i,doc in enumerate(sorted(expected_docs))} for name,values in cells.items()}}


def select(dataset,model):
    version={'polyie':'development_v3','mulms':'development_v2'}[dataset]
    source=ROOT/f'data/{dataset}/dev_inputs.json';inputs=json.loads(source.read_text())
    ids={x['doc_key'] for x in inputs};tag=model.replace('/','__')
    suffix='subset0' if dataset=='polyie' else 'perdoc0'
    paths=[ROOT/f'results/{dataset}/{tag}__{version}__repeat{repeat}_{suffix}_edit_controls.json' for repeat in [0,1,2]]
    missing=[str(path.relative_to(ROOT)) for path in paths if not path.exists()]
    if missing:
        print(json.dumps({'status':'not_ready','selection_written':False,'missing_complete_full_repeat_analyses':missing},indent=2))
        return False
    summaries=[json.loads(path.read_text()) for path in paths]
    if any(s['n_source_items']!=len(inputs) for s in summaries):raise ValueError('Full development scope required.')
    result=pool_repeat_summaries(summaries,ids,model=model,dataset=dataset,version=version)
    result['input_sha256']={str(path.relative_to(ROOT)):hashlib.sha256(path.read_bytes()).hexdigest()
                            for path in paths+[source]}
    result['selection_code_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    path=ROOT/'research/development_selections'/(dataset+'__'+tag+'.json');path.parent.mkdir(exist_ok=True)
    if path.exists():
        old=json.loads(path.read_text())
        if old!=result:raise ValueError('Previously frozen selection changed; do not silently overwrite.')
    else:
        path.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'status':'ready','path':str(path.relative_to(ROOT)),
                      'threshold':result['threshold'],'control':result['strongest_actual_three_call_control']},indent=2))
    return True


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--dataset',choices=['polyie','mulms'],required=True)
    ap.add_argument('--model',required=True);args=ap.parse_args();select(args.dataset,args.model)
