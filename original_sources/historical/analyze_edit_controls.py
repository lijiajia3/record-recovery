#!/usr/bin/env python3
"""Exploratory development analysis of predeclared offline edit controls."""
import argparse
from collections import defaultdict
import json
from pathlib import Path

import numpy as np
from edit_controls import document_controls, edit_identity
from pilot import ROOT, prf, read_json, write_json

SEEDS=list(range(20261003,20261103))


def analyze(dataset,model,repeat,subset,per_doc):
    assert dataset in {'polyie','mulms'}
    if dataset=='polyie':
        import polyie_experiment as task
        inputs=read_json(task.DATA/'dev_inputs.json')
        if subset:
            ids=sorted({x['doc_key'] for x in inputs},key=lambda x:task.digest('polyie-feasibility|'+x))[:subset]
            inputs=[x for x in inputs if x['doc_key'] in ids]
        def adapter_for(inp):
            em=task.entity_map(inp)
            return (lambda r:task.canonical(r,em),task.valid_key,task.edits_for,task.quote_valid,'group',inp['window_id'])
        def key(inp,r):return task.canonical(r,task.entity_map(inp))
        labels=read_json(task.DATA/'dev_gold.json')
        documents={d['doc_key']:d for d in read_json(task.DATA/'dev_documents.json')}
        def gold_for(doc,items):
            em=task.entity_map(documents[doc])
            return {task.canonical(r,em) for r in labels[doc]['eligible']}
    else:
        import mulms_experiment as task
        inputs=task.selected_inputs('dev',per_doc)
        def adapter_for(inp):
            return (lambda r:task.canonical(r,inp['text']),task.valid_key,task.edits_for,task.quote_valid,'edge',inp['id'])
        def key(inp,r):return inp['id'],task.canonical(r,inp['text'])
        labels=read_json(task.DATA/'dev_gold.json')
        def gold_for(doc,items):
            return {key(inp,r) for inp in items for r in labels[inp['id']]}
    by_doc=defaultdict(list);cache={}
    for inp in inputs:
        by_doc[inp['doc_key']].append(inp)
        item_id=adapter_for(inp)[-1]
        for stage in task.GENERATED:
            path=task.path_for(model,'dev',stage,inp,repeat)
            if not path.exists():raise FileNotFoundError('Incomplete development suite: '+str(path))
            cache[item_id,stage]=read_json(path)
    ids=sorted(by_doc)
    # Construct prediction controls before handing in any gold target set.
    predictions={};randomized={};budgets={};costs={}
    for stage in task.GENERATED[:-1]:
        predictions[stage+'_clean']={}
        for doc,items in by_doc.items():
            pred=set()
            for inp in items:
                adapter=adapter_for(inp);canonical,valid,*_=adapter
                pred|={key(inp,r) for r in cache[adapter[-1],stage]['relations'] if valid(canonical(r))}
            predictions[stage+'_clean'][doc]=pred
    for threshold in [3,4,5]:
        sem_name=f'semantic_clean_{threshold}';quote_name=f'quote_edits_clean_{threshold}'
        predictions[sem_name]={};predictions[quote_name]={};randomized[threshold]={};budgets[threshold]={}
        for doc,items in by_doc.items():
            def get(inp):
                item_id=adapter_for(inp)[-1]
                return tuple(cache[item_id,s] for s in ['retrieval','targeted','verify_edits'])
            distributions=[]
            for seed in SEEDS:
                out=document_controls(items,get,threshold,adapter_for=adapter_for,
                                      global_key=key,doc_id=dataset+'|'+doc,
                                      seed=seed,clean_old=True)
                distributions.append(set(out['budget_random']))
            predictions[sem_name][doc]=set(out['semantic'])
            predictions[quote_name][doc]=set(out['quote_edits'])
            randomized[threshold][doc]=distributions
            budgets[threshold][doc]={k:out[k] for k in ['matched_budgets','eligible_additions',
                                        'eligible_deletions','semantic_additions','semantic_deletions',
                                        'eligible_counts_by_stratum','random_identity_choice_bits']}
    # Gold is used only from this point to evaluate already selected graphs.
    golds={doc:gold_for(doc,by_doc[doc]) for doc in ids}
    def cell(p,g):return np.array([len(p&g),len(p-g),len(g-p)],dtype=float)
    result={'dataset':dataset,'model':model,'repeat':repeat,'generation_protocol':task.VERSION,
            'analysis':'exploratory_development; all random seeds retained',
            'n_documents':len(ids),'n_source_items':len(inputs),'document_ids':ids,
            'random_seeds':SEEDS,'matched_budget_scope':'Original source paper, unique final edits after window deduplication, operation and confidence stratum',
            'stages':{},'budgets':budgets}
    counts={}
    old=predictions['retrieval_clean']
    for name,preds in predictions.items():
        cells=np.array([cell(preds[d],golds[d]) for d in ids]);counts[name]=cells
        metrics=prf(cells.sum(axis=0));e=defaultdict(int)
        for d in ids:
            add,delete=preds[d]-old[d],old[d]-preds[d]
            e['correct_added']+=len(add&golds[d]);e['wrong_added']+=len(add-golds[d])
            e['correct_deleted']+=len(delete&golds[d]);e['wrong_deleted']+=len(delete-golds[d])
        metrics['edits_vs_clean_retrieval']=dict(e)
        base=prf(np.array([cell(old[d],golds[d]) for d in ids]).sum(axis=0))
        identity=edit_identity(base['tp'],base['fp'],base['fn'],e['correct_added'],e['wrong_added'],e['correct_deleted'],e['wrong_deleted'])
        assert abs(identity['f1_after']-metrics['f1'])<1e-12
        metrics['f1_edit_identity']=identity
        result['stages'][name]=metrics
    for threshold,distributions in randomized.items():
        cube=np.array([[cell(distributions[d][i],golds[d]) for d in ids] for i in range(len(SEEDS))])
        cells=cube.mean(axis=0);name=f'budget_random_clean_{threshold}_mean'
        counts[name]=cells;metrics=prf(cells.sum(axis=0))
        metrics.update({label:float(value) for label,value in zip(['tp','fp','fn'],cells.sum(axis=0))})
        metrics['counts_are_expected_over_fixed_seeds']=True
        sem=counts[f'semantic_clean_{threshold}']
        assert np.allclose(cube[:,:,:2].sum(axis=2),sem[:,:2].sum(axis=1)), 'Actual final record budget differs'
        seed_f1=[prf(row.sum(axis=0))['f1'] for row in cube]
        assert abs(np.mean(seed_f1)-metrics['f1'])<1e-12
        metrics['all_seed_f1']=seed_f1
        metrics['seed_f1_quantiles']=np.quantile(seed_f1,[.025,.5,.975]).tolist()
        result['stages'][name]=metrics
    def select(t):
        v=result['stages'][f'semantic_clean_{t}'];e=v['edits_vs_clean_retrieval']
        return v['f1'],-(e['wrong_added']+e['correct_deleted']),-sum(e.values()),t
    selected=max([3,4,5],key=select)
    # The string tie-break is deterministic, declared here before test scoring.
    strongest=max(['generic_twice_clean','source_confirmation_clean'],
                  key=lambda s:(result['stages'][s]['f1'],s))
    result['development_selected_clean_threshold']=selected
    result['development_strongest_three_call_control']=strongest
    result['descriptive_paired_contrasts']={}
    for b in [strongest,f'budget_random_clean_{selected}_mean']:
        a=f'semantic_clean_{selected}';ca,cb=counts[a],counts[b]
        rng=np.random.default_rng(20261003);samples=[]
        for _ in range(2000):
            idx=rng.integers(0,len(ids),len(ids))
            samples.append(100*(prf(ca[idx].sum(axis=0))['f1']-prf(cb[idx].sum(axis=0))['f1']))
        result['descriptive_paired_contrasts'][a+' vs '+b]={'delta_f1_points':100*(result['stages'][a]['f1']-result['stages'][b]['f1']),
            'paper_bootstrap_ci95':np.quantile(samples,[.025,.975]).tolist(),'confirmatory_test':False}
    for stage in task.GENERATED:
        outputs=[cache[adapter_for(inp)[-1],stage] for inp in inputs]
        usage={name:sum(int(x.get('usage',{}).get(name,0)) for x in outputs) for name in ['prompt_tokens','completion_tokens','total_tokens']}
        costs[stage]={'successful_paid_requests':sum(bool(x.get('response_id')) for x in outputs),
                      'failed_items':sum(bool(x.get('error')) for x in outputs),'reported_successful_usage':usage}
    result['generation_stage_accounting']=costs
    result['per_document_counts']={s:{d:c.tolist() for d,c in zip(ids,cells)} for s,cells in counts.items()}
    suffix=f'subset{subset}' if dataset=='polyie' else f'perdoc{per_doc}'
    path=ROOT/f'results/{dataset}/{model.replace("/","__")}__{task.VERSION}__repeat{repeat}_{suffix}_edit_controls.json'
    write_json(path,result)
    print(json.dumps({'path':str(path.relative_to(ROOT)),'selected_threshold':selected,'strongest_control':strongest,
                      'stages':{s:round(v['f1']*100,2) for s,v in result['stages'].items()},
                      'contrasts':result['descriptive_paired_contrasts']},indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--dataset',choices=['polyie','mulms'],required=True)
    p.add_argument('--model',default='deepseek-ai/DeepSeek-V3.2');p.add_argument('--repeat',type=int,default=0)
    p.add_argument('--subset',type=int,default=0);p.add_argument('--per-doc',type=int,default=4);a=p.parse_args()
    analyze(a.dataset,a.model,a.repeat,a.subset,a.per_doc)
