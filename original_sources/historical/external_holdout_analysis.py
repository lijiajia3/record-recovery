#!/usr/bin/env python3
"""Whole-source-paper analysis of complete, separately frozen holdout suites.

All method graphs/random identities are fixed before target loading. Three repeat
counts sum within a paper. No threshold/control selection or model call occurs.
"""
import argparse
from collections import defaultdict
import hashlib,importlib,json,os
from pathlib import Path

import numpy as np
from edit_controls import document_controls
from holdout_statistics import exact_paired_exchange,paired_paper_bootstrap,holm_adjust,f1_counts,exact_randomized_expectation
from source_edit_pools import source_edit_pools
from external_holdout_generation import cache_path,validate_setting,VERSION
from pilot import ROOT,read_json,write_json

SEEDS=list(range(20261003,20261103))


def cell(prediction,gold):
    prediction,gold=set(prediction),set(gold)
    return np.array([len(prediction&gold),len(prediction-gold),len(gold-prediction)],dtype=float)


def metrics(counts):
    tp,fp,fn=np.asarray(counts,dtype=float)
    return {'tp':float(tp),'fp':float(fp),'fn':float(fn),'f1':float(f1_counts(counts)),
            'precision':float(tp/(tp+fp)) if tp+fp else 0.,'recall':float(tp/(tp+fn)) if tp+fn else 0.}


def analyze(dataset,model):
    freeze,selection_path,selection=validate_setting(dataset,model)
    state_path=ROOT/'logs/holdout_jobs'/(dataset+'__'+model.replace('/','__')+'.json')
    if not state_path.exists():raise RuntimeError('No held-out job; no predictions or target loading allowed.')
    state=read_json(state_path)
    if state['status']!='completed' or state['planned_repeat_ids']!=[0,1,2]:
        raise RuntimeError('All three full held-out repeats must complete before analysis.')
    if state['selection_sha256']!=hashlib.sha256(selection_path.read_bytes()).hexdigest():
        raise ValueError('Development selection changed after first held-out request.')
    os.environ['POLYIE_PROTOCOL_VERSION']=freeze['source_development_versions']['polyie']
    os.environ['MULMS_PROTOCOL_VERSION']=freeze['source_development_versions']['mulms']
    task=importlib.import_module(dataset+'_experiment');inputs=read_json(task.DATA/'test_inputs.json')
    if dataset=='polyie':
        def adapter_for(inp):
            em=task.entity_map(inp)
            return lambda r:task.canonical(r,em),task.valid_key,task.edits_for,task.quote_valid,'group',inp['window_id']
        def key(inp,r):return task.canonical(r,task.entity_map(inp))
    else:
        def adapter_for(inp):
            return lambda r:task.canonical(r,inp['text']),task.valid_key,task.edits_for,task.quote_valid,'edge',inp['id']
        def key(inp,r):return inp['id'],task.canonical(r,inp['text'])
    by_doc=defaultdict(list)
    for inp in inputs:by_doc[inp['doc_key']].append(inp)
    ids=sorted(by_doc);threshold=selection['threshold'];cache={};cache_hashes={}
    graphs={};random_graphs={};pools={};budgets={};accounting={}
    for repeat in [0,1,2]:
        for inp in inputs:
            for stage in task.GENERATED:
                path=cache_path(dataset,model,stage,inp,repeat)
                if not path.exists():raise FileNotFoundError('Missing complete held-out item: '+str(path))
                out=read_json(path)
                if out.get('error'):raise RuntimeError('Incomplete/failed held-out arm; cannot silently score as empty.')
                cache[repeat,adapter_for(inp)[-1],stage]=out
                cache_hashes[str(path.relative_to(ROOT))]=hashlib.sha256(path.read_bytes()).hexdigest()
        for stage in task.GENERATED[:-1]:
            for mode in ['raw','clean']:
                name=stage+'_'+mode
                for doc,items in by_doc.items():
                    graph=set()
                    for inp in items:
                        canonical,valid,*_=adapter_for(inp)
                        output=cache[repeat,adapter_for(inp)[-1],stage]
                        graph|={key(inp,r) for r in output['relations'] if mode=='raw' or valid(canonical(r))}
                    graphs[repeat,name,doc]=graph
        for doc,items in by_doc.items():
            def get(inp):return tuple(cache[repeat,adapter_for(inp)[-1],stage] for stage in ['retrieval','targeted','verify_edits'])
            source_pools=source_edit_pools(items,get,threshold,adapter_for=adapter_for,global_key=key,clean_old=True)
            random=[]
            for seed in SEEDS:
                out=document_controls(items,get,threshold,adapter_for=adapter_for,global_key=key,
                                      doc_id=dataset+'|'+doc,seed=seed,clean_old=True)
                random.append(set(out['budget_random']))
            assert source_pools['old']==set(out['old'])
            for operation,field in [('add','add_strata'),('delete','delete_strata')]:
                for score,pool in source_pools[field].items():
                    assert len(pool)==out['eligible_counts_by_stratum'][operation][score]
            graphs[repeat,'semantic_clean',doc]=set(out['semantic'])
            graphs[repeat,'quote_edits_clean',doc]=set(out['quote_edits'])
            random_graphs[repeat,doc]=random;pools[repeat,doc]=source_pools
            budgets[str(repeat)+'|'+doc]={name:out[name] for name in ['matched_budgets','eligible_counts_by_stratum',
                         'random_identity_choice_bits','eligible_additions','eligible_deletions','semantic_additions','semantic_deletions']}
        accounting[str(repeat)]={}
        for stage in task.GENERATED:
            outputs=[cache[repeat,adapter_for(inp)[-1],stage] for inp in inputs]
            accounting[str(repeat)][stage]={
              'successful_completion_records':sum(bool(x.get('response_id')) for x in outputs),
              'no_edit_shortcuts':sum(bool(x.get('no_proposed_edits')) for x in outputs),
              'empty_role_shortcuts':sum(bool(x.get('source_only_empty_role_shortcut')) for x in outputs),
              'successful_completion_usage':{field:sum(x.get('usage',{}).get(field,0) for x in outputs)
                        for field in ['prompt_tokens','completion_tokens','total_tokens']},
              'note':'Failed/transient retries remain in separate whole-campaign request accounting; these are successful cached completion records only.'}
    # Graphs, pools and every randomized identity are fixed before this target read.
    labels=read_json(task.DATA/'test_gold.json')
    if dataset=='polyie':
        documents={x['doc_key']:x for x in read_json(task.DATA/'test_documents.json')}
        golds={doc:{task.canonical(r,task.entity_map(documents[doc])) for r in labels[doc]['eligible']} for doc in ids}
    else:golds={doc:{key(inp,r) for inp in items for r in labels[inp['id']]} for doc,items in by_doc.items()}
    names=sorted({name for _,name,_ in graphs});counts={};descriptions={};edits={};secondary={}
    for name in names:
        values=np.array([[sum(cell(graphs[repeat,name,doc],golds[doc])[i] for repeat in [0,1,2])
                          for i in range(3)] for doc in ids])
        counts[name]=values;descriptions[name]=metrics(values.sum(axis=0));edits[name]=defaultdict(int)
        for repeat in [0,1,2]:
            for doc in ids:
                old=graphs[repeat,'retrieval_clean',doc];pred=graphs[repeat,name,doc];gold=golds[doc]
                add,delete=pred-old,old-pred
                for field,value in [('correct_added',len(add&gold)),('wrong_added',len(add-gold)),
                                    ('correct_deleted',len(delete&gold)),('wrong_deleted',len(delete-gold))]:edits[name][field]+=value
        if dataset=='polyie':
            subgroup=lambda keys:{key for key in keys if task.valid_key(key) and bool(key[3])}
            secondary[name]={'condition_bearing_complete_groups':metrics(sum((cell(subgroup(graphs[r,name,d]),subgroup(golds[d]))
                                                      for r in [0,1,2] for d in ids),start=np.zeros(3))),
              'full_released_target_denominator_sensitivity':metrics(values.sum(axis=0)+np.array([0,0,3*sum(len(labels[d]['schema_excluded']) for d in ids)]))}
        else:
            secondary[name]={'complete_outgoing_measurement_root_graphs':metrics(sum((cell(task.observed_root_graphs(graphs[r,name,d]),task.observed_root_graphs(golds[d]))
                                                  for r in [0,1,2] for d in ids),start=np.zeros(3)))}
    random_cube=np.array([[sum((cell(random_graphs[r,d][index],golds[d]) for r in [0,1,2]),start=np.zeros(3))
                           for d in ids] for index in range(100)])
    counts['budget_random_clean_mean']=random_cube.mean(axis=0)
    descriptions['budget_random_clean_mean']=metrics(counts['budget_random_clean_mean'].sum(axis=0))
    assert np.allclose(random_cube[:,:,:2].sum(axis=2),counts['semantic_clean'][:,:2].sum(axis=1))
    seed_f1=[float(f1_counts(row.sum(axis=0))) for row in random_cube]
    assert abs(np.mean(seed_f1)-descriptions['budget_random_clean_mean']['f1'])<1e-12
    exact=[];quality_variance=0;ceilings=[];per_repeat_quality_variance={}
    for doc in ids:
        expected=np.zeros(3)
        for repeat in [0,1,2]:
            pool=pools[repeat,doc];budget=budgets[str(repeat)+'|'+doc]['matched_budgets']
            values=exact_randomized_expectation(pool['old'],golds[doc],pool['add_strata'],pool['delete_strata'],budget)
            expected+=np.array([values['expected_tp'],values['expected_fp'],values['expected_fn']]);quality_variance+=values['random_identity_tp_variance']
            per_repeat_quality_variance[str(repeat)+'|'+doc]=values['random_identity_tp_variance']
            adds=set().union(*pool['add_strata'].values());deletes=set().union(*pool['delete_strata'].values())
            oracle=(pool['old']-(deletes-golds[doc]))|(adds&golds[doc]);ceilings.append(cell(oracle,golds[doc]))
        exact.append(expected)
    counts['exact_uniform_budget_random_secondary']=np.array(exact)
    descriptions['exact_uniform_budget_random_secondary']=metrics(np.sum(exact,axis=0))
    contrasts={}
    for control in [selection['strongest_actual_three_call_control'],'budget_random_clean_mean']:
        values=exact_paired_exchange(counts['semantic_clean'],counts[control])
        values.update(paired_paper_bootstrap(counts['semantic_clean'],counts[control]))
        contrasts['semantic_clean vs '+control]=values
    corrected=holm_adjust([entry['raw_exact_p'] for entry in contrasts.values()])
    for entry,pvalue in zip(contrasts.values(),corrected):entry['setting_family_holm_p']=pvalue
    result={'dataset':dataset,'model':model,'generation_version':VERSION,'threshold_frozen_from_dev':threshold,
      'three_call_control_frozen_from_dev':selection['strongest_actual_three_call_control'],
      'source_papers':len(ids),'source_items_per_repeat':len(inputs),'repeat_ids':[0,1,2],
      'all_repeats_summed_within_paper':True,'test_targets_used_in_method_graphs':False,
      'raw_vs_common_structural_cleaning_reported':True,'stages':descriptions,
      'per_repeat_stage_metrics':{str(r):{name:metrics(sum((cell(graphs[r,name,d],golds[d]) for d in ids),start=np.zeros(3))) for name in names} for r in [0,1,2]},
      'edits_vs_clean_retrieval_summed_over_repeats':{name:dict(value) for name,value in edits.items()},
      'primary_contrasts':contrasts,'primary_family_scope':'two predeclared contrasts in this dataset/model; all eight-family sensitivity must be added when all four settings complete',
      'per_source_paper_count_vectors':{name:{doc:values[i].tolist() for i,doc in enumerate(ids)} for name,values in counts.items()},
      'all_fixed_seed_f1':seed_f1,'random_seeds':SEEDS,'per_repeat_paper_source_only_budgets':budgets,
      'secondary_descriptive_endpoints':secondary,
      'per_repeat_paper_conditional_tp_variance':per_repeat_quality_variance,
      'sum_per_repeat_tp_variances_descriptive_only':quality_variance,
      'joint_repeat_variance_is_not_inferred_from_that_sum':True,
      'fixed_100_seed_joint_repeat_tp_variance_descriptive':float(np.var(random_cube[:,:,0].sum(axis=1))),
      'random_repeat_dependence_note':'The same fixed random seeds are reused across repeats; all-seed pooled counts preserve their dependence. Per-repeat analytical variances cannot simply be added to obtain joint variance.',
      'gold_aware_unrestricted_pool_ceiling_nonimplementable':metrics(np.sum(ceilings,axis=0)),
      'oracle_is_method_or_ranking_performance':False,'successful_stage_accounting':accounting,
      'cache_sha256':cache_hashes,'selection_sha256':state['selection_sha256'],
      'protocol_freeze_sha256':state['protocol_freeze_sha256']}
    path=ROOT/f'results/{dataset}/{model.replace("/","__")}__{VERSION}__all3_summary.json'
    write_json(path,result);print(json.dumps({'path':str(path.relative_to(ROOT)),'primary_contrasts':contrasts},indent=2))
    return result


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--dataset',choices=['polyie','mulms'],required=True)
    ap.add_argument('--model',required=True);args=ap.parse_args();analyze(args.dataset,args.model)
