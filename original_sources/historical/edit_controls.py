"""Offline, matched-information controls for selective scientific edits.

Selection never reads gold. Document randomization holds the distinct additions,
deletions and confidence strata fixed after source-window deduplication.
"""
from collections import Counter
import hashlib
import math
import random


def eligible_edits(inp, old, proposal, verified, threshold, *, adapter):
    canonical,is_valid,edits_fn,quote_fn,field,input_id=adapter
    if verified.get('error'):
        return []
    candidates={e['edit_id']:e for e in edits_fn(inp,old,proposal)}
    frequency=Counter(v.get('edit_id') for v in verified['relations'] if isinstance(v,dict))
    eligible=[]
    for v in verified['relations']:
        if not isinstance(v,dict) or frequency[v.get('edit_id')]!=1:
            continue
        e=candidates.get(v.get('edit_id'));confidence=v.get('confidence')
        if e is None or type(confidence) is not int or not threshold<=confidence<=5:
            continue
        if v.get('judgment') not in {'supported','unsupported'}:
            continue  # Uncertainty and malformed judgments abstain in every arm.
        record=e[field]
        if not isinstance(record,dict):
            continue
        record=dict(record,evidence=v.get('evidence',''))
        if not is_valid(canonical(record)) or not quote_fn(inp,record,True):
            continue
        selected=(e['operation']=='add' and v['judgment']=='supported') or (
                  e['operation']=='delete' and v['judgment']=='unsupported')
        eligible.append({'id':e['edit_id'],'op':e['operation'],'record':record,'confidence':confidence,
                         'semantic_selected':selected})
    return eligible


def select_edits(inp, old, proposal, verified, threshold, *, adapter,
                 mode='semantic', seed=20261003, clean_old=False):
    """adapter=(canonical, is_valid, edits, quote_valid, record_field, input_id)."""
    canonical,is_valid,edits_fn,quote_fn,field,input_id=adapter
    accepted={canonical(r):r for r in old['relations']}
    if clean_old:
        accepted={k:r for k,r in accepted.items() if is_valid(k)}
    if verified.get('error'):
        return {'relations':list(accepted.values()),'verification_error':verified['error'],
                'fallback_to_parent':True}
    eligible=eligible_edits(inp,old,proposal,verified,threshold,adapter=adapter)
    chosen=[]
    if mode=='semantic':
        chosen=[e for e in eligible if e['semantic_selected']]
    elif mode=='quote_edits':
        chosen=eligible
    elif mode=='budget_random':
        for op in ['add','delete']:
            pool=sorted((e for e in eligible if e['op']==op),key=lambda e:e['id'])
            number=sum(e['semantic_selected'] for e in pool)
            local_seed=int(hashlib.sha256((str(seed)+'|'+input_id+'|'+op).encode()).hexdigest(),16)
            chosen.extend(random.Random(local_seed).sample(pool,number))
    else:
        raise ValueError(mode)
    for e in chosen:
        key=canonical(e['record'])
        if e['op']=='add':accepted[key]=e['record']
        else:accepted.pop(key,None)
    return {'relations':list(accepted.values()),'offline_control':mode,
            'structural_cleaning':clean_old,'threshold':threshold,
            'eligible_edits':len(eligible),'applied_edit_ids':[e['id'] for e in chosen],
            'selected_additions':sum(e['op']=='add' for e in chosen),
            'selected_deletions':sum(e['op']=='delete' for e in chosen)}


def document_controls(items, get_outputs, threshold, *, adapter_for, global_key,
                      doc_id, seed=20261003, clean_old=False):
    """Hold distinct graph edits fixed after overlapping-window deduplication.

    No gold argument exists. A deletion is structurally eligible only if every
    source window currently emitting that record supplies an eligible deletion.
    The confidence stratum is max eligible confidence for an addition and the
    min of per-copy maxima for deletion. Both ignore polarity. Randomization
    holds operation/count/confidence strata fixed within each source paper.
    """
    old_graph={};semantic_graph={};old_copies={};adds={};deletes={}
    for index,inp in enumerate(items):
        old,proposal,verified=get_outputs(inp)
        adapter=adapter_for(inp);canonical,is_valid,*_=adapter
        for r in old['relations']:
            if clean_old and not is_valid(canonical(r)):
                continue
            key=global_key(inp,r)
            old_graph[key]=r;old_copies.setdefault(key,set()).add(index)
        out=select_edits(inp,old,proposal,verified,threshold,adapter=adapter,clean_old=clean_old)
        for r in out['relations']:
            semantic_graph[global_key(inp,r)]=r
        for e in eligible_edits(inp,old,proposal,verified,threshold,adapter=adapter):
            key=global_key(inp,e['record']);score=e['confidence']
            if e['op']=='add':
                if key not in adds or score>adds[key]['confidence']:
                    adds[key]=dict(e,origin_input=adapter[-1])
            else:
                copies=deletes.setdefault(key,{})
                if index not in copies or score>copies[index]['confidence']:
                    copies[index]=dict(e,origin_input=adapter[-1])
    adds={k:e for k,e in adds.items() if k not in old_graph}
    deletes={k:dict(min((copies[i] for i in old_copies[k]),key=lambda e:e['confidence']))
             for k,copies in deletes.items() if k in old_graph and old_copies[k]<=copies.keys()}
    semantic_added=semantic_graph.keys()-old_graph.keys()
    semantic_deleted=old_graph.keys()-semantic_graph.keys()
    assert semantic_added<=adds.keys(),'Semantic addition lacks a structurally eligible source copy'
    assert semantic_deleted<=deletes.keys(),'Semantic deletion lacks evidence for every old copy'
    chosen={'add':set(),'delete':set()};budgets={};pools={};choice_bits=0.0
    for op,pool,selected in [('add',adds,semantic_added),('delete',deletes,semantic_deleted)]:
        budgets[op]={}
        pools[op]={}
        for score in range(threshold,6):
            keys=sorted((k for k,e in pool.items() if e['confidence']==score),key=repr)
            number=sum(pool[k]['confidence']==score for k in selected)
            budgets[op][str(score)]=number
            pools[op][str(score)]=len(keys)
            choice_bits+=math.log2(math.comb(len(keys),number))
            local_seed=int(hashlib.sha256((str(seed)+'|'+doc_id+'|'+op+'|'+str(score)).encode()).hexdigest(),16)
            chosen[op].update(random.Random(local_seed).sample(keys,number))
    randomized={k:r for k,r in old_graph.items() if k not in chosen['delete']}
    randomized.update({k:adds[k]['record'] for k in chosen['add']})
    quoted={k:r for k,r in old_graph.items() if k not in deletes}
    quoted.update({k:e['record'] for k,e in adds.items()})
    assert len(randomized.keys()-old_graph.keys())==len(semantic_added)
    assert len(old_graph.keys()-randomized.keys())==len(semantic_deleted)
    return {'old':old_graph,'semantic':semantic_graph,'quote_edits':quoted,
            'budget_random':randomized,'matched_budgets':budgets,
            'eligible_counts_by_stratum':pools,'random_identity_choice_bits':choice_bits,
            'eligible_additions':len(adds),'eligible_deletions':len(deletes),
            'semantic_additions':len(semantic_added),'semantic_deletions':len(semantic_deleted)}


def edit_identity(tp, fp, fn, correct_added, wrong_added, correct_deleted, wrong_deleted):
    """Exact F1 bookkeeping identity; no factual guarantee for an edit selector."""
    gold=tp+fn;predicted=tp+fp
    u=correct_added-correct_deleted
    v=correct_added+wrong_added-correct_deleted-wrong_deleted
    before=2*tp/(gold+predicted) if gold+predicted else 0
    after=2*(tp+u)/(gold+predicted+v) if gold+predicted+v else 0
    return {'delta_correct_records':u,'delta_predicted_records':v,
            'f1_before':before,'f1_after':after,'f1_difference':after-before,
            'gain_numerator':u*(gold+predicted)-tp*v}
