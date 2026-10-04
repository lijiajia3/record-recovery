"""Expose source-only globally eligible identities for independent diagnostics.

This independently reconstructs the declared pools without gold or performance.
It must match the frozen document_controls pool cardinalities and strata.
"""
from edit_controls import eligible_edits


def source_edit_pools(items,get_outputs,threshold,*,adapter_for,global_key,clean_old=True):
    old_graph={};copies={};additions={};deletions={}
    for index,inp in enumerate(items):
        old,proposal,verified=get_outputs(inp)
        adapter=adapter_for(inp);canonical,is_valid,*_=adapter
        for record in old['relations']:
            if clean_old and not is_valid(canonical(record)):continue
            key=global_key(inp,record);old_graph[key]=record;copies.setdefault(key,set()).add(index)
        for edit in eligible_edits(inp,old,proposal,verified,threshold,adapter=adapter):
            key=global_key(inp,edit['record']);score=edit['confidence']
            if edit['op']=='add':
                additions[key]=max(score,additions.get(key,0))
            else:
                per_copy=deletions.setdefault(key,{})
                per_copy[index]=max(score,per_copy.get(index,0))
    additions={key:score for key,score in additions.items() if key not in old_graph}
    deletions={key:min(scores[index] for index in copies[key])
               for key,scores in deletions.items() if key in old_graph and copies[key]<=scores.keys()}
    add_strata={str(score):{key for key,value in additions.items() if value==score}
                for score in range(threshold,6)}
    delete_strata={str(score):{key for key,value in deletions.items() if value==score}
                   for score in range(threshold,6)}
    return {'old':set(old_graph),'add_strata':add_strata,'delete_strata':delete_strata}

