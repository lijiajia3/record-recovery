"""Synthetic scope/tie checks; not scientific data or a repeated-run significance test."""
from copy import deepcopy
from pool_development_selection import pool_repeat_summaries
generated=['baseline','retrieval','generic','targeted','generic_twice','source_confirmation','verify_edits']
names=[n+'_clean' for n in generated[:-1]]+[n+str(t) for t in [3,4,5] for n in ['semantic_clean_','quote_edits_clean_']]+[f'budget_random_clean_{t}_mean' for t in [3,4,5]]
summaries=[]
for repeat in [0,1,2]:
 summaries.append({'repeat':repeat,'model':'model','dataset':'polyie','generation_protocol':'development_v3',
  'document_ids':['paperA','paperB'],'n_documents':2,'n_source_items':6,
  'generation_stage_accounting':{n:{'failed_items':0} for n in generated},
  'per_document_counts':{name:{'paperA':[2,1,1],'paperB':[1,1,2]} for name in names},
  'stages':{name:{'edits_vs_clean_retrieval':{'correct_added':0,'wrong_added':0,'correct_deleted':0,'wrong_deleted':0}} for name in names}})
args={'model':'model','dataset':'polyie','version':'development_v3'}
r=pool_repeat_summaries(summaries,['paperA','paperB'],**args)
assert r['threshold']==5 and r['strongest_actual_three_call_control']=='source_confirmation_clean'
assert r['per_source_paper_counts_summed_across_repeats']['semantic_clean_5']['paperA']==[6,3,3]
for mode in ['missing_repeat','missing_paper','failed_stage','wrong_version']:
 changed=deepcopy(summaries)
 if mode=='missing_repeat':changed.pop()
 elif mode=='missing_paper':changed[1]['document_ids'].pop()
 elif mode=='failed_stage':changed[0]['generation_stage_accounting']['baseline']['failed_items']=1
 else:changed[1]['generation_protocol']='development_v2'
 try:pool_repeat_summaries(changed,['paperA','paperB'],**args)
 except ValueError:pass
 else:raise AssertionError(mode+' incorrectly accepted')
print('PASS: pooled counts, deterministic ties, all repeats/papers/successful stages and version separation')
