"""Synthetic pipeline check, not scientific results and not billed inference.

Verify target loading happens after all source pools, frozen dev control is retained
when another test control is stronger, repeated generations remain TWO paper units,
and singleton matched pools give exactly identical random and semantic graphs.
"""
from contextlib import redirect_stdout
import hashlib,importlib,io,json
from pathlib import Path
import tempfile
from unittest.mock import patch

import external_holdout_analysis as analysis
from pilot import read_json,write_json

for dataset in ['polyie','mulms']:
 task=importlib.import_module(dataset+'_experiment')
 with tempfile.TemporaryDirectory() as directory:
  root=Path(directory);data=root/'data';data.mkdir();inputs=[];labels={};documents=[]
  for doc in ['paperA','paperB']:
   if dataset=='polyie':
    text='polymer modulus 12MPa 300K';tokens=text.split()
    ents=[{'id':'e'+str(i),'name':tokens[i],'type':kind,'start':i,'end':i+1}
          for i,kind in enumerate(['CN','PN','PV','Condition'])]
    inp={'doc_key':doc,'window_id':doc+'|0','start':0,'end':4,'text':text,'tokens':tokens,'entities':ents,'call_needed':True}
    good={'CN':['e0'],'PN':['e1'],'PV':['e2'],'Condition':['e3'],'evidence':text}
    wrong=dict(good,Condition=[])
    labels[doc]={'eligible':[good],'all_valid':[good],'schema_excluded':[]}
    documents.append({'doc_key':doc,'entities':ents})
   else:
    text='Sample shows hardness 12 MPa at 300 K.'
    inp={'id':doc+'|0','doc_key':doc,'text':text}
    good={'h':{'text':'shows','occurrence':0,'type':'MEASUREMENT'},
          't':{'text':'hardness','occurrence':0,'type':'PROPERTY'},'r':'measuresProperty','evidence':'shows hardness'}
    wrong=dict(good,r='usedAs');labels[inp['id']]=[good]
   inputs.append(inp)
  write_json(data/'test_inputs.json',inputs);write_json(data/'test_gold.json',labels)
  if dataset=='polyie':write_json(data/'test_documents.json',documents)
  def cache_path(dataset,model,stage,inp,repeat):
   return root/'cache'/str(repeat)/stage/(inp['doc_key']+'.json')
  for repeat in [0,1,2]:
   for inp in inputs:
    text=inp['text']
    if dataset=='polyie':
     good={'CN':['e0'],'PN':['e1'],'PV':['e2'],'Condition':['e3'],'evidence':text};wrong=dict(good,Condition=[])
    else:
     good={'h':{'text':'shows','occurrence':0,'type':'MEASUREMENT'},'t':{'text':'hardness','occurrence':0,'type':'PROPERTY'},'r':'measuresProperty','evidence':'shows hardness'};wrong=dict(good,r='usedAs')
    old={'relations':[wrong]};new={'relations':[good]}
    edits=task.edits_for(inp,old,new)
    verifier={'relations':[{'edit_id':e['edit_id'],'judgment':'supported' if e['operation']=='add' else 'unsupported',
                            'confidence':5,'evidence':text} for e in edits]}
    for stage in task.GENERATED:
     output=verifier if stage=='verify_edits' else new if stage in ['targeted','source_confirmation'] else old
     write_json(cache_path(dataset,'fixture-model',stage,inp,repeat),output)
  selection={'threshold':5,'strongest_actual_three_call_control':'generic_twice_clean'}
  selection_path=root/'synthetic_selection.json';write_json(selection_path,selection)
  state={'status':'completed','planned_repeat_ids':[0,1,2],
         'selection_sha256':hashlib.sha256(selection_path.read_bytes()).hexdigest(),
         'protocol_freeze_sha256':'synthetic fixture, no real protocol or evidence'}
  write_json(root/'logs/holdout_jobs'/f'{dataset}__fixture-model.json',state)
  made_pools=[0];original_pools=analysis.source_edit_pools
  def count_pools(*args,**kwargs):made_pools[0]+=1;return original_pools(*args,**kwargs)
  def guarded_read(path):
   if Path(path).name=='test_gold.json':assert made_pools[0]==6,'Labels loaded before all repeat/paper source pools were fixed'
   return read_json(path)
  with patch.object(analysis,'ROOT',root),patch.object(task,'DATA',data), \
       patch.object(analysis,'cache_path',cache_path),patch.object(analysis,'read_json',guarded_read), \
       patch.object(analysis,'source_edit_pools',count_pools), \
       patch.object(analysis,'validate_setting',lambda *_:({'source_development_versions':{'polyie':'development_v3','mulms':'development_v2'}},selection_path,selection)), \
       redirect_stdout(io.StringIO()):
   result=analysis.analyze(dataset,'fixture-model')
  assert result['stages']['semantic_clean']['f1']==1
  assert result['stages']['source_confirmation_clean']['f1']==1
  assert result['stages']['generic_twice_clean']['f1']==0
  primary=result['primary_contrasts']['semantic_clean vs generic_twice_clean']
  assert primary['permutation_units']==2 and primary['enumerated_assignments']==4
  assert primary['raw_exact_p']==.5 and primary['setting_family_holm_p']==1
  assert primary['delta_f1_points']==100
  assert result['primary_contrasts']['semantic_clean vs budget_random_clean_mean']['raw_exact_p']==1
  assert result['stages']['budget_random_clean_mean']['f1']==1
  assert result['stages']['exact_uniform_budget_random_secondary']['f1']==1
  assert all(b['random_identity_choice_bits']==0 for b in result['per_repeat_paper_source_only_budgets'].values())
  assert result['fixed_100_seed_joint_repeat_tp_variance_descriptive']==0
  assert len(result['per_repeat_stage_metrics'])==3
  print('PASS',dataset,': source-before-gold, actual selected-control retention, strict full-record correctness, three repeats clustered into two papers, 100 singleton random pools and analytical expectation')
