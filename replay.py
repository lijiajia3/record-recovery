"""Public saved-output semantic replay. No training, API or checkpoint loading.

This compact replay checks graph semantics and the original fixed statistics;
it does not repeat the historical checkpoint/cache/source-generation barrier.
That execution provenance is separate from the current public artifact manifest.
"""
from pathlib import Path
import argparse,json,hashlib,importlib.util,subprocess,sys
import numpy as np

def load(p,n):
 s=importlib.util.spec_from_file_location(n,p);m=importlib.util.module_from_spec(s);sys.modules[n]=m;s.loader.exec_module(m);return m

def read(r,n):return json.loads((r/n).read_bytes())
def verify(r):
 d=read(r,'PUBLIC_DATA_MANIFEST.json')
 for n,v in d['files'].items():
  p=r/n;h=hashlib.sha256()
  with p.open('rb') as f:
   for b in iter(lambda:f.read(1048576),b''):h.update(b)
  assert p.stat().st_size==v['size_bytes'] and h.hexdigest()==v['sha256'],n
 return len(d['files'])

def poly(r):
 m=load(r/'src/replay_polyie_neural_offline.py','poly_autonomous')
 dest='results/local_baseline/polyie_neural_family_v1';man=read(r,dest+'/generation_manifest.json');summary=read(r,dest+'/test_summary.json')
 docs=read(r,'data/polyie/test_documents.json');gold=read(r,'data/polyie/test_gold.json');entities={d['doc_key']:{e['id']:e for e in d['entities']} for d in docs};papers=man['paper_ids'];assert len(papers)==14
 vectors={};totals={}
 for arch in m.ARCH:
  pooled={p:dict.fromkeys(('tp','fp','fn'),0) for p in papers}
  for seed in m.SEEDS:
   graph=read(r,man['complete_graphs'][f'{arch}|{seed}']['path'])
   assert set(graph['graphs'])==set(papers)
   for p in papers:
    pred={m.canonical(x,entities[p]) for x in graph['graphs'][p]};g={m.canonical(x,entities[p]) for x in gold[p]['eligible']};c=m.count_sets(pred,g)
    m.assert_equal(c,summary['architecture_results'][arch]['all_seeds'][str(seed)]['covered_primary']['by_paper'][p],arch)
    for k in c:pooled[p][k]+=c[k]
  totals[arch]=m.summarize(pooled);m.assert_equal(totals[arch],summary['architecture_results'][arch]['pooled_primary'],arch)
  vectors[arch]=np.array([[pooled[p][k] for k in ('tp','fp','fn')] for p in papers])
 contrasts=[m.paired_statistics(vectors['typed'],vectors[c]) for c in ('mean','capacity_mean')]
 for x,p in zip(contrasts,m.holm([x['raw_exact_p'] for x in contrasts])):x['holm_p_two_planned_neural_contrasts']=p
 m.assert_equal(contrasts,summary['planned_contrasts'],'poly_contrasts')
 return {'papers':14,'graphs':9,'pooled':totals,'contrasts':contrasts}

def mu(r):
 m=load(r/'src/replay_mulms_neural_offline.py','mu_autonomous');dest=m.DEST;man=read(r,dest+'/generation_manifest.json');lock=read(r,m.LOCK);inputs=read(r,'data/mulms/test_inputs.json');gold,ner=m.targets(r,lock,inputs);papers=man['paper_ids'];assert len(papers)==7
 totals={};vectors={};contexts={}
 for a in (*m.ARCHITECTURES,*m.REFERENCES):
  if a in m.ARCHITECTURES:entries=man['relation_graphs'][a];expected=read(r,dest+'/test_summary.json')['architecture_results'][a]
  else:
   rd='results/local_baseline/mulms_'+a+'_reference_test_v1';entries=read(r,rd+'/generation_manifest.json')['graphs'];expected=read(r,rd+'/test_summary.json')
  seeds={};cx=[]
  for seed in m.SEEDS:
   g=read(r,entries[str(seed)]['path']);assert len(g['graphs'])==1114
   result=m.relation_counts(inputs,g['graphs'],gold,papers)
   # Compare every native paper and aggregate, not only manuscript totals.
   m.equal(result['by_paper'],expected['all_seeds'][str(seed)]['by_paper'],a)
   seeds[str(seed)]=result;cx.append(result['observed_measurement_root_exact_graph_descriptive'])
  pooled,vec=m.pooled_result(seeds,papers);m.equal(pooled['pooled_primary'],expected['pooled_primary'],a)
  totals[a]=pooled['pooled_primary'];vectors[a]=vec;contexts[a]=cx
 contrasts=[m.paired_statistics(vectors['typed'],vectors[c]) for c in ('mean','capacity_mean')]
 expected=read(r,dest+'/test_summary.json')['planned_contrasts']
 for x,p in zip(contrasts,m.holm([x['raw_exact_p'] for x in contrasts])):x['holm_p_two_planned_neural_contrasts']=p
 m.equal(contrasts,expected,'mu_contrasts')
 return {'papers':7,'relation_graphs':15,'pooled':totals,'complete_measurement_context_by_seed':contexts,'contrasts':contrasts}

def main():
 p=argparse.ArgumentParser();p.add_argument('--data',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();assert not a.out.exists(),'Use a new report directory';a.out.mkdir(parents=True)
 n=verify(a.data);result={'verified_files':n,'polyie':poly(a.data/'historical'),'mulms':mu(a.data/'historical'),'training_or_model_inference':False,'credential_access':False,'historical_full_barrier_repeated':False}
 subprocess.run([sys.executable,str(a.data/'expanded/src/replay_expanded_sccomics_saved_outputs_20261004.py'),'--root',str((a.data/'expanded').resolve()),'--out',str((a.out/'SC_replay.json').resolve())],check=True)
 subprocess.run([sys.executable,'REPRODUCE_OFFLINE.py'],cwd=a.data/'pilot',check=True)
 result['SC']=read(a.out,'SC_replay.json');result['pilot_13_arm_semantic_replay_passed']=True
 (a.out/'all_replays.json').write_text(json.dumps(result,indent=2)+'\n');print('PUBLIC_SAVED_OUTPUT_REPLAY_PASSED')
if __name__=='__main__':main()
