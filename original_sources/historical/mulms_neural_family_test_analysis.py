"""Annotation read only after the three-plus-nine complete source graph barrier."""
import json
from collections import defaultdict
import numpy as np
import pyarrow.parquet as pq
from mulms_neural_family import ROOT,FREEZE_REL,DEST_REL,SEEDS,ARCHITECTURES,sha,complete_family,span_key,TYPE_IDS,RELATION_IDS
from mulms_experiment import canonical,valid_key,observed_root_graphs
from holdout_statistics import f1_counts,exact_paired_exchange,paired_paper_bootstrap,holm_adjust
from polyie_local_baseline_training import write


def prf(c):
    tp,fp,fn=c['tp'],c['fp'],c['fn'];denominator=2*tp+fp+fn
    return dict(c,precision=tp/(tp+fp)if tp+fp else 0,recall=tp/(tp+fn)if tp+fn else 0,f1=2*tp/denominator if denominator else 0)


def total(rows):return prf({k:sum(c[k]for c in rows.values())for k in ['tp','fp','fn']})


def counts(predicted,gold):return {'tp':len(predicted&gold),'fp':len(predicted-gold),'fn':len(gold-predicted)}


def load_targets(freeze,inputs,root):
    # Caller must first validate every source graph/hash/checkpoint/selection.
    path=root/'data/mulms/test_gold.json';assert sha(path)==freeze['test_gold_sha256']
    edges=json.loads(path.read_text());ids={i['id']for i in inputs};assert set(edges)==ids
    source=root/'data/mulms/source/test.parquet';assert sha(source)==freeze['test_ner_source_sha256']
    rows=pq.read_table(source,columns=['doc_id','sentence','beginOffset','NER_labels']).to_pylist();ner={}
    lookup={i['id']:i for i in inputs}
    for row in rows:
        identity=row['doc_id']+'|'+str(row['beginOffset']);assert identity in lookup and identity not in ner
        assert row['sentence']==lookup[identity]['text']and row['doc_id']==lookup[identity]['doc_key']
        label=row['NER_labels'];assert len({len(v)for v in label.values()})==1;entities=[]
        for i in range(len(label['id'])):
            a,b,kind=int(label['begin'][i]),int(label['end'][i]),label['value'][i]
            assert kind in TYPE_IDS and 0<=a<b<=len(row['sentence'])and row['sentence'][a:b]==label['text'][i]
            entities.append({'start':a,'end':b,'type':kind,'text':label['text'][i]})
        assert len(entities)==len({span_key(e)for e in entities});ner[identity]=entities
    assert set(ner)==ids
    return edges,ner


def relation_counts(inputs,predictions,gold):
    papers=sorted({i['doc_key']for i in inputs});rows={p:{'tp':0,'fp':0,'fn':0}for p in papers}
    labels={r:{'tp':0,'fp':0,'fn':0}for r in RELATION_IDS};roots_pred=set();roots_gold=set()
    for inp in inputs:
        predicted={canonical(r,inp['text'])for r in predictions[inp['id']]}
        targets={canonical(r,inp['text'])for r in gold[inp['id']]};assert all(valid_key(k)for k in targets)
        c=counts(predicted,targets)
        for key in c:rows[inp['doc_key']][key]+=c[key]
        for label in RELATION_IDS:
            subset={canonical(r,inp['text'])for r in predictions[inp['id']]if r['r']==label}
            c=counts(subset,{k for k in targets if k[2]==label})
            for key in c:labels[label][key]+=c[key]
        roots_pred.update((inp['id'],k)for k in predicted);roots_gold.update((inp['id'],k)for k in targets)
    roots=prf(counts(observed_root_graphs(roots_pred),observed_root_graphs(roots_gold)))
    return {'by_paper':rows,'total':total(rows),'per_label':{k:prf(v)for k,v in labels.items()},
        'observed_measurement_root_exact_graph_descriptive':roots}


def analyze(root=ROOT):
    freeze,manifest,inputs,detectors,graphs=complete_family(root)
    edges,ner_gold=load_targets(freeze,inputs,root);papers=manifest['paper_ids']
    assert set(papers)=={i['doc_key']for i in inputs}and len(papers)==7
    detector_results={};pooled_ner={p:{'tp':0,'fp':0,'fn':0}for p in papers}
    for seed in SEEDS:
        rows={p:{'tp':0,'fp':0,'fn':0}for p in papers}
        for inp in inputs:
            c=counts({span_key(e)for e in detectors[seed]['graphs'][inp['id']]},{span_key(e)for e in ner_gold[inp['id']]})
            for k in c:rows[inp['doc_key']][k]+=c[k];pooled_ner[inp['doc_key']][k]+=c[k]
        detector_results[str(seed)]={'by_paper':rows,'total':total(rows)}
    results={};vectors={}
    for architecture in ARCHITECTURES:
        all_seeds={};pooled={p:{'tp':0,'fp':0,'fn':0}for p in papers}
        for seed in SEEDS:
            value=relation_counts(inputs,graphs[architecture,seed]['graphs'],edges);all_seeds[str(seed)]=value
            for p in papers:
                for k in pooled[p]:pooled[p][k]+=value['by_paper'][p][k]
        values=np.asarray([[pooled[p][k]for k in ['tp','fp','fn']]for p in papers],dtype=np.int64);vectors[architecture]=values
        rng=np.random.default_rng(20261003);indices=rng.integers(len(papers),size=(10000,len(papers)))
        ci=np.quantile(f1_counts(values[indices].sum(axis=1)),[.025,.975]).tolist()
        results[architecture]={'selected':freeze['relation_selections'][architecture]['chosen'],'all_seeds':all_seeds,
            'pooled_by_original_paper':pooled,'pooled_primary':total(pooled),'paper_bootstrap_95_f1_ci':ci,
            'pooled_per_label_descriptive':{r:prf({k:sum(all_seeds[str(s)]['per_label'][r][k]for s in SEEDS)for k in ['tp','fp','fn']})for r in RELATION_IDS}}
    contrasts=[]
    for control in ['mean','capacity_mean']:
        a,b=vectors['typed'],vectors[control]
        contrasts.append({'name':'typed_minus_'+control,**exact_paired_exchange(a,b),**paired_paper_bootstrap(a,b,resamples=10000,seed=20261003)})
    for row,adjusted in zip(contrasts,holm_adjust([r['raw_exact_p']for r in contrasts])):row['holm_p_two_planned_neural_contrasts']=adjusted
    report={'scope':'strict end-to-end directed typed-character relations; source-only predicted entities shared per seed',
        'test_papers':7,'test_sentences':1114,'paper_ids':papers,'architecture_results':results,'planned_contrasts':contrasts,
        'detector_descriptive':{'selected':freeze['detector_selection']['chosen'],'all_seeds':detector_results,'pooled_by_original_paper':pooled_ner,'pooled':total(pooled_ner)},
        'complete_detector_graph_barrier':3,'complete_relation_graph_barrier':9,'test_gold_loaded_only_after_all_source_graphs':True,
        'freeze_sha256':sha(root/FREEZE_REL),'generation_manifest_sha256':sha(root/DEST_REL/'generation_manifest.json'),
        'bootstrap_draws':10000,'bootstrap_seed':20261003,'seeds_are_independent_papers':False,
        'global_four_contrast_sensitivity_complete':False,'original_LLM_primary_family_changed':False}
    write(root/DEST_REL/'test_summary.json',report)
    print(json.dumps({'pooled_primary':{n:r['pooled_primary']for n,r in results.items()},'contrasts':contrasts},indent=2));return report


if __name__=='__main__':analyze()
