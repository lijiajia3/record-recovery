"""Nine-graph barrier then whole-paper scores and two predeclared contrasts."""
import json
import numpy as np
from polyie_local_adapter import ROOT,sha
from polyie_local_baseline_training import write
from polyie_neural_family import SEEDS,ARCHITECTURES,FREEZE_REL,DEST_REL,complete_family
from holdout_statistics import f1_counts,exact_paired_exchange,paired_paper_bootstrap,holm_adjust
import polyie_experiment as task


def prf(c):
    tp,fp,fn=c['tp'],c['fp'],c['fn']
    return dict(c,precision=tp/(tp+fp)if tp+fp else 0,recall=tp/(tp+fn)if tp+fn else 0,
        f1=2*tp/(2*tp+fp+fn)if 2*tp+fp+fn else 0)


def totals(rows):return prf({k:sum(c[k]for c in rows.values())for k in ['tp','fp','fn']})


def load_targets(freeze):
    # No caller may reach this function until all nine hashes pass.
    path=ROOT/'data/polyie/test_gold.json';assert sha(path)==freeze['test_gold_sha256']
    labels=json.loads(path.read_text())
    docs=json.loads((ROOT/'data/polyie/test_documents.json').read_text())
    return labels,{d['doc_key']:{e['id']:e for e in d['entities']}for d in docs}


def main():
    freeze,manifest,graphs=complete_family(ROOT)
    labels,em=load_targets(freeze);papers=manifest['paper_ids']
    assert set(em)==set(papers)and set(labels)==set(papers)
    results={};vectors={}
    for name in ARCHITECTURES:
        all_seeds={};pooled={doc:{'tp':0,'fp':0,'fn':0}for doc in papers}
        for seed in SEEDS:
            g=graphs[name,seed];primary={};full={};released={}
            for doc in papers:
                pred={task.canonical(r,em[doc])for r in g['graphs'][doc]}
                eligible={task.canonical(r,em[doc])for r in labels[doc]['eligible']}
                all_valid={task.canonical(r,em[doc])for r in labels[doc]['all_valid']}
                tp,fp,fn=task.counts_for(pred,eligible);primary[doc]={'tp':tp,'fp':fp,'fn':fn}
                tp,fp,fn=task.counts_for(pred,all_valid);full[doc]={'tp':tp,'fp':fp,'fn':fn}
                released_gold=all_valid|{('MALFORMED_RELEASED_GOLD',json.dumps(r,sort_keys=True))for r in labels[doc]['schema_excluded']}
                tp,fp,fn=task.counts_for(pred,released_gold);released[doc]={'tp':tp,'fp':fp,'fn':fn}
                for k in pooled[doc]:pooled[doc][k]+=primary[doc][k]
            all_seeds[str(seed)]={'covered_primary':{'by_paper':primary,'total':totals(primary)},
                'all_valid_full_document_sensitivity':{'by_paper':full,'total':totals(full)},
                'released_malformed_FN_sensitivity':{'by_paper':released,'total':totals(released)}}
        values=np.asarray([[pooled[d][k]for k in ['tp','fp','fn']]for d in papers],dtype=np.int64)
        vectors[name]=values;rng=np.random.default_rng(20261003)
        indices=rng.integers(len(papers),size=(10000,len(papers)))
        ci=np.quantile(f1_counts(values[indices].sum(axis=1)),[.025,.975]).tolist()
        results[name]={'selected':freeze['selections'][name]['chosen'],'all_seeds':all_seeds,
            'pooled_by_original_paper':pooled,'pooled_primary':totals(pooled),'paper_bootstrap_95_f1_ci':ci}
    contrasts=[]
    for control in ['mean','capacity_mean']:
        a,b=vectors['typed'],vectors[control]
        contrast={'name':'typed_minus_'+control,**exact_paired_exchange(a,b),
            **paired_paper_bootstrap(a,b,resamples=10000,seed=20261003)}
        contrasts.append(contrast)
    for row,adjusted in zip(contrasts,holm_adjust([r['raw_exact_p']for r in contrasts])):
        row['holm_p_two_planned_neural_contrasts']=adjusted
    report={'scope':'supervised combined role/position ablation, not feedback or exact author-model reproduction',
        'test_papers':14,'paper_ids':papers,'architecture_results':results,'planned_contrasts':contrasts,
        'complete_graph_barrier':9,'test_gold_loaded_only_after_all_nine_source_graphs':True,
        'freeze_sha256':sha(ROOT/FREEZE_REL),'generation_manifest_sha256':sha(ROOT/DEST_REL/'generation_manifest.json'),
        'bootstrap_draws':10000,'bootstrap_seed':20261003,'seeds_are_independent_papers':False,
        'original_LLM_primary_family_changed':False}
    write(ROOT/DEST_REL/'test_summary.json',report)
    print(json.dumps({'pooled_primary':{n:r['pooled_primary']for n,r in results.items()},'contrasts':contrasts},indent=2))
    return report


if __name__=='__main__':main()
