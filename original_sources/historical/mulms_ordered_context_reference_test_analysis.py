"""Descriptive-only classical reference, all three source graphs before Gold."""
import json
import numpy as np
from mulms_ordered_context_reference_test_generation import ROOT,LOCK,DEST,SEEDS,sha
from mulms_neural_family import complete_family
from mulms_local_adapter import span_key,RELATION_IDS
from mulms_experiment import endpoint_key
from mulms_neural_family_test_analysis import load_targets,relation_counts,total
from holdout_statistics import f1_counts
from polyie_local_baseline_training import write


def reference_complete():
    primary,main,inputs,detectors,_=complete_family(ROOT)
    lock=json.loads(LOCK.read_text())
    for name,digest in lock['file_sha256'].items():assert sha(ROOT/name)==digest,name
    assert lock['primary_freeze_sha256']==main['freeze_sha256']
    assert lock['primary_generation_manifest_sha256']==sha(ROOT/'results/local_baseline/mulms_neural_family_v1/generation_manifest.json')
    manifest=json.loads((DEST/'generation_manifest.json').read_text())
    assert manifest['status']=='all_three_reference_graphs_complete'and manifest['freeze_sha256']==sha(LOCK)
    assert set(manifest['graphs'])=={str(s)for s in SEEDS}and manifest['sentences_per_seed']==1114
    assert manifest['source_input_sha256']==sha(ROOT/'data/mulms/test_inputs.json')
    assert manifest['primary_generation_manifest_sha256']==lock['primary_generation_manifest_sha256']
    ids={i['id']for i in inputs};graphs={}
    for seed in SEEDS:
        entry=manifest['graphs'][str(seed)];assert sha(ROOT/entry['path'])==entry['sha256']
        graph=json.loads((ROOT/entry['path']).read_text())
        assert graph['seed']==seed and graph['architecture']=='ordered_context'and graph['annotation_content_used']is False
        assert graph['chosen']==lock['chosen']and graph['checkpoint_sha256']==lock['checkpoints'][str(seed)]['sha256']
        assert graph['freeze_sha256']==sha(LOCK)and graph['source_input_sha256']==manifest['source_input_sha256']
        assert graph['detector_graph_sha256']==main['detector_graphs'][str(seed)]['sha256']
        assert set(graph['graphs'])==ids==set(graph['candidate_pair_counts'])==set(graph['candidate_label_counts'])
        for inp in inputs:
            entity_keys={span_key(e)for e in detectors[seed]['graphs'][inp['id']]};number=len(entity_keys)**2
            assert graph['candidate_pair_counts'][inp['id']]==number and graph['candidate_label_counts'][inp['id']]==15*number
            records=graph['graphs'][inp['id']];assert len(records)==len({json.dumps(r,sort_keys=True)for r in records})
            for record in records:
                assert endpoint_key(record.get('h'),inp['text'])in entity_keys and endpoint_key(record.get('t'),inp['text'])in entity_keys
                assert record.get('r')in RELATION_IDS
        graphs[seed]=graph
    return primary,inputs,main['paper_ids'],lock,graphs


def main():
    primary,inputs,papers,lock,graphs=reference_complete()
    gold,_=load_targets(primary,inputs,ROOT)
    pooled={p:{'tp':0,'fp':0,'fn':0}for p in papers};seeds={}
    for seed in SEEDS:
        seeds[str(seed)]=relation_counts(inputs,graphs[seed]['graphs'],gold)
        for p in papers:
            for k in pooled[p]:pooled[p][k]+=seeds[str(seed)]['by_paper'][p][k]
    vectors=np.asarray([[pooled[p][k]for k in ['tp','fp','fn']]for p in papers])
    rng=np.random.default_rng(20261003);indices=rng.integers(len(papers),size=(10000,len(papers)))
    ci=np.quantile(f1_counts(vectors[indices].sum(axis=1)),[.025,.975]).tolist()
    result={'scope':'Additional same-architecture ordered-context reference; descriptive only, no new superiority hypothesis',
        'selected':lock['chosen'],'all_seeds':seeds,'pooled_by_original_paper':pooled,'pooled_primary':total(pooled),
        'paper_bootstrap_95_f1_ci':ci,'test_papers':7,'test_sentences':1114,'paper_ids':papers,
        'seeds_are_independent_papers':False,'test_gold_loaded_after_all_three_reference_graphs':True,
        'primary_summary_existed_before_reference_test_lock':lock['primary_summary_already_exists'],
        'fresh_independent_holdout_after_primary_score_claimed':False,'planned_primary_holm2_and_global4_modified':False,
        'freeze_sha256':sha(LOCK),'generation_manifest_sha256':sha(DEST/'generation_manifest.json')}
    write(DEST/'test_summary.json',result);print(json.dumps(result['pooled_primary']))


if __name__=='__main__':main()
