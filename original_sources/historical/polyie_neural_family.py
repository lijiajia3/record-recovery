"""Shared locked family metadata and nine-graph barrier; no annotation loader."""
import json
from polyie_local_adapter import ROOT, sha, population_count

SEEDS=(20261003,20261004,20261005)
ARCHITECTURES={
    'mean':('polyie_shared_context_v1','polyie_local_training_freeze.json'),
    'typed':('polyie_typed_interaction_v1','polyie_typed_training_freeze.json'),
    'capacity_mean':('polyie_capacity_mean_v1','polyie_capacity_training_freeze.json'),
}
FREEZE_REL='research/polyie_neural_family_test_freeze.json'
DEST_REL='results/local_baseline/polyie_neural_family_v1'


def output(root,name):return root/'results/local_baseline'/ARCHITECTURES[name][0]
def selection_path(root,name):return output(root,name)/'immutable_development_selection.json'


def validate_lock(root=ROOT):
    freeze_path=root/FREEZE_REL
    freeze=json.loads(freeze_path.read_text())
    assert freeze['architecture_ids']==list(ARCHITECTURES)
    assert freeze['seed_ids']==list(SEEDS)and freeze['required_complete_graphs']==9
    for name,expected in freeze['file_sha256'].items():assert sha(root/name)==expected,name
    for name in ARCHITECTURES:
        selection_file=selection_path(root,name);selected=json.loads(selection_file.read_text())
        state=json.loads((output(root,name)/'training_state.json').read_text())
        assert state['status']=='completed'and state['seeds']==list(SEEDS)and state['epochs']==3
        assert selected['seed_ids']==list(SEEDS)
        training=root/'research'/ARCHITECTURES[name][1]
        assert selected['training_freeze_sha256']==sha(training)
        assert freeze['selections'][name]['sha256']==sha(selection_file)
        assert state['selection_sha256']==sha(selection_file)
        assert freeze['selections'][name]['chosen']==selected['chosen']
        for seed in SEEDS:
            checkpoint=output(root,name)/f'seed{seed}'/f'epoch{selected["chosen"]["epoch"]}.pt'
            assert freeze['checkpoints'][name][str(seed)]['sha256']==sha(checkpoint)
    return freeze


def complete_family(root=ROOT):
    """Verify all nine source graphs before callers may interpret any Gold."""
    freeze=validate_lock(root);dest=root/DEST_REL
    manifest=json.loads((dest/'generation_manifest.json').read_text())
    assert manifest['freeze_sha256']==sha(root/FREEZE_REL)
    assert manifest['test_targets_read']is False
    assert manifest['test_source_sha256']==freeze['test_source_sha256']
    source_path=root/'data/polyie/test_inputs.json'
    assert sha(source_path)==freeze['test_source_sha256']
    source=json.loads(source_path.read_text())
    allowed={'doc_key','window_id','start','end','text','tokens','entities','call_needed'}
    assert len(source)==61 and all(set(inp)==allowed for inp in source)
    assert len({inp['window_id']for inp in source})==61
    papers=sorted({inp['doc_key']for inp in source});assert len(papers)==14
    expected_counts={p:sum(population_count(inp)for inp in source if inp['doc_key']==p)for p in papers}
    assert manifest['paper_ids']==papers and manifest['scope']=={'papers':14,'windows':61}
    assert manifest['architecture_ids']==list(ARCHITECTURES)and manifest['seed_ids']==list(SEEDS)
    expected={name+'|'+str(seed)for name in ARCHITECTURES for seed in SEEDS}
    assert set(manifest['complete_graphs'])==expected
    for path,digest in manifest['source_cache_sha256'].items():assert sha(root/path)==digest,path
    graphs={}
    for name in ARCHITECTURES:
        selected=freeze['selections'][name]
        for seed in SEEDS:
            identity=name+'|'+str(seed);entry=manifest['complete_graphs'][identity]
            path=root/entry['path'];assert sha(path)==entry['sha256'],identity
            graph=json.loads(path.read_text())
            assert graph['architecture']==name and graph['seed']==seed
            assert graph['test_targets_read']is False
            assert graph['epoch']==selected['chosen']['epoch']
            assert graph['threshold']==selected['chosen']['threshold']
            assert graph['selection_sha256']==selected['sha256']
            assert graph['checkpoint_sha256']==freeze['checkpoints'][name][str(seed)]['sha256']
            assert set(graph['graphs'])==set(manifest['paper_ids'])and len(graph['graphs'])==14
            assert set(graph['enumerated_candidates_by_paper'])==set(manifest['paper_ids'])
            assert all(isinstance(n,int)and n>=0 for n in graph['enumerated_candidates_by_paper'].values())
            assert all(len(graph['graphs'][p])<=graph['enumerated_candidates_by_paper'][p] for p in manifest['paper_ids'])
            graphs[name,seed]=graph
    assert all(g['enumerated_candidates_by_paper']==expected_counts for g in graphs.values())
    return freeze,manifest,graphs
