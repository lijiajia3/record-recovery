"""Freeze complete dev choices/checkpoints; copy prior Gold digest without parsing."""
from datetime import datetime,timezone
import json
from polyie_local_adapter import ROOT,sha
from polyie_neural_family import SEEDS,ARCHITECTURES,FREEZE_REL,output,selection_path
from polyie_local_baseline_training import write


def main():
    dest=ROOT/FREEZE_REL
    assert not dest.exists(),'Family lock already exists; never replace selected evidence'
    files={};selections={};checkpoints={}
    def add(path):files[str(path.relative_to(ROOT))]=sha(path)
    for name in ARCHITECTURES:
        out=output(ROOT,name);state=json.loads((out/'training_state.json').read_text())
        assert state['status']=='completed'and state['seeds']==list(SEEDS)and state['epochs']==3
        assert state['test_files_read']is False
        all_epochs=json.loads((out/'development_all_epochs.json').read_text())
        assert set(all_epochs)=={str(s)for s in SEEDS}
        assert all(set(epochs)=={'1','2','3'}for epochs in all_epochs.values())
        path=selection_path(ROOT,name);selected=json.loads(path.read_text())
        assert selected['seed_ids']==list(SEEDS)and state['selection_sha256']==sha(path)
        training=ROOT/'research'/ARCHITECTURES[name][1]
        assert selected['training_freeze_sha256']==sha(training)
        selections[name]={'path':str(path.relative_to(ROOT)),'sha256':sha(path),'chosen':selected['chosen']}
        add(path);add(out/'training_state.json');add(out/'development_all_epochs.json');add(training)
        for f,digest in json.loads(training.read_text())['file_sha256'].items():
            assert sha(ROOT/f)==digest,f
            if f in files:assert files[f]==digest
            files[f]=digest
        checkpoints[name]={}
        for seed in SEEDS:
            cp=out/f'seed{seed}'/f'epoch{selected["chosen"]["epoch"]}.pt';add(cp)
            checkpoints[name][str(seed)]={'path':str(cp.relative_to(ROOT)),'sha256':sha(cp)}
    for f in ['src/polyie_neural_family.py','src/lock_polyie_neural_family.py',
        'src/polyie_neural_family_test_generation.py','src/polyie_neural_family_test_analysis.py',
        'src/check_polyie_neural_family_barrier.py','src/holdout_statistics.py',
        'src/polyie_local_adapter.py','src/polyie_local_baseline_training.py',
        'src/polyie_typed_classifier.py','src/polyie_capacity_mean_classifier.py',
        'src/polyie_experiment.py','src/pilot.py','research/polyie_neural_family_test_protocol.md',
        'research/polyie_capacity_control_addendum.md','research/external_holdout_v1_freeze.json',
        'research/mulms_local_supervised_protocol.md',
        'research/polyie_local_source_cache_freeze.json','data/polyie/test_inputs.json','data/polyie/test_documents.json']:
        add(ROOT/f)
    # Bind the nested source-cache dependencies and the actual pinned model bytes,
    # not only a parent freeze or an unbound currently-present model manifest.
    source_freeze=json.loads((ROOT/'research/polyie_local_source_cache_freeze.json').read_text())
    for f,digest in source_freeze['file_sha256'].items():
        assert sha(ROOT/f)==digest,f
        if f in files:assert files[f]==digest
        files[f]=digest
    model_manifest=ROOT/'data/local_baselines/matscibert/source_manifest.json'
    for entry in json.loads(model_manifest.read_text())['files']:
        p=model_manifest.parent/entry['name'];assert sha(p)==entry['sha256'];add(p)
    # This immutable external digest existed before this neural family was designed.
    prior=json.loads((ROOT/'research/external_holdout_v1_freeze.json').read_text())
    assert files['data/polyie/test_inputs.json']==prior['file_sha256']['data/polyie/test_inputs.json']
    assert files['data/polyie/test_documents.json']==prior['file_sha256']['data/polyie/test_documents.json']
    value={'frozen_at_utc':datetime.now(timezone.utc).isoformat(),'architecture_ids':list(ARCHITECTURES),
        'seed_ids':list(SEEDS),'required_complete_graphs':9,'selections':selections,'checkpoints':checkpoints,
        'file_sha256':files,'test_gold_sha256':prior['file_sha256']['data/polyie/test_gold.json'],
        'test_source_sha256':files['data/polyie/test_inputs.json'],
        'test_gold_interpreted':False,'test_predictions_generated':False,
        'planned_neural_contrasts':['typed_minus_mean','typed_minus_capacity_mean'],'holm_family_size':2,
        'original_LLM_family_changed':False}
    write(dest,value);print(json.dumps({'freeze':str(dest.relative_to(ROOT)),'sha256':sha(dest),'files':len(files)}))


if __name__=='__main__':main()
