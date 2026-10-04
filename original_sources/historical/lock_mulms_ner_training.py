"""Freeze complete text caches and all detector supervision before first fit."""
from datetime import datetime,timezone
import json,platform
import torch,transformers
from mulms_local_adapter import ROOT,CACHE,MODEL_DIR,sha,source_cache_path
from polyie_local_baseline_training import write


def main():
    dest=ROOT/'research/mulms_local_ner_training_freeze.json';assert not dest.exists()
    files=['src/lock_mulms_ner_training.py','src/mulms_local_ner_training.py','src/check_mulms_ner_selection.py',
        'src/mulms_local_adapter.py','src/check_mulms_local_adapter.py','src/prepare_mulms_local_cache.py',
        'src/polyie_local_baseline_training.py','src/polyie_local_adapter.py','src/mulms_experiment.py',
        'research/mulms_local_supervised_protocol.md','research/mulms_local_ner_fitting_addendum.md',
        'research/mulms_local_source_cache_freeze.json','research/mulms_local_ner_source_review.json',
        'research/mulms_local_ner_representation_audit.json','data/local_baselines/matscibert/source_manifest.json',
        'data/mulms/train_inputs.json','data/mulms/dev_inputs.json','data/local_baselines/mulms/train_ner_gold.json',
        'data/local_baselines/mulms/dev_ner_gold.json','data/mulms/source/train.parquet','data/mulms/source/dev.parquet']
    source_freeze=json.loads((ROOT/'research/mulms_local_source_cache_freeze.json').read_text())
    for name,digest in source_freeze['file_sha256'].items():assert sha(ROOT/name)==digest,name
    for entry in json.loads((MODEL_DIR/'source_manifest.json').read_text())['files']:
        p=MODEL_DIR/entry['name'];assert sha(p)==entry['sha256'];files.append(str(p.relative_to(ROOT)))
    for split in ['train','dev']:
        p=CACHE/split/'state.json';state=json.loads(p.read_text())
        inputs=json.loads((ROOT/f'data/mulms/{split}_inputs.json').read_text())
        assert state['status']=='completed_source_only_cache'and state['sentences']==state['complete_sentences']==len(inputs)
        assert state['targets_read']is False and state['freeze_sha256']==sha(ROOT/'research/mulms_local_source_cache_freeze.json')
        assert set(state['cache_sha256'])=={str(source_cache_path(split,i).relative_to(ROOT))for i in inputs}
        for name,digest in state['cache_sha256'].items():assert sha(ROOT/name)==digest,name
        files.append(str(p.relative_to(ROOT)))
    value={'frozen_at_utc':datetime.now(timezone.utc).isoformat(),'stage':'three-seed nested-span detector before first fit',
        'seeds':[20261003,20261004,20261005],'epochs_per_seed':3,'test_data_allowed':False,
        'file_sha256':{f:sha(ROOT/f)for f in sorted(set(files))},'candidate_width_wordpieces':32,
        'environment':{'python':platform.python_version(),'torch':torch.__version__,'transformers':transformers.__version__,
            'device':'mps'if torch.backends.mps.is_available()else'cpu','float_dtype':'float32'},
        'relation_heads_or_test_stage_complete':False,'original_LLM_primary_family_changed':False}
    write(dest,value);print(json.dumps({'freeze_sha256':sha(dest),'files':len(value['file_sha256'])}))


if __name__=='__main__':main()
