"""Bind full detector/dev predictions and supervised-source files before fit."""
from datetime import datetime,timezone
import json,platform
import torch,transformers
from mulms_local_relation_training import ROOT,DEST,FREEZE,ARCHITECTURES,SEEDS,predicted_development_entities
from mulms_local_ner_development_generation import LOCK as NER_LOCK,GRAPHS as NER_GRAPHS,completed_detector
from mulms_local_ner_training import FREEZE as NER_FREEZE
from mulms_local_adapter import CACHE,MODEL_DIR,sha
from polyie_local_baseline_training import write


def main():
    assert not FREEZE.exists()and not DEST.exists(),'Do not replace a lock or prior fit'
    completed_detector();predicted=predicted_development_entities()
    for seed in SEEDS:
        assert len(predicted[seed])==1532
    files=['src/lock_mulms_relation_training.py','src/mulms_local_relation_training.py','src/mulms_relation_adapter.py',
        'src/check_mulms_relation_adapter.py','src/mulms_local_ner_development_generation.py','src/audit_mulms_relation_training_source.py',
        'research/mulms_relation_source_representation_addendum.md','research/mulms_relation_training_representation_audit.json',
        'research/mulms_local_relation_source_review.json','research/mulms_local_ner_training_freeze.json',
        'research/mulms_ner_development_generation_freeze.json','research/mulms_local_supervised_protocol.md',
        'data/mulms/train_gold.json','data/mulms/dev_gold.json','data/local_baselines/mulms/train_ner_gold.json',
        str((NER_GRAPHS/'generation_manifest.json').relative_to(ROOT))]
    files+=list(json.loads(NER_FREEZE.read_text())['file_sha256'])
    files+=list(json.loads(NER_LOCK.read_text())['file_sha256'])
    manifest=json.loads((NER_GRAPHS/'generation_manifest.json').read_text())
    files+=[value['path']for value in manifest['seeds'].values()]
    audit=json.loads((ROOT/'research/mulms_relation_training_representation_audit.json').read_text())
    assert audit['test_annotations_read']is False
    assert len(audit['splits']['train']['zero_wordpiece_overlap_entities'])==3
    assert not audit['splits']['dev']['zero_wordpiece_overlap_entities']
    assert all(not v['edges_missing_gold_ner_endpoint']for v in audit['splits'].values())
    assert audit['model_manifest_sha256']==sha(MODEL_DIR/'source_manifest.json')
    assert set(audit['tokenizer_sha256'])=={'tokenizer.json','tokenizer_config.json','vocab.txt'}
    for name,digest in audit['tokenizer_sha256'].items():assert sha(MODEL_DIR/name)==digest,name
    for split in ['train','dev']:
        paths={'inputs_sha256':f'data/mulms/{split}_inputs.json',
            'ner_sha256':f'data/local_baselines/mulms/{split}_ner_gold.json','edges_sha256':f'data/mulms/{split}_gold.json'}
        for field,name in paths.items():assert audit['splits'][split][field]==sha(ROOT/name),name
    review=json.loads((ROOT/'research/mulms_local_relation_source_review.json').read_text())
    # Independently reviewed source hashes must match, regardless of report layout.
    reviewed=review.get('file_sha256',review.get('source_sha256',{}))
    assert reviewed,'Reviewer must bind reviewed source files explicitly'
    for name,digest in reviewed.items():assert sha(ROOT/name)==digest,name
    files+=list(reviewed)
    lock={'frozen_at_utc':datetime.now(timezone.utc).isoformat(),'stage':'three full-pair supervised relation heads before fit',
        'file_sha256':{f:sha(ROOT/f)for f in sorted(set(files))},'architectures':list(ARCHITECTURES),'seeds':list(SEEDS),
        'all_architectures_same_detector_by_seed':True,'development_inference_uses_gold_entities':False,
        'train_empty_source_token_entities':3,'empty_token_policy':'fixed zero vector, preserve all supervised pairs and targets',
        'test_data_allowed':False,'environment':{'python':platform.python_version(),'torch':torch.__version__,
            'transformers':transformers.__version__,'device':'mps'if torch.backends.mps.is_available()else'cpu','float_dtype':'float32'},
        'nominal_capacity_control_is_equal_effective_capacity':False,'original_LLM_primary_family_changed':False}
    write(FREEZE,lock);print(json.dumps({'freeze_sha256':sha(FREEZE),'files':len(lock['file_sha256'])}))


if __name__=='__main__':main()
