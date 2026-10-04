"""Pre-fit binding of an additional declared directed classical reference."""
from datetime import datetime,timezone
import json
import mulms_ordered_context_reference_training as configured
from mulms_local_ner_development_generation import LOCK as NER_LOCK
from mulms_local_relation_training import predicted_development_entities
from mulms_local_adapter import ROOT,sha
from polyie_local_baseline_training import write


def main():
    trainer=configured.trainer;freeze_path=configured.REFERENCE_FREEZE;dest=configured.REFERENCE_DEST
    assert not freeze_path.exists()and not dest.exists()
    original=trainer.validate_freeze()
    predicted_development_entities()
    files=['src/mulms_ordered_context_reference.py','src/mulms_ordered_context_reference_training.py','src/lock_mulms_ordered_context_reference_training.py',
        'src/check_mulms_ordered_context_reference.py','src/check_mulms_ordered_context_reference_barrier.py','src/continue_mulms_ordered_context_reference.py',
        'src/mulms_ordered_context_reference_test_generation.py','src/mulms_ordered_context_reference_test_analysis.py',
        'research/mulms_task_matched_ordered_context_reference_protocol.md','research/mulms_ordered_context_reference_pretest_intent.json',
        'research/mulms_ordered_context_reference_source_review.json','research/mulms_ner_development_generation_freeze.json',
        'research/mulms_relation_training_representation_audit.json',
        'data/mulms/train_gold.json','data/mulms/dev_gold.json']
    files+=list(json.loads(NER_LOCK.read_text())['file_sha256'])
    files.append(str(trainer.FREEZE.relative_to(ROOT)));files+=list(original['file_sha256'])
    files+=list(json.loads((ROOT/'research/mulms_local_relation_source_review.json').read_text())['file_sha256'])
    dev_manifest=ROOT/'results/local_baseline/mulms_supervised_v1/ner/development_graphs/generation_manifest.json'
    files.append(str(dev_manifest.relative_to(ROOT)));manifest=json.loads(dev_manifest.read_text())
    files+=[r['path']for r in manifest['seeds'].values()]
    review=json.loads((ROOT/'research/mulms_ordered_context_reference_source_review.json').read_text())
    for name,digest in review['file_sha256'].items():assert sha(ROOT/name)==digest,name
    files+=list(review['file_sha256'])
    intent=json.loads((ROOT/'research/mulms_ordered_context_reference_pretest_intent.json').read_text())
    for name,digest in intent['file_sha256'].items():assert sha(ROOT/name)==digest,name
    value={'frozen_at_utc':datetime.now(timezone.utc).isoformat(),'architecture_ids':['ordered_context'],'seed_ids':list(trainer.SEEDS),
        'epochs_per_seed':3,'hyperparameter_or_selection_redesign_after_primary_MuLMS_test':False,
        'same_training_and_predicted_entities_as_original_three_heads':True,'same_nominal_head_architecture_with_ordered_context':True,
        'new_significance_contrasts':False,'file_sha256':{f:sha(ROOT/f)for f in sorted(set(files))}}
    write(freeze_path,value);print(json.dumps({'freeze_sha256':sha(freeze_path),'files':len(value['file_sha256'])}))


if __name__=='__main__':main()
