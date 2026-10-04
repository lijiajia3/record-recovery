"""Freeze three detectors/nine selected relation models before test sources."""
from datetime import datetime,timezone
import json
from mulms_neural_family import ROOT,FREEZE_REL,DEST_REL,SEEDS,ARCHITECTURES,sha
from mulms_local_ner_development_generation import completed_detector,LOCK as NER_LOCK
from mulms_local_ner_training import FREEZE as NER_FREEZE
from mulms_local_relation_training import DEST,FREEZE as REL_FREEZE,select,validate_freeze
from polyie_local_baseline_training import write


def main():
    output=ROOT/FREEZE_REL;assert not output.exists()and not (ROOT/DEST_REL).exists()
    validate_freeze();choice,ner_results=completed_detector()
    state=json.loads((DEST/'training_state.json').read_text());assert state['status']=='completed_all_three_architectures'
    selections=json.loads((DEST/'complete_selections.json').read_text())
    assert selections['status']=='all_three_architectures_complete'and selections['freeze_sha256']==sha(REL_FREEZE)
    assert set(selections['selections'])==set(ARCHITECTURES)
    files=['src/mulms_neural_family.py','src/lock_mulms_neural_family.py','src/mulms_neural_family_test_generation.py',
        'src/mulms_neural_family_test_analysis.py','src/check_mulms_neural_family_barrier.py','src/holdout_statistics.py',
        'src/joint_supervised_multiplicity.py','src/check_joint_supervised_multiplicity.py',
        'research/mulms_neural_family_source_review.json','research/mulms_local_relation_training_freeze.json',
        'research/mulms_local_ner_training_freeze.json','research/mulms_ner_development_generation_freeze.json',
        'data/mulms/test_inputs.json','data/mulms/task_manifest.json','research/external_holdout_v1_freeze.json',
        str((DEST/'complete_selections.json').relative_to(ROOT)),str((DEST/'training_state.json').relative_to(ROOT))]
    for parent in [NER_FREEZE,NER_LOCK,REL_FREEZE]:files+=list(json.loads(parent.read_text())['file_sha256'])
    detector_cp={};relation_choices={};relation_cp={}
    for seed in SEEDS:
        record=ner_results[str(seed)][str(choice['chosen']['epoch'])]
        detector_cp[str(seed)]={'path':record['checkpoint'],'sha256':record['checkpoint_sha256']};files.append(record['checkpoint'])
    for architecture in ARCHITECTURES:
        entry=selections['selections'][architecture];assert sha(ROOT/entry['path'])==entry['sha256']
        selected=json.loads((ROOT/entry['path']).read_text());all_path=DEST/architecture/'development_all_epochs.json'
        results=json.loads(all_path.read_text());expected=select(results)
        assert selected['chosen']==expected['chosen']and selected['all_pooled_candidates']==expected['all_pooled_candidates']
        assert selected['training_freeze_sha256']==sha(REL_FREEZE)and selected['test_data_read']is False
        relation_choices[architecture]=selected;relation_cp[architecture]={};files.extend([entry['path'],str(all_path.relative_to(ROOT))])
        for seed in SEEDS:
            for record in results[str(seed)].values():assert sha(ROOT/record['checkpoint'])==record['checkpoint_sha256']
            record=results[str(seed)][str(selected['chosen']['epoch'])]
            relation_cp[architecture][str(seed)]={'path':record['checkpoint'],'sha256':record['checkpoint_sha256']};files.append(record['checkpoint'])
    external=json.loads((ROOT/'research/external_holdout_v1_freeze.json').read_text())
    for name,digest in external['file_sha256'].items():assert sha(ROOT/name)==digest,name
    assert sha(ROOT/'data/mulms/test_inputs.json')==external['file_sha256']['data/mulms/test_inputs.json']
    original_source_freeze=ROOT/'research/mulms_development_v1_freeze.json'
    original_source=json.loads(original_source_freeze.read_text())
    assert sha(ROOT/'data/mulms/task_manifest.json')==original_source['file_sha256']['data/mulms/task_manifest.json']
    files.append(str(original_source_freeze.relative_to(ROOT)))
    files+=list(external['file_sha256'])
    gold_sha=external['file_sha256']['data/mulms/test_gold.json'];assert sha(ROOT/'data/mulms/test_gold.json')==gold_sha
    source_sha=json.loads((ROOT/'data/mulms/task_manifest.json').read_text())['splits']['test']['source_sha256']
    assert sha(ROOT/'data/mulms/source/test.parquet')==source_sha
    review=json.loads((ROOT/'research/mulms_neural_family_source_review.json').read_text());assert review['file_sha256']
    for name,digest in review['file_sha256'].items():assert sha(ROOT/name)==digest,name
    value={'frozen_at_utc':datetime.now(timezone.utc).isoformat(),'architecture_ids':list(ARCHITECTURES),'seed_ids':list(SEEDS),
        'required_complete_detector_graphs':3,'required_complete_relation_graphs':9,'detector_selection':choice,
        'detector_checkpoints':detector_cp,'relation_selections':relation_choices,'relation_checkpoints':relation_cp,
        'file_sha256':{f:sha(ROOT/f)for f in sorted(set(files))},'test_gold_sha256':gold_sha,'test_ner_source_sha256':source_sha,
        'test_annotation_interpretation_before_complete_family':False,'planned_contrasts':['typed_minus_mean','typed_minus_capacity_mean'],
        'holm_family_size':2,'global_two_domain_sensitivity_family_size':4,'original_LLM_primary_family_changed':False}
    write(output,value);print(json.dumps({'freeze_sha256':sha(output),'files':len(value['file_sha256'])}))


if __name__=='__main__':main()
