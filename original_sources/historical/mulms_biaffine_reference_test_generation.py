"""Three complete reference source graphs before its own annotation scoring."""
from datetime import datetime,timezone
import gc,json,os
import torch
from transformers import BertModel
from mulms_neural_family import ROOT,complete_family,load_test_cache,SEEDS,sha
from mulms_local_adapter import MODEL_DIR,OUTPUT
from mulms_biaffine_reference import BiaffineReference
from mulms_biaffine_reference_training import REFERENCE_DEST,REFERENCE_FREEZE
from mulms_relation_adapter import pair_candidates,records_from_probabilities
from mulms_local_relation_training import select
from polyie_local_baseline_training import write

LOCK=ROOT/'research/mulms_biaffine_reference_test_freeze.json'
DEST=ROOT/'results/local_baseline/mulms_biaffine_reference_test_v1'


def reference_training_complete():
    for name,digest in json.loads(REFERENCE_FREEZE.read_text())['file_sha256'].items():assert sha(ROOT/name)==digest,name
    state=json.loads((REFERENCE_DEST/'training_state.json').read_text());assert state['status']=='completed_biaffine_reference_three_seeds'
    manifest=json.loads((REFERENCE_DEST/'complete_selections.json').read_text())
    assert manifest['status']=='reference_architecture_complete'and set(manifest['selections'])=={'biaffine'}
    entry=manifest['selections']['biaffine'];assert sha(ROOT/entry['path'])==entry['sha256']
    choice=json.loads((ROOT/entry['path']).read_text());results=json.loads((REFERENCE_DEST/'biaffine/development_all_epochs.json').read_text())
    expected=select(results)
    assert choice['chosen']==expected['chosen']and choice['all_pooled_candidates']==expected['all_pooled_candidates']
    assert choice['training_freeze_sha256']==sha(REFERENCE_FREEZE)and state['freeze_sha256']==sha(REFERENCE_FREEZE)
    checkpoints={}
    for seed in SEEDS:
        for record in results[str(seed)].values():assert sha(ROOT/record['checkpoint'])==record['checkpoint_sha256']
        record=results[str(seed)][str(choice['chosen']['epoch'])]
        checkpoints[str(seed)]={'path':record['checkpoint'],'sha256':record['checkpoint_sha256']}
    return choice,checkpoints


def main():
    assert not DEST.exists()and not LOCK.exists(),'Never replace prior reference lock or graphs'
    primary,manifest,inputs,detectors,_=complete_family(ROOT);choice,checkpoints=reference_training_complete()
    files=['src/mulms_biaffine_reference_test_generation.py','src/mulms_biaffine_reference_test_analysis.py',
        'research/mulms_biaffine_reference_training_freeze.json',
        'results/local_baseline/mulms_supervised_v1/biaffine_reference/biaffine/immutable_development_selection.json',
        'results/local_baseline/mulms_supervised_v1/biaffine_reference/biaffine/development_all_epochs.json',
        'results/local_baseline/mulms_supervised_v1/biaffine_reference/training_state.json',
        'results/local_baseline/mulms_supervised_v1/biaffine_reference/complete_selections.json']
    files+=list(json.loads(REFERENCE_FREEZE.read_text())['file_sha256'])
    files+=[v['path']for v in checkpoints.values()]
    write(LOCK,{'frozen_at_utc':datetime.now(timezone.utc).isoformat(),'chosen':choice['chosen'],'checkpoints':checkpoints,
        'file_sha256':{f:sha(ROOT/f)for f in sorted(set(files))},'primary_freeze_sha256':manifest['freeze_sha256'],
        'primary_generation_manifest_sha256':sha(ROOT/'results/local_baseline/mulms_neural_family_v1/generation_manifest.json'),
        'primary_summary_already_exists':(ROOT/'results/local_baseline/mulms_neural_family_v1/test_summary.json').exists(),
        'test_annotation_content_used':False,'test_gold_sha256':primary['test_gold_sha256']})
    DEST.mkdir();state_path=DEST/'generation_state.json';state={'pid':os.getpid(),'status':'running_source_only_reference',
        'started_at_utc':datetime.now(timezone.utc).isoformat(),'freeze_sha256':sha(LOCK),'test_annotation_content_used':False};write(state_path,state)
    generated={};device='mps'if torch.backends.mps.is_available()else'cpu';torch.set_num_threads(2)
    try:
        for seed in SEEDS:
            checkpoint=checkpoints[str(seed)];base=BertModel.from_pretrained(str(MODEL_DIR),local_files_only=True,trust_remote_code=False,use_safetensors=True)
            model=BiaffineReference(base.encoder.layer[11]);del base;gc.collect()
            saved=torch.load(ROOT/checkpoint['path'],map_location='cpu',weights_only=True)
            assert saved['architecture']=='biaffine'and saved['seed']==seed and saved['epoch']==choice['chosen']['epoch']
            assert saved['freeze_sha256']==sha(REFERENCE_FREEZE);model.load_state_dict(saved['state_dict']);del saved;model=model.to(device).eval()
            graphs={};pair_counts={};label_counts={}
            with torch.no_grad():
                for number,inp in enumerate(inputs):
                    entities=detectors[seed]['graphs'][inp['id']];pairs=pair_candidates(entities);records=[]
                    if pairs:
                        cached=load_test_cache(inp);matrix=model.entities(cached,entities,device)
                        for start in range(0,len(pairs),2048):
                            batch=pairs[start:start+2048];p=torch.sigmoid(model.pairs(matrix,entities,batch,len(inp['text']))).cpu().numpy()
                            records+=records_from_probabilities(entities,batch,p,inp['text'],choice['chosen']['threshold'])
                    graphs[inp['id']]=records;pair_counts[inp['id']]=len(pairs);label_counts[inp['id']]=15*len(pairs)
                    with (OUTPUT/'generation_progress.jsonl').open('a')as stream:stream.write(json.dumps({'phase':'full_source_only_classical_reference',
                        'seed':seed,'sentence_number':number+1,'candidate_pairs':len(pairs),'at_utc':datetime.now(timezone.utc).isoformat()})+'\n')
            path=DEST/f'seed{seed}.json';write(path,{'seed':seed,'architecture':'biaffine','chosen':choice['chosen'],'graphs':graphs,
                'candidate_pair_counts':pair_counts,'candidate_label_counts':label_counts,'source_input_sha256':sha(ROOT/'data/mulms/test_inputs.json'),
                'detector_graph_sha256':manifest['detector_graphs'][str(seed)]['sha256'],'checkpoint_sha256':checkpoint['sha256'],
                'freeze_sha256':sha(LOCK),'annotation_content_used':False})
            generated[str(seed)]={'path':str(path.relative_to(ROOT)),'sha256':sha(path)}
            del model;gc.collect()
            if device=='mps':torch.mps.empty_cache()
        write(DEST/'generation_manifest.json',{'status':'all_three_reference_graphs_complete','graphs':generated,
            'freeze_sha256':sha(LOCK),'sentences_per_seed':len(inputs),'source_input_sha256':sha(ROOT/'data/mulms/test_inputs.json'),
            'primary_generation_manifest_sha256':sha(ROOT/'results/local_baseline/mulms_neural_family_v1/generation_manifest.json')})
        state.update(status='completed_all_three_reference_source_graphs',completed_at_utc=datetime.now(timezone.utc).isoformat());write(state_path,state)
    except BaseException as error:
        state.update(status='stopped_explicit_failure',error=str(error));write(state_path,state);raise


if __name__=='__main__':main()
