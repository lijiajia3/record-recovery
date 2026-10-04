"""Generate all three locked detector graphs from every dev source sentence."""
from datetime import datetime,timezone
import gc,json,os
import numpy as np
import torch
from transformers import BertModel
from mulms_local_adapter import ROOT,MODEL_DIR,OUTPUT,sha,SpanDetector,span_candidates,entities_from_probs
from mulms_local_ner_training import SEEDS,FREEZE,DEST,load_source,select,validate_sources
from polyie_local_baseline_training import write

LOCK=ROOT/'research/mulms_ner_development_generation_freeze.json'
GRAPHS=OUTPUT/'ner/development_graphs'


def completed_detector():
    validate_sources();state=json.loads((DEST/'training_state.json').read_text())
    assert state['status']=='completed'and state['test_data_read']is False and state['freeze_sha256']==sha(FREEZE)
    choice_path=DEST/'immutable_development_selection.json';choice=json.loads(choice_path.read_text())
    assert state['selection_sha256']==sha(choice_path)and choice['training_freeze_sha256']==sha(FREEZE)
    results=json.loads((DEST/'development_all_epochs.json').read_text());expected=select(results)
    assert choice['chosen']==expected['chosen']and choice['all_pooled_candidates']==expected['all_pooled_candidates']
    for seed in SEEDS:
        for entry in results[str(seed)].values():assert sha(ROOT/entry['checkpoint'])==entry['checkpoint_sha256']
    return choice,results


def lock():
    assert not LOCK.exists(),'Never replace detector prediction lock'
    choice,results=completed_detector();epoch=str(choice['chosen']['epoch'])
    files=['src/mulms_local_ner_development_generation.py','research/mulms_local_ner_training_freeze.json',
        'results/local_baseline/mulms_supervised_v1/ner/immutable_development_selection.json',
        'results/local_baseline/mulms_supervised_v1/ner/development_all_epochs.json','data/mulms/dev_inputs.json']
    files+=list(json.loads(FREEZE.read_text())['file_sha256'])
    files+=[results[str(seed)][epoch]['checkpoint']for seed in SEEDS]
    value={'frozen_at_utc':datetime.now(timezone.utc).isoformat(),'scope':'source-only predicted dev NER before relation fitting',
        'file_sha256':{f:sha(ROOT/f)for f in sorted(set(files))},'chosen':choice['chosen'],
        'seeds':list(SEEDS),'annotation_content_used_at_inference':False,'test_data_read':False}
    write(LOCK,value);return value


def main():
    assert not GRAPHS.exists(),'Prior complete or partial NER graph run exists; no overwrite'
    locked=lock()
    for name,digest in locked['file_sha256'].items():assert sha(ROOT/name)==digest,name
    choice,results=completed_detector();inputs=json.loads((ROOT/'data/mulms/dev_inputs.json').read_text())
    GRAPHS.mkdir(parents=True);state_path=GRAPHS/'state.json';state={'pid':os.getpid(),'status':'running',
        'started_at_utc':datetime.now(timezone.utc).isoformat(),'source_only':True,'freeze_sha256':sha(LOCK)};write(state_path,state)
    device='mps'if torch.backends.mps.is_available()else'cpu';torch.set_num_threads(2);manifest={}
    try:
        for seed in SEEDS:
            entry=results[str(seed)][str(choice['chosen']['epoch'])]
            base=BertModel.from_pretrained(str(MODEL_DIR),local_files_only=True,trust_remote_code=False,use_safetensors=True)
            model=SpanDetector(base.encoder.layer[11]);del base;gc.collect()
            checkpoint=torch.load(ROOT/entry['checkpoint'],map_location='cpu',weights_only=True)
            assert checkpoint['seed']==seed and checkpoint['epoch']==choice['chosen']['epoch']and checkpoint['freeze_sha256']==sha(FREEZE)
            model.load_state_dict(checkpoint['state_dict']);del checkpoint;model=model.to(device);model.eval();graphs={};counts={}
            with torch.no_grad():
                for number,inp in enumerate(inputs):
                    saved=load_source('dev',inp);candidates=span_candidates(saved['offsets']);entities=[]
                    if candidates:
                        tokens=model.tokens(saved,device)
                        for start in range(0,len(candidates),1024):
                            batch=candidates[start:start+1024];probabilities=torch.sigmoid(model.spans(tokens,batch)).cpu().numpy()
                            entities+=entities_from_probs(inp,batch,probabilities,choice['chosen']['threshold'])
                    assert len(entities)==len({(e['start'],e['end'],e['type'])for e in entities})
                    graphs[inp['id']]=sorted(entities,key=lambda e:(e['start'],e['end'],e['type']))
                    counts[inp['id']]=len(candidates)
                    with (OUTPUT/'generation_progress.jsonl').open('a')as stream:stream.write(json.dumps({'phase':'locked_dev_ner_source_generation',
                        'seed':seed,'sentence_number':number+1,'candidate_spans':len(candidates),'predicted_entities':len(entities),
                        'at_utc':datetime.now(timezone.utc).isoformat()})+'\n')
            path=GRAPHS/f'seed{seed}.json';write(path,{'seed':seed,'chosen':choice['chosen'],'graphs':graphs,
                'candidate_span_counts':counts,'source_input_sha256':sha(ROOT/'data/mulms/dev_inputs.json'),
                'checkpoint_sha256':entry['checkpoint_sha256'],'freeze_sha256':sha(LOCK),'annotation_content_used':False})
            manifest[str(seed)]={'path':str(path.relative_to(ROOT)),'sha256':sha(path)}
            del model;gc.collect()
            if device=='mps':torch.mps.empty_cache()
        assert set(manifest)=={str(seed)for seed in SEEDS}
        write(GRAPHS/'generation_manifest.json',{'status':'all_three_complete','seeds':manifest,'sentences_per_seed':len(inputs),
            'freeze_sha256':sha(LOCK),'test_data_read':False,'annotation_content_used':False})
        state.update(status='completed_all_three_source_graphs',completed_at_utc=datetime.now(timezone.utc).isoformat());write(state_path,state)
        print(json.dumps(state))
    except BaseException as error:
        state.update(status='stopped_explicit_failure',error=str(error));write(state_path,state);raise


if __name__=='__main__':main()
