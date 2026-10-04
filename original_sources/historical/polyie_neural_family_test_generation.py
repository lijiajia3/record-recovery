"""All nine locked source-only graphs, shared deterministic cache, no Gold reads."""
from datetime import datetime,timezone
import gc,json,os
import numpy as np
import torch
from transformers import BertModel,AutoTokenizer
from polyie_local_adapter import ROOT,MODEL_DIR,sha,validate_model,chunk_encoding,population_count,candidate_batches,record_for
from polyie_local_baseline_training import Classifier as MeanClassifier,write
from polyie_typed_classifier import TypedClassifier
from polyie_capacity_mean_classifier import CapacityMeanClassifier
from polyie_neural_family import SEEDS,ARCHITECTURES,FREEZE_REL,DEST_REL,validate_lock,output

CLASSIFIERS={'mean':MeanClassifier,'typed':TypedClassifier,'capacity_mean':CapacityMeanClassifier}
DEST=ROOT/DEST_REL


def event(value):
    row=dict(value,at_utc=datetime.now(timezone.utc).isoformat())
    with (DEST/'generation_progress.jsonl').open('a')as stream:stream.write(json.dumps(row)+'\n')
    print(json.dumps(row),flush=True)


def main():
    freeze=validate_lock();validate_model();freeze_path=ROOT/FREEZE_REL
    if (DEST/'generation_state.json').exists():raise RuntimeError('Prior generation exists; audit recovery rather than rerun.')
    DEST.mkdir(parents=True,exist_ok=True)
    state_path=DEST/'generation_state.json'
    write(state_path,{'status':'running','pid':os.getpid(),'started_at_utc':datetime.now(timezone.utc).isoformat(),
        'test_targets_read':False,'freeze_sha256':sha(freeze_path)})
    torch.set_num_threads(2);torch.manual_seed(20261003)
    device='mps'if torch.backends.mps.is_available()else'cpu'
    source=ROOT/'data/polyie/test_inputs.json';inputs=json.loads(source.read_text())
    allowed={'doc_key','window_id','start','end','text','tokens','entities','call_needed'}
    assert all(set(inp)==allowed for inp in inputs)
    papers=sorted({inp['doc_key']for inp in inputs});assert len(inputs)==61 and len(papers)==14
    tokenizer=AutoTokenizer.from_pretrained(str(MODEL_DIR),local_files_only=True,trust_remote_code=False)
    base=BertModel.from_pretrained(str(MODEL_DIR),local_files_only=True,trust_remote_code=False,use_safetensors=True).to(device).eval()
    cache_dir=DEST/'source_cache';cache_dir.mkdir(exist_ok=True);paths=[]
    with torch.no_grad():
        for number,inp in enumerate(inputs):
            chunks,positions=chunk_encoding(inp,tokenizer);tensors=[]
            for chunk in chunks:
                ids=torch.tensor([chunk['input_ids']],device=device)
                mask=base.get_extended_attention_mask(torch.ones_like(ids),ids.shape)
                hidden=base.embeddings(input_ids=ids,token_type_ids=torch.zeros_like(ids))
                for layer in base.encoder.layer[:11]:hidden=layer(hidden,attention_mask=mask)[0]
                tensors.append({'hidden':hidden.cpu().float(),'mask':mask.cpu().float()})
            path=cache_dir/f'{number}.pt';torch.save({'chunks':tensors,'positions':positions},path);paths.append(path)
            event({'phase':'test_source_cache','window':inp['window_id'],'chunks':len(chunks),'test_targets_read':False})
    last_layer=base.encoder.layer[11];del base;gc.collect()
    manifest={'freeze_sha256':sha(freeze_path),'test_source_sha256':sha(source),'test_targets_read':False,
        'device':device,'paper_ids':papers,'scope':{'papers':14,'windows':61},
        'architecture_ids':list(ARCHITECTURES),'seed_ids':list(SEEDS),'complete_graphs':{},
        'source_cache_sha256':{str(p.relative_to(ROOT)):sha(p)for p in paths}}
    for name in ARCHITECTURES:
        chosen=freeze['selections'][name]['chosen'];model=CLASSIFIERS[name](last_layer).to(device)
        for seed in SEEDS:
            checkpoint=output(ROOT,name)/f'seed{seed}'/f'epoch{chosen["epoch"]}.pt'
            assert sha(checkpoint)==freeze['checkpoints'][name][str(seed)]['sha256']
            saved=torch.load(checkpoint,map_location=device,weights_only=True)
            assert saved['seed']==seed and saved['epoch']==chosen['epoch']
            assert saved['freeze_sha256']==sha(ROOT/'research'/ARCHITECTURES[name][1])
            model.load_state_dict(saved['state_dict']);model.eval()
            docs={doc:[]for doc in papers};counts={doc:0 for doc in papers}
            with torch.no_grad():
                for inp,path in zip(inputs,paths):
                    expected=population_count(inp);enumerated=0
                    if expected:
                        if hasattr(model,'bind_source'):model.bind_source(inp)
                        features=torch.load(path,map_location='cpu',weights_only=True)
                        identities,matrix=model.entities(features,device)
                        for rows in candidate_batches(inp):
                            probabilities=torch.sigmoid(model.candidates(identities,matrix,rows)).cpu().numpy()
                            if not np.isfinite(probabilities).all():raise RuntimeError('Nonfinite probability; explicit failure')
                            enumerated+=len(rows)
                            for index in np.flatnonzero(probabilities>=chosen['threshold']):
                                docs[inp['doc_key']].append(dict(record_for(rows[index]),probability=float(probabilities[index])))
                    assert enumerated==expected,inp['window_id'];counts[inp['doc_key']]+=enumerated
                    event({'phase':'source_only_prediction','architecture':name,'seed':seed,
                        'window':inp['window_id'],'enumerated_candidates':enumerated,'test_targets_read':False})
            graph_path=DEST/f'{name}_seed{seed}.json'
            write(graph_path,{'architecture':name,'seed':seed,'epoch':chosen['epoch'],'threshold':chosen['threshold'],
                'test_targets_read':False,'checkpoint_sha256':sha(checkpoint),
                'selection_sha256':freeze['selections'][name]['sha256'],
                'enumerated_candidates_by_paper':counts,'graphs':docs})
            manifest['complete_graphs'][name+'|'+str(seed)]={'path':str(graph_path.relative_to(ROOT)),'sha256':sha(graph_path)}
        del model;gc.collect()
        if device=='mps':torch.mps.empty_cache()
    assert len(manifest['complete_graphs'])==9
    validate_lock();manifest['completed_at_utc']=datetime.now(timezone.utc).isoformat()
    write(DEST/'generation_manifest.json',manifest)
    write(state_path,dict(manifest,status='completed_source_only_generation'))


if __name__=='__main__':
    try:main()
    except BaseException as error:
        path=DEST/'generation_state.json'
        if path.exists():
            value=json.loads(path.read_text());value.update(status='stopped_explicit_failure',error=str(error),
                stopped_at_utc=datetime.now(timezone.utc).isoformat());write(path,value)
        raise
