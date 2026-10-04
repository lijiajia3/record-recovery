"""Source-only test cache, then all three detectors and all nine edge graphs."""
from datetime import datetime,timezone
import gc,json,os
import numpy as np
import torch
from transformers import BertModel,AutoTokenizer
from mulms_neural_family import ROOT,FREEZE_REL,DEST_REL,SEEDS,ARCHITECTURES,validate_lock,load_test_cache,validate_entities
from mulms_local_adapter import MODEL_DIR,CACHE,OUTPUT,sha,source_encoding,source_cache_path,SpanDetector,span_candidates,entities_from_probs
from mulms_relation_adapter import RelationHead,pair_candidates,records_from_probabilities
from polyie_local_baseline_training import write


def event(value):
    row=dict(value,at_utc=datetime.now(timezone.utc).isoformat())
    with (OUTPUT/'generation_progress.jsonl').open('a')as stream:stream.write(json.dumps(row)+'\n')
    if row.get('sentence_number',0)%100==0:print(json.dumps(row),flush=True)


def prepare_cache(inputs,device):
    dest=CACHE/'test';assert not dest.exists(),'Prior test cache exists; no overwrite or partial-as-complete reuse'
    dest.mkdir();state_path=dest/'state.json';state={'pid':os.getpid(),'status':'running_source_only_cache','split':'test',
        'created_at_utc':datetime.now(timezone.utc).isoformat(),'targets_read':False,'freeze_sha256':sha(ROOT/FREEZE_REL),
        'sentences':len(inputs),'cache_sha256':{}};write(state_path,state)
    try:
        tokenizer=AutoTokenizer.from_pretrained(str(MODEL_DIR),local_files_only=True,trust_remote_code=False,use_fast=True)
        base=BertModel.from_pretrained(str(MODEL_DIR),local_files_only=True,trust_remote_code=False,use_safetensors=True).to(device).eval()
        with torch.no_grad():
            for number,inp in enumerate(inputs):
                encoded=source_encoding(inp,tokenizer);chunks=[]
                for chunk in encoded['chunks']:
                    ids=torch.tensor([chunk['ids']],device=device);mask=base.get_extended_attention_mask(torch.ones_like(ids),ids.shape)
                    hidden=base.embeddings(input_ids=ids,token_type_ids=torch.zeros_like(ids))
                    for layer in base.encoder.layer[:11]:hidden=layer(hidden,attention_mask=mask)[0]
                    chunks.append({'start':chunk['start'],'end':chunk['end'],'hidden':hidden.cpu().float(),'mask':mask.cpu().float()})
                path=source_cache_path('test',inp);torch.save({'source_id':inp['id'],'wordpieces':encoded['wordpieces'],
                    'offsets':encoded['offsets'],'chunks':chunks,'targets_read':False,'freeze_sha256':sha(ROOT/FREEZE_REL)},path)
                state['cache_sha256'][str(path.relative_to(ROOT))]=sha(path)
                event({'phase':'source_only_supervised_test_cache','sentence_number':number+1,'wordpieces':encoded['wordpieces']})
                if(number+1)%100==0:write(state_path,state)
        state.update(status='completed_source_only_cache',complete_sentences=len(inputs),completed_at_utc=datetime.now(timezone.utc).isoformat());write(state_path,state)
        del base;gc.collect()
        if device=='mps':torch.mps.empty_cache()
    except BaseException as error:
        state.update(status='stopped_explicit_failure',error=str(error));write(state_path,state);raise


def load_fitted(kind,seed,checkpoint,chosen,device):
    assert sha(ROOT/checkpoint['path'])==checkpoint['sha256']
    base=BertModel.from_pretrained(str(MODEL_DIR),local_files_only=True,trust_remote_code=False,use_safetensors=True)
    model=SpanDetector(base.encoder.layer[11])if kind=='ner'else RelationHead(base.encoder.layer[11],kind)
    del base;gc.collect();saved=torch.load(ROOT/checkpoint['path'],map_location='cpu',weights_only=True)
    assert saved['seed']==seed and saved['epoch']==chosen['epoch']
    parent='research/mulms_local_ner_training_freeze.json'if kind=='ner'else'research/mulms_local_relation_training_freeze.json'
    assert saved['freeze_sha256']==sha(ROOT/parent)
    if kind!='ner':assert saved['architecture']==kind
    model.load_state_dict(saved['state_dict']);del saved;return model.to(device).eval()


def main():
    freeze=validate_lock();dest=ROOT/DEST_REL;assert not dest.exists(),'Prior complete/partial family exists; no overwrite'
    dest.mkdir();state_path=dest/'generation_state.json';state={'pid':os.getpid(),'status':'running_source_only_generation',
        'started_at_utc':datetime.now(timezone.utc).isoformat(),'annotation_content_used':False,'freeze_sha256':sha(ROOT/FREEZE_REL)};write(state_path,state)
    try:
        inputs=json.loads((ROOT/'data/mulms/test_inputs.json').read_text())
        assert len(inputs)==1114 and len({i['id']for i in inputs})==1114 and all(set(i)=={'id','doc_key','text'}for i in inputs)
        device='mps'if torch.backends.mps.is_available()else'cpu';torch.set_num_threads(2);prepare_cache(inputs,device)
        manifest={'status':'all_three_detector_and_nine_relation_graphs_complete','freeze_sha256':sha(ROOT/FREEZE_REL),
            'annotation_content_used':False,'source_sentences':len(inputs),'paper_ids':sorted({i['doc_key']for i in inputs}),
            'detector_graphs':{},'relation_graphs':{name:{}for name in ARCHITECTURES}}
        detector_chosen=freeze['detector_selection']['chosen']
        for seed in SEEDS:
            detector=load_fitted('ner',seed,freeze['detector_checkpoints'][str(seed)],detector_chosen,device)
            ner_graphs={};span_counts={}
            with torch.no_grad():
                for number,inp in enumerate(inputs):
                    saved=load_test_cache(inp);candidates=span_candidates(saved['offsets']);entities=[]
                    if candidates:
                        tokens=detector.tokens(saved,device)
                        for start in range(0,len(candidates),1024):
                            batch=candidates[start:start+1024];probabilities=torch.sigmoid(detector.spans(tokens,batch)).cpu().numpy()
                            entities+=entities_from_probs(inp,batch,probabilities,detector_chosen['threshold'])
                    validate_entities(inp,entities,candidates,saved['offsets'],detector_chosen['threshold'])
                    ner_graphs[inp['id']]=sorted(entities,key=lambda e:(e['start'],e['end'],e['type']));span_counts[inp['id']]=len(candidates)
                    event({'phase':'full_source_only_test_detector','seed':seed,'sentence_number':number+1,
                        'candidate_spans':len(candidates),'predicted_entities':len(entities)})
            ner_path=dest/f'detector_seed{seed}.json';write(ner_path,{'seed':seed,'chosen':detector_chosen,'graphs':ner_graphs,
                'candidate_span_counts':span_counts,'annotation_content_used':False,'source_input_sha256':sha(ROOT/'data/mulms/test_inputs.json'),
                'freeze_sha256':sha(ROOT/FREEZE_REL),'checkpoint_sha256':freeze['detector_checkpoints'][str(seed)]['sha256']})
            manifest['detector_graphs'][str(seed)]={'path':str(ner_path.relative_to(ROOT)),'sha256':sha(ner_path)}
            del detector;gc.collect()
            if device=='mps':torch.mps.empty_cache()
            for architecture in ARCHITECTURES:
                chosen=freeze['relation_selections'][architecture]['chosen'];checkpoint=freeze['relation_checkpoints'][architecture][str(seed)]
                model=load_fitted(architecture,seed,checkpoint,chosen,device);graphs={};pair_counts={};label_counts={}
                with torch.no_grad():
                    for number,inp in enumerate(inputs):
                        entities=ner_graphs[inp['id']];pairs=pair_candidates(entities);records=[]
                        if pairs:
                            saved=load_test_cache(inp);matrix=model.entities(saved,entities,device)
                            for start in range(0,len(pairs),2048):
                                batch=pairs[start:start+2048];probabilities=torch.sigmoid(model.pairs(matrix,entities,batch,len(inp['text']))).cpu().numpy()
                                records+=records_from_probabilities(entities,batch,probabilities,inp['text'],chosen['threshold'])
                        graphs[inp['id']]=records;pair_counts[inp['id']]=len(pairs);label_counts[inp['id']]=len(pairs)*15
                        event({'phase':'full_source_only_test_relations','architecture':architecture,'seed':seed,
                            'sentence_number':number+1,'candidate_pairs':len(pairs),'candidate_labels':len(pairs)*15,'predicted_edges':len(records)})
                path=dest/f'{architecture}_seed{seed}.json';write(path,{'architecture':architecture,'seed':seed,'chosen':chosen,'graphs':graphs,
                    'candidate_pair_counts':pair_counts,'candidate_label_counts':label_counts,'annotation_content_used':False,
                    'source_input_sha256':sha(ROOT/'data/mulms/test_inputs.json'),'checkpoint_sha256':checkpoint['sha256'],
                    'freeze_sha256':sha(ROOT/FREEZE_REL),'detector_graph_sha256':sha(ner_path)})
                manifest['relation_graphs'][architecture][str(seed)]={'path':str(path.relative_to(ROOT)),'sha256':sha(path)}
                del model;gc.collect()
                if device=='mps':torch.mps.empty_cache()
        assert len(manifest['detector_graphs'])==3 and all(len(v)==3 for v in manifest['relation_graphs'].values())
        write(dest/'generation_manifest.json',manifest)
        state.update(status='completed_all_three_detector_and_nine_relation_graphs',completed_at_utc=datetime.now(timezone.utc).isoformat());write(state_path,state)
        print(json.dumps(state))
    except BaseException as error:
        state.update(status='stopped_explicit_failure',error=str(error),stopped_at_utc=datetime.now(timezone.utc).isoformat());write(state_path,state);raise


if __name__=='__main__':main()
