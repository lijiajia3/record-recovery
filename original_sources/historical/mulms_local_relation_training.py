"""Three prospective relation architectures; complete pooled dev selection.

Gold spans are allowed for train supervision only. Every dev source receives
the same locked predicted entities for a seed, including entity-free sentences.
"""
from datetime import datetime,timezone
import gc,json,os,random
import numpy as np
import torch
from torch import nn
from transformers import BertModel
from mulms_local_adapter import ROOT,MODEL_DIR,OUTPUT,sha,span_key,span_candidates,TYPE_IDS,RELATION_IDS
from mulms_local_ner_training import load_source,SEEDS,validate_sources
from mulms_local_ner_development_generation import LOCK as NER_GRAPH_FREEZE,GRAPHS as NER_GRAPHS,completed_detector
from mulms_relation_adapter import RelationHead,pair_candidates,pair_supervision
from mulms_experiment import canonical,valid_key
from polyie_local_baseline_training import write

DEST=OUTPUT/'relations';FREEZE=ROOT/'research/mulms_local_relation_training_freeze.json'
ARCHITECTURES=('mean','typed','capacity_mean');EPOCHS=3
THRESHOLDS=(.5,.75,.9,.95,.975,.99,.995,.999,.9995,.9999)


def event(value):
    row=dict(value,at_utc=datetime.now(timezone.utc).isoformat())
    with (OUTPUT/'training_progress.jsonl').open('a')as stream:stream.write(json.dumps(row)+'\n')
    if row['phase']!='relation_training_update' or row.get('update_number',0)%250==0:print(json.dumps(row),flush=True)


def predicted_development_entities():
    manifest=json.loads((NER_GRAPHS/'generation_manifest.json').read_text())
    assert manifest['status']=='all_three_complete'and manifest['annotation_content_used']is False
    assert manifest['freeze_sha256']==sha(NER_GRAPH_FREEZE)and set(manifest['seeds'])=={str(s)for s in SEEDS}
    inputs=json.loads((ROOT/'data/mulms/dev_inputs.json').read_text());expected={i['id']for i in inputs};result={}
    choice,checkpoints=completed_detector();population={}
    for inp in inputs:
        saved=load_source('dev',inp);candidates=span_candidates(saved['offsets'])
        population[inp['id']]=(len(candidates),{(c[2],c[3])for c in candidates},saved['offsets'])
    for seed in SEEDS:
        entry=manifest['seeds'][str(seed)];path=ROOT/entry['path'];assert sha(path)==entry['sha256']
        graph=json.loads(path.read_text());assert set(graph['graphs'])==expected==set(graph['candidate_span_counts'])
        assert graph['seed']==seed and graph['chosen']==choice['chosen']and graph['annotation_content_used']is False
        assert graph['source_input_sha256']==sha(ROOT/'data/mulms/dev_inputs.json')and graph['freeze_sha256']==sha(NER_GRAPH_FREEZE)
        assert graph['checkpoint_sha256']==checkpoints[str(seed)][str(choice['chosen']['epoch'])]['checkpoint_sha256']
        for inp in inputs:
            count,boundaries,offsets=population[inp['id']];entities=graph['graphs'][inp['id']]
            assert graph['candidate_span_counts'][inp['id']]==count
            assert len(entities)==len({span_key(e)for e in entities})
            for entity in entities:
                assert (entity['start'],entity['end'])in boundaries and entity['type']in TYPE_IDS
                assert any(a<entity['end']and b>entity['start']for a,b in offsets),'Predicted dev mention has no source WordPiece'
                assert entity['text']==inp['text'][entity['start']:entity['end']]and entity['text']
                assert np.isfinite(entity['probability'])and choice['chosen']['threshold']<=entity['probability']<=1
        result[seed]=graph['graphs']
    return result


def validate_freeze():
    freeze=json.loads(FREEZE.read_text())
    for name,digest in freeze['file_sha256'].items():assert sha(ROOT/name)==digest,name
    validate_sources()
    for name,digest in json.loads(NER_GRAPH_FREEZE.read_text())['file_sha256'].items():assert sha(ROOT/name)==digest,name
    return freeze


def development_counts(model,inputs,labels,predicted_entities,device,name,seed,epoch):
    papers=sorted({i['doc_key']for i in inputs})
    counts={str(t):{p:{'tp':0,'fp':0,'fn':0}for p in papers}for t in THRESHOLDS};model.eval()
    with torch.no_grad():
        for number,inp in enumerate(inputs):
            entities=predicted_entities[inp['id']];pairs=pair_candidates(entities)
            gold={canonical(r,inp['text'])for r in labels[inp['id']]};assert all(valid_key(k)for k in gold)
            for threshold in THRESHOLDS:counts[str(threshold)][inp['doc_key']]['fn']+=len(gold)
            if pairs:
                saved=load_source('dev',inp);matrix=model.entities(saved,entities,device)
                for start in range(0,len(pairs),2048):
                    batch=pairs[start:start+2048]
                    probabilities=torch.sigmoid(model.pairs(matrix,entities,batch,len(inp['text']))).cpu().numpy()
                    if not np.isfinite(probabilities).all():raise RuntimeError('Nonfinite full-population dev relation probability')
                    indices=np.argwhere(probabilities>=THRESHOLDS[0]);values=np.asarray([probabilities[r,k]for r,k in indices])
                    correct=np.asarray([(span_key(entities[batch[r][0]]),span_key(entities[batch[r][1]]),RELATION_IDS[k])in gold
                        for r,k in indices],dtype=bool)
                    for threshold in THRESHOLDS:
                        included=values>=threshold;tp=int(np.sum(included&correct));fp=int(np.sum(included&~correct))
                        c=counts[str(threshold)][inp['doc_key']];c['tp']+=tp;c['fp']+=fp;c['fn']-=tp
            event({'phase':'full_relation_development_enumeration','architecture':name,'seed':seed,'epoch':epoch,
                'sentence_number':number+1,'candidate_pairs':len(pairs),'predicted_entities':len(entities)})
    return counts


def select(results):
    assert set(results)=={str(s)for s in SEEDS}and all(set(x)=={'1','2','3'}for x in results.values())
    candidates=[]
    for epoch in range(1,EPOCHS+1):
        for threshold in THRESHOLDS:
            total={k:sum(c[k]for s in SEEDS for c in results[str(s)][str(epoch)]['development_counts'][str(threshold)].values())for k in ['tp','fp','fn']}
            denom=2*total['tp']+total['fp']+total['fn']
            candidates.append({'epoch':epoch,'threshold':threshold,'pooled':dict(total,f1=2*total['tp']/denom if denom else 0)})
    chosen=max(candidates,key=lambda c:(c['pooled']['f1'],-c['pooled']['fp'],c['threshold'],-c['epoch']))
    return {'chosen':chosen,'all_pooled_candidates':candidates,'seed_ids':list(SEEDS),
        'rule':'pooled strict typed-character relation F1, fewer FP, higher threshold, earlier epoch; all seeds retained',
        'predicted_entities_used_at_development':True,'test_data_read':False}


def main():
    validate_freeze();assert not DEST.exists(),'Prior relation run exists; no overwrite'
    DEST.mkdir(parents=True);state_path=DEST/'training_state.json';torch.set_num_threads(2)
    device='mps'if torch.backends.mps.is_available()else'cpu';state={'pid':os.getpid(),'status':'running',
        'started_at_utc':datetime.now(timezone.utc).isoformat(),'freeze_sha256':sha(FREEZE),'device':device,
        'seeds':list(SEEDS),'architectures':list(ARCHITECTURES),'epochs_per_seed':EPOCHS,'test_data_read':False};write(state_path,state)
    try:
        inputs={s:json.loads((ROOT/f'data/mulms/{s}_inputs.json').read_text())for s in ['train','dev']}
        labels={s:json.loads((ROOT/f'data/mulms/{s}_gold.json').read_text())for s in ['train','dev']}
        entities=json.loads((ROOT/'data/local_baselines/mulms/train_ner_gold.json').read_text())
        entities={sid:sorted(values,key=span_key)for sid,values in entities.items()}
        assert set(entities)=={i['id']for i in inputs['train']}
        assert all(set(labels[s])=={i['id']for i in inputs[s]}for s in inputs)
        predicted=predicted_development_entities();all_choices={}
        for name in ARCHITECTURES:
            results={};directory=DEST/name;directory.mkdir()
            for seed in SEEDS:
                torch.manual_seed(seed);random.seed(seed);rng=np.random.default_rng(seed)
                base=BertModel.from_pretrained(str(MODEL_DIR),local_files_only=True,trust_remote_code=False,use_safetensors=True)
                model=RelationHead(base.encoder.layer[11],name).to(device);del base;gc.collect()
                optimizer=torch.optim.AdamW(model.parameters(),lr=2e-4,weight_decay=.01,foreach=False);results[str(seed)]={}
                for epoch in range(1,EPOCHS+1):
                    model.train();updates=0;empty=0;losses=[];positives=0;sampled=0
                    for number in rng.permutation(len(inputs['train'])):
                        inp=inputs['train'][int(number)];entity=entities[inp['id']]
                        if not entity:
                            assert not labels['train'][inp['id']],'Gold relation present without supervised entities'
                            empty+=1;event({'phase':'relation_no_candidate_pair','architecture':name,'seed':seed,'epoch':epoch,'sentence':inp['id']});continue
                        pairs,target=pair_supervision(entity,labels['train'][inp['id']],inp['text'],rng)
                        saved=load_source('train',inp);matrix=model.entities(saved,entity,device,allow_empty_source=True)
                        logits=model.pairs(matrix,entity,pairs,len(inp['text']))
                        loss=nn.functional.binary_cross_entropy_with_logits(logits,torch.from_numpy(target).to(device))
                        if not torch.isfinite(loss):raise RuntimeError('Nonfinite relation fitting loss; preserve failure')
                        optimizer.zero_grad();loss.backward();optimizer.step();updates+=1;losses.append(float(loss.detach().cpu()))
                        positives+=int(target.sum());sampled+=len(pairs)
                        event({'phase':'relation_training_update','architecture':name,'seed':seed,'epoch':epoch,
                            'sentence':inp['id'],'update_number':updates,'selected_pairs':len(pairs),'positive_labels':int(target.sum()),'loss':losses[-1]})
                    checkpoint_path=directory/f'seed{seed}'/f'epoch{epoch}.pt';checkpoint_path.parent.mkdir(parents=True,exist_ok=True)
                    torch.save({'state_dict':model.state_dict(),'architecture':name,'seed':seed,'epoch':epoch,'freeze_sha256':sha(FREEZE)},checkpoint_path)
                    counts=development_counts(model,inputs['dev'],labels['dev'],predicted[seed],device,name,seed,epoch)
                    results[str(seed)][str(epoch)]={'updates':updates,'no_candidate_pair_sentences':empty,'selected_training_pairs':sampled,
                        'positive_relation_label_targets':positives,'mean_sentence_loss':float(np.mean(losses)),
                        'checkpoint':str(checkpoint_path.relative_to(ROOT)),'checkpoint_sha256':sha(checkpoint_path),'development_counts':counts}
                    write(directory/'development_all_epochs.json',results);event({'phase':'relation_epoch_completed','architecture':name,'seed':seed,'epoch':epoch,'updates':updates})
                del optimizer,model;gc.collect()
                if device=='mps':torch.mps.empty_cache()
            choice=select(results);choice.update(training_freeze_sha256=sha(FREEZE),architecture=name)
            path=directory/'immutable_development_selection.json';assert not path.exists();write(path,choice)
            all_choices[name]={'path':str(path.relative_to(ROOT)),'sha256':sha(path)}
            event({'phase':'relation_architecture_completed','architecture':name,'choice':choice['chosen']})
        assert set(all_choices)==set(ARCHITECTURES)
        write(DEST/'complete_selections.json',{'status':'all_three_architectures_complete','selections':all_choices,'freeze_sha256':sha(FREEZE),'test_data_read':False})
        state.update(status='completed_all_three_architectures',completed_at_utc=datetime.now(timezone.utc).isoformat());write(state_path,state)
    except BaseException as error:
        state.update(status='stopped_explicit_failure',error=str(error),stopped_at_utc=datetime.now(timezone.utc).isoformat());write(state_path,state);raise


if __name__=='__main__':main()
