"""Fixed three-seed nested-span fitting and pooled-development selection only."""
from datetime import datetime,timezone
import gc,json,os,random
import numpy as np
import torch
from torch import nn
from transformers import BertModel
from mulms_local_adapter import ROOT,MODEL_DIR,CACHE,OUTPUT,sha,validate_model,source_cache_path,SpanDetector,span_candidates,span_supervision,span_key,TYPE_IDS
from polyie_local_baseline_training import write

DEST=OUTPUT/'ner'
FREEZE=ROOT/'research/mulms_local_ner_training_freeze.json'
SEEDS=(20261003,20261004,20261005)
EPOCHS=3
THRESHOLDS=(.1,.2,.3,.4,.5,.6,.7,.8,.9,.95,.99)


def event(value):
    row=dict(value,at_utc=datetime.now(timezone.utc).isoformat())
    with (OUTPUT/'training_progress.jsonl').open('a')as stream:stream.write(json.dumps(row)+'\n')
    if row['phase']!='training_update' or row.get('update_number',0)%100==0:print(json.dumps(row),flush=True)


def validate_sources():
    freeze=json.loads(FREEZE.read_text())
    for name,digest in freeze['file_sha256'].items():assert sha(ROOT/name)==digest,name
    validate_model();source_freeze=ROOT/'research/mulms_local_source_cache_freeze.json'
    for name,digest in json.loads(source_freeze.read_text())['file_sha256'].items():assert sha(ROOT/name)==digest,name
    for split in ['train','dev']:
        state=json.loads((CACHE/split/'state.json').read_text())
        inputs=json.loads((ROOT/f'data/mulms/{split}_inputs.json').read_text())
        expected={str(source_cache_path(split,inp).relative_to(ROOT))for inp in inputs}
        assert state['status']=='completed_source_only_cache'and state['sentences']==state['complete_sentences']
        assert state['sentences']==len(inputs)and set(state['cache_sha256'])==expected
        assert state['freeze_sha256']==sha(source_freeze)and state['targets_read']is False
        for name,digest in state['cache_sha256'].items():assert sha(ROOT/name)==digest,name
    return freeze


def load_source(split,inp):
    saved=torch.load(source_cache_path(split,inp),map_location='cpu',weights_only=True)
    assert saved['source_id']==inp['id']and saved['targets_read']is False
    assert saved['freeze_sha256']==sha(ROOT/'research/mulms_local_source_cache_freeze.json')
    return saved


def development_counts(model,inputs,labels,device,seed,epoch):
    papers=sorted({i['doc_key']for i in inputs});counts={str(t):{p:{'tp':0,'fp':0,'fn':0}for p in papers}for t in THRESHOLDS}
    model.eval()
    with torch.no_grad():
        for number,inp in enumerate(inputs):
            saved=load_source('dev',inp);candidates=span_candidates(saved['offsets']);gold={span_key(e)for e in labels[inp['id']]}
            for t in THRESHOLDS:counts[str(t)][inp['doc_key']]['fn']+=len(gold)
            if candidates:
                tokens=model.tokens(saved,device)
                for start in range(0,len(candidates),1024):
                    batch=candidates[start:start+1024]
                    probabilities=torch.sigmoid(model.spans(tokens,batch)).cpu().numpy()
                    if not np.isfinite(probabilities).all():raise RuntimeError('Nonfinite dev span probability; preserve failure')
                    indices=np.argwhere(probabilities>=THRESHOLDS[0])
                    values=np.asarray([probabilities[r,k]for r,k in indices])
                    correct=np.asarray([(batch[r][2],batch[r][3],TYPE_IDS[k])in gold for r,k in indices],dtype=bool)
                    for threshold in THRESHOLDS:
                        included=values>=threshold;tp=int(np.sum(included&correct));fp=int(np.sum(included&~correct))
                        c=counts[str(threshold)][inp['doc_key']];c['tp']+=tp;c['fp']+=fp;c['fn']-=tp
            event({'phase':'full_ner_development_enumeration','seed':seed,'epoch':epoch,
                'sentence':inp['id'],'candidate_spans':len(candidates),'sentence_number':number+1})
    return counts


def select(results):
    assert set(results)=={str(s)for s in SEEDS}and all(set(x)=={'1','2','3'}for x in results.values())
    candidates=[]
    for epoch in range(1,EPOCHS+1):
        for threshold in THRESHOLDS:
            total={k:sum(c[k]for s in SEEDS for c in results[str(s)][str(epoch)]['development_counts'][str(threshold)].values())for k in ['tp','fp','fn']}
            denominator=2*total['tp']+total['fp']+total['fn'];f1=2*total['tp']/denominator if denominator else 0
            candidates.append({'epoch':epoch,'threshold':threshold,'pooled':dict(total,f1=f1)})
    chosen=max(candidates,key=lambda c:(c['pooled']['f1'],-c['pooled']['fp'],c['threshold'],-c['epoch']))
    return {'seed_ids':list(SEEDS),'chosen':chosen,'all_pooled_candidates':candidates,
        'rule':'pooled F1 then fewerFP, higherthreshold, earlier epoch; no seed selection','test_data_read':False}


def main():
    validate_sources();DEST.mkdir(parents=True,exist_ok=True);state_path=DEST/'training_state.json'
    assert not state_path.exists(),'Prior detector run exists; no duplicate/overwrite'
    OUTPUT.mkdir(parents=True,exist_ok=True);torch.set_num_threads(2)
    device='mps'if torch.backends.mps.is_available()else'cpu'
    state={'pid':os.getpid(),'status':'running','device':device,'started_at_utc':datetime.now(timezone.utc).isoformat(),
        'seeds':list(SEEDS),'epochs':EPOCHS,'test_data_read':False,'freeze_sha256':sha(FREEZE)};write(state_path,state)
    results={}
    try:
        inputs={s:json.loads((ROOT/f'data/mulms/{s}_inputs.json').read_text())for s in ['train','dev']}
        labels={s:json.loads((ROOT/f'data/local_baselines/mulms/{s}_ner_gold.json').read_text())for s in ['train','dev']}
        assert all(set(labels[s])=={i['id']for i in inputs[s]}for s in inputs)
        for seed in SEEDS:
            torch.manual_seed(seed);random.seed(seed);rng=np.random.default_rng(seed)
            base=BertModel.from_pretrained(str(MODEL_DIR),local_files_only=True,trust_remote_code=False,use_safetensors=True)
            model=SpanDetector(base.encoder.layer[11]).to(device);del base;gc.collect()
            optimizer=torch.optim.AdamW(model.parameters(),lr=2e-4,weight_decay=.01,foreach=False);results[str(seed)]={}
            for epoch in range(1,EPOCHS+1):
                model.train();updates=0;empty=0;losses=[];positive_types=0;selected_spans=0
                for number in rng.permutation(len(inputs['train'])):
                    inp=inputs['train'][int(number)];saved=load_source('train',inp);candidates=span_candidates(saved['offsets'])
                    if not candidates:
                        empty+=1;event({'phase':'source_only_no_update','seed':seed,'epoch':epoch,'sentence':inp['id']});continue
                    selected,target=span_supervision(candidates,labels['train'][inp['id']],rng)
                    tokens=model.tokens(saved,device);logits=model.spans(tokens,[candidates[i]for i in selected])
                    loss=nn.functional.binary_cross_entropy_with_logits(logits,torch.from_numpy(target).to(device))
                    if not torch.isfinite(loss):raise RuntimeError('Nonfinite NER fitting loss; preserve failure')
                    optimizer.zero_grad();loss.backward();optimizer.step();updates+=1;losses.append(float(loss.detach().cpu()))
                    positive_types+=int(target.sum());selected_spans+=len(selected)
                    event({'phase':'training_update','model':'ner','seed':seed,'epoch':epoch,'sentence':inp['id'],
                        'selected_spans':len(selected),'positive_type_targets':int(target.sum()),'loss':losses[-1],'update_number':updates})
                cp=DEST/f'seed{seed}'/f'epoch{epoch}.pt';cp.parent.mkdir(parents=True,exist_ok=True)
                torch.save({'state_dict':model.state_dict(),'seed':seed,'epoch':epoch,'freeze_sha256':sha(FREEZE)},cp)
                counts=development_counts(model,inputs['dev'],labels['dev'],device,seed,epoch)
                results[str(seed)][str(epoch)]={'updates':updates,'source_only_no_update_sentences':empty,
                    'positive_type_targets':positive_types,'selected_training_spans':selected_spans,'mean_sentence_loss':float(np.mean(losses)),
                    'checkpoint':str(cp.relative_to(ROOT)),'checkpoint_sha256':sha(cp),'development_counts':counts}
                write(DEST/'development_all_epochs.json',results)
                event({'phase':'ner_epoch_completed','seed':seed,'epoch':epoch,'updates':updates})
            del optimizer,model;gc.collect()
            if device=='mps':torch.mps.empty_cache()
        choice=select(results);choice['training_freeze_sha256']=sha(FREEZE)
        path=DEST/'immutable_development_selection.json';assert not path.exists();write(path,choice)
        state.update(status='completed',completed_at_utc=datetime.now(timezone.utc).isoformat(),selection_sha256=sha(path));write(state_path,state)
        event({'phase':'ner_fitting_and_pooled_selection_completed','selection':choice['chosen']})
    except BaseException as error:
        state.update(status='stopped_explicit_failure',error=str(error),stopped_at_utc=datetime.now(timezone.utc).isoformat());write(state_path,state);raise


if __name__=='__main__':main()
