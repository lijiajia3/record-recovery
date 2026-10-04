"""Prospective role/position head ablation, derived from frozen mean/control fitting."""
from datetime import datetime, timezone
import gc
import json
import os
from pathlib import Path
import random

import numpy as np
import torch
from torch import nn
from transformers import BertModel

from polyie_local_adapter import (ROOT, MODEL_DIR, CACHE, ROLES, sha, cache_path,
    population_count, ordered_roles, is_owner, candidate_batches, negative_sample,
    record_for, key, validate_model)

OUTPUT=ROOT/'results/local_baseline/polyie_typed_interaction_v1'
FREEZE=ROOT/'research/polyie_typed_training_freeze.json'
SEEDS=(20261003,20261004,20261005)
THRESHOLDS=(.5,.75,.9,.95,.975,.99,.995,.999,.9995,.9999)
EPOCHS=3


def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix('.tmp');temp.write_text(json.dumps(value,indent=2)+'\n');temp.replace(path)


def event(value):
    row=dict(value,at_utc=datetime.now(timezone.utc).isoformat())
    with (OUTPUT/'training_progress.jsonl').open('a')as stream:stream.write(json.dumps(row)+'\n')
    print(json.dumps(row),flush=True)


class UnusedOriginalClassifier(nn.Module):
    def __init__(self,last_layer):
        super().__init__();self.last_layer=last_layer
        self.norm=nn.LayerNorm(768);self.dropout=nn.Dropout(.1);self.head=nn.Linear(768,1)

    def entities(self,saved,device):
        outputs=[];offsets=[];current=0
        for chunk in saved['chunks']:
            offsets.append(current)
            hidden=self.last_layer(chunk['hidden'].to(device),attention_mask=chunk['mask'].to(device))[0][0]
            outputs.append(hidden);current+=hidden.shape[0]
        identities=sorted(saved['positions'])
        pool=np.zeros((len(identities),current),dtype=np.float32)
        for row,identity in enumerate(identities):
            for chunk,position,weight in saved['positions'][identity]:pool[row,offsets[chunk]+position]+=weight
        matrix=torch.from_numpy(pool).to(device)@torch.cat(outputs,dim=0)
        return identities,matrix

    def candidates(self,identities,matrix,rows):
        index={identity:i+1 for i,identity in enumerate(identities)}
        indices=np.zeros((len(rows),4),dtype=np.int64)
        sizes=np.zeros((len(rows),1),dtype=np.float32)
        for number,ids in enumerate(rows):
            sizes[number]=len(ids)
            for role,identity in enumerate(ids):indices[number,role]=index[identity]
        full=torch.cat([matrix.new_zeros((1,768)),matrix],dim=0)
        means=full[torch.from_numpy(indices).to(matrix.device)].sum(dim=1)/torch.from_numpy(sizes).to(matrix.device)
        return self.head(self.dropout(self.norm(means))).flatten()


from polyie_typed_classifier import TypedClassifier as Classifier

def owner_positives(inp,labels):
    entities={e['id']:e for e in inp['entities']};rows=set()
    for relation in labels[inp['doc_key']]['eligible']:
        if not all(len(relation[r])==1 for r in ROLES[:3])or len(relation['Condition'])>1:continue
        ids=tuple(str(i)for role in ROLES for i in relation[role])
        if all(i in entities for i in ids)and is_owner(inp,[entities[i]for i in ids]):rows.add(ids)
    return rows


def validate_sources():
    freeze=json.loads(FREEZE.read_text())
    for name,expected in freeze['file_sha256'].items():assert sha(ROOT/name)==expected,name
    validate_model()
    source_freeze=ROOT/'research/polyie_local_source_cache_freeze.json'
    for name,expected in json.loads(source_freeze.read_text())['file_sha256'].items():assert sha(ROOT/name)==expected,name
    for split in('train','dev'):
        state=json.loads((CACHE/split/'state.json').read_text())
        assert state['status']=='completed_source_cache'and state['complete_windows']==state['windows']
        assert state['freeze_sha256']==sha(source_freeze)
        for name,expected in state['cache_sha256'].items():assert sha(ROOT/name)==expected,name
    return freeze


def load_window(split,inp):
    saved=torch.load(cache_path(split,inp),map_location='cpu',weights_only=True)
    assert saved['candidate_population']==population_count(inp)
    return saved


def development_counts(model,inputs,labels,device,seed,epoch):
    """Full source population; supplied labels enter only the scoring comparator."""
    docs=sorted({inp['doc_key']for inp in inputs})
    gold={doc:{key(r)for r in labels[doc]['eligible']}for doc in docs}
    counts={str(t):{doc:{'tp':0,'fp':0,'fn':len(gold[doc])}for doc in docs}for t in THRESHOLDS}
    model.eval()
    with torch.no_grad():
        for inp in inputs:
            expected=population_count(inp);enumerated=0
            if expected:
                model.bind_source(inp)
                identities,matrix=model.entities(load_window('dev',inp),device)
                for rows in candidate_batches(inp):
                    probabilities=torch.sigmoid(model.candidates(identities,matrix,rows)).cpu().numpy()
                    if not np.isfinite(probabilities).all():raise RuntimeError('Nonfinite candidate probability; no silent empty prediction')
                    enumerated+=len(rows)
                    qualifying=np.flatnonzero(probabilities>=THRESHOLDS[0])
                    scores=probabilities[qualifying]
                    correct=np.asarray([key(record_for(rows[i]))in gold[inp['doc_key']]for i in qualifying],dtype=bool)
                    for threshold in THRESHOLDS:
                        included=scores>=threshold;tp=int(np.sum(included&correct));fp=int(np.sum(included&~correct))
                        c=counts[str(threshold)][inp['doc_key']];c['tp']+=tp;c['fp']+=fp;c['fn']-=tp
            assert enumerated==expected,inp['window_id']
            event({'phase':'full_development_scoring','seed':seed,'epoch':epoch,
                   'window':inp['window_id'],'enumerated_candidates':enumerated})
    return counts


def full_document_sensitivity(counts,labels):
    result={}
    for threshold,docs in counts.items():
        result[threshold]={}
        for doc,c in docs.items():
            eligible={key(r)for r in labels[doc]['eligible']}
            full={key(r)for r in labels[doc]['all_valid']}
            assert eligible<=full
            # Every source candidate is in a fixed covered window; an uncovered
            # full-document gold group cannot match this adapter's predictions.
            result[threshold][doc]=dict(c,fn=c['fn']+len(full-eligible))
    return result


def select(all_results):
    assert set(all_results)=={str(seed)for seed in SEEDS}
    choices=[]
    for epoch in range(1,EPOCHS+1):
        for threshold in THRESHOLDS:
            tp=fp=fn=0
            for seed in SEEDS:
                rows=all_results[str(seed)][str(epoch)]['development_counts'][str(threshold)]
                for c in rows.values():tp+=c['tp'];fp+=c['fp'];fn+=c['fn']
            f1=2*tp/(2*tp+fp+fn)if 2*tp+fp+fn else 0
            choices.append({'epoch':epoch,'threshold':threshold,'tp':tp,'fp':fp,'fn':fn,'f1':f1})
    selected=max(choices,key=lambda x:(x['f1'],-x['fp'],x['threshold'],-x['epoch']))
    return {'chosen':selected,'all_candidates':choices,'selection_rule':'pooled3seeds F1, fewerFP, higherthreshold, earlierepoch',
            'seed_ids':list(SEEDS),'test_targets_read':False}


def main():
    validate_sources();OUTPUT.mkdir(parents=True,exist_ok=True);state_path=OUTPUT/'training_state.json'
    if state_path.exists():
        previous=json.loads(state_path.read_text())
        if previous['status']=='completed':raise RuntimeError('Frozen complete training already exists; no duplicate fitting.')
        try:os.kill(previous['pid'],0)
        except ProcessLookupError:raise RuntimeError('Prior interrupted training needs an explicit audited recovery, not a silent rerun.')
        else:raise RuntimeError('Training already live; no duplicate fitting.')
    torch.set_num_threads(2);device='mps'if torch.backends.mps.is_available()else'cpu'
    state={'pid':os.getpid(),'started_at_utc':datetime.now(timezone.utc).isoformat(),'status':'running',
        'device':device,'seeds':list(SEEDS),'epochs':EPOCHS,'test_files_read':False,'freeze_sha256':sha(FREEZE)}
    write(state_path,state)
    train_inputs=json.loads((ROOT/'data/polyie/train_inputs.json').read_text())
    dev_inputs=json.loads((ROOT/'data/polyie/dev_inputs.json').read_text())
    train_labels=json.loads((ROOT/'data/polyie/train_gold.json').read_text())
    dev_labels=json.loads((ROOT/'data/polyie/dev_gold.json').read_text())
    all_results={}
    try:
        for seed in SEEDS:
            torch.manual_seed(seed);random.seed(seed);rng=np.random.default_rng(seed)
            base=BertModel.from_pretrained(str(MODEL_DIR),local_files_only=True,trust_remote_code=False,use_safetensors=True)
            model=Classifier(base.encoder.layer[11]).to(device);del base;gc.collect()
            optimizer=torch.optim.AdamW(model.parameters(),lr=2e-4,weight_decay=.01,foreach=False)
            all_results[str(seed)]={}
            for epoch in range(1,EPOCHS+1):
                model.train();losses=[];updates=0;no_update=0;positive_total=0;negative_total=0
                for number in rng.permutation(len(train_inputs)):
                    inp=train_inputs[int(number)];population=population_count(inp)
                    if not population:
                        no_update+=1;event({'phase':'source_only_no_update','seed':seed,'epoch':epoch,'window':inp['window_id']});continue
                    positive=owner_positives(inp,train_labels)
                    negative=negative_sample(inp,positive,10*max(1,len(positive)),rng)
                    rows=sorted(positive)+negative
                    assert rows,'Nonempty owner-population produced empty BCE'
                    model.bind_source(inp)
                    identities,matrix=model.entities(load_window('train',inp),device)
                    logits=model.candidates(identities,matrix,rows)
                    target=torch.tensor([1.]*len(positive)+[0.]*len(negative),device=device)
                    loss=nn.functional.binary_cross_entropy_with_logits(logits,target)
                    if not torch.isfinite(loss):raise RuntimeError('Nonfinite training loss; preserve explicit failure')
                    optimizer.zero_grad();loss.backward();optimizer.step()
                    losses.append(float(loss.detach().cpu()));updates+=1
                    positive_total+=len(positive);negative_total+=len(negative)
                    event({'phase':'training_update','seed':seed,'epoch':epoch,'window':inp['window_id'],
                           'positives':len(positive),'negatives':len(negative),'loss':losses[-1]})
                checkpoint=OUTPUT/f'seed{seed}'/f'epoch{epoch}.pt';checkpoint.parent.mkdir(exist_ok=True)
                torch.save({'state_dict':model.state_dict(),'seed':seed,'epoch':epoch,'freeze_sha256':sha(FREEZE)},checkpoint)
                dev_counts=development_counts(model,dev_inputs,dev_labels,device,seed,epoch)
                row={'updates':updates,'source_only_no_update_windows':no_update,'positive_examples':positive_total,
                     'negative_examples':negative_total,'mean_window_loss':float(np.mean(losses)),
                     'checkpoint':str(checkpoint.relative_to(ROOT)),'checkpoint_sha256':sha(checkpoint),
                     'development_counts':dev_counts,
                     'full_document_all_valid_sensitivity':full_document_sensitivity(dev_counts,dev_labels)}
                all_results[str(seed)][str(epoch)]=row;write(OUTPUT/'development_all_epochs.json',all_results)
                event({'phase':'epoch_completed','seed':seed,'epoch':epoch,'updates':updates,'no_update_windows':no_update})
            del optimizer,model;gc.collect()
            if device=='mps':torch.mps.empty_cache()
        selection=select(all_results);selection['training_freeze_sha256']=sha(FREEZE)
        selection_path=OUTPUT/'immutable_development_selection.json'
        assert not selection_path.exists();write(selection_path,selection)
        state.update(status='completed',completed_at_utc=datetime.now(timezone.utc).isoformat(),
            selection_sha256=sha(selection_path));write(state_path,state)
        event({'phase':'training_and_pooled_selection_completed','selection':selection['chosen']})
    except BaseException as error:
        state.update(status='stopped_explicit_failure',error=str(error),stopped_at_utc=datetime.now(timezone.utc).isoformat())
        write(state_path,state);raise


if __name__=='__main__':main()
