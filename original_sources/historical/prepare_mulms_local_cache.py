"""Frozen source-only encoder cache for train/development; no annotation reads."""
from datetime import datetime,timezone
import argparse,json,os
import torch
from transformers import BertModel,AutoTokenizer
from mulms_local_adapter import ROOT,MODEL_DIR,CACHE,sha,validate_model,source_encoding,source_cache_path
from polyie_local_baseline_training import write

FREEZE=ROOT/'research/mulms_local_source_cache_freeze.json'


def main(split):
    assert split in ['train','dev']
    freeze=json.loads(FREEZE.read_text())
    for name,digest in freeze['file_sha256'].items():assert sha(ROOT/name)==digest,name
    validate_model();dest=CACHE/split;dest.mkdir(parents=True,exist_ok=True);state_path=dest/'state.json'
    assert not state_path.exists(),'Prior cache state exists; audit rather than duplicate/overwrite'
    inputs=json.loads((ROOT/f'data/mulms/{split}_inputs.json').read_text())
    assert all(set(inp)=={'id','doc_key','text'}for inp in inputs)
    assert len({source_cache_path(split,inp)for inp in inputs})==len(inputs)
    state={'pid':os.getpid(),'status':'running_source_only_cache','split':split,'sentences':len(inputs),
        'created_at_utc':datetime.now(timezone.utc).isoformat(),'targets_read':False,'freeze_sha256':sha(FREEZE),'cache_sha256':{}}
    write(state_path,state);torch.set_num_threads(2)
    device='mps'if torch.backends.mps.is_available()else'cpu'
    tokenizer=AutoTokenizer.from_pretrained(str(MODEL_DIR),local_files_only=True,trust_remote_code=False)
    base=BertModel.from_pretrained(str(MODEL_DIR),local_files_only=True,trust_remote_code=False,use_safetensors=True).to(device).eval()
    try:
        with torch.no_grad():
            for inp in inputs:
                encoded=source_encoding(inp,tokenizer);chunks=[]
                for c in encoded['chunks']:
                    ids=torch.tensor([c['ids']],device=device)
                    mask=base.get_extended_attention_mask(torch.ones_like(ids),ids.shape)
                    hidden=base.embeddings(input_ids=ids,token_type_ids=torch.zeros_like(ids))
                    for layer in base.encoder.layer[:11]:hidden=layer(hidden,attention_mask=mask)[0]
                    chunks.append({'start':c['start'],'end':c['end'],'hidden':hidden.cpu().float(),'mask':mask.cpu().float()})
                saved={'wordpieces':encoded['wordpieces'],'offsets':encoded['offsets'],'chunks':chunks,
                    'source_id':inp['id'],'targets_read':False,'freeze_sha256':sha(FREEZE)}
                path=source_cache_path(split,inp);torch.save(saved,path)
                state['cache_sha256'][str(path.relative_to(ROOT))]=sha(path)
                row={'phase':'source_only_cache','split':split,'source_id':inp['id'],'wordpieces':encoded['wordpieces'],
                    'complete_sentences':len(state['cache_sha256']),'targets_read':False,'at_utc':datetime.now(timezone.utc).isoformat()}
                with (dest/'cache_progress.jsonl').open('a')as stream:stream.write(json.dumps(row)+'\n')
                if len(state['cache_sha256'])%100==0:write(state_path,state);print(json.dumps(row),flush=True)
        state.update(status='completed_source_only_cache',completed_at_utc=datetime.now(timezone.utc).isoformat(),complete_sentences=len(inputs))
        write(state_path,state);print(json.dumps({'split':split,'status':state['status'],'sentences':len(inputs)}),flush=True)
    except BaseException as error:
        state.update(status='stopped_explicit_failure',error=str(error),stopped_at_utc=datetime.now(timezone.utc).isoformat());write(state_path,state);raise


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--split',choices=['train','dev'],required=True)
    main(ap.parse_args().split)
