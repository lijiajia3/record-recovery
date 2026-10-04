"""Train/dev-only endpoint/pooling representation audit before relation fit."""
import json
from collections import defaultdict
from transformers import AutoTokenizer
from mulms_local_adapter import ROOT,MODEL_DIR,source_encoding,span_key,sha,validate_model
from mulms_experiment import canonical,valid_key
from polyie_local_baseline_training import write


def main():
    validate_model()
    output={'test_annotations_read':False,'splits':{},'model_manifest_sha256':sha(MODEL_DIR/'source_manifest.json'),
        'tokenizer_sha256':{name:sha(MODEL_DIR/name)for name in ['tokenizer.json','tokenizer_config.json','vocab.txt']}}
    tokenizer=AutoTokenizer.from_pretrained(str(MODEL_DIR),local_files_only=True,use_fast=True)
    for split in ['train','dev']:
        inputs=json.loads((ROOT/f'data/mulms/{split}_inputs.json').read_text())
        entities=json.loads((ROOT/f'data/local_baselines/mulms/{split}_ner_gold.json').read_text())
        edges=json.loads((ROOT/f'data/mulms/{split}_gold.json').read_text())
        zero=[];missing=[];edge_count=0;candidate_count=0;no_edge=0;multi=0
        for inp in inputs:
            offsets=source_encoding(inp,tokenizer)['offsets'];keys={span_key(e)for e in entities[inp['id']]}
            groups=defaultdict(set);candidate_count+=len(keys)**2
            for entity in entities[inp['id']]:
                if not any(a<entity['end']and b>entity['start']for a,b in offsets):
                    zero.append({'id':inp['id'],'span':span_key(entity),'entity':entity,'text':inp['text'],'offsets':offsets})
            for record in edges[inp['id']]:
                key=canonical(record,inp['text']);assert valid_key(key);edge_count+=1
                if key[0]not in keys or key[1]not in keys:missing.append({'id':inp['id'],'key':key})
                groups[key[:2]].add(key[2])
            no_edge+=not groups;multi+=sum(len(v)>1 for v in groups.values())
        output['splits'][split]={'sentences':len(inputs),'edges':edge_count,'all_ordered_pairs_including_diagonal':candidate_count,
            'no_edge_sentences':no_edge,'multi_label_pairs':multi,'zero_wordpiece_overlap_entities':zero,'edges_missing_gold_ner_endpoint':missing,
            'inputs_sha256':sha(ROOT/f'data/mulms/{split}_inputs.json'),'ner_sha256':sha(ROOT/f'data/local_baselines/mulms/{split}_ner_gold.json'),
            'edges_sha256':sha(ROOT/f'data/mulms/{split}_gold.json'),
            'zero_overlap_policy':'retain exact entity and all pairs/targets, fixed zero768 training representation; predicted dev/test zero overlap forbidden'}
    write(ROOT/'research/mulms_relation_training_representation_audit.json',output)
    assert not any(v['edges_missing_gold_ner_endpoint']for v in output['splits'].values())
    assert not output['splits']['dev']['zero_wordpiece_overlap_entities']
    print(json.dumps({s:{k:v for k,v in c.items()if not isinstance(v,list)and not k.endswith('sha256')}for s,c in output['splits'].items()}))


if __name__=='__main__':main()
