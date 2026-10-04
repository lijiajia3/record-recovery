"""Whole-family source-only graph barrier, without a test-annotation loader."""
import json
import numpy as np
import torch
from transformers import AutoTokenizer
from mulms_local_adapter import ROOT,MODEL_DIR,OUTPUT,CACHE,sha,source_encoding,source_cache_path,span_candidates,span_key,TYPE_IDS,RELATION_IDS
from mulms_local_ner_training import SEEDS
from mulms_local_relation_training import ARCHITECTURES
from mulms_experiment import endpoint_key

FREEZE_REL='research/mulms_neural_family_test_freeze.json'
DEST_REL='results/local_baseline/mulms_neural_family_v1'


def validate_lock(root=ROOT):
    path=root/FREEZE_REL;freeze=json.loads(path.read_text())
    assert freeze['architecture_ids']==list(ARCHITECTURES)and freeze['seed_ids']==list(SEEDS)
    assert freeze['required_complete_detector_graphs']==3 and freeze['required_complete_relation_graphs']==9
    for name,digest in freeze['file_sha256'].items():assert sha(root/name)==digest,name
    return freeze


def validate_entities(inp,entities,candidates,offsets,threshold):
    assert len(entities)==len({span_key(e)for e in entities})
    boundaries={(c[2],c[3])for c in candidates}
    for entity in entities:
        assert (entity['start'],entity['end'])in boundaries and entity['type']in TYPE_IDS
        assert entity['text']==inp['text'][entity['start']:entity['end']]and entity['text']
        assert any(a<entity['end']and b>entity['start']for a,b in offsets),'Inference cannot use train empty-token fallback'
        assert np.isfinite(entity['probability'])and threshold<=entity['probability']<=1


def load_test_cache(inp,root=ROOT):
    path=root/str(source_cache_path('test',inp).relative_to(ROOT));saved=torch.load(path,map_location='cpu',weights_only=True)
    assert saved['source_id']==inp['id']and saved['targets_read']is False
    assert saved['freeze_sha256']==sha(root/FREEZE_REL)
    return saved


def complete_family(root=ROOT):
    # Every graph is required before any downstream caller may load Gold.
    freeze=validate_lock(root);dest=root/DEST_REL;manifest=json.loads((dest/'generation_manifest.json').read_text())
    assert manifest['status']=='all_three_detector_and_nine_relation_graphs_complete'
    assert manifest['freeze_sha256']==sha(root/FREEZE_REL)and manifest['annotation_content_used']is False
    assert set(manifest['detector_graphs'])=={str(s)for s in SEEDS}
    assert set(manifest['relation_graphs'])==set(ARCHITECTURES)
    assert all(set(v)=={str(s)for s in SEEDS}for v in manifest['relation_graphs'].values())
    inputs=json.loads((root/'data/mulms/test_inputs.json').read_text());ids={i['id']for i in inputs}
    assert len(ids)==len(inputs)==1114 and len({i['doc_key']for i in inputs})==7
    assert manifest['source_sentences']==len(inputs)
    assert len(manifest['paper_ids'])==7 and set(manifest['paper_ids'])=={i['doc_key']for i in inputs}
    state=json.loads((root/'results/local_baseline/mulms_supervised_v1/source_cache/test/state.json').read_text())
    assert state['status']=='completed_source_only_cache'and state['sentences']==state['complete_sentences']==len(inputs)
    assert state['freeze_sha256']==sha(root/FREEZE_REL)and state['targets_read']is False
    expected={str(source_cache_path('test',inp).relative_to(ROOT))for inp in inputs}
    assert set(state['cache_sha256'])==expected
    for name,digest in state['cache_sha256'].items():assert sha(root/name)==digest,name
    tokenizer=AutoTokenizer.from_pretrained(str(root/str(MODEL_DIR.relative_to(ROOT))),local_files_only=True,use_fast=True)
    population={}
    for inp in inputs:
        encoded=source_encoding(inp,tokenizer);saved=load_test_cache(inp,root)
        assert [list(x)for x in encoded['offsets']]==[list(x)for x in saved['offsets']]
        assert saved['wordpieces']==encoded['wordpieces']
        assert [(x['start'],x['end'])for x in saved['chunks']]==[(x['start'],x['end'])for x in encoded['chunks']]
        population[inp['id']]=(span_candidates(encoded['offsets']),encoded['offsets'])
    detector_graphs={};relation_graphs={}
    for seed in SEEDS:
        entry=manifest['detector_graphs'][str(seed)];assert sha(root/entry['path'])==entry['sha256']
        graph=json.loads((root/entry['path']).read_text())
        assert graph['seed']==seed and graph['freeze_sha256']==sha(root/FREEZE_REL)and graph['annotation_content_used']is False
        assert graph['chosen']==freeze['detector_selection']['chosen']
        assert graph['checkpoint_sha256']==freeze['detector_checkpoints'][str(seed)]['sha256']
        assert graph['source_input_sha256']==sha(root/'data/mulms/test_inputs.json')
        assert set(graph['graphs'])==ids==set(graph['candidate_span_counts'])
        for inp in inputs:
            candidates,offsets=population[inp['id']]
            assert graph['candidate_span_counts'][inp['id']]==len(candidates)
            validate_entities(inp,graph['graphs'][inp['id']],candidates,offsets,graph['chosen']['threshold'])
        detector_graphs[seed]=graph
        for architecture in ARCHITECTURES:
            entry=manifest['relation_graphs'][architecture][str(seed)]
            assert sha(root/entry['path'])==entry['sha256']
            rel=json.loads((root/entry['path']).read_text())
            assert rel['seed']==seed and rel['architecture']==architecture and rel['annotation_content_used']is False
            assert rel['chosen']==freeze['relation_selections'][architecture]['chosen']
            assert rel['checkpoint_sha256']==freeze['relation_checkpoints'][architecture][str(seed)]['sha256']
            assert rel['freeze_sha256']==sha(root/FREEZE_REL)and rel['source_input_sha256']==sha(root/'data/mulms/test_inputs.json')
            assert rel['detector_graph_sha256']==manifest['detector_graphs'][str(seed)]['sha256']
            assert set(rel['graphs'])==ids==set(rel['candidate_pair_counts'])==set(rel['candidate_label_counts'])
            for inp in inputs:
                count=len(graph['graphs'][inp['id']])**2
                assert rel['candidate_pair_counts'][inp['id']]==count
                assert rel['candidate_label_counts'][inp['id']]==count*len(RELATION_IDS)
                entities={span_key(e)for e in graph['graphs'][inp['id']]};records=rel['graphs'][inp['id']]
                assert len(records)==len({json.dumps(r,sort_keys=True)for r in records})
                for record in records:
                    assert endpoint_key(record.get('h'),inp['text'])in entities and endpoint_key(record.get('t'),inp['text'])in entities
                    assert record.get('r')in RELATION_IDS
            relation_graphs[architecture,seed]=rel
    return freeze,manifest,inputs,detector_graphs,relation_graphs
