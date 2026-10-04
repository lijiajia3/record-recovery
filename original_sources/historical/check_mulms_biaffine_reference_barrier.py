"""Invented reference gate: no annotation load before all three source graphs."""
from contextlib import redirect_stdout
import io,json,tempfile
from pathlib import Path
import mulms_biaffine_reference_test_analysis as analysis
from mulms_relation_adapter import source_endpoint
from mulms_local_adapter import RELATION_IDS,sha
import mulms_local_relation_training as original
import mulms_biaffine_reference_training as reference


def put(path,obj):path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(obj))


def main():
    assert original.ARCHITECTURES==('mean','typed','capacity_mean')
    assert original.DEST!=reference.REFERENCE_DEST and original.FREEZE!=reference.REFERENCE_FREEZE
    with tempfile.TemporaryDirectory()as folder:
        root=Path(folder);dest=root/'ref';papers=[f'p{i}'for i in range(7)]
        inputs=[{'id':f's{i}','doc_key':papers[i%7],'text':'A B'}for i in range(1114)]
        ents=[{'start':0,'end':1,'type':'MAT','text':'A'},{'start':2,'end':3,'type':'FORM','text':'B'}]
        correct=[{'h':source_endpoint(ents[0],'A B'),'t':source_endpoint(ents[1],'A B'),'r':r,'evidence':'A B'}for r in RELATION_IDS[:2]]
        first={f's{i}'for i in range(7)}
        gold={i['id']:(correct+[dict(correct[0],h={'text':'A','occurrence':0,'type':'VALUE'})]if i['id']in first else[])for i in inputs}
        primary_manifest={'freeze_sha256':'primary','paper_ids':papers,'detector_graphs':{str(s):{'sha256':f'ner{s}'}for s in analysis.SEEDS}}
        primary_path=root/'results/local_baseline/mulms_neural_family_v1/generation_manifest.json';put(primary_path,primary_manifest)
        put(root/'data/mulms/test_inputs.json',inputs);chosen={'epoch':1,'threshold':.5}
        bound=root/'choice.json';put(bound,chosen)
        lock={'chosen':chosen,'file_sha256':{'choice.json':sha(bound)},'primary_freeze_sha256':'primary',
            'primary_generation_manifest_sha256':sha(primary_path),'primary_summary_already_exists':True,
            'checkpoints':{str(s):{'sha256':f'cp{s}'}for s in analysis.SEEDS}}
        lp=root/'lock.json';put(lp,lock);manifest={'status':'all_three_reference_graphs_complete','freeze_sha256':sha(lp),
            'sentences_per_seed':1114,'source_input_sha256':sha(root/'data/mulms/test_inputs.json'),
            'primary_generation_manifest_sha256':sha(primary_path),'graphs':{}}
        for seed in analysis.SEEDS:
            p=dest/f'{seed}.json';put(p,{'seed':seed,'architecture':'biaffine','annotation_content_used':False,'chosen':chosen,
                'checkpoint_sha256':f'cp{seed}','freeze_sha256':sha(lp),'source_input_sha256':manifest['source_input_sha256'],
                'detector_graph_sha256':f'ner{seed}','candidate_pair_counts':{i['id']:4 for i in inputs},
                'candidate_label_counts':{i['id']:60 for i in inputs},'graphs':{i['id']:(correct if i['id']in first else[])for i in inputs}})
            manifest['graphs'][str(seed)]={'path':str(p.relative_to(root)),'sha256':sha(p)}
        mp=dest/'generation_manifest.json';put(mp,manifest)
        old=(analysis.ROOT,analysis.LOCK,analysis.DEST,analysis.complete_family,analysis.load_targets);calls=[]
        analysis.ROOT=root;analysis.LOCK=lp;analysis.DEST=dest
        analysis.complete_family=lambda _:(None,primary_manifest,inputs,{s:{'graphs':{i['id']:ents for i in inputs}}for s in analysis.SEEDS},None)
        def load(*_):calls.append(True);return gold,None
        analysis.load_targets=load
        def blocked():
            before=len(calls)
            try:analysis.main()
            except AssertionError:pass
            else:raise AssertionError('Reference opened Gold despite incomplete/tampered source scope')
            assert len(calls)==before
        try:
            original_manifest=json.loads(json.dumps(manifest));del manifest['graphs'][str(analysis.SEEDS[-1])];put(mp,manifest);blocked()
            manifest=json.loads(json.dumps(original_manifest));put(mp,manifest)
            oldbytes=bound.read_bytes();bound.write_bytes(oldbytes+b' ');blocked();bound.write_bytes(oldbytes)
            saved={}
            for entry in manifest['graphs'].values():
                p=root/entry['path'];saved[p]=p.read_bytes();g=json.loads(p.read_text())
                g['candidate_pair_counts']={i['id']:1 for i in inputs};g['candidate_label_counts']={i['id']:15 for i in inputs};put(p,g);entry['sha256']=sha(p)
            put(mp,manifest);blocked()
            for p,b in saved.items():p.write_bytes(b)
            put(mp,original_manifest)
            with redirect_stdout(io.StringIO()):analysis.main()
            result=json.loads((dest/'test_summary.json').read_text())
            assert len(calls)==1 and result['pooled_primary']['tp']==42 and result['pooled_primary']['fn']==21
            assert result['primary_summary_existed_before_reference_test_lock']is True
            assert result['planned_primary_holm2_and_global4_modified']is False
        finally:analysis.ROOT,analysis.LOCK,analysis.DEST,analysis.complete_family,analysis.load_targets=old
    print('PASS invented separate three-reference/source/checkpoint/hash/population barriers before Gold; unchanged primary globals, all FN, descriptive only')


if __name__=='__main__':main()
