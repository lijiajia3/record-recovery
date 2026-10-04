"""Invented seven-paper fixture: complete source barriers precede any Gold read."""
from contextlib import redirect_stdout
import io,json,tempfile
from pathlib import Path
import mulms_neural_family as family
import mulms_neural_family_test_analysis as analysis
from mulms_relation_adapter import source_endpoint
from mulms_local_adapter import sha,source_cache_path


def put(path,value):
    path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value))


def main():
    with tempfile.TemporaryDirectory()as directory:
        root=Path(directory);dest=root/family.DEST_REL;papers=[f'paper{i}'for i in range(7)]
        inputs=[{'id':f's{i}','doc_key':papers[i%7],'text':'A B'}for i in range(1114)]
        put(root/'data/mulms/test_inputs.json',inputs)
        entities=[{'start':0,'end':1,'type':'MAT','text':'A','probability':1.},
                  {'start':2,'end':3,'type':'FORM','text':'B','probability':1.}]
        selected={'epoch':1,'threshold':.5};cp={str(s):{'sha256':f'checkpoint{s}'}for s in family.SEEDS}
        freeze={'architecture_ids':list(family.ARCHITECTURES),'seed_ids':list(family.SEEDS),
            'required_complete_detector_graphs':3,'required_complete_relation_graphs':9,'file_sha256':{},
            'detector_selection':{'chosen':selected},'detector_checkpoints':cp,
            'relation_selections':{a:{'chosen':selected}for a in family.ARCHITECTURES},
            'relation_checkpoints':{a:cp for a in family.ARCHITECTURES}}
        bound=root/'invented_selection.json';put(bound,selected);freeze['file_sha256']['invented_selection.json']=sha(bound)
        put(root/family.FREEZE_REL,freeze);freeze_hash=sha(root/family.FREEZE_REL)
        cache={}
        for inp in inputs:
            path=root/str(source_cache_path('test',inp).relative_to(family.ROOT));path.parent.mkdir(parents=True,exist_ok=True)
            path.write_bytes(inp['id'].encode());cache[str(path.relative_to(root))]=sha(path)
        put(root/'results/local_baseline/mulms_supervised_v1/source_cache/test/state.json',
            {'status':'completed_source_only_cache','sentences':1114,'complete_sentences':1114,'freeze_sha256':freeze_hash,'targets_read':False,'cache_sha256':cache})
        manifest={'status':'all_three_detector_and_nine_relation_graphs_complete','freeze_sha256':freeze_hash,
            'annotation_content_used':False,'source_sentences':1114,'detector_graphs':{},'relation_graphs':{a:{}for a in family.ARCHITECTURES},'paper_ids':papers}
        correct=[{'h':source_endpoint(entities[0],'A B'),'t':source_endpoint(entities[1],'A B'),'r':r,'evidence':'A B'}for r in family.RELATION_IDS[:2]]
        invalid={'h':source_endpoint(entities[0],'A B'),'t':source_endpoint(entities[0],'A B'),'r':family.RELATION_IDS[0],'evidence':'A'}
        first={f's{i}'for i in range(7)};gold={i['id']:(correct+[dict(correct[0],h={'text':'A','occurrence':0,'type':'VALUE'})]if i['id']in first else[])for i in inputs}
        ner_gold={i['id']:(entities+[{'start':0,'end':1,'type':'VALUE','text':'A'}]if i['id']in first else entities)for i in inputs}
        for seed in family.SEEDS:
            path=dest/f'detector{seed}.json';put(path,{'seed':seed,'chosen':selected,'graphs':{i['id']:entities for i in inputs},
                'candidate_span_counts':{i['id']:3 for i in inputs},'freeze_sha256':freeze_hash,'annotation_content_used':False,
                'checkpoint_sha256':cp[str(seed)]['sha256'],'source_input_sha256':sha(root/'data/mulms/test_inputs.json')})
            manifest['detector_graphs'][str(seed)]={'path':str(path.relative_to(root)),'sha256':sha(path)}
            for a in family.ARCHITECTURES:
                p=dest/f'{a}{seed}.json';graphs={i['id']:(([invalid]+(correct if a=='typed'else[]))if i['id']in first else[])for i in inputs}
                graphs[inputs[-1]['id']]=[invalid]
                put(p,{'seed':seed,'architecture':a,'chosen':selected,'graphs':graphs,
                    'candidate_pair_counts':{i['id']:4 for i in inputs},'candidate_label_counts':{i['id']:60 for i in inputs},
                    'annotation_content_used':False,'freeze_sha256':freeze_hash,'checkpoint_sha256':cp[str(seed)]['sha256'],
                    'source_input_sha256':sha(root/'data/mulms/test_inputs.json'),'detector_graph_sha256':sha(path)})
                manifest['relation_graphs'][a][str(seed)]={'path':str(p.relative_to(root)),'sha256':sha(p)}
        mp=dest/'generation_manifest.json';put(mp,manifest)
        oldtokenizer,oldcache,oldtarget=family.AutoTokenizer,family.load_test_cache,analysis.load_targets;calls=[]
        class FakeTokenizer:
            cls_token_id=101;sep_token_id=102
            @staticmethod
            def from_pretrained(*_,**__):return FakeTokenizer()
            def __call__(self,*_,**__):return {'input_ids':[1,2],'offset_mapping':[(0,1),(2,3)]}
        family.AutoTokenizer=FakeTokenizer
        family.load_test_cache=lambda inp,root:{'offsets':[(0,1),(2,3)],'wordpieces':2,'chunks':[{'start':0,'end':2}]}
        def targets(*_):calls.append(True);return gold,ner_gold
        analysis.load_targets=targets
        def blocked():
            before=len(calls)
            try:analysis.analyze(root)
            except AssertionError:pass
            else:raise AssertionError('Incomplete/tampered family reached Gold')
            assert len(calls)==before,'Gold opened before whole-family gate'
        try:
            original=json.loads(json.dumps(manifest));del manifest['relation_graphs']['typed'][str(family.SEEDS[-1])];put(mp,manifest);blocked()
            manifest=json.loads(json.dumps(original));put(mp,manifest)
            manifest['paper_ids']=['wrong']*7;put(mp,manifest);blocked();manifest=json.loads(json.dumps(original));put(mp,manifest)
            manifest['source_sentences']=1000;put(mp,manifest);blocked();manifest=json.loads(json.dumps(original));put(mp,manifest)
            del manifest['detector_graphs'][str(family.SEEDS[-1])];put(mp,manifest);blocked();manifest=json.loads(json.dumps(original));put(mp,manifest)
            old=bound.read_bytes();bound.write_bytes(old+b' ');blocked();bound.write_bytes(old)
            entry=manifest['relation_graphs']['mean'][str(family.SEEDS[0])];p=root/entry['path'];old=p.read_bytes();p.write_bytes(old+b' ');blocked();p.write_bytes(old)
            # Even equally underreported full-pair populations in all nine heads
            # must fail the independently derived N^2, 15-label population.
            saved={}
            for by_seed in manifest['relation_graphs'].values():
                for entry in by_seed.values():
                    p=root/entry['path'];saved[p]=p.read_bytes();g=json.loads(p.read_text());g['candidate_pair_counts']={i['id']:1 for i in inputs}
                    g['candidate_label_counts']={i['id']:15 for i in inputs};put(p,g);entry['sha256']=sha(p)
            put(mp,manifest);blocked()
            for p,data in saved.items():p.write_bytes(data)
            put(mp,original)
            with redirect_stdout(io.StringIO()):result=analysis.analyze(root)
            assert len(calls)==1 and result['complete_detector_graph_barrier']==3 and result['complete_relation_graph_barrier']==9
            assert result['architecture_results']['typed']['pooled_primary']['tp']==42
            assert result['architecture_results']['typed']['pooled_primary']['fp']==24
            assert result['architecture_results']['typed']['pooled_primary']['fn']==21
            assert result['detector_descriptive']['pooled']['fn']==21
            for c in result['planned_contrasts']:
                assert c['permutation_units']==7 and c['enumerated_assignments']==128
                assert c['raw_exact_p']==2/128 and c['holm_p_two_planned_neural_contrasts']==4/128
        finally:family.AutoTokenizer,family.load_test_cache,analysis.load_targets=oldtokenizer,oldcache,oldtarget
    print('PASS: invented 3+9 graph/hash/source-population barriers before Gold; nested/missed NER FN, same-pair multilabel, self/no-edge FP, seven-paper Holm2')


if __name__=='__main__':main()
