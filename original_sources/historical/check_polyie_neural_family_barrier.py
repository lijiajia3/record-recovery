"""Invented14-paper fixture: every graph/hash before Gold, seeds cluster by paper."""
from contextlib import redirect_stdout
import io,json,tempfile
from pathlib import Path
import polyie_neural_family as family
import polyie_neural_family_test_analysis as analysis
from polyie_local_adapter import sha


def put(path,value):
    path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value))


with tempfile.TemporaryDirectory()as directory:
    root=Path(directory);analysis.ROOT=root;dest=root/family.DEST_REL
    freeze={'architecture_ids':list(family.ARCHITECTURES),'seed_ids':list(family.SEEDS),
        'required_complete_graphs':9,'file_sha256':{},'selections':{},'checkpoints':{}}
    papers=sorted(map(str,range(14)));docs=[];labels={};inputs=[]
    group={'CN':['1'],'PN':['2'],'PV':['3'],'Condition':[]};invalid=dict(group,CN=[])
    for doc in papers:
        docs.append({'doc_key':doc,'entities':[{'id':i,'type':r}for i,r in [('1','CN'),('2','PN'),('3','PV'),('4','PV')]]})
        labels[doc]={'eligible':[group],'all_valid':[group,dict(group,PV=['4'])],
            'schema_excluded':[{},{}]if doc=='0'else[]}
    for i in range(61):
        doc=papers[i%14]
        inputs.append({'doc_key':doc,'window_id':str(i),'start':0,'end':4,'text':'a b c d','tokens':['a','b','c','d'],
            'call_needed':True,'entities':[{'id':j,'type':r,'start':int(j)-1,'end':int(j)}for j,r in [('1','CN'),('2','PN'),('3','PV'),('4','PV')]]})
    source=root/'data/polyie/test_inputs.json';put(source,inputs);freeze['test_source_sha256']=sha(source)
    expected={d:2*sum(i['doc_key']==d for i in inputs)for d in papers}
    gold=root/'data/polyie/test_gold.json';put(gold,labels);freeze['test_gold_sha256']=sha(gold)
    put(root/'data/polyie/test_documents.json',docs)
    manifest={'test_targets_read':False,'test_source_sha256':sha(source),'scope':{'papers':14,'windows':61},
        'architecture_ids':list(family.ARCHITECTURES),'seed_ids':list(family.SEEDS),'paper_ids':papers,
        'complete_graphs':{},'source_cache_sha256':{}}
    for name in family.ARCHITECTURES:
        out=family.output(root,name);training=root/'research'/family.ARCHITECTURES[name][1];put(training,{})
        selected={'seed_ids':list(family.SEEDS),'chosen':{'epoch':2,'threshold':.9},'training_freeze_sha256':sha(training)}
        selection=family.selection_path(root,name);put(selection,selected)
        put(out/'training_state.json',{'status':'completed','seeds':list(family.SEEDS),'epochs':3,'selection_sha256':sha(selection)})
        freeze['selections'][name]={'sha256':sha(selection),'chosen':selected['chosen']};freeze['checkpoints'][name]={}
        for seed in family.SEEDS:
            cp=out/f'seed{seed}'/'epoch2.pt';cp.parent.mkdir(parents=True,exist_ok=True);cp.write_bytes(b'invented checkpoint')
            freeze['checkpoints'][name][str(seed)]={'sha256':sha(cp)}
            path=dest/f'{name}_{seed}.json'
            put(path,{'architecture':name,'seed':seed,'epoch':2,'threshold':.9,'test_targets_read':False,
                'selection_sha256':sha(selection),'checkpoint_sha256':sha(cp),
                'enumerated_candidates_by_paper':expected,
                'graphs':{d:([group,invalid]if name=='typed'else[invalid])for d in papers}})
            manifest['complete_graphs'][name+'|'+str(seed)]={'path':str(path.relative_to(root)),'sha256':sha(path)}
    freeze_path=root/family.FREEZE_REL;put(freeze_path,freeze);manifest['freeze_sha256']=sha(freeze_path)
    mp=dest/'generation_manifest.json';put(mp,manifest)
    original_loader=analysis.load_targets;target_calls=[]
    def tracked(f):target_calls.append(True);return original_loader(f)
    analysis.load_targets=tracked
    def blocked():
        before=len(target_calls)
        try:analysis.main()
        except AssertionError:pass
        else:raise AssertionError('Invalid family reached scoring')
        assert len(target_calls)==before,'Incomplete/tampered family opened Gold'
    missing=dict(manifest,complete_graphs={k:v for k,v in manifest['complete_graphs'].items()if k!='capacity_mean|20261005'})
    put(mp,missing);blocked();put(mp,manifest)
    path=root/manifest['complete_graphs']['mean|20261003']['path'];old=path.read_bytes();path.write_bytes(old+b' ')
    blocked();path.write_bytes(old)
    selection=family.selection_path(root,'typed');old=selection.read_bytes();selection.write_bytes(old+b' ')
    blocked();selection.write_bytes(old)
    # Identically underreported candidate counts across all nine graphs still fail
    # the independently derived source population check, before annotation reads.
    originals={}
    old_manifest=json.loads(json.dumps(manifest))
    for entry in manifest['complete_graphs'].values():
        p=root/entry['path'];originals[p]=p.read_bytes();g=json.loads(p.read_text())
        g['enumerated_candidates_by_paper']={d:2 for d in papers};put(p,g);entry['sha256']=sha(p)
    put(mp,manifest);blocked()
    for p,data in originals.items():p.write_bytes(data)
    manifest=old_manifest;put(mp,manifest)
    with redirect_stdout(io.StringIO()):result=analysis.main()
    assert len(target_calls)==1 and result['complete_graph_barrier']==9
    typed=result['architecture_results']['typed'];assert typed['pooled_primary']['tp']==42 and typed['pooled_primary']['fp']==42
    for seed in family.SEEDS:
        r=typed['all_seeds'][str(seed)]
        assert r['all_valid_full_document_sensitivity']['total']['fn']==14
        assert r['released_malformed_FN_sensitivity']['total']['fn']==15
    for c in result['planned_contrasts']:
        assert c['permutation_units']==14 and c['enumerated_assignments']==16384
        assert c['raw_exact_p']==2/16384 and c['holm_p_two_planned_neural_contrasts']==4/16384
    assert result['seeds_are_independent_papers']is False
print('PASS invented nine-graph/hash/selection barrier, invalid FP/full FN and Holm2 original-paper inference')
