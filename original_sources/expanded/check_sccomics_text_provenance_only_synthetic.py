"""Invented-only conformance checks; never invokes the actual source audit."""
from __future__ import annotations
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import tempfile
import time
from unittest.mock import patch

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('text_provenance',HERE/'audit_sccomics_text_provenance_only.py')
audit=importlib.util.module_from_spec(spec);spec.loader.exec_module(audit)

def put(path,data):
    path.parent.mkdir(parents=True,exist_ok=True)
    raw=data if isinstance(data,bytes) else (json.dumps(data,ensure_ascii=False,indent=2)+'\n').encode()
    path.write_bytes(raw);return hashlib.sha256(raw).hexdigest()

def fixture(root):
    base=' '.join(f'word{i}' for i in range(80))
    boundary=' '.join(f'boundary{i}' for i in range(22))
    fullwidth=''.join(chr(ord(c)+0xFEE0) if 33<=ord(c)<=126 else '\u3000' if c==' ' else c for c in base.upper())
    texts={1:base,2:' '.join(base.split()[:-3]+['changed77','changed78','changed79']),
      3:' '.join(base.split()[40:]+base.split()[:40]),4:'SHORT SOURCE',5:boundary,
      6:' '.join(reversed(base.split())),7:base,8:base,
      9:' '.join(boundary.split()[:-3]+['changed19','changed20','changed21']),10:'short\u3000source',
      101:fullwidth,104:'short\t\nsource',
      105:' '.join(boundary.split()[:-2]+['changed20','changed21']),
      106:' '.join(boundary.split()[:-3]+['changed19','changed20','changed21']),201:base}
    members={}
    folder=root/'data/sccomics_round4/source_v3/raw_text'
    for source_id in range(1,1001):
        text=texts.get(source_id,f'inventedshort{source_id}')
        path=folder/f'{source_id:04d}.txt';raw=text.encode();sha=put(path,raw)
        members[f'InventedAbstracts/{source_id:04d}.txt']={'canonical_path':str(path),'bytes':len(raw),'sha256':sha,'category':'raw_source_document'}
    sentinel=root/'data/sccomics_round4/source_v3/raw_annotations/0001.ann'
    put(sentinel,b'NEVER_OPEN_NEW_ANN_SENTINEL')
    manifest={'protocol_sha256':'a'*64,'source_review_sha256':'b'*64,'native_source_abstract_records':1000,
      'unique_normalized_dois':1000,'split_ids':copy.deepcopy(audit.NATIVE_SPLITS),
      'normalized_doi_by_id':{str(i):f'10.9999/invented{i}' for i in range(1,1001)},
      'official_archives':{audit.TEXT_ARCHIVE:{'members':members,'annotation_semantics_parsed':False,'official_sha256':'c'*64,'bytes':0,'archive':'invented_unopened_text_archive.zip'},
      audit.ANN_ARCHIVE:{'members':{'InventedAnnotations/0001.ann':{'canonical_path':str(sentinel),'sha256':'d'*64,'category':'unparsed_annotation_bytes'}},'annotation_semantics_parsed':False}},
      'all_saved_bytes_rehashed_after_all_copies':True,'semantic_annotation_read':False,'label_statistics_computed':False,
      'model_training_or_inference_performed':False,'paid_api_calls':0}
    manifest_path=root/'data/sccomics_round4/source_v3/source_acquisition_manifest.json';manifest_sha=put(manifest_path,manifest)
    poly=root/'data/polyie/source/train.json'
    poly_sha=put(poly,[{'entities':[{'label':'OLD_GOLD_SENTINEL','text':'Never decode this label value'}],
      'id':7,'relations':[[{'value':'OLD_GOLD_SENTINEL'}]],'text':base.split()},
      {'id':8,'text':['a','short','fixture'],'entities':{'escaped':'OLD_GOLD_SENTINEL\"\\'}}])
    import pyarrow as pa
    import pyarrow.parquet as pq
    mu=root/'data/mulms/source/train.parquet';mu.parent.mkdir(parents=True,exist_ok=True)
    pq.write_table(pa.table({'doc_id':['invented_mu1','invented_mu2'],'sentence':[base,'another short fixture'],
      'NER_labels':['OLD_PARQUET_GOLD_SENTINEL','OLD_PARQUET_GOLD_SENTINEL'],'relations':['NEVER_PROJECT','NEVER_PROJECT']}),mu)
    mu_sha=hashlib.sha256(mu.read_bytes()).hexdigest()
    metadata=root/'research/invented_official_metadata.csv'
    meta_sha=put(metadata,b'name,DOI,year,article_type,title\ninvented_mu1,https://doi.org/10.9999/invented1,2001,research-article,Invented fixture title\ninvented_mu2,,2002,review,Invented fixture title2\n')
    license_path=root/'research/invented_license.txt';license_sha=put(license_path,b'INVENTED fixture license; no assertion about real data')
    gate={'status':'synthetic_fixture_only','script_sha256':hashlib.sha256(Path(audit.__file__).read_bytes()).hexdigest(),
      'acquisition_manifest_sha256':manifest_sha,'source_stage_protocol_sha256':'a'*64,
      'annotation_semantics_allowed':False,'model_work_allowed':False,'root_confirmed_real_acquisition':False,
      'actual_execution_authorized':False,
      'fixed_similarity':{'word_ngram':5,'jaccard_numerator':4,'jaccard_denominator':5,'min_word_tokens':20},
      'legacy_source_inputs':[{'path':str(poly.relative_to(root)),'sha256':poly_sha,'bytes':poly.stat().st_size,'corpus':'POLYIE','split':'train','format':'json_tokens'},
       {'path':str(mu.relative_to(root)),'sha256':mu_sha,'bytes':mu.stat().st_size,'corpus':'MuLMS','split':'train','format':'parquet_sentences'}],
      'official_metadata_inputs':[{'path':str(metadata.relative_to(root)),'sha256':meta_sha,'format':'csv','fields':['name','DOI','year','article_type','title'],'corpus':'MuLMS'}],
      'license_inputs':[{'path':str(license_path.relative_to(root)),'sha256':license_sha,'declared_scope':'INVENTED fixture only','license_name':'INVENTED'}],
      'declared_native_text_license':'INVENTED fixture; not real source permission'}
    gate_path=root/'research/round4_sccomics_text_provenance_fixture_gate.json';put(gate_path,gate)
    return manifest_path,manifest,manifest_sha,gate_path,gate,sentinel,base


def main():
    started=time.monotonic();checks=[]
    with tempfile.TemporaryDirectory(prefix='INVENTED_sccomics_text_provenance_',dir=HERE.parent/'research') as tmp:
        root=Path(tmp);mp,manifest,msha,gp,gate,sentinel,base=fixture(root)
        output=root/'research/fixture_output'
        real_os_open=os.open;real_json_loads=json.loads
        opened_annotations=[];decoded_gold=[]
        def spy_open(path,*args,**kwargs):
            if Path(path).suffix=='.ann':opened_annotations.append(str(path));raise AssertionError('Annotation opened')
            return real_os_open(path,*args,**kwargs)
        def spy_loads(value,*args,**kwargs):
            if 'OLD_GOLD_SENTINEL' in (value.decode() if isinstance(value,bytes) else value):
                decoded_gold.append(True);raise AssertionError('Skipped Gold value semantically decoded')
            return real_json_loads(value,*args,**kwargs)
        import pyarrow.parquet as pq
        real_batches=pq.ParquetFile.iter_batches;projected_columns=[]
        def spy_batches(self,*args,**kwargs):
            projected_columns.append(kwargs.get('columns'))
            assert kwargs.get('columns')==['doc_id','sentence']
            return real_batches(self,*args,**kwargs)
        with patch.object(os,'open',spy_open),patch.object(json,'loads',spy_loads),patch.object(pq.ParquetFile,'iter_batches',spy_batches):
            result=audit.run_audit(root,mp,msha,gp,output,synthetic=True)
        pairs=[json.loads(l) for l in (output/'all_qualifying_pairs.jsonl').read_text().splitlines()]
        def present(kind,left,right):
            return [p for p in pairs if p['kind']==kind and p['left']['corpus']==p['right']['corpus']=='SC-CoMIcs' and
              {p['left']['source_id'],p['right']['source_id']}=={str(left),str(right)}]
        assert present('exact_native_source_bytes',1,201);checks.append('exact native byte duplicate retained across test/train')
        assert present('normalized_text',1,101) and not present('exact_native_source_bytes',1,101);checks.append('NFKC/fullwidth/casefold/whitespace duplicate distinct from raw bytes')
        assert present('word5gram_jaccard_ge_0.8',3,201);checks.append('reordered two-block near duplicate retained by ordered word5gram sets')
        boundary=present('word5gram_jaccard_ge_0.8',5,105);assert len(boundary)==1 and boundary[0]['intersection']==16 and boundary[0]['union']==20
        checks.append('exact threshold16/20=0.8 included')
        assert not present('word5gram_jaccard_ge_0.8',5,106);checks.append('below-threshold15/21 excluded')
        assert not present('word5gram_jaccard_ge_0.8',6,201);checks.append('reversing every word does not fake ordered-word5gram similarity')
        assert present('normalized_text',4,104) and not present('word5gram_jaccard_ge_0.8',4,104)
        assert any(r['corpus']=='SC-CoMIcs' and r['source_id']=='4' for r in result['short_text_units_near_judgment_unavailable'])
        checks.append('short exact-normalized match retained but near judgment unavailable')
        assert result['population_counts']['SC-CoMIcs']=={'train':{'text_units':800,'distinct_source_document_ids':800},'dev':{'text_units':100,'distinct_source_document_ids':100},'test':{'text_units':100,'distinct_source_document_ids':100}}
        assert result['native_split_unchanged'] and result['split_changed'] is False;checks.append('1000 IDs native800/100/100 unchanged')
        assert present('exact_native_source_bytes',1,8) and present('normalized_text',4,10) and present('word5gram_jaccard_ge_0.8',1,2)
        checks.append('within-test exact/normalized/near pairs all retained before real text reading')
        clusters=result['test_source_text_clusters'];groups=clusters['clusters']
        assert clusters['all_test_ids_present_exactly_once'] and clusters['native_test_record_count']==100
        assert clusters['source_text_cluster_count']==95
        core=next(g for g in groups if 1 in g['native_test_ids']);assert core['native_test_ids']==[1,2,3,7,8]
        assert next(g for g in groups if 4 in g['native_test_ids'])['native_test_ids']==[4,10]
        assert next(g for g in groups if 6 in g['native_test_ids'])['native_test_ids']==[6]
        assert next(g for g in groups if 5 in g['native_test_ids'])['native_test_ids']==[5]
        assert next(g for g in groups if 9 in g['native_test_ids'])['native_test_ids']==[9]
        checks.append('fixed100-node test graph components preserve isolated/repeated/near/short nodes')
        assert not clusters['gold_or_model_used'] and not clusters['external_or_crossfold_bridges_used'] and not clusters['resampling_or_statistical_test_performed']
        checks.append('test source-text grouping performs no Gold/model/resampling/independence certification')
        assert not opened_annotations and not decoded_gold and projected_columns==[['doc_id','sentence']]
        assert all(not e['path'].endswith('.ann') for e in result['file_open_log']);checks.append('new .ann sentinel never opened; JSON Gold values never decoded; Parquet only doc_id/sentence')
        assert any(p['kind']=='normalized_doi' and p['left']['corpus']!=p['right']['corpus'] for p in pairs)
        assert any(m['corpus']=='POLYIE' for m in result['missing_or_invalid_doi_documents'])
        assert result['official_metadata_distribution_by_native_block']['test']['year_counts']=={'2001':1}
        checks.append('DOI overlap uses bound official metadata; missing metadata/years preserved')
        assert not (output/'source_text_index.sqlite3').exists();checks.append('successful output removes temporary text-derived SQLite index')
        assert result['all_captured_inputs_final_rehashed']
        publication=json.loads((output/'output_manifest.json').read_text())
        for binding in publication['outputs']:
            raw=(output/binding['path']).read_bytes()
            assert hashlib.sha256(raw).hexdigest()==binding['sha256'] and len(raw)==binding['bytes']
        checks.append('all final input/code/gate/manifest checks and pair/unit/cluster/audit output hashes verified')
        # Exhaustive independent brute force over all eligible long fixture units.
        all_text={}
        for i in range(1,1001):
            raw=(root/f'data/sccomics_round4/source_v3/raw_text/{i:04d}.txt').read_text()
            tokens=audit.WORD.findall(audit.normalized_text(raw))
            if len(tokens)>=20:all_text[('SC-CoMIcs',str(i),audit.split_for(i))]={tuple(tokens[j:j+5]) for j in range(len(tokens)-4)}
        tokens=audit.WORD.findall(audit.normalized_text(base));grams={tuple(tokens[j:j+5]) for j in range(len(tokens)-4)}
        all_text[('POLYIE','7','train')]=grams;all_text[('MuLMS','invented_mu1:row0','train')]=grams
        expected=set()
        import itertools
        for (left,lg),(right,rg) in itertools.combinations(all_text.items(),2):
            if 'SC-CoMIcs' not in (left[0],right[0]) or (left[0]==right[0] and left[2]==right[2] and left[2]!='test'):continue
            intersection=len(lg&rg);union=len(lg|rg)
            if 5*intersection>=4*union:expected.add(frozenset((left,right)))
        actual={frozenset(((p['left']['corpus'],p['left']['source_id'],p['left']['split']),(p['right']['corpus'],p['right']['source_id'],p['right']['split']))) for p in pairs if p['kind']=='word5gram_jaccard_ge_0.8'}
        assert actual==expected;checks.append(f'all qualifying cross-fold/within-test/cross-corpus pairs retained: brute-force agreement({len(expected)} pairs)')

        def blocked(name,edit_manifest=None,edit_gate=None,expected_code=None,mutate=None,synthetic=True):
            changed=copy.deepcopy(manifest);changed_gate=copy.deepcopy(gate)
            if edit_manifest:edit_manifest(changed)
            if edit_gate:edit_gate(changed_gate)
            sha=put(mp,changed);changed_gate['acquisition_manifest_sha256']=sha;put(gp,changed_gate)
            if mutate:mutate()
            try:audit.run_audit(root,mp,sha,gp,root/f'research/blocked_{len(checks)}',synthetic=synthetic)
            except audit.AuditError as error:
                assert error.code==expected_code,(name,error.code,expected_code)
                checks.append(name+' rejected('+error.code+')')
            else:raise AssertionError(name+' was not rejected')
            put(mp,manifest);put(gp,gate)
        blocked('native split mutation',edit_manifest=lambda d:d['split_ids']['test'].pop(),expected_code='changed_native_split')
        blocked('missing source ID',edit_manifest=lambda d:d['official_archives'][audit.TEXT_ARCHIVE]['members'].pop('InventedAbstracts/1000.txt'),expected_code='incomplete_native_text_ids')
        blocked('duplicate numeric source ID',edit_manifest=lambda d:d['official_archives'][audit.TEXT_ARCHIVE]['members'].update({'InventedAbstracts/01.txt':copy.deepcopy(d['official_archives'][audit.TEXT_ARCHIVE]['members']['InventedAbstracts/0001.txt'])}),expected_code='duplicate_or_unexpected_source_id')
        blocked('source hash mutation',edit_manifest=lambda d:d['official_archives'][audit.TEXT_ARCHIVE]['members']['InventedAbstracts/0001.txt'].update(sha256='0'*64),expected_code='source_hash_or_size_mismatch')
        blocked('unreviewed script SHA',edit_gate=lambda d:d.update(script_sha256='0'*64),expected_code='unreviewed_script_hash')
        blocked('threshold mutation',edit_gate=lambda d:d['fixed_similarity'].update(min_word_tokens=19),expected_code='changed_similarity_contract')
        blocked('source scope mutated to Gold allowed',edit_gate=lambda d:d.update(annotation_semantics_allowed=True),expected_code='not_source_only_gate')
        blocked('real execution without explicit gate',synthetic=False,expected_code='protocol_not_authorized')
        blocked('real gate without root confirmation',edit_gate=lambda d:d.update(status='authorized_real_source_text_only_audit'),synthetic=False,expected_code='real_acquisition_not_confirmed')
        blocked('annotation filename injected as text',edit_manifest=lambda d:d['official_archives'][audit.TEXT_ARCHIVE]['members']['InventedAbstracts/0001.txt'].update(canonical_path=str(sentinel)),expected_code='forbidden_namespace')
        original=root/'data/sccomics_round4/source_v3/raw_text/0001.txt';original_raw=original.read_bytes()
        def link():original.unlink();original.symlink_to(sentinel)
        blocked('symlink text source to annotation',mutate=link,expected_code='symlink_path');original.unlink();put(original,original_raw)
        parent=root/'data/sccomics_round4/source_v3/raw_text';renamed=parent.with_name('fixture_text_original')
        def ancestor_link():parent.rename(renamed);parent.symlink_to(renamed,target_is_directory=True)
        blocked('symlink ancestor',mutate=ancestor_link,expected_code='symlink_path');parent.unlink();renamed.rename(parent)
        blocked('lexical traversal',edit_manifest=lambda d:d['official_archives'][audit.TEXT_ARCHIVE]['members']['InventedAbstracts/0001.txt'].update(canonical_path=str(root/'data/../outside.txt')),expected_code='unsafe_path')
        # Changes during copy/projection/commit must not leave a stale certificate.
        original_files={p:p.read_bytes() for p in [mp,gp,root/'data/polyie/source/train.json',root/'data/mulms/source/train.parquet',root/'research/invented_official_metadata.csv',root/'research/invented_license.txt']}
        failure_records=[]
        def failed_tamper(name,target,hook,code):
            destination=root/f'research/tamper_{len(checks)}'
            try:
                with patch.object(audit,target,hook):audit.run_audit(root,mp,msha,gp,destination,synthetic=True)
            except audit.AuditError as error:
                assert error.code==code,(name,error.code,code)
                assert not (destination/'audit.json').exists() and not (destination/'output_manifest.json').exists()
                failure=json.loads((destination/'failed_attempt.json').read_text());assert failure['completed_certificate'] is False
                failure_records.append({'fixture_case':name,**failure});checks.append(name+' rejected; failure preserved/no completed certificate')
            else:raise AssertionError(name+' produced a stale certificate')
            finally:
                for path,raw in original_files.items():path.write_bytes(raw)
        original_project=audit.project_json
        def during_json(access,path,fields):
            for n,row in enumerate(original_project(access,path,fields)):
                if '__source_projections__' in str(path) and n==0:Path(path).write_bytes(Path(path).read_bytes()+b' ')
                yield row
        failed_tamper('during-read projected JSON private-copy tamper','project_json',during_json,'source_hash_or_size_mismatch')
        original_parquet=audit.parquet_rows
        def during_parquet(access,path):
            for n,row in enumerate(original_parquet(access,path)):
                if n==0:Path(path).write_bytes(Path(path).read_bytes()+b' ')
                yield row
        failed_tamper('during-read projected Parquet private-copy tamper','parquet_rows',during_parquet,'source_hash_or_size_mismatch')
        original_snapshot=audit.private_source_snapshot
        def after_copy_original(access,original,binding,destination,number):
            copied=original_snapshot(access,original,binding,destination,number)
            if number==0:Path(original).write_bytes(Path(original).read_bytes()+b' ')
            return copied
        failed_tamper('post-copy original legacy source tamper','private_source_snapshot',after_copy_original,'source_hash_or_size_mismatch')
        def after_copy_gate(access,original,binding,destination,number):
            copied=original_snapshot(access,original,binding,destination,number)
            if number==0:gp.write_bytes(gp.read_bytes()+b' ')
            return copied
        failed_tamper('post-parse captured gate-byte tamper','private_source_snapshot',after_copy_gate,'source_hash_or_size_mismatch')
        def after_copy_manifest(access,original,binding,destination,number):
            copied=original_snapshot(access,original,binding,destination,number)
            if number==0:mp.write_bytes(mp.read_bytes()+b' ')
            return copied
        failed_tamper('post-parse captured acquisition-manifest-byte tamper','private_source_snapshot',after_copy_manifest,'source_hash_or_size_mismatch')
        def after_copy_license(access,original,binding,destination,number):
            copied=original_snapshot(access,original,binding,destination,number)
            if number==0:
                path=root/'research/invented_license.txt';path.write_bytes(path.read_bytes()+b' ')
            return copied
        failed_tamper('post-copy license-byte tamper','private_source_snapshot',after_copy_license,'source_hash_or_size_mismatch')
        real_module_file=audit.__file__;fake_module=root/'src/invented_audit_source.py';fake_module.parent.mkdir(parents=True);fake_module.write_bytes(Path(real_module_file).read_bytes())
        def after_copy_script(access,original,binding,destination,number):
            copied=original_snapshot(access,original,binding,destination,number)
            if number==0:fake_module.write_bytes(fake_module.read_bytes()+b' ')
            return copied
        try:
            audit.__file__=str(fake_module)
            failed_tamper('during-audit own-source-byte tamper on an invented clone','private_source_snapshot',after_copy_script,'script_changed_during_audit')
        finally:audit.__file__=real_module_file
        real_open=audit.SourceAccess.open
        def copy_read_tamper(access,path,kind):
            stream=real_open(access,path,kind)
            if kind!='old_source_raw_byte_snapshot_no_semantic_parse' or Path(path).suffix!='.json':return stream
            class ChangingStream:
                changed=False
                def __enter__(self):return self
                def __exit__(self,*args):stream.close()
                def read(self,n=-1):
                    if not self.changed:
                        Path(path).write_bytes(Path(path).read_bytes()+b' ');self.changed=True
                    return stream.read(n)
            return ChangingStream()
        # Adapt the generic failure runner to patch a class method, not real files.
        destination=root/f'research/tamper_{len(checks)}'
        try:
            with patch.object(audit.SourceAccess,'open',copy_read_tamper):audit.run_audit(root,mp,msha,gp,destination,synthetic=True)
        except audit.AuditError as error:
            assert error.code=='legacy_snapshot_hash_mismatch'
            failure=json.loads((destination/'failed_attempt.json').read_text());assert not failure['completed_certificate']
            assert not (destination/'audit.json').exists() and not (destination/'output_manifest.json').exists()
            failure_records.append({'fixture_case':'during-copy original-byte tamper',**failure});checks.append('during-copy original-byte tamper rejected; no completed certificate')
        else:raise AssertionError('Changed snapshot input accepted')
        finally:
            for path,raw in original_files.items():path.write_bytes(raw)
        # Preserve only explicitly INVENTED result snapshots, never temporary texts.
        out=HERE.parent/'research/round4_sccomics_text_provenance_synthetic_inputs';out.mkdir(exist_ok=True)
        for name in ['audit.json','all_qualifying_pairs.jsonl','all_source_units.jsonl','test_source_text_clusters.json','test_source_text_clusters.jsonl','output_manifest.json']:
            (out/('INVENTED_'+name)).write_bytes((output/name).read_bytes())
        summary={'identity':'invented_fixture_conformance_validation_not_actual_source_audit','status':'all_synthetic_checks_passed',
          'checks':checks,'check_count':len(checks),'qualifying_near_pairs':len(expected),'raw_source_fixture_records':1000,
          'native_split_counts':{'train':800,'dev':100,'test':100},'new_ann_open_count':len(opened_annotations),
          'native_test_record_count':clusters['native_test_record_count'],'synthetic_test_text_cluster_count':clusters['source_text_cluster_count'],
          'old_json_gold_value_decode_count':len(decoded_gold),'parquet_projected_columns':projected_columns,
          'real_sccomics_text_read':False,'real_annotations_or_model_outputs_read':False,'actual_execution_authorized':False,
          'script_sha256':hashlib.sha256(Path(audit.__file__).read_bytes()).hexdigest(),
          'checker_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'elapsed_seconds':time.monotonic()-started}
        (out/'INVENTED_tamper_failures.json').write_text(json.dumps(failure_records,indent=2)+'\n')
        (out/'synthetic_check_summary.json').write_text(json.dumps(summary,indent=2)+'\n')
        print(json.dumps(summary,indent=2))

if __name__=='__main__':main()
