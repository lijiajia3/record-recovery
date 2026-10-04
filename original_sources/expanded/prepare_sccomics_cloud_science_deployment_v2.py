"""SOURCE-defined minimal cloud deployment; root-only actual byte preparation.

Copies the fixed public texts/cache/cold files and TRAIN201..1000 annotations.
Does not copy DEV/TEST annotations, an API credential, project histories, or
installed runtime trees. The root must freeze and review this builder first.
No GPU/provider calls, Torch import, fitting, or annotation parsing.
"""
from __future__ import annotations
import argparse, hashlib, json, os, pathlib, re, types

SELF='src/prepare_sccomics_cloud_science_deployment_v2.py'
BRIDGE='src/sccomics_cloud_local_bridge_v1.py'
CLOUD='src/sccomics_cuda_cloud_v3.py'
CLOUD_SHA='fb3f2afc056ea260933a244828f1ec62181479eb8cc4024c29141a236f84a20f'
OPERATOR='src/check_sccomics_cuda_operators_v3.py'
OPERATOR_SHA='0d286015a905cd6707ea1bd9c50779382c98975d6486c7d19512b60dfabc400e'
HARDWARE='research/round4_sccomics_cloud_cuda_hardware_amendment.json'
MANIFEST='research/round4_sccomics_cloud_cuda_deployment_science_manifest_v1.json'

def cloud_source(root):
    p=pathlib.Path(root)/CLOUD
    if any(q.is_symlink() for q in (p,*p.parents)):raise ValueError('source alias')
    raw=p.read_bytes()
    if hashlib.sha256(raw).hexdigest()!=CLOUD_SHA:raise ValueError('immutable cloud source changed')
    m=types.ModuleType('_deployment_captured_cloud');m.__file__=str(p)
    exec(compile(raw,str(p),'exec'),m.__dict__);return m

def binding(c,files,name):
    h,size=hashlib.sha256(),0
    with files.opened(name) as fd:
        while chunk:=os.read(fd,1048576):h.update(chunk);size+=len(chunk)
    return {'sha256':h.hexdigest(),'size_bytes':size}

REVIEW_PATH='research/round4_sccomics_cloud_bridge_independent_source_review.json'

def metadata_descriptor(c,d,role):
    # The role is checked BEFORE any stat/read/copy of its payload. No filename
    # role can promote an ANN/cache/checkpoint/credential to metadata duty.
    c.closed(d,('path','sha256','size_bytes'));c.relative(d['path'])
    c.strict_binding({k:d[k] for k in ('sha256','size_bytes')})
    if role=='independent_source_review':
        c.require(d['path']==REVIEW_PATH,'fixed independent review metadata duty')
    elif role=='actual_operator_receipt':
        name=d['path'];lower=name.lower()
        c.require(re.fullmatch(r'research/round4_sccomics_[A-Za-z0-9_/-]+\.json',name) is not None and
            'cuda' in lower and 'operator' in lower and
            not any(word in lower for word in ('annotation','gold','credential','key','weights','checkpoint','cache','tokenizer','raw_text')),
            'finite CUDA operator research JSON duty; no data/model/secret role')
    else:raise ValueError('unknown metadata duty')
    return d

def capture(c,files,d,*,role):
    metadata_descriptor(c,d,role)
    return files.read(d['path'],{k:d[k] for k in ('sha256','size_bytes')})

def extract_inputs(c,objects):
    """Metadata projection only; no native ANN/text/cache contents are opened."""
    acq=objects['data/sccomics_round4/source_v3/source_acquisition_manifest.json']
    cache=objects['results/local_baseline/sccomics_native_v1/source_cache/completed_manifest.json']
    cold=objects['data/local_baselines/matscibert/source_manifest.json']
    c.require(acq.get('native_source_abstract_records')==1000 and acq.get('all_saved_bytes_rehashed_after_all_copies') is True,'actual original acquisition receipt')
    inputs={}
    for archive in acq['official_archives'].values():
        for member,row in archive['members'].items():
            if member.endswith('.txt'):name=c.text_path(int(member[:-4]))
            elif member.endswith('.ann') and int(member[:-4]) in c.SPLITS['train']:name=c.ann_path(int(member[:-4]))
            else:continue
            c.require(name not in inputs,'unique native deployed identity')
            inputs[name]={'sha256':row['sha256'],'size_bytes':row['bytes']}
    c.require(cache.get('status')=='completed_cold_source_text_cache_only' and
        cache.get('complete_native_records')==1000 and cache.get('all_captured_source_model_and_cache_hashes_size_final_checked') is True and
        cache.get('annotation_content_used_in_source_features') is False,'original complete source-only cache')
    for row in cache['cache_bindings']:
        i=row['native_id'];c.require(type(i) is int and 1<=i<=1000 and row['path']==c.cache_path(i) and
            row['source_sha256']==inputs[c.text_path(i)]['sha256'] and row['path'] not in inputs,'exact cached source/native identity')
        inputs[row['path']]={'sha256':row['sha256'],'size_bytes':row['bytes']}
    c.require(cold.get('revision')=='ced9d8f5f208712c4a90f98a246fe32155b29995' and
        cold.get('base_encoder_not_task_finetuned_checkpoint') is True and len(cold['files'])==7,'original cold checkpoint provenance')
    for row in cold['files']:
        name=c.COLD_ROOT+'/'+row['name'];c.require(row['name'] in c.COLD_NAMES and name not in inputs,'original cold exact7')
        inputs[name]={'sha256':row['sha256'],'size_bytes':row['bytes']}
    c.require(set(inputs)==c.data_population(),'exact2807 deployed inputs, no held ANN/key')
    for b in inputs.values():c.strict_binding(b)
    return inputs

def hardware(c,receipt,receipt_descriptor):
    c.require(receipt.get('identity')=='CUDA_ACTUAL_INVENTED_OPERATOR_COMPATIBILITY_NO_CORPUS_OR_PRETRAINED_WEIGHT' and
        receipt.get('passed') is True and receipt.get('runner_sha256')==CLOUD_SHA and
        receipt.get('helper_sha256')==OPERATOR_SHA and receipt.get('science')==c.SCIENCE_PINS,
        'actual current nine-operator receipt required, SOURCE fake is no authority')
    checks={'Torch25_weights_only_original_shaped_invented_cache','original_overlap_index_add_gradient_and_actual_BERT_4Dmask',
        'strict_same_build_dropout_overlap_repeat','original_span_head_complete_and_zero','safetensors_and_weights_only_invented_state'}|{'original_'+k+'_complete_and_zero' for k in c.HEADS}
    c.require(type(receipt.get('checks')) is list and len(receipt['checks'])==9 and
        {x.get('id') for x in receipt['checks']}==checks and all(x.get('passed') is True for x in receipt['checks']), 'all9 actual checks')
    versions=receipt['runtime_versions'];c.require(set(versions)=={'python','torch','numpy','transformers','tokenizers','safetensors','networkx'} and
        all(type(x) is str and x for x in versions.values()),'finite direct runtime versions')
    policy={'device':'cuda:0','dtype':'float32','deterministic':True,'cublas_workspace':':4096:8',
        'tf32':False,'autocast':False,'attention_policy':'library_default_recorded_before_fit'}
    c.require(receipt.get('cuda_policy')==policy,'strict unchanged CUDA policy')
    return {'status':'before_fit_cuda_hardware_amendment_after_actual_operator_check','runner_sha256':CLOUD_SHA,
        'science':c.SCIENCE_PINS,'runtime_versions':versions,'cuda_policy':policy,
        'source_cache_policy':'reuse_original_MPS_first11_source_cache_bytes_CUDA_trainable_lastblock',
        'operator_compatibility_receipt':receipt_descriptor,'expected_GPU':receipt['GPU'],
        'expected_compute_capability':receipt['compute_capability'],
        'expected_attention_implementation':receipt['attention_implementation'],
        'expected_last_block_class':receipt['last_block_class']}

def build(c,files,authority_raw):
    a=c.unique(authority_raw)
    c.closed(a,('version','status','root_execution_authorized','builder_source','local_bridge_source',
        'independent_source_review','actual_operator_receipt','licenses','output_directory'))
    c.require(type(a['version']) is int and a['version']==1 and a['status']=='FROZEN_BEFORE_CLOUD_SUPERVISED_FIT_DEPLOYMENT' and
        a['root_execution_authorized'] is True,'separate root before-fit deployment authority')
    c.require(a['builder_source']==binding(c,files,SELF) and a['local_bridge_source']==binding(c,files,BRIDGE),'current builder/bridge source pins')
    metadata_descriptor(c,a['independent_source_review'],'independent_source_review')
    metadata_descriptor(c,a['actual_operator_receipt'],'actual_operator_receipt')
    review=c.unique(capture(c,files,a['independent_source_review'],role='independent_source_review'))
    c.require(review.get('status')=='source_review_passed' and review.get('independent_from_implementer') is True and
        review.get('deployment_builder_sha256')==a['builder_source']['sha256'] and
        review.get('local_bridge_sha256')==a['local_bridge_source']['sha256'],'independent practical SOURCE review')
    op=c.unique(capture(c,files,a['actual_operator_receipt'],role='actual_operator_receipt'));h=hardware(c,op,a['actual_operator_receipt'])
    controls={}
    objects={}
    for name,wanted in c.PROVENANCE.items():
        b=binding(c,files,name);c.require(b['sha256']==wanted,'immutable provenance receipt');controls[name]=b;objects[name]=c.unique(files.read(name,b))
    inputs=extract_inputs(c,objects)
    for name,wanted in ((c.CONFIG,c.CONFIG_SHA),(c.NER_AMENDMENT,c.NER_AMENDMENT_SHA)):
        b=binding(c,files,name);c.require(b['sha256']==wanted,'immutable original science configuration');controls[name]=b
    science={n:binding(c,files,n) for n in c.SCIENCE_PINS};c.require(science==c.SCIENCE_PINS,'ten unchanged science sources')
    runner=binding(c,files,CLOUD);helper=binding(c,files,OPERATOR)
    c.require(runner['sha256']==CLOUD_SHA and helper['sha256']==OPERATOR_SHA,'stable reviewed cloud routes')
    c.require(type(a['licenses']) is dict and a['licenses'],'explicit public data/model/code licenses')
    for name,b in a['licenses'].items():
        c.relative(name);c.require(name.startswith('licenses/') and pathlib.PurePosixPath(name).suffix in ('.txt','.md','.json'),'closed license path duties');files.verify(name,b)
    out=a['output_directory'];c.relative(out);c.require(out.startswith('cloud_deployments/') and not files.path(out).exists(),'distinct no-overwrite deployment directory')
    target=files.path(out);target.mkdir(parents=True);bundle=c.Files(target)
    controls[HARDWARE]=c.binding(c.encoded(h))
    manifest={'version':1,'purpose':'sccomics_cloud_cuda_before_fit','runner':runner,'operator_helper':helper,
        'science':science,'inputs':inputs,'controls':controls,'licenses':a['licenses'],
        'runtime_versions':h['runtime_versions'],'cuda_policy':h['cuda_policy']};c.manifest_contract(manifest)
    population={CLOUD:runner,OPERATOR:helper,**science,**inputs,
        **{n:b for n,b in controls.items() if n!=HARDWARE},**a['licenses'],
        a['actual_operator_receipt']['path']:{k:a['actual_operator_receipt'][k] for k in ('sha256','size_bytes')}}
    for name,b in population.items():
        destination=bundle.path(name);destination.parent.mkdir(parents=True,exist_ok=True)
        files.copy(name,b,destination)
    bundle.new(HARDWARE,c.encoded(h));bundle.new(MANIFEST,c.encoded(manifest))
    # Validate exact copied identities and nine-operator gate against the new root.
    c.provenance_contract(bundle,manifest);c.validate_hardware(bundle,h,manifest)
    for name,b in population.items():files.verify(name,b);bundle.verify(name,b)
    for name,b in controls.items():bundle.verify(name,b)
    for name in (SELF,BRIDGE):files.verify(name,a['builder_source'] if name==SELF else a['local_bridge_source'])
    files.verify(a['independent_source_review']['path'],{k:a['independent_source_review'][k] for k in ('sha256','size_bytes')})
    members={**population,HARDWARE:controls[HARDWARE],MANIFEST:c.binding(c.encoded(manifest))}
    result={'status':'complete_original_bytes_minimal_cloud_deployment_not_fit_authority',
        'members':members,'source_manifest_path':MANIFEST,'source_manifest_sha256':members[MANIFEST]['sha256'],
        'native_public_texts':1000,'native_source_caches':1000,'TRAIN_annotations':800,'cold_files':7,
        'DEV_TEST_annotation_files_uploaded':0,'credential_files_uploaded':0,
        'actual_fit_performed':False,'installed_runtime_inventory_claimed':False,
        'scientific_configuration_unchanged':True}
    bundle.new('deployment_byte_population_receipt.json',c.encoded(result))
    return {'output_directory':out,'manifest':{'path':out+'/'+MANIFEST,**members[MANIFEST]},'population_count':len(members)}

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',required=True)
    p.add_argument('--authority',required=True);p.add_argument('--authority-sha256',required=True)
    p.add_argument('--prepare-root-frozen-cloud-deployment',action='store_true');a=p.parse_args()
    if not a.prepare_root_frozen_cloud_deployment:raise ValueError('default SOURCE-only; no actual input preparation')
    c=cloud_source(a.root);files=c.Files(a.root);c.relative(a.authority)
    c.require(a.authority.startswith('research/round4_sccomics_cloud_cuda_') and a.authority.endswith('.json'),'authority metadata only')
    b={'sha256':a.authority_sha256,'size_bytes':files.path(a.authority).stat().st_size};raw=files.read(a.authority,b)
    result=build(c,files,raw);files.verify(a.authority,b)
    print(json.dumps(result,sort_keys=True))

if __name__=='__main__':main()
