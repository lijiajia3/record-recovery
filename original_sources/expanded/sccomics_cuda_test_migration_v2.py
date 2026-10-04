"""SOURCE-defined TEST inference migration; default performs no data or GPU work.

Capture the immutable trained v5 runner; amend only its final constructor hardware
identity check. Reuse completed CP/DEV and the original manifest/test gate. A fresh
actual nine-operator receipt on the current GPU and a different SOURCE review are
required before TEST sources. No fitting, DEV generation, Gold, or selection CLI.
"""
from __future__ import annotations
import argparse, contextlib, hashlib, json, pathlib, re, types

SELF='src/sccomics_cuda_test_migration_v2.py'
RUNNER='src/sccomics_cuda_cloud_v5.py'
RUNNER_BINDING={'sha256':'391d24ba24d762ae1610c92b18fb471386f10b65b0a818a1b461f5488347fedd','size_bytes':56134}
HELPER='src/check_sccomics_cuda_operators_v5.py'
HELPER_BINDING={'sha256':'17b26f413f59edb4d61348138ec89637f90f44406b788e202b910a349785e05e','size_bytes':15189}
MANIFEST='research/round4_sccomics_cloud_cuda_deployment_science_manifest_v1.json'
MANIFEST_BINDING={'sha256':'82227e594a5a8dfeee1de6edcc1da659c600fdfeeb32f153cc9486439a8cb1f6','size_bytes':519193}
TEST_GATE='research/round4_sccomics_cloud_cuda_v5_test_sources_gate_20261004.json'
TEST_GATE_BINDING={'sha256':'20d10cb6a4eadb13e4ec37961eb67904fbfa900857cc06c17f3cf4b018f7392e','size_bytes':2203}
HARDWARE='research/round4_sccomics_cloud_cuda_hardware_amendment.json'
HARDWARE_BINDING={'sha256':'91291126d359c98a91ff722ce800f24fa730fccc91c2046a6378fd18e4ae6ceb','size_bytes':3695}
PRIOR_PREFIX='cloud_outputs/SC_NATIVE_FOLD1_V5_CUDA_20261004'
TEST_PREFIX='cloud_outputs/SC_NATIVE_FOLD1_V5_TEST_GPU_MIGRATION_V1_20261004'
REVIEW='research/round4_sccomics_test_gpu_migration_independent_source_review_20261004.json'
POLICY='research/round4_sccomics_test_gpu_migration_policy_20261004.json'
OPERATOR_PREFIX='research/round4_sccomics_test_gpu_migration_actual_compat_20261004/'
KNOWN_GPUS=('Tesla V100S-PCIE-32GB','Tesla V100-SXM2-32GB')
IDENTITY='sccomics_test_inference_gpu_migration_before_test_sources_v1'
ORIGINAL_REQUIRE="        require(self.runtime['GPU']==h['expected_GPU'] and self.runtime['compute_capability']==h['expected_compute_capability'] and self.runtime['actual_attention_implementation']==h['expected_attention_implementation'] and self.runtime['actual_last_block_class']==h['expected_last_block_class'],'actual model/hardware matches prior operator amendment')"
MIGRATED_REQUIRE='        TEST_INFERENCE_HARDWARE_VALIDATOR(self.runtime,h,manifest)'

class MigrationFailure(ValueError):pass

def require(value,message):
    if not value:raise MigrationFailure(message)
def binding(raw):return {'sha256':hashlib.sha256(raw).hexdigest(),'size_bytes':len(raw)}
def descriptor(name,wanted):return {'path':name,**wanted}
def captured_bytes(root,name,wanted):
    p=pathlib.Path(root).absolute()/name
    require(not any(x.is_symlink() for x in (p,*p.parents)),'no captured source alias')
    raw=p.read_bytes();require(binding(raw)==wanted,'immutable captured source binding');return raw

def effective_runner_source(raw):
    require(binding(raw)==RUNNER_BINDING,'original captured runner bytes')
    source=raw.decode('utf-8');require(source.count(ORIGINAL_REQUIRE)==1 and MIGRATED_REQUIRE not in source,'one exact final backend identity check')
    effective=source.replace(ORIGINAL_REQUIRE,MIGRATED_REQUIRE).encode()
    require(effective.decode().replace(MIGRATED_REQUIRE,ORIGINAL_REQUIRE).encode()==raw,'exact inverse original runner byte identity')
    return effective

def capture_runner(root):
    raw=captured_bytes(root,RUNNER,RUNNER_BINDING);effective=effective_runner_source(raw)
    api=types.ModuleType('_SC_CAPTURED_V5_TEST_GPU_MIGRATION');api.__file__=str(pathlib.Path(root).absolute()/RUNNER)
    exec(compile(effective,api.__file__,'exec'),api.__dict__)
    return api,effective

def metadata_descriptor(api,d,path=None,prefix=None):
    api.closed(d,('path','sha256','size_bytes'));api.relative(d['path']);api.strict_binding({k:d[k] for k in ('sha256','size_bytes')})
    if path is not None:require(d['path']==path,'fixed migration metadata duty')
    else:
        require(prefix is not None and d['path'].startswith(prefix) and d['path'].endswith('/operator_result.json') and
            not any(x in d['path'].lower() for x in ('annotation','gold','credential','key','weights','checkpoint','cache','tokenizer')),
            'distinct actual compatibility metadata duty')
    return d

def read_descriptor(api,files,d,path=None,prefix=None):
    metadata_descriptor(api,d,path,prefix);return files.read(d['path'],{k:d[k] for k in ('sha256','size_bytes')})

def literal_equal(value,wanted):
    # JSON equality must not equate bool/int/float or accept shape subclasses.
    if type(value) is not type(wanted):return False
    if type(wanted) is dict:return set(value)==set(wanted) and all(type(k) is str and literal_equal(value[k],wanted[k]) for k in wanted)
    if type(wanted) is list:return len(value)==len(wanted) and all(literal_equal(a,b) for a,b in zip(value,wanted))
    return type(wanted) in (str,int,float,bool,type(None)) and value==wanted

def expected_operator_details(api):
    details={
      'Torch25_weights_only_original_shaped_invented_cache':{'4D_masks':[[1,1,1,6],[1,1,1,6]],'original_cache_not_opened':True},
      'original_overlap_index_add_gradient_and_actual_BERT_4Dmask':{'strict_deterministic_index_add_forward_backward':True,'tokens':[6,768]},
      'strict_same_build_dropout_overlap_repeat':{'cross_MPS_CUDA_equivalence_claim':False,'same_build_seeded_bitwise_repeat':True},
      'amended_CPU_prefix_span_head_complete_and_zero':{'all_finite_parameter_gradients':True,'complete21toyspans':True,'empty_shape':[0,9],'prefix_backend':api.SPAN_PREFIX_BACKEND,'same_build_seeded_span_logits_and_gradients_bitwise_repeat':True},
      'safetensors_and_weights_only_invented_state':{'invented_state_only':True}}
    details.update({'original_'+k+'_complete_and_zero':{'all9orderedtoyrows':True,'empty_shape':[0,5]} for k in api.HEADS})
    return details

def validate_operator(api,receipt,manifest,training_hardware):
    api.closed(receipt,('ANN_source_text_cache_pretrained_weights_or_API_read','CUDA_RNG_state_sizes','GPU','attention_implementation','checks','compute_capability','cuda_policy','cudnn','helper_sha256','identity','last_block_class','passed','runner_sha256','runtime_versions','science','span_prefix_backend','torch_cuda_build'))
    require(receipt['identity']=='CUDA_ACTUAL_INVENTED_OPERATOR_COMPATIBILITY_NO_CORPUS_OR_PRETRAINED_WEIGHT' and receipt['passed'] is True and receipt['ANN_source_text_cache_pretrained_weights_or_API_read'] is False,'fresh actual invented operator pass, no source/data read')
    require(receipt['runner_sha256']==RUNNER_BINDING['sha256'] and receipt['helper_sha256']==HELPER_BINDING['sha256'] and manifest['runner']==RUNNER_BINDING and manifest['operator_helper']==HELPER_BINDING,'unchanged actual helper/runner identities')
    require(literal_equal(receipt['science'],api.SCIENCE_PINS) and literal_equal(receipt['science'],manifest['science']) and literal_equal(receipt['span_prefix_backend'],api.SPAN_PREFIX_BACKEND) and literal_equal(receipt['span_prefix_backend'],training_hardware['span_prefix_backend']),'raw science and effective CPU-prefix policy unchanged')
    api.closed(receipt['runtime_versions'],('python','torch','numpy','transformers','tokenizers','safetensors','networkx'))
    require(all(type(v) is str and v for v in receipt['runtime_versions'].values()) and receipt['runtime_versions']==manifest['runtime_versions']==training_hardware['runtime_versions'] and literal_equal(receipt['cuda_policy'],manifest['cuda_policy']) and literal_equal(receipt['cuda_policy'],training_hardware['cuda_policy']),'same literal runtime versions and strict CUDA policy')
    require(type(receipt['GPU']) is str and receipt['GPU'] in KNOWN_GPUS and training_hardware['expected_GPU']==KNOWN_GPUS[0],'only two observed GPU names; trained GPU provenance remains V100S')
    cap=receipt['compute_capability'];require(type(cap) is list and all(type(x) is int for x in cap) and cap==[7,0]==training_hardware['expected_compute_capability'],'exact observed compute capability')
    require(receipt['attention_implementation']==training_hardware['expected_attention_implementation']=='sdpa' and receipt['last_block_class']==training_hardware['expected_last_block_class']=='BertLayer','original attention and layer class')
    require(type(receipt['cudnn']) is int and receipt['cudnn']==90100 and receipt['torch_cuda_build']=='12.1' and literal_equal(receipt['CUDA_RNG_state_sizes'],[16]),'same original CUDA/cuDNN/RNG-build metadata')
    expected={'Torch25_weights_only_original_shaped_invented_cache','original_overlap_index_add_gradient_and_actual_BERT_4Dmask','strict_same_build_dropout_overlap_repeat','amended_CPU_prefix_span_head_complete_and_zero','safetensors_and_weights_only_invented_state'}|{'original_'+k+'_complete_and_zero' for k in api.HEADS}
    checks=receipt['checks'];require(type(checks) is list and len(checks)==9,'all nine actual checks')
    for item in checks:api.closed(item,('detail','id','passed'));require(type(item['id']) is str and item['passed'] is True and type(item['detail']) is dict,'each actual complete successful operation')
    require({x['id'] for x in checks}==expected,'exact complete nine-check identity')
    details=expected_operator_details(api)
    require(all(literal_equal(item['detail'],details[item['id']]) for item in checks),'exact complete original-helper operation detail schema and typed values')
    return receipt

def validate_runtime(api,runtime,training_hardware,manifest,receipt):
    validate_operator(api,receipt,manifest,training_hardware)
    require(type(runtime) is dict and all(literal_equal(runtime[key],receipt[target]) for key,target in
        (('GPU','GPU'),('compute_capability','compute_capability'),('actual_attention_implementation','attention_implementation'),('actual_last_block_class','last_block_class'),('versions','runtime_versions'),('policy','cuda_policy'),('torch_cuda_build','torch_cuda_build'),('cudnn','cudnn'))),'current backend must match new actual compatibility hardware with exact literal types')
    return {'training_GPU':training_hardware['expected_GPU'],'inference_GPU':runtime['GPU'],'compute_capability':runtime['compute_capability'],'cross_GPU_bitwise_equivalence_claimed':False}

def policy_contract(api,files,raw,effective):
    p=api.unique(raw);api.closed(p,('version','purpose','actual_root_execution_authorized','phase','driver','original_runner','original_helper','original_manifest','original_test_gate','original_training_hardware','actual_new_operator_receipt','prior_run_prefix','test_output_prefix','effective_runner_sha256','independent_source_review','fit_DEV_selection_or_Gold_permission','cross_GPU_bitwise_equivalence_claimed'))
    require(type(p['version']) is int and p['version']==1 and p['purpose']==IDENTITY and p['actual_root_execution_authorized'] is True and p['phase']=='test_sources','root test-only migration authority')
    require(p['fit_DEV_selection_or_Gold_permission'] is False and p['cross_GPU_bitwise_equivalence_claimed'] is False,'no fit/DEV/selection/Gold or crossGPU equality permission')
    require(p['prior_run_prefix']==PRIOR_PREFIX and p['test_output_prefix']==TEST_PREFIX and p['effective_runner_sha256']==binding(effective)['sha256'],'fixed separate TEST output and prior run identity')
    for field,name,wanted in [('original_runner',RUNNER,RUNNER_BINDING),('original_helper',HELPER,HELPER_BINDING),('original_manifest',MANIFEST,MANIFEST_BINDING),('original_test_gate',TEST_GATE,TEST_GATE_BINDING),('original_training_hardware',HARDWARE,HARDWARE_BINDING)]:
        metadata_descriptor(api,p[field],name);require(p[field]==descriptor(name,wanted),'immutable original training/stage binding');files.verify(name,wanted)
    metadata_descriptor(api,p['driver'],SELF);files.verify(SELF,{k:p['driver'][k] for k in ('sha256','size_bytes')})
    review=api.unique(read_descriptor(api,files,p['independent_source_review'],path=REVIEW))
    require(review.get('status')=='source_review_passed' and review.get('independent_from_implementer') is True and review.get('driver_sha256')==p['driver']['sha256'] and review.get('original_runner_sha256')==RUNNER_BINDING['sha256'] and review.get('effective_runner_sha256')==p['effective_runner_sha256'],'different migration SOURCE review exact identities')
    require(review.get('scope')=='test_inference_gpu_migration_source_only' and review.get('inverse_original_runner_bytes_exact') is True and review.get('all_scientific_test_operations_unchanged') is True and review.get('actual_test_source_generation') is False and review.get('scientific_execution_authority') is False,'SOURCE review is not an actual TEST pass or authority')
    metadata_descriptor(api,p['actual_new_operator_receipt'],prefix=OPERATOR_PREFIX)
    require(p['actual_new_operator_receipt']['path']!='research/round4_sccomics_cloud_operator_actual_20261004_v5/research/ACTUAL_CUDA_OPERATOR_V5_20261004/operator_result.json','no reused training compatibility result')
    return p

def blocked_fit(*args,**kwargs):raise MigrationFailure('TEST-only migration forbids fit, DEV generation and Gold callbacks')

@contextlib.contextmanager
def test_routes(api,policy,receipt,training_hardware,manifest,migration_binding):
    original_store=api.PhaseStore
    old_validator=api.__dict__.get('TEST_INFERENCE_HARDWARE_VALIDATOR');had_validator='TEST_INFERENCE_HARDWARE_VALIDATOR' in api.__dict__
    originals=[(api.CudaBackend,name,getattr(api.CudaBackend,name)) for name in ('fit_start','fit_epoch','gold')]+[(api.CloudStages,name,getattr(api.CloudStages,name)) for name in ('fit','development')]
    def hardware_validator(runtime,h,m):
        require(h==training_hardware and m==manifest,'same validated original hardware/manifest');return validate_runtime(api,runtime,h,m,receipt)
    class RoutedTestStore(original_store):
        def __init__(self,files,prefix,phase):
            require(prefix==PRIOR_PREFIX and phase=='test_sources' and policy['test_output_prefix']==TEST_PREFIX,'route only original test stage to distinct TEST output')
            super().__init__(files,TEST_PREFIX,phase)
        def finish(self,metadata):
            require('gpu_migration' not in metadata,'no overwritten migration metadata')
            return super().finish({**metadata,'gpu_migration':migration_binding})
    api.TEST_INFERENCE_HARDWARE_VALIDATOR=hardware_validator;api.PhaseStore=RoutedTestStore
    for cls,name,old in originals:setattr(cls,name,blocked_fit)
    try:yield
    finally:
        api.PhaseStore=original_store
        for cls,name,old in originals:setattr(cls,name,old)
        if had_validator:api.TEST_INFERENCE_HARDWARE_VALIDATOR=old_validator
        else:api.__dict__.pop('TEST_INFERENCE_HARDWARE_VALIDATOR',None)

def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--root',required=True);parser.add_argument('--policy-sha256',required=True);parser.add_argument('--execute-test-inference-migration',action='store_true');a=parser.parse_args(argv)
    require(a.execute_test_inference_migration,'default SOURCE-only; no actual GPU/data operation')
    require(re.fullmatch('[0-9a-f]{64}',a.policy_sha256) is not None,'explicit root frozen policy SHA')
    api,effective=capture_runner(a.root);files=api.Files(a.root)
    size=files.path(POLICY).stat(follow_symlinks=False).st_size;policy_raw=files.read(POLICY,{'sha256':a.policy_sha256,'size_bytes':size});policy=policy_contract(api,files,policy_raw,effective)
    manifest_raw=files.read(MANIFEST,MANIFEST_BINDING);gate_raw=files.read(TEST_GATE,TEST_GATE_BINDING);manifest,gate=api.authorize(files,manifest_raw,gate_raw,'test_sources')
    require(gate['run_prefix']==PRIOR_PREFIX and manifest_raw==api.encoded(manifest),'unchanged canonical training manifest and original stage gate')
    h=api.unique(files.read(HARDWARE,HARDWARE_BINDING));operator_raw=read_descriptor(api,files,policy['actual_new_operator_receipt'],prefix=OPERATOR_PREFIX);operator=validate_operator(api,api.unique(operator_raw),manifest,h)
    require(not files.path(TEST_PREFIX+'/test_sources').exists(),'new TEST output already claimed/failed; no overwrite or resume')
    migration={'identity':IDENTITY,'driver':policy['driver'],'policy':descriptor(POLICY,{'sha256':a.policy_sha256,'size_bytes':size}),'source_review':policy['independent_source_review'],'new_actual_operator_receipt':policy['actual_new_operator_receipt'],'original_runner':descriptor(RUNNER,RUNNER_BINDING),'effective_runner_sha256':binding(effective)['sha256'],'original_manifest_sha256':MANIFEST_BINDING['sha256'],'original_test_gate_sha256':TEST_GATE_BINDING['sha256'],'prior_run_prefix':PRIOR_PREFIX,'test_output_prefix':TEST_PREFIX,'training_GPU':h['expected_GPU'],'inference_GPU':operator['GPU'],'cross_GPU_bitwise_equivalence_claimed':False,'fit_DEV_selection_or_Gold_permission':False}
    def final_metadata():
        for name,wanted in [(POLICY,{'sha256':a.policy_sha256,'size_bytes':size}),(SELF,{k:policy['driver'][k] for k in ('sha256','size_bytes')}),(RUNNER,RUNNER_BINDING),(HELPER,HELPER_BINDING),(MANIFEST,MANIFEST_BINDING),(TEST_GATE,TEST_GATE_BINDING),(HARDWARE,HARDWARE_BINDING)]:files.verify(name,wanted)
        for d in (policy['independent_source_review'],policy['actual_new_operator_receipt']):files.verify(d['path'],{k:d[k] for k in ('sha256','size_bytes')})
        review=gate['source_review'];files.verify(review['path'],{k:review[k] for k in ('sha256','size_bytes')});api.validate_hardware(files,api.unique(files.read(HARDWARE,HARDWARE_BINDING)),manifest)
    final_metadata()
    try:
        with test_routes(api,policy,operator,h,manifest,migration),api.deployed_backend(files,manifest) as backend:
            hardware_identity=validate_runtime(api,backend.runtime,h,manifest,operator);migration.update(hardware_identity)
            result=api.CloudStages(files,manifest,gate,backend,PRIOR_PREFIX,final_metadata).run()
    except BaseException as error:
        if not files.path(TEST_PREFIX+'/test_sources').exists():files.new(TEST_PREFIX+'/test_sources/STARTUP_FAILED.json',api.encoded({'status':'failed_before_stage_no_completion_no_resume','type':type(error).__name__,'error':str(error),'gpu_migration':migration}))
        raise
    print(json.dumps({'phase':'test_sources','receipt':result,'scope':'migrated_full_TEST_source_outputs_no_held_Gold_score','training_GPU':h['expected_GPU'],'inference_GPU':operator['GPU']}));return 0

if __name__=='__main__':raise SystemExit(main())
