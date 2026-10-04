"""SOURCE-only adapter for original verify_test then score_test under new TEST route.

Capture immutable bridge3 and invert exactly two Population path expressions.
Require reviewed migration bindings and a separate completed source-only unlock
before any TEST Gold callback. Native reconstruction/scorers/statistics stay raw.
"""
from __future__ import annotations
import argparse, contextlib, hashlib, json, pathlib, tempfile, types

SELF='src/sccomics_local_test_migration_v4.py'
BRIDGE='src/sccomics_cloud_local_bridge_v3.py'
BRIDGE_BINDING={'sha256':'01fb2cc5b18239b1d2c07e988fea680237a0892314efaff83ed16a517ea59c02','size_bytes':37520}
DRIVER='src/sccomics_cuda_test_migration_v2.py'
DRIVER_BINDING={'sha256':'6d47fccd623de3e789ed3ed0b7d0729f1b33629996b47015a15aefb445b09701','size_bytes':18608}
REVIEW='research/round4_sccomics_local_test_migration_v3_independent_source_review_20261004.json'
IDENTITY='sccomics_local_test_migration_verify_then_score_v1'
OLD_PATH="d['path'] == run_prefix + '/' + phase + '/receipt.json'"
NEW_PATH="d['path'] == (TEST_MIGRATION_PREFIX if phase == 'test_sources' else run_prefix) + '/' + phase + '/receipt.json'"
OLD_OUTPUT="r.get('output_prefix') == run_prefix + '/' + phase"
NEW_OUTPUT="r.get('output_prefix') == (TEST_MIGRATION_PREFIX if phase == 'test_sources' else run_prefix) + '/' + phase"

class LocalMigrationFailure(ValueError):pass
def require(value,message):
    if not value:raise LocalMigrationFailure(message)
def binding(raw):return {'sha256':hashlib.sha256(raw).hexdigest(),'size_bytes':len(raw)}
def captured(root,name,wanted):
    p=pathlib.Path(root).absolute()/name;require(not any(x.is_symlink() for x in (p,*p.parents)),'captured adapter source alias');raw=p.read_bytes();require(binding(raw)==wanted,'immutable adapter dependency binding');return raw

def effective_bridge_source(raw):
    require(binding(raw)==BRIDGE_BINDING,'original bridge3 bytes');s=raw.decode();require(s.count(OLD_PATH)==s.count(OLD_OUTPUT)==1,'exact two Population route expressions')
    effective=s.replace(OLD_PATH,NEW_PATH).replace(OLD_OUTPUT,NEW_OUTPUT).encode();require(effective.decode().replace(NEW_PATH,OLD_PATH).replace(NEW_OUTPUT,OLD_OUTPUT).encode()==raw,'two-route inverse original bridge bytes');return effective

def capture_modules(root):
    dr=captured(root,DRIVER,DRIVER_BINDING);driver=types.ModuleType('_SC_SOURCE_MIGRATION_DRIVER');driver.__file__=str(pathlib.Path(root).absolute()/DRIVER);exec(compile(dr,driver.__file__,'exec'),driver.__dict__)
    raw=captured(root,BRIDGE,BRIDGE_BINDING);effective=effective_bridge_source(raw);bridge=types.ModuleType('_SC_CAPTURED_BRIDGE3_TEST_MIGRATION');bridge.__file__=str(pathlib.Path(root).absolute()/BRIDGE);bridge.TEST_MIGRATION_PREFIX=driver.TEST_PREFIX;exec(compile(effective,bridge.__file__,'exec'),bridge.__dict__)
    return driver,bridge,effective

def dcheck(c,d,*,exact=None,prefix=None):
    c.closed(d,('path','sha256','size_bytes'));c.relative(d['path']);c.strict_binding({k:d[k] for k in ('sha256','size_bytes')})
    if exact is not None:require(d['path']==exact,'fixed local migration metadata duty')
    else:require(prefix is not None and d['path'].startswith(prefix) and d['path'].endswith('.json'),'local migration JSON descriptor duty')
    return d
def read(c,files,d,**role):dcheck(c,d,**role);return files.read(d['path'],{k:d[k] for k in ('sha256','size_bytes')})

def route_prefix(driver,phase,prior):
    require(phase in ('fit_ner','ner_dev','fit_heads','pair_dev','test_sources') and prior==driver.PRIOR_PREFIX,'exact original phase/prior prefix')
    return driver.TEST_PREFIX if phase=='test_sources' else prior

def validate_returned_migration(driver,c,receipt,policy,policy_d,operator,original_hardware,manifest):
    require(receipt.get('phase')=='test_sources' and receipt.get('status')=='completed_cloud_source_phase_not_Gold_score' and receipt.get('output_prefix')==driver.TEST_PREFIX+'/test_sources' and receipt.get('manifest_sha256')==driver.MANIFEST_BINDING['sha256'],'full returned TEST identity under unchanged manifest')
    expected={'identity':driver.IDENTITY,'driver':policy['driver'],'policy':policy_d,'source_review':policy['independent_source_review'],'new_actual_operator_receipt':policy['actual_new_operator_receipt'],'original_runner':driver.descriptor(driver.RUNNER,driver.RUNNER_BINDING),'effective_runner_sha256':policy['effective_runner_sha256'],'original_manifest_sha256':driver.MANIFEST_BINDING['sha256'],'original_test_gate_sha256':driver.TEST_GATE_BINDING['sha256'],'prior_run_prefix':driver.PRIOR_PREFIX,'test_output_prefix':driver.TEST_PREFIX,'training_GPU':original_hardware['expected_GPU'],'inference_GPU':operator['GPU'],'cross_GPU_bitwise_equivalence_claimed':False,'fit_DEV_selection_or_Gold_permission':False,'compute_capability':[7,0]}
    require(driver.literal_equal(receipt.get('gpu_migration'),expected) and receipt.get('DEV_TEST_annotation_permission') is False and receipt.get('not_a_local_Gold_score_or_statistics_certificate') is True,'exact migration bindings; no Gold or scientific-certification promotion')
    driver.validate_runtime(c,receipt['runtime'],original_hardware,manifest,operator)
    return expected

def validate_source_unlock(driver,c,files,unlock_d,common,authority):
    require(unlock_d is not None and authority['test_gold_unlock_receipt'] is not None,'separate reviewed migration verification and original source unlock before score')
    unlock=c.unique(read(c,files,unlock_d,prefix='local_cloud_analysis/'))
    c.closed(unlock,('identity','status','migration_bindings','original_action_receipt','test_Gold_consulted'))
    require(unlock['identity']==IDENTITY and unlock['status']=='completed_migrated_verify_test' and driver.literal_equal(unlock['migration_bindings'],common) and unlock['test_Gold_consulted'] is False,'same completed source-only migration verification')
    original=c.unique(read(c,files,unlock['original_action_receipt'],prefix='local_cloud_analysis/'))
    require(original.get('status')=='completed_local_verify_test' and driver.literal_equal(original.get('result',{}).get('unlock_receipt'),authority['test_gold_unlock_receipt']) and driver.literal_equal(original.get('opened_annotation_events'),[]),'original verified complete source unlock, zero annotations')
    return unlock

@contextlib.contextmanager
def guarded_Gold(driver,bridge,c,files,action,unlock_d,common,authority,recheck):
    original=bridge.held_gold
    def guard(*args,**kwargs):
        require(action=='score_test','verify adapter never grants held Gold callbacks');validate_source_unlock(driver,c,files,unlock_d,common,authority);recheck();return original(*args,**kwargs)
    bridge.held_gold=guard
    try:yield
    finally:bridge.held_gold=original

def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--root',required=True);parser.add_argument('--amendment',required=True);parser.add_argument('--amendment-sha256',required=True);parser.add_argument('--action',choices=('verify_test','score_test'),required=True);parser.add_argument('--execute-reviewed-local-test-migration',action='store_true');args=parser.parse_args(argv)
    require(args.execute_reviewed_local_test_migration,'default SOURCE-only; no actual source/Gold/API/GPU execution')
    driver,bridge,effective=capture_modules(args.root);c=bridge.cloud_source(args.root);files=c.Files(args.root)
    c.relative(args.amendment);require(args.amendment.startswith('research/round4_sccomics_local_test_migration_') and args.amendment.endswith('.json'),'fixed amendment family duty')
    size=files.path(args.amendment).stat(follow_symlinks=False).st_size;amd={'path':args.amendment,'sha256':args.amendment_sha256,'size_bytes':size};raw=read(c,files,amd,prefix='research/round4_sccomics_local_test_migration_');a=c.unique(raw)
    c.closed(a,('version','purpose','actual_root_execution_authorized','action','adapter','original_bridge','driver','effective_bridge_sha256','independent_source_review','migration_policy','returned_test_receipt','original_local_action_authority','verified_migration_unlock_receipt'))
    require(type(a['version']) is int and a['version']==1 and a['purpose']==IDENTITY and a['actual_root_execution_authorized'] is True and a['action']==args.action,'root local migration action authority')
    dcheck(c,a['adapter'],exact=SELF);files.verify(SELF,{k:a['adapter'][k] for k in ('sha256','size_bytes')});require(a['original_bridge']=={'path':BRIDGE,**BRIDGE_BINDING} and a['driver']=={'path':DRIVER,**DRIVER_BINDING} and a['effective_bridge_sha256']==binding(effective)['sha256'],'unchanged bridge/driver and exact two-route effective source')
    review=c.unique(read(c,files,a['independent_source_review'],exact=REVIEW));require(review.get('status')=='source_review_passed' and review.get('independent_from_implementer') is True and review.get('adapter_sha256')==a['adapter']['sha256'] and review.get('original_bridge_sha256')==BRIDGE_BINDING['sha256'] and review.get('effective_bridge_sha256')==a['effective_bridge_sha256'],'different local adapter SOURCE review current bindings')
    require(review.get('scope')=='local_test_gpu_migration_source_only' and review.get('inverse_original_bridge_bytes_exact') is True and review.get('native_scorer_and_statistics_unchanged') is True and review.get('actual_test_Gold_read') is False and review.get('scientific_execution_authority') is False,'source reviewer does not create actual Gold/authority')
    policy_raw=read(c,files,a['migration_policy'],exact=driver.POLICY);runner_raw=files.read(driver.RUNNER,driver.RUNNER_BINDING);policy=driver.policy_contract(c,files,policy_raw,driver.effective_runner_source(runner_raw))
    original_raw=read(c,files,a['original_local_action_authority'],prefix='research/round4_sccomics_cloud_cuda_');auth,manifest=bridge.authority(c,files,original_raw,args.action)
    require(auth['run_prefix']==driver.PRIOR_PREFIX and auth['deployment_manifest']==driver.descriptor(driver.MANIFEST,driver.MANIFEST_BINDING),'original local authority manifest/prior chain')
    require(a['returned_test_receipt']==auth['phase_receipts']['test_sources'],'root-bound exact migrated TEST receipt')
    receipt=c.unique(read(c,files,a['returned_test_receipt'],exact=driver.TEST_PREFIX+'/test_sources/receipt.json'))
    h=c.unique(files.read(driver.HARDWARE,driver.HARDWARE_BINDING));operator=c.unique(driver.read_descriptor(c,files,policy['actual_new_operator_receipt'],prefix=driver.OPERATOR_PREFIX));driver.validate_operator(c,operator,manifest,h)
    migration=validate_returned_migration(driver,c,receipt,policy,a['migration_policy'],operator,h,manifest)
    common={'adapter':a['adapter'],'adapter_source_review':a['independent_source_review'],'original_bridge':a['original_bridge'],'effective_bridge_sha256':a['effective_bridge_sha256'],'driver':a['driver'],'migration_policy':a['migration_policy'],'returned_test_receipt':a['returned_test_receipt'],'gpu_migration':migration}
    if args.action=='verify_test':require(a['verified_migration_unlock_receipt'] is None and auth['test_gold_unlock_receipt'] is None,'verify first, no inherited unlock')
    else:validate_source_unlock(driver,c,files,a['verified_migration_unlock_receipt'],common,auth)
    def recheck():
        for name,wanted in [(SELF,{k:a['adapter'][k] for k in ('sha256','size_bytes')}),(BRIDGE,BRIDGE_BINDING),(DRIVER,DRIVER_BINDING)]:files.verify(name,wanted)
        for d in (amd,a['independent_source_review'],a['migration_policy'],a['returned_test_receipt'],a['original_local_action_authority'],policy['actual_new_operator_receipt'],policy['independent_source_review']):files.verify(d['path'],{k:d[k] for k in ('sha256','size_bytes')})
        if a['verified_migration_unlock_receipt'] is not None:files.verify(a['verified_migration_unlock_receipt']['path'],{k:a['verified_migration_unlock_receipt'][k] for k in ('sha256','size_bytes')})
    recheck();captures={n:files.read('src/'+n+'.py',manifest['science']['src/'+n+'.py']) for n in c.SCIENCE}
    with guarded_Gold(driver,bridge,c,files,args.action,a['verified_migration_unlock_receipt'],common,auth,recheck),tempfile.TemporaryDirectory(prefix='SC_LOCAL_SOURCE_TOKENIZER_') as temp,c.CapturedScience(captures) as science:
        backend=bridge.LocalSourceBackend(c,files,manifest,science,pathlib.Path(temp).resolve());original_receipt=bridge.execute(c,files,auth,manifest,backend,science,a['original_local_action_authority'])
    recheck();final={'identity':IDENTITY,'status':'completed_migrated_'+args.action,'migration_bindings':common,'original_action_receipt':original_receipt,'test_Gold_consulted':args.action=='score_test'}
    result=bridge.write_object(c,files,auth['output_prefix']+'/completed_migrated_local_action_receipt.json',final)
    print(json.dumps({'action':args.action,'receipt':result,'scope':'original_native_analysis_under_reviewed_TEST_migration_route'}));return 0

if __name__=='__main__':raise SystemExit(main())
