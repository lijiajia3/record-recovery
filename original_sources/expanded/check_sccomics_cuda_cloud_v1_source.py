"""SOURCE-only controller checks: owned memory files/fake backend, no Torch/SC."""
from __future__ import annotations
import ast,builtins,copy,hashlib,importlib.util,json,os,pathlib,tempfile,traceback

root=pathlib.Path(__file__).resolve().parents[1];output=root/'research/round4_sccomics_cuda_cloud_v1_inputs/INVENTED_SOURCE_v3';output.mkdir(exist_ok=False)
path=root/'src/sccomics_cuda_cloud_v1.py';source=path.read_bytes();(output/'runner.before.py.txt').write_bytes(source);operator=(root/'src/check_sccomics_cuda_operators_v1.py').read_bytes();(output/'operator.before.py.txt').write_bytes(operator)
rows=[];original=builtins.__import__;forbidden=[]
def guard(name,*args,**kwargs):
    if name=='torch' or name.startswith('torch.') or name=='numpy' or name.startswith('numpy.'):
        forbidden.append(name);raise AssertionError('SOURCE helper must not import numerical/NN runtime')
    return original(name,*args,**kwargs)
builtins.__import__=guard
try:
    spec=importlib.util.spec_from_file_location('_INVENTED_CUDA_STAGE_SOURCE',path);api=importlib.util.module_from_spec(spec);spec.loader.exec_module(api)
finally:builtins.__import__=original

def need(value,message):
    if not value:raise AssertionError(message)
def check(name,fn):
    try:detail=fn();rows.append({'id':name,'passed':True,'detail':detail})
    except BaseException as e:rows.append({'id':name,'passed':False,'type':type(e).__name__,'error':str(e),'traceback':traceback.format_exc()})
def reject(fn):
    try:fn()
    except (api.CloudIntegrityError,ValueError):return True
    raise AssertionError('expected failure')

check('SOURCE_loading_no_Torch_or_NumPy',lambda:need(not forbidden,'NN imports'))
check('exact2807deployed_data_roles_noheldANN',lambda:need(len(api.data_population())==2807 and all(not n.endswith('.ann') or int(pathlib.PurePosixPath(n).stem)>=201 for n in api.data_population()),'role scope'))
for i in (1,100,101,200):check('ann_permission_reject_'+str(i),lambda i=i:reject(lambda:api.ann_path(i)))
check('phase_population90fitNER360fitheads3000NER12000pair1501test',lambda:need([len(api.expected_phase_files(p)) for p in api.PHASES]==[90,3000,360,12000,1501],'complete phases'))
for raw in (b'{"x":1,"x":2}',b'{"x":NaN}',b'{"x":1e400}'):
    check('strict_JSON_'+raw.decode(),lambda raw=raw:reject(lambda:api.unique(raw)))
for value in ('../x','x/../x','/x','x//x','x\\x'):
    check('relative_path_reject_'+value,lambda value=value:reject(lambda:api.relative(value)))
check('strict_binding_boolsize_reject',lambda:reject(lambda:api.strict_binding({'sha256':'0'*64,'size_bytes':True})))

def grammar():
    for raw in (source,operator):ast.parse(raw,feature_version=(3,12))
    tree=ast.parse(source)
    need(not any(isinstance(n,ast.Import) and any(a.name in ('torch','numpy','transformers') for a in n.names) for n in tree.body),'top runtime imports')
    need(b'weights_only=True' in source and b'use_safetensors=True' in source and b'local_files_only=True' in source and b'trust_remote_code=False' in source,'tensor/cold loading controls')
    need(b'torch.use_deterministic_algorithms(True,warn_only=False)' in source and b'torch.cuda.get_rng_state_all()' in source,'determinism/RNG')
    need(b'from_pretrained' not in operator and b'open(ann' not in operator,'compatibility helper not loading originals');return True
check('Python312_source_AST_frozen_CUDA_controls',grammar)

def mention_rejections():
    d={'candidates':[[0,0,0,1]],'text_length':1}
    for m in ([{'type':'Main','segments':[[False,1]]}],[{'type':'Main','segments':[[0,2]]}],[{'type':'Other','segments':[[0,1]]}],[{'type':'Main','segments':[[0,1]]}]*2):reject(lambda m=m:api.strict_mentions(m,d))
    return True
check('locked_mention_bool_offset_unknown_and_duplicate_reject',mention_rejections)
def manifest_shape():
    b=api.binding(b'INVENTED');controls={api.CONFIG:{'sha256':api.CONFIG_SHA,'size_bytes':0},api.NER_AMENDMENT:{'sha256':api.NER_AMENDMENT_SHA,'size_bytes':0},'research/round4_sccomics_cloud_cuda_hardware_amendment.json':b,**{p:{'sha256':h,'size_bytes':0} for p,h in api.PROVENANCE.items()}}
    m={'version':1,'purpose':'sccomics_cloud_cuda_before_fit','runner':api.binding(source),'operator_helper':api.binding(operator),'science':api.SCIENCE_PINS,'inputs':{p:copy.deepcopy(b) for p in api.data_population()},'controls':controls,'licenses':{'licenses/INVENTED.md':b},'runtime_versions':{p:'INVENTED_VERSION_NOT_ACTUAL' for p in ('python','torch','numpy','transformers','tokenizers','safetensors','networkx')},'cuda_policy':{'device':'cuda:0','dtype':'float32','deterministic':True,'cublas_workspace':':4096:8','tf32':False,'autocast':False,'attention_policy':'library_default_recorded_before_fit'}}
    api.manifest_contract(m);bad=copy.deepcopy(m);bad['inputs']['data/sccomics_round4/source_v3/raw_annotations/0100.ann']=b;reject(lambda:api.manifest_contract(bad));bad=copy.deepcopy(m);bad['science'][next(iter(bad['science']))]=b;reject(lambda:api.manifest_contract(bad));bad=copy.deepcopy(m);bad['cuda_policy']['autocast']=True;reject(lambda:api.manifest_contract(bad));return {'shape_only_not_actual_receipt_authority':True}
check('manifest_extra_heldANN_changed_science_or_AMP_failclosed',manifest_shape)
def default_cli():
    import subprocess,sys
    c=subprocess.run([sys.executable,str(path),'--root','/INVENTED_DOES_NOT_EXIST','--manifest','research/INVENTED.json','--manifest-sha256','0'*64,'--gate','research/INVENTED.json','--gate-sha256','0'*64,'--run-prefix','cloud_outputs/INVENTED','--phase','fit_ner'],capture_output=True,text=True)
    (output/'default_denial.stdout').write_text(c.stdout);(output/'default_denial.stderr').write_text(c.stderr);need(c.returncode!=0 and 'default no actual' in c.stderr,'no actual default gate');return {'actual_exit_code':c.returncode,'resource_paths_unopened':True}
check('actual_default_CLI_denied_before_any_NN_or_input',default_cli)


with tempfile.TemporaryDirectory(prefix='INVENTED_CUDA_SOURCE_') as temporary:
    owned=pathlib.Path(temporary).resolve();disk=api.Files(owned)
    def paths():
        disk.new('safe.txt',b'INVENTED');(owned/'alias').symlink_to(owned,target_is_directory=True)
        reject(lambda:disk.read('alias/safe.txt',api.binding(b'INVENTED')));os.mkfifo(owned/'invented_fifo');reject(lambda:disk.read('invented_fifo',api.binding(b'')));return {'no_alias_or_FIFO_open':True}
    check('actual_owned_regular_symlink_FIFO_guards',paths)
    def tamper():
        disk.new('tamper.txt',b'INVENTED')
        try:
            with disk.opened('tamper.txt') as fd:(owned/'tamper.txt').write_bytes(b'INVENTED_CHANGED')
        except api.CloudIntegrityError:return True
        raise AssertionError('post-open mutation accepted')
    check('actual_owned_during_capture_tamper_reject',tamper)

    class MemoryFiles(api.Files):
        def __init__(self,folder):super().__init__(folder);self.memory={};self.reads=[]
        def new(self,name,raw):api.relative(name);need(name not in self.memory,'no overwrite');self.memory[name]=raw;return api.binding(raw)
        def read(self,name,wanted):
            api.relative(name);raw=self.memory[name];need(api.binding(raw)==wanted,'memory binding mismatch');self.reads.append(name);return raw
        def verify(self,name,wanted):self.read(name,wanted);return wanted
    class FakeBackend:
        def __init__(self,files,manifest,zero=False):
            self.files=files;self.manifest=manifest;self.zero=zero;self.descriptors={i:{'candidates':[[0,0,0,1]],'text_length':1,'source_sha256':manifest['inputs'][api.text_path(i)]['sha256']} for i in range(1,1001)};self.runtime={'identity':'INVENTED_NO_NN'};self.opened_annotations=[];self.active=None;self.document_budget=0;self.update_budget=0
        def fit_start(self,k,s):self.active={'identity':'INVENTED_MODEL','kind':k,'seed':s};return self.active
        def fit_epoch(self,h,k,s,e,log):
            self.document_budget+=800;self.update_budget+=100
            log({'identity':'INVENTED_TRAINING_NOT_ACTUAL','source_document_population':800,'optimizer_updates':100,'epoch':e})
            return f'INVENTED_CP:{k}:{s}:{e}'.encode(),{'complete_model_state_dict':True,'optimizer_moments_in_epoch_checkpoint':False,'implicit_resume':False,'optimizer_updates':[{'update':(e-1)*100+i} for i in range(1,101)],'rng':{'identity':'INVENTED'},'final_zero_LR_update_still_step':e==10}
        def release(self,h):h.clear();self.active=None
        def failure_state(self):return b'INVENTED_FAILED_PARTIAL' if self.active else None
        def inference_start(self,k,s,raw):self.active={'identity':'INVENTED_INFERENCE'};return self.active
        def cache(self,i):return None
        def verify_private_cold(self):return None
        def probabilities(self,h,k,s,i,mentions=None):
            rows=([[0,0,0,0,0,0,0,0,0]] if self.zero else [[0,0,0,1,0,0,0,0,0]]) if k=='span' else [[0,0,0,0,0] for _ in range(len(mentions)**2)]
            a={'source_id':i,'source_sha256':self.manifest['inputs'][api.text_path(i)]['sha256'],'source_only':True,'Gold_consulted':False,'candidate_cap':None,'inference_row_block_size':512,'probabilities':rows,'columns':9 if k=='span' else 5,'complete_rows':len(rows)}
            if k=='span':a['candidates']=[[0,0,0,1]]
            else:a.update(mentions=copy.deepcopy(mentions),diagnostics=self.donor_diagnostics(mentions,s,i))
            return a
        def donor_diagnostics(self,m,s,i):
            return {'complete_directed_rows':len(m)**2,'donor_sha256':api.sha(b'INVENTED_DONOR'+str(len(m)).encode()),'donor_fixed_points':len(m)**2,'source_mentions_sha256':api.sha(api.encoded(m)),'Gold_consulted':False,'inference_row_block_size':512,'candidate_cap':None}
        def detector(self,obj,threshold):return [{'type':'Main','segments':[[0,1]]}] if obj['probabilities'][0][3]>=threshold else []
        def detector_records(self,i,m):return [{'family':'T','id':'T1','type':'Main','segments':[[0,1]],'text':'X'}] if m else []
        def records(self,i,m,p,rt,et):return self.detector_records(i,m),{'identity':'INVENTED_EMISSION'}
    def fixture(zero=False):
        folder=owned/('INVENTED_zero' if zero else 'INVENTED_normal');folder.mkdir();files=MemoryFiles(folder)
        manifest={'runner':api.binding(source),'operator_helper':api.binding(operator),'science':api.SCIENCE_PINS,'inputs':{},'controls':{},'licenses':{},'runtime_versions':{},'cuda_policy':{},'version':1,'purpose':'INVENTED_INJECTED_FAKE_BACKEND_NOT_ACTUAL_GATE'}
        files.new(api.SELF,source);files.new('src/check_sccomics_cuda_operators_v1.py',operator)
        for n,b in api.SCIENCE_PINS.items():files.new(n,(root/n).read_bytes())
        for n in api.data_population():manifest['inputs'][n]=files.new(n,b'INVENTED_PAYLOAD_NEVER_REAL_DATA_OR_WEIGHT')
        return files,manifest,FakeBackend(files,manifest,zero)
    def pipeline(zero=False):
        files,manifest,backend=fixture(zero);prefix='cloud_outputs/INVENTED';prior={};bridges={};selectors={'NER':'1'*64,**{k:('2'*64) for k in api.HEADS}}
        for phase in api.PHASES:
            gate={'phase':phase,'prior_receipts':copy.deepcopy(prior),'bridge':None,'local_selector_receipt_sha256':{} if phase in ('fit_ner','ner_dev') else {'NER':selectors['NER']} if phase in ('fit_heads','pair_dev') else selectors}
            if phase in ('fit_heads','pair_dev'):gate['bridge']=bridges['ner']
            if phase=='test_sources':gate['bridge']=bridges['pair']
            runner=api.CloudStages(files,manifest,gate,backend,prefix,lambda:None);receipt=runner.run();prior[phase]=receipt
            if phase=='ner_dev':
                locked={str(s):{str(i):[] if zero else [{'type':'Main','segments':[[0,1]]}] for i in api.SPLITS['dev']} for s in api.SEEDS}
                value={'version':1,'kind':'local_ner_source_bridge','population_receipt_sha256':receipt['sha256'],'selector_source_sha256':api.SCIENCE_PINS['src/sccomics_development_selection.py']['sha256'],'local_grid_proof':{'complete_setting_count':80,'seed_ids':list(api.SEEDS),'development_source_ids':list(api.SPLITS['dev']),'Gold_kept_local':True,'local_checked_selector_receipt_sha256':selectors['NER']},'choice':{'epoch':1,'threshold':.5},'locked_mentions':locked}
                raw=api.encoded(value);name='cloud_bridges/INVENTED_NER.json';bridges['ner']={'path':name,**files.new(name,raw)}
            if phase=='pair_dev':
                proof={'complete_setting_count':250,'seed_ids':list(api.SEEDS),'development_source_ids':list(api.SPLITS['dev']),'Gold_kept_local':True,'local_checked_selector_receipt_sha256':'2'*64}
                value={'version':1,'kind':'local_all_pair_choices_source_bridge','ner_bridge':bridges['ner'],'population_receipt_sha256':receipt['sha256'],'selector_source_sha256':api.SCIENCE_PINS['src/sccomics_development_selection.py']['sha256'],'choices':{k:{'epoch':1,'relation_threshold':.5,'role_threshold':.5} for k in api.HEADS},'local_grid_proofs':{k:copy.deepcopy(proof) for k in api.HEADS}}
                name='cloud_bridges/INVENTED_PAIRS.json';bridges['pair']={'path':name,**files.new(name,api.encoded(value))}
        need(backend.document_budget==120000 and backend.update_budget==15000,'all frozen work arithmetic')
        final=api.unique(files.memory[prior['test_sources']['path']]);need(len(final['files'])==1501,'all test objects+pretestfreeze')
        held=[n for n in files.reads if n.endswith('.ann') and int(pathlib.PurePosixPath(n).stem)<=200];need(not held,'held ANN memory sentinel opened')
        need(all(api.unique(raw).get('test_Gold_consulted') is False for name,raw in files.memory.items() if '/test_sources/' in name and '/test_sources/' in name.split(prefix+'/test_sources/',1)[-1]),'held Gold claimed')
        summary={'identity':'INVENTED_MEMORY_PIPELINE_NOT_NN','zero_detector_population':zero,'artificial_document_budget':backend.document_budget,'artificial_updates':backend.update_budget,'all_five_phases_completed':True,'receipt_bindings':prior,'memory_artifact_count':len(files.memory),'heldANN_sentinel_reads':0,'actual_data_or_model_files_opened':False}
        (output/('zero_pipeline_summary.json' if zero else 'normal_pipeline_summary.json')).write_bytes(api.encoded(summary))
        return files,manifest,backend,prior,bridges,summary
    normal=None
    def run_normal():
        global normal;normal=pipeline();return normal[-1]
    check('full_fixed_population_INVENTED_normal_all_five_stages',run_normal)
    check('full_fixed_population_INVENTED_zero_NER_all_five_stages',lambda:pipeline(True)[-1])
    def malformed():
        files,manifest,backend,prior,bridges,_=normal;gate={'phase':'test_sources','prior_receipts':{k:v for k,v in prior.items() if k!='test_sources'},'bridge':bridges['pair'],'local_selector_receipt_sha256':{'NER':'1'*64,**{k:'2'*64 for k in api.HEADS}}};runner=api.CloudStages(files,manifest,gate,backend,'cloud_outputs/INVENTED',lambda:None)
        obj=backend.probabilities({},'span',api.SEEDS[0],101);cp=api.binding(runner.cp('span',api.SEEDS[0],1))
        for field,value in [('Gold_consulted',True),('complete_rows',True),('candidates',[[False,0,0,1]]),('probabilities',[[float('nan')]*9]),('source_id',True)]:
            bad=copy.deepcopy(obj);bad[field]=value;reject(lambda bad=bad:runner.source_record(bad,'span',api.SEEDS[0],1,101,cp))
        m=[{'type':'Main','segments':[[0,1]]}];obj=backend.probabilities({},'aligned',api.SEEDS[0],101,m);obj['probabilities']=[];reject(lambda:runner.source_record(obj,'aligned',api.SEEDS[0],1,101,cp,m));return True
    check('source_content_Gold_bool_N²_candidates_finite_ID_guards',malformed)
    def locked_changed():
        files,manifest,backend,prior,bridges,_=normal;gate={'phase':'pair_dev','prior_receipts':{k:prior[k] for k in ('fit_ner','ner_dev','fit_heads')},'bridge':bridges['ner'],'local_selector_receipt_sha256':{'NER':'1'*64}}
        runner=api.CloudStages(files,manifest,gate,backend,'cloud_outputs/INVENTED',lambda:None);b=api.unique(files.memory[bridges['ner']['path']]);b['locked_mentions'][str(api.SEEDS[0])]['101']=[];return reject(lambda:runner.validate_ner_bridge(b,api.encoded(b)))
    check('same_seed_locked_graph_content_swap_reject',locked_changed)
    def incomplete_prior():
        files,manifest,backend,prior,bridges,_=normal;gate={'phase':'test_sources','prior_receipts':{k:v for k,v in prior.items() if k not in ('test_sources','fit_heads')},'bridge':bridges['pair'],'local_selector_receipt_sha256':{}};return reject(lambda:api.CloudStages(files,manifest,gate,backend,'cloud_outputs/INVENTED',lambda:None))
    check('missing150checkpoint_family_prior_gate_reject',incomplete_prior)
    def failed_final():
        files,manifest,backend=fixture(False) if False else (MemoryFiles(owned/'INVENTED_failure'),None,None);files.root.mkdir();manifest={'runner':api.binding(source),'operator_helper':api.binding(operator),'science':{},'inputs':{},'controls':{},'licenses':{}};files.new(api.SELF,source);files.new('src/check_sccomics_cuda_operators_v1.py',operator);backend=FakeBackend.__new__(FakeBackend);backend.runtime={'identity':'INVENTED'};backend.opened_annotations=[];backend.active=None;backend.document_budget=0;backend.update_budget=0;backend.verify_private_cold=lambda:None
        gate={'phase':'fit_ner','prior_receipts':{},'bridge':None,'local_selector_receipt_sha256':{}};runner=api.CloudStages(files,manifest,gate,backend,'cloud_outputs/INVENTED_final_failure',lambda:(_ for _ in ()).throw(api.CloudIntegrityError('INVENTED captured gate replaced')))
        reject(runner.run);need('cloud_outputs/INVENTED_final_failure/fit_ner/receipt.json' not in files.memory,'stale completed receipt');need('cloud_outputs/INVENTED_final_failure/fit_ner/FAILED.json' in files.memory,'failure preserved');return True
    check('during_run_final_control_tamper_no_completed_receipt',failed_final)
    def partial_state():
        folder=owned/'INVENTED_failure_state';folder.mkdir();files=MemoryFiles(folder);manifest={'runner':api.binding(source),'operator_helper':api.binding(operator),'science':{},'inputs':{},'controls':{},'licenses':{}};files.new(api.SELF,source);files.new('src/check_sccomics_cuda_operators_v1.py',operator);backend=FakeBackend.__new__(FakeBackend);backend.runtime={};backend.opened_annotations=[];backend.active=None
        backend.fit_epoch=lambda *a:(_ for _ in ()).throw(api.CloudIntegrityError('INVENTED failed epoch'))
        runner=api.CloudStages(files,manifest,{'phase':'fit_ner','prior_receipts':{},'bridge':None},backend,'cloud_outputs/INVENTED_failed_fit',lambda:None);reject(runner.run)
        need(files.memory.get('cloud_outputs/INVENTED_failed_fit/fit_ner/failed_partial_model.pt')==b'INVENTED_FAILED_PARTIAL','partial model lost during release');return True
    check('failed_fit_captures_partial_before_release_no_empty_baseline',partial_state)

need(path.read_bytes()==source and (root/'src/check_sccomics_cuda_operators_v1.py').read_bytes()==operator,'sources changed while checking')
result={'identity':'IMPLEMENTER_ACTUAL_EXECUTED_SOURCE_AST_MEMORY_FIXTURES_NOT_NN','runner_sha256':api.sha(source),'operator_helper_sha256':api.sha(operator),'checks':rows,'total':len(rows),'passed':sum(r['passed'] for r in rows),'failed':sum(not r['passed'] for r in rows),'Torch_NumPy_actual_SC_ANN_cache_cold_weights_GPU_API_or_keys_read':False,'independent_source_review':'not performed by implementer'}
(output/'actual_results.json').write_bytes(api.encoded(result));print(json.dumps({k:result[k] for k in ('total','passed','failed','runner_sha256','operator_helper_sha256')}));raise SystemExit(int(result['failed']>0))
