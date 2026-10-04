"""INVENTED-only implementer checks for finite import/preparation helpers."""
from __future__ import annotations
import argparse
import contextlib
import copy
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import threading
import traceback
import types


def module(name,path):
    m=types.ModuleType(name);m.__file__=str(path);sys.modules[name]=m
    exec(compile(path.read_bytes(),m.__file__,'exec',dont_inherit=True),m.__dict__);return m


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--output-dir',required=True);args=ap.parse_args()
    root=Path(__file__).absolute().parents[1];out=root/args.output_dir
    if out.parent!=root/'research/round4_sccomics_runtime_dirfd_helpers_v2_inputs' or not out.name.startswith('INVENTED_') or out.exists():raise ValueError('fresh owned fixture output required')
    out.mkdir(parents=True)
    source_names=('src/check_sccomics_runtime_imports_source_only_v2.py','src/prepare_sccomics_supervised_runtime_v3_source_only.py',
                  'src/sccomics_supervised_stages_v2.py','src/sccomics_supervised_runtime_v2.py',
                  'src/check_sccomics_runtime_dirfd_helpers_v2_synthetic.py')
    source={n:(root/n).read_bytes() for n in source_names}
    for n,raw in source.items():
        p=out/'before'/n;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(raw)
    (out/'prospective_fixture_plan_before_execution.json').write_text(json.dumps({
        'identity':'INVENTED_HELPER_FIXTURE_PLAN_BEFORE_EXECUTION',
        'entry_modules':['INVENTED_library_'+str(i) for i in range(6)],
        'normal_module_source':['INVENTED_VALUE = '+str(i)+'\n' for i in range(6)],
        'no_real_runtime_library_model_or_corpus_operation':True,
        'source_digests':{n:hashlib.sha256(raw).hexdigest() for n,raw in source.items()},
        'cases_defined_in_captured_helper_source':True},sort_keys=True,indent=2)+'\n')
    smoke=module('_INVENTED_smoke_helper',root/source_names[0]);prep=module('_INVENTED_prep_helper',root/source_names[1])
    stage,runtime=smoke.load_captured_modules(root,{smoke.STAGE:source[smoke.STAGE],smoke.RUNTIME:source[smoke.RUNTIME]})
    results=[];specs=[]
    def check(name,function):
        try:detail=function();results.append({'id':name,'passed':True,'detail':detail})
        except BaseException as e:results.append({'id':name,'passed':False,'error_type':type(e).__name__,'error':str(e),'traceback':traceback.format_exc()})
    def reject(function):
        try:function()
        except BaseException as e:return {'rejected':True,'error_type':type(e).__name__,'error':str(e)}
        raise RuntimeError('expected rejection, accepted')
    @contextlib.contextmanager
    def invented(modification=None):
        with smoke.invented_source_import_capability() as cap:
            lib=cap.runtime_root/'lib';lib.mkdir()
            roles={};bindings={};bodies={}
            for i,name in enumerate(cap.entry_modules):
                raw=('INVENTED_VALUE = '+str(i)+'\n').encode();p=lib/(name+'.py');p.write_bytes(raw)
                relative=str(p.relative_to(cap.runtime_root));roles[relative]='package_code_native'
                bindings[relative]={**stage.binding(raw,'runtime'),'runtime_role':'package_code_native'};bodies[relative]=raw.decode()
            population={'policy':runtime.POLICY,'framework_root':str(cap.runtime_root),'stdlib_relative_path':'stdlib','site_relative_path':'lib','executable_relative_path':'bin/python3.13','package_roots':['lib'],
                        'file_roles':roles,'all_OS_GPU_or_system_dynamic_libraries_byte_bound':False}
            receipt={'status':'finite_runtime_code_only_source_receipt_not_fit_authority','runtime':stage.RUNTIME,'population':population,'file_bindings':bindings,
                     'source_contract_json_sha256':stage.V2_CONTRACT_SHA,'actual_model_weight_cache_or_SC_annotation_read':False,
                     'all_declared_files_start_and_final_hash_checked':True,'OS_GPU_byte_identity_claim':False,'platform_version_only':'INVENTED'}
            if modification:modification(cap,receipt)
            specs.append({'identity':'INVENTED_RUNTIME_MODULES_ONLY','module_source_text':bodies,'receipt':copy.deepcopy(receipt)})
            sys.path.insert(0,str(lib))
            try:yield cap,receipt
            finally:
                if str(lib) in sys.path:sys.path.remove(str(lib))
                for key in tuple(sys.modules):
                    if key.startswith(('INVENTED_library_','INVENTED_unlisted_')):sys.modules.pop(key,None)
    def make_guard(cap,receipt):return smoke.guard_class(stage,runtime)(receipt,cap)
    check('direct_capability_constructor_rejected',lambda:reject(lambda:smoke.SourceImportCapability(root,Path(sys.base_prefix).resolve(),False,smoke.ENTRIES,{},object())))
    def six_modules():
        with invented() as (cap,receipt):
            guard=make_guard(cap,receipt)
            b={n:stage.binding(source[n],'code') for n in (smoke.STAGE,smoke.RUNTIME,smoke.SELF)}
            guard.allow_science(smoke.source_module_inventory_adapter(root,b))
            answer=smoke.perform_imports(guard,cap)
            if len(answer['entry_libraries_imported'])!=6 or cap in stage._SCOPES:raise RuntimeError('population/registry differs')
            return {'six_owned_modules_imported':True,'no_fit_scope_registered':True,'finite_runtime_files':6,'no_real_entry_library_imported':True}
    check('six_INVENTED_entries_and_inventory_adapter',six_modules)
    def six_with_audit():
        with invented() as (cap,receipt):
            reads=tuple('INVENTED_runtime/'+n for n in receipt['file_bindings'])
            with smoke.verified_open_intent_adapter(stage,cap,reads,tuple(receipt['file_bindings'])) as adapter:
                state=smoke.project_and_network_audit(cap,reads,cap.project_root/'outputs',intent_adapter=adapter)
                try:
                    guard=make_guard(cap,receipt);answer=smoke.perform_imports(guard,cap)
                    if len(answer['entry_libraries_imported'])!=6 or state['denied_events']:raise RuntimeError('declared six-file audit population failed')
                    return {'complete_six_owned_imports_with_active_project_audit':True,'actual_libraries_or_model_constructed':False}
                finally:state['active']=False
    check('six_INVENTED_imports_with_active_project_and_network_audit',six_with_audit)
    @contextlib.contextmanager
    def dirfd_case():
        previous=os.getcwd();original_reader=stage.opened
        with invented() as (cap,receipt):
            raw=b'INVENTED_FRAMEWORK_BINARY_NOT_WEIGHTS';(cap.runtime_root/'Python').write_bytes(raw)
            receipt['population']['file_roles']['Python']='Python_framework_binary'
            receipt['file_bindings']['Python']={**stage.binding(raw,'runtime'),'runtime_role':'Python_framework_binary'}
            (cap.project_root/'Python').write_bytes(b'INVENTED_PROJECT_SHADOW_MUST_NOT_READ')
            bad=cap.project_root/'INVENTED_wrong_parent';bad.mkdir();(bad/'Python').write_bytes(b'INVENTED WRONG PARENT NOT READ')
            control=cap.project_root/'research/INVENTED/nested.json';control.parent.mkdir(parents=True);control.write_bytes(b'{"INVENTED":true}')
            reads=tuple('INVENTED_runtime/'+n for n in receipt['file_bindings'])+('research/INVENTED/nested.json',)
            try:
                os.chdir(cap.project_root) # Fixture reproduces the original failure; helper never changes CWD.
                with smoke.verified_open_intent_adapter(stage,cap,reads,tuple(receipt['file_bindings'])) as adapter:
                    state=smoke.project_and_network_audit(cap,reads,cap.project_root/'outputs',intent_adapter=adapter)
                    try:yield cap,receipt,state,raw
                    finally:state['active']=False
                if stage.opened is not original_reader:raise RuntimeError('original reader not restored')
            finally:os.chdir(previous)
    def framework_python():
        with dirfd_case() as (cap,r,state,expected):
            if stage.raw_read(cap.runtime_root,'Python')!=expected:raise RuntimeError('framework binary bytes differ')
            if stage.raw_read(cap.project_root,'research/INVENTED/nested.json')!=b'{"INVENTED":true}':raise RuntimeError('nested declared source failed')
            answer=smoke.perform_imports(make_guard(cap,r),cap)
            if len(answer['entry_libraries_imported'])!=6:raise RuntimeError('six population incomplete')
            return {'CWD_is_INVENTED_project':True,'framework_literal_Python_openat_succeeded':True,'nested_source_control_succeeded':True,'six_INVENTED_entry_imports_succeeded':True}
    check('dirfd_framework_Python_project_CWD_nested_and_complete_imports',framework_python)
    def shadow_project():
        with dirfd_case() as (cap,r,state,expected):
            a=reject(lambda:(cap.project_root/'Python').read_bytes())
            b=reject(lambda:stage.raw_read(cap.project_root,'Python'))
            return {'direct_absolute_project_shadow':a,'undeclared_stage_read':b,'allowed_framework_still_works':stage.raw_read(cap.runtime_root,'Python')==expected}
    check('dirfd_project_shadow_not_basename_exempt',shadow_project)
    def unknown_relative():
        with dirfd_case() as (cap,r,state,expected):
            descriptor=os.open(cap.runtime_root,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW)
            try:return reject(lambda:os.open('Python',os.O_RDONLY,dir_fd=descriptor))
            finally:os.close(descriptor)
    check('unverified_relative_openat_rejected',unknown_relative)
    def borrow_body():
        with dirfd_case() as (cap,r,state,expected):
            with stage.opened(cap.runtime_root,'Python') as opened:
                a=reject(lambda:os.open('Python',os.O_RDONLY))
                if os.read(opened,1000)!=expected:raise RuntimeError('legitimate fd changed')
            return dict(a,yield_body_cannot_borrow_intent=True)
    check('with_yield_same_name_call_cannot_borrow_context',borrow_body)
    def nested():
        with dirfd_case() as (cap,r,state,expected):
            with stage.opened(cap.runtime_root,'Python') as fd:
                inner=stage.raw_read(cap.project_root,'research/INVENTED/nested.json')
                if os.read(fd,1000)!=expected or inner!=b'{"INVENTED":true}':raise RuntimeError('nested declared read failed')
            return {'nested_contexts_return_to_outer_fd':True}
    check('nested_verified_reads_restore_context',nested)
    def exception_body():
        with dirfd_case() as (cap,r,state,expected):
            try:
                with stage.opened(cap.runtime_root,'Python'):raise RuntimeError('INVENTED BODY EXCEPTION')
            except RuntimeError as e:
                if str(e)!='INVENTED BODY EXCEPTION':raise
            if stage.raw_read(cap.runtime_root,'Python')!=expected:raise RuntimeError('exception left stale context')
            return {'exception_context_restored':True,'unrelated_relative_read_rejected':reject(lambda:open('Python','rb'))}
    check('exception_body_context_restoration',exception_body)
    def threaded():
        with dirfd_case() as (cap,r,state,expected):
            outcomes=[]
            def worker():
                try:
                    rejected=reject(lambda:os.open('Python',os.O_RDONLY))
                    value=stage.raw_read(cap.runtime_root,'Python')
                    outcomes.append({'rejected':rejected,'independent_verified_read':value==expected})
                except BaseException as e:outcomes.append({'error':repr(e)})
            with stage.opened(cap.runtime_root,'Python') as fd:
                thread=threading.Thread(target=worker);thread.start();thread.join(5)
                if thread.is_alive() or os.read(fd,1000)!=expected:raise RuntimeError('thread/outer read failed')
            if len(outcomes)!=1 or not outcomes[0].get('independent_verified_read'):raise RuntimeError(str(outcomes))
            return {'thread_cannot_borrow_outer_context':outcomes,'no_real_libraries_or_model':True}
    check('thread_context_isolation_and_independent_declared_read',threaded)
    def active_write():
        with dirfd_case() as (cap,r,state,expected):return reject(lambda:stage.raw_read(cap.project_root,'undeclared.txt'))
    check('intent_adapter_rejects_undeclared_root_relative_name',active_write)
    def write_intent():
        with dirfd_case() as (cap,r,state,expected):
            before=(cap.runtime_root/'Python').stat().st_size
            def attempt():
                with stage.opened(cap.runtime_root,'Python',write=True):pass
            answer=reject(attempt)
            if (cap.runtime_root/'Python').stat().st_size!=before:raise RuntimeError('write intent changed source')
            return dict(answer,active_adapter_read_only=True)
    check('intent_adapter_active_write_rejected_before_open',write_intent)
    def alias_leaf():
        with dirfd_case() as (cap,r,state,expected):
            path=cap.runtime_root/'Python';path.unlink();path.symlink_to(cap.project_root/'Python')
            return reject(lambda:stage.raw_read(cap.runtime_root,'Python'))
    check('declared_framework_leaf_alias_still_rejected',alias_leaf)
    def descriptor_identity():
        with dirfd_case() as (cap,r,state,expected):
            bad=cap.project_root/'INVENTED_wrong_parent'
            badfd=os.open(bad,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW);saved=[];previous_trace=sys.gettrace()
            code=stage.opened.__wrapped__.__closure__
            # Captured original generator is held inside wrapped's closure.
            original=next(c.cell_contents for c in code if callable(c.cell_contents) and hasattr(c.cell_contents,'__wrapped__'))
            original_code=original.__wrapped__.__code__
            def trace(frame,event,arg):
                if frame.f_code is original_code and event=='line' and 'flags' in frame.f_locals and not saved:
                    saved.append(frame.f_locals['descriptor']);frame.f_locals['descriptor']=badfd
                return trace
            try:
                sys.settrace(trace);answer=reject(lambda:stage.raw_read(cap.runtime_root,'Python'))
                if not saved or 'descriptor identity' not in answer['error']:raise RuntimeError('descriptor attack was not independently rejected: '+str(answer))
                return answer
            finally:
                sys.settrace(previous_trace)
                for fd in [badfd,*saved]:
                    try:os.close(fd)
                    except OSError:pass
    check('captured_descriptor_identity_mutation_rejected',descriptor_identity)
    def preloaded():
        with invented() as (cap,receipt):
            guard=make_guard(cap,receipt);sys.modules[cap.entry_modules[0]]=types.ModuleType(cap.entry_modules[0])
            return reject(lambda:smoke.perform_imports(guard,cap))
    check('preloaded_entry_is_not_fresh_import',preloaded)
    for attack in ('missing_file_binding','extra_file_binding','bad_bool_size','bad_role','ANN_labeled_metadata','license_labeled_weight','hash_tamper','runtime_bool_version','unknown_receipt_field'):
        def role_case(attack=attack):
            with invented() as (cap,r):
                first=next(iter(r['file_bindings']))
                if attack=='missing_file_binding':r['file_bindings'].pop(first)
                elif attack=='extra_file_binding':r['file_bindings']['unlisted.py']=copy.deepcopy(r['file_bindings'][first])
                elif attack=='bad_bool_size':r['file_bindings'][first]['size_bytes']=True
                elif attack=='bad_role':r['file_bindings'][first]['runtime_role']='unreviewed'
                elif attack in ('ANN_labeled_metadata','license_labeled_weight'):
                    name='LICENSE_weights.pt' if attack.startswith('license') else 'INVENTED_targets.ann'
                    raw=b'INVENTED_NOT_REAL_WEIGHTS_OR_ANNOTATIONS';(cap.runtime_root/name).write_bytes(raw)
                    r['population']['file_roles'][name]='distribution_metadata_license';r['file_bindings'][name]={**stage.binding(raw,'runtime'),'runtime_role':'distribution_metadata_license'}
                elif attack=='hash_tamper':(cap.runtime_root/first).write_bytes(b'INVENTED_DIFFERENT')
                elif attack=='runtime_bool_version':r['runtime']=dict(stage.RUNTIME,python=True)
                elif attack=='unknown_receipt_field':r['authorize_read_weights']=True
                return reject(lambda:make_guard(cap,r))
        check('receipt_'+attack,role_case)
    def outside_module():
        with invented() as (cap,r):
            p=cap.runtime_root/'lib/INVENTED_unlisted_probe.py';p.write_text("raise RuntimeError('INVENTED SENTINEL EXECUTED')\n")
            guard=make_guard(cap,r);guard.install()
            try:
                a=reject(lambda:__import__('INVENTED_unlisted_probe'))
                if 'SENTINEL EXECUTED' in a['error']:raise RuntimeError('unlisted loader executed')
                return a
            finally:guard.close()
    check('unknown_dynamic_file_rejected_before_loader',outside_module)
    def final_tamper():
        with invented() as (cap,r):
            guard=make_guard(cap,r);n=next(iter(r['file_bindings']));(cap.runtime_root/n).write_bytes(b'INVENTED_CHANGED_AFTER_CAPTURE')
            return reject(guard.validate_files)
    check('postcapture_runtime_hash_tamper',final_tamper)
    for suffix in ('.ann','.pt','.safetensors','.zip'):
        def audit_suffix(suffix=suffix):
            with invented() as (cap,r):
                target=cap.project_root/('INVENTED_SENTINEL'+suffix);target.write_bytes(b'INVENTED SENTINEL NOT READ')
                state=smoke.project_and_network_audit(cap,(),cap.project_root/'outputs')
                try:
                    a=reject(lambda:target.read_bytes())
                    if not state['denied_events']:raise RuntimeError('audit did not record rejection')
                    return dict(a,sentinel_content_not_read=True)
                finally:state['active']=False
        check('audit_blocks_'+suffix+'_sentinel',audit_suffix)
    def project_read():
        with invented() as (cap,r):
            target=cap.project_root/'INVENTED_future_text.txt';target.write_text('INVENTED NOT A REAL SOURCE')
            state=smoke.project_and_network_audit(cap,(),cap.project_root/'outputs')
            try:return reject(target.read_bytes)
            finally:state['active']=False
    check('audit_blocks_undeclared_project_text',project_read)
    for attack in ('output_traversal','output_symlink'):
        def output_audit(attack=attack):
            with invented() as (cap,r):
                output=cap.project_root/'outputs';output.mkdir();target=cap.project_root/'INVENTED_future_text.txt';target.write_text('INVENTED UNREAD SOURCE')
                if attack=='output_traversal':path=output/'..'/'INVENTED_future_text.txt'
                else:path=output/'INVENTED_alias.txt';path.symlink_to(target)
                state=smoke.project_and_network_audit(cap,(),output)
                try:return reject(path.read_bytes)
                finally:state['active']=False
        check('audit_blocks_'+attack,output_audit)
    def network():
        import socket
        with invented() as (cap,r):
            state=smoke.project_and_network_audit(cap,(),cap.project_root/'outputs')
            try:return reject(lambda:socket.getaddrinfo('INVENTED.invalid',443))
            finally:state['active']=False
    check('audit_blocks_network_before_resolution',network)
    def make_prep_root():
        temp=tempfile.TemporaryDirectory(prefix='INVENTED_prep_helper_');p=Path(temp.name).resolve()
        for name in (prep.STAGE,prep.RUNTIME,prep.SELF):
            q=p/name;q.parent.mkdir(parents=True,exist_ok=True);q.write_bytes(source[name])
        return temp,p
    self_sha=hashlib.sha256(source[prep.SELF]).hexdigest()
    def valid_output():
        temp,p=make_prep_root()
        try:
            captured=prep.capture_sources(p,prep.STAGE_SHA,prep.RUNTIME_SHA,self_sha)
            dest=prep.allocate_output(stage,p,prep.OUTPUT_PREFIX+'/ACTUAL_RUNTIME_SOURCE_PREP_1')
            if not dest.is_dir():raise RuntimeError('valid output failed')
            prep.verify_sources(p,captured);return {'valid_source_capture_and_output':True,'actual_installed_runtime_hash_not_called':True}
        finally:temp.cleanup()
    check('preparer_valid_declared_output_and_current_sources',valid_output)
    for attack in ('wrong_SHA','leaf_symlink','ancestor_symlink','FIFO','postcapture_tamper','noncanonical_output','outside_output'):
        def prep_case(attack=attack):
            temp,p=make_prep_root()
            try:
                if attack=='wrong_SHA':return reject(lambda:prep.capture_sources(p,'0'*64,prep.RUNTIME_SHA,self_sha))
                if attack in ('leaf_symlink','FIFO'):
                    f=p/prep.RUNTIME;f.unlink()
                    if attack=='leaf_symlink':f.symlink_to(p/prep.STAGE)
                    else:os.mkfifo(f)
                    return reject(lambda:prep.capture_sources(p,prep.STAGE_SHA,prep.RUNTIME_SHA,self_sha))
                if attack=='ancestor_symlink':
                    (p/'src').rename(p/'INVENTED_sources');(p/'src').symlink_to(p/'INVENTED_sources',target_is_directory=True)
                    return reject(lambda:prep.capture_sources(p,prep.STAGE_SHA,prep.RUNTIME_SHA,self_sha))
                if attack=='postcapture_tamper':
                    captured=prep.capture_sources(p,prep.STAGE_SHA,prep.RUNTIME_SHA,self_sha);(p/prep.RUNTIME).write_bytes(b'INVENTED CHANGED')
                    return reject(lambda:prep.verify_sources(p,captured))
                name='../escape' if attack=='noncanonical_output' else 'src/ACTUAL_RUNTIME_SOURCE_PREP_1'
                return reject(lambda:prep.allocate_output(stage,p,name))
            finally:temp.cleanup()
        check('preparer_'+attack,prep_case)
    for optimization in ('-O','PYTHONOPTIMIZE'):
        def optimized(optimization=optimization):
            code="import types,sys,pathlib\np=pathlib.Path("+repr(str(root/prep.SELF))+ ")\nm=types.ModuleType('_INVENTED_optimized_prep');m.__file__=str(p);sys.modules[m.__name__]=m\nexec(compile(p.read_bytes(),str(p),'exec'),m.__dict__)\ntry:m.capture_sources(pathlib.Path('/INVENTED_NOT_OPENED'),'0'*64,m.RUNTIME_SHA,'0'*64)\nexcept ValueError as e:print('ALWAYS_ON_REJECT',str(e))\nelse:raise RuntimeError('OPTIMIZED_AUTHORITY_BYPASS')\n"
            command=[sys.executable]+(['-O'] if optimization=='-O' else [])+['-c',code]
            env=dict(os.environ)
            if optimization=='PYTHONOPTIMIZE':env['PYTHONOPTIMIZE']='1'
            r=subprocess.run(command,env=env,stdout=subprocess.PIPE,stderr=subprocess.PIPE,timeout=10,text=True)
            if r.returncode!=0 or 'ALWAYS_ON_REJECT' not in r.stdout:raise RuntimeError(r.stdout+r.stderr)
            return {'captured_exit_code':r.returncode,'stdout':r.stdout,'stderr':r.stderr,'optimization':optimization,'no_source_or_runtime_file_open_by_invalid_gate':True}
        check('preparer_guards_survive_'+optimization,optimized)
    for change in ('success','during_preparation_tamper','symlink_output_ancestor'):
        def mock_preparation(change=change):
            temp,p=make_prep_root();original_loader=prep.load_modules
            try:
                captured=prep.capture_sources(p,prep.STAGE_SHA,prep.RUNTIME_SHA,self_sha)
                calls=[]
                class FakeRuntime:
                    @staticmethod
                    def prepare_receipt():
                        calls.append('INVENTED_ONLY_FAKE_PREPARE')
                        if change=='during_preparation_tamper':(p/prep.RUNTIME).write_bytes(b'INVENTED SOURCE CHANGED DURING MOCK PREPARATION')
                        return {'status':'finite_runtime_code_only_source_receipt_not_fit_authority','file_bindings':{'INVENTED_NEVER_REAL_RUNTIME.py':{'sha256':'0'*64,'size_bytes':3}},
                                'actual_model_weight_cache_or_SC_annotation_read':False,'all_declared_files_start_and_final_hash_checked':True,'OS_GPU_byte_identity_claim':False,
                                'INVENTED_mock_only_not_actual_finite_hash_receipt':True}
                prep.load_modules=lambda root,captured:(stage,FakeRuntime)
                output=prep.OUTPUT_PREFIX+'/ACTUAL_RUNTIME_SOURCE_PREP_1'
                if change=='symlink_output_ancestor':
                    outside=p/'INVENTED_outside';outside.mkdir();(p/'research').mkdir();(p/prep.OUTPUT_PREFIX).symlink_to(outside,target_is_directory=True)
                    a=reject(lambda:prep.run_captured(types.SimpleNamespace(output_dir=output),p,captured))
                    if calls or any(outside.iterdir()):raise RuntimeError('symlink destination wrote or called preparation')
                    return dict(a,no_write_before_guard=True,no_mock_prepare_call=True)
                status=prep.run_captured(types.SimpleNamespace(output_dir=output),p,captured)
                target=p/output;result=json.loads((target/'actual_preparation_result.json').read_bytes())
                if len(calls)!=1:raise RuntimeError('mock population call count differs')
                if change=='success':
                    if status!=0 or result['passed'] is not True:raise RuntimeError('complete mock preparation failed')
                else:
                    if status!=1 or result['passed'] is not False or (target/'finite_runtime_source_only_receipt.json').exists():raise RuntimeError('tampered input obtained hash certificate')
                shutil.copytree(target,out/('INVENTED_mock_preparer_'+change))
                return {'captured_function_return_code':status,'passed_field':result['passed'],'INVENTED_mock_runtime_only':True,
                        'actual_installed_population_hashing':False,'captured_source_changes_withhold_receipt':change=='during_preparation_tamper'}
            finally:prep.load_modules=original_loader;temp.cleanup()
        check('complete_mock_preparer_'+change,mock_preparation)
    def no_real_gate():
        self_digest=hashlib.sha256(source[smoke.SELF]).hexdigest()
        contract=root/smoke.CONTRACT;contract_sha=hashlib.sha256(contract.read_bytes()).hexdigest()
        if (root/smoke.REVIEW).exists():raise RuntimeError('distinct SOURCE review unexpectedly exists; do not run missing-gate check')
        r=subprocess.run([sys.executable,str(root/smoke.SELF),'--root',str(root),'--self-sha256',self_digest,'--receipt-sha256','0'*64,'--source-review-sha256','0'*64,'--contract-sha256',contract_sha,'--output-dir',smoke.OUTPUT_PREFIX+'/ACTUAL_IMPORT_CHECK_987654321'],stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True,timeout=10)
        if r.returncode==0 or (root/smoke.OUTPUT_PREFIX/'ACTUAL_IMPORT_CHECK_987654321').exists():raise RuntimeError('missing review did not stop before output/import')
        return {'captured_exit_code':r.returncode,'stdout':r.stdout,'stderr':r.stderr,'actual_entrypoint_library_import_or_runtime_hashing':False,'output_not_allocated':True}
    check('actual_default_missing_distinct_SOURCE_review_before_import_or_output',no_real_gate)
    if any((root/n).read_bytes()!=raw for n,raw in source.items()):raise RuntimeError('source changed during artificial suite')
    if hashlib.sha256((root/'src/prepare_sccomics_supervised_runtime_v2_source_only.py').read_bytes()).hexdigest()!='940abd90d0de917d7b6db8ebd69d021b9dadca7f21016f4fe56154f098f75810':raise RuntimeError('retained original940 changed')
    result={'identity':'INVENTED_IMPLEMENTER_SELF_CHECK_ONLY_NOT_DIFFERENT_SOURCE_REVIEW','source_bindings':{n:{'sha256':hashlib.sha256(raw).hexdigest(),'size_bytes':len(raw)} for n,raw in source.items()},
            'checks':results,'total':len(results),'passed':sum(r['passed'] for r in results),'failed':sum(not r['passed'] for r in results),
            'actual_SC_text_annotation_archive_model_weight_cache_prediction_fit_or_scientific_score':False,
            'actual_installed_runtime_population_byte_hash_or_real_six_library_import_smoke':False,'retained_original940_and_stable6bba_a46_unchanged':True}
    (out/'fixture_specs_captured_before_cases_saved_after.json').write_text(json.dumps(specs,sort_keys=True,indent=2)+'\n')
    (out/'actual_synthetic_results.json').write_text(json.dumps(result,sort_keys=True,indent=2)+'\n')
    files=[]
    for p in sorted(out.rglob('*')):
        if p.is_file() and p.name!='.DS_Store':
            raw=p.read_bytes();files.append({'path':str(p.relative_to(out)),'sha256':hashlib.sha256(raw).hexdigest(),'size_bytes':len(raw)})
    (out/'output_manifest.json').write_text(json.dumps({'identity':'OWNED_SOURCE_AND_INVENTED_HELPER_CHECK_OUTPUTS','bindings':files},sort_keys=True,indent=2)+'\n')
    print(json.dumps({'total':result['total'],'passed':result['passed'],'failed':result['failed'],'output_dir':args.output_dir},sort_keys=True))
    return 0 if result['failed']==0 else 1

if __name__=='__main__':raise SystemExit(main())
