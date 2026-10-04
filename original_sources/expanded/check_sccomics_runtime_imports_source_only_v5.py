"""Reviewed SOURCE-only six-library import smoke; never a fitting authority.

Default actual entry requires a separate review, contract and finite runtime
receipt. The distinct capability is never registered in stage._SCOPES. No
backend, corpus, tokenizer/model constructor, tensor/checkpoint loader or fit
method is called. Actual entry imports only the fixed six library entrypoints.
"""
from __future__ import annotations
import argparse
import builtins
import copy
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass,field
import hashlib
import importlib
import json
import os
from pathlib import Path
import stat
import sys
import tempfile
import traceback
import types

STAGE='src/sccomics_supervised_stages_v4.py'
RUNTIME='src/sccomics_supervised_runtime_v4.py'
SELF='src/check_sccomics_runtime_imports_source_only_v5.py'
STAGE_SHA='95c33eac2f286be3ffc0300a55e7afa3fc02c18068a893b80e9a204fa1686341'
RUNTIME_SHA='03f037451ec6516f3b63101b85923bbabffd9ebd560d2ffdb5a1f47dd62a4fbe'
RECEIPT='research/round4_sccomics_supervised_runtime_expanded_receipt_v1.json'
REVIEW='research/round4_sccomics_runtime_import_compatibility_v5_source_review.json'
CONTRACT='research/round4_sccomics_runtime_import_compatibility_v5_source_only_contract.json'
CONTRACT_MD=CONTRACT.removesuffix('.json')+'.md'
AMENDMENT='research/round4_sccomics_supervised_stages_v2_runtime_import_compatibility_prospective_amendment.json'
AMENDMENT_MD=AMENDMENT.removesuffix('.json')+'.md'
AMENDMENT_SHA='7450353904d1faa81aebdecd70028f268b852bc2931a08ded6dc2ef01281d3e2'
AMENDMENT_MD_SHA='31bbca03eecb469b602308bc16b4dbdeb938d9560ed4ddb928f41fad8d1588fc'
MISSING_ADDENDUM='research/round4_sccomics_runtime_missing_module_before_fit_corrective_addendum.json'
MISSING_ADDENDUM_MD=MISSING_ADDENDUM.removesuffix('.json')+'.md'
MISSING_ADDENDUM_SHA='c994a7fde4cb98e3493e2ac54d96ded9a2890bf60d4e6f5a23723f70b1e34d2f'
MISSING_ADDENDUM_MD_SHA='8a6be91d31d17ecb0f731058cf4845dd05d39c1290f5d12434719897a4779201'
DEPENDENCIES='research/round4_sccomics_runtime_optional_dependencies_before_fit_source_contract.json'
DEPENDENCIES_MD=DEPENDENCIES.removesuffix('.json')+'.md'
DEPENDENCIES_SHA='ce5bb7b1043db5d19fbcb8b20b5a9ab8555945362e2297c5109bf662e01b3209'
DEPENDENCIES_MD_SHA='d78a83e08742f9d7d7236b78e672c84d039ab8cb2a88bf71339d32f06b727710'
PLAN='research/round4_sccomics_runtime_optional_inventory_inputs/prospective_finite_distribution_plan.json'
PLAN_SHA='742a027436d3734c75a1154327f5d9a78fa3d1d8d344df4a5891a40238323eb5'
ENTRIES=('torch','numpy','tokenizers','transformers','networkx','safetensors')
OUTPUT_PREFIX='research/round4_sccomics_supervised_stages_v2_runtime_sourceonly_inputs'
_CAPABILITIES=set();_CONSTRUCT=object()


def demand(ok,message):
    if not ok:raise ValueError(message)


def bootstrap_read(root,name):
    p=Path(name)
    demand(type(name) is str and not p.is_absolute() and '..' not in p.parts and str(p)==name,'explicit canonical root-relative source path')
    root=Path(root).absolute();path=root/p
    demand(not any(q.is_symlink() for q in (path,*path.parents)),'source path/ancestor symlink')
    before=path.stat(follow_symlinks=False);demand(stat.S_ISREG(before.st_mode),'regular source before open')
    fd=os.open(path,os.O_RDONLY|os.O_NONBLOCK|os.O_NOFOLLOW)
    try:
        s=os.fstat(fd);demand((s.st_dev,s.st_ino)==(before.st_dev,before.st_ino),'source opened identity differs')
        chunks=[]
        while chunk:=os.read(fd,1048576):chunks.append(chunk)
        end=os.fstat(fd)
        demand((s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns)==(end.st_dev,end.st_ino,end.st_size,end.st_mtime_ns,end.st_ctime_ns),'source changed during capture')
        raw=b''.join(chunks);demand(len(raw)==s.st_size,'source size changed');return raw
    finally:os.close(fd)


def digest(raw):return hashlib.sha256(raw).hexdigest()


@dataclass(frozen=True,eq=False)
class SourceImportCapability:
    project_root:Path
    runtime_root:Path
    invented:bool
    entry_modules:tuple
    captured_bindings:dict=field(repr=False)
    token:object=field(repr=False)

    def __post_init__(self):
        demand(self.token is _CONSTRUCT,'capability construction is private')
        demand(type(self.invented) is bool,'typed source-only capability')
        demand(type(self.entry_modules) is tuple and len(self.entry_modules)==6,'six declared entry modules')
        if self.invented:
            demand(self.project_root.parent==Path(tempfile.gettempdir()).resolve() and self.project_root.name.startswith('INVENTED_sccomics_imports_'),'newly owned invented root')
            demand(self.runtime_root==self.project_root/'INVENTED_runtime' and all(n.startswith('INVENTED_library_') for n in self.entry_modules),'invented runtime cannot name actual framework/libraries')
        else:
            demand(self.runtime_root==Path(sys.base_prefix).resolve() and self.entry_modules==ENTRIES,'actual fixed framework/six entries')
        _CAPABILITIES.add(self)


@contextmanager
def invented_source_import_capability():
    with tempfile.TemporaryDirectory(prefix='INVENTED_sccomics_imports_') as directory:
        root=Path(directory).resolve();(root/'INVENTED_runtime').mkdir()
        cap=SourceImportCapability(root,root/'INVENTED_runtime',True,tuple('INVENTED_library_'+str(i) for i in range(6)),{},_CONSTRUCT)
        try:yield cap
        finally:_CAPABILITIES.remove(cap)


def guard_class(stage,runtime):
    """Distinct authority; only reuse finite import/inventory/check methods.

    Inherited inventory needs a boolean invented flag at self.scope. That
    attribute holds SourceImportCapability, never stage._Scope or fit authority.
    """
    class SourceOnlyGuard(runtime.BoundRuntime):
        def __init__(self,receipt,capability):
            stage.require(capability in _CAPABILITIES and isinstance(capability,SourceImportCapability),'genuine separate source-only authority')
            stage.require(capability not in stage._SCOPES,'source capability must never be registered as fitting authority')
            stage.require(type(receipt) is dict and receipt.get('status')=='finite_runtime_code_only_source_receipt_not_fit_authority' and
                          stage.identical(receipt.get('runtime'),stage.RUNTIME) and receipt.get('source_contract_json_sha256')==stage.V4_DEPENDENCIES_SHA and
                          receipt.get('actual_model_weight_cache_or_SC_annotation_read') is False and
                          receipt.get('all_declared_files_start_and_final_hash_checked') is True and receipt.get('OS_GPU_byte_identity_claim') is False,'finite source-only receipt identity')
            required={'status','runtime','population','file_bindings','source_contract_json_sha256','actual_model_weight_cache_or_SC_annotation_read','all_declared_files_start_and_final_hash_checked','OS_GPU_byte_identity_claim','platform_version_only'}
            stage.require(required<=set(receipt) and set(receipt)<=required|{'source_only_preparation_source_bindings','actual_complete_freeze_final_gate_or_fitting_authority_created'},'closed source-only receipt fields')
            if 'actual_complete_freeze_final_gate_or_fitting_authority_created' in receipt:stage.require(receipt['actual_complete_freeze_final_gate_or_fitting_authority_created'] is False,'receipt never grants fitting')
            self.receipt=copy.deepcopy(receipt);self.scope=capability;self.events=[];self.active=False;self.sciences=[]
            selected=receipt['population'];self.root=stage.root_path(selected['framework_root'])
            stage.require(self.root==capability.runtime_root,'declared source-only runtime root')
            stage.require(selected.get('policy')==runtime.POLICY and selected.get('all_OS_GPU_or_system_dynamic_libraries_byte_bound') is False,'finite scope, not all system bytes')
            self.inputs=copy.deepcopy(receipt['file_bindings'])
            stage.require(type(self.inputs) is dict and set(self.inputs)==set(selected['file_roles']),'exact finite file population')
            for name,b in self.inputs.items():
                stage.relative(name);p=Path(name);r=selected['file_roles'][name]
                stage.require(type(b) is dict and set(b)=={'sha256','size_bytes','duty','runtime_role'} and b['duty']=='runtime' and b['runtime_role']==r,'typed runtime binding/role')
                stage.strict_binding({k:b[k] for k in ('sha256','size_bytes','duty')})
                if r in {'package_code_native','stdlib_code_native'}:stage.require(p.suffix in {'.py','.so','.dylib'} and '__pycache__' not in p.parts,'code/native excludes model/cache/archive/annotation/pyc')
                elif r=='distribution_metadata_license':stage.require(p.name in runtime.METADATA or runtime.license_name(p,directory='licenses' in p.parts),'closed metadata document')
                elif r in {'package_license','Python_license'}:stage.require(runtime.license_name(p),'closed license document')
                elif r=='Python_executable':stage.require(name==selected['executable_relative_path'] and p.name in {'python','python3','python3.13'},'fixed executable')
                elif r=='Python_framework_binary':stage.require(name=='Python','canonical framework binary')
                else:raise stage.StageIntegrityError('unreviewed runtime role')
            for name in (selected['stdlib_relative_path'],selected['site_relative_path'],selected['executable_relative_path'],*selected['package_roots']):stage.relative(name)
            self.namespaces=[self.root/name for name in selected['package_roots']]
            self.validate_files()
            if not capability.invented:
                self._claim_meta_path()
                try:stage.require(stage.identical(selected,runtime.population()),'actual metadata population changed; no automatic extension')
                finally:self._release_meta_path()
    return SourceOnlyGuard


@contextmanager
def verified_open_intent_adapter(stage,capability,allowed_project_reads,runtime_names):
    """Read-only intent for the exact captured stage.opened implementation.

    CPython's open audit tuple omits dir_fd. Match only that implementation's
    frame and exact next root/ancestor/leaf operation. No intent is active
    while the opened descriptor is yielded to its caller. All actual paths
    still pass the unchanged suffix/project/output policy.
    """
    demand(capability in _CAPABILITIES,'genuine separate SOURCE capability')
    original=stage.opened;original_code=original.__wrapped__.__code__
    current=ContextVar('SC_source_only_verified_open_intent',default=None)
    allowed={(capability.project_root,n) for n in allowed_project_reads}
    allowed|={(capability.runtime_root,n) for n in runtime_names}
    for root,name in allowed:stage.root_path(root);stage.relative(name)
    class Adapter:
        def resolve(self,args):
            context=current.get();frame=sys._getframe(2)
            if context is None or frame.f_code is not original_code:return None
            demand(stage.opened is wrapped,'captured file reader adapter changed')
            demand(frame.f_locals.get('root')==context['root'] and frame.f_locals.get('parts')==context['parts'] and frame.f_locals.get('write') is False,'exact captured file-reader arguments')
            index=context['index'];operations=context['operations']
            demand(index<len(operations),'unexpected extra open from captured reader')
            operation=operations[index];path=os.fsdecode(args[0]);flags=args[2] if len(args)>2 and type(args[2]) is int else None
            demand(path==operation['raw_path'] and flags is not None and (flags&~os.O_CLOEXEC)==operation['flags'],'exact captured open path/flags sequence')
            if index:
                descriptor=frame.f_locals.get('descriptor')
                demand(type(descriptor) is int,'captured directory descriptor')
                actual=os.fstat(descriptor);parent=operation['resolved_path'].parent
                demand(not any(p.is_symlink() for p in (parent,*parent.parents)),'directory intent ancestor alias')
                expected=os.stat(parent,follow_symlinks=False)
                demand(stat.S_ISDIR(actual.st_mode) and (actual.st_dev,actual.st_ino)==(expected.st_dev,expected.st_ino),'captured directory descriptor identity differs')
            context['index']+=1
            return operation['resolved_path']
    adapter=Adapter()
    @contextmanager
    def wrapped(root,name,*,write=False):
        root=stage.root_path(root);parts=stage.relative(name)
        demand(write is False and (root,name) in allowed,'active SOURCE file-reader intent is read-only and explicitly declared')
        directory_flags=os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW
        operations=[{'raw_path':str(root),'resolved_path':root,'flags':directory_flags}]
        parent=root
        for component in parts[:-1]:
            parent=parent/component;operations.append({'raw_path':component,'resolved_path':parent,'flags':directory_flags})
        operations.append({'raw_path':parts[-1],'resolved_path':parent/parts[-1],'flags':os.O_RDONLY|os.O_NOFOLLOW|os.O_NONBLOCK})
        context={'root':root,'parts':parts,'operations':operations,'index':0};token=current.set(context)
        try:
            with original(root,name,write=False) as descriptor:
                demand(context['index']==len(operations),'complete exact file-reader open sequence was not audited')
                paused=current.set(None)
                try:yield descriptor
                finally:current.reset(paused)
        finally:current.reset(token)
    stage.opened=wrapped
    try:yield adapter
    finally:
        changed=stage.opened is not wrapped
        stage.opened=original
        demand(not changed,'SOURCE file reader replaced during adapter scope')


def project_and_network_audit(capability,allowed_project_reads,owned_output,*,intent_adapter=None):
    """Deny project data/model access and Python socket operations during smoke.

    Not a universal native OS I/O sandbox. Code/native provenance and ordinary
    resource limitations remain exactly those in the finite runtime contract.
    """
    state={'active':True,'denied_events':[]};project=capability.project_root
    denied_suffixes={'.ann','.zip','.tar','.gz','.pt','.pth','.ckpt','.safetensors','.bin'}
    allowed={project/n for n in allowed_project_reads}
    def audit(event,args):
        if not state['active']:return
        if event.startswith('socket.') and event in {'socket.connect','socket.connect_ex','socket.getaddrinfo','socket.bind','socket.sendto'}:
            state['denied_events'].append({'kind':'network','event':event});raise PermissionError('SOURCE-only smoke forbids network')
        if event!='open' or not args or isinstance(args[0],int):return
        flags=args[2] if len(args)>2 and type(args[2]) is int else 0
        directory_metadata=(flags&os.O_DIRECTORY)!=0 and (flags&(os.O_WRONLY|os.O_RDWR|os.O_CREAT|os.O_APPEND))==0
        verified=intent_adapter.resolve(args) if intent_adapter is not None else None
        raw_path=os.fsdecode(args[0])
        if verified is None and not Path(raw_path).is_absolute() and not directory_metadata:
            state['denied_events'].append({'kind':'unverified_relative_file_open'});raise PermissionError('SOURCE-only smoke rejects relative file opens without verified directory intent')
        # Keep the OS spelling, including '..', until symlink metadata resolution.
        # normpath/abspath would erase alias/.. semantics before classifying its
        # true target. Relative files require exact intent above; the existing
        # read-only directory exception is anchored without lexical collapse.
        spelling=Path(raw_path)
        p=verified if verified is not None else (spelling if spelling.is_absolute() else Path.cwd()/spelling)
        if p.suffix.lower() in denied_suffixes:
            state['denied_events'].append({'kind':'model_corpus_archive','suffix':p.suffix.lower(),'path_scope':'raw_spelling'});raise PermissionError('SOURCE-only smoke forbids weights/cache/annotations/archives')
        try:resolved=p.resolve(strict=False)
        except (OSError,RuntimeError) as error:
            state['denied_events'].append({'kind':'unresolved_path_metadata','error_type':type(error).__name__});raise PermissionError('SOURCE-only smoke cannot establish actual path metadata') from error
        if resolved.suffix.lower() in denied_suffixes:
            state['denied_events'].append({'kind':'model_corpus_archive','suffix':resolved.suffix.lower(),'path_scope':'resolved_target'});raise PermissionError('SOURCE-only smoke forbids weights/cache/annotations/archives')
        if p.is_relative_to(project) or resolved.is_relative_to(project):
            if p!=resolved or any(q.is_symlink() for q in (p,*p.parents)):
                state['denied_events'].append({'kind':'project_symlink_or_noncanonical_spelling'});raise PermissionError('SOURCE-only smoke rejects project/output aliases')
            if not directory_metadata and p not in allowed and not p.is_relative_to(owned_output):
                state['denied_events'].append({'kind':'undeclared_project_file','relative_path':str(p.relative_to(project))});raise PermissionError('SOURCE-only smoke forbids undeclared project file')
    sys.addaudithook(audit)
    return state


def perform_imports(guard,capability):
    demand(capability in _CAPABILITIES and guard.scope is capability,'genuine separate source-only scope')
    demand(not any(n==e or n.startswith(e+'.') for n in sys.modules for e in capability.entry_modules),'entry libraries already loaded; no honest fresh import smoke')
    guard.install();loaded=[]
    try:
        for name in capability.entry_modules:
            module=importlib.import_module(name);loaded.append({'module':name,'file':getattr(module,'__file__',None)})
        result=guard.validate()
        return {'entry_libraries_imported':loaded,'runtime_validation':result,'source_only':True,
                'tokenizer_model_or_backend_constructed':False,'all_loaded_definitions_reexecuted_claim':False,'all_OS_GPU_bytes_claim':False}
    finally:guard.close()


def source_module_inventory_adapter(root,source_bindings):
    """Only the three already captured source paths, with bootstrap hashes."""
    demand(set(source_bindings)=={STAGE,RUNTIME,SELF},'exact captured source adapter')
    return types.SimpleNamespace(files={n:Path(root)/n for n in source_bindings},
                                 bound=types.SimpleNamespace(root=Path(root),inputs=copy.deepcopy(source_bindings)))


def load_captured_modules(root,captured):
    modules={};original=builtins.__import__
    names={'sccomics_supervised_stages_v4':STAGE,'sccomics_supervised_runtime_v4':RUNTIME}
    def local_import(name,globals=None,locals=None,fromlist=(),level=0):
        if name.startswith('sccomics_'):
            demand(level==0 and name in names,'closed smoke source imports');return get(name)
        return original(name,globals,locals,fromlist,level)
    def get(name):
        if name not in modules:
            m=types.ModuleType('_source_import_'+name);m.__file__=str(root/names[name]);modules[name]=m;sys.modules[m.__name__]=m
            m.__dict__['__builtins__']=dict(vars(builtins),__import__=local_import)
            exec(compile(captured[names[name]],m.__file__,'exec',dont_inherit=True),m.__dict__)
        return modules[name]
    return get('sccomics_supervised_stages_v4'),get('sccomics_supervised_runtime_v4')


def actual_captured_main(args,root,captured):
    stage,runtime=load_captured_modules(root,captured)
    fixed={AMENDMENT:AMENDMENT_SHA,AMENDMENT_MD:AMENDMENT_MD_SHA,
           MISSING_ADDENDUM:MISSING_ADDENDUM_SHA,MISSING_ADDENDUM_MD:MISSING_ADDENDUM_MD_SHA,
           DEPENDENCIES:DEPENDENCIES_SHA,DEPENDENCIES_MD:DEPENDENCIES_MD_SHA,PLAN:PLAN_SHA,
           CONTRACT:args.contract_sha256,REVIEW:args.source_review_sha256,RECEIPT:args.receipt_sha256}
    for name,wanted in fixed.items():
        stage.require(type(wanted) is str and len(wanted)==64 and all(c in '0123456789abcdef' for c in wanted),'explicit SHA256 required')
        raw=stage.raw_read(root,name);stage.require(stage.sha(raw)==wanted,'source-only gate/receipt bytes differ');captured[name]=raw
    review=stage.unique_json(captured[REVIEW]);contract=stage.unique_json(captured[CONTRACT]);receipt=stage.unique_json(captured[RECEIPT])
    stage.require(stage.unique_json(captured[MISSING_ADDENDUM])['md_sha256']==MISSING_ADDENDUM_MD_SHA and stage.V3_MISSING_SHA==MISSING_ADDENDUM_SHA and stage.V3_MISSING_MD_SHA==MISSING_ADDENDUM_MD_SHA,'fixed missing-module corrective SOURCE crossbinding')
    stage.require(stage.unique_json(captured[DEPENDENCIES])['md_sha256']==DEPENDENCIES_MD_SHA and stage.V4_DEPENDENCIES_SHA==DEPENDENCIES_SHA and stage.V4_RUNTIME_PLAN_SHA==PLAN_SHA and stage.unique_json(captured[DEPENDENCIES])['finite_distribution_plan_sha256']==PLAN_SHA,'fixed finite optional runtime correction crossbinding')
    stage.require(type(review) is dict and set(review)=={'status','independent_from_implementer','scope','bound_source_files','bound_protocol_files','entry_libraries','actual_fit_or_SC_semantic_permission'},'closed distinct SOURCE review schema')
    stage.require(review['status']=='source_only_review_passed' and review['independent_from_implementer'] is True and review['scope']=='finite_runtime_import_compatibility_only' and
                  review['actual_fit_or_SC_semantic_permission'] is False and stage.identical(review['entry_libraries'],list(ENTRIES)),'independent review only grants source import scope')
    stage.require(type(contract) is dict and set(contract)=={'identity','entry_libraries','helper_implementation','source_only','actual_fit_or_SC_semantic_permission','original_prospective_amendment_sha256','contract_md_sha256'},'closed prospective smoke contract')
    stage.require(contract['identity']=='PROSPECTIVE_SOURCE_ONLY_FINITE_RUNTIME_IMPORT_COMPATIBILITY' and contract['source_only'] is True and contract['actual_fit_or_SC_semantic_permission'] is False and
                  contract['helper_implementation']==SELF and contract['original_prospective_amendment_sha256']==AMENDMENT_SHA and stage.identical(contract['entry_libraries'],list(ENTRIES)),'source-only contract does not grant fit/Gold')
    raw=stage.raw_read(root,CONTRACT_MD);stage.require(stage.sha(raw)==contract['contract_md_sha256'],'contract Markdown differs');captured[CONTRACT_MD]=raw
    expected_source={n:stage.binding(captured[n],'code') for n in (STAGE,RUNTIME,SELF)}
    expected_protocol={n:stage.binding(captured[n],'source_only_protocol') for n in (CONTRACT,CONTRACT_MD,AMENDMENT,AMENDMENT_MD,MISSING_ADDENDUM,MISSING_ADDENDUM_MD,DEPENDENCIES,DEPENDENCIES_MD,PLAN)}
    stage.require(stage.identical(review['bound_source_files'],expected_source) and stage.identical(review['bound_protocol_files'],expected_protocol),'exact source/protocol review byte bindings')
    stage.relative(args.output_dir);output_name=args.output_dir;p=Path(output_name)
    stage.require(str(p.parent)==OUTPUT_PREFIX and p.name.startswith('ACTUAL_IMPORT_CHECK_') and p.name.removeprefix('ACTUAL_IMPORT_CHECK_').isdigit(),'closed distinct owned source-only output')
    if not (root/OUTPUT_PREFIX).exists():stage.new_directory(root,OUTPUT_PREFIX)
    stage.new_directory(root,output_name);out=root/output_name
    cap=SourceImportCapability(root,Path(sys.base_prefix).resolve(),False,ENTRIES,copy.deepcopy(expected_source),_CONSTRUCT)
    audit_state=None;guard=None;intent_manager=None;intent_active=False
    try:
        for name,raw in captured.items():
            relative=Path('before')/name
            for parent in reversed(relative.parents):
                if str(parent)!='.' and not (out/parent).exists():stage.new_directory(out,str(parent))
            dest=out/relative
            with stage.opened(out,str(dest.relative_to(out)),write=True) as fd:stage.write_all(fd,raw)
        intent_manager=verified_open_intent_adapter(stage,cap,tuple(captured),tuple(receipt['file_bindings']))
        intent_adapter=intent_manager.__enter__();intent_active=True
        audit_state=project_and_network_audit(cap,tuple(captured),out,intent_adapter=intent_adapter)
        guard=guard_class(stage,runtime)(receipt,cap)
        # Distinct source paths, already captured and byte-checked. This is not
        # a fitting science registry and grants no model/data capability.
        guard.allow_science(source_module_inventory_adapter(root,expected_source))
        result=perform_imports(guard,cap)
        for name,raw in captured.items():stage.require(stage.raw_read(root,name)==raw,'source-only input changed during smoke')
        result.update(identity='ACTUAL_FINITE_RUNTIME_IMPORT_COMPATIBILITY_ONLY_NOT_FIT_AUTHORITY',passed=True,
                      original_input_bindings={n:stage.binding(raw,'source_only_input') for n,raw in captured.items()},
                      stage_fitting_registry_unchanged=cap not in stage._SCOPES,
                      actual_SC_annotation_text_model_weight_cache_checkpoint_prediction_read=False,
                      actual_fitting_inference_or_scientific_statistic=False)
    except BaseException as error:
        result={'identity':'SOURCE_ONLY_RUNTIME_IMPORT_COMPATIBILITY_FAILED_NO_FIT_AUTHORITY','passed':False,'error_type':type(error).__name__,'error':str(error),'traceback':traceback.format_exc(),
                'stage_fitting_registry_unchanged':cap not in stage._SCOPES,'actual_fitting_inference_or_scientific_statistic':False}
    finally:
        try:
            if guard is not None:guard.close()
        finally:
            if audit_state is not None:audit_state['active']=False
            try:
                if intent_active:intent_manager.__exit__(None,None,None)
            finally:_CAPABILITIES.remove(cap)
    result['audit_denied_events']=copy.deepcopy(audit_state['denied_events']) if audit_state else []
    raw=stage.serialized(result)
    with stage.opened(out,'actual_import_compatibility_result.json',write=True) as fd:stage.write_all(fd,raw)
    files=[]
    for file in sorted(out.rglob('*')):
        if file.is_file():files.append({'path':str(file.relative_to(out)),**stage.digest_file(out,str(file.relative_to(out)))})
    with stage.opened(out,'output_manifest.json',write=True) as fd:stage.write_all(fd,stage.serialized({'identity':'SOURCE_ONLY_IMPORT_COMPATIBILITY_OUTPUTS_NOT_SCIENTIFIC_AUTHORITY','bindings':files}))
    print(json.dumps({'passed':result['passed'],'output_dir':output_name,'fit_or_SC_semantic_permission':False},sort_keys=True))
    return 0 if result['passed'] else 1


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('root','self-sha256','receipt-sha256','source-review-sha256','contract-sha256','output-dir'):p.add_argument('--'+name,required=True)
    args=p.parse_args();root=Path(args.root).absolute()
    captured={STAGE:bootstrap_read(root,STAGE),RUNTIME:bootstrap_read(root,RUNTIME),SELF:bootstrap_read(root,SELF)}
    demand(digest(captured[STAGE])==STAGE_SHA and digest(captured[RUNTIME])==RUNTIME_SHA and digest(captured[SELF])==args.self_sha256,'explicit immutable source bytes')
    demand(Path(__file__).absolute()==root/SELF,'fixed actual source helper path')
    module=types.ModuleType('_source_import_compatibility_captured');module.__file__=str(root/SELF);sys.modules[module.__name__]=module
    exec(compile(captured[SELF],module.__file__,'exec',dont_inherit=True),module.__dict__)
    return module.actual_captured_main(args,root,captured)


if __name__=='__main__':raise SystemExit(main())
