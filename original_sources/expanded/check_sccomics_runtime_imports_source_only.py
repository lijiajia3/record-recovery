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

STAGE='src/sccomics_supervised_stages_v2.py'
RUNTIME='src/sccomics_supervised_runtime_v2.py'
SELF='src/check_sccomics_runtime_imports_source_only.py'
STAGE_SHA='6bba0ec13e93490d680afb8b306cf9f997cf5d7ed86dabc241aaafc428702308'
RUNTIME_SHA='a46bd4fef4f8c8d9e2afe93cfdb96e286845f86a49c8113e43e847cec3a8f78a'
RECEIPT='research/round4_sccomics_supervised_runtime_receipt.json'
REVIEW='research/round4_sccomics_runtime_import_compatibility_source_review.json'
CONTRACT='research/round4_sccomics_runtime_import_compatibility_source_only_contract.json'
CONTRACT_MD=CONTRACT.removesuffix('.json')+'.md'
AMENDMENT='research/round4_sccomics_supervised_stages_v2_runtime_import_compatibility_prospective_amendment.json'
AMENDMENT_MD=AMENDMENT.removesuffix('.json')+'.md'
AMENDMENT_SHA='7450353904d1faa81aebdecd70028f268b852bc2931a08ded6dc2ef01281d3e2'
AMENDMENT_MD_SHA='31bbca03eecb469b602308bc16b4dbdeb938d9560ed4ddb928f41fad8d1588fc'
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
                          stage.identical(receipt.get('runtime'),stage.RUNTIME) and receipt.get('source_contract_json_sha256')==stage.V2_CONTRACT_SHA and
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
                sys.meta_path.insert(0,self)
                try:stage.require(stage.identical(selected,runtime.population()),'actual metadata population changed; no automatic extension')
                finally:
                    stage.require(sys.meta_path and sys.meta_path[0] is self,'metadata import guard changed');sys.meta_path.remove(self)
    return SourceOnlyGuard


def project_and_network_audit(capability,allowed_project_reads,owned_output):
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
        p=Path(os.path.normpath(os.path.abspath(os.fsdecode(args[0]))))
        if p.suffix.lower() in denied_suffixes:
            state['denied_events'].append({'kind':'model_corpus_archive','suffix':p.suffix.lower()});raise PermissionError('SOURCE-only smoke forbids weights/cache/annotations/archives')
        if p.is_relative_to(project):
            if any(q.is_symlink() for q in (p,*p.parents)):
                state['denied_events'].append({'kind':'project_symlink'});raise PermissionError('SOURCE-only smoke rejects project/output aliases')
            flags=args[2] if len(args)>2 and type(args[2]) is int else 0
            directory_metadata=(flags&os.O_DIRECTORY)!=0 and (flags&(os.O_WRONLY|os.O_RDWR|os.O_CREAT|os.O_APPEND))==0
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
    names={'sccomics_supervised_stages_v2':STAGE,'sccomics_supervised_runtime_v2':RUNTIME}
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
    return get('sccomics_supervised_stages_v2'),get('sccomics_supervised_runtime_v2')


def actual_captured_main(args,root,captured):
    stage,runtime=load_captured_modules(root,captured)
    fixed={AMENDMENT:AMENDMENT_SHA,AMENDMENT_MD:AMENDMENT_MD_SHA,
           CONTRACT:args.contract_sha256,REVIEW:args.source_review_sha256,RECEIPT:args.receipt_sha256}
    for name,wanted in fixed.items():
        stage.require(type(wanted) is str and len(wanted)==64 and all(c in '0123456789abcdef' for c in wanted),'explicit SHA256 required')
        raw=stage.raw_read(root,name);stage.require(stage.sha(raw)==wanted,'source-only gate/receipt bytes differ');captured[name]=raw
    review=stage.unique_json(captured[REVIEW]);contract=stage.unique_json(captured[CONTRACT]);receipt=stage.unique_json(captured[RECEIPT])
    stage.require(type(review) is dict and set(review)=={'status','independent_from_implementer','scope','bound_source_files','bound_protocol_files','entry_libraries','actual_fit_or_SC_semantic_permission'},'closed distinct SOURCE review schema')
    stage.require(review['status']=='source_only_review_passed' and review['independent_from_implementer'] is True and review['scope']=='finite_runtime_import_compatibility_only' and
                  review['actual_fit_or_SC_semantic_permission'] is False and stage.identical(review['entry_libraries'],list(ENTRIES)),'independent review only grants source import scope')
    stage.require(type(contract) is dict and set(contract)=={'identity','entry_libraries','helper_implementation','source_only','actual_fit_or_SC_semantic_permission','original_prospective_amendment_sha256','contract_md_sha256'},'closed prospective smoke contract')
    stage.require(contract['identity']=='PROSPECTIVE_SOURCE_ONLY_FINITE_RUNTIME_IMPORT_COMPATIBILITY' and contract['source_only'] is True and contract['actual_fit_or_SC_semantic_permission'] is False and
                  contract['helper_implementation']==SELF and contract['original_prospective_amendment_sha256']==AMENDMENT_SHA and stage.identical(contract['entry_libraries'],list(ENTRIES)),'source-only contract does not grant fit/Gold')
    raw=stage.raw_read(root,CONTRACT_MD);stage.require(stage.sha(raw)==contract['contract_md_sha256'],'contract Markdown differs');captured[CONTRACT_MD]=raw
    expected_source={n:stage.binding(captured[n],'code') for n in (STAGE,RUNTIME,SELF)}
    expected_protocol={n:stage.binding(captured[n],'source_only_protocol') for n in (CONTRACT,CONTRACT_MD,AMENDMENT,AMENDMENT_MD)}
    stage.require(stage.identical(review['bound_source_files'],expected_source) and stage.identical(review['bound_protocol_files'],expected_protocol),'exact source/protocol review byte bindings')
    stage.relative(args.output_dir);output_name=args.output_dir;p=Path(output_name)
    stage.require(str(p.parent)==OUTPUT_PREFIX and p.name.startswith('ACTUAL_IMPORT_CHECK_') and p.name.removeprefix('ACTUAL_IMPORT_CHECK_').isdigit(),'closed distinct owned source-only output')
    if not (root/OUTPUT_PREFIX).exists():stage.new_directory(root,OUTPUT_PREFIX)
    stage.new_directory(root,output_name);out=root/output_name
    cap=SourceImportCapability(root,Path(sys.base_prefix).resolve(),False,ENTRIES,copy.deepcopy(expected_source),_CONSTRUCT)
    audit_state=None;guard=None
    try:
        for name,raw in captured.items():
            relative=Path('before')/name
            for parent in reversed(relative.parents):
                if str(parent)!='.' and not (out/parent).exists():stage.new_directory(out,str(parent))
            dest=out/relative
            with stage.opened(out,str(dest.relative_to(out)),write=True) as fd:stage.write_all(fd,raw)
        audit_state=project_and_network_audit(cap,tuple(captured),out)
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
        if guard is not None:guard.close()
        if audit_state is not None:audit_state['active']=False
        _CAPABILITIES.remove(cap)
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
