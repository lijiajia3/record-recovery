"""Always-on reviewed SOURCE-only explicit optional runtime preparation, no fit authority.

Replaces the retained 940abd90 preparer, whose assert checks disappear under
Python -O. Every permission/integrity check here is unconditional. No corpus,
model/tokenizer constructor, weights, checkpoint, tensor cache or API is used.
"""
from __future__ import annotations
import argparse
import builtins
import hashlib
import json
import os
from pathlib import Path
import stat
import sys
import types
import traceback

STAGE='src/sccomics_supervised_stages_v4.py';RUNTIME='src/sccomics_supervised_runtime_v4.py'
SELF='src/prepare_sccomics_supervised_runtime_v4_source_only.py'
STAGE_SHA='95c33eac2f286be3ffc0300a55e7afa3fc02c18068a893b80e9a204fa1686341'
RUNTIME_SHA='03f037451ec6516f3b63101b85923bbabffd9ebd560d2ffdb5a1f47dd62a4fbe'
OUTPUT_PREFIX='research/round4_sccomics_supervised_stages_v4_runtime_sourceonly_inputs'


def demand(ok,message):
    if not ok:raise ValueError(message)


def checked_read(path):
    path=Path(path).absolute()
    demand(not any(p.is_symlink() for p in (path,*path.parents)),'source path or ancestor symlink')
    before=path.stat(follow_symlinks=False);demand(stat.S_ISREG(before.st_mode),'regular source before nonblocking open')
    fd=os.open(path,os.O_RDONLY|os.O_NONBLOCK|os.O_NOFOLLOW)
    try:
        start=os.fstat(fd);demand((start.st_dev,start.st_ino)==(before.st_dev,before.st_ino),'opened source identity changed')
        chunks=[]
        while chunk:=os.read(fd,1048576):chunks.append(chunk)
        end=os.fstat(fd)
        demand((start.st_dev,start.st_ino,start.st_size,start.st_mtime_ns,start.st_ctime_ns)==(end.st_dev,end.st_ino,end.st_size,end.st_mtime_ns,end.st_ctime_ns),'source changed while captured')
        raw=b''.join(chunks);demand(len(raw)==start.st_size,'captured source size changed');return raw
    finally:os.close(fd)


def capture_sources(root,stage_sha256,runtime_sha256,self_sha256):
    root=Path(root).absolute()
    demand(stage_sha256==STAGE_SHA and runtime_sha256==RUNTIME_SHA,'fixed reviewed runner/runtime identities')
    wanted={STAGE:stage_sha256,RUNTIME:runtime_sha256,SELF:self_sha256};captured={}
    for name,value in wanted.items():
        demand(type(value) is str and len(value)==64 and all(c in '0123456789abcdef' for c in value),'explicit source SHA256')
        raw=checked_read(root/name);demand(hashlib.sha256(raw).hexdigest()==value,'actual source digest differs');captured[name]=raw
    return captured


def load_modules(root,captured):
    modules={};original=builtins.__import__;names={'sccomics_supervised_stages_v4':STAGE,'sccomics_supervised_runtime_v4':RUNTIME}
    def captured_import(name,globals=None,locals=None,fromlist=(),level=0):
        if name.startswith('sccomics_'):
            demand(level==0 and name in names,'closed runtime preparation scientific imports');return get(name)
        return original(name,globals,locals,fromlist,level)
    def get(name):
        if name not in modules:
            m=types.ModuleType('_sourceprep_v2_'+name);m.__file__=str(root/names[name]);modules[name]=m;sys.modules[m.__name__]=m
            m.__dict__['__builtins__']=dict(vars(builtins),__import__=captured_import)
            exec(compile(captured[names[name]],m.__file__,'exec',dont_inherit=True),m.__dict__)
        return modules[name]
    return get('sccomics_supervised_stages_v4'),get('sccomics_supervised_runtime_v4')


def verify_sources(root,captured):
    for name,raw in captured.items():demand(checked_read(root/name)==raw,'source changed after capture/preparation')


def allocate_output(stage,root,output_name):
    stage.relative(output_name);name=output_name;p=Path(name)
    stage.require(str(p.parent)==OUTPUT_PREFIX and p.name.startswith('ACTUAL_RUNTIME_SOURCE_PREP_') and p.name.removeprefix('ACTUAL_RUNTIME_SOURCE_PREP_').isdigit(),'closed explicit source-only preparation output')
    if not (root/OUTPUT_PREFIX).exists():stage.new_directory(root,OUTPUT_PREFIX)
    stage.new_directory(root,name);return root/name


def run_captured(args,root,captured):
    stage,runtime=load_modules(root,captured)
    review_raw=checked_read(root/REVIEW)
    demand(type(args.source_review_sha256) is str and len(args.source_review_sha256)==64 and hashlib.sha256(review_raw).hexdigest()==args.source_review_sha256,'explicit actual independent preparation-review SHA')
    review=stage.unique_json(review_raw)
    stage.require(type(review) is dict and set(review)=={'status','independent_from_implementer','scope','bound_source_files','bound_protocol_files','entry_libraries','actual_fit_or_SC_semantic_permission'},'closed independent SOURCE preparation review')
    stage.require(review['status']=='source_only_review_passed' and review['independent_from_implementer'] is True and review['scope']=='finite_runtime_source_preparation_only' and review['entry_libraries']==[] and review['actual_fit_or_SC_semantic_permission'] is False,'preparation-only scope, no entry imports or fit/Gold')
    protocols={}
    for name,wanted in zip(PROTOCOL,PROTOCOL_SHA):
        raw=checked_read(root/name);demand(hashlib.sha256(raw).hexdigest()==wanted,'fixed prospective finite optional plan differs');protocols[name]=raw
    stage.require(stage.identical(review['bound_source_files'],{n:stage.binding(raw,'code') for n,raw in captured.items()}) and stage.identical(review['bound_protocol_files'],{n:stage.binding(raw,'source_only_protocol') for n,raw in protocols.items()}),'exact preparation source/protocol closed bindings')
    out=allocate_output(stage,root,args.output_dir)
    def write(name,raw):
        with stage.opened(out,name,write=True) as fd:stage.write_all(fd,raw)
    for name,raw in captured.items():write(Path(name).name+'.before.py.txt',raw)
    try:
        receipt=runtime.prepare_receipt()
        verify_sources(root,captured)
        demand(checked_read(root/REVIEW)==review_raw and all(checked_read(root/n)==raw for n,raw in protocols.items()),"preparation review/protocol changed after capture")
        stage.require(type(receipt) is dict and receipt.get('status')=='finite_runtime_code_only_source_receipt_not_fit_authority' and
                      receipt.get('actual_model_weight_cache_or_SC_annotation_read') is False and receipt.get('all_declared_files_start_and_final_hash_checked') is True and
                      receipt.get('OS_GPU_byte_identity_claim') is False,'source-only finite receipt, not fitting certificate')
        receipt['source_only_preparation_source_bindings']={n:stage.binding(raw,'source_only_code') for n,raw in captured.items()}
        receipt['actual_complete_freeze_final_gate_or_fitting_authority_created']=False
        write('finite_runtime_source_only_receipt.json',stage.serialized(receipt))
        summary={'identity':'FINITE_INSTALLED_RUNTIME_BYTE_PREPARATION_ONLY_NOT_FIT_AUTHORITY','passed':True,
                 'file_count':len(receipt['file_bindings']),'total_bytes':sum(b['size_bytes'] for b in receipt['file_bindings'].values()),
                 'source_input_start_final_bindings':{n:stage.binding(raw,'source_only_code') for n,raw in captured.items()},
                 'actual_SC_text_annotation_model_weight_cache_checkpoint_or_prediction_read':False,
                 'actual_fit_or_library_model_tokenizer_initialization':False,'all_OS_GPU_bytes_claim':False,
                 'receipt_sha256':stage.sha(stage.raw_read(out,'finite_runtime_source_only_receipt.json'))}
    except BaseException as error:
        summary={'identity':'SOURCE_ONLY_RUNTIME_PREPARATION_FAILED_NO_AUTHORITY','passed':False,'error_type':type(error).__name__,
                 'error':str(error),'traceback':traceback.format_exc(),'actual_SC_text_annotation_model_weight_cache_checkpoint_or_prediction_read':False,
                 'actual_fit_or_library_model_tokenizer_initialization':False}
    write('actual_preparation_result.json',stage.serialized(summary))
    files=[]
    for p in sorted(out.iterdir()):
        if p.is_file():files.append({'path':p.name,**stage.digest_file(out,p.name)})
    write('output_manifest.json',stage.serialized({'identity':'SOURCE_ONLY_RUNTIME_PREPARATION_OUTPUTS_NO_FIT_AUTHORITY','bindings':files}))
    print(json.dumps({'passed':summary['passed'],'output_dir':args.output_dir,'actual_fit_permission':False},sort_keys=True))
    return 0 if summary['passed'] else 1


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('output-dir','stage-sha256','runtime-sha256','self-sha256','source-review-sha256'):p.add_argument('--'+name,required=True)
    args=p.parse_args();root=Path(__file__).absolute().parents[1]
    captured=capture_sources(root,args.stage_sha256,args.runtime_sha256,args.self_sha256)
    module=types.ModuleType('_sourceprep_v2_captured_helper');module.__file__=str(root/SELF);sys.modules[module.__name__]=module
    exec(compile(captured[SELF],module.__file__,'exec',dont_inherit=True),module.__dict__)
    return module.run_captured(args,root,captured)


if __name__=='__main__':raise SystemExit(main())
