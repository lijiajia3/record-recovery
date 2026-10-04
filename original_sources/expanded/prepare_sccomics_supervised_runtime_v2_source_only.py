"""Prepare finite installed code/native hashes only; never fit/load task data.

Compiles the two explicitly SHA-bound current source modules from captured
bytes. This is not the supervised entry and creates no actual freeze/gate.
"""
from __future__ import annotations
import argparse
import builtins
import hashlib
import json
from pathlib import Path
import sys
import types
import traceback


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',required=True)
    parser.add_argument('--stage-sha256',required=True)
    parser.add_argument('--runtime-sha256',required=True)
    args=parser.parse_args();root=Path(__file__).resolve().parents[1]
    files={'sccomics_supervised_stages_v2':(root/'src/sccomics_supervised_stages_v2.py',args.stage_sha256),
           'sccomics_supervised_runtime_v2':(root/'src/sccomics_supervised_runtime_v2.py',args.runtime_sha256)}
    captured={}
    for name,(p,wanted) in files.items():
        assert not any(q.is_symlink() for q in (p,*p.parents)) and p.is_file()
        raw=p.read_bytes();assert hashlib.sha256(raw).hexdigest()==wanted;captured[name]=raw
    modules={};original=builtins.__import__
    def captured_import(name,globals=None,locals=None,fromlist=(),level=0):
        if name.startswith('sccomics_'):
            assert level==0 and name in captured
            return get(name)
        return original(name,globals,locals,fromlist,level)
    def get(name):
        if name not in modules:
            module=types.ModuleType('_sc_sourceonly_'+name);module.__file__=str(files[name][0]);modules[name]=module;sys.modules[module.__name__]=module
            module.__dict__['__builtins__']=dict(vars(builtins),__import__=captured_import)
            exec(compile(captured[name],module.__file__,'exec',dont_inherit=True),module.__dict__)
        return modules[name]
    stage=get('sccomics_supervised_stages_v2');runtime=get('sccomics_supervised_runtime_v2')
    out=Path(args.output_dir).absolute();prefix=root/'research/round4_sccomics_supervised_stages_v2_runtime_sourceonly_inputs'
    assert out.parent==prefix and not out.exists()
    if not prefix.exists():stage.new_directory(root,str(prefix.relative_to(root)))
    stage.root_path(out.parent)
    stage.new_directory(out.parent,out.name)
    def write(name,raw):
        with stage.opened(out,name,write=True) as fd:stage.write_all(fd,raw)
    for name,raw in captured.items():write(name+'.before.py.txt',raw)
    write('helper.before.py.txt',Path(__file__).read_bytes())
    try:
        receipt=runtime.prepare_receipt()
        for name,(p,wanted) in files.items():assert p.read_bytes()==captured[name] and hashlib.sha256(captured[name]).hexdigest()==wanted
        receipt['source_only_preparation_source_bindings']={str(p.relative_to(root)):{'sha256':wanted,'size_bytes':len(captured[name])} for name,(p,wanted) in files.items()}
        receipt['actual_complete_freeze_final_gate_or_fitting_authority_created']=False
        write('finite_runtime_source_only_receipt.json',stage.serialized(receipt))
        summary={'identity':'ACTUAL_INSTALLED_FINITE_CODE_NATIVE_HASH_PREPARATION_ONLY_NOT_FIT_AUTHORITY',
                 'passed':True,'files':len(receipt['file_bindings']),'total_bytes':sum(b['size_bytes'] for b in receipt['file_bindings'].values()),
                 'actual_SC_model_weight_cache_prediction_or_annotation_read':False,'all_OS_GPU_bytes_claim':False,
                 'receipt_sha256':stage.sha(stage.raw_read(out,'finite_runtime_source_only_receipt.json'))}
    except BaseException as error:
        summary={'identity':'SOURCE_ONLY_RUNTIME_PREPARATION_FAILED_NO_AUTHORITY','passed':False,'error_type':type(error).__name__,
                 'error':str(error),'traceback':traceback.format_exc(),'actual_SC_model_weight_cache_prediction_or_annotation_read':False}
    write('actual_preparation_result.json',stage.serialized(summary));print(json.dumps(summary,sort_keys=True))
    return 0 if summary['passed'] else 1

if __name__=='__main__':raise SystemExit(main())
