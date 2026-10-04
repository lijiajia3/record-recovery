"""Bind only SOURCE and explicitly owned artificial runtime-helper artifacts."""
from __future__ import annotations
import hashlib,json,os,stat
from pathlib import Path


def checked(path):
    if any(p.is_symlink() for p in (path,*path.parents)):raise ValueError('artifact symlink')
    before=path.stat(follow_symlinks=False)
    if not stat.S_ISREG(before.st_mode):raise ValueError('regular artifact required')
    fd=os.open(path,os.O_RDONLY|os.O_NONBLOCK|os.O_NOFOLLOW)
    try:
        start=os.fstat(fd);h=hashlib.sha256();size=0
        if (start.st_dev,start.st_ino)!=(before.st_dev,before.st_ino):raise ValueError('opened artifact changed')
        while chunk:=os.read(fd,1048576):h.update(chunk);size+=len(chunk)
        end=os.fstat(fd)
        if (start.st_dev,start.st_ino,start.st_size,start.st_mtime_ns,start.st_ctime_ns)!=(end.st_dev,end.st_ino,end.st_size,end.st_mtime_ns,end.st_ctime_ns) or size!=start.st_size:raise ValueError('artifact changed while read')
        return {'sha256':h.hexdigest(),'size_bytes':size}
    finally:os.close(fd)


def main():
    root=Path(__file__).absolute().parents[1];prefix='research/round4_sccomics_runtime_sourceonly_helpers_inputs'
    out=root/prefix/'STABLE_HELPER_HANDOFF'
    if out.exists():raise ValueError('retain old handoffs, use fresh output')
    out.mkdir()
    files={}
    def add(name):
        b=checked(root/name)
        if name in files and files[name]!=b:raise ValueError('binding changed')
        files[name]=b;return b
    final=prefix+'/INVENTED_v4/'
    summary=json.loads((root/final/'actual_synthetic_results.json').read_bytes())
    if summary['passed']!=37 or summary['failed']!=0 or summary['actual_installed_runtime_population_byte_hash_or_real_six_library_import_smoke'] is not False:raise ValueError('final artificial scope differs')
    for n,b in summary['source_bindings'].items():
        if add(n)!=b:raise ValueError('current source differs '+n)
    for version in ('INVENTED_v1','INVENTED_v2','INVENTED_v3','INVENTED_v4'):
        p=root/prefix/version;m=json.loads((p/'output_manifest.json').read_bytes())
        for b in m['bindings']:
            if add(str((p/b['path']).relative_to(root)))!={k:b[k] for k in ('sha256','size_bytes')}:raise ValueError('saved artificial history changed')
        add(str((p/'output_manifest.json').relative_to(root)))
    for directory in ('ACCIDENTAL_SAME_NAME_WRITE_HISTORY','INITIAL_CANDIDATE_VALID_OUTPUT_TYPE_FAILURE'):
        for p in sorted((root/prefix/directory).rglob('*')):
            if p.is_file() and p.name!='.DS_Store':add(str(p.relative_to(root)))
    controls=('research/round4_sccomics_runtime_import_compatibility_source_only_contract',
              'research/round4_sccomics_supervised_stages_v2_runtime_import_compatibility_prospective_amendment')
    for base in controls:
        for ext in ('.md','.json'):add(base+ext)
    add('research/round4_sccomics_runtime_sourceonly_helpers_implementation.md')
    retained='src/prepare_sccomics_supervised_runtime_v2_source_only.py'
    if add(retained)['sha256']!='940abd90d0de917d7b6db8ebd69d021b9dadca7f21016f4fe56154f098f75810':raise ValueError('retained original940 differs')
    add('src/build_sccomics_runtime_sourceonly_helpers_handoff.py')
    raw=Path(__file__).read_bytes();(out/'builder.before.py.txt').write_bytes(raw);add(str((out/'builder.before.py.txt').relative_to(root)))
    for n in summary['source_bindings']:
        p=out/'before'/n;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes((root/n).read_bytes());add(str(p.relative_to(root)))
    flat={'identity':'SOURCE_ONLY_HELPERS_AND_OWNED_ARTIFICIAL_BINDINGS_NOT_ACTUAL_RUNTIME_OR_FIT_CERTIFICATE',
          'bindings':[{'path':n,**b} for n,b in sorted(files.items())],'all_listed_hash_sizes_actually_reopened_checked':True,
          'actual_installed_runtime_bytes_real_library_import_SC_or_model_access':False,'fit_or_final_freeze_permission':False}
    manifest=out/'implementation_delivery_manifest.json';manifest.write_text(json.dumps(flat,sort_keys=True,indent=2)+'\n')
    report={'identity':'SOURCEONLY_HELPER_IMPLEMENTATION_HANDOFF_NOT_INDEPENDENT_SOURCE_REVIEW','status':'stable_helpers_ready_for_different_agent_review',
            'current_source_bindings':summary['source_bindings'],'implementation_report_md':files['research/round4_sccomics_runtime_sourceonly_helpers_implementation.md'],
            'flat_delivery_manifest':{'path':str(manifest.relative_to(root)),**checked(manifest),'binding_count':len(files)},
            'actual_artificial_history':[{'directory':'INVENTED_v1','total':31,'passed':31,'failed':0,'captured_exit_code':0},
                {'directory':'INVENTED_v2','total':34,'passed':34,'failed':0,'captured_exit_code':0},
                {'directory':'INVENTED_v3','total':35,'passed':34,'failed':1,'captured_exit_code':1,'reason':'read-only directory metadata audit rejected'},
                {'directory':'INVENTED_v4','total':37,'passed':37,'failed':0,'captured_exit_code':0}],
            'original940_retained_but_blocked_for_actual_use':files[retained],
            'original940_brief_same_name_candidate_and_exact_restoration_history_preserved':True,
            'valid_output_TypeError_candidate_and_actual_counterexample_preserved':True,
            'actual_INSTALLED_runtime_byte_preparation_executed':False,'actual_six_real_library_import_compatibility_executed':False,
            'actual_SC_text_ANN_ZIP_model_weight_cache_checkpoint_prediction_fit_API_or_scores':False,
            'no_source_capability_in_stage_fitting_registry':True,'different_source_review_passed':False,'actual_complete_freeze_gate_permission':False,
            'finite_scope_not_all_OS_GPU_or_all_loaded_definition_reexecution':True}
    p=root/'research/round4_sccomics_runtime_sourceonly_helpers_implementation.json'
    if p.exists():raise ValueError('retain old report')
    p.write_text(json.dumps(report,sort_keys=True,indent=2)+'\n')
    result={'identity':'ACTUAL_OWNED_SOURCE_HELPER_HANDOFF_BINDING_EXECUTION','passed':True,'flat_binding_count':len(files),
            'manifest':checked(manifest),'report_json':checked(p),'actual_runtime_or_fit_or_SC_permission':False}
    (out/'actual_builder_result.json').write_text(json.dumps(result,sort_keys=True,indent=2)+'\n')
    print(json.dumps(result,sort_keys=True));return 0

if __name__=='__main__':raise SystemExit(main())
