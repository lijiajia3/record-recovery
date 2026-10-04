"""Bind owned SOURCE/INVENTED artifacts without actual corpus/model access."""
from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path
import stat


def hash_file(path):
    assert not any(p.is_symlink() for p in (path,*path.parents))
    before=path.stat(follow_symlinks=False);assert stat.S_ISREG(before.st_mode)
    fd=os.open(path,os.O_RDONLY|os.O_NONBLOCK|os.O_NOFOLLOW)
    try:
        s=os.fstat(fd);assert (s.st_dev,s.st_ino)==(before.st_dev,before.st_ino)
        h=hashlib.sha256();size=0
        while chunk:=os.read(fd,1048576):h.update(chunk);size+=len(chunk)
        after=os.fstat(fd)
        assert (s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns)==(after.st_dev,after.st_ino,after.st_size,after.st_mtime_ns,after.st_ctime_ns)
        assert size==s.st_size
        return {'sha256':h.hexdigest(),'size_bytes':size}
    finally:os.close(fd)


def main():
    root=Path(__file__).absolute().parents[1]
    out=root/'research/round4_sccomics_supervised_stages_v2_sourceonly_inputs/STABLE_SOURCE_HANDOFF'
    assert not out.exists() and not any(p.is_symlink() for p in (out.parent,*out.parent.parents))
    out.mkdir()
    own=Path(__file__).read_bytes();(out/'builder.before.py.txt').write_bytes(own)
    fixture='research/round4_sccomics_supervised_stages_v2_synthetic_inputs/INVENTED_v2/'
    manifest_name=fixture+'output_manifest.json';summary_name=fixture+'actual_synthetic_results.json'
    originals=json.loads((root/manifest_name).read_bytes());summary=json.loads((root/summary_name).read_bytes())
    verification_name='research/round4_sccomics_supervised_stages_v2_sourceonly_inputs/POSTEXEC_INVENTED_v2_NONFINDER/actual_verification_result.json'
    checked=json.loads((root/verification_name).read_bytes())
    assert checked['passed'] is True and checked['all_nonFinder_artifact_hashes_and_sizes_matched'] is True
    assert summary['total']==summary['passed']==92 and summary['failed']==0
    assert summary['source_sha256']=='6bba0ec13e93490d680afb8b306cf9f997cf5d7ed86dabc241aaafc428702308'
    assert summary['source_bindings']['src/sccomics_supervised_runtime_v2.py']['sha256']=='a46bd4fef4f8c8d9e2afe93cfdb96e286845f86a49c8113e43e847cec3a8f78a'
    assert summary['real_SC_annotations_weights_cache_or_predictions_read'] is False
    bindings={};duties={}
    def add(name,duty,expected=None):
        p=Path(name);assert not p.is_absolute() and '..' not in p.parts and str(p)==name
        got=hash_file(root/name)
        if expected is not None:assert got=={k:expected[k] for k in ('sha256','size_bytes')}
        if name in bindings:assert bindings[name]==got
        bindings[name]=got;duties[name]=duty
        return got
    excluded=[]
    for item in originals['files']:
        if Path(item['path']).name=='.DS_Store':excluded.append(item);continue
        add(fixture+item['path'],'owned_INVENTED_fixture_or_output',item)
    assert len(excluded)==19 and len(bindings)==43739
    for n in (manifest_name,verification_name):add(n,'retained_actual_artificial_execution_or_verification')
    failed_check='research/round4_sccomics_supervised_stages_v2_sourceonly_inputs/POSTEXEC_INVENTED_v2/actual_verification_result.json'
    add(failed_check,'retained_failed_verification_Finder_metadata_change')
    for n,b in summary['source_bindings'].items():
        assert n.startswith('src/sccomics_') and n.endswith('.py');add(n,'stable_scientific_SOURCE',b)
    other_sources=(
        'src/check_sccomics_supervised_stages_v2_synthetic.py',
        'src/check_sccomics_runtime_resource_v2_synthetic.py',
        'src/prepare_sccomics_supervised_runtime_v2_source_only.py',
        'src/verify_sccomics_supervised_v2_artificial_bindings.py',
        'src/verify_sccomics_supervised_v2_artificial_bindings_v2.py',
        'src/build_sccomics_supervised_v2_source_handoff.py',
        'src/sccomics_supervised_stages.py',
        'src/check_sccomics_supervised_stages_synthetic.py')
    for n in other_sources:add(n,'SOURCE_helper_or_immutable_v1_history')
    assert bindings[other_sources[0]]['sha256']==summary['helper_sha256']
    controls=(
        'research/round4_sccomics_experimental_configuration_before_fit',
        'research/round4_sccomics_execution_details_before_fit',
        'research/round4_sccomics_pre_fit_ner_denominator_amendment',
        'research/round4_sccomics_supervised_stages_v2_before_fit_source_contract',
        'research/round4_sccomics_supervised_stages_v2_independent_findings_before_fit_addendum',
        'research/round4_sccomics_supervised_stages_v2_late_findings_statistics_addendum',
        'research/round4_sccomics_supervised_stages_v2_arithmetic_replacement_before_artificial',
        'research/round4_sccomics_supervised_stages_v2_runtime_and_fresh_dev_before_fit_addendum',
        'research/round4_sccomics_supervised_stages_v2_runtime_import_compatibility_prospective_amendment',
        'research/round4_sccomics_supervised_stages_sourceonly_implementation',
        'research/round4_sccomics_supervised_stages_independent_source_review',
        'research/round4_sccomics_test_analysis_independent_source_review',
        'research/round4_sccomics_independent_statistics_source_review')
    for n in controls:
        for ext in ('.md','.json'):add(n+ext,'SOURCE_protocol_or_retained_review')
    add('research/round4_sccomics_supervised_stages_v2_sourceonly_implementation.md','this_implementation_report')
    previous='research/round4_sccomics_supervised_stages_v2_synthetic_inputs/INVENTED_v1/'
    for name in ('actual_synthetic_results.json','output_manifest.json'):add(previous+name,'retained_first_artificial_run_manifest_or_summary_not_fresh_allfile_verification')
    # Only small owned failure/result/source snapshots; never follow declared
    # installed-framework filenames in metadata to read framework/model files.
    folders=(
        'research/round4_sccomics_supervised_stages_v2_runtime_compatibility_inputs/INVENTED_v1',
        'research/round4_sccomics_supervised_stages_v2_runtime_compatibility_inputs/INVENTED_v2',
        'research/round4_sccomics_supervised_stages_v2_runtime_compatibility_inputs/INVENTED_v3',
        'research/round4_sccomics_supervised_stages_v2_runtime_population_metadata_inputs/INVENTED_METADATA_CHECK_v1',
        'research/round4_sccomics_supervised_stages_v2_runtime_population_metadata_inputs/INSTALLED_METADATA_CHECK_v2',
        'research/round4_sccomics_supervised_stages_v2_runtime_population_metadata_inputs/INSTALLED_METADATA_CHECK_v3',
        'research/round4_sccomics_supervised_stages_v2_sourceonly_inputs/POSTEXEC_INVENTED_v2',
        'research/round4_sccomics_supervised_stages_v2_sourceonly_inputs/POSTEXEC_INVENTED_v2_NONFINDER')
    for directory in folders:
        for p in sorted((root/directory).rglob('*')):
            if p.is_file() and p.name!='.DS_Store':add(str(p.relative_to(root)),'owned_SOURCE_metadata_failure_or_verification_history')
    for n,b in summary['source_bindings'].items():
        dest=out/'before'/n;dest.parent.mkdir(parents=True,exist_ok=True);raw=(root/n).read_bytes()
        assert len(raw)==b['size_bytes'] and hashlib.sha256(raw).hexdigest()==b['sha256'];dest.write_bytes(raw)
        add(str(dest.relative_to(root)),'captured_stable_scientific_SOURCE_before')
    for n in other_sources[:6]:
        dest=out/'before'/n;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes((root/n).read_bytes());add(str(dest.relative_to(root)),'captured_helper_before')
    add(str((out/'builder.before.py.txt').relative_to(root)),'builder_before')
    assert own==Path(__file__).read_bytes()
    flat={'identity':'SOURCE_ONLY_IMPLEMENTATION_AND_OWNED_ARTIFICIAL_ARTIFACT_BINDINGS_NOT_FIT_AUTHORITY',
          'bindings':[{'path':n,**b,'duty':duties[n]} for n,b in sorted(bindings.items())],
          'all_flat_bindings_actually_reopened_sha256_size_checked':True,
          'original_manifest_Finder_bindings_excluded_from_flat_population':excluded,
          'Finder_binding_current_identities_record':verification_name,
          'actual_SC_model_cache_cold_weight_prediction_or_annotation_read':False,
          'actual_installed_runtime_population_bytes_hashed_by_this_builder':False,
          'complete_scientific_freeze_final_gate_created':False}
    manifest_path=out/'implementation_delivery_manifest.json';manifest_path.write_text(json.dumps(flat,sort_keys=True,indent=2)+'\n')
    summary_out={'identity':'SOURCEONLY_IMPLEMENTATION_HANDOFF_NOT_INDEPENDENT_REVIEW','status':'stable_source_ready_for_different_agent_SOURCE_review',
        'stable_source_bindings':summary['source_bindings'],
        'main_fixture_helper':{'path':other_sources[0],**bindings[other_sources[0]]},
        'implementation_report_md':{'path':'research/round4_sccomics_supervised_stages_v2_sourceonly_implementation.md',**bindings['research/round4_sccomics_supervised_stages_v2_sourceonly_implementation.md']},
        'delivery_manifest':{'path':str(manifest_path.relative_to(root)),**hash_file(manifest_path),'flat_binding_count':len(bindings)},
        'second_artificial_execution':{'summary':{'path':summary_name,**bindings[summary_name]},'total':92,'passed':92,'failed':0,'directly_captured_process_exit_code':None,'original_exit_capture_limitation':'Tool wrapper discarded asynchronous session ID; do not substitute helper return rule for an observed exit.'},
        'postexecution_verification_history':{'original_verifier_session':26853,'original_verifier_exit_code':1,'original_verifier_result':failed_check,
            'failed_original_bindings':['.DS_Store','INVENTED_zeroNER_root/.DS_Store'],
            'nonFinder_verifier_session':24093,'nonFinder_verifier_exit_code':0,'nonFinder_result':verification_name,
            'original_total_bindings':43758,'explicitly_excluded_Finder_bindings':19,'nonFinder_bindings_verified':43739,'nonFinder_mismatches':0,
            'all_original_bindings_matched_claim':False},
        'actual_runtime_byte_receipt_preparer':{'path':other_sources[2],**bindings[other_sources[2]],'executed':False,'different_SOURCE_review_passed':False},
        'actual_six_library_import_compatibility':{'status':'prospective dedicated source-only helper/authority and different review pending','executed':False,'known_real_namespace_failure':False},
        'different_agent_wrapper_runtime_SOURCE_review_passed':False,'complete_actual_scientific_freeze_final_gate_or_fitting_permission':False,
        'actual_SC_annotation_model_cache_weight_checkpoint_prediction_semantic_access':False,'actual_fit_inference_API_or_new_performance_statistic':False,
        'original_v1_and_failed_histories_preserved':True,'same_scientific_configuration':True,
        'count_replay_same_statistics_module_is_not_independent_algorithm':True,'b83_arithmetic_replay_shares_declared_streams':True,
        'no_all_OS_GPU_bytes_all_loaded_definitions_reexecution_or_physical_independence_claim':True}
    p=root/'research/round4_sccomics_supervised_stages_v2_sourceonly_implementation.json';assert not p.exists();p.write_text(json.dumps(summary_out,sort_keys=True,indent=2)+'\n')
    result={'identity':'SOURCE_HANDOFF_BUILDER_ACTUAL_EXECUTION_ONLY','flat_bindings':len(bindings),'delivery_manifest':hash_file(manifest_path),'implementation_json':hash_file(p),'passed':True,'actual_fitting_permission':False}
    (out/'builder_actual_result.json').write_text(json.dumps(result,sort_keys=True,indent=2)+'\n');print(json.dumps(result,sort_keys=True))
    return 0

if __name__=='__main__':raise SystemExit(main())
