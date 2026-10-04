"""Post-execution byte verification; excludes explicit Finder metadata only.

Original v1 verification retained two real Finder .DS_Store mismatches.
This new helper reports every original Finder binding and its current identity;
it does not call those original bindings matched.

Does not rerun fitting/scoring. Reads only manifest-enumerated owned invented
files and the source bindings declared by that fixture summary.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import stat


def digest(path):
    assert not any(p.is_symlink() for p in (path, *path.parents))
    before=path.stat(follow_symlinks=False)
    assert stat.S_ISREG(before.st_mode)
    fd=os.open(path,os.O_RDONLY|os.O_NONBLOCK|os.O_NOFOLLOW)
    try:
        opened=os.fstat(fd)
        assert (before.st_dev,before.st_ino)==(opened.st_dev,opened.st_ino)
        h=hashlib.sha256();size=0
        while chunk:=os.read(fd,1024*1024):h.update(chunk);size+=len(chunk)
        after=os.fstat(fd)
        assert (opened.st_dev,opened.st_ino,opened.st_size,opened.st_mtime_ns,opened.st_ctime_ns)==(after.st_dev,after.st_ino,after.st_size,after.st_mtime_ns,after.st_ctime_ns)
        assert size==opened.st_size
        return {'sha256':h.hexdigest(),'size_bytes':size}
    finally:os.close(fd)


def main():
    a=argparse.ArgumentParser(description=__doc__);a.add_argument('--fixture-dir',required=True);a.add_argument('--output-dir',required=True);args=a.parse_args()
    root=Path(__file__).absolute().parents[1]
    fixture=root/args.fixture_dir;out=root/args.output_dir
    assert fixture.parent==root/'research/round4_sccomics_supervised_stages_v2_synthetic_inputs' and fixture.name=='INVENTED_v2'
    assert out.parent==root/'research/round4_sccomics_supervised_stages_v2_sourceonly_inputs' and out.name=='POSTEXEC_INVENTED_v2_NONFINDER' and not out.exists()
    assert not any(p.is_symlink() for p in (fixture,*fixture.parents,out.parent,*out.parent.parents))
    out.mkdir(parents=True)
    (out/'helper.before.py.txt').write_bytes(Path(__file__).read_bytes())
    manifest_path=fixture/'output_manifest.json';summary_path=fixture/'actual_synthetic_results.json'
    before=digest(manifest_path);manifest=json.loads(manifest_path.read_bytes());summary=json.loads(summary_path.read_bytes())
    assert manifest['identity']=='INVENTED_FIXTURES_AND_ACTUAL_SYNTHETIC_OUTPUTS_ONLY'
    assert summary['identity']=='INVENTED_IMPLEMENTER_SELF_CHECK_ONLY_NOT_INDEPENDENT_REVIEW'
    assert summary['real_SC_annotations_weights_cache_or_predictions_read'] is False
    seen=set();failures=[];excluded=[];checked=0
    for item in manifest['files']:
        name=item['path'];p=Path(name)
        assert not p.is_absolute() and '..' not in p.parts and str(p)==name and name not in seen
        seen.add(name)
        if p.name=='.DS_Store':
            try:current=digest(fixture/name)
            except BaseException as e:current={'error_type':type(e).__name__,'error':str(e)}
            expected={k:item[k] for k in ('sha256','size_bytes')}
            excluded.append({'path':name,'scope':'Finder directory-layout metadata; not artificial source, scientific output, code, or gate',
                             'original_binding':expected,'current_binding':current,'original_binding_currently_matches':current==expected})
            continue
        checked+=1
        try:
            got=digest(fixture/name)
            assert got=={k:item[k] for k in ('sha256','size_bytes')}
        except BaseException as e:failures.append({'path':name,'type':type(e).__name__,'error':str(e)})
    assert 'actual_synthetic_results.json' in seen
    code=[]
    for name,binding in summary['source_bindings'].items():
        assert name.startswith('src/sccomics_') and name.endswith('.py') and '..' not in Path(name).parts
        got=digest(root/name);wanted={k:binding[k] for k in ('sha256','size_bytes')}
        code.append({'path':name,**got,'matched':got==wanted})
        if got!=wanted:failures.append({'path':name,'error':'current source differs from captured run'})
    helper=digest(root/'src/check_sccomics_supervised_stages_v2_synthetic.py')
    assert helper['sha256']==summary['helper_sha256']
    assert before==digest(manifest_path)
    result={'identity':'POST_EXECUTION_INVENTED_BYTE_BINDING_VERIFICATION_ONLY',
            'original_artifact_bindings':len(seen),'checked_artifact_bindings':checked,
            'all_nonFinder_artifact_hashes_and_sizes_matched':not failures,
            'all_original_artifact_bindings_matched':not failures and all(e['original_binding_currently_matches'] for e in excluded),
            'explicitly_excluded_Finder_metadata_bindings':excluded,
            'fixture_output_manifest':{'path':str(manifest_path.relative_to(root)),**before},
            'fixture_summary':{'path':str(summary_path.relative_to(root)),**digest(summary_path)},
            'current_scientific_source_bindings':code,'current_fixture_helper':helper,
            'artificial_checks':{'total':summary['total'],'passed':summary['passed'],'failed':summary['failed']},
            'original_fixture_process_exit_code':None,
            'original_exit_capture_limitation':'Original tool wrapper returned only output and discarded session ID; generated result reports 92/92, but exit code was not independently captured.',
            'this_is_not_independent_SOURCE_review':True,'actual_SC_model_cache_weight_annotation_or_prediction_read':False,
            'actual_fitting_or_new_statistical_evaluation':False,'failures':failures,'passed':not failures}
    raw=(json.dumps(result,sort_keys=True,ensure_ascii=False,indent=2)+'\n').encode()
    (out/'actual_verification_result.json').write_bytes(raw)
    print(json.dumps({k:result[k] for k in ('passed','checked_artifact_bindings','artificial_checks','failures')},sort_keys=True))
    return 0 if result['passed'] else 1

if __name__=='__main__':raise SystemExit(main())
