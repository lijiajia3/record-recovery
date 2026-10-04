"""Finite runtime metadata and reviewed INVENTED-only CUDA operator check.

No corpus, annotations, pretrained weights, network, installation or fitting.
Explicit execution flag required. Preserve stdout, stderr and actual returncode.
"""
import argparse
import hashlib
import importlib.metadata
import json
import pathlib
import platform
import subprocess
import sys
import zipfile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True)
    parser.add_argument('--execute-invented-only', action='store_true')
    args = parser.parse_args()
    if not args.execute_invented_only:
        raise SystemExit('Explicit invented-only authorization required')
    root = pathlib.Path(args.root).resolve(strict=True)
    output = root / 'research/ACTUAL_CUDA_OPERATOR_20261003'
    output.mkdir(exist_ok=False)
    receipt = {'scope': 'invented_operator_compatibility_not_SC_training',
               'corpus_or_ANN_or_pretrained_weights_or_API': False,
               'bootstrap_sha256': hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest()}
    returncode = 1
    try:
        manifest = json.loads((root / 'source_only_deployment_manifest.json').read_text())
        pins = manifest['source_members']
        checked = {}
        for name, binding in pins.items():
            path = root / name
            if path.is_symlink() or not path.is_file():
                raise ValueError('Non-regular source input: ' + name)
            raw = path.read_bytes()
            actual = {'sha256': hashlib.sha256(raw).hexdigest(), 'size_bytes': len(raw)}
            if actual != {k: binding[k] for k in actual}:
                raise ValueError('Source binding mismatch: ' + name)
            checked[name] = actual
        receipt['verified_original_members'] = checked
        versions = {'python': platform.python_version()}
        for name in ('torch', 'numpy', 'transformers', 'tokenizers', 'safetensors', 'networkx'):
            versions[name] = importlib.metadata.version(name)
        receipt['actual_declared_runtime_versions_before_operator'] = versions
        helper = 'src/check_sccomics_cuda_operators_v3.py'
        runner = 'src/sccomics_cuda_cloud_v3.py'
        review = 'research/round4_sccomics_cloud_cuda_independent_source_review.json'
        science = {name: binding for name, binding in checked.items()
                   if name.startswith('src/') and name not in (helper, runner)}
        if len(science) != 10:
            raise ValueError('Exactly ten scientific modules required')
        config = 'data/local_baselines/matscibert/config.json'
        spec = {'version': 1,
                'purpose': 'CUDA_ACTUAL_INVENTED_OPERATOR_COMPATIBILITY_NO_CORPUS_OR_PRETRAINED_WEIGHT',
                'root_actual_invented_check_authorized': True,
                'helper': checked[helper], 'runner': checked[runner], 'science': science,
                'config': {'path': config, **checked[config]},
                'runtime_versions': versions,
                'cuda_policy': {'device': 'cuda:0', 'dtype': 'float32', 'deterministic': True,
                                'cublas_workspace': ':4096:8', 'tf32': False, 'autocast': False,
                                'attention_policy': 'library_default_recorded_before_fit'},
                'independent_source_review': {'path': review, **checked[review]}}
        spec_path = root / 'research/round4_sccomics_cuda_operator_actual_spec_20261003.json'
        with spec_path.open('x') as handle:
            json.dump(spec, handle, indent=2, sort_keys=True)
            handle.write('\n')
        specification_sha = hashlib.sha256(spec_path.read_bytes()).hexdigest()
        receipt['specification_sha256'] = specification_sha
        argv = [sys.executable, helper, '--root', str(root), '--specification',
                str(spec_path.relative_to(root)), '--specification-sha256', specification_sha,
                '--output', 'research/ACTUAL_CUDA_OPERATOR_20261003/operator_result.json',
                '--execute-invented-operator-compatibility']
        receipt['argv'] = argv
        with (output / 'stdout.txt').open('xb') as out, (output / 'stderr.txt').open('xb') as err:
            process = subprocess.run(argv, cwd=root, stdout=out, stderr=err)
        returncode = process.returncode
        receipt['actual_operator_subprocess_returncode'] = returncode
        # Verify original source inputs again, with no source editing on either outcome.
        for name, binding in checked.items():
            raw = (root / name).read_bytes()
            if {'sha256': hashlib.sha256(raw).hexdigest(), 'size_bytes': len(raw)} != binding:
                raise ValueError('Source changed after operator invocation: ' + name)
        receipt['original_sources_still_identical'] = True
    except Exception as error:
        receipt['preparation_or_invocation_error'] = {'type': type(error).__name__, 'error': str(error)}
        returncode = 1
    receipt['bootstrap_returncode'] = returncode
    (output / 'actual_execution_receipt.json').write_text(json.dumps(receipt, indent=2, sort_keys=True)+'\n')
    bundle = root / 'ACTUAL_CUDA_OPERATOR_20261003_bundle.zip'
    with zipfile.ZipFile(bundle, 'x', compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(output.glob('*')):
            archive.write(path, str(path.relative_to(root)))
        spec_path = root / 'research/round4_sccomics_cuda_operator_actual_spec_20261003.json'
        if spec_path.is_file():
            archive.write(spec_path, str(spec_path.relative_to(root)))
    raw = bundle.read_bytes()
    print(json.dumps({'actual_bootstrap_returncode': returncode, 'bundle': str(bundle),
                      'size_bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest()}), flush=True)
    return returncode


if __name__ == '__main__':
    raise SystemExit(main())
