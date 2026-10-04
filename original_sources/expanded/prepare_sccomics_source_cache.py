"""Cold, source-text-only first-eleven-block cache for fixed native fold1.

Actual execution requires an independently reviewed, separately authorized cache
protocol. No annotation file, fitted checkpoint, API or credential loader.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gc
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import types

import torch
from transformers import AutoTokenizer, BertModel

SELF = 'src/prepare_sccomics_source_cache.py'
PRIMITIVE = 'src/sccomics_source_models.py'
PRIMITIVE_SHA = '8ff50b5bb88c3fa6296b76ff9f5dab63a43177e1c0c54178aea01dafc59900f0'
REVIEW = 'research/round4_sccomics_source_cache_independent_source_review.json'
GATE = 'research/round4_sccomics_source_cache_authorized_gate.json'
ACQUISITION = 'data/sccomics_round4/source_v3/source_acquisition_manifest.json'
TRAIN_AUDIT = 'research/round4_sccomics_train_schema_actual/audit.json'
TRAIN_CROSSCHECK = 'research/round4_sccomics_train_schema_actual_root_crosscheck.json'
MODEL_ROOT = 'data/local_baselines/matscibert'
MODEL_NAMES = ('README.md', 'config.json', 'model.safetensors', 'source_manifest.json',
               'special_tokens_map.json', 'tokenizer.json', 'tokenizer_config.json', 'vocab.txt')
SOURCE_REVISION = 'ced9d8f5f208712c4a90f98a246fe32155b29995'
ACQUISITION_SHA = 'a38f1637af74b15246b6b028d3d58b5836094e905d25845e5922668031fee1c7'
TRAIN_AUDIT_SHA = 'febaa67459457d97ea7eb6b617f4422f872693677bcb4218f984c7f0430b2d1f'
TRAIN_CROSSCHECK_SHA = '4ef4bfb4b8483f65612753b34d1a7515fab12509bf818ac679a12c1f039c29e5'
SPLITS = {'train': list(range(201, 1001)), 'dev': list(range(101, 201)), 'test': list(range(1, 101))}
METADATA = {SELF, PRIMITIVE, REVIEW, ACQUISITION, TRAIN_AUDIT, TRAIN_CROSSCHECK,
            'research/round4_sccomics_source_model_independent_review.json'}


def require(value, message):
    if not value:
        raise ValueError(message)


def sha_file(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def sha_bytes(raw):
    return hashlib.sha256(raw).hexdigest()


def unique_json(raw):
    def pairs(values):
        result = {}
        for k, v in values:
            require(k not in result, 'duplicate control JSON key')
            result[k] = v
        return result
    return json.loads(raw, object_pairs_hook=pairs, parse_constant=lambda x: (_ for _ in ()).throw(ValueError('nonfinite control JSON')))


def safe_path(root, name):
    require(root.is_absolute(), 'explicit root must be absolute')
    ancestor = Path(root.anchor)
    for part in root.parts[1:]:
        ancestor = ancestor / part
        require(not ancestor.is_symlink(), 'symlink root ancestor')
    p = Path(name)
    if p.is_absolute():
        require(p.is_relative_to(root), 'outside explicit root')
        p = p.relative_to(root)
    require(p.parts and all(x not in {'.', '..', ''} for x in p.parts), 'invalid lexical relative path')
    current = root
    for part in p.parts:
        current = current / part
        require(not current.is_symlink(), 'symlink input or output ancestor')
    return current


def serialized(value):
    return (json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n').encode('utf-8')


def write_new(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = serialized(value)
    with path.open('xb') as stream:
        stream.write(raw)


def execute(root, output_dir):
    root = Path(root).absolute()
    require(not root.is_symlink(), 'symlink explicit root')
    gate_path = safe_path(root, GATE)
    gate_raw = gate_path.read_bytes()
    gate = unique_json(gate_raw)
    require(type(gate.get('synthetic_fixture')) is bool, 'explicit fixture/actual scope required')
    invented = gate['synthetic_fixture']
    require(not invented or 'INVENTED' in str(root), 'fixture cannot use an actual-root path')
    require(gate.get('status') == ('INVENTED_authorized_source_text_cache_only' if invented else
                                  'authorized_source_text_cache_only') and gate.get('actual_execution_authorized') is True,
            'source cache is not authorized')
    require(gate.get('annotation_files_allowed') is False and gate.get('fitted_checkpoints_allowed') is False and
            gate.get('paid_API_or_credentials_allowed') is False and gate.get('model_fitting_allowed') is False,
            'wrong cache-stage permissions')
    require(gate.get('native_split_ids') == SPLITS, 'fixed native fold differs')
    bound = gate.get('file_sha256', {})
    expected = METADATA | {f'{MODEL_ROOT}/{name}' for name in MODEL_NAMES}
    require(set(bound) == expected, 'closed source cache input duties required')
    captures, inputs = {}, {}
    for name, digest in bound.items():
        path = safe_path(root, name)
        require(sha_file(path) == digest, f'bound cache input SHA differs: {name}')
        inputs[name] = {'sha256': digest, 'bytes': path.stat().st_size}
        if name in METADATA or name == f'{MODEL_ROOT}/source_manifest.json':
            captures[name] = path.read_bytes()
            require(sha_bytes(captures[name]) == digest, 'captured control source differs')
    require(bound[PRIMITIVE] == PRIMITIVE_SHA, 'source primitives changed')
    own_sha = bound[SELF]
    require(Path(__file__).absolute() == safe_path(root, SELF), 'executing source outside explicit root')
    independent = unique_json(captures[REVIEW])
    require(independent.get('status') == ('INVENTED_source_review_passed' if invented else 'source_only_review_passed') and
            independent.get('independent_from_implementer') is True and
            independent.get('reviewed_source_sha256') == own_sha, 'cache source independently reviewed version required')
    primitive_review = unique_json(captures['research/round4_sccomics_source_model_independent_review.json'])
    require(primitive_review.get('status') == 'source_only_review_passed', 'source model review incomplete')
    primitive_binding = primitive_review.get('file_sha256', {}).get(PRIMITIVE)
    require(primitive_binding == PRIMITIVE_SHA, 'source model review binding differs')
    train = unique_json(captures[TRAIN_AUDIT])
    require(train.get('status') == 'train_schema_audit_completed_only' and train.get('completed') is True and
            train.get('annotation_ids_opened') == SPLITS['train'] and not train.get('schema_failures') and
            train.get('dev_or_test_annotations_opened') is False, 'complete native TRAIN schema prerequisite missing')
    train_cross = unique_json(captures[TRAIN_CROSSCHECK])
    require(train_cross.get('audit_sha256') == bound[TRAIN_AUDIT] and
            train_cross.get('all_native_record_raw_bytes_and_fields_checked_against_saved_schema') is True,
            'independent completed TRAIN schema binding differs')
    acquisition = unique_json(captures[ACQUISITION])
    if invented:
        require(bound[ACQUISITION] != ACQUISITION_SHA and acquisition.get('synthetic_fixture') is True and
                train.get('synthetic_fixture') is True, 'fictional completion inputs must be explicitly fictional')
    else:
        require(bound[ACQUISITION] == ACQUISITION_SHA and bound[TRAIN_AUDIT] == TRAIN_AUDIT_SHA and
                bound[TRAIN_CROSSCHECK] == TRAIN_CROSSCHECK_SHA and
                acquisition.get('synthetic_fixture') is not True and train.get('synthetic_fixture') is False,
                'actual completed native provenance identities differ')
    require(acquisition.get('split_ids') == SPLITS and acquisition.get('native_source_abstract_records') == 1000,
            'source inventory differs')
    members = acquisition['official_archives']['SC-CoMIcs-Abstract1000_CC_BY-NC-3.0.zip']['members']
    require(set(members) == {f'{i:04}.txt' for i in range(1, 1001)}, 'source text member population differs')
    model_manifest = unique_json(captures[f'{MODEL_ROOT}/source_manifest.json'])
    require(model_manifest.get('revision') == SOURCE_REVISION and model_manifest.get('model') == 'm3rg-iitd/matscibert' and
            model_manifest.get('base_encoder_not_task_finetuned_checkpoint') is True,
            'cold model revision/identity differs')
    original = model_manifest.get('files', [])
    require(len(original) == 7 and {b['name'] for b in original} == set(MODEL_NAMES) - {'source_manifest.json'},
            'original cold model file inventory differs')
    for b in original:
        key = f'{MODEL_ROOT}/{b["name"]}'
        require(b['sha256'] == bound[key] and b['bytes'] == inputs[key]['bytes'],
                'cold model bytes differ from original download receipt')
    # Execute the same captured, source-reviewed primitive bytes. No prior-task
    # adapter/model module or uncontrolled .pyc is imported.
    module_name = '_sccomics_verified_source_cache_primitives'
    module = types.ModuleType(module_name)
    sys.modules[module_name] = module
    exec(compile(captures[PRIMITIVE], PRIMITIVE, 'exec'), module.__dict__)
    output = safe_path(root, output_dir)
    require(not output.exists(), 'existing cache evidence cannot be overwritten or resumed implicitly')
    output.mkdir(parents=True)
    state = {'status': 'running_cold_source_text_cache_only', 'pid': os.getpid(),
             'synthetic_fixture': invented,
             'started_at_utc': datetime.now(timezone.utc).isoformat(), 'gate_sha256': sha_bytes(gate_raw),
             'script_sha256': own_sha, 'native_split_ids': SPLITS, 'cache_bindings': [],
             'native_annotation_files_opened': False, 'fitted_checkpoints_loaded': False,
             'annotation_content_used_in_source_features': False, 'model_fitting_performed': False}
    write_new(output / 'started.json', state)
    torch.set_num_threads(2)
    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    state['device'] = device
    text_bindings = []
    try:
        # Cold model files are copied and checked before the local HF loader can
        # read them. It never reads an old fitted checkpoint or live original
        # weights after this snapshot; no remote-code/download permission.
        with tempfile.TemporaryDirectory(prefix='sccomics_cold_cache_model_') as temporary:
            snapshot = Path(temporary)
            for name in MODEL_NAMES:
                source = safe_path(root, f'{MODEL_ROOT}/{name}')
                destination = snapshot / name
                shutil.copyfile(source, destination)
                require(sha_file(destination) == bound[f'{MODEL_ROOT}/{name}'], 'cold model snapshot differs')
            tokenizer = AutoTokenizer.from_pretrained(str(snapshot), local_files_only=True, trust_remote_code=False)
            base = BertModel.from_pretrained(str(snapshot), local_files_only=True, trust_remote_code=False,
                                            use_safetensors=True).to(device).eval()
            require(len(base.encoder.layer) == 12, 'cold encoder block count differs')
            with torch.no_grad():
                for split, identifiers in SPLITS.items():
                    directory = output / split
                    directory.mkdir()
                    for i in identifiers:
                        text_path = safe_path(root, f'data/sccomics_round4/source_v3/raw_text/{i:04}.txt')
                        binding = members[text_path.name]
                        require(safe_path(root, binding['canonical_path']) == text_path and
                                binding.get('category') == 'raw_source_document', 'source path/category differs')
                        raw = text_path.read_bytes()
                        require(sha_bytes(raw) == binding['sha256'] and len(raw) == binding['bytes'], 'source text SHA/size differs')
                        text_bindings.append({'path': str(text_path.relative_to(root)),
                                              'sha256': sha_bytes(raw), 'bytes': len(raw)})
                        text = raw.decode('utf-8', 'strict')
                        encoded = module.source_encoding(text, tokenizer)
                        chunks = []
                        for c in encoded['chunks']:
                            ids = torch.tensor([c['ids']], device=device, dtype=torch.long)
                            mask = base.get_extended_attention_mask(torch.ones_like(ids), ids.shape)
                            hidden = base.embeddings(input_ids=ids, token_type_ids=torch.zeros_like(ids))
                            for layer in base.encoder.layer[:11]:
                                hidden = layer(hidden, attention_mask=mask)[0]
                            require(hidden.shape == (1, len(c['ids']), 768) and
                                    mask.shape == (1, 1, 1, len(c['ids'])) and
                                    bool(torch.isfinite(hidden).all()) and bool(torch.isfinite(mask).all()),
                                    'invalid/nonfinite source representations or mask')
                            stored_hidden, stored_mask = hidden.cpu().float(), mask.cpu().float()
                            require(bool(torch.isfinite(stored_hidden).all()) and bool(torch.isfinite(stored_mask).all()),
                                    'nonfinite final float32 source representations or mask')
                            chunks.append({'start': c['start'], 'end': c['end'],
                                           'hidden': stored_hidden, 'mask': stored_mask})
                        saved = {'source_id': i, 'split': split, 'source_sha256': sha_bytes(raw),
                                 'gate_sha256': sha_bytes(gate_raw), 'cold_revision': SOURCE_REVISION,
                                 'wordpieces': encoded['wordpieces'], 'offsets': encoded['offsets'], 'chunks': chunks,
                                 'source_only': True, 'annotation_content_used': False,
                                 'candidate_span_count': len(module.span_candidates(encoded['offsets']))}
                        path = directory / f'{i:04}.pt'
                        require(not path.exists(), 'cache collision')
                        torch.save(saved, path)
                        state['cache_bindings'].append({'native_id': i, 'split': split,
                            'path': str(path.relative_to(root)), 'sha256': sha_file(path), 'bytes': path.stat().st_size,
                            'wordpieces': saved['wordpieces'], 'chunks': len(chunks),
                            'candidate_span_count': saved['candidate_span_count'], 'source_sha256': sha_bytes(raw)})
                        event = {'phase': 'source_only_cache_record_completed', 'native_id': i, 'split': split,
                                 'completed_records': len(state['cache_bindings']), 'wordpieces': saved['wordpieces'],
                                 'at_utc': datetime.now(timezone.utc).isoformat()}
                        with (output / 'progress.jsonl').open('a') as stream:
                            stream.write(json.dumps(event) + '\n')
                        if len(state['cache_bindings']) % 50 == 0:
                            print(json.dumps(event), flush=True)
            del base, tokenizer
            gc.collect()
            if device == 'mps':
                torch.mps.empty_cache()
            for name in MODEL_NAMES:
                require(sha_file(snapshot / name) == bound[f'{MODEL_ROOT}/{name}'], 'used cold model snapshot changed')
        # Every source/control/model/cache is verified after all computation.
        # This is a finite byte-coherence check, not a continuous OS transaction.
        require(sha_file(gate_path) == sha_bytes(gate_raw), 'cache gate changed during execution')
        for name, b in inputs.items():
            path = safe_path(root, name)
            require(sha_file(path) == b['sha256'] and path.stat().st_size == b['bytes'], 'cache control/model input changed')
        for b in text_bindings + state['cache_bindings']:
            path = safe_path(root, b['path'])
            require(sha_file(path) == b['sha256'] and path.stat().st_size == b['bytes'], 'source/cache changed before completion')
        require(len(state['cache_bindings']) == 1000 and
                all([b['native_id'] for b in state['cache_bindings'] if b['split'] == s] == ids for s, ids in SPLITS.items()),
                'incomplete source cache population')
        for split, identifiers in SPLITS.items():
            directory = safe_path(root, str((output / split).relative_to(root)))
            require({p.name for p in directory.iterdir()} == {f'{i:04}.pt' for i in identifiers},
                    'extra or missing source cache member')
        require({p.name for p in output.iterdir()} == set(SPLITS) | {'started.json', 'progress.jsonl'},
                'extra or missing cache output member')
        state.update(status='INVENTED_completed_cold_source_text_cache_only' if invented else
                     'completed_cold_source_text_cache_only', completed_at_utc=datetime.now(timezone.utc).isoformat(),
                     input_bindings=inputs, source_text_bindings=text_bindings,
                     all_captured_source_model_and_cache_hashes_size_final_checked=True,
                     complete_native_records=1000, paid_API_or_credentials_used=False)
        manifest_path = output / 'completed_manifest.json'
        expected_manifest = serialized(state)
        write_new(manifest_path, state)
        actual_manifest_path = safe_path(root, str(manifest_path.relative_to(root)))
        require(actual_manifest_path.read_bytes() == expected_manifest, 'actual completed cache manifest bytes differ')
        print(json.dumps({'status': state['status'], 'records': 1000, 'device': device}), flush=True)
        return state
    except BaseException as exc:
        candidate = output / 'completed_manifest.json'
        if candidate.exists() or candidate.is_symlink():
            candidate.rename(output / 'completed_manifest.json.failed_pre_certificate')
        state.update(status='stopped_explicit_source_cache_failure', failure_type=type(exc).__name__,
                     failure=str(exc), stopped_at_utc=datetime.now(timezone.utc).isoformat())
        write_new(output / 'failed_attempt.json', state)
        raise


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root', required=True)
    ap.add_argument('--output-dir', required=True)
    args = ap.parse_args()
    execute(args.root, args.output_dir)
