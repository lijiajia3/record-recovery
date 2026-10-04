"""Practical local DEV bridges and complete returned TEST intake, before fit.

Import is stdlib/SOURCE only. Actual commands require a separately reviewed,
root-created authority. No local neural fitting or provider/cloud requests.
Cloud receipts certify supplied files, not independent physical truth.
"""
from __future__ import annotations
import argparse, contextlib, copy, hashlib, importlib.util, io, json, math
import os, pathlib, stat, sys, tempfile, types

SELF = 'src/sccomics_cloud_local_bridge_v2.py'
CLOUD = 'src/sccomics_cuda_cloud_v4.py'
CLOUD_SHA = 'a99bb8a4d03da964e6b7705e807fd3d54659568dd885720da6263b686d635fcc'
OPERATOR = 'src/check_sccomics_cuda_operators_v4.py'
OPERATOR_SHA = 'e62e48cc5b6154f79bd6b8666a8659878a1d5b417499af8f7bd6b6416c9dd1ad'
CLUSTERS = 'research/round4_sccomics_text_provenance_actual/test_source_text_clusters.json'
CLUSTERS_SHA = 'd84888434ecfcedcb59cb2a7060e63c21a68f324a55e37b13570280febec0e6d'
HISTORY = 'results/local_baseline/joint_supervised_global_holm4.json'
HISTORY_SHA = 'c89995e473f182f7d767a0c526088541242944d2097b12807183ab967ca5bfa0'

def cloud_source(root):
    """Execute only the previously reviewed captured stdlib cloud definition."""
    p = pathlib.Path(root) / CLOUD
    if any(q.is_symlink() for q in (p, *p.parents)):
        raise ValueError('cloud source alias')
    raw = p.read_bytes()
    if hashlib.sha256(raw).hexdigest() != CLOUD_SHA:
        raise ValueError('immutable cloud SOURCE changed')
    m = types.ModuleType('_sccomics_captured_cloud_v4')
    m.__file__ = str(p)
    exec(compile(raw, str(p), 'exec'), m.__dict__)
    return m

def descriptor(c, files, name):
    c.relative(name)
    with files.opened(name) as fd:
        import os
        h, size = hashlib.sha256(), 0
        while chunk := os.read(fd, 1048576):
            h.update(chunk); size += len(chunk)
    return {'path': name, 'sha256': h.hexdigest(), 'size_bytes': size}

def content_binding(d):
    return {k: d[k] for k in ('sha256', 'size_bytes')}

def captured(c, files, d):
    c.closed(d, ('path', 'sha256', 'size_bytes'))
    c.relative(d['path'])
    c.require(d['path'].endswith('.json') and d['path'].startswith((
        'research/', 'cloud_outputs/', 'cloud_bridges/', 'local_cloud_analysis/', 'cloud_deployments/')),
        'closed JSON metadata/returned-source descriptor duty, no weight/credential path')
    return files.read(d['path'], content_binding(d))

def source_bindings(c, files):
    names = {SELF, CLOUD, OPERATOR, *c.SCIENCE_PINS}
    bindings = {n: content_binding(descriptor(c, files, n)) for n in names}
    c.require(bindings[CLOUD]['sha256'] == CLOUD_SHA and
              bindings[OPERATOR]['sha256'] == OPERATOR_SHA and
              {n: bindings[n] for n in c.SCIENCE_PINS} == c.SCIENCE_PINS,
              'unchanged cloud/operator/ten scientific SOURCE identities')
    return bindings

def exact_phase_disk(c, files, prefix, expected):
    root = files.path(prefix); found = set()
    for directory, directories, names in os.walk(root, followlinks=False):
        for name in directories:
            p = pathlib.Path(directory) / name
            c.require(not p.is_symlink() and stat.S_ISDIR(p.stat(follow_symlinks=False).st_mode), 'ordinary phase directory')
        for name in names:
            p = pathlib.Path(directory) / name
            c.require(not p.is_symlink() and stat.S_ISREG(p.stat(follow_symlinks=False).st_mode), 'regular returned phase file, no alias/FIFO')
            if name == '.DS_Store': continue # Ordinary OS metadata only, no semantic payload role.
            found.add(p.relative_to(root).as_posix())
    c.require(found == set(expected) | {'receipt.json'}, 'complete phase disk population, unexpected files stop')

def authority(c, files, raw, action):
    a = c.unique(raw)
    c.closed(a, ('version', 'status', 'root_execution_authorized', 'action',
        'sources', 'independent_source_review', 'deployment_manifest',
        'run_prefix', 'phase_receipts', 'bridges', 'local_receipts',
        'output_prefix', 'test_gold_unlock_receipt', 'analysis_inputs'))
    c.require(type(a['version']) is int and a['version'] == 1 and
        a['status'] == 'FROZEN_BEFORE_CLOUD_SUPERVISED_FIT' and
        a['root_execution_authorized'] is True and a['action'] == action,
        'root actual authority; SOURCE/INVENTED checks never authorize data')
    c.require(a['sources'] == source_bindings(c, files), 'finite current SOURCE pins')
    r = c.unique(captured(c, files, a['independent_source_review']))
    c.require(r.get('status') == 'source_review_passed' and
        r.get('independent_from_implementer') is True and
        r.get('local_bridge_sha256') == a['sources'][SELF]['sha256'],
        'different-agent practical bridge SOURCE review')
    manifest_raw = captured(c, files, a['deployment_manifest'])
    m = c.unique(manifest_raw); c.manifest_contract(m)
    c.require(manifest_raw == c.encoded(m) and m['runner'] == a['sources'][CLOUD]
        and m['operator_helper'] == a['sources'][OPERATOR], 'canonical cloud manifest')
    for group in ('science', 'controls', 'licenses'):
        for name, b in m[group].items(): files.verify(name, b)
    c.provenance_contract(files, m)
    h = 'research/round4_sccomics_cloud_cuda_hardware_amendment.json'
    c.validate_hardware(files, c.unique(files.read(h, m['controls'][h])), m)
    for n in ('run_prefix', 'output_prefix'):
        c.relative(a[n])
    c.require(a['run_prefix'].startswith('cloud_outputs/') and
        a['output_prefix'].startswith('local_cloud_analysis/'), 'separate local/cloud output paths')
    c.require(not files.path(a['output_prefix']).exists(), 'no overwrite or implicit resume')
    if action == 'score_test':
        c.require(set(a['analysis_inputs']) == {'source_clusters','historical_raw'}, 'separate fixed analysis input descriptors')
        for k, name, wanted in (('source_clusters',CLUSTERS,CLUSTERS_SHA),('historical_raw',HISTORY,HISTORY_SHA)):
            d = a['analysis_inputs'][k]
            c.closed(d,('path','sha256','size_bytes'))
            c.require(d['path'] == name and d['sha256'] == wanted, 'unchanged original cluster/history controls')
            files.verify(name,content_binding(d))
    else:
        c.require(a['analysis_inputs'] == {}, 'no unused later-stage control permission')
    return a, m

def analysis_controls(c, files, controls):
    d = controls['source_clusters']; clusters = c.unique(files.read(d['path'], content_binding(d)))
    c.require(type(clusters.get('native_test_record_count')) is int and clusters['native_test_record_count'] == 100 and
        type(clusters.get('source_text_cluster_count')) is int and clusters['source_text_cluster_count'] == 100 and
        clusters.get('gold_or_model_used') is False and clusters.get('native_split_changed') is False and
        clusters.get('same_experiment_or_pretraining_independence_certified') is False, 'original source-text clusters, no physical independence claim')
    rows = [{'cluster_id':r['cluster_id'],'source_ids':r['native_test_ids']} for r in clusters['clusters']]
    c.require(len(rows) == 100 and all(type(r['cluster_id']) is str and r['cluster_id'] and not any(x in r['cluster_id'] for x in '\r\n\t') and
        type(r['source_ids']) is list and len(r['source_ids']) == 1 and type(r['source_ids'][0]) is int for r in rows) and
        len({r['cluster_id'] for r in rows}) == 100 and {r['source_ids'][0] for r in rows} == set(c.SPLITS['test']), 'original exact100 singleton cluster identities')
    d = controls['historical_raw']; raw = files.read(d['path'], content_binding(d))
    c.require(c.sha(raw) == HISTORY_SHA, 'same historically exposed c899 source')
    history = {name:{'numerator':1,'denominator':denominator,'source_sha256':HISTORY_SHA}
        for name,denominator in zip(('polyie:typed_minus_mean','polyie:typed_minus_capacity_mean',
            'mulms:typed_minus_mean','mulms:typed_minus_capacity_mean'),(8192,8192,64,64))}
    return rows, history

class Population:
    """Whole returned phase byte population, before any held-Gold access."""
    def __init__(self, c, files, manifest, run_prefix, descriptors, phases, backend):
        self.c, self.files, self.manifest, self.backend = c, files, manifest, backend
        c.require(type(descriptors) is dict and set(descriptors) == set(phases),
                  'exact phase predecessor population')
        self.descriptors, self.receipts, self.checkpoint_bindings = copy.deepcopy(descriptors), {}, {}
        for phase in c.PHASES:
            if phase not in phases: continue
            d = descriptors[phase]
            c.require(d['path'] == run_prefix + '/' + phase + '/receipt.json', 'same cloud run receipt')
            r = c.unique(captured(c, files, d))
            c.require(r.get('version') == 1 and
                r.get('status') == 'completed_cloud_source_phase_not_Gold_score' and
                r.get('phase') == phase and r.get('output_prefix') == run_prefix + '/' + phase and
                r.get('manifest_sha256') == c.sha(c.encoded(manifest)) and
                r.get('DEV_TEST_annotation_permission') is False and
                r.get('not_a_local_Gold_score_or_statistics_certificate') is True and
                type(r.get('files')) is dict and set(r['files']) == c.expected_phase_files(phase),
                'returned complete source phase identity, never a Gold score')
            prior = {p: descriptors[p] for p in c.PHASES[:c.PHASES.index(phase)]}
            c.require(r.get('prior_receipts') == prior, 'exact immutable predecessor chain')
            for n, b in r['files'].items(): files.verify(r['output_prefix'] + '/' + n, b)
            exact_phase_disk(c, files, r['output_prefix'], r['files'])
            self.receipts[phase] = r
        self.stage = c.CloudStages.__new__(c.CloudStages)
        self.stage.files, self.stage.manifest, self.stage.backend = files, manifest, backend
        self.stage.prior, self.stage.prior_bindings = self.receipts, descriptors

    def old(self, phase, name): return self.stage.old(phase, name)
    def cp(self, kind, seed, epoch):
        raw = self.stage.cp(kind, seed, epoch)
        self.checkpoint_bindings[kind, seed, epoch] = self.c.binding(raw)
        return raw
    def probability(self, kind, seed, epoch, identifier, mentions=None):
        phase = 'ner_dev' if kind == 'span' else 'pair_dev'
        obj = self.c.unique(self.old(phase, self.c.probability_name(kind, seed, epoch, identifier)))
        key = kind, seed, epoch
        if key not in self.checkpoint_bindings: self.cp(kind, seed, epoch)
        return self.stage.source_record(obj, kind, seed, epoch, identifier,
            self.checkpoint_bindings[key], mentions)
    def recheck(self):
        for phase, r in self.receipts.items():
            self.files.verify(self.descriptors[phase]['path'], content_binding(self.descriptors[phase]))
            for n, b in r['files'].items(): self.files.verify(r['output_prefix'] + '/' + n, b)
            exact_phase_disk(self.c, self.files, r['output_prefix'], r['files'])

class LocalSourceBackend:
    """Read source-only cache/tokenizer; never construct a neural model/optimizer."""
    def __init__(self, c, files, manifest, science, temporary):
        import torch, numpy, transformers
        self.c, self.files, self.manifest, self.science = c, files, manifest, science
        self.torch, self.numpy = torch, numpy
        self.core, self.models = science['sccomics_fitting_core'], science['sccomics_source_models']
        self.projection = science['sccomics_training_projection']
        self.descriptors, self.opened_annotations = {}, []
        cold = pathlib.Path(temporary) / 'tokenizer'; cold.mkdir()
        for name in c.COLD_NAMES:
            if name == 'model.safetensors': continue
            files.copy(c.COLD_ROOT + '/' + name, manifest['inputs'][c.COLD_ROOT + '/' + name], cold / name)
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(str(cold),
            local_files_only=True, trust_remote_code=False, use_fast=True, token=False)
    def text(self, i): return self.files.read(self.c.text_path(i), self.manifest['inputs'][self.c.text_path(i)])
    def cache(self, i):
        c = self.c
        raw = self.files.read(c.cache_path(i), self.manifest['inputs'][c.cache_path(i)])
        saved = self.torch.load(io.BytesIO(raw), map_location='cpu', weights_only=True)
        self.core._validate_cache(saved); text = self.text(i)
        split = next(s for s, ids in c.SPLITS.items() if i in ids)
        c.require(type(saved.get('source_id')) is int and saved['source_id'] == i and
            saved.get('split') == split and saved.get('source_sha256') == c.sha(text), 'original source cache identity')
        enc = self.tokenizer(text.decode('utf-8'), add_special_tokens=False,
                             truncation=False, return_offsets_mapping=True)
        offsets = [list(x) for x in enc['offset_mapping']]
        c.require(saved['wordpieces'] == len(enc['input_ids']) and
            [list(x) for x in saved['offsets']] == offsets, 'original complete tokenizer/cache offsets')
        candidates = [list(x) for x in self.models.span_candidates(offsets)]
        c.require(type(saved.get('candidate_span_count')) is int and
            saved['candidate_span_count'] == len(candidates), 'source candidate population')
        self.descriptors[i] = {'candidates': candidates, 'text_length': len(text.decode('utf-8')),
                              'wordpieces': len(offsets), 'source_sha256': c.sha(text)}
        return saved
    def donor_diagnostics(self, mentions, seed, i):
        donor = self.models.donor_permutation(mentions, str(i), self.manifest['inputs'][self.c.text_path(i)]['sha256'], seed)
        return {'complete_directed_rows': len(mentions)**2,
            'donor_sha256': self.c.sha(donor.astype('<i8').tobytes()),
            'donor_fixed_points': int(self.numpy.count_nonzero(donor == self.numpy.arange(len(donor)))),
            'source_mentions_sha256': self.core.json_digest(mentions), 'Gold_consulted': False,
            'inference_row_block_size': 512, 'candidate_cap': None}
    def detector(self, obj, threshold):
        # Frozen function expects a two-dimensional NumPy array even for 0 rows.
        values = self.numpy.asarray(obj['probabilities'], dtype=float).reshape((len(obj['candidates']), 9))
        return self.core.predicted_mentions(obj['candidates'], values, threshold)
    def detector_records(self, i, mentions):
        text = self.text(i).decode('utf-8')
        return [{'family': 'T', 'id': f'T{j+1}', 'type': m['type'],
                 'segments': copy.deepcopy(m['segments']),
                 'text': ' '.join(text[a:b] for a, b in m['segments'])} for j, m in enumerate(mentions)]
    def records(self, i, mentions, rows, rt, et):
        n = len(mentions)
        self.c.require(type(rows) is list and len(rows) == n*n and
            all(type(r) is list and len(r) == 5 for r in rows), 'strict N² before zero reshape')
        matrix = self.numpy.asarray(rows, dtype=float).reshape((n*n, 5))
        return self.projection.emit_native_prediction_records(self.text(i).decode('utf-8'),
            mentions, matrix, relation_threshold=rt, role_threshold=et)

def annotation_bindings(c, files):
    name = next(n for n in c.PROVENANCE if 'acquisition' in n)
    rows = c.unique(files.read(name, {'sha256': c.PROVENANCE[name],
        'size_bytes': files.path(name).stat().st_size}))['official_archives']
    result = {}
    for archive in rows.values():
        for n, row in archive['members'].items():
            if n.endswith('.ann'):
                i = int(pathlib.PurePosixPath(n).stem)
                c.require(1 <= i <= 1000 and i not in result, 'complete unique native ANN metadata')
                result[i] = {'sha256': row['sha256'], 'size_bytes': row['bytes']}
    c.require(set(result) == set(range(1, 1001)), 'all1000 ANN metadata, no semantic read')
    return result

def held_gold(c, files, backend, science, ids, permitted, ann_bindings):
    c.require(tuple(ids) == tuple(c.SPLITS[permitted]), 'exact staged held-Gold permission')
    out = {}
    for i in ids:
        name = f'data/sccomics_round4/source_v3/raw_annotations/{i:04}.ann'
        raw = files.read(name, ann_bindings[i]); text = backend.text(i)
        backend.opened_annotations.append({'path': name, 'source_id': i,
            'permission': 'local_' + permitted + '_only_after_complete_returned_source',
            'sha256': c.sha(raw)})
        out[str(i)] = science['sccomics_native_graph'].GoldSource(text, raw)
    return out

def select_ner(c, population, science, gold_sources):
    native, selector = science['sccomics_native_graph'], science['sccomics_development_selection']
    gold = {i: native.parse_document(gold_sources[str(i)].text_utf8,
        gold_sources[str(i)].annotation_utf8, gold=True) for i in c.SPLITS['dev']}
    settings = []
    for e in range(1, 11):
        # One epoch population retained across its eight thresholds, not whole CP/cache RAM.
        source = {(s, i): population.probability('span', s, e, i) for s in c.SEEDS for i in c.SPLITS['dev']}
        for t in c.SPAN_THRESHOLDS:
            by = {str(s): {} for s in c.SEEDS}
            for s in c.SEEDS:
                for i in c.SPLITS['dev']:
                    mentions = population.backend.detector(source[s, i], t)
                    records = population.backend.detector_records(i, mentions)
                    pred = native.parse_prediction_records(population.backend.text(i), records)
                    by[str(s)][str(i)] = {'T_complete_native': selector.text_counts(pred, gold[i])}
            settings.append({'epoch': e, 'thresholds': [t], 'by_seed_document': by})
    result = selector.select_detector(settings); choice = {k: result['chosen'][k] for k in ('epoch', 'threshold')}
    locked = {str(s): {str(i): population.backend.detector(
        population.probability('span', s, choice['epoch'], i), choice['threshold'])
        for i in c.SPLITS['dev']} for s in c.SEEDS}
    return result, settings, choice, locked

def select_heads(c, population, science, gold_sources, locked):
    native, selector = science['sccomics_native_graph'], science['sccomics_development_selection']
    gold = {i: native.parse_document(gold_sources[str(i)].text_utf8,
        gold_sources[str(i)].annotation_utf8, gold=True) for i in c.SPLITS['dev']}
    results, all_settings, choices = {}, {}, {}
    for k in c.HEADS:
        settings, event_cache = [], {}
        for e in range(1, 11):
            source = {(s, i): population.probability(k, s, e, i, locked[str(s)][str(i)])
                      for s in c.SEEDS for i in c.SPLITS['dev']}
            for rt in c.PAIR_THRESHOLDS:
                for et in c.PAIR_THRESHOLDS:
                    by = {str(s): {} for s in c.SEEDS}
                    for s in c.SEEDS:
                        for i in c.SPLITS['dev']:
                            records, _ = population.backend.records(i, locked[str(s)][str(i)], source[s, i]['probabilities'], rt, et)
                            pred = native.parse_prediction_records(population.backend.text(i), records)
                            row = selector.primary_counts(pred, gold[i], event_cache=event_cache)
                            by[str(s)][str(i)] = {n: row[n] for n in selector.ENDPOINTS}
                    settings.append({'epoch': e, 'thresholds': [rt, et], 'by_seed_document': by})
        result = selector.select_pair_head(settings)
        results[k], all_settings[k] = result, settings
        choices[k] = {n: result['chosen'][n] for n in ('epoch', 'relation_threshold', 'role_threshold')}
    return results, all_settings, choices

def validate_development(c, p, ner=None):
    kinds = ('span',) if ner is None else c.KINDS
    for k in kinds:
        for s in c.SEEDS:
            for e in range(1, 11):
                p.cp(k, s, e)
                for i in c.SPLITS['dev']:
                    p.probability(k, s, e, i, None if k == 'span' else ner['locked_mentions'][str(s)][str(i)])

def validate_bridges(c, p, a):
    c.require(set(a['bridges']) == {'NER', 'pairs'} and set(a['local_receipts']) == {'NER', *c.HEADS}, 'all fixed choices/local grids')
    nr = captured(c, p.files, a['bridges']['NER']); ner = c.unique(nr)
    pr = captured(c, p.files, a['bridges']['pairs']); pair = c.unique(pr)
    p.stage.gate = {'local_selector_receipt_sha256': {k: d['sha256'] for k, d in a['local_receipts'].items()}}
    p.stage.bridge = pair
    p.stage.validate_ner_bridge(ner, nr); choices = p.stage.test_choices(nr)
    c.require(pair['ner_bridge'] == a['bridges']['NER'] and
        p.receipts['fit_heads']['selection_bridge_sha256'] == c.sha(nr) and
        p.receipts['pair_dev']['selection_bridge_sha256'] == c.sha(nr), 'same immutable NER bridge through heads')
    for k, d in a['local_receipts'].items():
        r = c.unique(captured(c, p.files, d))
        c.require(r.get('status') == 'complete_local_native_development_grid' and r.get('family') == k and
            r.get('source_selector_sha256') == p.manifest['science']['src/sccomics_development_selection.py']['sha256'] and
            r.get('complete_setting_count') == (80 if k == 'NER' else 250) and
            r.get('population_receipt_sha256') == p.descriptors['ner_dev' if k == 'NER' else 'pair_dev']['sha256'] and
            r.get('seed_ids') == list(c.SEEDS) and r.get('development_source_ids') == list(c.SPLITS['dev']) and
            r.get('test_Gold_consulted') is False, 'complete exact local grid receipt')
        settings = c.unique(captured(c, p.files, r['settings']))
        c.require(r['settings']['path'].startswith('local_cloud_analysis/'), 'local complete count grid stays local')
        selector = p.backend.science['sccomics_development_selection']
        actual = selector.select_detector(settings) if k == 'NER' else selector.select_pair_head(settings)
        c.require(actual == r['selector_result'], 'saved complete count-grid exact reselection')
        expected = {n: actual['chosen'][n] for n in (('epoch', 'threshold') if k == 'NER' else ('epoch','relation_threshold','role_threshold'))}
        c.require(expected == (ner['choice'] if k == 'NER' else choices[k]), 'same frozen choices')
    return ner, nr, pair, pr

def returned_test(c, p, a):
    """Fresh all150CP/all15kDEV/all1500TEST reconstruction, before TEST Gold."""
    ner, nr, pair, pr = validate_bridges(c, p, a)
    validate_development(c, p, ner)
    checkpoints = {c.checkpoint_name(k, s, e): c.binding(p.cp(k, s, e))
                   for k in c.KINDS for s in c.SEEDS for e in range(1, 11)}
    freeze = c.unique(p.old('test_sources', 'before_test_source_freeze.json'))
    expected = {'status': 'all150CP_all15000dev_source_content_and_local_choices_locked_before_test_source_generation',
        'checkpoint_bindings': checkpoints, 'prior_receipts': {k: a['phase_receipts'][k] for k in c.PHASES[:-1]},
        'NER_bridge_sha256': c.sha(nr), 'pair_bridge_sha256': c.sha(pr), 'NER_choice': ner['choice'],
        'pair_choices': pair['choices'], 'held_Gold_present_or_opened_on_cloud': False, 'local_grid_math_certified_here': False}
    c.require(freeze == expected and p.receipts['test_sources']['selection_bridge_sha256'] == c.sha(pr), 'all150 checkpoint/choice TEST source freeze')
    graphs = {k: {str(s): {} for s in c.SEEDS} for k in c.KINDS}; locked = {}
    for k in c.KINDS:
        for s in c.SEEDS:
            choice = ner['choice'] if k == 'span' else pair['choices'][k]
            cp = c.binding(p.cp(k, s, choice['epoch']))
            for i in c.SPLITS['test']:
                row = c.unique(p.old('test_sources', c.test_name(k, s, i)))
                allowed = {'source_probability','original_source_records','source_only','test_Gold_consulted','NER_choice'}
                if k != 'span': allowed |= {'emission_diagnostics','pair_choice'}
                c.closed(row, allowed)
                c.require(row['source_only'] is True and row['test_Gold_consulted'] is False and row['NER_choice'] == ner['choice'], 'no hidden TEST Gold')
                obj = p.stage.source_record(row['source_probability'], k, s, choice['epoch'], i, cp,
                    None if k == 'span' else locked[s, i])
                if k == 'span':
                    locked[s, i] = p.backend.detector(obj, choice['threshold'])
                    records = p.backend.detector_records(i, locked[s, i])
                else:
                    c.require(row['pair_choice'] == choice, 'same paired threshold/epoch choice')
                    records, diagnostics = p.backend.records(i, locked[s, i], obj['probabilities'], choice['relation_threshold'], choice['role_threshold'])
                    c.require(row['emission_diagnostics'] == diagnostics, 'same source-only emitter diagnostics')
                c.require(type(row['original_source_records']) is list and row['original_source_records'] == records,
                          'complete original source graph exactly rebuilt; [] only if actual emitter empty')
                graphs[k][str(s)][str(i)] = copy.deepcopy(records)
    p.recheck()
    return graphs

def finalize_inputs(c, files, a, manifest, authority_descriptor, opened_annotations):
    # No claim to read unopened held ANN bytes: only actually opened ANN rehashed.
    for n, b in a['sources'].items(): files.verify(n, b)
    for group in ('inputs', 'controls', 'licenses'):
        for n, b in manifest[group].items(): files.verify(n, b)
    for d in (a['deployment_manifest'], a['independent_source_review'], authority_descriptor): files.verify(d['path'], content_binding(d))
    for group in ('phase_receipts', 'bridges', 'local_receipts'):
        for d in a[group].values(): files.verify(d['path'], content_binding(d))
    for d in a['analysis_inputs'].values(): files.verify(d['path'], content_binding(d))
    if a['test_gold_unlock_receipt'] is not None:
        files.verify(a['test_gold_unlock_receipt']['path'],content_binding(a['test_gold_unlock_receipt']))
    for row in opened_annotations:
        files.verify(row['path'], {'sha256': row['sha256'], 'size_bytes': files.path(row['path']).stat().st_size})

def write_object(c, files, name, value):
    b = files.new(name, c.encoded(value))
    if not hasattr(files, '_local_created_bindings'): files._local_created_bindings = {}
    c.require(name not in files._local_created_bindings, 'distinct local output identity')
    files._local_created_bindings[name] = b
    return {'path': name, **b}

def execute(c, files, a, manifest, backend, science, authority_descriptor):
    action = a['action']; phases = {'select_ner': c.PHASES[:2], 'select_heads': c.PHASES[:4],
                                  'verify_test': c.PHASES, 'score_test': c.PHASES}[action]
    p = Population(c, files, manifest, a['run_prefix'], a['phase_receipts'], phases, backend)
    ann = annotation_bindings(c, files); out = a['output_prefix']; result = None
    if action == 'select_ner':
        c.require(a['bridges'] == {} and a['local_receipts'] == {} and a['test_gold_unlock_receipt'] is None, 'NER stage no extra later authority')
        validate_development(c, p); p.recheck() # Whole30CP/3000source content BEFORE first DEV annotation.
        gold = held_gold(c, files, backend, science, c.SPLITS['dev'], 'dev', ann)
        selected, settings, choice, locked = select_ner(c, p, science, gold)
        grid = write_object(c, files, out + '/NER_settings.json', settings)
        receipt = write_object(c, files, out + '/NER_selection_receipt.json', {
            'status': 'complete_local_native_development_grid','family': 'NER','complete_setting_count':80,
            'seed_ids':list(c.SEEDS),'development_source_ids':list(c.SPLITS['dev']),
            'source_selector_sha256':manifest['science']['src/sccomics_development_selection.py']['sha256'],
            'population_receipt_sha256':a['phase_receipts']['ner_dev']['sha256'],
            'settings':grid,'selector_result':selected,'test_Gold_consulted':False})
        bridge = {'version':1,'kind':'local_ner_source_bridge',
            'population_receipt_sha256':a['phase_receipts']['ner_dev']['sha256'],
            'selector_source_sha256':manifest['science']['src/sccomics_development_selection.py']['sha256'],
            'local_grid_proof':{'complete_setting_count':80,'seed_ids':list(c.SEEDS),
                'development_source_ids':list(c.SPLITS['dev']),'Gold_kept_local':True,
                'local_checked_selector_receipt_sha256':receipt['sha256']},
            'choice':choice,'locked_mentions':locked}
        bridge_name = 'cloud_bridges/' + a['run_prefix'][len('cloud_outputs/'):] + '/NER_bridge.json'
        result = {'NER_bridge':write_object(c, files, bridge_name, bridge), 'local_receipts':{'NER':receipt}}
    elif action == 'select_heads':
        c.require(set(a['bridges']) == {'NER'} and set(a['local_receipts']) == {'NER'} and a['test_gold_unlock_receipt'] is None, 'pair stage only NER predecessor')
        nr = captured(c, files, a['bridges']['NER']); ner = c.unique(nr)
        p.stage.gate = {'local_selector_receipt_sha256': {'NER':a['local_receipts']['NER']['sha256']}}
        p.stage.validate_ner_bridge(ner, nr)
        local = c.unique(captured(c, files, a['local_receipts']['NER']))
        c.require(local.get('status') == 'complete_local_native_development_grid' and local.get('family') == 'NER' and
            local.get('complete_setting_count') == 80 and local.get('seed_ids') == list(c.SEEDS) and
            local.get('development_source_ids') == list(c.SPLITS['dev']) and local.get('test_Gold_consulted') is False and
            local.get('population_receipt_sha256') == a['phase_receipts']['ner_dev']['sha256'] and
            local.get('source_selector_sha256') == manifest['science']['src/sccomics_development_selection.py']['sha256'], 'same complete80 NER selector predecessor')
        c.require(local['settings']['path'].startswith('local_cloud_analysis/'), 'NER count grid stays local')
        selected = science['sccomics_development_selection'].select_detector(c.unique(captured(c, files, local['settings'])))
        c.require(selected == local['selector_result'] and {k:selected['chosen'][k] for k in ('epoch','threshold')} == ner['choice'], 'all80 NER saved-grid exact reselection before pair Gold')
        c.require(p.receipts['fit_heads']['selection_bridge_sha256'] == c.sha(nr) and
            p.receipts['pair_dev']['selection_bridge_sha256'] == c.sha(nr), 'same NER before all120CP')
        validate_development(c, p, ner); p.recheck()
        gold = held_gold(c, files, backend, science, c.SPLITS['dev'], 'dev', ann)
        results, settings, choices = select_heads(c, p, science, gold, ner['locked_mentions'])
        proofs, receipts = {}, {}
        for k in c.HEADS:
            grid = write_object(c, files, out + '/' + k + '_settings.json', settings[k])
            receipts[k] = write_object(c, files, out + '/' + k + '_selection_receipt.json', {
                'status':'complete_local_native_development_grid','family':k,'complete_setting_count':250,
                'seed_ids':list(c.SEEDS),'development_source_ids':list(c.SPLITS['dev']),
                'source_selector_sha256':manifest['science']['src/sccomics_development_selection.py']['sha256'],
                'population_receipt_sha256':a['phase_receipts']['pair_dev']['sha256'],
                'settings':grid,'selector_result':results[k],'test_Gold_consulted':False})
            proofs[k] = {'complete_setting_count':250,'seed_ids':list(c.SEEDS),
                'development_source_ids':list(c.SPLITS['dev']),'Gold_kept_local':True,
                'local_checked_selector_receipt_sha256':receipts[k]['sha256']}
        bridge_name = 'cloud_bridges/' + a['run_prefix'][len('cloud_outputs/'):] + '/pair_bridge.json'
        result = {'pair_bridge':write_object(c, files, bridge_name, {
            'version':1,'kind':'local_all_pair_choices_source_bridge','ner_bridge':a['bridges']['NER'],
            'population_receipt_sha256':a['phase_receipts']['pair_dev']['sha256'],
            'selector_source_sha256':manifest['science']['src/sccomics_development_selection.py']['sha256'],
            'choices':choices,'local_grid_proofs':proofs}), 'local_receipts':receipts}
    else:
        graphs = returned_test(c, p, a)
        graph_file = write_object(c, files, out + '/rebuilt_complete_source_graphs.json', graphs)
        unlock = {'status':'all150CP_allchoices_all15000DEV_all1500TEST_source_content_rebuilt_before_TEST_Gold',
            'deployment_manifest':a['deployment_manifest'],'phase_receipts':a['phase_receipts'],
            'bridges':a['bridges'],'local_receipts':a['local_receipts'],'seed_ids':list(c.SEEDS),
            'test_source_ids':list(c.SPLITS['test']),'source_graph_population':1500,
            'source_graphs':graph_file,'test_Gold_consulted':False}
        if action == 'verify_test':
            c.require(a['test_gold_unlock_receipt'] is None, 'verify is first source-only unlock')
            result = {'unlock_receipt':write_object(c, files, out + '/TEST_Gold_unlock_receipt.json', unlock)}
        else:
            d = a['test_gold_unlock_receipt']; c.require(d is not None, 'separate root-bound source unlock before TEST Gold')
            prior = c.unique(captured(c, files, d)); saved = c.unique(captured(c, files, prior['source_graphs']))
            c.require({k:v for k,v in prior.items() if k != 'source_graphs'} == {k:v for k,v in unlock.items() if k != 'source_graphs'} and saved == graphs, 'same exact source-only unlock, rebuilt again before Gold')
            clusters, history = analysis_controls(c, files, a['analysis_inputs'])
            p.recheck() # All source-only gates have passed. This is the first TEST ANN permission.
            gold = held_gold(c, files, backend, science, c.SPLITS['test'], 'test', ann)
            supplied = (gold, graphs['span'], {k:graphs[k] for k in c.HEADS}, clusters, history)
            module = science['sccomics_test_analysis']
            primary = module.analyze_test_population(*supplied)
            primary_d = write_object(c, files, out + '/primary_test_analysis.json', primary)
            count_replay = module.replay_test_population(*supplied, primary)
            c.require(count_replay.get('primary_witnesses_validated_against_independent_finite_closures') is True,
                'different parser/scorer complete count and witness replay')
            count_d = write_object(c, files, out + '/independent_native_count_replay.json', count_replay)
            arithmetic = science['sccomics_independent_statistics_v2'].replay_statistics(
                primary['cluster_statistics_input'], expected_cluster_ids=[r['cluster_id'] for r in clusters],
                historical_raw=history, primary_statistics=primary['statistics'])
            arithmetic_d = write_object(c, files, out + '/independent_statistics_replay.json', arithmetic)
            result = {'primary_analysis':primary_d,'different_native_count_replay':count_d,
                'different_statistical_arithmetic_replay':arithmetic_d,
                'random_samples_independent_from_primary':False,'source_cluster_count':100,
                'three_fits_are_not_N300':True,'physical_truth_or_independent_experiments_certified':False,
                'paper_peer_review_or_ranking_performed':False}
    p.recheck(); finalize_inputs(c, files, a, manifest, authority_descriptor, backend.opened_annotations)
    for name, wanted in getattr(files, '_local_created_bindings', {}).items(): files.verify(name, wanted)
    return write_object(c, files, out + '/completed_local_action_receipt.json', {
        'status':'completed_local_' + action,'actual_scientific_fit_performed_here':False,
        'authority':authority_descriptor,'result':result,'opened_annotation_events':backend.opened_annotations,
        'output_bindings_before_completion':copy.deepcopy(getattr(files,'_local_created_bindings',{})),
        'unopened_held_ANN_bytes_rehashed':False,'whole_installed_runtime_byte_inventory_claimed':False})

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',required=True)
    p.add_argument('--authority',required=True);p.add_argument('--authority-sha256',required=True)
    p.add_argument('--action',choices=('select_ner','select_heads','verify_test','score_test'),required=True)
    p.add_argument('--execute-root-frozen-local-bridge',action='store_true');a=p.parse_args()
    if not a.execute_root_frozen_local_bridge: raise ValueError('default SOURCE-only: no actual cache/NN/held-Gold access')
    c=cloud_source(a.root);files=c.Files(a.root);c.relative(a.authority)
    c.require(a.authority.startswith('research/round4_sccomics_cloud_cuda_') and a.authority.endswith('.json'),'authority metadata path')
    d={'path':a.authority,'sha256':a.authority_sha256,'size_bytes':files.path(a.authority).stat().st_size}
    raw=captured(c,files,d);auth,manifest=authority(c,files,raw,a.action)
    captures={n:files.read('src/'+n+'.py',manifest['science']['src/'+n+'.py']) for n in c.SCIENCE}
    with tempfile.TemporaryDirectory(prefix='SC_LOCAL_SOURCE_TOKENIZER_') as temp,c.CapturedScience(captures) as science:
        backend=LocalSourceBackend(c,files,manifest,science,pathlib.Path(temp).resolve())
        result=execute(c,files,auth,manifest,backend,science,d)
    print(json.dumps(result,sort_keys=True))

if __name__=='__main__':main()
