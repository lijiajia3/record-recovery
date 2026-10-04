"""Prospective SC training surrogates and source-only output emitter.

No filesystem, model, tokenizer, scorer, API, or previous-task loader. A caller
must separately authorize native TRAIN Gold before passing a NativeDocument.
Binary pair supervision is explicitly lossy relative to native multiset graphs;
the complete original Gold document is never returned as a filtered population.
"""
from __future__ import annotations

from collections import Counter
import hashlib
import json
import re

import numpy as np

TEXT_TYPES = tuple(sorted(('Characterization', 'Material', 'Property', 'Element',
                          'Process', 'Value', 'SC', 'Main', 'Doping')))
R_TYPES = ('Condition', 'Equivalent', 'Target')
ROLE_TYPES = ('Dopant', 'Site')
PAIR_OUTPUTS = R_TYPES + ROLE_TYPES


def _key(mention):
    return mention['type'], tuple(tuple(x) for x in mention['segments'])


def _require_gold(doc):
    if not doc.gold or any(not r.valid for r in doc.records):
        raise ValueError('Only an intact, separately authorized TRAIN Gold document')


def unique_training_mentions(doc):
    """Merge identical T features for supervision only, retaining ID crosswalk.

    Unknown valid native T types remain in this source population. Their neural
    type features are explicitly unavailable, rather than relabeled. Native
    T occurrences are still scored separately by the unchanged native scorer.
    """
    _require_gold(doc)
    mentions, lookup, id_rows = [], {}, {}
    for record in doc.family('T'):
        key = record.mention_key()
        if key not in lookup:
            lookup[key] = len(mentions)
            mentions.append({'type': record.label,
                             'segments': [list(x) for x in record.segments]})
        id_rows[record.identifier] = lookup[key]
    return mentions, id_rows


def project_pair_supervision(doc):
    """All N² directed rows and five binary targets, without type pruning.

    Native E endpoints are projected to their T triggers for this training
    surrogate. Numeric role suffixes project to their base role only here.
    One event per Doping trigger with flat T roles is the declared output family;
    nested roles, duplicate roles/events and other valid native structures remain
    in full evaluation Gold even when this projection cannot express them.
    """
    mentions, id_rows = unique_training_mentions(doc)
    n = len(mentions)
    target = np.zeros((n * n, len(PAIR_OUTPUTS)), dtype=np.float32)
    diagnostics = Counter()
    ids = doc.by_id

    def row(identifier):
        record = ids[identifier]
        return id_rows[identifier if record.family == 'T' else record.trigger]

    positive_occurrences = 0
    for record in doc.family('R'):
        diagnostics['native_R_occurrences'] += 1
        if record.label not in R_TYPES:
            diagnostics['unsupported_R_label_occurrences'] += 1
            continue
        h, t = row(record.arg1), row(record.arg2)
        k = R_TYPES.index(record.label)
        diagnostics['R_event_endpoint_occurrences'] += int(
            ids[record.arg1].family == 'E' or ids[record.arg2].family == 'E')
        diagnostics['collapsed_R_target_occurrences'] += int(target[h * n + t, k] == 1)
        target[h * n + t, k] = 1
        positive_occurrences += 1
    trigger_events = Counter()
    for record in doc.family('E'):
        diagnostics['native_E_occurrences'] += 1
        if record.label != 'Doping' or ids[record.trigger].label != 'Doping':
            diagnostics['unsupported_event_or_trigger_occurrences'] += 1
            diagnostics['unsupported_event_role_occurrences'] += len(record.roles)
            continue
        h = row(record.trigger)
        trigger_events[h] += 1
        if not record.roles:
            diagnostics['zero_role_Doping_events'] += 1
        for role, endpoint in record.roles:
            diagnostics['native_supported_event_role_occurrences'] += 1
            role_base = next((base for base in ROLE_TYPES
                             if role == base or re.fullmatch(re.escape(base) + r'[0-9]+', role)), None)
            if role_base is None:
                diagnostics['unsupported_role_name_occurrences'] += 1
                continue
            diagnostics['numeric_suffix_role_occurrences'] += int(role != role_base)
            diagnostics['nested_role_endpoint_occurrences'] += int(ids[endpoint].family == 'E')
            t = row(endpoint)
            k = len(R_TYPES) + ROLE_TYPES.index(role_base)
            diagnostics['collapsed_role_target_occurrences'] += int(target[h * n + t, k] == 1)
            target[h * n + t, k] = 1
            positive_occurrences += 1
    diagnostics['multiple_event_same_trigger_excess'] = sum(max(0, x - 1) for x in trigger_events.values())
    diagnostics['native_T_occurrences'] = len(doc.family('T'))
    diagnostics['unique_T_training_features'] = n
    diagnostics['duplicate_T_feature_occurrences'] = len(doc.family('T')) - n
    diagnostics['unknown_T_training_features'] = sum(m['type'] not in TEXT_TYPES for m in mentions)
    diagnostics['positive_projected_occurrences_before_binary_merge'] = positive_occurrences
    diagnostics['positive_binary_cells'] = int(target.sum())
    diagnostics['native_AUX_occurrences'] = len(doc.family('AUX'))
    return mentions, target, dict(diagnostics)


def project_span_supervision(doc, candidates):
    """Targets for source-derived contiguous candidate boundaries only.

    No Gold candidate insertion or span hull conversion. Duplicate occurrences
    share one training cell; their native multiplicity is not changed in Gold.
    """
    _require_gold(doc)
    boundaries = [(int(c[2]), int(c[3])) for c in candidates]
    if len(boundaries) != len(set(boundaries)):
        raise ValueError('Duplicate source candidate boundaries')
    lookup = {x: i for i, x in enumerate(boundaries)}
    targets = np.zeros((len(candidates), len(TEXT_TYPES)), dtype=np.float32)
    diagnostics = Counter()
    for record in doc.family('T'):
        diagnostics['native_T_occurrences'] += 1
        if record.label not in TEXT_TYPES:
            diagnostics['unsupported_T_type_occurrences'] += 1
            continue
        if len(record.segments) != 1 or record.segments[0] not in lookup:
            diagnostics['unrepresented_T_occurrences'] += 1
            continue
        row, kind = lookup[record.segments[0]], TEXT_TYPES.index(record.label)
        diagnostics['collapsed_T_target_occurrences'] += int(targets[row, kind] == 1)
        targets[row, kind] = 1
        diagnostics['representable_T_occurrences'] += 1
    diagnostics['positive_binary_cells'] = int(targets.sum())
    return targets, dict(diagnostics)


def sampled_training_rows(targets, *, seed, epoch, source_id, task,
                          negative_ratio=4, minimum_negatives=64, maximum_negatives=1024):
    """Same deterministic supervised sample across heads, not a donor policy.

    Include every positive row and sample negative rows without replacement.
    Complete feature donors must be assigned before calling this function.
    """
    targets = np.asarray(targets)
    if targets.ndim != 2 or not np.isfinite(targets).all() or not np.isin(targets, [0, 1]).all():
        raise ValueError('Finite binary target matrix required')
    if (type(seed) is not int or type(epoch) is not int or epoch < 1 or
            task not in {'span', 'pair'} or any(type(x) is not int or x < 0 for x in
            (negative_ratio, minimum_negatives, maximum_negatives)) or maximum_negatives < minimum_negatives):
        raise ValueError('Invalid prospective sampling configuration')
    positive = np.flatnonzero(targets.any(axis=1))
    negative = np.flatnonzero(~targets.any(axis=1))
    payload = json.dumps({'seed': seed, 'epoch': epoch, 'source_id': str(source_id),
                          'task': task, 'policy': 'sccomics_training_sample_v1'},
                         sort_keys=True, separators=(',', ':')).encode()
    rng = np.random.default_rng(int.from_bytes(hashlib.sha256(payload).digest()[:16], 'big'))
    count = min(len(negative), maximum_negatives, max(minimum_negatives, negative_ratio * len(positive)))
    selected = np.concatenate([positive, rng.choice(negative, size=count, replace=False)])
    selected.sort()
    return selected.astype(np.int64)


def emit_native_prediction_records(source_text, mentions, probabilities, *,
                                   relation_threshold, role_threshold):
    """Source-only fixed emitter: one flat Doping E per predicted trigger.

    Every supplied predicted T is emitted, then all qualifying directed R cells
    including diagonal cells. R endpoints referring to a Doping trigger use its
    emitted E. E roles are unindexed Dopant/Site with T endpoints, at most one
    occurrence per trigger/role/target; no Gold-based role cap or type restriction.
    Doping triggers with zero predicted roles still emit a valid empty-role E.
    Unknown/malformed mentions or probabilities stop generation, never get dropped.
    """
    if not isinstance(source_text, str):
        raise TypeError('Unmodified Unicode source text required')
    if any(not np.isfinite(t) or not 0 <= t <= 1 for t in (relation_threshold, role_threshold)):
        raise ValueError('Thresholds must be finite and within [0,1]')
    n = len(mentions)
    probabilities = np.asarray(probabilities)
    if (probabilities.shape != (n * n, len(PAIR_OUTPUTS)) or
            not np.isfinite(probabilities).all() or
            ((probabilities < 0) | (probabilities > 1)).any()):
        raise ValueError('Complete finite N² × five probability population required')
    records, seen = [], set()
    for i, mention in enumerate(mentions):
        if (set(mention) != {'type', 'segments'} or mention['type'] not in TEXT_TYPES or
                not isinstance(mention['segments'], list) or not mention['segments'] or
                any(not isinstance(s, list) or len(s) != 2 or any(type(v) is not int for v in s) or
                    not 0 <= s[0] < s[1] <= len(source_text) for s in mention['segments'])):
            raise ValueError('Malformed or unavailable predicted T')
        segments = mention['segments']
        if any(max(a, c) < min(b, d) for index, (a, b) in enumerate(segments)
               for c, d in segments[index + 1:]):
            raise ValueError('Overlapping segments inside one predicted mention')
        identity = _key(mention)
        if identity in seen:
            raise ValueError('Duplicate predicted T identity')
        seen.add(identity)
        records.append({'family': 'T', 'id': f'T{i + 1}', 'type': mention['type'],
                        'segments': [list(s) for s in segments],
                        'text': ' '.join(source_text[a:b] for a, b in segments)})
    events = {i: f'E{j + 1}' for j, i in enumerate(i for i, m in enumerate(mentions) if m['type'] == 'Doping')}
    for h, identifier in events.items():
        roles = [{'role': kind, 'target': f'T{t + 1}'}
                 for k, kind in enumerate(ROLE_TYPES, len(R_TYPES)) for t in range(n)
                 if probabilities[h * n + t, k] >= role_threshold]
        records.append({'family': 'E', 'id': identifier, 'type': 'Doping',
                        'trigger': f'T{h + 1}', 'roles': roles})
    relation_number = 0
    for h in range(n):
        for t in range(n):
            for k, kind in enumerate(R_TYPES):
                if probabilities[h * n + t, k] >= relation_threshold:
                    relation_number += 1
                    records.append({'family': 'R', 'id': f'R{relation_number}', 'type': kind,
                                    'arg1': events.get(h, f'T{h + 1}'),
                                    'arg2': events.get(t, f'T{t + 1}')})
    diagnostics = {'source_mentions': n, 'complete_directed_pair_rows': n * n,
                   'emitted_T': n, 'emitted_E': len(events), 'emitted_R': relation_number,
                   'qualifying_roles_at_nonDoping_heads_not_in_output_family': int(sum(
                       np.sum(probabilities[h * n:(h + 1) * n, len(R_TYPES):] >= role_threshold)
                       for h in range(n) if h not in events)),
                   'Gold_consulted_during_emission': False,
                   'nested_E_roles_or_identical_role_target_multiplicity_supported': False,
                   'same_role_to_distinct_T_targets_supported': True}
    return records, diagnostics
