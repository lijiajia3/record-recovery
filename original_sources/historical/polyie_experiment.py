#!/usr/bin/env python3
"""Strict role-aware POLYIE development, public given spans, SiliconFlow only.

Inputs and gold are separate. Source-only windows and all gold coverage are
fixed at preparation. Never use this development command to silently score test.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import re

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from pilot import ROOT, append, begin_attempt, call, prf, read_json, write_json

DATA = ROOT / 'data/polyie'
ROLES = ('CN', 'PN', 'PV', 'Condition')
WIDTH, STRIDE = 1000, 800
VERSION = os.environ.get('POLYIE_PROTOCOL_VERSION', 'development_v3')
GENERATED = ('baseline', 'retrieval', 'generic', 'targeted', 'generic_twice',
             'source_confirmation', 'verify_edits')
EVALUATED = GENERATED[:-1] + ('quote_gate', 'semantic_gate_3',
                              'semantic_gate_4', 'semantic_gate_5',
                              'verifier_quote_gate_4', 'random_judgment_gate_4',
                              'anchored_quote_gate', 'string_semantic_gate_4')
LEGACY_DEFINITION = '''Extract complete scientific measurement groups between GIVEN
entity spans. CN is chemical/material name, PN property name, PV property value,
Condition the explicitly associated measurement/test/processing condition.
Every group requires CN, PN and PV; Condition is empty only when no provided
condition is linked to that group. Preserve multiple material/value/condition
bindings and ranges. Do not transfer a condition to another experiment, material,
value or property just because it is nearby. Do not invent missing values.
Use only given identifiers with their given roles. ES has no output role.
Output a single JSON object with a relations list. Each group has CN, PN, PV and
Condition lists of given identifier strings, and evidence copied from SOURCE.
Use compact JSON without indentation. Evidence must be actual source words,
never a schema placeholder. Do not enumerate unsupported Cartesian products.
Role lists may contain multiple identifiers when they form ONE complete released
measurement group, rather than an unrelated Cartesian product. Quote every
endpoint surface mention; return all groups and [] when there are none.
INVENTED FORMAT EXAMPLE ONLY (never output its identifiers):
Source: Material A has modulus 12 MPa at 300 K.
Given entities: X1 CN Material A; X2 PN modulus; X3 PV 12 MPa; X4 Condition 300 K.
Output: {"relations":[{"CN":["X1"],"PN":["X2"],"PV":["X3"],"Condition":["X4"],"evidence":"Material A has modulus 12 MPa at 300 K."}]}'''

TASK_SCHEMA = '''Extract mention-level POLYIE relation groups using only the supplied
entity IDs. Each group records ONE property-name mention linked to ONE property-
value mention for the material or material system actually described, plus only
the condition mention(s) that qualify that value. A group is a specific property-
value binding, not a bag of every entity in an experiment or paragraph.
Use exactly one PN ID and one PV ID in each complete group. Keep separate groups
for different property-value bindings, including lists aligned by respectively
and distinct values under distinct conditions. The same CN, PN, PV or Condition
ID may be reused across separate groups when the source supports it. Do not merge
groups merely because they share an endpoint.
CN: select the provided mention(s) identifying the material/system of this binding.
A blend/composite name annotated as one CN remains that single ID. Multiple CN IDs
are justified only when the source requires those mentions jointly for the same
material-system binding, not because names co-occur or refer to the same chemical.
Condition: attach only supplied condition mention(s) constraining this particular
value. Do not inherit all nearby conditions or combine all values with all conditions.
Use an empty Condition list when no supplied condition is supported.
Preserve original IDs and occurrences. Identical words at different spans are
different mentions. Do not canonicalize all occurrences to the first mention or
add all aliases to a group. Deduplicate only identical role-ID sets, not groups
that merely have identical words or values. Given entity rows have columns
[ID, type, absolute_token_start, absolute_token_end, surface_text, local_context].
Token spans are half open. The context contains up to two source tokens before
and after that occurrence; it is not an extra annotation.
Cross-sentence binding is allowed when supported by SOURCE. Resolve the actual
material-property-value-condition alignment; nearness or a quote containing all
names is insufficient. Do not invent IDs, split supplied entities, add missing
units or infer unreported values. Supplied PV may be a bare number or qualitative
value. Return no relation for an unsupported candidate. ES has no output role.'''

DEFINITION = (TASK_SCHEMA + '''
Output one compact JSON object with a relations list. Each group has CN, PN, PV,
Condition lists of identifier strings and evidence copied verbatim from SOURCE.
Quote every endpoint occurrence. Return all supported groups and [] when none.
Evidence must contain actual source words, never a schema placeholder.
INVENTED FORMAT EXAMPLE ONLY (never output its identifiers):
Source: Material A has modulus 12 MPa at 300 K.
Given entities: X1 CN Material A; X2 PN modulus; X3 PV 12 MPa; X4 Condition 300 K.
Output: {"relations":[{"CN":["X1"],"PN":["X2"],"PV":["X3"],"Condition":["X4"],"evidence":"Material A has modulus 12 MPa at 300 K."}]}'''
              if VERSION.startswith('development_v3') else LEGACY_DEFINITION)


def prompt_entities(inp):
    """Only source-derived occurrence context; never inspect target relations."""
    if not VERSION.startswith('development_v3'):
        return inp['entities']
    tokens = inp['tokens']
    start = inp['start']
    visible = []
    for entity in inp['entities']:
        if entity['type'] not in ROLES:
            continue
        a, b = entity['start'] - start, entity['end'] - start
        assert 0 <= a < b <= len(tokens)
        visible.append([entity['id'], entity['type'], entity['start'], entity['end'],
                        entity['name'], ' '.join(tokens[max(0, a-2):min(len(tokens), b+2)])])
    return visible


def groups_for_prompt(records):
    # Preserve all endpoint choices, including invalid ones, for fair review.
    # Full parent evidence remains in raw caches; SOURCE is shared by all arms.
    return [{r: record.get(r) for r in ROLES} if isinstance(record, dict) else record
            for record in records] if VERSION.startswith('development_v3') else records


def prompt_json(value):
    return json.dumps(value, ensure_ascii=False, separators=(',', ':'))


def digest(value):
    return hashlib.sha256(value if isinstance(value, bytes) else str(value).encode()).hexdigest()


def clean(value):
    return ' '.join(str(value).casefold().split())


def entity_map(inp):
    return {str(e['id']): e for e in inp['entities']}


def canonical(record, ents=None):
    """Invalid outputs remain false-positive records, never filtered away."""
    if not isinstance(record, dict):
        return ('INVALID', json.dumps(record, sort_keys=True, ensure_ascii=False))
    fields = []
    for role in ROLES:
        value = record.get(role)
        if not isinstance(value, list) or any(not isinstance(x, (str, int)) for x in value):
            return ('INVALID', json.dumps(record, sort_keys=True, ensure_ascii=False))
        ids = tuple(sorted(set(map(str, value))))
        if role != 'Condition' and not ids:
            return ('INVALID', json.dumps(record, sort_keys=True, ensure_ascii=False))
        if ents is not None and any(i not in ents or ents[i]['type'] != role for i in ids):
            return ('INVALID', json.dumps(record, sort_keys=True, ensure_ascii=False))
        fields.append(ids)
    return tuple(fields)


def valid_key(k):
    return len(k) == 4 and all(isinstance(x, tuple) for x in k)


def gold_record(ids, ents):
    out = {r: [] for r in ROLES}
    invalid = []
    for i in ids:
        i = str(i)
        if i not in ents or ents[i]['type'] not in ROLES:
            invalid.append(i)
        else:
            out[ents[i]['type']].append(i)
    out={role:sorted(set(values)) for role,values in out.items()}
    k = canonical(out, ents)
    if invalid or not valid_key(k):
        return None, {'ids': list(map(str, ids)), 'unsupported_ids': invalid,
                      'missing_roles': [r for r in ROLES[:3] if not out[r]]}
    return out, None


def prepare():
    audit = {'window_width_tokens': WIDTH, 'window_stride_tokens': STRIDE,
             'source_boundary_rule': 'token-only; independent of relations',
             'splits': {}}
    training = []
    for split in ('train', 'dev', 'test'):
        source = DATA / 'source' / f'{split}.json'
        docs = read_json(source)
        windows, labels, doc_inputs = [], {}, []
        counts = Counter()
        anomalies = []
        ids_seen = set()
        for doc in docs:
            doc_id = str(doc['id'])
            assert doc_id not in ids_seen, 'duplicate source document ID'
            ids_seen.add(doc_id)
            tokens = doc['text']
            assert isinstance(tokens, list) and all(isinstance(t, str) for t in tokens)
            ents = []
            for e in doc['entities']:
                a, b = int(e['start_offset']), int(e['end_offset'])
                assert 0 <= a < b <= len(tokens), ('invalid span', doc_id, e['id'])
                ents.append({'id': str(e['id']), 'type': e['label'],
                             'start': a, 'end': b, 'name': ' '.join(tokens[a:b])})
            em = {e['id']: e for e in ents}
            assert len(em) == len(ents), 'duplicate entity ID'
            dw = []
            for start in range(0, len(tokens), STRIDE):
                end = min(start + WIDTH, len(tokens))
                visible = [e for e in ents if e['start'] >= start and e['end'] <= end]
                inp = {'doc_key': doc_id, 'window_id': f'{doc_id}|{start}:{end}',
                       'start': start, 'end': end,
                       'text': ' '.join(tokens[start:end]), 'tokens': tokens[start:end],
                       'entities': visible}
                inp['call_needed'] = all(any(e['type'] == role for e in visible) for role in ROLES[:3])
                dw.append(inp)
            all_valid, eligible, invalid = {}, {}, []
            for ids in doc['relations']:
                counts['released_groups'] += 1
                rec, bad = gold_record(ids, em)
                if bad is not None:
                    bad['doc_key'] = doc_id
                    invalid.append(bad)
                    anomalies.append(bad)
                    counts['schema_excluded'] += 1
                    continue
                k = canonical(rec, em)
                all_valid[k] = rec
                endpoints = set(i for role in ROLES for i in rec[role])
                if any(endpoints.issubset(entity_map(w)) for w in dw):
                    eligible[k] = rec
            counts['valid_unique_groups'] += len(all_valid)
            counts['covered_unique_groups'] += len(eligible)
            counts['uncovered_unique_groups'] += len(all_valid) - len(eligible)
            counts['condition_groups_covered'] += sum(bool(k[3]) for k in eligible)
            labels[doc_id] = {'eligible': list(eligible.values()),
                              'all_valid': list(all_valid.values()),
                              'schema_excluded': invalid}
            doc_inputs.append({'doc_key': doc_id, 'entities': ents})
            windows.extend(dw)
            if split == 'train':
                for w in dw:
                    visible = entity_map(w)
                    targets = [r for k, r in all_valid.items()
                               if all(i in visible for role in ROLES for i in r[role])]
                    quoted_targets = []
                    for r in targets:
                        ids = [i for role in ROLES for i in r[role]]
                        a = min(visible[i]['start'] for i in ids) - w['start']
                        b = max(visible[i]['end'] for i in ids) - w['start']
                        quoted_targets.append(dict(r, evidence=' '.join(w['tokens'][a:b])))
                    if w['call_needed']:
                        training.append({'input': w, 'relations': quoted_targets})
        counts['documents'] = len(docs)
        counts['windows'] = len(windows)
        counts['paid_call_windows_per_stage'] = sum(w['call_needed'] for w in windows)
        write_json(DATA / f'{split}_inputs.json', windows)
        write_json(DATA / f'{split}_documents.json', doc_inputs)
        write_json(DATA / f'{split}_gold.json', labels)
        audit['splits'][split] = {'source_sha256': digest(source.read_bytes()),
                                  'counts': dict(counts), 'schema_anomalies': anomalies,
                                  'document_ids': sorted(ids_seen)}
    # Cross-split IDs and exact source-text duplication are checked separately.
    ids = {s: set(audit['splits'][s]['document_ids']) for s in audit['splits']}
    audit['cross_split_doc_id_overlap'] = {a + '/' + b: sorted(ids[a] & ids[b])
        for a, b in [('train', 'dev'), ('train', 'test'), ('dev', 'test')]}
    texts = {s: {digest(' '.join(d['text'])) for d in read_json(DATA / 'source' / f'{s}.json')}
             for s in ids}
    audit['cross_split_exact_text_overlap'] = {a + '/' + b: len(texts[a] & texts[b])
        for a, b in [('train', 'dev'), ('train', 'test'), ('dev', 'test')]}
    assert not any(audit['cross_split_doc_id_overlap'].values())
    assert not any(audit['cross_split_exact_text_overlap'].values())
    write_json(DATA / 'training_examples.json', training)
    write_json(DATA / 'scope_audit.json', audit)
    print(json.dumps({s: audit['splits'][s]['counts'] for s in ids}, indent=2), flush=True)


class Retriever:
    def __init__(self):
        self.bank = read_json(DATA / 'training_examples.json')
        self.vec = TfidfVectorizer(lowercase=True, token_pattern=r'(?u)\b\w\w+\b')
        self.matrix = self.vec.fit_transform(x['input']['text'] for x in self.bank)

    def get(self, inp):
        scores = (self.matrix @ self.vec.transform([inp['text']]).T).toarray().ravel()
        picked, seen = [], set()
        for i in np.argsort(-scores, kind='stable'):
            item = self.bank[int(i)]
            doc = item['input']['doc_key']
            if doc in seen:
                continue
            seen.add(doc)
            picked.append({'text': item['input']['text'],
                           'entities': prompt_entities(item['input']),
                           'relations': groups_for_prompt(item['relations'])})
            if len(picked) == 2:
                break
        return picked


def extraction_prompt(inp, examples=None):
    assert not any(k in inp for k in ['gold', 'relations', 'eligible'])
    p = DEFINITION + '\nSOURCE AND GIVEN ENTITIES:\n' + prompt_json(
        {'text': inp['text'], 'entities': prompt_entities(inp)})
    if examples is not None:
        p += '\nTRAINING EXAMPLES FROM OTHER PAPERS:\n' + prompt_json(examples)
    return p


def path_for(model, split, stage, inp, repeat):
    return ROOT / 'results/polyie' / ('cache_' + VERSION) / model.replace('/', '__') / split / f'repeat{repeat}' / stage / (digest(inp['window_id'])[:20] + '.json')


def quote_valid(inp, record, anchored=False):
    em = entity_map(inp)
    k = canonical(record, em)
    if not valid_key(k):
        return False
    q = clean(record.get('evidence', ''))
    literal = bool(q) and q in clean(inp['text']) and all(
        clean(em[i]['name']) in q for ids in k for i in ids)
    if not literal or not anchored:
        return literal
    # Map normalized quote occurrences to the actual supplied span offsets.
    # Same surface names at another position cannot validate these identifiers.
    bounds, chunks, cursor = [], [], 0
    for token in inp['tokens']:
        text = clean(token)
        if text and chunks:
            cursor += 1
        bounds.append((cursor, cursor + len(text)))
        if text:
            chunks.append(text)
            cursor += len(text)
    source = ' '.join(chunks)
    start = source.find(q)
    while start >= 0:
        end = start + len(q)
        if all(start <= bounds[em[i]['start'] - inp['start']][0] and
               bounds[em[i]['end'] - inp['start'] - 1][1] <= end
               for ids in k for i in ids):
            return True
        start = source.find(q, start + 1)
    return False


def edits_for(inp, old, proposal):
    em = entity_map(inp)
    a = {canonical(r, em): r for r in old['relations']}
    b = {canonical(r, em): r for r in proposal['relations']}
    edits = []
    for operation, keys, records in [('add', b.keys() - a.keys(), b), ('delete', a.keys() - b.keys(), a)]:
        for k in sorted(keys, key=repr):
            edits.append({'edit_id': digest(operation + repr(k))[:16],
                          'operation': operation, 'group': records[k]})
    return edits


def quote_gate(inp, old, proposal, anchored=False):
    em = entity_map(inp)
    a = {canonical(r, em): r for r in old['relations']}
    for k in list(a):
        if not valid_key(k):
            del a[k]
    for r in proposal['relations']:
        if quote_valid(inp, r, anchored):
            a[canonical(r, em)] = r
    return {'relations': list(a.values()), 'offline_gate': 'quote_only'}


def semantic_gate(inp, old, proposal, verified, threshold, mode='semantic', anchored=False):
    em = entity_map(inp)
    accepted = {canonical(r, em): r for r in old['relations']}
    edits = {e['edit_id']: e for e in edits_for(inp, old, proposal)}
    applied = []
    if verified.get('error'):
        return {'relations': old['relations'], 'fallback_to_parent': True,
                'verification_error': verified['error']}
    # Duplicate verifier answers are ambiguous and therefore abstained.
    frequency = Counter(v.get('edit_id') for v in verified['relations'] if isinstance(v, dict))
    for v in verified['relations']:
        if not isinstance(v, dict) or frequency[v.get('edit_id')] != 1:
            continue
        e = edits.get(v.get('edit_id'))
        confidence = v.get('confidence')
        if e is None or type(confidence) is not int or not threshold <= confidence <= 5:
            continue
        rec = e['group']
        if not isinstance(rec, dict):
            continue
        quoted = dict(rec, evidence=v.get('evidence', ''))
        if not quote_valid(inp, quoted, anchored):
            continue
        k = canonical(rec, em)
        judgment = v.get('judgment')
        if mode == 'random_judgment':
            judgment = ('supported' if int(digest('polyie-negative-control-20261003|' +
                         inp['window_id'] + '|' + e['edit_id']), 16) % 2 else 'unsupported')
        if e['operation'] == 'add' and (judgment == 'supported' or mode == 'verifier_quote'):
            accepted[k] = quoted
            applied.append(e['edit_id'])
        elif e['operation'] == 'delete' and judgment == 'unsupported' and mode != 'verifier_quote':
            accepted.pop(k, None)
            applied.append(e['edit_id'])
    return {'relations': list(accepted.values()), 'offline_gate': 'semantic',
            'confidence_threshold': threshold, 'applied_edit_ids': applied}


def run_stage(model, split, stage, workers, repeat, subset=0):
    assert split == 'dev', 'Test generation requires a separately frozen holdout implementation.'
    assert VERSION.startswith('development_v3'), 'Legacy generation uses its archived exact source; current engine is development_v3.'
    attempt = begin_attempt('polyie_' + VERSION + '_' + stage + f'_repeat{repeat}', model, split)
    inputs = read_json(DATA / f'{split}_inputs.json')
    if subset:
        ids = sorted({w['doc_key'] for w in inputs}, key=lambda x: digest('polyie-feasibility|' + x))[:subset]
        inputs = [w for w in inputs if w['doc_key'] in ids]
    retriever = Retriever() if stage != 'baseline' else None
    def work(inp):
        path = path_for(model, split, stage, inp, repeat)
        if path.exists():
            return read_json(path)
        if not inp['call_needed']:
            out = {'relations': [], 'source_only_empty_role_shortcut': True}
            write_json(path, out)
            return out
        def parent(s):
            p = path_for(model, split, s, inp, repeat)
            if not p.exists():
                raise RuntimeError('Missing parent cache: ' + str(p))
            return read_json(p)
        old = None
        p = extraction_prompt(inp, retriever.get(inp) if retriever else None)
        if stage in ('baseline', 'retrieval'):
            pass
        elif stage in ('generic', 'targeted'):
            old = parent('retrieval')
            p += '\nCURRENT COMPLETE GROUPS:\n' + prompt_json(groups_for_prompt(old['relations']))
            if stage == 'generic':
                p += '\nReview carefully for incorrect groups and omissions. Return a COMPLETE corrected list with exact evidence.'
            else:
                p += ('\nAudit each complete material/property/value/condition binding against the source. '
                      'Check whether conditions qualify this exact measurement or a different experiment. '
                      'Check all provided materials and values, range endpoints, negations and contrasts. '
                      'Preserve correct groups; remove unsupported bindings; recover explicitly missing groups. '
                      'Return a COMPLETE corrected list with exact evidence.')
        elif stage in ('generic_twice', 'source_confirmation'):
            old = parent('generic' if stage == 'generic_twice' else 'targeted')
            p += '\nCURRENT COMPLETE GROUPS:\n' + prompt_json(groups_for_prompt(old['relations']))
            if stage == 'generic_twice':
                p += '\nReview carefully again for incorrect groups and omissions. Return a COMPLETE corrected list with exact evidence.'
            else:
                p += ('\nPerform final source confirmation of every complete group. '
                      'Check each role and the scope of every condition, distinguish separate experiments, '
                      'verify quotes and recover omissions. Return a COMPLETE corrected list with exact evidence.')
        elif stage == 'verify_edits':
            old, proposed = parent('retrieval'), parent('targeted')
            edits = edits_for(inp, old, proposed)
            if not edits:
                out = {'relations': [], 'no_proposed_edits': True}
                write_json(path, out)
                return out
            p = (TASK_SCHEMA + '\nJudge candidate edits to scientific measurement groups using ONLY the source and GIVEN entities. '
                 'Evaluate the entire CN/PN/PV/Condition binding, including absence of conditions. '
                 'A quote containing all names is necessary but does not by itself prove their binding. '
                 'For BOTH operations the judgment describes factual support of the supplied GROUP, '
                 'not whether you approve the edit. A supported addition can be accepted; a supported '
                 'old group must be preserved. For a deletion, judge the OLD full binding; delete only when the '
                 'source establishes wrong association or a contradicting binding, not just uncertainty. '
                 'Use supported, unsupported, or uncertain. Give an ordinal confidence integer 1 to 5: '
                 '1 guess, 2 weak, 3 plausible, 4 strong explicit evidence, 5 unambiguous explicit evidence. '
                 'Return compact JSON with a relations list. Each judgment object has edit_id, '
                 'judgment (supported, unsupported or uncertain), confidence (integer 1 through 5), '
                 'and evidence (a string of actual copied SOURCE words covering every group endpoint occurrence). '
                 'Never put instructions or placeholder text in evidence. '
                 'Return each edit ID exactly once. Do not modify candidate groups or invent new ones.\nSOURCE:\n'
                 + prompt_json({'text': inp['text'], 'entities': prompt_entities(inp)})
                 + '\nTRAINING EXAMPLES FROM OTHER PAPERS:\n' + prompt_json(retriever.get(inp))
                 + '\nCANDIDATE EDITS:\n' + prompt_json([
                     dict(e, group=groups_for_prompt([e['group']])[0]) for e in edits]))
        else:
            raise ValueError(stage)
        out = call(model, p, path)
        if out.get('error') and old is not None and stage != 'verify_edits':
            out['relations'] = old['relations']
            out['fallback_to_parent'] = True
            write_json(path, out)
        return out
    failed = 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(work, x) for x in inputs]
        for n, f in enumerate(as_completed(futures), 1):
            out = f.result()
            failed += bool(out.get('error'))
            if n % 10 == 0 or n == len(inputs):
                print(model, split, stage, f'{n}/{len(inputs)}', f'failed={failed}', flush=True)
    append(ROOT / 'logs/attempts.jsonl', {'event': 'complete', 'attempt': attempt,
                                         'failed_windows': failed, 'repeat': repeat})
    return failed


def collect(model, split, stage, inputs, repeat):
    records = defaultdict(dict)
    failures = 0
    for inp in inputs:
        def get(s):
            return read_json(path_for(model, split, s, inp, repeat))
        if stage == 'quote_gate':
            out = quote_gate(inp, get('retrieval'), get('targeted'))
        elif stage == 'anchored_quote_gate':
            out = quote_gate(inp, get('retrieval'), get('targeted'), True)
        elif stage == 'string_semantic_gate_4':
            out = semantic_gate(inp, get('retrieval'), get('targeted'), get('verify_edits'), 4)
        elif stage in ('verifier_quote_gate_4', 'random_judgment_gate_4'):
            mode = 'verifier_quote' if stage.startswith('verifier_quote') else 'random_judgment'
            out = semantic_gate(inp, get('retrieval'), get('targeted'),
                                get('verify_edits'), 4, mode, True)
        elif stage.startswith('semantic_gate_'):
            out = semantic_gate(inp, get('retrieval'), get('targeted'),
                                get('verify_edits'), int(stage[-1]), anchored=True)
        else:
            out = get(stage)
        failures += bool(out.get('error') or out.get('verification_error'))
        for r in out['relations']:
            k = canonical(r, entity_map(inp))
            records[inp['doc_key']][k] = r
    return records, failures


def counts_for(pred, gold):
    return [len(pred & gold), len(pred - gold), len(gold - pred)]


def evaluate(model, split, repeat, subset=0):
    assert split == 'dev', 'No unfrozen test evaluation.'
    inputs = read_json(DATA / f'{split}_inputs.json')
    if subset:
        ids = sorted({w['doc_key'] for w in inputs}, key=lambda x: digest('polyie-feasibility|' + x))[:subset]
        inputs = [w for w in inputs if w['doc_key'] in ids]
    ids = sorted({w['doc_key'] for w in inputs})
    labels = read_json(DATA / f'{split}_gold.json')
    docs = {d['doc_key']: d for d in read_json(DATA / f'{split}_documents.json')}
    result = {'model': model, 'split': split, 'repeat': repeat, 'protocol_version': VERSION,
              'n_documents': len(ids), 'document_ids': ids,
              'analysis_status': 'exploratory_development', 'stages': {}}
    sets, counts = {}, {}
    for stage in EVALUATED:
        predicted, failed = collect(model, split, stage, inputs, repeat)
        cells, ccells, corecells, fullcells, releasedcells = [], [], [], [], []
        ps, gs, binding = [], [], Counter()
        for doc in ids:
            em = entity_map(docs[doc])
            p = set(predicted[doc])
            g = {canonical(r, em) for r in labels[doc]['eligible']}
            ga = {canonical(r, em) for r in labels[doc]['all_valid']}
            # Invalid gold gets an unreachable unique key; it remains a FN.
            gr = ga | {('MALFORMED_RELEASED_GOLD', json.dumps(r, sort_keys=True))
                       for r in labels[doc]['schema_excluded']}
            cells.append(counts_for(p, g))
            fullcells.append(counts_for(p, ga))
            releasedcells.append(counts_for(p, gr))
            cp = {k for k in p if valid_key(k) and k[3]}
            # Malformed claimed condition records are also condition false positives.
            cp |= {k for k in p if not valid_key(k) and isinstance(predicted[doc][k], dict)
                   and predicted[doc][k].get('Condition')}
            cg = {k for k in g if k[3]}
            ccells.append(counts_for(cp, cg))
            project = lambda z: {k[:3] if valid_key(k) else k for k in z}
            corecells.append(counts_for(project(p), project(g)))
            gold_cores = {k[:3] for k in g}
            for k in p:
                if valid_key(k) and k[:3] in gold_cores:
                    binding['predicted_groups_with_correct_core'] += 1
                    binding['full_binding_correct'] += k in g
            ps.append(p)
            gs.append(g)
        sets[stage] = ps
        counts[stage] = np.array(cells)
        m = prf(counts[stage].sum(axis=0))
        m.update({'condition_bearing': prf(np.array(ccells).sum(axis=0)),
                  'core_projection': prf(np.array(corecells).sum(axis=0)),
                  'full_valid_document_sensitivity': prf(np.array(fullcells).sum(axis=0)),
                  'full_released_sensitivity': prf(np.array(releasedcells).sum(axis=0)),
                  'condition_binding_given_correct_core': dict(binding),
                  'failed_windows': failed})
        if stage not in ('baseline', 'retrieval'):
            edit = Counter()
            for old, new, g in zip(sets['retrieval'], ps, gs):
                edit.update({'correct_deleted': len((old - new) & g),
                             'wrong_deleted': len((old - new) - g),
                             'correct_added': len((new - old) & g),
                             'wrong_added': len((new - old) - g)})
            m['edits_vs_retrieval'] = dict(edit)
        rng = np.random.default_rng(20261003)
        b = [prf(counts[stage][rng.integers(0, len(ids), len(ids))].sum(axis=0))['f1']
             for _ in range(2000)]
        m['descriptive_document_bootstrap_f1_ci95'] = np.quantile(b, [.025, .975]).tolist()
        result['stages'][stage] = m
    thresholds = [3, 4, 5]
    def choice(t):
        v = result['stages'][f'semantic_gate_{t}']
        e = v['edits_vs_retrieval']
        return (v['f1'], -(e['wrong_added'] + e['correct_deleted']),
                -sum(e.values()), t)
    result['development_selected_threshold'] = max(thresholds, key=choice)
    result['per_document_counts'] = {s: {d: c.tolist() for d, c in zip(ids, cells)}
                                     for s, cells in counts.items()}
    output = ROOT / 'results/polyie' / (model.replace('/', '__') + '__' + VERSION + f'__dev_repeat{repeat}' + (f'_subset{subset}' if subset else '') + '_summary.json')
    write_json(output, result)
    print(json.dumps({'summary': str(output.relative_to(ROOT)),
                      'development_selected_threshold': result['development_selected_threshold'],
                      'stages': {s: {'f1': v['f1'], 'condition_f1': v['condition_bearing']['f1'],
                                     'edits': v.get('edits_vs_retrieval')}
                                 for s, v in result['stages'].items()}}, indent=2), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('action', choices=['prepare', 'run', 'evaluate'])
    ap.add_argument('--model', default='Qwen/Qwen3-8B')
    ap.add_argument('--split', choices=['dev'], default='dev')
    ap.add_argument('--stages', default=','.join(GENERATED))
    ap.add_argument('--workers', type=int, default=6)
    ap.add_argument('--repeat', type=int, default=0)
    ap.add_argument('--subset', type=int, default=0)
    ap.add_argument('--halt-on-failure', action='store_true',
                    help='Development feasibility only: halt later stages after failed windows, retaining all outputs.')
    a = ap.parse_args()
    if a.action == 'prepare':
        prepare()
    elif a.action == 'run':
        for stage in a.stages.split(','):
            failed=run_stage(a.model, a.split, stage, a.workers, a.repeat, a.subset)
            if failed and a.halt_on_failure:
                raise SystemExit('Feasibility stopped before later stages; failed outputs retained, no method comparison.')
    else:
        evaluate(a.model, a.split, a.repeat, a.subset)


if __name__ == '__main__':
    main()
