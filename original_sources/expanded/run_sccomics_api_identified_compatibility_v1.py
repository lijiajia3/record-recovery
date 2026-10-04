"""Root-only identified INVENTED six-slot compatibility; no corpus authority.

Default refuses before authority/source/key I/O. The distinct closed technical
authority and different-agent source proof must precede actual paid requests.
No source-manifest, annotation, model, scorer or experimental metric is opened.
An empty targeted critic deliberately leaves identical T/S payloads: each has
its own stage intent, ledger and real request, never reuse T as S.
"""
from __future__ import annotations
import argparse
import contextlib
import importlib.util
from pathlib import Path
import sys
import time
import types

SELF = 'src/run_sccomics_api_identified_compatibility_v1.py'
CONTROLLER = 'src/sccomics_api_feedback_pipeline_v3.py'
TECH_ROOT = 'research/round4_sccomics_api_identified_compatibility_v1'
AUTHORITY = TECH_ROOT + '/before_call_authority.json'
SOURCE = TECH_ROOT + '/INPUTS/source.txt'
DOC_ID = 1000001
REPEAT = 0
PURPOSE = 'SC_NATIVE_IDENTIFIED_INVENTED_6_SLOT_COMPATIBILITY'

def implementation():
    # Source loader only; no corpus/network/credential read on import.
    path = Path(__file__).absolute().parent / 'sccomics_api_feedback_pipeline_v3.py'
    module = types.ModuleType('_IDENTIFIED_API_SOURCE_ROUTER'); module.__file__ = str(path)
    exec(compile(path.read_bytes(), str(path), 'exec'), module.__dict__)
    return module

def proposed_authority(source_binding, source_pins, transport_pins, review, run_name):
    """Inert proposal only, root must bind/review/freeze before actual use."""
    a = implementation()
    intents = [{'study_id': 'SC_NATIVE_API_FEEDBACK_V1', 'split': 'invented', 'document_id': DOC_ID,
                'model': m, 'repeat': REPEAT, 'stage': s} for m in a.MODELS for s in a.SLOTS]
    return {'version': 1, 'purpose': PURPOSE, 'state': 'PROPOSED_NOT_AUTHORITY',
            'technical_request_authority': False, 'scientific_request_authority': False,
            'source': {'path': SOURCE, **source_binding}, 'source_sha256_pins': source_pins,
            'transport_source_sha256_pins': transport_pins, 'closed_independent_source_review': review,
            'allowed_intents': intents, 'run_name': run_name,
            'model_payload_profiles': {a.MODELS[0]: {}, a.MODELS[1]: {'reasoning_effort': 'low'}},
            'limits': dict(a.LIMITS), 'retry_policy': dict(a.RETRY_POLICY),
            'corpus_or_Gold_or_performance_authority': False}

def validate(a, files, authority_raw, execute_actual, invented):
    p = a.unique(authority_raw)
    a.closed(p, proposed_authority({}, {}, {}, {}, '').keys())
    a.require(type(p['version']) is int and p['version'] == 1 and p['purpose'] == PURPOSE,
              'identified technical purpose')
    a.require(p['state'] == ('INVENTED_TECH_SOURCE_FIXTURE' if invented else 'FROZEN_BEFORE_IDENTIFIED_INVENTED_API') and
              p['technical_request_authority'] is (not invented) and p['scientific_request_authority'] is False and
              p['corpus_or_Gold_or_performance_authority'] is False, 'technical only; no corpus/Gold')
    run = p['run_name']; a.relative(run)
    a.require('/' not in run and run.startswith('ACTUAL_') and len(run) <= 80, 'distinct owned technical run name')
    source = p['source']; a.closed(source, ('path', 'sha256', 'size_bytes'))
    a.require(source['path'] == SOURCE, 'fixed owned invented source file')
    raw = files.read(SOURCE, {k: source[k] for k in ('sha256', 'size_bytes')})
    a.require(raw.startswith(b'INVENTED_'), 'explicit invented synthetic input, no corpus')
    raw.decode('utf8', 'strict')
    wanted = proposed_authority({k: source[k] for k in ('sha256', 'size_bytes')},
                               p['source_sha256_pins'], p['transport_source_sha256_pins'],
                               p['closed_independent_source_review'], run)
    for key in ('model_payload_profiles', 'limits', 'retry_policy', 'allowed_intents'):
        a.require(a.encoded(p[key]) == a.encoded(wanted[key]), 'exact frozen technical population/profile/' + key)
    a.require(type(p['source_sha256_pins']) is dict and set(p['source_sha256_pins']) == {*a.SOURCES, SELF}, 'exact nine source roles')
    captures = {}
    for name, sha in p['source_sha256_pins'].items():
        a.sha(sha); b = files.read(name); a.require(a.digest(b) == sha, 'actual source identity'); captures[name] = b
    a.require(p['transport_source_sha256_pins'] == {name: p['source_sha256_pins'][name] for name in a.TRANSPORT_SOURCES}, 'same v3 four transport pins')
    row = p['closed_independent_source_review']; a.closed(row, ('path', 'sha256', 'size_bytes'))
    a.relative(row['path']); a.require(row['path'].startswith('research/') and row['path'].endswith('.json'), 'SOURCE review role')
    review = a.unique(files.read(row['path'], {k: row[k] for k in ('sha256', 'size_bytes')}))
    a.require(review.get('status') == 'source_review_passed' and review.get('independent_from_implementer') is True and
              review.get('reviewed_source_sha256_pins') == p['source_sha256_pins'], 'closed independently checked current nine-source proof')
    return p, raw, captures

def run(root, authority_bytes, *, execute_actual=False, adapter=None):
    a = implementation(); invented = adapter is not None
    a.require((invented and not execute_actual) or (not invented and execute_actual), 'default refuses paid/source I/O')
    files = a.Files(root)
    if invented:
        a.require(files.root.name.startswith('INVENTED_'), 'owned fake root')
        a.closed(adapter, ('exchange', 'key_reader', 'pacer', 'sleep'))
        a.require(all(callable(adapter[k]) for k in ('exchange', 'key_reader', 'sleep')), 'no actual fake defaults')
    a.require(files.read(AUTHORITY) == authority_bytes, 'captured fixed before-call technical authority')
    p, source, captures = validate(a, files, authority_bytes, execute_actual, invented)
    prefix = TECH_ROOT + '/' + p['run_name']; authority_sha = a.digest(authority_bytes)
    before = {AUTHORITY: a.binding(authority_bytes), SOURCE: a.binding(source)}
    row = p['closed_independent_source_review']; before[row['path']] = {k: row[k] for k in ('sha256', 'size_bytes')}
    outputs = {}; summaries = []
    def put(name, obj):
        outputs[name] = files.obj(prefix + '/' + name, obj)
    def raw_put(name, raw): outputs[name] = files.put(prefix + '/' + name, raw)
    def recheck():
        for name, row in before.items(): files.read(name, row)
        for name, wanted in p['source_sha256_pins'].items():
            a.require(a.digest(files.read(name)) == wanted, 'technical source changed')
    with a.captured_modules({name: captures[name] for name in a.SOURCES}, root=files.root) as modules:
        t = modules['sccomics_api_transport_v1']; transport = modules['sccomics_api_scientific_transport_v3']
        tickets = modules['sccomics_api_feedback_tickets_v2']; compiler = modules['sccomics_api_native_projection_v1']
        try:
            for model in a.MODELS:
                base = f'units/{a.MODELS.index(model)}'; raw = {}; critics = {}; calls = {}
                for slot in a.slot_order(model, REPEAT, DOC_ID):
                    if slot == 'B': messages = tickets.build_baseline_messages(source)
                    elif slot.endswith('critic'): messages = tickets.build_critic_messages(source, raw['B'], slot[0])
                    else: messages = tickets.build_repair_messages(source, raw['B'], critics['G' if slot[0] == 'G' else 'T'],
                                slot[0], model=model, source_id=str(DOC_ID), repeat=REPEAT)
                    payload = t.serialized(tickets.payload_from_messages(model, messages, slot))
                    intent = {'study_id': 'SC_NATIVE_API_FEEDBACK_V1', 'protocol_sha256': authority_sha,
                              'split': 'invented', 'document_id': DOC_ID, 'model': model, 'repeat': REPEAT, 'stage': slot}
                    freeze = {'version': 1, 'purpose': 'identified_scientific_API_request', 'endpoint': t.ENDPOINT,
                              'protocol_sha256': authority_sha, 'intent': intent, 'payload_sha256': a.digest(payload),
                              'source_sha256': a.digest(source), 'source_sha256_pins': p['transport_source_sha256_pins'],
                              'limits': {**a.LIMITS, 'reserved_tokens': len(payload) + (2048 if slot.endswith('critic') else 8192)},
                              'root_authorized_scientific_request': not invented,
                              'request_mode': 'identified_invented_compatibility'}
                    freeze_raw = t.serialized(freeze); attempt = 0; attempts = []
                    while True:
                        destination = files.path(prefix + f'/{base}/calls/{slot}/attempts/{attempt:06}')
                        kwargs = {'execute_actual': not invented}
                        if invented: kwargs.update(exchange=adapter['exchange'], key_reader=adapter['key_reader'], pacer=adapter['pacer'])
                        result = transport.run_attempt(files.root, payload, source, freeze_raw, destination, attempt=attempt, **kwargs)
                        attempts.append({'attempt': attempt, 'intent_identity': result['binding']['intent_identity'],
                                         'passed': result['passed'], 'retryable': result['retryable_transport_failure'],
                                         'binding': result['binding'],
                                         'manifest': a.binding(files.read(destination.relative_to(files.root).as_posix() + '/output_manifest.json'))})
                        if result['passed']: break
                        if not result['retryable_transport_failure']: raise a.FamilyStopped('technical terminal slot: ' + slot)
                        wait_name = f'{base}/calls/{slot}/retry_wait_{attempt:06}.json'
                        if files.path(prefix + '/' + wait_name).exists():
                            wait = a.unique(files.read(prefix + '/' + wait_name)); calculated_at = wait['calculation_time_unix']
                        else: calculated_at = time.time()
                        delay = a.retry_delay(attempt, result, now=calculated_at)
                        put(wait_name, {'attempt': attempt, 'seconds': delay, 'calculation_time_unix': calculated_at,
                                       'same_scientific_identity': result['binding']['intent_identity']})
                        (adapter['sleep'] if invented else time.sleep)(delay); attempt += 1
                    raw[slot] = result['raw_content'].encode('utf8', 'strict')
                    raw_put(f'{base}/raw/{slot}.json', raw[slot]); put(f'{base}/calls/{slot}/completed.json',
                            {'intent': intent, 'payload_sha256': a.digest(payload), 'attempts': attempts, 'cache_is_additional_repeat': False})
                    calls[slot] = {'intent_identity': result['binding']['intent_identity'], 'payload_sha256': a.digest(payload), 'actual_API_attempt_started': result['actual_API_attempt_started']}
                    if slot.endswith('critic'):
                        artifact = tickets.compile_critic(source, raw['B'], raw[slot], slot[0]); put(f'{base}/critics/{slot[0]}.json', artifact.to_json())
                        artifact.require_feedback(); critics[slot[0]] = artifact
                    else:
                        compiled = compiler.compile_graph_json(source, raw[slot]); put(f'{base}/graphs/{slot[0]}.json', compiled.to_json()); compiled.require_document()
                for arm in ('T', 'S'): put(f'{base}/correspondence/{arm}.json', tickets.correspondence(critics['T'], arm, model=model, source_id=str(DOC_ID), repeat=REPEAT).to_json())
                put(f'{base}/T_S_identity_check.json', {'T_S_exact_shared_critic': True, 'T_critic_sha256': a.digest(raw['T_critic']),
                    'T_S_same_payload': calls['T_repair']['payload_sha256'] == calls['S_repair']['payload_sha256'],
                    'T_S_distinct_request_intents': calls['T_repair']['intent_identity'] != calls['S_repair']['intent_identity'],
                    'T_and_S_each_completed_slot': True, 'cached_T_is_NOT_S': True})
                summaries.append({'model': model, 'document_id': DOC_ID, 'repeat': REPEAT, 'calls': calls})
            recheck()
            for name, row in outputs.items(): files.read(prefix + '/' + name, row)
            # Technical wire artifacts are re-opened, not merely derived graphs.
            for model_summary in summaries:
                base = f"units/{a.MODELS.index(model_summary['model'])}"
                for slot in a.SLOTS:
                    complete = a.unique(files.read(prefix + f'/{base}/calls/{slot}/completed.json'))
                    for item in complete['attempts']:
                        destination = files.path(prefix + f"/{base}/calls/{slot}/attempts/{item['attempt']:06}")
                        files.read(destination.relative_to(files.root).as_posix() + '/output_manifest.json', item['manifest'])
                        observed = transport.reopen_attempt(destination, item['binding'])
                    a.require(observed['passed'] and observed['raw_content'].encode('utf8') ==
                              files.read(prefix + f'/{base}/raw/{slot}.json'), 'technical full wire raw output identity')
            summary = {'status': 'complete_identified_INVENTED_compatibility_not_corpus_or_performance',
                       'authority_sha256': authority_sha, 'logical_slots': 12, 'graphs': 8, 'critics': 4,
                       'models': summaries, 'derived_outputs': outputs, 'source_sha256_pins': p['source_sha256_pins'],
                       'ANN_Gold_NN_or_corpus_access': False, 'thinking_false_honored_not_certified': True}
            files.obj(prefix + '/COMPLETE.json', summary); recheck(); return summary
        except BaseException as error:
            files.obj(prefix + '/FAILED_' + str(time.time_ns()) + '.json', {'status': 'technical_failed_no_empty_baseline_or_corpus_result',
                      'error_type': type(error).__name__, 'error': str(error), 'completed_model_summaries': summaries})
            raise

def main():
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('--root', required=True)
    p.add_argument('--root-execute-frozen-identified-invented-compatibility', action='store_true'); args = p.parse_args()
    if not args.root_execute_frozen_identified_invented_compatibility: raise ValueError('default refuses paid/source I/O')
    a = implementation(); files = a.Files(args.root)
    result = run(files.root, files.read(AUTHORITY), execute_actual=True)
    print(a.encoded({'status': result['status'], 'logical_slots': result['logical_slots'], 'corpus_result': False}).decode())
    return 0

if __name__ == '__main__': raise SystemExit(main())
