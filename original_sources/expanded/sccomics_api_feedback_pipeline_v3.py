"""Frozen source-only API generation, not annotation loading or scoring.

Default CLI refuses. Actual root authority is external and must bind these
bytes, the source-only manifest and different-agent SOURCE reviews. The owned
INVENTED adapter executes the same complete populations with fake HTTP only.
The reviewed native compiler uses parse_emissions from a module containing
scoring definitions; this controller never calls those scoring functions.
"""
from __future__ import annotations
import argparse
import contextlib
import datetime
import email.utils
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import re
import stat
import sys
import time
import types

SELF = 'src/sccomics_api_feedback_pipeline_v3.py'
PROTOCOL = 'research/sccomics_api_feedback_v1/protocol_before_corpus.json'
SOURCE_MANIFEST = 'research/sccomics_api_feedback_v1/source_text_manifest.json'
TEST_GATE = 'research/sccomics_api_feedback_v1/test_source_generation_gate.json'
ACQUISITION = 'data/sccomics_round4/source_v3/source_acquisition_manifest.json'
OUTPUT = 'results/sccomics_api_feedback_v1'
TEXT_ROOT = 'data/sccomics_round4/source_v3/raw_text'
MODELS = ('deepseek-ai/DeepSeek-V4-Pro', 'zai-org/GLM-5.3')
REPEATS = (0, 1, 2)
IDS = {'development': tuple(range(101, 113)), 'test': tuple(range(1, 101))}
SLOTS = ('B', 'G_critic', 'G_repair', 'T_critic', 'T_repair', 'S_repair')
ARMS = ('B', 'G', 'T', 'S')
MODULES = ('siliconflow_pacing', 'siliconflow_fair_pacing',
           'sccomics_api_transport_v1', 'sccomics_api_scientific_transport_v3',
           'sccomics_native_graph', 'sccomics_api_native_projection_v1',
           'sccomics_api_feedback_tickets_v2')
SOURCES = (SELF, *('src/' + n + '.py' for n in MODULES))
TRANSPORT_SOURCES = ('src/sccomics_api_scientific_transport_v3.py',
                     'src/sccomics_api_transport_v1.py',
                     'src/siliconflow_pacing.py', 'src/siliconflow_fair_pacing.py')
LIMITS = {'idle_timeout_s': 120, 'wall_timeout_s': 1800,
          'max_response_bytes': 67108864, 'pacing_policy': 'shared_fair_conservative_v1'}
ORDER_SCHEME = 'SC_NATIVE_API_DEPENDENCY_READY_SHA256_v1'
RETRY_POLICY = {'retryable': 'completed_HTTP429_HTTP5xx_or_network_only',
                'attempt_limit': None, 'base_seconds': 5, 'max_backoff_seconds': 300,
                'honor_retry_after': True, 'same_scientific_identity': True}

class PipelineIntegrityError(ValueError): pass
class FamilyStopped(RuntimeError): pass

def require(ok, message):
    if not ok: raise PipelineIntegrityError(message)

def digest(raw): return hashlib.sha256(raw).hexdigest()

def finite(obj):
    stack = [obj]
    while stack:
        x = stack.pop()
        if type(x) is dict:
            require(all(type(k) is str for k in x), 'JSON string keys'); stack.extend(x.values())
        elif type(x) is list: stack.extend(x)
        elif type(x) is float: require(math.isfinite(x), 'finite JSON numbers')
        else: require(type(x) in (str, int, bool, type(None)), 'JSON primitives')
    return obj

def encoded(obj):
    finite(obj)
    return (json.dumps(obj, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + '\n').encode()

def unique(raw):
    def pairs(items):
        d = {}
        for k, v in items:
            require(k not in d, 'duplicate JSON key'); d[k] = v
        return d
    def bad(_): raise PipelineIntegrityError('nonfinite JSON constant')
    return finite(json.loads(raw.decode('utf8', 'strict'), object_pairs_hook=pairs, parse_constant=bad))

def closed(obj, keys): require(type(obj) is dict and set(obj) == set(keys), 'closed object schema')
def sha(value): require(type(value) is str and re.fullmatch('[0-9a-f]{64}', value) is not None, 'SHA256 type')
def binding(raw): return {'sha256': digest(raw), 'size_bytes': len(raw)}
def check_binding(row):
    closed(row, ('sha256', 'size_bytes')); sha(row['sha256'])
    require(type(row['size_bytes']) is int and row['size_bytes'] >= 0, 'byte size type')

def relative(name):
    require(type(name) is str and name and not name.startswith('/') and '\\' not in name,
            'root-relative path')
    p = Path(name)
    require(all(s not in ('', '.', '..') for s in name.split('/')) and p.as_posix() == name,
            'canonical relative path without traversal')
    return p

class Files:
    """Regular no-alias files only; no arbitrary directory or annotation glob."""
    def __init__(self, root):
        self.root = Path(root).absolute()
        require('..' not in Path(root).parts, 'root traversal')
        self._ancestors(self.root)
        require(self.root.is_dir(), 'existing root directory')
    def _ancestors(self, p):
        for q in (p, *p.parents):
            require(not q.is_symlink(), 'path or ancestor symbolic link')
    def path(self, name):
        p = self.root / relative(name); self._ancestors(p); return p
    def regular_metadata(self, name):
        """Check an exempt file's real regular identity without reading content."""
        p = self.path(name)
        before = p.stat(follow_symlinks=False)
        require(stat.S_ISREG(before.st_mode), 'only ordinary regular metadata may be exempt')
        fd = os.open(p, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
        try:
            opened = os.fstat(fd); now = p.stat(follow_symlinks=False)
            require(stat.S_ISREG(opened.st_mode), 'metadata opened regular file')
            identity = lambda s: (s.st_dev, s.st_ino, s.st_mode, s.st_size, s.st_mtime_ns)
            require(identity(before) == identity(opened) == identity(now), 'metadata identity changed while checked')
            self._ancestors(p)
        finally: os.close(fd)
    def read(self, name, wanted=None):
        p = self.path(name)
        fd = os.open(p, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
        try:
            st = os.fstat(fd); require(stat.S_ISREG(st.st_mode), 'regular file required')
            pieces = []
            while True:
                b = os.read(fd, 1048576)
                if not b: break
                pieces.append(b)
            end = os.fstat(fd); now = p.stat(follow_symlinks=False)
            require((st.st_dev, st.st_ino, st.st_size, st.st_mtime_ns) ==
                    (end.st_dev, end.st_ino, end.st_size, end.st_mtime_ns) ==
                    (now.st_dev, now.st_ino, now.st_size, now.st_mtime_ns), 'file changed while captured')
            self._ancestors(p); raw = b''.join(pieces)
            if wanted is not None:
                check_binding(wanted); require(binding(raw) == wanted, 'file SHA/size mismatch: ' + name)
            return raw
        finally: os.close(fd)
    def put(self, name, raw):
        require(type(raw) is bytes, 'raw bytes')
        p = self.path(name)
        if p.exists():
            require(self.read(name) == raw, 'immutable artifact changed: ' + name); return binding(raw)
        p.parent.mkdir(parents=True, exist_ok=True); self._ancestors(p)
        fd = os.open(p, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
        try:
            with os.fdopen(fd, 'wb') as f: f.write(raw); f.flush(); os.fsync(f.fileno())
        except BaseException: raise
        require(self.read(name) == raw, 'new artifact captured identity'); return binding(raw)
    def obj(self, name, obj): return self.put(name, encoded(obj))

@contextlib.contextmanager
def captured_modules(captures, *, root):
    """Execute the captured source closure, never stale scientific .pyc."""
    files = Files(root)
    saved = {n: sys.modules.get(n) for n in MODULES}
    try:
        result = {}
        for name in MODULES:
            mod = types.ModuleType(name); mod.__file__ = str(files.path('src/' + name + '.py'))
            sys.modules[name] = mod
            exec(compile(captures['src/' + name + '.py'], mod.__file__, 'exec'), mod.__dict__)
            result[name] = mod
        yield result
    finally:
        for name, old in saved.items():
            if old is None: sys.modules.pop(name, None)
            else: sys.modules[name] = old

def contract_template(source_pins, source_manifest_binding, reviews):
    """Return an inert proposal. It grants no request or corpus authority."""
    return {'version': 1, 'source_sha256_pins': source_pins,
            'source_manifest': {'path': SOURCE_MANIFEST, **source_manifest_binding},
            'independent_source_reviews': reviews, 'models': list(MODELS), 'repeats': list(REPEATS),
            'development_ids': list(IDS['development']), 'test_ids': list(IDS['test']),
            'slots': list(SLOTS), 'arms': list(ARMS), 'order_scheme': ORDER_SCHEME,
            'retry_policy': dict(RETRY_POLICY), 'transport_limits': dict(LIMITS),
            'compiler_policy': 'lossless_complete_native_invalid_occurrences_retained',
            'critic_policy': 'required_G_and_exact_shared_T_no_B_fallback',
            'graph_cap': None, 'Gold_or_scoring_authority': False,
            'non_scientific_OS_metadata': 'ignore_only_regular_nofollow_noalias_.DS_Store_filename',
            'model_payload_profiles': {MODELS[0]: {}, MODELS[1]: {'reasoning_effort': 'low'}}}

def validate_contract(c):
    closed(c, contract_template({}, {'sha256': '0' * 64, 'size_bytes': 0}, []).keys())
    wanted = contract_template(c['source_sha256_pins'],
                              {k: c['source_manifest'][k] for k in ('sha256', 'size_bytes')},
                              c['independent_source_reviews'])
    require(encoded(c) == encoded(wanted) and type(c['version']) is int and c['version'] == 1, 'exact fixed scientific controller contract')
    require(type(c['source_sha256_pins']) is dict and set(c['source_sha256_pins']) == set(SOURCES),
            'exact eight focused source pins')
    for x in c['source_sha256_pins'].values(): sha(x)
    closed(c['source_manifest'], ('path', 'sha256', 'size_bytes'))
    check_binding({k: c['source_manifest'][k] for k in ('sha256', 'size_bytes')})
    require(type(c['independent_source_reviews']) is list and len(c['independent_source_reviews']) >= 1,
            'different-agent SOURCE reviews required')

def source_manifest(files, raw, invented):
    m = unique(raw)
    closed(m, ('version', 'purpose', 'acquisition_manifest', 'text_members'))
    require(type(m['version']) is int and m['version'] == 1 and
            m['purpose'] == 'SC_API_SOURCE_TEXT_ONLY_1000_NATIVE_RECORDS', 'source manifest role')
    a = m['acquisition_manifest']; closed(a, ('path', 'sha256', 'size_bytes'))
    require(a['path'] == ACQUISITION, 'fixed acquisition metadata path')
    ar = files.read(a['path'], {k: a[k] for k in ('sha256', 'size_bytes')}); acquisition = unique(ar)
    require(acquisition.get('native_source_abstract_records') == 1000, 'complete source metadata population')
    members = m['text_members']; require(type(members) is dict and set(members) == {str(i) for i in range(1, 1001)}, 'exact 1000 text IDs')
    documented = {}
    # Metadata only: .ann member bytes are never opened or enumerated from disk.
    for archive in acquisition['official_archives'].values():
        for name, row in archive['members'].items():
            if not name.endswith('.txt'): continue
            require(re.fullmatch('[0-9]{4}\\.txt', name) is not None, 'canonical documented text name')
            identifier = str(int(name[:-4])); require(identifier not in documented, 'duplicate native source ID')
            canonical = str(files.root / TEXT_ROOT / name)
            require(row['category'] == 'raw_source_document' and row['canonical_path'] == canonical,
                    'source receipt exact text path/category')
            documented[identifier] = {'path': TEXT_ROOT + '/' + name,
                                     'sha256': row['sha256'], 'size_bytes': row['bytes']}
    require(documented == members, 'source manifest exactly projected acquisition text metadata')
    for identifier, row in members.items():
        closed(row, ('path', 'sha256', 'size_bytes'))
        require(row['path'] == f'{TEXT_ROOT}/{int(identifier):04}.txt', 'text-only exact source path')
        check_binding({k: row[k] for k in ('sha256', 'size_bytes')})
    return m, ar

def unit_order(phase):
    rows = [(m, r, i) for m in MODELS for r in REPEATS for i in IDS[phase]]
    return sorted(rows, key=lambda x: digest(encoded({'scheme': ORDER_SCHEME, 'phase': phase,
                                                      'model': x[0], 'repeat': x[1], 'document_id': x[2]})))

def slot_order(model, repeat, identifier):
    dependencies = {'B': set(), 'G_critic': {'B'}, 'T_critic': {'B'},
                    'G_repair': {'G_critic'}, 'T_repair': {'T_critic'}, 'S_repair': {'T_critic'}}
    complete, order = set(), []
    while len(order) != 6:
        ready = [s for s in SLOTS if s not in complete and dependencies[s] <= complete]
        chosen = min(ready, key=lambda s: digest(encoded({'scheme': ORDER_SCHEME, 'model': model,
                        'repeat': repeat, 'document_id': identifier, 'slot': s})))
        order.append(chosen); complete.add(chosen)
    return order

def unit_path(phase, model, repeat, identifier):
    return f'{OUTPUT}/{phase}/units/{MODELS.index(model)}/{repeat}/{identifier:04}'

def retry_delay(attempt, result, now=None):
    require(type(attempt) is int and attempt >= 0, 'retry attempt index')
    seconds = min(300, 5 * 2 ** min(attempt, 6))
    raw = result.get('safe_response_headers', {}).get('retry-after')
    if raw is not None:
        try:
            value = float(raw)
            if math.isfinite(value): seconds = max(seconds, max(0, value))
        except (TypeError, ValueError):
            try:
                target = email.utils.parsedate_to_datetime(raw)
                if target.tzinfo is None: target = target.replace(tzinfo=datetime.timezone.utc)
                seconds = max(seconds, max(0, target.timestamp() - (time.time() if now is None else now)))
            except (TypeError, ValueError, OverflowError): pass
    require(math.isfinite(seconds), 'finite retry delay'); return seconds

class Session:
    def __init__(self, files, protocol_raw, manifest_raw, phase, modules, invented, adapter):
        self.files, self.protocol_raw, self.manifest_raw = files, protocol_raw, manifest_raw
        self.phase, self.modules, self.invented, self.adapter = phase, modules, invented, adapter
        self.protocol = unique(protocol_raw); self.contract = self.protocol['controller_contract']
        self.protocol_sha = digest(protocol_raw); self.t = modules['sccomics_api_transport_v1']
        self.transport = modules['sccomics_api_scientific_transport_v3']
        self.tickets = modules['sccomics_api_feedback_tickets_v2']
        self.compiler = modules['sccomics_api_native_projection_v1']
        self.manifest, self.acquisition_raw = source_manifest(files, manifest_raw, invented)
        self.metadata = {PROTOCOL: binding(protocol_raw), SOURCE_MANIFEST: binding(manifest_raw),
                         ACQUISITION: binding(self.acquisition_raw)}
        self.texts = {}
    def text(self, i):
        row = self.manifest['text_members'][str(i)]
        raw = self.files.read(row['path'], {k: row[k] for k in ('sha256', 'size_bytes')})
        raw.decode('utf8', 'strict'); self.texts[row['path']] = binding(raw); return raw
    def verify_inputs(self):
        for name, row in {**self.metadata, **self.texts}.items(): self.files.read(name, row)
        for name, wanted in self.contract['source_sha256_pins'].items():
            require(digest(self.files.read(name)) == wanted, 'source changed during controller run')
    def put(self, name, obj): self.files.obj(name, obj)
    def request(self, model, repeat, i, slot, source, messages, base, replay=False):
        payload_raw = self.t.serialized(self.tickets.payload_from_messages(model, messages, slot))
        intent = {'study_id': 'SC_NATIVE_API_FEEDBACK_V1', 'protocol_sha256': self.protocol_sha,
                  'split': 'invented' if self.invented else self.phase, 'document_id': i,
                  'model': model, 'repeat': repeat, 'stage': slot}
        freeze = {'version': 1, 'purpose': 'identified_scientific_API_request', 'endpoint': self.t.ENDPOINT,
                  'protocol_sha256': self.protocol_sha, 'intent': intent, 'payload_sha256': digest(payload_raw),
                  'source_sha256': digest(source), 'source_sha256_pins': self.protocol['transport_source_sha256_pins'],
                  'limits': {**LIMITS, 'reserved_tokens': len(payload_raw) + (2048 if slot.endswith('critic') else 8192)},
                  'root_authorized_scientific_request': not self.invented, 'request_mode': 'corpus'}
        freeze_raw = self.t.serialized(freeze); identity = self.transport.intent_identity(intent)
        call_path = base + '/calls/' + slot + '/call.json'
        attempts = []; n = 0
        while True:
            attempt_path = base + f'/calls/{slot}/attempts/{n:06}'
            expected_binding = {'intent_identity': identity, 'intent': intent, 'attempt': n,
                'payload_sha256': digest(payload_raw), 'source_sha256': digest(source), 'freeze_sha256': digest(freeze_raw)}
            manifest_path = self.files.path(attempt_path + '/output_manifest.json')
            if replay or manifest_path.exists():
                require(manifest_path.exists(), 'missing completed required attempt/output')
                result = self.transport.reopen_attempt(self.files.path(attempt_path), expected_binding)
            else:
                kwargs = {'execute_actual': not self.invented}
                if self.invented:
                    kwargs.update(exchange=self.adapter['exchange'], key_reader=self.adapter['key_reader'], pacer=self.adapter['pacer'])
                result = self.transport.run_attempt(self.files.root, payload_raw, source, freeze_raw,
                    self.files.path(attempt_path), attempt=n, **kwargs)
            attempts.append({'attempt': n, 'path': attempt_path,
                             'manifest': binding(self.files.read(attempt_path + '/output_manifest.json')),
                             'passed': result['passed'], 'retryable': result['retryable_transport_failure']})
            if result['passed']:
                require(type(result.get('raw_content')) is str, 'successful exact raw assistant content required')
                obj = {'intent': intent, 'intent_identity': identity, 'controller_phase': self.phase,
                       'payload_sha256': digest(payload_raw), 'source_sha256': digest(source),
                       'freeze_sha256': digest(freeze_raw), 'final_attempt': n, 'attempts': attempts,
                       'cache_is_additional_repeat': False}
                self.put(call_path, obj)
                return result['raw_content'].encode('utf8', 'strict'), obj
            if not result['retryable_transport_failure']:
                self.put(call_path, {'intent': intent, 'intent_identity': identity, 'controller_phase': self.phase,
                                     'terminal_failed': True, 'attempts': attempts, 'no_baseline_fallback': True})
                raise FamilyStopped('terminal required API slot: ' + slot)
            wait_path = base + f'/calls/{slot}/retry_waits/{n:06}.json'
            if self.files.path(wait_path).exists():
                wait = unique(self.files.read(wait_path))
                closed(wait, ('attempt', 'seconds', 'calculation_time_unix', 'same_identity', 'retry_policy'))
                require(type(wait['calculation_time_unix']) is float and math.isfinite(wait['calculation_time_unix']), 'captured retry wait timestamp')
                calculated_at = wait['calculation_time_unix']
            else:
                require(not replay, 'missing required retry wait artifact'); calculated_at = time.time()
            delay = retry_delay(n, result, now=calculated_at)
            self.put(wait_path, {'attempt': n, 'seconds': delay, 'calculation_time_unix': calculated_at,
                                 'same_identity': identity, 'retry_policy': RETRY_POLICY})
            if not replay and not self.files.path(base + f'/calls/{slot}/attempts/{n+1:06}/output_manifest.json').exists():
                (self.adapter['sleep'] if self.invented else time.sleep)(delay)
            n += 1  # deliberately no retry-count cap and no changed scientific payload
    def unit(self, model, repeat, i, replay=False):
        source = self.text(i); base = unit_path(self.phase, model, repeat, i)
        self.files.put(base + '/source.txt', source)
        order = slot_order(model, repeat, i)
        identity = {'phase': self.phase, 'document_id': i, 'model': model, 'repeat': repeat,
                    'source_sha256': digest(source), 'protocol_sha256': self.protocol_sha}
        self.put(base + '/identity_and_order.json', {'identity': identity, 'slot_order': order, 'scheme': ORDER_SCHEME})
        raw, critics, calls = {}, {}, {}
        for slot in order:
            if slot == 'B': messages = self.tickets.build_baseline_messages(source)
            elif slot.endswith('critic'):
                messages = self.tickets.build_critic_messages(source, raw['B'], slot[0])
            else:
                arm = slot[0]; artifact = critics['G' if arm == 'G' else 'T']
                messages = self.tickets.build_repair_messages(source, raw['B'], artifact, arm,
                    model=model, source_id=str(i), repeat=repeat)
            response, call = self.request(model, repeat, i, slot, source, messages, base, replay)
            raw[slot] = response; calls[slot] = call
            self.files.put(base + '/raw/' + slot + '.json', response)
            if slot.endswith('critic'):
                artifact = self.tickets.compile_critic(source, raw['B'], response, slot[0])
                self.put(base + '/critics/' + slot[0] + '.json', artifact.to_json())
                try: artifact.require_feedback()
                except self.tickets.CriticTerminalFailure as error:
                    raise FamilyStopped('terminal required critic artifact: ' + slot) from error
                critics[slot[0]] = artifact
            else:
                compilation = self.compiler.compile_graph_json(source, response)
                arm = 'B' if slot == 'B' else slot[0]
                self.put(base + '/graphs/' + arm + '.json', compilation.to_json())
                try: compilation.require_document()
                except self.compiler.ProjectionTerminalFailure as error:
                    raise FamilyStopped('terminal required complete graph artifact: ' + slot) from error
        for mode in ('T', 'S'):
            mapping = self.tickets.correspondence(critics['T'], mode, model=model, source_id=str(i), repeat=repeat)
            self.put(base + '/correspondence/' + mode + '.json', mapping.to_json())
        self.files.put(base + '/correspondence/unchanged_ticket_text.json', critics['T'].ticket_text_bytes)
        self.put(base + '/T_S_shared_critic.json', {'identity': identity,
            'baseline_sha256': digest(raw['B']), 'T_critic_sha256': digest(raw['T_critic']),
            'T_S_ticket_text_sha256': digest(critics['T'].ticket_text_bytes),
            'T_S_exact_critic_shared': True, 'repair_template_shared': True,
            'only_correspondence_payload_changes': True, 'independent_S_critic_call': False})
        return {'identity': identity, 'base': base, 'slot_order': order,
                'calls': {s: digest(self.files.read(base + '/calls/' + s + '/call.json')) for s in SLOTS},
                'graphs': {a: digest(self.files.read(base + '/graphs/' + a + '.json')) for a in ARMS},
                'critics': {a: digest(self.files.read(base + '/critics/' + a + '.json')) for a in ('G', 'T')}}
    def phase_population(self, replay=False):
        return [self.unit(m, r, i, replay) for m, r, i in unit_order(self.phase)]
    def expected_files(self, population):
        result = set()
        for unit in population:
            base = unit['base']
            result.update(base + '/' + name for name in ('source.txt', 'identity_and_order.json', 'T_S_shared_critic.json',
                          'correspondence/T.json', 'correspondence/S.json', 'correspondence/unchanged_ticket_text.json'))
            result.update(base + '/raw/' + slot + '.json' for slot in SLOTS)
            result.update(base + '/graphs/' + arm + '.json' for arm in ARMS)
            result.update(base + '/critics/' + arm + '.json' for arm in ('G', 'T'))
            for slot in SLOTS:
                call_path = base + '/calls/' + slot + '/call.json'; result.add(call_path)
                call = unique(self.files.read(call_path))
                for attempt in call['attempts']:
                    result.update(attempt['path'] + '/' + name for name in
                                  ('request.json', 'source.txt', 'freeze.json', 'response_raw.safe', 'result.json', 'output_manifest.json'))
                    if not attempt['passed']:
                        result.add(base + f"/calls/{slot}/retry_waits/{attempt['attempt']:06}.json")
        return result
    def snapshot_files(self, population):
        base = self.files.path(OUTPUT + '/' + self.phase)
        result = {}
        for current, dirs, names in os.walk(base, followlinks=False):
            for d in dirs: require(not (Path(current) / d).is_symlink(), 'owned output directory alias')
            for name in names:
                path = Path(current) / name; relative_name = path.relative_to(self.files.root).as_posix()
                if name == '.DS_Store':
                    self.files.regular_metadata(relative_name)
                    continue  # ordinary regular OS metadata only, never aliases/FIFO
                if relative_name == OUTPUT + '/' + self.phase + '/source_barrier.json':
                    self.files.regular_metadata(relative_name)
                    continue  # only this exact planned phase-root barrier path
                require(not name.startswith('FAILED_'), 'terminal failure prohibits completion')
                result[relative_name] = binding(self.files.read(relative_name))
        require(set(result) == self.expected_files(population), 'exact owned phase output population; no extra/missing objects')
        return result
    def barrier(self, population, save):
        n = len(IDS[self.phase]) * 6
        require(len(population) == n and len({tuple(x['identity'][k] for k in ('model', 'repeat', 'document_id')) for x in population}) == n,
                'exact complete model/repeat/source population')
        self.verify_inputs()
        obj = {'version': 1, 'status': 'complete_source_generation_no_Gold_or_scores',
               'protocol_sha256': self.protocol_sha, 'source_manifest_sha256': digest(self.manifest_raw),
               'phase': self.phase, 'source_ids': list(IDS[self.phase]), 'models': list(MODELS),
               'repeats': list(REPEATS), 'logical_true_calls': n * 6, 'native_graphs': n * 4,
               'critic_artifacts': n * 2, 'complete_units': population,
               'source_files': self.texts, 'files': self.snapshot_files(population),
               'source_recompiled_and_tickets_sham_requests_rebuilt': True,
               'ignored_OS_metadata_policy': self.contract['non_scientific_OS_metadata'],
               'invalid_individual_native_occurrences_retained': True,
               'Gold_read_or_scoring_performed': False, 'additional_repeat_from_cache': False,
               'source_sha256_pins': self.contract['source_sha256_pins']}
        path = OUTPUT + '/' + self.phase + '/source_barrier.json'
        if save: self.put(path, obj)
        else: require(self.files.read(path) == encoded(obj), 'fresh complete source barrier differs')
        return obj

def verify_reviews(files, c):
    covered = set()
    for row in c['independent_source_reviews']:
        closed(row, ('path', 'sha256', 'size_bytes')); relative(row['path'])
        require(row['path'].startswith('research/') and row['path'].endswith('.json'), 'SOURCE review metadata path only')
        review = unique(files.read(row['path'], {k: row[k] for k in ('sha256', 'size_bytes')}))
        require(review.get('status') == 'source_review_passed' and review.get('independent_from_implementer') is True,
                'different-agent SOURCE passed identity')
        pins = review.get('reviewed_source_sha256_pins')
        require(type(pins) is dict and pins and set(pins) <= set(SOURCES), 'review exact focused source roles')
        for name, value in pins.items():
            require(value == c['source_sha256_pins'][name], 'different review current source SHA'); covered.add(name)
    require(covered == set(SOURCES), 'all eight controller/ticket/compiler/native/transport/pacing sources reviewed')

@contextlib.contextmanager
def study_lock(files):
    p = files.path(OUTPUT + '/study.lock'); p.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(p, os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW, 0o600)
    try:
        try: fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error: raise PipelineIntegrityError('another generation controller owns study') from error
        yield
    finally: fcntl.flock(fd, fcntl.LOCK_UN); os.close(fd)

def run_phase(root, protocol_bytes, source_manifest_bytes, phase, *, execute_actual=False,
              adapter=None, test_gate_bytes=None):
    """Generate one complete phase; no implicit paid admission or held-Gold stage.

    Fake adapter exact fields: exchange/key_reader/pacer/sleep. Its root name must
    start INVENTED_; inputs still use the fixed scientific population. The fake
    transport intent says invented and never invokes the external credential.
    """
    require(phase in IDS, 'development or test phase only')
    invented = adapter is not None
    require((invented and execute_actual is False) or (not invented and execute_actual is True),
            'default refuses actual requests/source access')
    files = Files(root)
    if invented:
        require(files.root.name.startswith('INVENTED_'), 'owned INVENTED root')
        closed(adapter, ('exchange', 'key_reader', 'pacer', 'sleep'))
        require(all(callable(adapter[k]) for k in ('exchange', 'key_reader', 'sleep')), 'explicit fake I/O, no defaults')
    p = unique(protocol_bytes)
    require(p.get('state') == ('INVENTED_FROZEN_SOURCE_FIXTURE' if invented else 'FROZEN_BEFORE_CORPUS_API') and
            p.get('scientific_request_authority') is (not invented), 'external root frozen protocol authority')
    c = p['controller_contract']; validate_contract(c)
    require(binding(source_manifest_bytes) == {k: c['source_manifest'][k] for k in ('sha256', 'size_bytes')}, 'manifest protocol bytes')
    require(files.read(PROTOCOL) == protocol_bytes and files.read(SOURCE_MANIFEST) == source_manifest_bytes,
            'provided inputs match captured canonical authority paths')
    require(p.get('transport_source_sha256_pins') == {n: c['source_sha256_pins'][n] for n in TRANSPORT_SOURCES}, 'fixed v2 transport pins')
    captures = {}
    for name, wanted in c['source_sha256_pins'].items():
        raw = files.read(name); require(digest(raw) == wanted, 'actual captured source pin'); captures[name] = raw
    verify_reviews(files, c)
    review_bindings = {r['path']: {k: r[k] for k in ('sha256', 'size_bytes')} for r in c['independent_source_reviews']}
    with study_lock(files), captured_modules(captures, root=files.root) as modules:
        session = Session(files, protocol_bytes, source_manifest_bytes, phase, modules, invented, adapter)
        session.metadata.update(review_bindings)
        try:
            if phase == 'test':
                require(type(test_gate_bytes) is bytes, 'distinct root after-DEV test gate required')
                require(files.read(TEST_GATE) == test_gate_bytes, 'canonical test gate bytes')
                g = unique(test_gate_bytes)
                closed(g, ('version', 'purpose', 'protocol_sha256', 'source_manifest_sha256',
                           'development_barrier', 'root_after_development_authorized_test_sources', 'Gold_authority'))
                require(type(g['version']) is int and g['version'] == 1 and g['purpose'] == 'SC_API_AFTER_DEV_SOURCE_ONLY_TEST_GATE' and
                        g['protocol_sha256'] == digest(protocol_bytes) and g['source_manifest_sha256'] == digest(source_manifest_bytes) and
                        g['root_after_development_authorized_test_sources'] is True and g['Gold_authority'] is False,
                        'test gate same immutable protocol/source no Gold')
                db = g['development_barrier']; closed(db, ('path', 'sha256', 'size_bytes'))
                require(db['path'] == OUTPUT + '/development/source_barrier.json', 'same study DEV barrier')
                files.read(db['path'], {k: db[k] for k in ('sha256', 'size_bytes')})
                dev = Session(files, protocol_bytes, source_manifest_bytes, 'development', modules, invented, adapter)
                population = dev.phase_population(replay=True); dev.barrier(population, save=False)
                session.metadata.update(dev.metadata); session.texts.update(dev.texts)
                session.metadata[TEST_GATE] = binding(test_gate_bytes)
                session.metadata[db['path']] = {k: db[k] for k in ('sha256', 'size_bytes')}
            population = session.phase_population()
            # Reopen raw attempts, rebuild all requests/critics/sham/compilations,
            # not merely filenames or a previous successful status flag.
            rebuilt = session.phase_population(replay=True)
            require(rebuilt == population, 'fresh complete generation identity replay')
            result = session.barrier(rebuilt, save=True)
            session.verify_inputs()
            return result
        except BaseException as error:
            failure = {'status': 'whole_planned_family_stopped_no_completion_or_B_fallback', 'phase': phase,
                       'protocol_sha256': digest(protocol_bytes), 'error_type': type(error).__name__,
                       'error': str(error), 'Gold_read_or_scoring_performed': False}
            try: session.verify_inputs()
            except BaseException as guard: failure['final_input_integrity_error'] = type(guard).__name__ + ': ' + str(guard)
            path = OUTPUT + '/' + phase + '/FAILED_' + str(time.time_ns()) + '.json'
            files.obj(path, failure)
            raise

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True); parser.add_argument('--phase', choices=tuple(IDS), required=True)
    parser.add_argument('--root-execute-frozen-api-source-generation', action='store_true')
    args = parser.parse_args()
    require(args.root_execute_frozen_api_source_generation, 'default refuses API/source access; root external authority required')
    files = Files(args.root)
    result = run_phase(files.root, files.read(PROTOCOL), files.read(SOURCE_MANIFEST), args.phase,
                       execute_actual=True, test_gate_bytes=files.read(TEST_GATE) if args.phase == 'test' else None)
    print(json.dumps({'status': result['status'], 'phase': args.phase, 'logical_true_calls': result['logical_true_calls'],
                      'Gold_read_or_scores': False}))
    return 0

if __name__ == '__main__': raise SystemExit(main())
