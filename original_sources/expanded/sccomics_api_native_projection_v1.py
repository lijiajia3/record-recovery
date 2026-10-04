"""In-memory native API JSON -> family-declared Emission compiler.

No file/key/corpus/API/model loader or scorer is called. Caller supplies the
unchanged source and raw assistant JSON after its own complete source barrier.
This is not the deferred flat-one-event/trigger emitter: every emitted object,
event instance, raw role occurrence and original identifier is retained.

compile_graph_json() returns a Compilation. A malformed identifiable object is
one invalid Emission in its list's native family. A non-JSON/top-shape artifact
returns terminal_graph_artifact_failure with no manufactured graph; callers
must fail the planned family, never replace this with an empty prediction.
"""
from __future__ import annotations

import base64
from dataclasses import dataclass
import hashlib
import json
import math
import re
from typing import Any

from sccomics_native_graph import (Emission, NativeDocument, Registry,
                                  document_json, parse_emissions)


TRAIN_ENTITY_LABELS = frozenset(('Characterization', 'Material', 'Property',
                                'Element', 'Process', 'Value', 'SC', 'Main', 'Doping'))
TRAIN_REGISTRY = Registry(entity_labels=TRAIN_ENTITY_LABELS,
                          event_labels=frozenset(('Doping',)),
                          relation_labels=frozenset(('Condition', 'Equivalent', 'Target')),
                          event_roles=frozenset(('Dopant', 'Site')),
                          allow_numeric_role_suffix=True)
FAMILIES = {'entities': 'T', 'relations': 'R', 'events': 'E'}
ATOM = re.compile(r'[^\s:;]+\Z')
ENDPOINT = re.compile(r'[TE][1-9][0-9]*\Z')


class ProjectionTerminalFailure(ValueError):
    """The raw artifact cannot supply a complete native prediction population."""
    def __init__(self, compilation: 'Compilation'):
        super().__init__('terminal API graph artifact: ' + ','.join(compilation.diagnostics))
        self.compilation = compilation


@dataclass(frozen=True)
class Compilation:
    source_bytes: bytes
    raw_json_bytes: bytes
    emissions: tuple[Emission, ...] | None
    document: NativeDocument | None
    record_diagnostics: tuple[dict, ...]
    diagnostics: tuple[str, ...]
    status: str

    def require_document(self) -> NativeDocument:
        if self.document is None or self.emissions is None:
            raise ProjectionTerminalFailure(self)
        return self.document

    def to_json(self) -> dict:
        """Lossless raw bytes plus diagnostics; no annotation/file I/O."""
        return {
            'status': self.status,
            'source_sha256': hashlib.sha256(self.source_bytes).hexdigest(),
            'raw_assistant_JSON_utf8_base64': base64.b64encode(self.raw_json_bytes).decode('ascii'),
            'raw_assistant_JSON_sha256': hashlib.sha256(self.raw_json_bytes).hexdigest(),
            'terminal_failure': self.document is None,
            'diagnostics': list(self.diagnostics),
            'record_diagnostics': list(self.record_diagnostics),
            'emissions': None if self.emissions is None else [
                {'family': e.family, 'declared_label': e.declared_label,
                 'payload_base64': base64.b64encode(e.payload).decode('ascii')} for e in self.emissions],
            'native_document': None if self.document is None else document_json(self.document),
            'Gold_or_performance_scoring_performed': False,
            'registry': {'entity_labels': sorted(TRAIN_REGISTRY.entity_labels),
                         'event_labels': sorted(TRAIN_REGISTRY.event_labels),
                         'relation_labels': sorted(TRAIN_REGISTRY.relation_labels),
                         'event_roles': sorted(TRAIN_REGISTRY.event_roles),
                         'allow_numeric_role_suffix': TRAIN_REGISTRY.allow_numeric_role_suffix},
        }


def _require(value: bool, message: str) -> None:
    if not value:
        raise ValueError(message)


def _utf8(value: bytes | str, *, source: bool) -> bytes:
    if type(value) is bytes:
        raw = value
    elif type(value) is str:
        raw = value.encode('utf8', 'strict')
    else:
        raise TypeError('source/assistant JSON must be unchanged bytes or str')
    if source:
        raw.decode('utf8', 'strict')
    return raw


def _strict_json(raw: bytes) -> Any:
    def unique(items):
        result = {}
        for key, value in items:
            _require(key not in result, 'duplicate_JSON_key')
            result[key] = value
        return result
    def nonfinite(_):
        raise ValueError('nonfinite_JSON_constant')
    obj = json.loads(raw.decode('utf8', 'strict'), object_pairs_hook=unique,
                     parse_constant=nonfinite)
    stack = [obj]
    while stack:
        value = stack.pop()
        if type(value) is dict:
            stack.extend(value.values())
        elif type(value) is list:
            stack.extend(value)
        elif type(value) is float:
            _require(math.isfinite(value), 'nonfinite_JSON_number')
    return obj


def _atom(value: Any, pattern: re.Pattern, message: str) -> str:
    _require(type(value) is str and pattern.fullmatch(value) is not None, message)
    value.encode('utf8', 'strict')  # Surrogates cannot enter a native line.
    return value


def _quote_anchor(source: str, span: Any) -> tuple[int, int]:
    _require(type(span) is dict and set(span) == {'quote', 'occurrence'}, 'span_object_schema')
    quote, occurrence = span['quote'], span['occurrence']
    _require(type(quote) is str and quote != '', 'quote_nonempty_string')
    _require(type(occurrence) is int and occurrence >= 0, 'occurrence_nonnegative_integer')
    at = -1
    for _ in range(occurrence + 1):
        at = source.find(quote, at + 1)  # Counts overlapping exact occurrences.
        _require(at >= 0, 'quote_occurrence_unavailable')
    return at, at + len(quote)


def _render(obj: Any, family: str, source: str) -> tuple[bytes, dict]:
    _require(type(obj) is dict, 'record_not_object')
    keys = ({'id', 'type', 'spans'} if family == 'T' else
            {'id', 'type', 'head', 'tail'} if family == 'R' else
            {'id', 'type', 'trigger', 'roles'})
    _require(set(obj) == keys, 'record_closed_schema')
    identifier = _atom(obj['id'], re.compile(family + r'[1-9][0-9]*\Z'), 'id_lexical_schema')
    label = _atom(obj['type'], ATOM, 'label_lexical_schema')
    info = {'id': identifier, 'declared_label': label, 'rendered': True}
    if family == 'T':
        _require(type(obj['spans']) is list and obj['spans'], 'nonempty_spans_list')
        segments = [_quote_anchor(source, span) for span in obj['spans']]
        offsets = ';'.join(f'{a} {b}' for a, b in segments)
        # Uniform offset-only T representation. The reviewed native parser
        # treats the informational reference field as optional; this avoids
        # inventing FP for legal cross-LF/CR spans without conditional rewriting.
        # All original quotes/occurrences/source bytes stay in raw diagnostics.
        line = f'{identifier}\t{label} {offsets}'
        info.update(segments=[list(s) for s in segments],
                    informational_reference_field='uniformly_omitted_offset_only')
    elif family == 'R':
        head = _atom(obj['head'], ENDPOINT, 'relation_head_lexical_schema')
        tail = _atom(obj['tail'], ENDPOINT, 'relation_tail_lexical_schema')
        line = f'{identifier}\t{label} Arg1:{head} Arg2:{tail}'
    else:
        trigger = _atom(obj['trigger'], re.compile(r'T[1-9][0-9]*\Z'), 'trigger_lexical_schema')
        _require(type(obj['roles']) is list, 'roles_list_schema')
        roles = []
        for role in obj['roles']:
            _require(type(role) is dict and set(role) == {'type', 'target'}, 'role_object_schema')
            role_type = _atom(role['type'], ATOM, 'role_type_lexical_schema')
            target = _atom(role['target'], ENDPOINT, 'role_target_lexical_schema')
            roles.append(f'{role_type}:{target}')
        line = f'{identifier}\t{label}:{trigger}' + (' ' + ' '.join(roles) if roles else '')
    return line.encode('utf8', 'strict'), info


def compile_graph_json(source_utf8: bytes | str, assistant_json_utf8: bytes | str) -> Compilation:
    """Compile the exact assistant graph JSON; fixed TRAIN Registry, no I/O.

    Use response.raw_content, not a filtered transport graph. Top key order and
    each family's list order survive; forward/nested/cyclic references resolve
    only in parse_emissions. All lexical atoms are checked before Brat rendering.
    Unknown labels are emitted literally and become native unsupported-label FP;
    they are never mapped/deleted. No Gold denominator is constructed here.
    Source UTF-8 corruption or non-string API misuse raises, rather than counting
    a model record. Malformed graph bytes/top-level shape returns terminal failure.
    """
    source_bytes = _utf8(source_utf8, source=True)
    source = source_bytes.decode('utf8', 'strict')
    try:
        raw = _utf8(assistant_json_utf8, source=False)
    except UnicodeError:
        # A supplied Python string may contain a lone surrogate. There were no
        # original bytes in that interface; surrogatepass retains its exact
        # codepoints reversibly while marking this as a terminal non-UTF8 input.
        raw = assistant_json_utf8.encode('utf8', 'surrogatepass')
        return Compilation(source_bytes, raw, None, None, (),
                           ('terminal_invalid_UTF8_string',
                            'raw_string_reversibly_encoded_utf8_surrogatepass',
                            'complete_native_population_unavailable'),
                           'terminal_graph_artifact_failure')
    try:
        graph = _strict_json(raw)
        _require(type(graph) is dict and set(graph) == set(FAMILIES), 'complete_three_family_top_object')
        _require(all(type(v) is list for v in graph.values()), 'every_family_must_be_a_list')
    except (ValueError, UnicodeError, RecursionError) as error:
        return Compilation(source_bytes, raw, None, None, (),
                           ('terminal_' + type(error).__name__, str(error), 'complete_native_population_unavailable'),
                           'terminal_graph_artifact_failure')
    emissions, diagnostics = [], []
    for list_name, objects in graph.items():
        family = FAMILIES[list_name]
        for list_index, obj in enumerate(objects):
            label = obj.get('type') if type(obj) is dict and type(obj.get('type')) is str else None
            raw_object = json.dumps(obj, ensure_ascii=True, separators=(',', ':'), allow_nan=False)
            try:
                payload, info = _render(obj, family, source)
            except (ValueError, TypeError, KeyError, UnicodeError) as error:
                # Exactly one declared-family invalid occurrence. Retain an
                # original lexically safe native ID even for malformed content:
                # an ID alone fails the header, but lets native resolution
                # propagate invalid dependencies and ambiguous duplicate IDs.
                # Full raw JSON remains losslessly archived in diagnostics.
                original_id = obj.get('id') if type(obj) is dict else None
                retain_id = (type(original_id) is str and
                             re.fullmatch(r'[TRE][1-9][0-9]*', original_id) is not None)
                payload = original_id.encode('ascii') if retain_id else raw_object.encode('ascii')
                info = {'rendered': False, 'malformed_object': True,
                        'reason': str(error), 'declared_label': label,
                        'original_ID_retained_for_native_invalid_resolution': retain_id}
            emissions.append(Emission(family, payload, label))
            diagnostics.append(dict(info, family=family, list_name=list_name,
                                    list_index=list_index, raw_object_JSON_ascii=raw_object))
    native = parse_emissions(source_bytes, emissions, registry=TRAIN_REGISTRY)
    _require(len(native.records) == len(emissions), 'one_occurrence_per_known_family_object')
    for family in FAMILIES.values():
        _require(len(native.family(family)) == sum(e.family == family for e in emissions),
                 'native_family_count_conservation')
    for info, record in zip(diagnostics, native.records):
        info.update(native_valid=record.valid, native_issues=list(record.issues))
    status = ('compiled_prediction_with_invalid_records' if any(not r.valid for r in native.records)
              else 'compiled_prediction')
    return Compilation(source_bytes, raw, tuple(emissions), native,
                       tuple(diagnostics), (), status)
