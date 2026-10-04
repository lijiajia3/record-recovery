"""Prospective B/G/T/S prompts and source-only ticket correspondence.

Pure supplied-memory functions: no corpus, annotation, key, model, API, file or
scorer loader. Not a scientific freeze or execution authority. Every raw critic
artifact and every ticket occurrence survive; terminal artifacts never become
an empty critic or baseline fallback. T/S share EXACT critic/text/template bytes.
"""
from __future__ import annotations

import base64
from collections import Counter, defaultdict
from dataclasses import dataclass, replace
import hashlib
import json
import math
import re
from typing import Any

MODELS = ('deepseek-ai/DeepSeek-V4-Pro', 'zai-org/GLM-5.3')
CALL_SLOTS = ('B', 'G_critic', 'G_repair', 'T_critic', 'T_repair', 'S_repair')
FAMILIES = ('T', 'R', 'E')
OPERATIONS = ('add', 'modify', 'delete')
MAX_TICKETS = 12  # Critic feedback budget, NEVER a final graph/Gold population cap.
GRAPH_TOKENS, CRITIC_TOKENS = 8192, 2048

TASK_DEFINITION = """Extract only explicitly stated scientific information from the unchanged source.
Return the complete native graph as one JSON object with exactly entities, relations, events lists.
The text-bound labels are Characterization (characterization method), Material (material mention),
Property (property expression), Element (element/composition mention), Process (processing expression),
Value (explicit value), SC (superconductivity property expression), Main (the subject material),
and Doping (an explicitly stated doping/substitution event trigger). Main is distinct from Element.
These are annotation-task labels, not physical truth or a license to infer unreported facts.
Equivalent is ORDERED property-name -> value (not symmetric equivalence/coreference).
Condition is ORDERED value -> Doping event. Target is ORDERED value/information/event -> material.
A Doping event has an explicit type, a text-bound trigger, and the complete actually stated role
occurrences Dopant and/or Site. Do not require both roles, invent missing roles, or infer them from
a nearby material or a formula. Preserve zero-role events, multiple events sharing one trigger,
multiple same-role targets, repeated raw roles (including numeric suffixes), and T/E references.
Preserve forward/nested/cyclic event references and multiple relation labels on the same directed pair.
Every T/R/E instance has its own original ID (T1/R1/E1 etc.); do not merge distinct instances.
Entities use spans=[{quote: exact literal source substring, occurrence: 0-based integer}]. Count
overlapping occurrences in increasing Unicode-codepoint start order. Preserve discontinuous segment
order; never replace segments by their hull, sort, merge, or normalize source whitespace/CRLF/Unicode.
Relations use id,type,head,tail, with ordered T/E endpoints. Events use id,type,trigger,roles, with
roles=[{type: raw role name, target: T/E ID}]. Unknown literal labels must not be silently mapped.
No final graph size cap. Legitimate empty lists are allowed only when extraction warrants them.
Return JSON only; no Markdown fences, explanation, or character-offset arithmetic.
"""

GRAPH_FORMAT = """JSON shape (strings illustrate format, not facts or mandatory graph size):
{"entities":[{"id":"T1","type":"Main","spans":[{"quote":"literal substring","occurrence":0}]}],
 "relations":[{"id":"R1","type":"Target","head":"E1","tail":"T1"}],
 "events":[{"id":"E1","type":"Doping","trigger":"T2","roles":[{"type":"Dopant","target":"T3"}]}]}
All reference IDs must refer to the returned complete graph; return all three lists.
"""

_STRING = {'type': 'string', 'minLength': 1}
_WITNESS = {'type': 'object', 'properties': {'quote': _STRING,
             'occurrence': {'type': 'integer', 'minimum': 0}},
            'required': ['quote', 'occurrence'], 'additionalProperties': False}
CRITIC_SCHEMA = {'type': 'object', 'properties': {'tickets': {'type': 'array', 'maxItems': MAX_TICKETS,
    'items': {'type': 'object', 'properties': {
        'family': {'enum': list(FAMILIES)}, 'operation': {'enum': list(OPERATIONS)},
        'issue': _STRING, 'recommendation': _STRING, 'uncertainty': _STRING,
        'target_ids': {'type': 'array', 'items': _STRING},
        'source_witnesses': {'type': 'array', 'items': _WITNESS}},
        'required': ['family', 'operation', 'issue', 'recommendation', 'uncertainty',
                     'target_ids', 'source_witnesses'], 'additionalProperties': False}}},
    'required': ['tickets'], 'additionalProperties': False}

BASELINE_PROMPT = """Produce the complete native extraction graph from the whole source.
Inspect all source positions, including information with no cue-word match. Do not use outside
knowledge to invent facts. Resolve source ambiguity explicitly through conservative literal extraction.
"""
GENERIC_CRITIC_PROMPT = """Review the WHOLE unchanged source against the baseline graph carefully.
Check omitted/mistyped/discontinuous T mentions and Doping triggers; missed complete E instances,
zero-role/shared-trigger events, wrong/missing/additional Dopant or Site roles and their binding;
missing/wrong/extra R instances, Equivalent/Condition/Target directions, endpoints, and literal support.
Deletion, addition, and modification are all allowed; do not assume the baseline is correct.
You may cite exact source witnesses and recognize events absent from the baseline. Look beyond
already predicted objects and review source positions with no baseline mention.
Return the closed critic JSON schema. Prioritize at most 12 tickets; 0 is allowed when no concern is
found. This limit is feedback budget, not a cap on final objects or the number of true source objects.
Each ticket has family T/R/E, operation add/modify/delete, issue, recommendation, uncertainty,
target_ids (existing baseline objects affected; [] is allowed for an addition), source_witnesses
(exact quote + 0-based occurrence; [] is allowed when you cannot find a witness).
State uncertainty and do not fabricate a source quotation. Return JSON only.
"""
TARGETED_CRITIC_PROMPT = """Review the WHOLE unchanged source against the baseline graph carefully.
Check omitted/mistyped/discontinuous T mentions and Doping triggers; missed complete E instances,
zero-role/shared-trigger events, wrong/missing/additional Dopant or Site roles and their binding;
missing/wrong/extra R instances, Equivalent/Condition/Target directions, endpoints, and literal support.
Deletion, addition, and modification are all allowed; do not assume the baseline is correct.
The source-only cue table lists every occurrence of a fixed reminder vocabulary. A cue is NOT an
annotated trigger, Gold entity, evidence of an event, or an exhaustive list. Review unmatched source
positions too. Explicitly consider missing events, missing roles, and multiple events near one trigger.
For each concern, localize the affected baseline IDs or an addition's source position using exact
source witnesses. Separate recommendation wording from its target/witness correspondence.
Return the closed critic JSON schema. Prioritize at most 12 tickets; 0 is allowed when no concern is
found. This limit is feedback budget, not a cap on final objects or the number of true source objects.
Each ticket has family T/R/E, operation add/modify/delete, issue, recommendation, uncertainty,
target_ids (existing baseline objects affected; [] is allowed for an addition), source_witnesses
(exact quote + 0-based occurrence; [] is allowed when you cannot find a witness).
State uncertainty and do not fabricate a source quotation. Return JSON only.
"""
GENERIC_REPAIR_PROMPT = """Return a complete revised native graph from the whole source, baseline, and
raw review. You may add, modify, or delete objects, including objects omitted by the reviewer.
Review statements can be wrong. Assess them against the source; do not automatically accept them.
Preserve correct baseline objects and all legitimate shared/zero-role/multiple events. Do not infer
missing scientific facts. No output-count cap and no automatic baseline fallback. Return JSON only.
"""
LOCALIZED_REPAIR_PROMPT = """Return a complete revised native graph from the whole source, baseline,
ticket text, and the separate correspondence table. Ticket slots are anonymous bookkeeping IDs.
Use each slot's table entry to locate its targets and witnesses, then assess the recommendation
against the complete source. Mechanical checks establish only quote location/ID availability,
NOT entailment, correct binding, completeness, physical truth, or permission to accept an edit.
Review statements, tables, or malformed tickets can be wrong. You may add, modify, or delete
objects, including objects omitted by the tickets, and may reject a proposed edit after source review.
Preserve correct baseline objects and all legitimate shared/zero-role/multiple events. Do not infer
missing scientific facts. No output-count cap and no automatic baseline fallback. Return JSON only.
"""

PROMPT_TEXTS = {'task_definition.txt': TASK_DEFINITION, 'graph_format.txt': GRAPH_FORMAT,
               'baseline.txt': BASELINE_PROMPT, 'generic_critic.txt': GENERIC_CRITIC_PROMPT,
               'targeted_critic.txt': TARGETED_CRITIC_PROMPT, 'generic_repair.txt': GENERIC_REPAIR_PROMPT,
               'localized_repair_shared_T_S.txt': LOCALIZED_REPAIR_PROMPT}


class TicketInputError(ValueError):
    pass


class CriticTerminalFailure(ValueError):
    def __init__(self, artifact: 'FeedbackArtifact'):
        super().__init__('terminal critic artifact: ' + artifact.status)
        self.artifact = artifact


def _require(ok: bool, message: str) -> None:
    if not ok:
        raise TicketInputError(message)


def _digest(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _json(value: Any) -> bytes:
    # Canonical ASCII escaping retains even malformed JSON string codepoints in
    # diagnostics; never replace/drop a raw ticket because it contains a surrogate.
    return json.dumps(value, ensure_ascii=True, separators=(',', ':'), allow_nan=False).encode('utf8')


def _raw(value: bytes | str) -> bytes:
    _require(type(value) in (bytes, str), 'supplied bytes/string required')
    return value if type(value) is bytes else value.encode('utf8', 'strict')


def _strict(raw: bytes) -> Any:
    def pairs(items):
        result = {}
        for key, value in items:
            _require(key not in result, 'duplicate JSON key')
            result[key] = value
        return result
    def invalid(_):
        raise TicketInputError('nonfinite JSON constant')
    value = json.loads(raw.decode('utf8', 'strict'), object_pairs_hook=pairs, parse_constant=invalid)
    stack = [value]
    while stack:
        item = stack.pop()
        if type(item) is dict:
            stack.extend(item.values())
        elif type(item) is list:
            stack.extend(item)
        elif type(item) is float:
            _require(math.isfinite(item), 'nonfinite JSON number')
    return value


def _inputs(source_utf8: bytes | str, baseline_json_utf8: bytes | str) -> tuple[bytes, bytes, dict]:
    source, baseline = _raw(source_utf8), _raw(baseline_json_utf8)
    source.decode('utf8', 'strict')
    graph = _strict(baseline)
    _require(type(graph) is dict and set(graph) == {'entities', 'relations', 'events'},
             'baseline complete native top object required')
    _require(all(type(v) is list for v in graph.values()), 'baseline complete family lists required')
    # Keep all invalid native objects too; no format cleaning or native score here.
    return source, baseline, graph


def source_cues(source_utf8: bytes | str) -> list[dict]:
    """Every source-only reminder match, without caps or a Gold/trigger claim."""
    source = _raw(source_utf8).decode('utf8', 'strict')
    patterns = (('doping', r'dop(?:ing|ed|ant)\w*'), ('substitution', r'substitut\w*'),
                ('oxygen_deficiency', r'oxygen[\s-]*deficien\w*'),
                ('vacancy', r'vacanc\w*'), ('intercalation', r'intercalat\w*'))
    results = []
    for kind, pattern in patterns:
        for match in re.finditer('(?=(' + pattern + '))', source, re.IGNORECASE):
            start, end = match.start(1), match.end(1)
            results.append({'kind': kind, 'start': start, 'end': end, 'literal': source[start:end]})
    return sorted(results, key=lambda row: (row['start'], row['end'], row['kind']))


def _quote(source: str, witness: Any) -> list[int]:
    _require(type(witness) is dict and set(witness) == {'quote', 'occurrence'}, 'witness schema')
    quote, occurrence = witness['quote'], witness['occurrence']
    _require(type(quote) is str and bool(quote), 'nonempty witness quote')
    _require(type(occurrence) is int and occurrence >= 0, '0-based witness occurrence integer')
    at = -1
    for _ in range(occurrence + 1):
        at = source.find(quote, at + 1)
        _require(at >= 0, 'witness occurrence unavailable')
    return [at, at + len(quote)]


def _ticket_schema(ticket: Any) -> None:
    required = {'family', 'operation', 'issue', 'recommendation', 'uncertainty', 'target_ids', 'source_witnesses'}
    _require(type(ticket) is dict and set(ticket) == required, 'closed ticket object')
    _require(ticket['family'] in FAMILIES and type(ticket['family']) is str, 'native ticket family')
    _require(ticket['operation'] in OPERATIONS and type(ticket['operation']) is str, 'ticket operation')
    for key in ('issue', 'recommendation', 'uncertainty'):
        _require(type(ticket[key]) is str and bool(ticket[key]), 'nonempty ' + key)
        ticket[key].encode('utf8', 'strict')
    _require(type(ticket['target_ids']) is list and all(type(v) is str and bool(v) for v in ticket['target_ids']), 'target ID list')
    for target in ticket['target_ids']:
        target.encode('utf8', 'strict')
    _require(type(ticket['source_witnesses']) is list, 'source witness list')
    for witness in ticket['source_witnesses']:
        _require(type(witness) is dict and set(witness) == {'quote', 'occurrence'}, 'witness closed schema')
        _require(type(witness['quote']) is str and bool(witness['quote']), 'nonempty witness quote')
        witness['quote'].encode('utf8', 'strict')
        _require(type(witness['occurrence']) is int and witness['occurrence'] >= 0, 'witness 0-based integer')


@dataclass(frozen=True)
class FeedbackArtifact:
    source_bytes: bytes
    baseline_bytes: bytes
    raw_critic_bytes: bytes
    ticket_text_bytes: bytes | None
    aligned_table_bytes: bytes | None
    diagnostics_bytes: bytes
    status: str
    critic_kind: str = 'unspecified'

    def require_feedback(self) -> 'FeedbackArtifact':
        if self.ticket_text_bytes is None or self.aligned_table_bytes is None:
            raise CriticTerminalFailure(self)
        return self

    def to_json(self) -> dict:
        return {'status': self.status, 'critic_kind': self.critic_kind, 'source_sha256': _digest(self.source_bytes),
                'baseline_sha256': _digest(self.baseline_bytes), 'critic_sha256': _digest(self.raw_critic_bytes),
                'unchanged_source_utf8_base64': base64.b64encode(self.source_bytes).decode('ascii'),
                'raw_baseline_base64': base64.b64encode(self.baseline_bytes).decode('ascii'),
                'raw_critic_base64': base64.b64encode(self.raw_critic_bytes).decode('ascii'),
                'ticket_text': None if self.ticket_text_bytes is None else json.loads(self.ticket_text_bytes),
                'aligned_correspondence': None if self.aligned_table_bytes is None else json.loads(self.aligned_table_bytes),
                'diagnostics': json.loads(self.diagnostics_bytes), 'Gold_or_fact_scoring': False,
                'automatic_baseline_fallback': False}


def parse_critic(source_utf8: bytes | str, baseline_json_utf8: bytes | str,
                 critic_json_utf8: bytes | str) -> FeedbackArtifact:
    """Retain every ticket; whole-artifact shape/budget errors are terminal.

    Caller must already establish transport success/stop (never length). This
    raw supplied-memory parser neither infers nor changes HTTP/finish metadata.
    """
    source, baseline, graph = _inputs(source_utf8, baseline_json_utf8)
    try:
        raw = _raw(critic_json_utf8)
    except UnicodeError:
        raw = critic_json_utf8.encode('utf8', 'surrogatepass')
        return FeedbackArtifact(source, baseline, raw, None, None,
                                _json({'terminal': 'invalid Unicode string', 'raw_string_encoding': 'surrogatepass'}),
                                'terminal_critic_artifact_failure')
    try:
        obj = _strict(raw)
        _require(type(obj) is dict and set(obj) == {'tickets'}, 'complete critic top object')
        _require(type(obj['tickets']) is list, 'tickets list required')
        _require(len(obj['tickets']) <= MAX_TICKETS, 'critic exceeds fixed 12-ticket budget; do not truncate')
    except (ValueError, UnicodeError, RecursionError) as error:
        return FeedbackArtifact(source, baseline, raw, None, None,
                                _json({'terminal': type(error).__name__, 'message': str(error)}),
                                'terminal_critic_artifact_failure')
    ids = Counter()
    id_family = {}
    for list_name, family in (('entities', 'T'), ('relations', 'R'), ('events', 'E')):
        for record in graph[list_name]:
            identifier = record.get('id') if type(record) is dict else None
            if type(identifier) is str:
                ids[identifier] += 1
                id_family[identifier] = family
    text_rows, table_rows, diagnostics = [], [], []
    decoded = source.decode('utf8')
    for i, ticket in enumerate(obj['tickets']):
        slot = f'S{i:06d}'
        raw_ticket = json.dumps(ticket, ensure_ascii=True, separators=(',', ':'), allow_nan=False)
        schema_issue = None
        try:
            _ticket_schema(ticket)
            text_row = {key: ticket[key] for key in ('family', 'operation', 'issue', 'recommendation', 'uncertainty')}
        except (ValueError, TypeError, KeyError) as error:
            schema_issue = str(error)
            text_row = {'family': ticket.get('family') if type(ticket) is dict else None,
                        'operation': ticket.get('operation') if type(ticket) is dict else None,
                        'raw_invalid_ticket_JSON_ascii': raw_ticket}
        text_rows.append(dict(slot=slot, **text_row))
        targets = ticket.get('target_ids') if type(ticket) is dict else None
        witnesses = ticket.get('source_witnesses') if type(ticket) is dict else None
        checks = {'schema_issue': schema_issue, 'target_ID_checks': [], 'witness_checks': [],
                  'witness_location_is_not_entailment': True}
        if type(targets) is list:
            for target in targets:
                lexical = type(target) is str and re.fullmatch(r'[TRE][1-9][0-9]*', target) is not None
                unique = lexical and ids[target] == 1
                family_match = (unique and (text_row.get('operation') == 'add' or
                                           id_family[target] == text_row.get('family')))
                checks['target_ID_checks'].append({'original': target, 'lexical': lexical,
                    'baseline_unique_ID_present': bool(unique), 'affected_family_matches_or_add_context': bool(family_match),
                    'native_validity_not_checked': True})
        else:
            checks['target_list_issue'] = 'missing or non-list target_ids retained'
        if type(witnesses) is list:
            for j, witness in enumerate(witnesses):
                try:
                    anchor = _quote(decoded, witness)
                    checks['witness_checks'].append({'index': j, 'available': True, 'anchor': anchor})
                except (ValueError, TypeError, KeyError) as error:
                    checks['witness_checks'].append({'index': j, 'available': False, 'issue': str(error)})
            if not witnesses:
                checks['no_exact_source_witness'] = True
        else:
            checks['witness_list_issue'] = 'missing or non-list source_witnesses retained'
        table_rows.append({'slot': slot, 'target_ids': targets, 'source_witnesses': witnesses,
                           'mechanical_diagnostics': checks})
        diagnostics.append({'slot': slot, 'schema_issue': schema_issue, 'original_ticket_JSON_ascii': raw_ticket})
    return FeedbackArtifact(source, baseline, raw, _json(text_rows), _json(table_rows),
                            _json({'ticket_count': len(text_rows), 'records': diagnostics,
                                   'invalid_ticket_count': sum(d['schema_issue'] is not None for d in diagnostics)}),
                            'feedback_available_with_diagnostics')


@dataclass(frozen=True)
class Correspondence:
    source_sha256: str
    baseline_sha256: str
    critic_sha256: str
    table_bytes: bytes
    routing_diagnostics_bytes: bytes

    def to_json(self) -> dict:
        return {'source_sha256': self.source_sha256, 'baseline_sha256': self.baseline_sha256,
                'critic_sha256': self.critic_sha256, 'table': json.loads(self.table_bytes),
                'routing_diagnostics': json.loads(self.routing_diagnostics_bytes)}


def _identity(model: str, source_id: str, repeat: int) -> None:
    _require(model in MODELS, 'fixed candidate account model')
    _require(type(source_id) is str and bool(source_id), 'explicit source ID')
    _require(type(repeat) is int and repeat in (0, 1, 2), 'retain exactly three request repeats 0/1/2')


def _permutation(size: int, seed: bytes) -> list[int]:
    """Deterministic SHA Fisher-Yates; rejection sampling avoids modulus bias."""
    values, counter = list(range(size)), 0
    for i in range(size - 1, 0, -1):
        n = i + 1
        bound = (1 << 256) - ((1 << 256) % n)
        while True:
            value = int.from_bytes(hashlib.sha256(seed + counter.to_bytes(8, 'big')).digest(), 'big')
            counter += 1
            if value < bound:
                break
        j = value % n
        values[i], values[j] = values[j], values[i]
    return values


def correspondence(artifact: FeedbackArtifact, mode: str, *, model: str,
                   source_id: str, repeat: int) -> Correspondence:
    artifact.require_feedback()
    _require(mode in ('T', 'S'), 'localized correspondence mode T/S')
    _identity(model, source_id, repeat)
    rows = json.loads(artifact.aligned_table_bytes)
    text_rows = json.loads(artifact.ticket_text_bytes)
    strata = defaultdict(list)
    unstratified = []
    for index, text_row in enumerate(text_rows):
        family, operation = text_row.get('family'), text_row.get('operation')
        if family in FAMILIES and operation in OPERATIONS:
            strata[(family, operation)].append(index)
        else:
            unstratified.append(index)  # Retained fixed slots, not silently erased.
    result = list(rows)
    diagnostic = []
    identity = _json({'scheme': 'SC_API_TICKET_SHA_SHAM_v1', 'model': model, 'source_id': source_id,
                      'repeat': repeat, 'source_sha256': _digest(artifact.source_bytes)})
    for (family, operation), indices in sorted(strata.items()):
        seed = identity + b'\0' + family.encode() + b'\0' + operation.encode()
        donors = _permutation(len(indices), seed) if mode == 'S' else list(range(len(indices)))
        for local_index, donor in enumerate(donors):
            destination, origin = indices[local_index], indices[donor]
            result[destination] = dict(rows[origin], slot=rows[destination]['slot'])
        diagnostic.append({'family': family, 'operation': operation, 'population': len(indices),
                           'donor_local_indices': donors,
                           'fixed_points': sum(i == j for i, j in enumerate(donors)),
                           'can_permute': len(indices) > 1})
    return Correspondence(_digest(artifact.source_bytes), _digest(artifact.baseline_bytes),
                          _digest(artifact.raw_critic_bytes), _json(result),
                          _json({'mode': mode, 'identity_sha256': _digest(identity), 'strata': diagnostic,
                                 'unstratified_fixed_slots': unstratified,
                                 'ticket_text_bytes_unchanged': True,
                                 'all_mapping_payloads_preserved': True,
                                 'routing_not_selected_by_Gold': True}))


def _messages(instructions: str, sections: list[tuple[str, str]], *, graph: bool) -> tuple[dict, dict]:
    system = TASK_DEFINITION + ('\n' + GRAPH_FORMAT if graph else '') + '\n' + instructions
    user = '\n\n'.join(f'===== {name} =====\n{value}\n===== END {name} =====' for name, value in sections)
    return ({'role': 'system', 'content': system}, {'role': 'user', 'content': user})


def baseline_messages(source_utf8: bytes | str) -> tuple[dict, dict]:
    source = _raw(source_utf8).decode('utf8', 'strict')
    return _messages(BASELINE_PROMPT, [('UNCHANGED_SOURCE', source)], graph=True)


def critic_messages(source_utf8: bytes | str, baseline_json_utf8: bytes | str,
                    mode: str) -> tuple[dict, dict]:
    source, baseline, _ = _inputs(source_utf8, baseline_json_utf8)
    _require(mode in ('G', 'T'), 'exactly G/T critic; S reuses EXACT T critic')
    sections = [('UNCHANGED_SOURCE', source.decode('utf8')), ('RAW_BASELINE_GRAPH', baseline.decode('utf8')),
                ('CRITIC_JSON_SCHEMA', _json(CRITIC_SCHEMA).decode('utf8'))]
    if mode == 'T':
        sections.append(('SOURCE_ONLY_CUES_ALL_POSITIONS', _json(source_cues(source)).decode('utf8')))
    return _messages(GENERIC_CRITIC_PROMPT if mode == 'G' else TARGETED_CRITIC_PROMPT, sections, graph=False)


def generic_repair_messages(artifact: FeedbackArtifact) -> tuple[dict, dict]:
    artifact.require_feedback()
    # G gets its raw critic and strong review template; no T cue/map mechanical module.
    return _messages(GENERIC_REPAIR_PROMPT,
                     [('UNCHANGED_SOURCE', artifact.source_bytes.decode('utf8')),
                      ('RAW_BASELINE_GRAPH', artifact.baseline_bytes.decode('utf8')),
                      ('RAW_GENERIC_CRITIC', artifact.raw_critic_bytes.decode('utf8'))], graph=True)


def localized_repair_messages(artifact: FeedbackArtifact, routing: Correspondence) -> tuple[dict, dict]:
    artifact.require_feedback()
    _require((routing.source_sha256, routing.baseline_sha256, routing.critic_sha256) ==
             (_digest(artifact.source_bytes), _digest(artifact.baseline_bytes), _digest(artifact.raw_critic_bytes)),
             'routing must bind this same EXACT T critic/source/baseline')
    # Deliberately no arm name, sham diagnostic, different wording or routing seed in prompt.
    return _messages(LOCALIZED_REPAIR_PROMPT,
                     [('UNCHANGED_SOURCE', artifact.source_bytes.decode('utf8')),
                      ('RAW_BASELINE_GRAPH', artifact.baseline_bytes.decode('utf8')),
                      ('UNCHANGED_TICKET_TEXT', artifact.ticket_text_bytes.decode('utf8')),
                      ('CORRESPONDENCE_TABLE', routing.table_bytes.decode('utf8'))], graph=True)


def request_spec(model: str, source_id: str, repeat: int, slot: str,
                 source_utf8: bytes | str, messages: tuple[dict, dict]) -> dict:
    """Candidate payload+distinct cache identity, NOT transport/execution authority."""
    _identity(model, source_id, repeat)
    _require(slot in CALL_SLOTS, 'six unique slots; no independent S critic')
    source = _raw(source_utf8)
    source.decode('utf8', 'strict')
    payload = payload_from_messages(model, messages, slot)
    _require(source.decode('utf8') in messages[1]['content'], 'complete unchanged source present')
    identity = {'scheme': 'SC_API_FEEDBACK_REQUEST_v1', 'model': model, 'source_id': source_id,
                'repeat': repeat, 'slot': slot, 'source_sha256': _digest(source),
                'payload_sha256': _digest(_json(payload))}
    return {'payload': payload, 'scientific_request_identity': identity,
            'request_identity_sha256': _digest(_json(identity)),
            # The synthetic transport's strict native_graph validator rejects
            # individual bad quotes/IDs. Scientific predictions instead need
            # raw JSON-object capture -> lossless native compiler -> FP counts.
            'transport_output_mode': 'json_object',
            'artifact_kind': 'critic_tickets' if slot.endswith('_critic') else 'native_graph',
            'candidate_not_scientifically_frozen': True,
            'thinking_disabled_is_NOT_established': True, 'implicit_retry_or_fallback': False}


def prompt_snapshot_bytes() -> dict[str, bytes]:
    """Caller may save fixed prompt snapshots; this function does no file I/O."""
    return {name: text.encode('utf8') for name, text in PROMPT_TEXTS.items()}


# Root controller adapters, with no import or change to any actual transport.
def build_baseline_messages(source_utf8: bytes | str) -> tuple[dict, dict]:
    return baseline_messages(source_utf8)


def build_critic_messages(source_utf8: bytes | str, baseline_raw: bytes | str,
                          kind: str) -> tuple[dict, dict]:
    return critic_messages(source_utf8, baseline_raw, kind)


def compile_critic(source_utf8: bytes | str, baseline_raw: bytes | str,
                   critic_raw: bytes | str, kind: str) -> FeedbackArtifact:
    _require(kind in ('G', 'T'), 'only G/T critics; S uses EXACT T artifact')
    return replace(parse_critic(source_utf8, baseline_raw, critic_raw), critic_kind=kind)


def build_repair_messages(source_utf8: bytes | str, baseline_raw: bytes | str,
                          critic_artifact: FeedbackArtifact, mappingmode: str, *,
                          model: str | None = None, source_id: str | None = None,
                          repeat: int | None = None) -> tuple[dict, dict]:
    """Exact root adapter. Save correspondence() with the same arguments separately.

    G requires its G artifact; T/S require that same repeat's T artifact. Return
    two role/content dictionaries, no hidden final graph or fallback. This is
    not a whole-cohort/transport success gate or a hostile object sandbox.
    """
    source, baseline, _ = _inputs(source_utf8, baseline_raw)
    _require(source == critic_artifact.source_bytes and baseline == critic_artifact.baseline_bytes,
             'repair source/baseline must match captured critic input EXACTLY')
    _require(mappingmode in ('G', 'T', 'S'), 'repair arm G/T/S')
    expected = 'G' if mappingmode == 'G' else 'T'
    _require(critic_artifact.critic_kind == expected, 'repair must use the correct captured critic kind')
    if mappingmode == 'G':
        return generic_repair_messages(critic_artifact)
    return localized_repair_messages(critic_artifact, correspondence(
        critic_artifact, mappingmode, model=model, source_id=source_id, repeat=repeat))


def payload_from_messages(model: str, messages: tuple[dict, dict], slot: str) -> dict:
    """Only proposed payload; root's scientific identity/freeze remains external."""
    _require(model in MODELS and slot in CALL_SLOTS, 'fixed model and six-call slot')
    _require(type(messages) is tuple and len(messages) == 2, 'two explicit fixed messages')
    for expected_role, message in zip(('system', 'user'), messages):
        _require(type(message) is dict and set(message) == {'role', 'content'} and
                 message['role'] == expected_role and type(message['content']) is str and
                 bool(message['content']), 'fixed system/user message schema')
    # No explicit n: the separately reviewed scientific transport uses model-specific
    # exact fields and validates a single returned choice. This is a requested
    # default, not proof of provider control compliance or a repeat/cache rule.
    payload = {'model': model, 'messages': list(messages), 'stream': True, 'temperature': 0.2,
               'enable_thinking': False, 'response_format': {'type': 'json_object'},
               'max_tokens': CRITIC_TOKENS if slot.endswith('_critic') else GRAPH_TOKENS}
    if model == 'zai-org/GLM-5.3':
        payload['reasoning_effort'] = 'low'  # model-specific official guide, still always-on reasoning
    return payload
