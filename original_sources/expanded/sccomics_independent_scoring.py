"""Independent in-memory native Brat scorer using only the standard library.

The parser, closure representation, graph-isomorphism search and global root
selection are implemented here. No production parser/scorer, training module,
file reader, model, cache, network client or CLI is imported. Callers own the
source/Gold/prediction authorization and byte-freeze barriers.

The primary R metric projects E endpoints to typed triggers; primary E credit
requires a common injective identifier map across complete matched closures.
The independent search counter differs from the original VF2 counter. It charges
candidate root pairs, node-candidate trials, completed local maps, global search
states and diagnostic augmenting states. Limits are inclusive and at most one
million charged operations per graph metric/document. No approximation or
Gold-population reduction is returned on budget/resource failure.
"""
from __future__ import annotations

import base64
from collections import Counter
from dataclasses import dataclass, replace
import hashlib
import json
from typing import Iterable, Mapping, Sequence


ENTITY_LABELS = frozenset({"Characterization", "Material", "Property", "Element",
                           "Process", "Value", "SC", "Main"})
EVENT_LABELS = frozenset({"Doping"})
RELATION_LABELS = frozenset({"Condition", "Equivalent", "Target"})
DEFAULT_MAX_SEARCH_STEPS = 1_000_000
ENDPOINT_ALIASES = {"R_trigger_projected_complete": "relations_trigger_projected",
                    "E_global_sharing_consistent_complete": "events_full_graph"}


class GoldIntegrityError(ValueError):
    """Complete supplied Gold is corrupt; scoring must stop."""


class MatchingBudgetExceeded(RuntimeError):
    """Independent exact search would exceed its inclusive operation limit."""


class MatchingResourceError(RuntimeError):
    """Exact search could not finish because of interpreter/memory resources."""


@dataclass(frozen=True)
class Registry:
    entity_labels: frozenset[str] = ENTITY_LABELS
    event_labels: frozenset[str] = EVENT_LABELS
    relation_labels: frozenset[str] = RELATION_LABELS
    event_roles: frozenset[str] = frozenset({"Dopant", "Site"})
    allow_numeric_role_suffix: bool = True

    def __post_init__(self):
        if type(self.allow_numeric_role_suffix) is not bool:
            raise TypeError("numeric suffix support must be explicit bool")
        for name in ("entity_labels", "event_labels", "relation_labels", "event_roles"):
            value = getattr(self, name)
            if not isinstance(value, frozenset) or any(not _atom(label) for label in value):
                raise TypeError("registry sets must be immutable UTF-8 label atoms")

    def allows_role(self, name: str) -> bool:
        if name in self.event_roles:
            return True
        if not self.allow_numeric_role_suffix:
            return False
        return any(name.startswith(base) and _digits(name[len(base):])
                   for base in self.event_roles)

    def role_allowed(self, name: str) -> bool:
        return self.allows_role(name)


@dataclass(frozen=True)
class Emission:
    family: str
    payload: bytes
    declared_label: str | None = None


@dataclass(frozen=True)
class Record:
    occurrence: int
    family: str
    identifier: str | None
    label: str | None
    raw: bytes
    valid: bool = True
    issues: tuple[str, ...] = ()
    segments: tuple[tuple[int, int], ...] = ()
    surface: str | None = None
    trigger: str | None = None
    roles: tuple[tuple[str, str], ...] = ()
    arg1: str | None = None
    arg2: str | None = None
    input_format: str = "brat"

    def mention_key(self):
        if self.family != "T" or not self.valid:
            raise ValueError("typed-segment identity requires a valid T")
        return self.label, self.segments


@dataclass(frozen=True)
class Document:
    source_bytes: bytes
    source_text: str
    annotation_bytes: bytes | None
    records: tuple[Record, ...]
    gold: bool
    registry: Registry
    artifact_status: str = "present"
    diagnostics: tuple[str, ...] = ()

    @property
    def by_id(self):
        return {r.identifier: r for r in self.records if r.valid and r.identifier is not None}

    def family(self, name: str, *, valid_only=False):
        return tuple(r for r in self.records if r.family == name and (r.valid or not valid_only))


@dataclass(frozen=True)
class GoldSource:
    text_utf8: bytes
    annotation_utf8: bytes


def _digits(value):
    return isinstance(value, str) and bool(value) and all("0" <= c <= "9" for c in value)


def _identifier(value):
    return (value == "*" or isinstance(value, str) and len(value) > 1 and
            ("A" <= value[0] <= "Z" or value[0] == "#") and _digits(value[1:]))


def _reference(value, allowed="TE"):
    return isinstance(value, str) and len(value) > 1 and value[0] in allowed and _digits(value[1:])


def _atom(value):
    return _utf8_string(value) and bool(value) and all(not c.isspace() and c not in ":;" for c in value)


def _utf8_string(value):
    if not isinstance(value, str):
        return False
    try:
        value.encode("utf-8", "strict")
        return True
    except UnicodeError:
        return False


def _source(value):
    try:
        if isinstance(value, str):
            raw = value.encode("utf-8", "strict")
        elif isinstance(value, bytes):
            raw = value
        else:
            raise TypeError("source must be original UTF-8 bytes or unchanged string")
        return raw, raw.decode("utf-8", "strict")
    except UnicodeError as exc:
        raise GoldIntegrityError("original source is not strict UTF-8") from exc


def _family(raw):
    first = raw.lstrip()[:1]
    return first.decode("ascii") if first in (b"T", b"R", b"E") else "AUX"


def _hint(raw):
    try:
        tokens = raw.decode("utf-8", "strict").split("\t", 1)[1].split()
        return tokens[0].split(":", 1)[0] if tokens else None
    except (UnicodeError, IndexError):
        return None


def _bad(n, family, raw, reason, identifier=None, label=None, input_format="brat"):
    return Record(n, family, identifier, label, raw, valid=False,
                  issues=(reason,), input_format=input_format)


def _span_issues(segments, source, surface):
    if not segments or any(len(pair) != 2 for pair in segments):
        raise ValueError("invalid_segment_syntax")
    if any(a < 0 or b < a or b > len(source) for a, b in segments):
        raise ValueError("invalid_offset_range")
    for i in range(len(segments)):
        a, b = segments[i]
        for j in range(i):
            c, d = segments[j]
            if a < d and c < b and max(a, c) < min(b, d):
                raise ValueError("overlapping_segments")
    reference = " ".join(source[a:b] for a, b in segments)
    return (("missing_reference_text",) if surface is None else
            () if surface == reference else ("reference_text_mismatch",))


def _brat_record(n, raw, source, declared=None, label_hint=None):
    family = declared or _family(raw)
    label = _hint(raw) or label_hint
    try:
        line = raw.decode("utf-8", "strict")
    except UnicodeError:
        return _bad(n, family, raw, "invalid_utf8", label=label_hint)
    if "\r" in line or "\n" in line:
        return _bad(n, family, raw, "embedded_line_break", label=label)
    header, separator, tail = line.partition("\t")
    if not separator or not _identifier(header):
        return _bad(n, family, raw, "invalid_record_header", header, label)
    actual = header[0] if header[0] in "TRE" else "AUX"
    if declared is not None and actual != declared:
        return _bad(n, family, raw, "declared_family_mismatch", header, label)
    if actual == "AUX":
        return Record(n, "AUX", header, label, raw)
    try:
        if actual == "T":
            body, third, surface = tail.partition("\t")
            typ, space, offsets = body.partition(" ")
            if not space or not _atom(typ):
                raise ValueError("invalid_T_label_or_span")
            segments = tuple(tuple(int(token) for token in part.split()) for part in offsets.split(";"))
            text = surface if third else None
            issues = _span_issues(segments, source, text)
            return Record(n, "T", header, typ, raw, segments=segments, surface=text, issues=issues)
        if "\t" in tail:
            raise ValueError("R_E_requires_two_tab_fields")
        tokens = tail.split()
        if actual == "R":
            if len(tokens) != 3 or not _atom(tokens[0]):
                raise ValueError("invalid_relation_syntax")
            arguments = {}
            for token in tokens[1:]:
                role, colon, target = token.partition(":")
                if not colon or role not in {"Arg1", "Arg2"} or role in arguments or not _reference(target):
                    raise ValueError("relation_requires_Arg1_Arg2_once")
                arguments[role] = target
            if set(arguments) != {"Arg1", "Arg2"}:
                raise ValueError("relation_requires_Arg1_Arg2_once")
            return Record(n, "R", header, tokens[0], raw, arg1=arguments["Arg1"], arg2=arguments["Arg2"])
        if not tokens:
            raise ValueError("missing_event_trigger")
        typ, colon, trigger = tokens[0].partition(":")
        if not colon or not _atom(typ) or not _reference(trigger, "T"):
            raise ValueError("invalid_event_trigger")
        roles = []
        for token in tokens[1:]:
            role, colon, target = token.partition(":")
            if not colon or not _atom(role) or not _reference(target):
                raise ValueError("invalid_event_role_endpoint")
            roles.append((role, target))
        return Record(n, "E", header, typ, raw, trigger=trigger, roles=tuple(roles))
    except (ValueError, TypeError) as exc:
        return _bad(n, actual, raw, str(exc), header, label)


def _resolve(records, registry, gold):
    id_counts = Counter(r.identifier for r in records if r.identifier is not None and r.identifier != "*")
    result = []
    for r in records:
        issues = list(r.issues)
        valid = r.valid
        if r.identifier is not None and r.identifier != "*" and id_counts[r.identifier] > 1:
            valid = False
            issues.append("duplicate_identifier")
        if r.valid and not gold and r.family != "AUX":
            supported = (registry.entity_labels | registry.event_labels if r.family == "T" else
                         registry.relation_labels if r.family == "R" else registry.event_labels)
            if r.label not in supported:
                valid = False
                issues.append("unsupported_prediction_label")
            if r.family == "E" and any(not registry.allows_role(role) for role, _ in r.roles):
                valid = False
                issues.append("unsupported_prediction_role")
        result.append(replace(r, valid=valid, issues=tuple(issues)))
    # Monotone invalidation reaches a fixed point even through cycles.
    while True:
        index = {r.identifier: r for r in result if r.valid and r.identifier is not None}
        changed = False
        next_records = []
        for r in result:
            if r.valid and r.family in {"E", "R"}:
                targets = ((r.trigger,) + tuple(target for _, target in r.roles) if r.family == "E"
                           else (r.arg1, r.arg2))
                issue = None
                if any(target not in index or index[target].family not in {"T", "E"} for target in targets):
                    issue = "dangling_or_invalid_dependency"
                elif r.family == "E" and index[r.trigger].family != "T":
                    issue = "event_trigger_not_T"
                if issue is not None:
                    r = replace(r, valid=False, issues=r.issues + (issue,))
                    changed = True
            next_records.append(r)
        result = next_records
        if not changed:
            break
    invalid = [r for r in result if not r.valid]
    if gold and invalid:
        raise GoldIntegrityError("malformed/ambiguous Gold: " + ";".join(
            f"{r.occurrence}:{r.identifier}:{','.join(r.issues)}" for r in invalid))
    warnings = tuple(f"{r.occurrence}:{r.identifier}:{issue}" for r in result if r.valid for issue in r.issues)
    return tuple(result), warnings


def _document(raw_source, text, annotations, records, gold, registry):
    records, diagnostics = _resolve(records, registry, gold)
    unclassified = tuple(f"unclassified_output_record:{r.occurrence}" for r in records if not r.valid and r.family == "AUX")
    status = ("output_failure" if unclassified else "present_with_invalid_records" if any(not r.valid for r in records) else "present")
    return Document(raw_source, text, annotations, records, gold, registry, status, diagnostics + unclassified)


def parse_document(source_text: bytes | str, annotations: bytes | None, *, gold=False, registry=None):
    reg = registry or Registry()
    raw, text = _source(source_text)
    if annotations is None:
        if gold:
            raise GoldIntegrityError("missing Gold annotations")
        return Document(raw, text, None, (), False, reg, "missing_prediction", ("missing_prediction_artifact",))
    if not isinstance(annotations, bytes):
        raise TypeError("annotation input must be bytes or missing prediction None")
    records = []
    for physical_line in annotations.split(b"\n"):
        line = physical_line[:-1] if physical_line.endswith(b"\r") else physical_line
        if line.strip():
            records.append(_brat_record(len(records), line, text))
    return _document(raw, text, annotations, records, gold, reg)


def parse_emissions(source_text, emissions: Iterable[Emission], *, registry=None):
    raw, text = _source(source_text)
    reg = registry or Registry()
    records = []
    for n, emitted in enumerate(emissions):
        if not isinstance(emitted, Emission) or emitted.family not in {"T", "R", "E", "AUX"}:
            raise TypeError("explicit T/R/E/AUX emission required")
        if not isinstance(emitted.payload, bytes) or (emitted.declared_label is not None and not isinstance(emitted.declared_label, str)):
            raise TypeError("emission bytes and optional string label required")
        records.append(_brat_record(n, emitted.payload, text, emitted.family, emitted.declared_label))
    return _document(raw, text, None, records, False, reg)


def _json_raw(obj):
    try:
        return json.dumps(dict(obj), ensure_ascii=True, allow_nan=False, sort_keys=True).encode("utf-8")
    except (TypeError, ValueError):
        return repr(obj).encode("utf-8", "backslashreplace")


def parse_prediction_records(source_text, objects: Sequence[Mapping], *, registry=None):
    """Direct typed JSON validation: strings are never interpolated into Brat."""
    raw_source, text = _source(source_text)
    reg = registry or Registry()
    records = []
    for n, obj in enumerate(objects):
        if not isinstance(obj, Mapping) or not isinstance(obj.get("family"), str) or obj["family"] not in {"T", "R", "E", "AUX"}:
            raise ValueError("JSON prediction needs an explicit T/R/E/AUX family")
        family = obj["family"]
        raw = _json_raw(obj)
        label = obj.get("type") if isinstance(obj.get("type"), str) else None
        try:
            if family == "AUX":
                if set(obj) != {"family", "raw_brat"} or not isinstance(obj["raw_brat"], str):
                    raise ValueError("AUX_JSON_schema")
                r = _brat_record(n, obj["raw_brat"].encode("utf-8", "strict"), text, "AUX")
                records.append(replace(r, raw=raw, input_format="json"))
                continue
            keys = ({"family", "id", "type", "segments", "text"} if family == "T" else
                    {"family", "id", "type", "arg1", "arg2"} if family == "R" else
                    {"family", "id", "type", "trigger", "roles"})
            if set(obj) != keys or not _reference(obj.get("id"), family) or not _atom(label):
                raise ValueError("JSON_keys_id_type_schema")
            common = dict(occurrence=n, family=family, identifier=obj["id"], label=label, raw=raw, input_format="json")
            if family == "T":
                spans = obj["segments"]
                if (not isinstance(spans, list) or not spans or not _utf8_string(obj["text"]) or
                    any(not isinstance(p, list) or len(p) != 2 or any(type(v) is not int for v in p) for p in spans) or
                    "\n" in obj["text"] or "\r" in obj["text"]):
                    raise ValueError("T_JSON_schema")
                segments = tuple(tuple(p) for p in spans)
                records.append(Record(**common, segments=segments, surface=obj["text"], issues=_span_issues(segments, text, obj["text"])))
            elif family == "R":
                if not _reference(obj["arg1"]) or not _reference(obj["arg2"]):
                    raise ValueError("R_JSON_endpoint_schema")
                records.append(Record(**common, arg1=obj["arg1"], arg2=obj["arg2"]))
            else:
                if not _reference(obj["trigger"], "T") or not isinstance(obj["roles"], list):
                    raise ValueError("E_JSON_trigger_roles_schema")
                roles = []
                for entry in obj["roles"]:
                    if (not isinstance(entry, Mapping) or set(entry) != {"role", "target"} or
                            not _atom(entry["role"]) or not _reference(entry["target"])):
                        raise ValueError("E_JSON_role_endpoint_schema")
                    roles.append((entry["role"], entry["target"]))
                records.append(Record(**common, trigger=obj["trigger"], roles=tuple(roles)))
        except (ValueError, TypeError, KeyError, UnicodeError) as exc:
            # A malformed JSON object has no validated native annotation ID.
            # Keep all original fields in raw bytes, plus declared family/type.
            records.append(_bad(n, family, raw, str(exc), label=label, input_format="json"))
    return _document(raw_source, text, None, records, False, reg)


def document_json(doc):
    return {"source_utf8_base64": base64.b64encode(doc.source_bytes).decode("ascii"),
            "annotation_utf8_base64": None if doc.annotation_bytes is None else base64.b64encode(doc.annotation_bytes).decode("ascii"),
            "gold": doc.gold, "artifact_status": doc.artifact_status, "diagnostics": list(doc.diagnostics),
            "records": [{"occurrence": r.occurrence, "family": r.family, "id": r.identifier, "type": r.label,
                         "segments": [list(p) for p in r.segments], "text": r.surface, "trigger": r.trigger,
                         "roles": [{"role": role, "target": target} for role, target in r.roles],
                         "arg1": r.arg1, "arg2": r.arg2, "valid": r.valid, "issues": list(r.issues),
                         "input_format": r.input_format, "raw_base64": base64.b64encode(r.raw).decode("ascii")}
                        for r in doc.records]}


@dataclass(frozen=True)
class _Closure:
    root: str
    colors: dict
    arcs: Counter
    signatures: dict


def _closure(doc, root):
    index = doc.by_id
    colors, arcs = {}, Counter()
    pending = [root.identifier]
    while pending:
        node = pending.pop()
        if node in colors:
            continue
        r = index[node]
        colors[node] = (("T", r.label, r.segments) if r.family == "T" else (r.family, r.label))
        links = (([(r.trigger, ("trigger",))] + [(target, ("role", role)) for role, target in r.roles]) if r.family == "E" else
                 [(r.arg1, ("arg", "Arg1")), (r.arg2, ("arg", "Arg2"))] if r.family == "R" else [])
        for target, role in links:
            arcs[(node, target, role)] += 1
            pending.append(target)
    signatures = {}
    for node, color in colors.items():
        outgoing = Counter()
        incoming = Counter()
        loops = Counter()
        for (a, b, role), count in arcs.items():
            if a == node:
                outgoing[role] += count
            if b == node:
                incoming[role] += count
            if a == b == node:
                loops[role] += count
        signatures[node] = (color, tuple(sorted(outgoing.items())), tuple(sorted(incoming.items())), tuple(sorted(loops.items())))
    return _Closure(root.identifier, colors, arcs, signatures)


class _Budget:
    def __init__(self, limit):
        if type(limit) is not int or not 1 <= limit <= DEFAULT_MAX_SEARCH_STEPS:
            raise ValueError("independent search limit must be integer 1..1000000; unlimited search is forbidden")
        self.limit = limit
        self.counts = Counter()

    def charge(self, kind):
        if sum(self.counts.values()) >= self.limit:
            raise MatchingBudgetExceeded(f"independent exact search would exceed {self.limit} charged operations")
        self.counts[kind] += 1

    def report(self):
        return {"search_steps": sum(self.counts.values()), "search_step_limit": self.limit,
                "search_step_components": dict(self.counts), "counter_implementation": "independent_standard_library_backtracking_not_VF2"}


def _arc_counter(closure, a, b):
    return Counter({role: n for (u, v, role), n in closure.arcs.items() if u == a and v == b})


def _maps(pred, gold, assigned, inverse, budget):
    if len(pred.colors) != len(gold.colors) or sum(pred.arcs.values()) != sum(gold.arcs.values()):
        return
    if Counter(pred.signatures.values()) != Counter(gold.signatures.values()):
        return
    pn, gn = set(pred.colors), set(gold.colors)
    if any(a in pn and b not in gn or b in gn and a not in pn for a, b in assigned.items()):
        return
    fixed = {a: b for a, b in assigned.items() if a in pn}
    if pred.root in fixed and fixed[pred.root] != gold.root or gold.root in inverse and inverse[gold.root] != pred.root:
        return
    fixed[pred.root] = gold.root
    if len(set(fixed.values())) != len(fixed) or any(pred.signatures[a] != gold.signatures[b] for a, b in fixed.items()):
        return
    if any(_arc_counter(pred, a, c) != _arc_counter(gold, b, d)
           for a, b in fixed.items() for c, d in fixed.items()):
        return
    domains = {a: [b for b in gold.colors if pred.signatures[a] == gold.signatures[b]] for a in pred.colors}

    def extend(mapping, used):
        if len(mapping) == len(pred.colors):
            budget.charge("completed_root_maps")
            yield dict(mapping)
            return
        remaining = [a for a in pred.colors if a not in mapping]
        a = min(remaining, key=lambda node: (sum(b not in used for b in domains[node]),
                                           -sum(n for (u, v, _), n in pred.arcs.items() if u == node and v in mapping or v == node and u in mapping)))
        for b in domains[a]:
            budget.charge("candidate_node_trials")
            if b in used:
                continue
            if any(_arc_counter(pred, a, c) != _arc_counter(gold, b, d) or
                   _arc_counter(pred, c, a) != _arc_counter(gold, d, b) for c, d in mapping.items()):
                continue
            mapping[a] = b
            yield from extend(mapping, used | {b})
            del mapping[a]
    yield from extend(fixed, set(fixed.values()))


def _pair_candidates(pgraphs, ggraphs, budget):
    options = {}
    for p in pgraphs:
        targets = []
        for g in ggraphs:
            budget.charge("candidate_root_pairs")
            if (p.colors[p.root] == g.colors[g.root] and len(p.colors) == len(g.colors) and
                    sum(p.arcs.values()) == sum(g.arcs.values()) and
                    Counter(p.signatures.values()) == Counter(g.signatures.values())):
                targets.append(g)
        options[p.root] = targets
    return options


def exact_graph_match(pred, gold, family="E", *, max_search_steps=DEFAULT_MAX_SEARCH_STEPS):
    if family not in {"E", "R"}:
        raise ValueError("full-graph family must be E or R")
    budget = _Budget(max_search_steps)
    try:
        p = [_closure(pred, r) for r in pred.family(family, valid_only=True)]
        g = [_closure(gold, r) for r in gold.family(family, valid_only=True)]
        options = _pair_candidates(p, g, budget)
        order = sorted((graph for graph in p if options[graph.root]), key=lambda graph: len(options[graph.root]))
        best_pairs, best_mapping = [], {}

        def visit(position, used_roots, mapping, inverse, pairs):
            nonlocal best_pairs, best_mapping
            budget.charge("global_combination_states")
            # This is a proof bound on roots, not an approximate graph score.
            # Once reached, extra automorphisms cannot improve the objective.
            upper = len(pairs) + min(len(order) - position, len(g) - len(used_roots))
            if upper <= len(best_pairs):
                return
            if position == len(order):
                best_pairs, best_mapping = list(pairs), dict(mapping)
                return
            current = order[position]
            for target in options[current.root]:
                if target.root in used_roots:
                    continue
                for extension in _maps(current, target, mapping, inverse, budget):
                    combined = {**mapping, **extension}
                    inv = {**inverse, **{b: a for a, b in extension.items()}}
                    visit(position + 1, used_roots | {target.root}, combined, inv, pairs + [(current.root, target.root)])
                    if len(best_pairs) >= upper:
                        return
            visit(position + 1, used_roots, mapping, inverse, pairs)

        visit(0, set(), {}, {}, [])
        return {"tp": len(best_pairs), "optimal_root_count_proven": True,
                "matched_roots": [list(pair) for pair in best_pairs],
                "identifier_mapping": best_mapping, "sharing_scope": "consistent_across_all_matched_root_closures", **budget.report()}
    except (RecursionError, MemoryError) as exc:
        raise MatchingResourceError("independent exact graph matching did not finish") from exc


def root_local_event_match(pred, gold, *, max_search_steps=DEFAULT_MAX_SEARCH_STEPS):
    budget = _Budget(max_search_steps)
    try:
        p = [_closure(pred, r) for r in pred.family("E", valid_only=True)]
        g = [_closure(gold, r) for r in gold.family("E", valid_only=True)]
        options = _pair_candidates(p, g, budget)
        adjacency = {}
        for current in p:
            adjacency[current.root] = [target.root for target in options[current.root]
                                      if next(_maps(current, target, {}, {}, budget), None) is not None]
        gold_to_pred = {}

        def augment(current, seen):
            budget.charge("diagnostic_augmenting_states")
            for target in adjacency[current]:
                if target in seen:
                    continue
                seen.add(target)
                if target not in gold_to_pred or augment(gold_to_pred[target], seen):
                    gold_to_pred[target] = current
                    return True
            return False

        for current in p:
            augment(current.root, set())
        pairs = [[a, b] for b, a in gold_to_pred.items()]
        return {"tp": len(pairs), "matched_roots": pairs, "sharing_scope": "within_each_root_closure_only", **budget.report()}
    except (RecursionError, MemoryError) as exc:
        raise MatchingResourceError("independent local diagnostic did not finish") from exc


def _counts(tp, prediction_count, gold_count):
    fp, fn = prediction_count - tp, gold_count - tp
    if min(tp, fp, fn) < 0:
        raise AssertionError("negative count violates population conservation")
    return {"tp": tp, "fp": fp, "fn": fn, "prediction_count": prediction_count, "gold_count": gold_count,
            "precision": tp / prediction_count if prediction_count else 0.0,
            "recall": tp / gold_count if gold_count else 0.0,
            "f1": 2 * tp / (prediction_count + gold_count) if prediction_count + gold_count else 0.0}


def _bag_metric(pred, gold, pkey, gkey):
    a = Counter(pkey(r) for r in pred if r.valid)
    b = Counter(gkey(r) for r in gold)
    return _counts(sum((a & b).values()), len(pred), len(gold))


def _projected_endpoint(doc, target):
    index = doc.by_id
    r = index[target]
    return r.mention_key() if r.family == "T" else index[r.trigger].mention_key()


def _relation_key(doc, r):
    return r.label, _projected_endpoint(doc, r.arg1), _projected_endpoint(doc, r.arg2)


def _category(r):
    known = ENTITY_LABELS | EVENT_LABELS if r.family == "T" else RELATION_LABELS if r.family == "R" else EVENT_LABELS
    return r.label if r.label in known else "OTHER_NATIVE"


def _inventory(doc):
    return {family: {"count": len(doc.family(family)), "valid_count": len(doc.family(family, valid_only=True)),
                     "invalid_count": sum(not r.valid for r in doc.family(family)),
                     "labels": dict(Counter(r.label or "<missing_label>" for r in doc.family(family)))}
            for family in ("T", "R", "E", "AUX")}


def score_document(pred, gold, *, include_full_relation_graph=True, include_root_local_diagnostic=True,
                   max_search_steps=DEFAULT_MAX_SEARCH_STEPS):
    if not isinstance(pred, Document) or not isinstance(gold, Document) or pred.gold or not gold.gold:
        raise ValueError("parsed prediction and Gold documents in their respective modes required")
    if pred.source_bytes != gold.source_bytes or pred.registry != gold.registry:
        raise ValueError("original source or registry differs")
    if gold.annotation_bytes is None or any(not r.valid for r in gold.records):
        raise GoldIntegrityError("invalid/missing Gold document cannot be scored")
    _Budget(max_search_steps)
    metrics = {}
    pt, gt = pred.family("T"), gold.family("T")
    for name, labels in (("entities_eight", ENTITY_LABELS), ("triggers", EVENT_LABELS),
                         ("combined_text_bound", ENTITY_LABELS | EVENT_LABELS)):
        metrics[name] = _bag_metric([r for r in pt if r.label in labels], [r for r in gt if r.label in labels],
                                    lambda r: r.mention_key(), lambda r: r.mention_key())
    metrics["text_bound_all_native"] = _bag_metric(pt, gt, lambda r: r.mention_key(), lambda r: r.mention_key())
    metrics["relations_trigger_projected"] = _bag_metric(pred.family("R"), gold.family("R"),
                                                       lambda r: _relation_key(pred, r), lambda r: _relation_key(gold, r))
    graph_matches = {}
    for family, name in (("E", "events_full_graph"), ("R", "relations_full_graph")):
        if family == "R" and not include_full_relation_graph:
            continue
        match = exact_graph_match(pred, gold, family, max_search_steps=max_search_steps)
        graph_matches[name] = match
        metrics[name] = _counts(match["tp"], len(pred.family(family)), len(gold.family(family)))
    if include_root_local_diagnostic:
        local = root_local_event_match(pred, gold, max_search_steps=max_search_steps)
        graph_matches["events_root_local_diagnostic"] = local
        metrics["events_root_local_diagnostic"] = _counts(local["tp"], len(pred.family("E")), len(gold.family("E")))
    tables = {}
    for family, name in (("T", "text_bound_all_native"), ("R", "relations_trigger_projected"), ("E", "events_full_graph")):
        pr, gr = pred.family(family), gold.family(family)
        rows = {}
        for label in sorted({_category(r) for r in pr + gr}):
            a, b = [r for r in pr if _category(r) == label], [r for r in gr if _category(r) == label]
            if family == "E":
                selected = graph_matches[name]["matched_roots"]
                tp = sum(_category(pred.by_id[p]) == label for p, g in selected)
                rows[label] = _counts(tp, len(a), len(b))
            elif family == "T":
                rows[label] = _bag_metric(a, b, lambda r: r.mention_key(), lambda r: r.mention_key())
            else:
                rows[label] = _bag_metric(a, b, lambda r: _relation_key(pred, r), lambda r: _relation_key(gold, r))
        for field in ("tp", "fp", "fn", "prediction_count", "gold_count"):
            if sum(row[field] for row in rows.values()) != metrics[name][field]:
                raise AssertionError("per-label and complete-family populations disagree")
        tables[family] = rows
    return {"metrics": metrics, "endpoint_aliases": dict(ENDPOINT_ALIASES), "by_label": tables,
            "graph_matches": graph_matches, "prediction_inventory": _inventory(pred), "gold_inventory": _inventory(gold),
            "invalid_prediction_records": [{"occurrence": r.occurrence, "family": r.family, "identifier": r.identifier,
                                             "label": r.label, "issues": list(r.issues)} for r in pred.records if not r.valid],
            "auxiliary": {"prediction_count": len(pred.family("AUX")), "gold_count": len(gold.family("AUX")),
                          "scope": "preserved_outside_T_R_E_metrics"},
            "artifact_status": pred.artifact_status, "gold_diagnostics": list(gold.diagnostics),
            "prediction_diagnostics": list(pred.diagnostics), "source_sha256": hashlib.sha256(gold.source_bytes).hexdigest(),
            "gold_annotation_sha256": hashlib.sha256(gold.annotation_bytes).hexdigest()}


def score_population(inventory, gold_sources, predictions, *, registry=None, include_full_relation_graph=True,
                     include_root_local_diagnostic=True, max_search_steps=DEFAULT_MAX_SEARCH_STEPS):
    if not inventory or isinstance(inventory, (str, bytes)) or any(not isinstance(i, str) or not i for i in inventory):
        raise ValueError("nonempty fixed inventory of document ID strings required")
    if len(inventory) != len(set(inventory)):
        raise ValueError("duplicate inventory identifier")
    if set(gold_sources) != set(inventory):
        raise GoldIntegrityError("complete Gold inventory differs")
    if not set(predictions) <= set(inventory):
        raise ValueError("prediction outside fixed population")
    reg = registry or Registry()
    # Validate the entire supplied Gold first; no partial population result.
    gold = {i: parse_document(gold_sources[i].text_utf8, gold_sources[i].annotation_utf8, gold=True, registry=reg) for i in inventory}
    reports = {}
    for i in inventory:
        emitted = predictions.get(i)
        pred = (parse_emissions(gold_sources[i].text_utf8, emitted, registry=reg) if isinstance(emitted, (list, tuple)) else
                parse_document(gold_sources[i].text_utf8, emitted, registry=reg))
        reports[i] = score_document(pred, gold[i], include_full_relation_graph=include_full_relation_graph,
                                    include_root_local_diagnostic=include_root_local_diagnostic, max_search_steps=max_search_steps)
    totals = {name: _counts(sum(r["metrics"][name]["tp"] for r in reports.values()),
                           sum(r["metrics"][name]["prediction_count"] for r in reports.values()),
                           sum(r["metrics"][name]["gold_count"] for r in reports.values()))
              for name in next(iter(reports.values()))["metrics"]}
    return {"inventory": list(inventory), "by_document": reports, "totals": totals,
            "missing_predictions": [i for i in inventory if predictions.get(i) is None],
            "endpoint_aliases": dict(ENDPOINT_ALIASES), "scope": "complete_supplied_native_T_R_E_not_auxiliary_or_physical_truth",
            "matching": "independent_exact_one_to_one_globally_consistent_root_graphs", "zero_denominator": "P_R_F1_zero"}
