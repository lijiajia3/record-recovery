"""Independent, in-memory SC-CoMIcs native Brat parser and exact graph scorer.

No corpus-file loader, label CLI, model, author converter, or training code is
imported. Callers are responsible for their prospective source/selection/Gold
barriers. ``parse_document`` accepts original UTF-8 text bytes and emitted
annotation bytes; ``parse_emissions`` additionally preserves the declared family
of malformed model outputs. ``score_population`` iterates a fixed inventory.

The event metric is *globally sharing-consistent*: maximum one-to-one root-event
credit with one consistent injective T/E identifier renaming across matched
root closures. This is stricter than independent root-graph membership tests.
``relations_trigger_projected`` is the separately named author-compatible view
(complete raw population and one-to-one multiplicities, not original-code/F1
reproduction). ``relations_full_graph`` additionally preserves endpoint kinds,
event closures and cross-root sharing. Exact search can be expensive: a supplied
search limit raises MatchingBudgetExceeded; it never returns an approximate
score or reduces the Gold population.

Unknown syntactically valid Gold T/R/E labels are retained. Default prediction
schema rejects unknown labels/roles as FP; a future TRAIN-audited, prospectively
frozen Registry can extend these sets. Auxiliaries are losslessly retained but
are explicitly outside T/R/E metrics, requiring a future declared scope policy.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, replace
import hashlib
import base64
import json
import re
from typing import Iterable, Mapping, Sequence

import networkx as nx


ENTITY_LABELS = frozenset({"Characterization", "Material", "Property", "Element",
                           "Process", "Value", "SC", "Main"})
EVENT_LABELS = frozenset({"Doping"})
RELATION_LABELS = frozenset({"Condition", "Equivalent", "Target"})
IDENTIFIER = re.compile(r"(?:[A-Z#][0-9]+|\*)\Z")
LABEL = re.compile(r"[^\s:;]+\Z")
ENDPOINT = re.compile(r"[TE][0-9]+\Z")
DEFAULT_MAX_SEARCH_STEPS = 1_000_000


class GoldIntegrityError(ValueError):
    """Gold/source corruption: no complete-population score may be issued."""


class MatchingBudgetExceeded(RuntimeError):
    """Exact matching did not complete; no partial score is valid."""


@dataclass(frozen=True)
class Registry:
    entity_labels: frozenset[str] = ENTITY_LABELS
    event_labels: frozenset[str] = EVENT_LABELS
    relation_labels: frozenset[str] = RELATION_LABELS
    event_roles: frozenset[str] = frozenset({"Dopant", "Site"})
    allow_numeric_role_suffix: bool = True

    def role_allowed(self, role: str) -> bool:
        if role in self.event_roles:
            return True
        return self.allow_numeric_role_suffix and any(
            re.fullmatch(re.escape(base) + r"[0-9]+", role)
            for base in self.event_roles)


@dataclass(frozen=True)
class Emission:
    """One raw model-emitted record; family survives malformed contents.

    family is T, R, E or AUX. payload contains one annotation line (no terminal
    newline required). An embedded newline is one invalid emitted object, not
    an opportunity to split it into extra predictions. Unknown output-family
    declarations raise an API error rather than inventing a scoring penalty.
    """
    family: str
    payload: bytes
    declared_label: str | None = None


@dataclass(frozen=True)
class NativeRecord:
    occurrence: int
    identifier: str | None
    family: str
    label: str | None
    raw: bytes
    segments: tuple[tuple[int, int], ...] = ()
    surface: str | None = None
    arg1: str | None = None
    arg2: str | None = None
    trigger: str | None = None
    roles: tuple[tuple[str, str], ...] = ()
    valid: bool = True
    issues: tuple[str, ...] = ()

    def mention_key(self) -> tuple:
        if self.family != "T" or not self.valid:
            raise ValueError("mention_key requires a valid T record")
        return self.label, self.segments


@dataclass(frozen=True)
class NativeDocument:
    source_bytes: bytes
    source_text: str
    annotation_bytes: bytes | None
    records: tuple[NativeRecord, ...]
    gold: bool
    registry: Registry
    artifact_status: str = "present"
    diagnostics: tuple[str, ...] = ()

    @property
    def by_id(self) -> dict[str, NativeRecord]:
        return {r.identifier: r for r in self.records if r.valid and r.identifier}

    def family(self, family: str, *, valid_only: bool = False) -> tuple[NativeRecord, ...]:
        return tuple(r for r in self.records if r.family == family and
                     (r.valid or not valid_only))


@dataclass(frozen=True)
class GoldSource:
    """Source object supplied only after caller's complete Gold-read barrier."""
    text_utf8: bytes
    annotation_utf8: bytes


def _source(source: bytes | str) -> tuple[bytes, str]:
    if isinstance(source, str):
        raw = source.encode("utf-8", "strict")
    elif isinstance(source, bytes):
        raw = source
    else:
        raise TypeError("source must be original UTF-8 bytes or unchanged str")
    try:
        return raw, raw.decode("utf-8", "strict")
    except UnicodeError as exc:
        raise GoldIntegrityError("source is not valid UTF-8") from exc


def _family_prefix(raw: bytes) -> str:
    # Leading whitespace can make syntax invalid without concealing a recoverable
    # emitted family. The family-declared JSON/Emission API is preferred.
    prefix = raw.lstrip()[:1]
    return chr(prefix[0]) if prefix in (b"T", b"R", b"E") else "AUX"


def _label_hint(raw: bytes) -> str | None:
    try:
        field = raw.decode("utf-8").split("\t", 1)[1].split()[0]
        return field.split(":", 1)[0]
    except (UnicodeError, IndexError):
        return None


def _invalid(n: int, raw: bytes, family: str, issue: str,
             identifier: str | None = None, label: str | None = None) -> NativeRecord:
    return NativeRecord(n, identifier, family, label, raw, valid=False,
                        issues=(issue,))


def _parse_one(n: int, raw: bytes, text: str, family: str | None = None,
               declared_label: str | None = None) -> NativeRecord:
    inferred = _family_prefix(raw)
    declared = family or inferred
    try:
        line = raw.decode("utf-8", "strict")
    except UnicodeError:
        return _invalid(n, raw, declared, "invalid_utf8", label=declared_label)
    if "\n" in line or "\r" in line:
        return _invalid(n, raw, declared, "embedded_line_break", label=declared_label or _label_hint(raw))
    fields = line.split("\t")
    ident = fields[0] if fields else None
    hint = _label_hint(raw) or declared_label
    if not ident or not IDENTIFIER.fullmatch(ident) or len(fields) < 2:
        return _invalid(n, raw, declared, "invalid_record_header", ident, hint)
    inferred = ident[0] if ident[0] in "TRE" else "AUX"
    if family is not None and family != inferred:
        return _invalid(n, raw, declared, "declared_family_mismatch", ident, hint)
    if inferred == "AUX":
        # Auxiliary/unknown-family payloads are opaque, not converted into R/E.
        return NativeRecord(n, ident, "AUX", hint, raw)
    try:
        if inferred == "T":
            # Third reference field is informational. Brat's pinned parser can
            # fill missing reference text; here it is retained as missing, with
            # a diagnostic, instead of mutating the annotation or Gold offsets.
            fields = line.split("\t", 2)
            label, span = fields[1].split(" ", 1)
            if not LABEL.fullmatch(label):
                raise ValueError("invalid_T_label")
            segments = tuple(tuple(int(x) for x in part.split())
                             for part in span.split(";"))
            if not segments or any(len(s) != 2 for s in segments):
                raise ValueError("invalid_segment_syntax")
            if any(a < 0 or b < a or b > len(text) for a, b in segments):
                raise ValueError("invalid_offset_range")
            # Segment order is part of identity; do not sort, merge or hull it.
            if any(max(a, c) < min(b, d) for i, (a, b) in enumerate(segments)
                   for c, d in segments[:i]):
                raise ValueError("overlapping_segments")
            reference = " ".join(text[a:b] for a, b in segments)
            surface = fields[2] if len(fields) == 3 else None
            issues = (("missing_reference_text",) if surface is None else
                      () if reference == surface else ("reference_text_mismatch",))
            return NativeRecord(n, ident, "T", label, raw, segments=segments,
                                surface=surface, issues=issues)
        if len(fields) != 2:
            raise ValueError("R_E_requires_two_tab_fields")
        tokens = fields[1].split()
        if inferred == "R":
            if len(tokens) != 3 or not LABEL.fullmatch(tokens[0]):
                raise ValueError("invalid_relation_syntax")
            arguments = [token.split(":", 1) for token in tokens[1:]]
            if any(len(a) != 2 for a in arguments):
                raise ValueError("invalid_relation_argument")
            if Counter(a[0] for a in arguments) != Counter({"Arg1": 1, "Arg2": 1}):
                raise ValueError("relation_requires_Arg1_Arg2_once")
            args = dict(arguments)
            if any(not ENDPOINT.fullmatch(v) for v in args.values()):
                raise ValueError("invalid_relation_endpoint_identifier")
            return NativeRecord(n, ident, "R", tokens[0], raw,
                                arg1=args["Arg1"], arg2=args["Arg2"])
        if not tokens:
            raise ValueError("missing_event_trigger")
        head = tokens[0].split(":", 1)
        if len(head) != 2 or not LABEL.fullmatch(head[0]) or not re.fullmatch(r"T[0-9]+", head[1]):
            raise ValueError("invalid_event_trigger")
        roles = []
        for token in tokens[1:]:
            role = token.split(":", 1)
            if len(role) != 2 or not LABEL.fullmatch(role[0]) or not ENDPOINT.fullmatch(role[1]):
                raise ValueError("invalid_event_role_endpoint")
            roles.append(tuple(role))
        return NativeRecord(n, ident, "E", head[0], raw,
                            trigger=head[1], roles=tuple(roles))
    except (ValueError, TypeError) as exc:
        return _invalid(n, raw, inferred, str(exc), ident, hint)


def _resolve(records: Sequence[NativeRecord], *, gold: bool,
             registry: Registry) -> tuple[tuple[NativeRecord, ...], tuple[str, ...]]:
    counts = Counter(r.identifier for r in records if r.identifier and r.identifier != "*")
    out = []
    for r in records:
        issues = list(r.issues)
        invalid = not r.valid
        if r.identifier and r.identifier != "*" and counts[r.identifier] > 1:
            issues.append("duplicate_identifier")
            invalid = True
        if not gold and r.valid:
            known = (registry.entity_labels | registry.event_labels if r.family == "T"
                     else registry.relation_labels if r.family == "R"
                     else registry.event_labels if r.family == "E" else None)
            if known is not None and r.label not in known:
                issues.append("unsupported_prediction_label")
                invalid = True
            if r.family == "E" and any(not registry.role_allowed(role) for role, _ in r.roles):
                issues.append("unsupported_prediction_role")
                invalid = True
        out.append(replace(r, valid=not invalid, issues=tuple(issues)))
    # Invalid dependencies propagate through arbitrary event chains/cycles.
    while True:
        index = {r.identifier: r for r in out if r.valid and r.identifier}
        changed = False
        new = []
        for r in out:
            issue = None
            if r.valid and r.family in {"R", "E"}:
                endpoints = ((r.arg1, r.arg2) if r.family == "R" else
                             (r.trigger,) + tuple(e for _, e in r.roles))
                if any(e not in index or index[e].family not in {"T", "E"}
                       for e in endpoints):
                    issue = "dangling_or_invalid_dependency"
                elif r.family == "E" and index[r.trigger].family != "T":
                    issue = "event_trigger_not_T"
            if issue:
                r = replace(r, valid=False, issues=r.issues + (issue,))
                changed = True
            new.append(r)
        out = new
        if not changed:
            break
    fatal = [f"{r.occurrence}:{r.identifier}:{','.join(r.issues)}" for r in out if not r.valid]
    if gold and fatal:
        raise GoldIntegrityError("malformed or ambiguous Gold: " + "; ".join(fatal))
    warnings = tuple(f"{r.occurrence}:{r.identifier}:{issue}"
                     for r in out for issue in r.issues if r.valid)
    return tuple(out), warnings


def parse_document(source_text: bytes | str, annotations: bytes | None, *,
                   gold: bool = False, registry: Registry | None = None) -> NativeDocument:
    """Parse unchanged UTF-8 bytes, never files. None is missing prediction only.

    Gold missing/corrupt records raise GoldIntegrityError. Invalid prediction
    lines remain one FP record each in their inferred T/R/E family. UTF-8 errors
    with no recognizable family remain explicit output failures; no arbitrary
    artifact-FP penalty is manufactured. Use parse_emissions to declare family
    when even an invalid record header is not recoverable.
    """
    reg = registry or Registry()
    source_bytes, text = _source(source_text)
    if annotations is None:
        if gold:
            raise GoldIntegrityError("missing Gold annotations")
        return NativeDocument(source_bytes, text, None, (), False, reg,
                              "missing_prediction", ("missing_prediction_artifact",))
    if not isinstance(annotations, bytes):
        raise TypeError("annotations must be bytes or None")
    records = []
    # Split only physical LF records. CRLF is preserved in annotation_bytes;
    # one terminal CR is syntax, not an alteration of source text offsets.
    for line in annotations.split(b"\n"):
        raw = line[:-1] if line.endswith(b"\r") else line
        if not raw.strip():
            continue
        records.append(_parse_one(len(records), raw, text))
    resolved, warnings = _resolve(records, gold=gold, registry=reg)
    failures = tuple(f"unclassified_output_record:{r.occurrence}" for r in resolved
                     if not r.valid and r.family == "AUX")
    status = "output_failure" if failures else "present_with_invalid_records" if any(
        not r.valid for r in resolved) else "present"
    return NativeDocument(source_bytes, text, annotations, resolved, gold, reg,
                          status, warnings + failures)


def parse_emissions(source_text: bytes | str, emissions: Iterable[Emission], *,
                    registry: Registry | None = None) -> NativeDocument:
    """Prediction API; exactly one family-declared occurrence per emission."""
    reg = registry or Registry()
    raw, text = _source(source_text)
    records = []
    for n, emission in enumerate(emissions):
        if not isinstance(emission, Emission) or emission.family not in {"T", "R", "E", "AUX"}:
            raise TypeError("emission must be Emission with T/R/E/AUX family")
        if not isinstance(emission.payload, bytes):
            raise TypeError("emission payload must be bytes")
        if emission.declared_label is not None and not isinstance(emission.declared_label, str):
            raise TypeError("declared_label must be str or None")
        records.append(_parse_one(n, emission.payload, text, emission.family, emission.declared_label))
    resolved, warnings = _resolve(records, gold=False, registry=reg)
    return NativeDocument(raw, text, None, resolved, False, reg,
                          "present_with_invalid_records" if any(not r.valid for r in resolved)
                          else "present", warnings)


def parse_prediction_records(source_text: bytes | str, objects: Sequence[Mapping], *,
                             registry: Registry | None = None) -> NativeDocument:
    """Stable JSON-object emitter interface, retaining malformed family records.

    T: family,id,type,segments (list of [start,end]),text.
    R: family,id,type,arg1,arg2. E: family,id,type,trigger,roles (list of
    {role,target}; never a dictionary keyed by role). AUX: family,raw_brat.
    Exact keys are required: malformed shapes become one invalid FP emission in
    the declared T/R/E family; the raw JSON is preserved. A missing/unknown family
    is an output-interface failure and raises rather than inventing its family.
    The original JSON records must be archived by the caller alongside outputs;
    ``document_json`` supplies lossless parsed-record bytes and native fields.
    """
    emissions = []
    for obj in objects:
        if not isinstance(obj, Mapping) or obj.get("family") not in {"T", "R", "E", "AUX"}:
            raise ValueError("each JSON emission needs an explicit T/R/E/AUX family")
        family = obj["family"]
        try:
            if family == "AUX":
                if set(obj) != {"family", "raw_brat"} or not isinstance(obj["raw_brat"], str):
                    raise ValueError("AUX JSON schema")
                raw = obj["raw_brat"].encode("utf-8")
            else:
                keys = ({"family", "id", "type", "segments", "text"} if family == "T"
                        else {"family", "id", "type", "arg1", "arg2"} if family == "R"
                        else {"family", "id", "type", "trigger", "roles"})
                if set(obj) != keys or any(not isinstance(obj[k], str) for k in ("id", "type")):
                    raise ValueError("JSON keys/id/type schema")
                # Validate JSON lexical atoms BEFORE rendering Brat. Whitespace,
                # TAB/colon payloads must never inject an alternate span/role or
                # silently become a different valid native target.
                if not re.fullmatch(family + r"[0-9]+", obj["id"]) or not LABEL.fullmatch(obj["type"]):
                    raise ValueError("JSON id/type lexical schema")
                if family == "T":
                    spans = obj["segments"]
                    if (not isinstance(spans, list) or not spans or
                            not isinstance(obj["text"], str) or
                            any(not isinstance(s, list) or len(s) != 2 or
                                any(type(v) is not int for v in s) for s in spans)):
                        raise ValueError("T JSON schema")
                    offsets = ";".join(f"{a} {b}" for a, b in spans)
                    line = f"{obj['id']}\t{obj['type']} {offsets}\t{obj['text']}"
                elif family == "R":
                    if any(not isinstance(obj[k], str) or not ENDPOINT.fullmatch(obj[k])
                           for k in ("arg1", "arg2")):
                        raise ValueError("R JSON schema")
                    line = f"{obj['id']}\t{obj['type']} Arg1:{obj['arg1']} Arg2:{obj['arg2']}"
                else:
                    if (not isinstance(obj["trigger"], str) or
                            not re.fullmatch(r"T[0-9]+", obj["trigger"]) or
                            not isinstance(obj["roles"], list)):
                        raise ValueError("E JSON schema")
                    for role in obj["roles"]:
                        if (not isinstance(role, Mapping) or set(role) != {"role", "target"} or
                                any(not isinstance(role[k], str) for k in ("role", "target")) or
                                not LABEL.fullmatch(role["role"]) or
                                not ENDPOINT.fullmatch(role["target"])):
                            raise ValueError("E role JSON schema")
                    args = " ".join(f"{r['role']}:{r['target']}" for r in obj["roles"])
                    line = f"{obj['id']}\t{obj['type']}:{obj['trigger']}" + (" " + args if args else "")
                raw = line.encode("utf-8")
        except (ValueError, TypeError, KeyError, UnicodeError):
            try:
                raw = json.dumps(dict(obj), ensure_ascii=True, allow_nan=False,
                                 sort_keys=True).encode("utf-8")
            except (TypeError, ValueError):
                raw = repr(obj).encode("utf-8", "backslashreplace")
        emissions.append(Emission(family, raw, obj.get("type") if isinstance(obj.get("type"), str) else None))
    return parse_emissions(source_text, emissions, registry=registry)


def document_json(doc: NativeDocument) -> dict:
    """JSON-serializable lossless representation; no real-file read/write CLI."""
    return {"source_utf8_base64": base64.b64encode(doc.source_bytes).decode("ascii"),
            "annotation_utf8_base64": (base64.b64encode(doc.annotation_bytes).decode("ascii")
                                       if doc.annotation_bytes is not None else None),
            "gold": doc.gold, "artifact_status": doc.artifact_status,
            "diagnostics": list(doc.diagnostics),
            "records": [{"occurrence": r.occurrence, "id": r.identifier, "family": r.family,
                         "type": r.label, "segments": [list(s) for s in r.segments],
                         "text": r.surface, "arg1": r.arg1, "arg2": r.arg2,
                         "trigger": r.trigger,
                         "roles": [{"role": role, "target": target} for role, target in r.roles],
                         "raw_base64": base64.b64encode(r.raw).decode("ascii"),
                         "valid": r.valid, "issues": list(r.issues)} for r in doc.records]}


def _endpoint_projected(doc: NativeDocument, identifier: str) -> tuple:
    r = doc.by_id[identifier]
    return r.mention_key() if r.family == "T" else doc.by_id[r.trigger].mention_key()


def _relation_key(doc: NativeDocument, r: NativeRecord) -> tuple:
    return r.label, _endpoint_projected(doc, r.arg1), _endpoint_projected(doc, r.arg2)


def _node_match(a: dict, b: dict) -> bool:
    return a["identity"] == b["identity"] and a["root"] == b["root"]


def _edge_match(a: dict, b: dict) -> bool:
    # NetworkX categorical_multiedge_match uses sets; own Counter keeps parallel
    # raw-role occurrences and trigger/argument edge-kind multiplicities exact.
    return Counter(v["identity"] for v in a.values()) == Counter(v["identity"] for v in b.values())


def rooted_graph(doc: NativeDocument, root: NativeRecord) -> nx.MultiDiGraph:
    """Finite closure graph; IDs are node handles, never semantic attributes.

    A visited set supports forward refs, nested E and cycles. Distinct IDs with
    identical T spans stay distinct nodes. R roots are supported only for the
    separately named full-graph endpoint metric. Source sharing is not flattened.
    """
    if not root.valid or root.family not in {"E", "R"}:
        raise ValueError("root must be valid E/R")
    index = doc.by_id
    graph = nx.MultiDiGraph()
    todo = [root.identifier]
    visited = set()
    while todo:
        ident = todo.pop()
        if ident in visited:
            continue
        visited.add(ident)
        r = index[ident]
        identity = ("T", r.mention_key()) if r.family == "T" else (r.family, r.label)
        graph.add_node(ident, identity=identity, root=ident == root.identifier)
        if r.family == "E":
            edges = [(r.trigger, ("trigger",))] + [(endpoint, ("role", role))
                                                       for role, endpoint in r.roles]
        elif r.family == "R":
            edges = [(r.arg1, ("arg", "Arg1")), (r.arg2, ("arg", "Arg2"))]
        else:
            edges = []
        for endpoint, label in edges:
            graph.add_edge(ident, endpoint, identity=label)
            todo.append(endpoint)
    return graph


@dataclass
class _Budget:
    limit: int | None
    used: int = 0

    def tick(self) -> None:
        self.used += 1
        if self.limit is not None and self.used > self.limit:
            raise MatchingBudgetExceeded(f"exact search exceeded {self.limit} steps")


class _CountedMultiDiGraphMatcher(nx.algorithms.isomorphism.MultiDiGraphMatcher):
    """Include each VF2 candidate-node feasibility state in the exact budget."""

    def __init__(self, *args, budget: _Budget, **kwargs):
        self._budget = budget
        super().__init__(*args, **kwargs)

    def syntactic_feasibility(self, node1, node2):
        self._budget.tick()
        return super().syntactic_feasibility(node1, node2)


def exact_graph_match(pred: NativeDocument, gold: NativeDocument, family: str, *,
                      max_search_steps: int | None = DEFAULT_MAX_SEARCH_STEPS) -> dict:
    """Exact maximum root credit with a global, injective identifier mapping.

    Candidate root isomorphisms use labeled directed multigraph VF2; a finite
    branch-and-bound search combines only compatible node maps. Unmatched roots
    do not impose extra sharing constraints. No score is returned on exhaustion.
    This is annotation identity recovery, not physical truth or a model ceiling.
    """
    if family not in {"E", "R"}:
        raise ValueError("family must be E/R")
    if max_search_steps is not None and (type(max_search_steps) is not int or max_search_steps < 1):
        raise ValueError("max_search_steps must be a positive integer or None")
    p = pred.family(family, valid_only=True)
    g = gold.family(family, valid_only=True)
    pg = {r.identifier: rooted_graph(pred, r) for r in p}
    gg = {r.identifier: rooted_graph(gold, r) for r in g}
    budget = _Budget(max_search_steps)
    choices = {}
    for pr in p:
        opts = []
        for gr in g:
            budget.tick()
            if pr.label != gr.label:
                continue
            a, b = pg[pr.identifier], gg[gr.identifier]
            if len(a) != len(b) or a.number_of_edges() != b.number_of_edges():
                continue
            matcher = _CountedMultiDiGraphMatcher(
                a, b, budget=budget, node_match=_node_match, edge_match=_edge_match)
            for mapping in matcher.isomorphisms_iter():
                budget.tick()
                opts.append((gr.identifier, mapping))
        choices[pr.identifier] = opts
    order = sorted(p, key=lambda r: (len(choices[r.identifier]), r.occurrence))
    best_pairs = []
    best_map = {}

    def visit(k: int, used_roots: set, mapping: dict, inverse: dict, pairs: list) -> None:
        nonlocal best_pairs, best_map
        budget.tick()
        if len(pairs) + len(order) - k <= len(best_pairs):
            return
        if k == len(order):
            best_pairs, best_map = list(pairs), dict(mapping)
            return
        ident = order[k].identifier
        for gold_root, extension in choices[ident]:
            if gold_root in used_roots:
                continue
            if any((a in mapping and mapping[a] != b) or
                   (b in inverse and inverse[b] != a) for a, b in extension.items()):
                continue
            merged, inv = dict(mapping), dict(inverse)
            merged.update(extension)
            inv.update({b: a for a, b in extension.items()})
            visit(k + 1, used_roots | {gold_root}, merged, inv,
                  pairs + [(ident, gold_root)])
        visit(k + 1, used_roots, mapping, inverse, pairs)

    visit(0, set(), {}, {}, [])
    return {"tp": len(best_pairs), "matched_roots": [list(pair) for pair in best_pairs],
            "identifier_mapping": best_map, "search_steps": budget.used,
            "search_step_limit": max_search_steps,
            "sharing_scope": "consistent_across_all_matched_root_closures"}


def root_local_event_match(pred: NativeDocument, gold: NativeDocument, *,
                           max_search_steps: int | None = DEFAULT_MAX_SEARCH_STEPS) -> dict:
    """Descriptive root-local diagnostic; no cross-root sharing constraint.

    Each root closure itself preserves sharing/cycles. Maximum bipartite matching
    is one-to-one over E occurrences. This is neither the primary global graph
    definition nor an exact reproduction of the author's dictionary scorer.
    """
    if max_search_steps is not None and (type(max_search_steps) is not int or max_search_steps < 1):
        raise ValueError("max_search_steps must be a positive integer or None")
    budget = _Budget(max_search_steps)
    p = pred.family("E", valid_only=True)
    g = gold.family("E", valid_only=True)
    graph = nx.Graph()
    for r in p:
        graph.add_node(("P", r.identifier), bipartite=0)
    for r in g:
        graph.add_node(("G", r.identifier), bipartite=1)
    pg = {r.identifier: rooted_graph(pred, r) for r in p}
    gg = {r.identifier: rooted_graph(gold, r) for r in g}
    for pr in p:
        for gr in g:
            budget.tick()
            matcher = _CountedMultiDiGraphMatcher(
                pg[pr.identifier], gg[gr.identifier], budget=budget,
                node_match=_node_match, edge_match=_edge_match)
            if matcher.is_isomorphic():
                graph.add_edge(("P", pr.identifier), ("G", gr.identifier))
    matched = nx.algorithms.bipartite.maximum_matching(
        graph, top_nodes={("P", r.identifier) for r in p})
    pairs = sorted((a[1], b[1]) for a, b in matched.items() if a[0] == "P")
    return {"tp": len(pairs), "matched_roots": [list(pair) for pair in pairs],
            "search_steps": budget.used, "search_step_limit": max_search_steps,
            "sharing_scope": "within_each_root_closure_only"}


def _counts(tp: int, pred_count: int, gold_count: int) -> dict:
    fp, fn = pred_count - tp, gold_count - tp
    if min(tp, fp, fn) < 0:
        raise AssertionError("count conservation failure")
    return {"tp": tp, "fp": fp, "fn": fn,
            "prediction_count": pred_count, "gold_count": gold_count,
            "precision": tp / pred_count if pred_count else 0.0,
            "recall": tp / gold_count if gold_count else 0.0,
            "f1": 2 * tp / (pred_count + gold_count) if pred_count + gold_count else 0.0}


def _multiset_metric(pred: Sequence[NativeRecord], gold: Sequence[NativeRecord],
                     pred_key, gold_key) -> dict:
    pc = Counter(pred_key(r) for r in pred if r.valid)
    gc = Counter(gold_key(r) for r in gold)
    return _counts(sum((pc & gc).values()), len(pred), len(gold))


def _category(r: NativeRecord, registry: Registry) -> str:
    # Schema support may be prospectively extended, but the native eight/nine
    # configured populations and OTHER_NATIVE reporting do not silently expand.
    known = (ENTITY_LABELS | EVENT_LABELS if r.family == "T"
             else RELATION_LABELS if r.family == "R" else EVENT_LABELS)
    return r.label if r.label in known else "OTHER_NATIVE"


def score_document(pred: NativeDocument, gold: NativeDocument, *,
                   include_full_relation_graph: bool = True,
                   max_search_steps: int | None = DEFAULT_MAX_SEARCH_STEPS) -> dict:
    """Score the complete supplied raw Gold T/R/E population, with invalid FP.

    P/R/F1 are zero on their zero denominators. Every native label is additionally
    accounted for in an all-native family metric and per-category counts; eight
    entity/trigger/combined-nine scores intentionally have different populations.
    Both documents must share original bytes and Registry. Auxiliary records are
    retained/reported without silently claiming auxiliary-target evaluation.
    """
    if not gold.gold or pred.gold:
        raise ValueError("requires Gold and prediction documents in their respective modes")
    if pred.source_bytes != gold.source_bytes or pred.registry != gold.registry:
        raise ValueError("source/Registry mismatch")
    reg = gold.registry
    pt, gt = pred.family("T"), gold.family("T")
    metrics = {}
    groups = {
        "entities_eight": ENTITY_LABELS,
        "triggers": EVENT_LABELS,
        "combined_text_bound": ENTITY_LABELS | EVENT_LABELS,
    }
    for name, labels in groups.items():
        metrics[name] = _multiset_metric([r for r in pt if r.label in labels],
                                        [r for r in gt if r.label in labels],
                                        lambda r: r.mention_key(), lambda r: r.mention_key())
    metrics["text_bound_all_native"] = _multiset_metric(
        pt, gt, lambda r: r.mention_key(), lambda r: r.mention_key())
    pr, gr = pred.family("R"), gold.family("R")
    metrics["relations_trigger_projected"] = _multiset_metric(
        pr, gr, lambda r: _relation_key(pred, r), lambda r: _relation_key(gold, r))
    matches = {}
    for family, metric in (("E", "events_full_graph"), ("R", "relations_full_graph")):
        if family == "R" and not include_full_relation_graph:
            continue
        matching = exact_graph_match(pred, gold, family, max_search_steps=max_search_steps)
        matches[metric] = matching
        metrics[metric] = _counts(matching["tp"], len(pred.family(family)), len(gold.family(family)))
    local = root_local_event_match(pred, gold, max_search_steps=max_search_steps)
    matches["events_root_local_diagnostic"] = local
    metrics["events_root_local_diagnostic"] = _counts(local["tp"], len(pred.family("E")), len(gold.family("E")))
    by_label = {}
    for family, metric in (("T", "text_bound_all_native"), ("R", "relations_trigger_projected"),
                           ("E", "events_full_graph")):
        pred_records, gold_records = pred.family(family), gold.family(family)
        labels = sorted({_category(r, reg) for r in pred_records + gold_records})
        rows = {}
        for label in labels:
            p = [r for r in pred_records if _category(r, reg) == label]
            g = [r for r in gold_records if _category(r, reg) == label]
            if family == "T":
                rows[label] = _multiset_metric(p, g, lambda r: r.mention_key(), lambda r: r.mention_key())
            elif family == "R":
                rows[label] = _multiset_metric(p, g, lambda r: _relation_key(pred, r),
                                              lambda r: _relation_key(gold, r))
            else:
                pindex = pred.by_id
                tp = sum(_category(pindex[a], reg) == label
                         for a, _ in matches[metric]["matched_roots"])
                rows[label] = _counts(tp, len(p), len(g))
        if any(sum(row[k] for row in rows.values()) != metrics[metric][k]
               for k in ("tp", "fp", "fn", "prediction_count", "gold_count")):
            raise AssertionError("label/family count conservation failure")
        by_label[family] = rows
    return {"metrics": metrics, "by_label": by_label, "graph_matches": matches,
            "invalid_prediction_records": [
                {"occurrence": r.occurrence, "family": r.family, "identifier": r.identifier,
                 "label": r.label, "issues": list(r.issues)} for r in pred.records if not r.valid],
            "auxiliary": {"gold_count": len(gold.family("AUX")),
                          "prediction_count": len(pred.family("AUX")),
                          "scope": "losslessly_preserved_not_scored_as_T_R_E"},
            "artifact_status": pred.artifact_status,
            "gold_diagnostics": list(gold.diagnostics), "prediction_diagnostics": list(pred.diagnostics),
            "source_sha256": hashlib.sha256(gold.source_bytes).hexdigest(),
            "gold_annotation_sha256": hashlib.sha256(gold.annotation_bytes).hexdigest()}


def score_population(inventory: Sequence[str], gold_sources: Mapping[str, GoldSource],
                     predictions: Mapping[str, bytes | Sequence[Emission] | None], *,
                     registry: Registry | None = None,
                     include_full_relation_graph: bool = True,
                     max_search_steps: int | None = DEFAULT_MAX_SEARCH_STEPS) -> dict:
    """Manifest-first population; missing predictions are empty, never skipped.

    Requires exact Gold inventory and refuses out-of-inventory predictions.
    Every Gold document is parsed/validated before any document scoring; malformed
    Gold raises instead of returning a reduced-population result. This API does
    not certify source barriers: callers must freeze/check their own input bytes.
    """
    if not inventory or any(not isinstance(i, str) or not i for i in inventory):
        raise ValueError("nonempty inventory with string IDs required")
    if len(inventory) != len(set(inventory)):
        raise ValueError("duplicate inventory ID")
    if set(gold_sources) != set(inventory):
        raise GoldIntegrityError("Gold inventory mismatch")
    if not set(predictions).issubset(set(inventory)):
        raise ValueError("out-of-inventory prediction")
    reg = registry or Registry()
    gold = {i: parse_document(gold_sources[i].text_utf8, gold_sources[i].annotation_utf8,
                              gold=True, registry=reg) for i in inventory}
    reports = {}
    for i in inventory:
        payload = predictions.get(i)
        pred = (parse_emissions(gold_sources[i].text_utf8, payload, registry=reg)
                if isinstance(payload, (tuple, list)) else
                parse_document(gold_sources[i].text_utf8, payload, registry=reg))
        reports[i] = score_document(pred, gold[i], include_full_relation_graph=include_full_relation_graph,
                                    max_search_steps=max_search_steps)
    totals = {}
    for name in next(iter(reports.values()))["metrics"]:
        tp = sum(r["metrics"][name]["tp"] for r in reports.values())
        pc = sum(r["metrics"][name]["prediction_count"] for r in reports.values())
        gc = sum(r["metrics"][name]["gold_count"] for r in reports.values())
        totals[name] = _counts(tp, pc, gc)
    return {"inventory": list(inventory), "by_document": reports, "totals": totals,
            "missing_predictions": [i for i in inventory if predictions.get(i) is None],
            "scope": "complete_supplied_native_T_R_E_not_auxiliary_or_physical_truth",
            "matching": "exact_one_to_one_globally_sharing_consistent_root_graphs",
            "zero_denominator": "P_R_F1_zero"}
