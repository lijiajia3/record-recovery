#!/usr/bin/env python3
"""TRAIN-only native schema inventory; no fit, predictions or model scoring.

The CLI is deliberately closed until a separately reviewed authorization gate
binds this source, the reviewed parser and completed source-text provenance.
Only manifest-declared 0201..1000 annotation/text pairs can be opened. The
INVENTED fixture gate is separate and cannot authorize a real acquisition.
"""
from __future__ import annotations

import argparse
import base64
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import sys
import types

TRAIN = tuple(range(201, 1001))
SPLITS = {"train": list(TRAIN), "dev": list(range(101, 201)), "test": list(range(1, 101))}
PARSER_SHA = "b59306e02e7b8132e16366f70c729ee63df83ed10ee39e25ab0c38f74e321b53"
PARSER_REVIEW_SHA = "4fb146e0aa5210ec0b551b127292f475c6594944ab2505a9080cd991e3208e31"
ACTUAL_ACQUISITION_SHA = "a38f1637af74b15246b6b028d3d58b5836094e905d25845e5922668031fee1c7"
PARSER_PATH = "src/sccomics_native_graph.py"
SELF_PATH = "src/audit_sccomics_train_schema_only.py"
PARSER_REVIEW_PATH = "research/round4_sccomics_native_graph_independent_source_review.json"
BRIDGE_PATH = "research/round4_sccomics_train_schema_bridge_protocol.json"
ACTUAL_GATE_PATH = "research/round4_sccomics_train_schema_authorized_gate.json"
ACTUAL_WRAPPER_REVIEW_PATH = "research/round4_sccomics_train_schema_independent_source_review.json"
ACTUAL_ACQUISITION_PATH = "data/sccomics_round4/source_v3/source_acquisition_manifest.json"
ACTUAL_TEXT_PATHS = {
    "audit": "research/round4_sccomics_text_provenance_actual/audit.json",
    "output_manifest": "research/round4_sccomics_text_provenance_actual/output_manifest.json",
    "root_crosscheck": "research/round4_sccomics_actual_text_provenance_root_crosscheck.json"}
ACTUAL_TEXT_SHA = {
    "audit": "0ccad5261aed0e37ec052d049fc82e8ea7d0c4b1881c08dd1be7acb529ed3eb1",
    "output_manifest": "dfbc7ac80868bd66ff5bbadc5f87b2e9aca179f46b1ad63b643a0fe1c67c250d",
    "root_crosscheck": "494bd5d5b6284c401b7b67e75287b376985f06ea7255729e4d33880958b4e3d5"}
INVENTED_GATE_PATH = "research/INVENTED_authorized_gate.json"
INVENTED_WRAPPER_PATH = "research/INVENTED_wrapper_review.json"
INVENTED_ACQUISITION_PATH = "data/INVENTED_source/source_acquisition_manifest.json"
INVENTED_TEXT_PATHS = {k: "research/INVENTED_text_completion/" + name for k, name in
                       (("audit", "audit.json"), ("output_manifest", "output_manifest.json"),
                        ("root_crosscheck", "root_crosscheck.json"))}
TEXT_OUTPUT_NAMES = frozenset({"all_qualifying_pairs.jsonl", "all_source_units.jsonl",
                             "test_source_text_clusters.json", "test_source_text_clusters.jsonl", "audit.json"})


class AuditBlocked(ValueError):
    pass


def require(condition, message):
    if not condition:
        raise AuditBlocked(message)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def json_bytes(obj):
    return (json.dumps(obj, ensure_ascii=False, sort_keys=True, indent=2,
                       allow_nan=False) + "\n").encode("utf-8")


def unique_object(pairs):
    result = {}
    for k, v in pairs:
        require(k not in result, f"duplicate JSON key: {k}")
        result[k] = v
    return result


def load_json(raw):
    return json.loads(raw.decode("utf-8", "strict"), object_pairs_hook=unique_object,
                      parse_constant=lambda x: (_ for _ in ()).throw(AuditBlocked(x)))


class Access:
    """Central no-glob/no-archive reader, input byte bindings and final recheck."""
    def __init__(self, root):
        self.root = Path(os.path.abspath(os.fspath(root)))
        self._no_links(self.root)
        require(self.root.is_dir(), "root must be an existing directory")
        self.bindings = {}
        self.log = []
        self.allowed_data = set()

    @staticmethod
    def _no_links(path):
        chain = list(reversed(path.parents)) + [path]
        for part in chain:
            require(not part.is_symlink(), f"symlink/ancestor rejected: {part}")

    def path(self, value):
        value = os.fspath(value)
        require(not any(c in value for c in "*?[]"), "glob characters rejected")
        p = Path(value)
        require(".." not in p.parts, "parent traversal rejected")
        p = p if p.is_absolute() else self.root / p
        p = Path(os.path.abspath(p))
        require(p.is_relative_to(self.root), f"path outside explicit root: {p}")
        self._no_links(p)
        return p

    def relative(self, value):
        return str(self.path(value).relative_to(self.root))

    @staticmethod
    def _open_read_fd(path):
        # Walk from / through pinned directory descriptors. Leaf-only
        # O_NOFOLLOW would still leave an ancestor replacement race.
        directory = os.open(path.anchor, os.O_RDONLY | os.O_DIRECTORY | getattr(os, "O_NOFOLLOW", 0))
        try:
            for component in path.parts[1:-1]:
                next_directory = os.open(component, os.O_RDONLY | os.O_DIRECTORY |
                                         getattr(os, "O_NOFOLLOW", 0), dir_fd=directory)
                os.close(directory)
                directory = next_directory
            return os.open(path.name, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0), dir_fd=directory)
        finally:
            os.close(directory)

    def read(self, value, *, expected=None, size=None, purpose="metadata", capture=True):
        p = self.path(value)
        rel = str(p.relative_to(self.root))
        require(p.suffix.lower() not in {".zip", ".gz", ".tar", ".bz2", ".xz"}, "archives forbidden")
        if p.suffix.lower() == ".ann" or "raw_annotations" in p.parts or "raw_text" in p.parts:
            require(rel in self.allowed_data, f"non-TRAIN or unauthorized raw data read: {rel}")
        fd = self._open_read_fd(p)
        with os.fdopen(fd, "rb") as stream:
            before = os.fstat(stream.fileno())
            require(stat.S_ISREG(before.st_mode), f"not a regular file: {rel}")
            raw = stream.read()
            after = os.fstat(stream.fileno())
        self._no_links(p)
        require((before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) ==
                (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns), f"changed during read: {rel}")
        digest = sha(raw)
        require(expected is None or digest == expected, f"SHA mismatch: {rel}")
        require(size is None or len(raw) == size, f"byte size mismatch: {rel}")
        binding = {"path": rel, "sha256": digest, "bytes": len(raw)}
        if capture:
            require(rel not in self.bindings or self.bindings[rel] == binding,
                    f"changed between reads: {rel}")
            self.bindings[rel] = binding
        self.log.append({**binding, "purpose": purpose})
        return raw

    def final_recheck(self):
        for b in tuple(self.bindings.values()):
            self.read(b["path"], expected=b["sha256"], size=b["bytes"],
                      purpose="final_input_rehash", capture=False)


def parser_from_captured(raw):
    # Execute only the exact bytes just checked against the independent review.
    # Ordinary import would introduce a separate, unbound second file read.
    name = "_sccomics_verified_train_schema_native"
    module = types.ModuleType(name)
    module.__file__ = PARSER_PATH
    sys.modules[name] = module
    exec(compile(raw, PARSER_PATH, "exec"), module.__dict__)
    return module


def metadata_gates(access, acquisition_path, protocol_path):
    own = access.read(SELF_PATH, purpose="auditor_source")
    require(Path(__file__).absolute() == access.path(SELF_PATH), "running source is outside explicit root")
    # These paths identify predeclared metadata duties, not arbitrary files
    # that a gate can rename/reclassify as metadata to expand read permission.
    protocol_rel = access.relative(protocol_path)
    require(protocol_rel in {ACTUAL_GATE_PATH, INVENTED_GATE_PATH}, "protocol path is not a declared source gate duty")
    protocol_raw = access.read(protocol_path, purpose="captured_authorization_gate")
    gate = load_json(protocol_raw)
    invented = gate.get("synthetic_fixture") is True
    require(protocol_rel == (INVENTED_GATE_PATH if invented else ACTUAL_GATE_PATH), "gate duty/fixture scope mismatch")
    expected_status = "INVENTED_authorized_train_schema_only" if invented else "authorized_train_schema_only"
    require(gate.get("status") == expected_status and gate.get("actual_execution_authorized") is True,
            "TRAIN schema audit is not authorized by this gate")
    require(gate.get("train_semantic_ids") == list(TRAIN), "gate must specify exactly IDs201..1000")
    for field in ("dev_semantics_allowed", "test_semantics_allowed", "model_training_or_inference_allowed",
                  "paid_api_or_credentials_allowed"):
        require(gate.get(field) is False, f"gate requires explicit false: {field}")
    require(gate.get("audit_script_sha256") == sha(own), "auditor source not prospectively bound")
    file_map = gate.get("file_sha256")
    require(isinstance(file_map, dict), "gate requires input file_sha256 map")
    acq_rel = access.relative(acquisition_path)
    require(acq_rel == (INVENTED_ACQUISITION_PATH if invented else ACTUAL_ACQUISITION_PATH),
            "acquisition path is not its declared metadata duty")
    required = {SELF_PATH, PARSER_PATH, PARSER_REVIEW_PATH, BRIDGE_PATH, acq_rel}
    completion = gate.get("text_completion", {})
    require(completion == (INVENTED_TEXT_PATHS if invented else ACTUAL_TEXT_PATHS),
            "text completion paths are not declared metadata duties")
    required.update(completion.values())
    wrapper_path = gate.get("wrapper_independent_review_path", "")
    require(wrapper_path == (INVENTED_WRAPPER_PATH if invented else ACTUAL_WRAPPER_REVIEW_PATH),
            "wrapper review path is not its declared source review duty")
    required.add(wrapper_path)
    # Closed set: no weights, credentials, checkpoint, caches, annotations,
    # archives or extra files may become readable by a metadata declaration.
    require(set(file_map) == required, "gate must contain exact required metadata duty set; extras forbidden")
    if not invented:
        require(all(file_map[ACTUAL_TEXT_PATHS[k]] == digest for k, digest in ACTUAL_TEXT_SHA.items()),
                "actual text completion identity differs from already completed source audit")
    captured = {}
    for path, digest in file_map.items():
        require(isinstance(digest, str) and re.fullmatch(r"[0-9a-f]{64}", digest), "invalid input SHA")
        captured[path] = access.read(path, expected=digest, purpose="bound_source_gate_input")
    require(file_map[SELF_PATH] == sha(own), "gate own source mismatch")
    require(file_map[PARSER_PATH] == PARSER_SHA, "native parser is not reviewed version")
    parser_review = load_json(captured[PARSER_REVIEW_PATH])
    require(parser_review.get("status") == "source_only_review_passed" and
            parser_review.get("reviewed_source_sha256") == PARSER_SHA, "native parser review not passed")
    if not invented:
        require(file_map[PARSER_REVIEW_PATH] == PARSER_REVIEW_SHA, "native review identity changed")
    wrapper = load_json(captured[wrapper_path])
    require(wrapper.get("status") == ("INVENTED_wrapper_review_passed" if invented else "source_only_review_passed")
            and wrapper.get("reviewed_source_sha256") == sha(own), "independent wrapper review missing/stale")
    require(wrapper.get("independent_from_implementer") is True or invented, "wrapper review is not independent")
    bridge = load_json(captured[BRIDGE_PATH])
    require(bridge.get("train_semantic_ids") == list(TRAIN) and
            bridge.get("source_stage_phase2_sequence_prospectively_amended") is True,
            "narrow TRAIN-only chronology bridge missing")
    acquisition_raw = captured[acq_rel]
    manifest = load_json(acquisition_raw)
    if invented:
        require(manifest.get("synthetic_fixture") is True and
                "INVENTED" in str(access.root) and sha(acquisition_raw) != ACTUAL_ACQUISITION_SHA,
                "INVENTED gate cannot authorize a real acquisition")
    else:
        require(file_map[acq_rel] == ACTUAL_ACQUISITION_SHA and not manifest.get("synthetic_fixture"),
                "actual acquisition identity differs")
    require(manifest.get("split_ids") == SPLITS and manifest.get("native_source_abstract_records") == 1000,
            "native800/100/100 acquisition inventory differs")
    require(manifest.get("all_saved_bytes_rehashed_after_all_copies") is True and
            manifest.get("semantic_annotation_read") is False, "raw acquisition state is not source-only")
    audit = load_json(captured[completion["audit"]])
    outputs = load_json(captured[completion["output_manifest"]])
    crosscheck = load_json(captured[completion["root_crosscheck"]])
    require(audit.get("status") == "source_text_provenance_audit_completed_only" and
            audit.get("synthetic_fixture") is invented and audit.get("all_captured_inputs_final_rehashed") is True,
            "completed text provenance gate missing")
    require(audit.get("acquisition_manifest_sha256") == sha(acquisition_raw) and
            audit.get("annotation_semantics_parsed") is False and audit.get("native_split_unchanged") is True,
            "text audit scope/acquisition mismatch")
    require(audit.get("population_counts", {}).get("SC-CoMIcs") == {
        split: {"text_units": len(ids), "distinct_source_document_ids": len(ids)}
        for split, ids in SPLITS.items()}, "text completion native population mismatch")
    require(outputs.get("status") == "source_text_only_outputs_complete" and
            outputs.get("input_final_rehash_passed") is True and outputs.get("synthetic_fixture") is invented and
            outputs.get("acquisition_manifest_sha256") == sha(acquisition_raw), "text output receipt mismatch")
    require(crosscheck.get("actual_audit_sha256") == sha(captured[completion["audit"]]) and
            crosscheck.get("actual_output_manifest_sha256") == sha(captured[completion["output_manifest"]]) and
            crosscheck.get("all5outputs_hash_size_checked") is True and
            crosscheck.get("new_annotation_semantics_parsed") is False, "text root crosscheck mismatch")
    entries = outputs.get("outputs")
    require(isinstance(entries, list) and len(entries) == 5 and
            all(isinstance(b, dict) and set(b) == {"path", "sha256", "bytes"} for b in entries),
            "text receipt must contain exact five complete output entries")
    require({b["path"] for b in entries} == TEXT_OUTPUT_NAMES and
            len({b["path"] for b in entries}) == len(entries),
            "text receipt output names missing/duplicate/undeclared")
    require(all(isinstance(b["sha256"], str) and re.fullmatch(r"[0-9a-f]{64}", b["sha256"]) and
                type(b["bytes"]) is int and b["bytes"] >= 0 for b in entries),
            "text receipt output SHA/size schema invalid")
    audit_entry = next(b for b in entries if b["path"] == "audit.json")
    require(audit_entry["sha256"] == sha(captured[completion["audit"]]) and
            audit_entry["bytes"] == len(captured[completion["audit"]]), "text receipt audit identity differs")
    for b in entries:
        path = str(Path(completion["output_manifest"]).parent / b["path"])
        access.read(path, expected=b["sha256"], size=b["bytes"], purpose="completed_text_output_byte_check")
    # Historical acquisition source bindings are provenance, not a new file
    # permission list. The actual manifest has a fixed independently checked
    # identity. Do not reopen its old scripts, Mu metadata or archived config
    # ZIPs in this narrow phase; only runtime captured inputs are rehashed.
    return gate, manifest, parser_from_captured(captured[PARSER_PATH]), sha(protocol_raw), sha(acquisition_raw)


def train_inventory(access, manifest, acquisition_path):
    base = access.path(acquisition_path).parent
    archives = manifest["official_archives"]
    spec = [("SC-CoMIcs-Abstract1000_CC_BY-NC-3.0.zip", "txt", "raw_text", "raw_source_document"),
            ("SC-CoMIcs-Annotation1000_CC_BY_4.0.zip", "ann", "raw_annotations", "unparsed_annotation_bytes")]
    result = {i: {} for i in TRAIN}
    for name, ext, directory, category in spec:
        archive = archives[name]
        require(archive.get("annotation_semantics_parsed") is False, "acquisition semantics flag differs")
        members = archive["members"]
        require(set(members) == {f"{i:04}.{ext}" for i in range(1, 1001)}, "native archive member metadata incomplete")
        for i in TRAIN:
            b = members[f"{i:04}.{ext}"]
            expected = base / directory / f"{i:04}.{ext}"
            require(access.path(b["canonical_path"]) == expected and b.get("category") == category,
                    f"canonical TRAIN path/category mismatch: {i}/{ext}")
            require(type(b.get("bytes")) is int and b["bytes"] >= 0 and
                    isinstance(b.get("sha256"), str) and re.fullmatch(r"[0-9a-f]{64}", b["sha256"]),
                    "invalid TRAIN byte binding")
            result[i][ext] = b
    # Whitelist all TRAIN pairs only after the complete metadata population check.
    access.allowed_data = {access.relative(b["canonical_path"]) for pair in result.values() for b in pair.values()}
    return result


def duplicates(records, key):
    groups = defaultdict(list)
    for r in records:
        groups[key(r)].append(r.identifier)
    return [{"signature": repr(k), "distinct_ids": ids, "occurrences": len(ids)}
            for k, ids in groups.items() if len(ids) > 1]


def describe_document(doc, native, identifier):
    index = doc.by_id
    counts = Counter()
    labels = defaultdict(Counter)
    unknown = defaultdict(Counter)
    details = []
    for r in doc.records:
        counts[r.family] += 1
        labels[r.family][r.label or "<missing_label>"] += 1
        known = (native.ENTITY_LABELS | native.EVENT_LABELS if r.family == "T" else
                 native.RELATION_LABELS if r.family == "R" else
                 native.EVENT_LABELS if r.family == "E" else frozenset())
        if r.label not in known:
            unknown[r.family][r.label or "<missing_label>"] += 1
        details.append({"id": r.identifier, "occurrence": r.occurrence, "family": r.family,
                        "label": r.label, "segments": [list(x) for x in r.segments],
                        "surface": r.surface, "arg1": r.arg1, "arg2": r.arg2, "trigger": r.trigger,
                        "roles": [list(x) for x in r.roles], "issues": list(r.issues),
                        "raw_record_base64": base64.b64encode(r.raw).decode("ascii")})
    mentions, relations, events, aux = (doc.family(f) for f in ("T", "R", "E", "AUX"))
    def category(target):
        q = index[target]
        return q.family + ":" + q.label
    span_diagnostics = {
        "zero_length_segment_ids": [r.identifier for r in mentions if any(a == b for a, b in r.segments)],
        "discontinuous_ids": [r.identifier for r in mentions if len(r.segments) > 1],
        "unsorted_segment_ids": [r.identifier for r in mentions if tuple(sorted(r.segments)) != r.segments],
        "surface_mismatch_ids": [r.identifier for r in mentions if "reference_text_mismatch" in r.issues],
        "missing_surface_ids": [r.identifier for r in mentions if "missing_reference_text" in r.issues],
    }
    role_hist, base_hist, endpoint_hist = Counter(), Counter(), Counter()
    role_details, trigger_groups = [], defaultdict(list)
    graph = native.nx.DiGraph()
    graph.add_nodes_from(r.identifier for r in events)
    for r in events:
        trigger_groups[r.trigger].append(r.identifier)
        exact = Counter(role for role, _ in r.roles)
        bases = Counter(re.sub(r"[0-9]+$", "", role) for role, _ in r.roles)
        for role, target in r.roles:
            role_hist[role] += 1
            base_hist[re.sub(r"[0-9]+$", "", role)] += 1
            endpoint_hist[f"{role}|{category(target)}"] += 1
            if index[target].family == "E":
                graph.add_edge(r.identifier, target)
        role_details.append({"event_id": r.identifier, "type": r.label, "trigger_id": r.trigger,
                             "trigger_type": index[r.trigger].label, "role_count": len(r.roles),
                             "missing_source_defined_base_roles": [x for x in ("Dopant", "Site") if not bases[x]],
                             "repeated_exact_role_names": {k: v for k, v in exact.items() if v > 1},
                             "repeated_base_role_names": {k: v for k, v in bases.items() if v > 1},
                             "numeric_suffixed_roles": [role for role, _ in r.roles if re.search(r"[0-9]+$", role)],
                             "unknown_roles": [role for role, _ in r.roles if not native.Registry().role_allowed(role)],
                             "nested_E_targets": [target for _, target in r.roles if index[target].family == "E"],
                             "T_targets": [target for _, target in r.roles if index[target].family == "T"]})
    cycles = [sorted(c) for c in native.nx.strongly_connected_components(graph)
              if len(c) > 1 or any(graph.has_edge(n, n) for n in c)]
    pair_labels = defaultdict(list)
    relation_endpoints = Counter()
    for r in relations:
        pair_labels[(r.arg1, r.arg2)].append({"id": r.identifier, "label": r.label})
        relation_endpoints[f"{r.label}|{category(r.arg1)}->{category(r.arg2)}"] += 1
    duplicate = {
        "T_label_segment_signatures": duplicates(mentions, lambda r: (r.label, r.segments)),
        "R_label_ordered_endpoint_ids": duplicates(relations, lambda r: (r.label, r.arg1, r.arg2)),
        "E_label_trigger_and_role_multiset_ids": duplicates(events, lambda r: (r.label, r.trigger, tuple(sorted(r.roles)))),
        "AUX_payload_after_identifier": duplicates(aux, lambda r: r.raw.split(b"\t", 1)[1]),
    }
    # Explicit source-defined directed motifs. Never expand event endpoints to
    # triggers, nearest mentions, or an inferred missing material association.
    equivalents = [r for r in relations if r.label == "Equivalent" and
                   category(r.arg1) == "T:SC" and category(r.arg2) == "T:Value"]
    conditions = [r for r in relations if r.label == "Condition" and
                  category(r.arg1) == "T:Value" and category(r.arg2) in {"T:Doping", "E:Doping"}]
    targets = [r for r in relations if r.label == "Target" and
               category(r.arg1) in {"T:SC", "T:Value", "T:Doping", "E:Doping"} and
               category(r.arg2) in {"T:Element", "T:Main"}]
    chains = [{"equivalent_id": e.identifier, "condition_id": c.identifier,
               "SC_id": e.arg1, "value_id": e.arg2, "doping_endpoint_id": c.arg2}
              for e in equivalents for c in conditions if e.arg2 == c.arg1]
    return {"native_id": identifier, "status": "native_schema_parsed", "source_chars": len(doc.source_text),
            "record_counts": dict(counts), "labels": {k: dict(v) for k, v in labels.items()},
            "source_unknown_labels_retained": {k: dict(v) for k, v in unknown.items()},
            "all_native_records": details, "T_diagnostics": span_diagnostics,
            "duplicate_signatures_distinct_ids_retained": duplicate,
            "event_role_occurrences": dict(role_hist), "event_role_base_occurrences": dict(base_hist),
            "event_role_endpoint_types": dict(endpoint_hist), "event_diagnostics": role_details,
            "shared_trigger_events": [{"trigger_id": k, "event_ids": v} for k, v in trigger_groups.items() if len(v) > 1],
            "nested_event_cycle_components": sorted(cycles), "ordered_relation_endpoint_types": dict(relation_endpoints),
            "ordered_pair_multiple_labels": [{"arg1": a, "arg2": b, "relations": values}
                for (a, b), values in pair_labels.items() if len({x["label"] for x in values}) > 1],
            "AUX_policy": "All opaque native AUX occurrences retained; annotation.conf declares no attribute labels; no metric scope assigned",
            "explicit_source_motifs": {"SC_to_Value_Equivalent": [r.identifier for r in equivalents],
                "Value_to_Doping_Condition": [r.identifier for r in conditions],
                "info_to_Element_or_Main_Target": [r.identifier for r in targets],
                "SC_Value_Doping_chains_same_native_Value_ID": chains},
            "motif_interpretation": "Training graph description only; no new query Gold, causal/physical truth or independent adjudication"}


def write_exclusive(path, raw):
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0), 0o600)
    with os.fdopen(fd, "wb") as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())


def execute(root, acquisition_manifest, protocol, output_dir):
    access = Access(root)
    output = access.path(output_dir)
    require(not output.exists(), "output directory already exists; existing evidence is never overwritten")
    output.parent.mkdir(parents=True, exist_ok=True)
    access._no_links(output.parent)
    output.mkdir()
    report = {"status": "blocked_or_failed_train_schema_audit", "completed": False,
              "training_or_model_scoring_performed": False, "dev_or_test_annotations_opened": False,
              "annotation_ids_opened": [], "schema_failures": []}
    try:
        gate, manifest, native, gate_sha, acquisition_sha = metadata_gates(access, acquisition_manifest, protocol)
        inventory = train_inventory(access, manifest, acquisition_manifest)
        totals, label_totals, unknown_totals, motif_totals = Counter(), defaultdict(Counter), defaultdict(Counter), Counter()
        diagnostics = Counter()
        doc_bindings, empty = [], []
        partial = output / "all_train_documents.jsonl.partial"
        fd = os.open(partial, os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0), 0o600)
        with os.fdopen(fd, "wb") as stream:
            for identifier in TRAIN:
                pair = inventory[identifier]
                # Original bytes are captured once. Parsing never reopens corpus
                # files; full final rehash catches later replacement/tampering.
                text = access.read(pair["txt"]["canonical_path"], expected=pair["txt"]["sha256"],
                                   size=pair["txt"]["bytes"], purpose="TRAIN_source_text_capture")
                ann = access.read(pair["ann"]["canonical_path"], expected=pair["ann"]["sha256"],
                                  size=pair["ann"]["bytes"], purpose="TRAIN_annotation_capture")
                report["annotation_ids_opened"].append(identifier)
                binding = {"native_id": identifier, "text_sha256": sha(text), "annotation_sha256": sha(ann),
                           "text_bytes": len(text), "annotation_bytes": len(ann)}
                doc_bindings.append(binding)
                try:
                    doc = native.parse_document(text, ann, gold=True)
                    entry = describe_document(doc, native, identifier)
                    if not doc.records:
                        empty.append(identifier)
                    totals.update(entry["record_counts"])
                    for k, v in entry["labels"].items():
                        label_totals[k].update(v)
                    for k, v in entry["source_unknown_labels_retained"].items():
                        unknown_totals[k].update(v)
                    for k, v in entry["explicit_source_motifs"].items():
                        motif_totals[k] += len(v)
                    for k, v in entry["T_diagnostics"].items():
                        diagnostics[k] += len(v)
                    diagnostics["shared_trigger_groups"] += len(entry["shared_trigger_events"])
                    diagnostics["nested_event_cycle_components"] += len(entry["nested_event_cycle_components"])
                    diagnostics["ordered_pair_multiple_labels"] += len(entry["ordered_pair_multiple_labels"])
                except native.GoldIntegrityError as exc:
                    entry = {"native_id": identifier, "status": "fatal_native_schema_error", "error": str(exc),
                             "annotation_preserved_at_original_bound_path": pair["ann"]["canonical_path"]}
                    report["schema_failures"].append(entry)
                stream.write((json.dumps({**entry, "byte_binding": binding}, ensure_ascii=False, sort_keys=True) + "\n").encode())
            stream.flush()
            os.fsync(stream.fileno())
        # Check every captured input, including gate/manifest/source/review JSON
        # and all TRAIN pairs. No document is removed for diagnostic problems.
        access.final_recheck()
        report.update({"synthetic_fixture": gate.get("synthetic_fixture") is True,
            "created_at_utc": datetime.now(timezone.utc).isoformat(), "protocol_sha256": gate_sha,
            "acquisition_manifest_sha256": acquisition_sha, "auditor_script_sha256": access.bindings[SELF_PATH]["sha256"],
            "native_parser_sha256": PARSER_SHA, "complete_declared_population": list(TRAIN),
            "transitive_acquisition_source_bindings_not_reopened": manifest.get("input_file_sha256", {}),
            "source_archive_license_bindings": {name: {k: archive.get(k) for k in
                ("url", "official_sha256", "bytes")} for name, archive in manifest["official_archives"].items()},
            "declared_native_text_license": "CC BY-NC 3.0", "declared_native_annotation_license": "CC BY 4.0",
            "documents_opened": len(report["annotation_ids_opened"]), "zero_record_document_ids": empty,
            "record_counts": dict(totals), "labels": {k: dict(v) for k, v in label_totals.items()},
            "source_unknown_labels_retained": {k: dict(v) for k, v in unknown_totals.items()},
            "diagnostic_occurrence_counts": dict(diagnostics), "explicit_source_motif_occurrence_counts": dict(motif_totals),
            "native_split_changed": False, "unsupported_valid_Gold_removed": False,
            "model_metrics_or_search_budget_assigned": False, "tokenizer_or_weights_read": False,
            "physical_truth_expert_or_independent_query_Gold_certified": False,
            "input_byte_bindings": list(access.bindings.values()), "all_inputs_final_rehashed": True,
            "native_document_byte_bindings": doc_bindings, "file_open_log": access.log,
            "interpretation": "TRAIN population schema/structure description only; future registry/AUX/budget/model protocol remains pending"})
        require(not report["schema_failures"], "native schema errors prevent a completion certificate; full800 diagnostic inventory retained")
        require(report["annotation_ids_opened"] == list(TRAIN), "complete TRAIN population required")
        documents = output / "all_train_documents.jsonl"
        access._no_links(partial)
        os.rename(partial, documents)
        doc_raw = access.read(documents, purpose="derived_document_output_binding", capture=False)
        report.update({"status": "INVENTED_train_schema_fixture_completed" if report["synthetic_fixture"] else "train_schema_audit_completed_only",
                       "completed": True, "derived_document_output": {"path": documents.name, "sha256": sha(doc_raw), "bytes": len(doc_raw)}})
        audit_path = output / "audit.json"
        write_exclusive(audit_path, json_bytes(report))
        audit_raw = access.read(audit_path, purpose="derived_audit_output_binding", capture=False)
        access.final_recheck()
        # Reopen every derived output immediately before publishing the last
        # certificate; failure converts the audit to failed evidence instead.
        access.read(documents, expected=sha(doc_raw), size=len(doc_raw), purpose="final_output_rehash", capture=False)
        access.read(audit_path, expected=sha(audit_raw), size=len(audit_raw), purpose="final_output_rehash", capture=False)
        receipt = {"status": "INVENTED_train_schema_outputs_complete" if report["synthetic_fixture"] else "train_schema_only_outputs_complete",
                   "synthetic_fixture": report["synthetic_fixture"], "all_inputs_final_rehashed": True,
                   "auditor_source_sha256": report["auditor_script_sha256"], "protocol_sha256": gate_sha,
                   "acquisition_manifest_sha256": acquisition_sha, "native_ids": list(TRAIN),
                   "outputs": [report["derived_document_output"], {"path": "audit.json", "sha256": sha(audit_raw), "bytes": len(audit_raw)}],
                   "no_fit_dev_test_model_scoring_or_physical_validation": True}
        write_exclusive(output / "output_manifest.json", json_bytes(receipt))
        return report
    except Exception as exc:
        report.update({"status": "blocked_or_failed_train_schema_audit", "completed": False,
                       "failure_type": type(exc).__name__, "failure": str(exc),
                       "input_byte_bindings": list(access.bindings.values()), "file_open_log": access.log})
        if (output / "audit.json").exists():
            os.rename(output / "audit.json", output / "audit.json.failed_pre_certificate")
        if (output / "output_manifest.json").exists():
            os.rename(output / "output_manifest.json", output / "output_manifest.json.failed_pre_certificate")
        write_exclusive(output / "failed_attempt.json", json_bytes(report))
        raise


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for arg in ("root", "acquisition-manifest", "protocol", "output-dir"):
        p.add_argument("--" + arg, required=True)
    a = p.parse_args()
    try:
        r = execute(a.root, a.acquisition_manifest, a.protocol, a.output_dir)
        print(json.dumps({"status": r["status"], "documents": r["documents_opened"], "completed": r["completed"]}))
    except Exception as exc:
        print(f"TRAIN schema audit blocked/failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
