#!/usr/bin/env python3
"""Execute only INVENTED TRAIN-schema fixtures; never read real SC annotations."""
from __future__ import annotations
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import sys


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def save(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def fixture(root, workspace, annotation=None):
    root.mkdir(parents=True)
    for name in ("audit_sccomics_train_schema_only.py", "sccomics_native_graph.py"):
        (root / "src").mkdir(exist_ok=True)
        shutil.copyfile(workspace / "src" / name, root / "src" / name)
    text = b"Tc 40K doped Fe main ABC DEF"
    sample = (b"T1\tSC 0 2\tTc\nT2\tValue 3 6\t40K\nT3\tDoping 7 12\tdoped\n"
              b"T4\tElement 13 15\tFe\nT5\tMain 16 20\tmain\n"
              b"T6\tMain 16 20\tmain\nT7\tMystery 21 21\t\n"
              b"T8\tProcess 25 27;21 24\tDE ABC\nT9\tMaterial 21 24\twrong\nT10\tProperty 21 24\n"
              b"R1\tEquivalent Arg1:T1 Arg2:T2\nR2\tEquivalent Arg1:T1 Arg2:T2\n"
              b"R3\tTarget Arg1:T1 Arg2:T2\nR4\tCondition Arg1:T2 Arg2:T3\n"
              b"R5\tTarget Arg1:T2 Arg2:T5\nR6\tOddRel Arg1:E1 Arg2:T4\n"
              b"E1\tDoping:T3 Dopant:T4 Dopant:T4 Site2:T5 Extra:E2\n"
              b"E2\tDoping:T3 Site:E1\nE3\tDoping:T3\nE4\tDoping:T3\nE5\tOddEvent:T3 Lone:T4\n"
              b"A1\tConfidence E1 high\nA2\tConfidence E1 high\n*\tEquiv T5 T6\n")
    acquisition_dir = root / "data" / "INVENTED_source"
    archives = {}
    for name, ext, folder, category in (
        ("SC-CoMIcs-Abstract1000_CC_BY-NC-3.0.zip", "txt", "raw_text", "raw_source_document"),
        ("SC-CoMIcs-Annotation1000_CC_BY_4.0.zip", "ann", "raw_annotations", "unparsed_annotation_bytes")):
        members = {}
        (acquisition_dir / folder).mkdir(parents=True)
        for i in range(1, 1001):
            raw = text if ext == "txt" else (sample if i == 201 else b"")
            if i < 201 and ext == "ann":
                raw = b"INVENTED_HELD_SEMANTIC_SENTINEL_MUST_NEVER_OPEN\n"
            if i == 201 and ext == "ann" and annotation is not None:
                raw = annotation
            path = acquisition_dir / folder / f"{i:04}.{ext}"
            path.write_bytes(raw)
            members[path.name] = {"canonical_path": str(path), "bytes": len(raw), "sha256": sha(raw), "category": category}
        archives[name] = {"annotation_semantics_parsed": False, "members": members,
                          "url": "https://example.invalid/INVENTED", "archive": "NEVER_OPEN.zip",
                          "official_sha256": "0" * 64, "bytes": 0}
    split = {"train": list(range(201, 1001)), "dev": list(range(101, 201)), "test": list(range(1, 101))}
    acquisition = {"synthetic_fixture": True, "native_source_abstract_records": 1000,
                   "split_ids": split, "official_archives": archives,
                   "all_saved_bytes_rehashed_after_all_copies": True, "semantic_annotation_read": False,
                   "input_file_sha256": {}}
    acq_path = acquisition_dir / "source_acquisition_manifest.json"
    save(acq_path, acquisition)
    acq_sha = sha(acq_path.read_bytes())
    parser_path = root / "src" / "sccomics_native_graph.py"
    audit_path = root / "src" / "audit_sccomics_train_schema_only.py"
    parser_review = "research/round4_sccomics_native_graph_independent_source_review.json"
    save(root / parser_review, {"status": "source_only_review_passed", "reviewed_source_sha256": sha(parser_path.read_bytes()),
                               "synthetic_fixture": True, "actual_independent_review_certified": False})
    bridge = "research/round4_sccomics_train_schema_bridge_protocol.json"
    save(root / bridge, {"train_semantic_ids": split["train"], "source_stage_phase2_sequence_prospectively_amended": True,
                        "status": "INVENTED_bridge_not_real_authorization"})
    wrapper = "research/INVENTED_wrapper_review.json"
    save(root / wrapper, {"status": "INVENTED_wrapper_review_passed", "reviewed_source_sha256": sha(audit_path.read_bytes()),
                         "synthetic_fixture": True, "independent_from_implementer": False})
    text_dir = root / "research" / "INVENTED_text_completion"
    text_audit = {"status": "source_text_provenance_audit_completed_only", "synthetic_fixture": True,
                  "all_captured_inputs_final_rehashed": True, "acquisition_manifest_sha256": acq_sha,
                  "annotation_semantics_parsed": False, "native_split_unchanged": True,
                  "population_counts": {"SC-CoMIcs": {k: {"text_units": len(v), "distinct_source_document_ids": len(v)} for k, v in split.items()}}}
    save(text_dir / "audit.json", text_audit)
    outputs = []
    for name in ("all_qualifying_pairs.jsonl", "all_source_units.jsonl", "test_source_text_clusters.json", "test_source_text_clusters.jsonl"):
        (text_dir / name).write_bytes(b"INVENTED_METADATA_ONLY\n")
        outputs.append({"path": name, "sha256": sha((text_dir / name).read_bytes()), "bytes": (text_dir / name).stat().st_size})
    outputs.append({"path": "audit.json", "sha256": sha((text_dir / "audit.json").read_bytes()), "bytes": (text_dir / "audit.json").stat().st_size})
    save(text_dir / "output_manifest.json", {"status": "source_text_only_outputs_complete", "synthetic_fixture": True,
          "input_final_rehash_passed": True, "acquisition_manifest_sha256": acq_sha, "outputs": outputs})
    save(text_dir / "root_crosscheck.json", {"actual_audit_sha256": sha((text_dir / "audit.json").read_bytes()),
        "actual_output_manifest_sha256": sha((text_dir / "output_manifest.json").read_bytes()),
        "all5outputs_hash_size_checked": True, "new_annotation_semantics_parsed": False})
    completion = {k: str((text_dir / filename).relative_to(root)) for k, filename in
                  (("audit", "audit.json"), ("output_manifest", "output_manifest.json"), ("root_crosscheck", "root_crosscheck.json"))}
    required = ["src/audit_sccomics_train_schema_only.py", "src/sccomics_native_graph.py", parser_review, bridge, wrapper,
                str(acq_path.relative_to(root)), *completion.values()]
    gate = {"status": "INVENTED_authorized_train_schema_only", "synthetic_fixture": True,
            "actual_execution_authorized": True, "train_semantic_ids": split["train"],
            "audit_script_sha256": sha(audit_path.read_bytes()), "wrapper_independent_review_path": wrapper,
            "text_completion": completion, "file_sha256": {p: sha((root / p).read_bytes()) for p in required},
            "dev_semantics_allowed": False, "test_semantics_allowed": False,
            "model_training_or_inference_allowed": False, "paid_api_or_credentials_allowed": False}
    gate_path = root / "research" / "INVENTED_authorized_gate.json"
    save(gate_path, gate)
    return acq_path, gate_path


def load_module(path):
    spec = importlib.util.spec_from_file_location("INVENTED_train_schema_audit", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--workspace", required=True)
    p.add_argument("--output-dir", required=True)
    args = p.parse_args()
    workspace, destination = Path(args.workspace).absolute(), Path(args.output_dir).absolute()
    if destination.exists() or "INVENTED" not in str(destination):
        raise ValueError("fresh explicit INVENTED output path required")
    destination.mkdir(parents=True)
    shutil.copyfile(__file__, destination / "before_checker.py.txt")
    shutil.copyfile(workspace / "src" / "audit_sccomics_train_schema_only.py", destination / "before_auditor.py.txt")
    findings = []

    def run(name, mutate=None, injection=None, expected_failure=None, annotation=None, require_no_annotation_open=False):
        root = destination / ("INVENTED_" + name)
        acquisition, gate_path = fixture(root, workspace, annotation)
        before = root / "fixture_before"
        before.mkdir()
        for pth in (acquisition, gate_path, root / "data/INVENTED_source/raw_annotations/0201.ann",
                    root / "data/INVENTED_source/raw_text/0201.txt"):
            shutil.copyfile(pth, before / (pth.name + ".txt"))
        if mutate:
            mutate(root, acquisition, gate_path)
        module = load_module(root / "src" / "audit_sccomics_train_schema_only.py")
        reads = []
        original_open = module.os.open
        def guarded_open(path, flags, *a, **kw):
            if str(path).endswith(".ann"):
                numeric = int(Path(path).stem)
                if numeric < 201:
                    raise AssertionError("HELD annotation sentinel opened")
                reads.append(numeric)
            return original_open(path, flags, *a, **kw)
        module.os.open = guarded_open
        if injection:
            injection(module, root, acquisition, gate_path)
        try:
            result = module.execute(root, acquisition, gate_path, "outputs")
            failure = None
        except Exception as exc:
            result, failure = None, f"{type(exc).__name__}: {exc}"
        finally:
            module.os.open = original_open
        passed = ((expected_failure is None and result is not None) or
                  (expected_failure is not None and failure is not None and expected_failure in failure))
        receipt = root / "outputs" / "output_manifest.json"
        if expected_failure is not None:
            passed = passed and not receipt.exists()
        if require_no_annotation_open:
            passed = passed and not reads
        findings.append({"name": name, "passed": passed, "expected_failure": expected_failure,
                         "actual_failure": failure, "annotation_read_ids_unique": sorted(set(reads)),
                         "required_annotation_gate_before_open": require_no_annotation_open,
                         "held_annotation_sentinel_opened": False, "completed_output_receipt_present": receipt.exists()})
        return root, result

    good, result = run("rich_schema")
    first = json.loads((good / "outputs" / "all_train_documents.jsonl").read_text().splitlines()[0])
    tests = {
        "full800_and799_empty": result and result["documents_opened"] == 800 and len(result["zero_record_document_ids"]) == 799,
        "all_T_R_E_AUX_occurrences": first["record_counts"] == {"T": 10, "R": 6, "E": 5, "AUX": 3},
        "unknown_T_R_E_AUX_retained": first["source_unknown_labels_retained"] == {"T": {"Mystery": 1}, "R": {"OddRel": 1}, "E": {"OddEvent": 1}, "AUX": {"Confidence": 2, "Equiv": 1}},
        "zero_discontinuous_unsorted": first["T_diagnostics"]["zero_length_segment_ids"] == ["T7"] and first["T_diagnostics"]["unsorted_segment_ids"] == ["T8"],
        "surface_missing_mismatch": first["T_diagnostics"]["surface_mismatch_ids"] == ["T9"] and first["T_diagnostics"]["missing_surface_ids"] == ["T10"],
        "duplicate_distinct_T_ids": first["duplicate_signatures_distinct_ids_retained"]["T_label_segment_signatures"][0]["distinct_ids"] == ["T5", "T6"],
        "repeated_roles_and_suffix": first["event_diagnostics"][0]["repeated_exact_role_names"] == {"Dopant": 2} and first["event_diagnostics"][0]["numeric_suffixed_roles"] == ["Site2"],
        "nested_cycles_and_shared_trigger": first["nested_event_cycle_components"] == [["E1", "E2"]] and len(first["shared_trigger_events"][0]["event_ids"]) == 5,
        "ordered_multilabel_relation": len(first["ordered_pair_multiple_labels"]) == 1 and first["ordered_relation_endpoint_types"]["OddRel|E:Doping->T:Element"] == 1,
        "literal_motifs_no_nearestMain": len(first["explicit_source_motifs"]["SC_Value_Doping_chains_same_native_Value_ID"]) == 2 and first["explicit_source_motifs"]["info_to_Element_or_Main_Target"] == ["R5"],
    }
    for name, passed in tests.items():
        findings.append({"name": name, "passed": bool(passed), "INVENTED_semantics_only": True})

    def gate_change(key, value):
        def change(root, acq, gate):
            j = json.loads(gate.read_bytes()); j[key] = value; save(gate, j)
        return change
    def modify_acquisition(callback):
        def change(root, acq, gate):
            j = json.loads(acq.read_bytes()); callback(j, root); save(acq, j)
            # Rebuild ONLY invented metadata so the adversarial path/member
            # cases reach their intended inventory gate, rather than failing
            # the deliberately earlier text-completion binding check.
            g = json.loads(gate.read_bytes())
            td = root / Path(g["text_completion"]["audit"]).parent
            ta = json.loads((td / "audit.json").read_bytes())
            ta["acquisition_manifest_sha256"] = sha(acq.read_bytes()); save(td / "audit.json", ta)
            om = json.loads((td / "output_manifest.json").read_bytes())
            om["acquisition_manifest_sha256"] = sha(acq.read_bytes())
            for b in om["outputs"]:
                b["sha256"] = sha((td / b["path"]).read_bytes()); b["bytes"] = (td / b["path"]).stat().st_size
            save(td / "output_manifest.json", om)
            cross = json.loads((td / "root_crosscheck.json").read_bytes())
            cross["actual_audit_sha256"] = sha((td / "audit.json").read_bytes())
            cross["actual_output_manifest_sha256"] = sha((td / "output_manifest.json").read_bytes())
            save(td / "root_crosscheck.json", cross)
            for path in [str(acq.relative_to(root)), *g["text_completion"].values()]:
                g["file_sha256"][path] = sha((root / path).read_bytes())
            save(gate, g)
        return change
    def tamper_after_read(target_kind):
        def injection(module, root, acq, gate):
            original = module.Access.read
            changed = False
            def read(self, value, **kw):
                nonlocal changed
                raw = original(self, value, **kw)
                if not changed and kw.get("purpose") == "TRAIN_annotation_capture":
                    target = (Path(value) if target_kind == "annotation" else gate if target_kind == "gate" else acq)
                    target.write_bytes(target.read_bytes() + b" ")
                    changed = True
                return raw
            module.Access.read = read
        return injection

    run("not_authorized", gate_change("actual_execution_authorized", False), expected_failure="not authorized")
    run("dev_permission", gate_change("dev_semantics_allowed", True), expected_failure="explicit false")
    run("train_id_includes_test", gate_change("train_semantic_ids", [1, *range(202, 1001)]), expected_failure="exactly IDs")
    run("own_hash", gate_change("audit_script_sha256", "0" * 64), expected_failure="source not prospectively bound")
    run("invented_gate_cannot_actual", gate_change("synthetic_fixture", False), expected_failure="fixture scope mismatch")
    run("split_mutation", modify_acquisition(lambda j, r: j["split_ids"]["train"].__setitem__(0, 1)), expected_failure="inventory differs")
    run("path_test_alias", modify_acquisition(lambda j, r: j["official_archives"]["SC-CoMIcs-Annotation1000_CC_BY_4.0.zip"]["members"]["0201.ann"].__setitem__("canonical_path", str(r / "data/INVENTED_source/raw_annotations/0001.ann"))), expected_failure="canonical TRAIN path")
    run("path_traversal", modify_acquisition(lambda j, r: j["official_archives"]["SC-CoMIcs-Annotation1000_CC_BY_4.0.zip"]["members"]["0201.ann"].__setitem__("canonical_path", "../0001.ann")), expected_failure="parent traversal")
    run("missing_member", modify_acquisition(lambda j, r: j["official_archives"]["SC-CoMIcs-Annotation1000_CC_BY_4.0.zip"]["members"].pop("1000.ann")), expected_failure="metadata incomplete")
    run("glob_path", modify_acquisition(lambda j, r: j["official_archives"]["SC-CoMIcs-Annotation1000_CC_BY_4.0.zip"]["members"]["0201.ann"].__setitem__("canonical_path", "data/INVENTED_source/raw_annotations/*.ann")), expected_failure="glob characters")
    def extra_bound_input(relative, raw):
        def change(root, acq, gate):
            path = root / relative
            if not path.exists():
                path.parent.mkdir(parents=True, exist_ok=True); path.write_bytes(raw)
            j = json.loads(gate.read_bytes()); j["file_sha256"][relative] = sha(path.read_bytes()); save(gate, j)
        return change
    run("held_bound_input_sentinel", extra_bound_input("data/INVENTED_source/raw_annotations/0001.ann", b""), expected_failure="extras forbidden", require_no_annotation_open=True)
    run("archive_bound_input", extra_bound_input("data/INVENTED_source/NEVER_OPEN.zip", b"INVENTED"), expected_failure="extras forbidden", require_no_annotation_open=True)
    for name, path in (("weight_extra", "INVENTED_model_cache.pt"),
                       ("credential_JSON_extra", "research/INVENTED_credentials.json"),
                       ("checkpoint_extra", "research/INVENTED_checkpoint.bin"),
                       ("safe_extension_cache_extra", "research/INVENTED_model_cache.txt")):
        run(name, extra_bound_input(path, b"INVENTED_NEVER_READ"), expected_failure="extras forbidden", require_no_annotation_open=True)
    def receipt_change(callback):
        def change(root, acq, gate):
            g = json.loads(gate.read_bytes()); td = root / Path(g["text_completion"]["audit"]).parent
            receipt = json.loads((td / "output_manifest.json").read_bytes()); callback(receipt, td)
            save(td / "output_manifest.json", receipt)
            cross = json.loads((td / "root_crosscheck.json").read_bytes())
            cross["actual_output_manifest_sha256"] = sha((td / "output_manifest.json").read_bytes()); save(td / "root_crosscheck.json", cross)
            for k in ("output_manifest", "root_crosscheck"):
                path = g["text_completion"][k]; g["file_sha256"][path] = sha((root / path).read_bytes())
            save(gate, g)
        return change
    run("receipt_empty", receipt_change(lambda j, td: j.__setitem__("outputs", [])), expected_failure="exact five", require_no_annotation_open=True)
    run("receipt_missing", receipt_change(lambda j, td: j["outputs"].pop()), expected_failure="exact five", require_no_annotation_open=True)
    run("receipt_duplicate", receipt_change(lambda j, td: j["outputs"].__setitem__(0, dict(j["outputs"][1]))), expected_failure="missing/duplicate/undeclared", require_no_annotation_open=True)
    for name, path in (("receipt_weights", "model_cache.pt"), ("receipt_traversal", "../audit.json"),
                       ("receipt_held_annotation", "../../data/INVENTED_source/raw_annotations/0001.ann")):
        run(name, receipt_change(lambda j, td, path=path: j["outputs"][0].__setitem__("path", path)), expected_failure="missing/duplicate/undeclared", require_no_annotation_open=True)
    run("receipt_audit_rebound", receipt_change(lambda j, td: j["outputs"][-1].__setitem__("sha256", "0" * 64)), expected_failure="audit identity differs", require_no_annotation_open=True)
    run("receipt_bool_bytes", receipt_change(lambda j, td: j["outputs"][0].__setitem__("bytes", True)), expected_failure="SHA/size schema", require_no_annotation_open=True)
    def missing_real_output(j, td): (td / j["outputs"][0]["path"]).unlink()
    run("receipt_file_missing", receipt_change(missing_real_output), expected_failure="FileNotFoundError", require_no_annotation_open=True)
    def old_parser(root, acq, gate):
        path = root / "src/sccomics_native_graph.py"; path.write_bytes(path.read_bytes() + b"\n# INVENTED change\n")
        j = json.loads(gate.read_bytes()); j["file_sha256"]["src/sccomics_native_graph.py"] = sha(path.read_bytes()); save(gate, j)
    run("unreviewed_parser_change", old_parser, expected_failure="not reviewed version")
    def raw_hash(root, acq, gate):
        path = root / "data/INVENTED_source/raw_annotations/0201.ann"; path.write_bytes(path.read_bytes() + b" ")
    run("raw_hash", raw_hash, expected_failure="SHA mismatch")
    def link(root, acq, gate):
        path = root / "data/INVENTED_source/raw_annotations/0201.ann"; path.unlink(); path.symlink_to(root / "data/INVENTED_source/raw_annotations/0001.ann")
    run("symlink", link, expected_failure="symlink/ancestor")
    def ancestor_link(root, acq, gate):
        path = root / "data/INVENTED_source/raw_annotations"; moved = path.with_name("annotations_moved"); path.rename(moved); path.symlink_to(moved, target_is_directory=True)
    run("ancestor_symlink", ancestor_link, expected_failure="symlink/ancestor")
    def stale_review(root, acq, gate):
        g = json.loads(gate.read_bytes()); pth = root / g["wrapper_independent_review_path"]
        j = json.loads(pth.read_bytes()); j["reviewed_source_sha256"] = "0" * 64; save(pth, j)
        g["file_sha256"][str(pth.relative_to(root))] = sha(pth.read_bytes()); save(gate, g)
    run("stale_independent_review", stale_review, expected_failure="review missing/stale")
    def bad_text(root, acq, gate):
        g = json.loads(gate.read_bytes()); pth = root / g["text_completion"]["audit"]
        j = json.loads(pth.read_bytes()); j["all_captured_inputs_final_rehashed"] = False; save(pth, j)
        g["file_sha256"][str(pth.relative_to(root))] = sha(pth.read_bytes()); save(gate, g)
    run("text_completion_missing", bad_text, expected_failure="text provenance gate missing")
    run("annotation_tamper_after_capture", injection=tamper_after_read("annotation"), expected_failure="SHA mismatch")
    run("gate_tamper_after_capture", injection=tamper_after_read("gate"), expected_failure="SHA mismatch")
    run("manifest_tamper_after_capture", injection=tamper_after_read("manifest"), expected_failure="SHA mismatch")
    def during_read(module, root, acq, gate):
        original_read = module.Access.read
        original_fdopen = module.os.fdopen
        used = False
        def read(self, value, **kw):
            nonlocal used
            if used or kw.get("purpose") != "TRAIN_annotation_capture":
                return original_read(self, value, **kw)
            used = True
            class ChangingStream:
                def __init__(self, wrapped): self.wrapped = wrapped
                def __enter__(self): self.wrapped.__enter__(); return self
                def __exit__(self, *args): return self.wrapped.__exit__(*args)
                def fileno(self): return self.wrapped.fileno()
                def read(self):
                    raw = self.wrapped.read()
                    Path(value).write_bytes(raw + b"DURING_READ_TAMPER")
                    return raw
            module.os.fdopen = lambda fd, *a, **k: ChangingStream(original_fdopen(fd, *a, **k))
            try:
                return original_read(self, value, **kw)
            finally:
                module.os.fdopen = original_fdopen
        module.Access.read = read
    run("during_read_tamper", injection=during_read, expected_failure="changed during read")
    def output_tamper(module, root, acq, gate):
        original = module.Access.read
        def read(self, value, **kw):
            raw = original(self, value, **kw)
            if kw.get("purpose") == "derived_document_output_binding":
                Path(value).write_bytes(raw + b"TAMPER")
            return raw
        module.Access.read = read
    run("derived_output_tamper", injection=output_tamper, expected_failure="SHA mismatch")
    for name, ann in (
        ("duplicate_native_ID", b"T1\tMain 0 2\tTc\nT1\tMain 0 2\tTc\n"),
        ("dangling_endpoint", b"T1\tMain 0 2\tTc\nR1\tTarget Arg1:T1 Arg2:E9\n"),
        ("bad_offset", b"T1\tMain 0 99\tTc\n"),
        ("invalid_UTF8", b"T1\tMain 0 2\t\xff\n"),
    ):
        run(name, annotation=ann, expected_failure="schema errors prevent")
    bindings = []
    # This walk inventories our artificial fixtures only. Production auditor
    # has no walk/glob/archive reader; no real workspace data is inspected here.
    for path in sorted(destination.rglob("*")):
        if path.is_file() and not path.is_symlink():
            bindings.append({"path": str(path.relative_to(destination)), "sha256": sha(path.read_bytes()), "bytes": path.stat().st_size})
    report = {"scope": "INVENTED only; implementer self-check, not independent review or actual TRAIN audit",
              "checks": findings, "passed": sum(bool(x["passed"]) for x in findings), "total": len(findings),
              "actual_corpus_annotations_opened": False, "training_or_model_scoring_performed": False,
              "fixture_byte_bindings": bindings}
    save(destination / "synthetic_execution_report.json", report)
    print(json.dumps({k: report[k] for k in ("passed", "total", "actual_corpus_annotations_opened")}))
    return 0 if all(x["passed"] for x in findings) else 1


if __name__ == "__main__":
    raise SystemExit(main())
