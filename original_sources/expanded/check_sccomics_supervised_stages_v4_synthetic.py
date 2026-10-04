"""Invented stage/I/O fixtures only; no actual corpus/model input option."""
from __future__ import annotations

import argparse
import copy
from dataclasses import FrozenInstanceError
import hashlib
import io
import json
from pathlib import Path
import shutil
import sys
import tempfile
import traceback
import types

import sccomics_supervised_stages_v4 as st
import sccomics_supervised_runtime_v4 as rt
import importlib
import importlib.readers
import os
import time
import marshal
import struct
import numpy as np


def dump(path,obj):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_bytes(st.serialized(obj))


def build(scope,source_raw):
    inputs={}
    def put(name,raw,duty):
        path=scope.root/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(raw)
        inputs[name]=st.binding(raw,duty)
    for name,raw in source_raw.items():put(name,raw,"code")
    clusters={"native_test_record_count":100,"source_text_cluster_count":100,"gold_or_model_used":False,"native_split_changed":False,"same_experiment_or_pretraining_independence_certified":False,
              "clusters":[{"cluster_id":f"INVENTED_cluster_{i:04}","native_test_ids":[i]} for i in range(1,101)]}
    put(st.CLUSTERS,st.serialized(clusters),"control")
    put(st.HISTORY,b"INVENTED historical metadata; frozen literal values from prospective configuration only","control")
    for name in st.COLD_NAMES:put(st.COLD_ROOT+"/"+name,b"INVENTED_COLD_FILE_NO_REAL_WEIGHTS_"+name.encode(),"cold")
    for split,ids in st.SPLITS.items():
        for identifier in ids:
            text=f"Doped Fe INVENTED source {identifier}\n".encode()
            # Unavailable native T label proves complete-T FN denominator.
            ann=b"T1\tDoping 0 5\tDoped\nE1\tDoping:T1\n"
            if split!="train":
                ann+=b"T2\tINVENTED_UNKNOWN_TYPE 6 8\tFe\nR1\tTarget Arg1:T1 Arg2:T2\n"
            put(f"data/sccomics_round4/source_v3/raw_text/{identifier:04}.txt",text,"text")
            put(f"data/sccomics_round4/source_v3/raw_annotations/{identifier:04}.ann",ann,"annotation")
            cached=st.serialized({"INVENTED":True,"source_id":identifier,"source_sha256":st.sha(text),"source_only":True,"annotation_content_used":False,"wordpieces":2,"offsets":[[0,5],[6,8]],"candidate_span_count":3})
            put(f"results/local_baseline/sccomics_native_v1/source_cache/{split}/{identifier:04}.pt",cached,"cache")
    return st.BoundIO(scope,inputs)


class InventedBackend:
    """No real model/fit; deterministic artificial state and source values.

    Native parsing, complete selection and primary match operate only on this
    newly generated fictional corpus. Cold snapshots/cache capture are genuine
    I/O checks using invented bytes; no actual HF weights are consumed.
    """
    def __init__(self,bound,science):
        self.bound=bound;self.science=science
        self.native=science.get("sccomics_native_graph")
        self.selector=science.get("sccomics_development_selection")
        self.projection=science.get("sccomics_training_projection")
        self.models=science.get("sccomics_source_models")
        self.core=science.get("sccomics_fitting_core")
        self.loads=[];self.fit_starts=[];self.fail_after_documents=None;self.zero_ner=False
        self.cold=bound.private_snapshot(tuple(st.COLD_ROOT+"/"+n for n in st.COLD_NAMES),bound.root/"private_cold")
        self.validate()

    def validate(self):
        self.science.validate()
        for name,path in self.cold.items():
            assert st.digest_file(path.parent,path.name)=={k:self.bound.inputs[name][k] for k in ("sha256","size_bytes")}

    def text(self,identifier):
        return self.bound.read(f"data/sccomics_round4/source_v3/raw_text/{identifier:04}.txt","text")

    def cache(self,identifier):
        split=next(s for s,ids in st.SPLITS.items() if identifier in ids)
        raw=self.bound.read(f"results/local_baseline/sccomics_native_v1/source_cache/{split}/{identifier:04}.pt","cache")
        # Mock weights-only deserializer receives checked BytesIO, not a path.
        stream=io.BytesIO(raw)
        assert isinstance(stream,io.BytesIO)
        value=st.unique_json(stream.read())
        assert value["INVENTED"] and value["source_id"]==identifier and value["source_sha256"]==st.sha(self.text(identifier))
        return value

    def gold(self,identifier):
        raw=self.bound.read(f"data/sccomics_round4/source_v3/raw_annotations/{identifier:04}.ann","annotation")
        self.loads.append(identifier)
        return self.native.parse_document(self.text(identifier),raw,gold=True)

    def fit_start(self,kind,seed):
        self.validate()
        self.fit_starts.append((kind,seed))
        # Mock original loader's exact offline flag contract, all12 blocks and
        # independent same-seed last-block copy; no HF/torch model is loaded.
        flags={"local_files_only":True,"trust_remote_code":False,"use_safetensors":True}
        original=[{"INVENTED_cold_block":i} for i in range(12)]
        last=copy.deepcopy(original[11])
        assert last=={"INVENTED_cold_block":11} and flags=={"local_files_only":True,"trust_remote_code":False,"use_safetensors":True}
        return {"kind":kind,"seed":seed,"last":last}, {"INVENTED_no_actual_training":True,"cold_start":True,"seed":seed,
            "rng":"INVENTED_reset_marker","loader_flags":flags,"implicit_resume":False}

    def fit_epoch(self,handle,kind,seed,epoch,log):
        order=self.core.document_order(seed,epoch,st.SPLITS["train"])
        seen=0;updates=[]
        for offset in range(0,800,8):
            group=order[offset:offset+8];update=(epoch-1)*100+offset//8+1
            for identifier in group:
                self.gold(identifier);self.cache(identifier)
                log({"event":"train_document_backward_completed","kind":kind,"seed":seed,"epoch":epoch,"source_id":identifier,
                     "update":update,"diagnostics":{"INVENTED_no_model_loss_or_backward":True}})
                seen+=1
                if self.fail_after_documents is not None and seen==self.fail_after_documents:
                    raise MemoryError("INVENTED_fixed_failure_after_document_record")
            row={"event":"optimizer_update_completed","kind":kind,"seed":seed,"epoch":epoch,"update":update,
                 "source_ids":group,"lr_factor":self.core.lr_factor(update),"INVENTED_no_optimizer_executed":True}
            updates.append(row);log(row)
        raw=st.serialized({"INVENTED_complete_task_state_not_torch_weights":True,"kind":kind,"seed":seed,"epoch":epoch})
        return raw,{"complete_model_state_dict":True,"optimizer_moments_in_epoch_checkpoint":False,
                    "rng":"INVENTED_epoch_rng_marker","optimizer_updates":updates,"INVENTED_only":True}

    def failure_state(self,handle):
        return b"INVENTED_PARTIAL_FAILED_STATE",{"failed_partial_state_not_a_successful_checkpoint":True}

    def release(self,handle):
        handle.clear()

    def inference_start(self,kind,seed,checkpoint_raw):
        obj=st.unique_json(checkpoint_raw)
        assert obj["kind"]==kind and obj["seed"]==seed and obj["INVENTED_complete_task_state_not_torch_weights"]
        return {"kind":kind,"seed":seed,"INVENTED_source_inference_mock":True}

    def source_probabilities(self,handle,kind,seed,identifier,mentions=None):
        self.cache(identifier)
        if kind=="span":
            candidates=[[0,0,0,5],[0,1,0,8],[1,1,6,8]]
            values=[[0.]*9 for _ in candidates];values[0][self.models.TEXT_TYPES.index("Doping")]=0. if self.zero_ner else .6
            result={"candidates":candidates,"probabilities":values,"complete_rows":3,"columns":9}
        else:
            result={"mentions":copy.deepcopy(mentions),"probabilities":[[0.]*5 for _ in range(len(mentions)**2)],
                    "complete_rows":len(mentions)**2,"columns":5,"diagnostics":{"INVENTED_full_directed_population":True}}
        return dict(result,source_id=identifier,source_sha256=st.sha(self.text(identifier)),source_only=True,Gold_consulted=False,
                    candidate_cap=None,inference_row_block_size=512)

    def source_descriptor(self,identifier):
        saved=self.cache(identifier)
        return {"source_id":identifier,"source_sha256":saved["source_sha256"],"wordpieces":saved["wordpieces"],"offsets":saved["offsets"],
                "candidate_span_count":saved["candidate_span_count"],"fresh_original_local_tokenizer_boundary_check":True}

    def detector_mentions(self,obj,threshold):
        return self.core.predicted_mentions(obj["candidates"],obj["probabilities"],threshold)

    def detector_document(self,identifier,mentions):
        raw=self.text(identifier);text=raw.decode()
        records=[{"family":"T","id":f"T{i+1}","type":m["type"],"segments":m["segments"],
                  "text":" ".join(text[a:b] for a,b in m["segments"])} for i,m in enumerate(mentions)]
        return records,self.native.parse_prediction_records(raw,records)

    def emitted(self,identifier,mentions,probabilities,rt,et):
        text=self.text(identifier)
        count=len(mentions)**2
        assert type(probabilities) is list and len(probabilities)==count and all(type(row) is list and len(row)==5 and all(type(x) in (int,float) and np.isfinite(x) and 0<=x<=1 for x in row) for row in probabilities)
        matrix=np.asarray(probabilities,dtype=float).reshape((count,5))
        records,diagnostics=self.projection.emit_native_prediction_records(text.decode(),mentions,matrix,relation_threshold=rt,role_threshold=et)
        self.last_emission_diagnostics=copy.deepcopy(diagnostics)
        return records,self.native.parse_prediction_records(text,records)


def run_analysis(controller,bound):
    assert not(set(st.SPLITS["test"])&bound.annotation_ids_read)
    receipt=controller.score_and_statistics()
    assert controller.phase=="primary_test_analysis_complete_only"
    assert set(bound.annotation_ids_read)==set(range(1,1001))
    assert {r["stage"] for r in bound.annotation_first_reads if r["native_id"]<=100}=={"scoring_test_gold"}
    result=controller.store.object("analysis/primary_complete_test_objects.json")
    assert result["source_cluster_count"]==100 and len(result["statistics"]["primary_comparisons"])==6
    assert result["three_fit_pooled_count_scores"]["NER"]["T_complete_native"]["counts"]["fn"]==300
    assert result["independent_statistical_algorithm"] is False
    return {"INVENTED_only":True,"first100_test_ANN_reads_after_fresh_complete_barrier":True,
            "all_unknown_T_FN_retained_across3fits":300,"primary6_and_historical10":True,"receipt":receipt}


def run_replay(controller):
    receipt=controller.independent_count_replay()
    assert controller.phase=="controlled_stages_and_native_counts_finished_only"
    result=controller.store.object("analysis/independent_native_count_replay.json")
    assert result["primary_witnesses_validated_against_independent_finite_closures"] is True
    assert result["independent_statistical_algorithm"] is False
    return {"INVENTED_only":True,"distinct_native_parser_and_counts_and_witnesses":True,"same_statistics_module_honestly_recomputed":True,"receipt":receipt}


def run_independent_statistics(controller):
    receipt=controller.independent_statistics_replay()
    assert controller.phase=="controlled_stages_and_all_replays_finished_only"
    result=controller.store.object("analysis/independent_statistics_replay.json")
    assert result["independent_statistical_arithmetic"] is True and result["production_statistics_module_imported"] is False
    assert result["raw_integer_bootstrap_replicates_verified"]==10000 and result["bootstrap_intervals_verified"]==14
    assert result["fixed_RNG_library_and_stream_shared_by_design"] is True
    assert result["actual_source_or_execution_barriers_certified"] is False
    return {"INVENTED_only":True,"distinct_integer_Fraction_tail_Holm_and_percentile_arithmetic":True,
            "fixed_random_streams_replayed_not_new_samples":True,"no_new_statistical_policy_or_P":True,"receipt":receipt}


def invented_runtime(scope):
    root=scope.root/"INVENTED_runtime";root.mkdir()
    files={"bin/python3.13":b"INVENTED_EXECUTABLE_BYTES_NOT_RUN", "lib/python3.13/INVENTED_allowed.py":b"VALUE='captured_original'\n"}
    roles={"bin/python3.13":"Python_executable","lib/python3.13/INVENTED_allowed.py":"stdlib_code_native"}
    bindings={}
    for name,raw in files.items():
        p=root/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(raw)
        bindings[name]=dict(st.binding(raw,"runtime"),runtime_role=roles[name])
    population={"policy":rt.POLICY,"framework_root":str(root),"stdlib_relative_path":"lib/python3.13", "site_relative_path":"lib/python3.13/site-packages",
                "executable_relative_path":"bin/python3.13","package_roots":[],"file_roles":roles,"all_OS_GPU_or_system_dynamic_libraries_byte_bound":False}
    return {"status":"finite_runtime_code_only_source_receipt_not_fit_authority","runtime":st.RUNTIME,"population":population,
            "file_bindings":bindings,"source_contract_json_sha256":st.V4_DEPENDENCIES_SHA}


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("--output-dir",required=True)
    args=parser.parse_args();output=Path(args.output_dir).absolute()
    assert not output.exists(),"historical synthetic outputs never overwritten"
    output.mkdir(parents=True)
    assert not any(p.is_symlink() for p in (output,*output.parents))
    workspace=Path(__file__).resolve().parents[1]
    source={n:(workspace/n).read_bytes() for n in st.SCIENCE+(st.SELF,)+st.FINAL_ANALYSIS}
    (output/"runner_before.py").write_bytes(source[st.SELF]);(output/"helper_before.py").write_bytes(Path(__file__).read_bytes())
    for name,raw in source.items():
        (output/(Path(name).name+".before_snapshot")).write_bytes(raw)
    results=[]
    def check(name,function):
        try:
            detail=function();results.append({"id":name,"passed":True,"detail":detail})
        except Exception as error:
            results.append({"id":name,"passed":False,"error":str(error),"error_type":type(error).__name__,"traceback":traceback.format_exc()})
    def rejected(function,types_=(st.StageIntegrityError,OSError,ValueError)):
        try:function()
        except types_ as error:return {"rejected":True,"error_type":type(error).__name__,"message":str(error)}
        raise AssertionError("unsafe/incomplete stage accepted")
    check("actual_default_missing_full_freeze",lambda:rejected(lambda:st.authorize_actual(workspace,freeze_sha256="0"*64,gate_sha256="0"*64)))
    for value in ("../escape","/absolute","a/../b","a//b","a\\b","a*","a?","a:b","a\x00b"):
        check("reject_path_"+repr(value),lambda value=value:rejected(lambda:st.relative(value)))
    check("reject_JSON_duplicate",lambda:rejected(lambda:st.unique_json(b'{"a":1,"a":2}')))
    check("reject_JSON_NaN",lambda:rejected(lambda:st.unique_json(b'{"a":NaN}')))
    check("reject_JSON_exponent_overflow",lambda:rejected(lambda:st.unique_json(b'{"nested":[{"x":1e400}]}')))
    check("reject_direct_invented_scope_constructor",lambda:rejected(lambda:st._Scope(workspace,True,"outputs",None,object())))

    def actual_entry_mock():
        """Run actual entry orchestration with only owned mock authority/I/O.

        This does not create an actual gate and never invokes TorchBackend,
        cold/task reads or fits. It exposes /var alias and old runner-name bugs.
        """
        captures=[];calls=[]
        class Bound:
            def read(self,n,d):assert (n,d)==(st.RUNTIME_RECEIPT,"control");return b"{}"
            def validate_all(self):calls.append("bound-final")
        class Runtime:
            def __init__(self,receipt,scope):assert receipt=={};self.active=False
            def allow_science(self,science):calls.append("allow-science")
            def install(self):self.active=True
            def validate(self):calls.append("runtime-final")
            def close(self):self.active=False
        class Controller:
            def __init__(self,bound,backend,science):self.phase="INVENTED_mock_pretest_finished"
            def run_pretest(self):calls.append("pretest-only");return {"INVENTED_no_fit":True}
        class Backend:
            def __init__(self,bound,science,parent,runtime):
                assert not any(p.is_symlink() for p in (parent,*parent.parents));calls.append("mock-backend-only")
        class Science:
            def __init__(self,bound,directory):
                st.root_path(directory.parent);captures.append(directory)
            def get(self,name):
                calls.append(name)
                if name=="sccomics_supervised_stages_v4":return types.SimpleNamespace(_SCOPES=st._SCOPES,TorchBackend=Backend,StageController=Controller)
                if name=="sccomics_supervised_runtime_v4":return types.SimpleNamespace(BoundRuntime=Runtime)
                raise AssertionError("wrong captured runner module")
            def close(self):calls.append("science-close")
        original_authorize,original_science=st.authorize_actual,st.CapturedScience
        with st.invented_workspace() as fake:
            st.authorize_actual=lambda *a,**k:(fake,Bound());st.CapturedScience=Science
            # Main legitimately consumes this mocked scope; keep owner cleanup.
            try:assert st.main(["--root",str(fake.root),"--freeze-sha256","0"*64,"--final-gate-sha256","0"*64])==0
            finally:
                st.authorize_actual=original_authorize;st.CapturedScience=original_science;st._SCOPES.add(fake)
        assert captures and "sccomics_supervised_stages_v4" in calls and "pretest-only" in calls
        return {"actual_entry_orchestration_only_mocked":True,"canonical_private_temp_path_validated":True,
                "correct_captured_v2_runner_module":True,"actual_gate_created_or_fit_or_model_SC_read":False}
    check("actual_main_owned_mock_canonical_temp_and_captured_v2_module",actual_entry_mock)
    def package_resource_interface():
        with tempfile.TemporaryDirectory(prefix="INVENTED_package_resource_") as directory:
            package=Path(directory).resolve();path=package/"__init__.py";raw=b"INVENTED='package only'\n";path.write_bytes(raw)
            (package/"resource.txt").write_text("INVENTED RESOURCE ONLY")
            class Guard:
                def capture(self,name):assert Path(name)==path;return raw
            reader=rt.PythonLoader(Guard(),str(path)).get_resource_reader("INVENTED_package")
            assert reader.files()==package and reader.files().joinpath("resource.txt").read_text()=="INVENTED RESOURCE ONLY"
            return {"INVENTED_only":True,"stdlib_FileReader_path_interface":True,"actual_library_resources_or_model_backend_not_certified":True}
    check("runtime_PythonLoader_plain_INVENTED_package_resource_interface",package_resource_interface)
    def vendored_license_role():
        for first,last in (("distribution_metadata_license","package_license"),("package_license","distribution_metadata_license")):
            roles={"INVENTED_vendor.dist-info/LICENSE":first};rt.claim_role(roles,"INVENTED_vendor.dist-info/LICENSE",last)
            assert roles=={"INVENTED_vendor.dist-info/LICENSE":"distribution_metadata_license"}
        result=rejected(lambda:rt.claim_role({"INVENTED.py":"package_code_native"},"INVENTED.py","distribution_metadata_license"))
        return dict(result,same_license_file_two_metadata_discovery_routes_no_permission_expansion=True)
    check("runtime_vendored_dist_info_license_role_deduplicated_only",vendored_license_role)
    def canonical_framework_alias():
        with tempfile.TemporaryDirectory(prefix="INVENTED_framework_alias_") as d:
            root=Path(d).resolve();(root/"Python").write_bytes(b"INVENTED_FRAMEWORK_BINARY_NOT_EXECUTED")
            p=root/rt.FRAMEWORK_METADATA_ALIAS;p.parent.mkdir(parents=True);p.symlink_to(root/"Python")
            row=rt.framework_alias_metadata(root,p)
            assert row["canonical_target"]=="Python" and row["import_or_task_read_via_alias_permitted"] is False
            result=rejected(lambda:st.raw_read(root,rt.FRAMEWORK_METADATA_ALIAS))
            p.unlink();(root/"wrong").write_bytes(b"wrong");p.symlink_to(root/"wrong")
            rejected(lambda:rt.framework_alias_metadata(root,p))
            return dict(result,one_fixed_framework_metadata_alias_only=True,wrong_target_rejected=True,task_or_code_read_via_alias_still_rejected=True)
    check("runtime_one_framework_alias_metadata_canonical_binary_not_read_alias",canonical_framework_alias)

    with st.invented_workspace() as rt_scope:
        receipt=invented_runtime(rt_scope);runtime=rt.BoundRuntime(receipt,rt_scope)
        def runtime_good():
            sys.path.insert(0,str(runtime.root/"lib/python3.13"));runtime.install()
            try:
                module=importlib.import_module("INVENTED_allowed")
                assert module.VALUE=="captured_original"
                assert runtime.validate()["actual_file_hashes_checked"]==2
                return {"finite_runtime_files":2,"captured_source_no_pyc":True,"OS_GPU_bytes_claim":False}
            finally:runtime.close();sys.path.pop(0);sys.modules.pop("INVENTED_allowed",None)
        check("runtime_declared_python_import",runtime_good)
        def runtime_pyc():
            p=runtime.root/"lib/python3.13/INVENTED_allowed.py";status=p.stat()
            cache=Path(importlib.util.cache_from_source(str(p)));cache.parent.mkdir(exist_ok=True)
            poisoned=compile("VALUE='POISONED_STALE_PYC'\n",str(p),"exec")
            cache.write_bytes(importlib.util.MAGIC_NUMBER+struct.pack("<III",0,int(status.st_mtime),status.st_size)+marshal.dumps(poisoned))
            sys.path.insert(0,str(p.parent));runtime.install()
            try:
                module=importlib.import_module("INVENTED_allowed")
                assert module.VALUE=="captured_original"
                return {"valid_timestamp_poisoned_pyc_ignored":True,"actual_captured_Python_source_used":True}
            finally:runtime.close();sys.path.pop(0);sys.modules.pop("INVENTED_allowed",None)
        check("runtime_poisoned_valid_timestamp_pyc_ignored",runtime_pyc)
        def runtime_capture_then_replace():
            p=runtime.root/"lib/python3.13/INVENTED_allowed.py";raw=p.read_bytes();original=runtime.capture
            def replace(filename):
                captured=original(filename);p.write_bytes(b"raise RuntimeError('LIVE_CODE_MUST_NOT_EXECUTE')\n");return captured
            runtime.capture=replace;sys.path.insert(0,str(p.parent));runtime.install()
            try:
                module=importlib.import_module("INVENTED_allowed");assert module.VALUE=="captured_original"
                result=rejected(runtime.validate_files)
                return dict(result,captured_original_executed=True,post_capture_live_change_blocks_receipt=True)
            finally:runtime.close();sys.path.pop(0);sys.modules.pop("INVENTED_allowed",None);runtime.capture=original;p.write_bytes(raw)
        check("runtime_captured_source_ignores_live_replacement_and_final_hash_blocks",runtime_capture_then_replace)
        def runtime_unknown():
            marker=rt_scope.root/"UNAUTHORIZED_IMPORT_EXECUTED"
            p=runtime.root/"lib/python3.13/INVENTED_unbound.py"
            p.write_text("from pathlib import Path\nPath("+repr(str(marker))+").write_text('bad')\n")
            sys.path.insert(0,str(p.parent));runtime.install()
            try:
                result=rejected(lambda:importlib.import_module("INVENTED_unbound"));assert not marker.exists();return dict(result,sentinel_not_executed=True)
            finally:runtime.close();sys.path.pop(0);sys.modules.pop("INVENTED_unbound",None)
        check("runtime_unlisted_dynamic_import_before_loader",runtime_unknown)
        def runtime_weight(role,name):
            bad=copy.deepcopy(receipt);bad["population"]["file_roles"][name]=role
            bad["file_bindings"][name]={"sha256":"0"*64,"size_bytes":1,"duty":"runtime","runtime_role":role}
            return rejected(lambda:rt.BoundRuntime(bad,rt_scope))
        for role,name in (("package_code_native","INVENTED_weights.pt"),("distribution_metadata_license","INVENTED_credentials.json"),
                          ("stdlib_code_native","INVENTED_archive.zip"),("package_license","LICENSE_weights.pt"),
                          ("distribution_metadata_license","licenses/INVENTED_credentials.json"),("Python_license","LICENSE_archive.zip")):
            check("runtime_reject_role_"+name,lambda role=role,name=name:runtime_weight(role,name))
        def changed_runtime():
            p=runtime.root/"lib/python3.13/INVENTED_allowed.py";raw=p.read_bytes();p.write_bytes(b"VALUE='changed'\n")
            try:return rejected(runtime.validate_files)
            finally:p.write_bytes(raw)
        check("runtime_final_bytes_tamper",changed_runtime)
        shutil.copytree(rt_scope.root,output/"INVENTED_runtime_fixture")

    with st.invented_workspace(gold_analysis_allowed=True) as scope:
        bound=build(scope,source)
        science=st.CapturedScience(bound,scope.root/"private_source")
        backend=InventedBackend(bound,science)
        controller=st.StageController(bound,backend,science=science)
        check("startup_all_source_hashes_no_unopened_ANN",lambda: {
            "receipt_only_annotations":len(bound.validate_all()["unopened_annotations_receipt_path_size_only"]),
            "assert_no_ANN_open":not bound.annotation_ids_read} if not bound.annotation_ids_read else (_ for _ in ()).throw(AssertionError("ANN opened")))
        check("test_semantics_rejected_created",lambda:rejected(lambda:backend.gold(1)))
        check("dev_semantics_rejected_created",lambda:rejected(lambda:backend.gold(101)))
        check("TRAIN_semantics_rejected_outside_fit",lambda:rejected(lambda:backend.gold(201)))
        def spoof_phase():
            old=controller.phase;controller.phase="selecting_ner";controller.busy=True
            try:return rejected(lambda:backend.gold(101))
            finally:controller.phase=old;controller.busy=False
        check("phase_string_alone_does_not_unlock_dev",spoof_phase)
        def spoof_callback():
            bound.permission=lambda i:"INVENTED_FORGED_PERMISSION"
            return rejected(lambda:backend.gold(1))
        check("public_callback_cannot_unlock_test",spoof_callback)
        def extra_input():
            bad=copy.deepcopy(bound.inputs);bad["research/INVENTED_model_weights.pt"]={"sha256":"0"*64,"size_bytes":1,"duty":"control"}
            return rejected(lambda:st.BoundIO(scope,bad))
        check("extra_weight_labeled_metadata_closed_gate",extra_input)
        check("cannot_skip_to_select_ner",lambda:rejected(controller.select_ner))
        check("cannot_skip_to_test_generation",lambda:rejected(controller.generate_test_sources))
        check("cannot_skip_to_test_barrier",lambda:rejected(controller.test_barrier))
        check("cannot_score_before_barrier",lambda:rejected(controller.score_and_statistics))
        check("frozen_scope_root_cannot_be_repointed",lambda:rejected(lambda:setattr(scope,"root",workspace),(FrozenInstanceError,)))
        def fifo_input():
            p=scope.root/"INVENTED_fifo";os.mkfifo(p);start=time.monotonic()
            try:
                result=rejected(lambda:st.raw_read(scope.root,p.name))
                assert time.monotonic()-start<1
                return dict(result,nonblocking_elapsed_seconds=time.monotonic()-start)
            finally:p.unlink()
        check("FIFO_rejected_before_blocking_open",fifo_input)
        def prewrite_snapshot():
            target=scope.root/"INVENTED_escape_target";target.mkdir()
            alias=scope.root/"INVENTED_destination_alias";alias.symlink_to(target,target_is_directory=True)
            try:
                result=rejected(lambda:bound.private_snapshot((st.SCIENCE[0],),alias/"new_snapshot"))
                assert not(target/"new_snapshot").exists()
                return dict(result,no_external_directory_or_bytes_created=True)
            finally:alias.unlink()
        check("snapshot_symlink_ancestor_rejected_before_mkdir_or_write",prewrite_snapshot)
        def complete_pipeline():
            barrier=controller.run_pretest()
            assert controller.phase=="pretest_source_barrier_passed_only"
            result=controller.store.object("test_source_barrier.json")
            assert result["complete150_epoch_checkpoints"]==150 and result["complete_test_source_objects"]==1500
            assert result["test_annotation_semantic_permission_enabled"] is False
            assert set(bound.annotation_ids_read)==set(range(101,1001))
            assert {e["stage"] for e in bound.annotation_first_reads if e["native_id"]<201}=={"selecting_ner"}
            assert backend.fit_starts==[(k,s) for k in st.KINDS for s in st.SEEDS]
            grid=controller.store.object("choices/ner_complete80_grid.json")
            assert len(grid)==80
            assert all(len(s["by_seed_document"])==3 and all(len(v)==100 for v in s["by_seed_document"].values()) for s in grid)
            assert grid[0]["by_seed_document"][str(st.SEEDS[0])]["101"]["T_complete_native"]=={"tp":1,"fp":0,"fn":1}
            assert controller.store.object("choices/dev_complete_native_gold_inventory.json")["101"]["unknown_T_label_counts"]=={"INVENTED_UNKNOWN_TYPE":1}
            assert all(len(controller.store.object(f"choices/{k}_complete250_grid.json"))==250 for k in st.ARCHITECTURES)
            assert len([n for n in controller.store.bindings if n.startswith("checkpoints/")])==150
            assert len([n for n in controller.store.bindings if n.startswith("dev_ner_probabilities/")])==3000
            assert len([n for n in controller.store.bindings if n.startswith("dev_pair_probabilities/")])==12000
            assert len([n for n in controller.store.bindings if n.startswith("test_source_graphs/")])==1500
            return {"INVENTED_only":True,"complete150_dummy_CP":150,"full80_and4x250_grids":True,
                    "dev_unknown_T_retained_FN":True,"complete1500_test_source_JSON":1500,"test_ANN_opened":0,"barrier":barrier}
        check("complete_INVENTED_pipeline_no_real_fit",complete_pipeline)
        def corrupt_dev_content(role,kind,epoch,mutation):
            name=st.artifact_name(role,kind,st.SEEDS[0],epoch,101)+".json";b=copy.deepcopy(controller.store.bindings[name]);p=scope.root/b["path"];raw=p.read_bytes()
            obj=st.unique_json(raw);mutation(obj);bad=st.serialized(obj);p.write_bytes(bad)
            controller.store.bindings[name]={"path":b["path"],"sha256":st.sha(bad),"size_bytes":len(bad)}
            try:
                result=rejected(controller._verify_development_source_population)
                assert not(set(st.SPLITS["test"])&bound.annotation_ids_read)
                return dict(result,test_ANN_still_unopened=True,fresh_content_not_just_byte_hash=True)
            finally:p.write_bytes(raw);controller.store.bindings[name]=b
        check("fresh_dev_verification_rejects_bound_wrong_span_population",lambda:corrupt_dev_content("dev_ner_probabilities","span",1,lambda o:o["candidates"].reverse()))
        check("fresh_dev_verification_rejects_bound_changed_locked_graph",lambda:corrupt_dev_content("locked_dev_mentions","span",None,lambda o:o.update(original_source_records=[])))
        check("fresh_dev_verification_rejects_bound_wrong_same_seed_head_mentions",lambda:corrupt_dev_content("dev_pair_probabilities","aligned",1,lambda o:o["mentions"][0].update(type="Main")))
        span_name=st.artifact_name("dev_ner_probabilities","span",st.SEEDS[0],1,101)+".json"
        def bad_candidates(mutator):
            obj=controller.store.object(span_name);mutator(obj)
            return rejected(lambda:controller._source_record(obj,"span",st.SEEDS[0],1,101))
        mutations={"duplicate":lambda o:o["candidates"].__setitem__(1,o["candidates"][0]),
                   "negative":lambda o:o["candidates"][0].__setitem__(2,-1),
                   "out_of_text":lambda o:o["candidates"][0].__setitem__(3,100000),
                   "bool_offset":lambda o:o["candidates"][0].__setitem__(2,False),
                   "float_offset":lambda o:o["candidates"][0].__setitem__(2,0.0),
                   "wrong_length":lambda o:o["candidates"][0].pop(),
                   "inverted":lambda o:o["candidates"][0].__setitem__(2,6),
                   "reordered":lambda o:o["candidates"].reverse(),
                   "empty_population":lambda o:o.update(candidates=[],probabilities=[],complete_rows=0),
                   "float_rows":lambda o:o.update(complete_rows=3.0),
                   "float_computational_block":lambda o:o.update(inference_row_block_size=512.0),
                   "wrong_optional_seed":lambda o:o.update(seed=st.SEEDS[1])}
        for name,mutation in mutations.items():check("reject_source_candidates_"+name,lambda mutation=mutation:bad_candidates(mutation))
        def bad_head_mentions():
            name=st.artifact_name("dev_pair_probabilities","aligned",st.SEEDS[0],1,101)+".json";obj=controller.store.object(name)
            locked=copy.deepcopy(obj["mentions"]);obj["mentions"][0]["type"]="INVENTED_UNSUPPORTED"
            return rejected(lambda:controller._source_record(obj,"aligned",st.SEEDS[0],1,101,locked_mentions=locked))
        check("reject_pair_source_changed_locked_type",bad_head_mentions)
        def float_locked_offsets():
            name=st.artifact_name("dev_pair_probabilities","aligned",st.SEEDS[0],1,101)+".json";obj=controller.store.object(name)
            locked=copy.deepcopy(obj["mentions"]);obj["mentions"][0]["segments"][0][0]=0.0
            return rejected(lambda:controller._source_record(obj,"aligned",st.SEEDS[0],1,101,locked_mentions=locked))
        check("reject_pair_source_float_offsets_equal_to_locked_int",float_locked_offsets)
        def native_Main_Element_identity_and_complete_rows():
            name=st.artifact_name("dev_pair_probabilities","aligned",st.SEEDS[0],1,101)+".json";obj=controller.store.object(name)
            mentions=[{"type":"Doping","segments":[[0,5]]},{"type":"Main","segments":[[6,8]]},{"type":"Element","segments":[[6,8]]}]
            probabilities=[[0.]*5 for _ in range(9)];probabilities[1][3]=.8;probabilities[2][3]=.8
            obj.update(mentions=copy.deepcopy(mentions),complete_rows=9,probabilities=probabilities)
            checked=controller._source_record(obj,"aligned",st.SEEDS[0],1,101,locked_mentions=mentions)
            records,_=backend.emitted(101,mentions,checked["probabilities"],.5,.5)
            assert [r["type"] for r in records if r["family"]=="T"]==["Doping","Main","Element"]
            event=[r for r in records if r["family"]=="E"];assert len(event)==1
            assert event[0]["roles"]==[{"role":"Dopant","target":"T2"},{"role":"Dopant","target":"T3"}]
            assert backend.last_emission_diagnostics["complete_directed_pair_rows"]==9
            return {"INVENTED_only":True,"Main_and_Element_distinct_native_types_at_same_span_retained":True,
                    "all9_directed_rows_including_diagonal":True,"same_role_distinct_targets_preserved":True,
                    "no_physical_truth_or_actual_query_Gold":True,"records":records}
        check("complete3mention9row_Main_Element_and_multiple_role_targets",native_Main_Element_identity_and_complete_rows)
        def bad_head_shape(value):
            name=st.artifact_name("dev_pair_probabilities","aligned",st.SEEDS[0],1,101)+".json";obj=controller.store.object(name)
            obj["probabilities"]=value
            return rejected(lambda:controller._source_record(obj,"aligned",st.SEEDS[0],1,101,locked_mentions=obj["mentions"]))
        for name,value in (("flat",[0.]*5),("short",[[0.]*4]),("extra",[[0.]*6]),("bool",[[False]*5]),("empty_nonzero_N",[])):
            check("reject_nonempty_pair_shape_"+name,lambda value=value:bad_head_shape(value))
        def no_Gold_authority():
            old=scope.gold_analysis_allowed;object.__setattr__(scope,"gold_analysis_allowed",False)
            try:
                result=rejected(controller._verify_gold_barrier_and_capture_predictions)
                assert not(set(st.SPLITS["test"])&bound.annotation_ids_read)
                return dict(result,full_source_barrier_does_not_override_separate_Gold_permission=True)
            finally:object.__setattr__(scope,"gold_analysis_allowed",old)
        check("full_source_barrier_still_requires_separate_Gold_scope",no_Gold_authority)
        def malicious_content(mutation):
            name=st.artifact_name("test_source_graphs","aligned",st.SEEDS[0],None,1)+".json";b=copy.deepcopy(controller.store.bindings[name]);p=scope.root/b["path"];raw=p.read_bytes()
            obj=st.unique_json(raw);mutation(obj);bad=st.serialized(obj);p.write_bytes(bad)
            controller.store.bindings[name]={"path":b["path"],"sha256":st.sha(bad),"size_bytes":len(bad)}
            try:
                result=rejected(lambda:controller._verify_complete_source_population(prior_barrier=False))
                assert not(set(st.SPLITS["test"])&bound.annotation_ids_read)
                return dict(result,test_ANN_still_unopened=True)
            finally:p.write_bytes(raw);controller.store.bindings[name]=b
        mutations={"Gold_flag":lambda o:o.update(test_Gold_consulted=True),"nonlist_records":lambda o:o.update(original_source_records={}),
                   "source_ID":lambda o:o["source_probability_record"].update(source_id=2),
                   "source_SHA":lambda o:o["source_probability_record"].update(source_sha256="0"*64),
                   "CP_identity":lambda o:o["source_probability_record"]["checkpoint"].update(sha256="0"*64),
                   "same_seed_NER":lambda o:o["same_seed_detector_source"].update(sha256="0"*64)}
        for name,mutation in mutations.items():check("source_barrier_rejects_bound_content_"+name,lambda mutation=mutation:malicious_content(mutation))
        def empty_choices():
            old=controller.choices;controller.choices={}
            try:return rejected(lambda:controller._verify_complete_source_population(prior_barrier=False))
            finally:controller.choices=old
        check("source_barrier_rejects_empty_choices",empty_choices)
        check("test_semantics_still_rejected_after_barrier",lambda:rejected(lambda:backend.gold(1)))
        check("cannot_replay_before_primary_analysis",lambda:rejected(controller.independent_count_replay))
        check("cannot_independent_statistics_before_native_replay",lambda:rejected(controller.independent_statistics_replay))
        check("immutable_output_collision",lambda:rejected(lambda:controller.store.json("choices/ner.json",{})))
        def missing_population():
            name=st.artifact_name("test_source_graphs","biaffine",st.SEEDS[-1],None,100)+".json"
            old=controller.store.bindings.pop(name)
            try:return rejected(lambda:controller._require_population("test_source_graphs",st.KINDS,"test",epochs=False))
            finally:controller.store.bindings[name]=old
        check("barrier_rejects_missing_one_of1500",missing_population)
        def changed_artifact():
            name="choices/ner.json";path=scope.root/controller.store.bindings[name]["path"];old=path.read_bytes();path.write_bytes(b"{}\n")
            try:return rejected(lambda:controller.store.read(name))
            finally:path.write_bytes(old)
        check("artifact_POST_write_tamper_rejected",changed_artifact)
        def captured_code_ABA():
            name="src/sccomics_fitting_core.py";path=scope.root/name;old=path.read_bytes()
            path.write_bytes(b"raise RuntimeError('INVENTED_LIVE_CODE_REPLACEMENT')\n")
            assert science.get("sccomics_fitting_core").lr_factor(1)==.01
            path.write_bytes(old)
            bound.validate_all();science.validate()
            return {"executed_captured_original_source_not_live_replacement":True,"ABA_live_path_restored":True}
        check("captured_source_ignores_live_ABA_and_pyc",captured_code_ABA)
        def input_tamper():
            name="data/sccomics_round4/source_v3/raw_text/0001.txt";path=scope.root/name;old=path.read_bytes();path.write_bytes(b"X"*len(old))
            try:return rejected(lambda:bound.validate_all())
            finally:path.write_bytes(old)
        check("final_all_text_POST_read_tamper_rejected",input_tamper)
        def private_tamper():
            p=science.files["src/sccomics_fitting_core.py"];old=p.read_bytes();p.write_bytes(b"raise RuntimeError('INVENTED_PRIVATE_TAMPER')\n")
            try:return rejected(science.validate)
            finally:p.write_bytes(old)
        check("private_snapshot_POST_copy_tamper_rejected",private_tamper)
        def symlink_leaf():
            name="data/sccomics_round4/source_v3/raw_text/0001.txt";p=scope.root/name;raw=p.read_bytes();p.unlink();p.symlink_to(scope.root/"outside_sentinel")
            try:return rejected(lambda:bound.read(name,"text"))
            finally:p.unlink();p.write_bytes(raw)
        check("symlink_leaf_rejected",symlink_leaf)
        def symlink_ancestor():
            directory=scope.root/"data/sccomics_round4/source_v3/raw_text";saved=directory.with_name("saved_raw_text")
            directory.rename(saved);directory.symlink_to(saved,target_is_directory=True)
            try:return rejected(lambda:bound.read("data/sccomics_round4/source_v3/raw_text/0001.txt","text"))
            finally:directory.unlink();saved.rename(directory)
        check("symlink_ancestor_rejected",symlink_ancestor)
        # Preserve every invented native input and actual generated artifact.
        bound.validate_all();controller.store.validate();science.validate();backend.validate()
        check("complete_primary_analysis_after_fresh_barrier",lambda:run_analysis(controller,bound))
        check("independent_count_replay_separate_stage",lambda:run_replay(controller))
        check("independent_statistical_arithmetic_separate_stage",lambda:run_independent_statistics(controller))
        shutil.copytree(scope.root,output/"INVENTED_fixture_root")
        science.close()

    with st.invented_workspace() as scope:
        bound=build(scope,source);science=st.CapturedScience(bound,scope.root/"private_source")
        backend=InventedBackend(bound,science);backend.fail_after_documents=3
        controller=st.StageController(bound,backend,science=science)
        def failed_epoch():
            rejected(controller.fit_ner,(MemoryError,))
            assert controller.phase=="failed" and not any(n.startswith("checkpoints/") for n in controller.store.bindings)
            name=st.artifact_name("fit_trajectory","span",st.SEEDS[0],1)+".jsonl"
            records=controller.store.read(name).splitlines()
            assert len(records)==3
            assert st.artifact_name("failed_fit_states","span",st.SEEDS[0],1)+".pt" in controller.store.bindings
            assert controller.store.object("failure.json")["baseline_or_primary_family_success"] is False
            return {"raw3_partial_document_records_preserved":True,"failed_partial_state_preserved":True,"successful_CP":0,"empty_success_not_manufactured":True}
        check("failure_after3_documents_preserves_state_and_logs",failed_epoch)
        check("failure_cannot_resume_as_empty_baseline",lambda:rejected(controller.fit_ner))
        shutil.copytree(scope.root,output/"INVENTED_failure_root")
        science.close()

    # A second complete, manually constructed population produces zero NER in
    # every source. It exercises the legitimate [] -> (0,5) boundary through
    # all complete head grids, test source objects, full Gold FN and replay.
    with st.invented_workspace(gold_analysis_allowed=True) as zero_scope:
        bound=build(zero_scope,source);science=st.CapturedScience(bound,zero_scope.root/"private_source")
        backend=InventedBackend(bound,science);backend.zero_ner=True
        controller=st.StageController(bound,backend,science=science)
        def complete_zero_ner():
            controller.run_full_pipeline()
            assert controller.phase=="controlled_stages_and_all_replays_finished_only"
            assert len([n for n in controller.store.bindings if n.startswith("test_source_graphs/")])==1500
            assert all(controller.store.object(st.artifact_name("test_source_graphs",k,s,None,i)+".json")["original_source_records"]==[]
                       for k in st.KINDS for s in st.SEEDS for i in st.SPLITS["test"])
            primary=controller.store.object("analysis/primary_complete_test_objects.json")
            assert primary["three_fit_pooled_count_scores"]["NER"]["T_complete_native"]["counts"]=={"tp":0,"fp":0,"fn":600}
            for k in st.ARCHITECTURES:
                for endpoint in ("R_trigger_projected_complete","E_global_sharing_consistent_complete"):
                    assert primary["three_fit_pooled_count_scores"]["heads"][k][endpoint]["counts"]=={"tp":0,"fp":0,"fn":300}
            return {"INVENTED_only":True,"complete150CP_15000dev_1500test_sources":True,"zero_NER_not_dropped_or_substituted":True,
                    "strict_empty_pair_shape_restored0x5":True,"all_unknown_and_known_native_Gold_FN_preserved":True}
        check("complete_zero_NER_population_all_stages_and_replay",complete_zero_ner)
        shutil.copytree(zero_scope.root,output/"INVENTED_zeroNER_root");science.close()

    assert all((workspace/n).read_bytes()==raw for n,raw in source.items()),"scientific source changed during artificial check; retain run but no stable certificate"
    summary={"identity":"INVENTED_IMPLEMENTER_SELF_CHECK_ONLY_NOT_INDEPENDENT_REVIEW","real_SC_annotations_weights_cache_or_predictions_read":False,
             "real_HF_or_task_fit_executed":False,"real_performance_P_or_rank_assigned":False,
             "source_sha256":st.sha(source[st.SELF]),"helper_sha256":st.sha(Path(__file__).read_bytes()),
             "source_bindings":{n:st.binding(raw,"code") for n,raw in source.items()},
             "checks":results,"total":len(results),"passed":sum(r["passed"] for r in results),"failed":sum(not r["passed"] for r in results)}
    dump(output/"actual_synthetic_results.json",summary)
    files=[]
    for p in sorted(output.rglob("*")):
        if p.is_file():
            assert not p.is_symlink();raw=p.read_bytes();files.append({"path":str(p.relative_to(output)),"sha256":st.sha(raw),"size_bytes":len(raw)})
    dump(output/"output_manifest.json",{"identity":"INVENTED_FIXTURES_AND_ACTUAL_SYNTHETIC_OUTPUTS_ONLY","files":files,"all_files_reopened_and_hashed":True})
    print(json.dumps({"output_dir":str(output),"total":summary["total"],"passed":summary["passed"],"failed":summary["failed"],"source_sha256":summary["source_sha256"]},sort_keys=True),flush=True)
    return 0 if summary["failed"]==0 else 1


if __name__=="__main__":raise SystemExit(main())
