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

import sccomics_supervised_stages as st


def dump(path,obj):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_bytes(st.serialized(obj))


def build(scope,source_raw):
    inputs={}
    def put(name,raw,duty):
        path=scope.root/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(raw)
        inputs[name]=st.binding(raw,duty)
    for name,raw in source_raw.items():put(name,raw,"code")
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
            cached=st.serialized({"INVENTED":True,"source_id":identifier,"source_sha256":st.sha(text),"source_only":True,"annotation_content_used":False})
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
        self.loads=[];self.fit_starts=[];self.fail_after_documents=None
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
            values=[[0.]*9 for _ in candidates];values[0][self.models.TEXT_TYPES.index("Doping")]=.6
            result={"candidates":candidates,"probabilities":values,"complete_rows":3,"columns":9}
        else:
            result={"mentions":copy.deepcopy(mentions),"probabilities":[[0.]*5 for _ in range(len(mentions)**2)],
                    "complete_rows":len(mentions)**2,"columns":5,"diagnostics":{"INVENTED_full_directed_population":True}}
        return dict(result,source_id=identifier,source_sha256=st.sha(self.text(identifier)),source_only=True,Gold_consulted=False,
                    candidate_cap=None,inference_row_block_size=512)

    def detector_mentions(self,obj,threshold):
        return self.core.predicted_mentions(obj["candidates"],obj["probabilities"],threshold)

    def detector_document(self,identifier,mentions):
        raw=self.text(identifier);text=raw.decode()
        records=[{"family":"T","id":f"T{i+1}","type":m["type"],"segments":m["segments"],
                  "text":" ".join(text[a:b] for a,b in m["segments"])} for i,m in enumerate(mentions)]
        return records,self.native.parse_prediction_records(raw,records)

    def emitted(self,identifier,mentions,probabilities,rt,et):
        text=self.text(identifier)
        records,diagnostics=self.projection.emit_native_prediction_records(text.decode(),mentions,probabilities,relation_threshold=rt,role_threshold=et)
        self.last_emission_diagnostics=copy.deepcopy(diagnostics)
        return records,self.native.parse_prediction_records(text,records)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("--output-dir",required=True)
    args=parser.parse_args();output=Path(args.output_dir).absolute()
    assert not output.exists(),"historical synthetic outputs never overwritten"
    output.mkdir(parents=True)
    assert not any(p.is_symlink() for p in (output,*output.parents))
    workspace=Path(__file__).resolve().parents[1]
    source={n:(workspace/n).read_bytes() for n in st.SCIENCE+(st.SELF,)}
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
    check("reject_direct_invented_scope_constructor",lambda:rejected(lambda:st._Scope(workspace,True,"outputs",None,object())))

    with st.invented_workspace() as scope:
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
        check("cannot_score_unimplemented_before_barrier",lambda:rejected(controller.score_and_statistics,(st.UnimplementedStageError,)))
        check("frozen_scope_root_cannot_be_repointed",lambda:rejected(lambda:setattr(scope,"root",workspace),(FrozenInstanceError,)))
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
        check("test_semantics_still_rejected_after_barrier",lambda:rejected(lambda:backend.gold(1)))
        check("scoring_explicitly_unimplemented_after_barrier",lambda:rejected(controller.score_and_statistics,(st.UnimplementedStageError,)))
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
