"""Controlled pretest SC stages; default command refuses prospective authority.

This module imports no scientific library/model at startup. Actual use requires
an explicit complete scientific freeze and final gate with byte-bound source
review. Invented contexts are newly allocated private temporary roots only.
The currently implemented endpoint is a pretest source barrier, NOT scoring or
completed science. Test Gold scoring/statistics is explicitly unimplemented.
"""
from __future__ import annotations

import argparse
import ast
import builtins
from contextlib import contextmanager
from dataclasses import dataclass, field
import copy
import gc
import hashlib
import importlib
import io
import json
import math
import os
from pathlib import Path, PurePosixPath
import platform
import stat
import sys
import tempfile
import types

SELF = "src/sccomics_supervised_stages.py"
CONFIG = "research/round4_sccomics_experimental_configuration_before_fit.json"
EXECUTION = "research/round4_sccomics_execution_details_before_fit.json"
AMENDMENT = "research/round4_sccomics_pre_fit_ner_denominator_amendment.json"
PROSPECTIVE = {
    CONFIG: "742fb7011f2162cb935bd6de3068e38a8c0f3a67c58df8ff266dc73a8814103c",
    EXECUTION: "5c5891319150baa57f8c6eaf454a344a37b43ab3d740a75cbdf6b82cde31a205",
    AMENDMENT: "e699e091bac59b1f14c5204c2eedee3b6a40a376f45a10229ccf6697f818d9fc",
}
# The amendment hash is validated against explicit root-provided SHA in the
# complete freeze too. A changed source review requires a retained new freeze;
# core/selector version hashes are deliberately not hardcoded here.
SCIENCE = tuple("src/" + name + ".py" for name in (
    "sccomics_source_models", "sccomics_training_projection", "sccomics_native_graph",
    "sccomics_fitting_core", "sccomics_development_selection", "sccomics_primary_matching_v2", "sccomics_statistics"))
FINAL_ANALYSIS = ("src/sccomics_independent_scoring.py", "src/sccomics_test_analysis.py")
ACQUISITION = "data/sccomics_round4/source_v3/source_acquisition_manifest.json"
ACQUISITION_SHA = "a38f1637af74b15246b6b028d3d58b5836094e905d25845e5922668031fee1c7"
CACHE_RECEIPT = "results/local_baseline/sccomics_native_v1/source_cache/completed_manifest.json"
CACHE_RECEIPT_SHA = "fe7b66198773d49e7652c7be64d3f5923fdba27c528cb73303f0b95a36194139"
COLD_ROOT = "data/local_baselines/matscibert"
COLD_RECEIPT = COLD_ROOT + "/source_manifest.json"
COLD_NAMES = ("README.md", "config.json", "model.safetensors", "special_tokens_map.json",
              "tokenizer.json", "tokenizer_config.json", "vocab.txt")
FREEZE_PATH = "research/round4_sccomics_complete_scientific_freeze.json"
GATE_PATH = "research/round4_sccomics_supervised_final_gate.json"
SOURCE_REVIEW = "research/round4_sccomics_supervised_complete_source_review.json"
RUNTIME_RECEIPT = "research/round4_sccomics_supervised_runtime_receipt.json"
RUNTIME = {"python":"3.13.7", "numpy":"2.2.6", "torch":"2.9.1", "transformers":"4.57.3",
           "tokenizers":"0.22.1", "networkx":"3.6.1"}
SEEDS = (20261013,20261014,20261015)
ARCHITECTURES = ("aligned","permuted","ordered_context","biaffine")
KINDS = ("span",) + ARCHITECTURES
SPLITS = {"train":tuple(range(201,1001)),"dev":tuple(range(101,201)),"test":tuple(range(1,101))}
SPAN_THRESHOLDS = (.05,.1,.25,.5,.75,.9,.95,.99)
PAIR_THRESHOLDS = (.5,.75,.9,.95,.99)
_SCOPES = set()
_SCOPE_CONSTRUCTION = object()


class StageIntegrityError(ValueError):
    pass


class UnimplementedStageError(RuntimeError):
    pass


def require(ok, message):
    if not ok:
        raise StageIntegrityError(message)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def serialized(obj):
    return (json.dumps(obj,sort_keys=True,separators=(",",":"),ensure_ascii=True,allow_nan=False)+"\n").encode()


def unique_json(raw):
    def pairs(items):
        d={}
        for k,v in items:
            require(k not in d,"duplicate JSON key")
            d[k]=v
        return d
    def invalid(value):
        raise StageIntegrityError("nonfinite JSON constant")
    return json.loads(raw.decode("utf-8","strict"),object_pairs_hook=pairs,parse_constant=invalid)


def relative(value):
    require(type(value) is str and value and "\\" not in value and "\x00" not in value,
            "explicit POSIX root-relative path required")
    parts=PurePosixPath(value).parts
    require(not PurePosixPath(value).is_absolute() and all(p not in (".","..") for p in parts)
            and value=="/".join(parts) and not any(c in value for c in "*?[]:"),"unsafe/noncanonical path")
    return parts


def root_path(root):
    root=Path(root).absolute()
    require(root.is_dir() and not any(p.is_symlink() for p in (root,*root.parents)),"root/ancestor symlink or missing root")
    return root


@contextmanager
def opened(root,name,*,write=False):
    """openat/no-follow prevents ancestor and leaf symlink race escapes."""
    parts=relative(name)
    root=root_path(root)
    descriptor=os.open(root,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW)
    try:
        for component in parts[:-1]:
            try:
                next_fd=os.open(component,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW,dir_fd=descriptor)
            except FileNotFoundError:
                require(write,"missing input ancestor")
                os.mkdir(component,0o700,dir_fd=descriptor)
                next_fd=os.open(component,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW,dir_fd=descriptor)
            os.close(descriptor)
            descriptor=next_fd
        flags=(os.O_WRONLY|os.O_CREAT|os.O_EXCL if write else os.O_RDONLY)|os.O_NOFOLLOW
        leaf=os.open(parts[-1],flags,0o600,dir_fd=descriptor)
        try:
            require(stat.S_ISREG(os.fstat(leaf).st_mode),"regular file required")
            yield leaf
        finally:
            os.close(leaf)
    finally:
        os.close(descriptor)


def raw_read(root,name):
    with opened(root,name) as fd:
        pieces=[]
        while chunk:=os.read(fd,1024*1024):
            pieces.append(chunk)
        return b"".join(pieces)


def new_directory(root,name):
    parts=relative(name);root=root_path(root)
    descriptor=os.open(root,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW)
    try:
        for i,component in enumerate(parts):
            try:
                os.mkdir(component,0o700,dir_fd=descriptor)
            except FileExistsError:
                require(i<len(parts)-1,"fresh output directory required")
            next_fd=os.open(component,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW,dir_fd=descriptor)
            os.close(descriptor);descriptor=next_fd
    finally:
        os.close(descriptor)


def digest_file(root,name):
    with opened(root,name) as fd:
        digest=hashlib.sha256()
        size=0
        while chunk:=os.read(fd,1024*1024):
            digest.update(chunk);size+=len(chunk)
        return {"sha256":digest.hexdigest(),"size_bytes":size}


def binding(raw,duty):
    return {"sha256":sha(raw),"size_bytes":len(raw),"duty":duty}


def strict_binding(b):
    require(type(b) is dict and set(b)=={"sha256","size_bytes","duty"},"binding schema")
    require(type(b["sha256"]) is str and len(b["sha256"])==64 and all(c in "0123456789abcdef" for c in b["sha256"]),"SHA256 required")
    require(type(b["size_bytes"]) is int and b["size_bytes"]>=0 and type(b["duty"]) is str,"binding size/duty")


def path_duty(name,duty):
    relative(name)
    if duty=="code":
        require(name in SCIENCE+(SELF,)+FINAL_ANALYSIS,"closed imported/final scientific-source population")
    elif duty=="control":
        require(name in tuple(PROSPECTIVE)+(ACQUISITION,CACHE_RECEIPT,COLD_RECEIPT,SOURCE_REVIEW,RUNTIME_RECEIPT,FREEZE_PATH,GATE_PATH),"closed metadata duties")
    elif duty=="cold":
        require(name in tuple(COLD_ROOT+"/"+n for n in COLD_NAMES),"seven original cold files only")
    elif duty in {"text","annotation"}:
        suffix="txt" if duty=="text" else "ann"
        directory="raw_text" if duty=="text" else "raw_annotations"
        require(name in {f"data/sccomics_round4/source_v3/{directory}/{i:04}.{suffix}" for i in range(1,1001)},"native source path population")
    elif duty=="cache":
        require(name in {f"results/local_baseline/sccomics_native_v1/source_cache/{s}/{i:04}.pt" for s,ids in SPLITS.items() for i in ids},"cold text-cache path population")
    else:
        raise StageIntegrityError("undeclared I/O duty")


@dataclass(frozen=True,eq=False)
class _Scope:
    root:Path
    invented:bool
    output_prefix:str
    authority_sha256:str|None
    token:object=field(repr=False)

    def __post_init__(self):
        require(self.token is _SCOPE_CONSTRUCTION,"scope construction is not public authority")
        object.__setattr__(self,"root",root_path(self.root))
        if self.invented:
            require(self.root.parent==Path(tempfile.gettempdir()).resolve() and
                    self.root.name.startswith("INVENTED_sccomics_stages_"),"new owned temporary invented root required")
        _SCOPES.add(self)


@contextmanager
def invented_workspace():
    """No caller-selected invented root can grant access to a real checkout."""
    with tempfile.TemporaryDirectory(prefix="INVENTED_sccomics_stages_") as directory:
        scope=_Scope(Path(directory).resolve(),True,"outputs",None,_SCOPE_CONSTRUCTION)
        try:
            yield scope
        finally:
            _SCOPES.remove(scope)


class BoundIO:
    def __init__(self,scope,inputs):
        require(scope in _SCOPES,"genuine actual-proof or newly allocated invented scope required")
        self.scope=scope;self.root=scope.root;self.inputs=copy.deepcopy(inputs)
        self.annotation_ids_read=set();self.annotation_first_reads=[];self._controller=None
        for name,b in self.inputs.items():
            strict_binding(b);path_duty(name,b["duty"])
        required_code=SCIENCE+(SELF,)+(tuple() if scope.invented else FINAL_ANALYSIS)
        require(set(n for n,b in inputs.items() if b["duty"]=="code")==set(required_code),"complete captured scientific source closure")
        require(set(n for n,b in inputs.items() if b["duty"]=="cold")=={COLD_ROOT+"/"+n for n in COLD_NAMES},"complete seven cold files")
        for duty in ("text","annotation","cache"):
            require(sum(b["duty"]==duty for b in inputs.values())==1000,"complete1000 source/cache/annotation bindings required")

    def read(self,name,duty):
        require(name in self.inputs and self.inputs[name]["duty"]==duty,"unbound file or wrong duty")
        if duty=="annotation":
            identifier=int(Path(name).stem)
            require(self._controller is not None,"no stage-bound semantic reader")
            stage=self._controller._annotation_permission(identifier)
            require(type(stage) is str and stage,"annotation semantic permission not enabled")
        raw=raw_read(self.root,name)
        b=self.inputs[name]
        require(len(raw)==b["size_bytes"] and sha(raw)==b["sha256"],f"captured bytes identity differs: {name}")
        if duty=="annotation" and identifier not in self.annotation_ids_read:
            event={"native_id":identifier,"stage":stage,"captured_raw_sha256":sha(raw),"size_bytes":len(raw),
                   "first_semantic_read_in_this_run":True,"prior_acquisition_byte_exposure_not_denied":True}
            self.annotation_first_reads.append(event);self.annotation_ids_read.add(identifier)
        return raw

    def bind_controller(self,controller):
        require(self._controller is None and controller.io is self,"one immutable controller ownership per input ledger")
        self._controller=controller

    def validate_all(self):
        """Unopened held ANN are metadata-only; never claim their raw rehash."""
        actual_hashes=[];annotation_receipt_only=[]
        for name,b in self.inputs.items():
            if b["duty"]=="annotation" and int(Path(name).stem) not in self.annotation_ids_read:
                # No open() and no content read at this boundary; lstat each
                # canonical ancestor/leaf, retaining the original receipt SHA.
                path=self.root.joinpath(*relative(name))
                require(not any(p.is_symlink() for p in (path,*path.parents)),"annotation metadata symlink")
                status=path.stat()
                require(stat.S_ISREG(status.st_mode) and status.st_size==b["size_bytes"],"unopened annotation path/size differs")
                annotation_receipt_only.append(name)
            else:
                actual=digest_file(self.root,name)
                require(actual=={k:b[k] for k in ("sha256","size_bytes")},f"startup/final input changed: {name}")
                actual_hashes.append(name)
        return {"all_nonannotation_inputs_and_opened_annotations_actual_rehashed":actual_hashes,
                "unopened_annotations_receipt_path_size_only":annotation_receipt_only,
                "all_annotations_actual_rehashed_claim":False}

    def private_snapshot(self,names,destination):
        """Stream once into private bytes, validate before any consumer load."""
        destination=Path(destination)
        require(not destination.exists(),"private snapshot collision")
        destination.mkdir(mode=0o700)
        mappings={}
        for name in names:
            require(name in self.inputs and self.inputs[name]["duty"] in {"code","cold"},"snapshot only scientific source/original cold")
            target=destination/Path(name).name
            with opened(self.root,name) as source_fd, target.open("xb") as sink:
                digest=hashlib.sha256();size=0
                while chunk:=os.read(source_fd,1024*1024):
                    sink.write(chunk);digest.update(chunk);size+=len(chunk)
                sink.flush();os.fsync(sink.fileno())
            require(digest.hexdigest()==self.inputs[name]["sha256"] and size==self.inputs[name]["size_bytes"],"captured private snapshot identity differs")
            require(target.read_bytes() if self.inputs[name]["duty"]=="code" else True,"empty scientific source")
            require(digest_file(destination,target.name)=={"sha256":digest.hexdigest(),"size_bytes":size},"private snapshot final bytes differ")
            mappings[name]=target
        return mappings


class CapturedScience:
    """Compile actual captured bytes with private local import resolution.

    No project sys.path, sys.modules scientific cache, .pyc or source-path reread
    supplies scientific definitions. stdlib/site-package dependencies remain
    separately version/origin checked by the actual backend.
    """
    def __init__(self,bound_io,directory):
        self.files=bound_io.private_snapshot(tuple(n for n,b in bound_io.inputs.items() if b["duty"]=="code"),directory)
        self.raw={Path(n).stem:p.read_bytes() for n,p in self.files.items()}
        require(all(sha(self.raw[Path(n).stem])==bound_io.inputs[n]["sha256"] and
                    len(self.raw[Path(n).stem])==bound_io.inputs[n]["size_bytes"] for n in self.files),
                "actual captured scientific bytes differ before compilation")
        self.modules={};self.prefix="_sccomics_captured_"+sha(serialized({n:sha(r) for n,r in self.raw.items()}))[:16]
        for name,raw in self.raw.items():
            for node in ast.walk(ast.parse(raw)):
                if isinstance(node,ast.ImportFrom) and (node.module or "").startswith("sccomics_"):
                    require(node.level==0 and node.module in self.raw,"uncaptured scientific import")
                if isinstance(node,ast.Import):
                    require(all(not a.name.startswith("sccomics_") or a.name in self.raw for a in node.names),"uncaptured scientific import")

    def get(self,name):
        require(name in self.raw,"unbound scientific module")
        if name in self.modules:
            return self.modules[name]
        qualified=self.prefix+"."+name
        module=types.ModuleType(qualified);module.__file__=str(self.files["src/"+name+".py"])
        self.modules[name]=module;sys.modules[qualified]=module
        original=builtins.__import__
        def captured_import(import_name,globals=None,locals=None,fromlist=(),level=0):
            if import_name.startswith("sccomics_"):
                require(level==0 and import_name in self.raw,"unbound scientific import")
                return self.get(import_name)
            return original(import_name,globals,locals,fromlist,level)
        private_builtins=dict(vars(builtins),__import__=captured_import)
        module.__dict__["__builtins__"]=private_builtins
        try:
            exec(compile(self.raw[name],module.__file__,"exec",dont_inherit=True),module.__dict__)
        except BaseException:
            del self.modules[name];sys.modules.pop(qualified,None)
            raise
        return module

    def validate(self):
        for name,path in self.files.items():
            require(sha(path.read_bytes())==sha(self.raw[Path(name).stem]),"captured scientific snapshot changed")

    def close(self):
        for name in self.modules:
            sys.modules.pop(self.prefix+"."+name,None)


class Artifacts:
    def __init__(self,scope):
        require(scope in _SCOPES,"unproven artifact scope")
        self.root=scope.root;self.prefix=scope.output_prefix;relative(self.prefix)
        directory=self.root.joinpath(*relative(self.prefix))
        require(not directory.exists(),"new run output required; no overwrite/implicit resume")
        new_directory(self.root,self.prefix)
        self.bindings={}

    def write(self,name,raw):
        relative(name);require(type(raw) is bytes,"artifact raw bytes required")
        path=self.prefix+"/"+name
        require(name not in self.bindings,"artifact overwrite")
        with opened(self.root,path,write=True) as fd:
            view=memoryview(raw)
            while view:
                count=os.write(fd,view);require(count>0,"incomplete artifact write");view=view[count:]
            os.fsync(fd)
        b={"path":path,"sha256":sha(raw),"size_bytes":len(raw)}
        require(digest_file(self.root,path)=={k:b[k] for k in ("sha256","size_bytes")},"written artifact bytes differ")
        self.bindings[name]=b
        return b

    def json(self,name,obj):
        return self.write(name,serialized(obj))

    def read(self,name):
        require(name in self.bindings,"missing required source artifact")
        b=self.bindings[name];raw=raw_read(self.root,b["path"])
        require(sha(raw)==b["sha256"] and len(raw)==b["size_bytes"],"source artifact changed")
        return raw

    def object(self,name):
        return unique_json(self.read(name))

    def validate(self):
        for name in self.bindings:
            self.read(name)
        return copy.deepcopy(self.bindings)

    def journal(self,name):
        return StreamingJournal(self,name)


class StreamingJournal:
    """Durable partial document/update history survives failed epochs."""
    def __init__(self,store,name):
        relative(name);require(name not in store.bindings,"journal collision")
        self.store=store;self.name=name;self.path=store.prefix+"/"+name
        self.manager=opened(store.root,self.path,write=True);self.fd=self.manager.__enter__()
        self.digest=hashlib.sha256();self.size=0;self.closed=False

    def append(self,value):
        require(not self.closed,"closed journal")
        raw=serialized(value);view=memoryview(raw)
        while view:
            n=os.write(self.fd,view);require(n>0,"partial trajectory write");view=view[n:]
        self.digest.update(raw);self.size+=len(raw)
        # Each completed backward is recoverable; this is a fixed durability
        # policy, not a result-driven logging budget.
        os.fsync(self.fd)

    def close(self):
        if not self.closed:
            os.fsync(self.fd);self.manager.__exit__(None,None,None);self.closed=True
            b={"path":self.path,"sha256":self.digest.hexdigest(),"size_bytes":self.size}
            require(digest_file(self.store.root,self.path)=={k:b[k] for k in ("sha256","size_bytes")},"partial/full trajectory bytes differ")
            self.store.bindings[self.name]=b
        return self.store.bindings[self.name]


def _controls_from_receipts(root,captures):
    """Only explicit original receipt metadata expands native data duties."""
    acq=unique_json(captures[ACQUISITION]);cache=unique_json(captures[CACHE_RECEIPT]);cold=unique_json(captures[COLD_RECEIPT])
    require(acq.get("native_source_abstract_records")==1000 and acq.get("split_ids")=={k:list(v) for k,v in SPLITS.items()},"native source/fold receipt differs")
    groups={}
    for archive in acq["official_archives"].values():
        for entry in archive["members"].values():
            category=entry.get("category")
            if category in {"raw_source_document","unparsed_annotation_bytes"}:
                canonical=Path(entry["canonical_path"])
                name=str(canonical.relative_to(root)) if canonical.is_absolute() else str(canonical)
                duty="text" if category=="raw_source_document" else "annotation"
                path_duty(name,duty)
                require(name not in groups,"duplicate native receipt path")
                groups[name]={"sha256":entry["sha256"],"size_bytes":entry["bytes"],"duty":duty}
    require(cache.get("status")=="completed_cold_source_text_cache_only" and cache.get("complete_native_records")==1000
            and cache.get("all_captured_source_model_and_cache_hashes_size_final_checked") is True,"actual source-cache completion required")
    for entry in cache["cache_bindings"]:
        name=entry["path"];path_duty(name,"cache")
        require(name not in groups,"duplicate cache input")
        identifier=entry["native_id"];split=entry["split"]
        require(type(identifier) is int and identifier in SPLITS[split] and name.endswith(f"/{split}/{identifier:04}.pt"),"cache source/fold identity")
        require(entry["source_sha256"]==groups[f"data/sccomics_round4/source_v3/raw_text/{identifier:04}.txt"]["sha256"],"cache/native text SHA differs")
        groups[name]={"sha256":entry["sha256"],"size_bytes":entry["bytes"],"duty":"cache"}
    require(cold.get("revision")=="ced9d8f5f208712c4a90f98a246fe32155b29995" and
            cold.get("base_encoder_not_task_finetuned_checkpoint") is True and len(cold.get("files",[]))==7,"original cold model receipt")
    require({r["name"] for r in cold["files"]}==set(COLD_NAMES),"seven original cold names differ")
    for entry in cold["files"]:
        name=COLD_ROOT+"/"+entry["name"]
        groups[name]={"sha256":entry["sha256"],"size_bytes":entry["bytes"],"duty":"cold"}
    return groups


def authorize_actual(root,*,freeze_sha256,gate_sha256):
    """Only complete actual freeze + separately pinned final human/root gate.

    All current prospective protocols deliberately fail at status validation.
    This function reads control/source-review metadata before any cold/cache or
    annotation bytes. No fixture flag can redirect the actual authority path.
    """
    root=root_path(root)
    freeze_raw=raw_read(root,FREEZE_PATH);gate_raw=raw_read(root,GATE_PATH)
    require(sha(freeze_raw)==freeze_sha256 and sha(gate_raw)==gate_sha256,"explicit final authority SHA differs")
    freeze,gate=unique_json(freeze_raw),unique_json(gate_raw)
    require(freeze.get("status")=="actual_complete_scientific_freeze" and freeze.get("complete") is True
            and freeze.get("synthetic_fixture") is False,"no complete actual scientific freeze; prospective documents cannot authorize fitting")
    require(set(gate)=={"status","freeze_sha256","stage","output_relative_path","actual_fitting_allowed","paid_API_or_credentials_allowed","synthetic_fixture"},"closed final stage gate")
    require(gate["status"]=="root_authorized_actual_supervised_stage_run" and gate["freeze_sha256"]==freeze_sha256 and
            gate["stage"]=="pretest_source" and gate["actual_fitting_allowed"] is True and
            gate["paid_API_or_credentials_allowed"] is False and gate["synthetic_fixture"] is False,"wrong actual stage authority")
    require(gate["output_relative_path"]=="results/local_baseline/sccomics_native_v1/supervised_v1","fixed fresh supervised output path required")
    require(freeze.get("runtime")==RUNTIME and freeze.get("native_split_ids")=={k:list(v) for k,v in SPLITS.items()},"frozen runtime/folds differ")
    require(freeze.get("implemented_final_scoring_and_independent_replay") is True,"complete external scoring/replay still required before any actual fit")
    inputs=freeze.get("input_bindings")
    require(type(inputs) is dict,"closed input metadata required")
    controls=set(PROSPECTIVE)|{ACQUISITION,CACHE_RECEIPT,COLD_RECEIPT,SOURCE_REVIEW,RUNTIME_RECEIPT}
    actual_code=SCIENCE+(SELF,)+FINAL_ANALYSIS
    require(set(inputs)==controls|set(actual_code),"closed complete source/control/scoring/replay bindings")
    captures={}
    for name,b in inputs.items():
        strict_binding(b);path_duty(name,b["duty"])
        require(b["duty"]==("code" if name in actual_code else "control"),"source/control duty differs")
        raw=raw_read(root,name)
        require(sha(raw)==b["sha256"] and len(raw)==b["size_bytes"],"captured proof/source bytes differ")
        captures[name]=raw
    for name,digest in PROSPECTIVE.items():
        require(inputs[name]["sha256"]==digest,"prospective configuration/amendment identity differs")
    require(inputs[ACQUISITION]["sha256"]==ACQUISITION_SHA and inputs[CACHE_RECEIPT]["sha256"]==CACHE_RECEIPT_SHA,"actual source/cold-cache receipt identity differs")
    require(unique_json(captures[RUNTIME_RECEIPT]).get("runtime")==RUNTIME,"runtime receipt differs")
    review=unique_json(captures[SOURCE_REVIEW])
    require(review.get("status")=="source_only_review_passed" and review.get("independent_from_implementer") is True
            and review.get("all_individual_source_reviews_and_artificial_checks_verified") is True
            and review.get("reviewed_source_sha256_by_path")=={n:inputs[n]["sha256"] for n in actual_code},"complete independent imported/scoring/replay-source review required")
    scope=_Scope(root,False,gate["output_relative_path"],freeze_sha256,_SCOPE_CONSTRUCTION)
    material=_controls_from_receipts(root,captures)
    inputs={**inputs,**material,FREEZE_PATH:binding(freeze_raw,"control"),GATE_PATH:binding(gate_raw,"control")}
    bound=BoundIO(scope,inputs)
    bound.validate_all()
    return scope,bound


def artifact_name(role,kind,seed,epoch=None,source_id=None):
    require(kind in KINDS and type(seed) is int and seed in SEEDS,"fixed kind/seed")
    name=f"{role}/{kind}/{seed}"
    if epoch is not None:
        require(type(epoch) is int and 1<=epoch<=10,"fixed epoch")
        name+=f"/epoch{epoch:02}"
    if source_id is not None:
        require(type(source_id) is int and 1<=source_id<=1000,"native ID")
        name+=f"/{source_id:04}"
    return name


class TorchBackend:
    """Real backend callable only from actual complete proof; not tested on data.

    Invented fixtures use a separate fake backend, never this cold/model loader.
    Every fitted model is freshly initialized from a private original snapshot.
    """
    def __init__(self,bound,science,temporary_parent):
        require(not bound.scope.invented and bound.scope.authority_sha256,"actual proof required before scientific-library/model load")
        self.bound=bound;self.science=science;self.modules={}
        require(platform.python_version()==RUNTIME["python"],"fixed Python runtime")
        for name,version in RUNTIME.items():
            if name=="python":continue
            module=importlib.import_module(name)
            origin=Path(module.__file__).absolute()
            require(getattr(module,"__version__",None)==version and not origin.is_relative_to(bound.root),"runtime version or project-shadow library differs")
            self.modules[name]=module
        self.torch=self.modules["torch"]
        require(self.torch.backends.mps.is_available(),"fixed serial MPS runtime unavailable; no silent fallback")
        self.device="mps"
        self.core=science.get("sccomics_fitting_core");self.models=science.get("sccomics_source_models")
        self.projection=science.get("sccomics_training_projection");self.native=science.get("sccomics_native_graph")
        self.selector=science.get("sccomics_development_selection")
        self.cold_directory=Path(temporary_parent)/"cold"
        self.cold_map=bound.private_snapshot(tuple(COLD_ROOT+"/"+n for n in COLD_NAMES),self.cold_directory)
        base=self.modules["transformers"].BertModel.from_pretrained(str(self.cold_directory),
            local_files_only=True,trust_remote_code=False,use_safetensors=True)
        require(len(base.encoder.layer)==12,"original12 blocks required")
        self.original_last=copy.deepcopy(base.encoder.layer[11]).cpu()
        del base;gc.collect()

    def validate(self):
        for name,path in self.cold_map.items():
            require(digest_file(path.parent,path.name)=={k:self.bound.inputs[name][k] for k in ("sha256","size_bytes")},"used original cold snapshot changed")
        self.science.validate()

    def cold_model(self,kind,seed):
        self.core.seed_fit(seed)
        self.validate()
        # Independent deep copy of the retained original last block. The full
        # original is read only once from the checked private local snapshot;
        # no prior fit mutates the CPU cold template and no checkpoint resumes.
        last=copy.deepcopy(self.original_last)
        model=(self.models.SpanDetector(last) if kind=="span" else self.models.DirectedHead(last,kind)).to(self.device)
        return model

    def rng(self):
        numpy=self.modules["numpy"];random=importlib.import_module("random")
        ns=numpy.random.get_state()
        return {"python_random":random.getstate(),"numpy":{"algorithm":ns[0],"state":ns[1].tolist(),
            "position":ns[2],"has_gauss":ns[3],"cached_gaussian":ns[4]},
            "torch_cpu":self.torch.get_rng_state().tolist(),"torch_mps":self.torch.mps.get_rng_state().tolist()}

    def cache(self,identifier):
        split=next(s for s,ids in SPLITS.items() if identifier in ids)
        raw=self.bound.read(f"results/local_baseline/sccomics_native_v1/source_cache/{split}/{identifier:04}.pt","cache")
        # Captured bytes, never a checked-then-reopened live model/cache path.
        saved=self.torch.load(io.BytesIO(raw),map_location="cpu",weights_only=True)
        self.core._validate_cache(saved)
        expected=self.bound.inputs[f"data/sccomics_round4/source_v3/raw_text/{identifier:04}.txt"]["sha256"]
        require(saved["source_id"]==identifier and saved["split"]==split and saved["source_sha256"]==expected,"cache/source identity differs")
        return saved

    def text(self,identifier):
        return self.bound.read(f"data/sccomics_round4/source_v3/raw_text/{identifier:04}.txt","text")

    def gold(self,identifier):
        raw=self.bound.read(f"data/sccomics_round4/source_v3/raw_annotations/{identifier:04}.ann","annotation")
        return self.native.parse_document(self.text(identifier),raw,gold=True)

    def fit_start(self,kind,seed):
        model=self.cold_model(kind,seed);optimizer=self.core.make_optimizer(model)
        metadata={"optimizer_partitions":[{k:v for k,v in g.items() if k!="params"} for g in optimizer.param_groups],
                  "rng":self.rng(),"cold_start":True,"implicit_resume":False}
        return {"model":model,"optimizer":optimizer},metadata

    def fit_epoch(self,handle,kind,seed,epoch,log_event):
        def source(identifier):
            require(identifier in SPLITS["train"],"TRAIN800 only")
            return self.gold(identifier),self.cache(identifier)
        summaries=self.core.train_epoch(handle["model"],handle["optimizer"],source,kind=kind,seed=seed,epoch=epoch,
            device=self.device,log_event=log_event,source_ids=SPLITS["train"],total_updates=1000,warmup=100)
        require(len(summaries)==100 and summaries[-1]["update"]==epoch*100,"complete fixed update population")
        payload={k:v.detach().cpu() for k,v in handle["model"].state_dict().items()}
        saved=io.BytesIO();self.torch.save(payload,saved)
        return saved.getvalue(),{"rng":self.rng(),"optimizer_updates":summaries,"complete_model_state_dict":True,
            "optimizer_moments_in_epoch_checkpoint":False,"final_scheduled_update_zero_lr_still_step":epoch==10}

    def release(self,handle):
        handle.clear();gc.collect();self.torch.mps.empty_cache()

    def failure_state(self,handle):
        saved=io.BytesIO()
        self.torch.save({k:v.detach().cpu() for k,v in handle["model"].state_dict().items()},saved)
        return saved.getvalue(),{"rng":self.rng(),"failed_partial_state_not_a_successful_checkpoint":True}

    def inference_start(self,kind,seed,checkpoint_raw):
        model=self.cold_model(kind,seed)
        weights=self.torch.load(io.BytesIO(checkpoint_raw),map_location="cpu",weights_only=True)
        model.load_state_dict(weights,strict=True)
        del weights
        return {"model":model}

    def source_probabilities(self,handle,kind,seed,identifier,mentions=None):
        model=handle["model"]
        saved=self.cache(identifier)
        text=self.text(identifier)
        if kind=="span":
            candidates,values=self.core.span_probabilities(model,saved,self.device)
            require(len(candidates)==saved["candidate_span_count"],"all source-derived span candidates required")
            result={"candidates":[list(c) for c in candidates],"probabilities":values.tolist(),"complete_rows":len(candidates),"columns":9}
        else:
            require(type(mentions) is list,"locked source detector mentions required")
            values,diagnostics=self.core.pair_probabilities(model,saved,mentions,seed=seed,device=self.device,text_length=len(text.decode("utf-8","strict")))
            result={"mentions":copy.deepcopy(mentions),"probabilities":values.tolist(),"complete_rows":len(mentions)**2,"columns":5,"diagnostics":diagnostics}
        return dict(result,source_id=identifier,source_sha256=sha(text),source_only=True,Gold_consulted=False,
            candidate_cap=None,inference_row_block_size=512)

    def detector_mentions(self,obj,threshold):
        return self.core.predicted_mentions(obj["candidates"],obj["probabilities"],threshold)

    def emitted(self,identifier,mentions,probabilities,rt,et):
        raw=self.text(identifier)
        records,diagnostics=self.projection.emit_native_prediction_records(raw.decode("utf-8","strict"),mentions,probabilities,
            relation_threshold=rt,role_threshold=et)
        self.last_emission_diagnostics=copy.deepcopy(diagnostics)
        return records,self.native.parse_prediction_records(raw,records)

    def detector_document(self,identifier,mentions):
        raw=self.text(identifier);text=raw.decode("utf-8","strict")
        records=[{"family":"T","id":f"T{i+1}","type":m["type"],"segments":copy.deepcopy(m["segments"]),
                  "text":" ".join(text[a:b] for a,b in m["segments"])} for i,m in enumerate(mentions)]
        return records,self.native.parse_prediction_records(raw,records)


class StageController:
    """Serial 30NER→120head fitting/generation/selection/source-barrier API.

    No stage skips, implicit resume, partial successful family or file-loader
    callbacks selected by an annotation outcome. Failure permanently stops this
    controller and preserves all preceding immutable artifacts.
    """
    def __init__(self,bound,backend,*,science=None):
        require(bound.scope in _SCOPES,"unproven scope")
        require(bound.scope.invented or isinstance(backend,TorchBackend),"actual run cannot install an arbitrary fake backend")
        self.io=bound;self.backend=backend;self.science=science;self.store=Artifacts(bound.scope)
        self.phase="created";self.busy=False;self.events=[];self.choices={}
        self.io.bind_controller(self)
        self.store.json("started.json",{"scope":"INVENTED" if bound.scope.invented else "actual_authorized_pretest_stages_only",
            "authority_sha256":bound.scope.authority_sha256,"test_Gold_scoring_implemented":False,
            "prior_exposure":"Original acquisition byte identity and priorTRAIN800 semantics already exposed; current-run stage reads are separately recorded",
            "fixed_seeds":list(SEEDS),"fixed_native_splits":{s:list(ids) for s,ids in SPLITS.items()},
            "configurations":PROSPECTIVE,"final_source_hashes":{n:b["sha256"] for n,b in bound.inputs.items() if b["duty"]=="code"}})

    def _annotation_permission(self,identifier):
        require(self.busy,"no semantic read outside an actively ordered stage")
        if identifier in SPLITS["train"] and self.phase in {"fitting_ner","fitting_heads"}:
            return self.phase
        if identifier in SPLITS["dev"] and self.phase in {"selecting_ner","selecting_heads"}:
            # A changed phase string alone cannot expose Gold. Exact artifact
            # inventories must exist, and the stage function rehashed all those
            # populations before the first semantic call.
            role="dev_ner_probabilities" if self.phase=="selecting_ner" else "dev_pair_probabilities"
            kinds=("span",) if self.phase=="selecting_ner" else ARCHITECTURES
            expected={artifact_name(role,k,s,e,i)+".json" for k in kinds for s in SEEDS for e in range(1,11) for i in SPLITS["dev"]}
            require({n for n in self.store.bindings if n.startswith(role+"/")}==expected,"semantic-read prerequisite source population incomplete")
            return self.phase
        # This version implements no test semantic loader. Even a complete
        # pretest barrier does not silently unlock a scoring implementation.
        raise StageIntegrityError("annotation is not permitted in this exact stage")

    def _event(self,event,**detail):
        self.events.append({"sequence":len(self.events),"event":event,"phase":self.phase,**detail})

    def _stage(self,expected,running,finished,function):
        require(not self.busy and self.phase==expected,"serial exact stage order; no skipped/resumed/reentrant stage")
        self.busy=True;self.phase=running;self._event("stage_started")
        try:
            result=function()
            self.io.validate_all();self.store.validate()
            if self.science is not None:self.science.validate()
            self.backend.validate()
            self.phase=finished;self._event("stage_finished")
            self.store.json(f"stage_receipts/{finished}.json",{"phase":finished,"events":copy.deepcopy(self.events),
                "annotation_first_semantic_reads_this_run":copy.deepcopy(self.io.annotation_first_reads),
                "current_artifacts":copy.deepcopy(self.store.bindings),"real_scientific_completion_claim":False})
            return result
        except BaseException as error:
            self._event("stage_explicit_failure",failure_type=type(error).__name__,failure=str(error))
            self.phase="failed"
            # Existing artifacts are never renamed into successful empty
            # outputs. A single immutable failure record preserves raw history.
            try:
                self.store.json("failure.json",{"phase":"failed","failure_type":type(error).__name__,"failure":str(error),
                    "events":self.events,"partial_artifacts":self.store.bindings,
                    "annotation_first_semantic_reads_this_run":self.io.annotation_first_reads,
                    "baseline_or_primary_family_success":False})
            except BaseException:
                pass
            raise
        finally:
            self.busy=False

    def _require_population(self,role,kinds,split=None,epochs=True,suffix=".json"):
        expected={artifact_name(role,k,s,e,i)+suffix for k in kinds for s in SEEDS
                  for e in (range(1,11) if epochs else (None,))
                  for i in (SPLITS[split] if split else (None,))}
        actual={name for name in self.store.bindings if name.startswith(role+"/")}
        require(actual==expected,f"incomplete/extra population at {role}")
        for name in sorted(expected):self.store.read(name)
        return expected

    def _fit(self,kinds):
        for kind in kinds:
            for seed in SEEDS:
                # Every fit starts independently from original cold, never a
                # selected detector/head CP or the preceding fit's model state.
                handle,initial=self.backend.fit_start(kind,seed)
                self.store.json(artifact_name("fit_initial",kind,seed)+".json",initial)
                self._event("fit_cold_started",kind=kind,seed=seed)
                try:
                    for epoch in range(1,11):
                        docs=[];updates=[]
                        journal=self.store.journal(artifact_name("fit_trajectory",kind,seed,epoch)+".jsonl")
                        def log(entry):
                            require(type(entry) is dict and entry.get("kind")==kind and entry.get("seed")==seed
                                    and entry.get("epoch")==epoch,"training trajectory identity")
                            e=copy.deepcopy(entry)
                            if entry.get("event")=="train_document_backward_completed":
                                identifier=entry.get("source_id")
                                require(type(identifier) is int and identifier in SPLITS["train"],"trajectoryTRAIN800 only")
                                e["input_hashes"]={d:self.io.inputs[path]["sha256"] for d,path in (
                                    ("text",f"data/sccomics_round4/source_v3/raw_text/{identifier:04}.txt"),
                                    ("annotation",f"data/sccomics_round4/source_v3/raw_annotations/{identifier:04}.ann"),
                                    ("cache",f"results/local_baseline/sccomics_native_v1/source_cache/train/{identifier:04}.pt"))}
                                docs.append(identifier)
                            require(entry.get("event") in {"train_document_backward_completed","optimizer_update_completed"},"unrecognized training trajectory event")
                            if entry["event"]=="optimizer_update_completed":updates.append(e)
                            journal.append(e)
                        try:
                            weights,metadata=self.backend.fit_epoch(handle,kind,seed,epoch,log)
                        except BaseException as error:
                            journal.close()
                            try:
                                state,state_metadata=self.backend.failure_state(handle)
                                self.store.write(artifact_name("failed_fit_states",kind,seed,epoch)+".pt",state)
                                self.store.json(artifact_name("failed_fit_states",kind,seed,epoch)+".json",dict(state_metadata,
                                    failure_type=type(error).__name__,failure=str(error),successful_checkpoint=False))
                            except BaseException as state_error:
                                self.store.json(artifact_name("failed_fit_states",kind,seed,epoch)+".state_capture_failure.json",{
                                    "failure_type":type(state_error).__name__,"failure":str(state_error),"successful_checkpoint":False})
                            raise
                        finally:
                            trajectory=journal.close()
                        require(len(docs)==800 and set(docs)==set(SPLITS["train"]),"all800 uniqueTRAIN rows per epoch")
                        require(len(updates)==100 and [e["update"] for e in updates]==list(range((epoch-1)*100+1,epoch*100+1)),"complete100 updates each epoch")
                        require(all(len(e["source_ids"])==8 and len(set(e["source_ids"]))==8 for e in updates)
                                and [i for e in updates for i in e["source_ids"]]==docs,"eight-document accumulation trajectory differs")
                        require(type(weights) is bytes and weights and metadata.get("complete_model_state_dict") is True
                                and metadata.get("optimizer_moments_in_epoch_checkpoint") is False,"complete epoch weights-only state required")
                        cp=self.store.write(artifact_name("checkpoints",kind,seed,epoch)+".pt",weights)
                        self.store.json(artifact_name("checkpoint_metadata",kind,seed,epoch)+".json",{
                            "kind":kind,"seed":seed,"epoch":epoch,"authority_sha256":self.io.scope.authority_sha256,
                            "checkpoint":cp,"trajectory":trajectory,"metadata":metadata,"implicit_resume":False})
                        self._event("epoch_checkpoint_completed",kind=kind,seed=seed,epoch=epoch,checkpoint_sha256=cp["sha256"])
                except BaseException as error:
                    failed_name=artifact_name("failed_fit_states",kind,seed,epoch)+".pt"
                    if failed_name not in self.store.bindings:
                        try:
                            state,state_metadata=self.backend.failure_state(handle)
                            self.store.write(failed_name,state)
                            self.store.json(artifact_name("failed_fit_states",kind,seed,epoch)+".post_validation.json",dict(state_metadata,
                                failure_type=type(error).__name__,failure=str(error),successful_checkpoint=False))
                        except BaseException as state_error:
                            self._event("failed_state_capture_explicit_failure",failure_type=type(state_error).__name__,failure=str(state_error))
                    raise
                finally:
                    self.backend.release(handle)

    def fit_ner(self):
        def run():
            self._fit(("span",))
            self._require_population("checkpoints",("span",),suffix=".pt")
            return {"complete_NER_epoch_checkpoints":30}
        return self._stage("created","fitting_ner","ner_fitted",run)

    def _source_record(self,obj,kind,seed,epoch,identifier):
        require(type(obj) is dict and obj.get("source_only") is True and obj.get("Gold_consulted") is False and
                type(obj.get("source_id")) is int and obj.get("source_id")==identifier and obj.get("candidate_cap") is None and obj.get("inference_row_block_size")==512,
                "complete source-only probability identity required")
        expected=self.io.inputs[f"data/sccomics_round4/source_v3/raw_text/{identifier:04}.txt"]["sha256"]
        require(obj.get("source_sha256")==expected,"source probability/text SHA differs")
        rows=obj.get("probabilities");columns=9 if kind=="span" else 5
        require(type(rows) is list and type(obj.get("columns")) is int and obj.get("columns")==columns and
                type(obj.get("complete_rows")) is int and obj.get("complete_rows")==len(rows),"probability row/column population")
        require(all(type(r) is list and len(r)==columns and all(type(x) in (int,float) and math.isfinite(x) and 0<=x<=1 for x in r) for r in rows),"nonfinite/invalid complete probabilities")
        if kind=="span":
            require(type(obj.get("candidates")) is list and len(obj["candidates"])==len(rows),"all span candidate rows")
        else:
            require(type(obj.get("mentions")) is list and len(rows)==len(obj["mentions"])**2,"all directedN² rows including diagonal")
        return dict(obj,kind=kind,seed=seed,epoch=epoch,
                    checkpoint=self.store.bindings[artifact_name("checkpoints",kind,seed,epoch)+".pt"])

    def _generate_development(self,kinds):
        role="dev_ner_probabilities" if kinds==("span",) else "dev_pair_probabilities"
        for kind in kinds:
            for seed in SEEDS:
                for epoch in range(1,11):
                    cp=self.store.read(artifact_name("checkpoints",kind,seed,epoch)+".pt")
                    handle=self.backend.inference_start(kind,seed,cp)
                    try:
                        for identifier in SPLITS["dev"]:
                            mentions=None if kind=="span" else self.store.object(artifact_name("locked_dev_mentions","span",seed,None,identifier)+".json")["mentions"]
                            obj=self.backend.source_probabilities(handle,kind,seed,identifier,mentions)
                            self.store.json(artifact_name(role,kind,seed,epoch,identifier)+".json",self._source_record(obj,kind,seed,epoch,identifier))
                    finally:
                        self.backend.release(handle)
        self._require_population(role,kinds,"dev")

    def generate_ner_development(self):
        return self._stage("ner_fitted","generating_ner_dev_sources","ner_dev_sources_complete",
                           lambda:self._generate_development(("span",)))

    def select_ner(self):
        def run():
            self._require_population("dev_ner_probabilities",("span",),"dev")
            # This is the first permitted new-run dev annotation semantics.
            gold={i:self.backend.gold(i) for i in SPLITS["dev"]}
            known=set(self.backend.models.TEXT_TYPES)
            self.store.json("choices/dev_complete_native_gold_inventory.json",{
                str(i):{"native_T_occurrences":len(gold[i].family("T")),
                    "unknown_T_label_counts":{label:sum(r.label==label for r in gold[i].family("T"))
                        for label in sorted({r.label for r in gold[i].family("T") if r.label not in known})},
                    "native_R_occurrences":len(gold[i].family("R")),"native_E_occurrences":len(gold[i].family("E")),
                    "native_AUX_occurrences":len(gold[i].family("AUX")),
                    "raw_annotation_sha256":self.io.inputs[f"data/sccomics_round4/source_v3/raw_annotations/{i:04}.ann"]["sha256"],
                    "unknown_valid_T_retained_in_denominator":True} for i in SPLITS["dev"]})
            grids=[]
            for epoch in range(1,11):
                for threshold in SPAN_THRESHOLDS:
                    by_seed={}
                    for seed in SEEDS:
                        rows={}
                        for identifier in SPLITS["dev"]:
                            obj=self.store.object(artifact_name("dev_ner_probabilities","span",seed,epoch,identifier)+".json")
                            mentions=self.backend.detector_mentions(obj,threshold)
                            _,pred=self.backend.detector_document(identifier,mentions)
                            rows[str(identifier)]={"T_complete_native":self.backend.selector.text_counts(pred,gold[identifier])}
                        by_seed[str(seed)]=rows
                    grids.append({"epoch":epoch,"thresholds":[threshold],"by_seed_document":by_seed})
            choice=self.backend.selector.select_detector(grids)
            require(len(choice["all_pooled_candidates"])==80,"complete80 detector grid required")
            self.store.json("choices/ner_complete80_grid.json",grids)
            self.store.json("choices/ner.json",choice);self.choices["span"]=choice["chosen"]
            epoch=choice["chosen"]["epoch"];threshold=choice["chosen"]["threshold"]
            for seed in SEEDS:
                for identifier in SPLITS["dev"]:
                    name=artifact_name("dev_ner_probabilities","span",seed,epoch,identifier)+".json"
                    obj=self.store.object(name);mentions=self.backend.detector_mentions(obj,threshold)
                    records,_=self.backend.detector_document(identifier,mentions)
                    self.store.json(artifact_name("locked_dev_mentions","span",seed,None,identifier)+".json",{
                        "seed":seed,"source_id":identifier,"source_sha256":obj["source_sha256"],"mentions":mentions,
                        "original_source_records":records,"detector_epoch":epoch,"detector_threshold":threshold,
                        "source_probabilities":self.store.bindings[name],"Gold_consulted_for_emission":False})
            self._require_population("locked_dev_mentions",("span",),"dev",epochs=False)
            return choice
        return self._stage("ner_dev_sources_complete","selecting_ner","ner_selected_and_dev_locked",run)

    def fit_heads(self):
        def run():
            self._fit(ARCHITECTURES)
            self._require_population("checkpoints",KINDS,suffix=".pt")
            return {"complete_all_epoch_checkpoints":150}
        return self._stage("ner_selected_and_dev_locked","fitting_heads","all150_fitted",run)

    def generate_pair_development(self):
        return self._stage("all150_fitted","generating_pair_dev_sources","all_pair_dev_sources_complete",
                           lambda:self._generate_development(ARCHITECTURES))

    def select_heads(self):
        def run():
            self._require_population("dev_pair_probabilities",ARCHITECTURES,"dev")
            gold={i:self.backend.gold(i) for i in SPLITS["dev"]}
            # Exact completed T/E fingerprint cache is the reviewed selector's
            # mechanism; no approximate/partial or Gold-type-pruned shortcut.
            event_cache={};all_choices={}
            for kind in ARCHITECTURES:
                grids=[]
                for epoch in range(1,11):
                    for rt in PAIR_THRESHOLDS:
                        for et in PAIR_THRESHOLDS:
                            by_seed={}
                            for seed in SEEDS:
                                rows={}
                                for identifier in SPLITS["dev"]:
                                    obj=self.store.object(artifact_name("dev_pair_probabilities",kind,seed,epoch,identifier)+".json")
                                    _,pred=self.backend.emitted(identifier,obj["mentions"],obj["probabilities"],rt,et)
                                    scored=self.backend.selector.primary_counts(pred,gold[identifier],event_cache=event_cache)
                                    rows[str(identifier)]={e:scored[e] for e in self.backend.selector.ENDPOINTS}
                                by_seed[str(seed)]=rows
                            grids.append({"epoch":epoch,"thresholds":[rt,et],"by_seed_document":by_seed})
                choice=self.backend.selector.select_pair_head(grids)
                require(len(choice["all_pooled_candidates"])==250,"complete250 settings for every architecture")
                self.store.json(f"choices/{kind}_complete250_grid.json",grids)
                self.store.json(f"choices/{kind}.json",choice)
                self.choices[kind]=choice["chosen"];all_choices[kind]=choice
            require(set(self.choices)==set(KINDS),"complete detector/four-head choices")
            return all_choices
        return self._stage("all_pair_dev_sources_complete","selecting_heads","all_choices_complete",run)

    def freeze_pretest(self):
        def run():
            self._require_population("checkpoints",KINDS,suffix=".pt")
            self._require_population("dev_ner_probabilities",("span",),"dev")
            self._require_population("dev_pair_probabilities",ARCHITECTURES,"dev")
            self._require_population("locked_dev_mentions",("span",),"dev",epochs=False)
            require(set(self.choices)==set(KINDS),"all final choices before test sources")
            checked=self.io.validate_all();artifacts=self.store.validate();self.backend.validate()
            if self.science:self.science.validate()
            return self.store.json("pretest_scientific_source_freeze.json",{
                "status":"all150_checkpoints_choices_development_and_sources_frozen_before_test_generation",
                "authority_sha256":self.io.scope.authority_sha256,"input_bindings":self.io.inputs,
                "actual_rehash_boundary":checked,"immutable_artifacts":artifacts,"all_common_choices":self.choices,
                "test_annotation_semantics_read_this_run":False,"scientific_completion_claim":False})
        return self._stage("all_choices_complete","freezing_pretest_sources","pretest_frozen",run)

    def generate_test_sources(self):
        def run():
            frozen=self.store.object("pretest_scientific_source_freeze.json")
            require(frozen["all_common_choices"]==self.choices,"frozen choices changed")
            for name,b in frozen["immutable_artifacts"].items():
                require(self.store.bindings.get(name)==b,"frozen artifact identity changed");self.store.read(name)
            detector=self.choices["span"];epoch=detector["epoch"];threshold=detector["threshold"]
            for seed in SEEDS:
                handle=self.backend.inference_start("span",seed,self.store.read(artifact_name("checkpoints","span",seed,epoch)+".pt"))
                try:
                    for identifier in SPLITS["test"]:
                        obj=self._source_record(self.backend.source_probabilities(handle,"span",seed,identifier),"span",seed,epoch,identifier)
                        mentions=self.backend.detector_mentions(obj,threshold);records,_=self.backend.detector_document(identifier,mentions)
                        self.store.json(artifact_name("test_source_graphs","span",seed,None,identifier)+".json",{
                            "source_probability_record":obj,"mentions":mentions,"original_source_records":records,
                            "detector_threshold":threshold,"test_Gold_consulted":False})
                finally:self.backend.release(handle)
            for kind in ARCHITECTURES:
                choice=self.choices[kind];epoch=choice["epoch"]
                for seed in SEEDS:
                    handle=self.backend.inference_start(kind,seed,self.store.read(artifact_name("checkpoints",kind,seed,epoch)+".pt"))
                    try:
                        for identifier in SPLITS["test"]:
                            ner=self.store.object(artifact_name("test_source_graphs","span",seed,None,identifier)+".json")
                            obj=self._source_record(self.backend.source_probabilities(handle,kind,seed,identifier,ner["mentions"]),kind,seed,epoch,identifier)
                            records,_=self.backend.emitted(identifier,ner["mentions"],obj["probabilities"],choice["relation_threshold"],choice["role_threshold"])
                            self.store.json(artifact_name("test_source_graphs",kind,seed,None,identifier)+".json",{
                                "source_probability_record":obj,"original_source_records":records,"test_Gold_consulted":False,
                                "emission_diagnostics":copy.deepcopy(self.backend.last_emission_diagnostics),
                                "same_seed_detector_source":self.store.bindings[artifact_name("test_source_graphs","span",seed,None,identifier)+".json"],
                                "relation_threshold":choice["relation_threshold"],"role_threshold":choice["role_threshold"]})
                    finally:self.backend.release(handle)
            self._require_population("test_source_graphs",KINDS,"test",epochs=False)
            require(not(set(SPLITS["test"])&self.io.annotation_ids_read),"test annotation permission violated")
            return {"complete15x100_original_source_JSON_with_full_probabilities":1500}
        return self._stage("pretest_frozen","generating_test_sources","all_test_sources_complete",run)

    def test_barrier(self):
        def run():
            self._require_population("checkpoints",KINDS,suffix=".pt")
            self._require_population("test_source_graphs",KINDS,"test",epochs=False)
            freeze=self.store.object("pretest_scientific_source_freeze.json")
            require(freeze["all_common_choices"]==self.choices and set(self.choices)==set(KINDS),"full pretest choices binding")
            require(not(set(SPLITS["test"])&self.io.annotation_ids_read),"no test semantics before complete source barrier")
            boundary=self.io.validate_all();artifacts=self.store.validate();self.backend.validate()
            if self.science:self.science.validate()
            return self.store.json("test_source_barrier.json",{"status":"complete_test_source_barrier_passed_only",
                "complete150_epoch_checkpoints":150,"complete_test_source_objects":1500,"all_choices":self.choices,
                "actual_input_rehash_boundary":boundary,"immutable_artifacts":artifacts,
                "annotation_first_semantic_reads_this_run":self.io.annotation_first_reads,
                "test_annotation_semantic_permission_enabled":False,"final_scoring_implemented_here":False,
                "actual_model_performance_or_science_completion_claim":False})
        return self._stage("all_test_sources_complete","verifying_complete_test_source_barrier","pretest_source_barrier_passed_only",run)

    def score_and_statistics(self):
        # Never parse any test annotation to discover an unimplemented stage.
        raise UnimplementedStageError("This source version ends at the complete pretest source barrier; separately reviewed/frozen Gold scoring and statistical execution are not implemented here")

    def run_pretest(self):
        for function in (self.fit_ner,self.generate_ner_development,self.select_ner,self.fit_heads,
                         self.generate_pair_development,self.select_heads,self.freeze_pretest,
                         self.generate_test_sources,self.test_barrier):
            function()
        return self.store.bindings["test_source_barrier.json"]


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root",required=True)
    parser.add_argument("--freeze-sha256",required=True)
    parser.add_argument("--final-gate-sha256",required=True)
    args=parser.parse_args(argv)
    # Missing/current prospective authority fails before any model/cache or
    # annotation-byte loader and before any output directory is created.
    scope,bound=authorize_actual(args.root,freeze_sha256=args.freeze_sha256,gate_sha256=args.final_gate_sha256)
    with tempfile.TemporaryDirectory(prefix="sccomics_private_actual_") as directory:
        science=CapturedScience(bound,Path(directory)/"source")
        try:
            # Execute the verified runner definitions too, bypassing a possibly
            # stale imported runner .pyc or a post-capture live-file replacement.
            runner=science.get("sccomics_supervised_stages")
            # Scope registrations belong to the captured module, not the live
            # bootstrap. Transfer only this already validated actual authority.
            runner._SCOPES.add(scope)
            backend=runner.TorchBackend(bound,science,directory)
            controller=runner.StageController(bound,backend,science=science)
            result=controller.run_pretest()
            print(json.dumps({"status":"pretest_source_barrier_only","barrier":result},sort_keys=True),flush=True)
        finally:
            science.close();_SCOPES.discard(scope)
    return 0


if __name__=="__main__":
    raise SystemExit(main())
