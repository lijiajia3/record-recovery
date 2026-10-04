"""Finite installed-runtime byte identity; no task/model/corpus file loader.

Population policy was declared in the v2 pre-fit SOURCE contract. This module
never selects weights, annotations, archives, credentials or data/cache suffixes.
The runtime receipt is not actual fitting authority. OS/GPU libraries are outside
its byte population. Python new-file imports compile captured checked source;
native loading has finite pre/post checks, not a hostile-ABA transaction proof.
"""
from __future__ import annotations
import copy
import importlib.abc
import importlib.machinery
import importlib.metadata
from pathlib import Path, PurePosixPath
import platform
import sys
import sysconfig
import sccomics_supervised_stages_v2 as stage

SEEDS=("torch","numpy","tokenizers","transformers","networkx","safetensors")
OPTIONAL=("scipy","scikit-learn","pillow","torchvision","librosa","torchaudio","accelerate","optree","setuptools")
METADATA=frozenset({"METADATA","WHEEL","RECORD","top_level.txt"})
POLICY="SC_v2_finite_framework_stdlib_selected_distribution_code_native_licenses_metadata_v1"
LICENSE_SUFFIXES=frozenset({"",".txt",".md",".rst",".html",".rtf"})
FRAMEWORK_METADATA_ALIAS="lib/python3.13/config-3.13-darwin/libpython3.13.dylib"


def license_name(path,*,directory=False):
    """Plain-document license roles cannot grant data/secret/weight access."""
    p=Path(path)
    return (p.suffix.lower() in LICENSE_SUFFIXES and
            (directory or p.name.lower().startswith(("license","notice","copying"))))


def claim_role(files,name,role):
    previous=files.get(name)
    if {previous,role}=={"distribution_metadata_license","package_license"}:
        # A vendored dist-info license can occur in both the distribution
        # RECORD and the selected enclosing package. It is the same file,
        # with the more specific closed metadata/license role, not extra I/O.
        files[name]="distribution_metadata_license"
    else:
        stage.require(previous is None or previous==role,"runtime role collision: "+name)
        files[name]=role


def framework_alias_metadata(framework,path):
    """One known installed-framework alias; never a permitted import/read path.

    Its canonical Python binary is separately byte-bound. Other aliases,
    including every caller/task/model path, still fail the nonsymlink guard.
    """
    p=Path(path).absolute();framework=Path(framework)
    if str(p.relative_to(framework))!=FRAMEWORK_METADATA_ALIAS:return None
    stage.require(p.is_symlink() and p.resolve()==framework/"Python" and
                  not any(q.is_symlink() for q in ((framework/"Python"),*(framework/"Python").parents)) and
                  (framework/"Python").is_file(),"fixed framework alias/regular canonical target differs")
    return {"metadata_path":FRAMEWORK_METADATA_ALIAS,"canonical_target":"Python",
            "alias_content_not_opened_or_hashed":True,"canonical_target_requires_separate_byte_binding":True,
            "import_or_task_read_via_alias_permitted":False}


def population():
    """Metadata-only discovery, no scientific imports or code/binary contents."""
    from packaging.requirements import Requirement
    framework=Path(sys.base_prefix).resolve();stdlib=Path(sysconfig.get_path("stdlib")).resolve()
    site=Path(sysconfig.get_path("purelib")).resolve()
    stage.require(framework.is_absolute() and not framework.is_relative_to(Path.cwd()) and
                  stdlib.is_relative_to(framework) and site.is_relative_to(stdlib),"installed canonical Python framework outside project")
    pending=list(SEEDS);absent=[]
    for name in OPTIONAL:
        try:importlib.metadata.distribution(name)
        except importlib.metadata.PackageNotFoundError:absent.append(name)
        else:pending.append(name)
    distributions={};roots=set();files={};aliases=[]
    def add(p,role):
        p=Path(p).absolute();stage.require(p.is_relative_to(framework),"runtime path outside framework")
        name=str(p.relative_to(framework));stage.relative(name)
        stage.require(not any(x.is_symlink() for x in (p,*p.parents)) and p.is_file(),"runtime regular nonsymlink path")
        claim_role(files,name,role)
    while pending:
        name=pending.pop().lower().replace("_","-").replace(".","-")
        if name in distributions:continue
        d=importlib.metadata.distribution(name);record=tuple(d.files or ())
        stage.require(record,"distribution file metadata missing")
        distributions[name]={"version":d.version,"metadata_directory":str(Path(d._path).absolute().relative_to(framework))}
        for raw in d.requires or ():
            req=Requirement(raw)
            if req.marker is None or req.marker.evaluate({"extra":""}):pending.append(req.name)
        for item in record:
            parts=PurePosixPath(str(item)).parts
            if not parts or any(p in {"..","."} for p in parts):continue
            p=Path(d.locate_file(item)).absolute()
            if not p.is_relative_to(site):continue
            if any(part.endswith(".dist-info") for part in parts):
                if p.name in METADATA or license_name(p,directory="licenses" in parts):
                    add(p,"distribution_metadata_license")
            elif p.suffix in {".py",".so",".dylib"} and "__pycache__" not in p.parts:
                roots.add(str((site/parts[0]).relative_to(framework)))
    for root in sorted(roots):
        p=framework/root
        for child in (p.rglob("*") if p.is_dir() else (p,)):
            if not child.is_file() or "__pycache__" in child.parts:continue
            if child.suffix in {".py",".so",".dylib"}:add(child,"package_code_native")
            elif license_name(child):add(child,"package_license")
    for p in stdlib.rglob("*"):
        if p.is_relative_to(site) or "__pycache__" in p.parts:continue
        if p.is_file() and p.suffix in {".py",".so",".dylib"}:
            alias=framework_alias_metadata(framework,p)
            if alias is not None:aliases.append(alias)
            else:add(p,"stdlib_code_native")
    executable=Path(sys.executable).resolve();add(executable,"Python_executable")
    if (framework/"Python").is_file():add(framework/"Python","Python_framework_binary")
    for name in ("LICENSE.txt","LICENSE","Resources/English.lproj/Documentation/LICENSE.txt"):
        if (framework/name).is_file():add(framework/name,"Python_license")
    return {"policy":POLICY,"framework_root":str(framework),"stdlib_relative_path":str(stdlib.relative_to(framework)),
        "site_relative_path":str(site.relative_to(framework)),"executable_relative_path":str(executable.relative_to(framework)),
        "invocation_executable_alias":sys.executable,"package_roots":sorted(roots),"selected_distributions":dict(sorted(distributions.items())),
        "absent_optional_distributions":sorted(absent),"file_roles":dict(sorted(files.items())),
        "fixed_framework_alias_metadata_no_read_permission":aliases,"all_OS_GPU_or_system_dynamic_libraries_byte_bound":False}


def prepare_receipt():
    """Source-only streamed code/native/metadata hashes, no model/task I/O."""
    selected=population();root=stage.root_path(selected["framework_root"])
    stage.require(platform.python_version()==stage.RUNTIME["python"] and
                  all(selected["selected_distributions"].get(name,{}).get("version")==version for name,version in stage.RUNTIME.items() if name!="python"),
                  "installed metadata/runtime versions differ before any runtime population hash")
    bindings={n:dict(stage.digest_file(root,n),duty="runtime",runtime_role=r) for n,r in selected["file_roles"].items()}
    for n,b in bindings.items():stage.require(stage.digest_file(root,n)=={k:b[k] for k in ("sha256","size_bytes")},"runtime changed during preparation")
    return {"status":"finite_runtime_code_only_source_receipt_not_fit_authority","runtime":stage.RUNTIME,"population":selected,
        "file_bindings":bindings,"source_contract_json_sha256":stage.V2_CONTRACT_SHA,"actual_model_weight_cache_or_SC_annotation_read":False,
        "all_declared_files_start_and_final_hash_checked":True,"OS_GPU_byte_identity_claim":False,"platform_version_only":platform.platform()}


class PythonLoader(importlib.abc.Loader):
    def __init__(self,guard,filename):self.guard=guard;self.filename=filename;self.path=filename
    def create_module(self,spec):return None
    def get_filename(self,name):return self.filename
    def is_package(self,name):return Path(self.filename).name=="__init__.py"
    def get_code(self,name):return compile(self.guard.capture(self.filename),self.filename,"exec",dont_inherit=True)
    def get_source(self,name):return self.guard.capture(self.filename).decode("utf-8","strict")
    def get_resource_reader(self,name):
        # Normal package resources are not silently added to the finite code/
        # native byte proof. This preserves library APIs without claiming that
        # all resources or external OS certificate files are byte verified.
        from importlib.readers import FileReader
        return FileReader(self)
    def exec_module(self,module):
        exec(self.get_code(module.__name__),module.__dict__)
        self.guard.events.append({"module":module.__name__,"origin":self.filename,"loader":"captured_Python_source_no_pyc"})


class NativeLoader(importlib.abc.Loader):
    def __init__(self,guard,original,filename):self.guard=guard;self.original=original;self.filename=filename
    def create_module(self,spec):
        self.guard.check_file(self.filename);result=self.original.create_module(spec);self.guard.check_file(self.filename);return result
    def exec_module(self,module):
        self.guard.check_file(self.filename);self.original.exec_module(module);self.guard.check_file(self.filename)
        self.guard.events.append({"module":module.__name__,"origin":self.filename,"loader":"declared_native_pre_post_file_SHA"})


class BlockedLoader(importlib.abc.Loader):
    def __init__(self,name):self.name=name
    def create_module(self,spec):raise stage.StageIntegrityError("undeclared runtime import before loader: "+self.name)
    def exec_module(self,module):raise stage.StageIntegrityError("undeclared runtime import before loader: "+self.name)


class BoundRuntime(importlib.abc.MetaPathFinder):
    def __init__(self,receipt,scope):
        stage.require(scope in stage._SCOPES,"genuine final-authorized or newly allocated invented scope")
        stage.require(type(receipt) is dict and receipt.get("status")=="finite_runtime_code_only_source_receipt_not_fit_authority" and
                      receipt.get("runtime")==stage.RUNTIME and receipt.get("source_contract_json_sha256")==stage.V2_CONTRACT_SHA,"finite runtime receipt schema")
        self.receipt=copy.deepcopy(receipt);self.scope=scope;self.events=[];self.active=False;self.sciences=[]
        selected=receipt["population"];self.root=stage.root_path(selected["framework_root"])
        if scope.invented:stage.require(self.root==scope.root/"INVENTED_runtime","invented runtime only newly owned subtree")
        else:stage.require(self.root==Path(sys.base_prefix).resolve(),"actual runtime must be declared running Python framework")
        stage.require(selected.get("policy")==POLICY and selected.get("all_OS_GPU_or_system_dynamic_libraries_byte_bound") is False,"finite OS/GPU scope")
        self.inputs=copy.deepcopy(receipt["file_bindings"])
        stage.require(type(self.inputs) is dict and set(self.inputs)==set(selected["file_roles"]),"complete exact finite runtime file population")
        for name,b in self.inputs.items():
            stage.relative(name)
            stage.require(type(b) is dict and set(b)=={"sha256","size_bytes","duty","runtime_role"} and b["duty"]=="runtime" and
                          b["runtime_role"]==selected["file_roles"][name],"runtime binding/role")
            stage.strict_binding({k:b[k] for k in ("sha256","size_bytes","duty")});p=Path(name);r=b["runtime_role"]
            if r in {"package_code_native","stdlib_code_native"}:stage.require(p.suffix in {".py",".so",".dylib"} and "__pycache__" not in p.parts,"no weights/cache/pyc runtime code")
            elif r=="distribution_metadata_license":stage.require(p.name in METADATA or license_name(p,directory="licenses" in p.parts),"closed runtime metadata")
            elif r in {"package_license","Python_license"}:stage.require(license_name(p),"closed plain license bytes")
            elif r=="Python_executable":stage.require(name==selected["executable_relative_path"] and p.name in {"python","python3","python3.13"},"declared Python executable")
            elif r=="Python_framework_binary":stage.require(name=="Python","declared framework binary")
            else:raise stage.StageIntegrityError("undeclared runtime role")
        self.namespaces=[self.root/x for x in selected["package_roots"]]
        for x in (selected["stdlib_relative_path"],selected["site_relative_path"],selected["executable_relative_path"],*selected["package_roots"]):stage.relative(x)
        self.validate_files()
        if not scope.invented:
            # Marker/dependency metadata interpretation imports packaging. At
            # actual startup even that new import must use declared checked
            # source, before the complete metadata population is rederived.
            sys.meta_path.insert(0,self)
            try:stage.require(selected==population(),"current metadata population differs from prospective plan")
            finally:
                stage.require(sys.meta_path and sys.meta_path[0] is self,"metadata-phase import guard changed")
                sys.meta_path.remove(self)

    def _name(self,filename):
        p=Path(filename).absolute();stage.require(p.is_relative_to(self.root),"module outside finite runtime root")
        name=str(p.relative_to(self.root));stage.relative(name);stage.require(name in self.inputs,"module outside predeclared runtime files");return name
    def check_file(self,filename):
        n=self._name(filename);b=self.inputs[n]
        stage.require(stage.digest_file(self.root,n)=={k:b[k] for k in ("sha256","size_bytes")},"runtime code/native changed");return n
    def capture(self,filename):
        n=self._name(filename);raw=stage.raw_read(self.root,n);b=self.inputs[n]
        stage.require(len(raw)==b["size_bytes"] and stage.sha(raw)==b["sha256"] and Path(n).suffix==".py","captured runtime source differs");return raw
    def allow_science(self,science):self.sciences.append(science)
    def find_spec(self,fullname,path=None,target=None):
        for finder in (importlib.machinery.BuiltinImporter,importlib.machinery.FrozenImporter):
            spec=finder.find_spec(fullname,path,target)
            if spec is not None:return spec
        spec=importlib.machinery.PathFinder.find_spec(fullname,path,target)
        if spec is None:return importlib.machinery.ModuleSpec(fullname,BlockedLoader(fullname))
        if spec.origin is None:
            locations=list(spec.submodule_search_locations or ())
            stage.require(locations and all(any(Path(p).absolute()==r or Path(p).absolute().is_relative_to(r) for r in self.namespaces) for p in locations),"undeclared runtime namespace")
            self.events.append({"module":fullname,"loader":"declared_namespace","search_locations":locations});return spec
        self.check_file(spec.origin)
        if isinstance(spec.loader,importlib.machinery.SourceFileLoader):spec.loader=PythonLoader(self,spec.origin)
        elif isinstance(spec.loader,importlib.machinery.ExtensionFileLoader):spec.loader=NativeLoader(self,spec.loader,spec.origin)
        else:raise stage.StageIntegrityError("unreviewed runtime filesystem loader; no pyc/custom fallback")
        return spec
    def validate_files(self):
        for n,b in self.inputs.items():stage.require(stage.digest_file(self.root,n)=={k:b[k] for k in ("sha256","size_bytes")},"finite runtime startup/final bytes changed")
        return {"actual_file_hashes_checked":len(self.inputs),"finite_scope_policy":POLICY,"all_OS_GPU_bytes_claim":False}
    def loaded_inventory(self):
        rows=[]
        for name,module in tuple(sys.modules.items()):
            if module is None:continue
            filename=getattr(module,"__file__",None);spec=getattr(module,"__spec__",None);origin=getattr(spec,"origin",None)
            if filename and any(Path(filename)==p for s in self.sciences for p in s.files.values()):rows.append({"module":name,"kind":"separately_captured_science"});continue
            if filename and any(Path(filename)==s.bound.root/n for s in self.sciences for n in s.files):
                for s in self.sciences:
                    for n in s.files:
                        if Path(filename)==s.bound.root/n:
                            actual=stage.digest_file(s.bound.root,n)
                            stage.require(actual=={k:s.bound.inputs[n][k] for k in ("sha256","size_bytes")},"verified bootstrap/source path changed")
                rows.append({"module":name,"kind":"separately_verified_bootstrap_science_path_actual_definitions_use_private_capture"});continue
            if origin in {"built-in","frozen"} or name in sys.builtin_module_names:rows.append({"module":name,"kind":"builtin_frozen_Python_framework"});continue
            if filename:
                if self.scope.invented and not Path(filename).absolute().is_relative_to(self.root):continue
                rows.append({"module":name,"kind":"declared_file_module","relative_file":self.check_file(filename)});continue
            locations=list(getattr(module,"__path__",()) or ())
            if locations:
                if self.scope.invented and not any(Path(p).absolute().is_relative_to(self.root) for p in locations):continue
                stage.require(all(any(Path(p).absolute()==r or Path(p).absolute().is_relative_to(r) for r in self.namespaces) for p in locations),"loaded namespace outside runtime")
                rows.append({"module":name,"kind":"declared_namespace"});continue
            parents=[sys.modules.get(name[:i]) for i,c in enumerate(name) if c=='.']
            parent=next((p for p in reversed(parents) if p is not None and getattr(p,"__file__",None) and Path(p.__file__).suffix in {".so",".dylib"}),None)
            if parent is not None:rows.append({"module":name,"kind":"native_generated_child_no_independent_file","native_parent":self.check_file(parent.__file__)});continue
            if self.scope.invented:continue
            raise stage.StageIntegrityError("loaded module lacks declared byte attribution: "+name)
        return rows
    def install(self):
        stage.require(not self.active,"one runtime guard");self.validate_files();self.loaded_inventory();sys.meta_path.insert(0,self);self.active=True
    def validate(self):return {**self.validate_files(),"loaded_modules":self.loaded_inventory(),"new_import_checks":copy.deepcopy(self.events)}
    def close(self):
        if self.active:
            stage.require(sys.meta_path and sys.meta_path[0] is self,"runtime import guard changed")
            sys.meta_path.remove(self);self.active=False
