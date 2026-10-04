"""Finite installed-runtime byte identity; no task/model/corpus file loader.

Population policy and explicit106 optional closure were declared before fit in
the distinct optional-dependency SOURCE contract. Original v2/v3 remain history. This module
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
import threading
import sccomics_supervised_stages_v4 as stage

SEEDS=("torch","numpy","tokenizers","transformers","networkx","safetensors")
OPTIONAL=("scipy","scikit-learn","pillow","torchvision","librosa","torchaudio","accelerate","optree","setuptools")
METADATA=frozenset({"METADATA","WHEEL","RECORD","top_level.txt"})
POLICY="SC_v4_finite_explicit_106_installed_soft_import_distribution_code_native_licenses_metadata_v1"
DECLARED_DISTRIBUTIONS={'aiohappyeyeballs': {'metadata_directory': 'lib/python3.13/site-packages/aiohappyeyeballs-2.6.1.dist-info', 'version': '2.6.1'}, 'aiohttp': {'metadata_directory': 'lib/python3.13/site-packages/aiohttp-3.13.2.dist-info', 'version': '3.13.2'}, 'aiosignal': {'metadata_directory': 'lib/python3.13/site-packages/aiosignal-1.4.0.dist-info', 'version': '1.4.0'}, 'annotated-doc': {'metadata_directory': 'lib/python3.13/site-packages/annotated_doc-0.0.4.dist-info', 'version': '0.0.4'}, 'annotated-types': {'metadata_directory': 'lib/python3.13/site-packages/annotated_types-0.7.0.dist-info', 'version': '0.7.0'}, 'anyio': {'metadata_directory': 'lib/python3.13/site-packages/anyio-4.11.0.dist-info', 'version': '4.11.0'}, 'asttokens': {'metadata_directory': 'lib/python3.13/site-packages/asttokens-3.0.0.dist-info', 'version': '3.0.0'}, 'attrs': {'metadata_directory': 'lib/python3.13/site-packages/attrs-25.4.0.dist-info', 'version': '25.4.0'}, 'beautifulsoup4': {'metadata_directory': 'lib/python3.13/site-packages/beautifulsoup4-4.14.2.dist-info', 'version': '4.14.2'}, 'certifi': {'metadata_directory': 'lib/python3.13/site-packages/certifi-2025.8.3.dist-info', 'version': '2025.8.3'}, 'cffi': {'metadata_directory': 'lib/python3.13/site-packages/cffi-2.0.0.dist-info', 'version': '2.0.0'}, 'charset-normalizer': {'metadata_directory': 'lib/python3.13/site-packages/charset_normalizer-3.4.4.dist-info', 'version': '3.4.4'}, 'click': {'metadata_directory': 'lib/python3.13/site-packages/click-8.3.1.dist-info', 'version': '8.3.1'}, 'contourpy': {'metadata_directory': 'lib/python3.13/site-packages/contourpy-1.3.3.dist-info', 'version': '1.3.3'}, 'cycler': {'metadata_directory': 'lib/python3.13/site-packages/cycler-0.12.1.dist-info', 'version': '0.12.1'}, 'cython': {'metadata_directory': 'lib/python3.13/site-packages/cython-3.2.4.dist-info', 'version': '3.2.4'}, 'datasets': {'metadata_directory': 'lib/python3.13/site-packages/datasets-4.8.5.dist-info', 'version': '4.8.5'}, 'decorator': {'metadata_directory': 'lib/python3.13/site-packages/decorator-5.2.1.dist-info', 'version': '5.2.1'}, 'dill': {'metadata_directory': 'lib/python3.13/site-packages/dill-0.4.1.dist-info', 'version': '0.4.1'}, 'distro': {'metadata_directory': 'lib/python3.13/site-packages/distro-1.9.0.dist-info', 'version': '1.9.0'}, 'executing': {'metadata_directory': 'lib/python3.13/site-packages/executing-2.2.1.dist-info', 'version': '2.2.1'}, 'fastapi': {'metadata_directory': 'lib/python3.13/site-packages/fastapi-0.128.0.dist-info', 'version': '0.128.0'}, 'filelock': {'metadata_directory': 'lib/python3.13/site-packages/filelock-3.20.3.dist-info', 'version': '3.20.3'}, 'flatbuffers': {'metadata_directory': 'lib/python3.13/site-packages/flatbuffers-25.12.19.dist-info', 'version': '25.12.19'}, 'fonttools': {'metadata_directory': 'lib/python3.13/site-packages/fonttools-4.59.2.dist-info', 'version': '4.59.2'}, 'frozenlist': {'metadata_directory': 'lib/python3.13/site-packages/frozenlist-1.8.0.dist-info', 'version': '1.8.0'}, 'fsspec': {'metadata_directory': 'lib/python3.13/site-packages/fsspec-2025.12.0.dist-info', 'version': '2025.12.0'}, 'ftfy': {'metadata_directory': 'lib/python3.13/site-packages/ftfy-6.3.1.dist-info', 'version': '6.3.1'}, 'h11': {'metadata_directory': 'lib/python3.13/site-packages/h11-0.16.0.dist-info', 'version': '0.16.0'}, 'hf-xet': {'metadata_directory': 'lib/python3.13/site-packages/hf_xet-1.2.0.dist-info', 'version': '1.2.0'}, 'httpcore': {'metadata_directory': 'lib/python3.13/site-packages/httpcore-1.0.9.dist-info', 'version': '1.0.9'}, 'httpx': {'metadata_directory': 'lib/python3.13/site-packages/httpx-0.28.1.dist-info', 'version': '0.28.1'}, 'huggingface-hub': {'metadata_directory': 'lib/python3.13/site-packages/huggingface_hub-0.36.0.dist-info', 'version': '0.36.0'}, 'idna': {'metadata_directory': 'lib/python3.13/site-packages/idna-3.11.dist-info', 'version': '3.11'}, 'ipython': {'metadata_directory': 'lib/python3.13/site-packages/ipython-9.6.0.dist-info', 'version': '9.6.0'}, 'ipython-pygments-lexers': {'metadata_directory': 'lib/python3.13/site-packages/ipython_pygments_lexers-1.1.1.dist-info', 'version': '1.1.1'}, 'jedi': {'metadata_directory': 'lib/python3.13/site-packages/jedi-0.19.2.dist-info', 'version': '0.19.2'}, 'jinja2': {'metadata_directory': 'lib/python3.13/site-packages/jinja2-3.1.6.dist-info', 'version': '3.1.6'}, 'jiter': {'metadata_directory': 'lib/python3.13/site-packages/jiter-0.14.0.dist-info', 'version': '0.14.0'}, 'joblib': {'metadata_directory': 'lib/python3.13/site-packages/joblib-1.5.3.dist-info', 'version': '1.5.3'}, 'kiwisolver': {'metadata_directory': 'lib/python3.13/site-packages/kiwisolver-1.4.9.dist-info', 'version': '1.4.9'}, 'lxml': {'metadata_directory': 'lib/python3.13/site-packages/lxml-6.1.0.dist-info', 'version': '6.1.0'}, 'markdown-it-py': {'metadata_directory': 'lib/python3.13/site-packages/markdown_it_py-4.0.0.dist-info', 'version': '4.0.0'}, 'markupsafe': {'metadata_directory': 'lib/python3.13/site-packages/markupsafe-3.0.3.dist-info', 'version': '3.0.3'}, 'matplotlib': {'metadata_directory': 'lib/python3.13/site-packages/matplotlib-3.10.6.dist-info', 'version': '3.10.6'}, 'matplotlib-inline': {'metadata_directory': 'lib/python3.13/site-packages/matplotlib_inline-0.2.1.dist-info', 'version': '0.2.1'}, 'mdurl': {'metadata_directory': 'lib/python3.13/site-packages/mdurl-0.1.2.dist-info', 'version': '0.1.2'}, 'mpmath': {'metadata_directory': 'lib/python3.13/site-packages/mpmath-1.3.0.dist-info', 'version': '1.3.0'}, 'multidict': {'metadata_directory': 'lib/python3.13/site-packages/multidict-6.7.0.dist-info', 'version': '6.7.0'}, 'multiprocess': {'metadata_directory': 'lib/python3.13/site-packages/multiprocess-0.70.19.dist-info', 'version': '0.70.19'}, 'networkx': {'metadata_directory': 'lib/python3.13/site-packages/networkx-3.6.1.dist-info', 'version': '3.6.1'}, 'numpy': {'metadata_directory': 'lib/python3.13/site-packages/numpy-2.2.6.dist-info', 'version': '2.2.6'}, 'onnxruntime': {'metadata_directory': 'lib/python3.13/site-packages/onnxruntime-1.28.0.dist-info', 'version': '1.28.0'}, 'openai': {'metadata_directory': 'lib/python3.13/site-packages/openai-2.42.0.dist-info', 'version': '2.42.0'}, 'opencv-contrib-python': {'metadata_directory': 'lib/python3.13/site-packages/opencv_contrib_python-4.13.0.90.dist-info', 'version': '4.13.0.90'}, 'opencv-python': {'metadata_directory': 'lib/python3.13/site-packages/opencv_python-4.12.0.88.dist-info', 'version': '4.12.0.88'}, 'packaging': {'metadata_directory': 'lib/python3.13/site-packages/packaging-25.0.dist-info', 'version': '25.0'}, 'pandas': {'metadata_directory': 'lib/python3.13/site-packages/pandas-2.3.2.dist-info', 'version': '2.3.2'}, 'parso': {'metadata_directory': 'lib/python3.13/site-packages/parso-0.8.5.dist-info', 'version': '0.8.5'}, 'pexpect': {'metadata_directory': 'lib/python3.13/site-packages/pexpect-4.9.0.dist-info', 'version': '4.9.0'}, 'pillow': {'metadata_directory': 'lib/python3.13/site-packages/pillow-12.3.0.dist-info', 'version': '12.3.0'}, 'prompt-toolkit': {'metadata_directory': 'lib/python3.13/site-packages/prompt_toolkit-3.0.52.dist-info', 'version': '3.0.52'}, 'propcache': {'metadata_directory': 'lib/python3.13/site-packages/propcache-0.4.1.dist-info', 'version': '0.4.1'}, 'protobuf': {'metadata_directory': 'lib/python3.13/site-packages/protobuf-6.33.0.dist-info', 'version': '6.33.0'}, 'psutil': {'metadata_directory': 'lib/python3.13/site-packages/psutil-7.1.2.dist-info', 'version': '7.1.2'}, 'ptyprocess': {'metadata_directory': 'lib/python3.13/site-packages/ptyprocess-0.7.0.dist-info', 'version': '0.7.0'}, 'pulp': {'metadata_directory': 'lib/python3.13/site-packages/pulp-3.3.1.dist-info', 'version': '3.3.1'}, 'pure-eval': {'metadata_directory': 'lib/python3.13/site-packages/pure_eval-0.2.3.dist-info', 'version': '0.2.3'}, 'pyarrow': {'metadata_directory': 'lib/python3.13/site-packages/pyarrow-22.0.0.dist-info', 'version': '22.0.0'}, 'pycparser': {'metadata_directory': 'lib/python3.13/site-packages/pycparser-2.23.dist-info', 'version': '2.23'}, 'pydantic': {'metadata_directory': 'lib/python3.13/site-packages/pydantic-2.12.5.dist-info', 'version': '2.12.5'}, 'pydantic-core': {'metadata_directory': 'lib/python3.13/site-packages/pydantic_core-2.41.5.dist-info', 'version': '2.41.5'}, 'pygments': {'metadata_directory': 'lib/python3.13/site-packages/pygments-2.19.2.dist-info', 'version': '2.19.2'}, 'pyparsing': {'metadata_directory': 'lib/python3.13/site-packages/pyparsing-3.2.3.dist-info', 'version': '3.2.3'}, 'python-dateutil': {'metadata_directory': 'lib/python3.13/site-packages/python_dateutil-2.9.0.post0.dist-info', 'version': '2.9.0.post0'}, 'pytz': {'metadata_directory': 'lib/python3.13/site-packages/pytz-2025.2.dist-info', 'version': '2025.2'}, 'pyyaml': {'metadata_directory': 'lib/python3.13/site-packages/pyyaml-6.0.3.dist-info', 'version': '6.0.3'}, 'regex': {'metadata_directory': 'lib/python3.13/site-packages/regex-2025.11.3.dist-info', 'version': '2025.11.3'}, 'requests': {'metadata_directory': 'lib/python3.13/site-packages/requests-2.32.5.dist-info', 'version': '2.32.5'}, 'rich': {'metadata_directory': 'lib/python3.13/site-packages/rich-14.3.0.dist-info', 'version': '14.3.0'}, 'safetensors': {'metadata_directory': 'lib/python3.13/site-packages/safetensors-0.7.0.dist-info', 'version': '0.7.0'}, 'scikit-learn': {'metadata_directory': 'lib/python3.13/site-packages/scikit_learn-1.8.0.dist-info', 'version': '1.8.0'}, 'scipy': {'metadata_directory': 'lib/python3.13/site-packages/scipy-1.16.3.dist-info', 'version': '1.16.3'}, 'setuptools': {'metadata_directory': 'lib/python3.13/site-packages/setuptools-80.9.0.dist-info', 'version': '80.9.0'}, 'six': {'metadata_directory': 'lib/python3.13/site-packages/six-1.17.0.dist-info', 'version': '1.17.0'}, 'sniffio': {'metadata_directory': 'lib/python3.13/site-packages/sniffio-1.3.1.dist-info', 'version': '1.3.1'}, 'soupsieve': {'metadata_directory': 'lib/python3.13/site-packages/soupsieve-2.8.dist-info', 'version': '2.8'}, 'stack-data': {'metadata_directory': 'lib/python3.13/site-packages/stack_data-0.6.3.dist-info', 'version': '0.6.3'}, 'starlette': {'metadata_directory': 'lib/python3.13/site-packages/starlette-0.50.0.dist-info', 'version': '0.50.0'}, 'sympy': {'metadata_directory': 'lib/python3.13/site-packages/sympy-1.14.0.dist-info', 'version': '1.14.0'}, 'threadpoolctl': {'metadata_directory': 'lib/python3.13/site-packages/threadpoolctl-3.6.0.dist-info', 'version': '3.6.0'}, 'timm': {'metadata_directory': 'lib/python3.13/site-packages/timm-1.0.22.dist-info', 'version': '1.0.22'}, 'tokenizers': {'metadata_directory': 'lib/python3.13/site-packages/tokenizers-0.22.1.dist-info', 'version': '0.22.1'}, 'torch': {'metadata_directory': 'lib/python3.13/site-packages/torch-2.9.1.dist-info', 'version': '2.9.1'}, 'torchvision': {'metadata_directory': 'lib/python3.13/site-packages/torchvision-0.24.1.dist-info', 'version': '0.24.1'}, 'tqdm': {'metadata_directory': 'lib/python3.13/site-packages/tqdm-4.67.1.dist-info', 'version': '4.67.1'}, 'traitlets': {'metadata_directory': 'lib/python3.13/site-packages/traitlets-5.14.3.dist-info', 'version': '5.14.3'}, 'transformers': {'metadata_directory': 'lib/python3.13/site-packages/transformers-4.57.3.dist-info', 'version': '4.57.3'}, 'typing-extensions': {'metadata_directory': 'lib/python3.13/site-packages/typing_extensions-4.15.0.dist-info', 'version': '4.15.0'}, 'typing-inspection': {'metadata_directory': 'lib/python3.13/site-packages/typing_inspection-0.4.2.dist-info', 'version': '0.4.2'}, 'tzdata': {'metadata_directory': 'lib/python3.13/site-packages/tzdata-2025.2.dist-info', 'version': '2025.2'}, 'urllib3': {'metadata_directory': 'lib/python3.13/site-packages/urllib3-2.5.0.dist-info', 'version': '2.5.0'}, 'uvicorn': {'metadata_directory': 'lib/python3.13/site-packages/uvicorn-0.40.0.dist-info', 'version': '0.40.0'}, 'wcwidth': {'metadata_directory': 'lib/python3.13/site-packages/wcwidth-0.2.14.dist-info', 'version': '0.2.14'}, 'xxhash': {'metadata_directory': 'lib/python3.13/site-packages/xxhash-3.7.0.dist-info', 'version': '3.7.0'}, 'yarl': {'metadata_directory': 'lib/python3.13/site-packages/yarl-1.22.0.dist-info', 'version': '1.22.0'}}
DECLARED_MARKER_ENVIRONMENT={'implementation_name': 'cpython', 'implementation_version': '3.13.7', 'os_name': 'posix', 'platform_machine': 'arm64', 'platform_python_implementation': 'CPython', 'platform_release': '24.6.0', 'platform_system': 'Darwin', 'platform_version': 'Darwin Kernel Version 24.6.0: Mon Jan 19 22:01:13 PST 2026; root:xnu-11417.140.69.708.3~1/RELEASE_ARM64_T8122', 'python_full_version': '3.13.7', 'python_version': '3.13', 'sys_platform': 'darwin'}
DECLARED_REQUESTED_EXTRAS={'fsspec': ['http']}
SHARED_MODULE_COOWNERS={'cv2': ['opencv-contrib-python', 'opencv-python']}
_META_PATH_OWNER=None
_META_PATH_LOCK=threading.RLock()
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
    from packaging.markers import default_environment
    stage.require(default_environment()==DECLARED_MARKER_ENVIRONMENT,"prospective runtime dependency marker environment differs")
    framework=Path(sys.base_prefix).resolve();stdlib=Path(sysconfig.get_path("stdlib")).resolve()
    site=Path(sysconfig.get_path("purelib")).resolve()
    stage.require(framework.is_absolute() and not framework.is_relative_to(Path.cwd()) and
                  stdlib.is_relative_to(framework) and site.is_relative_to(stdlib),"installed canonical Python framework outside project")
    pending=list(sorted(DECLARED_DISTRIBUTIONS));absent=[]
    distributions={};roots=set();files={};aliases=[];module_owners={}
    def add(p,role):
        p=Path(p).absolute();stage.require(p.is_relative_to(framework),"runtime path outside framework")
        name=str(p.relative_to(framework));stage.relative(name)
        stage.require(not any(x.is_symlink() for x in (p,*p.parents)) and p.is_file(),"runtime regular nonsymlink path")
        claim_role(files,name,role)
    while pending:
        name=pending.pop().lower().replace("_","-").replace(".","-")
        if name in distributions:continue
        stage.require(name in DECLARED_DISTRIBUTIONS,"undeclared runtime distribution; no population expansion")
        matches=list(importlib.metadata.distributions(name=name,path=[str(site)]))
        stage.require(len(matches)==1,"missing or duplicate installed distribution metadata")
        d=matches[0];record=tuple(d.files or ())
        stage.require(record,"distribution file metadata missing")
        identity={"version":d.version,"metadata_directory":str(Path(d._path).absolute().relative_to(framework))}
        stage.require(identity==DECLARED_DISTRIBUTIONS[name],"explicit installed runtime distribution identity differs")
        distributions[name]=identity
        # Only declaration-time dependency-requested extras participate. No new
        # extra or installed optional package can enlarge this fixed population.
        for raw in d.requires or ():
            req=Requirement(raw)
            active=req.marker is None or any(req.marker.evaluate({"extra":e}) for e in ("",*DECLARED_REQUESTED_EXTRAS.get(name,())))
            if active:
                child=req.name.lower().replace("_","-").replace(".","-")
                stage.require(child in DECLARED_DISTRIBUTIONS and not req.url,"undeclared mandatory runtime dependency")
                stage.require(set(req.extras)<=set(DECLARED_REQUESTED_EXTRAS.get(child,())),"undeclared requested runtime dependency extras")
                stage.require(not req.specifier or req.specifier.contains(DECLARED_DISTRIBUTIONS[child]["version"],prereleases=True),"incompatible declared runtime dependency")
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
                if len(parts)>1:module_owners.setdefault(parts[0],set()).add(name)
                elif p.suffix==".py":module_owners.setdefault(p.stem,set()).add(name)
    stage.require(set(distributions)==set(DECLARED_DISTRIBUTIONS),"complete explicit distribution population")
    shared={m:sorted(v) for m,v in module_owners.items() if len(v)>1}
    stage.require(shared==SHARED_MODULE_COOWNERS,"unresolved runtime code module coownership")
    stage.require(not shared or (str((site/"cv2").relative_to(framework)) in roots and (site/"cv2").resolve()==site/"cv2" and not (site/"cv2").is_symlink()),"declared shared canonical cv2 root differs")
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
        "explicit_dependency_requested_extras":copy.deepcopy(DECLARED_REQUESTED_EXTRAS),"shared_module_coowners":shared,
        "shared_origin_statement":"current canonical cv2 filesystem bytes once plus both metadata claims; not independent wheel purity",
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
        "file_bindings":bindings,"source_contract_json_sha256":stage.V4_DEPENDENCIES_SHA,"actual_model_weight_cache_or_SC_annotation_read":False,
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


class MissingLoader(importlib.abc.Loader):
    """Terminal absent result: natural ImportError semantics, no finder fallback."""
    def __init__(self,name):self.name=name
    def create_module(self,spec):raise ModuleNotFoundError("No module named "+repr(self.name),name=self.name)
    def exec_module(self,module):raise ModuleNotFoundError("No module named "+repr(self.name),name=self.name)


class BoundRuntime(importlib.abc.MetaPathFinder):
    def __init__(self,receipt,scope):
        stage.require(scope in stage._SCOPES,"genuine final-authorized or newly allocated invented scope")
        stage.require(type(receipt) is dict and receipt.get("status")=="finite_runtime_code_only_source_receipt_not_fit_authority" and
                      receipt.get("runtime")==stage.RUNTIME and receipt.get("source_contract_json_sha256")==stage.V4_DEPENDENCIES_SHA,"finite runtime receipt schema")
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
            self._claim_meta_path()
            try:stage.require(selected==population(),"current metadata population differs from prospective plan")
            finally:self._release_meta_path()

    def _claim_meta_path(self):
        global _META_PATH_OWNER
        with _META_PATH_LOCK:
            stage.require(_META_PATH_OWNER is None and not hasattr(self,"_owned_meta_path"),"nested or concurrent runtime guard ownership")
            stage.require(type(sys.meta_path) is list,"ordinary complete original meta_path list required")
            self._original_meta_path=sys.meta_path;self._original_entries=tuple(sys.meta_path)
            self._owner_thread=threading.get_ident();self._owned_meta_path=[self]
            _META_PATH_OWNER=self;sys.meta_path=self._owned_meta_path
    def _verify_meta_path(self):
        stage.require(_META_PATH_OWNER is self and threading.get_ident()==self._owner_thread and
                      sys.meta_path is self._owned_meta_path and len(sys.meta_path)==1 and sys.meta_path[0] is self,
                      "exclusive runtime meta_path/thread ownership changed")
    def _release_meta_path(self):
        global _META_PATH_OWNER
        with _META_PATH_LOCK:
            self._verify_meta_path()
            stage.require(len(self._original_meta_path)==len(self._original_entries) and
                          all(a is b for a,b in zip(self._original_meta_path,self._original_entries)),"original meta_path entries changed")
            sys.meta_path=self._original_meta_path;_META_PATH_OWNER=None
            del self._owned_meta_path,self._original_meta_path,self._original_entries,self._owner_thread
    def __enter__(self):self.install();return self
    def __exit__(self,exc_type,exc,tb):self.close();return False
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
        self._verify_meta_path()
        for finder in (importlib.machinery.BuiltinImporter,importlib.machinery.FrozenImporter):
            spec=finder.find_spec(fullname,path,target)
            if spec is not None:return spec
        spec=importlib.machinery.PathFinder.find_spec(fullname,path,target)
        if spec is None:return None
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
        stage.require(not self.active,"one runtime guard");self.validate_files();self.loaded_inventory();self._claim_meta_path();self.active=True
    def validate(self):return {**self.validate_files(),"loaded_modules":self.loaded_inventory(),"new_import_checks":copy.deepcopy(self.events)}
    def close(self):
        if self.active:
            self._release_meta_path();self.active=False
