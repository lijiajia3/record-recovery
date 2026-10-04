"""Pure metadata closure for a prospective finite optional runtime population.

No package source executes; the output proposes a declaration, not a receipt.
Optional extras are not silently enabled; exact dependency-requested extras are
closed and recorded as explicit metadata edges before any preparation. Only platform-valid unconditional
requirements are closed; lexically observed optional seeds are explicit.
"""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import re
from packaging.requirements import Requirement
from packaging.markers import default_environment
from packaging.version import Version

INVENTORY='research/round4_sccomics_runtime_optional_inventory_inputs/SOURCE_DIAGNOSTIC_202610032014/inventory.json'
INVENTORY_SHA='2ada0ae4c0de9eb9790926f64ce04b0f5a1d15d727631264a3616b19b6d05877'
OLD='research/round4_sccomics_supervised_runtime_receipt.json'
OLD_SHA='dd5e65a4134ab97d71454907b71bf4aea3f7654dbbc66291dbd5ae38344d9cb4'
COOWNERS={'cv2':['opencv-contrib-python','opencv-python']}


def require(ok,message):
    if not ok:raise ValueError(message)


def norm(name):return re.sub(r'[-_.]+','-',name).lower()


def derive(metadata,original,candidates,*,environment):
    """Fail closed on missing/incompatible mandatory dependency metadata."""
    require(type(metadata) is dict and type(original) is set and type(candidates) is set,'exact metadata/set inputs')
    selected=set(original)|set(candidates);pending=list(sorted(selected));reasons=[];requested_extras={n:set() for n in selected};processed={}
    while pending:
        name=pending.pop(0);require(name in metadata,'missing installed selected distribution: '+name)
        item=metadata[name];require(type(item) is dict and type(item.get('version')) is str,'installed version identity')
        Version(item['version'])
        active_extras=requested_extras[name]
        previous=processed.get(name)
        if previous is not None and previous==active_extras:continue
        processed[name]=set(active_extras)
        for raw in item['requires_dist']:
            req=Requirement(raw);active=req.marker is None or any(req.marker.evaluate(dict(environment,extra=e)) for e in ('',*sorted(active_extras)))
            if not active:continue
            child=norm(req.name)
            require(child in metadata,'missing platform-valid unconditional dependency: '+name+' -> '+child)
            require(not req.url,'unresolved direct-URL installed dependency origin')
            require(not req.specifier or req.specifier.contains(Version(metadata[child]['version']),prereleases=True),
                    'incompatible installed mandatory dependency: '+name+' -> '+str(req))
            reasons.append({'requiring_distribution':name,'required_distribution':child,'requirement':raw,'explicitly_requested_extras':sorted(req.extras)})
            before=set(requested_extras.get(child,set()));requested_extras.setdefault(child,set()).update(req.extras)
            if child not in selected:selected.add(child);pending.append(child)
            elif requested_extras[child]!=before:pending.append(child)
    return sorted(selected),reasons,{n:sorted(x) for n,x in requested_extras.items() if x}


def main():
    root=Path(__file__).absolute().parents[1]
    def get(name,sha):
        p=root/name;require(not any(x.is_symlink() for x in (p,*p.parents)),'metadata alias')
        raw=p.read_bytes();require(hashlib.sha256(raw).hexdigest()==sha,'fixed metadata diagnostic binding');return json.loads(raw)
    inventory=get(INVENTORY,INVENTORY_SHA);old=get(OLD,OLD_SHA)
    environment=default_environment()
    names,edges,extras=derive(inventory['all_installed_distribution_metadata'],set(old['population']['selected_distributions']),set(inventory['added_candidates']),environment=environment)
    moduleowners={}
    for name in names:
        for module in inventory['all_installed_distribution_metadata'][name]['top_level_code_modules']:
            moduleowners.setdefault(module,[]).append(name)
    shared={m:sorted(v) for m,v in moduleowners.items() if len(v)>1}
    require(shared==COOWNERS,'unresolved installed top-level shared-origin metadata ambiguity: '+str(shared))
    result={'identity':'PROSPECTIVE_OPTIONAL_RUNTIME_FINITE_DISTRIBUTION_PLAN_NOT_RECEIPT_OR_IMPORT_AUTHORITY',
        'old_receipt':{'path':OLD,'sha256':OLD_SHA},'source_inventory':{'path':INVENTORY,'sha256':INVENTORY_SHA},
        'selection':'original31 plus24 lexical installed soft-import candidates plus platform-valid unconditional dependencies; finite superset, not minimal executed import proof',
        'selected_distribution_count':len(names),'selected_distributions':{n:{k:inventory['all_installed_distribution_metadata'][n][k] for k in ('version','metadata_directory')} for n in names},
        'mandatory_dependency_edges':edges,'marker_environment':environment,'extras_automatically_enabled':False,'only_dependency_explicitly_requested_extras':extras,
        'shared_module_coowners':shared,'shared_cv2_statement':'Both installed metadata claims retained; one canonical current shared filesystem code/native image. No claim of independent wheel purity or arbitrary winner.',
        'literal_probe_distribution_mapping_notes':{'bs4':['beautifulsoup4'],'PIL':['pillow'],'sklearn':['scikit-learn'],'google.protobuf':['protobuf'],'cv2':COOWNERS['cv2'],
          'quark':['amd-quark'],'triton':['triton','pytorch-triton'],'faiss':['faiss','faiss-cpu','faiss-gpu'],'optimum.quanto':['optimum-quanto','optimum-quanto-underscore metadata alias mentioned by source']},
        'uninstalled_module_mapping_notes_do_not_grant_permission':True,
        'static_inventory_not_complete_runtime_import_proof':True,'actual_six_entry_import_SC_NN_or_receipt_preparation':False}
    output=root/'research/round4_sccomics_runtime_optional_inventory_inputs/prospective_finite_distribution_plan.json'
    require(not output.exists(),'do not replace retained plan')
    output.write_text(json.dumps(result,sort_keys=True,indent=2)+'\n')
    print(json.dumps({'selected_distribution_count':len(names),'names':names,'plan_sha256':hashlib.sha256(output.read_bytes()).hexdigest(),'actual_import_or_receipt_preparation':False}))
    return 0


if __name__=='__main__':raise SystemExit(main())
