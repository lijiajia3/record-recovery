"""SOURCE-only static inventory of installed optional runtime module origins.

Does not import Torch or any of the six entry packages, execute their source,
prepare a runtime receipt, initialize a model, or access corpus/model files.
This diagnostic report is not a new runtime/import/fitting authorization.
"""
from __future__ import annotations
import argparse
import ast
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path, PurePosixPath
import re
import stat
import sys
import sysconfig

OLD = 'research/round4_sccomics_supervised_runtime_receipt.json'
OLD_SHA = 'dd5e65a4134ab97d71454907b71bf4aea3f7654dbbc66291dbd5ae38344d9cb4'
PREFIX = 'research/round4_sccomics_runtime_optional_inventory_inputs'
SEEDS = frozenset(('torch','numpy','tokenizers','transformers','networkx','safetensors'))


def require(ok, message):
    if not ok:
        raise ValueError(message)


def normalize(name):
    return re.sub(r'[-_.]+', '-', name).lower()


def capture(path):
    p=Path(path)
    require(not any(x.is_symlink() for x in (p,*p.parents)), 'SOURCE metadata/code alias')
    before=p.stat(follow_symlinks=False)
    require(stat.S_ISREG(before.st_mode), 'regular SOURCE metadata/code required')
    fd=os.open(p,os.O_RDONLY|os.O_NONBLOCK|os.O_NOFOLLOW)
    try:
        start=os.fstat(fd)
        require((start.st_dev,start.st_ino)==(before.st_dev,before.st_ino), 'opened SOURCE identity changed')
        parts=[]
        while b:=os.read(fd,1048576):parts.append(b)
        end=os.fstat(fd); after=p.stat(follow_symlinks=False)
        keys=('st_dev','st_ino','st_size','st_mtime_ns','st_ctime_ns')
        require(all(getattr(start,k)==getattr(end,k)==getattr(after,k) for k in keys), 'SOURCE changed during capture')
        raw=b''.join(parts)
        require(len(raw)==start.st_size, 'SOURCE captured size changed')
        return raw, {'sha256':hashlib.sha256(raw).hexdigest(),'size_bytes':len(raw)}
    finally:os.close(fd)


def static_names(raw, filename):
    """All lexical absolute imports + recognized literal module probes.

    Includes imports inside functions/guards; reports potential execution only.
    No assertion that every lexical branch executes on the scientific path.
    """
    tree=ast.parse(raw,filename=filename)
    rows=[]
    for node in ast.walk(tree):
        if isinstance(node,ast.Import):
            rows.extend({'module':a.name.split('.')[0], 'line':node.lineno,'kind':'absolute_import'} for a in node.names)
        elif isinstance(node,ast.ImportFrom) and node.level==0 and node.module:
            rows.append({'module':node.module.split('.')[0], 'line':node.lineno,'kind':'absolute_from_import'})
        elif isinstance(node,ast.Call):
            fn=node.func
            name=fn.id if isinstance(fn,ast.Name) else fn.attr if isinstance(fn,ast.Attribute) else ''
            recognized=name in {'find_spec','import_module','__import__','_check_module_exists','_is_package_available'}
            if recognized and node.args and isinstance(node.args[0],ast.Constant) and type(node.args[0].value) is str:
                value=node.args[0].value
                if value and not value.startswith('.') and re.fullmatch(r'[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*',value):
                    rows.append({'module':value.split('.')[0],'line':node.lineno,'kind':'literal_probe:'+name})
    return rows


def run(root, output):
    raw, old_binding=capture(root/OLD)
    require(old_binding['sha256']==OLD_SHA, 'immutable old receipt differs')
    old=json.loads(raw); framework=Path(old['population']['framework_root'])
    site=Path(sysconfig.get_path('purelib')).resolve()
    require(framework==Path(sys.base_prefix).resolve() and site==framework/old['population']['site_relative_path'], 'running framework metadata differs')
    distributions={}; modules={}; meta_bindings={}
    for d in importlib.metadata.distributions(path=[str(site)]):
        name=normalize(d.metadata['Name'])
        require(name not in distributions, 'duplicate installed distribution metadata')
        dp=Path(d._path).absolute()
        require(dp.is_relative_to(site), 'installed metadata outside exact framework site')
        for leaf in ('METADATA','RECORD','top_level.txt'):
            if (dp/leaf).is_file():
                _, b=capture(dp/leaf); meta_bindings[str((dp/leaf).relative_to(framework))]=b
        record=tuple(d.files or ())
        roots=set()
        for item in record:
            parts=PurePosixPath(str(item)).parts
            if not parts or any(x in {'.','..'} for x in parts) or parts[0].endswith(('.dist-info','.data')):continue
            if len(parts)==1 and parts[0].endswith('.py'):module=Path(parts[0]).stem
            elif len(parts)>1 and Path(parts[-1]).suffix in {'.py','.so','.dylib'}:module=parts[0]
            else:continue
            if re.fullmatch(r'[A-Za-z_]\w*',module):
                roots.add(module); modules.setdefault(module,set()).add(name)
        distributions[name]={'version':d.version,'metadata_directory':str(dp.relative_to(framework)),
                             'requires_dist':list(d.requires or ()), 'top_level_code_modules':sorted(roots)}
    rows=[]; errors=[]; code_bindings={}
    site_prefix=old['population']['site_relative_path']+'/'
    # Fixed lexical source scope: six entry-package directories already in dd5.
    for name,b in sorted(old['file_bindings'].items()):
        if b.get('runtime_role')!='package_code_native' or not name.endswith('.py') or not name.startswith(site_prefix):continue
        top=name[len(site_prefix):].split('/')[0]
        if top not in SEEDS:continue
        source,actual=capture(framework/name)
        require(actual=={k:b[k] for k in ('sha256','size_bytes')}, 'old declared SOURCE file changed')
        code_bindings[name]=actual
        try:
            for r in static_names(source,name):
                r['source']=name
                r['installed_distributions']=sorted(modules.get(r['module'],()))
                r['stdlib_top_level']=r['module'] in sys.stdlib_module_names
                rows.append(r)
        except (SyntaxError,UnicodeError) as e:
            errors.append({'source':name,'error_type':type(e).__name__,'message':str(e)})
    candidates={}
    old_names=set(old['population']['selected_distributions'])
    for r in rows:
        for name in r['installed_distributions']:
            if name in old_names:continue
            slot=candidates.setdefault(name,{'version':distributions[name]['version'],'evidence_count':0,'first_examples':[],'literal_probe_evidence_count':0})
            slot['evidence_count']+=1
            if r['kind'].startswith('literal_probe:'):slot['literal_probe_evidence_count']+=1
            if len(slot['first_examples'])<12:slot['first_examples'].append(r)
    for name,b in {**meta_bindings,**code_bindings}.items():
        require(capture(framework/name)[1]==b, 'final SOURCE/metadata hash differs')
    require(capture(root/OLD)[1]==old_binding, 'old receipt changed')
    result={'identity':'STATIC_INSTALLED_OPTIONAL_RUNTIME_SOURCE_INVENTORY_NOT_RECEIPT_OR_IMPORT_AUTHORITY',
        'old_receipt':{'path':OLD,**old_binding},'framework_root':str(framework),'selected_source_scope':sorted(SEEDS),
        'installed_distribution_count':len(distributions),'source_python_file_count':len(code_bindings),
        'lexical_import_probe_evidence_count':len(rows),'parse_errors':errors,'added_candidates':dict(sorted(candidates.items())),
        'all_installed_distribution_metadata':dict(sorted(distributions.items())),
        'code_bindings':code_bindings,'metadata_bindings':meta_bindings,
        'candidate_statement':'Lexical potential dependencies, not imported/executed or claimed all necessary; proposed finite list requires review before preparation.',
        'actual_six_entry_import_NN_model_tokenizer_SC_gold_cache_weight_API_or_fit':False,
        'receipt_prepared_or_import_scope_extended':False,'all_listed_SOURCE_metadata_start_final_checked':True}
    output.mkdir()
    (output/'inventory.json').write_text(json.dumps(result,sort_keys=True,indent=2)+'\n')
    with (output/'lexical_import_probe_evidence.jsonl').open('w') as f:
        for r in rows:f.write(json.dumps(r,sort_keys=True)+'\n')
    print(json.dumps({'installed':len(distributions),'source_files':len(code_bindings),'candidate_count':len(candidates),'probe_candidates':[n for n,v in candidates.items() if v['literal_probe_evidence_count']], 'parse_errors':len(errors),'actual_import_or_fit':False},sort_keys=True))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output-dir',required=True);args=p.parse_args()
    root=Path(__file__).absolute().parents[1];parts=args.output_dir.split('/')
    require(type(args.output_dir) is str and not Path(args.output_dir).is_absolute() and all(x not in {'','.','..'} for x in parts), 'relative diagnostic output')
    require(str(Path(args.output_dir).parent)==PREFIX and Path(args.output_dir).name.startswith('SOURCE_DIAGNOSTIC_'),'closed diagnostic output')
    parent=root/PREFIX
    require(parent.is_dir() and not any(x.is_symlink() for x in (parent,*parent.parents)), 'existing nonsymlink diagnostic parent')
    out=root/args.output_dir;require(not out.exists(), 'do not overwrite history')
    run(root,out);return 0


if __name__=='__main__':raise SystemExit(main())
