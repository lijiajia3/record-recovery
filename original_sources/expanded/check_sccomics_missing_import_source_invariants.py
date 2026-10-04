"""SOURCE text/AST-only repair comparison; never import scientific modules."""
from __future__ import annotations
import ast
import hashlib
import json
from pathlib import Path


def definitions(text):
    lines = text.splitlines(keepends=True)
    return {n.name: ''.join(lines[n.lineno-1:n.end_lineno]) for n in ast.parse(text).body
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))}


def assignments(text):
    return {tuple(t.id for t in n.targets if isinstance(t, ast.Name)): ast.dump(n.value, include_attributes=False)
            for n in ast.parse(text).body if isinstance(n, ast.Assign)}


def main():
    root = Path(__file__).absolute().parents[1]
    target = root / 'research/round4_sccomics_runtime_missing_helpers_v4_inputs/standalone_source_invariant_check.json'
    if target.exists() or any(p.is_symlink() for p in (target.parent, *target.parent.parents)):
        raise ValueError('fresh explicit SOURCE result required')
    names = ('src/sccomics_supervised_stages_v2.py', 'src/sccomics_supervised_stages_v3.py',
             'src/sccomics_supervised_runtime_v2.py', 'src/sccomics_supervised_runtime_v3.py',
             'src/check_sccomics_runtime_imports_source_only_v3.py', 'src/check_sccomics_runtime_imports_source_only_v4.py',
             'src/check_sccomics_missing_import_source_invariants.py')
    raw = {}
    for name in names:
        p = root / name
        if any(q.is_symlink() for q in (p, *p.parents)) or not p.is_file():
            raise ValueError('regular nonsymlink SOURCE required')
        raw[name] = p.read_bytes()
    text = {n: b.decode() for n, b in raw.items()}
    stage0, stage1 = text[names[0]], text[names[1]]
    d0, d1 = definitions(stage0), definitions(stage1)
    changed = {n for n in d0.keys() | d1.keys() if d0.get(n) != d1.get(n)}
    if changed != {'authorize_actual', 'main', 'path_duty'}:
        raise ValueError('unexpected scientific definition change')
    a0, a1 = assignments(stage0), assignments(stage1)
    changed_assignments = {k for k in a0.keys() & a1.keys() if a0[k] != a1[k]}
    extra_assignments = set(a1) - set(a0)
    if changed_assignments != {('SELF',), ('SCIENCE',)} or extra_assignments != {('V3_MISSING',), ('V3_MISSING_MD',), ('V3_MISSING_SHA',), ('V3_MISSING_MD_SHA',)} or set(a0) - set(a1):
        raise ValueError('scientific global constant changed')
    r0, r1 = text[names[2]], text[names[3]]
    insertion = '''class MissingLoader(importlib.abc.Loader):
    """Terminal absent result: natural ImportError semantics, no finder fallback."""
    def __init__(self,name):self.name=name
    def create_module(self,spec):raise ModuleNotFoundError("No module named "+repr(self.name),name=self.name)
    def exec_module(self,module):raise ModuleNotFoundError("No module named "+repr(self.name),name=self.name)


class BoundRuntime(importlib.abc.MetaPathFinder):'''
    expected = r0.replace('import sccomics_supervised_stages_v2 as stage', 'import sccomics_supervised_stages_v3 as stage').replace(
        'class BoundRuntime(importlib.abc.MetaPathFinder):', insertion).replace(
        'if spec is None:return importlib.machinery.ModuleSpec(fullname,BlockedLoader(fullname))',
        'if spec is None:return importlib.machinery.ModuleSpec(fullname,MissingLoader(fullname))')
    if r1 != expected:
        raise ValueError('runtime changed outside missing-loader and route')
    helper0, helper1 = definitions(text[names[4]]), definitions(text[names[5]])
    for n in ('verified_open_intent_adapter', 'project_and_network_audit'):
        if helper0[n] != helper1[n]:
            raise ValueError('dirfd/alias policy changed')
    if any((root / n).read_bytes() != b for n, b in raw.items()):
        raise ValueError('SOURCE changed during static comparison')
    result = {'identity': 'ACTUAL_STATIC_SOURCE_REPAIR_INVARIANT_COMPARISON_NOT_FIT_REVIEW',
              'source_bindings': {n: {'sha256': hashlib.sha256(b).hexdigest(), 'size_bytes': len(b)} for n, b in raw.items()},
              'controller_changed_definitions': sorted(changed), 'unchanged_controller_function_class_count': len(d0)-len(changed),
              'all_other_controller_scientific_constants_identical': True,
              'only_runtime_self_routes_and_corrective_provenance_added': True,
              'runtime_exactly_route_MissingLoader_and_terminal_absent_branch': True,
              'dirfd_and_raw_resolved_alias_policy_source_segments_byte_identical': True,
              'actual_SC_installed_runtime_model_or_ANN_read': False, 'fitting_or_import_authority': False}
    target.write_text(json.dumps(result, sort_keys=True, indent=2) + '\n')
    print(json.dumps({'passed': True, 'unchanged_controller_function_class_count': result['unchanged_controller_function_class_count']}, sort_keys=True))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
