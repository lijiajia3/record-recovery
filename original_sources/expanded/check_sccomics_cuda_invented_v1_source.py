"""SOURCE/AST and pure invented arithmetic; deliberately forbids Torch import."""
import ast,builtins,hashlib,json,pathlib,runpy,subprocess,sys,traceback
root=pathlib.Path(__file__).resolve().parents[1];out=root/'research/round4_sccomics_cuda_invented_inputs/SOURCE_CHECKS_v1';out.mkdir(exist_ok=False)
source=root/'src/benchmark_sccomics_cuda_invented_v1.py';raw=source.read_bytes();(out/'script.before.py.txt').write_bytes(raw);rows=[]
def need(v,m):
 if not v:raise AssertionError(m)
def check(name,fn):
 try:rows.append({'id':name,'passed':True,'detail':fn()})
 except BaseException as e:rows.append({'id':name,'passed':False,'type':type(e).__name__,'error':str(e),'traceback':traceback.format_exc()})
original_import=builtins.__import__;torch_attempts=[]
def guarded(name,*a,**k):
 if name=='torch' or name.startswith('torch.'):
  torch_attempts.append(name);raise AssertionError('SOURCE check must not import Torch')
 return original_import(name,*a,**k)
builtins.__import__=guarded
try:module=runpy.run_path(str(source),run_name='_SOURCE_not_main')
finally:builtins.__import__=original_import
check('module_loading_no_Torch_NN',lambda:need(not torch_attempts,'Torch attempted'))
p=module['plan']()
check('150_epoch_120000_doc_15000_update_arithmetic',lambda:need(p['training_budget_reference']=={'fits':15,'epochs_per_fit':10,'checkpoint_count':150,'documents_per_epoch':800,'document_passes':120000,'optimizer_updates':15000},'budget mismatch'))
check('full_510WP_width32_population15824',lambda:need(len(module['candidates'](510))==15824,'candidate count'))
check('short126WP_population3536',lambda:need(len(module['candidates'](126))==3536,'candidate count'))
check('zero_population_retained',lambda:need(module['candidates'](0)==[],'zero population'))
for name,kwargs in [('warmups_bool',{'warmups':True}),('repeats_zero',{'repeats':0}),('budget_inf',{'budget_seconds':float('inf')}),('attention_invalid',{'attention':'flash_assumed'})]:
 def rejected(kwargs=kwargs):
  try:module['plan'](**kwargs)
  except ValueError:return True
  raise AssertionError('invalid inputs accepted')
 check(name,rejected)
def extrapolation():
 train=[{'case':c,'median_seconds':2.0} for c in p['training_cases']];infer=[{'case':c,'median_seconds':.25} for c in p['generation_cases']];e=module['extrapolate'](train,infer)
 need(e['sum_representative_training_compute_seconds']==30000,'5 systems × 3fits ×10epochs×100updates×INVENTED2sec')
 need(e['sum_representative_generation_compute_seconds']==4125,'5 systems×3300docs×INVENTED.25sec');return {'supplied_seconds_are_artificial_not_measurements':True}
check('hand_counted_fake_timing_unit_extrapolation',extrapolation)
def ast_contract():
 tree=ast.parse(raw);imports=[n for n in tree.body if isinstance(n,(ast.Import,ast.ImportFrom))]
 need(not any((isinstance(n,ast.Import) and any(a.name=='torch' for a in n.names)) or (isinstance(n,ast.ImportFrom) and n.module=='torch') for n in imports),'Torch top import')
 run=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='run_cuda');need(any(isinstance(n,ast.Import) and any(a.name=='torch' for a in n.names) for n in ast.walk(run)),'explicit runtime-only Torch branch')
 calls=[n for n in ast.walk(tree) if isinstance(n,ast.Call)];need(not any(isinstance(n.func,ast.Attribute) and n.func.attr in ('load','from_pretrained','urlopen') for n in calls),'data/weights/network loader')
 need(raw.count(b'torch.cuda.synchronize()')>=3,'before/after timing synchronization')
 need(b'foreach=False' in raw and b'ACCUMULATION=8' in raw and b'range(0,source[\'total_rows\'],512)' in raw,'update/block shape');return True
check('AST_delayed_Torch_no_weight_network_loader_synchronization',ast_contract)
# Separate actual dry-run process, still no torch branch or synthetic CUDA.
command=[sys.executable,str(source),'--output',str(out/'actual_dry_run.json')];completed=subprocess.run(command,text=True,capture_output=True);(out/'actual_dry_run.stdout').write_text(completed.stdout);(out/'actual_dry_run.stderr').write_text(completed.stderr)
check('actual_dry_run_CLI_exit0_without_NN',lambda:need(completed.returncode==0 and json.loads((out/'actual_dry_run.json').read_text())['actual_NN_execution'] is False,'dry run failure'))
need(source.read_bytes()==raw,'source changed during test');result={'identity':'ACTUAL_SOURCE_AST_PURE_ARITHMETIC_NOT_GPU_TIMING','script_sha256':hashlib.sha256(raw).hexdigest(),'checks':rows,'total':len(rows),'passed':sum(r['passed'] for r in rows),'failed':sum(not r['passed'] for r in rows),'actual_GPU_or_Torch_or_SC_or_weights_or_API':False}
(out/'actual_results.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:result[k] for k in ('script_sha256','total','passed','failed')}));raise SystemExit(int(result['failed']>0))
