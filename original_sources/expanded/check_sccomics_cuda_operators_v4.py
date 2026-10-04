"""Root-run invented CUDA operator compatibility, not fit/data/model loading.

No Torch import by module import/default CLI. Explicit source-reviewed root spec
is required. Original source_models/core operators run on invented containers;
all checks fail closed, with failures saved. No ANN/source cache/cold weights.
"""
from __future__ import annotations
import argparse,copy,hashlib,importlib.util,io,json,os,pathlib,platform,sys,traceback

SELF='src/check_sccomics_cuda_operators_v4.py'
IDENTITY='CUDA_ACTUAL_INVENTED_OPERATOR_COMPATIBILITY_NO_CORPUS_OR_PRETRAINED_WEIGHT'

class OperatorFailure(RuntimeError):
    def __init__(self,error,result):super().__init__(str(error));self.result=result

def implementation():
    p=pathlib.Path(__file__).resolve().parent/'sccomics_cuda_cloud_v4.py';spec=importlib.util.spec_from_file_location('_CUDA_SOURCE_CONTROLLER',p);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module

def metadata_correction_review(api,files,specification):
    # The original review covers v3 operations. A distinct focused review must
    # bind v4 routes and the exact metadata conversion; v3 is never called v4.
    review=specification['independent_source_review'];api.closed(review,('path','sha256','size_bytes'));api.relative(review['path'])
    api.require(review['path']=='research/round4_sccomics_cuda_metadata_correction_independent_source_review.json','fixed distinct metadata corrective review duty')
    r=api.unique(files.read(review['path'],{k:review[k] for k in ('sha256','size_bytes')}))
    api.require(r.get('status')=='source_review_passed' and r.get('independent_from_implementer') is True and r.get('operator_helper_sha256')==specification['helper']['sha256'] and r.get('runner_sha256')==specification['runner']['sha256'],'different SOURCE-reviewed v4 helper/source')
    api.require(r.get('scope')=='cuda_metadata_str_normalization_v4_source_only' and r.get('operation_AST_matches_v3_except_torch_version_normalization') is True and r.get('finite_JSON_guard_unchanged') is True and r.get('actual_nine_operator_compatibility') is False and r.get('scientific_execution_authority') is False,'focused corrective review is SOURCE only, no invented actual pass')
    prior=r.get('prior_operator_source_review');api.closed(prior,('path','sha256','size_bytes'))
    api.require(prior=={'path':'research/round4_sccomics_cloud_cuda_independent_source_review.json','sha256':'47770906b14ff457f3c13f5fcfa11f092b5285a5a2cd8d782c442757a4fafe34','size_bytes':99548},'immutable original v3 source review descriptor')
    p=api.unique(files.read(prior['path'],{k:prior[k] for k in ('sha256','size_bytes')}))
    api.require(p.get('status')=='source_review_passed' and p.get('independent_from_implementer') is True and p.get('operator_helper_sha256')=='0d286015a905cd6707ea1bd9c50779382c98975d6486c7d19512b60dfabc400e' and p.get('runner_sha256')=='fb3f2afc056ea260933a244828f1ec62181479eb8cc4024c29141a236f84a20f','original different-source reviewed operation identity')
    return review

def actual_check(api,files,specification):
    api.closed(specification,('version','purpose','root_actual_invented_check_authorized','helper','runner','science','config','runtime_versions','cuda_policy','independent_source_review'))
    api.require(type(specification['version']) is int and specification['version']==1 and specification['purpose']==IDENTITY and specification['root_actual_invented_check_authorized'] is True,'invented operator-only actual authority')
    api.require(specification['science']==api.SCIENCE_PINS,'ten unchanged source identities')
    files.verify(SELF,specification['helper']);files.verify(api.SELF,specification['runner'])
    review=metadata_correction_review(api,files,specification)
    config=specification['config'];api.closed(config,('path','sha256','size_bytes'));api.require(config['path']==api.COLD_ROOT+'/config.json','only original cold config metadata, not weights');api.require({k:config[k] for k in ('sha256','size_bytes')}=={'sha256': '2756f064f4f474e06da793dcc85994e876eaf845eb3f9c75968c2e8924c3aaa8', 'size_bytes': 620},'original MatSciBERT config byte identity');raw_config=files.read(config['path'],{k:config[k] for k in ('sha256','size_bytes')});cfg=api.unique(raw_config)
    captures={n:files.read('src/'+n+'.py',specification['science']['src/'+n+'.py']) for n in api.SCIENCE}
    api.require(specification['cuda_policy']=={'device':'cuda:0','dtype':'float32','deterministic':True,'cublas_workspace':':4096:8','tf32':False,'autocast':False,'attention_policy':'library_default_recorded_before_fit'},'same deterministic policy')
    old=os.environ.get('CUBLAS_WORKSPACE_CONFIG');api.require(old is None or old==':4096:8','workspace policy');os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8';os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1'
    result={'identity':IDENTITY,'passed':False,'science':specification['science'],'cuda_policy':specification['cuda_policy'],'helper_sha256':specification['helper']['sha256'],'runner_sha256':specification['runner']['sha256'],'checks':[],'ANN_source_text_cache_pretrained_weights_or_API_read':False}
    with api.CapturedScience(captures) as science:
        import torch,numpy,transformers,tokenizers,safetensors,networkx
        versions={'python':platform.python_version(),'torch':str(torch.__version__),'numpy':numpy.__version__,'transformers':transformers.__version__,'tokenizers':tokenizers.__version__,'safetensors':safetensors.__version__,'networkx':networkx.__version__}
        api.require(versions==specification['runtime_versions'],'exact declared runtime versions');api.require(not torch.cuda.is_initialized() and torch.cuda.is_available(),'fresh CUDA required');api.require(not torch.is_autocast_enabled(),'no actual CUDA autocast')
        torch.use_deterministic_algorithms(True,warn_only=False);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True;torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.set_float32_matmul_precision('highest')
        models=science['sccomics_source_models'];core=science['sccomics_fitting_core'];torch.manual_seed(20261013);torch.cuda.manual_seed_all(20261013)
        config_object=transformers.BertConfig(**cfg);base=transformers.BertModel(config_object);api.require(len(base.encoder.layer)==12 and config_object.hidden_size==768 and config_object.num_attention_heads==12 and config_object.intermediate_size==3072,'original BERT dimensions');layer=copy.deepcopy(base.encoder.layer[11]).cuda().float();attention=getattr(base.config,'_attn_implementation',None);del base
        result.update(runtime_versions=versions,GPU=torch.cuda.get_device_name(0),compute_capability=list(torch.cuda.get_device_capability(0)),attention_implementation=attention,last_block_class=type(layer).__name__,torch_cuda_build=torch.version.cuda,cudnn=torch.backends.cudnn.version())
        def check(name,fn):
            try:detail=fn();torch.cuda.synchronize();result['checks'].append({'id':name,'passed':True,'detail':detail})
            except BaseException as error:result['checks'].append({'id':name,'passed':False,'type':type(error).__name__,'error':str(error),'traceback':traceback.format_exc()});raise OperatorFailure(error,result) from error
        source=b'INVENTED_SIX_WORDPIECES';saved={'source_id':201,'split':'train','source_sha256':api.sha(source),'wordpieces':6,'offsets':[(i,i+1) for i in range(6)],'candidate_span_count':21,'source_only':True,'annotation_content_used':False,'chunks':[{'start':0,'end':4,'hidden':torch.randn(1,6,768),'mask':torch.zeros(1,1,1,6)},{'start':2,'end':6,'hidden':torch.randn(1,6,768),'mask':torch.zeros(1,1,1,6)}]}
        def container():
            b=io.BytesIO();torch.save(saved,b);loaded=torch.load(io.BytesIO(b.getvalue()),map_location='cpu',weights_only=True);core._validate_cache(loaded);api.require(loaded['wordpieces']==6 and torch.equal(loaded['chunks'][0]['hidden'],saved['chunks'][0]['hidden']),'weights_only invented original-shaped container');return {'original_cache_not_opened':True,'4D_masks':[list(c['mask'].shape) for c in loaded['chunks']]}
        check('Torch25_weights_only_original_shaped_invented_cache',container)
        def overlap():
            layer.eval();tokens=models.token_features(layer,saved,'cuda:0');values=[]
            with torch.no_grad():
                for c in saved['chunks']:values.append(layer(c['hidden'].cuda(),attention_mask=c['mask'].cuda())[0][0,1:5])
                expected=torch.stack([values[0][0],values[0][1],(values[0][2]+values[1][0])/2,(values[0][3]+values[1][1])/2,values[1][2],values[1][3]])
            api.require(tokens.shape==(6,768) and torch.allclose(tokens,expected,atol=1e-6,rtol=1e-5),'overlap averaged exact source operator/reference')
            layer.zero_grad(set_to_none=True);tokens.square().mean().backward();api.require(all(p.grad is not None and bool(torch.isfinite(p.grad).all()) for p in layer.parameters()),'strict index_add backward with actual BERT4Dmask');return {'tokens':[6,768],'strict_deterministic_index_add_forward_backward':True}
        check('original_overlap_index_add_gradient_and_actual_BERT_4Dmask',overlap)
        def repeatability():
            layer.train();results=[]
            for _ in range(2):
                torch.manual_seed(20261013);torch.cuda.manual_seed_all(20261013);layer.zero_grad(set_to_none=True);t=models.token_features(layer,saved,'cuda:0');t.square().mean().backward();results.append((t.detach().cpu(),{n:p.grad.detach().cpu().clone() for n,p in layer.named_parameters()}))
            api.require(torch.equal(results[0][0],results[1][0]) and all(torch.equal(results[0][1][n],results[1][1][n]) for n in results[0][1]),'same build seeded repeat equality');return {'same_build_seeded_bitwise_repeat':True,'cross_MPS_CUDA_equivalence_claim':False}
        check('strict_same_build_dropout_overlap_repeat',repeatability)
        def span_and_empty():
            model=models.SpanDetector(copy.deepcopy(layer)).to('cuda:0').float();tokens=models.token_features(model.last_layer,saved,'cuda:0');candidates=models.span_candidates(saved['offsets']);logits=model.spans(tokens,candidates);api.require(logits.shape==(21,9),'all toy span rows');logits.square().mean().backward();empty=model.spans(tokens.new_empty((0,768)),[]);api.require(empty.shape==(0,9),'zero NER output shape retained');return {'complete21toyspans':True,'empty_shape':[0,9]}
        check('original_span_head_complete_and_zero',span_and_empty)
        mentions=[{'type':'Main','segments':[[0,1]]},{'type':'Doping','segments':[[2,3]]},{'type':'Element','segments':[[4,6]]}]
        for kind in api.HEADS:
            def paired(kind=kind):
                model=models.DirectedHead(copy.deepcopy(layer),kind).to('cuda:0').float();tokens=models.token_features(model.last_layer,saved,'cuda:0');matrix,empty=models.mention_matrix(tokens,saved['offsets'],mentions);api.require(not empty,'nonempty synthetic coverage');pairs=models.pair_population(mentions);features=models.explicit_features(mentions,pairs,6);donor=models.donor_permutation(mentions,'201',api.sha(source),20261013)
                logits=model.pairs(matrix,mentions,pairs,6,full_features=features,donor=donor);api.require(logits.shape==(9,5),'full N²×5 including diagonal');logits.square().mean().backward();empty=model.pairs(matrix.new_empty((0,768)),[],[],6,full_features=numpy.empty((0,23),dtype=numpy.float32),donor=numpy.empty(0,dtype=numpy.int64));api.require(empty.shape==(0,5),'zero pairs shape');return {'all9orderedtoyrows':True,'empty_shape':[0,5]}
            check('original_'+kind+'_complete_and_zero',paired)
        def state_format():
            from safetensors.torch import save,load
            state={n:t.detach().cpu().contiguous() for n,t in layer.state_dict().items()};safe=load(save(state));api.require(set(safe)==set(state) and all(torch.equal(safe[n],state[n]) for n in state),'safetensors invented state')
            b=io.BytesIO();torch.save(state,b);parsed=torch.load(io.BytesIO(b.getvalue()),map_location='cpu',weights_only=True);api.require(all(torch.equal(parsed[n],state[n]) for n in state),'epoch tensor format');return {'invented_state_only':True}
        check('safetensors_and_weights_only_invented_state',state_format)
        result['CUDA_RNG_state_sizes']=[len(s) for s in torch.cuda.get_rng_state_all()]
        result['passed']=all(c['passed'] for c in result['checks']) and len(result['checks'])==9
    for name,b in specification['science'].items():files.verify(name,b)
    files.verify(SELF,specification['helper']);files.verify(api.SELF,specification['runner']);files.verify(config['path'],{k:config[k] for k in ('sha256','size_bytes')});files.verify(review['path'],{k:review[k] for k in ('sha256','size_bytes')})
    return result

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',required=True);p.add_argument('--specification',required=True);p.add_argument('--specification-sha256',required=True);p.add_argument('--output',required=True);p.add_argument('--execute-invented-operator-compatibility',action='store_true');a=p.parse_args();api=implementation()
    api.require(a.execute_invented_operator_compatibility,'default no Torch/NN/operator/input access');files=api.Files(a.root);api.relative(a.specification);api.relative(a.output);api.require(a.specification.startswith('research/round4_sccomics_cuda_operator_') and a.specification.endswith('.json'),'compatibility specification metadata path');api.require(a.output.startswith('research/'),'distinct operator receipt')
    size=files.path(a.specification).stat(follow_symlinks=False).st_size;raw=files.read(a.specification,{'sha256':a.specification_sha256,'size_bytes':size});specification=api.unique(raw)
    try:
        result=actual_check(api,files,specification);files.verify(a.specification,{'sha256':a.specification_sha256,'size_bytes':size})
    except BaseException as error:
        result=error.result if isinstance(error,OperatorFailure) else {'identity':IDENTITY,'passed':False}
        result.update(error_type=type(error).__name__,error=str(error),traceback=traceback.format_exc(),not_a_fit=True)
    files.new(a.output,api.encoded(result));print(json.dumps({'passed':result['passed'],'output':a.output,'scope':'invented_operators_only'}));return 0 if result['passed'] else 1
if __name__=='__main__':raise SystemExit(main())
