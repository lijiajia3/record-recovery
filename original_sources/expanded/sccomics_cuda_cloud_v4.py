"""Distinct practical cloud CUDA stages, SOURCE implementation only until frozen.

No Torch import or data/weight access on import. Cloud never receives held ANN.
Ten science modules stay unchanged; normal installed runtime versions are pinned
and recorded, not claimed as a whole-OS or hostile-process security sandbox.
"""
from __future__ import annotations
import argparse,contextlib,copy,gc,hashlib,importlib,importlib.abc,importlib.util,io,json,math,os,pathlib,platform,random,re,stat,sys,tempfile,time,types

SELF='src/sccomics_cuda_cloud_v4.py'
SCIENCE=tuple('sccomics_'+n for n in ('source_models','training_projection','native_graph','fitting_core','development_selection','primary_matching_v2','statistics','independent_statistics_v2','independent_scoring','test_analysis'))
SCIENCE_PINS={'src/sccomics_source_models.py': {'sha256': '8ff50b5bb88c3fa6296b76ff9f5dab63a43177e1c0c54178aea01dafc59900f0', 'size_bytes': 11092}, 'src/sccomics_training_projection.py': {'sha256': '08d9b2bd079c9be4a0ee1ece208105521dce15a741800750c19f4b5dcb34b7fa', 'size_bytes': 12431}, 'src/sccomics_native_graph.py': {'sha256': 'b59306e02e7b8132e16366f70c729ee63df83ed10ee39e25ab0c38f74e321b53', 'size_bytes': 38267}, 'src/sccomics_fitting_core.py': {'sha256': '4bb8be6ea316de1cf18498a9436bf4b089413be8b2a6579013763a29f5cb0359', 'size_bytes': 13903}, 'src/sccomics_development_selection.py': {'sha256': '03335c910d1a7939801b7e1493deb62d18c2a7c39be40f199d6667771500c317', 'size_bytes': 8688}, 'src/sccomics_primary_matching_v2.py': {'sha256': '53167cf381710b4ee8b67960caf6cf297fd977c78e2c462917e2fd0f606c779d', 'size_bytes': 4361}, 'src/sccomics_statistics.py': {'sha256': '9b92e1e684dd0690589c5b497c18bdaec535f299fcb949c6b4f17e57dc361322', 'size_bytes': 20215}, 'src/sccomics_independent_statistics_v2.py': {'sha256': 'b83c9959b567a016d2177d862c4347465c4f346ac4f4081680b9c5a1532b14c7', 'size_bytes': 20410}, 'src/sccomics_independent_scoring.py': {'sha256': '27846cb4792ac7485d2ca41378ff5c0530330d7ae940ce5ca2913dabe92cf367', 'size_bytes': 35839}, 'src/sccomics_test_analysis.py': {'sha256': '565f8ee4cd93e3ee045f5d74964b34692ece64f4b2e8c047bb17c98fcf0e7eb8', 'size_bytes': 26639}}
SEEDS=(20261013,20261014,20261015);HEADS=('aligned','permuted','ordered_context','biaffine');KINDS=('span',*HEADS)
SPLITS={'train':tuple(range(201,1001)),'dev':tuple(range(101,201)),'test':tuple(range(1,101))}
COLD_ROOT='data/local_baselines/matscibert';COLD_NAMES=('README.md','config.json','model.safetensors','special_tokens_map.json','tokenizer.json','tokenizer_config.json','vocab.txt')
CONFIG='research/round4_sccomics_experimental_configuration_before_fit.json';CONFIG_SHA='742fb7011f2162cb935bd6de3068e38a8c0f3a67c58df8ff266dc73a8814103c'
NER_AMENDMENT='research/round4_sccomics_pre_fit_ner_denominator_amendment.json';NER_AMENDMENT_SHA='e699e091bac59b1f14c5204c2eedee3b6a40a376f45a10229ccf6697f818d9fc'
PROVENANCE={
 'data/sccomics_round4/source_v3/source_acquisition_manifest.json':'a38f1637af74b15246b6b028d3d58b5836094e905d25845e5922668031fee1c7',
 'results/local_baseline/sccomics_native_v1/source_cache/completed_manifest.json':'fe7b66198773d49e7652c7be64d3f5923fdba27c528cb73303f0b95a36194139',
 'data/local_baselines/matscibert/source_manifest.json':'ffdcf65ccf91c8cce33d380701928fa039ffe20edeb3375b68b5bbf6152c6b0c'}
PHASES=('fit_ner','ner_dev','fit_heads','pair_dev','test_sources')
SPAN_THRESHOLDS=(.05,.1,.25,.5,.75,.9,.95,.99);PAIR_THRESHOLDS=(.5,.75,.9,.95,.99)

class CloudIntegrityError(ValueError):pass

def require(value,message):
    if not value:raise CloudIntegrityError(message)
def sha(raw):return hashlib.sha256(raw).hexdigest()
def finite(value):
    stack=[value]
    while stack:
        v=stack.pop()
        if type(v) is dict:require(all(type(k) is str for k in v),'JSON string keys');stack.extend(v.values())
        elif type(v) is list:stack.extend(v)
        elif type(v) is float:require(math.isfinite(v),'finite JSON numbers')
        else:require(type(v) in (str,int,bool,type(None)),'JSON primitives only')
    return value
def encoded(obj):return (json.dumps(finite(obj),sort_keys=True,indent=2,allow_nan=False)+'\n').encode()
def unique(raw):
    def pairs(items):
        result={}
        for k,v in items:require(k not in result,'duplicate JSON key');result[k]=v
        return result
    def bad(_):raise CloudIntegrityError('nonfinite JSON constant')
    try:return finite(json.loads(raw,object_pairs_hook=pairs,parse_constant=bad))
    except (UnicodeError,json.JSONDecodeError,RecursionError) as e:raise CloudIntegrityError('JSON syntax/encoding') from None
def closed(obj,fields):require(type(obj) is dict and set(obj)==set(fields),'closed metadata fields')
def integer(value,minimum=0):require(type(value) is int and value>=minimum,'strict integer range')
def relative(value):
    require(type(value) is str and value and '\\' not in value and '\x00' not in value and not value.startswith('/'),'relative path')
    p=pathlib.PurePosixPath(value);require(str(p)==value and all(x not in ('.','..','') for x in value.split('/')),'canonical nontraversing relative path');return p.parts
def strict_binding(obj):
    closed(obj,('sha256','size_bytes'));require(type(obj['sha256']) is str and re.fullmatch('[0-9a-f]{64}',obj['sha256']) is not None,'SHA256 required');integer(obj['size_bytes'])
def binding(raw):return {'sha256':sha(raw),'size_bytes':len(raw)}
def text_path(identifier):return f'data/sccomics_round4/source_v3/raw_text/{identifier:04}.txt'
def ann_path(identifier):require(identifier in SPLITS['train'],'cloud ANN permission TRAIN201..1000 only');return f'data/sccomics_round4/source_v3/raw_annotations/{identifier:04}.ann'
def cache_path(identifier):
    split=next(s for s,ids in SPLITS.items() if identifier in ids);return f'results/local_baseline/sccomics_native_v1/source_cache/{split}/{identifier:04}.pt'
def data_population():
    return {text_path(i) for i in range(1,1001)}|{cache_path(i) for i in range(1,1001)}|{ann_path(i) for i in SPLITS['train']}|{COLD_ROOT+'/'+n for n in COLD_NAMES}

class Files:
    def __init__(self,root):
        self.root=pathlib.Path(root).absolute();require('..' not in pathlib.Path(root).parts and not any(p.is_symlink() for p in (self.root,*self.root.parents)),'nonalias root');self.open_events=[]
    def path(self,name):
        relative(name);p=self.root.joinpath(*relative(name));require(not any(q.is_symlink() for q in (p,*p.parents)),'no symlink leaf/ancestor');return p
    @contextlib.contextmanager
    def opened(self,name):
        p=self.path(name);before=p.stat(follow_symlinks=False);require(stat.S_ISREG(before.st_mode),'regular file; no FIFO/device')
        fd=os.open(p,os.O_RDONLY|os.O_NONBLOCK|os.O_NOFOLLOW)
        try:
            start=os.fstat(fd);require(stat.S_ISREG(start.st_mode) and (start.st_dev,start.st_ino)==(before.st_dev,before.st_ino),'opened identity');yield fd
            end=os.fstat(fd);after=self.path(name).stat(follow_symlinks=False)
            require(all(getattr(start,k)==getattr(end,k)==getattr(after,k) for k in ('st_dev','st_ino','st_size','st_mtime_ns','st_ctime_ns')),'file changed during capture')
        finally:os.close(fd)
    def read(self,name,wanted):
        strict_binding(wanted)
        with self.opened(name) as fd:
            chunks=[]
            while b:=os.read(fd,1048576):chunks.append(b)
            raw=b''.join(chunks)
        require(binding(raw)==wanted,'captured bytes differ from frozen binding');self.open_events.append({'path':name,'mode':'captured bytes','sha256':sha(raw)});return raw
    def verify(self,name,wanted):
        strict_binding(wanted);h=hashlib.sha256();size=0
        with self.opened(name) as fd:
            while b:=os.read(fd,1048576):h.update(b);size+=len(b)
        require({'sha256':h.hexdigest(),'size_bytes':size}==wanted,'streamed hash differs');return wanted
    def copy(self,name,wanted,destination):
        strict_binding(wanted);h=hashlib.sha256();size=0
        require(not destination.exists() and not any(p.is_symlink() for p in (destination,*destination.parents)),'private snapshot destination')
        with self.opened(name) as fd,destination.open('xb') as target:
            while b:=os.read(fd,1048576):h.update(b);size+=len(b);target.write(b)
        require({'sha256':h.hexdigest(),'size_bytes':size}==wanted,'snapshot hash differs')
    def new(self,name,raw):
        p=self.path(name);p.parent.mkdir(parents=True,exist_ok=True);require(not any(q.is_symlink() for q in (p,*p.parents)),'new output alias')
        fd=os.open(p,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600)
        try:
            at=0
            while at<len(raw):at+=os.write(fd,raw[at:])
            os.fsync(fd)
        finally:os.close(fd)
        return binding(raw)

class CapturedScience:
    def __init__(self,captures):self.captures=captures;self.modules={};self.finder=None
    def __enter__(self):
        require(set(self.captures)==set(SCIENCE) and not any(n in sys.modules for n in SCIENCE),'ten fresh captured science modules')
        owner=self
        class Loader(importlib.abc.Loader):
            def create_module(self,spec):return None
            def exec_module(self,module):
                module.__file__='captured-source://'+module.__name__+'.py';exec(compile(owner.captures[module.__name__],module.__file__,'exec'),module.__dict__);owner.modules[module.__name__]=module
        class Finder(importlib.abc.MetaPathFinder):
            def find_spec(self,name,path=None,target=None):
                if name in owner.captures:return importlib.util.spec_from_loader(name,Loader())
                if name.startswith('sccomics_'):raise CloudIntegrityError('uncaptured scientific import')
                return None
        self.finder=Finder();sys.meta_path.insert(0,self.finder)
        try:
            for n in SCIENCE:importlib.import_module(n)
            return self.modules
        except BaseException:self.__exit__(*sys.exc_info());raise
    def __exit__(self,*args):
        if self.finder in sys.meta_path:sys.meta_path.remove(self.finder)
        for name in SCIENCE:sys.modules.pop(name,None)

def manifest_contract(manifest):
    closed(manifest,('version','purpose','runner','operator_helper','science','inputs','controls','licenses','runtime_versions','cuda_policy'))
    require(type(manifest['version']) is int and manifest['version']==1 and manifest['purpose']=='sccomics_cloud_cuda_before_fit','manifest identity')
    strict_binding(manifest['runner']);strict_binding(manifest['operator_helper']);require(type(manifest['science']) is dict and set(manifest['science'])=={'src/'+n+'.py' for n in SCIENCE},'exact ten scientific SOURCE files')
    require(manifest['science']==SCIENCE_PINS,'unchanged ten scientific SOURCE byte identities')
    require(type(manifest['inputs']) is dict and set(manifest['inputs'])==data_population(),'exact 1000 text/cache,800TRAIN ANN,seven cold files; no held ANN')
    require(type(manifest['controls']) is dict and set(manifest['controls'])=={CONFIG,NER_AMENDMENT,'research/round4_sccomics_cloud_cuda_hardware_amendment.json',*PROVENANCE},'original science plus distinct hardware amendment')
    require(manifest['controls'][CONFIG]['sha256']==CONFIG_SHA and manifest['controls'][NER_AMENDMENT]['sha256']==NER_AMENDMENT_SHA,'old science identities immutable')
    require(all(manifest['controls'][p]['sha256']==h for p,h in PROVENANCE.items()),'original source/cache/cold provenance metadata immutable')
    require(type(manifest['licenses']) is dict and bool(manifest['licenses']),'explicit deployed licenses')
    for name in manifest['licenses']:relative(name);require(name.startswith('licenses/') and pathlib.PurePosixPath(name).suffix in ('.txt','.md','.json'),'license/source documentation only')
    for group in ('science','inputs','controls','licenses'):
        for name,b in manifest[group].items():relative(name);strict_binding(b)
    versions=manifest['runtime_versions'];require(type(versions) is dict and set(versions)=={'python','torch','numpy','transformers','tokenizers','safetensors','networkx'} and all(type(v) is str and v for v in versions.values()),'explicit finite runtime versions')
    closed(manifest['cuda_policy'],('device','dtype','deterministic','cublas_workspace','tf32','autocast','attention_policy'))
    require(manifest['cuda_policy']=={'device':'cuda:0','dtype':'float32','deterministic':True,'cublas_workspace':':4096:8','tf32':False,'autocast':False,'attention_policy':'library_default_recorded_before_fit'},'fixed proposed CUDA policy')
    return manifest

def authorize(files,manifest_raw,gate_raw,phase):
    require(phase in PHASES,'declared phase');manifest=manifest_contract(unique(manifest_raw));gate=unique(gate_raw)
    closed(gate,('version','purpose','actual_root_execution_authorized','manifest_sha256','phase','run_prefix','source_review','hardware_amendment_sha256','prior_receipts','bridge','local_selector_receipt_sha256'))
    require(type(gate['version']) is int and gate['version']==1 and gate['purpose']=='sccomics_cloud_cuda_stage_gate' and gate['actual_root_execution_authorized'] is True,'default no actual stage authority')
    require(gate['manifest_sha256']==sha(manifest_raw) and gate['phase']==phase,'exact phase/manifest authority');relative(gate['run_prefix']);require(gate['run_prefix'].startswith('cloud_outputs/'),'bound run prefix')
    require(gate['hardware_amendment_sha256']==manifest['controls']['research/round4_sccomics_cloud_cuda_hardware_amendment.json']['sha256'],'hardware amendment binding')
    review=gate['source_review'];closed(review,('path','sha256','size_bytes'));relative(review['path']);require(review['path'].startswith('research/') and review['path'].endswith('.json'),'source review metadata role');raw=files.read(review['path'],{k:review[k] for k in ('sha256','size_bytes')});r=unique(raw)
    require(r.get('status')=='source_review_passed' and r.get('independent_from_implementer') is True and r.get('runner_sha256')==manifest['runner']['sha256'] and r.get('manifest_sha256')==sha(manifest_raw) and r.get('operator_helper_sha256')==manifest['operator_helper']['sha256'],'different-source review exact identities')
    require(type(gate['local_selector_receipt_sha256']) is dict,'explicit root-checked local selector receipt hashes')
    expected_local=set() if phase in ('fit_ner','ner_dev') else {'NER'} if phase in ('fit_heads','pair_dev') else {'NER',*HEADS}
    require(set(gate['local_selector_receipt_sha256'])==expected_local and all(type(h) is str and re.fullmatch('[0-9a-f]{64}',h) for h in gate['local_selector_receipt_sha256'].values()),'exact stage local selection proof identities')
    require(type(gate['prior_receipts']) is dict,'explicit prior receipts');require(gate['bridge'] is None or type(gate['bridge']) is dict,'explicit bridge metadata')
    files.verify(SELF,manifest['runner']);files.verify('src/check_sccomics_cuda_operators_v3.py',manifest['operator_helper'])
    for group in ('science','controls','licenses'):
        for name,wanted in manifest[group].items():files.verify(name,wanted)
    provenance_contract(files,manifest)
    hardware_name='research/round4_sccomics_cloud_cuda_hardware_amendment.json'
    hardware=unique(files.read(hardware_name,manifest['controls'][hardware_name]));validate_hardware(files,hardware,manifest)
    return manifest,gate

def provenance_contract(files,manifest):
    objects={p:unique(files.read(p,manifest['controls'][p])) for p in PROVENANCE}
    acq=objects['data/sccomics_round4/source_v3/source_acquisition_manifest.json'];cache=objects['results/local_baseline/sccomics_native_v1/source_cache/completed_manifest.json'];cold=objects['data/local_baselines/matscibert/source_manifest.json']
    require(acq.get('native_source_abstract_records')==1000 and acq.get('all_saved_bytes_rehashed_after_all_copies') is True,'original acquisition byte receipt')
    native={}
    for archive in acq['official_archives'].values():
        for member,d in archive['members'].items():
            if member.endswith('.txt'):name=text_path(int(member[:-4]))
            elif member.endswith('.ann') and int(member[:-4]) in SPLITS['train']:name=ann_path(int(member[:-4]))
            else:continue
            require(name not in native,'duplicate native source member');native[name]={'sha256':d['sha256'],'size_bytes':d['bytes']}
    require(set(native)=={text_path(i) for i in range(1,1001)}|{ann_path(i) for i in SPLITS['train']},'complete original text/TRAIN byte bindings')
    require(cache.get('status')=='completed_cold_source_text_cache_only' and cache.get('complete_native_records')==1000 and cache.get('all_captured_source_model_and_cache_hashes_size_final_checked') is True and cache.get('annotation_content_used_in_source_features') is False,'original all1000 unannotated source cache receipt')
    for d in cache['cache_bindings']:
        i=d['native_id'];require(type(i) is int and 1<=i<=1000 and d['path']==cache_path(i) and d['source_sha256']==native[text_path(i)]['sha256'],'native cached source identity');name=cache_path(i);require(name not in native,'duplicate source cache');native[name]={'sha256':d['sha256'],'size_bytes':d['bytes']}
    require(cold.get('revision')=='ced9d8f5f208712c4a90f98a246fe32155b29995' and cold.get('base_encoder_not_task_finetuned_checkpoint') is True and len(cold['files'])==7,'original MatSciBERT revision/weights')
    for d in cold['files']:
        name=COLD_ROOT+'/'+d['name'];require(d['name'] in COLD_NAMES and name not in native,'cold file population');native[name]={'sha256':d['sha256'],'size_bytes':d['bytes']}
    require(native==manifest['inputs'],'deployment exact original text/cache/TRAIN/cold bytes, no substitute recoding')
    return native

def validate_hardware(files,hardware,manifest):
    closed(hardware,('status','runner_sha256','science','runtime_versions','cuda_policy','source_cache_policy','operator_compatibility_receipt','expected_GPU','expected_compute_capability','expected_attention_implementation','expected_last_block_class'))
    require(hardware['status']=='before_fit_cuda_hardware_amendment_after_actual_operator_check' and hardware['runner_sha256']==manifest['runner']['sha256'] and hardware['science']==manifest['science'] and hardware['runtime_versions']==manifest['runtime_versions'] and hardware['cuda_policy']==manifest['cuda_policy'],'hardware prefit identities')
    require(hardware['source_cache_policy']=='reuse_original_MPS_first11_source_cache_bytes_CUDA_trainable_lastblock','cache provenance policy unchanged')
    d=hardware['operator_compatibility_receipt'];closed(d,('path','sha256','size_bytes'));relative(d['path']);require(d['path'].startswith('research/') and d['path'].endswith('.json'),'operator receipt metadata path')
    r=unique(files.read(d['path'],{k:d[k] for k in ('sha256','size_bytes')}))
    require(r.get('identity')=='CUDA_ACTUAL_INVENTED_OPERATOR_COMPATIBILITY_NO_CORPUS_OR_PRETRAINED_WEIGHT' and r.get('passed') is True and r.get('science')==SCIENCE_PINS and r.get('runtime_versions')==manifest['runtime_versions'] and r.get('cuda_policy')==manifest['cuda_policy'] and r.get('helper_sha256')==manifest['operator_helper']['sha256'] and r.get('runner_sha256')==manifest['runner']['sha256'],'actual operator compatibility gate, no SOURCE-only substitution')
    expected_checks={'Torch25_weights_only_original_shaped_invented_cache','original_overlap_index_add_gradient_and_actual_BERT_4Dmask','strict_same_build_dropout_overlap_repeat','original_span_head_complete_and_zero','safetensors_and_weights_only_invented_state'}|{'original_'+k+'_complete_and_zero' for k in HEADS}
    checks=r.get('checks');require(type(checks) is list and len(checks)==9 and all(type(c) is dict and c.get('passed') is True for c in checks) and {c.get('id') for c in checks}==expected_checks,'all nine actual compatibility checks, no mere passed flag')
    require(r.get('GPU')==hardware['expected_GPU'] and r.get('compute_capability')==hardware['expected_compute_capability'] and r.get('attention_implementation')==hardware['expected_attention_implementation'] and r.get('last_block_class')==hardware['expected_last_block_class'],'recorded hardware/kernel identity')
    return hardware

class CudaBackend:
    def __init__(self,files,manifest,science,temporary):
        self.files,self.manifest,self.science=files,manifest,science;self.input_bindings=manifest['inputs'];self.opened_annotations=[];self.descriptors={};self.active=None
        existing=os.environ.get('CUBLAS_WORKSPACE_CONFIG');require(existing is None or existing==':4096:8','fixed cublas policy conflict');os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
        os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1'
        import torch,numpy,transformers,tokenizers,safetensors,networkx
        self.torch,self.numpy=torch,numpy;self.core=science['sccomics_fitting_core'];self.models=science['sccomics_source_models'];self.native=science['sccomics_native_graph'];self.projection=science['sccomics_training_projection']
        versions={'python':platform.python_version(),'torch':str(torch.__version__),'numpy':numpy.__version__,'transformers':transformers.__version__,'tokenizers':tokenizers.__version__,'safetensors':safetensors.__version__,'networkx':networkx.__version__}
        require(versions==manifest['runtime_versions'],'actual installed runtime differs from declared versions');require(not torch.is_autocast_enabled(),'no actual CUDA autocast');require(not torch.cuda.is_initialized(),'fresh CUDA initialization policy');require(torch.cuda.is_available(),'CUDA unavailable; no fallback')
        torch.use_deterministic_algorithms(True,warn_only=False);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True;torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.set_float32_matmul_precision('highest');self.device='cuda:0'
        self.runtime={'versions':versions,'GPU':torch.cuda.get_device_name(0),'compute_capability':list(torch.cuda.get_device_capability(0)),'memory_bytes':torch.cuda.get_device_properties(0).total_memory,'torch_cuda_build':torch.version.cuda,'cudnn':torch.backends.cudnn.version(),'policy':manifest['cuda_policy'],'installed_runtime_bytes_not_whole_system_certified':True}
        cold=pathlib.Path(temporary)/'cold';cold.mkdir()
        for name in COLD_NAMES:files.copy(COLD_ROOT+'/'+name,self.input_bindings[COLD_ROOT+'/'+name],cold/name)
        self.tokenizer=transformers.AutoTokenizer.from_pretrained(str(cold),local_files_only=True,trust_remote_code=False,use_fast=True,token=False)
        for identifier in range(1,1001):self.cache(identifier)
        self.runtime['all1000_source_cache_tokenizer_boundaries_checked_before_first_fit']=True
        base=transformers.BertModel.from_pretrained(str(cold),local_files_only=True,trust_remote_code=False,use_safetensors=True,token=False)
        require(len(base.encoder.layer)==12,'original twelve blocks');self.original_last=copy.deepcopy(base.encoder.layer[11]).cpu();self.runtime['actual_attention_implementation']=getattr(base.config,'_attn_implementation',None);self.runtime['actual_last_block_class']=type(self.original_last).__name__;del base;gc.collect()
        self.private_cold=cold
        h=unique(files.read('research/round4_sccomics_cloud_cuda_hardware_amendment.json',manifest['controls']['research/round4_sccomics_cloud_cuda_hardware_amendment.json']))
        require(self.runtime['GPU']==h['expected_GPU'] and self.runtime['compute_capability']==h['expected_compute_capability'] and self.runtime['actual_attention_implementation']==h['expected_attention_implementation'] and self.runtime['actual_last_block_class']==h['expected_last_block_class'],'actual model/hardware matches prior operator amendment')
    def text(self,i):return self.files.read(text_path(i),self.input_bindings[text_path(i)])
    def cache(self,i):
        raw=self.files.read(cache_path(i),self.input_bindings[cache_path(i)]);saved=self.torch.load(io.BytesIO(raw),map_location='cpu',weights_only=True);self.core._validate_cache(saved)
        split=next(s for s,ids in SPLITS.items() if i in ids);source=self.text(i)
        require(type(saved.get('source_id')) is int and saved['source_id']==i and saved.get('split')==split and saved.get('source_sha256')==sha(source),'cache/source/fold identity')
        if i not in self.descriptors:
            enc=self.tokenizer(source.decode('utf-8'),add_special_tokens=False,truncation=False,return_offsets_mapping=True)
            offsets=[list(x) for x in enc['offset_mapping']];require(saved['wordpieces']==len(enc['input_ids']) and [list(x) for x in saved['offsets']]==offsets,'original source tokenizer/cache boundaries')
            candidates=[list(x) for x in self.models.span_candidates(offsets)];require(type(saved.get('candidate_span_count')) is int and saved['candidate_span_count']==len(candidates),'complete span population')
            self.descriptors[i]={'candidates':candidates,'text_length':len(source.decode('utf-8')),'wordpieces':len(offsets),'source_sha256':sha(source)}
        return saved
    def gold(self,i):
        require(i in SPLITS['train'],'no DEV/TEST ANN on cloud');raw=self.files.read(ann_path(i),self.input_bindings[ann_path(i)]);self.opened_annotations.append({'source_id':i,'stage_permission':'TRAIN fit callback','sha256':sha(raw)});return self.native.parse_document(self.text(i),raw,gold=True)
    def rng(self):
        n=self.numpy.random.get_state();return {'python':json.loads(json.dumps(random.getstate())),'numpy':{'kind':n[0],'state':n[1].tolist(),'position':n[2],'has_gauss':n[3],'cached_gaussian':n[4]},'torch_cpu':self.torch.get_rng_state().tolist(),'torch_cuda_all':[s.tolist() for s in self.torch.cuda.get_rng_state_all()]}
    def model(self,kind,seed):
        require(kind in KINDS and seed in SEEDS,'declared fit identity');self.core.seed_fit(seed);self.torch.cuda.manual_seed_all(seed);last=copy.deepcopy(self.original_last);m=self.models.SpanDetector(last) if kind=='span' else self.models.DirectedHead(last,kind);return m.to(self.device).float()
    def fit_start(self,kind,seed):
        self.active={'model':self.model(kind,seed)};self.active['optimizer']=self.core.make_optimizer(self.active['model']);return self.active
    def verify_private_cold(self):
        private=Files(self.private_cold)
        for name in COLD_NAMES:private.verify(name,self.input_bindings[COLD_ROOT+'/'+name])
    def donor_diagnostics(self,mentions,seed,i):
        donor=self.models.donor_permutation(mentions,str(i),self.input_bindings[text_path(i)]['sha256'],seed)
        return {'complete_directed_rows':len(mentions)**2,'donor_sha256':sha(donor.astype('<i8').tobytes()),'donor_fixed_points':int(self.numpy.count_nonzero(donor==self.numpy.arange(len(donor)))),'source_mentions_sha256':self.core.json_digest(mentions),'Gold_consulted':False,'inference_row_block_size':512,'candidate_cap':None}
    def fit_epoch(self,handle,kind,seed,epoch,log):
        updates=self.core.train_epoch(handle['model'],handle['optimizer'],lambda i:(self.gold(i),self.cache(i)),kind=kind,seed=seed,epoch=epoch,device=self.device,log_event=log,source_ids=SPLITS['train'],total_updates=1000,warmup=100)
        require(len(updates)==100 and updates[-1]['update']==epoch*100,'full original update budget');b=io.BytesIO();self.torch.save({k:v.detach().cpu() for k,v in handle['model'].state_dict().items()},b)
        return b.getvalue(),{'rng':self.rng(),'optimizer_partitions':json.loads(json.dumps([{k:v for k,v in g.items() if k!='params'} for g in handle['optimizer'].param_groups])),'optimizer_updates':updates,'complete_model_state_dict':True,'optimizer_moments_in_epoch_checkpoint':False,'implicit_resume':False,'final_zero_LR_update_still_step':epoch==10}
    def inference_start(self,kind,seed,raw):
        model=self.model(kind,seed);state=self.torch.load(io.BytesIO(raw),map_location='cpu',weights_only=True);model.load_state_dict(state,strict=True);self.active={'model':model};return self.active
    def release(self,handle):handle.clear();self.active=None;gc.collect();self.torch.cuda.empty_cache()
    def failure_state(self):
        if not self.active:return None
        b=io.BytesIO();self.torch.save({k:v.detach().cpu() for k,v in self.active['model'].state_dict().items()},b);return b.getvalue()
    def probabilities(self,handle,kind,seed,i,mentions=None):
        saved=self.cache(i);source=self.text(i)
        if kind=='span':
            candidates,values=self.core.span_probabilities(handle['model'],saved,self.device);obj={'candidates':[list(c) for c in candidates],'probabilities':values.tolist(),'columns':9,'complete_rows':len(candidates)}
        else:
            require(type(mentions) is list,'same-seed detector mentions');values,diagnostics=self.core.pair_probabilities(handle['model'],saved,mentions,seed=seed,device=self.device,text_length=len(source.decode('utf-8')));obj={'mentions':copy.deepcopy(mentions),'probabilities':values.tolist(),'columns':5,'complete_rows':len(mentions)**2,'diagnostics':diagnostics}
        return dict(obj,source_id=i,source_sha256=sha(source),source_only=True,Gold_consulted=False,candidate_cap=None,inference_row_block_size=512)
    def detector(self,obj,threshold):return self.core.predicted_mentions(obj['candidates'],obj['probabilities'],threshold)
    def records(self,i,mentions,probabilities,rt,et):
        n=len(mentions);require(type(probabilities) is list and len(probabilities)==n*n and all(type(r) is list and len(r)==5 and all(type(x) in (int,float) and math.isfinite(x) and 0<=x<=1 for x in r) for r in probabilities),'full strict N²×5 before reshape')
        matrix=self.numpy.asarray(probabilities,dtype=float).reshape((n*n,5));records,diag=self.projection.emit_native_prediction_records(self.text(i).decode('utf-8'),mentions,matrix,relation_threshold=rt,role_threshold=et);return records,diag
    def detector_records(self,i,mentions):
        text=self.text(i).decode('utf-8');return [{'family':'T','id':f'T{j+1}','type':m['type'],'segments':copy.deepcopy(m['segments']),'text':' '.join(text[a:b] for a,b in m['segments'])} for j,m in enumerate(mentions)]

def checkpoint_name(kind,seed,epoch):return f'checkpoints/{kind}/{seed}/{epoch:02}.pt'
def checkpoint_meta(kind,seed,epoch):return f'checkpoint_metadata/{kind}/{seed}/{epoch:02}.json'
def trajectory_name(kind,seed,epoch):return f'trajectories/{kind}/{seed}/{epoch:02}.jsonl'
def probability_name(kind,seed,epoch,i):return f'probabilities/{kind}/{seed}/{epoch:02}/{i:04}.json'
def test_name(kind,seed,i):return f'test_sources/{kind}/{seed}/{i:04}.json'
def expected_phase_files(phase):
    if phase in ('fit_ner','fit_heads'):
        kinds=('span',) if phase=='fit_ner' else HEADS
        return {name(k,s,e) for k in kinds for s in SEEDS for e in range(1,11) for name in (checkpoint_name,checkpoint_meta,trajectory_name)}
    if phase in ('ner_dev','pair_dev'):
        kinds=('span',) if phase=='ner_dev' else HEADS
        return {probability_name(k,s,e,i) for k in kinds for s in SEEDS for e in range(1,11) for i in SPLITS['dev']}
    if phase=='test_sources':return {test_name(k,s,i) for k in KINDS for s in SEEDS for i in SPLITS['test']}|{'before_test_source_freeze.json'}
    raise CloudIntegrityError('unknown phase')

class PhaseStore:
    def __init__(self,files,prefix,phase):
        relative(prefix);require(prefix.startswith('cloud_outputs/'),'distinct cloud_outputs run directory');self.files=files;self.prefix=prefix;self.phase=phase;self.base=prefix+'/'+phase;self.outputs={}
        p=files.path(self.base);require(not p.exists(),'no phase overwrite/resume, including failed phase');p.mkdir(parents=True)
    def write(self,name,raw):
        relative(name);require(name not in self.outputs,'no output overwrite');b=self.files.new(self.base+'/'+name,raw);self.outputs[name]=b;return b
    def object(self,name,obj):return self.write(name,encoded(obj))
    def finish(self,metadata):
        require(set(self.outputs)==expected_phase_files(self.phase),'complete exact phase output population')
        for name,b in self.outputs.items():self.files.verify(self.base+'/'+name,b)
        receipt={'version':1,'status':'completed_cloud_source_phase_not_Gold_score','phase':self.phase,'output_prefix':self.base,'files':self.outputs,**metadata}
        raw=encoded(receipt);b=self.files.new(self.base+'/receipt.json',raw);return {'path':self.base+'/receipt.json',**b}
    def failed(self,error,partial=None):
        if partial is not None:self.files.new(self.base+'/failed_partial_model.pt',partial)
        self.files.new(self.base+'/FAILED.json',encoded({'status':'failed_no_phase_completion_no_resume','type':type(error).__name__,'error':str(error),'successful_outputs_before_failure':self.outputs,'failed_partial_is_not_epoch_checkpoint':partial is not None}))

def strict_mentions(mentions,descriptor):
    require(type(mentions) is list,'mention list')
    eligible={(c[2],c[3]) for c in descriptor['candidates']};seen=set()
    for m in mentions:
        closed(m,('type','segments'));require(type(m['type']) is str and m['type'] in ('Characterization','Material','Property','Element','Process','Value','SC','Main','Doping'),'native detector type')
        seg=m['segments'];require(type(seg) is list and len(seg)==1 and type(seg[0]) is list and len(seg[0])==2 and all(type(x) is int for x in seg[0]) and tuple(seg[0]) in eligible,'strict source-defined contiguous detector candidate')
        identity=m['type'],tuple(seg[0]);require(identity not in seen,'duplicate detector mention identity');seen.add(identity)
    return mentions

class CloudStages:
    def __init__(self,files,manifest,gate,backend,run_prefix,metadata_recheck):
        self.files,self.manifest,self.gate,self.backend,self.run_prefix=files,manifest,gate,backend,run_prefix;self.prior={};self.prior_bindings=gate['prior_receipts'];self.bridge=None;self.bridge_raw=None;self.metadata_recheck=metadata_recheck;self.failed_partial=None;self.failure_state_error=None
        required={'fit_ner':set(),'ner_dev':{'fit_ner'},'fit_heads':{'fit_ner','ner_dev'},'pair_dev':{'fit_ner','ner_dev','fit_heads'},'test_sources':{'fit_ner','ner_dev','fit_heads','pair_dev'}}[gate['phase']]
        require(set(self.prior_bindings)==required,'exact predecessor stages')
        for phase,descriptor in self.prior_bindings.items():
            closed(descriptor,('path','sha256','size_bytes'));relative(descriptor['path']);require(descriptor['path']==run_prefix+'/'+phase+'/receipt.json','same run predecessor receipt path');b={k:descriptor[k] for k in ('sha256','size_bytes')};r=unique(files.read(descriptor['path'],b))
            require(r.get('status')=='completed_cloud_source_phase_not_Gold_score' and r.get('phase')==phase and r.get('manifest_sha256')==sha(encoded(manifest)),'prior success/manifest identity')
            require(r.get('output_prefix')==run_prefix+'/'+phase and type(r.get('files')) is dict and set(r['files'])==expected_phase_files(phase),'complete prior population')
            for name,wanted in r['files'].items():relative(name);files.verify(r['output_prefix']+'/'+name,wanted)
            self.prior[phase]=r
        if gate['phase'] in ('fit_heads','pair_dev','test_sources'):
            require(gate['bridge'] is not None,'locally selected bridge required');self.bridge_raw=self._external(gate['bridge']);self.bridge=unique(self.bridge_raw)
        else:require(gate['bridge'] is None,'no early selection bridge')
    def _external(self,b):
        closed(b,('path','sha256','size_bytes'));relative(b['path']);require(b['path'].startswith('cloud_bridges/') and b['path'].endswith('.json'),'source-only local bridge path');return self.files.read(b['path'],{k:b[k] for k in ('sha256','size_bytes')})
    def old(self,phase,name):return self.files.read(self.prior[phase]['output_prefix']+'/'+name,self.prior[phase]['files'][name])
    def cp(self,kind,seed,epoch):
        phase='fit_ner' if kind=='span' else 'fit_heads';raw=self.old(phase,checkpoint_name(kind,seed,epoch));meta=unique(self.old(phase,checkpoint_meta(kind,seed,epoch)))
        require(meta.get('kind')==kind and type(meta.get('seed')) is int and meta['seed']==seed and type(meta.get('epoch')) is int and meta['epoch']==epoch and meta.get('checkpoint')==binding(raw),'checkpoint metadata identity')
        require(meta.get('metadata',{}).get('complete_model_state_dict') is True and meta['metadata'].get('optimizer_moments_in_epoch_checkpoint') is False and meta['metadata'].get('implicit_resume') is False,'complete weights-only CP; never resume');return raw
    def source_record(self,obj,kind,seed,epoch,i,checkpoint,mentions=None):
        base={'source_id','source_sha256','source_only','Gold_consulted','candidate_cap','inference_row_block_size','probabilities','columns','complete_rows'}
        expected=base|({'candidates'} if kind=='span' else {'mentions','diagnostics'})|{'kind','seed','epoch','checkpoint'}
        require(type(obj) is dict and set(obj)<=expected and base<=set(obj),'source fields; no hidden Gold')
        require(type(obj['source_id']) is int and obj['source_id']==i and obj['source_sha256']==self.manifest['inputs'][text_path(i)]['sha256'] and obj['source_only'] is True and obj['Gold_consulted'] is False and obj['candidate_cap'] is None and type(obj['inference_row_block_size']) is int and obj['inference_row_block_size']==512,'original source identity/full population/no Gold')
        if i not in self.backend.descriptors:self.backend.cache(i)
        desc=self.backend.descriptors[i];rows=obj['probabilities'];cols=9 if kind=='span' else 5
        require(type(rows) is list and type(obj['columns']) is int and obj['columns']==cols and type(obj['complete_rows']) is int and obj['complete_rows']==len(rows),'strict row population')
        require(all(type(r) is list and len(r)==cols and all(type(x) in (int,float) and math.isfinite(x) and 0<=x<=1 for x in r) for r in rows),'finite complete probabilities')
        if kind=='span':
            values=obj.get('candidates');require(type(values) is list and all(type(c) is list and len(c)==4 and all(type(v) is int for v in c) for c in values) and values==desc['candidates'] and len(rows)==len(desc['candidates']),'strict native complete source span candidates/order')
        else:
            strict_mentions(mentions,desc);strict_mentions(obj.get('mentions'),desc)
            require(obj['mentions']==mentions and len(rows)==len(mentions)**2,'same-seed locked mentions/full N²')
            require(obj.get('diagnostics')==self.backend.donor_diagnostics(mentions,seed,i),'complete donor/source fingerprint diagnostics')
        for key,value in (('kind',kind),('seed',seed),('epoch',epoch),('checkpoint',checkpoint)):
            if key in obj:require(type(obj[key]) is type(value) and obj[key]==value,'source metadata changed')
        return dict(obj,kind=kind,seed=seed,epoch=epoch,checkpoint=checkpoint)
    def validate_ner_bridge(self,value,raw):
        closed(value,('version','kind','population_receipt_sha256','selector_source_sha256','local_grid_proof','choice','locked_mentions'))
        require(type(value['version']) is int and value['version']==1 and value['kind']=='local_ner_source_bridge' and value['population_receipt_sha256']==self.prior_bindings['ner_dev']['sha256'] and value['selector_source_sha256']==self.manifest['science']['src/sccomics_development_selection.py']['sha256'],'exact NER source/selector bridge')
        proof=value['local_grid_proof'];closed(proof,('complete_setting_count','seed_ids','development_source_ids','Gold_kept_local','local_checked_selector_receipt_sha256'))
        require(type(proof['complete_setting_count']) is int and proof['complete_setting_count']==80 and proof['seed_ids']==list(SEEDS) and proof['development_source_ids']==list(SPLITS['dev']) and proof['Gold_kept_local'] is True and re.fullmatch('[0-9a-f]{64}',proof['local_checked_selector_receipt_sha256']) is not None,'local complete80-grid receipt; no cloud Gold math claim')
        require(proof['local_checked_selector_receipt_sha256']==self.gate['local_selector_receipt_sha256']['NER'],'root-bound local detector-grid receipt')
        c=value['choice'];closed(c,('epoch','threshold'));require(type(c['epoch']) is int and 1<=c['epoch']<=10 and type(c['threshold']) in (int,float) and c['threshold'] in SPAN_THRESHOLDS,'fixed NER grid choice')
        locked=value['locked_mentions'];require(type(locked) is dict and set(locked)=={str(s) for s in SEEDS},'all same-seed dev graphs')
        for s in SEEDS:
            require(type(locked[str(s)]) is dict and set(locked[str(s)])=={str(i) for i in SPLITS['dev']},'all100 locked development sources')
            cp=binding(self.cp('span',s,c['epoch']))
            for i in SPLITS['dev']:
                obj=unique(self.old('ner_dev',probability_name('span',s,c['epoch'],i)));self.source_record(obj,'span',s,c['epoch'],i,cp)
                expected=self.backend.detector(obj,c['threshold']);strict_mentions(locked[str(s)][str(i)],self.backend.descriptors[i]);require(locked[str(s)][str(i)]==expected,'locked graph reconstructed from exact source probabilities, no nearest material or fabricated links')
        return value
    def ner_bridge(self):
        if self.gate['phase']=='test_sources':
            descriptor=self.bridge['ner_bridge'];raw=self._external(descriptor);value=unique(raw)
            require(descriptor['sha256']==self.prior['fit_heads']['selection_bridge_sha256'],'same NER bridge used before all heads')
        else:raw=self.bridge_raw;value=self.bridge
        value=self.validate_ner_bridge(value,raw)
        if self.gate['phase']=='pair_dev':require(sha(raw)==self.prior['fit_heads']['selection_bridge_sha256'],'fixed NER bridge across phases')
        return value,raw
    def test_choices(self,ner_raw):
        b=self.bridge;closed(b,('version','kind','ner_bridge','population_receipt_sha256','selector_source_sha256','choices','local_grid_proofs'))
        require(type(b['version']) is int and b['version']==1 and b['kind']=='local_all_pair_choices_source_bridge' and b['population_receipt_sha256']==self.prior_bindings['pair_dev']['sha256'] and b['selector_source_sha256']==self.manifest['science']['src/sccomics_development_selection.py']['sha256'],'fixed pair population/selector')
        require(b['ner_bridge']['sha256']==sha(ner_raw) and set(b['choices'])==set(HEADS) and set(b['local_grid_proofs'])==set(HEADS),'all four local choices')
        for k in HEADS:
            c=b['choices'][k];closed(c,('epoch','relation_threshold','role_threshold'));require(type(c['epoch']) is int and 1<=c['epoch']<=10 and all(type(c[x]) in (int,float) and c[x] in PAIR_THRESHOLDS for x in ('relation_threshold','role_threshold')),'fixed complete250-grid choice')
            proof=b['local_grid_proofs'][k];closed(proof,('complete_setting_count','seed_ids','development_source_ids','Gold_kept_local','local_checked_selector_receipt_sha256'))
            require(type(proof['complete_setting_count']) is int and proof['complete_setting_count']==250 and proof['seed_ids']==list(SEEDS) and proof['development_source_ids']==list(SPLITS['dev']) and proof['Gold_kept_local'] is True and re.fullmatch('[0-9a-f]{64}',proof['local_checked_selector_receipt_sha256']) is not None,'all four local complete-grid receipts')
            require(proof['local_checked_selector_receipt_sha256']==self.gate['local_selector_receipt_sha256'][k],'root-bound local250-grid receipt')
        return b['choices']
    def capture_failure(self):
        try:self.failed_partial=self.backend.failure_state()
        except BaseException as error:self.failure_state_error={'type':type(error).__name__,'error':str(error)}
    def fit(self,store,kinds):
        for kind in kinds:
            for seed in SEEDS:
                handle=self.backend.fit_start(kind,seed)
                try:
                    for epoch in range(1,11):
                        events=[];raw,metadata=self.backend.fit_epoch(handle,kind,seed,epoch,events.append)
                        b=store.write(checkpoint_name(kind,seed,epoch),raw);store.object(checkpoint_meta(kind,seed,epoch),{'kind':kind,'seed':seed,'epoch':epoch,'checkpoint':b,'metadata':metadata})
                        store.write(trajectory_name(kind,seed,epoch),b''.join(encoded(e).replace(b'\n',b'')+b'\n' for e in events))
                except BaseException:
                    self.capture_failure();raise
                finally:self.backend.release(handle)
    def development(self,store,kinds,ner=None):
        for kind in kinds:
            for seed in SEEDS:
                for epoch in range(1,11):
                    raw=self.cp(kind,seed,epoch);handle=self.backend.inference_start(kind,seed,raw)
                    try:
                        for i in SPLITS['dev']:
                            mentions=None if kind=='span' else ner['locked_mentions'][str(seed)][str(i)]
                            obj=self.backend.probabilities(handle,kind,seed,i,mentions);obj=self.source_record(obj,kind,seed,epoch,i,binding(raw),mentions);store.object(probability_name(kind,seed,epoch,i),obj)
                    finally:self.backend.release(handle)
    def fresh_development(self,ner):
        # Exact full content is revalidated before test source generation. No
        # local Gold scorer or grid recomputation happens on the cloud.
        for kind in KINDS:
            phase='ner_dev' if kind=='span' else 'pair_dev'
            for seed in SEEDS:
                for epoch in range(1,11):
                    cp=binding(self.cp(kind,seed,epoch))
                    for i in SPLITS['dev']:
                        mentions=None if kind=='span' else ner['locked_mentions'][str(seed)][str(i)]
                        obj=unique(self.old(phase,probability_name(kind,seed,epoch,i)));self.source_record(obj,kind,seed,epoch,i,cp,mentions)
    def test(self,store,ner,ner_raw):
        choices=self.test_choices(ner_raw);self.fresh_development(ner)
        cp_population={checkpoint_name(k,s,e):binding(self.cp(k,s,e)) for k in KINDS for s in SEEDS for e in range(1,11)}
        require(len(cp_population)==150,'all150 complete CP before any test source output')
        freeze={'status':'all150CP_all15000dev_source_content_and_local_choices_locked_before_test_source_generation','checkpoint_bindings':cp_population,'prior_receipts':self.prior_bindings,'NER_bridge_sha256':sha(ner_raw),'pair_bridge_sha256':sha(self.bridge_raw),'NER_choice':ner['choice'],'pair_choices':choices,'held_Gold_present_or_opened_on_cloud':False,'local_grid_math_certified_here':False}
        store.object('before_test_source_freeze.json',freeze);locked_test={}
        for seed in SEEDS:
            c=ner['choice'];raw=self.cp('span',seed,c['epoch']);handle=self.backend.inference_start('span',seed,raw)
            try:
                for i in SPLITS['test']:
                    obj=self.backend.probabilities(handle,'span',seed,i);obj=self.source_record(obj,'span',seed,c['epoch'],i,binding(raw));mentions=self.backend.detector(obj,c['threshold']);locked_test[(seed,i)]=mentions
                    store.object(test_name('span',seed,i),{'source_probability':obj,'original_source_records':self.backend.detector_records(i,mentions),'source_only':True,'test_Gold_consulted':False,'NER_choice':c})
            finally:self.backend.release(handle)
        for kind in HEADS:
            c=choices[kind]
            for seed in SEEDS:
                raw=self.cp(kind,seed,c['epoch']);handle=self.backend.inference_start(kind,seed,raw)
                try:
                    for i in SPLITS['test']:
                        mentions=locked_test[(seed,i)];obj=self.backend.probabilities(handle,kind,seed,i,mentions);obj=self.source_record(obj,kind,seed,c['epoch'],i,binding(raw),mentions);records,diag=self.backend.records(i,mentions,obj['probabilities'],c['relation_threshold'],c['role_threshold'])
                        store.object(test_name(kind,seed,i),{'source_probability':obj,'original_source_records':records,'emission_diagnostics':diag,'source_only':True,'test_Gold_consulted':False,'NER_choice':ner['choice'],'pair_choice':c})
                finally:self.backend.release(handle)
    def run(self):
        phase=self.gate['phase'];store=PhaseStore(self.files,self.run_prefix,phase)
        try:
            ner=None;ner_raw=None
            if phase in ('fit_heads','pair_dev','test_sources'):ner,ner_raw=self.ner_bridge()
            if phase=='fit_ner':self.fit(store,('span',))
            elif phase=='ner_dev':self.development(store,('span',))
            elif phase=='fit_heads':self.fit(store,HEADS)
            elif phase=='pair_dev':self.development(store,HEADS,ner)
            elif phase=='test_sources':self.test(store,ner,ner_raw)
            else:raise CloudIntegrityError('unimplemented phase')
            # Final all direct deployed sources/inputs and output hashes; no
            # claim about bytes of undeclared installed runtime/system libraries.
            self.files.verify(SELF,self.manifest['runner']);self.files.verify('src/check_sccomics_cuda_operators_v3.py',self.manifest['operator_helper'])
            for group in ('science','inputs','controls','licenses'):
                for name,wanted in self.manifest[group].items():self.files.verify(name,wanted)
            for phase,r in self.prior.items():
                d=self.prior_bindings[phase];self.files.verify(d['path'],{k:d[k] for k in ('sha256','size_bytes')})
                for name,wanted in r['files'].items():self.files.verify(r['output_prefix']+'/'+name,wanted)
            if self.gate['bridge'] is not None:
                d=self.gate['bridge'];self.files.verify(d['path'],{k:d[k] for k in ('sha256','size_bytes')})
                if self.gate['phase']=='test_sources':
                    d=self.bridge['ner_bridge'];self.files.verify(d['path'],{k:d[k] for k in ('sha256','size_bytes')})
            self.backend.verify_private_cold();self.metadata_recheck()
            return store.finish({'manifest_sha256':sha(encoded(self.manifest)),'prior_receipts':self.prior_bindings,'selection_bridge_sha256':sha(self.bridge_raw) if self.bridge_raw else None,'runtime':self.backend.runtime,'opened_TRAIN_annotation_events':self.backend.opened_annotations,'DEV_TEST_annotation_permission':False,'not_a_local_Gold_score_or_statistics_certificate':True})
        except BaseException as error:
            if self.failed_partial is None:self.capture_failure()
            if self.failure_state_error is not None:self.files.new(store.base+'/failure_state_capture_error.json',encoded(self.failure_state_error))
            store.failed(error,self.failed_partial);raise

@contextlib.contextmanager
def deployed_backend(files,manifest):
    # Startup source/data byte verification is authorized only after the new
    # actual gate; it is byte identity, not annotation semantic parsing.
    for name,wanted in manifest['inputs'].items():files.verify(name,wanted)
    captures={n:files.read('src/'+n+'.py',manifest['science']['src/'+n+'.py']) for n in SCIENCE}
    existing=os.environ.get('CUBLAS_WORKSPACE_CONFIG');require(existing is None or existing==':4096:8','cublas policy');os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8';os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1'
    with tempfile.TemporaryDirectory(prefix='SC_CUDA_PRIVATE_COLD_') as temporary,CapturedScience(captures) as science:
        backend=CudaBackend(files,manifest,science,pathlib.Path(temporary).resolve())
        yield backend

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',required=True);p.add_argument('--manifest',required=True);p.add_argument('--manifest-sha256',required=True);p.add_argument('--gate',required=True);p.add_argument('--gate-sha256',required=True);p.add_argument('--run-prefix',required=True);p.add_argument('--phase',choices=PHASES,required=True);p.add_argument('--execute-frozen-cloud-stage',action='store_true');a=p.parse_args()
    require(a.execute_frozen_cloud_stage,'default no actual fit/NN/data/cold/cache access; no final gate exists yet')
    files=Files(a.root);relative(a.manifest);relative(a.gate)
    require(a.manifest.startswith('research/round4_sccomics_cloud_cuda_') and a.manifest.endswith('.json') and a.gate.startswith('research/round4_sccomics_cloud_cuda_') and a.gate.endswith('.json'),'cloud manifest/gate metadata paths only')
    # Required hashes bind parsed raw metadata; argument sizes are acquired by
    # metadata only, then regular captured bytes and SHA are checked.
    def metadata(name,wanted):
        require(re.fullmatch('[0-9a-f]{64}',wanted) is not None,'explicit control SHA');size=files.path(name).stat(follow_symlinks=False).st_size;return files.read(name,{'sha256':wanted,'size_bytes':size})
    manifest_raw=metadata(a.manifest,a.manifest_sha256);gate_raw=metadata(a.gate,a.gate_sha256);manifest,gate=authorize(files,manifest_raw,gate_raw,a.phase)
    require(manifest_raw==encoded(manifest),'canonical fixed manifest bytes required')
    def final_metadata():
        metadata(a.manifest,a.manifest_sha256);metadata(a.gate,a.gate_sha256);r=gate['source_review'];files.verify(r['path'],{k:r[k] for k in ('sha256','size_bytes')})
        h='research/round4_sccomics_cloud_cuda_hardware_amendment.json';validate_hardware(files,unique(files.read(h,manifest['controls'][h])),manifest)
    require(a.run_prefix==gate['run_prefix'],'exact frozen cloud run prefix');require(not files.path(a.run_prefix+'/'+a.phase).exists(),'phase already claimed/failed; no resource-loading retry')
    try:
        with deployed_backend(files,manifest) as backend:receipt=CloudStages(files,manifest,gate,backend,a.run_prefix,final_metadata).run()
    except BaseException as error:
        if not files.path(a.run_prefix+'/'+a.phase).exists():
            files.new(a.run_prefix+'/'+a.phase+'/STARTUP_FAILED.json',encoded({'status':'failed_before_stage_no_completion_no_resume','type':type(error).__name__,'error':str(error)}))
        raise
    print(json.dumps({'phase':a.phase,'receipt':receipt,'scope':'cloud_source_outputs_no_held_Gold_score'}));return 0
if __name__=='__main__':raise SystemExit(main())
