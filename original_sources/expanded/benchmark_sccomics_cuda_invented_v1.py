"""Root-run CUDA workload timing on invented tensors only, no corpus/model files.

The random last-block proxy mirrors BERT-base dimensions and the repository's
span/pair head dimensions. It is NOT a fit, performance test, full encoder, cold
weight loading, scientific runtime adapter or authorization to run the study.
Default dry-run imports no Torch. Actual CUDA toy timing requires an explicit flag.
"""
from __future__ import annotations
import argparse,hashlib,json,math,pathlib,platform,statistics,time

WIDTH=32;HIDDEN=768;HEADS=12;INTERMEDIATE=3072;SPAN_TYPES=9;PAIR_TYPES=5;FEATURE_DIM=23
ACCUMULATION=8;DOCUMENTS=800;SEEDS=3;EPOCHS=10;TOTAL_EPOCH_CHECKPOINTS=150
ARCHITECTURES=('aligned','permuted','ordered_context','biaffine')

def require(value,message):
    if not value:raise ValueError(message)
def candidates(wordpieces):
    require(type(wordpieces) is int and wordpieces>=0,'nonnegative integer WordPiece count')
    return [(a,b) for a in range(wordpieces) for b in range(a,min(a+WIDTH,wordpieces))]
def plan(warmups=1,repeats=3,budget_seconds=120,attention='eager'):
    require(type(warmups) is int and 0<=warmups<=10,'warmups integer 0..10')
    require(type(repeats) is int and 1<=repeats<=20,'repeats integer 1..20')
    require(type(budget_seconds) in (int,float) and math.isfinite(budget_seconds) and 1<=budget_seconds<=300,'finite wall budget 1..300')
    require(attention in ('eager','sdpa'),'explicit toy attention implementation')
    training=[{'kind':'span','tokens':512,'mentions':0,'training_rows':r} for r in (64,512,1536)]
    training += [{'kind':k,'tokens':512,'mentions':n,'training_rows':min(n*n,r)} for k in ARCHITECTURES for n,r in ((8,64),(32,1024),(128,1536))]
    generation=[{'kind':'span','tokens':t,'mentions':0,'complete_rows':len(candidates(t-2))} for t in (128,512)]
    generation += [{'kind':k,'tokens':512,'mentions':n,'complete_rows':n*n} for k in ARCHITECTURES for n in (8,32,128)]
    return {'scope':'INVENTED_TENSOR_TIMING_NOT_SCIENTIFIC_FIT','dtype':'float32','attention':attention,'warmups':warmups,'repeats':repeats,'budget_seconds':budget_seconds,
        'training_cases':training,'generation_cases':generation,'accumulation_documents':ACCUMULATION,
        'chunk_semantics':'one invented chunk; tokens include two special tokens; actual long multi-chunk documents not measured',
        'training_budget_reference':{'fits':15,'epochs_per_fit':10,'checkpoint_count':150,'documents_per_epoch':800,'document_passes':120000,'optimizer_updates':15000},
        'source_generation_reference':{'NER_dev_documents':3000,'pair_dev_documents_per_architecture':3000,'NER_test_documents':300,'pair_test_documents_per_architecture':300},
        'row_semantics':'training rows are invented workload scenarios, not measured label support or a fixed scientific row cap; inference retains all defined toy rows in blocks of 512'}

def extrapolate(training,generation):
    """Deterministic arithmetic on supplied toy seconds; no fitted-study timing."""
    def selected(rows,kind,field):
        found=[r for r in rows if r['case']['kind']==kind and r['case']['tokens']==512 and (r['case']['training_rows']==512 if kind=='span' and field=='training_rows' else r['case'].get('mentions')==32 if kind!='span' else True)]
        return found[0] if len(found)==1 else None
    train={};infer={}
    for kind in ('span',*ARCHITECTURES):
        r=selected(training,kind,'training_rows')
        if r:train[kind]={'representative_case':r['case'],'toy_compute_seconds_for_3fits_10epochs_800docs':r['median_seconds']*(SEEDS*EPOCHS*(DOCUMENTS//ACCUMULATION))}
        r=selected(generation,kind,'complete_rows')
        if r:infer[kind]={'representative_case':r['case'],'toy_compute_seconds_for_3300_generation_documents':r['median_seconds']*3300}
    return {'identity':'SCENARIO_EXTRAPOLATION_NOT_ACTUAL_STUDY_MEASUREMENT','training_scenarios':train,'source_generation_scenarios':infer,
        'sum_representative_training_compute_seconds':sum(r['toy_compute_seconds_for_3fits_10epochs_800docs'] for r in train.values()) if len(train)==5 else None,
        'sum_representative_generation_compute_seconds':sum(r['toy_compute_seconds_for_3300_generation_documents'] for r in infer.values()) if len(infer)==5 else None,
        'excluded':'first-eleven-block cache construction, model/tokenizer downloads, real document lengths/multiple chunks, supervision/feature construction and actual row distributions, JSON/probability output and device transfers to host except explicit measured copies, 150 checkpoint serialization, per-document logs, dev grid search, native graph matching/statistics, cloud setup, storage/network; not an end-to-end completion guarantee'}

def build_model(torch,kind,attention):
    nn=torch.nn;F=torch.nn.functional
    class Block(nn.Module):
        def __init__(self):
            super().__init__();self.query=nn.Linear(HIDDEN,HIDDEN);self.key=nn.Linear(HIDDEN,HIDDEN);self.value=nn.Linear(HIDDEN,HIDDEN);self.attention_output=nn.Linear(HIDDEN,HIDDEN);self.attention_norm=nn.LayerNorm(HIDDEN,eps=1e-12);self.intermediate=nn.Linear(HIDDEN,INTERMEDIATE);self.output=nn.Linear(INTERMEDIATE,HIDDEN);self.output_norm=nn.LayerNorm(HIDDEN,eps=1e-12);self.dropout=nn.Dropout(.1)
        def forward(self,x):
            b,l,_=x.shape
            def heads(m):return m(x).view(b,l,HEADS,HIDDEN//HEADS).transpose(1,2)
            q,k,v=heads(self.query),heads(self.key),heads(self.value)
            if attention=='eager':a=self.dropout((q@k.transpose(-1,-2)/math.sqrt(HIDDEN//HEADS)).softmax(dim=-1))@v
            else:a=F.scaled_dot_product_attention(q,k,v,dropout_p=.1 if self.training else 0.)
            a=a.transpose(1,2).contiguous().view(b,l,HIDDEN);x=self.attention_norm(x+self.dropout(self.attention_output(a)))
            return self.output_norm(x+self.dropout(self.output(F.gelu(self.intermediate(x)))))
    class Model(nn.Module):
        def __init__(self):
            super().__init__();self.last_layer=Block();self.norm=nn.LayerNorm(HIDDEN)
            if kind=='span':
                self.width=nn.Embedding(WIDTH+1,32);self.span_head=nn.Sequential(nn.Linear(HIDDEN*3+32,128),nn.GELU(),nn.Dropout(.1),nn.Linear(128,SPAN_TYPES))
            elif kind=='biaffine':
                self.types=nn.Embedding(SPAN_TYPES+1,32);self.head=nn.Sequential(nn.Linear(800,128),nn.GELU(),nn.Dropout(.1));self.tail=nn.Sequential(nn.Linear(800,128),nn.GELU(),nn.Dropout(.1));self.biaffine=nn.Parameter(torch.empty(PAIR_TYPES,129,129));nn.init.xavier_uniform_(self.biaffine)
            else:
                self.content=nn.Sequential(nn.Linear(1536,128),nn.GELU());self.geometry=nn.Sequential(nn.Linear(FEATURE_DIM,32),nn.GELU());self.output=nn.Sequential(nn.Linear(160,64),nn.GELU(),nn.Dropout(.1),nn.Linear(64,PAIR_TYPES))
        def tokens(self,hidden):return self.last_layer(hidden)[0,1:-1]
        def rows(self,tokens,source,rows):
            if kind=='span':
                x=self.norm(tokens);sums=torch.cat([x.new_zeros((1,HIDDEN)),x.cumsum(0)]);a=source['a'][rows];b=source['b'][rows];width=b-a+1
                mean=(sums[b+1]-sums[a])/width[:,None];return self.span_head(torch.cat([x[a],x[b],mean,self.width(width)],1))
            matrix=source['matrix'];x=self.norm(matrix);h=source['h'][rows];t=source['t'][rows]
            if kind=='biaffine':
                z=torch.cat([x,self.types(source['types'])],1);hs=self.head(z);ts=self.tail(z);hs=torch.cat([hs,hs.new_ones((len(hs),1))],1);ts=torch.cat([ts,ts.new_ones((len(ts),1))],1)
                return torch.einsum('ni,kij,nj->nk',hs[h],self.biaffine,ts[t])
            features=source['features'][source['donor'][rows]] if kind=='permuted' else source['features'][rows] if kind=='aligned' else x.new_zeros((len(rows),FEATURE_DIM))
            return self.output(torch.cat([self.content(torch.cat([x[h],x[t]],1)),self.geometry(features)],1))
    return Model()

def invented_source(torch,case):
    # Entire population is synthetic, with no source IDs, text or annotations.
    device='cuda';w=case['tokens']-2;n=case['mentions'];source={'hidden_cpu':torch.randn(1,case['tokens'],HIDDEN,dtype=torch.float32)}
    if case['kind']=='span':
        values=candidates(w);source.update(a=torch.tensor([a for a,b in values],device=device),b=torch.tensor([b for a,b in values],device=device));total=len(values);classes=SPAN_TYPES
    else:
        pool=torch.zeros(n,w,dtype=torch.float32)
        for i in range(n):a=(i*7)%w;b=min(w,a+1+(i%8));pool[i,a:b]=1/(b-a)
        pairs=[(h,t) for h in range(n) for t in range(n)];total=n*n;classes=PAIR_TYPES
        source.update(pool_cpu=pool,h=torch.tensor([h for h,t in pairs],device=device),t=torch.tensor([t for h,t in pairs],device=device),types=torch.arange(n,device=device)%SPAN_TYPES,
                      features=torch.randn(total,FEATURE_DIM,device=device),donor=torch.randperm(total,device=device))
    source['total_rows']=total;source['rows']=torch.randperm(total,device=device)[:case.get('training_rows',total)];source['targets']=torch.randint(0,2,(len(source['rows']),classes),device=device).float()
    return source

def optimizer(torch,model):
    grouped={}
    for name,p in model.named_parameters():
        enc=name.startswith('last_layer.');decay=p.ndim>1 and not name.endswith('.bias');grouped.setdefault((enc,decay),[]).append(p)
    return torch.optim.AdamW([{'params':v,'lr':2e-5 if enc else 1e-3,'weight_decay':.01 if decay else 0.} for (enc,decay),v in grouped.items()],foreach=False)

def run_cuda(configuration):
    import torch  # Only the root-run explicitly authorized CUDA toy branch.
    require(torch.cuda.is_available(),'CUDA required; no CPU/MPS fallback')
    torch.manual_seed(20261003);torch.cuda.manual_seed_all(20261003)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    started=time.monotonic();training=[];generation=[];failures=[];timing_budget_reached=False
    def timed(fn):
        values=[]
        for i in range(configuration['warmups']+configuration['repeats']):
            if time.monotonic()-started>=configuration['budget_seconds']:return values,False
            torch.cuda.synchronize();begin=time.perf_counter();fn();torch.cuda.synchronize();seconds=time.perf_counter()-begin
            if i>=configuration['warmups']:values.append(seconds)
        return values,True
    for phase,cases,saved in (('training',configuration['training_cases'],training),('generation',configuration['generation_cases'],generation)):
        for case in cases:
            if time.monotonic()-started>=configuration['budget_seconds']:timing_budget_reached=True;break
            try:
                model=build_model(torch,case['kind'],configuration['attention']).cuda().float();source=invented_source(torch,case);opt=optimizer(torch,model) if phase=='training' else None
                torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()
                def document():
                    hidden=source['hidden_cpu'].to('cuda');tokens=model.tokens(hidden)
                    if case['kind']!='span':source['pool']=source['pool_cpu'].to('cuda');source['matrix']=source['pool']@tokens
                    return tokens
                if phase=='training':
                    model.train()
                    def execute():
                        opt.zero_grad(set_to_none=True)
                        for _ in range(ACCUMULATION):
                            logits=model.rows(document(),source,source['rows']);require(bool(torch.isfinite(logits).all()),'nonfinite invented logits')
                            loss=torch.nn.functional.binary_cross_entropy_with_logits(logits,source['targets']);require(bool(torch.isfinite(loss)),'nonfinite toy loss');float(loss.detach().cpu());(loss/ACCUMULATION).backward()
                        torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True);opt.step();require(all(bool(torch.isfinite(p).all()) for p in model.parameters()),'nonfinite toy parameters')
                else:
                    model.eval()
                    def execute():
                        with torch.no_grad():
                            tokens=document()
                            for at in range(0,source['total_rows'],512):
                                rows=torch.arange(at,min(source['total_rows'],at+512),device='cuda');logits=model.rows(tokens,source,rows);require(bool(torch.isfinite(logits).all()),'nonfinite toy logits');logits.sigmoid().detach().cpu()
                values,complete=timed(execute)
                saved.append({'case':case,'unit':'seconds per eight-document optimizer update' if phase=='training' else 'seconds per complete invented source document','raw_seconds':values,'median_seconds':statistics.median(values) if values else None,'mean_seconds':statistics.mean(values) if values else None,'measurement_repeats_complete':complete,'peak_allocated_bytes':torch.cuda.max_memory_allocated(),'peak_reserved_bytes':torch.cuda.max_memory_reserved(),'parameter_count':sum(p.numel() for p in model.parameters())})
                if not complete:timing_budget_reached=True;break
                del model,source,opt;torch.cuda.empty_cache()
            except Exception as e:failures.append({'phase':phase,'case':case,'type':type(e).__name__,'error':str(e)});break
        if failures or timing_budget_reached:break
    complete=len(training)==len(configuration['training_cases']) and len(generation)==len(configuration['generation_cases']) and not failures and not timing_budget_reached
    measured_train=[r for r in training if r['measurement_repeats_complete']];measured_generation=[r for r in generation if r['measurement_repeats_complete']]
    return {'identity':'ACTUAL_INVENTED_CUDA_TIMING_ONLY_NOT_SCIENTIFIC_FIT','completed_all_toy_cases':complete,'runtime':{'python':platform.python_version(),'torch':torch.__version__,'torch_cuda_build':torch.version.cuda,'gpu':torch.cuda.get_device_name(),'compute_capability':list(torch.cuda.get_device_capability()),'gpu_total_memory_bytes':torch.cuda.get_device_properties(0).total_memory,'dtype':'float32','autocast':False,'TF32':False,'attention':configuration['attention']},'configuration':configuration,'training_measurements':training,'source_generation_measurements':generation,'elapsed_seconds':time.monotonic()-started,'timing_budget_reached':timing_budget_reached,'failures':failures,'extrapolation':extrapolate(measured_train,measured_generation),'SC_text_ANN_model_cache_weights_API_or_scientific_results_used':False,
        'fidelity_limits':'random BERT-base 768/12/3072 last block, not original checkpoint or asserted identical library kernels; lower 11 blocks/cache construction excluded; source_models head dimensions preserved; one CPU->GPU hidden chunk, full toy candidate population, 512-row inference, BCE/eight-doc accumulation/AdamW foreach=False/clipping/finite checks; synthetic features prepared outside timing, actual Python/NumPy construction and real irregular segments excluded; fixed two learning rates used, not full scientific warmup/decay trajectory'}

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--execute-invented-cuda',action='store_true');p.add_argument('--output',required=True);p.add_argument('--warmups',type=int,default=1);p.add_argument('--repeats',type=int,default=3);p.add_argument('--budget-seconds',type=float,default=120);p.add_argument('--attention',choices=('eager','sdpa'),default='eager');a=p.parse_args()
    configuration=plan(a.warmups,a.repeats,a.budget_seconds,a.attention);out=pathlib.Path(a.output)
    require('..' not in out.parts and not out.exists() and not any(v.is_symlink() for v in (out,*out.parents)),'fresh nonalias output')
    result=run_cuda(configuration) if a.execute_invented_cuda else {'identity':'SOURCE_DRY_RUN_NO_TORCH_IMPORT_OR_GPU','configuration':configuration,'actual_NN_execution':False}
    result['script_sha256']=hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest();out.parent.mkdir(parents=True,exist_ok=True)
    with out.open('x') as f:json.dump(result,f,indent=2,allow_nan=False);f.write('\n')
    print(json.dumps({'identity':result['identity'],'output':str(out),'completed_all_toy_cases':result.get('completed_all_toy_cases')}));return 0 if not a.execute_invented_cuda or result['completed_all_toy_cases'] else 1
if __name__=='__main__':raise SystemExit(main())
