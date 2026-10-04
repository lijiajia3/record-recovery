"""Separate whole-swap/Fraction replay for supplied API twelve-family counts.

No primary/old statistics, scorer, API, model, file or corpus loader is imported.
Fixed NumPy PCG64 streams are intentionally shared, not new random samples.
Different arithmetic implementation; same implementer/model assistance, with a
different-agent SOURCE review still needed. No actual cohort/barrier certificate.
"""
from __future__ import annotations

import copy
from fractions import Fraction
import hashlib
import math
import json
import platform

import numpy as np

MODELS = ('deepseek-ai/DeepSeek-V4-Pro', 'zai-org/GLM-5.3')
ARMS = ('B', 'G', 'T', 'S')
REPEATS = (0, 1, 2)
ENDPOINTS = ('R_trigger_projected_complete', 'E_global_sharing_consistent_complete')
CONTRASTS = tuple((f'SC_API:{model}:{endpoint}:T_minus_{other}', model, endpoint, other)
                  for model in MODELS for endpoint in ENDPOINTS for other in ('B', 'G', 'S'))
IDS = tuple(row[0] for row in CONTRASTS)
OLD_IDS = ('polyie:typed_minus_mean', 'polyie:typed_minus_capacity_mean',
           'mulms:typed_minus_mean', 'mulms:typed_minus_capacity_mean')
OLD_P = (Fraction(1,8192), Fraction(1,8192), Fraction(1,64), Fraction(1,64))
OLD_SHA = 'c89995e473f182f7d767a0c526088541242944d2097b12807183ab967ca5bfa0'
MC, BOOT = 100000, 10000
CI_ABSOLUTE_TOLERANCE = 1e-14


class APIStatisticsReplayError(ValueError):
    pass


def need(ok, message):
    if not ok:
        raise APIStatisticsReplayError(message)


def integer(value, *, positive=False):
    need(type(value) is int and value >= (1 if positive else 0), 'original nonnegative Python integer required')
    return value


def vector(value):
    need(type(value) in (list, tuple) and len(value) == 3, 'count triple required')
    return tuple(integer(v) for v in value)


def f1(value):
    t, fp, fn = vector(value)
    return Fraction(2*t,2*t+fp+fn) if 2*t+fp+fn else Fraction(0)


def total(rows):
    materialized = tuple(rows)
    return tuple(sum(row[j] for row in materialized) for j in range(3))


def canonical(obj):
    return json.dumps(obj,sort_keys=True,separators=(',',':'),ensure_ascii=True,allow_nan=False).encode('utf8')


def same(a,b):
    if type(a) is not type(b):
        return False
    if type(a) is dict:
        return set(a)==set(b) and all(same(a[k],b[k]) for k in a)
    if type(a) in (list,tuple):
        return len(a)==len(b) and all(same(x,y) for x,y in zip(a,b))
    return a==b


def _capture(value):
    """Only primitive JSON-like trees; reject cycles/custom callbacks before copy."""
    def inspect(obj,active):
        kind=type(obj)
        need(kind in (dict,list,tuple,str,int,float,bool,type(None)), 'primitive supplied report only')
        if kind is float:
            need(math.isfinite(obj),'finite supplied float')
        if kind in (dict,list,tuple):
            need(id(obj) not in active,'cyclic supplied report is not JSON')
            if kind is dict:
                need(all(type(k)is str for k in obj),'string JSON keys')
            active.add(id(obj))
            for child in (obj.values() if kind is dict else obj):
                inspect(child,active)
            active.remove(id(obj))
    inspect(value,set())
    return copy.deepcopy(value)


def fraction_record(row,expected):
    need(type(row)is dict and set(row)=={'numerator','denominator','decimal'},'closed rational record')
    need(type(row['numerator'])is int and type(row['denominator'])is int and row['denominator']>0 and
         (row['numerator'],row['denominator'])==(expected.numerator,expected.denominator),
         'exact normalized rational numerator/denominator differs')
    need(type(row['decimal'])is float and math.isfinite(row['decimal']) and row['decimal']==float(expected),
         'rational display/type differs')


def _metric(row,value):
    t,fp,fn=vector(value)
    need(type(row)is dict and set(row)=={'counts','precision','recall','f1'} and same(row['counts'],[t,fp,fn]),
         'complete metric/count identity differs')
    fraction_record(row['precision'],Fraction(t,t+fp) if t+fp else Fraction(0))
    fraction_record(row['recall'],Fraction(t,t+fn) if t+fn else Fraction(0))
    fraction_record(row['f1'],f1((t,fp,fn)))


def _ordered_ids(value):
    need(type(value)in(list,tuple) and len(value)>0 and all(type(v)is str and v and not any(c in v for c in '\r\n\t')for v in value),
         'explicit lexical ordered source cluster inventory required')
    need(len(set(value))==len(value),'duplicate source cluster ID')
    return tuple(value)


def validate_counts(clusters,ordered_ids,historical):
    ids=_ordered_ids(ordered_ids)
    need(type(clusters)in(list,tuple) and len(clusters)==len(ids),'complete count population')
    base={(model,arm,endpoint):[] for model in MODELS for arm in ARMS for endpoint in ENDPOINTS}
    repetitions={(model,arm,endpoint,repeat):[0,0,0] for model in MODELS for arm in ARMS for endpoint in ENDPOINTS for repeat in REPEATS}
    captured=[]
    for cluster,identifier in zip(clusters,ids):
        need(type(cluster)is dict and set(cluster)=={'source_cluster_id','gold_support_by_endpoint','models'} and
             type(cluster['source_cluster_id'])is str and cluster['source_cluster_id']==identifier,'cluster schema/order')
        support=cluster['gold_support_by_endpoint']
        need(type(support)is dict and set(support)==set(ENDPOINTS),'both Gold supports')
        gold={endpoint:integer(support[endpoint])for endpoint in ENDPOINTS}
        need(type(cluster['models'])is dict and set(cluster['models'])==set(MODELS),'exact two models')
        c={'source_cluster_id':identifier,'gold_support_by_endpoint':gold,'models':{}}
        for model in MODELS:
            arms=cluster['models'][model]
            need(type(arms)is dict and set(arms)==set(ARMS),'all four arms')
            c['models'][model]={}
            for arm in ARMS:
                endpoints=arms[arm]
                need(type(endpoints)is dict and set(endpoints)==set(ENDPOINTS),'both endpoints every arm')
                c['models'][model][arm]={}
                for endpoint in ENDPOINTS:
                    rows=endpoints[endpoint]
                    need(type(rows)in(list,tuple) and len(rows)==3,'all three original repeats')
                    kept=[]
                    triples=[]
                    for row,repeat in zip(rows,REPEATS):
                        need(type(row)is dict and set(row)=={'repeat','tp','fp','fn'} and
                             type(row['repeat'])is int and row['repeat']==repeat,'repeat identity/type/order')
                        triple=vector((row['tp'],row['fp'],row['fn']))
                        need(triple[0]+triple[2]==gold[endpoint],'shared original Gold denominator')
                        triples.append(triple)
                        kept.append(dict(repeat=repeat,tp=triple[0],fp=triple[1],fn=triple[2]))
                        for j in range(3):repetitions[model,arm,endpoint,repeat][j]+=triple[j]
                    base[model,arm,endpoint].append(total(triples))
                    c['models'][model][arm][endpoint]=kept
        captured.append(c)
    need(type(historical)is dict and set(historical)==set(OLD_IDS),'all original historical four')
    old={}
    for name,value in zip(OLD_IDS,OLD_P):
        row=historical[name]
        need(type(row)is dict and set(row)=={'numerator','denominator','source_sha256'} and
             type(row['numerator'])is int and type(row['denominator'])is int and
             (row['numerator'],row['denominator'])==(value.numerator,value.denominator) and
             type(row['source_sha256'])is str and row['source_sha256']==OLD_SHA,'unchanged historical raw identity')
        old[name]=value
    return ids,captured,base,repetitions,old


def _stream(payload):
    raw=canonical(payload);sha=hashlib.sha256(raw).digest();seed=int.from_bytes(sha,'big')
    return np.random.Generator(np.random.PCG64(seed)),{'payload':payload,'sorted_compact_json_utf8':raw.decode('utf8'),
        'payload_sha256':sha.hex(),'digest_to_seed':'256-bit unsigned big-endian integer','bit_generator':'PCG64',
        'numpy_version':np.__version__,'seed_integer':seed}


def whole_swap(a,b,ids,name):
    """Rebuild exchanged whole triples and compare Fraction magnitudes directly."""
    need(name in IDS and len(a)==len(b)==len(ids),'complete paired source population')
    for x,y in zip(a,b):
        vector(x);vector(y);need(x[0]+x[2]==y[0]+y[2],'paired native Gold support')
    left,right=total(a),total(b)
    signed=f1(left)-f1(right);observed=abs(signed)
    fixed=total(x for x,y in zip(a,b)if x==y)
    changing=tuple(i for i,(x,y)in enumerate(zip(a,b))if x!=y)
    if len(ids)<=20:
        assignments=1<<len(ids);width=(len(ids)+7)//8;digest=hashlib.sha256()
        masks=range(assignments)
        def bits(mask):
            digest.update(mask.to_bytes(width,'little'))
            return tuple(bool(mask & (1<<i))for i in changing)
        add=0;denominator=assignments
    else:
        generator,metadata=_stream({'policy':'sccomics_API_cluster_swap_v1','seed':20261031,
                                   'ordered_cluster_ids':list(ids),'contrast_id':name})
        masks=generator.integers(0,2,size=(MC,len(ids)),dtype=np.uint8)
        digest=hashlib.sha256(masks.tobytes(order='C'))
        def bits(mask):return tuple(bool(mask[i])for i in changing)
        assignments=MC;add=1;denominator=MC+1
    hits=0
    for mask in masks:
        swaps=bits(mask)
        x=total((fixed,*(b[i]if switch else a[i]for i,switch in zip(changing,swaps))))
        y=total((fixed,*(a[i]if switch else b[i]for i,switch in zip(changing,swaps))))
        hits+=abs(f1(x)-f1(y))>=observed
    if len(ids)<=20:
        metadata={'branch':'exhaustive_all_masks','assignments':assignments,'mask_stream_sha256':digest.hexdigest(),
                  'serialization':'ascending integer masks; packed little-endian; cluster0 is least-significant bit',
                  'includes_identity_and_complement':True,'plus_one_added':False}
    else:
        metadata.update({'shape':[MC,len(ids)],'dtype':'uint8','serialization':'C-order one byte per Bernoulli bit',
                         'mask_stream_sha256':digest.hexdigest(),'assignments_with_replacement':True,
                         'independent_fair_Bernoulli_bits':True,'branch':'fixed_conservative_Monte_Carlo',
                         'assignments':MC,'plus_one_added':True,'exact_or_unbiased_p_claim':False})
    raw_den=(2*left[0]+left[1]+left[2] or 1)*(2*right[0]+right[1]+right[2] or 1)
    unnormalized=signed*raw_den
    need(unnormalized.denominator==1,'observed unnormalized representation must be integral')
    return {'T':left,'reference':right,'signed':signed,'absolute':observed,'hits':hits,
            'raw_numerator':hits+add,'raw_denominator':denominator,'stream':metadata,
            'observed_numerator':unnormalized.numerator,'observed_denominator':raw_den}


def _holm(raw,names):
    need(set(raw)==set(names) and len(set(names))==len(names) and all(type(raw[n])is Fraction and 0<=raw[n]<=1 for n in names),
         'complete exact closed multiplicity family')
    order=sorted(range(len(names)),key=lambda i:(raw[names[i]],i))
    values={}
    for rank,index in enumerate(order):
        value=min(Fraction(1),max((len(names)-j)*raw[names[order[j]]]for j in range(rank+1)))
        values[names[index]]={'rank':rank+1,'adjusted':value,'reject':value<=Fraction(1,20)}
    return values


def _check_holm(family,raw,names):
    fields={'method','family_size','alpha','ordered_family_ids','tie_order','entries'}
    need(type(family)is dict and set(family)==fields and family['method']=='Holm_step_down_Bonferroni' and
         type(family['family_size'])is int and family['family_size']==len(names) and
         same(family['ordered_family_ids'],list(names)) and family['tie_order']=='original declared family order',
         'Holm procedure/family identity')
    fraction_record(family['alpha'],Fraction(1,20))
    expected=_holm(raw,names)
    need(type(family['entries'])is list and len(family['entries'])==len(names),'complete Holm entries')
    for row,name in zip(family['entries'],names):
        want=expected[name]
        need(type(row)is dict and set(row)=={'id','rank','raw','adjusted','reject_at_fixed_alpha'} and
             row['id']==name and type(row['rank'])is int and row['rank']==want['rank'] and
             type(row['reject_at_fixed_alpha'])is bool and row['reject_at_fixed_alpha']==want['reject'],'Holm entry identity/type')
        fraction_record(row['raw'],raw[name]);fraction_record(row['adjusted'],want['adjusted'])


def linear_percentile(values,q):
    need(type(values)is list and len(values)>0 and all(type(v)is float and math.isfinite(v)for v in values),'full finite percentile population')
    ordered=sorted(values);position=(len(ordered)-1)*q;left=math.floor(position);right=min(left+1,len(ordered)-1)
    return ordered[left]+(position-left)*(ordered[right]-ordered[left])


def _bootstrap(base,ids,report):
    generator,stream=_stream({'policy':'sccomics_API_cluster_bootstrap_v1','seed':20261032,'ordered_cluster_ids':list(ids)})
    draws=generator.integers(0,len(ids),size=(BOOT,len(ids)),dtype=np.int64)
    index_hash=hashlib.sha256(draws.astype('<u8',copy=False).tobytes(order='C')).hexdigest()
    stream.update({'shape':[BOOT,len(ids)],'sampling_with_replacement':True,'serialization':'C-order little-endian unsigned64 indices',
                   'index_stream_sha256':index_hash})
    fields={'draws','source_cluster_count','all_three_repeats_retained_with_source','common_paired_indices_all_models_arms_endpoints','stream',
            'model_arm_F1_intervals','signed_difference_intervals','replicate_pooled_count_vectors','replicate_count_vectors_sha256',
            'replicate_count_serialization','raw_exact_reconstruction','zero_denominator_f1','zero_replicates_dropped','interpretation'}
    need(type(report)is dict and set(report)==fields and type(report['draws'])is int and report['draws']==BOOT and
         type(report['source_cluster_count'])is int and report['source_cluster_count']==len(ids) and
         same(report['stream'],stream) and report['all_three_repeats_retained_with_source']is True and
         report['common_paired_indices_all_models_arms_endpoints']is True and report['zero_replicates_dropped']is False and
         type(report['zero_denominator_f1'])is int and report['zero_denominator_f1']==0 and
         report['replicate_count_serialization']=='sorted compact ASCII JSON; original Python integer counts' and
         report['raw_exact_reconstruction']=='Each replicate F1 and signed difference reconstructs from retained integer counts as Fraction; only final linear percentile quantiles use float64' and
         report['interpretation']=='Conditional source-cluster resampling of retained API outputs; not algorithm/random-seed/provider-drift/simultaneous/physical uncertainty',
         'common complete bootstrap procedure identity/type')
    rows=report['replicate_pooled_count_vectors']
    need(type(rows)is list and len(rows)==BOOT,'all10000 raw integer bootstrap rows')
    values={key:[]for key in base};deltas={name:[]for name in IDS};reconstructed=[]
    for index,draw in enumerate(draws):
        row={model:{arm:{}for arm in ARMS}for model in MODELS};ratios={}
        for key,pool in base.items():
            v=total(pool[int(i)]for i in draw)
            model,arm,endpoint=key;row[model][arm][endpoint]=list(v);ratios[key]=f1(v);values[key].append(float(ratios[key]))
        need(same(row,rows[index]),'raw bootstrap integer count/type mismatch at row '+str(index))
        reconstructed.append(row)
        for name,model,endpoint,reference in CONTRASTS:deltas[name].append(float(ratios[model,'T',endpoint]-ratios[model,reference,endpoint]))
    digest=hashlib.sha256(canonical(reconstructed)).hexdigest()
    need(type(report['replicate_count_vectors_sha256'])is str and report['replicate_count_vectors_sha256']==digest,'integer replicate-count hash differs')
    intervals=report['model_arm_F1_intervals']
    need(type(intervals)is dict and set(intervals)==set(MODELS),'both model interval populations')
    for model in MODELS:
        need(type(intervals[model])is dict and set(intervals[model])==set(ARMS),'all arm intervals')
        for arm in ARMS:need(type(intervals[model][arm])is dict and set(intervals[model][arm])==set(ENDPOINTS),'both endpoint intervals')
    need(type(report['signed_difference_intervals'])is dict and set(report['signed_difference_intervals'])==set(IDS),'all12 signed intervals')
    errors=[]
    for key,samples in list(values.items())+list(deltas.items()):
        interval=intervals[key[0]][key[1]][key[2]]if type(key)is tuple else report['signed_difference_intervals'][key]
        need(type(interval)is dict and set(interval)=={'lower','upper','confidence_level','method','pointwise_conditional_only'} and
             type(interval['confidence_level'])is float and interval['confidence_level']==.95 and
             interval['method']=='percentile_linear' and interval['pointwise_conditional_only']is True,'pointwise conditional linear CI identity')
        for field,q in (('lower',.025),('upper',.975)):
            need(type(interval[field])is float and math.isfinite(interval[field]),'finite float64 display interval')
            error=abs(interval[field]-linear_percentile(samples,q));errors.append(error)
            need(error<=CI_ABSOLUTE_TOLERANCE,'manual linear percentile differs')
    return {'raw_integer_rows_verified':BOOT,'intervals_verified':28,'maximum_absolute_quantile_error':max(errors),
            'index_stream_sha256':index_hash,'replicate_count_vectors_sha256':digest}


def replay_statistics(clusters,*,expected_cluster_ids,historical_raw,primary_statistics,
                      expected_cluster_count=100,allow_artificial_population=False):
    """Full exact procedure/arithmetic replay of c630 supplied output schema.

    Default prospective test has100 original frozen sources; caller may explicitly
    select12 for pilot. Other K need the disclosed artificial flag. Neither flag
    nor counts establish actual source/Gold/graph barriers or an API permission.
    """
    need(platform.python_version()=='3.13.7' and np.__version__=='2.2.6','fixed Python/NumPy stream runtime')
    integer(expected_cluster_count,positive=True)
    need(type(allow_artificial_population)is bool and (allow_artificial_population or expected_cluster_count in(12,100)),
         'explicit prospective100test/12pilot or artificial-count scope')
    ids,captured,base,repetitions,old=validate_counts(clusters,expected_cluster_ids,historical_raw)
    need(len(ids)==expected_cluster_count,'expected original source-cluster count (not3repeat units)')
    p=_capture(primary_statistics)
    keys={'status','policy','ordered_cluster_ids','ordered_cluster_inventory_sha256','source_cluster_count','models','arms','request_repeats','endpoints',
          'inferential_N_not_multiplied_by_repeats_or_models','input_count_objects','input_count_objects_sha256','pooled_count_vectors_by_source_cluster',
          'pooled_summaries','individual_request_repeat_summaries','primary_comparisons','holm12_primary','holm16_historically_informed_sensitivity',
          'paired_conditional_bootstrap','runtime','source_Gold_graph_population_execution_or_semantic_barriers_certified','independent_replay_claim','limitations'}
    need(type(p)is dict and set(p)==keys and p['status']=='complete_statistics_for_explicit_supplied_API_counts_only' and
         p['policy']=='sccomics_API_native_cluster_statistics_v1' and same(p['ordered_cluster_ids'],list(ids)) and
         p['ordered_cluster_inventory_sha256']==hashlib.sha256(canonical(list(ids))).hexdigest() and
         type(p['source_cluster_count'])is int and p['source_cluster_count']==len(ids) and
         same(p['models'],list(MODELS)) and same(p['arms'],list(ARMS)) and same(p['request_repeats'],list(REPEATS)) and
         same(p['endpoints'],list(ENDPOINTS)) and p['inferential_N_not_multiplied_by_repeats_or_models']is True and
         same(p['input_count_objects'],captured) and p['input_count_objects_sha256']==hashlib.sha256(canonical(captured)).hexdigest() and
         same(p['runtime'],{'python':'3.13.7','numpy':'2.2.6'}) and
         p['source_Gold_graph_population_execution_or_semantic_barriers_certified']is False and p['independent_replay_claim']is False,
         'primary whole procedure/count population/runtime identity differs')
    need(same(p['limitations'],[
        'Caller must establish frozen actual source clusters, complete graphs and original native scoring before first Gold',
        'Three-repeat pooled counts are not an ensemble or mean-repeat F1; request repetitions are not independent original sources',
        'The two API families are not independent source samples or a proof of independent pretraining',
        'Whole-cluster swap null requires joint label exchangeability, not merely equal F1 or a causal intervention',
        'Pointwise bootstrap conditions on retained outputs/inventory; does not cover algorithm randomness, provider drift or annotation truth',
        'Historical16 is informed sensitivity; no retroactive confirmatory claim or old Holm4/family change',
        'No outcome-based reseeding, draw-budget, family, branch, alpha or source exclusion is supported']), 'scientific limitation scope differs')
    for field in ('pooled_count_vectors_by_source_cluster','pooled_summaries','individual_request_repeat_summaries'):
        need(type(p[field])is dict and set(p[field])==set(MODELS),'all model summaries/pools')
        for model in MODELS:
            need(type(p[field][model])is dict and set(p[field][model])==set(ARMS),'all arm summaries/pools')
            for arm in ARMS:need(type(p[field][model][arm])is dict and set(p[field][model][arm])==set(ENDPOINTS),'both endpoint summaries/pools')
    for key,pool in base.items():
        model,arm,endpoint=key;rows=p['pooled_count_vectors_by_source_cluster'][model][arm][endpoint]
        need(type(rows)is list and len(rows)==len(ids),'complete source count pools')
        need(all(vector(a)==b for a,b in zip(rows,pool)),'three-repeat source pool counts differ')
        _metric(p['pooled_summaries'][model][arm][endpoint],total(pool))
        repeat_rows=p['individual_request_repeat_summaries'][model][arm][endpoint]
        need(type(repeat_rows)is list and len(repeat_rows)==3,'all three repeat summaries')
        for row,repeat in zip(repeat_rows,REPEATS):
            need(type(row)is dict and set(row)=={'repeat','counts','precision','recall','f1'} and type(row['repeat'])is int and row['repeat']==repeat,
                 'per-repeat summary identity/type')
            _metric({k:v for k,v in row.items()if k!='repeat'},repetitions[model,arm,endpoint,repeat])
    originals=p['primary_comparisons']
    need(type(originals)is list and len(originals)==12,'all twelve primary comparisons')
    output=[];raw={}
    comparison_keys={'contrast_id','source_cluster_count','pooled_T_counts','pooled_reference_counts','T_f1','reference_f1','signed_T_minus_reference',
                     'absolute_statistic','observed_unreduced_difference','inclusive_exact_crossproduct_ties','hits','raw_unreduced_p_fraction','raw_p','stream','null',
                     'model','endpoint','reference_arm'}
    for original,(name,model,endpoint,reference)in zip(originals,CONTRASTS):
        need(type(original)is dict and set(original)==comparison_keys and original['contrast_id']==name and original['model']==model and
             original['endpoint']==endpoint and original['reference_arm']==reference and type(original['source_cluster_count'])is int and
             original['source_cluster_count']==len(ids) and original['inclusive_exact_crossproduct_ties']is True and
             original['null']=='joint system-label exchangeability under whole-source-cluster swaps conditional on retained API outputs/inventory; equal F1 alone is insufficient',
             'complete ordered contrast identity')
        row=whole_swap(base[model,'T',endpoint],base[model,reference,endpoint],ids,name)
        need(same(original['pooled_T_counts'],list(row['T'])) and same(original['pooled_reference_counts'],list(row['reference'])) and
             type(original['hits'])is int and original['hits']==row['hits'] and same(original['stream'],row['stream']) and
             same(original['raw_unreduced_p_fraction'],{'numerator':row['raw_numerator'],'denominator':row['raw_denominator']}) and
             same(original['observed_unreduced_difference'],{'numerator':row['observed_numerator'],'denominator':row['observed_denominator']}),
             'independent whole-swap integer/procedure/stream mismatch: '+name)
        value=Fraction(row['raw_numerator'],row['raw_denominator']);raw[name]=value
        for field,expected in [('T_f1',f1(row['T'])),('reference_f1',f1(row['reference'])),('signed_T_minus_reference',row['signed']),
                               ('absolute_statistic',row['absolute']),('raw_p',value)]:fraction_record(original[field],expected)
        output.append({'id':name,'hits':row['hits'],'raw_numerator':row['raw_numerator'],'raw_denominator':row['raw_denominator'],
                       'signed':{'numerator':row['signed'].numerator,'denominator':row['signed'].denominator},'stream_sha256':row['stream']['mask_stream_sha256']})
    _check_holm(p['holm12_primary'],raw,IDS)
    history=p['holm16_historically_informed_sensitivity']
    extra={'historically_informed_sensitivity_only','retroactive_confirmatory_control_claim','historical_source_sha256',
           'historical_rerandomization_or_old_family_replacement','historical_exact_raw_records'}
    need(type(history)is dict and set(history)=={'method','family_size','alpha','ordered_family_ids','tie_order','entries'}|extra and
         history['historically_informed_sensitivity_only']is True and history['retroactive_confirmatory_control_claim']is False and
         history['historical_rerandomization_or_old_family_replacement']is False and history['historical_source_sha256']==OLD_SHA and
         type(history['historical_exact_raw_records'])is dict and set(history['historical_exact_raw_records'])==set(OLD_IDS),
         'unchanged historically informed16 sensitivity scope')
    _check_holm({k:v for k,v in history.items()if k not in extra},{**raw,**old},IDS+OLD_IDS)
    for name,value in old.items():
        record=history['historical_exact_raw_records'][name]
        need(type(record)is dict and set(record)=={'numerator','denominator','decimal','source_sha256'} and record['source_sha256']==OLD_SHA,'historical source identity')
        fraction_record({k:v for k,v in record.items()if k!='source_sha256'},value)
    bootstrap=_bootstrap(base,ids,p['paired_conditional_bootstrap'])
    return {'status':'separate_arithmetic_replay_passed_for_explicit_supplied_API_counts_only',
            'source_cluster_count':len(ids),'ordered_cluster_ids':list(ids),'population_mode':'OWN_artificial'if allow_artificial_population else 'prospective_'+str(expected_cluster_count),
            'separate_whole_swap_Fraction_manual_quantile_and_closed_Holm_code':True,
            'primary_or_old_statistics_module_imported':False,'same_fixed_PCG64_stream_deterministic_replay_not_new_random_sampling':True,
            'independent_human_or_different_implementer_review_claim':False,'actual_source_Gold_graph_barriers_certified':False,
            'contrasts':output,'holm12_and_historical16_exactly_recomputed':True,'bootstrap':bootstrap,
            'quantile_absolute_tolerance':CI_ABSOLUTE_TOLERANCE,'integer_counts_rational_tails_Holm_tolerance':0,
            'runtime':{'python':platform.python_version(),'numpy':np.__version__},
            'limitations':['Pure supplied counts/report do not establish actual100 sources, complete native Gold, API execution or truth',
                           'Same fixed library/stream is shared by protocol, not independent RNG evidence',
                           'Same model-assisted implementer wrote primary and replay; genuinely different-agent SOURCE review is pending',
                           'Conditional cluster inference and pointwise bounds retain original scientific assumptions']}
