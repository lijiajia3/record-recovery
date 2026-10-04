#!/usr/bin/env python3
"""Sentence-level end-to-end typed-span relations; no gold entities at inference."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from sklearn.feature_extraction.text import TfidfVectorizer

from pilot import ROOT, append, begin_attempt, call, prf, read_json, write_json
from polyie_experiment import digest

DATA = ROOT / 'data/mulms'
VERSION = os.environ.get('MULMS_PROTOCOL_VERSION','development_v2')
TYPES = {'MAT','FORM','INSTRUMENT','DEV','NUM','UNIT','RANGE','VALUE','CITE',
         'PROPERTY','TECHNIQUE','SAMPLE','MEASUREMENT'}
LABELS = {'hasForm','measuresProperty','propertyValue','conditionProperty','usedAs',
          'conditionSampleFeatures','conditionPropertyValue','usesTechnique',
          'measuresPropertyValue','usedTogether','conditionEnvironment',
          'conditionInstrument','usedIn','takenFrom','dopedBy'}
CLOSURE = {'measuresPropertyValue','conditionPropertyValue'}
GENERATED = ['baseline','retrieval','generic','targeted','generic_twice',
             'source_confirmation','verify_edits']
EVALUATED = GENERATED[:-1] + ['quote_gate','anchored_quote_gate','semantic_gate_3',
                            'semantic_gate_4','semantic_gate_5',
                            'verifier_quote_gate_4','random_judgment_gate_4',
                            'string_semantic_gate_4']
DEFINITIONS = '''Extract directed material-science relations from the SOURCE
sentence. You must identify the entity mentions and types yourself; no annotated
entities are supplied. Copy exact substrings including their original case.
Each endpoint is {"text":actual_source_substring,"occurrence":zero_based_exact_match_index,"type":entity_type}.
Occurrence counts all overlapping exact substring matches left-to-right.
Types: MAT specific or generic material; FORM material form or morphology
(e.g. thin film, gas/liquid, cubic, crystals); SAMPLE named specimen,
batch, component or computational model; DEV target product/device studied in the
paper, not a test instrument or arbitrary subcomponent; INSTRUMENT test
instrument or analysis equipment/software; TECHNIQUE characterization technique,
not an arbitrary synthesis process; PROPERTY measured physical/chemical property
or a property defining a setting condition;
MEASUREMENT stated measurement/characterization process or trigger, including
technical descriptions and captions without a numeric result; VALUE full value
expression spanning NUM/RANGE and its reported units, with nested NUM/RANGE/UNIT
allowed. A list sharing one unit or a bound may be one VALUE as in the examples;
do not invent missing units or require an Arabic digit in every VALUE.
NUM measurement-related number, not a date or table/figure index; UNIT associated
unit including %; RANGE numeric interval or uncertainty expression including
bounds and intervening range words/symbols; CITE numeric reference marker
(when accompanied by author names, mark the numeric reference only). Nested
mentions may have different types. Preserve source boundaries shown by training
examples; there is no uniform verb-only or full-auxiliary trigger-span rule.
Relations (head -> tail):
measuresProperty: measurement -> measured property.
propertyValue: property -> associated value.
measuresPropertyValue: measurement -> measured result value.
conditionSampleFeatures: measurement -> tested material or sample.
conditionProperty: measurement -> property defining a measurement condition.
conditionPropertyValue: measurement -> condition value.
conditionEnvironment: measurement -> stated environment.
conditionInstrument: measurement -> instrument.
usesTechnique: measurement -> technique.
takenFrom: measurement -> reference from which this measurement SETTING is
borrowed/adapted, not every nearby citation.
hasForm: material/sample -> form.
usedIn: material -> device in which it is used.
usedAs: specific material -> generic material class or functional category
(such as catalyst, often itself typed MAT).
dopedBy: base material -> dopant.
usedTogether: first/aggregate material -> material mixed or adjacent in the same
experiment without synthesis into a new material. A newly synthesized composite
is a nested MAT, not an arbitrary co-occurrence edge. Retain stated direction,
do not symmetrize.
These are annotation conventions, not rigid endpoint-type impossibilities.
Do not invent measurements, values or parameters absent from the sentence.
Do not create parameter edges for purely qualitative outcome descriptions.
Absence of a numeric value alone does not make a technical measurement process
qualitative; retain stated technical slots and independent material relations.
Return compact JSON {"relations":[{"h":endpoint_object,"t":endpoint_object,
"r":relation_label,"evidence":actual_copied_source_words}]}. Quote both actual
endpoint occurrences. Never copy schema placeholders. Return [] when none.
INVENTED FORMAT EXAMPLE ONLY, not a scientific data record:
Source: The modulus was measured at 300 K.
Output: {"relations":[{"h":{"text":"was measured","occurrence":0,"type":"MEASUREMENT"},"t":{"text":"modulus","occurrence":0,"type":"PROPERTY"},"r":"measuresProperty","evidence":"The modulus was measured at 300 K."}]}'''
TASK_SCHEMA = DEFINITIONS.split('Return compact JSON')[0]
if VERSION.startswith('development_v2'):
    OCCURRENCE_GUIDANCE = '''Occurrence is the zero-based index of an EXACT match
of the entire copied endpoint substring, not a word position or character offset.
If that substring occurs only once in SOURCE, occurrence MUST be 0.
Use exactly these type names: MAT, FORM, INSTRUMENT, DEV, NUM, UNIT, RANGE,
VALUE, CITE, PROPERTY, TECHNIQUE, SAMPLE, MEASUREMENT. A relation name is not
an entity type. Preserve spelling/case/direction and don't add a CONDITION type.
FORMAT EXAMPLE for repeated substrings only: in "A rose; A fell", endpoint
text "A" has occurrence 0 for the first A and 1 for the second A.
These instructions do not provide any real annotated entity or relation.'''
    DEFINITIONS = OCCURRENCE_GUIDANCE + '\n' + DEFINITIONS
    TASK_SCHEMA = OCCURRENCE_GUIDANCE + '\n' + TASK_SCHEMA


def occurrences(text, substring):
    if not isinstance(substring,str) or not substring:
        return []
    found=[];i=text.find(substring)
    while i>=0:
        found.append(i);i=text.find(substring,i+1)
    return found


def endpoint_key(endpoint, text):
    if not isinstance(endpoint,dict) or endpoint.get('type') not in TYPES:
        return None
    index=endpoint.get('occurrence')
    if type(index) is not int or index<0:
        return None
    found=occurrences(text,endpoint.get('text'))
    if index>=len(found):
        return None
    start=found[index]
    return (start,start+len(endpoint['text']),endpoint['type'])


def canonical(record, text):
    if isinstance(record,dict):
        h=endpoint_key(record.get('h'),text);t=endpoint_key(record.get('t'),text)
        label=record.get('r')
        if h is not None and t is not None and h!=t and label in LABELS:
            return h,t,label
    return ('INVALID',json.dumps(record,sort_keys=True,ensure_ascii=False))


def valid_key(k):
    return len(k)==3 and isinstance(k[0],tuple) and isinstance(k[1],tuple)


def observed_root_graphs(edges):
    """Exact outgoing graphs for observed, typed MEASUREMENT heads only."""
    roots=defaultdict(set)
    for sentence,key in edges:
        if valid_key(key) and key[0][2]=='MEASUREMENT':
            roots[(sentence,key[0])].add((key[1],key[2]))
    return {(sentence,head,frozenset(tails))
            for (sentence,head),tails in roots.items()}


def source_endpoint(e,text):
    a,b=int(e['begin']),int(e['end']);surface=e['text']
    assert text[a:b]==surface
    return {'text':surface,'occurrence':occurrences(text,surface).index(a),'type':e['value']}


def prepare():
    manifest={'protocol_version':VERSION,'generation_inputs':'source text and document ID only',
              'gold_entities_supplied':False,'splits':{}}
    bank=[]
    for split in ['train','dev','test']:
        file=DATA/'source'/f'{split}.parquet';rows=pq.read_table(file).to_pylist()
        inputs=[];gold={};ids=set();counts=Counter()
        for row in rows:
            text=row['sentence'];doc=str(row['doc_id']);offset=int(row['beginOffset'])
            item={'id':doc+'|'+str(offset),'doc_key':doc,'text':text}
            assert item['id'] not in gold
            inputs.append(item);ids.add(doc)
            ner=row['NER_labels'];ents=[dict(zip(ner,vals)) for vals in zip(*(ner[k] for k in ner))]
            em={e['id']:e for e in ents};rels=row['relations'];records=[]
            for h,t,r in zip(rels['ne_id_gov'],rels['ne_id_dep'],rels['label']):
                assert r in LABELS and h in em and t in em
                he,te=em[h],em[t]
                a=min(int(he['begin']),int(te['begin']));b=max(int(he['end']),int(te['end']))
                rec={'h':source_endpoint(he,text),'t':source_endpoint(te,text),'r':r,
                     'evidence':text[a:b]}
                assert valid_key(canonical(rec,text))
                records.append(rec)
            unique={canonical(r,text):r for r in records}
            counts.update(sentences=1,raw_edges=len(records),unique_edges=len(unique),
                          primitive_edges=sum(k[-1] not in CLOSURE for k in unique),
                          no_gold_edges=int(not unique))
            gold[item['id']]=list(unique.values())
            if split=='train' and text.strip() and len(text)<=1500:
                bank.append({'input':item,'relations':list(unique.values())})
        write_json(DATA/f'{split}_inputs.json',inputs)
        write_json(DATA/f'{split}_gold.json',gold)
        manifest['splits'][split]={'counts':dict(counts),'documents':sorted(ids),
                                  'source_sha256':digest(file.read_bytes())}
    write_json(DATA/'training_examples.json',bank)
    write_json(DATA/'task_manifest.json',manifest)
    print(json.dumps({s:x['counts'] for s,x in manifest['splits'].items()},indent=2))


def selected_inputs(split,per_doc):
    assert split=='dev','Holdout generation/evaluation requires a separately frozen implementation.'
    items=read_json(DATA/f'{split}_inputs.json')
    if not per_doc:
        return items
    by_doc=defaultdict(list)
    for x in items:by_doc[x['doc_key']].append(x)
    return [x for doc in sorted(by_doc)
            for x in sorted(by_doc[doc],key=lambda y:digest('mulms-feasibility|'+y['id']+'|'+y['text']))[:per_doc]]


class Retriever:
    def __init__(self):
        self.bank=read_json(DATA/'training_examples.json')
        self.vec=TfidfVectorizer(ngram_range=(1,2),min_df=1)
        self.matrix=self.vec.fit_transform(x['input']['text'] for x in self.bank)
    def get(self,inp):
        sim=(self.matrix@self.vec.transform([inp['text']]).T).toarray().ravel()
        seen=set();examples=[]
        for i in np.argsort(-sim,kind='stable'):
            x=self.bank[int(i)];doc=x['input']['doc_key']
            if doc in seen:continue
            seen.add(doc);examples.append({'text':x['input']['text'],'relations':x['relations']})
            if len(examples)==3:break
        return examples


def prompt(inp,examples=None):
    assert set(inp)=={'id','doc_key','text'},'Gold fields must not enter inference.'
    p=DEFINITIONS+'\nSOURCE:\n'+inp['text']
    if examples is not None:p+='\nTRAINING EXAMPLES FROM OTHER PAPERS:\n'+json.dumps(examples,ensure_ascii=False)
    return p


def path_for(model,split,stage,inp,repeat):
    return ROOT/'results/mulms'/('cache_'+VERSION)/model.replace('/','__')/split/f'repeat{repeat}'/stage/(digest(inp['id'])[:20]+'.json')


def quote_valid(inp,record,anchored=False):
    k=canonical(record,inp['text'])
    if not valid_key(k):return False
    q=record.get('evidence')
    if not isinstance(q,str) or not q:return False
    # Literal means the exact source characters; do not infer paraphrases.
    positions=occurrences(inp['text'],q)
    if not positions or record['h']['text'] not in q or record['t']['text'] not in q:return False
    if not anchored:return True
    return any(a<=min(k[0][0],k[1][0]) and a+len(q)>=max(k[0][1],k[1][1]) for a in positions)


def edits_for(inp,old,new):
    a={canonical(r,inp['text']):r for r in old['relations']};b={canonical(r,inp['text']):r for r in new['relations']}
    edits=[]
    for op,keys,m in [('add',b.keys()-a.keys(),b),('delete',a.keys()-b.keys(),a)]:
        for k in sorted(keys,key=repr):edits.append({'edit_id':digest(op+repr(k))[:16],'operation':op,'edge':m[k]})
    return edits


def gate(inp,old,new,verified=None,threshold=4,mode='semantic',anchored=True):
    a={canonical(r,inp['text']):r for r in old['relations']}
    if mode=='quote':
        for k in list(a):
            if not valid_key(k):del a[k]
        for r in new['relations']:
            if quote_valid(inp,r,anchored):a[canonical(r,inp['text'])]=r
        return {'relations':list(a.values())}
    if verified.get('error'):
        return {'relations':old['relations'],'verification_error':verified['error'],'fallback_to_parent':True}
    edits={e['edit_id']:e for e in edits_for(inp,old,new)}
    freq=Counter(v.get('edit_id') for v in verified['relations'] if isinstance(v,dict))
    for v in verified['relations']:
        if not isinstance(v,dict) or freq[v.get('edit_id')]!=1:continue
        e=edits.get(v.get('edit_id'));confidence=v.get('confidence')
        if e is None or type(confidence) is not int or not threshold<=confidence<=5:continue
        r=e['edge']
        if not isinstance(r,dict):continue
        r=dict(r,evidence=v.get('evidence',''))
        if not quote_valid(inp,r,anchored):continue
        k=canonical(r,inp['text']);judgment=v.get('judgment')
        if mode=='random':judgment='supported' if int(digest('mulms-negative-control-20261003|'+inp['id']+'|'+e['edit_id']),16)%2 else 'unsupported'
        if e['operation']=='add' and (judgment=='supported' or mode=='verifier_quote'):a[k]=r
        elif e['operation']=='delete' and judgment=='unsupported' and mode!='verifier_quote':a.pop(k,None)
    return {'relations':list(a.values())}


def run_stage(model,stage,workers,repeat,per_doc):
    assert VERSION.startswith('development_v2'), 'Legacy generation uses its archived exact source; current engine is development_v2.'
    inputs=selected_inputs('dev',per_doc);retriever=Retriever() if stage!='baseline' else None
    attempt=begin_attempt('mulms_'+VERSION+'_'+stage+f'_repeat{repeat}',model,'dev')
    def work(inp):
        path=path_for(model,'dev',stage,inp,repeat)
        if path.exists():return read_json(path)
        def get(s):return read_json(path_for(model,'dev',s,inp,repeat))
        p=prompt(inp,retriever.get(inp) if retriever else None);old=None
        if stage in ['baseline','retrieval']:pass
        elif stage in ['generic','targeted','generic_twice','source_confirmation']:
            parent={'generic':'retrieval','targeted':'retrieval','generic_twice':'generic','source_confirmation':'targeted'}[stage]
            old=get(parent);p+='\nCURRENT COMPLETE EXTRACTION:\n'+json.dumps(old['relations'],ensure_ascii=False)
            if stage in ['generic','generic_twice']:p+='\nReview carefully for wrong edges and missing relations. Return a COMPLETE corrected list with exact evidence.'
            else:p+=('\nAudit source support, head/tail direction, exact entity occurrence and type, '
                     'measurement result versus condition, separate experiments, ranges and negations. '
                     'Preserve correct edges and recover explicit omissions. Return a COMPLETE corrected list with exact evidence.')
        elif stage=='verify_edits':
            old=get('retrieval');proposed=get('targeted');edits=edits_for(inp,old,proposed)
            if not edits:
                out={'relations':[],'no_proposed_edits':True};write_json(path,out);return out
            p=(TASK_SCHEMA+'\nJudge proposed edits to directed scientific relations using ONLY SOURCE. '
               'Check the exact head/tail occurrences, entity types, direction and label semantics, '
               'including condition versus measured result. A quote covering names does not prove binding. '
               'For BOTH operations the judgment is factual support of the supplied EDGE, '
               'not whether you approve the edit. Judge deletions by support of the OLD edge. '
               'Delete only when the source establishes an unsupported association, not uncertainty. '
               'Return one compact JSON object with a relations list. Every judgment has edit_id, '
               'judgment (supported, unsupported or uncertain), confidence (integer 1 through 5), '
               'and evidence (a string of actual copied SOURCE words, never a placeholder). '
               'Confidence 1 guess,2 weak,3 plausible,'
               '4 strong explicit evidence,5 unambiguous explicit evidence. Return each edit ID once, '
               'with quote covering the actual endpoint occurrences. Never invent another candidate.\n'
               'SOURCE:\n'+inp['text']+'\nTRAINING EXAMPLES FROM OTHER PAPERS:\n'
               +json.dumps(retriever.get(inp),ensure_ascii=False)+'\nEDITS:\n'+json.dumps(edits,ensure_ascii=False))
        else:raise ValueError(stage)
        out=call(model,p,path)
        if out.get('error') and old is not None and stage!='verify_edits':
            out['relations']=old['relations'];out['fallback_to_parent']=True;write_json(path,out)
        return out
    failures=0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for n,f in enumerate(as_completed([pool.submit(work,x) for x in inputs]),1):
            failures+=bool(f.result().get('error'))
            if n%20==0 or n==len(inputs):print(model,'MuLMS',stage,f'{n}/{len(inputs)}',f'failed={failures}',flush=True)
    append(ROOT/'logs/attempts.jsonl',{'event':'complete','attempt':attempt,'failed_sentences':failures})
    return failures


def evaluate(model,repeat,per_doc):
    inputs=selected_inputs('dev',per_doc);gold=read_json(DATA/'dev_gold.json')
    ids=sorted({x['doc_key'] for x in inputs});preds={};golds=defaultdict(set)
    for inp in inputs:
        golds[inp['doc_key']]|={(inp['id'],canonical(r,inp['text'])) for r in gold[inp['id']]}
    result={'protocol_version':VERSION,'model':model,'repeat':repeat,'analysis':'exploratory_development',
            'n_documents':len(ids),'n_sentences':len(inputs),'gold_entities_supplied':False,'stages':{}}
    for stage in EVALUATED:
        ps=defaultdict(set);failures=0
        for inp in inputs:
            def get(s):return read_json(path_for(model,'dev',s,inp,repeat))
            if stage in GENERATED:out=get(stage)
            else:
                old,new=get('retrieval'),get('targeted')
                if stage in ['quote_gate','anchored_quote_gate']:out=gate(inp,old,new,mode='quote',anchored=stage=='anchored_quote_gate')
                else:
                    mode='verifier_quote' if stage=='verifier_quote_gate_4' else 'random' if stage=='random_judgment_gate_4' else 'semantic'
                    out=gate(inp,old,new,get('verify_edits'),int(stage[-1]),mode,stage!='string_semantic_gate_4')
            failures+=bool(out.get('error') or out.get('verification_error'))
            ps[inp['doc_key']]|={(inp['id'],canonical(r,inp['text'])) for r in out['relations']}
        preds[stage]=ps;cells=[];primitive=[];measurement=[];nonmeasurement=[];root_cells=[];incident=[]
        invalid_count=0;label_counts={label:np.zeros(3,dtype=int) for label in LABELS}
        for d in ids:
            p,g=ps[d],golds[d]
            cells.append([len(p&g),len(p-g),len(g-p)])
            pp={k for k in p if not valid_key(k[1]) or k[1][-1] not in CLOSURE};gg={k for k in g if k[1][-1] not in CLOSURE}
            primitive.append([len(pp&gg),len(pp-gg),len(gg-pp)])
            invalid_count+=sum(not valid_key(k[1]) for k in p)
            pm={k for k in p if valid_key(k[1]) and k[1][0][2]=='MEASUREMENT'}
            gm={k for k in g if k[1][0][2]=='MEASUREMENT'}
            measurement.append([len(pm&gm),len(pm-gm),len(gm-pm)])
            pn={k for k in p if valid_key(k[1]) and k[1][0][2]!='MEASUREMENT'}
            gn=g-gm
            nonmeasurement.append([len(pn&gn),len(pn-gn),len(gn-pn)])
            pr,gr=observed_root_graphs(p),observed_root_graphs(g)
            root_cells.append([len(pr&gr),len(pr-gr),len(gr-pr)])
            pe={(s,end) for s,k in p if valid_key(k) for end in k[:2]}
            ge={(s,end) for s,k in g for end in k[:2]}
            incident.append([len(pe&ge),len(pe-ge),len(ge-pe)])
            for label in LABELS:
                pl={k for k in p if valid_key(k[1]) and k[1][-1]==label}
                gl={k for k in g if k[1][-1]==label}
                label_counts[label]+=np.array([len(pl&gl),len(pl-gl),len(gl-pl)])
        cells=np.array(cells);v=prf(cells.sum(axis=0));v['without_direct_value_labels']=prf(np.array(primitive).sum(axis=0));v['failed_sentences']=failures
        v['measurement_head_edges']=prf(np.array(measurement).sum(axis=0))
        v['other_valid_head_edges']=prf(np.array(nonmeasurement).sum(axis=0))
        v['observed_outgoing_measurement_graphs']=prf(np.array(root_cells).sum(axis=0))
        v['predicted_edge_incident_typed_mentions']=prf(np.array(incident).sum(axis=0))
        v['invalid_edges_in_primary_fp']=invalid_count
        v['per_label']={label:prf(counts) for label,counts in sorted(label_counts.items())}
        v['secondary_scope']='Head-class, graph, mention and label strata use valid edges; unresolvable edges stay FP in primary. Graphs contain only released outgoing sentence-view edges; mention recall is not full NER recall.'
        if stage not in ['baseline','retrieval']:
            e=Counter()
            for d in ids:
                old,new,g=preds['retrieval'][d],ps[d],golds[d]
                e.update(correct_deleted=len((old-new)&g),wrong_deleted=len((old-new)-g),
                         correct_added=len((new-old)&g),wrong_added=len((new-old)-g))
            v['edits_vs_retrieval']=dict(e)
        v['per_document_counts']={d:c.tolist() for d,c in zip(ids,cells)}
        rng=np.random.default_rng(20261003)
        bs=[prf(cells[rng.integers(0,len(ids),len(ids))].sum(axis=0))['f1'] for _ in range(2000)]
        v['descriptive_document_bootstrap_f1_ci95']=np.quantile(bs,[.025,.975]).tolist();result['stages'][stage]=v
    def score(t):
        v=result['stages'][f'semantic_gate_{t}'];e=v['edits_vs_retrieval']
        return v['f1'],-(e['correct_deleted']+e['wrong_added']),-sum(e.values()),t
    result['development_selected_threshold']=max([3,4,5],key=score)
    path=ROOT/'results/mulms'/(model.replace('/','__')+'__'+VERSION+f'__repeat{repeat}_perdoc{per_doc}_summary.json')
    write_json(path,result)
    print(json.dumps({'summary':str(path.relative_to(ROOT)),'stages':{s:round(v['f1']*100,2) for s,v in result['stages'].items()}},indent=2))


def main():
    ap=argparse.ArgumentParser();ap.add_argument('action',choices=['prepare','run','evaluate'])
    ap.add_argument('--model',default='Qwen/Qwen3-8B');ap.add_argument('--stages',default=','.join(GENERATED))
    ap.add_argument('--workers',type=int,default=4);ap.add_argument('--repeat',type=int,default=0)
    ap.add_argument('--per-doc',type=int,default=4);ap.add_argument('--halt-on-failure',action='store_true');a=ap.parse_args()
    if a.action=='prepare':prepare()
    elif a.action=='run':
        for stage in a.stages.split(','):
            failed=run_stage(a.model,stage,a.workers,a.repeat,a.per_doc)
            if failed and a.halt_on_failure:raise SystemExit('Feasibility stopped; failed responses retained.')
    else:evaluate(a.model,a.repeat,a.per_doc)


if __name__=='__main__':main()
