"""Root-only two-model six-slot INVENTED API compatibility; never a study score.

No released source, annotation, corpus loader or Gold/scorer is used. Uses the
reviewed synthetic transport, whose request identity suffices for this ONE
technical repeat; scientific repeats use separately reviewed scientific v2.
"""
from pathlib import Path
import concurrent.futures
import hashlib,json,os,time
import sccomics_api_transport_v1 as t
import sccomics_api_native_projection_v1 as c
import sccomics_api_feedback_tickets_v1 as f

ROOT=Path(__file__).absolute().parents[1]
OUTPUT=ROOT/'research/round4_sccomics_actual_invented_full_feedback_tech_20261003'
SOURCE=b'Copper-doped FeSe was measured at 20 K. The second doping procedure named no dopant.'
PINNED={
 'src/sccomics_api_transport_v1.py':'7b98d93f7d3435f90c2f0353c884b188119aea6c46e63618ebc254afde98f6d9',
 'src/sccomics_api_native_projection_v1.py':'b9087585fea8afa1ab03091bc2a537ec46335cd63301e3a6d0522d5512a8e9d4',
 'src/sccomics_api_feedback_tickets_v1.py':'5c6caeecc04b3093f76caa484bb5735a5414aa0b07c5b6238836c80b3c54c970',
 'src/sccomics_native_graph.py':'b59306e02e7b8132e16366f70c729ee63df83ed10ee39e25ab0c38f74e321b53'}
def check_sources():
 for p,w in PINNED.items():t.require(t.digest(t.checked_bytes(ROOT/p))==w,'reviewed technical input source differs')
def save(path,obj):t.write_new(path,t.serialized(obj))
def one(model,tag):
 path=OUTPUT/tag;path.mkdir()
 stage_results={};graphs={};critics={}
 def request(slot,messages):
  check_sources();payload=f.payload_from_messages(model,messages,slot);raw=t.serialized(payload)
  directory=path/slot;directory.mkdir()
  freeze={'version':1,'purpose':'synthetic_format_probe','source_is_invented':True,'endpoint':t.ENDPOINT,'payload_sha256':t.digest(raw),'source_sha256':t.digest(SOURCE),'client_sha256':PINNED[t.CLIENT_PATH],'pacing_source_sha256':{p:t.digest(t.checked_bytes(ROOT/p)) for p in t.PACING_PATHS},'output_mode':'json_object','schema_sha256':t.digest(t.serialized({'type':'object'})),'limits':{'idle_timeout_s':120,'wall_timeout_s':1800,'max_response_bytes':67108864,'reserved_tokens':len(raw)+payload['max_tokens'],'pacing_policy':'shared_fair_conservative_v1'},'root_authorized_actual_synthetic_probe':True}
  save(directory/'request_before_call.json',payload);save(directory/'freeze_before_call.json',freeze)
  t.write_new(directory/'invented_source.txt',SOURCE)
  result=t.run_probe(ROOT,raw,SOURCE,t.serialized(freeze),directory/'actual_output',execute_actual=True)
  compact={'passed_transport':result['passed'],'model_requested':model,'slot':slot,'actual_API_attempt_started':result['actual_API_attempt_started'],'finish_reason':result.get('finish_reason'),'http_status':result.get('http_status'),'usage':result.get('usage'),'latency':result['latency_seconds_including_admission'],'error':result.get('error')}
  save(directory/'compact_actual_stage.json',compact);stage_results[slot]=compact
  t.require(result['passed'],'technical transport terminal failure; preserve without same-profile retry: '+slot)
  t.require(result['reported_models'] and all(x==model for x in result['reported_models']),'exact reported model needed')
  content=result['raw_content'].encode('utf8');t.write_new(directory/'assistant_content.raw.txt',content)
  if slot.endswith('critic'):
   kind=slot[0];a=f.compile_critic(SOURCE,graphs['B'],content,kind)
   save(directory/'compiled_critic.json',a.to_json());a.require_feedback();critics[kind]=a
  else:
   a=c.compile_graph_json(SOURCE,content);save(directory/'compiled_native.json',a.to_json());a.require_document();graphs[slot]=content
  return content
 try:
  baseline=request('B',f.build_baseline_messages(SOURCE))
  for kind in ('G','T'):request(kind+'_critic',f.build_critic_messages(SOURCE,baseline,kind))
  for mode in ('G','T','S'):
   art=critics['G' if mode=='G' else 'T']
   if mode!='G':save(path/(mode+'_correspondence_before_call.json'),f.correspondence(art,mode,model=model,source_id='INVENTED-TECH-ONE',repeat=0).to_json())
   request(mode+'_repair',f.build_repair_messages(SOURCE,baseline,art,mode,model=model,source_id='INVENTED-TECH-ONE',repeat=0))
  status={'passed':True,'all_six_unique_actual_slots':list(stage_results),'scientific_Gold_score':False,'model':model,'stage_results':stage_results}
 except BaseException as error:
  status={'passed':False,'model':model,'error_type':type(error).__name__,'error':str(error),'scientific_Gold_score':False,'stage_results':stage_results}
 check_sources();save(path/'final_actual_technical_status.json',status)
 return status
def main():
 import argparse
 parser=argparse.ArgumentParser();parser.add_argument('--execute-root-invented-two-model-six-slot-probe',action='store_true');args=parser.parse_args()
 t.require(args.execute_root_invented_two_model_six_slot_probe,'default no API authority')
 check_sources();t.require(not OUTPUT.exists(),'single actual technical campaign, no implicit retry')
 OUTPUT.mkdir()
 reviews=['research/round4_sccomics_api_transport_independent_source_review.json','research/round4_sccomics_api_native_projection_independent_source_review.json','research/round4_sccomics_api_feedback_tickets_independent_source_review.json']
 for p in reviews:
  d=t.unique_json(t.checked_bytes(ROOT/p))
  for name,w in d['file_sha256'].items():t.require(t.digest(t.checked_bytes(ROOT/name))==w,'current independent review binding differs')
 save(OUTPUT/'root_before_all_calls_authorization.json',{'purpose':'pure_invented_full_feedback_technical_compatibility','actual_paid_provider':'SiliconFlow','models':list(t.MODELS),'max_actual_success_calls':12,'actual_request_repeats':1,'source_is_invented':True,'source_sha256':t.digest(SOURCE),'source':SOURCE.decode(),'source_sha256_pins':PINNED,'script_sha256':t.digest(t.checked_bytes(ROOT/'src/run_sccomics_invented_feedback_tech_v1.py')),'review_JSON_sha256':{p:t.digest(t.checked_bytes(ROOT/p)) for p in reviews},'held_Gold_or_corpus_read':False,'scientific_performance':False,'parameters_GLM_thinking_off_not_established':True,'authorized_at_unix':time.time()})
 with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
  jobs=[executor.submit(one,m,tag) for m,tag in zip(t.MODELS,('deepseek_v4_pro','glm_5_3'))]
  rows=[job.result() for job in jobs]
 files={str(p.relative_to(OUTPUT)):{'sha256':t.digest(t.checked_bytes(p)),'size_bytes':p.stat().st_size} for p in OUTPUT.rglob('*') if p.is_file()}
 save(OUTPUT/'actual_technical_campaign_summary.json',{'scope':'ACTUAL_INVENTED_PIPELINE_FORMAT_ONLY_NO_GOLD_OR_SCIENTIFIC_GAIN','models':rows,'passed':all(x['passed'] for x in rows),'calls_that_started':sum(stage['actual_API_attempt_started'] for x in rows for stage in x['stage_results'].values()),'files':files})
 print(json.dumps({'passed':all(x['passed'] for x in rows),'models':[{'model':x['model'],'passed':x['passed'],'completed_slots':len(x['stage_results'])} for x in rows],'output':str(OUTPUT)}))
 return 0 if all(x['passed'] for x in rows) else 1
if __name__=='__main__':raise SystemExit(main())
