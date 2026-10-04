"""Wait for primary GPU fitting, then run the predeclared reference sequentially."""
from datetime import datetime,timezone
import json,os,subprocess,sys,time
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1];STATE=ROOT/'logs/mulms_biaffine_reference_pipeline_state.json'


def stamp():return datetime.now(timezone.utc).isoformat()
def save(value):
    tmp=STATE.with_suffix('.tmp');tmp.write_text(json.dumps(value,indent=2)+'\n');tmp.replace(STATE)


def main():
    assert not STATE.exists(),'Prior reference orchestration exists; no duplicate'
    state={'pid':os.getpid(),'status':'waiting_for_primary_relation_fit','started_at_utc':stamp(),
        'waiting_counts_as_useful_work':False,'completed_steps':[]};save(state)
    try:
        main_state=ROOT/'logs/mulms_supervised_pipeline_state.json'
        while True:
            current=json.loads(main_state.read_text());assert current['status']!='stopped_explicit_failure','Primary failed; preserve explicit failure'
            if any(s['name']=='fit_three_relation_architectures'for s in current['completed_steps']):break
            os.kill(current['pid'],0);time.sleep(10)
        for name,script in [('freeze_reference_fit','src/lock_mulms_biaffine_reference_training.py'),
            ('fit_reference_three_seeds','src/mulms_biaffine_reference_training.py'),
            ('generate_reference_all_three','src/mulms_biaffine_reference_test_generation.py'),
            ('score_reference_descriptive_only','src/mulms_biaffine_reference_test_analysis.py')]:
            if name=='generate_reference_all_three':
                while True:
                    current=json.loads(main_state.read_text())
                    assert current['status']!='stopped_explicit_failure','Primary source graph pipeline failed; no reference Gold access'
                    if any(s['name']=='generate_all_source_test_graphs'for s in current['completed_steps']):break
                    os.kill(current['pid'],0);time.sleep(10)
            log=ROOT/'logs'/f'mulms_pipeline_{name}.log';assert not log.exists()
            state.update(status='running_authorized_step',current_step=name,step_started_at_utc=stamp());save(state)
            with log.open('w')as stream:
                child=subprocess.Popen([sys.executable,script],cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT)
                state.update(child_pid=child.pid,child_script=script,child_log=str(log.relative_to(ROOT)));save(state);code=child.wait()
            assert code==0,f'{name} explicit failure exit {code}; preserve child log'
            state['completed_steps'].append({'name':name,'script':script,'completed_at_utc':stamp(),'exit_code':code});state.pop('child_pid',None);save(state)
        state.update(status='completed_classical_reference',completed_at_utc=stamp());save(state)
    except BaseException as error:
        state.update(status='stopped_explicit_failure',error=str(error),stopped_at_utc=stamp());save(state);raise


if __name__=='__main__':main()
