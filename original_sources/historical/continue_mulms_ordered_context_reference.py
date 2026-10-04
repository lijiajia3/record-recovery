"""Sequential extra control after classical reference; waiting earns no credit."""
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
STATE = ROOT / 'logs/mulms_ordered_context_reference_pipeline_state.json'


def stamp():
    return datetime.now(timezone.utc).isoformat()


def save(state):
    temp = STATE.with_suffix('.tmp')
    temp.write_text(json.dumps(state, indent=2) + '\n')
    temp.replace(STATE)


def main():
    assert not STATE.exists(), 'Do not duplicate reference coordinator'
    state = {'pid': os.getpid(), 'status': 'waiting_for_complete_classical_reference',
        'started_at_utc': stamp(), 'waiting_counts_as_useful_work': False, 'completed_steps': []}
    save(state)
    try:
        prior_path = ROOT / 'logs/mulms_biaffine_reference_pipeline_state.json'
        while True:
            prior = json.loads(prior_path.read_text())
            assert prior['status'] != 'stopped_explicit_failure', 'Classical reference failed; preserve failure'
            if prior['status'] == 'completed_classical_reference':
                break
            os.kill(prior['pid'], 0)
            time.sleep(10)
        for name, script in [
            ('freeze_reference_fit', 'src/lock_mulms_ordered_context_reference_training.py'),
            ('fit_reference_three_seeds', 'src/mulms_ordered_context_reference_training.py'),
            ('generate_reference_all_three', 'src/mulms_ordered_context_reference_test_generation.py'),
            ('score_reference_descriptive_only', 'src/mulms_ordered_context_reference_test_analysis.py')]:
            log = ROOT / 'logs' / ('mulms_ordered_context_pipeline_' + name + '.log')
            assert not log.exists()
            state.update(status='running_authorized_step', current_step=name, step_started_at_utc=stamp())
            save(state)
            with log.open('w') as stream:
                child = subprocess.Popen([sys.executable, script], cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT)
                state.update(child_pid=child.pid, child_script=script, child_log=str(log.relative_to(ROOT)))
                save(state)
                code = child.wait()
            assert code == 0, name + ' explicit failure, exit ' + str(code)
            state['completed_steps'].append({'name': name, 'completed_at_utc': stamp(), 'exit_code': code})
            state.pop('child_pid', None)
            save(state)
        state.update(status='completed_ordered_context_reference', completed_at_utc=stamp())
        save(state)
    except BaseException as error:
        state.update(status='stopped_explicit_failure', error=str(error), stopped_at_utc=stamp())
        save(state)
        raise


if __name__ == '__main__':
    main()
