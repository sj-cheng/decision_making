"""Continue the preregistered design after an independently corrected audit."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMBA_NUM_THREADS'):
    os.environ[key]='1'
import fcntl
import json
import time
from pathlib import Path
import run_search_value_validation as experiment
from audit_search_value_validation_v2 import audit


def main():
    out=experiment.OUT
    with (experiment.ROOT/'current/.l45_search_value_timing.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        config=experiment.freeze()
        # Preserve both the first auditor and its complete failed result.
        corrected=out/'development/postrun_audit_v2.json'
        reviewed=json.loads(corrected.read_text()) if corrected.exists() else audit(out,'development')
        if not reviewed['passed']:
            experiment.save(out/'final_decision.json',dict(stop=True,reason='corrected_development_audit_failed',
                neural_training=False,deployment_changed=False));return
        dev=json.loads((out/'development/statistics.json').read_text())
        proceed=bool(dev['decision']['development_continue'])
        registration=out/'corrective_audit_registration.json'
        names=('audit_search_value_validation_v2.py','tests/test_search_value_audit_v2.py',
            'L45_SEARCH_VALUE_AUDIT_CORRECTION_20261026.md','confirm_search_value_validation.py')
        record=dict(role='verification correction only; algorithm and design unchanged',
            source_hashes={str(experiment.ROOT/'code'/n):experiment.sha(experiment.ROOT/'code'/n) for n in names},
            original_audit_sha256=experiment.sha(out/'development/postrun_audit.json'),
            original_decision_sha256=experiment.sha(out/'decision.json'),
            corrected_development_audit_sha256=experiment.sha(corrected),
            development_results_sha256=experiment.sha(out/'development/results.json'),
            old_experiment_sha256=experiment.sha(out/'experiment.json'),
            confirmation_games_previously_evaluated=len(list((out/'confirmation').glob('episodes/*.json'))),
            gate_unchanged=True,algorithm_changed=False,parameters_changed=False,
            training=False,deployment_changed=False)
        if registration.exists():
            saved=json.loads(registration.read_text())
            assert all(experiment.sha(p)==h for p,h in saved['source_hashes'].items())
        else:experiment.save(registration,record)
        experiment.save(out/'development_gate.json',dict(continue_to_confirmation=proceed,decision=dev['decision'],
            confirmation_outcomes_read=False,audit='postrun_audit_v2.json',audit_correction_only=True))
        if proceed:
            # Warming happens after the independent verifier, outside all
            # decision timers, and before the first real confirmation game.
            stamp=time.perf_counter();task=experiment.tasks('development')[0]
            packet=json.loads(Path(task['packet_path']).read_text())
            for method in experiment.METHODS:
                actor,_=experiment.initialize(packet,method,task['seed'])
                actor.act(experiment.observation(packet['history'][-1]),fixed_simulations=1)
            if not (out/'confirmation_warmup.json').exists():
                experiment.save(out/'confirmation_warmup.json',dict(seconds=time.perf_counter()-stamp,
                    excluded_from_decision_timing=True,algorithm_parameters_unchanged=True))
            confirmation=experiment.run_stage('confirmation')
            if confirmation is None:return
            checked=audit(out,'confirmation')
            if not checked['passed']:
                experiment.save(out/'final_decision.json',dict(stop=True,reason='confirmation_audit_failed',
                    neural_training=False,deployment_changed=False));return
            decision=confirmation['decision']
        else:decision=dev['decision']
        supported=bool(proceed and decision['rootpuct_followup'])
        experiment.save(out/'final_decision.json',dict(stop=not supported,confirmation_run=proceed,
            development_gate_passed=proceed,statistical_decision=decision,
            authoritative_audit_version=2,original_audit_failure_preserved=True,
            reason='unchanged_prespecified_grouped_endpoints',neural_training=False,deployment_changed=False))
        experiment.write_report(out)
        experiment.save(out/'status.json',dict(stage='complete',development_games=120,confirmation_games=240 if proceed else 0,
            stop=not supported,neural_training=False,deployment_changed=False))


if __name__=='__main__':main()
