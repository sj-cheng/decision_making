"""Exact arithmetic reuse, separate execution version, immutable algorithms."""
import sys,time,json,shutil
from pathlib import Path
import run_rollout_replan_experiment as experiment
import solvers.rollout_geometry_exact as fast

OLD_OUT=experiment.OUT
OUT=experiment.ROOT/'current/l45_rollout_replan_exact_20261026'
_freeze=experiment.freeze
_preflight=experiment.preflight
_prepare=experiment.prepare


def prepare():
    if not (OUT/'bank_frozen.json').exists():
        assert not (OLD_OUT/'confirmation/results.json').exists()
        assert not list((OLD_OUT/'confirmation/episodes').glob('*.json'))
        OUT.mkdir(parents=True,exist_ok=True)
        shutil.copytree(OLD_OUT/'bank',OUT/'bank')
        for name in ('bank_frozen.json','bank_acceptance.json'):shutil.copyfile(OLD_OUT/name,OUT/name)
    return _prepare()


def freeze():
    result=_freeze()
    names=('solvers/rollout_geometry_exact.py','launch_rollout_replan_exact.py','tests/test_rollout_geometry_exact.py','L45_ROLLOUT_REPLAN_EXACT_EXECUTION_20261026.md')
    revision=dict(role='exact execution reuse; rollout policy unchanged at equal complete simulations',
        source_hashes={str(experiment.ROOT/'code'/n):experiment.sha(experiment.ROOT/'code'/n) for n in names},
        preceding_execution_stop_sha256=experiment.sha(OLD_OUT/'execution_stop_record.json'),
        preceding_development_is_partial=True,confirmation_bank_originally_frozen_sha256=experiment.sha(OLD_OUT/'bank_frozen.json'),
        confirmation_evaluated_before_revision=False,parameters_changed=False,neural_training=False,deployment_changed=False)
    path=OUT/'execution_revision.json'
    if path.exists():assert json.loads(path.read_text())==revision
    else:experiment.save(path,revision)
    return result


def preflight():
    started=time.perf_counter();experiment.warmup()
    entry=experiment.development_entries()[0];packet=experiment.inputs(dict(stage='development',entry=entry))[0]
    own,_=experiment.initialize(packet,'T+Geo-R',17)
    own.act(experiment.observation(packet['history'][-1]),fixed_simulations=2)
    experiment.save(OUT/'pipeline_warmup.json',dict(seconds=time.perf_counter()-started,source_sha256=experiment.sha(__file__),
        scope='both kernels warmed under serial timing lock; no confirmation observations'))
    return _preflight()


if __name__=='__main__':
    fast.install();experiment.OUT=OUT
    experiment.prepare=prepare;experiment.freeze=freeze;experiment.preflight=preflight
    if len(sys.argv)<2:sys.argv.append('pipeline')
    experiment.main()
