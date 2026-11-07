"""Independent 30-epoch pilot, reusing immutable public-only research modules."""
import os
for _key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMBA_NUM_THREADS'):
    os.environ[_key] = '1'
import argparse
import fcntl
import hashlib
import importlib.util
import json
from pathlib import Path
import signal
import tempfile
import time
import unittest
import numpy as np

import run_candidate_prior_pilot as pilot
from run_frozen_d_closed_loop import save, sha, legal_action
from corner_estimator_scenarios import construct
from verify_l45_occlusion_scenarios import verify_case
from audit_frozen_d_search_interface import observation, independent_traces
from learning.candidate_prior import Network, mixed_prior, geometry_scores
from solvers.frozen_d_search_interface import loader

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'current/l45_candidate_prior_30e_20261026'
SEED = 92020261026
PROTOCOL = ROOT/'code/L45_CANDIDATE_PRIOR_30E_20261026.md'
pilot.OUT, pilot.BASE_SEED, pilot.EPOCHS = OUT, SEED, 30
START = time.monotonic()


def emit(**values):
    pilot.status(**values)


def _limit(signum, frame):
    save(OUT/'budget_stop.json', dict(stop=True, reason='registered_compute_cap',
        elapsed_seconds=time.monotonic()-START, complete=False))
    raise TimeoutError('Registered compute budget exhausted; preserve all completed work')


def arm(seconds=None):
    left = 7200.-(time.monotonic()-START)
    signal.setitimer(signal.ITIMER_REAL, max(.01, min(left, seconds or left)))


def configure():
    if (OUT/'experiment.json').exists():
        return verify()
    assert not (OUT/'evaluation').exists()
    known_maps, known_histories = set(), set()
    for path in (ROOT/'current').glob('**/controller_inputs/*.json'):
        data = json.loads(path.read_text())
        if 'map' in data and 'history' in data:
            known_maps.add(hashlib.sha256(json.dumps(data['map'], sort_keys=True).encode()).hexdigest())
            known_histories.add(hashlib.sha256(json.dumps(dict(map=data['map'], history=data['history']), sort_keys=True).encode()).hexdigest())
    previous_counts = dict(maps=len(known_maps), histories=len(known_histories))
    entries, checks = [], []
    for si, (split, repetitions) in enumerate((('train', 8), ('validation', 2), ('test', 4))):
        for fi, family in enumerate(('long_wall', 'wall_corner', 'multi_route')):
            for rep in range(repetitions):
                H = 4 if rep % 2 == 0 else 6
                speed = 1.1 if rep % 4 < 2 else 1.3
                seed = int(np.random.SeedSequence([SEED, si, fi, rep]).generate_state(1)[0])
                rng = np.random.default_rng(seed)
                height = float(rng.uniform(6.2, 6.9) if H == 4 else rng.uniform(8.15, 8.85))
                env, packet = construct(family, H, speed, seed, height, rep % 2, pause=rep % 2)
                check = verify_case(env, packet)
                assert check['passed'], (env['id'], check)
                md = hashlib.sha256(json.dumps(packet['map'], sort_keys=True).encode()).hexdigest()
                hd = hashlib.sha256(json.dumps(dict(map=packet['map'], history=packet['history']), sort_keys=True).encode()).hexdigest()
                assert md not in known_maps and hd not in known_histories
                known_maps.add(md); known_histories.add(hd)
                folder = OUT/'bank'/split
                for kind, data in (('controller_inputs', packet), ('environment', env)):
                    save(folder/kind/(env['id']+'.json'), data)
                entries.append(dict(id=env['id'], family=family, split=split, map_sha256=md,
                    geometry_history_sha256=hd, packet_path=str(folder/'controller_inputs'/(env['id']+'.json')),
                    environment_path=str(folder/'environment'/(env['id']+'.json'))))
                checks.append(check)
    save(OUT/'bank/manifest.json', dict(entries=entries, counts=dict(train=24, validation=6, test=12),
        previous_input_bank_counts=previous_counts, test_outcomes_read=False,
        split_unit='entire map instance and all histories/worlds'))
    save(OUT/'bank_acceptance.json', dict(passed=True, geometry_only=True, records=checks))
    old = ROOT/'current/l45_candidate_prior_train_20261026'
    old_config = json.loads((old/'experiment.json').read_text())
    sources = dict(old_config['source_hashes'])
    for name in ('run_candidate_prior_30e.py', 'L45_CANDIDATE_PRIOR_30E_20261026.md',
                 'analyze_candidate_prior_30e.py', 'audit_search_value_validation_v2.py'):
        path = ROOT/'code'/name
        sources[str(path)] = sha(path)
    protected = dict(old_config['protected_hashes'])
    for folder in (old, ROOT/'current/l45_search_value_20261026'):
        # Preserve all existing delivered artifacts, not only the old conclusion.
        for path in folder.rglob('*'):
            if path.is_file():
                protected[str(path)] = sha(path)
    config = dict(version='candidate_prior_30e_v1', training_seeds=list(pilot.SEEDS), epochs=30,
        train_maps=24, validation_maps=6, test_maps=12, worlds=2, methods=list(pilot.METHODS),
        evaluation_games=216, budget_ms=200., total_wall_cap_seconds=7200, training_cap_seconds=900,
        training_device='single-thread CPU', inference_device='single-thread NumPy CPU',
        source_hashes=sources, protected_hashes=protected,
        bank_hashes={str(p): sha(p) for p in (OUT/'bank').rglob('*.json')},
        protocol=str(PROTOCOL), previous_80_epoch_results_known=True,
        previous_results_used_for_tuning=False, new_training_from_scratch=True,
        frozen_D_changed=False, deployment_changed=False, ppo=False, value_network=False,
        primary_comparisons=['PUCT-L - PUCT-G', 'PUCT-L - NET_same_seed', 'PUCT-L - G'],
        primary_metric='failure-penalized completion steps', bootstrap_groups=12, bootstrap_draws=10000,
        bootstrap_seed=SEED+1, simultaneous_family_size=3,
        source_of_labels='immutable G, hard argmax plus softmax scores',
        checkpoint='minimum validation loss within 30 epochs, no test selection',
        search=dict(horizon=8, discount=.95, cpuct=1.5, max_simulations=4096,
            prior_scope='root only; deeper uniform', uniform_floor=.1,
            reward='first rediscovery + capture', opponent='original RootPUCT3000'),
        continuation='all three simultaneous cost upper bounds <0, mean <=-1, no observed capture drop, '
            'all per-seed differences <0, audit passes and no full-decision 200ms overruns')
    save(OUT/'experiment.json', config)
    for name in sources:
        path = Path(name)
        target = OUT/'source_snapshot'/path.relative_to(ROOT/'code')
        target.parent.mkdir(parents=True, exist_ok=True); target.write_bytes(path.read_bytes())
    (OUT/'PROTOCOL.md').write_bytes(PROTOCOL.read_bytes())
    emit(stage='registered', train_maps=24, validation_maps=6, test_maps=12, epochs=30, games=216)
    return config


def verify():
    loader.verify()
    config = json.loads((OUT/'experiment.json').read_text())
    for key in ('source_hashes', 'protected_hashes', 'bank_hashes'):
        for path, digest in config[key].items():
            assert sha(path) == digest, ('Frozen file changed', path)
    return config


def tests_module():
    spec = importlib.util.spec_from_file_location('prior_interface_tests', ROOT/'code/tests/test_candidate_prior.py')
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


def interface_check(temporary=False):
    module = tests_module()
    records = []
    for family in ('long_wall', 'wall_corner', 'multi_route'):
        entry = next(e for e in pilot.entries('validation') if e['family'] == family)
        packet = json.loads(Path(entry['packet_path']).read_text()); obs = observation(packet['history'][-1])
        baseline, _ = pilot.initialize(packet, 'G', 719); baseline.act(obs)
        for method in pilot.METHODS:
            repeats = []
            for _ in range(2):
                c, _ = pilot.initialize(packet, method, 719)
                action = c.act(obs, fixed_simulations=8)
                repeats.append(module.signature(c, action))
                for name in ('particles', 'particle_weights', 'weights', 'ids', 'rng'):
                    np.testing.assert_array_equal(getattr(c.estimator, name), getattr(baseline.estimator, name))
                traces = np.concatenate(c.traces) if c.traces else np.zeros((0, 10))
                assert not np.any(independent_traces(traces, c.public.boxes, c.public.speed))
                assert legal_action(obs.position, action, c.public.speed, c.public.boxes)
                roots_prior = c.last_stats.get('root_prior', [])
                if roots_prior:
                    assert np.all(np.isfinite(roots_prior)) and min(roots_prior) > 0
                    assert abs(sum(roots_prior)-1.) < 1e-12
            assert repeats[0] == repeats[1]
            c, setup = pilot.initialize(packet, method, 719)
            c.budget_ms = max(0., 200.-setup['total_ms'])
            stamp = time.perf_counter(); action = c.act(obs)
            elapsed = 1000*(time.perf_counter()-stamp)
            assert legal_action(obs.position, action, c.public.speed, c.public.boxes)
            records.append(dict(family=family, method=method, D_exact=True, replay_exact=True,
                action_legal=True, simulation_legal=True, normalized_prior=True,
                act_ms=elapsed, initialization=setup, first_total_ms=setup['total_ms']+elapsed,
                simulations=c.last_stats['simulations']))
    result = dict(passed=True, records=records, test_outcomes_read=False,
        random_temporary_network=temporary, actual_budget_overruns=sum(r['first_total_ms'] > 200. for r in records))
    save(OUT/('pretraining_interface.json' if temporary else 'preflight.json'), result)
    return result


def pretrain():
    module = tests_module()
    stamp = time.perf_counter()
    suite = unittest.defaultTestLoader.loadTestsFromModule(module)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    save(OUT/'unit_tests.json', dict(passed=result.wasSuccessful(), tests_run=result.testsRun,
        elapsed_seconds=time.perf_counter()-stamp))
    assert result.wasSuccessful()
    # Check actual public-history/weighted-D interfaces before any training.
    with tempfile.TemporaryDirectory(prefix='candidate-prior-30e-') as folder:
        path = Path(folder)/'random.npz'; module.write_toy(path)
        original = pilot.model_path
        pilot.model_path = lambda seed: path
        try:
            interface_check(temporary=True)
        finally:
            pilot.model_path = original
    emit(stage='pretraining_interface_passed')


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('command', choices=('prepare', 'pipeline', 'verify', 'analyze'))
    args = ap.parse_args(); OUT.mkdir(parents=True, exist_ok=True)
    signal.signal(signal.SIGALRM, _limit); arm()
    with (ROOT/'current/.l45_search_value_timing.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        configure()
        if args.command == 'prepare':
            return
        if args.command in ('analyze', 'verify'):
            from analyze_candidate_prior_30e import analyze, delivery
            if args.command == 'analyze': analyze()
            delivery(); return
        if (OUT/'decision.json').exists():
            from analyze_candidate_prior_30e import delivery
            delivery(); return
        emit(stage='starting', total_wall_cap_seconds=7200)
        pretrain()
        pilot.collect(); verify()
        arm(900); pilot.train(); arm(); verify()
        pilot.preflight = interface_check
        pilot.evaluate(); verify()
        from audit_search_value_validation_v2 import audit
        checked = audit(OUT, 'evaluation')
        assert checked['passed'], checked['errors'][:10]
        from analyze_candidate_prior_30e import analyze, delivery
        analyze(); delivery()
        emit(stage='complete', elapsed_seconds=time.monotonic()-START, training_complete=True,
            evaluation_games=216, deployment_changed=False)


if __name__ == '__main__':
    main()
