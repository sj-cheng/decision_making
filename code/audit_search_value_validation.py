"""Independent post-run checks for the frozen-D search-value validation.

This module imports no experimental runner or controller.  Real hidden state
is used solely for post-run motion/sensor/capture checks.  The matched target
is reconstructed from its public packet and registered random seed.
"""
import os
for _name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
              'NUMBA_NUM_THREADS'):
    os.environ[_name] = '1'

import argparse
import collections
import hashlib
import json
from pathlib import Path

import numpy as np

from estimator_audit_geometry import blocked, point_legal
from solvers.frozen_d_search_interface import loader, Observation
from search_value_model import ModelMatchedTarget

ROOT = Path(__file__).resolve().parents[1]


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _path(value, output, stage=None):
    path = Path(value)
    if path.is_absolute():
        return path
    candidates = [output/path, ROOT/path]
    if stage is not None:
        candidates.insert(0, output/stage/path)
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def _observation(item):
    return Observation(int(item['step']), np.asarray(item['position'], float),
                       None if item['target'] is None else
                       np.asarray(item['target'], float), int(item['remaining']))


def _visible(observer, target, boxes):
    return bool(np.sum((observer-target)**2) <= 36.+1e-12
                and not blocked(observer, target, boxes))


def _legal(start, action, limit, boxes):
    end = start+action
    return bool(np.all(np.isfinite(action)) and point_legal(start, boxes)
                and point_legal(end, boxes)
                and np.max(np.abs(action)) <= limit+2e-7
                and not blocked(start, end, boxes))


def _any_errors(value):
    if isinstance(value, dict):
        return any(_any_errors(v) for v in value.values())
    if isinstance(value, (list, tuple)):
        return any(_any_errors(v) for v in value)
    return bool(value)


def _quantiles(values):
    if not values:
        return dict(n=0, mean=None, p95=None, maximum=None)
    a = np.asarray(values, float)
    return dict(n=len(values), mean=float(a.mean()),
                p95=float(np.quantile(a, .95)), maximum=float(a.max()))


def audit(output, stage):
    """Audit all saved episodes; collect errors instead of dropping failures.

    Required task inputs are packet_path and seed.  Rows normally contain
    episode, and may carry optional episode_sha256/trajectory_sha256 fields.
    Optional metadata checks are explicitly counted in checks_performed.
    """
    output = Path(output).resolve()
    loaded = json.loads((output/stage/'results.json').read_text())
    rows = loaded if isinstance(loaded, list) else loaded['rows']
    errors, frames, diagnostics = [], [], []
    checks = collections.Counter()
    pair_values = {}
    matched_initials, matched_paths = {}, {}

    def error(kind, key, step=None, **detail):
        item = dict(kind=kind, key=key, **detail)
        if step is not None:
            item['step'] = int(step)
        errors.append(item)

    for number, row in enumerate(rows):
        key = row.get('key', 'row_'+str(number))
        try:
            episode_path = _path(row['episode'], output, stage)
            payload = json.loads(episode_path.read_text())
            task, trace = payload['task'], payload['trace']
            key = task.get('key', key)
            if 'episode_sha256' in row:
                checks['episode_hashes'] += 1
                if _sha(episode_path) != row['episode_sha256']:
                    error('episode_hash', key)
            if 'trajectory_sha256' in row:
                checks['trajectory_hashes'] += 1
                actual = hashlib.sha256(json.dumps(trace, sort_keys=True).encode()).hexdigest()
                if actual != row['trajectory_sha256']:
                    error('trajectory_hash', key)
            packet = json.loads(_path(task['packet_path'], output, stage).read_text())
            boxes = np.asarray(packet['map'], float).reshape(-1, 2, 2)
            speed = float(packet['motion']['pursuer_axis_limit'])
            initial = np.asarray(payload['initial_state'], float).reshape(-1)
            final = np.asarray(payload['final_state'], float).reshape(-1)
            summary = payload.get('summary', row)
            opponent = task.get('opponent', row.get('opponent', 'unknown'))
            group = task.get('group', task.get('history_id', row.get('history_id')))
            world = task.get('world', row.get('world'))
            method = task.get('method', row.get('method', 'unknown'))
            identity = (str(opponent), str(group), str(world))
            if str(opponent) == 'rootpuct' and 'environment_path' in task:
                environment = json.loads(_path(task['environment_path'], output, stage).read_text())
                environment_world = next(w for w in environment['worlds'] if w['id'] == world)
                prescribed = np.asarray(environment_world['handoff_full_state'], float).reshape(-1)
                if not np.array_equal(initial, prescribed):
                    error('rootpuct_prescribed_initial_state', key)
                checks['environment_initial_templates'] += 1
            elif str(opponent) == 'matched':
                # Example8's 13 coordinates are four robot positions, public
                # clock, then four activity flags; no A/B hidden template is
                # read to reconstruct the matched diagnostic initial state.
                prescribed = np.zeros(13)
                prescribed[2:4] = [-9.8, -9.8]
                prescribed[4:6] = packet['history'][-1]['position']
                prescribed[6:8] = [9.8, 9.8]
                prescribed[8] = packet['history'][-1]['step']
                prescribed[9:] = [1., 0., 1., 0.]
                if not np.array_equal(initial[2:], prescribed[2:]):
                    error('matched_public_state_template', key)
                checks['matched_public_initial_templates'] += 1
            failure = summary.get('failure', row.get('failure'))
            if failure is not None:
                error('recorded_engineering_failure', key, detail=failure)
            if not trace and failure is None and initial[8] < 80:
                error('empty_nonterminal_episode', key)
            own = loader.load_estimator(boxes, int(task['seed']))
            for item in packet['history'][:-1]:
                own.update(_observation(item))
            target = None
            matched_meta = payload.get('matched_initial')
            if matched_meta is not None:
                seed = int(task.get('target_seed', matched_meta.get('seed', task['seed'])))
                draw = int(task.get('draw_index', matched_meta.get('draw_index',
                               0 if str(world) == 'A' else 1 if str(world) == 'B' else 0)))
                target = ModelMatchedTarget(packet, seed, draw)
                expected_meta = dict(index=int(target.initial_index),
                                     mode=int(target.initial_mode),
                                     position=target.position.tolist())
                for name, expected in expected_meta.items():
                    found = matched_meta.get(name, matched_meta.get('initial_'+name))
                    if found is None or not np.array_equal(np.asarray(found), np.asarray(expected)):
                        error('matched_initial_'+name, key, expected=expected, actual=found)
                if not np.array_equal(initial[:2], target.position):
                    error('matched_initial_position', key)
                if identity in matched_initials and expected_meta != matched_initials[identity]:
                    error('matched_initial_not_paired', key)
                matched_initials[identity] = expected_meta
                checks['matched_public_sampler_replays'] += 1
            elif str(opponent).lower().startswith('matched'):
                error('matched_sampler_metadata_missing', key)
            last, first_seen = initial.copy(), None
            all_legal = True
            for index, frame in enumerate(trace):
                at = np.asarray(frame['state'], float).reshape(-1)
                nxt = np.asarray(frame['next_state'], float).reshape(-1)
                obs = frame['observation']
                action = np.asarray(frame['own_action'], float).reshape(2)
                other = np.asarray(frame['opponent_action'], float).reshape(2)
                step = obs['step']
                if not np.array_equal(at, last):
                    error('state_chain', key, step)
                if (int(step) != int(at[8]) or int(obs['remaining']) != 80-int(at[8])
                        or not np.array_equal(obs['position'], at[4:6])):
                    error('public_clock_or_position', key, step)
                unexpected = set(obs)-{'step', 'position', 'target', 'remaining'}
                if unexpected:
                    error('unexpected_actor_observation_fields', key, step,
                          fields=sorted(unexpected))
                seen = _visible(at[4:6], at[:2], boxes)
                if seen != (obs['target'] is not None):
                    error('pre_sensor', key, step)
                if obs['target'] is not None and not np.array_equal(obs['target'], at[:2]):
                    error('positive_measurement', key, step)
                particles, weights = own.update(_observation(obs))
                if (not np.array_equal(particles, frame['particles'])
                        or not np.array_equal(weights, frame['weights'])):
                    error('D_public_replay', key, step)
                if 'estimator_sha256' in frame:
                    signature = hashlib.sha256(own.particles.tobytes()+
                        own.particle_weights.tobytes()+own.rng.tobytes()).hexdigest()
                    if signature != frame['estimator_sha256']:
                        error('D_public_rng_signature', key, step)
                checks['D_public_frame_replays'] += 1
                recorded_weights = np.asarray(frame['weights'], float)
                if (recorded_weights.shape != (64,) or not np.all(np.isfinite(recorded_weights))
                        or np.any(recorded_weights < 0.)
                        or abs(float(recorded_weights.sum())-1.) > 1e-12):
                    error('importance_weights', key, step)
                lawful = (_legal(at[4:6], action, speed, boxes)
                          and _legal(at[:2], other, 1., boxes))
                all_legal = all_legal and lawful
                if not lawful:
                    error('executed_action', key, step)
                if (not np.allclose(nxt[4:6], at[4:6]+action, atol=1e-10, rtol=0.)
                        or not np.allclose(nxt[:2], at[:2]+other, atol=1e-10, rtol=0.)
                        or nxt[8] != at[8]+1):
                    error('joint_transition', key, step)
                post = _visible(nxt[4:6], nxt[:2], boxes)
                if post != bool(frame['post_seen']):
                    error('post_sensor', key, step)
                if post and first_seen is None:
                    first_seen = index+1
                distance_capture = bool(np.linalg.norm(nxt[:2]-nxt[4:6]) < 1.)
                flags_capture = bool(nxt[9] < .5 and nxt[11] < .5)
                if (bool(frame['capture']) != flags_capture
                        or bool(frame['capture']) != distance_capture
                        or bool(frame['capture']) and not all_legal):
                    error('capture', key, step)
                if _any_errors(frame.get('simulation_errors', [])):
                    error('reported_simulation_error', key, step,
                          detail=frame['simulation_errors'])
                if target is not None:
                    if not np.array_equal(target.position, at[:2]):
                        error('matched_pre_position', key, step)
                    predicted_target = target.step()
                    if not np.array_equal(predicted_target, nxt[:2]):
                        error('matched_transition', key, step)
                    target_identity = identity+(index,)
                    coordinates = nxt[:2].tolist()
                    if target_identity in matched_paths and coordinates != matched_paths[target_identity]:
                        error('matched_path_not_paired', key, step)
                    matched_paths[target_identity] = coordinates
                    checks['matched_target_transitions'] += 1
                statistics = frame.get('statistics', {})
                simulations = int(statistics.get('simulations', 0))
                if (obs['target'] is not None or method == 'G') and simulations:
                    error('search_on_common_geometry_branch', key, step)
                if 'root_visits' in statistics and sum(statistics['root_visits']) != simulations:
                    error('root_visit_accounting', key, step)
                attempted = int(statistics.get('attempted_simulations', simulations+
                                statistics.get('budget_aborted_simulations', 0)))
                for name in ('root_counts', 'terminations'):
                    if name in statistics and sum(statistics[name]) != attempted:
                        error(name+'_accounting', key, step)
                own_ms = float(frame['own_ms'])
                charged = float(statistics.get('charged_initialization_ms', 0.))
                budget = float(task.get('budget_ms', statistics.get('registered_budget_ms',
                               statistics.get('nominal_budget_ms', summary.get('budget_ms',
                               2000. if str(method).endswith('H') else 200.)))))
                if not np.isfinite(own_ms) or own_ms < 0:
                    error('invalid_timing', key, step)
                frames.append(dict(key=key, group=group, world=world, opponent=opponent,
                                   method=method, step=step, hidden=obs['target'] is None,
                                   first_decision=index == 0,
                                   own_ms=own_ms, first_total_ms=own_ms+charged,
                                   budget_ms=budget, simulations=simulations,
                                   simulated_transitions=int(frame.get('simulated_transitions', 0))))
                checks['independent_motion_sensor_capture_frames'] += 1
                last = nxt
            if not np.array_equal(final, last):
                error('final_state', key)
            if trace:
                first = trace[0]
                paired = dict(initial=payload['initial_state'], seed=task['seed'],
                              observation=first['observation'], particles=first['particles'],
                              weights=first['weights'], opponent_first=first.get('opponent_raw',
                              first['opponent_action']))
                for name in ('opponent_seed', 'environment_seed', 'target_seed', 'draw_index'):
                    if name in task:
                        paired[name] = task[name]
                if identity in pair_values and paired != pair_values[identity]:
                    error('common_initial_state_root_D_or_first_opponent', key)
                pair_values[identity] = paired
                checks['paired_initial_roots'] += 1
            if 'first_rediscovery' in summary and summary['first_rediscovery'] != first_seen:
                error('rediscovery_summary', key)
            if 'first_seen' in summary and summary['first_seen'] != first_seen:
                error('rediscovery_summary', key)
            captured = bool(trace and trace[-1]['capture'] and failure is None)
            if 'capture' in summary and bool(summary['capture']) != captured:
                error('capture_summary', key)
            original_remaining = 80-int(initial[8])
            if ('original_remaining' in summary
                    and summary['original_remaining'] != original_remaining):
                error('remaining_summary', key)
            expected_cost = len(trace) if captured else original_remaining
            if 'cost' in summary and summary['cost'] != expected_cost:
                error('completion_cost_summary', key)
            expected_rediscovery = first_seen if first_seen is not None else original_remaining
            if ('rediscovery_cost' in summary
                    and summary['rediscovery_cost'] != expected_rediscovery):
                error('rediscovery_cost_summary', key)
            if 'executed_steps' in summary and summary['executed_steps'] != len(trace):
                error('executed_steps_summary', key)
            if ('simulated_transitions' in summary
                    and summary['simulated_transitions'] != sum(
                        int(f.get('simulated_transitions', 0)) for f in trace)):
                error('simulated_transition_summary', key)
            diagnostics.append(dict(key=key, group=group, world=world, method=method,
                                    opponent=opponent, first_rediscovery=first_seen,
                                    capture=captured, engineering_failure=failure is not None,
                                    failure_type=None if captured else
                                    'engineering_failure' if failure is not None else
                                    'not_rediscovered' if first_seen is None else
                                    'late_rediscovery' if first_seen > (80-initial[8])-8 else
                                    'rediscovered_not_captured'))
        except Exception as exc:
            error('audit_exception', key, detail=repr(exc))
        if (number+1) % 20 == 0:
            print(json.dumps(dict(stage='audit_'+stage, completed=number+1,
                                  total=len(rows))), flush=True)

    summary = {}
    for opponent, method in sorted({(str(f['opponent']), str(f['method'])) for f in frames}):
        fs = [f for f in frames if str(f['opponent']) == opponent and str(f['method']) == method]
        hidden = [f for f in fs if f['hidden']]
        summary[opponent+'__'+method] = dict(
            decisions=len(fs), hidden_decisions=len(hidden),
            own_timing_ms=_quantiles([f['own_ms'] for f in fs]),
            hidden_timing_ms=_quantiles([f['own_ms'] for f in hidden]),
            hidden_simulations=_quantiles([f['simulations'] for f in hidden]),
            over_nominal_budget=sum(f['own_ms'] > f['budget_ms'] for f in fs),
            first_charged_over_nominal_budget=sum(f['first_decision'] and
                f['first_total_ms'] > f['budget_ms'] for f in fs),
            over_200ms=sum(f['own_ms'] > 200. for f in fs))
    result = dict(stage=stage, passed=not errors, episodes=len(rows),
                  groups=len({str(d['group']) for d in diagnostics}), frames=len(frames),
                  errors=errors, checks_performed=dict(checks), summary=summary,
                  simulated_transitions=sum(f['simulated_transitions'] for f in frames),
                  episode_diagnostics=diagnostics,
                  notes=[
                      'Actual D is replayed from public history, never hidden truth.',
                      'Matched target retains its initial graph after rediscovery; only pre-first-rediscovery model matching is claimed.',
                      'Reported simulation_errors audit online trace checks; full independent simulation trajectories are not fabricated from summaries.',
                      'Budget excesses are reported, not silently removed from outcome samples.'])
    destination = output/stage/'postrun_audit.json'
    destination.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--stage', required=True)
    args = parser.parse_args()
    report = audit(args.output, args.stage)
    print(json.dumps(dict(passed=report['passed'], errors=len(report['errors']),
                          episodes=report['episodes'], frames=report['frames'])))
    raise SystemExit(0 if report['passed'] else 1)
