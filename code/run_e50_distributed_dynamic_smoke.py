"""Dynamic two-agent distributed E50 smoke with local sensors and MINCO prefix."""
import argparse
import json
import math
from pathlib import Path
import numpy as np

from solvers.e50_camera import CameraGimbalTracker
from solvers.e50_distributed_geometry import evaluate_minco_endpoint
from solvers.e50_distributed_runtime import (AgentObservation, FactorisedMCTS,
    LocalPlan, PeerBus, PeerMailbox, make_local_geometry_mcts, make_peer_message)
from solvers.e50_leaf_value import LeafValueNet
from solvers.e50_lidar_map import LidarPlanningMap, raycast_scan


class Belief:
    def __init__(self): self.particles = np.zeros((0, 2)); self.weights = np.zeros(0)


def _single_target_action(target, pursuers, boxes):
    """Choose a legal one-target motion-model action for lifecycle probing."""
    target = np.asarray(target, float); pursuers = np.asarray(pursuers, float)
    away = target - pursuers[int(np.argmin(np.linalg.norm(target - pursuers, axis=1)))]
    norm = float(np.linalg.norm(away))
    unit = away / norm if norm > 1e-9 else np.array([1., 0.])
    tangent = np.array([-unit[1], unit[0]])
    directions = [unit, tangent, -tangent, -unit,
                  np.array([1., 0.]), np.array([-1., 0.]),
                  np.array([0., 1.]), np.array([0., -1.])]
    from e49_rules import _legal
    for direction in directions:
        endpoint = np.clip(target + np.asarray(direction, float), -9.0, 9.0)
        if _legal(endpoint, boxes):
            return endpoint - target
    return np.zeros(2, dtype=float)


def _project_target_state(env, state, targets):
    """Keep simulator-only target state inside E49's legal world bounds."""
    projected = 0
    state_view = np.asarray(state).reshape(-1, 1)
    targets = np.asarray(targets, float).reshape(2, 2).copy()
    from e49_rules import _legal
    for tid, robot in enumerate(env.evaders):
        # Keep a margin from the legal world boundary and move an invalid
        # target to the nearest sampled free point.  RootPUCT can otherwise
        # leave a single active target inside an obstacle, after which the
        # native search correctly reports no valid root action.
        original = targets[tid].copy()
        point = np.clip(original, -8.0, 8.0)
        if not _legal(point, env.obstacles):
            candidates = []
            for radius in np.linspace(.25, 3.0, 12):
                for angle in np.linspace(0., 2. * np.pi, 16, endpoint=False):
                    candidate = np.clip(point + radius * np.array([np.cos(angle), np.sin(angle)]), -8.0, 8.0)
                    if _legal(candidate, env.obstacles):
                        candidates.append(candidate)
                if candidates:
                    break
            if candidates:
                point = min(candidates, key=lambda x: float(np.linalg.norm(x - point)))
        if not np.allclose(point, original):
            projected += 1
        targets[tid] = point
        state_view[env.state_idxs[robot], 0] = point
    return projected, targets


def _safe_executed_prefix(start, positions, local_map):
    """Execute the longest prefix supported by the current local LiDAR map."""
    boxes = np.asarray(local_map.obstacle_boxes(), float).reshape(-1, 2, 2)
    previous = np.asarray(start, float).copy()
    truncated = False
    for point in np.asarray(positions, float):
        inside = bool(len(boxes) and np.any(((point[None, :] >= boxes[:, :, 0]) &
                                             (point[None, :] <= boxes[:, :, 1])).all(axis=1)))
        outside = bool(np.any(point < -9.5) or np.any(point > 9.5))
        if inside or outside or not np.isfinite(point).all():
            truncated = True
            break
        previous = point.copy()
    return previous, truncated


def run(case_path, output, cycles=20, lidar_rays=256, seed=17, delay=0, drop_cycles=(), rootpuct=False, rootpuct_simulations=20, leaf_value_checkpoint=None):
    case = json.loads(Path(case_path).read_text())
    boxes = np.asarray(case['map'], float).reshape(-1, 2, 2)
    state = np.asarray(case['worlds'][0]['initial'], float).reshape(-1)
    pursuers = state[4:8].reshape(2, 2).copy()
    targets = state[:4].reshape(2, 2).copy()
    opponent = None
    env = None
    if rootpuct:
        from e49_rules import create_environment, make_opponent
        env = create_environment(boxes, horizon=int(case.get('horizon', 80)))
        opponent = make_opponent(env, int(seed), simulations=int(rootpuct_simulations))
    trackers = [CameraGimbalTracker(1, fov_deg=60., gimbal_limit_deg=90.,
                                    sweep_rate_deg_s=960., max_range=20., seed=seed+i) for i in range(2)]
    maps = [LidarPlanningMap(max_age=2., max_points=512, inflation=.05,
                             max_boxes=24, merge_distance=.25) for _ in range(2)]
    beliefs = [Belief() for _ in range(2)]
    # Evaluate the full local candidate set before selecting an endpoint.  A
    # one-simulation smoke budget can only evaluate the first candidate and is
    # therefore unsuitable for a movement/capture result.
    value_net = (LeafValueNet.load(leaf_value_checkpoint)
                 if leaf_value_checkpoint is not None else LeafValueNet())
    controllers = {i: make_local_geometry_mcts(value_net=value_net, simulations=8,
                                                seed=seed+i, duration=2., max_candidates=8,
                                                candidate_samples=128) for i in range(2)}
    mailboxes = {i: PeerMailbox(i) for i in range(2)}
    bus = PeerBus((0, 1), delay=int(delay), drop_cycles=drop_cycles)
    previous = {0: None, 1: None}; seq = {0: 0, 1: 0}; logs=[]; captures=[]
    captured_targets = set()
    belief_reset_cycle = -1
    belief_last_seen = {0: -1, 1: -1}
    headings = {0: 0., 1: 0.}
    active_e = np.ones(2, dtype=bool)
    active_p = np.ones(2, dtype=bool)
    rootpuct_failures = 0
    rootpuct_error_cycles = []
    partial_active_fallbacks = 0
    single_active_model_steps = 0
    target_state_projections = 0
    terminal_opponent_skips = 0
    prefix_truncations = 0
    for cycle in range(int(cycles)):
        delivered = bus.deliver(cycle, mailboxes)
        observations = {}; plans = {}; sensor_stats = {}
        minco_records = {}
        for aid in (0, 1):
            scan = raycast_scan(pursuers[aid], boxes, frame_id='map', stamp=float(cycle),
                                num_rays=int(lidar_rays), max_range=10.)
            maps[aid].update(pursuers[aid], scan)
            frame = trackers[aid].step(pursuers[aid].reshape(1, 2), targets, [active_p[aid]],
                                       active_e, [headings[aid]], boxes, dt=1.)
            reports = [x[0] for x in frame.reports if x and x[0] is not None]
            if reports:
                # A frame may contain reports for both evaders.  Keeping only
                # reports[0] silently discarded the second target and made
                # the 2v2 belief effectively single-target.
                beliefs[aid].particles = np.asarray(reports, float).reshape(-1, 2)
                beliefs[aid].weights = np.ones(len(reports), dtype=float)
                belief_last_seen[aid] = int(cycle)
            elif belief_last_seen[aid] < 0 or int(cycle) - belief_last_seen[aid] > 5:
                beliefs[aid].particles = np.zeros((0, 2), dtype=float)
                beliefs[aid].weights = np.zeros(0, dtype=float)
            sensor_stats[str(aid)] = dict(visible_targets=int(np.sum(frame.visible)),
                                          belief_particles=int(len(beliefs[aid].particles)),
                                          camera_state=str(frame.state[0]),
                                          gimbal_yaw=float(frame.gimbal_yaw[0]))
            peer_messages = mailboxes[aid].valid(cycle)
            shared_particles = [np.asarray(m.beliefs, float).reshape(-1, 2)
                                for m in peer_messages
                                if len(m.beliefs) and int(m.cycle) > belief_reset_cycle]
            if shared_particles:
                merged = [beliefs[aid].particles] + shared_particles
                merged = np.concatenate([x for x in merged if len(x)], axis=0)
                # Keep the belief exchange distributed and bounded.  A peer's
                # report is evidence, not simulator truth; duplicate reports
                # are harmless and are removed only geometrically here.
                if len(merged):
                    merged = np.unique(np.round(merged, 6), axis=0)
                    beliefs[aid].particles = merged[-8:]
                    beliefs[aid].weights = np.ones(len(beliefs[aid].particles), dtype=float)
            obs = AgentObservation(aid, cycle, float(cycle), pursuers[aid].copy(),
                                   np.zeros(2), headings[aid], maps[aid], beliefs[aid],
                                   peer_messages,
                                   None if previous[aid] is None else previous[aid].action)
            observations[aid] = obs
            plans[aid] = controllers[aid].act(obs)
        for aid in (0, 1):
            plan = plans[aid]
            seq[aid] += 1
            # Send a bounded recent support set.  A long-lived trajectory of
            # stale reports is not a current target belief and would make the
            # teammate chase an already-visited location.
            recent_particles = (beliefs[aid].particles[-8:]
                                if (cycle > belief_reset_cycle + 1 and
                                    belief_last_seen[aid] >= 0 and
                                    cycle - belief_last_seen[aid] <= 2)
                                else np.zeros((0, 2)))
            belief_payload = tuple(tuple(float(v) for v in point)
                                   for point in recent_particles.tolist())
            bus.publish(make_peer_message(aid, seq[aid], cycle, float(cycle),
                                          pursuers[aid], np.zeros(2), headings[aid],
                                          beliefs=belief_payload, intent=plan.endpoint))
            previous[aid] = plan
        # Execute only the first second of each two-second MINCO plan.
        for aid in (0, 1):
            if not active_p[aid]: continue
            track = evaluate_minco_endpoint(pursuers[aid], plans[aid].endpoint,
                                            duration=2., sample_dt=.02, max_speed=2.0,
                                            max_acceleration=10., proxy=maps[aid])
            minco_records[str(aid)] = dict(feasible=bool(track['feasible']),
                                           collision=bool(track['collision']),
                                           max_speed=float(track['max_speed']),
                                           max_acceleration=float(track['max_acceleration']))
            midpoint = min(len(track['positions']) - 1, int(round(1.0 / .02)))
            executed, truncated = _safe_executed_prefix(pursuers[aid],
                                                         track['positions'][:midpoint + 1],
                                                         maps[aid])
            displacement = executed - pursuers[aid]
            if float(np.linalg.norm(displacement)) > 1e-6:
                headings[aid] = float(math.atan2(displacement[1], displacement[0]))
            pursuers[aid] = executed
            prefix_truncations += int(truncated)
        if opponent is not None:
            if np.asarray(state).ndim == 1:
                state[4:8] = pursuers.ravel()
            else:
                state[4:8, 0] = pursuers.ravel()
            state_column = np.asarray(state).reshape(-1, 1)
            projected, targets = _project_target_state(env, state, targets)
            target_state_projections += int(projected)
            state_column = np.asarray(state).reshape(-1, 1)
            evader_active = [env.is_active(state_column, robot) for robot in env.evaders]
            if float(np.asarray(state).reshape(-1)[env.time_idx]) >= float(env.tf):
                evader_action = np.zeros((2, 2), dtype=float)
                terminal_opponent_skips += 1
            elif any(evader_active):
                # The binding accepts inactive evaders.  Keep an explicit
                # legal one-target motion model only to seed a valid state;
                # RootPUCT still searches the resulting one-active-target
                # state on every cycle.
                seed_action = np.zeros((2, 2), dtype=float)
                for tid, is_active in enumerate(evader_active):
                    if is_active:
                        seed_action[tid] = _single_target_action(targets[tid], pursuers, boxes)
                try:
                    searched = opponent.search(env, state.reshape(-1, 1).copy(), turn=0)
                    evader_action = np.asarray(searched.best_action[:4], float).reshape(2, 2)
                except Exception as exc:
                    rootpuct_failures += 1
                    rootpuct_error_cycles.append(dict(cycle=int(cycle), error=repr(exc),
                                                      active=[bool(x) for x in evader_active],
                                                      state_active=np.asarray(state)[env.active_idxs].reshape(-1).tolist(),
                                                      state_time=float(np.asarray(state).reshape(-1)[env.time_idx]),
                                                      targets=targets.tolist(), pursuers=pursuers.tolist()))
                    evader_action = seed_action
                if not all(evader_active):
                    single_active_model_steps += 1
            else:
                evader_action = np.zeros((2, 2), dtype=float)
            full = np.zeros((env.action_dim, 1), dtype=float)
            full[env.action_idxs[0], 0] = evader_action[0]
            full[env.action_idxs[1], 0] = evader_action[1]
            # The MINCO prefix above is the pursuer's executed motion.  Do
            # not feed the same displacement into E49.step a second time;
            # the environment step is used here for evader dynamics and
            # capture bookkeeping only.
            for aid in (0, 1):
                full[env.action_idxs[2 + aid], 0] = 0.
            state = env.step(state, full, 1.)
            for event in getattr(env, 'last_capture_events', []):
                tid = int(event['evader'])
                if tid not in captured_targets:
                    captured_targets.add(tid)
                    captures.append(dict(cycle=cycle, pursuer=int(event['pursuer']),
                                         target=tid, source='rootpuct'))
                    for belief in beliefs:
                        belief.particles = np.zeros((0, 2), dtype=float)
                        belief.weights = np.zeros(0, dtype=float)
                    belief_reset_cycle = int(cycle)
                    belief_last_seen = {0: -1, 1: -1}
            targets = state[:4].reshape(2, 2).copy()
            pursuers = state[4:8].reshape(2, 2).copy()
            active_e = state[env.active_idxs[0:2], 0] > .5
        else:
            targets[0] = targets[0] + np.array([.35, .12])
            targets[1] = targets[1] + np.array([-.28, .16])
            targets = np.clip(targets, -8.5, 8.5)
        for aid in (0, 1):
            for tid in range(2):
                if active_e[tid] and np.linalg.norm(pursuers[aid] - targets[tid]) < .45:
                    active_e[tid] = False
                    if tid not in captured_targets:
                        captured_targets.add(tid)
                        captures.append(dict(cycle=cycle, pursuer=aid, target=tid))
                        for belief in beliefs:
                            belief.particles = np.zeros((0, 2), dtype=float)
                            belief.weights = np.zeros(0, dtype=float)
                        belief_reset_cycle = int(cycle)
                        belief_last_seen = {0: -1, 1: -1}
                    # Manual execution-side capture must be reflected in the
                    # RootPUCT state before the next search; otherwise E49
                    # resurrects the target and emits duplicate captures.
                    if env is not None:
                        state[env.active_idxs[tid], 0] = 0.
        pursuer_collision = False
        for point in pursuers:
            for box in boxes:
                if box[0, 0] <= point[0] <= box[0, 1] and box[1, 0] <= point[1] <= box[1, 1]:
                    pursuer_collision = True
        logs.append(dict(cycle=cycle, delivered=int(delivered), captures=list(captures),
                         pursuer_collision=bool(pursuer_collision), minco=minco_records,
                         sensor_stats=sensor_stats,
                         plans={str(a): dict(endpoint=plans[a].endpoint.tolist(),
                                             action=plans[a].action.tolist(),
                                             value=float(plans[a].value), visits=int(plans[a].visits)) for a in (0,1)},
                         pursuers=pursuers.tolist(), targets=targets.tolist(),
                         peer_ages={str(a): mailboxes[a].ages(cycle) for a in (0,1)}))
    result = dict(stage='e50_distributed_dynamic_smoke', case=str(case_path), cycles=int(cycles),
                  lidar_rays=int(lidar_rays), delay=int(delay), drop_cycles=list(drop_cycles),
                  rootpuct=bool(rootpuct), rootpuct_simulations=int(rootpuct_simulations),
                  leaf_value_checkpoint=None if leaf_value_checkpoint is None else str(leaf_value_checkpoint),
                  rootpuct_failures=int(rootpuct_failures),
                  rootpuct_error_cycles=rootpuct_error_cycles,
                  partial_active_fallbacks=int(partial_active_fallbacks),
                  single_active_model_steps=int(single_active_model_steps),
                  target_state_projections=int(target_state_projections),
                  terminal_opponent_skips=int(terminal_opponent_skips),
                  prefix_truncations=int(prefix_truncations),
                  truth_used_only_sensor=True, captures=captures, logs=logs)
    output = Path(output); output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    return result


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--case', required=True); p.add_argument('--output', required=True)
    p.add_argument('--cycles', type=int, default=20); p.add_argument('--lidar-rays', type=int, default=256)
    p.add_argument('--delay', type=int, default=0); p.add_argument('--drop-cycle', type=int, action='append', default=[])
    p.add_argument('--rootpuct', action='store_true'); p.add_argument('--rootpuct-simulations', type=int, default=20)
    p.add_argument('--leaf-value-checkpoint', default=None)
    a = p.parse_args(); print(json.dumps(run(a.case, a.output, a.cycles, a.lidar_rays, delay=a.delay, drop_cycles=a.drop_cycle, rootpuct=a.rootpuct, rootpuct_simulations=a.rootpuct_simulations, leaf_value_checkpoint=a.leaf_value_checkpoint), ensure_ascii=False))
