"""Dynamic two-agent distributed E50 smoke with local sensors and MINCO prefix."""
import sys
from pathlib import Path
_CODE_DIR = Path(__file__).resolve().parents[1]
if str(_CODE_DIR) not in sys.path:
    sys.path.insert(0, str(_CODE_DIR))
import repo_paths
repo_paths.install()
import argparse
import json
import math
import os
import time
from pathlib import Path
import numpy as np

from solvers.e50_camera import CameraGimbalTracker
from solvers.e50_distributed_geometry import (DEFAULT_MAX_ACCELERATION, SHARED_SPEED_LIMIT, SHARED_ACCELERATION_LIMIT, evaluate_minco_endpoint,
    validate_max_acceleration)
from solvers.e50_distributed_runtime import (AgentObservation, FactorisedMCTS,
    LocalPlan, PeerBus, PeerMailbox, make_local_geometry_mcts, make_peer_message)
from solvers.e50_leaf_value import LeafValueNet
from solvers.e50_lidar_map import LidarPlanningMap, raycast_scan, points_to_obstacle_boxes


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


def run(case_path, output, cycles=20, lidar_rays=256, seed=17, delay=0, drop_cycles=(), rootpuct=False, rootpuct_simulations=20, leaf_value_checkpoint=None, e52=False, e52_checkpoint=None, e52_mode="both", e52_depth=None, e52_budget=256, e52_time_ms=None, e52_nae_strategy='legacy', offline_branch=None, e52_advantage_checkpoint=None, *, geometric_evader_speed=None, evader_minco=False, max_acceleration=None, shared_max_speed=None, evader_hmcts=False, evader_window=8, e53_depth=3, e53_simulations=256, pursuer_structure="local", pursuer_speed_cap=None, decision_hook=None, audit_nae=False, evader_cruise=False, environment_horizon_floor=0, legacy_policy=None, nar_override=None, nae_override=None, waypoint_generator=None, search_experiment=None, evader_factory=None, direct_policy=None):
    if direct_policy is not None and (not e52 or pursuer_structure!='local' or legacy_policy is not None):
        raise ValueError('direct waypoint policy requires local E52 execution')
    if evader_cruise and not evader_hmcts:
        raise ValueError("cruise candidates require the H-MCTS evader")
    if pursuer_structure not in ("local", "coordinated", "joint_oracle", "joint_shared", "joint_global"):
        raise ValueError("unknown pursuer structure")
    if e52_depth is None:
        e52_depth = 8 if pursuer_structure == "joint_shared" else 3
    full_joint = pursuer_structure in ("joint_shared", "joint_global")
    if full_joint and (not e52 or offline_branch is not None):
        raise ValueError("joint search requires E52 online planning")
    if evader_minco and geometric_evader_speed is None and shared_max_speed is None:
        shared_max_speed = SHARED_SPEED_LIMIT
    if shared_max_speed is not None:
        if not evader_minco or not np.isfinite(shared_max_speed) or shared_max_speed <= 0:
            raise ValueError('shared max speed requires dual MINCO and a positive finite limit')
        if geometric_evader_speed is not None and geometric_evader_speed != shared_max_speed:
            raise ValueError('conflicting evader and shared speed limits')
        geometric_evader_speed = shared_max_speed
        if max_acceleration is None:
            max_acceleration = SHARED_ACCELERATION_LIMIT
    pursuer_max_speed = 2. if shared_max_speed is None else shared_max_speed
    if pursuer_speed_cap is not None:
        if shared_max_speed is None or not 0 < pursuer_speed_cap <= shared_max_speed:
            raise ValueError("pursuer_speed_cap requires a positive cap within the shared motion limit")
        pursuer_max_speed = float(pursuer_speed_cap)
    if evader_hmcts and (rootpuct or offline_branch is not None):
        raise ValueError('evader H-MCTS cannot be combined with RootPUCT or offline branch collection')
    if geometric_evader_speed is not None and (rootpuct or offline_branch is not None):
        raise ValueError('geometric evader cannot be combined with RootPUCT or offline branch collection')
    if evader_hmcts and not evader_minco:
        raise ValueError('evader H-MCTS requires MINCO execution')
    if evader_minco and (not e52 or (geometric_evader_speed is None and not evader_hmcts)):
        raise ValueError('evader MINCO requires the E52 geometric opponent')
    if max_acceleration is not None:
        if not evader_minco:
            raise ValueError('a shared max acceleration requires MINCO execution for both sides')
        max_acceleration = validate_max_acceleration(max_acceleration)
    execution_limit = DEFAULT_MAX_ACCELERATION if max_acceleration is None else max_acceleration
    output = Path(output); output.parent.mkdir(parents=True, exist_ok=True)
    progress = output.with_suffix('.progress.json')
    started = time.time()
    def publish_progress(completed, state='running'):
        temporary = progress.with_suffix('.tmp')
        temporary.write_text(json.dumps(dict(pid=os.getpid(),state=state,case=str(case_path),
            mode=e52_mode if e52 else 'e50',completed_cycles=completed,total_cycles=int(cycles),
            elapsed_seconds=time.time()-started,updated_at=time.time())))
        temporary.replace(progress)
    publish_progress(0)
    case = json.loads(Path(case_path).read_text())
    boxes = np.asarray(case['map'], float).reshape(-1, 2, 2)
    state = np.asarray(case['worlds'][0]['initial'], float).reshape(-1)
    pursuers = state[4:8].reshape(2, 2).copy()
    targets = state[:4].reshape(2, 2).copy()
    opponent = None
    env = None
    opponent_policy = 'scripted'
    if rootpuct:
        from e49_rules import create_environment, make_opponent
        effective_horizon = max(int(case.get('horizon', 80)), int(cycles), environment_horizon_floor)
        env = create_environment(boxes, horizon=effective_horizon)
        opponent = make_opponent(env, int(seed), simulations=int(rootpuct_simulations))
        opponent_policy = 'rootpuct'
    elif geometric_evader_speed is not None or evader_hmcts:
        from e49_rules import _legal, create_environment
        from solvers.e49_geometric_evader import GeometricEvader
        from solvers.e53_evader_hmcts import EvaderHMCTS
        effective_horizon = max(int(case.get('horizon', 80)), int(cycles), environment_horizon_floor)
        if not all(_legal(target, boxes) for target in targets):
            raise ValueError('geometric evader requires legal initial positions')
        if geometric_evader_speed is None:
            geometric_evader_speed = shared_max_speed or SHARED_SPEED_LIMIT
        opponent = ((evader_factory or EvaderHMCTS)(evader_speed=geometric_evader_speed,
                                 max_acceleration=max_acceleration or SHARED_ACCELERATION_LIMIT,
                                 depth=e53_depth, simulations=e53_simulations,
                                 window=evader_window, seed=seed, cruise=evader_cruise) if evader_hmcts else
                    GeometricEvader(evader_speed=geometric_evader_speed,
                                    predicted_pursuer_speed=shared_max_speed))
        env = create_environment(boxes, horizon=effective_horizon, evader_speed=geometric_evader_speed)
        # Historical starts contain elapsed observation ages, which remain valid
        # when the episode clock is rebased to the start of this game.
        if evader_hmcts:
            state[env.time_idx] = 0.
            opponent.prepare(env, targets[0])
        opponent_policy = 'evader_hmcts' if evader_hmcts else 'geometric'
    if evader_minco:
        from solvers.e50_evader_minco import EvaderMinco
        from solvers.e50_interfaces import sample_quintic_at
        samples = int(round(1. / env.capture_dt))
        capture_times = np.linspace(0., 1., samples + 1)
        evader_executor = EvaderMinco(boxes, geometric_evader_speed, samples,
                                     max_acceleration=max_acceleration, continuous_speed=shared_max_speed is not None)
    trackers = [CameraGimbalTracker(1, fov_deg=60., gimbal_limit_deg=90.,
                                    sweep_rate_deg_s=960., max_range=20., seed=seed+i) for i in range(2)]
    maps = [LidarPlanningMap(max_age=2., max_points=512, inflation=.05,
                             max_boxes=24, merge_distance=.25) for _ in range(2)]
    if e52:
        from solvers.e52_belief import TargetBelief
        beliefs = [TargetBelief(i) for i in range(2)]
    else:
        beliefs = [Belief() for _ in range(2)]
    # Evaluate the full local candidate set before selecting an endpoint.  A
    # one-simulation smoke budget can only evaluate the first candidate and is
    # therefore unsuitable for a movement/capture result.
    value_net = (LeafValueNet.load(leaf_value_checkpoint)
                 if leaf_value_checkpoint is not None else LeafValueNet())
    if e52:
        from solvers.e52_hmcts import HierarchicalMCTS
        from solvers.e52_nar_nae import NARNAE
        nar_nae = NARNAE.load(e52_checkpoint) if e52_checkpoint else None
        if nar_nae is not None:
            nar_nae.torch.set_num_threads(1)
        advantage = None
        if e52_advantage_checkpoint is not None:
            from solvers.e52_advantage import AdvantageMLP
            if e52_nae_strategy not in ('leaf_residual', 'budget_leaf_residual') or e52_mode not in ('both', 'nae'):
                raise ValueError('branch advantage requires leaf_residual with NAE enabled')
            advantage = AdvantageMLP.load(e52_advantage_checkpoint)
            advantage.torch.set_num_threads(1)
        controllers = {i: HierarchicalMCTS(nar=nar_nae if e52_mode in ("both", "nar") else None,
                                            nae=(advantage or nar_nae) if e52_mode in ("both", "nae") else None,
                                            high_simulations=e52_budget, max_candidates=8, depth=e52_depth,
                                            time_budget_ms=e52_time_ms, duration=2.,
                                            nae_strategy=e52_nae_strategy,
                                            max_acceleration=max_acceleration, max_speed=pursuer_max_speed if shared_max_speed is not None else None) for i in range(2)}
    else:
        controllers = {i: make_local_geometry_mcts(value_net=value_net, simulations=8,
                                                    seed=seed+i, duration=2., max_candidates=8,
                                                    candidate_samples=128) for i in range(2)}
    if nar_override is not None:
        for controller in controllers.values():
            controller.nar = nar_override
    for controller in controllers.values():
        if nae_override is not None:
            controller.nae = nae_override
        if waypoint_generator is not None:
            controller.waypoint_generator = waypoint_generator
    if audit_nae:
        for controller in controllers.values():
            controller.audit_ranking = True
    mailboxes = {i: PeerMailbox(i) for i in range(2)}
    bus = PeerBus((0, 1), delay=int(delay), drop_cycles=drop_cycles)
    previous = {0: None, 1: None}; seq = {0: 0, 1: 0}; logs=[]; captures=[]
    captured_targets = set()
    belief_reset_cycle = -1
    belief_last_seen = {0: -1, 1: -1}
    headings = {0: 0., 1: 0.}
    velocities = np.zeros((2, 2)); accelerations = np.zeros((2, 2))
    evader_velocities = np.zeros((2, 2)); evader_accelerations = np.zeros((2, 2))
    active_e = np.ones(2, dtype=bool)
    active_p = np.ones(2, dtype=bool)
    rootpuct_failures = 0
    rootpuct_error_cycles = []
    partial_active_fallbacks = 0
    single_active_model_steps = 0
    target_state_projections = 0
    terminal_opponent_skips = 0
    braking_tails = {}
    braking_remainders = {}
    prefix_truncations = 0
    time_team = (direct_policy is not None or getattr(search_experiment, 'measure_team_latency', False))
    for cycle in range(int(cycles)):
        if time_team:
            decision_start = time.perf_counter()
        delivered = bus.deliver(cycle, mailboxes)
        observations = {}; plans = {}; sensor_stats = {}
        minco_records = {}
        evader_records = {}
        evader_search = {}
        camera_visible = np.zeros((2, 2), bool)
        if evader_minco:
            pursuer_paths = np.repeat(pursuers[None, :, :], samples + 1, axis=0)
            evader_paths = np.repeat(targets[None, :, :], samples + 1, axis=0)
        for aid in (0, 1):
            scan = raycast_scan(pursuers[aid], boxes, frame_id='map', stamp=float(cycle),
                                num_rays=int(lidar_rays), max_range=10.)
            maps[aid].update(pursuers[aid], scan)
            frame = trackers[aid].step(pursuers[aid].reshape(1, 2), targets, [active_p[aid]],
                                       active_e, [headings[aid]], boxes, dt=1.)
            camera_visible[aid] = frame.visible[0]
            reports = [x[0] for x in frame.reports if x and x[0] is not None]
            if e52:
                beliefs[aid].update([(tid, x[0]) for tid,x in enumerate(frame.reports) if x and x[0] is not None], float(cycle))
            else:
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
            if e52:
                beliefs[aid].merge(peer_messages, float(cycle))
            else:
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
                                   velocities[aid].copy() if e52 else np.zeros(2), headings[aid], maps[aid], beliefs[aid],
                                   peer_messages,
                                   None if previous[aid] is None else previous[aid].action,
                                   acceleration=accelerations[aid].copy() if e52 else None)
            observations[aid] = obs
            if not full_joint and legacy_policy is None:
                if search_experiment is not None:
                    search_experiment.begin(controllers[aid], obs, len(captured_targets))
                if direct_policy is None:
                    plans[aid] = controllers[aid].act(obs)
                else:
                    started = time.perf_counter()
                    plans[aid] = direct_policy.act(obs, scan, len(captured_targets))
                    controllers[aid].last_search = dict(root=[], simulations=0, evaluations=0,
                        elapsed_ms=(time.perf_counter()-started)*1000)
            if offline_branch is not None:
                distances = np.linalg.norm(obs.belief.particles - obs.pose, axis=1)
                sensor_stats[str(aid)]['nearest_belief'] = float(min(distances)) if len(distances) else None
        coordination = dict(mode=pursuer_structure, changed=False)
        if legacy_policy is not None:
            measured = points_to_obstacle_boxes(np.concatenate([m.points for m in maps]),
                                                inflation=.05, max_boxes=24, merge_distance=.25)
            reports = [[targets[j].tolist() if camera_visible[i,j] else None for i in range(2)] for j in range(2)]
            if cycle == 0 and getattr(legacy_policy, 'initial_target_fix', False):
                reports = [[targets[j].tolist() if active_e[j] and active_p[i] else None
                            for i in range(2)] for j in range(2)]
            camera = dict(gimbal_yaw=[sensor_stats[str(i)]['gimbal_yaw'] for i in range(2)],
                          headings=[headings[i] for i in range(2)], fov_rad=np.deg2rad(60.), max_range=20.)
            plans = legacy_policy(cycle, pursuers, active_p, active_e, reports, camera, measured)
            for controller in controllers.values():
                controller.last_search = dict(method=legacy_policy.method)
            coordination['mode'] = 'legacy_team'
        elif full_joint:
            from solvers.e54_joint_search import joint_plans
            world = (boxes, targets[active_e], evader_velocities[active_e]) if pursuer_structure == "joint_global" else None
            plans, coordination = joint_plans(observations, controllers, world)
        elif e52 and pursuer_structure != "local":
            from solvers.e54_pursuer_coordination import coordinate
            plans_list, coordination = coordinate(
                [plans[0], plans[1]],
                [controllers[0].last_search, controllers[1].last_search],
                [beliefs[0], beliefs[1]],
                oracle_targets=targets[active_e] if pursuer_structure == "joint_oracle" else None,
                mode="oracle" if pursuer_structure == "joint_oracle" else "belief")
            plans = {i: plans_list[i] for i in (0, 1)}
        if time_team:
            coordination['team_decision_ms'] = (time.perf_counter() - decision_start)*1000
        if decision_hook is not None:
            decision_hook(cycle, observations, controllers, plans)
        if offline_branch is not None and offline_branch.cycle == cycle:
            # Both agents plan from their own observations before the single
            # offline intervention. Native RNG is reconstructed by replaying
            # the same prefix, rather than copying an opaque C++ wrapper.
            offline_branch.apply(
                observations, plans, controllers,
                (state, trackers, maps, beliefs, bus, mailboxes, previous, seq,
                 velocities, accelerations, headings, active_e, active_p, captures,
                 (opponent.calls, opponent.get_rng_state(env)) if opponent else None),
            )
        for aid in (0, 1):
            plan = plans[aid]
            seq[aid] += 1
            # Send a bounded recent support set.  A long-lived trajectory of
            # stale reports is not a current target belief and would make the
            # teammate chase an already-visited location.
            if e52:
                belief_payload = beliefs[aid].payload()
            else:
                recent_particles = (beliefs[aid].particles[-8:]
                                    if (cycle > belief_reset_cycle + 1 and
                                        belief_last_seen[aid] >= 0 and
                                        cycle - belief_last_seen[aid] <= 2)
                                    else np.zeros((0, 2)))
                belief_payload = tuple(tuple(float(v) for v in point)
                                       for point in recent_particles.tolist())
            bus.publish(make_peer_message(aid, seq[aid], cycle, float(cycle),
                                          pursuers[aid], velocities[aid] if e52 else np.zeros(2), headings[aid],
                                          beliefs=belief_payload, intent=plan.endpoint,
                                          map_delta=tuple(tuple(float(v) for v in point) for point in maps[aid].free_space_endpoints(max_candidates=4)) if e52 else ()))
            previous[aid] = plan
        # Execute only the first second of each two-second MINCO plan.
        for aid in (0, 1):
            if not active_p[aid]: continue
            incoming_speed = float(np.linalg.norm(velocities[aid]))
            track = evaluate_minco_endpoint(pursuers[aid], plans[aid].endpoint,
                                            duration=plans[aid].duration, sample_dt=.02, max_speed=pursuer_max_speed,
                                            pass_through=plans[aid].pass_through,
                                            continuous_speed=shared_max_speed is not None,
                                            max_acceleration=execution_limit, proxy=maps[aid],
                                            continuous_acceleration=max_acceleration is not None,
                                            start_velocity=velocities[aid] if e52 else None,
                                            start_acceleration=accelerations[aid] if e52 else None)
            if e52:
                from solvers.e52_hmcts import LocalSnapshot
                track["collision"] = not LocalSnapshot(maps[aid]).path_safe(track["positions"])
                track["feasible"] = bool(track["feasible"] and not track["collision"])
            minco_records[str(aid)] = dict(feasible=bool(track['feasible']),
                                           collision=bool(track['collision']), duration=plans[aid].duration,
                                           max_speed=float(track['max_speed']),
                                           max_acceleration=float(track['max_acceleration']))
            midpoint = min(len(track['positions']) - 1, int(round(1.0 / .02)))
            executed, truncated = _safe_executed_prefix(pursuers[aid],
                                                         track['positions'][:midpoint + 1],
                                                         maps[aid])
            if e52 and not track["feasible"]:
                # An invalid trajectory is recorded and never executed.
                executed = pursuers[aid].copy(); truncated = True
            if evader_minco and not truncated:
                points, v, a = sample_quintic_at(track['coefficients'],
                                                np.minimum(capture_times, plans[aid].duration))
                pursuer_paths[:, aid] = points
                track['positions'], track['velocities'], track['accelerations'] = points, v, a
                midpoint = samples
                executed = points[-1]
            used_tail = False
            if shared_max_speed is not None:
                tail = braking_tails.pop(aid, None)
                remainder = braking_remainders.pop(aid, None)
                if truncated and tail is not None and LocalSnapshot(maps[aid]).path_safe(tail[0]):
                    if remainder is not None:
                        braking_tails[aid] = remainder
                    points, v, a = tail
                    track['positions'], track['velocities'], track['accelerations'] = tail
                    pursuer_paths[:, aid] = points
                    midpoint = samples
                    executed, truncated, used_tail = points[-1], False, True
                elif not truncated and plans[aid].pass_through:
                    # The unused waypoint tail now ends in motion. Cache a
                    # separate validated braking curve from the next boundary.
                    controller = controllers[aid]
                    for endpoint, duration in controller._stops(executed, v[-1], a[-1]):
                        stop = evaluate_minco_endpoint(executed, endpoint, duration=duration,
                            start_velocity=v[-1], start_acceleration=a[-1], max_speed=pursuer_max_speed,
                            max_acceleration=execution_limit, continuous_speed=True, continuous_acceleration=True)
                        if stop['feasible'] and LocalSnapshot(maps[aid]).path_safe(stop['positions']):
                            braking_tails[aid] = sample_quintic_at(stop['coefficients'], np.minimum(capture_times, duration))
                            if duration > 1.:
                                braking_remainders[aid] = sample_quintic_at(stop['coefficients'], np.minimum(1.+capture_times, duration))
                            break
                elif not truncated and plans[aid].duration > 1.:
                    # The second half of the previously accepted curve ends at rest.
                    tail_times = np.minimum(1. + capture_times, plans[aid].duration)
                    braking_tails[aid] = sample_quintic_at(track['coefficients'], tail_times)
                minco_records[str(aid)]['continued_braking_tail'] = used_tail
                minco_records[str(aid)]['pass_through'] = plans[aid].pass_through and not used_tail
                minco_records[str(aid)]['executed_sampled_peak_speed'] = (
                    None if truncated else float(np.linalg.norm(track['velocities'][:midpoint+1], axis=1).max()))
            velocities[aid] = track["velocities"][midpoint] if e52 and not truncated else np.zeros(2)
            accelerations[aid] = track["accelerations"][midpoint] if e52 and not truncated else np.zeros(2)
            minco_records[str(aid)]["executed_positions"] = track["positions"][:midpoint+1].tolist() if not truncated else [executed.tolist()]
            if max_acceleration is not None:
                minco_records[str(aid)].update(
                    without_search_candidate=plans[aid].candidate_id < 0,
                    speed_reset=incoming_speed if truncated else 0.,
                    executed_sampled_peak_acceleration=(None if truncated else
                        float(np.linalg.norm(track['accelerations'][:midpoint+1], axis=1).max())))
            displacement = executed - pursuers[aid]
            if float(np.linalg.norm(displacement)) > 1e-6:
                headings[aid] = float(math.atan2(displacement[1], displacement[0]))
            pursuers[aid] = executed
            prefix_truncations += int(truncated)
        if opponent is not None:
            if not evader_minco:
                np.asarray(state).reshape(-1, 1)[4:8, 0] = pursuers.ravel()
            if rootpuct:
                projected, targets = _project_target_state(env, state, targets)
                target_state_projections += int(projected)
            state_column = np.asarray(state).reshape(-1, 1)
            evader_active = [env.is_active(state_column, robot) for robot in env.evaders]
            if float(np.asarray(state).reshape(-1)[env.time_idx]) >= float(env.tf):
                evader_action = np.zeros((2, 2), dtype=float)
                terminal_opponent_skips += 1
            elif any(evader_active):
                try:
                    if evader_hmcts:
                        opponent.set_camera_state(trackers, [observations[i].heading for i in (0, 1)], camera_visible)
                        opponent.set_execution_state(velocity=evader_velocities,
                                                     acceleration=evader_accelerations)
                    searched = opponent.search(env, state.reshape(-1, 1).copy(), turn=0)
                    evader_action = np.asarray(searched.best_action[:4], float).reshape(2, 2)
                    if evader_hmcts:
                        evader_search = dict(getattr(searched, 'metadata', opponent.last_search))
                except Exception as exc:
                    if not rootpuct:
                        raise
                    rootpuct_failures += 1
                    rootpuct_error_cycles.append(dict(cycle=int(cycle), error=repr(exc),
                                                      active=[bool(x) for x in evader_active],
                                                      state_active=np.asarray(state)[env.active_idxs].reshape(-1).tolist(),
                                                      state_time=float(np.asarray(state).reshape(-1)[env.time_idx]),
                                                      targets=targets.tolist(), pursuers=pursuers.tolist()))
                    evader_action = np.zeros((2, 2), dtype=float)
                    for tid, is_active in enumerate(evader_active):
                        if is_active:
                            evader_action[tid] = _single_target_action(targets[tid], pursuers, boxes)
                if not all(evader_active):
                    single_active_model_steps += 1
            else:
                evader_action = np.zeros((2, 2), dtype=float)
            if evader_minco:
                for tid, is_active in enumerate(evader_active):
                    if is_active:
                        prefix, mode = evader_executor.prefix(tid, targets[tid], evader_action[tid], waypoint=evader_hmcts, cruise_track=(opponent.selected_cruise[tid] if evader_cruise else None))
                        evader_paths[:, tid] = prefix['positions']
                        evader_velocities[tid] = prefix['velocities'][-1]
                        evader_accelerations[tid] = prefix['accelerations'][-1]
                        evader_records[str(tid)] = dict(mode=mode, **{k:v.tolist() for k,v in prefix.items()})
                state = env.step_trajectories(state, pursuer_paths, evader_paths)
            else:
                # Pursuers have already advanced. Only apply evader displacements.
                full = np.zeros((env.action_dim, 1), dtype=float)
                full[env.action_idxs[0], 0] = evader_action[0]
                full[env.action_idxs[1], 0] = evader_action[1]
                state = env.step(state, full, 1.)
            for event in getattr(env, 'last_capture_events', []):
                tid = int(event['evader'])
                if tid not in captured_targets:
                    captured_targets.add(tid)
                    capture = dict(cycle=cycle, pursuer=int(event['pursuer']), target=tid,
                                   source=opponent_policy)
                    if evader_minco:
                        capture['time'] = event['time']
                        record = evader_records[str(tid)]
                        for key in ('positions', 'velocities', 'accelerations'):
                            record[key] = record[key][:event['sample'] + 1]
                        evader_executor.tail.pop(tid, None)
                    captures.append(capture)
                    for belief in beliefs:
                        if e52: belief.capture(tid)
                        else:
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
        for aid in (() if evader_minco else (0, 1)):
            for tid in range(2):
                if active_e[tid] and np.linalg.norm(pursuers[aid] - targets[tid]) < .45:
                    active_e[tid] = False
                    if tid not in captured_targets:
                        captured_targets.add(tid)
                        captures.append(dict(cycle=cycle, pursuer=aid, target=tid))
                        for belief in beliefs:
                            if e52: belief.capture(tid)
                            else:
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
        if e52:
            # World geometry is used here only for the evaluation metric.
            for record in minco_records.values():
                points = np.array(record["executed_positions"])
                if len(boxes) and np.any(((points[:, None] >= boxes[None, :, :, 0]) & (points[:, None] <= boxes[None, :, :, 1])).all(axis=-1)):
                    pursuer_collision = True
        prediction_metrics = {}
        # Compare the one-step predictions used by each search with the
        # physical state after this cycle.  Pursuer beliefs are unlabeled, so
        # report nearest-target errors and threshold hit rates.
        for aid in (0, 1):
            predicted = controllers[aid].last_search.get('predicted_targets', []) if e52 else []
            predicted = np.asarray(predicted, float).reshape(-1, 2)
            if len(predicted) and len(targets):
                errors = np.min(np.linalg.norm(predicted[:, None] - targets[None, :, :], axis=2), axis=1)
                prediction_metrics[f'pursuer_{aid}_target'] = dict(
                    count=int(len(errors)), mean_m=float(np.mean(errors)),
                    median_m=float(np.median(errors)),
                    within_1m=float(np.mean(errors <= 1.)),
                    within_2m=float(np.mean(errors <= 2.)),
                    predicted=predicted.tolist())
        if evader_hmcts and evader_search.get('predicted_pursuers') is not None:
            predicted = np.asarray(evader_search['predicted_pursuers'], float).reshape(2, 2)
            errors = np.linalg.norm(predicted - pursuers, axis=1)
            prediction_metrics['evader_pursuer'] = dict(
                count=int(len(errors)), mean_m=float(np.mean(errors)),
                median_m=float(np.median(errors)),
                within_1m=float(np.mean(errors <= 1.)),
                within_2m=float(np.mean(errors <= 2.)),
                predicted=predicted.tolist())
        logs.append(dict(cycle=cycle, delivered=int(delivered), captures=list(captures),
                         pursuer_collision=bool(pursuer_collision), minco=minco_records,
                         sensor_stats=sensor_stats,
                         coordination=coordination,
                         prediction_metrics=prediction_metrics,
                         search={str(a): controllers[a].last_search for a in (0,1)} if e52 else {},
                         beliefs={str(a): beliefs[a].payload() for a in (0,1)} if e52 else {},
                         plans={str(a): dict(endpoint=plans[a].endpoint.tolist(),
                                             action=plans[a].action.tolist(),
                                             value=float(plans[a].value), duration=plans[a].duration, visits=int(plans[a].visits), candidate_id=int(plans[a].candidate_id)) for a in (0,1)},
                         pursuers=pursuers.tolist(), targets=targets.tolist(),
                         peer_ages={str(a): mailboxes[a].ages(cycle) for a in (0,1)}))
        if evader_minco:
            logs[-1]['evader_minco'] = evader_records
        if evader_hmcts:
            logs[-1]['evader_search'] = evader_search
        publish_progress(cycle+1)
        if len(captured_targets) == len(targets):
            break
    result = dict(stage='e50_distributed_dynamic_smoke', case=str(case_path), cycles=len(logs), requested_cycles=int(cycles),
                  e52_nae_strategy=e52_nae_strategy,
                  e52_advantage_checkpoint=str(e52_advantage_checkpoint) if e52_advantage_checkpoint else None,
                  terminal=bool(len(captured_targets)==len(targets)),
                  lidar_rays=int(lidar_rays), delay=int(delay), drop_cycles=list(drop_cycles),
                  rootpuct=bool(rootpuct), rootpuct_simulations=int(rootpuct_simulations) if opponent_policy != 'geometric' else None,
                  opponent_policy=opponent_policy,
                  evader_speed_per_axis=None if env is None or evader_minco else env.current_evader_speed_lim,
                  evader_minco=bool(evader_minco),
                  pursuer_max_speed=pursuer_max_speed, shared_max_speed=shared_max_speed,
                  evader_max_speed=geometric_evader_speed if evader_minco else None,
                  max_acceleration=execution_limit, continuous_acceleration=max_acceleration is not None,
                  environment_horizon=int(env.tf) if env is not None else None,
                  leaf_value_checkpoint=None if leaf_value_checkpoint is None else str(leaf_value_checkpoint),
                  e52=bool(e52), e52_mode=e52_mode, seed=int(seed), e52_depth=e52_depth, e52_budget=e52_budget, e52_time_ms=e52_time_ms, e52_checkpoint=None if e52_checkpoint is None else str(e52_checkpoint),
                  e53_depth=int(e53_depth), e53_simulations=int(e53_simulations), evader_window=int(evader_window), evader_cruise=evader_cruise,
                  pursuer_structure=pursuer_structure,
                  rootpuct_failures=int(rootpuct_failures),
                  rootpuct_error_cycles=rootpuct_error_cycles,
                  partial_active_fallbacks=int(partial_active_fallbacks),
                  single_active_model_steps=int(single_active_model_steps),
                  target_state_projections=int(target_state_projections),
                  terminal_opponent_skips=int(terminal_opponent_skips),
                  prefix_truncations=int(prefix_truncations),
                  truth_used_only_sensor=(pursuer_structure not in ("joint_oracle", "joint_global")), captures=captures, logs=logs)
    output = Path(output); output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    publish_progress(len(logs), state="complete")
    return result


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--case', required=True); p.add_argument('--output', required=True)
    p.add_argument('--cycles', type=int, default=20); p.add_argument('--lidar-rays', type=int, default=256)
    p.add_argument('--delay', type=int, default=0); p.add_argument('--drop-cycle', type=int, action='append', default=[])
    p.add_argument('--rootpuct', action='store_true'); p.add_argument('--rootpuct-simulations', type=int, default=20)
    p.add_argument('--geometric-evader-speed', type=float)
    p.add_argument('--evader-minco', action='store_true',
                   help='execute geometric evader actions as rolling MINCO trajectories')
    p.add_argument('--evader-hmcts', action='store_true',
                   help='use the privileged sparse-reward H-MCTS evader baseline')
    p.add_argument('--evader-cruise', action='store_true', help='include speed-conditioned cruise waypoints in evader H-MCTS')
    p.add_argument('--evader-window', type=int, default=8)
    p.add_argument('--e53-depth', type=int, default=3)
    p.add_argument('--e53-simulations', type=int, default=256)
    p.add_argument('--pursuer-structure', choices=('local', 'coordinated', 'joint_oracle', 'joint_shared', 'joint_global'), default='local')
    p.add_argument('--shared-max-speed', type=float, help='planar speed limit for both MINCO teams (m/s)')
    p.add_argument('--max-acceleration', type=float,
                   help='shared continuous planar acceleration cap in m/s² (requires E52 and --evader-minco)')
    p.add_argument('--leaf-value-checkpoint', default=None)
    p.add_argument('--e52', action='store_true'); p.add_argument('--e52-checkpoint', default=None)
    from solvers.e52_hmcts import HierarchicalMCTS
    p.add_argument('--e52-nae-strategy', choices=HierarchicalMCTS.NAE_STRATEGIES, default='legacy')
    a = p.parse_args()
    result = run(a.case, a.output, a.cycles, a.lidar_rays, delay=a.delay, drop_cycles=a.drop_cycle,
                 rootpuct=a.rootpuct, rootpuct_simulations=a.rootpuct_simulations,
                 leaf_value_checkpoint=a.leaf_value_checkpoint, e52=a.e52,
                 e52_checkpoint=a.e52_checkpoint, e52_nae_strategy=a.e52_nae_strategy,
                 geometric_evader_speed=a.geometric_evader_speed, evader_minco=a.evader_minco,
                 max_acceleration=a.max_acceleration, shared_max_speed=a.shared_max_speed,
                 evader_hmcts=a.evader_hmcts, evader_window=a.evader_window, evader_cruise=a.evader_cruise,
                 e53_depth=a.e53_depth, e53_simulations=a.e53_simulations,
                 pursuer_structure=a.pursuer_structure)
    print(json.dumps(result, ensure_ascii=False))
