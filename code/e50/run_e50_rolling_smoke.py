"""Two-cycle rolling endpoint-trajectory smoke for existing E49 controllers.

This is an execution continuity diagnostic: the pursuer executes the first
second of a two-second endpoint trajectory and replans. Evaders remain at the
frozen diagnostic state, so no scientific capture claim is made.
"""
import sys
from pathlib import Path
_CODE_DIR = Path(__file__).resolve().parents[1]
if str(_CODE_DIR) not in sys.path:
    sys.path.insert(0, str(_CODE_DIR))
import repo_paths
repo_paths.install()
import argparse
import json
import traceback
from pathlib import Path
import numpy as np

METHODS = ('A2', 'A3', 'A4', 'AG', 'GEOM_SEARCH', 'RouteAPF')


def _collision_prefix(positions, boxes, end):
    """Collision flag restricted to the actually executed prefix."""
    points = np.asarray(positions, dtype=float)[:int(end) + 1]
    boxes = np.asarray(boxes, dtype=float).reshape(-1, 2, 2)
    if not len(boxes):
        return False
    inside = ((points[:, :, None, :] >= boxes[None, None, :, :, 0]) &
              (points[:, :, None, :] <= boxes[None, None, :, :, 1]))
    return bool(np.any(inside.all(axis=-1)))


def _apply_evader_step(position, requested, boxes):
    """Apply only a legal evader step and report what was actually applied."""
    position = np.asarray(position, dtype=float)
    requested = np.asarray(requested, dtype=float)
    candidate = position + requested
    legal = (np.all(candidate >= -10.) and np.all(candidate <= 10.) and
             not any(np.all(candidate >= box[:, 0]) and
                     np.all(candidate <= box[:, 1]) for box in boxes))
    return (requested.copy(), True) if legal else (np.zeros(2, dtype=float), False)


def _mask_history_observation(frame, boxes, *, lidar_obstacles=None, lidar_rays=256,
                              target_radius=.25):
    """Backward-compatible history audit helper.

    The rolling camera path does not call this function; it remains for old
    LiDAR bridge tests and historical replay tools.
    """
    from solvers.frozen_d_search_interface import sensor_visible
    from solvers.e50_lidar_map import lidar_target_visibility
    result = dict(frame)
    positions = np.asarray(frame['positions'], dtype=float)
    visibility = None
    if lidar_obstacles is not None:
        targets = np.full((len(frame.get('active_e', [])), 2), np.nan)
        for j in range(len(targets)):
            for row in frame.get('reports', []):
                if j < len(row) and row[j] is not None:
                    targets[j] = row[j]; break
        known = np.flatnonzero(np.isfinite(targets).all(axis=1))
        visibility = np.zeros((len(positions), len(targets)), bool)
        if len(known):
            visibility[:, known] = lidar_target_visibility(
                positions, targets[known], lidar_obstacles,
                num_rays=int(lidar_rays), target_radius=float(target_radius))
    reports = []
    for j, row in enumerate(frame.get('reports', [])):
        filtered = []
        for i, report in enumerate(row):
            if report is None:
                filtered.append(None); continue
            keep = (bool(visibility[i, j]) if visibility is not None else
                    sensor_visible(positions[i], np.asarray(report, float), boxes))
            filtered.append(report if keep else None)
        reports.append(filtered)
    result['reports'] = reports
    if lidar_obstacles is not None:
        result['visibility_model'] = 'lidar'
    return result



def install_camera_belief_hooks():
    # E49 installs the historical G/W belief classes process-locally.
    # Redirect their imported negative-observation predicate for this
    # deployment run; the frozen known-map benchmark keeps its original
    # continuous map-line-of-sight semantics.
    from solvers.e45_team_adapter import team_hidden_mask_camera
    import learning.e45g_ppo as _g_belief
    import learning.e45w_support_recovery as _w_belief
    import e45c_belief_diagnostics as _diagnostic
    import solvers.e45f_support_refinement as _refinement
    import solvers.corner_estimator as _corner_estimator
    _g_belief.team_hidden_mask = team_hidden_mask_camera
    _w_belief.team_hidden_mask = team_hidden_mask_camera
    _diagnostic.team_hidden_mask = team_hidden_mask_camera
    _diagnostic.sensor_visible = lambda *args, **kwargs: True
    _refinement.team_hidden_mask = team_hidden_mask_camera
    # E49 may clone the diagnostic update function with a copied globals
    # dictionary; redirect that copy as well for camera FOV negatives.
    if hasattr(_diagnostic, 'RecordedTeamBelief'):
        _diagnostic.RecordedTeamBelief.update_team.__globals__['team_hidden_mask'] = team_hidden_mask_camera
    # A positive LiDAR return has already passed the physical occlusion
    # test.  Do not recheck it with the legacy local-map line-of-sight
    # predicate inside the frozen estimator.
    _corner_estimator.visible = lambda *args, **kwargs: True
    _corner_estimator.LIDAR_MODE = True


def run(case_path, output, cycles=2, pursuer_speed=1.5, pursuer_acceleration=6., unknown_map=False,
        command_delay=0, drop_cycles=(), pose_delay=0, moving_evader=False,
        lidar_rays=256, lidar_inflation=.05, grid_inflation=.05,
        lidar_target_radius=.25,
        rootpuct_evader=False, rootpuct_simulations=100, rootpuct_seed=17,
        methods=None, cross_cycle_smoothing=False, max_action_delta=1.2,
        geometry_evader=False, gate_full_horizon=False, fixed_simulations=2):
    from e49_controllers import install, make_controller
    from run_e49_benchmark import observation
    from e49_rules import create_environment
    from solvers.e50_compat_smoke import replay_endpoint_semantic
    from solvers.e50_minco_audit import obstacle_aware_minco_action
    from solvers.e50_lidar_map import LidarPlanningMap, raycast_scan
    from solvers.e50_local_obstacle import LocalObstacleProxy
    from solvers.e50_local_obstacle import project_action_to_local_scan
    from solvers.e50_camera import CameraGimbalTracker
    from solvers.e50_local_scan_memory import LocalScanMemory
    from solvers.e50_trajectory_smoothing import rate_limit_displacement
    if unknown_map:
        install_camera_belief_hooks()
    opponent_factory = None
    if rootpuct_evader:
        from e49_rules import make_opponent as opponent_factory
    geometry_factory = None
    if geometry_evader:
        from solvers.e49_geometric_evader import GeometricEvader as geometry_factory

    case_path, output = Path(case_path), Path(output)
    case = json.loads(case_path.read_text())
    truth_boxes = np.asarray(case['map'], float).reshape(-1, 2, 2)
    controller_boxes = np.zeros((0, 2, 2), dtype=float) if unknown_map else truth_boxes
    controller_case = dict(case, map=controller_boxes.tolist()) if unknown_map else case
    horizon = int(case.get('horizon', 80))
    install(horizon, output.parent)
    if unknown_map:
        # Patch after E49's process-local source cloning.  The loader uses the
        # frozen runtime namespace, and LiDAR-positive measurements must not be
        # rejected by its local-map legal/visibility checks.
        import importlib
        frozen_corner = importlib.import_module('l45_corner_shared_20261025.corner_estimator')
        if not getattr(frozen_corner, '_e50_lidar_update', False):
            original_update = frozen_corner.Estimator.update
            # E49 may have replaced this method with a cloned function whose
            # globals are a copied dictionary.  Patch that dictionary too;
            # changing the module attributes alone does not reach it.
            original_update.__globals__['legal'] = lambda *args, **kwargs: True
            original_update.__globals__['visible'] = lambda *args, **kwargs: True
            def lidar_update(self, sensor_obs, _corner=frozen_corner, _orig=original_update):
                if sensor_obs.target is None:
                    return _orig(self, sensor_obs)
                old_legal, old_visible = _corner.legal, _corner.visible
                _corner.legal = lambda *args, **kwargs: True
                _corner.visible = lambda *args, **kwargs: True
                try:
                    return _orig(self, sensor_obs)
                finally:
                    _corner.legal, _corner.visible = old_legal, old_visible
            frozen_corner.Estimator.update = lidar_update
            frozen_corner._e50_lidar_update = True
        # The frozen update function resolves these names from its module
        # globals (including the E49-cloned variant).  Keep the LiDAR mode
        # active for the whole process so a local planner patch cannot reject
        # an already identified target on a later reacquisition.
        frozen_corner.legal = lambda *args, **kwargs: True
        frozen_corner.visible = lambda *args, **kwargs: True
    env = create_environment(truth_boxes, horizon=horizon)
    env.current_pursuer_speed_lim = float(pursuer_speed)
    env.update_action_lims()
    for robot in env.pursuers:
        env.state_lims[env.acc_idxs[robot], :] = [[-pursuer_acceleration, pursuer_acceleration]] * 2
    rows = []
    command_delay = int(command_delay)
    drop_cycles = tuple(int(x) for x in drop_cycles)
    pose_delay = int(pose_delay)
    if command_delay < 0 or pose_delay < 0 or any(x < 0 for x in drop_cycles):
        raise ValueError('invalid communication perturbation')
    selected_methods = tuple(METHODS if methods is None else methods)
    for method in selected_methods:
        task = dict(arm=method, seed=17, train_seed=17, budget_ms=200., fixed=int(fixed_simulations),
                    horizon=horizon, out=str(output.parent))
        try:
            # Known-map runs keep the frozen global geometry.  Unknown-map
            # runs rebuild the legacy geometry adapter from the LiDAR memory
            # before every decision, so the controller plans on measured local
            # obstacle boxes instead of an empty map.
            controller = None
            if not unknown_map:
                controller = make_controller(controller_case, task)
            state = np.asarray(case['worlds'][0]['initial'], float).reshape(-1, 1)
            if controller is not None:
                for frame in case['history'][:-1]:
                    controller.observe(frame)
            cycles_out = []
            pending = [[] for _ in range(2)]
            state_history = []
            previous_action = None
            headings = np.zeros(2, dtype=float)
            camera_tracker = CameraGimbalTracker(2, fov_deg=60., gimbal_limit_deg=90.,
                                                 sweep_rate_deg_s=960., max_range=20.)
            camera_public_history = []
            # Rebuild the public prehistory through the simulated camera.  The
            # truth states are used only inside the sensor simulator.
            scripted = case.get('worlds', [{}])[0].get('scripted_states', [])
            for index, frame in enumerate(case['history'][:-1]):
                if index < len(scripted):
                    hs = np.asarray(scripted[index], float).reshape(-1, 1)
                    hp = hs[4:8, 0].reshape(2, 2)
                    ht = hs[:4, 0].reshape(2, 2)
                    ha = hs[11:13, 0] > .5
                    he = hs[9:11, 0] > .5
                    hh = np.zeros(2, dtype=float)
                    if index > 0:
                        prev = np.asarray(scripted[index - 1], float).reshape(-1, 1)[4:8, 0].reshape(2, 2)
                        delta = hp - prev
                        for k in range(2):
                            if np.linalg.norm(delta[k]) > 1e-8:
                                hh[k] = np.arctan2(delta[k, 1], delta[k, 0])
                    hf = camera_tracker.step(hp, ht, ha, he, hh, truth_boxes, dt=1.)
                    # The legacy E49 estimator requires a positive support
                    # bootstrap.  The strict unknown-map distributed runner
                    # uses its own public prior path; keep this compatibility
                    # entrypoint's historical bootstrap unchanged.
                    hist = dict(frame, visibility_model='camera_fov', camera=dict(
                        gimbal_yaw=hf.gimbal_yaw.tolist(), headings=hh.tolist(),
                        fov_rad=0., max_range=20., state=list(hf.state),
                        lost_reason=list(hf.lost_reason), history_bootstrap=True))
                    camera_public_history.append(hist)
                else:
                    camera_public_history.append(dict(frame, visibility_model='camera_fov',
                                                      camera=dict(gimbal_yaw=[0., 0.], headings=[0., 0.],
                                                                  fov_rad=np.deg2rad(60.), max_range=20.,
                                                                  state=['SEARCH', 'SEARCH'], lost_reason=[None, None])))
            scan_memory = [LocalScanMemory(resolution=.1, max_age=2.) for _ in range(2)]
            # Shared map-frame LiDAR hit memory is the planner input.  The
            # per-robot memories above remain for the independent safety gate.
            planning_map = LidarPlanningMap(
                max_age=2., max_points=512, inflation=float(lidar_inflation),
                max_boxes=24, merge_distance=.25)
            if unknown_map:
                # Seed the local map from the public prehistory.  Only range
                # returns are retained; the controller never receives the
                # global obstacle list.
                for frame in case['history'][:-1]:
                    stamp = float(frame.get('step', 0))
                    for local, position in enumerate(frame['positions']):
                        scan = raycast_scan(position, truth_boxes, frame_id='map',
                                            stamp=stamp, num_rays=int(lidar_rays),
                                            max_range=10.)
                        planning_map.update(position, scan)
                        scan_memory[local].update(position, scan)
            if geometry_evader:
                opponent = geometry_factory()
            else:
                opponent = opponent_factory(env, int(rootpuct_seed), simulations=int(rootpuct_simulations)) \
                    if rootpuct_evader else None
            rootpuct_failures = 0
            for cycle in range(int(cycles)):
                state_history.append(state.copy())
                observed_state = state_history[max(0, len(state_history) - 1 - pose_delay)].copy()
                # Keep the decision clock current while positions/velocities
                # come from the delayed pose packet.
                observed_state[env.time_idx, 0] = state[env.time_idx, 0]
                # The simulated LiDAR is the only geometry source in the
                # unknown-map path.  Build a conservative rectangle adapter
                # from occupied hit cells before asking the controller to act.
                planning_scans = []
                planning_proxies = []
                if unknown_map:
                    for local, robot in enumerate(env.pursuers):
                        start = observed_state[env.state_idxs[robot], 0]
                        scan = raycast_scan(start, truth_boxes, frame_id='map',
                                            stamp=float(state[env.time_idx, 0]),
                                            num_rays=int(lidar_rays), max_range=10.)
                        planning_scans.append(scan)
                        scan_memory[local].update(start, scan)
                        planning_map.update(start, scan)
                        planning_proxies.append(LocalObstacleProxy.from_scan(
                            start, scan, inflation=float(lidar_inflation)))
                    # Convert only measured range returns into the legacy
                    # rectangle map interface.  Max-range rays and unknown
                    # cells do not become free-space or obstacle tokens.
                    local_boxes = planning_map.obstacle_boxes()
                    controller_case_cycle = dict(case, map=local_boxes.tolist())
                    try:
                        if method == 'RouteAPF':
                            # RouteAPF's frozen belief anchor is map-global;
                            # local LiDAR patches can invalidate it during
                            # history replay. Keep its compatibility model
                            # empty and retain measured geometry in the
                            # downstream MINCO/LiDAR safety layer.
                            controller_case_cycle = dict(case, map=[])
                        controller = make_controller(controller_case_cycle, task)
                    except RuntimeError as exc:
                        # RouteAPF's frozen public-state lattice requires its
                        # belief anchor to lie exactly on a free local-map
                        # cell.  A sparse LiDAR patch can transiently cover
                        # that anchor even though the robot itself is legal.
                        # Retry with an empty compatibility map; the live
                        # LiDAR safety/MINCO layer remains active and records
                        # the conservative fallback instead of aborting the
                        # whole long-horizon audit.
                        if method != 'RouteAPF' or 'No lawful public anchor' not in str(exc):
                            raise
                        controller_case_cycle = dict(case, map=[])
                        controller = make_controller(controller_case_cycle, task)
                    for previous_obs in camera_public_history:
                        controller.observe(previous_obs)
                    planning_boxes = local_boxes
                else:
                    planning_boxes = controller_boxes
                if unknown_map:
                    camera_frame = camera_tracker.step(
                        observed_state[env.state_idxs[2:4], 0].reshape(2, 2),
                        observed_state[env.state_idxs[0:2], 0].reshape(2, 2),
                        observed_state[env.active_idxs[2:4], 0] > .5,
                        observed_state[env.active_idxs[0:2], 0] > .5,
                        headings, truth_boxes, dt=1.)
                    obs = observation(observed_state, planning_boxes, horizon,
                                      camera_frame=camera_frame, headings=headings)
                else:
                    camera_frame = None
                    obs = observation(observed_state, planning_boxes, horizon)
                controller_action = np.asarray(controller.act(obs, fixed_simulations=int(fixed_simulations)), float)
                action = controller_action.copy()
                if cross_cycle_smoothing and previous_action is not None:
                    action = rate_limit_displacement(action, previous_action,
                                                     max_delta=float(max_action_delta))
                minco_obstacle_plan = dict(applied=False, reason='known_map')
                if unknown_map:
                    planner_action = np.zeros((env.action_dim, 1))
                    planner_action[env.action_idxs[2], 0] = action[0]
                    planner_action[env.action_idxs[3], 0] = action[1]
                    planner_action, minco_obstacle_plan = obstacle_aware_minco_action(
                        env, state, planner_action, planning_boxes, dt=2., sample_dt=.02, robots=(2, 3))
                    action = np.asarray([planner_action[env.action_idxs[2], 0],
                                         planner_action[env.action_idxs[3], 0]], dtype=float)
                requested = action.copy()
                if cycle in drop_cycles:
                    delivered = np.zeros_like(action)
                    comm_event = 'packet_dropped'
                else:
                    for robot in range(2):
                        pending[robot].append((cycle + command_delay, action[robot].copy()))
                    delivered = np.zeros_like(action)
                    for robot in range(2):
                        due = [x for x in pending[robot] if x[0] <= cycle]
                        pending[robot] = [x for x in pending[robot] if x[0] > cycle]
                        if due:
                            delivered[robot] = due[-1][1]
                    comm_event = 'command_delivered' if np.any(delivered) else 'stop_degraded'
                # Smooth against what actually reached the executor, not a
                # command that may have been delayed or dropped in transit.
                previous_action = delivered.copy()
                if not np.any(state[env.active_idxs[0:2], 0] > .5):
                    comm_event = 'post_capture_stop'
                full = np.zeros((env.action_dim, 1))
                full[env.action_idxs[2], 0] = delivered[0]
                full[env.action_idxs[3], 0] = delivered[1]
                audit, feasible = replay_endpoint_semantic(env, state, full, dt=2., robots=(2, 3), sample_dt=.02)
                truth_plan_collision = bool(audit.collision)
                midpoint = int(np.argmin(np.abs(audit.times - 1.)))
                truth_prefix_collision = _collision_prefix(audit.positions, truth_boxes, midpoint)
                # In unknown-map mode the truth map is audit-only. The
                # execution gate may use only the current LiDAR proxies.
                full_lidar_safe = True
                prefix_lidar_safe = True
                lidar_proxies = []
                for local, robot in enumerate(env.pursuers):
                    start = state[env.state_idxs[robot], 0]
                    scan = raycast_scan(start, truth_boxes, frame_id='map',
                                        stamp=float(state[env.time_idx, 0]),
                                        num_rays=int(lidar_rays), max_range=10.)
                    proxy = LocalObstacleProxy.from_scan(start, scan, inflation=float(lidar_inflation))
                    lidar_proxies.append(proxy)
                    scan_memory[local].update(start, scan)
                    frame_full_safe = bool(all(
                        proxy.endpoint_safe(point, interpolation=bool(unknown_map))
                        for point in audit.positions[:, local]))
                    frame_prefix_safe = bool(all(
                        proxy.endpoint_safe(point, interpolation=bool(unknown_map))
                        for point in audit.positions[:midpoint + 1, local]))
                    if unknown_map:
                        frame_full_safe = frame_full_safe and scan_memory[local].grid_trajectory_safe(
                            audit.positions[:, local], inflation=float(grid_inflation))
                        frame_prefix_safe = frame_prefix_safe and scan_memory[local].grid_trajectory_safe(
                            audit.positions[:midpoint + 1, local], inflation=float(grid_inflation))
                    full_lidar_safe = full_lidar_safe and frame_full_safe
                    prefix_lidar_safe = prefix_lidar_safe and frame_prefix_safe
                # Replanning occurs at the one-second prefix.  A collision in
                # the unexecuted two-second tail is diagnostic only unless the
                # caller explicitly requests the old full-horizon gate.
                gate_lidar_safe = full_lidar_safe if gate_full_horizon else prefix_lidar_safe
                # LiDAR is planning input and audit input only.  It no longer
                # overrides the method action or inserts a stop command.
                requested_plan_collision = False
                safety_override = None
                lidar_safe = []
                for local, robot in enumerate(env.pursuers):
                    start = state[env.state_idxs[robot], 0]
                    scan = raycast_scan(start, truth_boxes, frame_id='map', stamp=float(state[env.time_idx, 0]),
                                        num_rays=int(lidar_rays), max_range=10.)
                    proxy = LocalObstacleProxy.from_scan(start, scan, inflation=float(lidar_inflation))
                    prefix_safe = bool(all(proxy.endpoint_safe(point, interpolation=bool(unknown_map))
                                           for point in audit.positions[:midpoint+1, local]))
                    if unknown_map:
                        prefix_safe = prefix_safe and scan_memory[local].grid_trajectory_safe(
                            audit.positions[:midpoint+1, local], inflation=float(grid_inflation))
                    lidar_safe.append(prefix_safe)
                # If the measured LiDAR corridor rejects the actually
                # executable prefix, do not send the unsafe action onward.
                # Re-evaluate a zero command through the same trajectory
                # backend so collision and distance accounting reflect what
                # the vehicle really executes.
                if unknown_map and not prefix_lidar_safe:
                    starts = np.asarray([state[env.state_idxs[r], 0] for r in env.pursuers]).reshape(2, 2)
                    projected = project_action_to_local_scan(starts, delivered, lidar_proxies)
                    safety_override = 'local_action_projection' if np.any(np.abs(projected) > 1e-9) else 'stop_lidar_prefix'
                    delivered = projected
                    full = np.zeros((env.action_dim, 1))
                    full[env.action_idxs[2], 0] = delivered[0]
                    full[env.action_idxs[3], 0] = delivered[1]
                    audit, feasible = replay_endpoint_semantic(
                        env, state, full, dt=2., robots=(2, 3), sample_dt=.02)
                    truth_plan_collision = bool(audit.collision)
                    truth_prefix_collision = _collision_prefix(audit.positions, truth_boxes, midpoint)
                for local, robot in enumerate(env.pursuers):
                    state[env.state_idxs[robot], 0] = audit.positions[midpoint, local]
                    state[env.vel_idxs[robot], 0] = audit.velocities[midpoint, local]
                    state[env.acc_idxs[robot], 0] = audit.accelerations[midpoint, local]
                    if np.linalg.norm(audit.velocities[midpoint, local]) > 1e-8:
                        headings[local] = np.arctan2(audit.velocities[midpoint, local, 1],
                                                     audit.velocities[midpoint, local, 0])
                capture_events = []
                rootpuct_evader_step = None
                rootpuct_evader_requested_step = None
                if moving_evader:
                    # Deterministic legal evader motion for task-layer smoke;
                    # it is not the frozen E49 opponent benchmark.
                    from e49_rules import capture_pairs
                    pursuer_positions = audit.positions[midpoint]
                    evader_positions = state[:4, 0].reshape(2, 2).copy()
                    initial_pairs = capture_pairs(
                        pursuer_positions, evader_positions,
                        state[env.active_idxs[2:4], 0] > .5,
                        state[env.active_idxs[0:2], 0] > .5, truth_boxes)
                    for pursuer, evader in initial_pairs:
                        state[env.active_idxs[evader], 0] = 0.
                        capture_events.append(dict(pursuer=int(pursuer), evader=int(evader), phase='cycle_start'))
                    if geometry_evader:
                        try:
                            raw = np.asarray(opponent.search(env, state.copy(), turn=0).best_action,
                                             dtype=float).reshape(-1)[:4].reshape(2, 2)
                        except Exception:
                            rootpuct_failures += 1
                            raw = np.zeros((2, 2), dtype=float)
                        for evader in range(2):
                            if state[env.active_idxs[evader], 0] <= .5:
                                continue
                            step = np.clip(raw[evader], -1.5, 1.5)
                            applied, legal = _apply_evader_step(
                                evader_positions[evader], step, truth_boxes)
                            if legal:
                                evader_positions[evader] += applied
                            if rootpuct_evader_step is None:
                                rootpuct_evader_step = []
                                rootpuct_evader_requested_step = []
                            rootpuct_evader_requested_step.append(step.tolist())
                            rootpuct_evader_step.append(applied.tolist())
                    else:
                        for evader in range(2):
                            if state[env.active_idxs[evader], 0] <= .5:
                                continue
                            if opponent is not None:
                                try:
                                    raw = np.asarray(opponent.search(env, state.copy(), turn=0).best_action,
                                                      dtype=float).reshape(-1)[:4].reshape(2, 2)
                                except Exception:
                                    rootpuct_failures += 1
                                    raw = np.zeros((2, 2), dtype=float)
                                step = np.clip(raw[evader], -1.5, 1.5)
                            else:
                                delta = evader_positions[evader] - pursuer_positions[evader % 2]
                                norm = float(np.linalg.norm(delta))
                                step = (1.5 * delta / norm) if norm > 1e-9 else np.zeros(2)
                            applied, legal = _apply_evader_step(
                                evader_positions[evader], step, truth_boxes)
                            if legal:
                                evader_positions[evader] += applied
                            if rootpuct_evader_step is None:
                                rootpuct_evader_step = []
                                rootpuct_evader_requested_step = []
                            rootpuct_evader_requested_step.append(step.tolist())
                            rootpuct_evader_step.append(applied.tolist())
                    state[:4, 0] = evader_positions.ravel()
                    pairs = capture_pairs(pursuer_positions, evader_positions,
                                           state[env.active_idxs[2:4], 0] > .5,
                                           state[env.active_idxs[0:2], 0] > .5, truth_boxes)
                    for pursuer, evader in pairs:
                        state[env.active_idxs[evader], 0] = 0.
                        capture_events.append(dict(pursuer=int(pursuer), evader=int(evader)))
                state[env.time_idx, 0] += 1.
                pursuer_positions = state[4:8, 0].reshape(2, 2)
                target_positions = state[:4, 0].reshape(2, 2)
                min_target_distance = float(np.min(
                    np.linalg.norm(pursuer_positions[:, None, :] - target_positions[None, :, :], axis=2)))
                executed_collision = _collision_prefix(audit.positions, truth_boxes, midpoint)
                cycles_out.append(dict(cycle=cycle, action=action.tolist(), feasible=bool(feasible),
                                      collision=bool(executed_collision),
                                      planned_trajectory_collision=requested_plan_collision,
                                      truth_planned_trajectory_collision=truth_plan_collision,
                                      truth_executed_prefix_collision=truth_prefix_collision,
                                      lidar_full_trajectory_safe=bool(full_lidar_safe),
                                      lidar_executed_prefix_safe=bool(prefix_lidar_safe),
                                      gate_full_horizon=bool(gate_full_horizon),
                                      safety_override=safety_override,
                                      minco_obstacle_plan=minco_obstacle_plan,
                                      controller_action=controller_action.tolist(),
                                      requested_action=requested.tolist(),
                                      delivered_action=delivered.tolist(),
                                      communication_event=comm_event,
                                      planning_ms=float(getattr(controller, 'last_stats', {}).get('total_ms', 0.)),
                                      minco_max_speed=float(audit.max_speed),
                                      minco_max_acceleration=float(audit.max_acceleration),
                                      executed_distance=float(np.linalg.norm(audit.positions[midpoint] - audit.positions[0], axis=1).sum()),
                                      capture_events=capture_events,
                                      min_target_distance=min_target_distance,
                                      evader_step=rootpuct_evader_step,
                                      evader_requested_step=rootpuct_evader_requested_step,
                                      rootpuct_search_failures=int(rootpuct_failures),
                                      midpoint_position=audit.positions[midpoint].tolist(),
                                      midpoint_velocity=audit.velocities[midpoint].tolist(),
                                      midpoint_acceleration=audit.accelerations[midpoint].tolist(),
                                      endpoint_position=audit.positions[-1].tolist(),
                                      # Only the first rolling second is sent to
                                      # the vehicle.  The remaining planned tail
                                      # is retained separately for diagnostics;
                                      # mixing it into the executed path makes
                                      # replanning look like a sharp turn.
                                      executed_trajectory_positions=audit.positions[:midpoint + 1].tolist(),
                                      executed_trajectory_times=audit.times[:midpoint + 1].tolist(),
                                      planned_trajectory_positions=audit.positions.tolist(),
                                      planned_trajectory_times=audit.times.tolist(),
                                      evader_position=state[:4, 0].reshape(2, 2).tolist(),
                                      active_p=state[env.active_idxs[2:4], 0].tolist(),
                                      active_e=state[env.active_idxs[0:2], 0].tolist(),
                                      observation=obs,
                                      camera_state=(None if camera_frame is None else dict(
                                          gimbal_yaw=camera_frame.gimbal_yaw.tolist(),
                                          state=list(camera_frame.state),
                                          lost_reason=list(camera_frame.lost_reason),
                                          visible=camera_frame.visible.tolist(),
                                          covered=camera_frame.covered.tolist(),
                                          measurements=camera_frame.measurements)),
                                      lidar_scans=[dict(frame_id=scan.frame_id,
                                                        stamp=float(scan.stamp),
                                                        angles=scan.angles.tolist(),
                                                        ranges=scan.ranges.tolist(),
                                                        max_range=float(scan.max_range))
                                                    for scan in planning_scans],
                                      lidar_hit_count=[int(np.sum(scan.ranges < scan.max_range - 1e-9))
                                                       for scan in planning_scans],
                                      local_lidar_safe=lidar_safe,
                                      lidar_planning_input=bool(unknown_map),
                                      local_obstacle_count=int(len(planning_boxes)),
                                      local_obstacle_boxes=planning_boxes.tolist()))
                if unknown_map:
                    camera_public_history.append(obs)
            rows.append(dict(method=method, valid=True, cycles=cycles_out))
        except Exception as exc:
            rows.append(dict(method=method, valid=False, error=repr(exc),
                             traceback=traceback.format_exc()))
    result = dict(stage='rolling_endpoint_camera_minco_planning_smoke', case=str(case_path),
                  horizon_seconds=2., execute_seconds=1., cycles=cycles,
                  pursuer_speed=pursuer_speed, pursuer_acceleration=pursuer_acceleration,
                  command_delay=command_delay, drop_cycles=list(drop_cycles),
                  pose_delay=pose_delay,
                  moving_evader=bool(moving_evader),
                  rootpuct_evader=bool(rootpuct_evader),
                  geometry_evader=bool(geometry_evader),
                  gate_full_horizon=bool(gate_full_horizon),
                  rootpuct_simulations=int(rootpuct_simulations),
                  lidar_rays=int(lidar_rays), lidar_inflation=float(lidar_inflation),
                  lidar_target_radius=float(lidar_target_radius),
                  grid_inflation=float(grid_inflation),
                  cross_cycle_smoothing=bool(cross_cycle_smoothing),
                  max_action_delta=float(max_action_delta),
                  fixed_simulations=int(fixed_simulations),
                  frozen_evader=not bool(moving_evader), unknown_map_claim=bool(unknown_map),
                  controller_received_full_map=not bool(unknown_map),
                  controller_received_lidar_local_map=bool(unknown_map),
                  lidar_truth_only_for_safety=False,
                  local_lidar_safety_audit=True,
                  lidar_action_safety_gate=False,
                  minco_obstacle_planning=bool(unknown_map),
                  target_visibility_model='camera_fov' if unknown_map else 'legacy',
                  methods=list(selected_methods), rows=rows)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--case', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--cycles', type=int, default=2)
    parser.add_argument('--pursuer-speed', type=float, default=1.5)
    parser.add_argument('--pursuer-acceleration', type=float, default=6.)
    parser.add_argument('--unknown-map', action='store_true')
    parser.add_argument('--command-delay', type=int, default=0)
    parser.add_argument('--drop-cycles', type=int, nargs='*', default=[])
    parser.add_argument('--pose-delay', type=int, default=0)
    parser.add_argument('--moving-evader', action='store_true')
    parser.add_argument('--lidar-rays', type=int, default=256)
    parser.add_argument('--lidar-inflation', type=float, default=.05)
    parser.add_argument('--lidar-target-radius', type=float, default=.25)
    parser.add_argument('--grid-inflation', type=float, default=.05)
    parser.add_argument('--rootpuct-evader', action='store_true')
    parser.add_argument('--rootpuct-simulations', type=int, default=100)
    parser.add_argument('--rootpuct-seed', type=int, default=17)
    parser.add_argument('--geometry-evader', action='store_true')
    parser.add_argument('--gate-full-horizon', action='store_true')
    parser.add_argument('--fixed-simulations', type=int, default=2)
    parser.add_argument('--cross-cycle-smoothing', action='store_true')
    parser.add_argument('--max-action-delta', type=float, default=1.2)
    parser.add_argument('--methods', nargs='*', default=None)
    args = parser.parse_args()
    print(json.dumps(run(args.case, args.output, cycles=args.cycles,
                         pursuer_speed=args.pursuer_speed,
                         pursuer_acceleration=args.pursuer_acceleration,
                         unknown_map=args.unknown_map,
                         command_delay=args.command_delay,
                         drop_cycles=args.drop_cycles,
                         pose_delay=args.pose_delay,
                         moving_evader=args.moving_evader,
                         lidar_rays=args.lidar_rays,
                         lidar_inflation=args.lidar_inflation,
                         lidar_target_radius=args.lidar_target_radius,
                         grid_inflation=args.grid_inflation,
                         rootpuct_evader=args.rootpuct_evader,
                         rootpuct_simulations=args.rootpuct_simulations,
                         rootpuct_seed=args.rootpuct_seed,
                         geometry_evader=args.geometry_evader,
                         gate_full_horizon=args.gate_full_horizon,
                         fixed_simulations=args.fixed_simulations,
                         cross_cycle_smoothing=args.cross_cycle_smoothing,
                         max_action_delta=args.max_action_delta,
                         methods=args.methods),
                     ensure_ascii=False, indent=2))
