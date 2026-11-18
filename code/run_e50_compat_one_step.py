"""One-step compatibility diagnostic for existing E49 method controllers.

This intentionally labels the controller as known-map: it is used to isolate
MINCO execution and LiDAR safety effects before the sensor-only adapter is
integrated into the controllers.
"""
import argparse
import json
from pathlib import Path
import numpy as np


METHODS = ('A2', 'A3', 'A4', 'AG', 'GEOM_SEARCH', 'RouteAPF')


def run(case_path, output, methods=METHODS, fixed_simulations=2,
        pursuer_speed=None, pursuer_acceleration=None):
    from e49_controllers import install, make_controller
    from run_e49_benchmark import observation
    from solvers.e50_compat_smoke import compare_execution, replay_endpoint_semantic
    from solvers.e50_minco_corrected import CorrectedClosedLoopMincoPlanner
    from e49_rules import create_environment

    case_path, output = Path(case_path), Path(output)
    case = json.loads(case_path.read_text())
    output.parent.mkdir(parents=True, exist_ok=True)
    install(int(case.get('horizon', 80)), output.parent)
    state = np.asarray(case['worlds'][0]['initial'], dtype=float).reshape(-1, 1)
    boxes = np.asarray(case['map'], dtype=float).reshape(-1, 2, 2)
    env = create_environment(boxes, horizon=int(case.get('horizon', 80)))
    if pursuer_speed is not None:
        env.current_pursuer_speed_lim = float(pursuer_speed)
        env.update_action_lims()
    if pursuer_acceleration is not None:
        for robot in env.pursuers:
            env.state_lims[env.acc_idxs[robot], :] = [[-float(pursuer_acceleration), float(pursuer_acceleration)]] * 2
    rows = []
    for method in methods:
        task = dict(arm=method, seed=17, train_seed=17, budget_ms=200.,
                    fixed=int(fixed_simulations), horizon=int(case.get('horizon', 80)),
                    out=str(output.parent))
        try:
            controller = make_controller(case, task)
            for frame in case['history'][:-1]:
                controller.observe(frame)
            obs = observation(state, boxes, task['horizon'])
            action = np.asarray(controller.act(obs, fixed_simulations=fixed_simulations), float)
            full = np.zeros((env.action_dim, 1))
            full[env.action_idxs[2], 0] = action[0]
            full[env.action_idxs[3], 0] = action[1]
            audit = compare_execution(env, state, full, robots=(2, 3), sample_dt=.05)
            corrected = compare_execution(env, state, full, robots=(2, 3), sample_dt=.05,
                                          minco_planner_cls=CorrectedClosedLoopMincoPlanner)
            endpoint_horizons = {}
            for horizon in (1., 2., 3.):
                endpoint, endpoint_feasible = replay_endpoint_semantic(
                    env, state, full, dt=horizon, robots=(2, 3), sample_dt=.05)
                endpoint_horizons[str(horizon)] = dict(
                    feasible=bool(endpoint_feasible), collision=endpoint.collision,
                    max_speed=endpoint.max_speed,
                    max_acceleration=endpoint.max_acceleration)
            rows.append(dict(method=method, valid=True, action=action.tolist(),
                             controller_stats=getattr(controller, 'last_stats', {}),
                             endpoint_delta=audit.endpoint_delta.tolist(),
                             corrected_endpoint_delta=corrected.endpoint_delta.tolist(),
                             endpoint_horizons=endpoint_horizons,
                             minco_raw_targets=(audit.minco.raw_targets.tolist() if audit.minco.raw_targets is not None else None),
                             minco_feasible_targets=(audit.minco.feasible_targets.tolist() if audit.minco.feasible_targets is not None else None),
                             minco_error_to_feasible=audit.minco_error_to_feasible.tolist(),
                             direct_max_speed=audit.direct.max_speed,
                             minco_max_speed=audit.minco.max_speed,
                             minco_max_acceleration=audit.minco.max_acceleration,
                             direct_collision=audit.direct.collision,
                             minco_collision=audit.minco.collision,
                             direct_lidar_safe=audit.direct_lidar_safe.tolist(),
                             minco_lidar_safe=audit.minco_lidar_safe.tolist()))
        except Exception as exc:
            rows.append(dict(method=method, valid=False, error=repr(exc)))
    result = dict(stage='known_map_controller_minco_diagnostic', case=str(case_path),
                  methods=list(methods), fixed_simulations=int(fixed_simulations), rows=rows,
                  unknown_map_claim=False, pursuer_speed=pursuer_speed,
                  pursuer_acceleration=pursuer_acceleration,
                  note='Controller still receives E49 map; execution-only diagnostic.')
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2, default=lambda x: x.item() if hasattr(x, 'item') else str(x))+'\n')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--case', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--pursuer-speed', type=float, default=None)
    parser.add_argument('--pursuer-acceleration', type=float, default=None)
    args = parser.parse_args()
    print(json.dumps(run(args.case, args.output, pursuer_speed=args.pursuer_speed,
                         pursuer_acceleration=args.pursuer_acceleration), ensure_ascii=False, indent=2))
