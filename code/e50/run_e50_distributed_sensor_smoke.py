"""Sensor-to-independent-MCTS E50 smoke.

This is the first E50 entry point that keeps camera, LiDAR memory and belief
per pursuer.  Truth geometry is used only inside the sensor simulator; the
controllers receive AgentObservation objects and peer messages only.
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
from pathlib import Path
import numpy as np

from solvers.e50_camera import CameraGimbalTracker
from solvers.e50_distributed_runtime import AgentObservation, DistributedEpisodeRunner, make_local_geometry_mcts
from solvers.e50_leaf_value import LeafValueNet
from solvers.e50_lidar_map import LidarPlanningMap, raycast_scan


class PublicBelief:
    def __init__(self, particles=None, weights=None):
        self.particles = np.zeros((0, 2), dtype=float) if particles is None else np.asarray(particles, float).reshape(-1, 2)
        self.weights = np.zeros(len(self.particles), dtype=float) if weights is None else np.asarray(weights, float).reshape(-1)


def run(case_path, output, cycles=8, lidar_rays=256, seed=17):
    case = json.loads(Path(case_path).read_text())
    truth_boxes = np.asarray(case['map'], float).reshape(-1, 2, 2)
    initial = np.asarray(case['worlds'][0]['initial'], float).reshape(-1)
    pursuers = initial[4:8].reshape(2, 2).copy()
    targets = initial[:4].reshape(2, 2).copy()
    active_p = np.ones(2, dtype=bool); active_e = np.ones(2, dtype=bool)
    trackers = [CameraGimbalTracker(1, fov_deg=60., gimbal_limit_deg=90.,
                                    sweep_rate_deg_s=960., max_range=20., seed=seed+i)
                for i in range(2)]
    maps = [LidarPlanningMap(max_age=2., max_points=512, inflation=.05,
                             max_boxes=24, merge_distance=.25) for _ in range(2)]
    beliefs = [PublicBelief() for _ in range(2)]
    controllers = {i: make_local_geometry_mcts(value_net=LeafValueNet(),
                                                simulations=8, seed=seed+i,
                                                duration=2.) for i in range(2)}

    def observation_fn(agent_id, cycle, peer_messages, previous):
        position = pursuers[agent_id].copy()
        scan = raycast_scan(position, truth_boxes, frame_id='map', stamp=float(cycle),
                            num_rays=int(lidar_rays), max_range=10.)
        maps[agent_id].update(position, scan)
        frame = trackers[agent_id].step(position.reshape(1, 2), targets, [True], active_e,
                                        [0.], truth_boxes, dt=1.)
        reports = [x for x in frame.reports if x and x[0] is not None]
        if reports:
            beliefs[agent_id] = PublicBelief([reports[0][0]], [1.])
        return AgentObservation(agent_id, cycle, float(cycle), position, np.zeros(2),
                                0., maps[agent_id], beliefs[agent_id], tuple(peer_messages),
                                None if previous is None else previous.action)

    runner = DistributedEpisodeRunner(controllers, delay=0, ttl=2)
    logs = runner.run(int(cycles), observation_fn)
    result = dict(stage='e50_distributed_sensor_smoke', case=str(case_path), cycles=int(cycles),
                  lidar_rays=int(lidar_rays), agents=2, truth_used_only_sensor=True,
                  independent_local_maps=True, independent_camera_trackers=True,
                  independent_beliefs=True, logs=logs)
    output = Path(output); output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2, default=lambda x: x.tolist() if hasattr(x, 'tolist') else str(x)) + '\n')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--case', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--cycles', type=int, default=8)
    parser.add_argument('--lidar-rays', type=int, default=256)
    args = parser.parse_args()
    print(json.dumps(run(args.case, args.output, args.cycles, args.lidar_rays), ensure_ascii=False))
