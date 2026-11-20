"""Rolling quintic execution for the full-map geometric simulation opponent."""
import numpy as np

from local_observation import clear_segments
from solvers.e50_distributed_geometry import (DEFAULT_MAX_ACCELERATION, evaluate_minco_endpoint,
    reachable_waypoint, braking_waypoints, validate_max_acceleration)
from solvers.e50_interfaces import sample_quintic_at
from solvers.e49_geometric_evader import ACTIONS


class EvaderMinco:
    """Replan for two seconds and keep a safe tail for the next cycle."""

    def __init__(self, boxes, max_speed, samples_per_cycle, *, max_acceleration=None, continuous_speed=False):
        self.continuous_speed = continuous_speed
        self.boxes = boxes
        self.max_speed = max_speed
        self.continuous_acceleration = max_acceleration is not None
        self.max_acceleration = (DEFAULT_MAX_ACCELERATION if max_acceleration is None
                                 else validate_max_acceleration(max_acceleration))
        self.samples = samples_per_cycle
        self.displacements = max_speed * ACTIONS / np.maximum(1., np.linalg.norm(ACTIONS, axis=1, keepdims=True))
        self.tail = {}

    def _safe(self, points):
        bound = 9. if self.continuous_speed else 10.
        return (not np.any(np.abs(points) > bound) and
                (not len(self.boxes) or clear_segments(points[:-1], points[1:], self.boxes).all()))

    def _braking_tail(self, pose, velocity, acceleration):
        for endpoint, duration in braking_waypoints(pose, velocity, acceleration, self.max_acceleration, 2.):
            stop = evaluate_minco_endpoint(pose, endpoint, duration=duration,
                start_velocity=velocity, start_acceleration=acceleration, max_speed=self.max_speed,
                max_acceleration=self.max_acceleration, continuous_speed=True, continuous_acceleration=True)
            if not stop['feasible'] or not self._safe(stop['positions']):
                continue
            cycles = int(np.ceil(duration))
            times = np.minimum(np.arange(cycles*self.samples+1)/self.samples, duration)
            arrays = sample_quintic_at(stop['coefficients'], times)
            if self._safe(arrays[0]):
                return dict(zip(('positions', 'velocities', 'accelerations'), arrays))
        return None

    def prefix(self, agent, position, preferred_displacement):
        n = self.samples
        previous = self.tail.get(agent)
        velocity = np.zeros(2) if previous is None else previous['velocities'][0]
        acceleration = np.zeros(2) if previous is None else previous['accelerations'][0]
        alternatives = self.displacements
        if np.linalg.norm(preferred_displacement) > 1e-9:
            alternatives = alternatives[np.any(alternatives != 0., axis=1)]
        order = np.argsort(np.linalg.norm(alternatives - preferred_displacement, axis=1), kind='stable')
        for action in alternatives[order]:
            endpoint = position + action
            if self.continuous_speed:
                endpoint = reachable_waypoint(position, endpoint, velocity, 2., self.max_speed, self.max_acceleration)
            moving = self.continuous_speed and np.linalg.norm(action) > 1e-9
            for through in ((True, False) if moving else (False,)):
                track = evaluate_minco_endpoint(position, endpoint, duration=2., sample_dt=1./n,
                    start_velocity=velocity, start_acceleration=acceleration,
                    max_speed=self.max_speed, max_acceleration=self.max_acceleration,
                    continuous_acceleration=self.continuous_acceleration, continuous_speed=self.continuous_speed,
                    pass_through=through)
                if not track['feasible'] or not self._safe(track['positions']):
                    continue
                keys = ('positions', 'velocities', 'accelerations')
                boundary = [track[k][n] for k in keys]
                stop = None
                if self.continuous_speed and max(np.linalg.norm(x) for x in boundary[1:]) >= 1e-6:
                    stop = self._braking_tail(*boundary)
                    if stop is None:
                        continue
                self.tail[agent] = stop if through and stop is not None else {k:track[k][n:] for k in keys}
                return {k:track[k][:n+1] for k in keys}, 'pass_through' if through else 'replan'

        if previous is not None:
            if len(previous['positions']) > n+1:
                self.tail[agent] = {k:v[n:] for k,v in previous.items()}
            else:
                del self.tail[agent]
            return {k:v[:n+1] for k,v in previous.items()}, 'continue'
        hold = {'positions': np.repeat(position[None, :], n+1, axis=0),
                'velocities': np.zeros((n+1, 2)), 'accelerations': np.zeros((n+1, 2))}
        return hold, 'hold'
