"""Conservative local obstacle proxy built from range-only LiDAR rays.

This adapter intentionally does not pretend that a rectangle map can be
reconstructed from one scan. It protects a trajectory against the nearest
measured returns and treats angular gaps conservatively by using the smaller
of the neighboring beam ranges.
"""
from dataclasses import dataclass
import numpy as np

from solvers.e50_interfaces import LidarScan


@dataclass(frozen=True)
class LocalObstacleProxy:
    sensor_position: np.ndarray
    angles: np.ndarray
    ranges: np.ndarray
    max_range: float
    inflation: float = .25

    def __post_init__(self):
        position = np.asarray(self.sensor_position, dtype=float).copy()
        angles = np.asarray(self.angles, dtype=float).copy()
        ranges = np.asarray(self.ranges, dtype=float).copy()
        if position.shape != (2,) or angles.ndim != 1 or ranges.shape != angles.shape:
            raise ValueError('invalid local obstacle proxy shapes')
        if not np.all(np.isfinite(position)) or not np.all(np.isfinite(angles)):
            raise ValueError('non-finite local obstacle proxy')
        if self.max_range <= 0 or self.inflation < 0:
            raise ValueError('invalid local obstacle limits')
        if np.any(~np.isfinite(ranges)) or np.any(ranges < 0) or np.any(ranges > self.max_range):
            raise ValueError('invalid local ranges')
        if len(angles) < 2:
            raise ValueError('at least two beams are required')
        object.__setattr__(self, 'sensor_position', position)
        object.__setattr__(self, 'angles', angles)
        object.__setattr__(self, 'ranges', ranges)

    @classmethod
    def from_scan(cls, sensor_position, scan: LidarScan, inflation=.25):
        if not isinstance(scan, LidarScan):
            raise TypeError('scan must be a LidarScan')
        return cls(sensor_position, scan.angles, scan.ranges, scan.max_range, inflation)

    def hit_points(self):
        directions = np.column_stack((np.cos(self.angles), np.sin(self.angles)))
        return self.sensor_position[None, :] + directions * self.ranges[:, None]

    def _neighbor_limit(self, angle):
        # The scan is cyclic. The smaller neighboring return is used so a
        # narrow obstacle between beams cannot be treated as free space.
        unwrapped = np.unwrap(self.angles)
        a = float(angle)
        while a < unwrapped[0]:
            a += 2*np.pi
        while a >= unwrapped[-1] + 2*np.pi/len(unwrapped):
            a -= 2*np.pi
        idx = int(np.searchsorted(unwrapped, a, side='right') - 1) % len(unwrapped)
        nxt = (idx + 1) % len(unwrapped)
        return float(min(self.ranges[idx], self.ranges[nxt]))

    def _interpolated_limit(self, angle):
        """Interpolate adjacent returns for dense scans.

        This keeps the beam-local geometry instead of extending the nearest
        return across the whole angular sector. A caller still supplies its
        own inflation margin for robot footprint and calibration uncertainty.
        """
        unwrapped = np.unwrap(self.angles)
        a = float(angle)
        while a < unwrapped[0]:
            a += 2*np.pi
        period = 2*np.pi
        while a >= unwrapped[-1] + period / len(unwrapped):
            a -= period
        idx = int(np.searchsorted(unwrapped, a, side='right') - 1) % len(unwrapped)
        nxt = (idx + 1) % len(unwrapped)
        left = unwrapped[idx]
        right = unwrapped[nxt] if nxt > idx else unwrapped[idx] + period
        alpha = np.clip((a - left) / max(right - left, 1e-12), 0., 1.)
        return float((1. - alpha) * self.ranges[idx] + alpha * self.ranges[nxt])

    def endpoint_safe(self, point, *, interpolation=False):
        point = np.asarray(point, dtype=float)
        if point.shape != (2,) or not np.all(np.isfinite(point)):
            return False
        delta = point - self.sensor_position
        distance = float(np.linalg.norm(delta))
        if distance <= 1e-9:
            return True
        angle = np.arctan2(delta[1], delta[0])
        limit = self._interpolated_limit(angle) if interpolation else self._neighbor_limit(angle)
        return distance + self.inflation < limit

    def segment_safe(self, start, end, samples=32, *, interpolation=False):
        start, end = np.asarray(start, dtype=float), np.asarray(end, dtype=float)
        if start.shape != (2,) or end.shape != (2,) or int(samples) < 2:
            raise ValueError('invalid segment')
        points = np.linspace(start, end, int(samples))
        return bool(all(self.endpoint_safe(point, interpolation=interpolation) for point in points))


def proxy_from_scan(sensor_position, scan, inflation=.25):
    """Typed convenience wrapper used by local geometry adapters."""
    return LocalObstacleProxy.from_scan(sensor_position, scan, inflation)


def project_action_to_local_scan(starts, action, proxies, *, radii=(.25, .5, .75, 1.0),
                                 samples=24):
    """Project a joint displacement onto locally measured-safe actions.

    Besides shortening the requested vector and trying the four cardinal
    directions, generate small tangential moves around the nearest LiDAR hit
    points.  The old cardinal-only fallback could stop in front of a wall even
    when a measured left/right bypass was immediately available.
    """
    starts = np.asarray(starts, dtype=float)
    action = np.asarray(action, dtype=float)
    if starts.shape != (2, 2) or action.shape != (2, 2) or len(proxies) != 2:
        raise ValueError('expected starts/action (2,2) and two proxies')
    result = np.zeros_like(action)
    for robot in range(2):
        desired = action[robot]
        norm = float(np.linalg.norm(desired))
        direction = desired / norm if norm > 1e-9 else np.zeros(2)
        base_candidates = [np.zeros(2)]
        if norm > 1e-9:
            base_candidates.extend(direction * min(norm, radius) for radius in radii)
        # Add axis-aligned escape directions for blocked goal vectors.
        base_candidates.extend(np.asarray(v, dtype=float) * radius
                               for radius in radii for v in ((1., 0.), (-1., 0.), (0., 1.), (0., -1.)))
        bypass_candidates = []
        if norm > 1e-9:
            # Rotate the desired direction by moderate angles.  These are
            # cheap local waypoint substitutes and remain meaningful when the
            # obstacle patch is only a sparse scan reconstruction.
            for angle in (np.pi / 8., -np.pi / 8., np.pi / 4., -np.pi / 4.,
                          np.pi / 2., -np.pi / 2.):
                c, s = np.cos(angle), np.sin(angle)
                rotated = np.array([c * direction[0] - s * direction[1],
                                    s * direction[0] + c * direction[1]])
                bypass_candidates.extend(rotated * min(norm, radius) for radius in radii)
        # Add tangential directions around the nearest measured returns.  A
        # bounded number keeps projection cost independent of a 256-beam scan.
        hits = proxies[robot].hit_points()
        if len(hits):
            distances = np.linalg.norm(hits - starts[robot][None, :], axis=1)
            for hit in hits[np.argsort(distances)[:8]]:
                radial = hit - starts[robot]
                radial_norm = float(np.linalg.norm(radial))
                if radial_norm <= 1e-9:
                    continue
                radial /= radial_norm
                tangents = (np.array([-radial[1], radial[0]]),
                            np.array([radial[1], -radial[0]]))
                bypass_candidates.extend(tangent * radius for tangent in tangents for radius in radii)
        valid_base = [candidate for candidate in base_candidates
                 if proxies[robot].segment_safe(starts[robot], starts[robot] + candidate,
                                                samples=samples, interpolation=True)]
        # Prefer the old direct/axis semantics whenever they are safe.  Use
        # angular bypass candidates only when the requested motion is blocked
        # by the local scan; this prevents a harmless short direct step from
        # being replaced by a diagonal detour.
        nonzero_base = [candidate for candidate in valid_base
                        if float(np.linalg.norm(candidate)) > 1e-9]
        bypass_valid = [candidate for candidate in bypass_candidates
                        if proxies[robot].segment_safe(starts[robot], starts[robot] + candidate,
                                                       samples=samples, interpolation=True)]
        valid = nonzero_base or bypass_valid or valid_base
        if valid:
            result[robot] = min(valid, key=lambda candidate: float(np.linalg.norm(candidate - desired)))
    return result
