"""Small LiDAR scan adapter for the E50 simulation/deployment boundary.

The real robot may provide an already-built SLAM occupancy grid. This module
also supports replaying point scans in simulation, so both paths produce the
same LocalMapSnapshot contract.
"""
import numpy as np
from dataclasses import dataclass, field

from solvers.e50_interfaces import LidarScan, LocalMapSnapshot


def _boxes_from_points(points, inflation=.12, max_boxes=24, merge_distance=.45,
                       sensor_position=None):
    """Build conservative axis-aligned obstacle patches from hit points.

    A range scan is not a global obstacle list.  This conversion keeps only
    measured returns and groups nearby returns into small rectangles that the
    legacy E49 geometry routines can consume.  Unobserved space therefore
    remains absent from the map instead of silently becoming occupied.
    """
    points = np.asarray(points, dtype=float)
    if points.size == 0:
        return np.zeros((0, 2, 2), dtype=float)
    if points.ndim != 2 or points.shape[1] != 2 or not np.isfinite(points).all():
        raise ValueError('points must be finite with shape (N,2)')
    inflation = float(inflation)
    max_boxes = int(max_boxes)
    merge_distance = float(merge_distance)
    if inflation < 0 or max_boxes < 1 or merge_distance <= 0:
        raise ValueError('invalid obstacle patch parameters')

    # Connected components in the hit cloud.  The threshold is deliberately
    # conservative: neighbouring returns on one wall form one patch, while
    # distant walls stay separate.
    remaining = set(range(len(points)))
    groups = []
    threshold = merge_distance + 2.0 * inflation
    # Batch pairwise distances once; avoid O(N²) Python/numpy scalar calls.
    adjacent = np.linalg.norm(points[:, None] - points[None, :], axis=2) <= threshold
    while remaining:
        seed = remaining.pop()
        group = [seed]
        frontier = [seed]
        while frontier:
            i = frontier.pop()
            near = [j for j in remaining if adjacent[i, j]]
            for j in near:
                remaining.remove(j); frontier.append(j); group.append(j)
        groups.append(group)

    pose = None if sensor_position is None else np.asarray(sensor_position, float)

    def patch(group):
        measured_low = points[group].min(axis=0)
        measured_high = points[group].max(axis=0)
        low = measured_low - inflation
        high = measured_high + inflation
        if pose is not None and np.all(pose >= low) and np.all(pose <= high):
            # Reduce padding only on the strongest separating axis, keeping all
            # measured returns. A box enclosing ego even without padding is
            # split below.
            gap = np.maximum(measured_low - pose, pose - measured_high)
            axis = int(np.argmax(gap))
            if gap[axis] > 0.:
                if pose[axis] < measured_low[axis]:
                    low[axis] = (pose[axis] + measured_low[axis]) * .5
                else:
                    high[axis] = (pose[axis] + measured_high[axis]) * .5
        return np.column_stack((low, high))

    # A connected cloud can wrap around the sensor. Split its AABB into
    # directional patches so the bounding box does not fill the free space.
    if pose is not None:
        refined = []
        for group in groups:
            box = patch(group)
            if np.all(pose >= box[:, 0]) and np.all(pose <= box[:, 1]):
                delta = points[group] - pose
                axis = np.argmax(np.abs(delta), axis=1)
                side = (delta[np.arange(len(group)), axis] >= 0).astype(int)
                labels = 2 * axis + side
                for label in np.unique(labels):
                    refined.append(np.asarray(group)[labels == label].tolist())
            else:
                refined.append(group)
        groups = refined
    boxes = [patch(g) for g in groups]
    # E49 networks reserve 24 map tokens.  Merge the closest patches if a
    # dense scan would otherwise exceed that fixed capacity.
    while len(boxes) > max_boxes:
        centers = np.asarray([(b[:, 0] + b[:, 1]) * .5 for b in boxes])
        dist = np.linalg.norm(centers[:, None] - centers[None, :], axis=2)
        dist[np.diag_indices_from(dist)] = np.inf
        if pose is not None:
            # Prefer capacity merges that do not close the measured free
            # space around ego. If the capacity is too small to preserve it,
            # retaining a conservative containing patch is safer than loss
            # of real returns.
            lows = np.minimum(np.asarray(boxes)[:, None, :, 0], np.asarray(boxes)[None, :, :, 0])
            highs = np.maximum(np.asarray(boxes)[:, None, :, 1], np.asarray(boxes)[None, :, :, 1])
            contains_pose = ((pose >= lows) & (pose <= highs)).all(axis=-1)
            safe_dist = np.where(contains_pose, np.inf, dist)
            if np.isfinite(safe_dist).any():
                dist = safe_dist
        i, j = np.unravel_index(np.argmin(dist), dist.shape)
        low = np.minimum(boxes[i][:, 0], boxes[j][:, 0])
        high = np.maximum(boxes[i][:, 1], boxes[j][:, 1])
        boxes[i] = np.column_stack((low, high)); boxes.pop(j)
    return np.asarray(boxes, dtype=float).reshape(-1, 2, 2)


def scan_to_obstacle_boxes(sensor_position, scan: LidarScan, *, inflation=.12,
                            max_boxes=24, merge_distance=.45):
    """Convert one range-only LiDAR scan into local obstacle patches.

    Returned boxes are in the same map frame as ``sensor_position``.  A
    max-range ray contributes no obstacle point; it only provides free-ray
    evidence to a separate occupancy memory.
    """
    if not isinstance(scan, LidarScan):
        raise TypeError('scan must be LidarScan')
    sensor = np.asarray(sensor_position, dtype=float)
    if sensor.shape != (2,) or not np.isfinite(sensor).all():
        raise ValueError('sensor_position must be finite with shape (2,)')
    angles = scan.angles
    valid = scan.ranges < scan.max_range - 1e-9
    if not np.any(valid):
        return np.zeros((0, 2, 2), dtype=float)
    points = sensor[None, :] + np.column_stack((np.cos(angles[valid]),
                                                 np.sin(angles[valid]))) * scan.ranges[valid, None]
    return _boxes_from_points(points, inflation=inflation, max_boxes=max_boxes,
                              merge_distance=merge_distance)


def points_to_obstacle_boxes(points, *, inflation=.12, max_boxes=24,
                             merge_distance=.45, sensor_position=None):
    """Convert accumulated map-frame LiDAR hits into planner obstacle patches."""
    return _boxes_from_points(points, inflation=inflation, max_boxes=max_boxes,
                              merge_distance=merge_distance, sensor_position=sensor_position)


@dataclass
class LidarPlanningMap:
    """Small rolling shared map used to feed legacy planners.

    The map stores only measured hit endpoints in the common map frame.  It
    intentionally has no ``free`` default: cells/regions that no ray has
    measured stay unknown and are not turned into obstacle patches.  The
    planner receives conservative patches through :meth:`obstacle_boxes`.
    """
    max_age: float = 2.0
    max_points: int = 512
    inflation: float = .12
    max_boxes: int = 24
    merge_distance: float = .45
    _points: np.ndarray = field(default_factory=lambda: np.zeros((0, 2), dtype=float),
                                init=False, repr=False)
    _stamps: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=float),
                                init=False, repr=False)
    _sensor_position: np.ndarray = field(default_factory=lambda: np.zeros(2, dtype=float),
                                         init=False, repr=False)
    _has_sensor_position: bool = field(default=False, init=False, repr=False)
    _last_scan: object = field(default=None, init=False, repr=False)

    def __post_init__(self):
        if self.max_age <= 0 or self.max_points < 1 or self.inflation < 0:
            raise ValueError('invalid lidar planning map parameters')

    @property
    def points(self):
        return self._points.copy()

    def update(self, sensor_position, scan: LidarScan, *, stamp=None):
        if not isinstance(scan, LidarScan):
            raise TypeError('scan must be LidarScan')
        sensor = np.asarray(sensor_position, dtype=float)
        if sensor.shape != (2,) or not np.isfinite(sensor).all():
            raise ValueError('sensor_position must be finite with shape (2,)')
        # Keep the current pose so a hit caused by a robot sitting exactly on
        # an obstacle boundary can be removed from the planner patches below.
        # Ray casting returns a zero-range hit at that boundary; inflating that
        # hit would otherwise create a self-obstacle and make MINCO reject a
        # stationary/recovery trajectory forever.
        self._sensor_position = sensor.copy()
        self._has_sensor_position = True
        self._last_scan = scan
        now = float(scan.stamp if stamp is None else stamp)
        if not np.isfinite(now):
            raise ValueError('stamp must be finite')
        valid = (scan.ranges > 1e-9) & (scan.ranges < scan.max_range - 1e-9)
        if np.any(valid):
            points = sensor[None, :] + np.column_stack((np.cos(scan.angles[valid]),
                                                         np.sin(scan.angles[valid]))) \
                * scan.ranges[valid, None]
            self._points = np.concatenate((self._points, points), axis=0)
            self._stamps = np.concatenate((self._stamps,
                                           np.full(len(points), now, dtype=float)))
        keep = ((now - self._stamps) <= float(self.max_age)) & (self._stamps <= now + 1e-9)
        self._points, self._stamps = self._points[keep], self._stamps[keep]
        # Keep newest points per coarse cell first, then enforce the budget.
        if len(self._points) > self.max_points:
            order = np.argsort(self._stamps)[::-1][:self.max_points]
            self._points, self._stamps = self._points[order], self._stamps[order]

    def obstacle_boxes(self):
        points = self._points
        pose = self._sensor_position if self._has_sensor_position else None
        if pose is not None and len(points):
            # Remove actual zero-range self returns, not the rectangular
            # envelope of arbitrary distant measurements around the robot.
            points = points[np.linalg.norm(points - pose, axis=1) > 1e-9]
        return points_to_obstacle_boxes(points, inflation=self.inflation,
                                        max_boxes=self.max_boxes,
                                        merge_distance=self.merge_distance,
                                        sensor_position=pose)

    def free_space_endpoints(self, max_distance=2., min_clearance=.4,
                             max_candidates=12):
        """Return frontier endpoints supported by recent free LiDAR rays.

        A local obstacle patch is only made from hit points, so a geometry
        planner can otherwise end up with no candidate when all tangent
        endpoints intersect conservative patches.  Free rays provide the
        complementary observation: they are valid only for the latest scan
        and are still checked by the planner's segment safety gate.
        """
        if self._last_scan is None or not self._has_sensor_position:
            return []
        scan = self._last_scan
        ranges = np.asarray(scan.ranges, float)
        angles = np.asarray(scan.angles, float)
        distance = float(max_distance)
        clearance = float(min_clearance)
        if distance <= 0 or clearance < 0 or ranges.shape != angles.shape:
            return []
        usable = np.isfinite(ranges) & (ranges > clearance + .25)
        if not np.any(usable):
            return []
        # Keep the longest, angularly separated rays so a dense 256-ray scan
        # does not consume all endpoint slots with near-duplicates.
        order = np.flatnonzero(usable)[np.argsort(ranges[usable])[::-1]]
        out = []
        for idx in order:
            direction = np.array([np.cos(angles[idx]), np.sin(angles[idx])])
            step = min(distance, max(.25, float(ranges[idx] - clearance)))
            point = self._sensor_position + direction * step
            if not any(float(np.linalg.norm(point - x)) < .25 for x in out):
                out.append(point)
            if len(out) >= int(max_candidates):
                break
        return out


def raycast_lidar(sensor_position, obstacles, *, num_rays=256, max_range=10.,
                  angle_offset=0., noise_std=0., rng=None):
    """Return 2-D ray distances to the nearest axis-aligned obstacle.

    ``obstacles`` uses the repository convention: (N, 2, 2), with x/y lower
    and upper bounds. The result is a public sensor observation; no target
    truth is included. A later real-robot adapter can expose the same arrays
    from the existing SLAM/LiDAR stack.
    """
    sensor = np.asarray(sensor_position, dtype=float)
    boxes = np.asarray(obstacles, dtype=float).reshape(-1, 2, 2)
    if sensor.shape != (2,) or num_rays < 1 or max_range <= 0:
        raise ValueError('invalid lidar parameters')
    if not np.all(np.isfinite(sensor)) or not np.all(np.isfinite(boxes)):
        raise ValueError('non-finite lidar geometry')
    angles = float(angle_offset) + np.arange(num_rays) * (2*np.pi/num_rays)
    directions = np.column_stack((np.cos(angles), np.sin(angles)))
    distances = np.full(num_rays, float(max_range))
    for k, direction in enumerate(directions):
        for box in boxes:
            t0, t1 = -np.inf, np.inf
            valid = True
            for axis in range(2):
                if abs(direction[axis]) < 1e-12:
                    if not (box[axis, 0] <= sensor[axis] <= box[axis, 1]):
                        valid = False
                        break
                else:
                    a = (box[axis, 0] - sensor[axis]) / direction[axis]
                    b = (box[axis, 1] - sensor[axis]) / direction[axis]
                    t0, t1 = max(t0, min(a, b)), min(t1, max(a, b))
            if valid and t1 >= max(t0, 0.):
                distances[k] = min(distances[k], max(0., t0))
    if noise_std:
        generator = np.random.default_rng() if rng is None else rng
        distances += generator.normal(0., float(noise_std), size=num_rays)
        distances = np.clip(distances, 0., max_range)
    return angles, distances


def raycast_scan(sensor_position, obstacles, *, frame_id='base_link', stamp=0.,
                 num_rays=256, max_range=10., angle_offset=0., noise_std=0., rng=None):
    """Return the typed scan object used by the deployment-facing interface."""
    angles, distances = raycast_lidar(sensor_position, obstacles, num_rays=num_rays,
                                      max_range=max_range, angle_offset=angle_offset,
                                      noise_std=noise_std, rng=rng)
    return LidarScan(frame_id, float(stamp), angles, distances, float(max_range))


def lidar_target_visibility(sensor_positions, targets, obstacles, *, num_rays=256,
                            max_range=10., target_radius=.25, angle_offset=0.):
    """Return the target reports a discrete 2-D LiDAR can produce.

    The result has shape ``(num_sensors, num_targets)``.  A target is reported
    only when at least one of the actual LiDAR rays intersects its vehicle
    disc before the nearest static obstacle (and before another target).  The
    returned visibility is deliberately a sensor result; it does not expose
    the ray ranges or the map to the controller.  Once detected, the caller
    may publish the simulator's exact target centre as the identified target
    report, matching the deployment contract used by the legacy controllers.
    """
    sensors = np.asarray(sensor_positions, dtype=float).reshape(-1, 2)
    targets = np.asarray(targets, dtype=float).reshape(-1, 2)
    boxes = np.asarray(obstacles, dtype=float).reshape(-1, 2, 2)
    if num_rays < 1 or max_range <= 0 or target_radius <= 0:
        raise ValueError('invalid lidar target parameters')
    if not (np.all(np.isfinite(sensors)) and np.all(np.isfinite(targets))
            and np.all(np.isfinite(boxes))):
        raise ValueError('non-finite lidar target geometry')
    angles = float(angle_offset) + np.arange(int(num_rays)) * (2*np.pi/int(num_rays))
    directions = np.column_stack((np.cos(angles), np.sin(angles)))
    result = np.zeros((len(sensors), len(targets)), dtype=bool)

    def ray_box_distance(sensor, direction, box):
        t0, t1 = -np.inf, np.inf
        for axis in range(2):
            if abs(direction[axis]) < 1e-12:
                if not (box[axis, 0] <= sensor[axis] <= box[axis, 1]):
                    return np.inf
            else:
                a = (box[axis, 0] - sensor[axis]) / direction[axis]
                b = (box[axis, 1] - sensor[axis]) / direction[axis]
                t0, t1 = max(t0, min(a, b)), min(t1, max(a, b))
        return max(0., t0) if t1 >= max(t0, 0.) else np.inf

    for si, sensor in enumerate(sensors):
        for direction in directions:
            obstacle_t = min((ray_box_distance(sensor, direction, box)
                              for box in boxes), default=np.inf)
            candidates = []
            for ti, target in enumerate(targets):
                delta = target - sensor
                along = float(np.dot(delta, direction))
                perp2 = float(np.dot(delta, delta) - along * along)
                if along < -target_radius or perp2 > target_radius * target_radius:
                    continue
                root = max(0., along - np.sqrt(max(0., target_radius**2 - perp2)))
                if root <= max_range + 1e-9:
                    candidates.append((root, ti))
            if not candidates:
                continue
            # A range-returning LiDAR sees the first return on this ray.
            candidates.sort()
            first_t = candidates[0][0]
            for target_t, ti in candidates:
                if target_t > first_t + 1e-6:
                    break
                if target_t <= obstacle_t + 1e-6:
                    result[si, ti] = True
    return result


def _cell(point, origin, resolution, shape):
    ij = np.floor((np.asarray(point) - origin) / resolution).astype(int)
    return ij if 0 <= ij[0] < shape[1] and 0 <= ij[1] < shape[0] else None


def _ray_cells(start, end, origin, resolution, shape):
    a = _cell(start, origin, resolution, shape)
    b = _cell(end, origin, resolution, shape)
    if a is None or b is None:
        return ()
    steps = int(max(abs(int(b[0])-int(a[0])), abs(int(b[1])-int(a[1])))) + 1
    xs = np.rint(np.linspace(a[0], b[0], steps)).astype(int)
    ys = np.rint(np.linspace(a[1], b[1], steps)).astype(int)
    return tuple((int(y), int(x)) for x, y in zip(xs, ys))


def scan_to_snapshot(points, sensor_position, *, frame_id='map', stamp=0., resolution=.1,
                     shape=(200, 200), origin=(-10., -10.), max_range=10.):
    """Rasterize a map-frame 2-D point scan into occupancy and unknown masks.

    Rays up to each returned hit are known free; cells outside all rays remain
    unknown. Points outside the configured rolling window are ignored.
    """
    points = np.asarray(points, dtype=float)
    sensor = np.asarray(sensor_position, dtype=float)
    if points.ndim != 2 or points.shape[1] != 2 or sensor.shape != (2,):
        raise ValueError('points must be (N,2) and sensor_position must be (2,)')
    if resolution <= 0 or max_range <= 0 or len(shape) != 2 or min(shape) < 2:
        raise ValueError('invalid map parameters')
    origin = np.asarray(origin, dtype=float)
    occupancy = np.zeros(shape, dtype=np.float32)
    unknown = np.ones(shape, dtype=bool)
    for point in points:
        delta = point - sensor
        distance = float(np.linalg.norm(delta))
        if not np.isfinite(distance) or distance <= 1e-9 or distance > max_range:
            continue
        cells = _ray_cells(sensor, point, origin, resolution, shape)
        if not cells:
            continue
        for cell in cells[:-1]:
            unknown[cell] = False
            occupancy[cell] = 0.
        endpoint = cells[-1]
        unknown[endpoint] = False
        occupancy[endpoint] = 1.
    return LocalMapSnapshot(frame_id, float(stamp), float(resolution), origin, occupancy, unknown)


def unknown_fraction(snapshot):
    """Return the fraction of cells not observed by the current scan."""
    return float(np.mean(snapshot.unknown))
