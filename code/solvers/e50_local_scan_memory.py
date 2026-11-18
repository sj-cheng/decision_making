"""Rolling obstacle-hit memory for the E50 local LiDAR adapter."""
from dataclasses import dataclass
import numpy as np

from solvers.e50_interfaces import LidarScan
from solvers.e50_lidar_map import scan_to_snapshot


@dataclass
class LocalScanMemory:
    resolution: float = .2
    max_age: float = 2.
    max_points: int = 256

    def __post_init__(self):
        if self.resolution <= 0 or self.max_age <= 0 or self.max_points < 1:
            raise ValueError('invalid scan memory parameters')
        self.points = np.zeros((0, 2), dtype=float)
        self.stamps = np.zeros(0, dtype=float)
        self.occupied = {}
        self.free = {}

    def update(self, sensor_position, scan: LidarScan, *, yaw=0., stamp=None):
        if not isinstance(scan, LidarScan):
            raise TypeError('scan must be a LidarScan')
        sensor = np.asarray(sensor_position, dtype=float)
        if sensor.shape != (2,) or not np.all(np.isfinite(sensor)):
            raise ValueError('invalid sensor position')
        now = float(scan.stamp if stamp is None else stamp)
        angles = scan.angles + float(yaw)
        points = sensor[None, :] + np.column_stack((np.cos(angles), np.sin(angles))) * scan.ranges[:, None]
        valid = np.isfinite(points).all(axis=1) & (scan.ranges < scan.max_range - 1e-9)
        points, stamps = points[valid], np.full(int(valid.sum()), now)
        if len(points):
            self.points = np.concatenate((self.points, points), axis=0)
            self.stamps = np.concatenate((self.stamps, stamps), axis=0)
        # Keep a small local occupancy grid with directional ray evidence.
        for angle, distance in zip(angles, scan.ranges):
            end = sensor + np.array([np.cos(angle), np.sin(angle)]) * float(distance)
            count = max(1, int(np.ceil(float(distance) / self.resolution)))
            for point in np.linspace(sensor, end, count, endpoint=False):
                cell = tuple(np.floor(point / self.resolution).astype(int))
                self.free[cell] = now
            if distance < scan.max_range - 1e-9:
                cell = tuple(np.floor(end / self.resolution).astype(int))
                self.occupied[cell] = now
        keep = (now - self.stamps <= self.max_age) & (self.stamps <= now + 1e-9)
        self.points, self.stamps = self.points[keep], self.stamps[keep]
        # Merge hits in a resolution cell, retaining the newest observation.
        cells = np.floor(self.points / self.resolution).astype(int)
        selected = {}
        for i, cell in enumerate(map(tuple, cells)):
            if cell not in selected or self.stamps[i] >= self.stamps[selected[cell]]:
                selected[cell] = i
        ids = np.asarray(list(selected.values()), dtype=int)
        ids = ids[np.argsort(self.stamps[ids])[::-1]][:self.max_points]
        self.points, self.stamps = self.points[ids], self.stamps[ids]
        self.occupied = {cell: stamp for cell, stamp in self.occupied.items()
                         if now - stamp <= self.max_age and stamp <= now + 1e-9}
        self.free = {cell: stamp for cell, stamp in self.free.items()
                     if now - stamp <= self.max_age and stamp <= now + 1e-9}

    def snapshot(self, *, frame_id='map', stamp=0., origin=(-10., -10.), shape=(200, 200)):
        occupancy = scan_to_snapshot(self.points, np.asarray(origin, float), frame_id=frame_id,
                                     stamp=stamp, resolution=self.resolution, shape=shape,
                                     origin=origin, max_range=max(np.linalg.norm(self.points-np.asarray(origin), axis=1).max(initial=1.), 1.))
        return occupancy

    def obstacle_boxes(self, *, inflation=0.05, max_boxes=24):
        """Return conservative axis-aligned boxes built from measured hit cells.

        This is the bridge for legacy E49 planners whose geometry interface is a
        list of rectangles rather than a raster occupancy grid.  Only occupied
        LiDAR cells are exported; unobserved cells remain unknown and are not
        reported as free space.
        """
        if inflation < 0 or int(max_boxes) < 1:
            raise ValueError("invalid inflation or max_boxes")
        if not self.occupied:
            return np.zeros((0, 2, 2), dtype=float)
        cells = set(self.occupied)
        components = []
        while cells:
            seed = cells.pop(); stack = [seed]; component = [seed]
            while stack:
                x, y = stack.pop()
                for neighbour in ((x+1,y), (x-1,y), (x,y+1), (x,y-1)):
                    if neighbour in cells:
                        cells.remove(neighbour); stack.append(neighbour)
                        component.append(neighbour)
            components.append(component)
        # Preserve the largest measured components under the legacy capacity.
        components.sort(key=len, reverse=True)
        boxes = []
        r = float(self.resolution); pad = float(inflation)
        for component in components[:int(max_boxes)]:
            xy = np.asarray(component, dtype=float)
            lo = xy.min(axis=0) * r - pad
            hi = (xy.max(axis=0) + 1.) * r + pad
            boxes.append(np.stack((lo, hi), axis=1))
        return np.asarray(boxes, dtype=float).reshape(-1, 2, 2)

    def endpoint_safe(self, point, *, inflation=.12):
        """Conservative point check against recent measured hit points."""
        point = np.asarray(point, dtype=float)
        if point.shape != (2,) or not np.all(np.isfinite(point)) or inflation < 0:
            return False
        if not len(self.points):
            return True
        return bool(np.min(np.linalg.norm(self.points - point[None, :], axis=1)) > float(inflation))

    def trajectory_safe(self, points, *, inflation=.12):
        points = np.asarray(points, dtype=float)
        if points.ndim != 2 or points.shape[1] != 2:
            raise ValueError('points must have shape (N,2)')
        return bool(all(self.endpoint_safe(p, inflation=inflation) for p in points))

    def grid_trajectory_safe(self, points, *, inflation=.05):
        """Check only measured occupied cells, preserving ray directionality."""
        points = np.asarray(points, dtype=float)
        if points.ndim != 2 or points.shape[1] != 2 or inflation < 0:
            raise ValueError('invalid points or inflation')
        radius = int(np.ceil(float(inflation) / self.resolution))
        for point in points:
            cell = np.floor(point / self.resolution).astype(int)
            for dx in range(-radius, radius + 1):
                for dy in range(-radius, radius + 1):
                    if (int(cell[0] + dx), int(cell[1] + dy)) in self.occupied:
                        return False
        return True
