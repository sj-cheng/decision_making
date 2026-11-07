"""Causal team observations, anonymous tracking and versioned episode specs.

Only ``observe`` crosses the environment/agent boundary. Trackers and planners
accept TeamObservation, never a real simulator state. All random streams are
explicit and serializable; sampled states in planning are hypotheses.
"""
from dataclasses import dataclass, asdict
from itertools import permutations
import hashlib
import json
import numpy as np
from functools import lru_cache

PROTOCOL = "example8_local_shared_known_map_swept_v1"
SCHEMA = "lidar270_anonymous_belief23_v1"
BASE_DIM = 270
TRACK_DIM = 9  # relative xy, velocity xy, variance xy, age, visible, existence
OBS_DIM = BASE_DIM + 2 * TRACK_DIM + 3 + 2
TEAMS = ((0, 1), (2, 3))


@lru_cache(maxsize=1)
def native_geometry():
    from cpp.build import bindings
    return bindings if hasattr(bindings,'local_geometry_legal') else None


def segment_interval(start, end, box):
    lo, hi = 0.0, 1.0
    for axis in range(2):
        d = float(end[axis] - start[axis])
        if abs(d) < 1e-12:
            if start[axis] < box[axis, 0] or start[axis] > box[axis, 1]:
                return None
        else:
            a, b = (box[axis] - start[axis]) / d
            lo, hi = max(lo, min(a, b)), min(hi, max(a, b))
    return (lo, hi) if lo <= hi else None


def first_collision_point(start, end, obstacles):
    first, result = 2.0, np.array(end, dtype=float, copy=True)
    for box in obstacles:
        hit = segment_interval(start, end, box)
        if hit is not None and hit[0] < first:
            first = hit[0]
            result = np.clip(start + (end - start) * min(hit[1], first + 1e-5), box[:, 0], box[:, 1])
    return result


def legal_points(points, obstacles, bounds, margin=1e-6):
    points = np.asarray(points)
    native=native_geometry()
    if native is not None:
        return np.asarray(native.local_geometry_legal(points.reshape(-1,2),obstacles,bounds,margin)).reshape(points.shape[:-1])
    valid = ((points > bounds[:, 0] + margin) & (points < bounds[:, 1] - margin)).all(axis=-1)
    for box in obstacles:
        valid &= ~((points >= box[:, 0] - margin) & (points <= box[:, 1] + margin)).all(axis=-1)
    return valid


def clear_segments(starts, ends, obstacles):
    native=native_geometry()
    if native is not None:
        return np.asarray(native.local_geometry_clear(starts,ends,obstacles)).reshape(-1)
    clear = np.ones(len(starts), dtype=bool)
    delta = ends - starts
    for box in obstacles:
        low = np.zeros(len(starts)); high = np.ones(len(starts)); possible = np.ones(len(starts), bool)
        for axis in range(2):
            moving = np.abs(delta[:, axis]) > 1e-12
            a = (box[axis,0] - starts[:,axis]) / np.where(moving,delta[:,axis],1.)
            b = (box[axis,1] - starts[:,axis]) / np.where(moving,delta[:,axis],1.)
            low = np.maximum(low,np.where(moving,np.minimum(a,b),0.))
            high = np.minimum(high,np.where(moving,np.maximum(a,b),1.))
            possible &= moving | ((starts[:,axis]>=box[axis,0]) & (starts[:,axis]<=box[axis,1]))
        clear &= ~(possible & (low<=high))
    return clear


def visible_points(points, origins, obstacles, max_range):
    """Union of visibility from active teammates, with vectorized box slabs."""
    points = np.asarray(points).reshape(-1, 2)
    native=native_geometry()
    if native is not None:
        return np.asarray(native.local_geometry_visible(points,np.asarray(origins).reshape(-1,2),obstacles,max_range)).reshape(-1)
    seen = np.zeros(len(points), dtype=bool)
    for origin in origins:
        delta = points - origin
        visible = np.linalg.norm(delta, axis=1) <= max_range
        for box in obstacles:
            low = np.zeros(len(points)); high = np.ones(len(points)); possible = np.ones(len(points), bool)
            for axis in range(2):
                moving = np.abs(delta[:, axis]) > 1e-12
                denom = np.where(moving, delta[:, axis], 1.)
                a = (box[axis, 0] - origin[axis]) / denom
                b = (box[axis, 1] - origin[axis]) / denom
                low = np.maximum(low, np.where(moving, np.minimum(a, b), 0.))
                high = np.minimum(high, np.where(moving, np.maximum(a, b), 1.))
                possible &= moving | ((origin[axis] >= box[axis, 0]) & (origin[axis] <= box[axis, 1]))
            visible &= ~(possible & (low <= high) & (low < 1. - 1e-6))
        seen |= visible
    return seen


@dataclass
class TeamObservation:
    team: int
    time: float
    positions: np.ndarray
    active: np.ndarray
    limits: np.ndarray
    base: np.ndarray
    detections: np.ndarray
    remaining: int


def observe(problem, state, team, cpp=None):
    """Sensor-only environment adapter; detection identity is never exposed.

    Decode range/bearing from the same semantic beams available to the actor.
    The old six memory scalars are rebuilt from the causal tracker below.
    """
    cpp = cpp if cpp is not None else getattr(problem, '_local_cpp', None)
    robots = TEAMS[team]
    positions = np.array([state[problem.state_idxs[r], 0] for r in robots])
    active = np.array([problem.is_active(state, r) for r in robots], dtype=bool)
    base = np.zeros((2, BASE_DIM), np.float32)
    detections = []; detection_origins = []
    extent = float(problem.lidar_max_range())
    for local, robot in enumerate(robots):
        enc = (cpp.policy_encoding(state, robot) if cpp is not None else problem.policy_encoding(state, robot))
        base[local, :264] = np.asarray(enc).reshape(-1)[:264]
        if not active[local]:
            continue
        ranges = base[local, 8:264].reshape(64, 4)[:, 1]
        # Keep distinct hits on adjoining beams when their inferred centers
        # differ; two anonymous objects may share an angular neighborhood.
        for beam in np.flatnonzero(ranges < 1. - 1e-6):
            point = positions[local] + extent * ranges[beam] * problem.lidar_directions[beam]
            if not any(np.linalg.norm(point - old) < 0.65 for old in detections):
                detections.append(point)
                detection_origins.append(positions[local])
    # A measurement can lie slightly inside a box after angular quantization.
    # Project it to a nearby legal point rather than consult the true target.
    boxes = [np.asarray(b) for b in problem.obstacles]
    bounds = problem.state_lims[:2]
    clean = []
    for point,origin in zip(detections,detection_origins):
        if not legal_points(point[None], boxes, bounds)[0]:
            offset=point-origin;distance=np.linalg.norm(offset);angle=np.arctan2(offset[1],offset[0])
            angles=angle+np.array([0.,.5,-.5,1.,-1.])*np.pi/64.
            candidates=origin+distance*np.stack((np.cos(angles),np.sin(angles)),-1)
            good = candidates[legal_points(candidates, boxes, bounds)]
            if not len(good):
                # Angular quantization can hide the tiny feasible part of a
                # beam cone at an arena/box corner. Project the measurement,
                # never the true object position, onto known free space.
                point=np.clip(point,bounds[:,0]+1e-3,bounds[:,1]-1e-3)
                candidates=[point]
                for box in boxes:
                    for axis in range(2):
                        for side in range(2):
                            candidate=point.copy();candidate[axis]=box[axis,side]+(-1e-3 if side==0 else 1e-3)
                            candidates.append(candidate)
                candidates=np.asarray(candidates)
                good=candidates[legal_points(candidates,boxes,bounds)]
                if not len(good):
                    raise RuntimeError('Known map has no legal support for a positive sensor return')
                good=good[np.argsort(np.linalg.norm(good-point,axis=1))]
            point = good[0]
        if not any(np.linalg.norm(point - old) < 0.65 for old in clean):
            clean.append(point)
    # At most two opponents; prefer detections supported by more than one
    # teammate. Selection and ordering depend solely on sensor output.
    clean = sorted(clean, key=lambda p: min(np.linalg.norm(p - q) for q in positions[active])) if active.any() else []
    limits = np.array([problem.action_lims[problem.action_idxs[r]] for r in robots])
    return TeamObservation(team, float(state[problem.time_idx, 0]), positions, active,
                           limits, base, np.asarray(clean[:2]).reshape(-1, 2), int(active.sum()))


class TeamBelief:
    """Bootstrap joint position/velocity particles with anonymous track slots."""
    def __init__(self, obstacles, bounds, team, seed=0, particles=64, max_range=None):
        self.obstacles = [np.array(b, dtype=float) for b in obstacles]
        self.bounds = np.array(bounds, dtype=float)
        self.team = int(team)
        self.max_speed = 1.2 if self.team == 0 else 1.0
        self.n = int(particles)
        self.rng = np.random.default_rng(seed)
        self.max_range = float(max_range or np.linalg.norm(self.bounds[:, 1] - self.bounds[:, 0]))
        self.particles = np.zeros((self.n, 2, 4), dtype=float)
        self.age = np.full(2, 80.)
        self.visible = np.zeros(2, bool)
        self.existence = np.ones(2)
        self.time = None
        self.last_observation = None
        self.reacquisition_times = []
        self.recovery_count = 0

    def negative_visible(self, points, obs):
        """A farther object on an occupied semantic beam is not ruled out.

        The sensor stores only the nearest opposing return per beam. Treating
        its whole geometric field as empty incorrectly eliminates a second
        anonymous object sharing that beam.
        """
        points = np.asarray(points).reshape(-1,2)
        native=native_geometry()
        if native is not None:
            return np.asarray(native.local_geometry_negative(points,obs.positions,obs.active.astype(float),obs.base,self.obstacles,self.max_range)).reshape(-1)
        result = np.zeros(len(points),bool)
        for local in np.flatnonzero(obs.active):
            origin=obs.positions[local]
            delta=points.astype(np.float32).astype(float)-origin.astype(np.float32).astype(float)
            beam=np.rint(np.arctan2(delta[:,1],delta[:,0])*64/(2*np.pi)).astype(int)%64
            packet=obs.base[local,8:264].reshape(64,4)[beam]
            distance=np.linalg.norm(delta,axis=1)
            seen=(distance>1e-8)&(distance<=self.max_range)&(distance<packet[:,2]*self.max_range-1e-5)
            seen &= np.abs(distance-packet[:,0]*self.max_range)>1e-5
            observed=packet[:,1]*self.max_range
            blocked=(observed<self.max_range*(1.-1e-6)) & (distance>=observed-.65)
            result |= seen & ~blocked
        return result

    def prior(self, count, obs, detections=()):
        points = []
        origins = obs.positions[obs.active]
        for _ in range(100):
            draw = self.rng.uniform(self.bounds[:, 0] + .01, self.bounds[:, 1] - .01, (max(64, count * 4), 2))
            valid = legal_points(draw, self.obstacles, self.bounds)
            seen = self.negative_visible(draw, obs)
            if len(detections):
                near = np.min(np.linalg.norm(draw[:, None] - np.asarray(detections)[None], axis=-1), axis=1) < .8
                seen &= ~near
            points.extend(draw[valid & ~seen].tolist())
            if len(points) >= count:
                return np.asarray(points[:count])
        if points:
            return np.asarray(points)[self.rng.integers(len(points),size=count)]
        # Narrow beam shadows at corners may have very little prior mass.
        # Add deterministic geometry support, still enforcing observations.
        corners=[]
        for box in self.obstacles:
            for x in box[0]:
                for y in box[1]:
                    for scale in (.001,.01,.05,.1,.25):
                        corners.extend(np.array([x,y])+scale*np.array([[-1.,-1.],[-1.,1.],[1.,-1.],[1.,1.]]))
        if corners:
            corners=np.asarray(corners)
            corners=corners[legal_points(corners,self.obstacles,self.bounds)&~self.negative_visible(corners,obs)]
            if len(corners):
                return corners[self.rng.integers(len(corners),size=count)]
        # When the whole arena is observable the posterior must lie near a
        # measurement. There is deliberately no hidden-state fallback.
        if len(detections):
            draw = np.asarray(detections)[self.rng.integers(len(detections), size=count)]
            return np.array(draw, copy=True)
        raise RuntimeError('No observation-consistent opponent support; sensor/model mismatch: '
                           f'team={obs.team} t={obs.time} remaining={obs.remaining} detections={obs.detections.tolist()} origins={obs.positions.tolist()}')

    def update(self, obs):
        if obs.team != self.team:
            raise ValueError('Cannot fuse an opposing team observation')
        if self.time is not None and obs.time < self.time:
            raise ValueError('Reset belief between episodes')
        if self.time == obs.time:
            self.last_observation = obs
            return self
        dt = 0. if self.time is None else obs.time - self.time
        old_mean = self.particles[:, :, :2].mean(0)
        old_visible = self.visible.copy()
        old_age = self.age.copy()
        if self.time is None and len(obs.detections) < obs.remaining:
            for slot in range(2):
                self.particles[:, slot, :2] = self.prior(self.n, obs, obs.detections)
        elif dt > 0:
            self.particles[:, :, 2:] = np.clip(.8 * self.particles[:, :, 2:] + self.rng.normal(0, .25, (self.n, 2, 2)), -self.max_speed, self.max_speed)
            for slot in range(2):
                start = self.particles[:, slot, :2].copy()
                proposed = start + dt * self.particles[:, slot, 2:]
                good = legal_points(proposed, self.obstacles, self.bounds) & clear_segments(start,proposed,self.obstacles)
                self.particles[good, slot, :2] = proposed[good]
                self.particles[~good, slot, 2:] = 0.
            self.age += dt
        self.visible[:] = False
        count = min(obs.remaining, len(obs.detections))
        detected = obs.detections[:count]
        # Assign anonymous measurements to previous slots, never robot IDs.
        means = self.particles[:, :, :2].mean(0)
        assignment = min(permutations(range(2), count),
                         key=lambda slots: sum(np.linalg.norm(means[s] - d) for s, d in zip(slots, detected))) if count else ()
        for slot, detection in zip(assignment, detected):
            if self.time is not None and not old_visible[slot] and old_age[slot] < 80.:
                self.reacquisition_times.append(float(old_age[slot] + dt))
            velocity = np.clip((detection - old_mean[slot]) / max(dt, 1.), -self.max_speed, self.max_speed) if self.time is not None and old_age[slot] < 4 else np.zeros(2)
            distance = min(np.linalg.norm(detection-q) for q in obs.positions[obs.active])
            sigma = max(.03, distance*np.pi/64./np.sqrt(3.))
            measured = detection + self.rng.normal(0.,sigma,(self.n,2))
            good = legal_points(measured,self.obstacles,self.bounds) & visible_points(measured,obs.positions[obs.active],self.obstacles,self.max_range)
            measured[~good] = detection
            self.particles[:, slot, :2] = measured
            self.particles[:, slot, 2:] = velocity
            self.visible[slot] = True
            self.age[slot] = 0.
        alive_slots = list(assignment) + [s for s in range(2) if s not in assignment]
        self.existence[:] = 0.
        self.existence[alive_slots[:obs.remaining]] = 1.
        origins = obs.positions[obs.active]
        for slot in alive_slots[:obs.remaining]:
            if self.visible[slot]:
                continue
            positions = self.particles[:, slot, :2]
            valid = legal_points(positions, self.obstacles, self.bounds)
            seen = self.negative_visible(positions, obs)
            # A slot representing an undetected object cannot duplicate a
            # currently visible detection (unless measurements merged).
            valid &= ~seen
            good = np.flatnonzero(valid)
            if len(good):
                self.particles[:, slot] = self.particles[self.rng.choice(good, self.n), slot]
            else:
                self.recovery_count += 1
                self.particles[:, slot, :2] = self.prior(self.n, obs, detected)
                self.particles[:, slot, 2:] = 0.
        self.time = obs.time
        self.last_observation = obs
        return self

    def features(self, obs, local, without_belief=False):
        result = np.zeros(OBS_DIM, np.float32)
        result[:BASE_DIM] = obs.base[local]
        mean = self.particles.mean(0)
        var = self.particles[:, :, :2].var(0)
        live = np.flatnonzero(self.existence > .5)
        if len(live):
            slot = min(live, key=lambda s: np.linalg.norm(mean[s, :2] - obs.positions[local]))
            result[264:270] = np.r_[(mean[slot, :2] - obs.positions[local]) / 20.,
                                        mean[slot, 2:] / 2.4, min(self.age[slot] / 80., 1.), self.visible[slot]]
        cursor = BASE_DIM
        for slot in range(2):
            result[cursor:cursor + TRACK_DIM] = np.r_[
                (mean[slot, :2] - obs.positions[local]) / 20., mean[slot, 2:] / 2.4,
                var[slot] / 400., min(self.age[slot] / 80., 1.), self.visible[slot], self.existence[slot]]
            cursor += TRACK_DIM
        other = 1 - local
        result[cursor:cursor + 3] = np.r_[(obs.positions[other] - obs.positions[local]) / 20., obs.active[other]]
        result[-2:] = obs.limits[local, :, 1] / 1.2
        if without_belief:
            result[264:BASE_DIM + 2 * TRACK_DIM] = 0.
        return np.clip(result, -1., 1.)

    def batch(self, obs, without_belief=False):
        from cpp.build import bindings
        if hasattr(bindings, 'local_belief_features'):
            result = np.asarray(bindings.local_belief_features(obs.base,obs.positions,obs.active.astype(np.float32),
                    obs.limits.reshape(4,2),self.particles.reshape(self.n,8),self.age,
                    self.visible.astype(np.float32),self.existence)).copy()
            if without_belief:
                result[:,264:BASE_DIM+2*TRACK_DIM]=0.
            return result
        return np.array([self.features(obs, i, without_belief) for i in range(2)])

    def state_dict(self):
        import copy
        return copy.deepcopy(self.__dict__)

    def load_state_dict(self, state):
        import copy
        self.__dict__.update(copy.deepcopy(state))


@dataclass
class EpisodeSpec:
    seed: int
    obstacles: list
    initial_state: list
    action_lims: list
    control_lims: list
    layout: str
    protocol: str = PROTOCOL

    @property
    def fingerprint(self):
        return hashlib.sha256(json.dumps(asdict(self), sort_keys=True).encode()).hexdigest()

    def restore(self):
        from problems.problem import get_problem
        if self.protocol != PROTOCOL:
            raise ValueError('Unsupported local-observation episode protocol')
        p = get_problem('example8')
        p.obstacles = [np.asarray(b) for b in self.obstacles]
        p.swept_collision = True
        p.use_minco_dynamics = False
        p.evader_spawn_near_obstacles = False
        p.action_lims = np.asarray(self.action_lims)
        p.control_lims = np.asarray(self.control_lims)
        p.current_evader_speed_lim = float(p.control_lims[0, 1])
        p.current_pursuer_speed_lim = float(p.control_lims[4, 1])
        p.clear_closed_loop_cache()
        return p, np.asarray(self.initial_state).copy()


def connected_map(obstacles, bounds):
    """Conservative half-meter free-space flood fill for generated maps."""
    axes = [np.arange(lo + .25, hi, .5) for lo, hi in bounds]
    grid = np.stack(np.meshgrid(*axes, indexing='ij'), -1)
    free = legal_points(grid, obstacles, bounds)
    cells = np.argwhere(free)
    if not len(cells):
        return False
    seen = {tuple(cells[0])}; stack = list(seen)
    while stack:
        x, y = stack.pop()
        for nx, ny in ((x-1,y),(x+1,y),(x,y-1),(x,y+1)):
            if 0 <= nx < free.shape[0] and 0 <= ny < free.shape[1] and free[nx,ny] and (nx,ny) not in seen:
                seen.add((nx,ny)); stack.append((nx,ny))
    return len(seen) == len(cells)


def random_map(rng, bounds):
    for _ in range(200):
        boxes = []
        for _ in range(int(rng.integers(2, 5))):
            size = rng.uniform(1.5, 3., 2)
            center = rng.uniform(bounds[:,0] + 3., bounds[:,1] - 3.)
            box = np.stack((center - size / 2, center + size / 2), -1)
            if any(np.all(box[:,0] < old[:,1] + 1.) and np.all(old[:,0] < box[:,1] + 1.) for old in boxes):
                break
            boxes.append(box)
        else:
            if connected_map(boxes, bounds):
                return boxes
    raise RuntimeError('Unable to generate connected obstacle layout')


def make_episode(seed, layout='two', require_occluded=False):
    from benchmark_baselines import configure_problem
    # initialize() uses legacy numpy RNG; preserve the caller's RNG exactly.
    old = np.random.get_state()
    try:
        np.random.seed(int(seed))
        p = configure_problem('none' if layout == 'random' else layout)
        p.swept_collision = True
        if layout == 'random':
            p.obstacles = random_map(np.random.default_rng(seed), p.state_lims[:2])
        for _ in range(2000):
            state = p.initialize()
            if not require_occluded or any(len(observe(p, state, t).detections) < 2 for t in range(2)):
                return EpisodeSpec(int(seed), [b.tolist() for b in p.obstacles], state.tolist(),
                                   p.action_lims.tolist(), p.control_lims.tolist(), layout)
        raise RuntimeError('Unable to sample requested occlusion stratum')
    finally:
        np.random.set_state(old)


def curriculum_episode(seed, progress):
    rng = np.random.default_rng(seed)
    weights = [.1, .45, .45, 0.] if progress < .2 else [.1, .2, .2, .5]
    layout = rng.choice(['none', 'two', 'four', 'random'], p=weights)
    return make_episode(seed, layout, require_occluded=(progress >= .2 and layout != 'none' and rng.random() < .5))
