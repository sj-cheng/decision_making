"""Causal one-target controllers. No controller API accepts an environment state.

The sensor is separate from the historical 64-beam protocol. Numeric kernels use
Numba to fit belief updates and root-sampled open-loop search in 50 ms.
"""
import copy
from dataclasses import dataclass
import time
import numpy as np
from numba import njit


@njit(cache=True)
def uniform(rng):
    rng[0] = (1664525 * rng[0] + 1013904223) % 4294967296
    return (rng[0] + .5) / 4294967296.


@njit(cache=True)
def legal(q, boxes, count):
    count[0] += 1
    if np.any(q < -10.) or np.any(q > 10.):
        return False
    for b in boxes:
        if b[0, 0] <= q[0] <= b[0, 1] and b[1, 0] <= q[1] <= b[1, 1]:
            return False
    return True


@njit(cache=True)
def clear(a, b, boxes, count):
    count[0] += 1
    for box in boxes:
        lo, hi = 0., 1.
        hit = True
        for axis in range(2):
            d = b[axis] - a[axis]
            if abs(d) < 1e-12:
                if a[axis] < box[axis, 0] or a[axis] > box[axis, 1]:
                    hit = False
            else:
                x, y = (box[axis, 0] - a[axis]) / d, (box[axis, 1] - a[axis]) / d
                lo, hi = max(lo, min(x, y)), min(hi, max(x, y))
        if hit and lo <= hi:
            return False
    return True


@njit(cache=True)
def visible(at, q, boxes, count):
    return np.sum((q-at)**2) <= 36. + 1e-12 and clear(at, q, boxes, count)


@njit(cache=True)
def safe(at, action, limit, boxes, count):
    return np.all(np.abs(action) <= limit + 1e-10) and legal(at+action, boxes, count) and clear(at, at+action, boxes, count)


@njit(cache=True)
def edge(a, b, speed):
    return int(np.ceil(np.max(np.abs(a-b))/speed - 1e-9))


@njit(cache=True)
def graph(vertices, boxes, speed, count):
    n = len(vertices)
    dist = np.full((n, n), 1000000, np.int64)
    for i in range(n):
        for j in range(n):
            if clear(vertices[i], vertices[j], boxes, count):
                dist[i, j] = edge(vertices[i], vertices[j], speed)
    for k in range(n):
        for i in range(n):
            for j in range(n):
                dist[i, j] = min(dist[i, j], dist[i, k]+dist[k, j])
    return dist


@njit(cache=True)
def route(at, target, speed, vertices, dist, boxes, count):
    best, first = 1000000, -1
    if clear(at, target, boxes, count):
        best = edge(at, target, speed)
    n = len(vertices)
    tail = np.full(n, 1000000, np.int64)
    for j in range(n):
        if clear(vertices[j], target, boxes, count):
            cost = edge(vertices[j], target, speed)
            for i in range(n):
                tail[i] = min(tail[i], dist[i, j]+cost)
    for i in range(n):
        if clear(at, vertices[i], boxes, count):
            cost = edge(at, vertices[i], speed)+tail[i]
            if cost < best:
                best, first = cost, i
    return best, first


@njit(cache=True)
def move(at, target, speed, vertices, dist, boxes, count):
    _, first = route(at, target, speed, vertices, dist, boxes, count)
    waypoint = vertices[first] if first >= 0 else target
    delta = waypoint-at
    a = delta / max(1., np.max(np.abs(delta))/speed)
    return a if safe(at, a, speed, boxes, count) else np.zeros(2)


@njit(cache=True)
def modal(particles):
    # Fixed unit cells; ties break lexicographically. Return a center sample,
    # never a mean that may lie inside an obstacle.
    winner, mass, key_best = 0, -1, 1000000
    for i in range(len(particles)):
        x, y = np.floor(particles[i, 0]), np.floor(particles[i, 1])
        key = int((x+10)*21+y+10)
        n, center = 0, np.zeros(2)
        for j in range(len(particles)):
            if np.floor(particles[j, 0]) == x and np.floor(particles[j, 1]) == y:
                n += 1
                center += particles[j]
        if n > mass or (n == mass and key < key_best):
            mass, key_best = n, key
            center /= n
            distance = 1e100
            for j in range(len(particles)):
                if np.floor(particles[j, 0]) == x and np.floor(particles[j, 1]) == y:
                    d = np.sum((particles[j]-center)**2)
                    if d < distance:
                        distance, winner = d, j
    return particles[winner].copy()


@njit(cache=True)
def predict(particles, observer, boxes, rng, count):
    survivors = np.empty_like(particles)
    n = 0
    for q in particles:
        candidate = q + np.array([2*uniform(rng)-1, 2*uniform(rng)-1])
        if legal(candidate, boxes, count) and clear(q, candidate, boxes, count) and not visible(observer, candidate, boxes, count):
            survivors[n] = candidate
            n += 1
    if n:
        for i in range(len(particles)):
            particles[i] = survivors[min(n-1, int(uniform(rng)*n))]
    return n


@njit(cache=True)
def recover(last, observers, boxes, rng, count):
    # Sequential recovery retains complete legal ancestral paths and applies
    # every empty observation. Auxiliary samples keep a 64-particle posterior.
    pool = np.empty((512, 2))
    for attempt in range(8):
        pool[:] = last
        good = True
        for observer in observers:
            if not predict(pool, observer, boxes, rng, count):
                good = False
                break
        if good:
            return pool[:64].copy(), 64
    return np.zeros((64, 2)), 0


@njit(cache=True)
def guided(at, particles, speed, vertices, dist, edist, boxes, count, enriched):
    actions = np.zeros((3, 2))
    sources = np.zeros(3, np.int64)
    actions[0] = move(at, modal(particles), speed, vertices, dist, boxes, count)
    n = 1
    if not enriched:
        return actions, sources, n
    best, chosen = -1., -1
    for i in range(len(vertices)):
        steps, _ = route(at, vertices[i], speed, vertices, dist, boxes, count)
        if steps <= 0 or steps >= 1000000:
            continue
        mass = 0
        for q in particles:
            mass += int(visible(vertices[i], q, boxes, count))
        score = mass / len(particles) / steps
        if score > best and mass:
            best, chosen = score, i
    if chosen >= 0:
        actions[n] = move(at, vertices[chosen], speed, vertices, dist, boxes, count)
        sources[n] = 1
        n += 1
    masses = np.zeros(len(vertices))
    for q in particles:
        _, first = route(at, q, speed, vertices, dist, boxes, count)
        if first >= 0:
            ours, _ = route(at, vertices[first], speed, vertices, dist, boxes, count)
            theirs, _ = route(q, vertices[first], 1., vertices, edist, boxes, count)
            if ours <= theirs:
                masses[first] += 1.
    if len(masses) and np.max(masses) > 0:
        chosen = np.argmax(masses)
        actions[n] = move(at, vertices[chosen], speed, vertices, dist, boxes, count)
        sources[n] = 2
        n += 1
    return actions, sources, n


@njit(cache=True)
def proposal(at, base, speed, boxes, rng, count):
    if uniform(rng) < .5:
        noise = np.empty(2)
        for k in range(2):
            noise[k] = np.sqrt(-2*np.log(uniform(rng)))*np.cos(2*np.pi*uniform(rng))*.1*speed
        result = np.minimum(speed, np.maximum(-speed, base+noise))
        return (result if safe(at, result, speed, boxes, count) else base), True
    for _ in range(1000):
        result = speed*np.array([2*uniform(rng)-1, 2*uniform(rng)-1])
        if safe(at, result, speed, boxes, count):
            return result, False
    return np.zeros(2), False


@njit(cache=True)
def simulation(at0, particles0, last0, history0, speed, remaining, vertices, dist, boxes,
               root_actions, root_sources, root_n, rng, count, parents, actions, sources, visits, totals, tree_size):
    # Each node indexes an action history, never a sampled state or observation.
    particles = particles0.copy()
    target = particles[min(63, int(uniform(rng)*64))].copy()
    at = at0.copy()
    last = last0.copy()
    history = np.empty((len(history0)+remaining, 2))
    history[:len(history0)] = history0
    hn = len(history0)
    node, pn, expanded = 0, 1, False
    path = np.zeros(remaining+1, np.int64)
    rewards = np.zeros(remaining)
    steps = 0
    for depth in range(min(100, remaining)):
        if not expanded:
            width = max(1, int(np.ceil(2*np.sqrt(max(1, visits[node])))))
            children = 0
            for c in range(1, tree_size):
                children += int(parents[c] == node)
            if children < width:
                base = move(at, modal(particles), speed, vertices, dist, boxes, count)
                source = 0
                if depth == 0:
                    k = min(root_n-1, int(uniform(rng)*root_n))
                    base, source = root_actions[k], root_sources[k]
                action, isguided = proposal(at, base, speed, boxes, rng, count)
                child = tree_size
                tree_size += 1
                parents[child], actions[child] = node, action
                sources[child] = source if isguided else 3
                expanded = True
            else:
                child, best = -1, -1e100
                for c in range(1, tree_size):
                    if parents[c] == node:
                        score = totals[c]/max(1, visits[c]) + np.sqrt(np.log(visits[node]+1)/max(1, visits[c]))
                        if score > best:
                            child, best = c, score
                action = actions[child].copy()
            node = child
            path[pn] = node
            pn += 1
        else:
            base = move(at, modal(particles), speed, vertices, dist, boxes, count)
            action, _ = proposal(at, base, speed, boxes, rng, count)
        # Simulated opponent has a public full-position sensor, matching the
        # information advantage of the real Search Only opponent. Its model
        # is an inexpensive evasive rollout, identical for T and V. It never
        # receives our already selected action.
        away = target-at
        away /= max(1., np.max(np.abs(away)))
        other, _ = proposal(target, away, 1., boxes, rng, count)
        d0 = np.linalg.norm(target-at)
        nxt, end = at+action, target+other
        steps += 1
        count[1] += 1
        if not safe(at, action, speed, boxes, count):
            rewards[depth] = -1.
            break
        if not safe(target, other, 1., boxes, count):
            break
        d1 = np.linalg.norm(end-nxt)
        if d1 < 1.:
            rewards[depth] = .5*(1.-(80-remaining+depth+1)/80.)
            break
        rewards[depth] = .03*min(1., max(-1., (d0-d1)/4.))
        at, target = nxt, end
        # Only this sensor result, never target itself, reaches G's belief.
        if visible(at, target, boxes, count):
            particles[:] = target
            last, hn = target.copy(), 0
        else:
            history[hn] = at
            hn += 1
            if not predict(particles, at, boxes, rng, count):
                new, support = recover(last, history[:hn], boxes, rng, count)
                if support:
                    particles[:] = new
                else:
                    # An impossible/too rare simulated history has no value
                    # cutoff; retain no hidden-world truth and stop rollout.
                    break
    # Depth-specific returns; no value network, no information bonus.
    for j in range(pn):
        visits[path[j]] += 1
        totals[path[j]] += np.sum(rewards[max(0, j-1):steps])
    return tree_size


@dataclass
class Observation:
    step: int
    position: np.ndarray
    target: object
    remaining: int


class PublicMap:
    def __init__(self, boxes, speed):
        self.boxes = np.asarray(boxes, float).reshape(-1, 2, 2)
        self.speed = float(speed)
        vertices = []
        for b in self.boxes:
            for x in range(2):
                for y in range(2):
                    vertices.append([b[0, x]+(.05 if x else -.05), b[1, y]+(.05 if y else -.05)])
        vertices += [[-9.95,-9.95],[-9.95,9.95],[9.95,-9.95],[9.95,9.95],
                     [0.,-9.95],[0.,9.95],[-9.95,0.],[9.95,0.]]
        count = np.zeros(2, np.int64)
        self.vertices = np.array([v for v in np.array(vertices) if legal(v, self.boxes, count)])
        self.dist = graph(self.vertices, self.boxes, self.speed, count)
        self.edist = graph(self.vertices, self.boxes, 1., count)

    def observe(self, step, pursuer, target):
        count = np.zeros(2, np.int64)
        seen = visible(pursuer, target, self.boxes, count)
        return Observation(step, np.array(pursuer, copy=True), np.array(target, copy=True) if seen else None, 80-step)


class Controller:
    def __init__(self, public, method, seed, simulations=1000, deadline_ms=50.):
        if method not in ('L','G','T','V','O'):
            raise ValueError(method)
        self.public, self.method = public, method
        self.rng = np.array([int(seed) % 4294967296], np.int64)
        # Estimator stream is separate from search, so common histories and
        # estimator states yield identical posteriors for G/T/V.
        self.filter_rng = np.array([(int(seed)+60020261024) % 4294967296], np.int64)
        self.particles = np.zeros((64, 2))
        self.last = None
        self.history = []
        self.observations = []
        self.recoveries = 0
        self.simulations, self.deadline_ms = simulations, deadline_ms
        self.patrol_seen = set()
        self.patrol_goal = None
        self.last_stats = {}

    def snapshot(self):
        return copy.deepcopy({k:v for k,v in self.__dict__.items() if k != 'public'})

    def restore(self, snapshot):
        self.__dict__.update(copy.deepcopy(snapshot))

    def update(self, obs, count):
        self.observations.append(dict(step=obs.step, position=obs.position.tolist(),
                                      target=None if obs.target is None else obs.target.tolist()))
        if obs.target is not None:
            self.last = obs.target.copy()
            self.history = []
            self.particles[:] = obs.target
            self.patrol_seen.clear()
            self.patrol_goal = None
        else:
            if self.last is None:
                raise ValueError('Initialization requires a legal positive observation')
            self.history.append(obs.position.copy())
            if self.method != 'L':
                if not predict(self.particles, obs.position, self.public.boxes, self.filter_rng, count):
                    self.recoveries += 1
                    particles, support = recover(self.last, np.asarray(self.history), self.public.boxes, self.filter_rng, count)
                    if not support:
                        raise RuntimeError('No lawful belief recovery support')
                    self.particles[:] = particles

    def act(self, obs, fixed_simulations=None):
        started = time.perf_counter()
        p, count = self.public, np.zeros(2, np.int64)
        self.update(obs, count)
        base = move(obs.position, self.last if self.method == 'L' else modal(self.particles),
                    p.speed, p.vertices, p.dist, p.boxes, count)
        if self.method == 'L' and obs.target is None and (np.linalg.norm(obs.position-self.last)<.05 or self.patrol_goal is not None):
            if self.patrol_goal is not None and np.linalg.norm(obs.position-p.vertices[self.patrol_goal]) < .05:
                self.patrol_seen.add(self.patrol_goal)
                self.patrol_goal = None
            if self.patrol_goal is None:
                if len(self.patrol_seen) == len(p.vertices): self.patrol_seen.clear()
                options = [(route(obs.position,q,p.speed,p.vertices,p.dist,p.boxes,count)[0],i)
                           for i,q in enumerate(p.vertices) if i not in self.patrol_seen]
                self.patrol_goal = min(options)[1]
            base = move(obs.position,p.vertices[self.patrol_goal],p.speed,p.vertices,p.dist,p.boxes,count)
        completed, root = 0, []
        selected_source = 'direct'
        generation_ms = 0.
        if self.method in ('T','V'):
            stamp = time.perf_counter()
            guided_actions, guided_sources, n = guided(obs.position,self.particles,p.speed,p.vertices,p.dist,p.edist,p.boxes,count,self.method=='V')
            generation_ms = 1000*(time.perf_counter()-stamp)
            cap = self.simulations if fixed_simulations is None else fixed_simulations
            parents = np.full(cap+1,-1,np.int64)
            actions = np.zeros((cap+1,2))
            sources = np.zeros(cap+1,np.int64)
            visits = np.zeros(cap+1,np.int64)
            totals = np.zeros(cap+1)
            size = 1
            for _ in range(cap):
                if fixed_simulations is None and self.deadline_ms > 0 and 1000*(time.perf_counter()-started) >= self.deadline_ms:
                    break
                size = simulation(obs.position,self.particles,self.last,np.asarray(self.history).reshape(-1,2),p.speed,
                    obs.remaining,p.vertices,p.dist,p.boxes,guided_actions,guided_sources,n,self.rng,count,
                    parents,actions,sources,visits,totals,size)
                completed += 1
            candidates = [i for i in range(1,size) if parents[i]==0 and safe(obs.position,actions[i],p.speed,p.boxes,count)]
            if candidates:
                best = max(candidates,key=lambda i:totals[i]/max(1,visits[i]))
                base = actions[best].copy()
                selected_source = ('pursuit','reobserve','exit','uniform')[sources[best]]
            else:
                selected_source = 'fallback_G'
            root = [dict(action=actions[i].tolist(),source=('pursuit','reobserve','exit','uniform')[sources[i]],
                         visits=int(visits[i]),value=float(totals[i]/max(1,visits[i]))) for i in range(1,size) if parents[i]==0]
        latency = 1000*(time.perf_counter()-started)
        self.last_stats = dict(simulations=completed,transitions=int(count[1]),geometry_queries=int(count[0]),
            latency_ms=latency,overrun=bool(self.method in ('T','V') and self.deadline_ms>0 and latency>self.deadline_ms),
            candidate_ms=generation_ms,source=selected_source,root=root,recoveries=self.recoveries)
        if not safe(obs.position,base,p.speed,p.boxes,count):
            raise RuntimeError('Controller generated illegal real action')
        return base


def warmup():
    for boxes in ([],[[[-1.,1.],[-3.,3.]]]):
        p = PublicMap(boxes,1.1)
        for method in ('L','G','T','V','O'):
            c = Controller(p,method,123,simulations=2,deadline_ms=0)
            c.act(p.observe(0,np.array([-8.,4.]),np.array([-2.1,4.])),fixed_simulations=2)
            c.act(Observation(1,np.array([-8.9,4.]),None,79),fixed_simulations=2)
