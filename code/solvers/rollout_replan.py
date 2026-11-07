"""Single-mechanism research variant: replan G geometry in rollout.

Frozen original algorithms are imported unchanged. The target sample never
enters the public geometry callback; cache uses exact own-position keys.
"""
import time
import numpy as np
from numba import njit, objmode
from solvers.frozen_d_search_interface import (
    Controller as OriginalController, COMPASS, root_sample, target_step,
    geometric_candidates, belief_view, move, safe, sensor_visible)

VERSION = "frozen_D_rollout_replan_v1"
_active = None


def make_context(controller, started, fixed):
    e,p=controller.estimator,controller.public
    return dict(samples=e.particles.copy(), weights=e.particle_weights.copy(),
        labels=None if e.graph is None else e.regions[e.ids].copy(),speed=p.speed,
        vertices=p.vertices,dist=p.dist,boxes=p.boxes,motion_boxes=controller.motion_boxes,
        deadline=float('inf') if fixed else started+controller.budget_ms/1000.,
        cache={},cache_enabled=controller.cache_enabled,hits=0,misses=0,query_ms=0.,max_query_ms=5.)


def step_allowed():
    return time.perf_counter() < _active['deadline']-.002


def rollout_action(at):
    ctx=_active
    key=(float(at[0]),float(at[1]))
    if ctx['cache_enabled'] and key in ctx['cache']:
        ctx['hits']+=1
        return ctx['cache'][key].copy(),step_allowed()
    # Conservative query admission, with all work charged to this decision.
    if time.perf_counter()+(max(5.,2*ctx['max_query_ms'])+2.)/1000. >= ctx['deadline']:
        return np.zeros(2),False
    stamp=time.perf_counter()
    _,action=geometric_candidates(at,ctx['samples'],ctx['weights'],ctx['labels'],ctx['speed'],
        ctx['vertices'],ctx['dist'],ctx['boxes'],ctx['motion_boxes'])
    elapsed=1000*(time.perf_counter()-stamp)
    ctx['misses']+=1;ctx['query_ms']+=elapsed
    ctx['max_query_ms']=max(ctx['max_query_ms'],elapsed)
    if ctx['cache_enabled']:ctx['cache'][key]=action.copy()
    return action,step_allowed()


@njit(cache=True)
def one_simulation(at0, speed, horizon, points, links, lengths, probabilities,
                   ids, weights, modes, roots, base, boxes, motion_boxes, rng,
                   children, tree_actions, visits, totals, size):
    sample, target, mode = root_sample(ids, weights, modes, rng)
    at = at0.copy()
    node = 0
    path = np.zeros(horizon+1, np.int64)
    pathn = 1
    expanded, seen_once = False, False
    rewards = np.zeros(horizon)
    trace = np.zeros((horizon, 10))
    count = np.zeros(2, np.int64)
    steps, reason = 0, 1  # reason=1: registered planning leaf; 0: capture
    for depth in range(horizon):
        with objmode(allowed="boolean"):
            allowed = step_allowed()
        if not allowed:
            return size, trace[:steps].copy(), sample, 2
        if not expanded:
            options = roots if node == 0 else COMPASS*speed
            child, chosen, best = -1, -1, -1e100
            for k in range(len(options)):
                if not safe(at, options[k], speed, motion_boxes, count):
                    continue
                c = children[node, k]
                if c < 0:
                    chosen, child = k, size
                    children[node, k] = child
                    tree_actions[child] = options[k]
                    size += 1
                    expanded = True
                    break
                value = totals[c]/max(1, visits[c])+np.sqrt(2*np.log(visits[node]+1)/max(1, visits[c]))
                if value > best:
                    best, chosen, child = value, k, c
            if child < 0:
                raise RuntimeError('No lawful tree action, including stop')
            action = options[chosen].copy()
            node = child
            path[pathn] = node
            pathn += 1
        else:
            with objmode(action="float64[:]", allowed="boolean"):
                action, allowed = rollout_action(at)
            if not allowed:
                return size, trace[:steps].copy(), sample, 2
        if not safe(at, action, speed, motion_boxes, count):
            raise RuntimeError('Illegal simulated own action')
        new_target, mode = target_step(target, mode, links, lengths, probabilities, rng)
        if not safe(points[target], points[new_target]-points[target], 1., boxes, count):
            raise RuntimeError('Illegal simulated motion edge')
        nxt = at+action
        seen = sensor_visible(nxt, points[new_target], boxes)
        capture = np.linalg.norm(nxt-points[new_target]) < 1.
        rewards[depth] = int(seen and not seen_once)+int(capture)
        trace[depth, :2], trace[depth, 2:4] = at, nxt
        trace[depth, 4:6], trace[depth, 6:8] = points[target], points[new_target]
        trace[depth, 8], trace[depth, 9] = int(seen), int(capture)
        seen_once = seen_once or seen
        at, target = nxt, new_target
        steps += 1
        if capture:
            reason = 0
            break
    for j in range(pathn):
        value = 0.
        for t in range(steps-1, max(0, j-1)-1, -1):
            value = rewards[t]+.95*value
        visits[path[j]] += 1
        totals[path[j]] += value
    return size, trace[:steps].copy(), sample, reason


class Controller(OriginalController):
    def __init__(self, boxes, speed, method, seed, budget_ms=200., simulations=1000, cache_enabled=True):
        if method not in ('T-R','T+Geo-R'):
            raise ValueError(method)
        super().__init__(boxes,speed,method.removesuffix('-R'),seed,budget_ms,simulations)
        self.research_method=method
        self.cache_enabled=cache_enabled

    def act(self, obs, fixed_simulations=None):
        start = time.perf_counter()
        global _active
        self.estimator.update(obs)
        update_ms = 1000*(time.perf_counter()-start)
        e, p = self.estimator, self.public
        stamp = time.perf_counter()
        if obs.target is not None:
            count = np.zeros(2, np.int64)
            base = move(obs.position, obs.target, p.speed, p.vertices, p.dist, p.boxes, count)
            roots, view = base.reshape(1, 2), None
        else:
            view = belief_view(e)
            roots, base = geometric_candidates(obs.position, e.particles, e.particle_weights,
                e.regions[e.ids], p.speed, p.vertices, p.dist, p.boxes, self.motion_boxes)
            if self.method == 'T':
                roots = COMPASS*p.speed
        candidate_ms = 1000*(time.perf_counter()-stamp)
        stamp = time.perf_counter()
        completed, size = 0, 1
        cap = self.simulations if fixed_simulations is None else fixed_simulations
        self.traces = []
        root_counts = np.zeros(64, np.int64)
        terminations = np.zeros(3, np.int64)
        action = base.copy()
        visits, totals = None, None
        # If public update/candidates have exhausted the budget, return G's
        # checked fallback and record an overrun, never hide estimation cost.
        eligible = (view is not None and self.method != 'G' and
                    (fixed_simulations is not None or 1000*(time.perf_counter()-start) < self.budget_ms-2.))
        prepare_ms = 0.
        context = make_context(self, start, fixed_simulations is not None)
        _active = context
        completed_attempts = 0
        if eligible:
            children = np.full((cap+1, 64), -1, np.int32)
            tree_actions = np.zeros((cap+1, 2))
            visits, totals = np.zeros(cap+1, np.int64), np.zeros(cap+1)
            prepare_ms = 1000*(time.perf_counter()-stamp)
            points, links, lengths, probabilities, *_ = e.graph
            for _ in range(cap):
                if fixed_simulations is None and 1000*(time.perf_counter()-start) >= self.budget_ms-2.:
                    break
                size, trace, sample, reason = one_simulation(obs.position, p.speed, min(8, obs.remaining),
                    points, links, lengths, probabilities, *view, roots, base, p.boxes, self.motion_boxes, self.rng,
                    children, tree_actions, visits, totals, size)
                self.traces.append(trace)
                completed_attempts += 1
                root_counts[sample] += 1
                terminations[reason] += 1
                if reason == 2:
                    break
                completed += 1
            if completed:
                candidates = children[0][children[0] >= 0]
                winner = candidates[np.argmax(visits[candidates])]
                action = tree_actions[winner].copy()
        search_ms = 1000*(time.perf_counter()-stamp)
        stamp = time.perf_counter()
        if not safe(obs.position, action, p.speed, p.boxes, np.zeros(2, np.int64)):
            raise RuntimeError('Illegal final action')
        final_check_ms = 1000*(time.perf_counter()-stamp)
        total_ms = 1000*(time.perf_counter()-start)
        self.last_stats = dict(update_ms=update_ms, build_ms=e.last_stats['build_ms'],
            candidate_ms=candidate_ms, tree_prepare_ms=prepare_ms, search_including_prepare_ms=search_ms,
            final_check_ms=final_check_ms, total_ms=total_ms, over_50ms=total_ms > 50.,
            simulations=completed, root_counts=root_counts.tolist(), terminations=terminations.tolist(),
            root_visits=[] if visits is None else visits[children[0][children[0] >= 0]].tolist(),
            root_totals=[] if totals is None else totals[children[0][children[0] >= 0]].tolist(),
            rollout_cache_hits=context['hits'], rollout_cache_misses=context['misses'],
            rollout_queries_ms=context['query_ms'], budget_aborted_simulations=int(terminations[2]),
            attempted_simulations=completed_attempts,
            root_candidates=len(roots), source='visible_geometry' if obs.target is not None else
                'weighted_uct_probe' if completed else 'weighted_geometry', planning_depth=min(8, obs.remaining))
        return action
