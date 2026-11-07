"""Version 1 local-search reliability repair, isolated from frozen experiments.

The controller accepts sensor observations only. Search target samples come
from the public-history particle estimator, never an environment state. A
failed simulation is an engineering failure, not a zero-valued terminal.
No network, new reward or geometric candidate enrichment is introduced here.
"""
import time

import numpy as np
from numba import njit

from solvers.local_pursuit import (
    Controller as HistoricalController, move, modal, uniform, safe, visible,
    predict, recover,
)


VERSION = "local_rollout_legality_v1"
CAPTURE, HORIZON, OWN_INVALID, OPPONENT_INVALID, BELIEF_FAILURE = range(5)
TERMINATIONS = ("capture", "horizon", "own_invalid", "opponent_invalid", "belief_failure")


class SimulationReliabilityError(RuntimeError):
    """The search cannot return a scientifically usable action/value."""


@njit(cache=True)
def checked_safe(at, action, speed, boxes, count):
    return (np.all(np.isfinite(at)) and np.all(np.isfinite(action))
            and np.isfinite(speed) and speed > 0
            and safe(at, action, speed, boxes, count))


@njit(cache=True)
def proposal(at, base, speed, boxes, rng, count, audit):
    # audit: repaired bases, rejected noisy draws, exhausted uniform sampling,
    # invariant failures. Keep the historical 50/50 mixture and draw order
    # whenever the historical proposal and fallback were already lawful.
    if not checked_safe(at, np.zeros(2), speed, boxes, count):
        audit[3] += 1
        raise ValueError("Invalid simulated starting position or speed")
    if uniform(rng) < .5:
        noise = np.empty(2)
        for k in range(2):
            noise[k] = np.sqrt(-2*np.log(uniform(rng)))*np.cos(2*np.pi*uniform(rng))*.1*speed
        result = np.minimum(speed, np.maximum(-speed, base+noise))
        if checked_safe(at, result, speed, boxes, count):
            return result, True
        audit[1] += 1
        if checked_safe(at, base, speed, boxes, count):
            return base.copy(), True
        audit[0] += 1
        # Stop is always checked too: never return an unchecked base or point.
        return np.zeros(2), True
    for _ in range(1000):
        result = speed*np.array([2*uniform(rng)-1, 2*uniform(rng)-1])
        if checked_safe(at, result, speed, boxes, count):
            return result, False
    audit[2] += 1
    return np.zeros(2), False


@njit(cache=True)
def world_step(at, target, action, speed, boxes, rng, count, absolute_step, audit):
    """Joint propagation; no invalid transition is applied or scored."""
    if not checked_safe(at, action, speed, boxes, count):
        return at.copy(), target.copy(), 0., OWN_INVALID, False
    if not checked_safe(target, np.zeros(2), 1., boxes, count):
        return at.copy(), target.copy(), 0., OPPONENT_INVALID, False
    away = target-at
    away /= max(1., np.max(np.abs(away)))
    other, _ = proposal(target, away, 1., boxes, rng, count, audit)
    if not checked_safe(target, other, 1., boxes, count):
        return at.copy(), target.copy(), 0., OPPONENT_INVALID, False
    nxt, end = at+action, target+other
    count[1] += 1
    d0, d1 = np.linalg.norm(target-at), np.linalg.norm(end-nxt)
    if d1 < 1.:
        return nxt, end, .5*(1.-absolute_step/80.), CAPTURE, False
    reward = .03*min(1., max(-1., (d0-d1)/4.))
    return nxt, end, reward, -1, visible(nxt, end, boxes, count)


@njit(cache=True)
def simulation(at0, particles0, last0, history0, speed, remaining, vertices, dist,
               boxes, rng, count, parents, actions, visits, totals, size,
               terminations, audit, lengths):
    particles = particles0.copy()
    target = particles[min(len(particles)-1, int(uniform(rng)*len(particles)))].copy()
    at, last = at0.copy(), last0.copy()
    horizon = min(100, remaining)
    history = np.empty((len(history0)+horizon, 2))
    history[:len(history0)] = history0
    hn, node, pn, expanded = len(history0), 0, 1, False
    path = np.zeros(horizon+1, np.int64)
    rewards = np.zeros(horizon)
    reason, steps = HORIZON, 0
    for depth in range(horizon):
        if not expanded:
            width = max(1, int(np.ceil(2*np.sqrt(max(1, visits[node])))))
            children = 0
            for c in range(1, size):
                children += int(parents[c] == node)
            if children < width:
                base = move(at, modal(particles), speed, vertices, dist, boxes, count)
                action, _ = proposal(at, base, speed, boxes, rng, count, audit)
                child = size
                size += 1
                parents[child], actions[child] = node, action
                expanded = True
            else:
                child, best = -1, -1e100
                for c in range(1, size):
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
            action, _ = proposal(at, base, speed, boxes, rng, count, audit)
        nxt, end, reward, status, seen = world_step(
            at, target, action, speed, boxes, rng, count, 80-remaining+depth+1, audit)
        if status >= OWN_INVALID:
            reason = status
            break
        rewards[depth] = reward
        steps += 1
        if status == CAPTURE:
            reason = CAPTURE
            break
        at, target = nxt, end
        # Nothing after the last legal transition will be used. In particular,
        # do not mislabel an unnecessary final filter failure as early cutoff.
        if depth+1 == horizon:
            break
        if seen:
            particles[:] = target
            last, hn = target.copy(), 0
        else:
            history[hn] = at
            hn += 1
            if not predict(particles, at, boxes, rng, count):
                fresh, support = recover(last, history[:hn], boxes, rng, count)
                if not support:
                    reason = BELIEF_FAILURE
                    break
                particles[:] = fresh
    terminations[reason] += 1
    lengths[0] += steps
    lengths[1] += horizon
    # Never backpropagate a fabricated terminal value after estimator failure.
    if reason < OWN_INVALID:
        for j in range(pn):
            visits[path[j]] += 1
            totals[path[j]] += np.sum(rewards[max(0, j-1):steps])
    return size, reason


class Controller(HistoricalController):
    """G/T reliability baseline; visible targets share direct local geometry.

    This is a new algorithm version, not an exact replay of historical T.
    T+Geo and neural priors are intentionally gated on the reliability audit.
    """
    def __init__(self, public, method, seed, simulations=1000, deadline_ms=50.):
        if method not in ("G", "T"):
            raise ValueError("Reliability stage supports G and T only")
        if simulations <= 0 or not np.isfinite(deadline_ms) or deadline_ms < 0:
            raise ValueError("Invalid simulation/time budget")
        super().__init__(public, method, seed, simulations, deadline_ms)
        self.version = VERSION

    def update(self, obs, count):
        p = self.public
        if (np.shape(obs.position) != (2,) or not np.all(np.isfinite(obs.position))
                or not isinstance(obs.step, (int, np.integer)) or not 0 <= obs.step <= 80
                or obs.remaining != 80-obs.step
                or not checked_safe(obs.position, np.zeros(2), p.speed, p.boxes, count)):
            raise ValueError("Invalid public observation position or clock")
        if obs.target is not None:
            if (np.shape(obs.target) != (2,) or not np.all(np.isfinite(obs.target))
                    or not checked_safe(obs.target, np.zeros(2), 1., p.boxes, count)
                    or not visible(obs.position, obs.target, p.boxes, count)):
                raise ValueError("Positive observation violates the public sensor")
        if self.observations:
            previous = self.observations[-1]
            at = np.asarray(previous["position"])
            if (obs.step != previous["step"]+1
                    or not checked_safe(at, obs.position-at, p.speed, p.boxes, count)):
                raise ValueError("Public history is not consecutive lawful motion")
        super().update(obs, count)

    def act(self, obs, fixed_simulations=None):
        started = time.perf_counter()
        if obs.remaining <= 0 or obs.remaining != 80-obs.step:
            raise ValueError("Remaining time must preserve the original clock")
        cap = self.simulations if fixed_simulations is None else fixed_simulations
        if not isinstance(cap, (int, np.integer)) or cap <= 0:
            raise ValueError("Simulation budget must be a positive integer")
        if not np.all(np.isfinite(obs.position)) or (obs.target is not None and not np.all(np.isfinite(obs.target))):
            raise ValueError("Non-finite sensor observation")
        p, count = self.public, np.zeros(2, np.int64)
        self.last_stats = dict(version=VERSION, status="updating_estimator")
        try:
            self.update(obs, count)
        except RuntimeError as exc:
            self.last_stats.update(status="estimator_failure", error=str(exc))
            raise SimulationReliabilityError("Real public-history estimator failed") from exc
        target = obs.target if obs.target is not None else modal(self.particles)
        action = move(obs.position, target, p.speed, p.vertices, p.dist, p.boxes, count)
        terminations, audit, lengths = np.zeros(5, np.int64), np.zeros(4, np.int64), np.zeros(2, np.int64)
        completed, roots, failure = 0, [], None
        if self.method == "T" and obs.target is None:
            parents, actions = np.full(cap+1, -1, np.int64), np.zeros((cap+1, 2))
            visits, totals, size = np.zeros(cap+1, np.int64), np.zeros(cap+1), 1
            for _ in range(cap):
                if fixed_simulations is None and self.deadline_ms > 0 and 1000*(time.perf_counter()-started) >= self.deadline_ms:
                    break
                size, reason = simulation(
                    obs.position, self.particles, self.last, np.asarray(self.history).reshape(-1, 2),
                    p.speed, obs.remaining, p.vertices, p.dist, p.boxes, self.rng, count,
                    parents, actions, visits, totals, size, terminations, audit, lengths)
                if reason >= OWN_INVALID:
                    failure = TERMINATIONS[reason]
                    break
                completed += 1
            ids = [i for i in range(1, size) if parents[i] == 0 and visits[i] > 0]
            if ids:
                best = max(ids, key=lambda i: totals[i]/visits[i])
                action = actions[best].copy()
            roots = [dict(action=actions[i].tolist(), visits=int(visits[i]),
                          value=float(totals[i]/visits[i])) for i in ids]
        valid_action = checked_safe(obs.position, action, p.speed, p.boxes, count)
        if not valid_action:
            failure = "real_action_invalid"
        latency = 1000*(time.perf_counter()-started)
        self.last_stats = dict(
            version=VERSION, status="failed" if failure else "passed", failure=failure,
            simulations=completed, simulation_attempts=int(terminations.sum()),
            terminations={k: int(v) for k, v in zip(TERMINATIONS, terminations)},
            repaired_bases=int(audit[0]), rejected_noise=int(audit[1]), uniform_exhausted=int(audit[2]),
            simulated_steps=int(lengths[0]), available_steps=int(lengths[1]),
            transitions=int(count[1]), geometry_queries=int(count[0]), latency_ms=latency,
            source="visible_geometry" if obs.target is not None else self.method,
            deadline_fallback=bool(self.method == "T" and obs.target is None and completed == 0 and not failure),
            root=roots, recoveries=self.recoveries,
        )
        if failure:
            raise SimulationReliabilityError(f"Search rejected: {failure}")
        return action
