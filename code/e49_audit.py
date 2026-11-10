"""Independent sampled-event replay and outcome-blind E49 diagnostics.

Uses the audit orientation/edge geometry, not the simulation transition kernel.
Only the environment-side evaluator sees hidden target positions.
"""
from __future__ import annotations

import numpy as np
from numba import njit
from estimator_audit_geometry import blocked, point_legal


@njit(cache=True)
def replay_motion(positions, targets, actions, enemy, ap, ae, boxes, capture_dt=.01):
    pp, ee = positions.copy(), targets.copy()
    alive = ae.copy()
    times = np.full(2, -1.)
    captors = np.full(2, -1, np.int64)
    count = int(round(1. / capture_dt))
    for k in range(count + 1):
        t = k / count
        pp = positions + actions * t
        for j in range(2):
            if alive[j]:
                ee[j] = targets[j] + enemy[j] * t
        used = np.zeros(2, np.bool_)
        for _ in range(2):
            best, bi, bj = 1., -1, -1
            for i in range(2):
                if not ap[i] or used[i]:
                    continue
                for j in range(2):
                    if not alive[j]:
                        continue
                    d = np.sqrt(np.sum((pp[i]-ee[j])**2))
                    if d < best and not blocked(pp[i], ee[j], boxes):
                        best, bi, bj = d, i, j
            if bi < 0:
                break
            used[bi] = True
            alive[bj] = False
            times[bj], captors[bj] = t, bi
    return pp, ee, ap.copy(), alive, times, captors


def audit_frame(frame, boxes, capture_dt=.01):
    state = np.asarray(frame['state'], float)
    nxt = np.asarray(frame['next_state'], float)
    action = np.asarray(frame['action'], float)
    enemy = np.asarray(frame['opponent_action'], float)
    ap, ae = state[11:13] > .5, state[9:11] > .5
    if not ap.all() or not (nxt[11:13] > .5).all():
        raise RuntimeError('E49 pursuer retired')
    obs=frame['observation']
    np.testing.assert_array_equal(obs['active_p'],ap)
    np.testing.assert_array_equal(obs['active_e'],ae)
    np.testing.assert_allclose(obs['positions'],state[4:8].reshape(2,2),atol=0,rtol=0)
    if obs['step']!=state[8]:raise RuntimeError('Observation clock mismatch')
    for j in range(2):
        for i in range(2):
            p=state[4+2*i:6+2*i];e=state[2*j:2*j+2]
            seen=bool(ap[i] and ae[j] and np.sum((p-e)**2)<=36.+1e-12 and not blocked(p,e,boxes))
            report=obs['reports'][j][i]
            if seen!=(report is not None):raise RuntimeError('Independent observation visibility mismatch')
            if seen:np.testing.assert_allclose(report,e,atol=0,rtol=0)
    for team, at, delta, active, speed in (
        ('pursuer', state[4:8].reshape(2, 2), action, ap, 1.),
        ('evader', state[:4].reshape(2, 2), enemy, ae, 1.5),
    ):
        if delta.shape != (2, 2) or not np.isfinite(delta).all():
            raise RuntimeError('Malformed '+team+' action')
        for j in range(2):
            if not active[j] and np.max(np.abs(delta[j])) > 1e-10:
                raise RuntimeError('Inactive actor moved')
            # The entire issued action must be legal, even if capture stops it.
            end = at[j]+delta[j]
            if (np.max(np.abs(delta[j])) > speed+2e-7
                    or not point_legal(at[j], boxes)
                    or not point_legal(end, boxes) or blocked(at[j], end, boxes)):
                raise RuntimeError('Independent illegal '+team+' motion')
    pp, ee, nap, nae, times, captors = replay_motion(
        state[4:8].reshape(2, 2), state[:4].reshape(2, 2),
        action, enemy, ap, ae, boxes, capture_dt)
    np.testing.assert_allclose(nxt[:4], ee.ravel(), atol=2e-8, rtol=0)
    np.testing.assert_allclose(nxt[4:8], pp.ravel(), atol=2e-8, rtol=0)
    np.testing.assert_array_equal(nxt[9:11] > .5, nae)
    np.testing.assert_array_equal(nxt[11:13] > .5, nap)
    if abs(nxt[8]-state[8]-1.) > 1e-8:
        raise RuntimeError('Physical clock mismatch')
    return [dict(time=float(state[8]+times[j]), target=j, pursuer=int(captors[j]))
            for j in range(2) if times[j] >= 0]


def audit_planner(evidence, boxes, capture_dt=.01):
    """Planner supplies the original action, including captured motion tails."""
    for item in evidence:
        pp,ee,ap,ae,_,_=replay_motion(
            np.asarray(item['positions']),np.asarray(item['targets']),
            np.asarray(item['pursuer_actions']),np.asarray(item['evader_actions']),
            np.asarray(item['active_p']),np.asarray(item['active_e']),boxes,capture_dt)
        for actual,key in ((pp,'next_positions'),(ee,'next_targets')):
            np.testing.assert_allclose(actual,item[key],atol=2e-8,rtol=0)
        np.testing.assert_array_equal(ap,item['next_active_p'])
        np.testing.assert_array_equal(ae,item['next_active_e'])
    return len(evidence)


def diagnostics(trace, horizon):
    if not trace:
        raise RuntimeError('Empty E49 trajectory')
    events = sorted((event for f in trace for event in f['capture_events']),
                    key=lambda e: (e['time'], e['pursuer'], e['target']))
    first = events[0]['time'] if events else None
    final = events[-1]['time'] if len(events) == 2 else None
    remaining = np.asarray(trace[-1]['next_state'])[9:11] > .5
    runs = [0, 0]
    hidden = visible = post_visible = post_active = 0
    recovered, censored = [], []
    last_seen = [False, False]
    for f in trace:
        obs = f['observation']
        for j in range(2):
            if not obs['active_e'][j]:
                if runs[j]:
                    censored.append(dict(target=j, steps=runs[j], reason='captured'))
                    runs[j] = 0
                continue
            seen = any(q is not None for q in obs['reports'][j])
            visible += int(seen)
            hidden += int(not seen)
            if first is not None and obs['step'] >= first:
                post_active += 1
                post_visible += int(seen)
            if seen and runs[j]:
                recovered.append(dict(target=j, steps=runs[j]))
                runs[j] = 0
            elif not seen:
                runs[j] += 1
            last_seen[j] = seen
    for j in range(2):
        if runs[j]:
            censored.append(dict(target=j, steps=runs[j], reason='end_of_episode'))
    complete = not remaining.any()
    category = ('double_capture' if complete else 'no_first_capture' if first is None
                else 'post_capture_seen_not_caught' if post_visible else 'post_capture_no_positive')
    return dict(double_capture=bool(complete), captures=int(2-remaining.sum()),
                at_least_one=bool(events), captured_fraction=float((2-remaining.sum())/2),
                first_capture_time=first, final_capture_time=final,
                first_capture_remaining=None if first is None else horizon-first,
                post_capture_seconds=None if final is None else final-first,
                failure_category=category, hidden_target_decisions=hidden,
                visible_target_decisions=visible, post_capture_visible_decisions=post_visible,
                post_capture_active_decisions=post_active, rediscoveries=recovered,
                censored_occlusions=censored, capture_events=events,
                progress={str(fraction): dict(
                    first=bool(first is not None and first <= fraction*horizon),
                    double=bool(final is not None and final <= fraction*horizon))
                    for fraction in (.25, .5, .75, 1.)})
