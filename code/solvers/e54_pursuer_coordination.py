"""Small, auditable coordination layers for the E54 pursuit comparison.

The local H-MCTS trees remain unchanged.  The coordinator only chooses among
their already evaluated root candidates, so its information boundary is clear.
"""
from dataclasses import replace
import numpy as np


def _choices(plan, search):
    result = [(plan.endpoint.copy(), float(plan.value), int(plan.candidate_id),
               float(plan.duration), bool(plan.pass_through), int(plan.visits))]
    for i, row in enumerate(search.get("root", [])):
        if i == plan.candidate_id or not row["visits"]:
            continue
        result.append((np.asarray(row["endpoint"], float), float(row["value"]), i,
                       float(row["duration"]), bool(row["pass_through"]), int(row["visits"])))
    return result


def _shared_positions(beliefs):
    tracks = {}
    now = max(b.now for b in beliefs)
    for belief in beliefs:
        for tid, track in belief.tracks.items():
            if tid not in tracks or track.timestamp > tracks[tid].timestamp:
                tracks[tid] = track
    return np.asarray([t.position + (now-t.timestamp)*t.velocity
                       for t in tracks.values()], float).reshape(-1, 2)


def coordinate(plans, searches, beliefs, *, oracle_targets=None, mode="belief"):
    """Select a pair of local root candidates using a shared objective.

    ``belief`` uses only the union of exchanged particles. ``oracle`` is a
    simulation upper bound and may use true target positions. The function
    returns replacement plans and a compact audit record.
    """
    if mode not in ("belief", "oracle"):
        raise ValueError(mode)
    candidates = [_choices(plans[i], searches[i]) for i in (0, 1)]
    if mode == "oracle":
        targets = np.asarray(oracle_targets, float).reshape(-1, 2)
    else:
        targets = _shared_positions(beliefs)
    best = None
    for a in candidates[0]:
        for b in candidates[1]:
            endpoints = np.stack((a[0], b[0]))
            score = a[1] + b[1]
            if len(targets):
                nearest = np.linalg.norm(endpoints[:, None] - targets[None, :, :], axis=2).min(axis=1)
                score += float(np.sum(1.0 / (1.0 + nearest)))
                if len(targets) >= 2:
                    ids = np.argmin(np.linalg.norm(endpoints[:, None] - targets[None, :, :], axis=2), axis=1)
                    if ids[0] == ids[1]:
                        score -= 1.5
            separation = float(np.linalg.norm(endpoints[0] - endpoints[1]))
            score += min(separation, 4.) * .08
            if best is None or score > best[0]:
                best = (score, a, b)
    if best is None:
        return plans, dict(mode=mode, changed=False, candidates=[1, 1])
    _, a, b = best
    chosen = []
    for i, item in enumerate((a, b)):
        chosen.append(replace(plans[i], endpoint=item[0], action=plans[i].action + (item[0] - plans[i].endpoint),
                               value=item[1], visits=item[5],
                               candidate_id=item[2], duration=item[3], pass_through=item[4]))
    changed = any(np.linalg.norm(chosen[i].endpoint - plans[i].endpoint) > 1e-6 for i in (0, 1))
    return chosen, dict(mode=mode, changed=changed,
                        candidates=[len(x) for x in candidates],
                        endpoints=[x.endpoint.tolist() for x in chosen])
