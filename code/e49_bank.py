"""Outcome-blind E49 bank: public positive anchors and paired hidden histories.

Existing E46 geometry constructors supply layout families and scripted positions;
every transition is reconstructed with E49 dynamics before acceptance. No method,
controller, learned model, reward or closed-loop outcome is queried here.
"""
from __future__ import annotations

import argparse
from collections import Counter, deque
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from e45f_scenarios import ID_FAMILIES, OOD_FAMILIES, new_id_case, shifted_case
from solvers.e45_team_adapter import public_observation
from estimator_audit_geometry import blocked

ROOT = Path(__file__).resolve().parents[1]
BASE = 14920261108
TIERS = ('sparse', '16', '24')
COUNTS = dict(train=80, validation=8, development=8, test=28)


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False))
    temporary.replace(path)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def geometry_key(boxes):
    """Order-independent translation/quarter-turn key at 1e-4 precision."""
    boxes = np.asarray(boxes, float)
    center = (boxes[:, :, 0].min(axis=0)+boxes[:, :, 1].max(axis=0))/2
    # Explicit corner arrays avoid relying on obstacle order for the origin.
    points = np.array([[[x, y] for x in b[0] for y in b[1]] for b in boxes])-center
    signatures = []
    for _ in range(4):
        converted = np.stack((points.min(axis=1), points.max(axis=1)), axis=2)
        converted = np.round(converted, 4)
        converted[converted == 0.] = 0.  # Canonicalize IEEE negative zero.
        signatures.append(json.dumps(sorted(converted.tolist()), separators=(',', ':')))
        points = points[:, :, [1, 0]]*np.array([-1., 1.])
    return hashlib.sha256(min(signatures).encode()).hexdigest()


def make_observation(state, boxes, horizon=80):
    state = np.asarray(state).reshape(-1)
    result = public_observation(int(state[8]), state[4:8].reshape(2, 2),
        state[:4].reshape(2, 2), state[11:13] > .5, state[9:11] > .5,
        np.asarray(boxes, float))
    result['remaining'] = int(horizon) - result['step']
    return result


def geometry_audit(boxes):
    """Validate complete rectangles and exact positive-area free-space connectivity.

    The cell decomposition includes all rectangle edges. Four-neighbor open
    cells give an exact connectivity certificate for axis-aligned closed boxes;
    zero-width passages are excluded. Joined edges of a layout are allowed.
    """
    boxes = np.asarray(boxes, float)
    if boxes.ndim != 3 or boxes.shape[1:] != (2, 2) or not 1 <= len(boxes) <= 24:
        raise ValueError('Invalid obstacle capacity/shape')
    if not np.isfinite(boxes).all() or np.any(boxes[:, :, 0] >= boxes[:, :, 1]):
        raise ValueError('Invalid rectangle bounds')
    if np.any(np.abs(boxes) >= 10):
        raise ValueError('Obstacle reaches arena boundary')
    for i, box in enumerate(boxes):
        overlaps = np.minimum(box[:, 1], boxes[i+1:, :, 1]) - np.maximum(box[:, 0], boxes[i+1:, :, 0])
        if np.any(np.all(overlaps > 1e-10, axis=1)):
            raise ValueError('Obstacle interiors overlap')
    xs = np.unique(np.r_[-10., boxes[:, 0].ravel(), 10.])
    ys = np.unique(np.r_[-10., boxes[:, 1].ravel(), 10.])
    xx, yy = np.meshgrid((xs[1:]+xs[:-1])/2, (ys[1:]+ys[:-1])/2, indexing='ij')
    occupied = np.zeros(xx.shape, bool)
    for box in boxes:
        occupied |= (xx > box[0, 0]) & (xx < box[0, 1]) & (yy > box[1, 0]) & (yy < box[1, 1])
    free = ~occupied
    visited = np.zeros(free.shape, bool)
    start = tuple(np.argwhere(free)[0])
    queue = deque([start]); visited[start] = True
    while queue:
        x, y = queue.popleft()
        for a, b in ((x-1, y), (x+1, y), (x, y-1), (x, y+1)):
            if 0 <= a < free.shape[0] and 0 <= b < free.shape[1] and free[a, b] and not visited[a, b]:
                visited[a, b] = True; queue.append((a, b))
    if not np.array_equal(visited, free):
        raise ValueError('Disconnected positive-area free space')
    # Minimum separation between distinct disjoint rectangle components and
    # boundary; touching core wall pieces describe the same component.
    gaps = [float(np.min(10-np.abs(boxes)))]
    for i, a in enumerate(boxes):
        for b in boxes[i+1:]:
            gap = np.maximum(np.maximum(a[:, 0]-b[:, 1], b[:, 0]-a[:, 1]), 0.)
            distance = float(np.linalg.norm(gap))
            if distance > 1e-9: gaps.append(distance)
    return dict(obstacles=len(boxes), occupied_area=float(np.prod(boxes[:, :, 1]-boxes[:, :, 0], axis=1).sum()),
        arena_area=400., minimum_obstacle_or_boundary_clearance=min(gaps),
        clearance_definition='Euclidean separation of disjoint obstacle rectangles or arena boundary; joined core pieces excluded',
        connected=True, free_cell_count=int(free.sum()), connectivity='exact rectangle-edge cell decomposition')


def densify(case, tier, seed):
    boxes = np.asarray(case['map'], float).tolist()
    if tier == 'sparse': return np.asarray(boxes)
    desired = int(tier)
    rng = np.random.default_rng(seed)
    paths = []
    for world in case['worlds']:
        states = np.asarray(world['scripted_states'])[:, :8].reshape(-1, 4, 2)
        paths.extend((a, b) for before, after in zip(states[:-1], states[1:]) for a, b in zip(before, after))
        # Protect each initial target's actual positive sensor ray.
        for j, reports in enumerate(case['history'][0]['reports']):
            for i, report in enumerate(reports):
                if report is not None: paths.append((states[0, i+2], states[0, j]))
    for _ in range(20000):
        if len(boxes) == desired: return np.asarray(boxes)
        center = rng.uniform(-8.9, 8.9, 2)
        half = rng.uniform(.22, .65, 2)
        candidate = np.stack((center-half, center+half), axis=1)
        other = np.asarray(boxes)
        separation = np.maximum(candidate[:, 0], other[:, :, 0]) - np.minimum(candidate[:, 1], other[:, :, 1])
        if np.any(np.all(separation < .15, axis=1)): continue
        expanded = candidate.copy(); expanded[:, 0] -= .12; expanded[:, 1] += .12
        if any(blocked(np.asarray(a), np.asarray(b), expanded[None]) for a, b in paths): continue
        boxes.append(candidate.tolist())
    raise ValueError('Could not place requested obstacle count')


def replay(case, horizon=80):
    from e49_rules import create_environment
    boxes = np.asarray(case['map'], float)
    summary = geometry_audit(boxes)
    shared = None
    for world in case['worlds']:
        model = create_environment(boxes, horizon=horizon)
        source = np.asarray(world['scripted_states'], float)
        states = []
        for index, item in enumerate(source):
            if index == 0:
                state = np.zeros((model.state_dim, 1))
                state[:8, 0] = item[:8]
                state[8, 0] = -(len(source)-1)
                state[model.active_idxs, 0] = 1
                model._update_observation_memory(state, state, 0.)
            else:
                action = (item[:8]-source[index-1, :8]).reshape(8, 1)
                if np.max(np.abs(action[4:])) > 1.+1e-9 or np.max(np.abs(action[:4])) > 1.5+1e-9:
                    raise ValueError('Script violates E49 role speed')
                state = model.step(state, action, 1.)
                if not np.allclose(state[:8, 0], item[:8], rtol=0, atol=1e-9):
                    raise ValueError('Script blocked by full obstacle geometry')
                if model.last_capture_events: raise ValueError('Capture during E49 scripted history')
            if not model.is_valid(state) or model.get_capture_pairs(state) or not np.all(state[model.active_idxs] > .5):
                raise ValueError('Illegal E49 scripted state or capture')
            states.append(state[:, 0].tolist())
        history = [make_observation(s, boxes, horizon) for s in states]
        if shared is not None and history != shared: raise ValueError('Distinguishable AB/BA public histories')
        shared = history
        world.update(initial=states[-1], scripted_states=states, history=history)
    if not all(any(q is not None for q in reports) for reports in shared[0]['reports']):
        raise ValueError('Missing shared public positive anchor')
    if any(q is not None for reports in shared[-1]['reports'] for q in reports):
        raise ValueError('Handoff exposes hidden targets')
    if case['worlds'][0]['initial'][:4] == case['worlds'][1]['initial'][:4]:
        raise ValueError('AB/BA hidden worlds collapse')
    summary['handoff_step'] = 0
    summary['history_steps'] = len(shared)
    summary['last_seen_steps'] = [max(r['step'] for r in shared if any(q is not None for q in r['reports'][j])) for j in range(2)]
    case.update(history=shared, speed=1., evader_speed=1.5, horizon=horizon, geometry_summary=summary,
        acceptance='geometry, exact free-space connectivity, E49 replay, shared positive anchor and hidden AB/BA public history only; no policy outcome',
        history_clock='negative scripted steps end at handoff0; H pursuit steps begin at handoff',
        rules=dict(pursuer_axis_limit=1., evader_axis_limit=1.5, persistent_pursuers=True,
            capture_radius=1., capture_strict=True, capture_los=True, capture_sample_dt=.01))
    return case


def construct(family, index, split, tier, horizon=80):
    generated_split = split if split in ('train', 'validation') else 'test'
    namespace = BASE + 1000*tuple(COUNTS).index(split) + 100*TIERS.index(tier)
    generator = new_id_case if family in ID_FAMILIES else shifted_case
    case = generator(family, index, generated_split, namespace)
    case['map'] = densify(case, tier, np.random.SeedSequence([namespace, index, 490])).tolist()
    case.update(id=f'e49_{split}_{tier}_{family}_{index:04d}', split=split,
        suite='ID' if family in ID_FAMILIES else 'OOD', obstacle_tier=tier,
        construction=dict(seed_base=namespace, source_index=index, source='E46 geometry/history constructors', outcome_filter=False))
    return replay(case, horizon)


def allocation(split, tier):
    if split == 'test': return {family: 4 for family in (*ID_FAMILIES, *OOD_FAMILIES)}
    quantity = COUNTS[split]
    offset = (TIERS.index(tier) + tuple(COUNTS).index(split)) % len(ID_FAMILIES)
    return {family: quantity//3 + int((i-offset)%3 < quantity%3) for i, family in enumerate(ID_FAMILIES)}


def prior_geometry_keys(out):
    keys = set(); files = 0
    for path in sorted((ROOT/'current').glob('*/bank/**/*.json')):
        if out.resolve() in path.resolve().parents: continue
        try:
            case = json.loads(path.read_text())
            if isinstance(case, dict) and 'map' in case:
                keys.add(geometry_key(case['map'])); files += 1
        except (ValueError, TypeError, KeyError): continue
    return keys, files


def prepare(out, horizon=80):
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    manifest = out/'bank_manifest.json'
    if manifest.exists():
        config = json.loads(manifest.read_text())
        if config['horizon'] != horizon: raise ValueError('Bank H differs; explicitly replay/refreeze bank for changed H')
        for entry in config['entries']:
            if sha(entry['path']) != entry['sha256']: raise ValueError('Bank hash mismatch: '+entry['id'])
        return config
    started = time.monotonic()
    allocation_plan = {f'{split}/{tier}': allocation(split, tier) for split in COUNTS for tier in TIERS}
    write(out/'bank_allocation.json', dict(base_seed=BASE, horizon=horizon, allocation_plan=allocation_plan,
        selection='Geometry, connectivity and E49 history legality only; no policy outcomes'))
    known, old_files = prior_geometry_keys(out)
    prior_count = len(known)
    entries = []; attempts = []
    for split in COUNTS:
        for tier in TIERS:
            distribution = allocation(split, tier)
            allocation_plan[f'{split}/{tier}'] = distribution
            for family, count in distribution.items():
                accepted = 0
                for index in range(20000):
                    if accepted == count: break
                    identifier = f'e49_{split}_{tier}_{family}_{index:04d}'
                    path = out/'bank'/split/(identifier+'.json')
                    try:
                        if path.exists():
                            case = json.loads(path.read_text())
                            if case.get('horizon') != horizon: raise ValueError('Partial case has different H')
                            replay(case, horizon)
                        else: case = construct(family, index, split, tier, horizon)
                        key = geometry_key(case['map'])
                        if key in known: raise ValueError('Duplicate translated/quarter-rotated geometry')
                    except (ValueError, AssertionError) as error:
                        attempts.append(dict(split=split, obstacle_tier=tier, family=family, index=index, accepted=False, reason=str(error)))
                        continue
                    known.add(key)
                    write(path, case)
                    entries.append(dict(id=identifier, path=str(path.resolve()), sha256=sha(path), family=family,
                        suite=case['suite'], split=split, obstacle_tier=tier, geometry_sha256=key,
                        obstacle_count=len(case['map'])))
                    attempts.append(dict(id=identifier, accepted=True, index=index))
                    accepted += 1
                    write(out/'bank_progress.json', dict(entries=len(entries), split=split, tier=tier, family=family,
                        accepted=accepted, required=count, elapsed_seconds=time.monotonic()-started))
                if accepted != count: raise RuntimeError(f'Cannot complete {split}/{tier}/{family}: {accepted}/{count}')
    if len(entries) != 372: raise AssertionError('Exact registered bank count not met')
    config = dict(experiment='E49', horizon=horizon, base_seed=BASE, entries=entries,
        allocation_plan=allocation_plan, historical_map_files=old_files, historical_unique_geometries=prior_count,
        outcome_filter=False, deduplication='translation/quarter-turn canonical geometry at1e-4 precision; historical banks and all E49 splits',
        historical_sources='read only', generation_seconds=time.monotonic()-started,
        splits=dict(Counter(entry['split'] for entry in entries)), attempts=attempts)
    write(manifest, config)
    return config


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, default=ROOT/'current/e49_persistent_pursuit_benchmark_20261108')
    parser.add_argument('--horizon', type=int, default=80)
    args = parser.parse_args()
    result = prepare(args.out, args.horizon)
    print(json.dumps({key: result[key] for key in ('splits', 'generation_seconds', 'historical_map_files')}))
