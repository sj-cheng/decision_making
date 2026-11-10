"""Eleven-method comparison on two scattered maps, without PPO.

Obstacles are axis-aligned. The reference pictures use rotated rectangles,
which this environment cannot represent, so the two maps match those pictures
in count, size and spread only. Every vehicle starts in the left half.
Each evader stays at least twice the capture radius from every pursuer,
and is seen by at least one pursuer at the first step. There is no 5.2 m cap.
Search decisions are not cut off at 200 ms. The original loop would otherwise
run 100000 simulations, so each search decision runs a fixed 64.
"""
from __future__ import annotations

import os
for _key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMBA_NUM_THREADS'):
    os.environ[_key] = '1'

import json
from pathlib import Path
import numpy as np

from e49_bank import geometry_audit, make_observation
from e49_rules import create_environment
from solvers.frozen_d_search_interface import sensor_visible
from run_e49_benchmark import batch, read, revision, sha, stable_seed, write
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'current/e49_scatter_left_nobudget_20261109'
PREVIOUS = ROOT/'current/e49_scatter_left_20261109'
FROZEN = ROOT/'current/e49_persistent_pursuit_benchmark_20261108/frozen_methods.json'
SPAWNS = 8
CAPTURE = 1.
MIN_SEPARATION = 2. * CAPTURE + 0.05
FIXED_SIMULATIONS = 64


def place_boxes(rng, count, gap, half_range):
    boxes = []
    for _ in range(50000):
        if len(boxes) == count:
            return np.asarray(boxes, float)
        center = rng.uniform(-8.2, 8.2, 2)
        half = rng.uniform(half_range[0], half_range[1], 2)
        box = np.stack((center-half, center+half), axis=1)
        if np.max(np.abs(box)) >= 9.55:
            continue
        if boxes:
            other = np.asarray(boxes)
            sep = np.maximum(box[:, 0], other[:, :, 0]) - np.minimum(box[:, 1], other[:, :, 1])
            if np.any(np.all(sep < gap, axis=1)):
                continue
        boxes.append(box.tolist())
    raise RuntimeError('Could not place obstacles')


def legal_point(point, boxes, margin=0.2):
    if np.any(np.abs(point) > 9.15):
        return False
    for box in boxes:
        if np.all(point >= box[:, 0]-margin) and np.all(point <= box[:, 1]+margin):
            return False
    return True


def sample_spawn(rng, boxes):
    for _ in range(20000):
        points = []
        ok = True
        for _agent in range(4):
            placed = None
            for _try in range(80):
                point = np.array([rng.uniform(-9.0, -0.4), rng.uniform(-8.8, 8.8)])
                if not legal_point(point, boxes):
                    continue
                if any(np.linalg.norm(point-other) < 1.3 for other in points):
                    continue
                placed = point
                break
            if placed is None:
                ok = False
                break
            points.append(placed)
        if not ok:
            continue
        pursuers, evaders = points[:2], points[2:]
        if any(np.linalg.norm(e-p) < MIN_SEPARATION for e in evaders for p in pursuers):
            continue
        # The belief refuses a target whose first report is empty. There is no
        # 5.2 m cap: a sighting anywhere inside the 6 m sensor is enough.
        if not all(any(sensor_visible(p, e, boxes) for p in pursuers) for e in evaders):
            continue
        return np.asarray(pursuers), np.asarray(evaders)
    raise RuntimeError('Could not place a legal left-half spawn')


def build_case(name, boxes, pursuers, evaders):
    env = create_environment(boxes, horizon=80)
    state = np.zeros((env.state_dim, 1))
    state[0:2, 0] = evaders[0]
    state[2:4, 0] = evaders[1]
    state[4:6, 0] = pursuers[0]
    state[6:8, 0] = pursuers[1]
    state[env.active_idxs, 0] = 1.
    if not env.is_valid(state) or env.get_capture_pairs(state):
        raise RuntimeError('Spawn is invalid or already captured')
    observation = make_observation(state[:, 0], boxes, 80)
    if observation['step'] != 0:
        raise RuntimeError('Handoff is not at t=0')
    return dict(id=name, family='scatter', pattern='left_half_random', suite='custom', split='custom',
                obstacle_tier=str(len(boxes)), map=np.asarray(boxes, float).tolist(), speed=1.,
                evader_speed=1.5, horizon=80, history=[observation],
                worlds=[dict(id='AB', initial=state[:, 0].tolist())],
                spawn=dict(pursuers=np.asarray(pursuers).tolist(), evaders=np.asarray(evaders).tolist()),
                           min_pursuer_evader=float(min(np.linalg.norm(e-p) for e in evaders for p in pursuers)),
                           max_nearest=float(max(min(np.linalg.norm(e-p) for p in pursuers) for e in evaders)))


def prepare():
    labels = ('scatter_open', 'scatter_dense')
    OUT.mkdir(parents=True, exist_ok=True)
    cases = []
    for label in labels:
        previous = json.loads((PREVIOUS/'maps'/f'{label}_00.json').read_text())
        boxes = np.asarray(previous['map'], float)
        geometry_audit(boxes)
        rng = np.random.default_rng(stable_seed('nobudget-spawns', label))
        for index in range(SPAWNS):
            pursuers, evaders = sample_spawn(rng, boxes)
            case = build_case(f'{label}_{index:02d}', boxes, pursuers, evaders)
            path = OUT/'maps'/f'{case["id"]}.json'
            write(path, case)
            cases.append(dict(id=case['id'], path=str(path), sha256=sha(path), label=label,
                              spawn=case['spawn'], obstacles=len(boxes)))
    write(OUT/'map_index.json', dict(spawns_per_map=SPAWNS, capture_radius=CAPTURE,
                                     min_separation=MIN_SEPARATION, max_nearest=None,
                                     fixed_simulations=FIXED_SIMULATIONS, time_budget_ms=None,
                                     left_half='x in [-9.0, -0.4]', cases=cases))
    return cases


def plot_maps():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    from matplotlib.patches import Rectangle
    font_manager.fontManager.addfont('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')
    plt.rcParams['font.family'] = 'Noto Sans CJK JP'
    index = read(OUT/'map_index.json')
    groups = {}
    for case in index['cases']:
        groups.setdefault(case['label'], []).append(case)
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 6.4))
    for ax, (label, cases) in zip(axes, groups.items()):
        data = read(cases[0]['path'])
        for b in data['map']:
            ax.add_patch(Rectangle((b[0][0], b[1][0]), b[0][1]-b[0][0], b[1][1]-b[1][0],
                                   facecolor='#bdbdbd', edgecolor='#3a3a3a', linewidth=0.5))
        for case in cases:
            spawn = case['spawn']
            p = np.asarray(spawn['pursuers']); e = np.asarray(spawn['evaders'])
            ax.scatter(p[:, 0], p[:, 1], s=18, c='#1f6feb', zorder=3)
            ax.scatter(e[:, 0], e[:, 1], s=18, c='#e07a2f', zorder=3)
        ax.axvline(0, color='#888888', linewidth=0.6, linestyle='--')
        ax.set_xlim(-10.5, 10.5); ax.set_ylim(-10.5, 10.5); ax.set_aspect('equal')
        ax.set_title(f'{label}  {len(data["map"])} 块  {len(cases)} 组出生点')
    fig.suptitle('蓝=追捕者  橙=逃跑者  全部在左半边')
    fig.tight_layout()
    fig.savefig(OUT/'maps_and_spawns.png', dpi=140)
    plt.close(fig)


def free_cpus(limit):
    import subprocess
    cores = {}
    for cpu in range(os.cpu_count() or 1):
        topo = Path(f'/sys/devices/system/cpu/cpu{cpu}/topology')
        if not topo.exists():
            continue
        key = ((topo/'physical_package_id').read_text().strip(), (topo/'core_id').read_text().strip())
        cores.setdefault(key, []).append(cpu)
    pinned = set()
    listing = subprocess.check_output(['ps', '-u', str(os.getuid()), '-o', 'pid,pcpu,comm', '--no-headers'], text=True)
    for line in listing.splitlines():
        pid, used, command = line.split(None, 2)
        if 'python' not in command or float(used) < 40:
            continue
        cpus = os.sched_getaffinity(int(pid))
        if len(cpus) <= 4:
            pinned |= cpus
    free = [cpu for group in cores.values() for cpu in group if cpu not in pinned]
    if len(free) < 4:
        raise RuntimeError('Fewer than 4 idle logical CPUs')
    return sorted(free)[:limit]


def specs():
    chosen = []
    for spec in read(FROZEN)['methods']:
        if spec['arm'] == 'PPO_C':
            continue
        item = dict(spec)
        if item.get('checkpoint'):
            item['checkpoint_sha256'] = sha(item['checkpoint'])
        chosen.append(item)
    return chosen


def tasks(cases):
    code = revision()
    for case in cases:
        seed = stable_seed('scatter-left-20261109', case['sha256'])
        for spec in specs():
            task = dict(spec, stage='scatter', case=case['path'], group=case['id'], case_sha256=case['sha256'],
                        family='scatter', suite='custom', split='custom', obstacle_tier=case['label'],
                        world='AB', stream=0, seed=seed, horizon=80, budget_ms=1e12,
                        fixed=FIXED_SIMULATIONS, source_revision=code)
            yield task


def summarize():
    rows = [read(path) for path in (OUT/'results/scatter').glob('*.json')]
    ok = [row for row in rows if row.get('valid')]
    arms = []
    for row in ok:
        if row['task']['arm'] not in arms:
            arms.append(row['task']['arm'])
    lines = [f'valid {len(ok)} / {len(rows)}']
    table = []
    for arm in arms:
        own = [row for row in ok if row['task']['arm'] == arm]
        seeds = sorted({row['task']['train_seed'] for row in own})
        per_seed = []
        for seed in seeds:
            part = [row for row in own if row['task']['train_seed'] == seed]
            per_seed.append(float(np.mean([row['summary']['double_capture'] for row in part])))
        table.append(dict(arm=arm, episodes=len(own), seeds=len(seeds),
                          double_capture=float(np.mean(per_seed)),
                          at_least_one=float(np.mean([row['summary']['at_least_one'] for row in own]))))
        lines.append(f"{arm:16s} {table[-1]['double_capture']:.1%}  episodes {len(own)}")
    write(OUT/'summary.json', dict(methods=table))
    (OUT/'summary.txt').write_text('\n'.join(lines)+'\n')
    return lines


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=('prepare', 'run', 'summarize'))
    parser.add_argument('--workers', type=int, default=32)
    args = parser.parse_args()
    if args.command == 'prepare':
        prepare(); plot_maps(); print('prepared', OUT)
    elif args.command == 'run':
        cases = read(OUT/'map_index.json')['cases']
        cpus = free_cpus(args.workers)
        write(OUT/'workers.json', dict(requested=args.workers, used=cpus))
        batch(OUT, list(tasks(cases)), cpus)
        print('\n'.join(summarize()))
    else:
        print('\n'.join(summarize()))


if __name__ == '__main__':
    main()
