"""Eleven methods, geometric escape, both teams limited to 1.0 per axis.

Same two scattered maps. New left-half spawns: every pair of vehicles is at
least 4 m apart, and each evader is still seen once so the pursuer belief can
start. The frozen transition kernel still clips evaders at 1.5; this opponent
never uses a step outside [-1, 1], so the executed speed is 1.0. Pursuer
beliefs keep the frozen 1.5 motion model.
"""
from __future__ import annotations

import os
for _key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMBA_NUM_THREADS'):
    os.environ[_key] = '1'

import argparse
from pathlib import Path
import numpy as np

import run_e49_benchmark as bench
import run_e49_scatter_left as scatter
from e49_bank import geometry_audit, make_observation
from e49_rules import create_environment
from solvers.frozen_d_search_interface import sensor_visible

ROOT = scatter.ROOT
OUT = ROOT/'current/e49_scatter_equal_20261110'
PREVIOUS = ROOT/'current/e49_scatter_left_20261109'
SPAWNS = 8
MIN_PAIR = 4.
CAPTURE_GAP = 2.05
ESCAPE_SPEED = 1.


def sample_spawn(rng, boxes):
    for _ in range(4000):
        points = []
        placed = True
        for _agent in range(4):
            point = None
            for _try in range(250):
                candidate = np.array([rng.uniform(-9.0, -0.4), rng.uniform(-8.8, 8.8)])
                if not scatter.legal_point(candidate, boxes):
                    continue
                if any(np.linalg.norm(candidate-other) < MIN_PAIR for other in points):
                    continue
                point = candidate
                break
            if point is None:
                placed = False
                break
            points.append(point)
        if not placed:
            continue
        pursuers, evaders = np.asarray(points[:2]), np.asarray(points[2:])
        gap = min(np.linalg.norm(e-p) for e in evaders for p in pursuers)
        if gap < CAPTURE_GAP:
            continue
        if not all(any(sensor_visible(p, e, boxes) for p in pursuers) for e in evaders):
            continue
        pair = min(np.linalg.norm(a-b) for i, a in enumerate(points) for b in points[i+1:])
        return pursuers, evaders, float(pair), float(gap)
    raise RuntimeError('Could not place a spread left-half spawn')


def build_case(name, boxes, pursuers, evaders, pair, gap):
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
    nearest = float(max(min(np.linalg.norm(e-p) for p in pursuers) for e in evaders))
    return dict(id=name, family='scatter', pattern='left_half_spread', suite='custom', split='custom',
                obstacle_tier=str(len(boxes)), map=np.asarray(boxes, float).tolist(), speed=1.,
                evader_speed=ESCAPE_SPEED, horizon=80, history=[observation],
                worlds=[dict(id='AB', initial=state[:, 0].tolist())],
                spawn=dict(pursuers=np.asarray(pursuers).tolist(), evaders=np.asarray(evaders).tolist(),
                           min_pair=pair, min_pursuer_evader=gap, max_nearest=nearest))


def prepare():
    OUT.mkdir(parents=True, exist_ok=True)
    cases = []
    for label in ('scatter_open', 'scatter_dense'):
        boxes = np.asarray(bench.read(PREVIOUS/'maps'/f'{label}_00.json')['map'], float)
        geometry_audit(boxes)
        rng = np.random.default_rng(bench.stable_seed('equal-speed-spread', label))
        for index in range(SPAWNS):
            pursuers, evaders, pair, gap = sample_spawn(rng, boxes)
            case = build_case(f'{label}_{index:02d}', boxes, pursuers, evaders, pair, gap)
            path = OUT/'maps'/f'{case["id"]}.json'
            bench.write(path, case)
            cases.append(dict(id=case['id'], path=str(path), sha256=bench.sha(path), label=label,
                              spawn=case['spawn'], obstacles=len(boxes)))
    bench.write(OUT/'map_index.json', dict(spawns_per_map=SPAWNS, min_pair=MIN_PAIR,
                                           capture_gap=CAPTURE_GAP, escape_speed=ESCAPE_SPEED,
                                           pursuer_speed=1., fixed_simulations=scatter.FIXED_SIMULATIONS,
                                           left_half='x in [-9.0, -0.4]', cases=cases))
    plot_maps()
    return cases


def plot_maps():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    from matplotlib.patches import Rectangle
    font_manager.fontManager.addfont('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')
    plt.rcParams['font.family'] = 'Noto Sans CJK JP'
    index = bench.read(OUT/'map_index.json')
    groups = {}
    for case in index['cases']:
        groups.setdefault(case['label'], []).append(case)
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 6.4))
    for ax, (label, cases) in zip(axes, groups.items()):
        data = bench.read(cases[0]['path'])
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
    fig.suptitle('同速 1.0  蓝=追捕者  橙=逃跑者  全部在左半边，两两至少 4 米')
    fig.tight_layout()
    fig.savefig(OUT/'maps_and_spawns.png', dpi=140)
    plt.close(fig)


def make_escape(env, seed, simulations=3000):
    from solvers.scatter_escape_evader import ScatterEscapeEvader
    return ScatterEscapeEvader(ESCAPE_SPEED)


def make_controller(case, task):
    import e49_controllers
    if task['arm'] != 'VORONOI':
        return _original_controller(case, task)
    from solvers.voronoi_pursuit import VoronoiController
    boxes = np.asarray(case['map'], float).reshape(-1, 2, 2)
    return VoronoiController(boxes, 1., int(task.get('seed', 17)))


def worker_init(queue):
    global _original_controller
    import e49_rules
    import e49_controllers
    e49_rules.make_opponent = make_escape
    _original_controller = e49_controllers.make_controller
    e49_controllers.make_controller = make_controller
    _original_init(queue)


_original_controller = None


_original_init = bench.worker_init


def tasks(cases):
    code = bench.revision()
    for case in cases:
        seed = bench.stable_seed('equal-speed-20261110', case['sha256'])
        for spec in scatter.specs()+[dict(name='VORONOI', arm='VORONOI', train_seed=17)]:
            yield dict(spec, stage='equal', case=case['path'], group=case['id'], case_sha256=case['sha256'],
                       family='scatter', suite='custom', split='custom', obstacle_tier=case['label'],
                       world='AB', stream=0, seed=seed, horizon=80, budget_ms=1e12,
                       fixed=scatter.FIXED_SIMULATIONS, source_revision=code,
                       opponent='geometry_escape_1.0', pursuer_speed=1., evader_speed=ESCAPE_SPEED)


def run(workers):
    cases = bench.read(OUT/'map_index.json')['cases']
    cpus = scatter.free_cpus(workers)
    bench.write(OUT/'workers.json', dict(requested=workers, used=cpus))
    pending = [task for task in tasks(cases)
               if not (OUT/'results'/task['stage']/(bench.identity(task)+'.json')).exists()]
    if not pending:
        return
    import concurrent.futures as cf
    import multiprocessing as mp
    context = mp.get_context('spawn')
    queue = context.Queue()
    for cpu in cpus:
        queue.put(cpu)
    done = 0
    with cf.ProcessPoolExecutor(len(cpus), mp_context=context, initializer=worker_init, initargs=(queue,)) as pool:
        futures = {pool.submit(bench.execute, task, str(OUT)): task for task in pending}
        for future in cf.as_completed(futures):
            row = future.result(); done += 1
            task = futures[future]
            bench.event(OUT, 'equal', scheduled=len(pending), completed=done, last_method=task['arm'],
                        last_valid=row['valid'], last_seconds=row['seconds'])


def summarize():
    rows = [bench.read(path) for path in (OUT/'results/equal').glob('*.json')]
    ok = [row for row in rows if row.get('valid')]
    table = []
    for arm in sorted({row['task']['arm'] for row in ok}):
        own = [row for row in ok if row['task']['arm'] == arm]

        def seeded(key, keep=lambda row: True):
            means = []
            for seed in sorted({row['task']['train_seed'] for row in own}):
                part = [row for row in own if row['task']['train_seed'] == seed and keep(row)]
                means.append(float(np.mean([key(row) for row in part])))
            return float(np.mean(means))
        table.append(dict(arm=arm, episodes=len(own),
                          double_capture=seeded(lambda r: float(r['summary']['double_capture'])),
                          at_least_one=seeded(lambda r: float(r['summary']['at_least_one'])),
                          decisions=seeded(lambda r: r['summary']['decisions']),
                          open=seeded(lambda r: float(r['summary']['double_capture']), lambda r: 'open' in r['task']['group']),
                          dense=seeded(lambda r: float(r['summary']['double_capture']), lambda r: 'dense' in r['task']['group'])))
    failures = [dict(arm=row['task']['arm'], group=row['task']['group'], train_seed=row['task']['train_seed'],
                     error=row.get('error')) for row in rows if not row.get('valid')]
    bench.write(OUT/'summary.json', dict(valid=len(ok), total=len(rows), engineering_failures=failures, methods=table))
    lines = [f'valid {len(ok)} / {len(rows)}  failures {len(failures)}']
    for row in sorted(table, key=lambda item: -item['double_capture']):
        lines.append(f"{row['arm']:16s} double {row['double_capture']:6.1%}  one {row['at_least_one']:6.1%}  "
                     f"steps {row['decisions']:5.1f}  open {row['open']:6.1%}  dense {row['dense']:6.1%}")
    (OUT/'summary.txt').write_text('\n'.join(lines)+'\n')
    return lines


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=('prepare', 'run', 'summarize'))
    parser.add_argument('--workers', type=int, default=32)
    args = parser.parse_args()
    if args.command == 'prepare':
        prepare(); print('prepared', OUT)
    elif args.command == 'run':
        run(args.workers)
        print('\n'.join(summarize()))
    else:
        print('\n'.join(summarize()))


if __name__ == '__main__':
    main()
