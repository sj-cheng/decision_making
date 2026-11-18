"""Make a reusable old-style 11-method GIF from a full E50 rolling run."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.patches import Rectangle, Wedge
import numpy as np


METHODS = ('A2', 'A3', 'A4', 'AG', 'GEOM_SEARCH', 'RouteAPF',
           'IG', 'SINGLE_PRIOR', 'A5', 'MAPPO_RouteAPF', 'PPO_C')
COLORS = ('#2166ac', '#1b9e77')
EVADER_COLORS = ('#e66101', '#b2182b')


def _draw_map(ax, boxes):
    for b in boxes:
        b = np.asarray(b, dtype=float)
        ax.add_patch(Rectangle((b[0, 0], b[1, 0]), b[0, 1] - b[0, 0],
                               b[1, 1] - b[1, 0], facecolor='#d1d5db',
                               edgecolor='#6b7280', linewidth=.55, zorder=0))
    ax.set(xlim=(-10, 10), ylim=(-10, 10), aspect='equal')
    ax.grid(alpha=.15, linewidth=.4)


def _path_at(row, initial_targets, cycle, fraction):
    pursuer = [[], []]
    evader = [initial_targets[i].copy() for i in range(2)]
    evader_history = [[initial_targets[i].copy()] for i in range(2)]
    cycles = row.get('cycles', [])
    for c in cycles[:cycle]:
        tr = np.asarray(c.get('executed_trajectory_positions', []), dtype=float)
        if tr.ndim == 3 and tr.shape[1:] == (2, 2):
            for i in range(2): pursuer[i].append(tr[:, i])
        target = np.asarray(c.get('evader_position', []), dtype=float)
        if target.shape == (2, 2):
            evader = [target[i].copy() for i in range(2)]
            for i in range(2): evader_history[i].append(target[i].copy())
    if cycle < len(cycles):
        c = cycles[cycle]
        tr = np.asarray(c.get('executed_trajectory_positions', []), dtype=float)
        if tr.ndim == 3 and tr.shape[1:] == (2, 2):
            n = max(1, int(round((len(tr) - 1) * float(fraction))) + 1)
            for i in range(2): pursuer[i].append(tr[:n, i])
        if fraction >= .999:
            target = np.asarray(c.get('evader_position', []), dtype=float)
            if target.shape == (2, 2):
                evader = [target[i].copy() for i in range(2)]
                for i in range(2): evader_history[i].append(target[i].copy())
    paths = [np.concatenate(v) if v else np.zeros((0, 2)) for v in pursuer]
    return paths, np.asarray(evader), [np.asarray(v) for v in evader_history]


def make_gif(case_path, result_path, output, fps=10):
    case = json.loads(Path(case_path).read_text())
    result = json.loads(Path(result_path).read_text())
    boxes = np.asarray(case['map'], dtype=float).reshape(-1, 2, 2)
    initial = np.asarray(case['worlds'][0]['initial'], dtype=float)
    targets = initial[:4].reshape(2, 2)
    rows = {r['method']: r for r in result['rows'] if r.get('valid')}
    methods = [m for m in METHODS if m in rows]
    cycles = max((len(rows[m].get('cycles', [])) for m in methods), default=0)
    fig, axes = plt.subplots(4, 3, figsize=(12, 15), squeeze=False)
    axes = list(axes.flat)
    for ax in axes[len(methods):]: ax.axis('off')

    def update(frame):
        cycle = min(cycles - 1, frame // 2) if cycles else 0
        fraction = 1.0 if frame % 2 else .45
        for ax, method in zip(axes, methods):
            ax.clear(); _draw_map(ax, boxes)
            paths, evaders, evader_paths = _path_at(rows[method], targets, cycle, fraction)
            current_cycle = rows[method].get('cycles', [])[cycle] if cycle < len(rows[method].get('cycles', [])) else {}
            camera = current_cycle.get('camera_state') or {}
            states = camera.get('state', ['SEARCH', 'SEARCH'])
            gimbal = camera.get('gimbal_yaw', [0., 0.])
            lost_reasons = camera.get('lost_reason', [None, None])
            midpoint_velocity = np.asarray(current_cycle.get('midpoint_velocity', []), dtype=float)
            for i in range(2):
                if len(paths[i]):
                    ax.plot(paths[i][:, 0], paths[i][:, 1], color=COLORS[i], lw=1.6, label=f'P{i}')
                    ax.scatter(*paths[i][-1], color=COLORS[i], s=22, zorder=4)
                    if midpoint_velocity.shape == (2, 2) and np.linalg.norm(midpoint_velocity[i]) > 1e-8:
                        heading = float(np.arctan2(midpoint_velocity[i, 1], midpoint_velocity[i, 0]))
                    elif len(paths[i]) > 1 and np.linalg.norm(paths[i][-1] - paths[i][-2]) > 1e-8:
                        delta = paths[i][-1] - paths[i][-2]
                        heading = float(np.arctan2(delta[1], delta[0]))
                    else:
                        heading = 0.
                    center = heading + float(gimbal[i] if i < len(gimbal) else 0.)
                    status = states[i] if i < len(states) else 'SEARCH'
                    if i < len(lost_reasons) and lost_reasons[i]:
                        status = f'{status}/{str(lost_reasons[i])[:4]}'
                    color = '#2ca25f' if status == 'TRACK' else '#9ecae1'
                    ax.add_patch(Wedge(tuple(paths[i][-1]), 3.0,
                                       np.degrees(center - np.pi / 6.),
                                       np.degrees(center + np.pi / 6.),
                                       facecolor=color, edgecolor=color,
                                       alpha=.16, linewidth=.7, zorder=1))
                    ax.text(paths[i][-1, 0] + .12, paths[i][-1, 1] + .12,
                            f'C{i}:{status}', fontsize=6,
                            color='#14532d' if status == 'TRACK' else '#1f4e79')
                if len(evader_paths[i]) > 1:
                    ax.plot(evader_paths[i][:, 0], evader_paths[i][:, 1], '--',
                            color=EVADER_COLORS[i], lw=1.35, label=f'E{i}')
                ax.scatter(*evaders[i], color=EVADER_COLORS[i], s=30, marker='x', zorder=4)
            infeasible = not bool(current_cycle.get('feasible', True))
            collision = bool(current_cycle.get('collision', False))
            flags = (' | MINCO infeasible' if infeasible else '') + (' | COLLISION' if collision else '')
            ax.set_title(f'{method} | step {cycle + 12 + fraction:.1f}{flags}', fontsize=9)
            ax.tick_params(labelsize=7)
        fig.suptitle('E50 camera FOV + 256-ray LiDAR map + MINCO executed-prefix comparison', fontsize=15)
        return axes

    frames = max(1, cycles * 2)
    output = Path(output); output.parent.mkdir(parents=True, exist_ok=True)
    anim = FuncAnimation(fig, update, frames=frames, interval=1000 / float(fps), blit=False)
    anim.save(output, writer=PillowWriter(fps=fps), dpi=100)
    plt.close(fig)
    return output


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--case', required=True)
    parser.add_argument('--result', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--fps', type=int, default=10)
    args = parser.parse_args()
    print(make_gif(args.case, args.result, args.output, args.fps))
