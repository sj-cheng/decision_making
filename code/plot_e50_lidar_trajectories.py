"""Plot one executed E50 LiDAR/MINCO episode for every method."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np


def _draw_scene(ax, boxes):
    for box in boxes:
        box = np.asarray(box, dtype=float)
        ax.add_patch(Rectangle((box[0, 0], box[1, 0]),
                               box[0, 1] - box[0, 0], box[1, 1] - box[1, 0],
                               facecolor='#d9d9d9', edgecolor='#555555',
                               linewidth=.7, alpha=.75, zorder=0))
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlim(-10, 10); ax.set_ylim(-10, 10)
    ax.grid(alpha=.15)


def _episode_paths(row):
    pursuer = [[], []]
    evader = [[], []]
    hits = []
    for cycle in row.get('cycles', []):
        traj = np.asarray(cycle.get('executed_trajectory_positions', []), dtype=float)
        times = np.asarray(cycle.get('executed_trajectory_times', []), dtype=float)
        # Older audit files stored the full two-second plan in this field.
        # Normalize them here so the figure always shows the actually executed
        # one-second rolling prefix.
        if len(times) == len(traj) and len(times) and times[-1] > 1.0 + 1e-9:
            keep = times <= 1.0 + 1e-9
            traj = traj[keep]
        if traj.ndim == 3 and traj.shape[1:] == (2, 2):
            for robot in range(2):
                pursuer[robot].append(traj[:, robot])
                if cycle.get('lidar_scans'):
                    scan = cycle['lidar_scans'][robot]
                    start = traj[0, robot]
                    angles = np.asarray(scan['angles'], dtype=float)
                    ranges = np.asarray(scan['ranges'], dtype=float)
                    mask = ranges < float(scan['max_range']) - 1e-9
                    if np.any(mask):
                        points = start[None, :] + np.column_stack((np.cos(angles[mask]),
                                                                    np.sin(angles[mask]))) * ranges[mask, None]
                        hits.append(points)
        target = np.asarray(cycle.get('evader_position', []), dtype=float)
        if target.shape == (2, 2):
            for robot in range(2):
                evader[robot].append(target[robot])
    pursuer = [np.concatenate(x) if x else np.zeros((0, 2)) for x in pursuer]
    evader = [np.asarray(x) if x else np.zeros((0, 2)) for x in evader]
    hits = np.concatenate(hits) if hits else np.zeros((0, 2))
    return pursuer, evader, hits


def plot(case_path, result_path, output_dir):
    case = json.loads(Path(case_path).read_text())
    result = json.loads(Path(result_path).read_text())
    boxes = np.asarray(case['map'], dtype=float).reshape(-1, 2, 2)
    output_dir = Path(output_dir); output_dir.mkdir(parents=True, exist_ok=True)
    colors = ('#1565c0', '#ef6c00')
    target_colors = ('#2e7d32', '#8e24aa')

    rows = [row for row in result.get('rows', []) if row.get('valid')]
    if not rows:
        raise RuntimeError('result contains no valid method rows')
    ncols, nrows = 3, int(np.ceil(len(rows) / 3))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.0*ncols, 5.0*nrows), squeeze=False)
    for index, row in enumerate(rows):
        ax = axes[index // ncols][index % ncols]
        _draw_scene(ax, boxes)
        pursuer, evader, hits = _episode_paths(row)
        if len(hits):
            ax.scatter(hits[:, 0], hits[:, 1], s=2, color='#90a4ae', alpha=.18, label='LiDAR hits')
        for robot in range(2):
            if len(pursuer[robot]):
                ax.plot(pursuer[robot][:, 0], pursuer[robot][:, 1], color=colors[robot], lw=1.8,
                        label=f'P{robot}')
                ax.scatter(*pursuer[robot][0], color=colors[robot], marker='o', s=24, zorder=4)
            if len(evader[robot]):
                ax.plot(evader[robot][:, 0], evader[robot][:, 1], '--', color=target_colors[robot], lw=1.2,
                        label=f'E{robot}')
        cycles = row.get('cycles', [])
        overrides = sum(c.get('safety_override') is not None for c in cycles)
        captures = sum(len(c.get('capture_events', [])) for c in cycles)
        ax.set_title(f"{row['method']}  safety={overrides}, captures={captures}")
        ax.set_xlabel('x [m]'); ax.set_ylabel('y [m]')
        ax.legend(fontsize=7, loc='upper right')
    for index in range(len(rows), nrows*ncols):
        axes[index // ncols][index % ncols].axis('off')
    fig.suptitle('E50 unknown-map LiDAR + MINCO executed trajectories', fontsize=15)
    fig.tight_layout()
    fig.savefig(output_dir / 'all_methods_overview.png', dpi=180, bbox_inches='tight')
    fig.savefig(output_dir / 'all_methods_overview.pdf', bbox_inches='tight')
    plt.close(fig)

    for row in rows:
        fig, ax = plt.subplots(figsize=(6.5, 6.5))
        _draw_scene(ax, boxes)
        pursuer, evader, hits = _episode_paths(row)
        if len(hits):
            ax.scatter(hits[:, 0], hits[:, 1], s=3, color='#90a4ae', alpha=.2, label='LiDAR hits')
        for robot in range(2):
            if len(pursuer[robot]):
                ax.plot(pursuer[robot][:, 0], pursuer[robot][:, 1], color=colors[robot], lw=2,
                        label=f'P{robot}')
                ax.scatter(*pursuer[robot][0], color=colors[robot], marker='o', s=35, zorder=4)
            if len(evader[robot]):
                ax.plot(evader[robot][:, 0], evader[robot][:, 1], '--', color=target_colors[robot], lw=1.4,
                        label=f'E{robot}')
        ax.set_title(f"{row['method']} — LiDAR/MINCO executed episode")
        ax.set_xlabel('x [m]'); ax.set_ylabel('y [m]'); ax.legend()
        fig.tight_layout()
        safe_name = str(row['method']).replace('/', '_')
        fig.savefig(output_dir / f'{safe_name}.png', dpi=200, bbox_inches='tight')
        plt.close(fig)
    return output_dir


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--case', required=True)
    parser.add_argument('--result', required=True)
    parser.add_argument('--output-dir', required=True)
    args = parser.parse_args()
    print(plot(args.case, args.result, args.output_dir))
