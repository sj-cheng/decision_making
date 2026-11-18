"""Compare raw and cross-cycle-smoothed executed E50 trajectories."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from plot_e50_lidar_trajectories import _draw_scene, _episode_paths


def plot(case_path, raw_path, smooth_path, output):
    case = json.loads(Path(case_path).read_text())
    raw = json.loads(Path(raw_path).read_text())
    smooth = json.loads(Path(smooth_path).read_text())
    boxes = np.asarray(case['map'], dtype=float).reshape(-1, 2, 2)
    raw_rows = {r['method']: r for r in raw['rows'] if r.get('valid')}
    smooth_rows = {r['method']: r for r in smooth['rows'] if r.get('valid')}
    methods = [m for m in raw_rows if m in smooth_rows]
    fig, axes = plt.subplots(len(methods), 2, figsize=(10, 3.4 * len(methods)), squeeze=False)
    colors = ('#1565c0', '#ef6c00')
    for i, method in enumerate(methods):
        for col, rows, label in ((0, raw_rows, 'raw'), (1, smooth_rows, 'cross-cycle smoothed')):
            ax = axes[i, col]; _draw_scene(ax, boxes)
            pursuer, _, _ = _episode_paths(rows[method])
            for robot in range(2):
                if len(pursuer[robot]):
                    ax.plot(pursuer[robot][:, 0], pursuer[robot][:, 1], color=colors[robot], lw=1.8,
                            label=f'P{robot}')
                    ax.scatter(*pursuer[robot][0], color=colors[robot], s=20, zorder=4)
            ax.set_title(f'{method} — {label}')
            ax.set_xlabel('x [m]'); ax.set_ylabel('y [m]')
            ax.legend(fontsize=7)
    fig.suptitle('Raw vs cross-cycle-smoothed executed pursuer trajectories', fontsize=15)
    fig.tight_layout()
    output = Path(output); output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180, bbox_inches='tight')
    fig.savefig(output.with_suffix('.pdf'), bbox_inches='tight')
    plt.close(fig)
    return output


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--case', required=True)
    parser.add_argument('--raw', required=True)
    parser.add_argument('--smooth', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    print(plot(args.case, args.raw, args.smooth, args.output))
