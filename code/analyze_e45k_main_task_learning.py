"""Read-only experiment analysis; writes separate development reports and plots."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import time

ROOT = Path(__file__).resolve().parents[1]


def read(path):
    return json.loads(path.read_text())


def analyze(root):
    cfg = read(root / 'experiment.json')
    state = read(root / 'status.json')
    runs = []
    lines = ['# E45-K1主任务学习能力：开发结果分析', '',
             f'更新时间：{time.strftime("%FT%T%z")}；协调器状态：{state["stage"]}。', '',
             '只使用训练/验证历史。选模沿用注册的验证平均原始效用，本文不改变训练或选模。', '',
             '|奖励|种子|新增步数|所选步|Root验证：初始→所选→最新|Root训练探针：初始→所选|固定D验证：初始→所选|',
             '|---|---:|---:|---:|---|---|---|']
    for variant in cfg['variants']:
        for seed in cfg['seeds']:
            folder = root / 'runs' / f'{variant}_{seed}'
            if not (folder / 'validation.json').exists():
                continue
            history = read(folder / 'validation.json')
            selected = read(folder / 'selected.json')
            status = read(folder / 'status.json')
            initial, latest = history[0], history[-1]
            def score(row, split='validation', opponent='rootpuct'):
                return row[split][opponent]['double_capture']
            updates = [read(p) for p in sorted((folder / 'updates').glob('*.json'))]
            record = dict(variant=variant, seed=seed, steps=status.get('steps', 0),
                          history=history, selected=selected,
                          initial_root=score(initial), selected_root=score(selected),
                          latest_root=score(latest), selected_train_root=score(selected, 'train_probe'))
            record['recent_training_root'] = None
            if updates:
                last = updates[-10:]
                rr = [u['by_opponent']['rootpuct'] for u in last]
                record['recent_training_root'] = dict(
                    captures=sum(r['captures'] for r in rr), episodes=sum(r['episodes'] for r in rr),
                    blocked_fraction_mean=sum(u['blocked_fraction'] for u in last)/len(last),
                    last_value_rmse=updates[-1]['value_rmse'])
            runs.append(record)
            lines.append(f'|{variant}|{seed}|{record["steps"]}|{selected["steps"]}|'
                         f'{score(initial):.1%}→{score(selected):.1%}→{score(latest):.1%}|'
                         f'{score(initial,"train_probe"):.1%}→{score(selected,"train_probe"):.1%}|'
                         f'{score(initial,opponent="fixed_D_kernel"):.1%}→{score(selected,opponent="fixed_D_kernel"):.1%}|')
    complete = state['stage'] == 'complete'
    lines += ['', '“所选”可能是早期检查点；最新检查点也单列，以显示后期退化。验证分数已用于选模，不能当独立测试或显著性证据。', '']
    if not complete:
        lines.append('**训练尚未完成，不判断本轮修复成功。** 各路可能处于不同步数，当前中间分数不能视为等预算最终比较。')
    else:
        lines.append('**本轮预算已完成；完成训练不等于已成为强MARL基线。**')
        for variant in cfg['variants']:
            rr = [r for r in runs if r['variant'] == variant]
            if len(rr) != len(cfg['seeds']):
                raise RuntimeError('Completed experiment has missing analysis records')
            train_gate = all(r['selected_train_root'] >= .5 for r in rr)
            val_mean = sum(r['selected_root'] for r in rr) / len(rr)
            initial_mean = sum(r['initial_root'] for r in rr) / len(rr)
            lines.append(f'- {variant}：Root验证平均 {initial_mean:.1%}→{val_mean:.1%}；'
                         f'训练探针门槛{"通过" if train_gate else "未通过"}，'
                         f'验证门槛{"通过" if val_mean >= .25 else "未通过"}。')
        lines += ['', '即使通过开发门槛，也需对照每个种子的改善、固定D退化及额外训练贡献；仅两个已有种子不能证明普遍增益。']
    if len(runs) == 4:
        lines += ['', '同一种子的捕获信用臂减去原奖励臂，在各自按同一规则所选的Root验证双捕率上的差：']
        for seed in cfg['seeds']:
            pair = {r['variant']:r for r in runs if r['seed'] == seed}
            delta = pair['capture_credit']['selected_root'] - pair['original_reward']['selected_root']
            lines.append(f'- 种子{seed}：{delta * 100:+.1f}个百分点。')
    lines += ['', '训练采样为随机策略，开发评估为确定性策略，两者成功率不能直接等同。', '',
              '|奖励/种子|最近最多10批随机训练Root双捕|批均安全缩幅|最近critic RMSE|',
              '|---|---|---:|---:|']
    for r in runs:
        d = r['recent_training_root']
        if d:
            lines.append(f'|{r["variant"]}/{r["seed"]}|{d["captures"]}/{d["episodes"]}|'
                         f'{d["blocked_fraction_mean"]:.2%}|{d["last_value_rmse"]:.3f}|')
    lines += ['', '![完整检查点学习曲线](learning_curves.png)', '',
              '这些曲线只表示开发表现，不含测试集，也不对重复使用的验证集给出确认性置信区间。', '',
              '判读：两臂都弱则继续固定主任务过拟合/表示/价值学习诊断；只有首捕提高则重点检查第二次捕获；'
              '训练强但验证弱则处理泛化。通过本轮后仍需更充分训练、更多种子、HASAC及独立冻结测试。', '',
              '[注册协议](PROTOCOL.md) · [进度](REPORT.md) · [逐批原始记录](runs/)']
    out = dict(updated_at=time.strftime('%FT%T%z'), pipeline_stage=state['stage'], complete=complete,
               runs=runs, held_out_test_used=False, strong_baseline_claim=False,
               analysis_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (root / 'learning_analysis.json').write_text(json.dumps(out, ensure_ascii=False, indent=2)+'\n')
    (root / '学习能力分析.md').write_text('\n'.join(lines)+'\n')
    os.environ.setdefault('MPLCONFIGDIR', '/tmp/e45k_analysis_matplotlib')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(14, 4), sharey=True)
    for r in runs:
        color = '#2563eb' if r['variant'] == 'original_reward' else '#d97706'
        style = '-' if r['seed'] == 17 else '--'
        for ax, split, opp, title in zip(axes, ['train_probe','validation','validation'],
                ['rootpuct','rootpuct','fixed_D_kernel'], ['Train probe: RootPUCT','Validation: RootPUCT','Validation: Fixed D']):
            ax.plot([h['steps']/1000 for h in r['history']],
                    [h[split][opp]['double_capture']*100 for h in r['history']],
                    color=color, linestyle=style, marker='o', label=f'{r["variant"]}, seed {r["seed"]}')
            ax.set_title(title); ax.set_xlabel('Additional transitions (thousands)'); ax.set_ylim(-3,103)
            ax.grid(alpha=.2)
    axes[0].set_ylabel('Original-task double capture (%)')
    axes[1].legend(loc='upper center', bbox_to_anchor=(.5,-.20), ncol=2, frameon=False, fontsize=8)
    fig.tight_layout(); fig.savefig(root/'learning_curves.png', dpi=180, bbox_inches='tight')
    fig.savefig(root/'learning_curves.pdf', bbox_inches='tight'); plt.close(fig)
    return state['stage']


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=ROOT/'current/e45k_main_task_learning_20261031')
    p.add_argument('--watch',action='store_true');args=p.parse_args();started=time.monotonic()
    while True:
        stage=analyze(args.output)
        print(time.strftime('%FT%T%z'),stage,flush=True)
        if not args.watch or stage in ('complete','failed') or time.monotonic()-started>9*3600:
            break
        time.sleep(60)


if __name__=='__main__':main()
