"""Descriptive post-run plots and network-vs-search comparisons; no selection."""
import os
os.environ.setdefault('MPLCONFIGDIR', '/tmp/mpl-candidate-prior')
import hashlib
import json
from pathlib import Path
import re
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'current/l45_candidate_prior_train_20261026'


def main():
    rows = json.loads((OUT/'evaluation/results.json').read_text())
    training = json.loads((OUT/'training.json').read_text())['records']
    stats = json.loads((OUT/'evaluation/statistics.json').read_text())
    summary = stats['summary']
    groups = sorted({r['group'] for r in rows})
    seeds = [r['seed'] for r in training]
    index = {(r['group'], r['world'], r['method']): r for r in rows}
    rng = np.random.default_rng(91320261026)
    indices = rng.integers(0, len(groups), (10000, len(groups)))
    comparisons = {}
    for a, b in (('NET', 'G'), ('PUCT-L', 'NET')):
        values = {}
        for metric in ('cost', 'rediscovery_cost', 'capture'):
            x = np.array([np.mean([float(index[(g, w, f'{a}{s}')][metric])-
                float(index[(g, w, f'{b}{s}' if b == 'NET' else b)][metric])
                for w in ('A', 'B') for s in seeds]) for g in groups])
            values[metric] = dict(mean=float(x.mean()), ordinary_ci95=np.quantile(x[indices].mean(-1), [.025, .975]).tolist())
        comparisons[a+'__'+b] = values
    figure_dir = OUT/'figures'; figure_dir.mkdir(exist_ok=True)
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.2), constrained_layout=True)
    for ax, record in zip(axes, training):
        logs = record['logs']; epochs = [r['epoch'] for r in logs]
        ax.plot(epochs, [r['train_loss'] for r in logs], label='Train')
        ax.plot(epochs, [r['validation_loss'] for r in logs], label='Held-out validation')
        ax.axvline(record['selected_epoch'], color='.5', linestyle=':', label='Selected epoch')
        ax.set(title=f'Seed {record["seed"]}', xlabel='Epoch', ylabel='Imitation loss')
    axes[0].legend(fontsize=8)
    for suffix in ('png', 'pdf'):
        fig.savefig(figure_dir/f'training.{suffix}', dpi=180)
    plt.close(fig)
    labels = ['G', 'Direct NN', 'PUCT uniform', 'PUCT geometry', 'PUCT learned']
    methods = [['G'], [f'NET{s}' for s in seeds], ['PUCT-U'], ['PUCT-G'], [f'PUCT-L{s}' for s in seeds]]
    cost = [np.mean([summary[m]['cost'] for m in ms]) for ms in methods]
    captures = [np.mean([summary[m]['captures']/summary[m]['games'] for m in ms]) for ms in methods]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), constrained_layout=True)
    colors = ['#555555', '#6699aa', '#ccaa66', '#9977aa', '#448866']
    axes[0].barh(labels, cost, color=colors)
    axes[0].set(xlabel='Failure-penalized completion steps (lower is better)', title='Held-out maps: cost')
    axes[1].barh(labels, np.array(captures)*100, color=colors)
    axes[1].set(xlim=(0, 105), xlabel='Observed capture rate (%)', title='Six history groups, two worlds')
    for ax in axes:
        ax.invert_yaxis()
    fig.suptitle('Exploratory pilot; neural arms average three seeds, not extra independent maps', fontsize=10)
    for suffix in ('png', 'pdf'):
        fig.savefig(figure_dir/f'closed_loop.{suffix}', dpi=180)
    plt.close(fig)
    cases = []
    for g in groups:
        for world in ('A', 'B'):
            base = index[(g, world, 'G')]
            selected = [index[(g, world, f'PUCT-L{s}')] for s in seeds]
            cases.append(dict(group=g, world=world, G_cost=base['cost'], G_capture=base['capture'],
                learned_mean_cost=float(np.mean([r['cost'] for r in selected])),
                learned_captures=sum(r['capture'] for r in selected),
                difference=float(np.mean([r['cost'] for r in selected])-base['cost']),
                G_episode=base['episode'], learned_episodes=[r['episode'] for r in selected]))
    diagnostics = dict(descriptive_only=True, auxiliary_comparisons=comparisons,
        cases_sorted_by_cost_loss=sorted(cases, key=lambda r: (-r['difference'], r['group'], r['world'])))
    (OUT/'descriptive_diagnosis.json').write_text(json.dumps(diagnostics, ensure_ascii=False, indent=2)+'\n')
    report = OUT/'REPORT.md'; text = report.read_text()
    marker = '\n## 训练表现与辅助消融\n'
    text = text.split(marker)[0]
    primary = ['|学习先验搜索相对基线|平均完成成本差|95%区间|捕获率差|', '|---|---:|---|---:|']
    for baseline, values in stats['comparisons'].items():
        interval = values['cost']['simultaneous_ci95'] or values['cost']['ci95']
        kind = '同时' if values['cost']['simultaneous_ci95'] is not None else '普通辅助'
        primary.append(f'|{baseline}|{values["cost"]["difference"]:+.2f}|[{interval[0]:+.2f}, {interval[1]:+.2f}] {kind}|{100*values["capture"]["difference"]:+.2f}pp|')
    start = text.index('|学习先验搜索相对基线|')
    end = text.index('\n独立逐帧审计通过', start)
    text = text[:start]+'\n'.join(primary)+'\n'+text[end:]
    learned_cost = float(np.mean([summary[f'PUCT-L{s}']['cost'] for s in seeds]))
    learned_captures = sum(summary[f'PUCT-L{s}']['captures'] for s in seeds)
    concrete = (f'**当前结论：本配置未超过强G。学习先验搜索三种子平均完成成本{learned_cost:.2f}步，'
                f'强G为{summary["G"]["cost"]:.2f}步；捕获{learned_captures}/36，强G为{summary["G"]["captures"]}/12。**')
    text = re.sub(r'\*\*当前结论：[^\n]+', concrete, text)
    lines = [marker, '五类方法汇总（神经方法合计三个种子；平均成本按种子等权）：', '',
        '|方法|捕获|平均完成成本|', '|---|---:|---:|']
    for label, ms in zip(('强G', '网络直接控制', '均匀先验搜索', '人工几何先验搜索', '学习先验搜索'), methods):
        captures = sum(summary[m]['captures'] for m in ms)
        games = sum(summary[m]['games'] for m in ms)
        mean_cost = float(np.mean([summary[m]['cost'] for m in ms]))
        lines.append(f'|{label}|{captures}/{games}|{mean_cost:.2f}|')
    lines += ['', '|种子|选中轮次/80|验证G动作一致率|验证G评分平均损失|', '|---|---:|---:|---:|']
    for r in training:
        lines.append(f'|{r["seed"]}|{r["selected_epoch"]}|{100*r["validation_hard_accuracy"]:.1f}%|{r["validation_G_score_regret"]:.3f}|')
    lines += ['', '验证动作一致率只是模仿诊断；G评分同分附近可有不同动作，不代表真实任务收益。', '',
        '|辅助比较（三种子平均）|完成成本差|普通95%区间|捕获差|', '|---|---:|---|---:|']
    for name, v in comparisons.items():
        lines.append(f'|{name}|{v["cost"]["mean"]:+.2f}|{v["cost"]["ordinary_ci95"]}|{100*v["capture"]["mean"]:+.2f}pp|')
    lines += ['', '辅助比较不参与冻结的继续判定；只用于区分模仿能力与搜索增量。', '',
        '![训练与验证损失](figures/training.png)', '', '![闭环成本与捕获](figures/closed_loop.png)', '',
        '可导出的PDF：[训练曲线](figures/training.pdf)、[闭环结果](figures/closed_loop.pdf)。', '',
        '逐初态正反例完整保留于[描述性诊断](descriptive_diagnosis.json)，不按这些结果重新选择模型或调参。']
    overruns = [r for r in rows if r['initial_total_ms'] > 200.]
    if overruns:
        lines += ['', '首次完整决策超时逐局保留：']
        for r in overruns:
            lines.append(f'- {r["method"]} / {r["group"]} / {r["world"]}：{r["initial_total_ms"]:.2f}ms。')
        lines += ['', '即使排除时限门槛，学习先验的平均成本和观测捕获率仍未达到相对强G的继续要求。']
    report.write_text(text+'\n'.join(lines)+'\n')
    # Preserve the postprocessing implementation alongside all delivered files.
    (OUT/'source_snapshot/summarize_candidate_prior_pilot.py').write_bytes(Path(__file__).read_bytes())
    paths = [p for p in OUT.rglob('*') if p.is_file() and p.name not in ('DELIVERY_SHA256.json', 'status.json', 'pipeline.log')]
    hashes = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    (OUT/'DELIVERY_SHA256.json').write_text(json.dumps(hashes, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps(dict(training=training and len(training), games=len(rows), auxiliary=comparisons)))


if __name__ == '__main__':
    main()
