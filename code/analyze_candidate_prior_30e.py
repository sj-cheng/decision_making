"""Frozen grouped analysis and delivery checks for the isolated 30-epoch pilot."""
import os
os.environ.setdefault('MPLCONFIGDIR', '/tmp/mpl-candidate-prior-30e')
import hashlib
import json
from pathlib import Path
import re
import numpy as np
from run_frozen_d_closed_loop import save, sha

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'current/l45_candidate_prior_30e_20261026'
SEEDS = (17, 29, 43)


def read(name):
    return json.loads((OUT/name).read_text())


def quantiles(values):
    return dict(n=len(values), mean=float(np.mean(values)), p95=float(np.quantile(values, .95)),
        maximum=float(max(values))) if values else dict(n=0, mean=None, p95=None, maximum=None)


def analyze():
    rows = read('evaluation/results.json'); config = read('experiment.json')
    bank = read('bank/manifest.json')['entries']; training = read('training.json')['records']
    groups = sorted(e['id'] for e in bank if e['split'] == 'test')
    index = {(r['group'], r['world'], r['method']): r for r in rows}
    assert len(index) == len(rows) == 216
    summary, timelines = {}, {}
    for method in config['methods']:
        selected = [r for r in rows if r['method'] == method]
        lat = [v for r in selected for v in r['latencies_ms']]
        first = [r['initial_total_ms'] for r in selected]
        traces = [f for r in selected for f in json.loads(Path(r['episode']).read_text())['trace']]
        phases = {}
        for phase in ('update_ms', 'build_ms', 'candidate_ms', 'inference_ms', 'prior_ms',
                      'tree_prepare_ms', 'geometry_preparation_ms', 'search_including_prepare_ms', 'final_check_ms'):
            phases[phase] = quantiles([f['statistics'].get(phase, 0.) for f in traces])
        timelines[method] = dict(phases=phases, initialization_static=quantiles([r['initialization']['static_setup_ms'] for r in selected]),
            history_replay=quantiles([r['initialization']['history_initialization_ms'] for r in selected]),
            opponent=quantiles([f['opponent_ms'] for f in traces]),
            environment=quantiles([f['environment_ms'] for f in traces]),
            joint_cycle=quantiles([f['cycle_ms'] for f in traces]),
            phase_note='build is part of update; prior/geometry/tree preparation overlap search; do not add overlapping phases')
        hidden = [f for f in traces if f['observation']['target'] is None]
        coverage = [sum(v > 0 for v in f['statistics']['root_visits'])/len(f['statistics']['root_visits'])
                    for f in hidden if f['statistics'].get('root_visits')]
        summary[method] = dict(games=len(selected), captures=sum(r['capture'] for r in selected),
            rediscovered=sum(r['first_seen'] is not None for r in selected),
            cost=float(np.mean([r['cost'] for r in selected])),
            rediscovery_cost=float(np.mean([r['rediscovery_cost'] for r in selected])),
            successful_rediscovery_time=quantiles([r['first_seen'] for r in selected if r['first_seen'] is not None]),
            decision_ms=quantiles(lat), initial_ms=quantiles(first), joint_cycle_ms=timelines[method]['joint_cycle'],
            decision_overruns=sum(v > 200. for v in lat), initial_overruns=sum(v > 200. for v in first),
            hidden_simulations=quantiles([v for r in selected for v in r['hidden_simulations']]),
            hidden_root_fraction_visited=quantiles(coverage), opponent_shields=sum(r['shield_count'] for r in selected),
            simulations=sum(r['simulated_transitions'] for r in selected))
    def diff(group, baseline, metric, seed=None):
        seeds = SEEDS if seed is None else (seed,)
        return float(np.mean([float(index[(group, w, f'PUCT-L{s}')][metric])-
            float(index[(group, w, f'NET{s}' if baseline == 'NET' else baseline)][metric])
            for w in ('A', 'B') for s in seeds]))
    bases = ('PUCT-G', 'NET', 'G', 'PUCT-U')
    metrics = ('cost', 'rediscovery_cost', 'capture')
    values = {b: {m: np.array([diff(g, b, m) for g in groups]) for m in metrics} for b in bases}
    indices = np.random.default_rng(config['bootstrap_seed']).integers(0, len(groups), (10000, len(groups)))
    boots = {b: {m: x[indices].mean(-1) for m, x in d.items()} for b, d in values.items()}
    radius = float(np.quantile(np.max(np.stack([np.abs(boots[b]['cost']-values[b]['cost'].mean())
        for b in bases[:3]]), axis=0), .95))
    comparisons = {b: {m: dict(difference=float(x.mean()), ordinary_ci95=np.quantile(boots[b][m], [.025, .975]).tolist(),
        simultaneous_ci95=[float(x.mean()-radius), float(x.mean()+radius)] if b in bases[:3] and m == 'cost' else None,
        group_differences=x.tolist()) for m, x in d.items()} for b, d in values.items()}
    per_seed = {str(s): {b: {m: float(np.mean([diff(g, b, m, s) for g in groups])) for m in metrics}
        for b in bases} for s in SEEDS}
    families = {}
    for family in ('long_wall', 'wall_corner', 'multi_route'):
        selected = [r for r in rows if r['family'] == family]
        families[family] = {m: dict(games=len(rs), captures=sum(r['capture'] for r in rs),
            rediscovered=sum(r['first_seen'] is not None for r in rs), cost=float(np.mean([r['cost'] for r in rs])),
            rediscovery_cost=float(np.mean([r['rediscovery_cost'] for r in rs])))
            for m in config['methods'] if (rs := [r for r in selected if r['method'] == m])}
    checked = read('evaluation/postrun_audit_v2.json')
    timing = not any(v['decision_overruns'] or v['initial_overruns'] for v in summary.values())
    reliable = {b: comparisons[b]['cost']['simultaneous_ci95'][1] < 0 for b in bases[:3]}
    practical = all(comparisons[b]['cost']['difference'] <= -1. and comparisons[b]['capture']['difference'] >= 0
                    for b in bases[:3])
    consistent = all(per_seed[str(s)][b]['cost'] < 0 for s in SEEDS for b in bases[:3])
    followup = all(reliable.values()) and practical and consistent and checked['passed'] and timing
    stats = dict(summary=summary, comparisons=comparisons, per_seed=per_seed, families=families,
        bootstrap=dict(groups=groups, draws=10000, simultaneous_radius=radius, primary_family=list(bases[:3]),
            unit='public initial history; keep worlds and all model seeds together'),
        fresh_test_instances=True, model_seeds_not_extra_independent_samples=True)
    save(OUT/'evaluation/statistics.json', stats); save(OUT/'evaluation/timing_breakdown.json', timelines)
    decision = dict(stop=True, completed=True, suggested_followup=bool(followup), reliable_cost_benefit=reliable,
        practical_guard=bool(practical), per_seed_consistency=bool(consistent), audit_passed=checked['passed'],
        timing_passed=timing, deployment_changed=False, training_complete=True, evaluation_complete=True,
        reason='bounded_pilot_finished_further_confirmation_required' if followup else 'bounded_pilot_no_reliable_complete_increment')
    save(OUT/'decision.json', decision)
    # Preserve every negative and positive paired case, without changing policy.
    cases, overruns = [], []
    for g in groups:
        for w in ('A', 'B'):
            base = index[(g, w, 'G')]
            for s in SEEDS:
                direct, search = index[(g, w, f'NET{s}')], index[(g, w, f'PUCT-L{s}')]
                cases.append(dict(group=g, family=base['family'], world=w, seed=s,
                    G_capture=base['capture'], G_cost=base['cost'], NET_capture=direct['capture'], NET_cost=direct['cost'],
                    search_capture=search['capture'], search_cost=search['cost'], search_rediscovery=search['first_seen'],
                    search_minus_G=search['cost']-base['cost'], search_minus_NET=search['cost']-direct['cost'],
                    observed_failure='never_rediscovered' if search['first_seen'] is None else
                        'rediscovered_without_capture' if not search['capture'] else 'captured',
                    episode=search['episode']))
    for r in rows:
        payload = json.loads(Path(r['episode']).read_text())
        if r['initial_total_ms'] > 200.:
            overruns.append(dict(key=r['key'], kind='initialization_plus_first_decision', ms=r['initial_total_ms']))
        for f in payload['trace']:
            if f['own_ms'] > 200.:
                overruns.append(dict(key=r['key'], step=f['step'], kind='decision', ms=f['own_ms']))
    save(OUT/'failure_diagnosis.json', dict(descriptive_only=True,
        causal_limit='No causal isolation of short horizon, model mismatch or Monte Carlo noise in this pilot.',
        cases=sorted(cases, key=lambda r: -r['search_minus_G']), overruns=overruns))
    lines = ['# 30轮几何候选先验网络：训练与独立闭环测试', '',
        '本轮实际完成3个种子的训练和216局闭环测试。采用新训练数据与全新测试实例，保留既有80轮试验及无网络搜索失败记录。', '',
        '冻结D、强G、RootPUCT3000对手、目标模型、奖励与8步深度；只训练候选评分网络。三种搜索共享PUCT实现，唯一机制差异是根先验。', '',
        '24张训练地图、6张验证地图、12张测试地图；按完整地图及其历史隔离。测试是三种已知拓扑的新实例，非新拓扑。A/B世界及训练种子均不增加独立统计样本量。', '',
        '|执行臂|合法捕获|重发现|失败惩罚完成步数|重发现成本|决策P95/最大ms|首次完整最大ms|决策/首次超时|',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    for m, s in summary.items():
        lines.append(f'|{m}|{s["captures"]}/{s["games"]}|{s["rediscovered"]}/{s["games"]}|{s["cost"]:.2f}|{s["rediscovery_cost"]:.2f}|'
            f'{s["decision_ms"]["p95"]:.2f}/{s["decision_ms"]["maximum"]:.2f}|{s["initial_ms"]["maximum"]:.2f}|{s["decision_overruns"]}/{s["initial_overruns"]}|')
    lines += ['', 'NET与PUCT-L的对应种子共用同一验证集选中的检查点。G无需等待；200ms是共同上限，低耗时本身是收益。', '',
        '|主要比较：学习先验搜索减基线|完成成本差|三项同时95%区间|结论|', '|---|---:|---|---|']
    labels = {'PUCT-G': '手工几何先验搜索', 'NET': '同种子网络直接控制', 'G': '强G'}
    for b in bases[:3]:
        v = comparisons[b]['cost']; lo, hi = v['simultaneous_ci95']
        verdict = '成本降低' if hi < 0 else '成本增加' if lo > 0 else '额外收益证据不足'
        lines.append(f'|{labels[b]}|{v["difference"]:+.2f}|[{lo:+.2f}, {hi:+.2f}]|{verdict}|')
    lines += ['', '区间采用12组公共历史配对bootstrap10000次，三项主终点共同校正。捕获率观察护栏不能证明非劣。', '',
        '|种子|训练轮次|选中轮次|训练/验证状态数|验证G动作一致率|验证评分损失|', '|---|---:|---:|---:|---:|---:|']
    for t in training:
        lines.append(f'|{t["seed"]}|{t["epochs"]}|{t["selected_epoch"]}|{t["train_states"]}/{t["validation_states"]}|'
            f'{100*t["validation_hard_accuracy"]:.1f}%|{t["validation_G_score_regret"]:.4f}|')
    lines += ['', '模型只按验证损失选择。动作一致率及训练损失用于检查模仿，不能证明搜索价值。', '',
        '|拓扑|G捕获/成本|NET三种子捕获/平均成本|学习搜索三种子捕获/平均成本|手工先验捕获/成本|', '|---|---|---|---|---|']
    for f, d in families.items():
        def desc(ms):
            return f'{sum(d[m]["captures"] for m in ms)}/{sum(d[m]["games"] for m in ms)} / {np.mean([d[m]["cost"] for m in ms]):.2f}'
        lines.append(f'|{f}|{desc(["G"])}|{desc([f"NET{s}" for s in SEEDS])}|{desc([f"PUCT-L{s}" for s in SEEDS])}|{desc(["PUCT-G"])}|')
    lines += ['', f'前置接口与逐帧审计通过：{checked["passed"]}；事后重放{checked["frames"]}帧，在线独立几何检查{checked["simulated_transitions"]}个模拟转移。'
        '模拟完整轨迹仅做在线检查并存散列，未声称事后重放所有模拟转移。',
        f'完整200ms时限通过：{timing}。实际超时记录{len(overruns)}条，全部保留。推理、初始化及历史回放均计入；一次性编译和接口预热另存unit_tests/pretraining记录。',
        '逐项耗时存在包含关系：build属于update，prior/几何预计算属于search，不能直接相加。对手/环境和联合周期独立报告于timing_breakdown。', '',
        '**停止决定：** '+('满足本轮进一步确认的预定条件；仅建议另行确认，本轮按预算结束。' if followup else
        '未满足预定的可靠额外收益条件，结束本配置，不扩大训练、重调测试或叠加模块。'), '',
        '失败可分为从未重发现及重发现后未捕获，逐局保留。候选模仿误差、短视奖励、目标模型与真实对手差异、有限模拟噪声可能共同作用；本轮未隔离它们，不能将相关现象断言为唯一原因，也不能据此否定所有NTE/MCTS。', '',
        '- [冻结协议](PROTOCOL.md)、[配置与源散列](experiment.json)、[地图与历史清单](bank/manifest.json)。',
        '- [训练前接口](pretraining_interface.json)、[正式模型接口](preflight.json)、[训练记录](training.json)、[模型散列](models_frozen.json)。',
        '- [逐局结果](evaluation/results.json)、[成组统计与各种子](evaluation/statistics.json)、[完整耗时](evaluation/timing_breakdown.json)。',
        '- [独立审计](evaluation/postrun_audit_v2.json)、[逐局失败与超时](failure_diagnosis.json)、[停止决定](decision.json)、[交付检查](delivery_checks.json)。', '',
        '![训练曲线](figures/training.png)', '', '![闭环对照](figures/closed_loop.png)', '',
        '导出：[训练PDF](figures/training.pdf)、[闭环PDF](figures/closed_loop.pdf)。']
    (OUT/'REPORT.md').write_text('\n'.join(lines)+'\n')
    plots(training, summary)
    emit = dict(stage='analyzed', decision=decision)
    print(json.dumps(emit, ensure_ascii=False), flush=True)


def plots(training, summary):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    folder = OUT/'figures'; folder.mkdir(exist_ok=True)
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.3), constrained_layout=True)
    for ax, r in zip(axes, training):
        logs = r['logs']; x = [v['epoch'] for v in logs]
        ax.plot(x, [v['train_loss'] for v in logs], label='Train')
        ax.plot(x, [v['validation_loss'] for v in logs], label='Validation')
        ax.axvline(r['selected_epoch'], color='.5', ls=':')
        ax.set(title=f'Seed {r["seed"]}', xlabel='Epoch', ylabel='G imitation loss')
    axes[0].legend()
    for ext in ('png', 'pdf'): fig.savefig(folder/f'training.{ext}', dpi=180)
    plt.close(fig)
    methods = [['G'], [f'NET{s}' for s in SEEDS], ['PUCT-U'], ['PUCT-G'], [f'PUCT-L{s}' for s in SEEDS]]
    labels = ['G', 'Direct network', 'Uniform PUCT', 'Geometry PUCT', 'Learned PUCT']
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), constrained_layout=True)
    axes[0].barh(labels, [np.mean([summary[m]['cost'] for m in ms]) for ms in methods])
    axes[1].barh(labels, [100*np.mean([summary[m]['captures']/summary[m]['games'] for m in ms]) for ms in methods])
    axes[0].set(xlabel='Failure-penalized completion steps (lower is better)')
    axes[1].set(xlabel='Observed legal capture rate (%)', xlim=(0, 105))
    for ax in axes: ax.invert_yaxis()
    fig.suptitle('12 independent public histories; neural arms average all three seeds', fontsize=10)
    for ext in ('png', 'pdf'): fig.savefig(folder/f'closed_loop.{ext}', dpi=180)
    plt.close(fig)


def delivery():
    config = read('experiment.json'); bank = read('bank/manifest.json')['entries']
    rows = read('evaluation/results.json'); models = read('models_frozen.json'); training = read('training.json')['records']
    checked = read('evaluation/postrun_audit_v2.json')
    checks = {}
    for key in ('source_hashes', 'protected_hashes', 'bank_hashes'):
        checks[key+'_unchanged'] = all(sha(p) == h for p, h in config[key].items())
    checks['models_frozen'] = all(sha(p) == h for p, h in models['files'].items())
    checks['data_frozen'] = all(sha(p) == h for p, h in read('data/manifest.json')['files'].items())
    checks['42_disjoint_maps_and_histories'] = len({e['map_sha256'] for e in bank}) == len({e['geometry_history_sha256'] for e in bank}) == 42
    expected = {(e['id'], w, m) for e in bank if e['split'] == 'test' for w in ('A', 'B') for m in config['methods']}
    actual = [(r['group'], r['world'], r['method']) for r in rows]
    checks['all216_unique_paired_games'] = len(actual) == len(set(actual)) == 216 and set(actual) == expected
    checks['all_episodes_intact'] = all(sha(r['episode']) == r['episode_sha256'] for r in rows)
    checks['all_games_no_engineering_failure'] = all(r['failure'] is None for r in rows)
    checks['pretraining_interface'] = read('pretraining_interface.json')['passed']
    checks['posttraining_interface'] = read('preflight.json')['passed']
    checks['unit_tests'] = read('unit_tests.json')['passed']
    checks['3_seeds_30_epochs_validation_only'] = {t['seed'] for t in training} == set(SEEDS) and all(
        len(t['logs']) == t['epochs'] == 30 and t['selected_epoch'] <= 30 and not t['test_read'] and
        t['selected_epoch'] == min(t['logs'], key=lambda v: v['validation_loss'])['epoch'] and
        t['numpy_export_max_logit_difference'] < .001 for t in training)
    checks['independent_D_motion_sensor_audit'] = checked['passed'] and checked['episodes'] == 216 and checked['groups'] == 12
    checks['no_deployment_ppo_value_changes'] = not any(config[k] for k in ('deployment_changed', 'ppo', 'value_network', 'frozen_D_changed'))
    report = (OUT/'REPORT.md').read_text()
    checks['report_links_exist'] = all((OUT/p).exists() for p in re.findall(r'\]\(([^)]+)\)', report) if '://' not in p)
    result = dict(passed=all(checks.values()), checks=checks, games=len(rows), independent_groups=12,
        decision=read('decision.json'), timing_pass_is_separate_from_artifact_integrity=True)
    save(OUT/'delivery_checks.json', result)
    handoff = ('# 30轮试验交付\n\n已完成3种子各30轮及216局全新实例闭环测试。'
        '全部结果见REPORT.md、decision.json、delivery_checks.json。保留旧80轮记录，'
        '不得基于本次测试调参或挑种子。复现检查：\n\n'
        '`/home/sjc/miniconda3/envs/dm_env_py311/bin/python code/run_candidate_prior_30e.py verify`\n\n'
        '本轮结束；不自动追加训练、修改估计器或部署。\n')
    (OUT/'HANDOFF.md').write_text(handoff)
    paths = [p for p in OUT.rglob('*') if p.is_file() and p.name not in ('DELIVERY_SHA256.json', 'pipeline.log', 'status.json')]
    old_manifest = read('DELIVERY_SHA256.json') if (OUT/'DELIVERY_SHA256.json').exists() else None
    # A verification invocation must not conceal any post-delivery mutation.
    if old_manifest is not None:
        assert all(sha(p) == h for p, h in old_manifest.items()), 'Delivered artifact changed'
    save(OUT/'DELIVERY_SHA256.json', {str(p): sha(p) for p in paths})
    print(json.dumps(dict(stage='delivery', **result), ensure_ascii=False), flush=True)
    assert result['passed'], result


if __name__ == '__main__':
    analyze(); delivery()
