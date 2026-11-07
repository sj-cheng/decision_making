"""Descriptive post-run evidence from complete search-value stages only.

Standard library only: never import a controller, run a simulator, inspect
partial caches, change outcomes, or tune a method.  Confirmation is eligible
only when its complete 240-row results.json exists and passes the design check.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT/'current/l45_search_value_20261026'
METHODS = ('G', 'OL200', 'FB200', 'OLH', 'FBH')
OPPONENTS = ('matched', 'rootpuct')
STAGE_GROUPS = {'development': 6, 'confirmation': 12}


def _mean(values):
    return None if not values else sum(values)/len(values)


def _quantile(values, probability):
    if not values:
        return None
    ordered = sorted(values)
    location = (len(ordered)-1)*probability
    lower = math.floor(location)
    upper = math.ceil(location)
    return ordered[lower]+(ordered[upper]-ordered[lower])*(location-lower)


def _distribution(values):
    return dict(count=len(values), mean=_mean(values),
                median=None if not values else statistics.median(values),
                minimum=None if not values else min(values),
                p95=_quantile(values, .95),
                maximum=None if not values else max(values))


def _read_complete_results(output, stage):
    path = output/stage/'results.json'
    if not path.exists():
        return None
    raw = path.read_bytes()
    rows = json.loads(raw)
    expected = STAGE_GROUPS[stage]*2*len(OPPONENTS)*len(METHODS)
    if not isinstance(rows, list) or len(rows) != expected:
        raise ValueError(f'{stage}: expected complete {expected}-row results.json')
    index = {}
    for row in rows:
        identity = (row['group'], row['world'], row['opponent'], row['method'])
        if identity in index:
            raise ValueError(f'{stage}: duplicated paired identity {identity}')
        index[identity] = row
    groups = sorted({r['group'] for r in rows})
    design = set(itertools.product(groups, ('A', 'B'), OPPONENTS, METHODS))
    if len(groups) != STAGE_GROUPS[stage] or set(index) != design:
        raise ValueError(f'{stage}: incomplete or unexpected paired stage design')
    return rows, index, path, hashlib.sha256(raw).hexdigest()


def _episode_path(value, output, stage):
    path = Path(value)
    if path.is_absolute():
        return path
    for candidate in (output/stage/path, output/path, ROOT/path):
        if candidate.exists():
            return candidate
    raise FileNotFoundError(value)


def _episode_diagnostic(row, payload):
    trace = payload['trace']
    hidden = [f for f in trace if f['observation']['target'] is None]
    poses = collections.Counter(tuple(f['state'][4:6]) for f in trace)
    stopped = lambda f: max(abs(x) for x in f['own_action']) < 1e-10
    path_length = sum(math.dist(f['state'][4:6], f['next_state'][4:6]) for f in trace)
    first = row['first_seen']
    failure_type = (None if row['capture'] else 'engineering_failure'
                    if row.get('failure') is not None else 'not_rediscovered'
                    if first is None else 'late_rediscovery'
                    if first > row['original_remaining']-8
                    else 'rediscovered_not_captured')
    return dict(key=row['key'], group=row['group'], world=row['world'],
                family=row.get('family'), method=row['method'],
                capture=bool(row['capture']), first_seen=first,
                completion_cost=row['cost'], rediscovery_cost=row['rediscovery_cost'],
                remaining=row['original_remaining'], executed_steps=len(trace),
                hidden_frames=len(hidden), hidden_stops=sum(stopped(f) for f in hidden),
                maximum_repeated_pose=max(poses.values()) if poses else 0,
                pursuer_path_length=path_length, failure_type=failure_type,
                episode=row['episode'])


def _method_summary(rows, frames):
    hidden = [f for f in frames if f['observation']['target'] is None]
    searched = [f for f in hidden if f['statistics']['simulations'] > 0]
    visits = [v for f in searched for v in f['statistics']['root_visits']]
    zero_returns = lambda f: (bool(f['statistics']['root_totals'])
                              and max(f['statistics']['root_totals']) == 0.)
    stopped = lambda f: max(abs(x) for x in f['own_action']) < 1e-10
    budget = lambda f: f['statistics']['registered_budget_ms']
    return dict(
        games=len(rows), capture=sum(bool(r['capture']) for r in rows),
        rediscovered=sum(r['first_seen'] is not None for r in rows),
        completion_cost=_mean([r['cost'] for r in rows]),
        rediscovery_cost=_mean([r['rediscovery_cost'] for r in rows]),
        hidden_frames=len(hidden), searched_hidden_frames=len(searched),
        hidden_simulations=_distribution([f['statistics']['simulations'] for f in hidden]),
        hidden_zero_simulation_frames=sum(f['statistics']['simulations'] == 0 for f in hidden),
        incomplete_legal_root_pool_frames=None if not searched else sum(
            sum(v > 0 for v in f['statistics']['root_visits']) < f['statistics']['root_candidates']
            for f in searched),
        only_single_visit_frames=None if not searched else sum(
            max(f['statistics']['root_visits'], default=0) == 1 for f in searched),
        minimum_completed_root_visit=None if not visits else min(visits),
        all_zero_root_return_frames=sum(zero_returns(f) for f in hidden),
        zero_return_stop_frames=sum(zero_returns(f) and stopped(f) for f in hidden),
        hidden_stops=sum(stopped(f) for f in hidden),
        positive_observation_branch_frames=sum(
            f['statistics'].get('positive_observation_nodes', 0) > 0 for f in hidden),
        positive_observation_nodes=sum(
            f['statistics'].get('positive_observation_nodes', 0) for f in hidden),
        negative_observation_nodes=sum(
            f['statistics'].get('negative_observation_nodes', 0) for f in hidden),
        exact_conditionings=sum(f['statistics'].get('exact_conditionings', 0) for f in hidden),
        budget_aborted_simulations=sum(
            f['statistics'].get('budget_aborted_simulations', 0) for f in hidden),
        registered_budgets_ms=sorted({budget(f) for f in frames}),
        actual_decision_ms=_distribution([f['own_ms'] for f in frames]),
        hidden_decision_ms=_distribution([f['own_ms'] for f in hidden]),
        initial_total_ms=_distribution([r['initial_total_ms'] for r in rows]),
        decisions_over_registered_budget=sum(f['own_ms'] > budget(f) for f in frames),
        decisions_over_200ms=sum(f['own_ms'] > 200. for f in frames))


def _format(value, digits=2):
    return 'N/A' if value is None else f'{value:.{digits}f}'


def _link(key, value, destination):
    path = _episode_path(value, destination.parents[1], destination.parent.name)
    relative = os.path.relpath(path, destination.parent)
    return f'[{key}]({Path(relative).as_posix()})'


def diagnose_stage(output, stage):
    output = Path(output).resolve()
    loaded = _read_complete_results(output, stage)
    if loaded is None:
        return None
    rows, index, results_path, results_sha = loaded
    root_rows = [r for r in rows if r['opponent'] == 'rootpuct']
    details, frames = {}, collections.defaultdict(list)
    for row in root_rows:
        path = _episode_path(row['episode'], output, stage)
        raw = path.read_bytes()
        if 'episode_sha256' in row and hashlib.sha256(raw).hexdigest() != row['episode_sha256']:
            raise ValueError('Saved episode hash mismatch: '+row['key'])
        payload = json.loads(raw)
        if payload['summary']['key'] != row['key']:
            raise ValueError('Episode/summary identity mismatch: '+row['key'])
        frames[row['method']].extend(payload['trace'])
        if row['method'] in ('G', 'FBH'):
            details[(row['group'], row['world'], row['method'])] = _episode_diagnostic(row, payload)
    method_summary = {m: _method_summary([r for r in root_rows if r['method'] == m], frames[m])
                      for m in METHODS}
    paired = []
    for group in sorted({r['group'] for r in root_rows}):
        for world in ('A', 'B'):
            a, b = details[(group, world, 'FBH')], details[(group, world, 'G')]
            paired.append(dict(group=group, world=world, family=a['family'],
                completion_difference=a['completion_cost']-b['completion_cost'],
                rediscovery_difference=a['rediscovery_cost']-b['rediscovery_cost'],
                common_rediscovery_delay=None if a['first_seen'] is None or b['first_seen'] is None
                    else a['first_seen']-b['first_seen'], FBH=a, G=b))
    common_delays = [p['common_rediscovery_delay'] for p in paired
                     if p['common_rediscovery_delay'] is not None]
    failures = sorted([p for p in paired if not p['FBH']['capture']],
                      key=lambda p: (-p['completion_difference'], p['group'], p['world']))[:3]
    rescued = sorted([p for p in paired if p['FBH']['capture'] and not p['G']['capture']],
                     key=lambda p: (p['completion_difference'], p['group'], p['world']))[:1]
    comparison = dict(
        paired_games=len(paired), public_history_groups=len({p['group'] for p in paired}),
        completion_difference=_mean([p['completion_difference'] for p in paired]),
        rediscovery_difference=_mean([p['rediscovery_difference'] for p in paired]),
        completion_slower=sum(p['completion_difference'] > 0 for p in paired),
        completion_tied=sum(p['completion_difference'] == 0 for p in paired),
        completion_faster=sum(p['completion_difference'] < 0 for p in paired),
        common_rediscovered_games=len(common_delays),
        common_rediscovery_delay=_distribution(common_delays),
        common_rediscovery_later=sum(x > 0 for x in common_delays),
        common_rediscovery_tied=sum(x == 0 for x in common_delays),
        common_rediscovery_earlier=sum(x < 0 for x in common_delays),
        G_success_became_FBH_failure=sum(p['G']['capture'] and not p['FBH']['capture'] for p in paired),
        G_failure_rescued_by_FBH=sum(p['FBH']['capture'] and not p['G']['capture'] for p in paired))
    interpretation = dict(
        directly_observable=[
            'Capture, failure-penalized completion/rediscovery costs, executed trajectories and pairing are logged outcomes.',
            'Completed root visits distinguish unvisited candidates from a finite-statistics search.',
            'Positive/negative observation nodes and conditionings document feedback execution, not its causal effectiveness.',
            'Changing the registered 200ms to 2000ms budget changes this controller and its achieved sample count.'],
        mechanisms_not_isolated=[
            'Eight physical planning steps may be short-sighted, but this study did not vary horizon independently.',
            'The fixed-root motion model does not condition target transitions on pursuer position, unlike the actual full-state RootPUCT opponent; its contribution to outcome loss was not separately measured.',
            'Full root visitation is not convergence; Monte Carlo ranking errors and observation-branch sparsity remain unmeasured possibilities.',
            'Different controllers induce different actual opponent trajectories; paired episodes are not a fixed-evader-path counterfactual.',
            'Geometry and feedback costs change simulation counts under a shared deadline; this is not a fixed-simulation causal comparison.',
            'These diagnostics do not modify the preregistered development/confirmation decisions or authorize training.'])
    result = dict(stage=stage, descriptive_only=True,
        evidence_status='development_screen_only' if stage == 'development' else 'complete_independent_confirmation',
        results_path=str(results_path), results_sha256=results_sha,
        complete_stage_games=len(rows), opponent='rootpuct', method_summary=method_summary,
        FBH_minus_G=comparison, paired_episodes=paired,
        representative_selection_rule='At most three FBH capture failures ranked by descending cost difference from G, then group/world; at most one G-failure rescue ranked by ascending cost difference, then group/world.',
        worst_failures=failures, rescue=rescued, interpretation=interpretation,
        no_new_significance_tests=True, controllers_changed=False, outcomes_changed=False)
    destination = output/stage/'failure_diagnosis.md'
    lines = [f'# {stage}：搜索价值验证事后失败诊断', '',
        '仅使用完整阶段结果与对应轨迹；不读取中途缓存、不调整方法，不更改预登记判断。以下计数和差值为描述性证据。', '',
        f"RootPUCT条件含{comparison['paired_games']}个配对初态、{comparison['public_history_groups']}组公共历史。", '',
        '|方法|合法捕获|重发现|完成成本|重发现成本|决策P95 ms|',
        '|---|---:|---:|---:|---:|---:|']
    for method in METHODS:
        s = method_summary[method]
        lines.append(f"|{method}|{s['capture']}/{s['games']}|{s['rediscovered']}/{s['games']}|"
                     f"{_format(s['completion_cost'])}|{_format(s['rediscovery_cost'])}|{_format(s['actual_decision_ms']['p95'])}|")
    lines += ['',
        f"FBH−G完成成本差{comparison['completion_difference']:+.3f}步，重发现成本差{comparison['rediscovery_difference']:+.3f}步；"
        f"逐初态成本更慢/相同/更快为{comparison['completion_slower']}/{comparison['completion_tied']}/{comparison['completion_faster']}。",
        f"双方均重发现的{len(common_delays)}局中，FBH更晚/相同/更早为"
        f"{comparison['common_rediscovery_later']}/{comparison['common_rediscovery_tied']}/{comparison['common_rediscovery_earlier']}。"
        f"G成功转为FBH失败{comparison['G_success_became_FBH_failure']}局，反向挽救{comparison['G_failure_rescued_by_FBH']}局。", '',
        '## 根评估、反馈执行与预算', '',
        '|方法|隐藏帧|完成模拟均值/最少|未遍历完根池|仅单次根访问|最少根访问|全零回报|隐藏停止|',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    for method in METHODS:
        s = method_summary[method]
        lines.append(f"|{method}|{s['hidden_frames']}|{_format(s['hidden_simulations']['mean'], 1)}/{_format(s['hidden_simulations']['minimum'], 0)}|"
            f"{_format(s['incomplete_legal_root_pool_frames'], 0)}|{_format(s['only_single_visit_frames'], 0)}|"
            f"{_format(s['minimum_completed_root_visit'], 0)}|{s['all_zero_root_return_frames']}|{s['hidden_stops']}|")
    s = method_summary['FBH']
    lines += ['',
        f"FBH在{s['positive_observation_branch_frames']}/{s['hidden_frames']}个隐藏决策创建正观测分支，"
        f"累计{s['positive_observation_nodes']}个正观测节点、{s['negative_observation_nodes']}个负观测节点、"
        f"{s['exact_conditionings']}次条件化。分支被执行不等于其已带来真实任务收益。",
        'G、OL200、FB200使用200ms上限；OLH、FBH使用2000ms诊断上限。高预算超过1秒高层周期，不是实时部署版本。'
        '更高预算下改善只说明该版本受计算资源影响，不能证明已达到收敛或已经超过强G。', '',
        '## 按固定规则选取的失败与挽救', '',
        '失败按FBH相对G成本差从大到小排序，最多三例；挽救按成本改善从大到小选一例。没有符合条件的例子则不补选。', '',
        '|类别|历史/世界|FBH/G捕获|FBH/G首次发现|FBH/G成本|FBH/G路径长|轨迹|',
        '|---|---|---:|---:|---:|---:|---|']
    for label, selected in (('失败', failures), ('挽救', rescued)):
        for pair in selected:
            a, b = pair['FBH'], pair['G']
            links = _link('FBH', a['episode'], destination)+' / '+_link('G', b['episode'], destination)
            lines.append(f"|{label}|{pair['group']} / {pair['world']}|{int(a['capture'])}/{int(b['capture'])}|"
                f"{a['first_seen']}/{b['first_seen']}|{a['completion_cost']}/{b['completion_cost']}|"
                f"{a['pursuer_path_length']:.2f}/{b['pursuer_path_length']:.2f}|{links}|")
    lines += ['', '## 可证实与尚未隔离的原因', '',
        '根访问和完整轨迹能检查候选遗漏、停止、绕行与发现过迟；分支计数能检查反馈是否实际执行。'
        '不能把根已访问解释为搜索已收敛，也不能把几何绕行更合理解释为真实捕获收益。', '',
        '8步规划短视、有限模拟的动作排名误差、观测分支稀疏、固定根运动模型与全状态反应式对手之间的失配，都尚未单独消融。'
        '当前日志不能确定各因素造成的损失份额。不同控制器产生不同逃跑方响应，不能当作固定未来轨迹的反事实比较。', '',
        '本诊断不计算新的显著性检验，不替代公共历史成组的预登记区间，也不授权教师生成或神经训练。', '']
    (output/stage/'failure_diagnosis.json').write_text(
        json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
    destination.write_text('\n'.join(lines))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument('--stage', choices=('development', 'confirmation', 'all'), default='all')
    args = parser.parse_args()
    for stage in STAGE_GROUPS if args.stage == 'all' else (args.stage,):
        result = diagnose_stage(args.output, stage)
        print(json.dumps(dict(stage=stage, status='no_complete_results' if result is None else 'written',
                              games=None if result is None else result['complete_stage_games']), ensure_ascii=False))


if __name__ == '__main__':
    main()
