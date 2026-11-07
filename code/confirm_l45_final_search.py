"""Evaluation-only confirmation of all frozen l45 PPO final checkpoints.

Reuses the unchanged original environment, observation encoding and C++ search.
The previous experiment is immutable; no training, selection or promotion.
"""
import os
for _name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[_name] = '1'
import argparse
import contextlib
import copy
import fcntl
import io
import json
import multiprocessing as mp
from pathlib import Path
import shutil
import time
import numpy as np
import torch
import train_l45_ppo as original

ROOT = original.ROOT
DEFAULT_SOURCE = ROOT/'current/l45_ppo_control_20261022'
DEFAULT_OUTPUT = ROOT/'current/l45_ppo_final_search_20261023'
SEEDS = (0, 1, 2)
LAYOUT_SPLITS = (('confirmation', 'four', 510), ('regression', 'none', 511))


def physical_hash(spec):
    return original.json_hash({k: spec[k] for k in
        ('layout', 'initial_state', 'action_lims', 'control_lims', 'evader_speed', 'pursuer_speed')})


def source_contract(source):
    source = Path(source)
    if original.read_json(source/'status.json')['phase'] != 'complete':
        raise ValueError('The parent experiment must be complete')
    experiment = original.read_json(source/'experiment.json')
    parent_args = original.parse_args(['--output', str(source), '--stage', 'evaluate',
                                      '--model-dir', experiment['source']])
    original.verify_experiment(parent_args, source)
    selection = original.read_json(source/'confirmation_selection.json')
    registry = [dict(label='baseline', seed=None, episode=0, models=str(source/'baseline'), checkpoint=None)]
    for kind in ('best', 'final'):
        for seed in SEEDS:
            checkpoint = source/f'seed{seed}/{kind}.pt'
            saved = original.load_torch(checkpoint)
            expected = selection[str(seed)]['best_episode'] if kind == 'best' else experiment['config']['episodes']
            if saved['episode'] != expected:
                raise ValueError(f'Unexpected frozen checkpoint: {checkpoint}')
            directory = source/f'seed{seed}/{kind}_models'
            for robot, name in enumerate(original.MODEL_NAMES[:4]):
                exported = original.load_torch(directory/name)
                if set(exported) != set(saved['actors'][robot]) or any(
                    not torch.equal(value, saved['actors'][robot][key]) for key, value in exported.items()):
                    raise ValueError(f'Export is not the frozen checkpoint: {directory/name}')
            registry.append(dict(label=f'seed{seed}_{kind}', seed=seed, episode=expected,
                                 models=str(directory), checkpoint=str(checkpoint),
                                 checkpoint_sha256=original.sha(checkpoint)))
    baseline = original.model_fingerprint(source/'baseline')
    for model in registry:
        model['model_hash'] = original.model_fingerprint(model['models'])
        for name in (*original.MODEL_NAMES[:2], original.MODEL_NAMES[-1]):
            if model['model_hash'][name] != baseline[name]:
                raise ValueError(f'Frozen evader or search value changed: {model["label"]}/{name}')
    return dict(parent_experiment_sha256=original.sha(source/'experiment.json'),
                parent_scenarios_sha256=original.sha(source/'scenarios.json'),
                parent_selection_sha256=original.sha(source/'confirmation_selection.json'),
                parent_results_sha256=original.sha(source/'results.json'), registry=registry)


def used_physical_scenes(source):
    prior = original.read_json(Path(source)/'scenarios.json')
    result = {physical_hash(s) for cases in prior.values() for s in cases}
    for seed in SEEDS:
        for path in sorted((Path(source)/f'seed{seed}/rollouts').glob('*.json')):
            result.update(physical_hash(row['spec']) for row in original.read_json(path))
    return result


def build_fresh_scenarios(source, four_count, none_count):
    used = used_physical_scenes(source); initial_used = len(used); result = {}
    for name, layout, phase in LAYOUT_SPLITS:
        count = four_count if layout == 'four' else none_count
        scenarios = []; index = 0
        while len(scenarios) < count:
            spec = original.make_scenario(original.scenario_seed(phase, 20261023, index), layout)
            index += 1; identity = physical_hash(spec)
            if identity in used:
                continue
            used.add(identity); scenarios.append(spec)
        result[name] = scenarios
    return result, dict(previous_physical_scenes=initial_used, new_scenes=four_count+none_count,
                        physical_overlap=0, namespace='scenario_seed(510/511, 20261023, index)')


def search_settings(args):
    return dict(search_budget=3 if args.smoke else 1000, search_only_budget=3 if args.smoke else 3000,
                search_depth=100, beta_policy=.5, beta_value=.1)


def signed_record(args):
    source = source_contract(args.source)
    record = dict(source=str(args.source), source_contract=source,
        config=dict(four_episodes=args.four_episodes, none_episodes=args.none_episodes,
                    diagnostic_pairs=args.diagnostic_pairs, search=search_settings(args), smoke=args.smoke,
                    seeds=SEEDS, opponents=original.OPPONENTS, roles=[1],
                    models_search='baseline + all 3 frozen best + all 3 frozen final',
                    models_policy='baseline + all 3 frozen final', selection='none',
                    primary='final search minus baseline search',
                    secondary='final minus same-seed best; policy/search transfer',
                    no_training=True, no_model_promotion=True),
        implementation=dict(evaluator=original.sha(__file__), parent=original.implementation_files()))
    return json.loads(json.dumps(record))


def verify(args):
    out = args.output; record = original.read_json(out/'experiment.json')
    if signed_record(args) != record:
        raise ValueError('Source, model, configuration, or implementation fingerprint changed')
    for model in record['source_contract']['registry']:
        if original.model_fingerprint(out/'models'/model['label']) != model['model_hash']:
            raise ValueError('Frozen model snapshot changed')
    expected = original.json_hash(dict(record=record, scenarios_sha256=original.sha(out/'scenarios.json')))
    if original.read_json(out/'signature.json')['sha256'] != expected:
        raise ValueError('Frozen evaluation scenarios changed')


def prepare(args):
    out = args.output
    if out == args.source or out.is_relative_to(args.source):
        raise ValueError('Use a separate sibling output; never modify the parent experiment')
    out.mkdir(parents=True, exist_ok=True)
    record = signed_record(args)
    if (out/'experiment.json').exists():
        if original.read_json(out/'experiment.json') != record:
            raise ValueError('Evaluation source/configuration fingerprint changed; use a new output')
    else:
        for model in record['source_contract']['registry']:
            target = out/'models'/model['label']; target.mkdir(parents=True, exist_ok=True)
            for name in original.MODEL_NAMES:
                shutil.copyfile(Path(model['models'])/name, target/name)
            if original.model_fingerprint(target) != model['model_hash']:
                raise ValueError('Frozen model copy mismatch')
        original.write_json(out/'experiment.json', record)
    if not (out/'scenarios.json').exists():
        scenes, overlap = build_fresh_scenarios(args.source, args.four_episodes, args.none_episodes)
        original.write_json(out/'scenarios.json', scenes)
        original.write_json(out/'scenario_independence.json', overlap)
    scenes = original.read_json(out/'scenarios.json')
    previous = used_physical_scenes(args.source)
    new = [physical_hash(s) for cases in scenes.values() for s in cases]
    if len(set(new)) != len(new) or previous.intersection(new):
        raise ValueError('Physical evaluation scenes overlap previous data')
    if len(scenes['confirmation']) != args.four_episodes or len(scenes['regression']) != args.none_episodes:
        raise ValueError('Evaluation scene counts changed')
    signature = original.json_hash(dict(record=record, scenarios_sha256=original.sha(out/'scenarios.json')))
    if (out/'signature.json').exists() and original.read_json(out/'signature.json')['sha256'] != signature:
        raise ValueError('Frozen evaluation scenarios changed')
    original.write_json(out/'signature.json', dict(sha256=signature))
    verify(args)
    return scenes, record


def build_tasks(args, scenarios, record):
    tasks = []; baseline = args.output/'models/baseline'
    for model in record['source_contract']['registry']:
        modes = ('search', 'policy') if model['label'] == 'baseline' or model['label'].endswith('_final') else ('search',)
        tasks.extend(original.evaluation_tasks(args.output/'models'/model['label'], model['label'], model['seed'],
                      baseline, scenarios, search_settings(args), modes=modes, roles=(1,)))
    if len(tasks) != len(scenarios)*len(original.OPPONENTS)*11:
        raise AssertionError('Expected seven search and four policy models')
    if len({t['key'] for t in tasks}) != len(tasks):
        raise ValueError('Duplicate evaluation tasks')
    return tasks


def strict_pairs(rows, candidate_kind, reference_kind, mode):
    def index(label):
        selected = [r for r in rows if r['label'] == label and r['mode'] == mode and r['role'] == 1]
        result = {}
        for row in selected:
            key = (row['layout'], row['episode_hash'], row['opponent'])
            if key in result:
                raise ValueError('Duplicate paired evaluation row')
            result[key] = row
        if not result:
            raise ValueError(f'Missing paired rows: {label}/{mode}')
        return result
    candidates = [index(f'seed{seed}_{candidate_kind}') for seed in SEEDS]
    references = [index('baseline' if reference_kind == 'baseline' else f'seed{seed}_{reference_kind}') for seed in SEEDS]
    keys = sorted(references[0])
    if any(set(item) != set(keys) for item in candidates+references):
        raise ValueError('Missing or unpaired evaluation rows')
    for candidate, reference in zip(candidates, references):
        for key in keys:
            if candidate[key]['search_seeds'] != reference[key]['search_seeds']:
                raise ValueError('Unpaired search random seeds')
    return keys, candidates, references


def paired_statistics(keys, differences):
    delta = np.asarray(differences, float)
    scene_keys = sorted({key[:2] for key in keys})
    scene_indices = [[i for i, key in enumerate(keys) if key[:2] == scene] for scene in scene_keys]
    clustered = np.array([[line[idx].mean() for idx in scene_indices] for line in delta])
    rng = np.random.default_rng(45020261023); draws = []
    for _ in range(4000):
        seeds = rng.integers(len(SEEDS), size=len(SEEDS))
        scenes = rng.integers(len(scene_keys), size=len(scene_keys))
        draws.append(float(clustered[np.ix_(seeds, scenes)].mean()))
    return dict(delta=float(delta.mean()), per_seed_delta=delta.mean(1).tolist(),
                ci95=[float(x) for x in np.percentile(draws, [2.5, 97.5])],
                n_scenes=len(scene_keys), observed_discordances=int(np.count_nonzero(delta)),
                per_opponent_delta={op: float(delta[:, [i for i,k in enumerate(keys) if k[2] == op]].mean())
                                    for op in original.OPPONENTS},
                note='Conditional on the single original l45. Scenes jointly resampled across opponents and PPO seeds; zero-discordance intervals do not prove equivalence.')


def comparison(rows, candidate='final', reference='baseline', mode='search'):
    keys, candidates, references = strict_pairs(rows, candidate, reference, mode)
    differences = [[c[key]['legal_double_capture']-r[key]['legal_double_capture'] for key in keys]
                   for c,r in zip(candidates,references)]
    result = paired_statistics(keys, differences)
    result.update(candidate=candidate, reference=reference, mode=mode,
                  own_illegal_delta=float(np.mean([[c[key]['pursuer_illegal']-r[key]['pursuer_illegal']
                                                    for key in keys] for c,r in zip(candidates,references)])))
    return result


def transfer_comparison(rows):
    keys, cs, rs = strict_pairs(rows, 'final', 'baseline', 'search')
    pk, cp, rp = strict_pairs(rows, 'final', 'baseline', 'policy')
    if keys != pk:
        raise ValueError('Policy/search transfer cases differ')
    diff = [[(c[key]['legal_double_capture']-r[key]['legal_double_capture']) -
             (p[key]['legal_double_capture']-q[key]['legal_double_capture']) for key in keys]
            for c,r,p,q in zip(cs,rs,cp,rp)]
    result = paired_statistics(keys,diff)
    result['meaning'] = '(final_search - baseline_search) - (final_policy - baseline_policy); diagnostic, not the primary criterion'
    return result


def model_summary(rows):
    groups = {}
    for row in rows:
        groups.setdefault((row['label'],row['mode']),[]).append(row)
    return [dict(label=label,mode=mode,**original.summarize(part)) for (label,mode),part in groups.items()]


def scientific_digest(row):
    return original.json_hash({k:v for k,v in row.items()
                               if 'latency' not in k and k not in ('key','timing_context','trace')})


def trace_evaluation(task):
    """Observe the existing search once; never sample or call an extra search."""
    original_run = original.run_episode
    trace = []

    def recorded_run(problem, team_solver, initial_state):
        own = team_solver.pursuer_solver.solver
        search = own.search
        actors = original.load_actors(problem, directory=task['models'])
        baseline = original.load_actors(problem, directory=task['baseline'])
        roots = []

        def recorded_search(p, state, turn=0):
            result = search(p, state, turn=turn)
            children = np.asarray(result.child_distribution)
            best = children[np.argsort(-children[:, -1])[:5]].tolist() if children.size else []
            roots.append(dict(turn=int(turn), child_visit_total=float(children[:, -1].sum()) if children.size else 0.,
                              value=np.asarray(result.value).tolist(), top_children=best))
            return result

        own.search = recorded_search

        class RecordingTeam:
            def policy(self, p, state):
                roots.clear()
                action = team_solver.policy(p, state)
                for name in ('last_joint_latency_ms', 'last_evader_latency_ms', 'last_pursuer_latency_ms'):
                    setattr(self, name, getattr(team_solver, name))
                policies = []
                with torch.no_grad():
                    for robot in original.TEAMS[1]:
                        obs = torch.tensor(p.policy_encoding(state, robot).reshape(-1), dtype=torch.float32)
                        limits = torch.tensor(p.action_lims[p.action_idxs[robot]], dtype=torch.float32)
                        current = original.distribution(actors[robot], obs, limits)
                        reference = original.distribution(baseline[robot], obs, limits)
                        active_targets = [e for e in p.evaders if p.is_active(state, e)]
                        xy = state[p.state_idxs[robot], 0][:2]
                        nearest = min(active_targets, key=lambda e: np.linalg.norm(
                            xy-state[p.state_idxs[e], 0][:2])) if active_targets else None
                        policies.append(dict(robot=int(robot), active=bool(p.is_active(state, robot)),
                            mean=current.mean.tolist(), std=current.stddev.tolist(),
                            baseline_mean=reference.mean.tolist(), baseline_std=reference.stddev.tolist(),
                            mean_shift=float(torch.linalg.vector_norm(current.mean-reference.mean)),
                            nearest_active_evader=None if nearest is None else int(nearest)))
                trace.append(dict(time=float(state[p.time_idx, 0]), state=state.reshape(-1).tolist(),
                                  action=np.asarray(action).reshape(-1).tolist(),
                                  active_evaders=int(p.active_evader_count(state)),
                                  policies=policies, roots=copy.deepcopy(roots)))
                return action

        try:
            return original_run(problem, RecordingTeam(), initial_state)
        finally:
            own.search = search

    original.run_episode = recorded_run
    try:
        row = original.evaluate_one(task)
    finally:
        original.run_episode = original_run
    if scientific_digest(row) != task['expected_scientific_digest']:
        raise ValueError('Passive diagnostic replay changed the scientific result')
    row['trace'] = trace
    return row


def diagnostic_tasks(args, scenes, rows):
    indexed = {}
    for row in rows:
        key = (row['layout'], row['episode_hash'], row['opponent'])
        indexed.setdefault((row['label'], row['mode']), {})[key] = row
    scene_map = {s['episode_hash']: s for cases in scenes.values() for s in cases}
    base = indexed[('baseline', 'search')]
    categories = ('final_gain', 'final_loss', 'final_over_best', 'best_over_final', 'policy_gain_without_search')
    selected = []; tasks = []; seen = set()
    for layout in ('four', 'none'):
        for category in categories:
            eligible = []
            for seed in SEEDS:
                final = indexed[(f'seed{seed}_final', 'search')]
                best = indexed[(f'seed{seed}_best', 'search')]
                policy = indexed[(f'seed{seed}_final', 'policy')]
                bp = indexed[('baseline', 'policy')]
                for key in sorted(base):
                    if key[0] != layout or (seed, key) in seen:
                        continue
                    b, f, v, p, q = [bool(r[key]['legal_double_capture']) for r in (base, final, best, policy, bp)]
                    match = dict(final_gain=f and not b, final_loss=b and not f,
                                 final_over_best=f and not v, best_over_final=v and not f,
                                 policy_gain_without_search=p and not q and not f)[category]
                    if match:
                        eligible.append((seed, key))
            if not eligible or len(selected) >= args.diagnostic_pairs:
                continue
            seed, key = eligible[0]; seen.add((seed, key))
            selected.append(dict(seed=seed, category=category, layout=layout, episode_hash=key[1], opponent=key[2]))
            for label in ('baseline', f'seed{seed}_best', f'seed{seed}_final'):
                task = original.evaluation_tasks(args.output/'models'/label, label,
                    None if label == 'baseline' else seed, args.output/'models/baseline',
                    [scene_map[key[1]]], search_settings(args), modes=('search',), roles=(1,),
                    opponents=(key[2],), timing_context='diagnostic_replay')[0]
                task['category'] = category
                task['expected_scientific_digest'] = scientific_digest(indexed[(label, 'search')][key])
                task['key'] = original.json_hash({k:v for k,v in task.items() if k != 'key'})
                tasks.append(task)
    return tasks, selected


def run_diagnostics(args, scenes, rows, pool, status):
    tasks, selected = diagnostic_tasks(args, scenes, rows)
    folder = args.output/'diagnostics'; folder.mkdir(exist_ok=True)
    manifest = [dict(key=t['key'], label=t['label'], category=t['category'],
                     episode_hash=t['spec']['episode_hash']) for t in tasks]
    if (folder/'manifest.json').exists() and original.read_json(folder/'manifest.json') != manifest:
        raise ValueError('Diagnostic manifest changed')
    original.write_json(folder/'manifest.json', manifest)
    original.write_json(folder/'selection.json', dict(cases=selected, selection='First sorted eligible pair per category/layout; diagnostic only, no model selection'))
    missing = [t for t in tasks if not (folder/(t['key']+'.json')).exists()]
    status('diagnostics', evaluation='diagnostics', completed=len(tasks)-len(missing), total=len(tasks))
    iterator = pool.imap_unordered(trace_evaluation, missing, chunksize=1)
    for i, row in enumerate(iterator, 1):
        original.write_json(folder/(row['key']+'.json'), row)
        status('diagnostics', evaluation='diagnostics', completed=len(tasks)-len(missing)+i, total=len(tasks))
    replay = []
    for task in tasks:
        row = original.read_json(folder/(task['key']+'.json'))
        if row['key'] != task['key'] or scientific_digest(row) != task['expected_scientific_digest']:
            raise ValueError('Diagnostic cache mismatch')
        after = [step for step in row['trace'] if step['active_evaders'] == 1]
        shifts = [p['mean_shift'] for step in after for p in step['policies'] if p['active']]
        replay.append(dict(label=row['label'], category=task['category'], layout=row['layout'],
                           episode_hash=row['episode_hash'], opponent=row['opponent'],
                           legal_double_capture=row['legal_double_capture'], first_capture_time=row['first_capture_time'],
                           one_capture_timeout=row['one_capture_timeout'], pursuer_illegal=row['pursuer_illegal'],
                           post_first_capture_steps=len(after),
                           post_first_capture_mean_shift=float(np.mean(shifts)) if shifts else None,
                           scientific_replay_matches=True))
    result = dict(cases=selected, replays=replay, all_replays_match=True,
                  note='Nearest active evader is a geometric proxy, not inferred intention. Root visits are observations of the original search calls.')
    original.write_json(folder/'summary.json', result)
    return result


def load_rows(out, split):
    folder = out/split
    manifest = original.read_json(folder/'manifest.json')
    if len({t['key'] for t in manifest}) != len(manifest):
        raise ValueError('Duplicate manifest keys')
    rows = [original.read_json(folder/(t['key']+'.json')) for t in manifest]
    if any(t['key'] != row['key'] for t, row in zip(manifest, rows)):
        raise ValueError('Result/manifest mismatch')
    return rows


def report(args):
    verify(args)
    results = dict(source=str(args.source), no_training=True, smoke=args.smoke,
                   full_deployment_validated=False, new_evader_guard_performed=False,
                   continue_formal=False, source_unchanged=True, comparisons={}, summaries={})
    for split, layout, _ in LAYOUT_SPLITS:
        rows = load_rows(args.output, split)
        expected = build_tasks(args, original.read_json(args.output/'scenarios.json')[split],
                               original.read_json(args.output/'experiment.json'))
        if {r['key'] for r in rows} != {t['key'] for t in expected}:
            raise ValueError('Report requires the entire frozen evaluation')
        result = dict(final_vs_baseline_search=comparison(rows),
                      final_vs_best_search=comparison(rows, reference='best'),
                      final_vs_baseline_policy=comparison(rows, mode='policy'),
                      search_policy_transfer=transfer_comparison(rows))
        results['comparisons'][layout] = result
        results['summaries'][layout] = model_summary(rows)
    main = results['comparisons']['four']['final_vs_baseline_search']
    results['pursuit_gain_supported'] = bool(not args.smoke and main['delta'] >= .05 and
        main['ci95'][0] > 0 and min(main['per_seed_delta']) >= 0 and main['own_illegal_delta'] <= .02)
    results['no_obstacle_point_margin_met'] = bool(not args.smoke and
        results['comparisons']['none']['final_vs_baseline_search']['delta'] >= -.03)
    diagnostic = original.read_json(args.output/'diagnostics/summary.json')
    results['diagnostic_replays'] = len(diagnostic['replays'])
    results['all_diagnostic_replays_match'] = diagnostic['all_replays_match']
    original.write_json(args.output/'results.json', results)
    lines = ['# l45 PPO 最终检查点独立搜索确认', '',
             '本轮仅评测已冻结的原 l45、三个验证最佳模型与三个最终模型，没有追加训练或重新选模型。',
             '三个 PPO 种子均从同一个原 l45 开始；区间以该 l45 为条件，不代表多个独立预训练模型。',
             '新场景的物理状态、速度参数已与上一轮训练、验证、确认数据去重。', '',
             f'小规模功能检查：{args.smoke}。正式搜索为 1,000 次模拟、深度 100、βπ=0.5、βv=0.1；Search Only 对手 3,000 次、深度 100。',
             '四障碍和无障碍均对三个固定原对手评测；同场景复用搜索种子。并行决策耗时不能用于声称部署加速。', '',
             '| 模型 | 训练局数 | 四障碍搜索 | 无障碍搜索 | 四障碍纯策略 | 无障碍纯策略 |',
             '|---|---:|---:|---:|---:|---:|']
    registry = original.read_json(args.output/'experiment.json')['source_contract']['registry']
    def rate(layout, label, mode):
        matches = [r for r in results['summaries'][layout] if r['label'] == label and r['mode'] == mode]
        return f"{matches[0]['success_rate']:.2%} ({matches[0]['success']}/{matches[0]['n']})" if matches else '未评测'
    for model in registry:
        label = model['label']
        lines.append(f"| {label} | {model['episode']} | {rate('four',label,'search')} | {rate('none',label,'search')} | {rate('four',label,'policy')} | {rate('none',label,'policy')} |")
    for layout in ('four', 'none'):
        lines.extend(['', f"## {'四障碍确认' if layout == 'four' else '无障碍回归'}", ''])
        for name, title in (('final_vs_baseline_search','最终模型搜索相对原 l45 搜索'),
                            ('final_vs_best_search','最终模型搜索相对同种子验证最佳搜索'),
                            ('final_vs_baseline_policy','最终纯策略相对原 l45 纯策略'),
                            ('search_policy_transfer','搜索增益减去纯策略增益（诊断项）')):
            item = results['comparisons'][layout][name]
            ci = item['ci95']
            deltas = '、'.join(f'{100*d:+.2f}' for d in item['per_seed_delta'])
            lines.append(f"- {title}：平均 {100*item['delta']:+.2f} 个百分点，配对 95% 区间 [{100*ci[0]:+.2f}, {100*ci[1]:+.2f}]；种子 0/1/2 分别 {deltas} 个百分点。")
        item = results['comparisons'][layout]['final_vs_baseline_search']
        lines.append(f"- 自身非法率变化：{100*item['own_illegal_delta']:+.2f} 个百分点。")
        lines.append('- 分对手搜索双捕率变化：' + '；'.join(f'{op} {100*d:+.2f} 个百分点' for op,d in item['per_opponent_delta'].items())+'。')
    lines.extend(['', '## 判断与限制', '',
        f"- 四障碍追捕提升证据达到预设门槛：{results['pursuit_gain_supported']}。",
        f"- 无障碍搜索平均下降不超过 3 个百分点：{results['no_obstacle_point_margin_met']}。",
        '- 本轮没有新增逃逸方完整搜索回归，也没有单进程部署耗时复测，因此完整部署门槛未验证；不自动替换原 l45、不启动 5 万局。',
        '- 上一轮 results.json 与未通过的质量门槛保留；最终模型没有追认为验证最佳模型。',
        f"- 分歧对局被动重放 {results['diagnostic_replays']} 局，科学指标逐项一致：{results['all_diagnostic_replays_match']}。详见 diagnostics/summary.json 和对应轨迹。",
        '- 分歧对局按确定规则挑选，可能偏向排序靠前的种子；只解释个案，不能估计失败原因总体占比。',
        '- 原基准指标、首捕后超时、双方非法与抓捕用时见 confirmation/summary.json、regression/summary.json。',
        '- 区间按场景成组、三个对手共同重采样，再重采样三个 PPO 种子。仅三个训练种子且共享预训练 l45；没有观测到差异不能证明等价。', ''])
    (args.output/'REPORT.md').write_text('\n'.join(lines), encoding='utf-8')
    verify(args)
    return results


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=DEFAULT_SOURCE)
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument('--workers', type=int, default=80)
    parser.add_argument('--four-episodes', type=int, default=200)
    parser.add_argument('--none-episodes', type=int, default=100)
    parser.add_argument('--diagnostic-pairs', type=int, default=12)
    parser.add_argument('--stage', choices=('pipeline','prepare','evaluate','report'), default='pipeline')
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args(argv)
    args.source = args.source.resolve(); args.output = args.output.resolve()
    if args.output == args.source or args.output.is_relative_to(args.source):
        parser.error('Output must be separate from the parent experiment')
    if args.smoke:
        if args.output == DEFAULT_OUTPUT.resolve():
            parser.error('--smoke requires a separate --output')
        args.four_episodes = args.none_episodes = 1
        args.diagnostic_pairs = 2
    if min(args.workers,args.four_episodes,args.none_episodes) < 1 or args.diagnostic_pairs < 0:
        parser.error('Workers/scenes must be positive and diagnostic pairs nonnegative')
    return args


def main(argv=None):
    args = parse_args(argv)
    torch.set_num_threads(1)
    args.output.mkdir(parents=True, exist_ok=True)
    lock = (args.output/'run.lock').open('a+')
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as error:
        raise RuntimeError('This output already has a running evaluator') from error
    clocks = {}
    def status(phase, **fields):
        now = time.monotonic()
        if phase in ('evaluation', 'diagnostics'):
            name = fields['evaluation']
            start, initial = clocks.setdefault(name, (now, fields['completed']))
            elapsed = now-start
            fields.update(elapsed_seconds=elapsed, rate=(fields['completed']-initial)/max(elapsed,1e-9))
        original.write_json(args.output/'status.json',
            dict(phase=phase,pid=os.getpid(),updated_unix=time.time(),no_training=True,**fields))
        print(json.dumps(dict(phase=phase,**fields), ensure_ascii=False), flush=True)
    try:
        status('checking')
        scenes, record = prepare(args)
        if args.stage == 'prepare':
            status('prepared')
            return
        if args.stage != 'report':
            with mp.get_context('spawn').Pool(args.workers) as pool:
                for split, _, _ in LAYOUT_SPLITS:
                    verify(args)
                    original.run_evaluation(args.output, split, build_tasks(args,scenes[split],record),pool,status)
                verify(args)
                rows = [r for split,_,_ in LAYOUT_SPLITS for r in load_rows(args.output,split)]
                run_diagnostics(args,scenes,rows,pool,status)
        status('summarizing')
        report(args)
        status('complete',continue_formal=False)
    except KeyboardInterrupt:
        status('interrupted')
        raise
    except Exception as error:
        status('failed',error=repr(error))
        raise
    finally:
        lock.close()


if __name__ == '__main__':
    main()
