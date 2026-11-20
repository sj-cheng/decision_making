"""Verify the expanded data split, frozen model, and paired traces."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
from statistics import median

import numpy as np

from run_e52_counterfactual_pilot import dynamics
from run_e52_train_nar_nae import initial_id


def read(path):
    return json.loads(path.read_text())


def require(condition, message):
    if not condition:
        raise ValueError(message)


def audit(root):
    dataset = read(root/'summary.json')
    manifest = read(root/'manifest.json')
    require(dataset['state']=='complete', 'dataset is still running')
    require(dataset['completed_groups']==dataset['requested_groups'], 'missing branch groups')
    require(not dataset['errors'], f"sampling encountered {len(dataset['errors'])} protocol errors")
    sources = {r['path']: r for r in manifest['cases']}
    require(len(sources)==len(manifest['cases']), 'duplicate case path')
    for path, source in sources.items():
        case = read(Path(path))
        require(hashlib.sha256(json.dumps(case['map']).encode()).hexdigest()==source['map_hash']
                and initial_id(case)==source['initial_id'], 'source map or initial state changed')
    for field in ('map_hash', 'initial_id', 'family'):
        splits = [{r[field] for r in sources.values() if r['split']==split}
                  for split in ('train','validation','test')]
        require(all(splits) and not any(a&b for i,a in enumerate(splits) for b in splits[i+1:]),
                f'{field} overlaps splits')

    groups = [g for g in dataset['groups'] if 'records' in g]
    coverage, categories = Counter(), Counter()
    first_action_invalid = subsequent_minco = collisions = 0
    for group in groups:
        source = sources[group['case']]
        require(group['split']==source['split'] and group['map_hash']==source['map_hash'],
                'label group does not match its source case')
        require(group['repeated_identically'] and group['snapshot'], 'missing replay verification')
        records = group['records']
        require(4<=len(records)<=8, 'candidate count outside 4–8')
        refs = [r for r in records if r['candidate_id']==group['reference']]
        require(len(refs)==1 and abs(refs[0]['advantage'])<1e-9, 'missing zero-advantage reference')
        require(all(abs(r['return_8']-refs[0]['return_8']-r['advantage'])<1e-9 for r in records),
                'advantage does not match branch return difference')
        cycle, aid = group['cycle'], str(group['aid'])
        directory = root/'branches'/Path(group['case']).stem/f"{group['condition']}_{cycle}_{aid}"
        reference = dynamics(read(directory/'reference.json'))
        require(reference==dynamics(read(directory/'repeat.json')), 'reference replay mismatch')
        teammate = str(1-int(aid))
        first_teammate_plan = reference[cycle]['plans'][teammate]
        for record in records:
            if record['candidate_id']==group['reference']:
                continue
            name = f"action_{record['candidate_id']:02d}.json"
            events = dynamics(read(directory/name))
            require(events[:cycle]==reference[:cycle], 'alternative branch changed its prefix')
            require(events[cycle]['plans'][teammate]==first_teammate_plan, 'teammate action leaked')
            require(events[cycle]['plans'][aid]['endpoint']==record['endpoint'], 'wrong action intervened')
        coverage[group['split']] += len(records)
        for r in records:
            categories[r['option']] += 1
            if r['features'][2]<.05:
                categories['near_stationary'] += 1
            first_action_invalid += not r['first_action_feasible']
            subsequent_minco += r['infeasible']
            collisions += r['collisions']
    require(sum(coverage.values())==dataset['labels'], 'dataset label count mismatch')

    evaluation = root/'evaluation'
    entry = read(evaluation/'manifest.json')
    summary = read(evaluation/'summary.json')
    rows = summary['rows']
    test = {path for path, source in sources.items() if source['split']=='test'}
    require(test==set(entry['test_cases']), 'evaluation changed the test split')
    require(hashlib.sha256((root/'advantage.pt').read_bytes()).hexdigest()==entry['checkpoint_sha256'],
            'evaluation checkpoint hash mismatch')
    expected = {(profile, Path(case).stem, arm) for profile in entry['profiles']
                for case in test for arm in entry['arms']}
    require(len(rows)==summary['completed']==summary['total']==len(expected)
            and {(r['profile'],r['case'],r['arm']) for r in rows}==expected,
            'missing or duplicated evaluation pairs')

    timings = {}
    for r in rows:
        trace = read(Path(r['trace']))
        profile, arm = r['profile'], r['arm']
        delay, drops = entry['profiles'][profile]
        require(trace['case'] in test and Path(trace['case']).stem==r['case'], 'trace case mismatch')
        require(trace['delay']==delay and trace['drop_cycles']==drops, 'communication profile mismatch')
        require(trace['requested_cycles']==trace['environment_horizon']==160 and trace['rootpuct_simulations']==20
                and trace['seed']==17 and trace['lidar_rays']==256 and trace['e52_budget']==256
                and trace['e52_depth']==3 and trace['e52_time_ms']==200, 'online evaluation settings mismatch')
        require(bool(trace['e52_advantage_checkpoint'])==(arm=='branch_advantage'), 'critic assignment mismatch')
        if arm != 'geometry':
            require(Path(trace['e52_checkpoint']).resolve().is_file(), 'missing frozen NAR checkpoint')
        if arm=='branch_advantage':
            require(Path(trace['e52_advantage_checkpoint']).resolve()==(root/'advantage.pt').resolve(),
                    'unexpected critic checkpoint path')
        for step in trace['logs']:
            for search in step['search'].values():
                require(search['nar_enabled']==(arm!='geometry')
                        and search['nae_enabled']==(arm in ('both_residual','branch_advantage')),
                        'network ablation mismatch')
                timings.setdefault((profile,arm), []).append(search['elapsed_ms'])

    paired_comparisons = {}
    for profile in entry['profiles']:
        profile_rows = [r for r in rows if r['profile']==profile]
        new = {r['case']:r for r in profile_rows if r['arm']=='branch_advantage'}
        paired_comparisons[profile] = {}
        for arm in ('geometry','nar','both_residual'):
            other = {r['case']:r for r in profile_rows if r['arm']==arm}
            comparable = [(r, other[case]) for case,r in new.items()]
            delta = [a['capture_cycles'][1]-b['capture_cycles'][1] for a,b in comparable
                     if a['captures']==b['captures']==2]
            paired_comparisons[profile][arm] = dict(
                double_capture_gain=sum(a['captures']==2 for a,_ in comparable)
                    - sum(b['captures']==2 for _,b in comparable),
                joint_success_pairs=len(delta), median_second_capture_delta=median(delta) if delta else None,
                faster=sum(x<0 for x in delta), equal=sum(x==0 for x in delta), slower=sum(x>0 for x in delta),
            )

    return dict(dataset_labels=dict(coverage), option_coverage=dict(categories),
                invalid_first_actions=first_action_invalid, branch_minco_infeasible=subsequent_minco,
                branch_collision_cycles=collisions, evaluations=len(rows),
                selected_model=read(root/'advantage.json'),
                paired_results=summary['results'],
                paired_comparisons=paired_comparisons,
                pooled_p95_ms={f'{p}/{a}':float(np.percentile(ms,95)) for (p,a),ms in timings.items()})


def render(result):
    labels = {'geometry': '纯 H-MCTS', 'nar': '仅 NAR', 'both_residual': '旧 NAE 限幅',
              'branch_advantage': '新 NAE 限幅'}
    lines = [
        '# E52 分支收益训练与闭环评测',
        '',
        f"共核验 {sum(result['dataset_labels'].values())} 条分支标签、{result['evaluations']} 局配对记录。",
        f"每个状态下的动作收益与同一几何基准相减，保留 {result['branch_minco_infeasible']} 次分支后续 MINCO 不可行记录。",
        '',
        '| 通信 | 方法 | 双捕获 | 第二次捕获中位周期（成功局） | 停滞步 | 碰撞周期 | MINCO 不可行 | 合并 P95 | 最大单局 P95 |',
        '| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |',
    ]
    for profile, block in result['paired_results'].items():
        for arm, row in block['by_arm'].items():
            timing = result['pooled_p95_ms'][f'{profile}/{arm}']
            cycle = row['median_double_capture_cycle'] if row['median_double_capture_cycle'] is not None else '—'
            lines.append(f"| {profile} | {labels[arm]} | {row['double_captures']}/{row['episodes']} | {cycle} | "
                         f"{row['stall_fraction']:.1%} | {row['collisions']} | {row['minco_infeasible']} | "
                         f"{timing:.1f} ms | {row['max_episode_p95_ms']:.1f} ms |")
    lines.extend(['', '逐局第二次捕获差值只统计两种方法都成功的配对。负值表示新 NAE 更早。'])
    for profile, comparisons in result['paired_comparisons'].items():
        for arm, row in comparisons.items():
            lines.append(f"- {profile} 对比 {labels[arm]}，双捕获差 {row['double_capture_gain']:+d} 局。"
                         f"共同成功 {row['joint_success_pairs']} 局，更快 {row['faster']}，持平 {row['equal']}，更慢 {row['slower']}，"
                         f"配对中位差 {row['median_second_capture_delta']} 周期。")
    lines.extend(['', '留出动作排序、按观测状态的误差和逐局字段见 `audit.json`。', ''])
    return '\n'.join(lines)


def audit_transformer(root):
    base = read(root/'evaluation/manifest.json')
    alternate = root/'transformer_evaluation'
    setup = read(alternate/'manifest.json')
    for name in ('test_cases','profiles','cycles','budget','depth','deadline_ms','old_checkpoint_sha256'):
        require(base[name]==setup[name], f'Transformer evaluation changed {name}')
    checkpoint = root/'advantage_transformer.pt'
    require(hashlib.sha256(checkpoint.read_bytes()).hexdigest()==setup['checkpoint_sha256'],
            'Transformer checkpoint differs from the evaluated checkpoint')
    summary = read(alternate/'summary.json')
    rows = summary['rows']
    keys = {(r['profile'],r['case'],r['arm']) for r in rows}
    expected = {(p,Path(c).stem,'branch_advantage') for p in base['profiles'] for c in base['test_cases']}
    require(keys==expected and len(rows)==summary['completed']==summary['total']==len(expected),
            'Transformer evaluation is incomplete')
    for row in rows:
        trace = read(Path(row['trace']))
        delay, drops = setup['profiles'][row['profile']]
        require(trace['case'] in base['test_cases'] and trace['delay']==delay
                and trace['drop_cycles']==drops and trace['seed']==17
                and trace['e52_budget']==256 and trace['e52_depth']==3
                and trace['requested_cycles']==trace['environment_horizon']==160
                and trace['e52_time_ms']==200
                and Path(trace['e52_advantage_checkpoint']).resolve()==checkpoint.resolve(),
                'Transformer trace settings mismatch')
    model = read(root/'advantage_transformer.json')
    require(model['architecture']=='transformer', 'wrong architecture metadata')
    return dict(training=model, results=summary['results'], evaluations=len(rows))


if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    args = parser.parse_args()
    result = audit(args.root)
    transformer = args.root/'transformer_evaluation'/'summary.json'
    if transformer.exists() and read(transformer)['completed']==read(transformer)['total']:
        result['transformer'] = audit_transformer(args.root)
    (args.root/'audit.json').write_text(json.dumps(result, indent=2)+'\n')
    (args.root/'实验核查.md').write_text(render(result))
    print(json.dumps(result,indent=2))
