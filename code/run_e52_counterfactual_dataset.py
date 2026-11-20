"""Collect replay-verified E52 candidate advantages across disjoint map families."""
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path

import numpy as np

from e52_nae_diagnostic_cases import ROOT, _perturb, _pose_key
from run_e52_counterfactual_pilot import Branch, collect, replay
from run_e52_train_nar_nae import initial_id


FAMILY_SPLIT = {
    'long_wall': 'train', 'wall_corner': 'train', 'multi_route': 'train',
    'post_field': 'train', 'staggered_gate': 'validation',
    'split_gate': 'test', 'u_courtyard': 'test',
}
SOURCE = ROOT / 'current/e49_persistent_pursuit_benchmark_20261108/engineering/cases'


def save(path, data):
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(data, indent=2) + '\n')
    temp.replace(path)


def make_cases(output, count):
    rng = np.random.default_rng(521123)
    rows, starts = [], set()
    for family, split in FAMILY_SPLIT.items():
        for density in ('sparse', '16', '24'):
            source = SOURCE / density / f'{family}.json'
            if not source.exists():
                continue
            original = json.loads(source.read_text())
            # The distributed runner uses world zero. Generate fresh starts
            # under that world and avoid mixing scripted history variants.
            original['worlds'] = original['worlds'][:1]
            for index in range(count):
                for _ in range(20):
                    proposal = _perturb(original, rng)
                    if proposal is not None and _pose_key(proposal[0]) not in starts:
                        break
                else:
                    raise RuntimeError(f'could not generate a distinct legal start for {source}')
                case, translation = proposal
                starts.add(_pose_key(case))
                case['id'] = f'{family}_{density}_{index:02d}'
                path = output / f"{case['id']}.json"
                save(path, case)
                map_hash = hashlib.sha256(json.dumps(case['map']).encode()).hexdigest()
                rows.append(dict(path=str(path), family=family, density=density, split=split,
                                 map_hash=map_hash, initial_id=initial_id(case),
                                 translation=translation, source=str(source)))
    for field in ('map_hash', 'initial_id'):
        sets = [{r[field] for r in rows if r['split']==split}
                for split in ('train', 'validation', 'test')]
        if not all(sets) or any(a & b for i,a in enumerate(sets) for b in sets[i+1:]):
            raise RuntimeError(f'{field} overlaps partitions')
    return rows


def select_states(trace, seed):
    pools = {k: [] for k in ('unknown', 'first_sighting', 'occluded', 'near_track')}
    was_visible = [False, False]
    ever_seen = [False, False]
    for row in trace['logs']:
        for aid in (0, 1):
            key = str(aid)
            stats = row['sensor_stats'][key]
            visible = stats['visible_targets'] > 0
            near = stats['nearest_belief']
            conditions = []
            if near is None:
                conditions.append('unknown')
            if visible and not ever_seen[aid]:
                conditions.append('first_sighting')
            if not visible and was_visible[aid] and near is not None:
                conditions.append('occluded')
            if visible and near is not None and near <= 2.5:
                conditions.append('near_track')
            if (row['cycle'] <= 152 and row['plans'][key]['candidate_id'] >= 0
                    and len(row['search'][key]['root']) >= 4):
                for condition in conditions:
                    pools[condition].append((row['cycle'], aid))
            was_visible[aid] = visible
            ever_seen[aid] |= visible
    rng = np.random.default_rng(seed)
    selected, used = [], set()
    for condition in ('first_sighting', 'occluded', 'near_track', 'unknown'):
        choices = [x for x in pools[condition] if x not in used]
        if choices:
            pair = choices[int(rng.integers(len(choices)))]
            selected.append((*pair, condition))
            used.add(pair)
    return selected


def scout(case, directory, seed):
    directory.mkdir(parents=True, exist_ok=True)
    trace = replay(case, directory/'scout.json', Branch(-1, 0), cycles=160)
    if trace['rootpuct_failures'] or trace['terminal_opponent_skips']:
        return [], dict(rootpuct_failures=trace['rootpuct_failures'],
                        terminal_skips=trace['terminal_opponent_skips'])
    return select_states(trace, seed), None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=36)
    parser.add_argument('--starts-per-map', type=int, default=2)
    args = parser.parse_args()
    root = args.output.resolve()
    if (root/'manifest.json').exists():
        raise SystemExit('use a fresh output directory')
    (root/'cases').mkdir(parents=True, exist_ok=True)
    cases = make_cases(root/'cases', args.starts_per_map)
    save(root/'manifest.json', dict(cases=cases, families=FAMILY_SPLIT,
                                    horizon=160, branch_horizon=8, offline_evaluations=128))
    summary = dict(state='scouting', requested_cases=len(cases), scouted=0,
                   requested_groups=0, completed_groups=0, labels=0, errors=[], groups=[])
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        scouts = {}
        for index, spec in enumerate(cases):
            case = Path(spec['path'])
            directory = root/'branches'/case.stem
            scouts[pool.submit(scout, case, directory, 521124+index)] = (case, directory, spec)
        jobs = {}
        for future in as_completed(scouts):
            case, directory, spec = scouts[future]
            states, issue = future.result()
            summary['scouted'] += 1
            if issue:
                summary['errors'].append(dict(case=str(case), phase='scout', **issue))
            for cycle, aid, condition in states:
                path = directory/f'{condition}_{cycle}_{aid}'
                jobs[pool.submit(collect, (case, path, cycle, aid, condition))] = (spec, path)
            summary['requested_groups'] = len(jobs)
            save(root/'summary.json', summary)
        summary['state'] = 'collecting'
        save(root/'summary.json', summary)
        for future in as_completed(jobs):
            spec, path = jobs[future]
            try:
                row = future.result()
            except RuntimeError as exc:
                summary['errors'].append(dict(case=spec['path'], path=str(path), reason=str(exc)))
            else:
                row.update(split=spec['split'], family=spec['family'], map_hash=spec['map_hash'])
                summary['groups'].append(row)
                summary['labels'] += len(row.get('records', []))
                if 'records' in row:
                    save(path/'labels.json', row)
            summary['completed_groups'] += 1
            save(root/'summary.json', summary)
            print(summary['completed_groups'], len(jobs), 'labels',summary['labels'],flush=True)
    summary['state'] = 'complete'
    summary['coverage'] = dict(Counter(f"{r['split']}/{r['condition']}" for r in summary['groups'] if 'records' in r))
    save(root/'summary.json', summary)


if __name__ == '__main__':
    main()
