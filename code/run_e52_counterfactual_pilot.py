"""Replay a seeded prefix and collect matched eight-cycle outcomes for legal actions."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import pickle

import numpy as np

from e52_nae_diagnostic_cases import generate_diagnostic_cases
from run_e50_distributed_dynamic_smoke import run
from solvers.e52_advantage import public_state


class Branch:
    def __init__(self, cycle, aid, candidate=None):
        self.cycle, self.aid, self.candidate = cycle, aid, candidate
        self.snapshot = None

    def apply(self, observations, plans, controllers, replay_state):
        self.snapshot = hashlib.sha256(pickle.dumps(replay_state, protocol=5)).hexdigest()
        obs, plan = observations[self.aid], plans[self.aid]
        self.teammate = plans[1-self.aid]
        self.reference = plan.candidate_id
        self.candidates = controllers[self.aid].last_search['root']
        self.public = public_state(obs)
        self.reward_tracks = obs.belief.payload()
        if self.candidate is not None:
            candidate = self.candidates[self.candidate]
            endpoint = np.asarray(candidate['endpoint'])
            plans[self.aid] = replace(plan, endpoint=endpoint, action=endpoint-obs.pose,
                                      duration=candidate['duration'],
                                      candidate_id=self.candidate, minco_feasible=True)


def replay(case, out, branch, *, cycles=None):
    return run(case, out, cycles=branch.cycle+8 if cycles is None else cycles, e52=True, e52_mode='geometry',
               rootpuct=True, rootpuct_simulations=20, seed=17, delay=1,
               drop_cycles=(2, 3), e52_budget=128, e52_time_ms=1e9,
               offline_branch=branch)


def dynamics(result):
    return [{key: value for key, value in row.items() if key != 'search'}
            for row in result['logs']]


def branch_result(data, branch):
    if data['rootpuct_failures'] or data['terminal_opponent_skips']:
        raise RuntimeError('opponent state or search invalid during branch')
    aid = str(branch.aid)
    rows = data['logs'][branch.cycle:]
    position = np.asarray(branch.public['pose'])
    tracks = branch.reward_tracks
    components = []
    seen = rows[0]['sensor_stats'][aid]['visible_targets'] > 0
    for row in rows:
        end = np.asarray(row['pursuers'][branch.aid])
        targets = np.asarray([target['position'] for target in tracks]).reshape(-1, 2)
        progress = (np.linalg.norm(targets-position, axis=1).min()
                    - np.linalg.norm(targets-end, axis=1).min()) if len(targets) else 0.
        visible = row['sensor_stats'][aid]['visible_targets'] > 0
        components.append(dict(captures=sum(c['cycle']==row['cycle'] for c in row['captures']),
                               progress=float(progress), reacquired=int(visible and not seen),
                               stall=int(np.linalg.norm(end-position)<.03)))
        position, seen, tracks = end, visible, row['beliefs'][aid]
    value = sum(.98**k * (4*c['captures']+.5*c['reacquired']+c['progress']-.1*c['stall']-.05)
                for k,c in enumerate(components))
    return dict(return_8=value, rewards=components,
                collisions=sum(r['pursuer_collision'] for r in rows),
                infeasible=sum(not m['feasible'] for r in rows for m in r['minco'].values()),
                first_action_feasible=rows[0]['minco'][aid]['feasible'])


def collect(task):
    case, out, cycle, aid, condition = task
    out.mkdir(parents=True, exist_ok=True)
    base = Branch(cycle, aid)
    ref = replay(case, out/'reference.json', base)
    if base.snapshot is None or base.reference < 0:
        return dict(case=str(case), cycle=cycle, aid=aid, skipped='no feasible reference action')
    duplicate = Branch(cycle, aid)
    same = replay(case, out/'repeat.json', duplicate)
    if duplicate.snapshot != base.snapshot or dynamics(same) != dynamics(ref):
        raise RuntimeError(f'repeated reference diverges: {case} cycle {cycle}')
    others = sorted((i for i in range(len(base.candidates)) if i != base.reference),
                    key=lambda i: base.candidates[i]['features'][2])
    chosen = [base.reference]
    if others:
        chosen += [others[k] for k in sorted(set(np.linspace(0, len(others)-1, min(7,len(others)), dtype=int)))]
    reference = branch_result(ref, base)
    records = []
    for i in chosen:
        if i == base.reference:
            result = reference
        else:
            branch = Branch(cycle, aid, i)
            data = replay(case, out/f'action_{i:02d}.json', branch)
            if branch.snapshot != base.snapshot or dynamics(data)[:cycle] != dynamics(ref)[:cycle]:
                raise RuntimeError(f'alternative changed prefix: {case} cycle {cycle}')
            before = ref['logs'][cycle]['plans'][str(1-aid)]
            after = data['logs'][cycle]['plans'][str(1-aid)]
            if before != after:
                raise RuntimeError('first-step teammate action changed')
            result = branch_result(data, branch)
        records.append(dict(candidate_id=i, **base.candidates[i], **result,
                            advantage=result['return_8']-reference['return_8']))
    item = dict(case=str(case), cycle=cycle, aid=aid, condition=condition, snapshot=base.snapshot,
                reference=base.reference, repeated_identically=True, public=base.public, records=records)
    (out/'labels.json').write_text(json.dumps(item, indent=2)+'\n')
    return item


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=18)
    parser.add_argument('--cycles', type=int, nargs='+', default=[0, 10])
    args = parser.parse_args()
    root = args.output.resolve()
    if (root/'summary.json').exists():
        raise SystemExit('use a fresh output directory')
    cases, _ = generate_diagnostic_cases(root/'cases', seed=521121)
    tasks = [(case, root/'branches'/case.stem/f'cycle_{cycle}', cycle, (i+j)%2, 'fixed')
             for i,case in enumerate(cases) for j,cycle in enumerate(args.cycles)]
    rows = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        jobs = [pool.submit(collect, task) for task in tasks]
        for job in as_completed(jobs):
            row = job.result()
            rows.append(row)
            summary = dict(completed=len(rows), requested=len(tasks),
                verified=sum(x.get('repeated_identically',False) for x in rows),
                labels=sum(len(x.get('records',[])) for x in rows), groups=rows)
            tmp = root/'summary.tmp'
            tmp.write_text(json.dumps(summary, indent=2)+'\n')
            tmp.replace(root/'summary.json')
            print({k:v for k,v in summary.items() if k!='groups'}, flush=True)


if __name__ == '__main__':
    main()
