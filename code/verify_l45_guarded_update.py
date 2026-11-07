"""Engineering replay of one saved S minibatch epoch, with and without hard KL.

No environment episodes, teacher searches, model export, or success-rate claim.
"""
import os
for name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[name] = '1'

import argparse
from pathlib import Path
import numpy as np
import torch
from torch.distributions import Normal, kl_divergence
import run_l45_search_feedback as s
from learning.kl_guard import accepted_actor_step


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=s.DEFAULT_OUTPUT)
    parser.add_argument('--output', type=Path, default=s.ROOT/'current/l45_completion_search_20261023/review/guarded_update.json')
    args = parser.parse_args(argv)
    args.source = args.source.resolve()
    torch.set_num_threads(1)
    s.verify(s.parse_args(['--output', str(args.source)]))
    directory = args.source/'training/S/seed0'
    manifest_path = directory/'batches/batch000000.json'
    if not manifest_path.exists():
        manifest_path = next(p for p in sorted((directory/'batches').glob('*.json')) if not p.name.endswith('.lock.json'))
    manifest = s.locked_read(manifest_path)
    episodes = [s.read_rollout(task) for task in manifest['tasks']]
    source = s.o.load_torch(manifest['tasks'][0]['snapshot'])
    initial = s.o.load_torch(args.source/'initial/source.pt')
    if source['episode'] != 0 or not s.equal_tree(source['actors'], initial['actors']):
        raise ValueError('Replay requires the first batch and its complete original Adam state')
    teacher_path = directory/'teachers/batch000000.json'
    teacher = s.locked_read(teacher_path)
    labels = [label for task in teacher['tasks'] for label in s.read_teacher(task)['labels']]
    rows = [row for episode in episodes for row in episode['rows']]
    tensor = lambda name: torch.tensor(np.asarray([r[name] for r in rows]), dtype=torch.float32)
    obs, limits, active, old_mean, old_std = [tensor(name) for name in ('obs', 'limits', 'active', 'old_mean', 'old_std')]
    targets, labelled = torch.zeros_like(old_mean), torch.zeros(len(rows), dtype=torch.bool)
    positions = {(ep['index'], step): offset+step for offset, ep in s._episode_offsets(episodes)
                 for step in range(len(ep['rows']))}
    for label in labels:
        index = positions[(label['index'], label['step'])]
        targets[index] = torch.tensor(label['target'])
        labelled[index] = True
    torch.manual_seed(46020261023)
    order = torch.randperm(len(rows))
    outputs = {}
    for guarded in (False, True):
        actors = s.o.load_actors(s.o.original_problem(), states=source['actors'])
        parameters = [p for robot in s.o.TEAMS[1] for p in actors[robot].parameters()]
        optimizer = torch.optim.Adam(parameters, lr=3e-5)
        optimizer.load_state_dict(initial['actor_optimizer'])
        frozen = {robot: {key: value.clone() for key, value in actors[robot].state_dict().items()} for robot in (0, 1)}

        def exact_kl():
            return sum(kl_divergence(Normal(old_mean[:, local], old_std[:, local]),
                s.o.distribution(actors[robot], obs[:, local], limits[:, local])).sum(-1)*active[:, local]
                for local, robot in enumerate(s.o.TEAMS[1])).mean()

        def search_error():
            with torch.no_grad():
                return float(s.search_loss(actors, obs, limits, active, targets, labelled))

        before = search_error()
        records = []
        for start in range(0, len(rows), 256):
            idx = order[start:start+256]
            if not bool(labelled[idx].any()):
                continue
            loss = lambda: s.search_loss(actors, obs[idx], limits[idx], active[idx], targets[idx], labelled[idx])
            if guarded:
                record = accepted_actor_step(parameters, optimizer, loss, exact_kl)
                records.append(record)
                if not record['accepted']:
                    break
            else:
                optimizer.zero_grad(set_to_none=True)
                loss().backward()
                torch.nn.utils.clip_grad_norm_(parameters, .5, error_if_nonfinite=True)
                optimizer.step()
                with torch.no_grad():
                    records.append(dict(accepted=True, final_kl=float(exact_kl())))
        unchanged = all(torch.equal(value, actors[robot].state_dict()[key])
                        for robot, weights in frozen.items() for key, value in weights.items())
        with torch.no_grad():
            final = float(exact_kl())
        if not unchanged or (guarded and any(r['final_kl'] > .02 for r in records)):
            raise RuntimeError('Frozen actors or hard acceptance invariant failed')
        outputs['guarded' if guarded else 'epoch_end_only'] = dict(
            search_error_before=before, search_error_after=search_error(), final_kl=final,
            attempted_minibatches=len(records), accepted_steps=sum(r['accepted'] for r in records),
            backtracked_trials=sum(max(0, len(r.get('attempts', []))-1) for r in records),
            evader_actors_unchanged=unchanged, steps=records)
    result = dict(scope='Engineering replay of first saved S batch, one deterministic minibatch epoch; not a training candidate',
        new_environment_episodes=0, new_teacher_queries=0, exported_models=0,
        samples=len(rows), teacher_labels=len(labels), minibatch=256, actor_lr=3e-5, maximum_kl=.02,
        source_sha256=s.o.sha(manifest['tasks'][0]['snapshot']),
        complete_initial_sha256=s.o.sha(args.source/'initial/source.pt'),
        manifest_sha256=s.o.sha(manifest_path), teacher_manifest_sha256=s.o.sha(teacher_path),
        source_experiment_sha256=s.o.sha(args.source/'experiment.json'),
        implementation={str(Path(__file__)): s.o.sha(__file__),
                        str(s.o.CODE/'learning/kl_guard.py'): s.o.sha(s.o.CODE/'learning/kl_guard.py')},
        outcomes=outputs, no_deployment_performance_claim=True)
    s.o.write_json(args.output, result)
    for name, output in outputs.items():
        print(name, {key: value for key, value in output.items() if key != 'steps'})
    return result


if __name__ == '__main__':
    main()
