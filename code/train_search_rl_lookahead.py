"""Overlap independent teams' PPO rollouts without changing batch/update order.

Same CLI as train_search_rl.py. A one-batch lookahead consumes spare workers
while the current team waits for search opponents. Each team is sampled with
its unchanged actor, and only committed batches advance the saved RNG stream.
Speculation is disabled when a current update changes the opponent pool.
"""
import hashlib
import json
from pathlib import Path
import shutil
import sys
import time

import numpy as np
import torch
import train_search_rl as trainer
from local_observation import TEAMS


class LookaheadRollouts:
    def __init__(self, output):
        self.output = Path(output)
        self.pending = None
        self.signatures = {}
        self.fallback = trainer.run_tasks

    def model_signature(self, payload, robots):
        key = (hashlib.sha256(payload).digest(), tuple(robots))
        if key not in self.signatures:
            state = trainer.unwire_weights(payload)
            digest = hashlib.sha256()
            digest.update(str(state['metadata'].get('without_belief', False)).encode())
            for robot in robots:
                for name, value in sorted(state['actors'][robot].items()):
                    array = value.detach().cpu().numpy()
                    digest.update(name.encode()); digest.update(str(array.shape).encode()); digest.update(array.tobytes())
            if len(self.signatures) >= 32:
                self.signatures.clear()
            self.signatures[key] = digest.hexdigest()
        return self.signatures[key]

    def signature(self, tasks):
        return [(tuple(sorted((key, value) for key, value in task.items() if key not in ('bundle','opponent'))),
                 self.model_signature(task['bundle'], TEAMS[task['role']]),
                 self.model_signature(task['opponent'], range(4))) for task in tasks]

    def predict_next(self, tasks):
        """Replay a private opponent-selector copy; never touch live RNGs."""
        path = self.output/'ppo_latest.pt'
        if not path.exists() or any(t.get('kind') != 'ppo' for t in tasks):
            return None
        saved = torch.load(path, map_location='cpu', weights_only=False)
        config = saved['config']; counters = saved['counters'].copy()
        role = tasks[0]['role']; count = len(tasks)
        expected = [trainer.episode_seed(config['seed'], 2, role, counters[role]+offset) for offset in range(count)]
        if [t['seed'] for t in tasks] != expected:
            return None
        before = counters[role]; counters[role] += count
        # The current role will add a newly trained opponent to the pool.
        # Its weights do not exist yet, so a future batch cannot be predicted.
        if before//500 != counters[role]//500:
            return None
        next_role = 0 if counters[0] <= counters[1] else 1
        if next_role == role or counters[next_role] >= config['episodes']:
            return None
        selector = np.random.default_rng(0)
        selector.bit_generator.state = saved['selector_rng']
        history = saved['history']
        for task in tasks:
            draw = selector.random()
            if .45 <= draw < .9 and history:
                selector.integers(len(history))
            if (draw >= .9) != task.get('search_opponent', False):
                return None
        count = min(config['episodes_per_update'], config['episodes']-counters[next_role])
        future = []
        for offset in range(count):
            index = counters[next_role]+offset; draw = selector.random()
            opposing = history[int(selector.integers(len(history)))] if .45 <= draw < .9 and history else saved['anchor']
            future.append(dict(seed=trainer.episode_seed(config['seed'], 2, next_role, index),
                               role=next_role, progress=index/max(config['episodes'], 1), kind='ppo',
                               bundle=tasks[0]['bundle'], opponent=trainer.wire_weights(opposing),
                               search_opponent=draw >= .9, opponent_budget=config['opponent_budget'],
                               budget=config['budget'], depth=config['depth']))
        return future

    def __call__(self, pool, tasks):
        if pool is None or not tasks or tasks[0].get('kind') != 'ppo':
            return self.fallback(pool, tasks)
        started = time.perf_counter()
        current_signature = self.signature(tasks)
        reused = self.pending is not None and self.pending['signature'] == current_signature
        if reused:
            current = self.pending['jobs']
        else:
            if self.pending is not None:
                # Safe fallback if any prediction disagrees with the actual
                # tasks. Speculative samples are never silently substituted.
                print('LOOKAHEAD discarded mismatched speculative batch', flush=True)
            current = [pool.apply_async(trainer.rollout_task, (task,)) for task in tasks]
        self.pending = None
        future = self.predict_next(tasks)
        if future:
            self.pending = dict(signature=self.signature(future),
                                jobs=[pool.apply_async(trainer.rollout_task, (task,)) for task in future])
        results = [job.get() for job in current]
        record = dict(role=tasks[0]['role'], episodes=len(tasks), reused_prefetch=reused,
                      prefetched_episodes=len(future) if future else 0,
                      wait_seconds=time.perf_counter()-started)
        with (self.output/'lookahead_metrics.jsonl').open('a') as log:
            log.write(json.dumps(record)+'\n')
        print('LOOKAHEAD '+json.dumps(record), flush=True)
        return results


def fork_dispatch_only(args, original_fork, base_hash):
    original_fork(args)
    path = Path(args.output)/'ppo_latest.pt'
    saved = torch.load(path, map_location='cpu', weights_only=False)
    if not saved.get('needs_revalidation'):
        return
    source_config = saved['fork']['source_config']
    source_base = source_config.get('base_implementation_hash', source_config.get('implementation_hash'))
    if source_base != base_hash:
        return
    # Dispatch-only changes leave validation semantics unchanged. Retain the
    # validated best models and training RNG; reinitializing validation models
    # here would otherwise consume RNG and defeat exact continuation.
    source = Path(args.fork_from).resolve().parent
    for pattern in ('validation_role*_ep*.json', 'fork_previous_best_role*.json'):
        for candidate in source.glob(pattern):
            shutil.copy2(candidate, Path(args.output)/candidate.name)
    saved['needs_revalidation'] = False
    saved['fork']['validation_reused_reason'] = 'Identical base implementation; only independent-role rollout dispatch changed.'
    trainer.write_json(Path(args.output)/'fork.json', saved['fork'])
    trainer.atomic_torch_save(saved, path)


def main():
    # Keep the established parser, PPO implementation, optimizer/checkpoint
    # code and pipeline. Only rollout dispatch and the recorded fingerprint
    # change. Spawned workers import this file without invoking main().
    import argparse
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--output', default=str(trainer.ROOT/'current/search_rl_v1/seed0'))
    options, _ = parser.parse_known_args()
    original_hash = trainer.implementation_hash
    original_configuration = trainer.configuration
    source = Path(__file__).read_bytes()
    base_hash = original_hash()
    backend_hash = hashlib.sha256(base_hash.encode()+source).hexdigest()
    trainer.implementation_hash = lambda: backend_hash
    def configuration(args):
        return dict(original_configuration(args), rollout_backend='paired_role_lookahead_v1', base_implementation_hash=base_hash)
    trainer.configuration = configuration
    original_fork = trainer.fork_ppo
    trainer.fork_ppo = lambda args: fork_dispatch_only(args, original_fork, base_hash)
    trainer.run_tasks = LookaheadRollouts(Path(options.output).resolve())
    trainer.main()


if __name__ == '__main__':
    main()
