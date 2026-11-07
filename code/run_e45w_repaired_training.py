"""Resume failed W learners with an explicit estimator revision and cost cap.

Model/critic/Adam and learner RNG resume. Environments start fresh registered
streams; failed partial rollouts are not reused. All spent sampling, including
lost progress, is charged to the original per-seed budget conservatively.
"""
import os
for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMBA_NUM_THREADS'):
    os.environ[key] = '1'
import argparse
import copy
from concurrent.futures import ProcessPoolExecutor
import json
import multiprocessing as mp
from pathlib import Path
import random
import shutil
import traceback

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.logger import configure
from stable_baselines3.common.vec_env import SubprocVecEnv
from learning.e45w_support_recovery import install, REVISION
if __name__ in ('__main__', '__mp_main__'):
    install()
import run_e45w_joint_training as original
from run_e45w_joint_training import (read, write, tensor_hash, TrainingAudit,
    SeedStreamEnv, _evaluate_job, score, preserve_rng, FORMAL_N_ENVS, FORMAL_N_STEPS)

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'current/e45w_joint_ppo_and_planning_20261103'
BLOCK = FORMAL_N_ENVS * FORMAL_N_STEPS
CAP = BLOCK * original.FORMAL_ROLLOUTS


def budget(source_status, checkpoint_step):
    # A worker could advance once more before the failed vector recv. Charge
    # a whole rollout containing that tick, rather than undercounting it.
    old = source_status.get('repair_budget', {})
    previous_lost = int(old.get('charged_lost_steps', 0))
    reported = int(source_status['counts']['team_transitions'])
    charged_attempt = ((reported + FORMAL_N_ENVS + BLOCK - 1) // BLOCK) * BLOCK
    lost = previous_lost + charged_attempt - checkpoint_step
    target = ((CAP - lost) // BLOCK) * BLOCK
    if target <= checkpoint_step:
        raise RuntimeError('Repair has no remaining authorized sampling budget')
    return dict(cap=CAP, checkpoint_step=checkpoint_step,
        source_reported_step=reported, charged_attempt_steps=charged_attempt,
        charged_lost_steps=lost, target_logical_steps=target,
        maximum_lifetime_team_steps=lost + target,
        includes_crash_tick_and_lost_partial_rollout=True)


def make_env(env_args, rank, seed, cores, checkpoint_step):
    def create():
        os.sched_setaffinity(0, {cores[rank % len(cores)]})
        torch.set_num_threads(1)
        initial = int(np.random.SeedSequence([seed, rank, checkpoint_step, 1250625]).generate_state(1)[0])
        return SeedStreamEnv(env_args, initial)
    return create


def restore_rng(states, device):
    random.setstate(states['python']); np.random.set_state(states['numpy'])
    torch.set_rng_state(states['torch'].cpu())
    if states.get('cuda') is not None:
        torch.cuda.set_rng_state(states['cuda']['state'].cpu(), device)


def init_eval(paths, evidence):
    install(evidence)
    original.init_evaluation_worker(paths)


class ResumeAudit(TrainingAudit):
    def __init__(self, *args, source, resume_audit, budget_info, **kwargs):
        super().__init__(*args, **kwargs)
        self.source = source; self.resume_audit = resume_audit; self.budget_info = budget_info
        self.counts = copy.deepcopy(resume_audit['counts'])

    def publish(self, status, **extra):
        return super().publish(status, repair_revision=REVISION,
            repair_budget=self.budget_info, resume_source=str(self.source),
            environment_exact_resume=False, **extra)

    def _on_training_start(self):
        # Revalidate every retained candidate under the new estimator. This
        # does not alter the loaded optimizer or select using final test data.
        original_audit = read(self.source / 'initialization_audit.json')
        self.initial_actor_hash = original_audit['actor_sha256']
        self.initial_critic_hash = original_audit['critic_sha256']
        write(self.directory / 'initialization_audit.json', original_audit)
        self.publish('revalidating_saved_candidates')
        for checkpoint in sorted((self.source / 'checkpoints').glob('*/model.zip')):
            step = int(checkpoint.parent.name)
            if step > self.model.num_timesteps:
                continue
            state = torch.load(checkpoint.parent / 'actor.pt', map_location='cpu', weights_only=False)
            with preserve_rng():
                stochastic = list(self.eval_pool.map(_evaluate_job, [(state, s) for s in self.eval_specs]))
                deterministic = list(self.eval_pool.map(_evaluate_job, [(state, s) for s in self.det_specs]))
            primary, diagnostic = score(stochastic), score(deterministic)
            self.counts['validation_episodes'] += len(stochastic) + len(deterministic)
            self.counts['validation_environment_steps'] += sum(r['decisions'] for r in stochastic + deterministic)
            record = dict(team_transitions=step, primary_mode='stochastic', stochastic=primary,
                deterministic_diagnostic=diagnostic, revalidated_saved_candidate=True,
                actor_sha256=tensor_hash(state), counts=copy.deepcopy(self.counts))
            with (self.directory / 'validation.jsonl').open('a') as f:
                f.write(json.dumps(record)+'\n')
            write(self.directory / 'validation_episodes' / f'{step:08d}.json',
                  dict(stochastic=stochastic, deterministic=deterministic))
            if self.best is None or primary['utility'] > self.best['utility']:
                self.best = dict(team_transitions=step, utility=primary['utility'],
                    double_capture_rate=primary['double_capture_rate'], mode='stochastic')
                shutil.copy2(checkpoint.parent / 'actor.pt', self.directory / 'selected_actor.pt')
                write(self.directory / 'selection.json', self.best | {'checkpoint':str(checkpoint)})
            if step == 0:
                shutil.copy2(checkpoint.parent / 'actor.pt', self.directory / 'initial_actor.pt')
            self.latest_evaluation = record; self.last_evaluated = step
            print(f'REVALIDATED {self.args.arm}{self.args.seed} teams={step} double={primary["double_captures"]}/{primary["episodes"]}', flush=True)
            self.publish('revalidating_saved_candidates')
        self.publish('training')


def run(args):
    manifest = read(original.MANIFEST); original.validate_manifest(manifest)
    source = Path(args.source).resolve()
    status = read(source / 'status.json')
    if status['status'] != 'failed':
        raise RuntimeError('Only a failed learner can be resumed here')
    checkpoints = sorted((source / 'checkpoints').glob('*/model.zip'))
    checkpoint = checkpoints[-1]; base = int(checkpoint.parent.name)
    budget_info = budget(status, base)
    directory = OUT / 'repaired_training' / args.run_name
    directory.mkdir(parents=True, exist_ok=False)
    cores = [int(x) for x in args.cpus.split(',')]
    os.sched_setaffinity(0, set(cores)); torch.set_num_threads(1)
    torch.cuda.set_device(torch.device(args.device))
    args.smoke = False; args.arm = status['arm']; args.seed = status['seed']
    install(directory / 'belief_failures')
    write(directory / 'requested_config.json', read(source/'requested_config.json') | dict(
        repair_revision=REVISION, resume_checkpoint=str(checkpoint),
        model_and_adam_restored=True, environments_exact_resume=False,
        fresh_env_seed_formula='SeedSequence([seed,rank,checkpoint_step,1250625])',
        eval_workers=args.eval_workers, repair_budget=budget_info))
    envs = pool = model = audit = None
    try:
        env_args = dict(case_paths=[r['path'] for r in manifest['train_entries']],
            opponents=['fixed_D_kernel','rootpuct'], root_simulations=3000,
            belief_evidence_dir=str(directory/'belief_failures'))
        envs = SubprocVecEnv([make_env(env_args, i, args.seed, cores, base) for i in range(FORMAL_N_ENVS)], start_method='spawn')
        pool = ProcessPoolExecutor(max_workers=args.eval_workers, mp_context=mp.get_context('spawn'),
            initializer=init_eval,
            initargs=([r['path'] for r in manifest['validation_entries']], str(directory/'evaluation_belief_failures')))
        model = PPO.load(checkpoint, env=envs, device=args.device)
        model.set_logger(configure(str(directory/'sb3_logs'), ['csv','json']))
        saved = torch.load(checkpoint.parent/'rng_audit.pt', map_location='cpu', weights_only=False)
        state = torch.load(checkpoint.parent/'actor.pt', map_location='cpu', weights_only=False)
        assert tensor_hash(model.policy.actor_state_numpy()) == tensor_hash(state)
        assert model.num_timesteps == base
        # Verify every Adam tensor and scalar against the actual zip payload.
        from stable_baselines3.common.save_util import load_from_zip_file
        _, parameters, _ = load_from_zip_file(checkpoint, device=args.device)
        actual = model.policy.optimizer.state_dict(); expected = parameters['policy.optimizer']
        for key, value in parameters['policy'].items():
            assert torch.equal(model.policy.state_dict()[key].cpu(), value.cpu())
        assert actual['param_groups'] == expected['param_groups']
        for key, fields in expected['state'].items():
            for field, value in fields.items():
                if torch.is_tensor(value):
                    assert torch.equal(actual['state'][key][field].cpu(), value.cpu())
                else: assert actual['state'][key][field] == value
        write(directory/'resume_integrity.json', dict(actor_equal=True, adam_all_tensors_equal=True,
            model_num_timesteps=base, checkpoint_sha256=original.sha(checkpoint),
            critic_restored_from_model_zip=True, environment_exact_resume=False))
        restore_rng(saved['rng'], args.device)
        audit = ResumeAudit(directory, manifest, args, pool, manifest['validation_entries'],
            source=source, resume_audit=saved, budget_info=budget_info)
        model.learn(total_timesteps=budget_info['target_logical_steps']-base,
                    callback=audit, reset_num_timesteps=False, log_interval=1)
        assert model.num_timesteps == budget_info['target_logical_steps']
        assert audit.counts['rollouts_collected'] == model.num_timesteps // BLOCK
        terminal = audit.evaluate(model.num_timesteps)
        torch.save(model.policy.actor_state_numpy(), directory/'terminal_actor.pt')
        write(directory/'terminal_checkpoint.json', dict(model=str(terminal/'model.zip'), team_transitions=model.num_timesteps))
        original.validate_manifest(manifest)
        audit.publish('complete', training_budget_audit_passed=True)
    except BaseException as exc:
        write(directory/'failure.json', dict(error=repr(exc), traceback=traceback.format_exc()))
        if audit is not None: audit.publish('failed', error=repr(exc))
        raise
    finally:
        # A dead vector worker must not deadlock close() while waiting for a
        # reply that can never arrive. Terminate the failed collection tree.
        if envs is not None:
            if (directory/'failure.json').exists():
                for process in envs.processes:
                    if process.is_alive(): process.terminate()
                for process in envs.processes: process.join(timeout=2)
            else: envs.close()
        if pool is not None: pool.shutdown(wait=True, cancel_futures=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', required=True)
    parser.add_argument('--run-name', required=True)
    parser.add_argument('--device', required=True)
    parser.add_argument('--cpus', required=True)
    parser.add_argument('--eval-workers', type=int, default=8)
    run(parser.parse_args())
