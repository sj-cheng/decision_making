"""PPO-only continuation of the original 270-input l45 policies.

This experiment deliberately does not use LocalController, TeamBelief, the
293-input actors, swept collision, or a new search implementation. Original
environment and C_PUCT files are read-only dependencies. The training critic
is never exported to search.
"""
import os
for _key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[_key] = '1'
os.environ.setdefault('MPLCONFIGDIR', '/tmp/l45_ppo_mpl')

import argparse
import contextlib
import copy
import fcntl
import hashlib
import io
import json
import multiprocessing as mp
from pathlib import Path
import random
import shutil
import sys
import time

import numpy as np
import torch
from torch import nn
from torch.distributions import Normal, kl_divergence

from problems.problem import get_problem
from learning.gaussian_policy_network import GaussianPolicyNetwork
from benchmark_baselines import (build_solver, configure_problem, problem_metadata,
                                 run_episode)
from solvers.baselines import RoleMaskedSolver, TeamSolver

CODE = Path(__file__).resolve().parent
ROOT = CODE.parent
DEFAULT_OUTPUT = ROOT / 'current/l45_ppo_control_20261022'
SOURCE_MODELS = ROOT / 'current/memory_observation_v3/models'
TEAMS = ((0, 1), (2, 3))
OPPONENTS = ('Policy Only', 'Search + Learning', 'Search Only')
MODEL_NAMES = tuple(f'model_policy_l45_i{i}.pt' for i in range(4)) + ('model_value_l45.pt',)
_WEIGHT_CACHE = {}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def json_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def write_json(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    os.replace(temporary, path)


def save_torch(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    torch.save(value, temporary); os.replace(temporary, path)


def read_json(path):
    return json.loads(Path(path).read_text())


def load_torch(path):
    return torch.load(path, map_location='cpu', weights_only=False)


def scenario_seed(phase, seed, index):
    return int(np.random.SeedSequence([20261022, 450270, phase, seed, index]).generate_state(1)[0])


def original_problem(layout='four'):
    if layout == 'four':
        p = get_problem('example8')
        p.use_minco_dynamics = False  # exactly train.py's original setting
    elif layout == 'none':
        p = configure_problem('none')
    else:
        raise ValueError('Only the original four-obstacle and no-obstacle maps are allowed')
    if p.policy_encoding_dim != 270 or p.state_dim != 53 or p.swept_collision:
        raise ValueError('Original l45 environment contract has changed')
    return p


def environment_description(layout):
    p = original_problem(layout)
    return dict(**problem_metadata(p, layout), state_dim=p.state_dim,
                policy_encoding_dim=p.policy_encoding_dim, tf=p.tf, dt=p.dt,
                gamma=p.gamma, desired_distance=p.desired_distance,
                init_min_dist=p.init_min_dist, swept_collision=p.swept_collision,
                evader_speed_lim_range=p.evader_speed_lim_range,
                pursuer_speed_lim_range=p.pursuer_speed_lim_range,
                rewards={key: getattr(p, key) for key in
                         ('evader_progress_reward_weight', 'pursuer_progress_reward_weight',
                          'evader_distance_reward_weight', 'cross_team_distance_reward_weight')})


def make_scenario(seed, layout='four'):
    previous = np.random.get_state()
    try:
        np.random.seed(seed); p = original_problem(layout); state = p.initialize()
        spec = dict(seed=int(seed), layout=layout, initial_state=state.tolist(),
                    action_lims=p.action_lims.tolist(), control_lims=p.control_lims.tolist(),
                    evader_speed=float(p.current_evader_speed_lim),
                    pursuer_speed=float(p.current_pursuer_speed_lim),
                    search_seeds=[scenario_seed(19, seed, role) for role in range(2)])
        spec['episode_hash'] = json_hash(spec)
        return spec
    finally:
        np.random.set_state(previous)


def restore_scenario(spec):
    check = {key: value for key, value in spec.items() if key != 'episode_hash'}
    if json_hash(check) != spec['episode_hash']:
        raise ValueError('Scenario fingerprint mismatch')
    p = original_problem(spec['layout'])
    p.action_lims = np.asarray(spec['action_lims'], float)
    p.control_lims = np.asarray(spec['control_lims'], float)
    p.current_evader_speed_lim = spec['evader_speed']
    p.current_pursuer_speed_lim = spec['pursuer_speed']
    p.clear_closed_loop_cache()
    return p, np.asarray(spec['initial_state'], float).copy()


def model_fingerprint(directory):
    return {name: sha(Path(directory) / name) for name in MODEL_NAMES}


def implementation_files():
    names = ['train_l45_ppo.py', 'benchmark_baselines.py', 'problems/problem.py',
             'problems/example8.py', 'learning/gaussian_policy_network.py',
             'learning/deterministic_value_network.py', 'learning/feedforward.py',
             'learning/oracles.py', 'solvers/c_puct.py', 'solvers/policy_solver.py',
             'solvers/baselines.py', 'solvers/solver.py', 'util.py',
             'cpp/solvers/puct_v1.hpp', 'cpp/problems/example8.hpp', 'cpp/problems/problem.hpp',
             'cpp/learning/gaussian_policy_network.hpp', 'cpp/bindings.cpp']
    return {name: sha(CODE / name) for name in names} | {
        str(path.relative_to(CODE)): sha(path) for path in (CODE / 'cpp/build').glob('*.so')}


def load_actors(problem, directory=None, states=None):
    actors = []
    for robot in range(4):
        actor = GaussianPolicyNetwork(problem, robot)
        weights = states[robot] if states is not None else cached_weights(Path(directory) / MODEL_NAMES[robot])
        actor.load_state_dict(weights, strict=True)
        if directory is not None:
            actor.path = str(Path(directory) / MODEL_NAMES[robot])
        actors.append(actor)
    return actors


def cached_weights(path):
    path = Path(path)
    key = (str(path), path.stat().st_mtime_ns, path.stat().st_size)
    if key not in _WEIGHT_CACHE:
        if len(_WEIGHT_CACHE) > 24:
            _WEIGHT_CACHE.clear()
        _WEIGHT_CACHE[key] = load_torch(path)
    return _WEIGHT_CACHE[key]


def actor_states(actors):
    return [copy.deepcopy(actor.state_dict()) for actor in actors]


def distribution(actor, observations, limits):
    single = observations.ndim == 1
    mean, logvar = actor(observations.unsqueeze(0) if single else observations, training=True)
    if single:
        mean, logvar = mean.squeeze(0), logvar.squeeze(0)
    center = .5 * (limits[..., 1] + limits[..., 0])
    half_range = .5 * (limits[..., 1] - limits[..., 0])
    physical_mean = center + half_range * torch.tanh(mean)
    std = torch.exp(.5 * logvar.clamp(-8., 2.))
    if not torch.isfinite(physical_mean).all() or not torch.isfinite(std).all():
        raise FloatingPointError('Nonfinite policy distribution')
    return Normal(physical_mean, std)


def critic_features(problem, state):
    low, high = problem.state_lims[:, 0], problem.state_lims[:, 1]
    physical = 2. * (np.asarray(state).reshape(-1) - low) / np.maximum(high - low, 1e-6) - 1.
    return np.r_[physical, problem.action_lims.reshape(-1) / 1.2].astype(np.float32)


class TrainingCritic(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(69, 128), nn.Tanh(), nn.Linear(128, 128),
                                 nn.Tanh(), nn.Linear(128, 1))

    def forward(self, value):
        return self.net(value).squeeze(-1)


def advantages(rewards, values, dones, gamma=1., lam=.95):
    result = np.zeros(len(rewards), np.float32); carry = 0.
    for i in reversed(range(len(rewards))):
        live = 1. - float(dones[i]); following = values[i+1] if i+1 < len(values) else 0.
        delta = rewards[i] + gamma * following * live - values[i]
        carry = delta + gamma * lam * live * carry; result[i] = carry
    return result, result + np.asarray(values, np.float32)


def joint_log_prob(actors, observations, raw, limits, active):
    return sum(distribution(actors[robot], observations[:, local], limits[:, local]).log_prob(
        raw[:, local]).sum(-1) * active[:, local] for local, robot in enumerate(TEAMS[1]))


def make_solver(problem, models, method, role, settings, seed):
    if settings['search_depth'] != 100:
        raise ValueError('This experiment preserves the original search depth 100')
    budget = settings['search_only_budget'] if method == 'Search Only' else settings['search_budget']
    with contextlib.redirect_stdout(io.StringIO()):
        solver = build_solver(problem, method, str(ROOT), budget,
                              model_dir=str(models), model_iteration=45,
                              beta_policy=.5, beta_value=.1)
    if method != 'Policy Only':
        solver.solver_wrapper.set_seed(int(seed))
    return RoleMaskedSolver(solver, TEAMS[role])


def rollout(task):
    """On-policy actor actions, original environment, frozen original opponent."""
    torch.set_num_threads(1)
    p, state = restore_scenario(task['spec'])
    checkpoint = cached_weights(task['snapshot'])
    if checkpoint['episode'] != task['snapshot_episode']:
        raise ValueError('Sampling snapshot changed while rollouts were running')
    actors = load_actors(p, states=checkpoint['actors'])
    opponent = make_solver(p, task['baseline'], task['opponent'], 0, task['settings'],
                           task['spec']['search_seeds'][0])
    torch.manual_seed(task['sampling_seed']); np.random.seed(task['sampling_seed']); random.seed(task['sampling_seed'])
    rows = []; captures = 0; first = None; second = None; rewards = []
    while not p.is_terminal(state) and len(rows) < int(p.tf / p.dt):
        obs = np.array([p.policy_encoding(state, robot).reshape(-1) for robot in TEAMS[1]], np.float32)
        limits = np.array([p.action_lims[p.action_idxs[robot]] for robot in TEAMS[1]], np.float32)
        active = np.array([p.is_active(state, robot) for robot in TEAMS[1]], np.float32)
        action = opponent.policy(p, state)
        raw = np.zeros((2, 2), np.float32); means = raw.copy(); stds = np.ones_like(raw); logp = 0.
        for local, robot in enumerate(TEAMS[1]):
            if active[local]:
                with torch.no_grad():
                    dist = distribution(actors[robot], torch.from_numpy(obs[local]), torch.from_numpy(limits[local]))
                    sampled = dist.sample(); logp += float(dist.log_prob(sampled).sum())
                raw[local] = sampled.numpy(); means[local] = dist.mean.numpy(); stds[local] = dist.stddev.numpy()
                action[p.action_idxs[robot], 0] = np.clip(raw[local], limits[local, :, 0], limits[local, :, 1])
        # Exactly the original benchmark ordering, including original reward().
        reward = p.reward(state, action); nxt = p.step(state, action, p.dt)
        done = bool(p.is_terminal(nxt)); team_reward = float(np.mean(reward[list(TEAMS[1]), 0]))
        rows.append(dict(obs=obs, limits=limits, active=active, raw=raw, old_logp=logp,
                         old_mean=means, old_std=stds, critic=critic_features(p, state),
                         reward=team_reward, done=done))
        rewards.append(team_reward)
        new_captures = 2 - p.active_evader_count(nxt)
        if new_captures > captures:
            if first is None: first = float(nxt[p.time_idx, 0])
            if new_captures == 2: second = float(nxt[p.time_idx, 0])
        captures = new_captures; state = nxt
    if rows and not rows[-1]['done']:
        raise RuntimeError('Unexpected nonterminal truncation; zero bootstrap would be invalid')
    illegal = violations(p, state)
    return dict(rows=rows, episode_hash=task['spec']['episode_hash'], opponent=task['opponent'],
                spec=task['spec'], snapshot_episode=checkpoint['episode'],
                stats=dict(steps=len(rows), legal_double_capture=int(captures == 2 and not any(illegal)),
                           pursuer_illegal=int(illegal[1]), evader_illegal=int(illegal[0]),
                           first_capture_time=first, capture_time=second, environment_return=sum(rewards)))


def violations(problem, state):
    return [any(problem.check_obstacle_collision(state, robot) or
                np.any(state[problem.state_idxs[robot], 0] < problem.state_lims[problem.state_idxs[robot], 0]) or
                np.any(state[problem.state_idxs[robot], 0] > problem.state_lims[problem.state_idxs[robot], 1])
                for robot in robots) for robots in TEAMS]


def update_ppo(actors, critic, actor_opt, critic_opt, episodes, config):
    rows = [row for episode in episodes for row in episode['rows']]
    if not rows: raise RuntimeError('No on-policy samples')
    tensor = lambda key: torch.tensor(np.asarray([row[key] for row in rows]), dtype=torch.float32)
    obs, limits, raw, active = (tensor(key) for key in ('obs', 'limits', 'raw', 'active'))
    x, old_logp, old_mean, old_std = (tensor(key) for key in ('critic', 'old_logp', 'old_mean', 'old_std'))
    if not all(torch.isfinite(t).all() for t in (obs, limits, raw, active, x, old_logp, old_mean, old_std)):
        raise FloatingPointError('Nonfinite on-policy samples')
    with torch.no_grad():
        recomputed = joint_log_prob(actors, obs, raw, limits, active)
        logp_error = float((recomputed-old_logp).abs().max())
        predicted = critic(x).numpy()
    if not np.isfinite(logp_error) or logp_error > 3e-4:
        raise RuntimeError(f'On-policy logprob mismatch: {logp_error}')
    advs = []; targets = []; offset = 0
    for episode in episodes:
        n = len(episode['rows']); part = episode['rows']
        adv, target = advantages([r['reward'] for r in part], predicted[offset:offset+n], [r['done'] for r in part])
        advs.extend(adv); targets.extend(target); offset += n
    adv = torch.tensor(np.asarray(advs), dtype=torch.float32)
    adv = (adv-adv.mean()) / adv.std(unbiased=False).clamp_min(1e-6)
    target = torch.tensor(np.asarray(targets), dtype=torch.float32)
    if not torch.isfinite(adv).all() or not torch.isfinite(target).all():
        raise FloatingPointError('Nonfinite GAE or returns')
    actor_parameters = [p for robot in TEAMS[1] for p in actors[robot].parameters()]
    actor_stopped = False; losses = []; value_losses = []; exact_kls = []; actor_epochs = 0
    for epoch in range(config['ppo_epochs']):
        order = torch.randperm(len(rows))
        for start in range(0, len(rows), config['minibatch']):
            idx = order[start:start+config['minibatch']]
            if not actor_stopped:
                logp = joint_log_prob(actors, obs[idx], raw[idx], limits[idx], active[idx])
                ratio = (logp-old_logp[idx]).exp()
                entropy = sum((distribution(actors[robot], obs[idx, local], limits[idx, local]).entropy().sum(-1)
                               * active[idx, local]).sum() / active[idx, local].sum().clamp_min(1.)
                              for local, robot in enumerate(TEAMS[1])) * .5
                loss = -torch.minimum(ratio*adv[idx], ratio.clamp(1-config['clip'], 1+config['clip'])*adv[idx]).mean()
                loss = loss - config['entropy'] * entropy
                if not torch.isfinite(loss): raise FloatingPointError('Nonfinite actor loss')
                actor_opt.zero_grad(); loss.backward()
                nn.utils.clip_grad_norm_(actor_parameters, .5, error_if_nonfinite=True); actor_opt.step()
                losses.append(float(loss.detach()))
            value_loss = .5 * (critic(x[idx])-target[idx]).square().mean()
            if not torch.isfinite(value_loss): raise FloatingPointError('Nonfinite critic loss')
            critic_opt.zero_grad(); value_loss.backward()
            nn.utils.clip_grad_norm_(critic.parameters(), .5, error_if_nonfinite=True); critic_opt.step()
            value_losses.append(float(value_loss.detach()))
        if not actor_stopped:
            actor_epochs += 1
            with torch.no_grad():
                exact = sum(kl_divergence(Normal(old_mean[:, local], old_std[:, local]),
                            distribution(actors[robot], obs[:, local], limits[:, local])).sum(-1) * active[:, local]
                            for local, robot in enumerate(TEAMS[1])).mean()
            exact_kls.append(float(exact))
            if not np.isfinite(exact_kls[-1]): raise FloatingPointError('Nonfinite exact KL')
            actor_stopped = exact_kls[-1] > config['target_kl']
    with torch.no_grad():
        ratio = (joint_log_prob(actors, obs, raw, limits, active)-old_logp).exp()
    return dict(samples=len(rows), actor_loss=float(np.mean(losses)), critic_loss=float(np.mean(value_losses)),
                exact_kl_by_epoch=exact_kls, actor_epochs=actor_epochs, kl_stopped=actor_stopped,
                clip_fraction=float(((ratio-1).abs() > config['clip']).float().mean()),
                old_logp_max_error=logp_error, mean_episode_reward=float(np.mean([e['stats']['environment_return'] for e in episodes])))


def rng_state():
    return dict(python=random.getstate(), numpy=np.random.get_state(), torch=torch.get_rng_state())


def restore_rng(value):
    random.setstate(value['python']); np.random.set_state(value['numpy']); torch.set_rng_state(value['torch'])


def export_models(states, target, baseline):
    target = Path(target); target.mkdir(parents=True, exist_ok=True)
    for robot, name in enumerate(MODEL_NAMES[:4]):
        if robot in TEAMS[0]:
            shutil.copyfile(Path(baseline) / name, target / name)
        else:
            save_torch(target / name, states[robot])
    shutil.copyfile(Path(baseline) / MODEL_NAMES[-1], target / MODEL_NAMES[-1])
    fingerprint = model_fingerprint(target)
    source = model_fingerprint(baseline)
    for name in (*MODEL_NAMES[:2], MODEL_NAMES[-1]):
        if fingerprint[name] != source[name]: raise AssertionError(f'Frozen model changed: {name}')
    write_json(target/'export.json', dict(models=fingerprint, schema='original_l45_270', critic_exported=False))
    return fingerprint


def evaluate_one(task):
    torch.set_num_threads(1)
    p, state = restore_scenario(task['spec']); role = task['role']
    method = 'Policy Only' if task['mode'] == 'policy' else 'Search + Learning'
    own = make_solver(p, task['models'], method, role, task['settings'], task['spec']['search_seeds'][role])
    other = make_solver(p, task['baseline'], task['opponent'], 1-role, task['settings'], task['spec']['search_seeds'][1-role])
    solver = TeamSolver(own if role == 0 else other, own if role == 1 else other)
    metric = run_episode(p, solver, state)
    illegal = metric['illegal_terminal']
    metric.update(legal_double_capture=int(metric['double_capture'] and not illegal),
                  legal_survival=int(metric['capture_count'] < 2 and metric['survival_time'] >= p.tf and not illegal),
                  one_capture_timeout=int(metric['capture_count'] == 1 and metric['survival_time'] >= p.tf and not illegal),
                  pursuer_illegal=int(metric['pursuer_obstacle_collision'] or metric['pursuer_out_of_bounds']),
                  evader_illegal=int(metric['evader_obstacle_collision'] or metric['evader_out_of_bounds']))
    metric['completion_cost'] = metric['capture_time'] if metric['legal_double_capture'] else float(p.tf)
    metric.update(key=task['key'], label=task['label'], training_seed=task.get('training_seed'),
                  mode=task['mode'], role=role, opponent=task['opponent'], layout=task['spec']['layout'],
                  episode_hash=task['spec']['episode_hash'], model_hash=task['model_hash'],
                  search_seeds=task['spec']['search_seeds'], timing_context=task['timing_context'])
    return metric


def summarize(rows):
    if not rows: return {}
    role = rows[0]['role']; side = 'pursuer' if role == 1 else 'evader'
    metric = 'legal_double_capture' if role == 1 else 'legal_survival'
    return dict(n=len(rows), success=sum(r[metric] for r in rows), success_rate=float(np.mean([r[metric] for r in rows])),
                own_illegal=sum(r[side+'_illegal'] for r in rows),
                opponent_illegal=sum(r[('evader' if role == 1 else 'pursuer')+'_illegal'] for r in rows),
                one_capture_timeout=sum(r['one_capture_timeout'] for r in rows),
                completion_cost=float(np.mean([r['completion_cost'] for r in rows])),
                environment_return=float(np.mean([r[side+'_return']*.5 for r in rows])),
                decision_ms=float(sum(r[side+'_latency_sum_ms'] for r in rows) /
                                  max(1, sum(r[side+'_latency_count'] for r in rows))))


def evaluation_tasks(models, label, train_seed, baseline, scenarios, settings,
                     modes=('policy',), roles=(1,), opponents=OPPONENTS, timing_context='concurrent_workers'):
    fingerprint = model_fingerprint(models); tasks = []
    for spec in scenarios:
        for mode in modes:
            for role in roles:
                for opponent in opponents:
                    task = dict(models=str(models), label=label, training_seed=train_seed, baseline=str(baseline),
                                spec=spec, settings=settings, mode=mode, role=role, opponent=opponent,
                                model_hash=fingerprint, timing_context=timing_context)
                    task['key'] = json_hash(task)
                    tasks.append(task)
    return tasks


def run_evaluation(out, name, tasks, pool, status):
    folder = Path(out)/name; folder.mkdir(parents=True, exist_ok=True)
    manifest = [dict(key=t['key'], label=t['label'], mode=t['mode'], role=t['role'], opponent=t['opponent'],
                     episode_hash=t['spec']['episode_hash'], model_hash=t['model_hash']) for t in tasks]
    if (folder/'manifest.json').exists() and read_json(folder/'manifest.json') != manifest:
        raise ValueError(f'Evaluation manifest changed: {folder}')
    write_json(folder/'manifest.json', manifest)
    missing = [t for t in tasks if not (folder/(t['key']+'.json')).exists()]
    status('evaluation', evaluation=name, completed=len(tasks)-len(missing), total=len(tasks))
    iterator = pool.imap_unordered(evaluate_one, missing, chunksize=1) if pool else map(evaluate_one, missing)
    for index, row in enumerate(iterator, 1):
        write_json(folder/(row['key']+'.json'), row)
        if index % 10 == 0 or index == len(missing):
            status('evaluation', evaluation=name, completed=len(tasks)-len(missing)+index, total=len(tasks))
    rows = [read_json(folder/(task['key']+'.json')) for task in tasks]
    for task, row in zip(tasks, rows):
        if row['key'] != task['key'] or row['model_hash'] != task['model_hash']:
            raise ValueError('Evaluation cache mismatch')
    groups = {}
    for row in rows:
        group = (row['label'], row['mode'], row['role'], row['opponent'], row['layout'])
        groups.setdefault(group, []).append(row)
    write_json(folder/'summary.json', [dict(label=g[0], mode=g[1], role=g[2], opponent=g[3], layout=g[4],
                                           **summarize(rs)) for g, rs in groups.items()])
    return rows


def selection_score(rows):
    # Each opponent has the same number of fixed scenes. Penalize invalid or
    # unfinished episodes by the full horizon, rather than their short duration.
    return (float(np.mean([r['legal_double_capture'] for r in rows])),
            -float(np.mean([r['pursuer_illegal'] for r in rows])),
            -float(np.mean([r['completion_cost'] for r in rows])))


def checkpoint_payload(actors, critic, actor_opt, critic_opt, completed, best_episode,
                       best_score, pending_validation, signature, last_metrics=None):
    return dict(actors=actor_states(actors), critic=critic.state_dict(),
                actor_optimizer=actor_opt.state_dict(), critic_optimizer=critic_opt.state_dict(),
                episode=completed, best_episode=best_episode, best_score=list(best_score),
                pending_validation=pending_validation, rng=rng_state(), signature=signature,
                last_metrics=last_metrics)


def train_seed(args, out, seed, manifest, baseline_rows, signature, pool, status):
    config = training_config(args); directory = Path(out)/f'seed{seed}'; directory.mkdir(exist_ok=True)
    baseline = Path(out)/'baseline'; p = original_problem('four')
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    actors = load_actors(p, directory=baseline); critic = TrainingCritic()
    for robot in TEAMS[0]:
        for parameter in actors[robot].parameters(): parameter.requires_grad_(False)
    actor_opt = torch.optim.Adam([v for robot in TEAMS[1] for v in actors[robot].parameters()], lr=args.lr)
    critic_opt = torch.optim.Adam(critic.parameters(), lr=args.critic_lr)
    latest = directory/'latest.pt'
    if latest.exists():
        saved = load_torch(latest)
        if saved['signature'] != signature: raise ValueError('Checkpoint configuration/source fingerprint changed')
        for actor, state in zip(actors, saved['actors']): actor.load_state_dict(state, strict=True)
        critic.load_state_dict(saved['critic']); actor_opt.load_state_dict(saved['actor_optimizer'])
        critic_opt.load_state_dict(saved['critic_optimizer']); restore_rng(saved['rng'])
        completed = saved['episode']; best_episode = saved['best_episode']; best_score = tuple(saved['best_score'])
        pending = saved['pending_validation']
    else:
        completed = 0; best_episode = 0; best_score = selection_score(baseline_rows); pending = False
        initial = checkpoint_payload(actors, critic, actor_opt, critic_opt, 0, 0, best_score, False, signature)
        save_torch(directory/'best.pt', initial)
        export_models(initial['actors'], directory/'best_models', baseline)
        save_torch(latest, initial)
    updates_this_run = 0
    while completed < args.episodes or pending:
        verify_experiment(args, out)
        if pending:
            models = directory/f'models_ep{completed}'
            export_models(actor_states(actors), models, baseline)
            tasks = evaluation_tasks(models, f'seed{seed}_ep{completed}', seed, baseline,
                                     manifest['validation'], settings(args))
            rows = run_evaluation(out, f'seed{seed}/validation_ep{completed}', tasks, pool, status)
            score = selection_score(rows)
            if score > best_score:
                best_score = score; best_episode = completed
                export_models(actor_states(actors), directory/'best_models', baseline)
                save_torch(directory/'best.pt', checkpoint_payload(actors, critic, actor_opt, critic_opt,
                           completed, best_episode, best_score, False, signature))
            pending = False
            save_torch(latest, checkpoint_payload(actors, critic, actor_opt, critic_opt,
                       completed, best_episode, best_score, False, signature))
            write_json(directory/'selection.json', dict(best_episode=best_episode, score=best_score,
                       current_episode=completed, current_score=score, selection_split='validation_only'))
        if completed == args.episodes: break
        if args.max_updates and updates_this_run >= args.max_updates:
            status('stopped_at_checkpoint', seed=seed, episode=completed); return False
        snapshot = directory/'sampling.pt'
        save_torch(snapshot, dict(actors=actor_states(actors), episode=completed))
        tasks = []
        for index in range(completed, completed+args.batch):
            spec = make_scenario(scenario_seed(1, seed, index), 'four')
            tasks.append(dict(spec=spec, snapshot=str(snapshot), snapshot_episode=completed,
                              baseline=str(baseline), sampling_seed=scenario_seed(2, seed, index),
                              opponent=('Policy Only', 'Search + Learning')[index % 2], settings=settings(args)))
        status('training', seed=seed, episode=completed, total=args.episodes, batch=args.batch)
        started = time.perf_counter()
        results = list(pool.imap(rollout, tasks, chunksize=1)) if pool else [rollout(t) for t in tasks]
        sampling_seconds = time.perf_counter()-started
        if any(e['snapshot_episode'] != completed for e in results): raise AssertionError('Stale on-policy batch')
        started = time.perf_counter(); metric = update_ppo(actors, critic, actor_opt, critic_opt, results, config)
        update_seconds = time.perf_counter()-started; completed += args.batch; updates_this_run += 1
        metric.update(seed=seed, episode=completed, sampling_seconds=sampling_seconds, update_seconds=update_seconds,
                      opponents={name: sum(e['opponent'] == name for e in results) for name in OPPONENTS[:2]})
        write_json(directory/f'updates/{completed:06d}.json', metric)
        write_json(directory/f'rollouts/{completed:06d}.json', [dict(spec=e['spec'], opponent=e['opponent'], stats=e['stats']) for e in results])
        pending = completed % args.eval_every == 0 or completed == args.episodes
        save_torch(latest, checkpoint_payload(actors, critic, actor_opt, critic_opt, completed,
                   best_episode, best_score, pending, signature, metric))
        write_json(directory/'progress.json', dict(episode=completed, total=args.episodes, best_episode=best_episode,
                                                   sampling_seconds=sampling_seconds, update_seconds=update_seconds))
        print(f'TRAIN seed={seed} episodes={completed}/{args.episodes} exact_KL={metric["exact_kl_by_epoch"]} sample_s={sampling_seconds:.1f}', flush=True)
    final = load_torch(latest); save_torch(directory/'final.pt', final)
    export_models(final['actors'], directory/'final_models', baseline)
    selected = load_torch(directory/'best.pt')
    write_json(directory/'selection.json', dict(best_episode=selected['episode'], score=selected['best_score'],
                                               current_episode=completed, selection_split='validation_only'))
    return True


def settings(args):
    return dict(search_budget=args.search_budget, search_only_budget=args.search_only_budget,
                search_depth=100, beta_policy=.5, beta_value=.1)


def training_config(args):
    return {key: getattr(args, key) for key in ('episodes', 'batch', 'ppo_epochs', 'minibatch', 'lr',
            'critic_lr', 'clip', 'entropy', 'target_kl', 'eval_every', 'validation_episodes',
            'confirmation_episodes', 'regression_episodes', 'guard_episodes', 'search_budget',
            'search_only_budget', 'seeds', 'smoke')} | dict(gamma=1., gae_lambda=.95, search_depth=100,
            training_opponent_mix={'Policy Only': .5, 'Search + Learning': .5},
            actor_schema='original_l45_270_physical_normal_then_clip', critic_input=69,
            actor_roles=[2, 3], search_value_frozen=True, reward='original_team_mean',
            environments={layout: environment_description(layout) for layout in ('four', 'none')})


def prepare(args, out):
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    source = Path(args.model_dir).resolve()
    if source == (out/'baseline').resolve(): raise ValueError('Source must be outside experiment output')
    record = dict(config=training_config(args), source=str(source), source_models=model_fingerprint(source),
                  implementation=implementation_files())
    record = json.loads(json.dumps(record))
    if (out/'experiment.json').exists():
        if read_json(out/'experiment.json') != record: raise ValueError('Experiment fingerprint/configuration differs; use a new output')
    else:
        if (out/'baseline').exists(): raise ValueError('Unregistered baseline output exists')
        (out/'baseline').mkdir()
        for name in MODEL_NAMES: shutil.copyfile(source/name, out/'baseline'/name)
        write_json(out/'experiment.json', record)
    if model_fingerprint(out/'baseline') != record['source_models']:
        raise ValueError('Frozen baseline models changed')
    if not (out/'scenarios.json').exists():
        manifest = {}
        definitions = [('validation', 31, args.validation_episodes, 'four'),
                       ('confirmation', 32, args.confirmation_episodes, 'four'),
                       ('regression', 33, args.regression_episodes, 'none'),
                       ('guard_four', 34, args.guard_episodes, 'four'),
                       ('guard_none', 35, args.guard_episodes, 'none')]
        for name, phase, count, layout in definitions:
            manifest[name] = [make_scenario(scenario_seed(phase, 0, i), layout) for i in range(count)]
        hashes = [s['episode_hash'] for cases in manifest.values() for s in cases]
        if len(set(hashes)) != len(hashes): raise AssertionError('Evaluation splits overlap')
        write_json(out/'scenarios.json', manifest)
    signature = json_hash(dict(experiment=record, scenarios=sha(out/'scenarios.json')))
    if (out/'signature.json').exists() and read_json(out/'signature.json')['sha256'] != signature:
        raise ValueError('Frozen scenarios changed')
    write_json(out/'signature.json', dict(sha256=signature))
    return read_json(out/'scenarios.json'), signature


def verify_experiment(args, out):
    record = read_json(Path(out)/'experiment.json')
    if (record['implementation'] != implementation_files() or
        record['source_models'] != model_fingerprint(args.model_dir) or
        record['source_models'] != model_fingerprint(Path(out)/'baseline')):
        raise ValueError('Original models or implementation changed during experiment')


def correctness_check(args, out, manifest):
    baseline = Path(out)/'baseline'; destination = Path(out)/'zero_update_models'
    p, state = restore_scenario(manifest['validation'][0]); actors = load_actors(p, directory=baseline)
    export_models(actor_states(actors), destination, baseline)
    errors = []
    for robot in range(4):
        obs = torch.tensor(p.policy_encoding(state, robot).reshape(-1), dtype=torch.float32)
        limits = torch.tensor(p.action_lims[p.action_idxs[robot]], dtype=torch.float32)
        with torch.no_grad():
            mean = distribution(actors[robot], obs, limits).mean.numpy()
        original = actors[robot].eval(p, state, robot, stochastic=False).reshape(-1)
        errors.append(float(np.max(np.abs(mean-original))))
    if max(errors) != 0.: raise AssertionError(f'Zero-update actor mismatch: {errors}')
    original = make_solver(p, baseline, 'Search + Learning', 1, settings(args), 12345)
    exported = make_solver(p, destination, 'Search + Learning', 1, settings(args), 12345)
    first = original.policy(p, state); second = exported.policy(p, state)
    np.testing.assert_array_equal(first, second)
    metric = dict(passed=True, actor_mean_max_error=max(errors), original_cpp_search_exact_match=True,
                  original_models_unchanged=model_fingerprint(args.model_dir) == model_fingerprint(baseline),
                  actor_input_dim=270, state_dim=p.state_dim, swept_collision=False,
                  search=settings(args), smoke=args.smoke)
    write_json(Path(out)/'correctness.json', metric)
    return metric


def profile(args, out, manifest, status):
    if (Path(out)/'profile.json').exists(): return read_json(Path(out)/'profile.json')
    baseline = Path(out)/'baseline'; durations = []
    status('profiling_serial_search', completed=0, total=12)
    for i in range(6):
        spec = manifest['validation'][i % len(manifest['validation'])]
        for role in (0, 1):
            p, state = restore_scenario(spec)
            solver = make_solver(p, baseline, 'Search + Learning', role, settings(args), spec['search_seeds'][role])
            tick = time.perf_counter(); solver.policy(p, state); durations.append((time.perf_counter()-tick)*1000.)
        status('profiling_serial_search', completed=2*(i+1), total=12)
    result = dict(search_decisions=len(durations), mean_ms=float(np.mean(durations)), p50_ms=float(np.median(durations)),
                  p95_ms=float(np.percentile(durations, 95)), timing_context='single_process_initial_decisions',
                  note='Initial-state decisions only; parallel episode throughput is measured separately during training.',
                  search=settings(args), smoke=args.smoke)
    write_json(Path(out)/'profile.json', result); return result


def confirmation(args, out, manifest, pool, status):
    baseline = Path(out)/'baseline'
    models = [('baseline', None, baseline)] + [(f'seed{seed}_best', seed, Path(out)/f'seed{seed}/best_models') for seed in args.seeds]
    for label, seed, directory in models[1:]:
        if not (Path(out)/f'seed{seed}/final.pt').exists(): raise ValueError('Training must finish before confirmation')
    # The selections are locked before any held-out evaluation is read.
    locked = {str(seed): read_json(Path(out)/f'seed{seed}/selection.json') for seed in args.seeds}
    lock_path = Path(out)/'confirmation_selection.json'
    if lock_path.exists() and read_json(lock_path) != locked: raise ValueError('Selection changed after confirmation started')
    write_json(lock_path, locked)
    for split in ('confirmation', 'regression'):
        tasks = []
        for label, seed, directory in models:
            tasks.extend(evaluation_tasks(directory, label, seed, baseline, manifest[split], settings(args), modes=('policy', 'search')))
        for seed in args.seeds:
            tasks.extend(evaluation_tasks(Path(out)/f'seed{seed}/final_models', f'seed{seed}_final', seed, baseline,
                                         manifest[split], settings(args)))
        run_evaluation(out, split, tasks, pool, status)
    for split in ('guard_four', 'guard_none'):
        tasks = []
        for label, seed, directory in models:
            tasks.extend(evaluation_tasks(directory, label, seed, baseline, manifest[split], settings(args), modes=('search',), roles=(0,)))
        run_evaluation(out, split, tasks, pool, status)
    # Real single-process episode timings; never infer these from pooled workers.
    serial_cases = manifest['confirmation'][:min(6, len(manifest['confirmation']))]
    tasks = []
    for label, seed, directory in models:
        tasks.extend(evaluation_tasks(directory, label, seed, baseline, serial_cases, settings(args),
                                     modes=('policy', 'search'), opponents=('Policy Only',), timing_context='single_process'))
    run_evaluation(out, 'serial_latency', tasks, None, status)


def load_evaluation(out, split):
    folder = Path(out)/split
    return [read_json(folder/(entry['key']+'.json')) for entry in read_json(folder/'manifest.json')]


def paired_comparison(rows, seeds, mode, role):
    baseline = {(r['episode_hash'], r['opponent']): r for r in rows if r['label'] == 'baseline' and r['mode'] == mode and r['role'] == role}
    metric = 'legal_double_capture' if role == 1 else 'legal_survival'
    side = 'pursuer' if role == 1 else 'evader'; differences = []; illegal = []
    for seed in seeds:
        selected = {(r['episode_hash'], r['opponent']): r for r in rows
                    if r['label'] == f'seed{seed}_best' and r['mode'] == mode and r['role'] == role}
        if set(selected) != set(baseline): raise ValueError('Unpaired evaluation cases')
        keys = sorted(baseline)
        differences.append([selected[key][metric]-baseline[key][metric] for key in keys])
        illegal.append([selected[key][side+'_illegal']-baseline[key][side+'_illegal'] for key in keys])
    delta = np.asarray(differences, float); invalid = np.asarray(illegal, float)
    # Resample scenes together across opponents AND across PPO seeds. All seeds
    # share the same pretrained l45; these intervals are conditional on l45.
    scenes = sorted({key[0] for key in keys}); index = {s: [i for i, key in enumerate(keys) if key[0] == s] for s in scenes}
    by_scene = np.asarray([[np.mean(line[index[s]]) for s in scenes] for line in delta])
    rng = np.random.default_rng(45020261022); draws = []
    for _ in range(4000):
        selected_seeds = rng.integers(len(seeds), size=len(seeds)); selected_scenes = rng.integers(len(scenes), size=len(scenes))
        draws.append(float(by_scene[np.ix_(selected_seeds, selected_scenes)].mean()))
    low, high = np.percentile(draws, [2.5, 97.5])
    return dict(mode=mode, role=role, success_delta=float(delta.mean()), per_seed_delta=delta.mean(1).tolist(),
                ci95=[float(low), float(high)], own_illegal_delta=float(invalid.mean()),
                n_scenes=len(scenes), opponents=len(OPPONENTS),
                observed_discordant_pairs=int(np.count_nonzero(delta)),
                note='Conditional on one pretrained l45; scenes shared across opponents/seeds. A zero bootstrap interval with no discordances does not establish equivalence.')


def final_report(args, out):
    out = Path(out); verify_experiment(args, out)
    comparisons = {split: paired_comparison(load_evaluation(out, split), args.seeds, 'search', 1 if split in ('confirmation', 'regression') else 0)
                   for split in ('confirmation', 'regression', 'guard_four', 'guard_none')}
    pure = paired_comparison(load_evaluation(out, 'confirmation'), args.seeds, 'policy', 1)
    main = comparisons['confirmation']
    gate = bool(not args.smoke and main['success_delta'] >= .05 and main['ci95'][0] > 0 and
                min(main['per_seed_delta']) >= 0 and main['own_illegal_delta'] <= .02 and
                all(comparisons[s]['success_delta'] >= -.03 for s in ('regression', 'guard_four', 'guard_none')))
    source = read_json(out/'experiment.json')['source_models']; integrity = []
    for seed in args.seeds:
        for name in ('best_models', 'final_models'):
            fp = model_fingerprint(out/f'seed{seed}'/name)
            frozen = all(fp[f] == source[f] for f in (*MODEL_NAMES[:2], MODEL_NAMES[-1]))
            if not frozen: raise AssertionError('A frozen actor or original search value changed')
            integrity.append(dict(seed=seed, export=name, evader_and_search_value_frozen=frozen))
    updates = [read_json(f) for seed in args.seeds for f in sorted((out/f'seed{seed}/updates').glob('*.json'))]
    result = dict(search_comparisons=comparisons, pure_comparison=pure, quality_gate_passed=gate,
                  continue_formal=False, frozen_models=integrity, source_models_unchanged=model_fingerprint(args.model_dir) == source,
                  implementation_unchanged=True, max_old_logp_error=max(r['old_logp_max_error'] for r in updates),
                  train_episodes_per_seed=args.episodes, seeds=args.seeds, pretrained_seed_count=1, smoke=args.smoke)
    write_json(out/'results.json', result)
    lines = ['# 原始 l45 直接追加 PPO：对照结果', '',
             '本实验只微调原 270 维追捕策略，原环境、原 C++ 搜索和原价值网络保持不变。',
             f'训练：每种子 {args.episodes} 局；PPO 种子 {args.seeds}。搜索：{args.search_budget} 次、深度 100、βπ=0.5、βv=0.1。',
             '三个 PPO 随机种子共享同一预训练 l45，不代表多个独立预训练种子。', '',
             '## 四障碍独立确认', '', '|模型|执行|对手|合法双捕|自身非法|对手非法|首捕后超时|',
             '|---|---|---|---:|---:|---:|---:|']
    for row in read_json(out/'confirmation/summary.json'):
        lines.append(f'|{row["label"]}|{row["mode"]}|{row["opponent"]}|{row["success"]}/{row["n"]}|{row["own_illegal"]}|{row["opponent_illegal"]}|{row["one_capture_timeout"]}|')
    lines += ['', '## 配对变化与判断', '', f'- 纯策略双捕率差：{pure["success_delta"]:+.1%}。',
              f'- 搜索双捕率差：{main["success_delta"]:+.1%}；三个 PPO 种子分别为 {main["per_seed_delta"]}；95%探索区间 {main["ci95"]}。',
              f'- 预设质量门槛：{gate}。不会自动启动 50,000 局，也不会根据确认集重新挑模型或调参。',
              '- [0,0] 的退化经验区间不能证明等效；原始 double_capture、合法双捕和对手非法均分别保留。',
              '- 单进程耗时见 serial_latency/summary.json；并发训练/评测耗时不能作为部署延迟。',
              '- 本轮没有降低搜索预算；不能将成功率变化直接表述为节省单步计算时间。', '',
              '配置及源码指纹见 experiment.json；固定场景见 scenarios.json；原模型保存在 baseline/；各 seed 的 best_models/ 可由原搜索直接加载。',
              '无障碍与逃逸回归结果及完整性校验见 results.json、regression/、guard_four/、guard_none/。']
    (out/'REPORT.md').write_text('\n'.join(lines)+'\n')
    return result


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stage', choices=('pipeline', 'prepare', 'check', 'profile', 'train', 'evaluate', 'report'), default='pipeline')
    p.add_argument('--model-dir', type=Path, default=SOURCE_MODELS)
    p.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    p.add_argument('--seeds', type=int, nargs='+', default=[0, 1, 2])
    p.add_argument('--workers', type=int, default=32)
    p.add_argument('--episodes', type=int, default=2048); p.add_argument('--batch', type=int, default=64)
    p.add_argument('--ppo-epochs', type=int, default=4); p.add_argument('--minibatch', type=int, default=256)
    p.add_argument('--lr', type=float, default=3e-5); p.add_argument('--critic-lr', type=float, default=3e-4)
    p.add_argument('--clip', type=float, default=.1); p.add_argument('--entropy', type=float, default=.001)
    p.add_argument('--target-kl', type=float, default=.02)
    p.add_argument('--eval-every', type=int, default=256)
    p.add_argument('--validation-episodes', type=int, default=60)
    p.add_argument('--confirmation-episodes', type=int, default=200)
    p.add_argument('--regression-episodes', type=int, default=100)
    p.add_argument('--guard-episodes', type=int, default=100)
    p.add_argument('--search-budget', type=int, default=1000); p.add_argument('--search-only-budget', type=int, default=3000)
    p.add_argument('--max-updates', type=int, default=0, help='Stop at an atomic checkpoint, without starting held-out evaluation')
    p.add_argument('--smoke', action='store_true', help='Explicit isolated functional check; never a scientific result')
    args = p.parse_args(argv)
    if args.smoke:
        if args.output.resolve() == DEFAULT_OUTPUT.resolve(): p.error('--smoke requires a separate --output')
        args.seeds = [0]; args.episodes = 4; args.batch = 2; args.eval_every = 2
        args.validation_episodes = args.confirmation_episodes = args.regression_episodes = args.guard_episodes = 1
        args.search_budget = args.search_only_budget = 3; args.ppo_epochs = 2
    for key in ('workers', 'episodes', 'batch', 'ppo_epochs', 'minibatch', 'eval_every', 'validation_episodes',
                'confirmation_episodes', 'regression_episodes', 'guard_episodes', 'search_budget', 'search_only_budget'):
        if getattr(args, key) <= 0: p.error(f'{key} must be positive')
    if args.episodes % args.batch or args.eval_every % args.batch or args.batch % 2:
        p.error('episodes and eval-every must be divisible by an even batch')
    if len(set(args.seeds)) != len(args.seeds) or min(args.seeds) < 0 or args.max_updates < 0:
        p.error('Invalid seeds or max-updates')
    if not all(np.isfinite(getattr(args, k)) and getattr(args, k) > 0
               for k in ('lr', 'critic_lr', 'target_kl')):
        p.error('Learning rates and target-kl must be finite and positive')
    if not 0 < args.clip < 1 or not np.isfinite(args.entropy) or args.entropy < 0:
        p.error('clip must be in (0,1); entropy must be finite and nonnegative')
    args.model_dir = args.model_dir.resolve(); args.output = args.output.resolve()
    return args


def main(argv=None):
    args = parse_args(argv); torch.set_num_threads(1)
    out = args.output; out.mkdir(parents=True, exist_ok=True)
    lock = (out/'run.lock').open('a')
    try: fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError: raise SystemExit('This output already has an active driver')
    def status(phase, **extra):
        value = dict(phase=phase, updated_unix=time.time(), pid=os.getpid(), **extra)
        write_json(out/'status.json', value); print('STATUS', json.dumps(value, ensure_ascii=False), flush=True)
    try:
        manifest, signature = prepare(args, out)
        if args.stage == 'prepare': status('prepared'); return
        verify_experiment(args, out)
        if args.stage in ('pipeline', 'check') and not (out/'correctness.json').exists():
            status('correctness_checks'); correctness_check(args, out, manifest)
        if args.stage == 'check': status('checks_passed'); return
        if args.stage in ('pipeline', 'profile'):
            profile(args, out, manifest, status)
        if args.stage == 'profile': status('profile_complete'); return
        if args.stage == 'report': final_report(args, out); status('complete', continue_formal=False); return
        context = mp.get_context('spawn')
        with context.Pool(args.workers) as pool:
            if args.stage in ('pipeline', 'train'):
                baseline = out/'baseline'
                tasks = evaluation_tasks(baseline, 'baseline', None, baseline, manifest['validation'], settings(args))
                initial_rows = run_evaluation(out, 'baseline_validation', tasks, pool, status)
                for seed in args.seeds:
                    if not train_seed(args, out, seed, manifest, initial_rows, signature, pool, status): return
            if args.stage in ('pipeline', 'evaluate'):
                confirmation(args, out, manifest, pool, status)
        if args.stage in ('pipeline', 'evaluate'):
            status('summarizing'); final_report(args, out); status('complete', continue_formal=False)
        else: status('training_complete', continue_formal=False)
    except BaseException as error:
        status('interrupted' if isinstance(error, KeyboardInterrupt) else 'failed', error=str(error))
        raise


if __name__ == '__main__':
    main()
