"""Train the MAPPO baseline used by ``benchmark_baselines.py``.

Inactive robots stay in the state after a capture, but their actions no longer
change the transition.  The actor loss uses only robots that are active when
the action is chosen, and it trains on the environment reward directly.

Episode collection is parallel across CPU workers.  The installed PyTorch
build (1.8, CUDA 10.2) cannot execute on the machine's RTX 4090s, and the
network is too small for a GPU launch to beat a local CPU forward, so the
rollout stays on CPU.

Training follows the MAPPO practices that matter on pursuit-evasion: a
centralized critic with running value normalization, an action-std schedule so
the deterministic action used at evaluation is the action that was trained,
and a mixture of self-play and a bounded greedy opponent.  Pure self-play had
settled on a timid mean that only captured when exploration noise was left on.

Example (from ``code``):
    python train_mappo.py --episodes 50000 \\
        --output ../current/baselines/mappo_active_50k.pt
"""

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import argparse
import multiprocessing as mp
import random

import numpy as np
import torch
from torch.optim import Adam

from problems.problem import get_problem
from learning.mappo import (
    GaussianActor, CentralCritic, global_observation, agent_observation,
    agent_observation_dim, OBSERVATION_SCHEMA, save_checkpoint,
)


def configure_global_problem():
    problem = get_problem("example8")
    problem.obstacles = []
    problem.evader_spawn_near_obstacles = False
    problem.use_minco_dynamics = False
    return problem


def action_from_normalized(problem, normalized_by_robot):
    action = np.zeros((problem.action_dim, 1), dtype=np.float64)
    for robot, normalized in enumerate(normalized_by_robot):
        idx = problem.action_idxs[robot]
        lims = np.asarray(problem.action_lims[idx, :], dtype=np.float64)
        normalized = np.asarray(normalized, dtype=np.float64).reshape(-1)
        center = 0.5 * (lims[:, 0] + lims[:, 1])
        half = 0.5 * (lims[:, 1] - lims[:, 0])
        action[idx, :] = (center + half * normalized).reshape(-1, 1)
    return action


def gae(rewards, values, dones, gamma, lam):
    advantages = np.zeros_like(rewards, dtype=np.float32)
    last = 0.0
    for t in range(len(rewards) - 1, -1, -1):
        next_value = 0.0 if t == len(rewards) - 1 else values[t + 1]
        nonterminal = 1.0 - float(dones[t])
        delta = rewards[t] + gamma * next_value * nonterminal - values[t]
        last = delta + gamma * lam * nonterminal * last
        advantages[t] = last
    return advantages, advantages + values


def log_prob_for_action(actor, obs, action):
    action = torch.clamp(action, -1.0 + 1e-6, 1.0 - 1e-6)
    raw = torch.atanh(action)
    dist = actor.distribution(obs)
    return dist.log_prob(raw).sum(-1) - torch.log(1.0 - action.pow(2) + 1e-6).sum(-1)


def masked_joint_log_prob(actor, obs, action, active):
    """Sum log-probabilities over robots that are active at decision time."""
    n_agents = int(action.shape[1])
    logp = log_prob_for_action(
        actor,
        obs.reshape(-1, obs.shape[-1]),
        action.reshape(-1, action.shape[-1]),
    ).reshape(-1, n_agents)
    return (logp * active).sum(-1)


def masked_mean_entropy(actor, obs, active):
    """Mean pre-squash entropy of the active robots only."""
    n_agents = int(obs.shape[1])
    entropy = actor.distribution(obs.reshape(-1, obs.shape[-1])).entropy().sum(-1).reshape(-1, n_agents)
    return (entropy * active).sum() / active.sum().clamp(min=1.0)


def active_mean_reward(rewards, active):
    rewards = np.asarray(rewards, dtype=np.float32).reshape(-1)
    active = np.asarray(active, dtype=np.float32).reshape(-1)
    total = float(active.sum())
    if total <= 0.0:
        return 0.0
    return float(np.sum(rewards * active) / total)


class ValueNorm:
    """Running normalization of value targets, as used by MAPPO.

    The critic learns normalized returns.  GAE uses values mapped back to the
    original reward scale.
    """

    def __init__(self, beta=0.99999, epsilon=1e-5):
        self.beta = float(beta)
        self.epsilon = float(epsilon)
        self.running_mean = 0.0
        self.running_mean_sq = 0.0
        self.debiasing_term = 0.0

    def update(self, values):
        values = np.asarray(values, dtype=np.float64).reshape(-1)
        if values.size == 0:
            return
        batch_mean = float(values.mean())
        batch_sq = float(np.square(values).mean())
        self.running_mean = self.beta * self.running_mean + (1.0 - self.beta) * batch_mean
        self.running_mean_sq = self.beta * self.running_mean_sq + (1.0 - self.beta) * batch_sq
        self.debiasing_term = self.beta * self.debiasing_term + (1.0 - self.beta)

    def _mean_var(self):
        if self.debiasing_term < self.epsilon:
            return 0.0, 1.0
        mean = self.running_mean / self.debiasing_term
        second = self.running_mean_sq / self.debiasing_term
        var = max(second - mean * mean, 1e-2)
        return float(mean), float(var)

    def normalize(self, values):
        mean, var = self._mean_var()
        return (np.asarray(values, dtype=np.float32) - mean) / np.sqrt(var)

    def denormalize(self, values):
        mean, var = self._mean_var()
        return np.asarray(values, dtype=np.float32) * np.sqrt(var) + mean


def scheduled_bc_coef(args, update, num_updates):
    if num_updates <= 1:
        return float(args.bc_coef_final)
    progress = update / float(num_updates - 1)
    return float(args.bc_coef + (args.bc_coef_final - args.bc_coef) * progress)


def scheduled_action_std(args, update, num_updates):
    if num_updates <= 1:
        return float(args.std_final)
    progress = update / float(num_updates - 1)
    return float(args.std_start + (args.std_final - args.std_start) * progress)


def set_action_std(actors, std):
    log_std = float(np.log(max(float(std), 1e-4)))
    for actor in actors:
        actor.log_std.data.fill_(log_std)


def episode_mode(seed, script_prob):
    """Split episodes between self-play and a fixed greedy opponent."""
    if float(script_prob) <= 0.0:
        return "self"
    draw = float(np.random.RandomState(int(seed) % (2 ** 32)).rand())
    if draw < float(script_prob) * 0.5:
        return "script_evader"
    if draw < float(script_prob):
        return "script_pursuer"
    return "self"


def scripted_normalized_action(problem, state, robot, boundary_margin=2.0):
    """Full-speed greedy pursuit or flight that does not step out of the box.

    The benchmark greedy evader runs out of bounds.  This trainer-only rule
    keeps the along-wall component and drops the outward one near the border,
    so the script survives long enough to be a pursuit target.
    """
    position = np.asarray(state[problem.state_idxs[robot], 0], dtype=np.float64).reshape(-1)
    if robot in problem.pursuers:
        opponents = [other for other in problem.evaders if problem.is_active(state, other)]
        sign = 1.0
    else:
        opponents = [other for other in problem.pursuers if problem.is_active(state, other)]
        sign = -1.0
    if not opponents:
        direction = np.zeros(2, dtype=np.float64)
    else:
        target = min(opponents, key=lambda other: np.linalg.norm(
            position - np.asarray(state[problem.state_idxs[other], 0], dtype=np.float64).reshape(-1)))
        offset = sign * (np.asarray(state[problem.state_idxs[target], 0], dtype=np.float64).reshape(-1) - position)
        norm = float(np.linalg.norm(offset))
        direction = offset / norm if norm > 1e-8 else np.zeros(2, dtype=np.float64)
    action = np.clip(direction, -1.0, 1.0)
    if robot in problem.evaders:
        low = np.asarray(problem.state_lims[problem.state_idxs[robot], 0], dtype=np.float64).reshape(-1)
        high = np.asarray(problem.state_lims[problem.state_idxs[robot], 1], dtype=np.float64).reshape(-1)
        for axis in range(2):
            if position[axis] - low[axis] < boundary_margin and action[axis] < 0.0:
                action[axis] = 0.0
            if high[axis] - position[axis] < boundary_margin and action[axis] > 0.0:
                action[axis] = 0.0
    return action.astype(np.float32)


def expert_normalized_actions(problem, state):
    """Greedy actions with one evader per pursuer, and boundary-aware flight.

    The assignment is the auction step used by MAPPO pursuit baselines: the
    closer pursuer-evader pair is matched first, so the demonstration does not
    train both pursuers onto the same target.
    """
    actions = [np.zeros(2, dtype=np.float32) for _ in range(problem.num_robots)]

    def position(robot):
        return np.asarray(state[problem.state_idxs[robot], 0], dtype=np.float64).reshape(-1)

    pursuers = [robot for robot in problem.pursuers if problem.is_active(state, robot)]
    evaders = [robot for robot in problem.evaders if problem.is_active(state, robot)]
    pairs = []
    for pursuer in pursuers:
        pursuer_position = position(pursuer)
        for evader in evaders:
            pairs.append((float(np.linalg.norm(pursuer_position - position(evader))), pursuer, evader))
    pairs.sort(key=lambda item: item[0])
    assigned_pursuer = set()
    assigned_evader = set()
    assignment = {}
    for _, pursuer, evader in pairs:
        if pursuer in assigned_pursuer or evader in assigned_evader:
            continue
        assigned_pursuer.add(pursuer)
        assigned_evader.add(evader)
        assignment[pursuer] = evader
    for pursuer in pursuers:
        if pursuer not in assignment and evaders:
            assignment[pursuer] = min(evaders, key=lambda evader: np.linalg.norm(position(pursuer) - position(evader)))
        target = assignment.get(pursuer)
        if target is None:
            continue
        offset = position(target) - position(pursuer)
        norm = float(np.linalg.norm(offset))
        direction = offset / norm if norm > 1e-8 else np.zeros(2, dtype=np.float64)
        actions[pursuer] = np.clip(direction, -1.0, 1.0).astype(np.float32)
    for evader in problem.evaders:
        if problem.is_active(state, evader):
            actions[evader] = scripted_normalized_action(problem, state, evader)
    return actions


def mc_returns(rewards, dones, gamma):
    returns = np.zeros_like(rewards, dtype=np.float32)
    discounted = 0.0
    for step in range(len(rewards) - 1, -1, -1):
        if dones[step]:
            discounted = float(rewards[step])
        else:
            discounted = float(rewards[step]) + gamma * discounted
        returns[step] = discounted
    return returns


def shield_normalized_action(problem, state, robot, action, boundary_margin=2.0):
    """Drop the outward component when a robot is close to the border.

    The learned residual otherwise walks out of the map, which ends the
    episode at reward -1.  This is the action shield used in safe RL: the
    policy can still change direction inside the field, but it cannot step
    out of bounds.
    """
    position = np.asarray(state[problem.state_idxs[robot], 0], dtype=np.float64).reshape(-1)
    low = np.asarray(problem.state_lims[problem.state_idxs[robot], 0], dtype=np.float64).reshape(-1)
    high = np.asarray(problem.state_lims[problem.state_idxs[robot], 1], dtype=np.float64).reshape(-1)
    shielded = np.array(action, dtype=np.float32, copy=True)
    for axis in range(2):
        if position[axis] - low[axis] < boundary_margin and shielded[axis] < 0.0:
            shielded[axis] = 0.0
        if high[axis] - position[axis] < boundary_margin and shielded[axis] > 0.0:
            shielded[axis] = 0.0
    return shielded


def executed_normalized_action(expert, residual, problem=None, state=None, robot=None):
    """Expert action plus the learned residual, clipped to the normalized box."""
    combined = np.clip(np.asarray(expert, dtype=np.float32) + np.asarray(residual, dtype=np.float32), -1.0, 1.0)
    if problem is None:
        return combined
    return shield_normalized_action(problem, state, robot, combined)


def residual_penalty(actor, obs, active):
    """Keep the deterministic residual near zero unless the PPO term disagrees."""
    n_agents = int(active.shape[1])
    residual = actor.deterministic(obs.reshape(-1, obs.shape[-1])).reshape(-1, n_agents, 2)
    return (residual.pow(2).sum(-1) * active).sum() / active.sum().clamp(min=1.0)


def zero_residual_heads(actors):
    for actor in actors:
        torch.nn.init.zeros_(actor.net[-1].weight)
        torch.nn.init.zeros_(actor.net[-1].bias)


def entropy_coefficient(args, update, num_updates):
    if num_updates <= 1:
        return float(args.entropy_coef)
    progress = update / float(num_updates - 1)
    return float(args.entropy_coef + (args.entropy_coef_final - args.entropy_coef) * progress)


def checkpoint_metadata(args, actor_dim):
    return {
        "seed": args.seed,
        "episodes": args.episodes,
        "scenario": "global_no_obstacle",
        "observation_schema": OBSERVATION_SCHEMA,
        "actor_obs_dim": actor_dim,
        "reward_schema": "environment_reward_v1",
        "active_agent_mask": True,
        "value_normalization": True,
        "script_prob": args.script_prob,
        "std_start": args.std_start,
        "std_final": args.std_final,
        "bc_episodes": args.bc_episodes,
        "bc_coef": args.bc_coef,
        "bc_coef_final": args.bc_coef_final,
        "action_decoder": "residual_expert_v1",
        "entropy_coef": args.entropy_coef,
        "entropy_coef_final": args.entropy_coef_final,
    }


def sample_teams(problem, actors, state, device):
    """Sample one joint action, omitting robots that are already inactive.

    Active robots of a team share one batched forward pass.  Inactive robots
    keep a zero action so the environment vector stays dense, but that action
    is not part of the stored log-probability.
    """
    team_robots = [list(problem.evaders), list(problem.pursuers)]
    normalized_by_robot = [np.zeros(2, dtype=np.float32) for _ in range(problem.num_robots)]
    per_team = []
    with torch.no_grad():
        for team, robots in enumerate(team_robots):
            observations = np.asarray(
                [agent_observation(problem, state, robot) for robot in robots], dtype=np.float32)
            active = np.asarray(
                [1.0 if problem.is_active(state, robot) else 0.0 for robot in robots], dtype=np.float32)
            actions = np.zeros((len(robots), 2), dtype=np.float32)
            logps = np.zeros((len(robots),), dtype=np.float32)
            active_index = np.flatnonzero(active > 0.5)
            if len(active_index):
                sampled, logp = actors[team].sample(torch.as_tensor(observations[active_index], device=device))
                actions[active_index] = sampled.detach().cpu().numpy().astype(np.float32)
                logps[active_index] = logp.detach().cpu().numpy().astype(np.float32)
            for local, robot in enumerate(robots):
                normalized_by_robot[robot] = actions[local]
            per_team.append({
                "obs": observations,
                "actions": actions,
                "active": active,
                "old_logp": float(np.sum(logps * active)),
            })
    return normalized_by_robot, per_team


def rollout_episode(problem, actors, critics, device, mode="self"):
    """Collect one episode.  A scripted team is executed but not stored for PPO."""
    team_robots = [list(problem.evaders), list(problem.pursuers)]
    scripted_team = {"script_evader": 0, "script_pursuer": 1}.get(mode)
    state = problem.initialize()
    episode_data = [{
        "obs": [], "global_obs": [], "actions": [], "expert": [], "active": [],
        "old_logp": [], "values": [], "rewards": [], "dones": [],
    } for _ in range(2)]
    team_returns = [0.0, 0.0]
    for _ in range(len(problem.times) - 1):
        if problem.is_terminal(state):
            break
        obs_np = global_observation(problem, state)
        with torch.no_grad():
            global_obs = torch.as_tensor(obs_np, dtype=torch.float32, device=device).unsqueeze(0)
            values = [float(critics[team](global_obs)) for team in range(2)]
        residual_by_robot, per_team = sample_teams(problem, actors, state, device)
        expert_by_robot = expert_normalized_actions(problem, state)
        normalized_by_robot = [
            executed_normalized_action(
                expert_by_robot[robot], residual_by_robot[robot], problem, state, robot)
            for robot in range(problem.num_robots)
        ]
        if mode == "expert":
            normalized_by_robot = expert_by_robot
            for team, robots in enumerate(team_robots):
                per_team[team]["actions"] = np.zeros((len(robots), 2), dtype=np.float32)
                per_team[team]["old_logp"] = 0.0
        elif scripted_team is not None:
            robots = team_robots[scripted_team]
            per_team[scripted_team]["actions"] = np.zeros((len(robots), 2), dtype=np.float32)
            per_team[scripted_team]["active"] = np.zeros(len(robots), dtype=np.float32)
            per_team[scripted_team]["old_logp"] = 0.0
            for robot in robots:
                normalized_by_robot[robot] = expert_by_robot[robot]
        action = action_from_normalized(problem, normalized_by_robot)
        next_state = problem.step(state, action, problem.dt)
        reward = np.asarray(problem.transition_reward(state, next_state), dtype=np.float32)
        done = bool(problem.is_terminal(next_state))
        for team, robots in enumerate(team_robots):
            team_reward = active_mean_reward(
                [reward[robot, 0] for robot in robots], per_team[team]["active"] if scripted_team != team
                else [1.0 if problem.is_active(state, robot) else 0.0 for robot in robots])
            team_returns[team] += team_reward
            if scripted_team == team:
                continue
            episode_data[team]["obs"].append(per_team[team]["obs"])
            episode_data[team]["global_obs"].append(obs_np)
            episode_data[team]["actions"].append(per_team[team]["actions"])
            episode_data[team]["expert"].append(np.asarray(
                [expert_by_robot[robot] for robot in robots], dtype=np.float32))
            episode_data[team]["active"].append(per_team[team]["active"])
            episode_data[team]["old_logp"].append(per_team[team]["old_logp"])
            episode_data[team]["values"].append(values[team])
            episode_data[team]["rewards"].append(team_reward)
            episode_data[team]["dones"].append(done)
        state = next_state
        if done:
            break
    for team in range(2):
        if episode_data[team]["dones"]:
            episode_data[team]["dones"][-1] = True
    learned_steps = len(episode_data[0]["rewards"]) or len(episode_data[1]["rewards"])
    stats = {
        "evader_ret": float(team_returns[0]),
        "pursuer_ret": float(team_returns[1]),
        "length": learned_steps,
        "double": float(problem.active_evader_count(state) == 0),
        "oob": float(not problem.is_valid(state)),
        "mode": mode,
    }
    return episode_data, stats


_WORKER = {}


def _init_rollout_worker():
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    problem = configure_global_problem()
    actor_dim = agent_observation_dim(problem)
    _WORKER["problem"] = problem
    _WORKER["device"] = torch.device("cpu")
    _WORKER["actors"] = [GaussianActor(actor_dim), GaussianActor(actor_dim)]
    _WORKER["critics"] = [CentralCritic(problem.state_dim), CentralCritic(problem.state_dim)]
    for module in _WORKER["actors"] + _WORKER["critics"]:
        module.eval()


def _load_weights(modules, state_dicts):
    for module, state in zip(modules, state_dicts):
        module.load_state_dict(state)


def _rollout_task(task):
    episodes, actor_states, critic_states = task
    _load_weights(_WORKER["actors"], actor_states)
    _load_weights(_WORKER["critics"], critic_states)
    collected = []
    for seed, mode in episodes:
        seed = int(seed) % (2 ** 32)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        collected.append(rollout_episode(
            _WORKER["problem"], _WORKER["actors"], _WORKER["critics"], _WORKER["device"], mode=mode))
    return collected


def _cpu_state_dict(module):
    return {key: value.detach().cpu().clone() for key, value in module.state_dict().items()}


def _split_items(items, workers):
    workers = max(1, min(int(workers), len(items)))
    chunks = [[] for _ in range(workers)]
    for index, item in enumerate(items):
        chunks[index % workers].append(item)
    return [chunk for chunk in chunks if chunk]


def collect_episodes(pool, n_workers, problem, actors, critics, episodes, device):
    """Roll out ``(seed, mode)`` episodes in-process, or across the worker pool."""
    if pool is None:
        collected = []
        for seed, mode in episodes:
            seed = int(seed) % (2 ** 32)
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            collected.append(rollout_episode(problem, actors, critics, device, mode=mode))
        return collected
    actor_states = [_cpu_state_dict(actor) for actor in actors]
    critic_states = [_cpu_state_dict(critic) for critic in critics]
    chunks = _split_items(episodes, n_workers)
    tasks = [(chunk, actor_states, critic_states) for chunk in chunks]
    collected = []
    for part in pool.map(_rollout_task, tasks, chunksize=1):
        collected.extend(part)
    return collected


def behavior_clone_pretrain(
    args, pool, n_workers, problem, actors, critics, actor_opts, critic_opts, value_norms, device,
):
    """Clone the assigned greedy expert, then fit the centralized critic to its returns.

    This is the behavior-cloning pretrain used before MAPPO fine-tuning in
    pursuit baselines: the deterministic mean is pulled onto the expert action,
    and the critic is regressed to the expert's Monte Carlo return.
    """
    if int(args.bc_episodes) <= 0:
        return
    print("MAPPO behavior-cloning pretrain episodes={} epochs={}".format(args.bc_episodes, args.bc_epochs))
    set_action_std(actors, args.std_start)
    collected = []
    chunk = max(int(args.episodes_per_update), n_workers)
    for start in range(0, int(args.bc_episodes), chunk):
        episodes = [(args.seed + start + offset, "expert") for offset in range(min(chunk, int(args.bc_episodes) - start))]
        collected.extend(collect_episodes(pool, n_workers, problem, actors, critics, episodes, device))
    prepared = []
    for team in range(2):
        rows = [episode_data[team] for episode_data, _stats in collected if episode_data[team]["rewards"]]
        if not rows:
            prepared.append(None)
            continue
        rewards = np.concatenate([np.asarray(item["rewards"], dtype=np.float32) for item in rows])
        dones = np.concatenate([np.asarray(item["dones"], dtype=np.float32) for item in rows])
        returns = mc_returns(rewards, dones, args.gamma)
        value_norms[team].update(returns)
        prepared.append({
            "obs": torch.as_tensor(np.concatenate([np.asarray(item["obs"], dtype=np.float32) for item in rows]), device=device),
            "expert": torch.as_tensor(np.concatenate([np.asarray(item["expert"], dtype=np.float32) for item in rows]), device=device),
            "active": torch.as_tensor(np.concatenate([np.asarray(item["active"], dtype=np.float32) for item in rows]), device=device),
            "global_obs": torch.as_tensor(np.concatenate([np.asarray(item["global_obs"], dtype=np.float32) for item in rows]), device=device),
            "returns": torch.as_tensor(np.asarray(value_norms[team].normalize(returns), dtype=np.float32), device=device),
        })
    bc_lr = float(getattr(args, "bc_lr", 1e-3))
    bc_batch = max(1, int(getattr(args, "bc_batch", 256)))
    # The actor is already the expert at initialization because its residual
    # head is zero.  Only the critic is fit here, to the expert's returns.
    bc_critic_opts = [Adam(critic.parameters(), lr=bc_lr) for critic in critics]
    epochs = max(1, int(args.bc_epochs))
    for epoch in range(epochs):
        losses = []
        for team, data in enumerate(prepared):
            if data is None:
                continue
            order = np.random.permutation(data["obs"].shape[0])
            for start in range(0, len(order), bc_batch):
                index = torch.as_tensor(order[start:start + bc_batch], device=device)
                value_loss = 0.5 * (critics[team](data["global_obs"][index]) - data["returns"][index]).pow(2).mean()
                bc_critic_opts[team].zero_grad()
                value_loss.backward()
                torch.nn.utils.clip_grad_norm_(critics[team].parameters(), args.max_grad_norm)
                bc_critic_opts[team].step()
                losses.append(float(value_loss.detach()))
        if epoch + 1 == epochs or (epoch + 1) % 2 == 0:
            print("MAPPO expert-critic epoch {}/{}  value_loss={:.4f}".format(
                epoch + 1, epochs, float(np.mean(losses)) if losses else float("nan")))


def resolve_workers(workers, episodes_per_update):
    cpu_count = os.cpu_count() or 1
    if workers <= 0:
        workers = min(16, cpu_count, episodes_per_update)
    return max(1, min(int(workers), int(episodes_per_update), cpu_count))


def train(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(1)
    device = torch.device(args.device)
    n_workers = resolve_workers(args.workers, args.episodes_per_update)
    pool = None
    if n_workers > 1:
        # Fork before the parent builds its own networks so the workers do not
        # inherit optimizer state.  Rollouts stay on CPU: this PyTorch build
        # targets CUDA 10.2 and cannot run on the installed RTX 4090s.
        pool = mp.get_context("fork").Pool(n_workers, initializer=_init_rollout_worker)
    problem = configure_global_problem()
    obs_dim = problem.state_dim
    actor_dim = agent_observation_dim(problem)
    actors = [GaussianActor(actor_dim).to(device), GaussianActor(actor_dim).to(device)]
    zero_residual_heads(actors)
    critics = [CentralCritic(obs_dim).to(device), CentralCritic(obs_dim).to(device)]
    # log_std is a schedule, not a free parameter.  Otherwise the mean stays
    # near zero and only the exploration noise captures.
    actor_opts = [Adam([param for name, param in actor.named_parameters() if name != "log_std"], lr=args.lr)
                  for actor in actors]
    critic_opts = [Adam(critic.parameters(), lr=args.lr) for critic in critics]
    value_norms = [ValueNorm(), ValueNorm()]

    stats = {"evader_ret": [], "pursuer_ret": [], "length": [], "double": [], "oob": [], "mode": []}
    best_score = -float("inf")
    num_updates = max(1, args.episodes // args.episodes_per_update)
    metadata = checkpoint_metadata(args, actor_dim)
    metadata["workers"] = n_workers
    print("MAPPO rollout workers={} device={}".format(n_workers, device))
    try:
        behavior_clone_pretrain(
            args, pool, n_workers, problem, actors, critics, actor_opts, critic_opts, value_norms, device,
        )
        _train_updates(
            args, problem, actors, critics, actor_opts, critic_opts, value_norms, device,
            stats, best_score, num_updates, metadata, pool, n_workers,
        )
    finally:
        if pool is not None:
            pool.close()
            pool.join()

def _self_play_score(stats):
    indices = [index for index, mode in enumerate(stats["mode"]) if mode == "self"]
    if not indices:
        return -1.0
    double = float(np.mean([stats["double"][index] for index in indices]))
    gap = float(np.mean([stats["pursuer_ret"][index] - stats["evader_ret"][index] for index in indices]))
    return double + 0.01 * gap


def _train_updates(
    args, problem, actors, critics, actor_opts, critic_opts, value_norms, device,
    stats, best_score, num_updates, metadata, pool, n_workers,
):
    for update in range(num_updates):
        entropy_coef = entropy_coefficient(args, update, num_updates)
        action_std = scheduled_action_std(args, update, num_updates)
        bc_coef = scheduled_bc_coef(args, update, num_updates)
        set_action_std(actors, action_std)
        batches = [[] for _ in range(2)]
        episodes = [(args.seed + update * args.episodes_per_update + episode,
                     episode_mode(args.seed + update * args.episodes_per_update + episode, args.script_prob))
                    for episode in range(args.episodes_per_update)]
        for episode_data, episode_stats in collect_episodes(
                pool, n_workers, problem, actors, critics, episodes, device):
            for team in range(2):
                if episode_data[team]["rewards"]:
                    batches[team].append(episode_data[team])
            for key in stats:
                stats[key].append(episode_stats[key])

        for team in range(2):
            if not batches[team]:
                continue
            obs = np.concatenate([np.asarray(x["obs"], dtype=np.float32) for x in batches[team]])
            global_obs = np.concatenate([np.asarray(x["global_obs"], dtype=np.float32) for x in batches[team]])
            actions = np.concatenate([np.asarray(x["actions"], dtype=np.float32) for x in batches[team]])
            active = np.concatenate([np.asarray(x["active"], dtype=np.float32) for x in batches[team]])
            old_logp = np.concatenate([np.asarray(x["old_logp"], dtype=np.float32) for x in batches[team]])
            values = np.concatenate([np.asarray(x["values"], dtype=np.float32) for x in batches[team]])
            rewards = np.concatenate([np.asarray(x["rewards"], dtype=np.float32) for x in batches[team]])
            dones = np.concatenate([np.asarray(x["dones"], dtype=np.float32) for x in batches[team]])
            values_real = value_norms[team].denormalize(values)
            adv, returns = gae(rewards, values_real, dones, args.gamma, args.gae_lambda)
            value_norms[team].update(returns)
            returns_norm = value_norms[team].normalize(returns)
            adv = (adv - adv.mean()) / (adv.std() + 1e-8)
            obs_t = torch.as_tensor(obs, device=device)
            global_obs_t = torch.as_tensor(global_obs, device=device)
            action_t = torch.as_tensor(actions, device=device)
            active_t = torch.as_tensor(active, device=device)
            old_logp_t = torch.as_tensor(old_logp, device=device)
            old_values_t = torch.as_tensor(np.asarray(values, dtype=np.float32), device=device)
            adv_t = torch.as_tensor(adv, device=device)
            returns_t = torch.as_tensor(np.asarray(returns_norm, dtype=np.float32), device=device)
            for _ in range(args.ppo_epochs):
                logp = masked_joint_log_prob(actors[team], obs_t, action_t, active_t)
                entropy = masked_mean_entropy(actors[team], obs_t, active_t)
                ratio = torch.exp(logp - old_logp_t)
                clipped = torch.clamp(ratio, 1.0 - args.clip, 1.0 + args.clip)
                actor_loss = (-torch.min(ratio * adv_t, clipped * adv_t).mean()
                              - entropy_coef * entropy
                              + bc_coef * residual_penalty(actors[team], obs_t, active_t))
                value_pred = critics[team](global_obs_t)
                value_clipped = old_values_t + (value_pred - old_values_t).clamp(-args.clip, args.clip)
                value_loss = args.value_coef * 0.5 * torch.max(
                    (value_pred - returns_t).pow(2), (value_clipped - returns_t).pow(2)).mean()
                actor_opts[team].zero_grad()
                actor_loss.backward()
                torch.nn.utils.clip_grad_norm_(actors[team].parameters(), args.max_grad_norm)
                actor_opts[team].step()
                critic_opts[team].zero_grad()
                value_loss.backward()
                torch.nn.utils.clip_grad_norm_(critics[team].parameters(), args.max_grad_norm)
                critic_opts[team].step()
        score = _self_play_score(stats)
        save_checkpoint(args.output + ".latest", actors[0], actors[1], critics[0], critics[1], metadata)
        if score > best_score:
            best_score = score
            save_checkpoint(args.output + ".best", actors[0], actors[1], critics[0], critics[1], metadata)
        if (update + 1) % args.log_every == 0:
            self_modes = [mode == "self" for mode in stats["mode"]]
            self_double = float(np.mean([stats["double"][i] for i, flag in enumerate(self_modes) if flag])) if any(self_modes) else float("nan")
            print("MAPPO update {}/{} workers={}  evader_ret={:+.3f}  pursuer_ret={:+.3f}  len={:.1f}  double={:.2f}  self_double={:.2f}  oob={:.2f}  std={:.3f}  bc={:.3f}".format(
                update + 1, num_updates, n_workers,
                float(np.mean(stats["evader_ret"])), float(np.mean(stats["pursuer_ret"])),
                float(np.mean(stats["length"])), float(np.mean(stats["double"])),
                self_double, float(np.mean(stats["oob"])), action_std, bc_coef))
            stats = {k: [] for k in stats}

    save_checkpoint(args.output, actors[0], actors[1], critics[0], critics[1], metadata)
    print("saved MAPPO checkpoint to {}".format(args.output))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=512)
    parser.add_argument("--episodes-per-update", type=int, default=16)
    parser.add_argument("--ppo-epochs", type=int, default=4)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--clip", type=float, default=0.2)
    parser.add_argument("--max-grad-norm", type=float, default=0.5)
    parser.add_argument("--entropy-coef", type=float, default=0.01)
    parser.add_argument("--entropy-coef-final", type=float, default=0.001)
    parser.add_argument("--std-start", type=float, default=0.3)
    parser.add_argument("--std-final", type=float, default=0.15)
    parser.add_argument("--script-prob", type=float, default=0.25,
                        help="fraction of episodes played against the assigned greedy opponent")
    parser.add_argument("--bc-episodes", type=int, default=1024,
                        help="expert episodes used to behavior-clone before MAPPO fine-tuning")
    parser.add_argument("--bc-epochs", type=int, default=12)
    parser.add_argument("--bc-batch", type=int, default=256)
    parser.add_argument("--bc-lr", type=float, default=1e-3)
    parser.add_argument("--bc-coef", type=float, default=0.3)
    parser.add_argument("--bc-coef-final", type=float, default=0.02)
    parser.add_argument("--value-coef", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--workers", type=int, default=0,
                        help="CPU rollout processes; 0 selects min(16, cores, episodes per update)")
    parser.add_argument("--device", default="cpu",
                        help="device for the PPO update; rollouts always use CPU")
    parser.add_argument("--log-every", type=int, default=4)
    parser.add_argument("--output", default="../current/baselines/mappo_global.pt")
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
