"""Small self-contained MAPPO baseline for the pursuit-evasion benchmark.

This implementation intentionally uses the repository's existing Problem
interface.  Actors are shared within each team and the critic is centralized
at team level.  It is meant as a reproducible baseline, not as a replacement
for the search-learning method.
"""

import os
import numpy as np
import torch
from torch import nn
from torch.distributions import Normal


OBSERVATION_SCHEMA = "ego_relative_v1"

def global_observation(problem, state):
    state = np.asarray(state, dtype=np.float32).reshape((-1, 1))
    low = np.asarray(problem.state_lims[:, 0:1], dtype=np.float32)
    high = np.asarray(problem.state_lims[:, 1:2], dtype=np.float32)
    scale = np.maximum(high - low, 1e-6)
    return np.clip((2.0 * (state - low) / scale - 1.0).reshape(-1), -1.0, 1.0)


def agent_observation(problem, state, robot):
    """Fixed-size decentralized observation with explicit identity.

    Positions are expressed relative to the ego robot, while the ego absolute
    position, alive mask, time, role and one-hot robot id remain available.
    The ordering is deterministic and therefore invariant to execution order.
    """
    state = np.asarray(state, dtype=np.float32).reshape((-1, 1))
    extent = np.maximum(np.asarray(problem.state_lims[:, 1] - problem.state_lims[:, 0], dtype=np.float32), 1e-6)
    pos = np.asarray([state[problem.state_idxs[i], 0] for i in range(problem.num_robots)], dtype=np.float32)
    ego = pos[int(robot)]
    rel = ((pos - ego) / extent[:2]).reshape(-1)
    low = np.asarray(problem.state_lims[problem.state_idxs[int(robot)], 0], dtype=np.float32)
    high = np.asarray(problem.state_lims[problem.state_idxs[int(robot)], 1], dtype=np.float32)
    absolute = 2.0 * (ego - low) / np.maximum(high - low, 1e-6) - 1.0
    alive = np.asarray([float(problem.is_active(state, i)) if hasattr(problem, "is_active") else 1.0
                        for i in range(problem.num_robots)], dtype=np.float32)
    time_value = np.asarray([2.0 * float(state[problem.time_idx, 0] / max(float(problem.tf), 1e-6)) - 1.0], dtype=np.float32)
    identity = np.eye(problem.num_robots, dtype=np.float32)[int(robot)]
    role = np.asarray([1.0 if int(robot) in problem.evaders else 0.0,
                       1.0 if int(robot) in problem.pursuers else 0.0], dtype=np.float32)
    return np.clip(np.concatenate([absolute, rel, alive, time_value, identity, role]), -1.0, 1.0)


def agent_observation_dim(problem):
    return int(2 + 2 * problem.num_robots + problem.num_robots + 1 + problem.num_robots + 2)


class GaussianActor(nn.Module):
    def __init__(self, obs_dim, action_dim=2, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, action_dim),
        )
        self.log_std = nn.Parameter(torch.full((action_dim,), -0.5))

    def distribution(self, obs):
        mean = self.net(obs)
        std = self.log_std.exp().expand_as(mean)
        return Normal(mean, std)

    def sample(self, obs):
        dist = self.distribution(obs)
        raw = dist.rsample()
        action = torch.tanh(raw)
        log_prob = dist.log_prob(raw).sum(-1) - torch.log(1.0 - action.pow(2) + 1e-6).sum(-1)
        return action, log_prob

    def deterministic(self, obs):
        return torch.tanh(self.net(obs))


class CentralCritic(nn.Module):
    def __init__(self, obs_dim, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, 1),
        )

    def forward(self, obs):
        return self.net(obs).squeeze(-1)


class MAPPOPolicySolver:
    solver_name = "MAPPO"
    supports_team_policy = True

    def __init__(self, problem, path=None, device="cpu", legacy=False):
        self.problem = problem
        # ``auto`` lets a benchmark worker use CUDA when the GPU is visible,
        # while still allowing the run to continue on CPU if the driver or
        # device mapping is temporarily unavailable.
        if str(device).lower() == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.legacy = bool(legacy)
        self.residual_expert = False
        obs_dim = int(problem.state_dim) if self.legacy else agent_observation_dim(problem)
        self.evader_actor = GaussianActor(obs_dim).to(self.device)
        self.pursuer_actor = GaussianActor(obs_dim).to(self.device)
        self.path = path
        if path is not None:
            self.load(path)

    def load(self, path):
        checkpoint = torch.load(path, map_location=self.device)
        metadata = checkpoint.get("metadata", {})
        if self.legacy:
            if metadata.get("observation_schema") is not None:
                raise ValueError("--legacy is only for checkpoints without an observation schema")
        elif metadata.get("observation_schema") not in (OBSERVATION_SCHEMA, None):
            raise ValueError("unsupported MAPPO observation schema: {}".format(metadata.get("observation_schema")))
        if not self.legacy and metadata.get("observation_schema") != OBSERVATION_SCHEMA:
            raise ValueError("legacy MAPPO checkpoint has no compatible ego observation schema")
        self.evader_actor.load_state_dict(checkpoint["evader_actor"])
        self.pursuer_actor.load_state_dict(checkpoint["pursuer_actor"])
        self.evader_actor.eval()
        self.pursuer_actor.eval()
        self.residual_expert = metadata.get("action_decoder") == "residual_expert_v1"

    def _scaled_action(self, normalized, robot, problem=None):
        problem = self.problem if problem is None else problem
        idx = problem.action_idxs[robot]
        lims = torch.as_tensor(problem.action_lims[idx, :], dtype=normalized.dtype, device=normalized.device)
        center = 0.5 * (lims[:, 0] + lims[:, 1])
        half = 0.5 * (lims[:, 1] - lims[:, 0])
        return center + half * normalized

    @torch.no_grad()
    def policy(self, problem, state, team=None):
        obs = torch.as_tensor(global_observation(problem, state), dtype=torch.float32, device=self.device).unsqueeze(0)
        action = np.zeros((problem.action_dim, 1), dtype=np.float64)
        team = None if team is None else set(int(robot) for robot in team)
        expert = None
        if self.residual_expert:
            from train_mappo import executed_normalized_action, expert_normalized_actions
            expert = expert_normalized_actions(problem, state)
        for robot in range(problem.num_robots):
            if team is not None and robot not in team:
                continue
            if hasattr(problem, "is_active") and not problem.is_active(state, robot):
                continue
            actor = self.evader_actor if robot in problem.evaders else self.pursuer_actor
            robot_obs = obs if self.legacy else torch.as_tensor(agent_observation(problem, state, robot), dtype=torch.float32, device=self.device).unsqueeze(0)
            normalized = actor.deterministic(robot_obs).squeeze(0)
            if expert is not None:
                normalized = torch.as_tensor(
                    executed_normalized_action(
                        expert[robot], normalized.detach().cpu().numpy(), problem, state, robot),
                    dtype=normalized.dtype, device=normalized.device)
            scaled = self._scaled_action(normalized, robot, problem).cpu().numpy().reshape(-1, 1)
            action[problem.action_idxs[robot], :] = scaled
        return action


def save_checkpoint(path, evader_actor, pursuer_actor, evader_critic, pursuer_critic, metadata=None):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    torch.save({
        "evader_actor": evader_actor.state_dict(),
        "pursuer_actor": pursuer_actor.state_dict(),
        "evader_critic": evader_critic.state_dict(),
        "pursuer_critic": pursuer_critic.state_dict(),
        "metadata": metadata or {},
    }, path)
