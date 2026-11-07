"""Root-sampled open-loop team MCTS. Real hidden states are not an input.

Actions, not particle identities, index nodes. Opponent actions and our
rollouts are generated from causal, simulated team observations. This is a
receding-horizon approximation, not a POMCP observation tree.
"""
import copy
import time
import numpy as np
import torch
from local_observation import TEAMS, TeamBelief, observe, legal_points, first_collision_point
from learning.local_rl import critic_features


class Node:
    def __init__(self, action=None):
        self.action = action; self.visits = 0; self.total = 0.; self.children = []


class LocalController:
    def __init__(self, model, bundle, team, seed=0, particles=64, without_belief=None):
        self.model = model  # public map/dynamics; never a real world state
        self.bundle = bundle; self.team = int(team)
        self.rng = np.random.default_rng(seed)
        self.belief = TeamBelief(model.obstacles, model.state_lims[:2], team, seed, particles)
        self.without_belief = bool(bundle.metadata.get('without_belief',False)) if without_belief is None else without_belief
        self.last_stats = {}

    def prepare(self, obs):
        self.belief.update(obs)
        return self.belief.batch(obs, self.without_belief)

    def actor_actions(self, obs, belief, stochastic=False):
        features = belief.batch(obs, self.without_belief)
        actions = np.zeros((2, 2)); raw = np.zeros((2, 2)); logp = np.zeros(2)
        for local, robot in enumerate(TEAMS[obs.team]):
            if obs.active[local]:
                actions[local], raw[local], logp[local] = self.bundle.actors[robot].action(features[local], obs.limits[local], self.rng if stochastic else None)
        return actions, raw, logp, features

    def act(self, obs, stochastic=False):
        self.prepare(obs)
        action, raw, logp, features = self.actor_actions(obs, self.belief, stochastic)
        self.last_stats = dict(simulations=0, transitions=0)
        return action, dict(raw=raw, old_logp=float(logp.sum()), features=features)


class BeliefMCTS(LocalController):
    def __init__(self, model, bundle, team, seed=0, particles=64, simulations=1000, depth=8,
                 beta_policy=.5, beta_value=0., without_belief=None, time_budget_ms=None):
        super().__init__(model, bundle, team, seed, particles, without_belief)
        self.simulations = int(simulations); self.depth = int(depth)
        self.beta_policy = float(beta_policy); self.beta_value = float(beta_value)
        self.time_budget_ms=time_budget_ms
        if beta_value and not bundle.metadata.get('value_validated', False):
            raise ValueError('Value cutoff requires an explicitly validated critic checkpoint')

    def hypothesis(self, obs):
        p = self.model
        state = np.zeros((p.state_dim, 1))
        state[p.time_idx, 0] = obs.time
        for i, robot in enumerate(TEAMS[self.team]):
            state[p.state_idxs[robot], 0] = obs.positions[i]
            state[p.active_idxs[robot], 0] = obs.active[i]
        particle = self.belief.particles[self.rng.integers(self.belief.n)]
        # Slot identities are bookkeeping only; no correspondence with real IDs.
        for i, robot in enumerate(TEAMS[1 - self.team]):
            point=particle[i,:2]
            if self.belief.existence[i]<.5 and not legal_points(point[None],p.obstacles,p.state_lims[:2])[0]:
                # No measurement ever existed for this now-absent slot. A
                # placeholder must not make a valid sampled world terminal.
                point=obs.positions[0]
            state[p.state_idxs[robot], 0] = point
            state[p.active_idxs[robot], 0] = self.belief.existence[i]
        # No hidden observation memory is copied from the real simulator.
        return state

    def safe(self, actions, obs):
        for i in range(2):
            if not obs.active[i]:
                continue
            end = obs.positions[i] + actions[i]
            if not legal_points(end[None], self.model.obstacles, self.model.state_lims[:2])[0]:
                return False
            if not np.allclose(first_collision_point(obs.positions[i], end, self.model.obstacles), end, atol=1e-7):
                return False
        return True

    def proposal(self, obs, belief):
        for _ in range(12):
            if self.rng.random() < self.beta_policy:
                action, _, _, _ = self.actor_actions(obs, belief, stochastic=True)
            else:
                action = self.rng.uniform(obs.limits[..., 0], obs.limits[..., 1]) * obs.active[:, None]
            if self.safe(action, obs):
                return action
        return np.zeros((2, 2))

    def act(self, obs, stochastic=False):
        started = time.perf_counter(); features = self.prepare(obs)
        root = Node(); transitions = 0; p = self.model
        # Explicit own limits; opponent limits are public maximum model bounds,
        # not the environment's randomized hidden episode parameters.
        for i, robot in enumerate(TEAMS[self.team]):
            p.action_lims[p.action_idxs[robot]] = obs.limits[i]
            p.control_lims[p.action_idxs[robot]] = obs.limits[i] / p.dt
        enable_fast_model(p)
        completed=0
        for _ in range(self.simulations):
            if completed and self.time_budget_ms is not None and (time.perf_counter()-started)*1000.>=self.time_budget_ms:
                break
            state = self.hypothesis(obs)
            ours = copy.deepcopy(self.belief)
            # All simulations have independent random streams, including
            # bootstrap sampling, while sharing the real posterior support.
            ours.rng = np.random.default_rng(int(self.rng.integers(2**31)))
            theirs = TeamBelief(p.obstacles, p.state_lims[:2], 1-self.team, int(self.rng.integers(2**31)), min(8, self.belief.n))
            node = root; path = [node]; rewards = []; expanded = False
            for depth in range(self.depth):
                if p.is_terminal(state):
                    break
                own_obs = obs if depth == 0 else observe(p, state, self.team)
                if depth:
                    ours.update(own_obs)
                other_obs = observe(p, state, 1-self.team); theirs.update(other_obs)
                if not expanded:
                    width = max(1, int(2 * max(node.visits, 1) ** .5))
                    if len(node.children) < width:
                        child = Node(self.proposal(own_obs, ours)); node.children.append(child); expanded = True
                    else:
                        child = max(node.children, key=lambda c: c.total / max(c.visits, 1) + np.sqrt(np.log(node.visits + 1) / max(c.visits, 1)))
                    own_action = child.action
                    node = child; path.append(node)
                else:
                    own_action = self.proposal(own_obs, ours)
                other_action, _, _, _ = self.actor_actions(other_obs, theirs)
                action = np.zeros((p.action_dim, 1))
                for team, local_action in ((self.team, own_action), (1-self.team, other_action)):
                    for i, robot in enumerate(TEAMS[team]):
                        action[p.action_idxs[robot], 0] = local_action[i]
                nxt = p.step(state, action, p.dt)
                reward = p.transition_reward(state, nxt)
                rewards.append(float(np.mean(reward[list(TEAMS[self.team]), 0])))
                state = nxt; transitions += 1
            tail = 0.
            if self.beta_value and not p.is_terminal(state) and self.rng.random() < self.beta_value:
                end_obs = observe(p, state, self.team); ours.update(end_obs)
                x = critic_features(p, state, ours.batch(end_obs, self.without_belief))
                with torch.no_grad():
                    tail = self.bundle.norms[self.team].denormalize(float(self.bundle.critics[self.team](torch.from_numpy(x))))
            # A node at depth d receives return from d onward. The root stores
            # the full return; do not back up the root prefix at deeper nodes.
            for d, visited in enumerate(path):
                visited.visits += 1; visited.total += sum(rewards[max(0, d-1):]) + tail
            completed+=1
        candidates = [c for c in root.children if self.safe(c.action, obs)]
        best = max(candidates, key=lambda c: c.total / max(c.visits, 1)).action if candidates else np.zeros((2, 2))
        self.last_stats = dict(simulations=completed, requested_simulations=self.simulations, transitions=transitions,
                               latency_ms=1000*(time.perf_counter()-started), root_children=len(root.children))
        return best, dict(features=features, root=[dict(action=c.action.tolist(), visits=c.visits, value=c.total/max(c.visits,1)) for c in root.children])


def public_model(obstacles):
    """Construct a simulation model using only public constants and the map."""
    from problems.problem import get_problem
    p = get_problem('example8'); p.obstacles = [np.array(b) for b in obstacles]
    p.swept_collision = True; p.use_minco_dynamics = False
    # Opponent motion is a model, not a read of real episode speed limits.
    for robot in range(4):
        upper = 1. if robot < 2 else 1.2
        p.action_lims[p.action_idxs[robot]] = np.array([[-upper, upper]] * 2)
        p.control_lims[p.action_idxs[robot]] = p.action_lims[p.action_idxs[robot]] / p.dt
    return p


def enable_fast_model(problem):
    """Use the tested C++ simulator/encoder, retaining the public Python API.

    Called after setting this decision's own action limits. The wrapper's
    signature includes map, limits and collision protocol.
    """
    from solvers.c_puct import C_PUCT
    if not hasattr(problem, '_local_adapter'):
        problem._local_adapter = C_PUCT(policy_oracle=[None]*4, number_simulations=1, solver_name='C_PUCT_V1')
    cpp = problem._local_adapter._problem_wrapper_for(problem)
    if not hasattr(cpp, 'transition_reward'):
        raise RuntimeError('Rebuild bindings with local protocol support')
    problem._local_cpp = cpp
    problem.step = lambda state, action, dt: np.asarray(cpp.step(state, action, dt)).reshape(-1, 1)
    problem.transition_reward = lambda state, nxt: np.asarray(cpp.transition_reward(state, np.zeros((8, 1)), nxt)).reshape(-1, 1)
    problem.is_terminal = lambda state: bool(cpp.is_terminal(state))
    return problem
