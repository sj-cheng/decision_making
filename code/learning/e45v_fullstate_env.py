"""E45-V privileged full-state pursuit diagnostic with frozen task physics.

This interface intentionally exposes true present state and fixed-D mode/model
parameters to the diagnostic Actor. It is not eligible for the public-observation
benchmark. Real LOS reports remain separate, never rewritten as truth reports.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np

try:
    from gymnasium.spaces import Box
except ImportError:  # The author dispatcher checks Box's class name only.
    from gym.spaces import Box

from corner_estimator_scenarios import original_model
from e45g_training_acceleration import public_geometry_cache
from learning.e45g_ppo import safe_action
from run_e45_team_pilot import FixedKernelTargets
from run_frozen_d_closed_loop import legal_action
from solvers.e45_team_adapter import public_observation
from solvers.e45b_joint_nte import simultaneous_capture, TIME_COST
from solvers.frozen_d_search_interface import PublicMap, route


MAX_BOXES = 8
MAX_NEIGHBORS = 49
STATE_DIM = 53
# Simulator state, padded map+presence, node IDs, 9-way current modes,
# graph anchors, relative positions, speed+remaining time, complete local
# next-position support with exact marginal edge probabilities and validity.
CENT_DIM = STATE_DIM + MAX_BOXES * 5 + 2 + 18 + 4 + 8 + 2
KERNEL_DIM = 2 * MAX_NEIGHBORS * 4
OBS_DIM = CENT_DIM + 3  # ego onehot[2], then ego active flag (must remain last)
ENCODING = {
    "simulator_state": [0, 53],
    "map_boxes_and_presence": [53, 93],
    "target_node_ids": [93, 95],
    "target_mode_onehot": [95, 113],
    "target_graph_anchors": [113, 117],
    "pursuer_target_relative_positions": [117, 125],
    "pursuer_speed_and_remaining": [125, 127],
    "ego_onehot": [CENT_DIM, CENT_DIM + 2],
    "ego_active": [OBS_DIM - 1, OBS_DIM],
}


def state_scale():
    scale = np.ones(STATE_DIM, dtype=np.float64)
    scale[:8] = 10.0
    scale[8] = 80.0
    scale[13:29] = 2.4
    for robot in range(4):
        start = 29 + 6 * robot
        scale[start:start + 2] = 20.0
        scale[start + 2:start + 4] = 2.4
        scale[start + 4] = 80.0
    return scale


class FullStatePursuitEnv:
    """HARL six-return environment; team done is distinct from retirement.

    Args: case_paths (required), opponent='fixed_D_kernel', max_boxes=8.
    ``seed(n)`` followed by reset uses n+1 like the frozen HARL MPE adapter.
    ``reset(case_path=..., world=..., episode_seed=...)`` fixes an engineering
    case without altering any underlying case file.
    """

    n_agents = 2

    def __init__(self, args):
        self.args = copy.deepcopy(args)
        if args.get("opponent", "fixed_D_kernel") != "fixed_D_kernel":
            raise ValueError("E45-V exposes fixed-D state; other opponent requires its own state audit")
        if int(args.get("max_boxes", MAX_BOXES)) != MAX_BOXES:
            raise ValueError("Frozen encoding requires max_boxes=8")
        if args.get("reward", "privileged_geometric_potential") != "privileged_geometric_potential":
            raise ValueError("V reward is explicitly adapted to a current-state geometric potential")
        paths = args.get("case_paths", [])
        self.case_paths = [str(Path(p["path"] if isinstance(p, dict) else p)) for p in paths]
        if not self.case_paths:
            raise ValueError("case_paths must be nonempty")
        self.world_ids = tuple(args.get("world_ids", ("AB", "BA")))
        # Static map+anchors and the frozen generator fully determine the
        # opponent transition graph. Formal V uses compact current state;
        # an expanded one-step support is optional for engineering inspection.
        self.include_transition_kernel = bool(args.get("include_transition_kernel", False))
        self.cent_dim = CENT_DIM + (KERNEL_DIM if self.include_transition_kernel else 0)
        self.obs_dim = self.cent_dim + 3
        self._cases = {}
        self._maps = {}
        self._seed = 0
        self._terminated = True
        self.state = None
        self.observation_space = [Box(-np.inf, np.inf, (self.obs_dim,), dtype=np.float32) for _ in range(2)]
        self.share_observation_space = [Box(-np.inf, np.inf, (self.cent_dim,), dtype=np.float32) for _ in range(2)]
        self.action_space = [Box(-1., 1., (2,), dtype=np.float32) for _ in range(2)]
        self.last_info = None
        self.trace = []

    def seed(self, seed):
        self._seed = int(seed)

    def _case(self, path):
        if path not in self._cases:
            self._cases[path] = json.loads(Path(path).read_text())
        return self._cases[path]

    def reset(self, case_path=None, world=None, episode_seed=None):
        self._seed += 1
        reset_rng = np.random.default_rng(self._seed)
        path = str(case_path) if case_path is not None else self.case_paths[int(reset_rng.integers(len(self.case_paths)))]
        case = self._case(path)
        worlds = [row for row in case["worlds"] if row["id"] in self.world_ids]
        if not worlds:
            raise ValueError("No allowed worlds in case")
        selected = next((row for row in worlds if row["id"] == world), None) if world is not None else worlds[int(reset_rng.integers(len(worlds)))]
        if selected is None:
            raise ValueError("Requested world not available")
        self.episode_seed = int(episode_seed) if episode_seed is not None else int(reset_rng.integers(1, 2**31 - 1))
        self.case_path, self.case, self.world = path, case, selected["id"]
        self.boxes = np.asarray(case["map"], dtype=np.float64)
        if self.boxes.shape[0] > MAX_BOXES:
            raise ValueError("Map exceeds frozen full-state capacity")
        self.speed = float(case["speed"])
        self.problem = original_model(self.boxes, self.speed)
        if self.problem.state_dim != STATE_DIM:
            raise ValueError("Simulator state shape changed")
        self.state = np.asarray(selected["initial"], dtype=np.float64).reshape(STATE_DIM, 1).copy()
        # Exact, argument-keyed graph/probability memoization only. Random root
        # sampling and every observation-conditioned posterior remain fresh.
        with public_geometry_cache():
            self.other = FixedKernelTargets(case, self.episode_seed)
        self.state[:4, 0] = self.other.positions().reshape(-1)
        if not self.problem.is_valid(self.state) or self.problem.is_terminal(self.state):
            raise ValueError("Invalid or already-terminal handover")
        key = (self.boxes.dtype.str, self.boxes.shape, self.boxes.tobytes(), self.speed)
        if key not in self._maps:
            self._maps[key] = PublicMap(self.boxes, self.speed)
        self.public_map = self._maps[key]
        self.start_step = int(self.state[8, 0])
        self.initial_remaining = 80 - self.start_step
        self._terminated = False
        self.first_capture = None
        self.decisions = 0
        self.raw_utility = 0.
        self.training_return = 0.
        self.blocked_total = 0
        self.trace = []
        self.phi = self.potential()
        self.initial_phi = self.phi
        self.initial_active_targets = int(self.active_e.sum())
        self.last_info = {"privileged_actor": True, "case": case["id"], "world": self.world,
                          "episode_seed": self.episode_seed, "step": self.start_step,
                          "remaining": self.initial_remaining, "public_observation": self.public_obs()}
        return self._observation()

    @property
    def active_e(self):
        return self.state[self.problem.active_idxs[:2], 0] > .5

    @property
    def active_p(self):
        return self.state[self.problem.active_idxs[2:], 0] > .5

    def public_obs(self):
        return public_observation(int(self.state[8, 0]), self.state[4:8, 0].reshape(2, 2),
                                  self.state[:4, 0].reshape(2, 2), self.active_p,
                                  self.active_e, self.boxes)

    def model_support(self):
        """All legal next positions and current one-step marginal probabilities.

        The latent mode is retained with probability .9, otherwise redrawn
        uniformly over nine modes. These probabilities do not consume RNG.
        The RNG's future state is deliberately never encoded.
        """
        support = np.zeros((2, MAX_NEIGHBORS, 4), dtype=np.float64)
        for j, graph in enumerate(self.other.graphs):
            if not self.active_e[j]:
                continue
            index, mode = int(self.other.ids[j]), int(self.other.modes[j])
            count = int(graph[2][index])
            if count > MAX_NEIGHBORS:
                raise ValueError("Opponent support exceeds declared capacity; no silent truncation")
            links = graph[1][index, :count]
            probabilities = graph[3][index, :count]
            marginal = .9 * probabilities[:, mode] + .1 * probabilities.mean(axis=1)
            np.testing.assert_allclose(marginal.sum(), 1., atol=1e-10, rtol=0)
            support[j, :count, :2] = graph[0][links] / 10.
            support[j, :count, 2] = marginal
            support[j, :count, 3] = 1.
        return support

    def _shared_features(self):
        map_rows = np.zeros((MAX_BOXES, 5), dtype=np.float64)
        map_rows[:len(self.boxes), :4] = self.boxes.reshape(-1, 4) / 10.
        map_rows[:len(self.boxes), 4] = 1.
        ids = np.zeros(2)
        modes = np.zeros((2, 9))
        anchors = np.zeros((2, 2))
        for j, graph in enumerate(self.other.graphs):
            anchors[j] = graph[0][int(graph[5])] / 10.
            if self.active_e[j]:
                index, mode = int(self.other.ids[j]), int(self.other.modes[j])
                # Active target physics and kernel state must agree exactly.
                np.testing.assert_allclose(self.state[2*j:2*j+2, 0], graph[0][index], atol=1e-10, rtol=0)
                ids[j] = index / max(1, len(graph[0]) - 1)
                modes[j, mode] = 1.
        relative = ((self.state[:4, 0].reshape(1, 2, 2)
                     - self.state[4:8, 0].reshape(2, 1, 2)) / 20.)
        shared = np.r_[self.state[:, 0] / state_scale(), map_rows.ravel(), ids,
                       modes.ravel(), anchors.ravel(), relative.ravel(), self.speed / 1.3,
                       (80. - self.state[8, 0]) / 80.]
        if self.include_transition_kernel:
            shared = np.r_[shared, self.model_support().ravel()]
        shared = shared.astype(np.float32)
        if shared.shape != (self.cent_dim,) or not np.isfinite(shared).all():
            raise ValueError("Invalid privileged state encoding")
        return shared

    def _observation(self):
        shared = self._shared_features()
        obs = np.stack([np.r_[shared, np.eye(2)[i], float(self.active_p[i])] for i in range(2)]).astype(np.float32)
        state = np.repeat(shared[None], 2, axis=0)
        assert obs.shape == (2, self.obs_dim) and state.shape == (2, self.cent_dim)
        return obs, state, None

    def potential(self):
        """Privileged state-only counterpart of J's unassigned path potential.

        Gamma is one. Thus the full shaped return remains raw utility minus
        initial Phi, apart from the already-declared shortened-intent penalty.
        This is explicitly not J's posterior-averaged public potential.
        """
        if self._terminated or not self.active_e.any():
            return 0.
        count = np.zeros(2, np.int64)
        value = 0.
        for j in np.flatnonzero(self.active_e):
            distance = 80.
            for i in np.flatnonzero(self.active_p):
                result = route(self.state[4+2*i:6+2*i, 0], self.state[2*j:2*j+2, 0],
                               self.speed, self.public_map.vertices, self.public_map.dist,
                               self.boxes, count)
                distance = min(distance, float(result[0]))
            value -= distance
        return float(.05 * value + .25 * (2 - self.active_e.sum()))

    def step(self, actions):
        if self._terminated:
            raise RuntimeError("Step after terminal; reset is required")
        actions = np.asarray(actions, dtype=np.float64)
        if actions.shape != (2, 2) or not np.isfinite(actions).all():
            raise ValueError("Expected finite normalized (2,2) action")
        if np.max(np.abs(actions)) > 1. + 2e-7:
            raise ValueError("Normalized action outside declared Box")
        old = self.state.copy()
        ap, ae = self.active_p.copy(), self.active_e.copy()
        intent = np.clip(actions, -1., 1.) * ap[:, None]
        physical, shortened = safe_action(old[4:8, 0].reshape(2, 2), intent, ap, self.speed, self.boxes)
        # Same execution-only <=2e-7 roundoff repair as E45-P, without pulling
        # its experiment launcher into the environment's dependency graph.
        physical = np.asarray(physical, dtype=np.float64)
        executed = np.clip(physical, -self.speed, self.speed)
        speed_roundoff = float(np.max(np.abs(executed - physical)))
        if speed_roundoff > 2e-7:
            raise RuntimeError("Unexpected non-roundoff overspeed")
        for i in range(2):
            if not legal_action(old[4+2*i:6+2*i, 0], executed[i], self.speed, self.boxes):
                raise RuntimeError("Illegal pursuer transition")
            if not ap[i] and np.any(executed[i] != 0.):
                raise RuntimeError("Retired pursuer moved")
        enemy = np.asarray(self.other.actions(ae), dtype=np.float64)
        for j in range(2):
            if not ae[j]: enemy[j] = 0.
            if not legal_action(old[2*j:2*j+2, 0], enemy[j], 1., self.boxes):
                raise RuntimeError("Illegal fixed-kernel target transition")
        joint = np.r_[enemy.ravel(), executed.ravel()][:, None]
        self.state = self.problem.step(old, joint, 1.)
        if not self.problem.is_valid(self.state):
            raise RuntimeError("Simulator produced invalid state")
        np.testing.assert_allclose(self.state[:8], old[:8] + joint, atol=1e-10, rtol=0)
        nap, nae, _ = simultaneous_capture(self.state[4:8, 0].reshape(2, 2),
                                           self.state[:4, 0].reshape(2, 2), ap, ae, self.boxes)
        np.testing.assert_array_equal(nap, self.active_p)
        np.testing.assert_array_equal(nae, self.active_e)
        captures = int(ae.sum() - nae.sum())
        complete = not nae.any()
        self._terminated = bool(self.problem.is_terminal(self.state))
        if self._terminated != (complete or int(self.state[8, 0]) >= 80):
            raise RuntimeError("Unexpected task termination")
        self.decisions += 1
        if captures and self.first_capture is None: self.first_capture = self.decisions
        raw = float(complete) - TIME_COST
        blocked = float(shortened / max(1, ap.sum()))
        next_phi = 0. if self._terminated else self.potential()
        reward = float(raw + next_phi - self.phi - .05 * blocked)
        before_phi, self.phi = self.phi, next_phi
        self.raw_utility += raw
        self.training_return += reward
        self.blocked_total += int(shortened)
        obs, state, available = self._observation()
        info = {"transition_active_mask": ap.astype(np.float32).tolist(),
                "next_active_mask": nap.astype(np.float32).tolist(),
                "privileged_actor": True, "raw_reward": raw, "phi": before_phi,
                "next_phi": next_phi, "blocked_fraction": blocked,
                "new_captures": captures, "public_observation": self.public_obs(),
                "normalized_intent": intent.tolist(), "physical_action": executed.tolist(),
                "opponent_action": enemy.tolist(), "speed_roundoff_correction": speed_roundoff,
                "true_terminal": self._terminated}
        if self._terminated:
            info["episode"] = {"case": self.case["id"], "world": self.world,
                               "episode_seed": self.episode_seed, "start_step": self.start_step,
                               "decisions": self.decisions, "utility": self.raw_utility,
                               "training_return": self.training_return,
                               "captures": self.initial_active_targets - int(nae.sum()),
                               "first_capture": self.first_capture, "double_capture": complete,
                               "completion_cost": self.decisions if complete else self.initial_remaining,
                               "reason": "double_capture" if complete else "timeout",
                               "initial_phi": self.initial_phi,
                               "blocked_total": self.blocked_total}
        self.last_info = info
        if self.args.get("record_trace", False):
            self.trace.append({"state": old[:, 0].tolist(), "next_state": self.state[:, 0].tolist(),
                               "info": copy.deepcopy(info)})
        # Both agent dones are TEAM done. Current-state activity is a separate
        # transition field; first capture must never create a team terminal.
        return obs, state, [[reward], [reward]], [self._terminated] * 2, [copy.deepcopy(info), copy.deepcopy(info)], available

    def get_avail_actions(self):
        return None

    def close(self):
        pass

