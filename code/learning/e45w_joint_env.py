"""Public-observation joint-action Gymnasium interface for E45-W.

The physical simulator and fixed opponents are the same ones used by the
frozen E45 episode driver.  Only the observation/action/training interface is
new: Single's structured public packet, four latent Gaussian coordinates,
and potential-only team reward shaping.  No hidden state enters observations.
"""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import gymnasium as gym
from gymnasium import spaces
import numpy as np

from corner_estimator_scenarios import original_model
from e45g_training_acceleration import public_geometry_cache
from learning.e45b_joint_network import KEYS, public_features
from learning.e45g_ppo import TeamBelief, safe_action
from run_e45_team_pilot import FixedKernelTargets
from run_frozen_d_closed_loop import legal_action
from solvers.e45_team_adapter import public_observation
from solvers.e45b_joint_nte import coverage_potential, simultaneous_capture, TIME_COST
from solvers.frozen_d_search_interface import PublicMap
from solvers.root_audit_puct import RootPUCT


OBS_SHAPES = {
    "maps": (8, 4), "map_mask": (8,), "agents": (2, 6),
    "particles": (2, 64, 3), "weights": (2, 64), "history": (16, 14),
    "history_mask": (16,), "public": (8,),
}
OPPONENTS = ("fixed_D_kernel", "rootpuct")
LATENT_GUARD = 1e6


class PublicFeatureState:
    """Single's exact normal-path public feature pipeline, without a tree."""
    def __init__(self, boxes, speed, seed):
        self.public = PublicMap(boxes, speed)
        # G adds only the published public-history fallback for an exhausted
        # finite lattice.  Normal-path arithmetic/export RNG matches F/Single.
        self.belief = TeamBelief(boxes, seed)
        self.records = []

    def observe(self, obs):
        self.belief.update(obs)
        self.records.append(copy.deepcopy(obs))

    def features(self, obs):
        samples = np.zeros((2, 64, 2)); weights = np.zeros((2, 64))
        for j, estimator in enumerate(self.belief.estimators):
            if obs["active_e"][j]:
                samples[j] = estimator.particles
                weights[j] = estimator.particle_weights
        return public_features(self.public.boxes, np.asarray(obs["positions"]),
            obs["active_p"], obs["active_e"], samples, weights,
            self.belief.known, self.records, self.public.speed, obs["remaining"])


class PublicJointPursuitEnv(gym.Env):
    """Two pursuers controlled by one four-dimensional latent action.

    Constructor args: ``case_paths`` (list of paths or manifest entries),
    ``opponents`` (default balanced sampling of both fixed opponents),
    ``root_simulations`` (formal default 3000), optional ``record_trace``.

    reset options can specify case_path/world/opponent/episode_seed and the
    public estimator's controller_seed.  Default controller_seed=episode_seed
    matches the frozen E45 episode driver.  No Torch/global RNG is reset.
    """
    metadata = {"render_modes": []}
    n_agents = 2

    def __init__(self, args=None, *, case_paths=None):
        super().__init__()
        self.args = copy.deepcopy(args or {})
        if case_paths is not None:
            self.args["case_paths"] = list(case_paths)
        paths = self.args.get("case_paths", [])
        self.case_paths = [str(Path(p["path"] if isinstance(p, dict) else p)) for p in paths]
        if not self.case_paths:
            raise ValueError("case_paths must be nonempty")
        self.opponents = tuple(self.args.get("opponents", OPPONENTS))
        if not self.opponents or not set(self.opponents) <= set(OPPONENTS):
            raise ValueError("Only fixed_D_kernel and rootpuct opponents are defined")
        self.root_simulations = int(self.args.get("root_simulations", 3000))
        if self.root_simulations < 1:
            raise ValueError("RootPUCT needs a positive registered simulation count")
        if self.args.get("belief_evidence_dir"):
            import e45c_belief_diagnostics as diagnostic
            diagnostic.EVIDENCE = Path(self.args["belief_evidence_dir"])
            diagnostic.EVIDENCE.mkdir(parents=True, exist_ok=True)
        # SB3 requires a finite Box.  This is only an engineering guard, never
        # an intended clipping transform; the policy asserts the same bound.
        self.action_space = spaces.Box(-LATENT_GUARD, LATENT_GUARD, shape=(4,), dtype=np.float32)
        self.observation_space = spaces.Dict({
            k: spaces.Box(-np.inf, np.inf, shape=OBS_SHAPES[k], dtype=np.float32) for k in KEYS
        })
        self._cases = {}
        self._terminated = True
        self.last_features = None
        self.last_info = {}
        self.trace = []

    def _case(self, path):
        if path not in self._cases:
            self._cases[path] = json.loads(Path(path).read_text())
        return self._cases[path]

    @property
    def active_p(self):
        return self.state[self.problem.active_idxs[2:], 0] > .5

    @property
    def active_e(self):
        return self.state[self.problem.active_idxs[:2], 0] > .5

    def public_obs(self):
        return public_observation(int(self.state[8, 0]), self.state[4:8, 0].reshape(2, 2),
            self.state[:4, 0].reshape(2, 2), self.active_p, self.active_e, self.boxes)

    def _encode(self, obs):
        features = self.own.features(obs)
        result = {k: np.asarray(features[k], dtype=np.float32).copy() for k in KEYS}
        for key, shape in OBS_SHAPES.items():
            if result[key].shape != shape or not np.isfinite(result[key]).all():
                raise RuntimeError("Malformed public observation: " + key)
        self.last_features = result
        return {k: v.copy() for k, v in result.items()}

    def _terminal_features(self, obs):
        # With both pursuers retired there is no observer to advance the
        # filter.  This terminal-only packet has actual public poses/clock and
        # lifecycle flags, zero target mass, and no hidden target coordinates.
        self.own.records.append(copy.deepcopy(obs))
        features = public_features(self.own.public.boxes, np.asarray(obs["positions"]),
            obs["active_p"], obs["active_e"], np.zeros((2, 64, 2)), np.zeros((2, 64)),
            [None, None], self.own.records, self.speed, obs["remaining"])
        self.last_features = {k: np.asarray(features[k], dtype=np.float32) for k in KEYS}
        return {k: v.copy() for k, v in self.last_features.items()}

    def potential(self):
        if self._terminated:
            return 0.
        # Use the estimator's exact float64 particle exports, as Single's
        # root potential does; do not round-trip normalized float32 packets.
        samples = np.zeros((2, 64, 2)); weights = np.zeros((2, 64))
        for j, e in enumerate(self.own.belief.estimators):
            if self.active_e[j]:
                samples[j] = e.particles; weights[j] = e.particle_weights
        return float(.05 * coverage_potential(self.state[4:8, 0].reshape(2, 2),
            self.active_p, self.active_e, samples, weights, self.speed,
            self.own.public.vertices, self.own.public.dist, self.boxes))

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        options = options or {}
        path = str(options.get("case_path", self.case_paths[int(self.np_random.integers(len(self.case_paths)))]))
        self.case_path = path; self.case = self._case(path)
        ids = [w["id"] for w in self.case["worlds"]]
        self.world = str(options.get("world", ids[int(self.np_random.integers(len(ids)))]))
        selected = next((w for w in self.case["worlds"] if w["id"] == self.world), None)
        if selected is None:
            raise ValueError("Unknown world")
        self.opponent = str(options.get("opponent", self.opponents[int(self.np_random.integers(len(self.opponents)))]))
        if self.opponent not in OPPONENTS:
            raise ValueError("Unknown opponent")
        self.episode_seed = int(options.get("episode_seed", self.np_random.integers(1, 2**31 - 1)))
        self.controller_seed = int(options.get("controller_seed", self.episode_seed))
        self.boxes = np.asarray(self.case["map"], dtype=np.float64)
        self.speed = float(self.case["speed"])
        self.problem = original_model(self.boxes, self.speed)
        self.state = np.asarray(selected["initial"], dtype=np.float64).reshape(-1, 1).copy()
        with public_geometry_cache():
            if self.opponent == "fixed_D_kernel":
                self.other = FixedKernelTargets(self.case, self.episode_seed)
                self.state[:4, 0] = self.other.positions().ravel()
            else:
                self.other = RootPUCT([None] * 4, None, self.root_simulations, 0., 0.)
                self.other.seeded_state(self.problem, self.episode_seed + 17, self.episode_seed + 29)
            self.own = PublicFeatureState(self.boxes, self.speed, self.controller_seed)
            for public in self.case["history"][:-1]:
                self.own.observe(public)
            current = self.public_obs()
            self.own.observe(current)
        if not self.problem.is_valid(self.state) or self.problem.is_terminal(self.state):
            raise RuntimeError("Invalid/already-terminal handover")
        self._terminated = False
        self.start_step = int(self.state[8, 0]); self.initial_remaining = 80 - self.start_step
        self.decisions = 0; self.first_capture = None
        self.raw_utility = self.training_return = 0.
        self.blocked_total = self.active_actuator_steps = self.opponent_shields = 0
        self.trace = []
        self.phi = self.potential(); self.initial_phi = self.phi
        result = self._encode(current)
        info = {"case": self.case["id"], "world": self.world, "opponent": self.opponent,
                "episode_seed": self.episode_seed, "controller_seed": self.controller_seed,
                "start_step": self.start_step, "remaining": self.initial_remaining,
                "public_observation": current, "privileged_actor": False,
                "initial_state_sha256": hashlib.sha256(self.state.tobytes()).hexdigest()}
        self.last_info = info
        return result, copy.deepcopy(info)

    def step(self, action):
        if self._terminated:
            raise RuntimeError("Step after terminal requires reset")
        latent = np.asarray(action, dtype=np.float64)
        if latent.shape != (4,) or not np.isfinite(latent).all():
            raise ValueError("Expected finite four-dimensional unsquashed latent action")
        if np.any(np.abs(latent) >= LATENT_GUARD):
            raise RuntimeError("Latent action reached finite library guard; no silent clipping")
        old = self.state.copy(); ap, ae = self.active_p.copy(), self.active_e.copy()
        normalized = np.tanh(latent.reshape(2, 2)) * ap[:, None]
        physical, shortened = safe_action(old[4:8, 0].reshape(2, 2), normalized,
                                           ap, self.speed, self.boxes)
        # The simulator clips tiny float32-axis roundoff; record that exact
        # execution without concealing materially illegal oversized actions.
        physical = np.asarray(physical, dtype=np.float64)
        executed = np.clip(physical, -self.speed, self.speed)
        roundoff = float(np.max(np.abs(executed - physical)))
        if roundoff > 2e-7:
            raise RuntimeError("Non-roundoff speed violation")
        for i in range(2):
            if not legal_action(old[4+2*i:6+2*i, 0], executed[i], self.speed, self.boxes):
                raise RuntimeError("Illegal pursuer action")
            if not ap[i] and np.any(executed[i] != 0.):
                raise RuntimeError("Retired pursuer action is not zero")
        if self.opponent == "fixed_D_kernel":
            enemy = np.asarray(self.other.actions(ae), dtype=np.float64)
        else:
            # pybind may expose the search result as a read-only view.  The
            # frozen episode driver copies raw before retirement/shield edits.
            enemy = np.asarray(self.other.search(self.problem, old.copy(), turn=0).best_action).reshape(-1)[:4].reshape(2, 2).copy()
        shield = 0
        for j in range(2):
            if not ae[j]:
                enemy[j] = 0.
            elif not legal_action(old[2*j:2*j+2, 0], enemy[j], 1., self.boxes):
                if self.opponent == "fixed_D_kernel":
                    raise RuntimeError("Illegal fixed-D action")
                enemy[j] = 0.; shield += 1
        joint = np.r_[enemy.ravel(), executed.ravel()][:, None]
        self.state = self.problem.step(old, joint, 1.)
        if not self.problem.is_valid(self.state):
            raise RuntimeError("Invalid physical successor")
        np.testing.assert_allclose(self.state[:8], old[:8] + joint, atol=1e-10, rtol=0)
        nap, nae, _ = simultaneous_capture(self.state[4:8, 0].reshape(2, 2),
            self.state[:4, 0].reshape(2, 2), ap, ae, self.boxes)
        np.testing.assert_array_equal(nap, self.active_p)
        np.testing.assert_array_equal(nae, self.active_e)
        captures = int(ae.sum() - nae.sum()); complete = not nae.any()
        self._terminated = bool(self.problem.is_terminal(self.state))
        if self._terminated != (complete or self.state[8, 0] >= 80):
            raise RuntimeError("Unexpected task terminal")
        self.decisions += 1
        if captures and self.first_capture is None:
            self.first_capture = self.decisions
        public = self.public_obs()
        if not self.active_p.any():
            result = self._terminal_features(public)
        else:
            self.own.observe(public)
            result = self._encode(public)
        next_phi = 0. if self._terminated else self.potential()
        raw = float(complete) - TIME_COST
        reward = float(raw + next_phi - self.phi)
        before_phi, self.phi = self.phi, next_phi
        self.raw_utility += raw; self.training_return += reward
        self.blocked_total += int(shortened); self.active_actuator_steps += int(ap.sum())
        self.opponent_shields += shield
        info = {"privileged_actor": False, "raw_reward": raw, "phi": before_phi,
                "next_phi": next_phi, "public_observation": public,
                "latent_action": latent.tolist(), "normalized_intent": normalized.tolist(),
                "physical_action": executed.tolist(), "opponent_action": enemy.tolist(),
                "transition_active_mask": ap.astype(np.float32).tolist(),
                "next_active_mask": nap.astype(np.float32).tolist(), "new_captures": captures,
                "blocked_fraction": float(shortened / max(1, ap.sum())),
                "blocked_penalty": 0., "opponent_shields": shield,
                "public_belief_refinements": len(self.own.belief.refinements),
                "speed_roundoff_correction": roundoff, "true_terminal": self._terminated}
        if self._terminated:
            summary = {"case": self.case["id"], "world": self.world, "opponent": self.opponent,
                "episode_seed": self.episode_seed, "controller_seed": self.controller_seed,
                "start_step": self.start_step, "decisions": self.decisions,
                "utility": self.raw_utility, "training_return": self.training_return,
                "initial_phi": self.initial_phi, "captures": 2 - int(nae.sum()),
                "first_capture": self.first_capture, "double_capture": complete,
                "completion_cost": self.decisions if complete else self.initial_remaining,
                "reason": "double_capture" if complete else "timeout",
                "blocked_total": self.blocked_total,
                "active_actuator_steps": self.active_actuator_steps,
                "blocked_fraction": self.blocked_total / max(1, self.active_actuator_steps),
                "opponent_shields": self.opponent_shields,
                "public_belief_refinements": len(self.own.belief.refinements),
                "r": self.training_return, "l": self.decisions}
            info["episode"] = summary
            info["task_episode"] = copy.deepcopy(summary)
            residual = self.training_return - (self.raw_utility - self.initial_phi)
            if abs(residual) > 2e-6:
                raise RuntimeError("Public potential did not telescope at true terminal")
        self.last_info = info
        if self.args.get("record_trace", False):
            self.trace.append({"state": old[:, 0].tolist(), "next_state": self.state[:, 0].tolist(),
                               "info": copy.deepcopy(info)})
        return result, reward, self._terminated, False, copy.deepcopy(info)

    def close(self):
        pass


# Explicit aliases for evaluation/registration code; there is one task class.
JointPursuitEnv = PublicJointPursuitEnv

