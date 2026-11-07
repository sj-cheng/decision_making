"""Meaningful observation/action/physical contracts for the new PPO bridge."""
import copy
import json
from pathlib import Path

import numpy as np
import pytest

from corner_estimator_scenarios import original_model
from learning.e45b_joint_network import KEYS
from learning.e45g_ppo import TeamBelief, safe_action
from learning.e45w_joint_env import PublicJointPursuitEnv, OBS_SHAPES, LATENT_GUARD
from run_e45_team_pilot import FixedKernelTargets
from run_frozen_d_closed_loop import legal_action
from solvers.root_audit_puct import RootPUCT

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def path():
    cfg = json.loads((ROOT / "current/e45m_hasac_task_20261101/experiment.json").read_text())
    return cfg["entries"][0]["path"]


def reset(path, opponent="fixed_D_kernel", world="AB", episode_seed=17, **args):
    env = PublicJointPursuitEnv({"case_paths": [path], "record_trace": True, **args})
    obs, info = env.reset(seed=19, options={"case_path": path, "world": world,
                         "opponent": opponent, "episode_seed": episode_seed})
    return env, obs, info


def test_public_packet_matches_single_root_exactly(path, monkeypatch):
    import solvers.e45b_joint_nte as joint
    monkeypatch.setattr(joint, "TeamBelief", TeamBelief)
    env, obs, info = reset(path)
    own = joint.Controller(env.boxes, env.speed, 17, use_policy=False, use_value=False, direct=True)
    for frame in env.case["history"][:-1]: own.observe(frame)
    current = env.public_obs(); own.observe(current)
    samples = np.zeros((2, 64, 2)); weights = np.zeros((2, 64))
    for j, est in enumerate(own.belief.estimators):
        samples[j] = est.particles; weights[j] = est.particle_weights
    own.remaining = current["remaining"]
    node = joint.Node(np.array(current["positions"]), current["active_p"], current["active_e"],
                      [None, None], list(own.belief.known), 0, list(own.records), samples, weights)
    expected = own.features(node)
    assert set(obs) == set(KEYS)
    for key in KEYS:
        assert obs[key].shape == OBS_SHAPES[key]
        np.testing.assert_array_equal(obs[key], np.asarray(expected[key], np.float32))
    assert env.observation_space.contains(obs)
    assert info["privileged_actor"] is False


def test_hidden_world_truth_does_not_enter_reset_observation(path):
    a, x, _ = reset(path, opponent="rootpuct", world="AB")
    b, y, _ = reset(path, opponent="rootpuct", world="BA")
    assert not np.array_equal(a.state[:4], b.state[:4])
    assert a.public_obs() == b.public_obs()
    for key in KEYS: np.testing.assert_array_equal(x[key], y[key])
    assert all(q is None for row in a.public_obs()["reports"] for q in row)


@pytest.mark.parametrize("opponent", ["fixed_D_kernel", "rootpuct"])
def test_same_physics_and_opponent_order_as_frozen_episode(path, opponent):
    env, _, _ = reset(path, opponent=opponent)
    p = original_model(env.boxes, env.speed)
    state = np.asarray(env.case["worlds"][0]["initial"], float).reshape(-1, 1).copy()
    if opponent == "fixed_D_kernel":
        other = FixedKernelTargets(env.case, 17); state[:4, 0] = other.positions().ravel()
    else:
        other = RootPUCT([None] * 4, None, 3000, 0., 0.); other.seeded_state(p, 34, 46)
    rng = np.random.default_rng(321)
    for _ in range(6):
        latent = rng.normal(size=4)
        ap, ae = state[p.active_idxs[2:], 0] > .5, state[p.active_idxs[:2], 0] > .5
        physical, _ = safe_action(state[4:8, 0].reshape(2, 2), np.tanh(latent.reshape(2, 2)) * ap[:, None],
                                  ap, env.speed, env.boxes)
        if opponent == "fixed_D_kernel": enemy = other.actions(ae)
        else: enemy = np.asarray(other.search(p, state.copy(), turn=0).best_action).reshape(-1)[:4].reshape(2, 2)
        for j in range(2):
            if not ae[j]: enemy[j] = 0.
            elif not legal_action(state[2*j:2*j+2, 0], enemy[j], 1., env.boxes): enemy[j] = 0.
        joint = np.r_[enemy.ravel(), np.clip(physical, -env.speed, env.speed).ravel()][:, None]
        state = p.step(state, joint, 1.)
        _, _, term, trunc, _ = env.step(latent)
        np.testing.assert_array_equal(env.state, state)
        assert not trunc
        if term: break


def test_first_capture_is_not_team_terminal_and_retired_action_stays_zero(path):
    env, _, _ = reset(path)
    # Engineer capture transitions through the real physical simulator.  Stop
    # the opponent for this unit test only; no formal dynamics is changed.
    env.other.actions = lambda active: np.zeros((2, 2))
    env.state[:8, 0] = [-7.7, -8., -8., 8., -8., -8., 8., 8.]
    obs, reward, terminal, truncated, info = env.step(np.zeros(4))
    assert not terminal and not truncated and info["new_captures"] == 1
    np.testing.assert_array_equal(env.active_p, [False, True])
    np.testing.assert_array_equal(env.active_e, [False, True])
    assert obs["agents"][0, 4] == 0
    retired = env.state[4:6].copy()
    env.step(np.array([3., -4., 0., 0.]))
    np.testing.assert_array_equal(env.state[4:6], retired)
    assert env.last_info["physical_action"][0] == [0., 0.]
    assert env.last_info["blocked_penalty"] == 0.


def test_true_horizon_and_potential_telescope(path):
    env, _, info = reset(path)
    start = env.start_step; remaining = 80 - start; rewards = []
    while True:
        _, reward, term, trunc, frame = env.step(np.zeros(4)); rewards.append(reward)
        assert not trunc and "TimeLimit.truncated" not in frame
        if term: break
    episode = frame["task_episode"]
    assert env.state[8, 0] == 80 or episode["double_capture"]
    assert episode["decisions"] <= remaining
    assert episode["start_step"] == start and frame["next_phi"] == 0.
    assert sum(rewards) == pytest.approx(episode["utility"] - episode["initial_phi"], abs=2e-6)
    with pytest.raises(RuntimeError): env.step(np.zeros(4))


def test_latent_action_is_not_clipped_and_invalid_input_fails(path):
    env, _, _ = reset(path)
    z = np.array([10., -10., 3., -3.])
    env.step(z)
    np.testing.assert_array_equal(env.last_info["latent_action"], z)
    np.testing.assert_allclose(env.last_info["normalized_intent"], np.tanh(z.reshape(2, 2)))
    with pytest.raises(ValueError): env.step(np.array([np.nan, 0., 0., 0.]))
    with pytest.raises(RuntimeError): env.step(np.full(4, LATENT_GUARD))


def test_readonly_root_action_can_be_retirement_masked_and_shielded(path):
    from types import SimpleNamespace
    env, _, _ = reset(path, opponent="rootpuct")
    # First produce the real 2v2 -> 1v1 capture transition, then receive
    # another read-only pybind-like result while one target is retired.
    env.state[:8, 0] = [-7.7, -8., -8., 8., -8., -8., 8., 8.]
    raw = np.array([100., 100., 100., 100., 0., 0., 0., 0.])
    raw.flags.writeable = False
    env.other.search = lambda *args, **kwargs: SimpleNamespace(best_action=raw)
    _, _, terminal, _, first = env.step(np.zeros(4))
    assert first["new_captures"] == 1 and not terminal
    _, _, _, _, info = env.step(np.zeros(4))
    assert info["opponent_action"][0] == [0., 0.]
    assert info["opponent_action"][1] == [0., 0.]
    assert info["opponent_shields"] == 1
    assert info["physical_action"][0] == [0., 0.]
    assert not raw.flags.writeable
