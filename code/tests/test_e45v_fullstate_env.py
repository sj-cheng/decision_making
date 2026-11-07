"""Transition-level contracts for the privileged diagnostic task interface."""
import json
from pathlib import Path

import numpy as np
import pytest

from learning.e45v_fullstate_env import (FullStatePursuitEnv, CENT_DIM, OBS_DIM,
                                          ENCODING, state_scale)
from run_e45_team_pilot import FixedKernelTargets
from corner_estimator_scenarios import original_model
from solvers.frozen_d_search_interface import PublicMap


ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def path():
    config = json.loads((ROOT / "current/e45m_hasac_task_20261101/experiment.json").read_text())
    return config["entries"][0]["path"]


def make_env(path, **kwargs):
    env = FullStatePursuitEnv({"case_paths": [path], "record_trace": True, **kwargs})
    env.seed(17)
    env.reset(case_path=path, world="AB", episode_seed=17)
    return env


def test_full_state_and_modes_are_visible_without_faking_reports(path):
    env = make_env(path)
    obs, central, avail = env._observation()
    assert obs.shape == (2, OBS_DIM) == (2, 130)
    assert central.shape == (2, CENT_DIM) == (2, 127)
    assert avail is None and env.initial_remaining == 70
    np.testing.assert_allclose(obs[0, :53], env.state[:, 0] / state_scale(), atol=2e-7)
    assert all(all(q is None for q in row) for row in env.public_obs()["reports"])
    old = obs.copy()
    env.other.modes[0] = (env.other.modes[0] + 1) % 9
    new = env._observation()[0]
    assert np.array_equal(old[:, :95], new[:, :95])
    assert not np.array_equal(old[:, 95:113], new[:, 95:113])
    np.testing.assert_array_equal(new[:, -1], [1., 1.])
    # Encoding excludes the opponent's RNG state and does not advance it.
    rngs = [r.copy() for r in env.other.rng]
    env._observation()
    for a, b in zip(rngs, env.other.rng): np.testing.assert_array_equal(a, b)


def test_compact_model_parameters_reconstruct_the_entire_fixed_graph(path):
    from l45_corner_shared_20261025.multimodal_estimator import make_graph
    from l45_corner_shared_20261025.corner_estimator import map_probabilities
    env = make_env(path)
    public = PublicMap(env.boxes, 1.)
    for graph in env.other.graphs:
        anchor = graph[0][int(graph[5])]
        rebuilt = list(make_graph(anchor, env.boxes))
        probabilities, _, _ = map_probabilities(rebuilt[0], rebuilt[1], rebuilt[2], anchor,
                                                env.boxes, public.vertices, public.dist)
        for index in (0, 1, 2): np.testing.assert_array_equal(rebuilt[index], graph[index])
        np.testing.assert_array_equal(probabilities, graph[3])


def test_expanded_support_has_all_edges_and_correct_marginal_probability(path):
    env = make_env(path, include_transition_kernel=True)
    obs, central, _ = env._observation()
    assert obs.shape == (2, 522) and central.shape == (2, 519)
    support = env.model_support()
    for j, graph in enumerate(env.other.graphs):
        index, mode = env.other.ids[j], env.other.modes[j]
        count = int(graph[2][index])
        assert support[j, :, 3].sum() == count
        assert support[j, :, 2].sum() == pytest.approx(1.)
        np.testing.assert_allclose(support[j, :count, :2] * 10., graph[0][graph[1][index, :count]])
        expected = .9 * graph[3][index, :count, mode] + .1 * graph[3][index, :count].mean(axis=1)
        np.testing.assert_allclose(support[j, :count, 2], expected)


def test_physics_and_fixed_opponent_match_existing_episode_order(path):
    from learning.e45g_ppo import safe_action
    env = make_env(path)
    case = json.loads(Path(path).read_text())
    reference = original_model(env.boxes, env.speed)
    other = FixedKernelTargets(case, 17)
    state = np.asarray(case["worlds"][0]["initial"], dtype=np.float64).reshape(-1, 1).copy()
    state[:4, 0] = other.positions().ravel()
    rng = np.random.default_rng(904)
    for _ in range(12):
        raw = rng.uniform(-1., 1., (2, 2))
        ap = state[reference.active_idxs[2:], 0] > .5
        ae = state[reference.active_idxs[:2], 0] > .5
        physical, _ = safe_action(state[4:8, 0].reshape(2, 2), raw * ap[:, None],
                                  ap, env.speed, env.boxes)
        enemy = other.actions(ae)
        enemy[~ae] = 0.
        joint = np.r_[enemy.ravel(), np.clip(physical, -env.speed, env.speed).ravel()][:, None]
        state = reference.step(state, joint, 1.)
        result = env.step(raw)
        np.testing.assert_array_equal(env.state, state)
        for x, y in zip(env.other.rng, other.rng): np.testing.assert_array_equal(x, y)
        assert not any(result[3])


def test_first_capture_does_not_terminate_and_retired_robots_do_not_move(path):
    env = make_env(path)
    # Engineering-only controlled state; physical transition/capture logic is
    # still the original Example8. Keep fixed targets stationary for this test.
    env.other.actions = lambda active: np.zeros((2, 2))
    env.state[4:6, 0] = env.state[:2, 0] + [.05, 0.]
    env.phi = env.potential()
    row = env.step(np.zeros((2, 2)))
    assert not any(row[3])
    assert row[4][0]["new_captures"] == 1
    assert row[4][0]["transition_active_mask"] == [1., 1.]
    assert sum(row[4][0]["next_active_mask"]) == 1
    retired_p = int(np.flatnonzero(~env.active_p)[0])
    retired_e = int(np.flatnonzero(~env.active_e)[0])
    old_p = env.state[4+2*retired_p:6+2*retired_p, 0].copy()
    old_e = env.state[2*retired_e:2*retired_e+2, 0].copy()
    row = env.step(np.ones((2, 2)) * .01)
    assert not any(row[3])
    assert row[4][0]["transition_active_mask"][retired_p] == 0.
    np.testing.assert_array_equal(env.state[4+2*retired_p:6+2*retired_p, 0], old_p)
    np.testing.assert_array_equal(env.state[2*retired_e:2*retired_e+2, 0], old_e)
    assert row[4][0]["normalized_intent"][retired_p] == [0., 0.]
    live_p, live_e = int(np.flatnonzero(env.active_p)[0]), int(np.flatnonzero(env.active_e)[0])
    env.state[4+2*live_p:6+2*live_p, 0] = env.state[2*live_e:2*live_e+2, 0] + [.05, 0.]
    env.phi = env.potential()
    row = env.step(np.zeros((2, 2)))
    assert all(row[3]) and row[4][0]["episode"]["double_capture"]
    assert row[4][0]["next_phi"] == 0.
    assert np.all(row[0][:, -1] == 0.)


def test_true_horizon_terminal_and_potential_telescoping(path):
    env = make_env(path)
    initial_remaining = env.initial_remaining
    total = 0.
    while True:
        row = env.step(np.zeros((2, 2)))
        total += row[2][0][0]
        assert "bad_transition" not in row[4][0]
        if all(row[3]): break
    assert env.decisions == initial_remaining
    assert env.state[8, 0] == 80
    assert row[4][0]["episode"]["reason"] == "timeout"
    assert row[4][0]["true_terminal"] and row[4][0]["next_phi"] == 0.
    penalty = .05 * sum(frame["info"]["blocked_fraction"] for frame in env.trace)
    assert total == pytest.approx(env.raw_utility - env.initial_phi - penalty, abs=1e-10)
    with pytest.raises(RuntimeError, match="Step after terminal"):
        env.step(np.zeros((2, 2)))

