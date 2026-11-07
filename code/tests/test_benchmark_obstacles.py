"""Check obstacle maps across initialization, simulation, search, and resume."""

import csv
import json
import os
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from benchmark_baselines import (
    build_solver, check_resume_signature, configure_problem, generate_initial_states,
    problem_metadata, run_episode, summarize,
    write_records,
)


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))


class FixedAction:
    def __init__(self, action):
        self.action = action

    def policy(self, problem, state):
        return self.action


class ObstacleBenchmarkTest(unittest.TestCase):
    def collision_setup(self, robot=0):
        problem = configure_problem("two")
        state = np.zeros((problem.state_dim, 1))
        state[problem.active_idxs, 0] = 1.0
        for i, position in enumerate(((-8, 7), (8, 7), (-8, -7), (8, -7))):
            state[problem.state_idxs[i], 0] = position
        # One legal step into the lower-left obstacle.
        state[problem.state_idxs[robot], 0] = (-4.5, -2.5)
        problem._update_observation_memory(state, state, 0.0)
        action = np.zeros((problem.action_dim, 1))
        action[problem.action_idxs[robot][0], 0] = 0.75
        return problem, state, action

    def test_paired_initial_states_are_legal_and_repeatable(self):
        for layout in ("none", "two", "four"):
            problem = configure_problem(layout)
            states = generate_initial_states(problem, [1000, 1001], 8)
            repeated = generate_initial_states(configure_problem(layout), [1000, 1001], 8)
            np.testing.assert_array_equal(states, repeated)
            for batch in states:
                for state in batch:
                    self.assertFalse(problem.is_terminal(state))
                    self.assertTrue(all(problem.has_spawn_clearance(state, robot)
                                        for robot in range(problem.num_robots)))

    def test_collision_is_reported_for_the_responsible_team(self):
        for robot, role in ((0, "evader"), (2, "pursuer")):
            problem, state, action = self.collision_setup(robot)
            self.assertTrue(problem.is_valid(state))
            metrics = run_episode(problem, FixedAction(action), state)
            self.assertEqual(metrics["steps"], 1)
            self.assertEqual(metrics["obstacle_collision"], 1)
            self.assertEqual(metrics[role + "_obstacle_collision"], 1)
            self.assertEqual(metrics["out_of_bounds"], 0)
            self.assertEqual(metrics["double_capture"], 0)
            self.assertEqual(summarize([metrics])["obstacle_collision"]["mean"], 1)

    def test_cpp_observations_and_collision_reward_match_simulator(self):
        problem, state, action = self.collision_setup()
        search = build_solver(problem, "Search Only", ROOT, 5)
        cpp = search._problem_wrapper_for(problem)
        for robot in range(problem.num_robots):
            np.testing.assert_allclose(
                np.asarray(cpp.policy_encoding(state, robot)).reshape(-1),
                problem.policy_encoding(state, robot).reshape(-1), atol=2e-5)
        np.testing.assert_allclose(cpp.normalized_reward(state, action).reshape(-1),
                                   problem.reward(state, action).reshape(-1), atol=2e-5)
        # The same move in the old empty map is legal and has no collision penalty.
        empty = configure_problem()
        self.assertTrue(empty.is_valid(empty.step(state, action, empty.dt)))
        self.assertLess(float(problem.reward(state, action)[0, 0]),
                        float(empty.reward(state, action)[0, 0]))

    def test_resume_rejects_different_maps_and_models_but_allows_workers(self):
        with tempfile.TemporaryDirectory() as output:
            os.makedirs(os.path.join(output, "chunks"))
            with open(os.path.join(output, "chunks", "test.seed1000.ep0.csv"), "w") as handle:
                handle.write("seed,episode\n1000,0\n")
            saved = problem_metadata(configure_problem("two"), "two")
            saved.update(model_dir="models_v7", workers=4)
            with open(os.path.join(output, "config.json"), "w") as handle:
                json.dump(saved, handle)
            check_resume_signature(output, dict(saved, workers=80))
            with self.assertRaises(SystemExit):
                check_resume_signature(output, dict(saved, model_dir="other_models"))
            with self.assertRaises(SystemExit):
                check_resume_signature(output, problem_metadata(configure_problem(), "none"))

    def test_legacy_no_obstacle_chunks_can_be_resumed_and_merged(self):
        with tempfile.TemporaryDirectory() as output:
            os.makedirs(os.path.join(output, "chunks"))
            chunk = os.path.join(output, "chunks", "test.seed1000.ep0.csv")
            with open(chunk, "w") as handle:
                handle.write("seed,episode\n1000,0\n")
            legacy = {"benchmark_protocol": "example8_global_no_obstacle_direct_v1"}
            with open(os.path.join(output, "config.json"), "w") as handle:
                json.dump(legacy, handle)
            check_resume_signature(output, problem_metadata(configure_problem(), "none"))
            problem, state, action = self.collision_setup()
            current = run_episode(problem, FixedAction(action), state)
            old = {key: value for key, value in current.items()
                   if "obstacle_collision" not in key and "out_of_bounds" not in key}
            # Keep the original combined diagnostic in the legacy record.
            old["collision_or_out_of_bounds"] = current["collision_or_out_of_bounds"]
            write_records(chunk, [old, current])
            with open(chunk) as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 2)
            self.assertEqual(rows[0]["obstacle_collision"], "")
            self.assertEqual(rows[1]["obstacle_collision"], "1")
            self.assertEqual(summarize([old, current])["obstacle_collision"]["n"], 1)


if __name__ == "__main__":
    unittest.main()
