"""Task identity, paired statistics and cost-accounting checks for E45-W."""
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
import run_e45w_planning_evaluation as w


class PlanningEvaluationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        case = self.root / "case.json"; case.write_text(json.dumps({"map": []}))
        self.row = dict(id="case1", path=str(case), family="long_wall", pattern="test",
                        split="test", suite="ID")

    def tearDown(self): self.temp.cleanup()

    def test_seed_paired_across_arm_budget_and_opponent(self):
        tasks = [w.request(self.row, "AB", opponent, arm, seed, "wall", budget, "formal")
                 for opponent in ("fixed_D_kernel", "rootpuct")
                 for arm, seed in w.ARMS for budget in w.WALL_BUDGETS]
        self.assertEqual(len({r["seed"] for r in tasks}), 1)
        self.assertEqual(len({w.artifact_id(r) for r in tasks}), 30)
        other = w.request(self.row, "BA", "rootpuct", "U", 0, "wall", 200, "formal")
        self.assertNotEqual(tasks[0]["seed"], other["seed"])

    def test_simulation_mode_does_not_claim_wall_limit(self):
        task = w.request(self.row, "AB", "fixed_D_kernel", "U", 0, "sim", 128, "formal")
        self.assertEqual(task["fixed"], 128)
        self.assertFalse(task["budget_enforced"])
        self.assertEqual(task["latency_reference_threshold_ms"], 200)

    def test_formal_task_counts(self):
        cfg = {"test_entries": [self.row | dict(id=f"case{i}") for i in range(42)],
               "validation_entries": [self.row]}
        for mode in ("wall", "sim"):
            self.assertEqual(len(w.make_tasks(cfg, mode)), 2520)
        tasks = w.make_tasks(cfg, "sim", smoke=True)
        self.assertEqual(len(tasks), 20)
        self.assertTrue(all(r["limit"] == 2 and r["fixed"] == 2 for r in tasks))

    def test_cost_ratio_counts_policy_cpu_only(self):
        row = dict(valid_episode=True, latencies_ms=[200, 200], cpu_ms=[180, 180],
                   arm="U", simulations=14)
        result = w.timing([row])
        self.assertAlmostEqual(result["cpu_wall"], .9)
        self.assertEqual(result["simulations"], 14)

    def test_group_paired_bootstrap_does_not_inflate_seed_units(self):
        rows = []
        for group in ("one", "two"):
            for opponent in ("fixed_D_kernel", "rootpuct"):
                for value in (50, 100, 200):
                    for arm, seed in w.ARMS:
                        for world in ("AB", "BA"):
                            rows.append(dict(suite="ID", opponent=opponent, control_value=value,
                                arm=arm, group=group, family="long_wall", train_seed=seed,
                                valid_episode=True, double_capture=True, utility=.9 if arm == "SINGLE_PRIOR" else .7,
                                completion_cost=40, latencies_ms=[20], cpu_ms=[19], construction_ms=1,
                                history_ms=[1], latency_reference_threshold_ms=value,
                                simulations=16, simulated_transitions=100, world=world))
        result = w.summarize(rows, self.root, "wall")
        self.assertEqual(len(result["contrasts"]), 12)
        for contrast in result["contrasts"]:
            self.assertEqual(contrast["groups"], 2)
            self.assertAlmostEqual(contrast["utility_difference"], .2)

    def test_invalid_cell_withholds_comparisons(self):
        result = w.summarize([dict(valid_episode=False)], self.root, "wall")
        self.assertFalse(result["complete"])
        self.assertEqual(result["contrasts"], [])

    def test_shared_g_filter_registration_preserves_real_opponent(self):
        import run_e45_team_pilot as pilot
        import solvers.e45_team_adapter as adapter
        import solvers.e45b_joint_nte as joint
        from learning.e45g_ppo import TeamBelief
        from learning.e45w_joint_env import TeamBelief as PPOBelief
        original_target_filter = pilot.TeamBelief
        w.configure(self.root)
        self.assertIs(adapter.TeamBelief, TeamBelief)
        self.assertIs(joint.TeamBelief, TeamBelief)
        self.assertIs(w.base.TeamBelief, TeamBelief)
        self.assertIs(PPOBelief, TeamBelief)
        self.assertIs(pilot.TeamBelief, original_target_filter)

    def test_normal_public_history_matches_ppo_and_old_f_arithmetic(self):
        from learning.e45w_joint_env import PublicFeatureState
        from solvers.e45f_support_refinement import RefinedTeamBelief
        cfg = w.read(w.ROOT / "current/e45m_hasac_task_20261101/experiment.json")
        case = w.read(cfg["entries"][0]["path"])
        w.configure(self.root)
        own = w.factory(case, dict(arm="U", seed=17, budget_ms=200))
        ppo = PublicFeatureState(case["map"], case["speed"], 17)
        old_f = RefinedTeamBelief(case["map"], 17)
        for obs in case["history"]:
            own.observe(obs); ppo.observe(obs); old_f.update(obs)
            self.assertEqual(own.belief.signature(), ppo.belief.signature())
            self.assertEqual(own.belief.signature(), old_f.signature())
            for a, b in zip(own.belief.estimators, ppo.belief.estimators):
                np.testing.assert_array_equal(a.rng, b.rng)
                if a.weights is not None: np.testing.assert_array_equal(a.weights, b.weights)
        self.assertEqual(own.belief.refinements, [])

    def test_public_support_fallback_matches_ppo(self):
        from learning.e45w_joint_env import PublicFeatureState
        fixture = w.read(Path(__file__).parent / "fixtures/e45f_empty_support_public.json")
        case = dict(map=fixture["boxes"], speed=1.1)
        w.configure(self.root)
        own = w.factory(case, dict(arm="U", seed=fixture["seed"], budget_ms=200))
        ppo = PublicFeatureState(case["map"], case["speed"], fixture["seed"])
        for obs in fixture["history"]:
            own.observe(obs); ppo.observe(obs)
            self.assertEqual(own.belief.signature(), ppo.belief.signature())
        self.assertTrue(own.belief.refinements)
        self.assertEqual(own.belief.refinements, ppo.belief.refinements)


if __name__ == "__main__": unittest.main()
