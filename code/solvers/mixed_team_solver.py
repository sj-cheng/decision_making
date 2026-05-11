import copy
import numpy as np
import time

from solvers.c_puct import C_PUCT
from solvers.random_solver import RandomSolver
from solvers.solver import Solver


class MixedTeamSolver(Solver):

	DEFAULT_METHOD_SETTINGS = {
		"random": {
			"number_simulations": 0,
			"search_depth": 0,
			"beta_policy": 0.0,
			"beta_value": 0.0,
		},
		"search_only": {
			"number_simulations": 2000,
			"search_depth": 100,
			"beta_policy": 0.0,
			"beta_value": 0.0,
		},
		"learning_only": {
			"number_simulations": 1,
			"search_depth": 1,
			"beta_policy": 1.0,
			"beta_value": 1.0,
		},
		"search_learning": {
			"number_simulations": 1000,
			"search_depth": 100,
			"beta_policy": 0.8,
			"beta_value": 0.8,
		},
	}

	def __init__(
		self,
		policy_oracle,
		value_oracle,
		team_methods=None,
		team_method_settings=None,
		number_simulations=1000,
		search_depth=100,
		C_pw=2.0,
		alpha_pw=0.5,
		C_exp=1.0,
		alpha_exp=0.25,
		beta_policy=0.8,
		beta_value=0.8,
		vis_on=False,
		c_puct_solver_name="C_PUCT_V1",
	):
		super(MixedTeamSolver, self).__init__()
		self.solver_name = "MixedTeam"
		self.policy_oracle = policy_oracle
		self.value_oracle = value_oracle
		self.team_methods = self._normalize_team_methods(team_methods)
		self.method_settings = self._build_method_settings(
			team_method_settings,
			number_simulations,
			search_depth,
			beta_policy,
			beta_value,
		)
		self.common_settings = {
			"C_pw": C_pw,
			"alpha_pw": alpha_pw,
			"C_exp": C_exp,
			"alpha_exp": alpha_exp,
			"vis_on": vis_on,
			"solver_name": c_puct_solver_name,
		}
		self.paper_tree_density_on = False
		self.paper_tree_topk_on = False
		self.paper_tree_capture_once = True
		self.paper_tree_capture_step = 0
		self.paper_tree_capture_turn = None
		self.current_step = None
		self.team_solvers = {}
		self.last_team_decision_times = {}

	def _normalize_team_methods(self, team_methods):
		default = {
			"evaders": "learning_only",
			"pursuers": "search_learning",
		}
		if team_methods is not None:
			default.update(team_methods)
		for team_name, method in default.items():
			if method not in self.DEFAULT_METHOD_SETTINGS:
				valid = ", ".join(sorted(self.DEFAULT_METHOD_SETTINGS.keys()))
				raise ValueError("Unknown {} method '{}'. Valid methods: {}".format(team_name, method, valid))
		return default

	def _build_method_settings(
		self,
		team_method_settings,
		number_simulations,
		search_depth,
		beta_policy,
		beta_value,
	):
		settings = copy.deepcopy(self.DEFAULT_METHOD_SETTINGS)
		settings["search_learning"].update({
			"number_simulations": number_simulations,
			"search_depth": search_depth,
			"beta_policy": beta_policy,
			"beta_value": beta_value,
		})
		if team_method_settings is not None:
			for method, overrides in team_method_settings.items():
				if method not in settings:
					valid = ", ".join(sorted(settings.keys()))
					raise ValueError("Unknown team method setting '{}'. Valid methods: {}".format(method, valid))
				settings[method].update(overrides)
		return settings

	def _require_teams(self, problem):
		if not hasattr(problem, "evaders") or not hasattr(problem, "pursuers"):
			raise ValueError("MixedTeamSolver requires problem.evaders and problem.pursuers.")
		return {
			"evaders": list(problem.evaders),
			"pursuers": list(problem.pursuers),
		}

	def _masked_policy_oracle(self, team):
		masked = [None for _ in range(len(self.policy_oracle))]
		for robot in team:
			if robot < len(self.policy_oracle):
				masked[robot] = self.policy_oracle[robot]
		return masked

	def _make_team_solver(self, team_name, team):
		method = self.team_methods[team_name]
		if method == "random":
			return RandomSolver()

		method_settings = self.method_settings[method]
		policy_oracle = self._masked_policy_oracle(team)
		if method_settings["beta_policy"] <= 0.0:
			policy_oracle = [None for _ in range(len(policy_oracle))]
		value_oracle = self.value_oracle if method_settings["beta_value"] > 0.0 else None

		solver = C_PUCT(
			policy_oracle=policy_oracle,
			value_oracle=value_oracle,
			search_depth=method_settings["search_depth"],
			number_simulations=method_settings["number_simulations"],
			C_pw=self.common_settings["C_pw"],
			alpha_pw=self.common_settings["alpha_pw"],
			C_exp=self.common_settings["C_exp"],
			alpha_exp=self.common_settings["alpha_exp"],
			beta_policy=method_settings["beta_policy"],
			beta_value=method_settings["beta_value"],
			vis_on=self.common_settings["vis_on"],
			solver_name=self.common_settings["solver_name"],
		)
		for attr in [
			"paper_tree_density_on",
			"paper_tree_topk_on",
			"paper_tree_capture_once",
			"paper_tree_capture_step",
			"paper_tree_capture_turn",
		]:
			setattr(solver, attr, getattr(self, attr))
		return solver

	def _get_team_solver(self, team_name, team):
		if team_name not in self.team_solvers:
			self.team_solvers[team_name] = self._make_team_solver(team_name, team)
		return self.team_solvers[team_name]

	def _copy_team_action(self, problem, state, target_action, source_action, team):
		for robot in team:
			robot_action_idxs = problem.action_idxs[robot]
			target_action[robot_action_idxs, 0] = source_action[robot_action_idxs, 0]
			if hasattr(problem, "is_active") and not problem.is_active(state, robot):
				target_action[robot_action_idxs, 0] = 0.0

	def _team_policy(self, problem, state, team_name, team):
		solver = self._get_team_solver(team_name, team)
		if hasattr(solver, "current_step"):
			solver.current_step = self.current_step
		if isinstance(solver, C_PUCT):
			result = solver.search(problem, state, turn=int(team[0]))
			action = np.zeros((problem.action_dim, 1))
			if result.success:
				action[:, 0] = np.asarray(result.best_action).reshape(-1)
			return action
		return solver.policy(problem, state)

	def policy(self, problem, state):
		teams = self._require_teams(problem)
		action = np.zeros((problem.action_dim, 1))
		self.last_team_decision_times = {}
		for team_name in ["evaders", "pursuers"]:
			team = teams[team_name]
			decision_start = time.perf_counter()
			team_action = self._team_policy(problem, state, team_name, team)
			self.last_team_decision_times[team_name] = time.perf_counter() - decision_start
			self._copy_team_action(problem, state, action, team_action, team)
		return action
