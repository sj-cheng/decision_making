
# standard 
import numpy as np 

# custom 
from solvers.solver import Solver 

class PolicySolver(Solver):
	supports_team_policy = True

	def __init__(self,policy_oracle,stochastic=False):
		self.policy_oracle = policy_oracle
		# A policy-only controller is an executable policy, so its default
		# output is the Gaussian mean.  Search expansion and state-generation
		# code can opt into sampling explicitly when exploration is wanted.
		self.stochastic = bool(stochastic)
		self.solver_name = "NeuralNetwork"

	def policy(self,problem,root_state,team=None):
		action = np.zeros((problem.action_dim,1))
		team = None if team is None else set(int(robot) for robot in team)
		for robot in range(problem.num_robots):
			if team is not None and robot not in team:
				continue
			if hasattr(problem, "is_active") and not problem.is_active(root_state, robot):
				continue
			robot_action_idxs = problem.action_idxs[robot] 
			policy = self.policy_oracle[robot]
			try:
				action_value = policy.eval(
					problem, root_state, robot, stochastic=self.stochastic
				)
			except TypeError:
				# Deterministic policy implementations do not expose the flag.
				action_value = policy.eval(problem, root_state, robot)
			action[robot_action_idxs,0] = action_value.squeeze()
		return action 
