import numpy as np

from solvers.solver import Solver


class RandomSolver(Solver):

	def __init__(self):
		super(RandomSolver, self).__init__()
		self.solver_name = "Random"

	def policy(self, problem, state):
		return np.asarray(problem.sample_action(), dtype=float).reshape((problem.action_dim, 1))
