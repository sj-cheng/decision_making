
# standard 
import numpy as np 

# custom 
from solvers.solver import Solver 

class PolicySolver(Solver):

	def __init__(self,policy_oracle):
		self.policy_oracle = policy_oracle
		self.solver_name = "NeuralNetwork"

	def policy(self,problem,root_state):
		action = np.zeros((problem.action_dim,1))
		for robot in range(problem.num_robots):
			robot_action_idxs = problem.action_idxs[robot] 
			action[robot_action_idxs,0] = self.policy_oracle[robot].eval(problem,root_state,robot).squeeze()
		return action 
