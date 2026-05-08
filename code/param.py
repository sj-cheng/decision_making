
import numpy as np 

class Param: 

	def __init__(self):

		# 
		self.parallel_on = True
		self.num_trials = 100
		self.fixed_initial_conditions = True
		self.initial_seed = 20260506
		self.run_seed = 20260506

		# names 
		self.problem_name = "example8" # e.g. example1, example2, example3, ...
		self.solver_name = "MixedTeam" # e.g. Empty, DARE, PUCT_V0, C_PUCT_V0, PUCT_V1, MixedTeam, ...
		# self.solver_name = "C_PUCT_V1" # e.g. Empty, DARE, PUCT_V0, C_PUCT_V0, PUCT_V1, MixedTeam, ...
		self.value_oracle_name = "deterministic" # ["deterministic","gaussian"]
		self.policy_oracle_name = "gaussian" # ["deterministic","gaussian"]

		# mixed-team experiment settings
		# example8: evaders=[0,1], pursuers=[2,3]
		self.team_methods = {
			"evaders": "search_only",
			"pursuers": "search_learning",
		}
		# available methods: random, search_only, learning_only, search_learning
		self.team_method_settings = {
			"random": {
				"number_simulations": 0,
				"search_depth": 0,
				"beta_policy": 0.0,
				"beta_value": 0.0,
			},
			"search_only": {
				"number_simulations": 3000,
				"search_depth": 200,
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
				"beta_policy": 0.5,
				"beta_value": 0.5,
			},
		}

		# oracles 
		oracles_on =True
		dirname = "../current/historymodels/1818_4zhang"
		#dirname = "/home/sjc/NTE/decision_making/current/models"
		#dirname = "/home/ben/projects/decision_making/saved/example9"
		# dirname = "/home/ben/projects/decision_making/saved/example6"

		n = 4 # num robots 
		l =39 # learning iteration 
		if oracles_on:
			self.policy_oracle_paths = ["{}/model_policy_l{}_i{}.pt".format(dirname,l,i) for i in range(n)]	
			self.value_oracle_path = "{}/model_value_l{}.pt".format(dirname,l)
		else:	
			self.policy_oracle_paths = [None]  
			self.value_oracle_path = None 

		# settings
		self.movie_on =False	
		self.pretty_plot_on = False

		# solver settings 
		self.number_simulations = 2000
		self.search_depth = 100
		self.C_pw = 2.0							# 每个节点最多能扩展多少个子节点
		self.alpha_pw = 0.5						# 访问次数增加时，子节点上限增长有多块 
		self.C_exp = 1.0						# 探索项的权重，越大越倾向于探索
		self.alpha_exp = 0.25					# 控制探索奖励随 N_parent 增长的速度
		self.beta_policy = 1.0					# 节点使用策略网络的概率
		self.beta_value = 1.0				# 使用价值网络的概率
		self.vis_on = False

	def to_dict(self):
		return self.__dict__
