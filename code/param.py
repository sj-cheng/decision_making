
import numpy as np 

class Param: 

	def __init__(self):

		# 
		self.parallel_on = False
		self.num_trials = 100 

		# names 
		self.problem_name = "example8" # e.g. example1, example2, example3, ...
		self.solver_name = "C_PUCT_V1" # e.g. Empty, DARE, PUCT_V0, C_PUCT_V0, PUCT_V1, ...
		self.value_oracle_name = "deterministic" # ["deterministic","gaussian"]
		self.policy_oracle_name = "gaussian" # ["deterministic","gaussian"]

		# oracles 
		oracles_on =True
		dirname = "../current/models"
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
		self.movie_on =True	
		self.pretty_plot_on = True
		# Toggle dense MINCO diagnostic pages in run.pdf.
		self.detailed_visualization_on = True

		# solver settings 
		self.number_simulations = 1000
		self.search_depth = 100
		self.C_pw = 2.0							# 每个节点最多能扩展多少个子节点
		self.alpha_pw = 0.5						# 访问次数增加时，子节点上限增长有多块 
		self.C_exp = 1.0						# 探索项的权重，越大越倾向于探索
		self.alpha_exp = 0.25					# 控制探索奖励随 N_parent 增长的速度
		self.beta_policy = 1					# 节点使用策略网络的概率
		self.beta_value = 1					# 使用价值网络的概率
		self.vis_on = False

	def to_dict(self):
		return self.__dict__
