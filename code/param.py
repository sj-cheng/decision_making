
import os
import numpy as np 

class Param: 

	def __init__(self):

		# 
		self.parallel_on = False
		self.num_trials = 50

		# names 
		self.problem_name = "example8" # e.g. example1, example2, example3, ...
		self.solver_name = "C_PUCT_V1" # e.g. Empty, DARE, PUCT_V0, C_PUCT_V0, PUCT_V1, ...
		self.value_oracle_name = "deterministic" # ["deterministic","gaussian"]
		self.policy_oracle_name = "gaussian" # ["deterministic","gaussian"]

		# Model selection.  Set model_family to "v2" or "v3" and choose an
		# explicit complete iteration.  Use None for model_iteration only when
		# you intentionally want the newest complete checkpoint automatically.
		oracles_on = True
		model_family = "v3"
		model_iteration = None

		n = 4 # num robots 
		model_roots = {
			"v2": os.path.abspath(os.path.join(os.path.dirname(__file__), "../current/memory_observation_v2/models")),
			"v3": os.path.abspath(os.path.join(os.path.dirname(__file__), "../current/memory_observation_v3/models")),
		}
		if model_family not in model_roots:
			raise ValueError("model_family must be one of: {}".format(sorted(model_roots)))
		dirname = model_roots[model_family]
		if model_iteration is None:
			complete_iterations = []
			for candidate in range(1000):
				if (
					os.path.exists(os.path.join(dirname, "model_value_l{}.pt".format(candidate)))
					and all(os.path.exists(os.path.join(dirname, "model_policy_l{}_i{}.pt".format(candidate, i))) for i in range(n))
				):
					complete_iterations.append(candidate)
			if not complete_iterations:
				raise FileNotFoundError("no complete model found under {}".format(dirname))
			l = max(complete_iterations)
		else:
			l = int(model_iteration)
			required = [os.path.join(dirname, "model_value_l{}.pt".format(l))]
			required += [os.path.join(dirname, "model_policy_l{}_i{}.pt".format(l, i)) for i in range(n)]
			missing = [path for path in required if not os.path.exists(path)]
			if missing:
				raise FileNotFoundError("incomplete model {} l{}; missing: {}".format(model_family, l, missing))
		# Keep the resolved checkpoint visible to callers and debugging output.
		self.model_family = model_family
		self.model_root = dirname
		self.model_iteration = l
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
		self.detailed_visualization_on = False
		# True: MINCO quintic trajectory planning; False: direct displacement-to-velocity propagation
		self.use_minco_dynamics = False

		# solver settings 
		self.number_simulations = 1000
		self.search_depth = 100
		self.C_pw = 2.0							# 每个节点最多能扩展多少个子节点
		self.alpha_pw = 0.5						# 访问次数增加时，子节点上限增长有多块 
		self.C_exp = 1.0						# 探索项的权重，越大越倾向于探索
		self.alpha_exp = 0.25					# 控制探索奖励随 N_parent 增长的速度
		self.beta_policy = 0.95					# 节点使用策略网络的概率
		self.beta_value = 0.1					# 使用价值网络的概率
		self.vis_on = False

	def to_dict(self):
		return self.__dict__
