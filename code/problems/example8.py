# standard 
import numpy as np 
import matplotlib.patches as patches
import matplotlib.pyplot as plt

from typing_extensions import override
# custom 
from problems.problem import Problem
from util import sample_vector, contains
import plotter 

# 2d single integrator pursuit evasion 1v1   
class Example8(Problem):

	def __init__(self): 
		super(Example8,self).__init__()

		self.t0 = 0
		self.tf = 80
		self.dt = 1
		self.gamma = 1.0
		self.num_robots = 4 

		self.r_max = 1
		self.r_min = 0
		self.name = "example8"
		self.position_idx = np.arange(2) 
		self.state_control_weight = 1e-5 
		self.action_semantics = "relative_position_delta"
		self.low_level_controller_name = "relative_displacement_to_velocity"
		self.desired_distance = 1.0
		self.init_min_dist = 2.0
		self.evaders = [0, 1]
		self.pursuers = [2, 3]
		self.turn_groups = [np.array([0, 1]), np.array([2, 3])]
		self.time_idx = 8
		self.active_idxs = [9, 10, 11, 12]

		self.state_dim = 13
		self.action_dim = 8
		self.state_idxs = [
			np.array([0,1]),  # E1
			np.array([2,3]),  # E2
			np.array([4,5]),  # P1
			np.array([6,7]),  # P2
		]
		self.action_idxs = [
			np.array([0,1]),  # E1 action
			np.array([2,3]),  # E2 action
			np.array([4,5]),  # P1 action
			np.array([6,7]),  # P2 action
		]
		#self.times = np.arange(self.t0,self.tf,self.dt)
		self.times = np.arange(self.t0,self.tf+self.dt,self.dt)
		self.policy_encoding_dim = self.state_dim
		self.value_encoding_dim = self.state_dim
 
		self.state_lims = np.array((
			(-10,10), 
			(-10,10), 
			(-10,10), 
			(-10,10),
			(-10,10), 
			(-10,10), 
			(-10,10), 
			(-10,10),
			(0,self.tf),
			(0,1),
			(0,1),
			(0,1),
			(0,1),
			))
		self.approx_dist = (self.state_lims[0,1] - self.state_lims[0,0])/10 

		self.evader_speed_lim_range = (1.4, 2.6)
		self.pursuer_speed_lim_range = (1.4, 2.6)
		self.current_evader_speed_lim = 2.0
		self.current_pursuer_speed_lim = 2.0

		self.update_action_lims()

	def update_action_lims(self):
		self.control_lims = np.array((
			(-self.current_evader_speed_lim, self.current_evader_speed_lim),
			(-self.current_evader_speed_lim, self.current_evader_speed_lim),

			(-self.current_evader_speed_lim, self.current_evader_speed_lim),
			(-self.current_evader_speed_lim, self.current_evader_speed_lim),

			(-self.current_pursuer_speed_lim, self.current_pursuer_speed_lim),
			(-self.current_pursuer_speed_lim, self.current_pursuer_speed_lim),

			(-self.current_pursuer_speed_lim, self.current_pursuer_speed_lim),
			(-self.current_pursuer_speed_lim, self.current_pursuer_speed_lim),
		))
		self.action_lims = self.dt * self.control_lims

	def randomize_speed_limits(self):
		self.current_evader_speed_lim = np.random.uniform(*self.evader_speed_lim_range)
		self.current_pursuer_speed_lim = np.random.uniform(*self.pursuer_speed_lim_range)
		self.update_action_lims()
		# self.action_lims = np.array((
		# 	# (-0.0,0.0),
		# 	# (-0.0,0.0),
		# 	# (-0.0,0.0),
		# 	# (-0.0,0.0),

		# 	(-1.0,1.0),
		# 	(-1.0,1.0),

		# 	(-1.0,1.0),
		# 	(-1.0,1.0),

		# 	(-1.0,1.0),
		# 	(-1.0,1.0),

		# 	(-1.0,1.0),
		# 	(-1.0,1.0),
		# 	))

		self.init_lims = np.array((
			(-8,8), (-8,8),
			(-8,8), (-8,8),
			(-8,8), (-8,8),
			(-8,8), (-8,8),
			(0,0),
			(1,1), (1,1), (1,1), (1,1),
			))

		self.Fc = np.array((
			(0,0), 
			(0,0), 
			))

		self.Bc = np.array((
			(1,0),
			(0,1),
			))

		self.Q = np.eye(2)
		self.Ru = self.state_control_weight * np.eye(2)

	def is_active(self, state, robot):
		return state[self.active_idxs[robot], 0] > 0.5

	def get_robot_speed_limit(self, robot):
		if robot in self.evaders:
			return self.current_evader_speed_lim
		return self.current_pursuer_speed_lim

	def get_robot_control_lims(self, robot):
		return self.control_lims[self.action_idxs[robot], :]

	def sample_action(self, state=None):
		return sample_vector(self.action_lims)

	def apply_velocity_limits(self, velocity, robot):
		control_lims = self.get_robot_control_lims(robot)
		return np.clip(
			velocity,
			control_lims[:, 0].reshape((-1, 1)),
			control_lims[:, 1].reshape((-1, 1)),
		)

	def action_to_velocity(self, action, robot, dt):
		delta_position = action[self.action_idxs[robot], :]
		dt = max(dt, 1e-8)
		desired_velocity = delta_position / dt
		return self.apply_velocity_limits(desired_velocity, robot)

	def propagate_robot_state(self, state, velocity, dt, robot):
		Fd = np.eye(len(self.state_idxs[robot])) + dt * self.Fc
		Bd = dt * self.Bc
		return np.dot(Fd, state[self.state_idxs[robot], :]) + np.dot(Bd, velocity)

	def active_evader_count(self, state):
		return sum(self.is_active(state, e) for e in self.evaders)
	def active_pursuer_count(self, state):
		return sum(self.is_active(state, p) for p in self.pursuers)
	def min_cross_team_dist(self, state):
		min_dist = np.inf
		for e in self.evaders:
			for p in self.pursuers:
				d = np.linalg.norm(state[self.state_idxs[e], :] - state[self.state_idxs[p], :])
				min_dist = min(min_dist, d)
		return min_dist
	def get_capture_pairs(self, state):
		candidates = []
		for p in self.pursuers:
			if not self.is_active(state, p):
				continue
			for e in self.evaders:
				if not self.is_active(state, e):
					continue
				d = np.linalg.norm(state[self.state_idxs[p], :] - state[self.state_idxs[e], :])
				if d < self.desired_distance:
					candidates.append((d, p, e))

		candidates.sort(key=lambda x: x[0]) #candidates里的元素是(d,p,e)，按照d排序，d越小越靠前

		matched_p = set()
		matched_e = set() 
		capture_pairs = []
		for _, p, e in candidates:
			if p in matched_p or e in matched_e:
				continue
			matched_p.add(p)
			matched_e.add(e)
			capture_pairs.append((p, e))

		return capture_pairs

	def reward(self,s,a): 
		reward = self.normalized_reward(s,a) 
		return reward

	@override
	def initialize(self):
		valid = False
		while not valid:
			self.randomize_speed_limits()
			state = sample_vector(self.init_lims)
			state[self.time_idx, 0] = 0.0
			for idx in self.active_idxs:
				state[idx, 0] = 1.0
			valid = not self.is_terminal(state) and (self.min_cross_team_dist(state) > 2*self.init_min_dist)
		return state

	def normalized_reward(self,s,a):
		s_next= self.step(s, a, self.dt)
		r1 = 0.0
		r2 = 0.0
		reward = np.array([[r1], [r1], [r2], [r2]], dtype=float)
		newly_captured = self.active_evader_count(s) - self.active_evader_count(s_next)
		if newly_captured > 0:
			t = min(s_next[self.time_idx, 0], self.tf)
			evader_reward = 0.5*newly_captured * (t / self.tf)	
			pursuer_reward = 0.5*newly_captured * (1.0 - t / self.tf)
			reward[self.evaders, 0] = evader_reward
			reward[self.pursuers, 0] = pursuer_reward
		if (s[self.time_idx, 0] < self.tf) and (s_next[self.time_idx, 0] >= self.tf):
			#t = min(s_next[self.time_idx, 0], self.tf)
			surviving_evaders = self.active_evader_count(s_next)
			reward[self.evaders, 0] += 0.5*surviving_evaders 
		for robot in range (self.num_robots):
			if not contains(s_next[self.state_idxs[robot],:],self.state_lims[self.state_idxs[robot],:]):
				reward[robot,0]= -1.0
		return reward

	def gt_action(self, state, robot):
		e_pos = state[self.state_idxs[0], :]
		p_pos = state[self.state_idxs[1], :]

		if robot == 0:
			direction = e_pos - p_pos
		else:
			direction = e_pos - p_pos

		norm = np.linalg.norm(direction)
		if norm < 1e-8:
			unit = np.zeros((2, 1))
		else:
			unit = direction / norm

		robot_action_idxs = self.action_idxs[robot]
		u_max = np.abs(self.action_lims[robot_action_idxs, 1]).reshape((-1, 1))

		if robot == 0:
			action = unit * u_max
		else:
			action = unit * u_max

		action = np.clip(
			action,
			self.action_lims[robot_action_idxs, 0].reshape((-1, 1)),
			self.action_lims[robot_action_idxs, 1].reshape((-1, 1)),
		)
		return action

	def gt_actions(self, state):
		actions = []
		for robot in range(self.num_robots):
			action = self.gt_action(state, robot)
			actions.append(action)
		return np.array(actions).reshape(-1, 1)

	def gt_value(self, state, robot):
		t = float(state[4, 0])

		if self.is_captured(state) or t >= self.tf:
			evader_value = min(t, self.tf) / self.tf
			pursuer_value = 1.0 - evader_value
			values = np.array([[evader_value], [pursuer_value]])
			return values[robot:robot+1, :]

		e_pos = state[self.state_idxs[0], :]
		p_pos = state[self.state_idxs[1], :]
		d = np.linalg.norm(e_pos - p_pos)

		evader_action_max = np.linalg.norm(self.action_lims[self.action_idxs[0], 1])
		pursuer_action_max = np.linalg.norm(self.action_lims[self.action_idxs[1], 1])
		closing_speed = max(evader_action_max + pursuer_action_max, 1e-8)

		t_to_capture = max((d - self.desired_distance) / closing_speed, 0.0)
		terminal_t = min(t + t_to_capture, self.tf)

		evader_value = terminal_t / self.tf
		pursuer_value = 1.0 - evader_value
		values = np.array([[evader_value], [pursuer_value]])
		return values[robot:robot+1, :]
	
	def is_captured(self, s):
		return len(self.get_capture_pairs(s)) > 0


	def step(self,s,a,dt):
		s_tp1 = np.array(s,copy=True)
		for robot in range(self.num_robots):
			s_tp1[self.active_idxs[robot], 0] = s[self.active_idxs[robot], 0]
			if not self.is_active(s, robot):
				s_tp1[self.state_idxs[robot],:] = s[self.state_idxs[robot],:]
				continue

			velocity = self.action_to_velocity(a, robot, dt)
			s_tp1[self.state_idxs[robot],:] = self.propagate_robot_state(s, velocity, dt, robot)
		s_tp1[self.time_idx, 0] = s[self.time_idx, 0] + dt
		for p, e in self.get_capture_pairs(s_tp1):
			s_tp1[self.active_idxs[p], 0] = 0.0
			s_tp1[self.active_idxs[e], 0] = 0.0
		return s_tp1 

	def render(self,states=None,fig=None,ax=None):
		# states, np array in [nt x state_dim]
  
		states = states.squeeze() if states is not None else None
		
		if fig == None or ax == None:
			fig,ax = plotter.make_fig()

		if states is not None:

			colors = plotter.get_n_colors(self.num_robots)
			for robot in range(self.num_robots):
				robot_state_idxs = self.state_idxs[robot] 

				ax.plot(states[:,robot_state_idxs[0]], states[:,robot_state_idxs[1]], color=colors[robot])
				ax.plot(states[0,robot_state_idxs[0]], states[0,robot_state_idxs[1]], color=colors[robot],marker='o')
				ax.plot(states[-1,robot_state_idxs[0]], states[-1,robot_state_idxs[1]], color=colors[robot],marker='s')
				
			# ax.set_aspect(lims[0,1]-lims[0,0] / lims[1,1]-lims[1,0])

				if robot in [0, 1]:
					circ = patches.Circle((states[-1,robot_state_idxs[0]], states[-1,robot_state_idxs[1]]), \
						self.desired_distance,facecolor='green',alpha=0.5)
					ax.add_patch(circ)

			for robot in range(self.num_robots):
				if robot == 0:
					label = "Evader1"
				elif robot == 1:
					label = "Evader2"
				elif robot == 2:
					label = "Pursuer1"
				elif robot == 3:
					label = "Pursuer2"
				ax.plot(np.nan,np.nan,color=colors[robot],label=label)
			ax.legend(loc='best')

		lims = self.state_lims
		ax.set_xlim((lims[0,0],lims[0,1]))
		ax.set_ylim((lims[1,0],lims[1,1]))
		ax.set_aspect( (lims[1,1]-lims[1,0]) / (lims[0,1]-lims[0,0]) )

		return fig,ax 

	def is_terminal(self,state):
		# return not self.is_valid(state)
		#return (not self.is_valid(state)) or self.is_captured(state) 
		return ((not self.is_valid(state)) or self.active_evader_count(state) == 0 or self.active_pursuer_count(state) == 0) or state[self.time_idx, 0] >= self.tf

	def is_valid(self,state):
		return contains(state,self.state_lims)

	def policy_encoding(self,state,robot):
		return state

	def value_encoding(self,state):
		return state 

	def plot_value_dataset(self,dataset,title):

		max_plots_per_robot = 10

		for robot in range(self.num_robots):
			groups = self.make_groups(dataset[0],dataset[1],robot)
			if len(groups) > max_plots_per_robot:
				groups = groups[0:max_plots_per_robot]

			for group in groups: 

				data = np.array([np.array(xi) for xi in group])
				encodings = data[:,0:self.state_dim]
				target = data[:,self.state_dim:]

				# contour
				if encodings.shape[0] > 100:
					fig,ax = plt.subplots() 
					robot_idxs = self.state_idxs[robot]
					pcm = ax.tricontourf(encodings[:,robot_idxs[0]],encodings[:,robot_idxs[1]],target[:,robot])
					fig.colorbar(pcm,ax=ax)
					ax.set_title("{} Value for Robot {}".format(title,robot))
					ax.set_xlim(self.state_lims[self.position_idx[0],:])
					ax.set_ylim(self.state_lims[self.position_idx[0],:])

				else:
					# scatter
					fig,ax = plt.subplots() 
					robot_idxs = self.state_idxs[robot]
					pcm = ax.scatter(encodings[:,robot_idxs[0]],encodings[:,robot_idxs[1]],c=target[:,robot])
					fig.colorbar(pcm,ax=ax)
					ax.set_title("{} Value for Robot {}".format(title,robot))
					ax.set_xlim(self.state_lims[self.position_idx[0],:])
					ax.set_ylim(self.state_lims[self.position_idx[0],:])

				# plot other agents 
				state = group[0][0:self.state_dim]
				for other_robot in range(self.num_robots):
					if other_robot != robot:
						other_robot_idxs = self.state_idxs[other_robot] 
						# Circle((x,y),radius)
						circ = patches.Circle((state[other_robot_idxs[0]], state[other_robot_idxs[1]]), \
							self.approx_dist,facecolor='gray',alpha=0.5)
						ax.add_patch(circ)
			
				self.render(fig=fig,ax=ax)

	def plot_policy_dataset(self,dataset,title,robot):

		max_plots = 10

		robot_idxs = self.state_idxs[robot]

		groups = self.make_groups(dataset[0],dataset[1],robot)
		if len(groups) > max_plots:
			groups = groups[0:max_plots]

		for group in groups: 
			fig,ax = plt.subplots() 

			data = np.array([np.array(xi) for xi in group])
			encodings = data[:,0:self.state_dim]
			target = data[:,self.state_dim:]
			# quiver 
			C = np.linalg.norm(target[:,0:2],axis=1)
			ax.quiver(encodings[:,robot_idxs[0]],encodings[:,robot_idxs[1]],\
				target[:,0],target[:,1])
			ax.scatter(encodings[:,robot_idxs[0]],encodings[:,robot_idxs[1]],c=C,s=2)
			ax.set_title("{} Policy for Robot {}".format(title,robot))
			ax.set_xlim(self.state_lims[self.position_idx[0],:])
			ax.set_ylim(self.state_lims[self.position_idx[0],:])

			# plot other agents 
			state = group[0][0:self.state_dim]
			for other_robot in range(self.num_robots):
				if other_robot != robot:
					other_robot_idxs = self.state_idxs[other_robot] 
					# Circle((x,y),radius)
					circ = patches.Circle((state[other_robot_idxs[0]], state[other_robot_idxs[1]]), \
						self.approx_dist,facecolor='gray',alpha=0.5)
					ax.add_patch(circ)
			self.render(fig=fig,ax=ax)


	def make_groups(self,encoding,target,robot):

		num_datapoints = encoding.shape[0]
		groups = [] # list of list of lists
		robot_idxs = self.state_idxs[robot] 
		not_robot_idxs = []
		for i in range(self.state_dim):
			if i not in robot_idxs:
				not_robot_idxs.append(i)

		# print('encoding.shape',encoding.shape)
		# print('target.shape',target.shape)

		for i in range(num_datapoints): 
			matched = False
			for group in groups: 
				# if self.isApprox(encoding[i][not_robot_idxs],group[0][0][not_robot_idxs]):
				if self.isApprox(encoding[i][not_robot_idxs],group[0][not_robot_idxs]):
					# group.append([encoding[i].tolist(),target[i].tolist()])
					group.append(np.concatenate((encoding[i],target[i])))
					matched = True
					break 
			if not matched: 
				# groups.append([encoding[i].tolist(),target[i].tolist()])
				groups.append([np.concatenate((encoding[i],target[i]))])
		return groups 

	def isApprox(self,s1,s2):
		return np.linalg.norm(s1-s2) < self.approx_dist 


	def pretty_plot(self,sim_result):

		value_plot_on = sim_result["instance"]["value_oracle"] is not None
		policy_plot_on = not all([a is None for a in sim_result["instance"]["policy_oracle"]])

		if value_plot_on or policy_plot_on:

			
			for robot in [0,1,2,3]:

				fig,ax = plt.subplots()
				inital_state = sim_result["states"][0]

				robot_idxs = self.state_idxs[robot]
				not_robot_idxs = []
				for i in range(self.state_dim):
					if i not in robot_idxs:
						not_robot_idxs.append(i)

				num_eval = 3000
				states = []
				for _ in range(num_eval):
					state = self.initialize()
					state[not_robot_idxs,:] = inital_state[not_robot_idxs,:]
					states.append(state)
				state_array = np.stack(states, axis=0)
				state_points = state_array[:, :, 0]

				# plot value func contours
				if sim_result["instance"]["value_oracle"] is not None:
					value_oracle = sim_result["instance"]["value_oracle"]
					values = []
					for state in states: 
						value = value_oracle.eval(self,state)
						values.append(value)
					values = np.array(values).squeeze(axis=2)

					pcm = ax.tricontourf(state_points[:,robot_idxs[0]],state_points[:,robot_idxs[1]],values[:,robot])
					fig.colorbar(pcm,ax=ax)	

				# plot policy function 
				if not all([a is None for a in sim_result["instance"]["policy_oracle"]]):
					policy_oracle = sim_result["instance"]["policy_oracle"]
					actions = []
					for state in states: 
						action = policy_oracle[robot].eval(self,state,robot)
						actions.append(action)
					actions = np.array(actions).squeeze(axis=2)
					ax.quiver(state_points[:,robot_idxs[0]],state_points[:,robot_idxs[1]],actions[:,0],actions[:,1])
					
				# plot final trajectory , obstacles and limits 
				self.render(fig=fig,ax=ax,states=sim_result["states"])
