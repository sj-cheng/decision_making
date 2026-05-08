

import numpy as np 
import os, subprocess
import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from matplotlib import cm	
from matplotlib.backends.backend_pdf import PdfPages 
from PyPDF2 import PdfFileMerger

# defaults
plt.rcParams.update({'font.size': 10})
plt.rcParams['lines.linewidth'] = 2.5

import matplotlib
matplotlib.use('Agg')


def has_figs():
	if len(plt.get_fignums()) > 0:
		return True
	else:
		return False


def save_figs(filename):
	file_dir,  file_name = os.path.split(filename)
	if len(file_dir) >0 and not (os.path.isdir(file_dir)):
		os.makedirs(file_dir)
	fn = os.path.join( os.getcwd(), filename)
	pp = PdfPages(fn)
	for i in plt.get_fignums():
		pp.savefig(plt.figure(i))
		plt.close(plt.figure(i))
	pp.close()


def open_figs(filename):
	pdf_path = os.path.join( os.getcwd(), filename)
	if os.path.exists(pdf_path):
		subprocess.call(["xdg-open", pdf_path])


def merge_figs(pdfs,result_fn):
	merger = PdfFileMerger()
	# write new one 
	for pdf in pdfs:
	    merger.append(pdf)
	merger.write(result_fn)
	merger.close()
	# delete old files 
	for pdf in pdfs: 
		os.remove(pdf)


def make_fig():
	return plt.subplots()


def make_3d_fig():
	from mpl_toolkits.mplot3d import Axes3D
	fig = plt.figure()
	ax = fig.add_subplot(111, projection='3d')
	return fig,ax 


def get_n_colors(n,cmap=None):
	colors = []
	cm_subsection = np.linspace(0, 1, n)
	if cmap is None:
		cmap = cm.tab20
	colors = [ cmap(x) for x in cm_subsection]
	return colors


def _get_problem_attr(problem, name, default=None):
	if isinstance(problem, dict):
		return problem.get(name, default)
	return getattr(problem, name, default)


def _as_time_matrix(data, expected_width=None):
	arr = np.asarray(data, dtype=float)
	if arr.size == 0:
		width = 0 if expected_width is None else expected_width
		return np.empty((0, width))
	if arr.ndim == 0:
		return arr.reshape((1, 1))
	if arr.ndim == 1:
		if expected_width is not None and arr.size == expected_width:
			return arr.reshape((1, expected_width))
		return arr.reshape((-1, 1))
	return arr.reshape((arr.shape[0], -1))


def _robot_label(problem, robot):
	evaders = list(_get_problem_attr(problem, "evaders", []))
	pursuers = list(_get_problem_attr(problem, "pursuers", []))
	if robot in evaders:
		if len(evaders) == 1:
			return "Evader"
		return "Evader {}".format(evaders.index(robot) + 1)
	if robot in pursuers:
		if len(pursuers) == 1:
			return "Pursuer"
		return "Pursuer {}".format(pursuers.index(robot) + 1)
	return "Robot {}".format(robot)


def _robot_position_indices(problem, robot):
	state_idxs = _get_problem_attr(problem, "state_idxs", [])
	if robot >= len(state_idxs):
		return np.array([], dtype=int)
	robot_state_idxs = np.asarray(state_idxs[robot], dtype=int)
	position_idx = np.asarray(
		_get_problem_attr(problem, "position_idx", np.arange(min(2, len(robot_state_idxs)))),
		dtype=int,
	).reshape(-1)
	position_idx = position_idx[position_idx < len(robot_state_idxs)]
	return robot_state_idxs[position_idx]


def _min_cross_team_distances(problem, states):
	evaders = list(_get_problem_attr(problem, "evaders", []))
	pursuers = list(_get_problem_attr(problem, "pursuers", []))
	if len(evaders) == 0 or len(pursuers) == 0 or states.shape[0] == 0:
		return None

	distances = []
	for state in states:
		min_dist = np.inf
		for evader in evaders:
			evader_idxs = _robot_position_indices(problem, evader)
			if len(evader_idxs) == 0:
				continue
			for pursuer in pursuers:
				pursuer_idxs = _robot_position_indices(problem, pursuer)
				num_dims = min(len(evader_idxs), len(pursuer_idxs))
				if num_dims == 0:
					continue
				dist = np.linalg.norm(state[evader_idxs[:num_dims]] - state[pursuer_idxs[:num_dims]])
				min_dist = min(min_dist, dist)
		if np.isfinite(min_dist):
			distances.append(min_dist)

	if len(distances) != states.shape[0]:
		return None
	return np.asarray(distances)


def _reward_groups(problem, num_robots):
	evaders = [robot for robot in list(_get_problem_attr(problem, "evaders", [])) if robot < num_robots]
	pursuers = [robot for robot in list(_get_problem_attr(problem, "pursuers", [])) if robot < num_robots]
	if len(evaders) > 0 or len(pursuers) > 0:
		groups = []
		if len(evaders) > 0:
			groups.append(("Evader team", evaders))
		if len(pursuers) > 0:
			groups.append(("Pursuer team", pursuers))
		return groups
	return [(_robot_label(problem, robot), [robot]) for robot in range(num_robots)]


def _style_summary_axis(ax):
	ax.grid(True, color="0.88", linewidth=0.7)
	ax.spines["top"].set_visible(False)
	ax.spines["right"].set_visible(False)
	ax.tick_params(labelsize=8)
	ax.title.set_fontsize(9)
	ax.xaxis.label.set_fontsize(8)
	ax.yaxis.label.set_fontsize(8)


def plot_sim_result(sim_result):
	times = sim_result["times"] # nt, 
	raw_states = sim_result["states"] # nt x state_dim
	raw_actions = sim_result["actions"] # nt-1 x action_dim
	raw_rewards = sim_result["rewards"] # nt-1,x num_robots
	problem = sim_result["instance"]["problem"] 
	times = np.asarray(times, dtype=float).reshape(-1)

	num_robots = int(_get_problem_attr(problem, "num_robots", 1))
	state_lims = np.asarray(_get_problem_attr(problem, "state_lims", []))
	action_lims = np.asarray(_get_problem_attr(problem, "action_lims", []))
	action_dim = action_lims.shape[0] if action_lims.ndim == 2 else None
	states = _as_time_matrix(raw_states, state_lims.shape[0] if state_lims.ndim == 2 else None)
	actions = _as_time_matrix(raw_actions, action_dim)
	rewards = _as_time_matrix(raw_rewards, num_robots)
	action_times = times[1:1 + actions.shape[0]]
	reward_times = times[1:1 + rewards.shape[0]]

	colors = get_n_colors(num_robots)
	labels = [_robot_label(problem, robot) for robot in range(num_robots)]
	panels = []

	position_idx = np.asarray(_get_problem_attr(problem, "position_idx", np.arange(2)), dtype=int).reshape(-1)
	position_names = ["x", "y", "z"]
	num_position_panels = min(len(position_idx), 3)
	for local_position_dim in range(num_position_panels):
		def plot_position(ax, local_position_dim=local_position_dim):
			for robot in range(num_robots):
				position_state_idxs = _robot_position_indices(problem, robot)
				if local_position_dim >= len(position_state_idxs):
					continue
				state_idx = position_state_idxs[local_position_dim]
				if state_idx >= states.shape[1]:
					continue
				ax.plot(
					times,
					states[:, state_idx],
					color=colors[robot],
					label=labels[robot],
					linewidth=1.7,
				)
			component = position_names[local_position_dim] if local_position_dim < len(position_names) else "p{}".format(local_position_dim)
			ax.set_title("{} position".format(component))
			ax.set_ylabel(component)
			ax.set_xlabel("time")
			if state_lims.ndim == 2 and position_idx[local_position_dim] < state_lims.shape[0]:
				lims = state_lims[position_idx[local_position_dim], :]
				if np.all(np.isfinite(lims)):
					ax.set_ylim((lims[0], lims[1]))
			ax.legend(loc="best", frameon=False, fontsize=7, ncol=2)
		panels.append(plot_position)

	if actions.shape[0] > 0:
		def plot_action_norm(ax):
			action_idxs = _get_problem_attr(problem, "action_idxs", [])
			for robot in range(num_robots):
				if robot >= len(action_idxs):
					continue
				robot_action_idxs = np.asarray(action_idxs[robot], dtype=int)
				robot_action_idxs = robot_action_idxs[robot_action_idxs < actions.shape[1]]
				if len(robot_action_idxs) == 0:
					continue
				action_norm = np.linalg.norm(actions[:, robot_action_idxs], axis=1)
				ax.plot(
					action_times,
					action_norm,
					color=colors[robot],
					label=labels[robot],
					linewidth=1.7,
				)
			ax.set_title("control magnitude")
			ax.set_ylabel(r"$\|u\|$")
			ax.set_xlabel("time")
			ax.legend(loc="best", frameon=False, fontsize=7, ncol=2)
		panels.append(plot_action_norm)

	min_distances = _min_cross_team_distances(problem, states)
	if min_distances is not None:
		def plot_min_distance(ax):
			ax.plot(times, min_distances, color="0.15", linewidth=1.9, label="min E-P distance")
			desired_distance = _get_problem_attr(problem, "desired_distance", None)
			if desired_distance is not None:
				ax.axhline(
					desired_distance,
					color="0.55",
					linestyle="--",
					linewidth=1.0,
					label="capture radius",
				)
			ax.set_title("minimum separation")
			ax.set_ylabel("distance")
			ax.set_xlabel("time")
			ax.legend(loc="best", frameon=False, fontsize=7)
		panels.append(plot_min_distance)

	if rewards.shape[0] > 0:
		reward_groups = _reward_groups(problem, min(num_robots, rewards.shape[1]))

		def plot_reward(ax):
			for i_group, (label, robot_idxs) in enumerate(reward_groups):
				robot_idxs = [robot for robot in robot_idxs if robot < rewards.shape[1]]
				if len(robot_idxs) == 0:
					continue
				group_reward = np.mean(rewards[:, robot_idxs], axis=1)
				ax.plot(
					reward_times,
					group_reward,
					color=colors[i_group % len(colors)],
					label=label,
					linewidth=1.7,
				)
			ax.set_title("instant reward")
			ax.set_ylabel("reward")
			ax.set_xlabel("time")
			ax.legend(loc="best", frameon=False, fontsize=7)
		panels.append(plot_reward)

		def plot_cumulative_reward(ax):
			for i_group, (label, robot_idxs) in enumerate(reward_groups):
				robot_idxs = [robot for robot in robot_idxs if robot < rewards.shape[1]]
				if len(robot_idxs) == 0:
					continue
				group_reward = np.mean(rewards[:, robot_idxs], axis=1)
				ax.plot(
					reward_times,
					np.cumsum(group_reward),
					color=colors[i_group % len(colors)],
					label=label,
					linewidth=1.7,
				)
			ax.set_title("cumulative reward")
			ax.set_ylabel("return")
			ax.set_xlabel("time")
			ax.legend(loc="best", frameon=False, fontsize=7)
		panels.append(plot_cumulative_reward)

	if len(panels) == 0:
		return

	ncols = 2 if len(panels) > 1 else 1
	nrows = int(math.ceil(len(panels) / ncols))
	fig,axs = plt.subplots(
		nrows=nrows,
		ncols=ncols,
		figsize=(7.2, max(2.2, 2.05 * nrows)),
		squeeze=False,
		constrained_layout=True,
	)
	axs = axs.reshape(-1)
	for ax, panel in zip(axs, panels):
		panel(ax)
		_style_summary_axis(ax)
	for ax in axs[len(panels):]:
		fig.delaxes(ax)


def plot_loss(losses):
	losses = np.array(losses)
	fig,ax = plt.subplots()
	ax.plot(losses[:,0],label="Train")
	ax.plot(losses[:,1],label="Test")
	ax.legend()
	if np.amin(losses) > 0:
		ax.set_yscale('log')
	ax.set_title("Losses")


def plot_tree_state(problem,tree_state,zoom_on=True):
	# tree state : nd array in [num_nodes x state_dim + 1]


	position_idxs = problem.position_idx

	if len(position_idxs) == 1: 
		fig,ax = plt.subplots() 

		plot_idx = np.arange(2)
		segments = []
		nodes = [] 
		for i_row,row in enumerate(tree_state):
			parentIdx = int(row[-1])
			nodes.append(row[plot_idx])
			if parentIdx >= 0:
				segments.append([row[plot_idx], tree_state[parentIdx][plot_idx]])

		ln_coll = matplotlib.collections.LineCollection(segments, linewidth=0.2, colors='k', alpha=0.2)
		ax.add_collection(ln_coll)
		problem.render(fig=fig,ax=ax)


	elif len(problem.position_idx) == 2: 
		fig,ax = plt.subplots()

		segments = [[] for _ in range(problem.num_robots)]
		nodes = [[] for _ in range(problem.num_robots)]
		for i_row,row in enumerate(tree_state):
			parentIdx = int(row[-1])

			for robot in range(problem.num_robots):
				robot_state_idxs = problem.state_idxs[robot]
				robot_position_idx = robot_state_idxs[position_idxs]
				nodes[robot].append(row[robot_position_idx])
				if parentIdx >= 0:
					segments[robot].append([row[robot_position_idx], tree_state[parentIdx][robot_position_idx]])

		# nodes = np.array(nodes[robot])
		for robot in range(problem.num_robots):
			ln_coll = matplotlib.collections.LineCollection(segments[robot], linewidth=0.2, colors='k', alpha=0.2)
			ax.add_collection(ln_coll)
			ax.scatter(nodes[robot][0][0],nodes[robot][0][1])


		if not zoom_on: 
			lims = problem.state_lims
			ax.set_xlim((lims[0,0],lims[0,1]))
			ax.set_ylim((lims[1,0],lims[1,1]))
			
		problem.render(fig=fig,ax=ax)


	elif len(problem.position_idx) == 3: 
		
		num_robots = problem.num_robots

		fig,ax = make_3d_fig()
		segments = [[] for _ in range(num_robots)]
		nodes = [[] for _ in range(num_robots)]
		for i_row,row in enumerate(tree_state):
			parentIdx = int(row[-1])

			for robot in range(num_robots):
				robot_state_idxs = problem.state_idxs[robot]
				robot_position_idx = robot_state_idxs[position_idxs]
				nodes[robot].append(row[robot_position_idx])
				if parentIdx >= 0:
					segments[robot].append([row[robot_position_idx], tree_state[parentIdx][robot_position_idx]])

		for robot in range(num_robots):
			ln_coll = Line3DCollection(segments[robot], linewidth=0.2, colors='k', alpha=0.2)
			ax.add_collection(ln_coll)

		lims = problem.state_lims
		ax.set_xlim((lims[0,0],lims[0,1]))
		ax.set_ylim((lims[1,0],lims[1,1]))
		ax.set_zlim((lims[2,0],lims[2,1]))
		ax.set_box_aspect((lims[0,1]-lims[0,0], lims[1,1]-lims[1,0], lims[2,1]-lims[2,0]))  
		problem.render(fig=fig,ax=ax)

	else: 
		print('tree plot dimension not supported')


def plot_value_dataset(problem,datasets,dataset_names):
	
	encoding_dim = problem.policy_encoding_dim
	target_dim = 1
	state_lims = problem.state_lims
	# action_lims = [0,1]

	for title,dataset in zip(dataset_names,datasets):
		encodings = dataset[0]
		target = dataset[1] 

		fig,ax = plt.subplots(nrows=2,ncols=max((encoding_dim,target_dim)),squeeze=False)
		for i_e in range(encoding_dim):
			ax[0,i_e].hist(encodings[:,i_e])
			if not (np.isinf(np.abs(state_lims[i_e,:])).any()):
				ax[0,i_e].set_xlim(state_lims[i_e,0],state_lims[i_e,1])
		ax[0,0].set_ylabel("Encoding")

		for i_t in range(target_dim):
			ax[1,i_t].hist(target[:,i_t])
			# ax[1,i_t].set_xlim(action_lims[i_t,0],action_lims[i_t,1])
		ax[1,0].set_ylabel("Target")
		fig.suptitle(title)

		problem.plot_value_dataset(dataset,title)



def plot_policy_dataset(problem,datasets,dataset_names,robot):
	# datapoints: [(encoding,target) ]
	# encoding: problem.policy_encoding(state)
	# target: robot_action 

	encoding_dim = problem.policy_encoding_dim
	target_dim = len(problem.action_idxs[robot]) 
	state_lims = problem.state_lims
	action_lims = problem.action_lims

	for title,dataset in zip(dataset_names,datasets):
		encodings = dataset[0]
		target = dataset[1]

		if title == "Eval":
			fig,ax = plt.subplots(nrows=3,ncols=max((encoding_dim,target_dim)),squeeze=False)
			for i_e in range(encoding_dim):
				ax[0,i_e].hist(encodings[:,i_e])
				if not (np.isinf(np.abs(state_lims[i_e,:])).any()):
					ax[0,i_e].set_xlim(state_lims[i_e,0],state_lims[i_e,1])
			ax[0,0].set_ylabel("Encoding")

			for i_t in range(target_dim):
				ax[1,i_t].hist(target[:,i_t])
				ax[1,i_t].set_xlim(action_lims[i_t,0],action_lims[i_t,1])
				ax[2,i_t].hist(target[:,i_t+target_dim])
			ax[1,0].set_ylabel("Mean")
			ax[2,0].set_ylabel("Variance")
			fig.suptitle(title)

		else: 
			fig,ax = plt.subplots(nrows=2,ncols=max((encoding_dim,target_dim)),squeeze=False)

			for i_e in range(encoding_dim):
				ax[0,i_e].hist(encodings[:,i_e])
				if not (np.isinf(np.abs(state_lims[i_e,:])).any()):
					ax[0,i_e].set_xlim(state_lims[i_e,0],state_lims[i_e,1])
			ax[0,0].set_ylabel("Encoding")

			for i_t in range(target_dim):
				ax[1,i_t].hist(target[:,i_t])
				ax[1,i_t].set_xlim(action_lims[i_t,0],action_lims[i_t,1])
			ax[1,0].set_ylabel("Target")
			fig.suptitle(title)

		problem.plot_policy_dataset(dataset,title,robot)


def plot_regression_test(results,render_on=True):
	# results: list of (instance, sim_result) pairs 

	# render each sim result 
	if render_on:
		for (param, sim_result) in results:
			fig,ax = sim_result["instance"]["problem"].render(states=sim_result["states"])
			fig.suptitle(param.key)
	else:
		param, sim_result = results[0]

	# for each problem, 
	# 	- plot duration per timestep across number of simulations for each solver 
	# 	- plot total reward ""
	duration_plot_data = np.zeros((
		len(param.problem_name_lst),
		len(param.number_simulations_lst),
		len(param.solver_name_lst),
		param.num_trial))
	reward_plot_data = np.zeros((
		len(param.problem_name_lst),
		len(param.number_simulations_lst),
		len(param.solver_name_lst),
		param.num_trial))
	for (param,sim_result) in results: 
		i_pn = param.problem_name_lst.index(param.problem_name)
		i_ns = param.number_simulations_lst.index(param.number_simulations)
		i_sn = param.solver_name_lst.index(param.solver_name)
		duration_plot_data[i_pn,i_ns,i_sn,param.trial] = sim_result["duration_per_timestep"]
		reward_plot_data[i_pn,i_ns,i_sn,param.trial] = np.sum(sim_result["rewards"][:,0])
	
	fig,axs = plt.subplots(ncols=len(param.problem_name_lst),nrows=1,squeeze=False)
	for i_pn, problem_name in enumerate(param.problem_name_lst):
		for i_sn, solver_name in enumerate(param.solver_name_lst):
			lns = axs[0,i_pn].plot(
				np.array(param.number_simulations_lst),
				np.mean(duration_plot_data[i_pn,:,i_sn,:],axis=1),
				label = solver_name)
			axs[0,i_pn].fill_between(
				np.array(param.number_simulations_lst),
				np.mean(duration_plot_data[i_pn,:,i_sn,:],axis=1) - np.std(duration_plot_data[i_pn,:,i_sn,:],axis=1),
				np.mean(duration_plot_data[i_pn,:,i_sn,:],axis=1) + np.std(duration_plot_data[i_pn,:,i_sn,:],axis=1),
				color = lns[0].get_color(),
				alpha = 0.5)
		axs[0,i_pn].set_xscale('log')
		axs[0,i_pn].set_yscale('log')
		axs[0,i_pn].set_title(problem_name)
		axs[0,i_pn].tick_params(axis='x', rotation=45)
	axs[0,0].legend(loc='best')
	axs[0,0].set_ylabel("WCT / Timestep")
	fig.suptitle("Computation Time Regression Test")

	fig,axs = plt.subplots(ncols=len(param.problem_name_lst),nrows=1,squeeze=False)
	for i_pn, problem_name in enumerate(param.problem_name_lst):
		for i_sn, solver_name in enumerate(param.solver_name_lst):
			lns = axs[0,i_pn].plot(
				np.array(param.number_simulations_lst),
				np.mean(reward_plot_data[i_pn,:,i_sn,:],axis=1),
				label = solver_name)
			axs[0,i_pn].fill_between(
				np.array(param.number_simulations_lst),
				np.mean(reward_plot_data[i_pn,:,i_sn,:],axis=1) - np.std(reward_plot_data[i_pn,:,i_sn,:],axis=1),
				np.mean(reward_plot_data[i_pn,:,i_sn,:],axis=1) + np.std(reward_plot_data[i_pn,:,i_sn,:],axis=1),
				color = lns[0].get_color(),
				alpha = 0.5)
		axs[0,i_pn].set_xscale('log')
		axs[0,i_pn].set_title(problem_name)
		axs[0,i_pn].tick_params(axis='x', rotation=45)
	axs[0,0].legend(loc='best')
	axs[0,0].set_ylabel("Total Reward")
	fig.suptitle("Total Reward Regression Test")


def make_movie(sim_result,instance,filename):

	from matplotlib import animation

	states = sim_result["states"]
	times = sim_result["times"] 

	if instance["problem"].name in ["example3","example4"]:
		fig,ax = make_3d_fig()
	else:
		fig,ax = make_fig()

	def init(): 
		ax.clear()
		ax.grid(True)

	# animate over trajectory
	def animate(i_t):
		init()
		#print(i_t/len(times))
		time_idxs = range(i_t) #times[0:i_t]
		states_i = states[time_idxs]
		if i_t < 2:
			return ln 
		else:
			instance["problem"].render(states=states_i,fig=fig,ax=ax)
		return ln 

	ln = ax.plot([],[],[])
	anim = animation.FuncAnimation(fig, animate, frames=len(times)+1, interval=1)
	anim.save(filename, fps=10, dpi=120)
