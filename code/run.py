
# standard 
import csv
import numpy as np 
import multiprocessing as mp
import itertools
import os
import time as time_module
from queue import Queue

# custom 
from param import Param 
from problems.problem import get_problem
from solvers.solver import get_solver 
from learning.oracles import get_oracles 
import plotter 
from util import init_tqdm, update_tqdm


def initialize_problem(problem, initial_seed=None):
	if initial_seed is None:
		return problem.initialize()
	np_state = np.random.get_state()
	np.random.seed(initial_seed)
	initial_state = problem.initialize()
	np.random.set_state(np_state)
	return initial_state


def make_instance(param, initial_seed=None):

	instance = dict() 

	problem = get_problem(param.problem_name)
	policy_oracle,value_oracle = get_oracles(problem,
		value_oracle_name = param.value_oracle_name,
		value_oracle_path = param.value_oracle_path,
		policy_oracle_name = param.policy_oracle_name,
		policy_oracle_paths = param.policy_oracle_paths
		)
	solver = get_solver(param.solver_name,
		policy_oracle=policy_oracle,
		value_oracle=value_oracle,
		search_depth=param.search_depth,
		number_simulations=param.number_simulations,
		C_pw=param.C_pw,
		alpha_pw=param.alpha_pw,
		C_exp=param.C_exp,
		alpha_exp=param.alpha_exp,
		beta_policy=param.beta_policy,
		beta_value=param.beta_value,
		vis_on=param.vis_on,
		team_methods=getattr(param, "team_methods", None),
		team_method_settings=getattr(param, "team_method_settings", None))
	configure_paper_tree_capture(solver, param)

	instance["policy_oracle"] = policy_oracle
	instance["value_oracle"] = value_oracle
	instance["problem"] = problem 
	instance["solver"] = solver 
	instance["initial_state"] = initialize_problem(problem, initial_seed=initial_seed)
	instance["initial_seed"] = initial_seed

	# instance["initial_state"] = np.array([
	# 	# [-1],[3], # state for single robot, 2d single integrator problems
	# 	# [1],[1],[1],[-2],[0],[0], # state for homicidal chauffeur 
	# 	[1],[1],[-2],[1],[np.pi],[0], # state for homicidal chauffeur 
	# 	])

	return instance 


def configure_paper_tree_capture(solver, param):
	attrs = [
		"paper_tree_density_on",
		"paper_tree_topk_on",
		"paper_tree_capture_once",
		"paper_tree_capture_step",
		"paper_tree_capture_turn",
	]
	for attr in attrs:
		if hasattr(param, attr):
			setattr(solver, attr, getattr(param, attr))
	if hasattr(solver, "team_solvers"):
		for team_solver in solver.team_solvers.values():
			configure_paper_tree_capture(team_solver, param)


def run_instance(rank,queue,total,instance,verbose=False,tqdm_on=True):
	# input: 
	#	- 
	# outputs:
	# 	- dict of sim result 

	times, states, actions, observations, rewards, decision_times = [],[],[],[],[],[]
	team_decision_times = {"evaders": [], "pursuers": []}

	if verbose:
		print('   running sim with... \n\t{} \n\t{} \n\t{}'.format(\
			instance["problem"],
			instance["solver"],
			instance["initial_state"]
			))

	problem = instance["problem"] 
	solver = instance["solver"] 
	curr_state = instance["initial_state"]

	# print('rank',rank)
	# print('total',total)
	if tqdm_on:	pbar = init_tqdm(rank,total)

	states.append(curr_state)
	times.append(problem.times[0])
	for step,time in enumerate(problem.times[1:]):

		if verbose and not tqdm_on: print('\t\t t = {}/{}'.format(step,len(problem.times)))
		
		decision_start = time_module.perf_counter()
		if hasattr(solver, "current_step"):
			solver.current_step = step
		action = solver.policy(problem,curr_state)
		decision_times.append(time_module.perf_counter() - decision_start)
		for team_name, elapsed in getattr(solver, "last_team_decision_times", {}).items():
			if team_name not in team_decision_times:
				team_decision_times[team_name] = []
			team_decision_times[team_name].append(elapsed)

		dt = problem.dt 
		if solver.solver_name in ["PUCT_V2","C_PUCT_V2"]:
			dt = action[-1,0]
			action = action[0:-1,:]

		reward = problem.reward(curr_state,action)
		next_state = problem.step(curr_state,action,dt)
		done = problem.is_terminal(next_state)

		times.append(time)
		states.append(next_state)
		actions.append(action)
		rewards.append(reward)

		if tqdm_on: update_tqdm(rank,1,queue,pbar)

		if done: 
			break 
		else: 
			curr_state = next_state

	if verbose:	print('completed sim.')
	if verbose: problem.render(states=np.array(states))

	sim_result = dict()
	sim_result["instance"] = instance
	sim_result["times"] = times 
	sim_result["states"] = np.array(states)
	sim_result["actions"] = np.array(actions)
	sim_result["rewards"] = np.array(rewards)
	sim_result["decision_times"] = np.array(decision_times)
	sim_result["team_decision_times"] = {
		team_name: np.array(times)
		for team_name, times in team_decision_times.items()
	}

	return sim_result

def make_trial_seeds(param, num_trials, name, random_max=2**31 - 1):
	if getattr(param, "fixed_initial_conditions", True):
		base_seed = int(getattr(param, name))
		return [base_seed + trial for trial in range(num_trials)]
	return [np.random.randint(random_max) for _ in range(num_trials)]


def worker_run_instance(rank,queue,num_trials,param,initial_seed,run_seed):
	np.random.seed(run_seed)
	instance = make_instance(param, initial_seed=initial_seed)
	instance["run_seed"] = run_seed
	total = num_trials * len(instance["problem"].times)
	sim_result = run_instance(rank,queue,total,instance)
	sim_result["initial_seed"] = initial_seed
	sim_result["run_seed"] = run_seed
	del sim_result["instance"]["solver"] # can't pickle bindings 
	return sim_result

def _worker_run_instance(arg):
	return worker_run_instance(*arg)


def _safe_label(text):
	return "".join(c if c.isalnum() or c in ["-", "_"] else "_" for c in str(text))


def _team_method_label(param, method):
	label = _safe_label(method)
	if method not in ["search_only", "search_learning"]:
		return label

	method_settings = getattr(param, "team_method_settings", {}).get(method, {})
	number_simulations = method_settings.get("number_simulations", None)
	if number_simulations is None and method == "search_learning":
		number_simulations = getattr(param, "number_simulations", None)
	if number_simulations is None:
		return label
	return "{}-numsim{}".format(label, _safe_label(number_simulations))


def get_run_label(param):
	if getattr(param, "solver_name", None) == "MixedTeam":
		team_methods = getattr(param, "team_methods", {})
		evader_method = team_methods.get("evaders", "unknown")
		pursuer_method = team_methods.get("pursuers", "unknown")
		return "evaders-{}_pursuers-{}".format(
			_team_method_label(param, evader_method),
			_team_method_label(param, pursuer_method),
		)
	return _safe_label(getattr(param, "solver_name", "run"))


def _state_matrix(states):
	states = np.asarray(states, dtype=float)
	if states.size == 0:
		return np.empty((0, 0))
	return states.reshape((states.shape[0], -1))


def _reward_matrix(rewards):
	rewards = np.asarray(rewards, dtype=float)
	if rewards.size == 0:
		return np.empty((0, 0))
	return rewards.reshape((rewards.shape[0], -1))


def _capture_stats(sim_result):
	problem = sim_result["instance"]["problem"]
	states = _state_matrix(sim_result["states"])
	times = np.asarray(sim_result["times"], dtype=float)
	if states.shape[0] == 0 or not hasattr(problem, "evaders") or not hasattr(problem, "active_idxs"):
		return 0, None

	initial_active = sum(states[0, problem.active_idxs[robot]] > 0.5 for robot in problem.evaders)
	capture_count = int(initial_active - sum(states[-1, problem.active_idxs[robot]] > 0.5 for robot in problem.evaders))
	capture_count = max(capture_count, 0)
	first_capture_time = None
	for i_state, state in enumerate(states):
		active_count = sum(state[problem.active_idxs[robot]] > 0.5 for robot in problem.evaders)
		if active_count < initial_active:
			first_capture_time = float(times[i_state]) if i_state < len(times) else float(i_state)
			break
	return capture_count, first_capture_time


def summarize_results(sim_results, param):
	num_trials = len(sim_results)
	team_methods = getattr(param, "team_methods", {})
	capture_counts = []
	success_capture_times = []
	pursuer_returns = []
	decision_times = []
	evader_decision_times = []
	pursuer_decision_times = []

	for sim_result in sim_results:
		problem = sim_result["instance"]["problem"]
		capture_count, first_capture_time = _capture_stats(sim_result)
		capture_counts.append(capture_count)
		if capture_count > 0 and first_capture_time is not None:
			success_capture_times.append(first_capture_time)

		rewards = _reward_matrix(sim_result["rewards"])
		if rewards.shape[0] > 0 and hasattr(problem, "pursuers"):
			pursuer_idxs = [robot for robot in problem.pursuers if robot < rewards.shape[1]]
			if len(pursuer_idxs) > 0:
				pursuer_returns.append(float(np.sum(rewards[:, pursuer_idxs])))

		decision_times.extend(np.asarray(sim_result.get("decision_times", []), dtype=float).reshape(-1).tolist())
		team_decision_times = sim_result.get("team_decision_times", {})
		evader_decision_times.extend(
			np.asarray(team_decision_times.get("evaders", []), dtype=float).reshape(-1).tolist()
		)
		pursuer_decision_times.extend(
			np.asarray(team_decision_times.get("pursuers", []), dtype=float).reshape(-1).tolist()
		)

	capture_counts = np.asarray(capture_counts, dtype=float)
	return {
		"evader_method": team_methods.get("evaders", ""),
		"pursuer_method": team_methods.get("pursuers", ""),
		"num_trials": num_trials,
		"fixed_initial_conditions": getattr(param, "fixed_initial_conditions", ""),
		"initial_seed": getattr(param, "initial_seed", ""),
		"run_seed": getattr(param, "run_seed", ""),
		"capture_success_rate": float(np.mean(capture_counts > 0)) if num_trials > 0 else 0.0,
		"capture_one_success_rate": float(np.mean(capture_counts == 1)) if num_trials > 0 else 0.0,
		"capture_two_success_rate": float(np.mean(capture_counts >= 2)) if num_trials > 0 else 0.0,
		"avg_capture_count": float(np.mean(capture_counts)) if num_trials > 0 else 0.0,
		"avg_success_capture_time": float(np.mean(success_capture_times)) if len(success_capture_times) > 0 else np.nan,
		"avg_pursuer_team_return": float(np.mean(pursuer_returns)) if len(pursuer_returns) > 0 else np.nan,
		"avg_step_decision_time_s": float(np.mean(decision_times)) if len(decision_times) > 0 else np.nan,
		"avg_evader_step_decision_time_s": float(np.mean(evader_decision_times)) if len(evader_decision_times) > 0 else np.nan,
		"avg_pursuer_step_decision_time_s": float(np.mean(pursuer_decision_times)) if len(pursuer_decision_times) > 0 else np.nan,
	}


def save_summary_csv(summary, filename):
	file_dir, _ = os.path.split(filename)
	if len(file_dir) > 0 and not os.path.isdir(file_dir):
		os.makedirs(file_dir)
	fieldnames = [
		"evader_method",
		"pursuer_method",
		"num_trials",
		"fixed_initial_conditions",
		"initial_seed",
		"run_seed",
		"capture_success_rate",
		"capture_one_success_rate",
		"capture_two_success_rate",
		"avg_capture_count",
		"avg_success_capture_time",
		"avg_pursuer_team_return",
		"avg_step_decision_time_s",
		"avg_evader_step_decision_time_s",
		"avg_pursuer_step_decision_time_s",
	]
	with open(filename, "w", newline="") as f:
		writer = csv.DictWriter(f, fieldnames=fieldnames)
		writer.writeheader()
		writer.writerow(summary)


def _find_paper_tree_solver(solver):
	if solver is None:
		return None
	if getattr(solver, "paper_tree_state", None) is not None:
		return solver
	if hasattr(solver, "team_solvers"):
		for team_solver in solver.team_solvers.values():
			found = _find_paper_tree_solver(team_solver)
			if found is not None:
				return found
	return None


def save_paper_tree_figures(sim_results, param, run_label):
	if not (
		getattr(param, "paper_tree_density_on", False)
		or getattr(param, "paper_tree_topk_on", False)
	):
		return

	tree_solver = None
	problem = None
	for sim_result in sim_results:
		instance = sim_result.get("instance", {})
		tree_solver = _find_paper_tree_solver(instance.get("solver", None))
		if tree_solver is not None:
			problem = instance.get("problem", None)
			break

	if tree_solver is None or problem is None:
		print("paper tree figure skipped: no captured tree found")
		return

	tree_state = tree_solver.paper_tree_state
	tree_info = getattr(tree_solver, "paper_tree_info", None)
	capture_suffix = "step{}_turn{}".format(
		getattr(tree_solver, "paper_tree_step", "unknown"),
		getattr(tree_solver, "paper_tree_turn", "unknown"),
	)

	if getattr(param, "paper_tree_density_on", False):
		density_path = "../current/plots/tree_density_{}_{}.pdf".format(run_label, capture_suffix)
		fig = plotter.plot_tree_density(
			problem,
			tree_state,
			tree_info=tree_info,
			bins=getattr(param, "paper_tree_density_bins", 65),
			weight_mode=getattr(param, "paper_tree_density_weight_mode", "log_visits"),
			title="Search density, {}".format(capture_suffix),
		)
		plotter.save_fig(fig, density_path)
		print("saved paper tree density to {}".format(density_path))

	if getattr(param, "paper_tree_topk_on", False):
		topk_path = "../current/plots/tree_topk_{}_{}.pdf".format(run_label, capture_suffix)
		fig = plotter.plot_tree_topk(
			problem,
			tree_state,
			tree_info=tree_info,
			top_fraction=getattr(param, "paper_tree_top_fraction", 0.05),
			max_edges=getattr(param, "paper_tree_top_max_edges", 700),
			min_visits=getattr(param, "paper_tree_top_min_visits", 1),
			title="Top search-tree branches, {}".format(capture_suffix),
		)
		plotter.save_fig(fig, topk_path)
		print("saved paper tree top-k to {}".format(topk_path))


if __name__ == '__main__':

	param = Param()

	print('running sim...')
	if param.parallel_on:
		pool = mp.Pool(mp.cpu_count() - 1)
		params = [Param() for _ in range(param.num_trials)]
		initial_seeds = make_trial_seeds(param, param.num_trials, "initial_seed")
		run_seeds = make_trial_seeds(param, param.num_trials, "run_seed")
		args = list(zip(
			itertools.count(), 
			itertools.repeat(mp.Manager().Queue()),
			itertools.repeat(param.num_trials),
			params,initial_seeds,run_seeds))
		sim_results = pool.imap_unordered(_worker_run_instance, args)
		# sim_results = pool.map(_worker, args)
		pool.close()
		pool.join()
		sim_results = list(sim_results)
	else:
		initial_seed = make_trial_seeds(param, 1, "initial_seed")[0]
		run_seed = make_trial_seeds(param, 1, "run_seed")[0]
		np.random.seed(run_seed)
		instance = make_instance(param, initial_seed=initial_seed)
		instance["run_seed"] = run_seed
		sim_results = [run_instance(0,Queue(),len(instance["problem"].times),instance,verbose=True)]
		sim_results[0]["initial_seed"] = initial_seed
		sim_results[0]["run_seed"] = run_seed

	if param.movie_on: 
		print('making movie...')
		plotter.make_movie(sim_results[0],sim_results[0]["instance"],"../current/plots/vid.gif")
		plotter.open_figs("../current/plots/vid.gif")	

	# save/load results
	# todo

	# plotting 
	print('plotting results...')
	run_label = get_run_label(param)
	save_paper_tree_figures(sim_results, param, run_label)
	for sim_result in sim_results:
		plotter.plot_sim_result(sim_result)
		sim_result["instance"]["problem"].render(states=sim_result["states"])
		#if param.pretty_plot_on and hasattr(sim_result["instance"]["problem"], 'pretty_plot') :
		#	sim_result["instance"]["problem"].pretty_plot(sim_result)

	plot_path = "../current/plots/run_{}.pdf".format(run_label)
	summary_path = "../current/plots/summary_{}.csv".format(run_label)
	summary = summarize_results(sim_results, param)
	save_summary_csv(summary, summary_path)
	plotter.save_figs(plot_path)
	plotter.open_figs(plot_path)
	print("saved summary to {}".format(summary_path))
