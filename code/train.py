
# standard 
import numpy as np 
import torch 
import random 
import pickle 
import tempfile
import itertools
import time 
import os
import multiprocessing as mp
from collections import deque
from tqdm import tqdm 
from queue import Queue, Empty 
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.nn import MSELoss

# custom
import plotter 
from problems.problem import get_problem
from solvers.solver import get_solver 
from learning.oracles import get_oracles
from run import run_instance
from util import write_dataset, get_dataset_fn, get_oracle_fn, format_dir, get_temp_fn, init_tqdm, update_tqdm

# solver 
num_simulations = 3000
search_depth = 200
C_pw = 2.0
alpha_pw = 0.5
C_exp = 1.0
alpha_exp = 0.25
beta_policy = 0.5
beta_value = 0.5
parallel_on = True
solver_name = "C_PUCT_V1"
# solver_name = "PUCT_V1"
problem_name = "example8"
use_minco_dynamics = False  # True: MINCO, False: direct displacement propagation
policy_oracle_name = "gaussian"
value_oracle_name = "deterministic"

dirname = "../current/models"
plot_on= False	
# learning
L = 80
resume_training = False  # True: resume from existing models in dirname; False: start from scratch (clears models)
mode = 1 # 0: weighted sum, 1: best child, 2: subsamples 
num_D_pi = 30000
# num_D_pi = 500
# num_D_pi = 200
num_pi_eval = 3000
num_D_v = 10000
num_v_eval = 2000
num_subsamples = 5
num_self_play_plots = 10
learning_rate = 7e-4
num_epochs = 200
# num_epochs = 100
batch_size = 4096
train_test_split = 0.8

# replay buffer
replay_buffer_multiplier = 20
policy_replay_buffers = None
value_replay_buffer = None


def split_work_count(total, num_workers):
	num_workers = max(1, int(num_workers))
	base = int(total) // num_workers
	remainder = int(total) % num_workers
	return [base + (1 if idx < remainder else 0) for idx in range(num_workers)]


def drain_progress_queue(queue, pbar):
	count = 0
	while True:
		try:
			count += queue.get_nowait()
		except Empty:
			break
	if count > 0:
		pbar.update(count)


def run_parallel_workers(worker_wrapper, args, queue, total, desc):
	if len(args) == 0:
		return
	with mp.Pool(len(args)) as pool:
		iterator = pool.imap_unordered(worker_wrapper, args, chunksize=1)
		completed = 0
		with tqdm(total=total, desc=desc) as pbar:
			while completed < len(args):
				try:
					iterator.next(0.2)
					completed += 1
				except mp.TimeoutError:
					pass
				drain_progress_queue(queue, pbar)
			drain_progress_queue(queue, pbar)


# MICE-like training
class Dataset(torch.utils.data.Dataset):
	
	def __init__(self, src_file, encoding_dim, target_dim, device='cpu'):
		with open(src_file, 'rb') as h:
			datapoints = np.load(h)
		self.X_np, self.target_np = datapoints[:,0:encoding_dim], datapoints[:,encoding_dim:]
		self.X_torch = torch.from_numpy(np.ascontiguousarray(self.X_np)).float()
		self.target_torch = torch.from_numpy(np.ascontiguousarray(self.target_np)).float()

	def __len__(self):
		return self.X_torch.shape[0]

	def __getitem__(self, idx):
		return int(idx)

	def get_batch(self, idx):
		if not torch.is_tensor(idx):
			idx = torch.as_tensor(idx, dtype=torch.long)
		return self.X_torch[idx, :], self.target_torch[idx, :]

	def to(self,device):
		self.X_torch = self.X_torch.to(device)
		self.target_torch = self.target_torch.to(device)
		return self


class TensorBatchLoader:
	def __init__(self, dataset, batch_size, shuffle, device='cpu'):
		self.dataset = dataset
		self.batch_size = max(1, int(batch_size))
		self.shuffle = bool(shuffle)
		self.size = len(dataset)

	def __iter__(self):
		if self.size == 0:
			return
		if self.shuffle:
			indices = torch.randperm(self.size, dtype=torch.long)
		else:
			indices = torch.arange(self.size, dtype=torch.long)
		for start in range(0, self.size, self.batch_size):
			batch_idx = indices[start:start+self.batch_size]
			yield self.dataset.get_batch(batch_idx)

	def __len__(self):
		if self.size == 0:
			return 0
		return (self.size + self.batch_size - 1) // self.batch_size

# policy demonstration functions 
def sample_reachable_state(problem, policy_oracle=None):
    state = problem.initialize()

    if policy_oracle is not None and any(p is not None for p in policy_oracle):
        rollout_solver = get_solver("NeuralNetwork", policy_oracle=policy_oracle)
    else:
        rollout_solver = None

    k = np.random.randint(0, len(problem.times) - 1)
    for _ in range(k):
        if problem.is_terminal(state):
            break
        if rollout_solver is None:
            action = problem.sample_action()
        else:
            action = rollout_solver.policy(problem, state)
        state = problem.step(state, action, problem.dt)

    return state

def worker_edp_wrapper(arg):
	return worker_edp(*arg)

def make_policy_datapoints(problem, robot, state, actions, num_visits):
	robot_action_idx = problem.action_idxs[robot]
	encoding = problem.policy_encoding(state, robot).squeeze()

	if mode == 0:
		robot_actions = np.array(actions)[:, robot_action_idx]
		target = np.average(robot_actions, weights=num_visits, axis=0)
		return [np.append(encoding, target)]
	elif mode == 1:
		target = np.array(actions[np.argmax(num_visits)])[robot_action_idx]
		return [np.append(encoding, target)]
	elif mode == 2:
		choice_idxs = np.random.choice(len(actions), num_subsamples, p=num_visits / np.sum(num_visits))
		return [
			np.append(encoding, np.array(actions[choice_idx])[robot_action_idx])
			for choice_idx in choice_idxs
		]
	raise ValueError("unsupported policy dataset mode: {}".format(mode))

def worker_edp(rank,queue,seed,fn,problem,robot,num_per_pool,policy_oracle,value_oracle):

	np.random.seed(seed)
	pbar = init_tqdm(rank,num_per_pool) if queue is None else None
	datapoints = []

	solver = get_solver(
			solver_name,
			policy_oracle=policy_oracle,
			value_oracle=value_oracle,
			search_depth=search_depth,
			number_simulations=num_simulations,
			C_pw=C_pw,
			alpha_pw=alpha_pw,
			C_exp=C_exp,
			alpha_exp=alpha_exp,
			beta_policy= beta_policy,
			beta_value = beta_value
			)
	
	robot_action_idx = problem.action_idxs[robot]
	count = 0
	failure_count = 0
 
	while count < num_per_pool:
		state = sample_reachable_state(problem, policy_oracle)
		if problem.is_terminal(state):
			continue
		if hasattr(problem, "is_active") and not problem.is_active(state, robot):
			continue
		root_node = solver.search(problem,state,turn=robot)

		if False:
			encoding = problem.policy_encoding(state,robot).squeeze()
			datapoint = np.append(encoding, problem.gt_actions(state)[robot_action_idx].flatten())
			datapoints.append(datapoint)
			count += 1
			update_tqdm(rank,1,queue,pbar)
			continue
		
  		# print('rank: {}, completion: {}, success: {}'.format(rank,len(datapoints)/num_per_pool,root_node.success))
		if root_node.success:
			actions,num_visits = solver.get_child_distribution(root_node)
			datapoints.extend(make_policy_datapoints(problem, robot, state, actions, num_visits))

			count += 1
			update_tqdm(rank,1,queue,pbar)
		else:
			failure_count += 1
	
	print('rank {} completed with {} datapoints and {} failures.'.format(rank,len(datapoints),failure_count))
	np.save(fn,np.array(datapoints))	
	return datapoints

def worker_edp_group_wrapper(arg):
	return worker_edp_group(*arg)

def worker_edp_group(rank, queue, seed, fns, problem, robot_group, num_per_pool, policy_oracle, value_oracle):
	np.random.seed(seed)
	robot_group = [int(robot) for robot in robot_group]
	turn = robot_group[0]
	pbar = init_tqdm(rank, num_per_pool * len(robot_group)) if queue is None else None
	datapoints_by_robot = {robot: [] for robot in robot_group}
	counts_by_robot = {robot: 0 for robot in robot_group}
	failure_count = 0
	search_count = 0

	solver = get_solver(
			solver_name,
			policy_oracle=policy_oracle,
			value_oracle=value_oracle,
			search_depth=search_depth,
			number_simulations=num_simulations,
			C_pw=C_pw,
			alpha_pw=alpha_pw,
			C_exp=C_exp,
			alpha_exp=alpha_exp,
			beta_policy= beta_policy,
			beta_value = beta_value
			)

	while any(counts_by_robot[robot] < num_per_pool for robot in robot_group):
		state = sample_reachable_state(problem, policy_oracle)
		if problem.is_terminal(state):
			continue

		needed_robots = [
			robot for robot in robot_group
			if counts_by_robot[robot] < num_per_pool
			and (not hasattr(problem, "is_active") or problem.is_active(state, robot))
		]
		if len(needed_robots) == 0:
			continue

		root_node = solver.search(problem, state, turn=turn)
		search_count += 1
		if root_node.success:
			actions, num_visits = solver.get_child_distribution(root_node)
			num_added = 0
			for robot in needed_robots:
				datapoints_by_robot[robot].extend(
					make_policy_datapoints(problem, robot, state, actions, num_visits)
				)
				counts_by_robot[robot] += 1
				num_added += 1
			update_tqdm(rank, num_added, queue, pbar)
		else:
			failure_count += 1

	for robot, fn in zip(robot_group, fns):
		np.save(fn, np.array(datapoints_by_robot[robot]))

	# print(
	# 	'rank {} group {} completed with counts {} from {} searches and {} failures.'.format(
	# 		rank, robot_group, counts_by_robot, search_count, failure_count
	# 	)
	# )
	return datapoints_by_robot


def make_expert_demonstration_pi(problem,robot,policy_oracle,value_oracle):
	global policy_replay_buffers

	start_time = time.time()
	print('making expert demonstration pi...')

	paths = []
	if parallel_on: 
		ncpu = max(1, mp.cpu_count() - 18)
		print("Total CPU:{}".format(ncpu))
		worker_counts = split_work_count(num_D_pi, ncpu)

		seeds = [] 
		for i in range(ncpu):
			# fd, path = tempfile.mkstemp()
			# path = path + ".npy"
			# fds.append(fd)
			path = get_temp_fn(dirname,i)
			paths.append(path)
			seeds.append(np.random.randint(10000))

		with mp.Manager() as manager:
			queue = manager.Queue()
			args = list(zip(itertools.count(), itertools.repeat(queue), seeds, paths, itertools.repeat(problem), \
				itertools.repeat(robot), worker_counts, itertools.repeat(policy_oracle), itertools.repeat(value_oracle)))
			run_parallel_workers(
				worker_edp_wrapper,
				args,
				queue,
				sum(worker_counts),
				"policy r{}".format(robot),
			)

	else: 
		# fd,path = tempfile.mkstemp()
		# path = path + ".npy"
		# fds.append(fd)
		path = get_temp_fn(dirname,0)
		seed = np.random.randint(10000)
		paths.append(path)
		worker_edp_wrapper((0,None,seed,path,problem,robot,num_D_pi,policy_oracle,value_oracle))
		# rank,queue,seed,fn,problem,robot,num_per_pool,policy_oracle,value_oracle
	new_datapoints = []
	for path in paths: 
	# for fd,path in list(zip(fds,paths)): 
		# os.close(fd) 
		new_datapoints.extend(list(np.load(path)))
		os.remove(path)

	if policy_replay_buffers is None:
		policy_replay_capacity = num_D_pi * replay_buffer_multiplier * (num_subsamples if mode == 2 else 1)
		policy_replay_buffers = [deque(maxlen=policy_replay_capacity) for _ in range(problem.num_robots)]

	policy_replay_buffers[robot].extend(new_datapoints)
	if len(policy_replay_buffers[robot]) == 0:
		raise RuntimeError('policy replay buffer is empty')

	base_samples = num_D_pi * (num_subsamples if mode == 2 else 1)
	num_policy_samples = min(5 * base_samples, len(policy_replay_buffers[robot]))
	
	buffer_list = list(policy_replay_buffers[robot])
	weights = np.arange(1, len(buffer_list) + 1)			# 按时间线性增长的权重，越新的数据权重越大
	weights = weights / np.sum(weights)
	indices = np.random.choice(len(buffer_list), size=num_policy_samples, replace=False, p=weights)
	datapoints = [buffer_list[i] for i in indices]

	split = int(len(datapoints)*train_test_split)
	robot_action_dim = len(problem.action_idxs[robot])
	train_dataset = datapoints_to_dataset(datapoints[0:split],"train_policy",\
		problem.policy_encoding_dim,robot_action_dim,robot=robot)
	test_dataset = datapoints_to_dataset(datapoints[split:],"test_policy",\
		problem.policy_encoding_dim,robot_action_dim,robot=robot)
	if plot_on:
		plotter.plot_policy_dataset(problem,\
			[[train_dataset.X_np,train_dataset.target_np],[test_dataset.X_np,test_dataset.target_np]],\
			["Train","Test"],robot)
		plotter.save_figs("{}/dataset_policy_l{}_i{}.pdf".format(dirname,l,robot))
	print('expert demonstration pi completed in {}s.'.format(time.time()-start_time))	
	return train_dataset, test_dataset

def make_expert_demonstration_pi_group(problem, robot_group, policy_oracle, value_oracle):
	global policy_replay_buffers

	robot_group = [int(robot) for robot in robot_group]
	start_time = time.time()
	print('making expert demonstration pi...')

	paths_by_worker = []
	if parallel_on:
		ncpu = max(1, mp.cpu_count() - 3)
		print("Total CPU:{}".format(ncpu))
		worker_counts = split_work_count(num_D_pi, ncpu)

		seeds = []
		for i in range(ncpu):
			worker_paths = []
			for robot in robot_group:
				worker_paths.append("{}/temp_{}_r{}.npy".format(dirname, i, robot))
			paths_by_worker.append(worker_paths)
			seeds.append(np.random.randint(10000))

		with mp.Manager() as manager:
			queue = manager.Queue()
			args = list(zip(
				itertools.count(),
				itertools.repeat(queue),
				seeds,
				paths_by_worker,
				itertools.repeat(problem),
				itertools.repeat(robot_group),
				worker_counts,
				itertools.repeat(policy_oracle),
				itertools.repeat(value_oracle),
			))
			run_parallel_workers(
				worker_edp_group_wrapper,
				args,
				queue,
				sum(worker_counts) * len(robot_group),
				"policy group {}".format(robot_group),
			)
	else:
		worker_paths = []
		for robot in robot_group:
			worker_paths.append("{}/temp_0_r{}.npy".format(dirname, robot))
		paths_by_worker.append(worker_paths)
		seed = np.random.randint(10000)
		worker_edp_group_wrapper((
			0, None, seed, worker_paths, problem, robot_group, num_D_pi, policy_oracle, value_oracle
		))

	new_datapoints_by_robot = {robot: [] for robot in robot_group}
	for worker_paths in paths_by_worker:
		for robot, path in zip(robot_group, worker_paths):
			new_datapoints_by_robot[robot].extend(list(np.load(path)))
			os.remove(path)

	if policy_replay_buffers is None:
		policy_replay_capacity = num_D_pi * replay_buffer_multiplier * (num_subsamples if mode == 2 else 1)
		policy_replay_buffers = [deque(maxlen=policy_replay_capacity) for _ in range(problem.num_robots)]

	datasets_by_robot = {}
	base_samples = num_D_pi * (num_subsamples if mode == 2 else 1)
	for robot in robot_group:
		policy_replay_buffers[robot].extend(new_datapoints_by_robot[robot])
		if len(policy_replay_buffers[robot]) == 0:
			raise RuntimeError('policy replay buffer is empty for robot {}'.format(robot))

		num_policy_samples = min(5 * base_samples, len(policy_replay_buffers[robot]))
		buffer_list = list(policy_replay_buffers[robot])
		weights = np.arange(1, len(buffer_list) + 1)
		weights = weights / np.sum(weights)
		indices = np.random.choice(len(buffer_list), size=num_policy_samples, replace=False, p=weights)
		datapoints = [buffer_list[i] for i in indices]

		split = int(len(datapoints) * train_test_split)
		robot_action_dim = len(problem.action_idxs[robot])
		train_dataset = datapoints_to_dataset(
			datapoints[0:split], "train_policy", problem.policy_encoding_dim, robot_action_dim, robot=robot
		)
		test_dataset = datapoints_to_dataset(
			datapoints[split:], "test_policy", problem.policy_encoding_dim, robot_action_dim, robot=robot
		)
		if plot_on:
			plotter.plot_policy_dataset(
				problem,
				[[train_dataset.X_np, train_dataset.target_np], [test_dataset.X_np, test_dataset.target_np]],
				["Train", "Test"],
				robot,
			)
			plotter.save_figs("{}/dataset_policy_l{}_i{}.pdf".format(dirname, l, robot))
		datasets_by_robot[robot] = (train_dataset, test_dataset)

	print('expert demonstration pi completed in {}s.'.format(time.time() - start_time))
	return datasets_by_robot


def datapoints_to_dataset(datapoints,oracle_name,encoding_dim,target_dim,robot=0):
	dataset_fn = get_dataset_fn(oracle_name,l,robot=robot)
	datapoints = np.array(datapoints)
	write_dataset(datapoints,dataset_fn)
	dataset = Dataset(dataset_fn,encoding_dim,target_dim)
	return dataset


# value estimate 
def worker_edv_wrapper(args):
	return worker_edv(*args)


def worker_edv(rank,queue,fn,seed,problem,num_states_per_pool,policy_oracle):

	solver = get_solver(
			"NeuralNetwork",
			policy_oracle=policy_oracle)

	instance = {
		"problem" : problem,
		"solver" : solver,
	}
	np.random.seed(seed)
	
	count = 0 
	pbar = init_tqdm(rank,num_states_per_pool) if queue is None else None
	datapoints = []
	while len(datapoints) < num_states_per_pool:	
		state = sample_reachable_state(problem, policy_oracle)
		if problem.is_terminal(state):
			continue
		instance["initial_state"] = state
  		# 使用和run.py一样的接口
		sim_result = run_instance(0,Queue(),0,instance,verbose=False,tqdm_on=False)		
		value = calculate_value(problem,sim_result)
		encoding = problem.value_encoding(state).squeeze()
		datapoint = np.append(encoding,value)
		datapoints.append(datapoint)
		update_tqdm(rank,1,queue,pbar)
	np.save(fn,np.array(datapoints))
	return datapoints


def make_expert_demonstration_v(problem, l): 
	global value_replay_buffer

	start_time = time.time()
	print('making value dataset...')

	_, policy_oracle_paths = get_oracle_fn(l,problem.num_robots)
	policy_oracle,_ = get_oracles(problem,
		policy_oracle_name = policy_oracle_name, 
		policy_oracle_paths = policy_oracle_paths
		)

	paths = []
	if parallel_on: 
		ncpu = max(1, mp.cpu_count() - 1)
		worker_counts = split_work_count(num_D_v, ncpu)
		seeds = [] 
		for i in range(ncpu):
			# _, path = tempfile.mkstemp()
			# paths.append(path + '.npy')
			paths.append(get_temp_fn(dirname,i))
			seeds.append(np.random.randint(10000))
		with mp.Manager() as manager:
			queue = manager.Queue()
			args = list(zip(itertools.count(), itertools.repeat(queue),paths, \
				seeds, itertools.repeat(problem), worker_counts, \
				itertools.repeat(policy_oracle) ))

			run_parallel_workers(
				worker_edv_wrapper,
				args,
				queue,
				sum(worker_counts),
				"value",
			)

	else:
		# _,path = tempfile.mkstemp()
		# paths.append(path + '.npy')
		paths = [get_temp_fn(dirname,0)]
		seed = np.random.randint(10000)
		worker_edv_wrapper((0,None,paths[0],seed,problem,num_D_v,policy_oracle))

	new_datapoints = []
	plot_count = 0 
	for path in paths:
		datapoints_i = np.load(path,allow_pickle=True)
		new_datapoints.extend(datapoints_i)
		os.remove(path)

	if value_replay_buffer is None:
		value_replay_capacity = num_D_v * replay_buffer_multiplier
		value_replay_buffer = deque(maxlen=value_replay_capacity)

	value_replay_buffer.extend(new_datapoints)
	if len(value_replay_buffer) == 0:
		raise RuntimeError('value replay buffer is empty')

	num_value_samples = min(5 * num_D_v, len(value_replay_buffer))
	
	buffer_list = list(value_replay_buffer)
	weights = np.arange(1, len(buffer_list) + 1)
	weights = weights / np.sum(weights)
	indices = np.random.choice(len(buffer_list), size=num_value_samples, replace=False, p=weights)
	datapoints = [buffer_list[i] for i in indices]

	split = int(len(datapoints)*train_test_split)
	train_dataset = datapoints_to_dataset(datapoints[0:split],"train_value",\
		problem.value_encoding_dim,problem.num_robots)
	test_dataset = datapoints_to_dataset(datapoints[split:],"test_value",\
		problem.value_encoding_dim,problem.num_robots)
	if plot_on:
		plotter.plot_value_dataset(problem,
			[[train_dataset.X_np,train_dataset.target_np],[test_dataset.X_np,test_dataset.target_np]],
			["Train","Test"])
		plotter.save_figs("{}/dataset_value_l{}.pdf".format(dirname,l))
	print('expert demonstration v completed in {}s.'.format(time.time()-start_time))	
	return train_dataset, test_dataset


def calculate_value(problem,sim_result):
	value = np.zeros((problem.num_robots,1))
	states = sim_result["states"]
	actions = sim_result["actions"]
	for step,(state,action) in enumerate(zip(states,actions)):
		reward = problem.normalized_reward(state,action)
		value += (problem.gamma ** step) * reward 
	return value 


def find_latest_complete_iteration(dirname, num_robots):
	import re
	import glob
	latest_l = -1
	value_files = glob.glob(os.path.join(dirname, "model_value_l*.pt"))
	policy_files = glob.glob(os.path.join(dirname, "model_policy_l*_i*.pt"))

	value_ls = set()
	for f in value_files:
		m = re.search(r"model_value_l(\d+)\.pt", os.path.basename(f))
		if m:
			value_ls.add(int(m.group(1)))

	policy_ls = [set() for _ in range(num_robots)]
	for f in policy_files:
		m = re.search(r"model_policy_l(\d+)_i(\d+)\.pt", os.path.basename(f))
		if m:
			l_val, i_val = int(m.group(1)), int(m.group(2))
			if 0 <= i_val < num_robots:
				policy_ls[i_val].add(l_val)

	for l_val in sorted(value_ls, reverse=True):
		is_complete = True
		for i in range(num_robots):
			if l_val not in policy_ls[i]:
				is_complete = False
				break
		if is_complete:
			latest_l = l_val
			break
	return latest_l

def train_model(problem,train_dataset,test_dataset,l,oracle_name,robot=0,resume_from_existing=False):
	start_time = time.time()
	print('training model...')

	device = "cuda" if torch.cuda.is_available() else "cpu"
	value_oracle_path, policy_oracle_paths = get_oracle_fn(l,problem.num_robots)

	if oracle_name == "policy":
		model_fn = policy_oracle_paths[robot]
		init_paths = [None for _ in range(problem.num_robots)]
		if resume_from_existing and os.path.exists(model_fn):
			print('Loading existing policy model from {} as starting point'.format(model_fn))
			init_paths[robot] = model_fn
		model, _ = get_oracles(problem,
			policy_oracle_name = policy_oracle_name,
			policy_oracle_paths = init_paths,
			force = True
			)
		model = model[robot]
	elif oracle_name == "value":
		model_fn = value_oracle_path
		init_path = model_fn if (resume_from_existing and os.path.exists(model_fn)) else None
		if init_path:
			print('Loading existing value model from {} as starting point'.format(init_path))
		_, model = get_oracles(problem,
			value_oracle_name = value_oracle_name,
			value_oracle_path = init_path,
			force = True
			)
	model.to(device)
	if device == "cuda":
		torch.backends.cudnn.benchmark = True
		torch.backends.cuda.matmul.allow_tf32 = True
		torch.backends.cudnn.allow_tf32 = True

	optimizer = torch.optim.AdamW(model.parameters(),lr=learning_rate)
	# scheduler = ReduceLROnPlateau(optimizer, 'min', \
	# 	factor=0.5, patience=50, min_lr=1e-4)

	train_size = len(train_dataset)
	test_size = len(test_dataset)
	if train_size == 0 or test_size == 0:
		raise RuntimeError("empty dataset: train_size={}, test_size={}".format(train_size, test_size))

	train_batch_size = min(batch_size, train_size)
	test_batch_size = min(256, test_size)

	train_loader = TensorBatchLoader(train_dataset,batch_size=train_batch_size,shuffle=True)
	test_loader = TensorBatchLoader(test_dataset,batch_size=test_batch_size,shuffle=False)
 
	train_dataset.to(device)
	test_dataset.to(device)

	losses = []
	best_test_loss = np.inf
	for epoch in tqdm(range(num_epochs)): 
		train_epoch_loss = train(model,optimizer,train_loader,device)
		test_epoch_loss = test(model,test_loader,device)
		# if scheduler is enabled, step it here
		# scheduler.step(test_epoch_loss)
		losses.append((train_epoch_loss,test_epoch_loss))
		if test_epoch_loss < best_test_loss:
			best_test_loss = test_epoch_loss
			state_dict_cpu = {k: v.detach().cpu() for k, v in model.state_dict().items()}
			torch.save(state_dict_cpu,model_fn)
	if 1:
		plotter.plot_loss(losses)
		plotter.save_figs("{}/losses_{}_l{}_i{}.pdf".format(dirname,oracle_name,l,robot))
	print('training model completed in {}s.'.format(time.time()-start_time))
	return 


def train(model,optimizer,loader,device):
	epoch_loss = 0
	loss_by_components = []
	num_batches = 0
	non_blocking = (device == "cuda")
	for step, (x,target) in enumerate(loader):
		# x = x.to(device, non_blocking=non_blocking)
		# target = target.to(device, non_blocking=non_blocking)
		optimizer.zero_grad(set_to_none=True)
		loss = model.loss_fnc(x,target)
		loss.backward()
		optimizer.step()
		epoch_loss += float(loss)
		num_batches += 1
	return epoch_loss/max(1,num_batches)


def test(model,loader,device):
	epoch_loss = 0
	loss_by_components = []
	num_batches = 0
	non_blocking = (device == "cuda")
	with torch.no_grad():
		for step, (x,target) in enumerate(loader):
			x = x.to(device, non_blocking=non_blocking)
			target = target.to(device, non_blocking=non_blocking)
			loss = model.loss_fnc(x,target)
			epoch_loss += float(loss)
			num_batches += 1
	return epoch_loss/max(1,num_batches)


def eval_value(problem,l):
	
	value_oracle_path, policy_oracle_paths = get_oracle_fn(l,problem.num_robots)

	_, value_oracle = get_oracles(problem,
		value_oracle_name = value_oracle_name,
		value_oracle_path = value_oracle_path
		)

	states = []
	values = []
	encodings = [] 
	for _ in range(num_v_eval):
		state = problem.initialize()
		encoding = problem.value_encoding(state)
		value = value_oracle.eval(problem,state)
		states.append(state)
		values.append(value)
		encodings.append(encoding.reshape((problem.value_encoding_dim,1)))

	states = np.array(states).squeeze(axis=2)
	values = np.array(values).squeeze(axis=2)
	encodings = np.array(encodings).squeeze(axis=2)
	if plot_on:
		plotter.plot_value_dataset(problem,[[encodings,values]],["Eval"])
		plotter.plot_value_dataset(problem,[[states,values]],["Eval"])
		plotter.save_figs("{}/value_eval_l{}.pdf".format(dirname,l))


def eval_policy(problem,l,robot):

	value_oracle_path, policy_oracle_paths = get_oracle_fn(l,problem.num_robots)
	for robot_i,path in enumerate(policy_oracle_paths):
		if robot_i != robot:
			policy_oracle_paths[robot_i] = None
	
	policy_oracles, _ = get_oracles(problem,
		policy_oracle_name = policy_oracle_name,
		policy_oracle_paths = policy_oracle_paths
		)
	policy_oracle = policy_oracles[robot]

	states = []
	encodings = []
	actions = []
	robot_action_dim = len(problem.action_idxs[robot]) 
	for _ in range(num_pi_eval):
		state = problem.initialize()
		encoding = problem.policy_encoding(state,robot)
		encoding = torch.tensor(encoding,dtype=torch.float32).squeeze().unsqueeze(0) # [batch_size x state_dim]
		mu, logvar = policy_oracle(encoding,training=True) # mu in [1 x robot_action_dim]
		if hasattr(policy_oracle, "sync_action_lims"):
			policy_oracle.sync_action_lims(problem)
		if hasattr(policy_oracle, "scale_action"):
			mu = policy_oracle.scale_action(mu)
		mu = mu.detach().numpy().reshape((robot_action_dim,1))
		sd = np.sqrt(np.exp(logvar.detach().numpy().reshape((robot_action_dim,1))))
		action = np.concatenate((mu,sd),axis=0)
		states.append(state)
		actions.append(action)
		encodings.append(encoding.detach().numpy().reshape((problem.policy_encoding_dim,1)))

	states = np.array(states).squeeze(axis=2)
	actions = np.array(actions).squeeze(axis=2)
	encodings = np.array(encodings).squeeze(axis=2)
	if plot_on:
		plotter.plot_policy_dataset(problem,[[states,actions]],["Eval"],robot)
		plotter.plot_policy_dataset(problem,[[encodings,actions]],["Eval"],robot)
		plotter.save_figs("{}/policy_eval_l{}_i{}.pdf".format(dirname,l,robot))


def self_play(problem,policy_oracle,value_oracle,l):
	solver = get_solver(
			"NeuralNetwork",
			policy_oracle=policy_oracle)

	instance = {
		"problem" : problem,
		"solver" : solver,
		"policy_oracle" : policy_oracle,
		"value_oracle" : value_oracle,
	}

	sim_results = [] 
	for _ in range(num_self_play_plots):
		state = problem.initialize()
		instance["initial_state"] = state
		sim_result = run_instance(0,Queue(),0,instance,verbose=False,tqdm_on=False)
		sim_results.append(sim_result)

	
	if 1:
		for sim_result in sim_results:
			plotter.plot_sim_result(sim_result)
			problem.render(states=sim_result["states"])

		if hasattr(problem, 'pretty_plot'):
			problem.pretty_plot(sim_results[0])	
		
		plotter.save_figs("{}/self_play_l{}.pdf".format(dirname,l))
		return sim_results


if __name__ == '__main__':

	problem = get_problem(problem_name)
	if hasattr(problem, 'use_minco_dynamics'):
		problem.use_minco_dynamics = use_minco_dynamics

	if resume_training:
		format_dir(clean_dirnames=["data"])
	else:
		format_dir(clean_dirnames=["data","models"])

	num_D_pi_samples = num_D_pi
	if mode == 2:
		num_D_pi_samples = num_D_pi*num_subsamples
	# if batch_size > np.min((num_D_pi_samples,num_D_v)) * (1-train_test_split):
	# 	batch_size = int(np.floor((np.min((num_D_pi_samples,num_D_v)) * train_test_split / 10)))
	# 	print('changing batch size to {}'.format(batch_size))

	start_l = 0
	if resume_training:
		latest_l = find_latest_complete_iteration(dirname, problem.num_robots)
		if latest_l >= 0:
			start_l = latest_l + 1
			print('Resuming training from iteration {} (found complete models up to l={})'.format(start_l, latest_l))
			if start_l >= L:
				print('Training already complete ({} iterations). Exiting.'.format(L))
				exit(0)
		else:
			print('No existing models found, starting from scratch.')

	# training
	for l in range(start_l, L):
		start_time = time.time()
		print('learning iteration: {}/{}...'.format(l,L))

		if l == 0:
			policy_oracle = [None for _ in range(problem.num_robots)]
			value_oracle = None
		else:
			value_oracle_path, policy_oracle_paths = get_oracle_fn(l-1,problem.num_robots)
			policy_oracle,value_oracle = get_oracles(problem,
				value_oracle_name = value_oracle_name,
				value_oracle_path = value_oracle_path,
				policy_oracle_name = policy_oracle_name,
				policy_oracle_paths = policy_oracle_paths
				)

			# 跑一次自对弈，生成example
			print('\t self play l/L: {}/{}...'.format(l,L))
			sim_results = self_play(problem,policy_oracle,value_oracle,l-1)


		if hasattr(problem, "turn_groups"):
			for robot_group in problem.turn_groups:
				robot_group = [int(robot) for robot in robot_group]
				for robot in robot_group:
					print('\t policy training iteration l/L, i/N: {}/{} {}/{}...'.format(\
						l,L,robot,problem.num_robots))
				datasets_by_robot = make_expert_demonstration_pi_group(
					problem, robot_group, policy_oracle, value_oracle
				)
				for robot in robot_group:
					train_dataset_pi, test_dataset_pi = datasets_by_robot[robot]
					train_model(
						problem,
						train_dataset_pi,
						test_dataset_pi,
						l,
						"policy",
						robot=robot,
						resume_from_existing=resume_training,
					)
					eval_policy(problem,l,robot)
		else:
			for robot in range(problem.num_robots):
				print('\t policy training iteration l/L, i/N: {}/{} {}/{}...'.format(\
					l,L,robot,problem.num_robots))
				train_dataset_pi, test_dataset_pi = make_expert_demonstration_pi(\
					problem,robot,policy_oracle,value_oracle)
				train_model(problem,train_dataset_pi,test_dataset_pi,l,"policy",robot=robot,resume_from_existing=resume_training)
				eval_policy(problem,l,robot)

		print('\t value training l/L: {}/{}'.format(l,L))
		train_dataset_v, test_dataset_v = make_expert_demonstration_v(problem, l)
		train_model(problem,train_dataset_v,test_dataset_v,l,"value",resume_from_existing=resume_training)
		eval_value(problem,l)
		print('complete learning iteration: {}/{} in {}s'.format(l,L,time.time()-start_time))
