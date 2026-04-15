# standard 
import numpy as np 
import matplotlib.patches as patches
import matplotlib.pyplot as plt

from typing_extensions import override
# custom 
from problems.problem import Problem
from util import sample_vector, contains
import plotter 

# 2d pursuit-evasion with per-step relative goal commands and MINCO-style rollout
class Example8(Problem):

	def __init__(self): 
		super(Example8,self).__init__()

		self.t0 = 0
		self.tf = 40
		self.dt = 0.5
		self.gamma = 1.0
		self.num_robots = 4 

		self.r_max = 1
		self.r_min = 0
		self.name = "example8"
		self.position_idx = np.arange(2) 
		self.state_control_weight = 1e-5 
		# Each robot action is a per-step local command offset delta. The
		# low-level controller converts the resulting command point into a
		# smooth quintic trajectory and evaluates that trajectory at t = dt.
		self.action_semantics = "relative_position_delta"
		self.low_level_controller_name = "relative_goal_to_minco_rollout"
		self.use_minco_rollout = True
		self.default_render_substeps = 10
		self.detailed_render_substeps = 30
		self.default_diagnostic_substeps = 50
		self.detailed_diagnostic_substeps = 200
		self.detailed_visualization_on = False
		self.render_substeps = self.default_render_substeps
		self.diagnostic_substeps = self.default_diagnostic_substeps
		self.desired_distance = 1.0
		self.init_min_dist = 2.0
		self.evaders = [0, 1]
		self.pursuers = [2, 3]
		self.turn_groups = [np.array([0, 1]), np.array([2, 3])]
		self.time_idx = 8
		self.active_idxs = [9, 10, 11, 12]
		self.evader_speed_lim_range = (2.0, 2.0)
		self.pursuer_speed_lim_range = (2.0, 2.0)

		self.evader_acc_lim =  3.0
		self.pursuer_acc_lim =  3.0
		self.state_dim = 29
		self.action_dim = 8
		self.state_idxs = [
			np.array([0,1]),  # E1
			np.array([2,3]),  # E2
			np.array([4,5]),  # P1
			np.array([6,7]),  # P2
		]
		self.vel_idxs = [
			np.array([13,14]),  # E1 velocity
			np.array([15,16]),  # E2 velocity
			np.array([17,18]),  # P1 velocity
			np.array([19,20]),  # P2 velocity
		]
		self.acc_idxs = [
			np.array([21,22]),  # E1 acceleration
			np.array([23,24]),  # E2 acceleration
			np.array([25,26]),  # P1 acceleration
			np.array([27,28]),  # P2 acceleration
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
			(-self.evader_speed_lim_range[1], self.evader_speed_lim_range[1]),
			(-self.evader_speed_lim_range[1], self.evader_speed_lim_range[1]),
			(-self.evader_speed_lim_range[1], self.evader_speed_lim_range[1]),
			(-self.evader_speed_lim_range[1], self.evader_speed_lim_range[1]),
			(-self.pursuer_speed_lim_range[1], self.pursuer_speed_lim_range[1]),
			(-self.pursuer_speed_lim_range[1], self.pursuer_speed_lim_range[1]),
			(-self.pursuer_speed_lim_range[1], self.pursuer_speed_lim_range[1]),
			(-self.pursuer_speed_lim_range[1], self.pursuer_speed_lim_range[1]),
			(-self.evader_acc_lim, self.evader_acc_lim),
			(-self.evader_acc_lim, self.evader_acc_lim),
			(-self.evader_acc_lim, self.evader_acc_lim),
			(-self.evader_acc_lim, self.evader_acc_lim),
			(-self.pursuer_acc_lim, self.pursuer_acc_lim),
			(-self.pursuer_acc_lim, self.pursuer_acc_lim),
			(-self.pursuer_acc_lim, self.pursuer_acc_lim),
			(-self.pursuer_acc_lim, self.pursuer_acc_lim),
			))
		self.approx_dist = (self.state_lims[0,1] - self.state_lims[0,0])/10 

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

		self.init_lims = np.array((
			(-8,8), (-8,8),
			(-8,8), (-8,8),
			(-8,8), (-8,8),
			(-8,8), (-8,8),
			(0,0),
			(1,1), (1,1), (1,1), (1,1),
			(0,0), (0,0),
			(0,0), (0,0),
			(0,0), (0,0),
			(0,0), (0,0),
			(0,0), (0,0),
			(0,0), (0,0),
			(0,0), (0,0),
			(0,0), (0,0),
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

	def get_minco_horizon(self, dt):
		return max(float(dt), 1e-8)

	def get_robot_action_lims(self, robot):
		return self.action_lims[self.action_idxs[robot], :]

	def get_robot_velocity_lims(self, robot):
		return self.get_robot_control_lims(robot)
	
	def get_robot_acceleration_lims(self, robot):
		return self.state_lims[self.acc_idxs[robot], :]
	
	def get_symmetric_abs_lims(self, lims):
		return np.minimum(-lims[:, 0], lims[:, 1]).reshape((-1, 1))

	def solve_end_pos_equ(self, horizon, t, rank, q0, vT, max_abs, minco_inv):
		bound_coeff = (self.construct_beta(t, rank).reshape((1, 6)) @ minco_inv).reshape((6,))
		q = np.vstack((q0, np.zeros((3, 2), dtype=float)))
		q[4, :] = vT.reshape((2,))
		posT_coeff = float(bound_coeff[3])
		const_center = -(bound_coeff.reshape((1, 6)) @ q).reshape((2, 1))
		const_val_max = const_center + max_abs
		const_val_min = const_center - max_abs
		eps = 1e-12
		if posT_coeff > eps:
			upper = const_val_max / posT_coeff
			lower = const_val_min / posT_coeff
		elif posT_coeff < -eps:
			upper = const_val_min / posT_coeff
			lower = const_val_max / posT_coeff
		else:
			upper = np.full((2, 1), np.inf)
			lower = np.full((2, 1), -np.inf)
		return upper, lower

	def solve_end_pos_bound(self, horizon, q0, vT, rank, max_abs, minco_inv, seg_count=7):
		upper = np.full((2, 1), np.inf)
		lower = np.full((2, 1), -np.inf)
		den = horizon / (2.0 * float(seg_count))
		for i in range(seg_count):
			t = (2*i + 1) * den
			pos_upper, pos_lower = self.solve_end_pos_equ(
				horizon, t, rank, q0, vT, max_abs, minco_inv)
			upper = np.minimum(upper, pos_upper)
			lower = np.maximum(lower, pos_lower)
		if rank == 2:
			pos_upper, pos_lower = self.solve_end_pos_equ(
				horizon, horizon, rank, q0, vT, max_abs, minco_inv)
			upper = np.minimum(upper, pos_upper)
			lower = np.maximum(lower, pos_lower)
		return upper, lower

	def project_terminal_command(self, p0, v0, a0, p_cmd_nominal, robot, horizon, minco_inv=None):
		q0 = np.vstack((
			p0.reshape((1, 2)),
			v0.reshape((1, 2)),
			a0.reshape((1, 2)),
		))
		velocity_lims = self.get_robot_velocity_lims(robot)
		vel_lower = velocity_lims[:, 0].reshape((-1, 1))
		vel_upper = velocity_lims[:, 1].reshape((-1, 1))
		position_lims = self.state_lims[self.state_idxs[robot], :]
		pos_lower = position_lims[:, 0].reshape((-1, 1))
		pos_upper = position_lims[:, 1].reshape((-1, 1))
		vel_abs = self.get_symmetric_abs_lims(velocity_lims)
		acc_abs = self.get_symmetric_abs_lims(self.get_robot_acceleration_lims(robot))
		vT_nominal = self.compute_terminal_velocity_ref(p0, p_cmd_nominal, robot, horizon)
		vT_base = np.clip(v0 + horizon * a0, vel_lower, vel_upper)
		if minco_inv is None:
			minco_inv = self.get_minco_matrix_inv(horizon)
		for alpha in (1.0,0.9,0.8,0.7, 0.6, 0.5, 0.4,0.3, 0.2, 0.1, 0.0):
			vT_candidate = np.clip(
				vT_base + alpha * (vT_nominal - vT_base),
				vel_lower,
				vel_upper,
			)
			vel_upper_p, vel_lower_p = self.solve_end_pos_bound(
				horizon, q0, vT_candidate, 1, vel_abs, minco_inv)
			acc_upper_p, acc_lower_p = self.solve_end_pos_bound(
				horizon, q0, vT_candidate, 2, acc_abs, minco_inv)
			feasible_lower = np.maximum.reduce((vel_lower_p, acc_lower_p, pos_lower))
			feasible_upper = np.minimum.reduce((vel_upper_p, acc_upper_p, pos_upper))
			if np.all(feasible_lower <= feasible_upper + 1e-8):
				pT_cmd = np.clip(p_cmd_nominal, feasible_lower, feasible_upper)
				return pT_cmd, vT_candidate
		pT_cmd = np.clip(p0, pos_lower, pos_upper)
		vT_cmd = np.clip(v0, vel_lower, vel_upper)
		return pT_cmd, vT_cmd
	def set_visualization_detail(self, detailed_on=False):
		self.detailed_visualization_on = bool(detailed_on)
		if self.detailed_visualization_on:
			self.render_substeps = self.detailed_render_substeps
			self.diagnostic_substeps = self.detailed_diagnostic_substeps
		else:
			self.render_substeps = self.default_render_substeps
			self.diagnostic_substeps = self.default_diagnostic_substeps

	def construct_beta(self, t, rank):
		beta_t = np.zeros(6, dtype=float)
		beta_coeff = np.zeros(6, dtype=float)
		beta_t[rank] = 1.0
		for i in range(rank + 1, 6):
			beta_t[i] = beta_t[i - 1] * t
		for i in range(rank, 6):
			coeff = 1.0
			for j in range(rank):
				coeff *= (i - j)
			beta_coeff[i] = coeff
		return beta_t * beta_coeff

	def construct_M(self, horizon):
		rows = []
		for t in (0.0, horizon):
			for rank in range(3):
				rows.append(self.construct_beta(t, rank))
		return np.vstack(rows)

	def construct_quintic_boundary_inverse(self, horizon):
		return np.linalg.inv(self.construct_M(horizon))

	def betabetaT_int(self, t, rank):
		beta = self.construct_beta(t, rank)
		bbT = np.outer(beta, beta)
		int_coeff = np.zeros((6, 6), dtype=float)
		for i in range(6):
			for j in range(6):
				denom = i + j - 2*rank + 1
				if denom > 0:
					int_coeff[i, j] = 1.0 / float(denom)
		return float(t) * bbT * int_coeff

	def get_minco_matrix_inv(self, horizon):
		M = self.construct_M(horizon)
		M_inv = np.linalg.inv(M)
		pJpC = self.betabetaT_int(horizon, 3)
		pQptQ = np.zeros((6, 6), dtype=float)
		pQptQ[-1, -1] = 1.0
		pCpQ = M_inv.T
		pJptQ = pQptQ @ pCpQ @ pJpC
		new_M = np.array(M, copy=True)
		new_M[-1, :] = pJptQ[-1, :]
		return np.linalg.inv(new_M)

	def solve_quintic_coeffs(self, p0, v0, a0, pT, vT, aT, boundary_inv):
		q = np.vstack((
			p0.reshape((1, 2)),
			v0.reshape((1, 2)),
			a0.reshape((1, 2)),
			pT.reshape((1, 2)),
			vT.reshape((1, 2)),
			aT.reshape((1, 2)),
		))
		return boundary_inv @ q

	def solve_minco_coeffs(self, p0, v0, a0, pT, vT, minco_inv):
		q = np.vstack((
			p0.reshape((1, 2)),
			v0.reshape((1, 2)),
			a0.reshape((1, 2)),
			pT.reshape((1, 2)),
			vT.reshape((1, 2)),
			np.zeros((1, 2), dtype=float),
		))
		return minco_inv @ q

	def compute_feasible_command(self, state, action, robot):
		p0 = state[self.state_idxs[robot], :]
		delta = action[self.action_idxs[robot], :]
		p_cmd_raw = p0 + delta
		position_lims = self.state_lims[self.state_idxs[robot], :]
		p_cmd_feasible = np.clip(
			p_cmd_raw,
			position_lims[:, 0].reshape((-1, 1)),
			position_lims[:, 1].reshape((-1, 1)),
		)
		return p_cmd_raw, p_cmd_feasible

	def compute_terminal_velocity_ref(self, p0, p_cmd_feasible, robot, horizon):
		velocity_lims = self.get_robot_velocity_lims(robot)
		desired_velocity = (p_cmd_feasible - p0) / max(float(horizon), 1e-8)
		return np.clip(
			desired_velocity,
			velocity_lims[:, 0].reshape((-1, 1)),
			velocity_lims[:, 1].reshape((-1, 1)),
		)

	def evaluate_rollout(self, coeffs, t):
		pos = (coeffs.T @ self.construct_beta(t, 0)).reshape((2, 1))
		vel = (coeffs.T @ self.construct_beta(t, 1)).reshape((2, 1))
		acc = (coeffs.T @ self.construct_beta(t, 2)).reshape((2, 1))
		return pos, vel, acc

	def rollout_quintic(self, coeffs, t):
		return self.evaluate_rollout(coeffs, t)

	def sample_render_trajectory(self, states, robot, substeps=None):
		states = np.atleast_2d(np.asarray(states).squeeze())
		if substeps is None:
			substeps = self.render_substeps
		if states.shape[0] == 0:
			return {
				"times": np.zeros((0,)),
				"positions": np.zeros((0, 2)),
				"velocities": np.zeros((0, 2)),
				"accelerations": np.zeros((0, 2)),
			}

		times = [float(states[0, self.time_idx])]
		positions = [states[0, self.state_idxs[robot]]]
		velocities = [states[0, self.vel_idxs[robot]]]
		accelerations = [states[0, self.acc_idxs[robot]]]

		for step_idx in range(states.shape[0] - 1):
			s0 = states[step_idx].reshape((-1, 1))
			s1 = states[step_idx + 1].reshape((-1, 1))
			horizon = max(float(s1[self.time_idx, 0] - s0[self.time_idx, 0]), 1e-8)
			sample_times = np.linspace(0.0, horizon, int(substeps) + 1)[1:]
			if not self.use_minco_rollout:
				position0 = s0[self.state_idxs[robot], :]
				velocity = (s1[self.state_idxs[robot], :] - position0) / horizon
				acceleration = np.zeros((2, 1), dtype=float)
				for t in sample_times:
					position = position0 + t * velocity
					times.append(float(s0[self.time_idx, 0] + t))
					positions.append(position[:, 0])
					velocities.append(velocity[:, 0])
					accelerations.append(acceleration[:, 0])
				continue

			boundary_inv = self.construct_quintic_boundary_inverse(horizon)
			coeffs = self.solve_quintic_coeffs(
				s0[self.state_idxs[robot], :],
				s0[self.vel_idxs[robot], :],
				s0[self.acc_idxs[robot], :],
				s1[self.state_idxs[robot], :],
				s1[self.vel_idxs[robot], :],
				s1[self.acc_idxs[robot], :],
				boundary_inv,
			)
			for t in sample_times:
				position, velocity, acceleration = self.rollout_quintic(coeffs, t)
				times.append(float(s0[self.time_idx, 0] + t))
				positions.append(position[:, 0])
				velocities.append(velocity[:, 0])
				accelerations.append(acceleration[:, 0])

		return {
			"times": np.asarray(times),
			"positions": np.asarray(positions),
			"velocities": np.asarray(velocities),
			"accelerations": np.asarray(accelerations),
		}

	def sample_render_positions(self, states, robot):
		return self.sample_render_trajectory(states, robot)["positions"]

	def normalize_render_actions(self, actions):
		if actions is None:
			return None
		actions = np.asarray(actions)
		if actions.size == 0:
			return np.zeros((0, self.action_dim), dtype=float)
		if actions.ndim == 3 and actions.shape[-1] == 1:
			return actions.reshape((actions.shape[0], actions.shape[1]))
		actions = np.squeeze(actions)
		if actions.ndim == 1:
			return actions.reshape((1, -1))
		return actions

	def get_rollout_target_point(self, state, action, robot, dt=None):
		state = np.asarray(state).reshape((-1, 1))
		action = np.asarray(action).reshape((-1, 1))
		if not self.is_active(state, robot):
			return None
		if dt is None:
			dt = self.dt
		safe_dt = max(float(dt), 1e-8)
		if not self.use_minco_rollout:
			_, p_cmd_feasible = self.compute_feasible_command(state, action, robot)
			return p_cmd_feasible
		horizon = self.get_minco_horizon(safe_dt)
		p0 = state[self.state_idxs[robot], :]
		v0 = state[self.vel_idxs[robot], :]
		a0 = state[self.acc_idxs[robot], :]
		_, p_cmd_feasible = self.compute_feasible_command(state, action, robot)
		minco_inv = self.get_minco_matrix_inv(horizon)
		pT_cmd, _ = self.project_terminal_command(
			p0, v0, a0, p_cmd_feasible, robot, horizon, minco_inv)
		return pT_cmd

	def get_render_target_segments(self, states, actions=None):
		states = np.atleast_2d(np.asarray(states).squeeze())
		actions = self.normalize_render_actions(actions)
		segment_data = []
		if states.size == 0 or states.shape[0] == 0:
			for _ in range(self.num_robots):
				segment_data.append({
					"starts": np.zeros((0, 2), dtype=float),
					"targets": np.zeros((0, 2), dtype=float),
				})
			return segment_data

		if actions is None:
			segment_count = max(states.shape[0] - 1, 0)
			for robot in range(self.num_robots):
				starts = []
				targets = []
				for seg_idx in range(segment_count):
					base_state = states[seg_idx].reshape((-1, 1))
					if not self.is_active(base_state, robot):
						continue
					starts.append(base_state[self.state_idxs[robot], 0])
					targets.append(np.array(states[seg_idx + 1, self.state_idxs[robot]], copy=True))
				segment_data.append({
					"starts": np.asarray(starts, dtype=float).reshape((-1, 2)) if starts else np.zeros((0, 2), dtype=float),
					"targets": np.asarray(targets, dtype=float).reshape((-1, 2)) if targets else np.zeros((0, 2), dtype=float),
				})
			return segment_data

		segment_count = min(actions.shape[0], states.shape[0])
		for robot in range(self.num_robots):
			starts = []
			targets = []
			for seg_idx in range(segment_count):
				base_state = states[seg_idx].reshape((-1, 1))
				target_point = self.get_rollout_target_point(base_state, actions[seg_idx], robot, dt=self.dt)
				if target_point is None:
					continue
				starts.append(base_state[self.state_idxs[robot], 0])
				targets.append(target_point[:, 0])
			segment_data.append({
				"starts": np.asarray(starts, dtype=float).reshape((-1, 2)) if starts else np.zeros((0, 2), dtype=float),
				"targets": np.asarray(targets, dtype=float).reshape((-1, 2)) if targets else np.zeros((0, 2), dtype=float),
			})
		return segment_data

	def rollout_robot_state(self, state, action, robot, dt):
		if not self.use_minco_rollout:
			safe_dt = max(float(dt), 1e-8)
			p0 = state[self.state_idxs[robot], :]
			v0 = state[self.vel_idxs[robot], :]
			delta = action[self.action_idxs[robot], :]
			p_cmd_raw = p0 + delta
			velocity_lims = self.get_robot_velocity_lims(robot)
			acc_lims = self.get_robot_acceleration_lims(robot)
			velocity = np.clip(
				delta / safe_dt,
				velocity_lims[:, 0].reshape((-1, 1)),
				velocity_lims[:, 1].reshape((-1, 1)),
			)
			position = p0 + safe_dt * velocity
			acceleration = np.clip(
				(velocity - v0) / safe_dt,
				acc_lims[:, 0].reshape((-1, 1)),
				acc_lims[:, 1].reshape((-1, 1)),
			)
			return p_cmd_raw, p_cmd_raw, position, velocity, acceleration

		safe_dt = max(float(dt), 1e-8)
		horizon = self.get_minco_horizon(safe_dt)
		p0 = state[self.state_idxs[robot], :]
		v0 = state[self.vel_idxs[robot], :]
		a0 = state[self.acc_idxs[robot], :]
		p_cmd_raw, p_cmd_nominal = self.compute_feasible_command(state, action, robot)
		minco_inv = self.get_minco_matrix_inv(horizon)
		pT_cmd, vT_cmd = self.project_terminal_command(
			p0, v0, a0, p_cmd_nominal, robot, horizon, minco_inv)
		coeffs = self.solve_minco_coeffs(
			p0, v0, a0, pT_cmd, vT_cmd, minco_inv)
		position, velocity, acceleration = self.rollout_quintic(coeffs, safe_dt)
		return p_cmd_raw, pT_cmd, position, velocity, acceleration

	def evaluate_rollout_segment(self, state, action, robot, t_eval, dt_total):
		safe_dt = max(float(dt_total), 1e-8)
		t = float(np.clip(t_eval, 0.0, safe_dt))
		p0 = state[self.state_idxs[robot], :]
		v0 = state[self.vel_idxs[robot], :]

		if not self.is_active(state, robot):
			return np.array(p0, copy=True), np.zeros((2, 1)), np.zeros((2, 1))

		if not self.use_minco_rollout:
			delta = action[self.action_idxs[robot], :]
			velocity_lims = self.get_robot_velocity_lims(robot)
			acc_lims = self.get_robot_acceleration_lims(robot)
			velocity = np.clip(
				delta / safe_dt,
				velocity_lims[:, 0].reshape((-1, 1)),
				velocity_lims[:, 1].reshape((-1, 1)),
			)
			position = p0 + t * velocity
			acceleration = np.clip(
				(velocity - v0) / safe_dt,
				acc_lims[:, 0].reshape((-1, 1)),
				acc_lims[:, 1].reshape((-1, 1)),
			)
			return position, velocity, acceleration

		a0 = state[self.acc_idxs[robot], :]
		_, p_cmd_nominal = self.compute_feasible_command(state, action, robot)
		minco_inv = self.get_minco_matrix_inv(safe_dt)
		pT_cmd, vT_cmd = self.project_terminal_command(
			p0, v0, a0, p_cmd_nominal, robot, safe_dt, minco_inv)
		coeffs = self.solve_minco_coeffs(
			p0, v0, a0, pT_cmd, vT_cmd, minco_inv)
		return self.rollout_quintic(coeffs, t)

	def interpolate_state(self, state, action, t_eval, dt_total):
		safe_dt = max(float(dt_total), 1e-8)
		t = float(np.clip(t_eval, 0.0, safe_dt))
		state_t = np.array(state, copy=True)
		state_t[self.time_idx, 0] = state[self.time_idx, 0] + t
		for robot in range(self.num_robots):
			state_t[self.active_idxs[robot], 0] = state[self.active_idxs[robot], 0]
			if not self.is_active(state, robot):
				state_t[self.state_idxs[robot], :] = state[self.state_idxs[robot], :]
				state_t[self.vel_idxs[robot], :] = 0.0
				state_t[self.acc_idxs[robot], :] = 0.0
				continue
			position, velocity, acceleration = self.evaluate_rollout_segment(
				state, action, robot, t, safe_dt)
			state_t[self.state_idxs[robot], :] = position
			state_t[self.vel_idxs[robot], :] = velocity
			state_t[self.acc_idxs[robot], :] = acceleration
		return state_t

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
			for vel_idxs in self.vel_idxs:
				state[vel_idxs, 0] = 0.0
			for acc_idxs in self.acc_idxs:
				state[acc_idxs, 0] = 0.0
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

	
	def is_captured(self, s):
		return len(self.get_capture_pairs(s)) > 0


	def step(self,s,a,dt):
		s_tp1 = np.array(s,copy=True)
		for robot in range(self.num_robots):
			s_tp1[self.active_idxs[robot], 0] = s[self.active_idxs[robot], 0]
			if not self.is_active(s, robot):
				s_tp1[self.state_idxs[robot],:] = s[self.state_idxs[robot],:]
				s_tp1[self.vel_idxs[robot],:] = 0.0
				s_tp1[self.acc_idxs[robot],:] = 0.0
				continue

			_, _, position, velocity, acceleration = self.rollout_robot_state(s, a, robot, dt)
			s_tp1[self.state_idxs[robot],:] = position
			s_tp1[self.vel_idxs[robot],:] = velocity
			s_tp1[self.acc_idxs[robot],:] = acceleration
		s_tp1[self.time_idx, 0] = s[self.time_idx, 0] + dt
		for p, e in self.get_capture_pairs(s_tp1):
			s_tp1[self.active_idxs[p], 0] = 0.0
			s_tp1[self.active_idxs[e], 0] = 0.0
			s_tp1[self.vel_idxs[p], :] = 0.0
			s_tp1[self.vel_idxs[e], :] = 0.0
			s_tp1[self.acc_idxs[p], :] = 0.0
			s_tp1[self.acc_idxs[e], :] = 0.0
		return s_tp1 

	def render(self,states=None,actions=None,fig=None,ax=None):
		# states, np array in [nt x state_dim]
  
		states = np.atleast_2d(np.asarray(states).squeeze()) if states is not None else None
		
		if fig == None or ax == None:
			fig,ax = plotter.make_fig()

		if states is not None:

			colors = plotter.get_n_colors(self.num_robots)
			target_segments = self.get_render_target_segments(states, actions=actions)
			show_target_legend = False
			for robot in range(self.num_robots):
				robot_state_idxs = self.state_idxs[robot] 
				render_positions = self.sample_render_positions(states, robot)
				discrete_positions = states[:, robot_state_idxs]
				segment_starts = target_segments[robot]["starts"]
				segment_targets = target_segments[robot]["targets"]

				ax.plot(render_positions[:,0], render_positions[:,1], color=colors[robot])
				ax.scatter(discrete_positions[:,0], discrete_positions[:,1], color=colors[robot], s=16, alpha=0.35)
				if segment_targets.shape[0] > 0:
					show_target_legend = True
					ax.scatter(
						segment_targets[:,0],
						segment_targets[:,1],
						marker='o',
						s=20,
						facecolors='white',
						edgecolors=colors[robot],
						linewidths=1.0,
						alpha=0.9,
						zorder=4,
					)
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
			if show_target_legend:
				ax.plot(
					np.nan,
					np.nan,
					marker='o',
					markerfacecolor='white',
					markeredgecolor='k',
					linestyle='None',
					label='solver pT',
				)
				ax.plot(np.nan, np.nan, color='k', marker='o', linestyle='None', alpha=0.35, label='stored state')
			ax.legend(loc='best')

		lims = self.state_lims
		ax.set_xlim((lims[0,0],lims[0,1]))
		ax.set_ylim((lims[1,0],lims[1,1]))
		ax.set_aspect( (lims[1,1]-lims[1,0]) / (lims[0,1]-lims[0,0]) )

		return fig,ax 

	def plot_run_diagnostics(self, sim_result):
		states = np.atleast_2d(np.asarray(sim_result["states"]).squeeze())
		if states.shape[0] == 0:
			return

		colors = plotter.get_n_colors(self.num_robots)
		labels = ["Evader1", "Evader2", "Pursuer1", "Pursuer2"]

		for robot in range(self.num_robots):
			trajectory = self.sample_render_trajectory(states, robot, substeps=self.diagnostic_substeps)
			sampled_times = trajectory["times"]
			sampled_vel = trajectory["velocities"]
			sampled_acc = trajectory["accelerations"]
			vel_norm = np.linalg.norm(sampled_vel, axis=1)
			acc_norm = np.linalg.norm(sampled_acc, axis=1)
			state_times = states[:, self.time_idx]
			state_vel = states[:, self.vel_idxs[robot]]
			state_acc = states[:, self.acc_idxs[robot]]
			state_vel_norm = np.linalg.norm(state_vel, axis=1)
			state_acc_norm = np.linalg.norm(state_acc, axis=1)
			vel_lims = self.state_lims[self.vel_idxs[robot], :]
			acc_lims = self.state_lims[self.acc_idxs[robot], :]
			vel_norm_lim = np.sqrt(np.sum(np.square(vel_lims[:, 1])))
			acc_norm_lim = np.sqrt(np.sum(np.square(acc_lims[:, 1])))

			series = [
				("vx", sampled_vel[:, 0], state_vel[:, 0], (vel_lims[0, 0], vel_lims[0, 1])),
				("vy", sampled_vel[:, 1], state_vel[:, 1], (vel_lims[1, 0], vel_lims[1, 1])),
				("|v|", vel_norm, state_vel_norm, (0.0, vel_norm_lim)),
				("ax", sampled_acc[:, 0], state_acc[:, 0], (acc_lims[0, 0], acc_lims[0, 1])),
				("ay", sampled_acc[:, 1], state_acc[:, 1], (acc_lims[1, 0], acc_lims[1, 1])),
				("|a|", acc_norm, state_acc_norm, (0.0, acc_norm_lim)),
			]
			sample_label = "MINCO internal samples" if self.use_minco_rollout else "direct interpolation samples"

			for series_name, sampled_values, state_values, y_lims in series:
				fig, ax = plt.subplots(figsize=(10, 4))
				ax.plot(sampled_times, sampled_values, color=colors[robot], alpha=0.25, linewidth=1.0)
				ax.scatter(sampled_times, sampled_values, color=colors[robot], s=10, alpha=0.85, label=sample_label)
				ax.scatter(state_times, state_values, color="black", s=26, marker="x", linewidths=0.9, label="stored states")
				ax.set_title("{} {} vs Time".format(labels[robot], series_name))
				ax.set_xlabel("time [s]")
				ax.set_ylabel(series_name)
				ax.set_ylim(y_lims)
				ax.grid(True, alpha=0.25)
				ax.legend(loc="best")
				fig.tight_layout()

	def is_terminal(self,state):
		# return not self.is_valid(state)
		#return (not self.is_valid(state)) or self.is_captured(state) 
		return ((not self.is_valid(state)) or self.active_evader_count(state) == 0 or self.active_pursuer_count(state) == 0) or state[self.time_idx, 0] >= self.tf

	def is_valid(self,state):
		for robot in range(self.num_robots):
			if not contains(state[self.state_idxs[robot], :], self.state_lims[self.state_idxs[robot], :]):
				return False
		return True

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
