# standard
import numpy as np
import matplotlib.patches as patches
import matplotlib.pyplot as plt
from scipy.linalg import solve_discrete_are, pinv

from typing_extensions import override
# custom
from problems.problem import Problem
from util import sample_vector, contains
import plotter

# namespace RSG_SIM
POLY_DIM = 2                           # 维度
POLY_CTRL_EFFORT = 3                   # Control Effort 次数，2->Accelerate, 3->Jerk
POLY_DEGREE = POLY_CTRL_EFFORT * 2 - 1 # 轨迹多项式次数


class PolyTraj:
	def __init__(self, coff=None):
		if coff is None:
			coff = np.zeros((POLY_DEGREE + 1, POLY_DIM), dtype=float)
		self.coff = np.array(coff, dtype=float)

	@staticmethod
	def construct_beta(t, rank):
		betaT = np.zeros(POLY_DEGREE + 1, dtype=float)
		beta_coff = np.zeros(POLY_DEGREE + 1, dtype=float)

		# beta:
		# [1 T T^2 T^3 ...] (rank=0)
		# [0 1 2*T 3*T^2 4*T^3 ...] (rank=1)
		# [0 0 2   2*3*T 3*4*T^2 ...] (rank=2)

		# 计算T有关的部分 [1 T T^2 T^3 ...]
		betaT[rank] = 1.0
		for i in range(rank + 1, POLY_DEGREE + 1):
			betaT[i] = betaT[i - 1] * t

		# 计算多项式系数 [1 1 1 1 1 1](rank=0) [0 1 2 3 4 ...](rank=1) [0 0 2 2*3 3*4 ...](rank=2)
		for i in range(rank, POLY_DEGREE + 1):
			coff = 1.0
			for j in range(rank):
				coff *= (i - j)
			beta_coff[i] = coff

		return betaT * beta_coff

	@staticmethod
	def construct_M(ts):
		M = np.zeros((POLY_DEGREE + 1, POLY_DEGREE + 1), dtype=float)
		row_idx = 0
		for i in range(len(ts)):
			for s in range(POLY_CTRL_EFFORT):
				M[row_idx, :] = PolyTraj.construct_beta(ts[i], s)
				row_idx += 1
		return M

	@staticmethod
	def init_by_qT(q0, qT, ts):
		M = PolyTraj.construct_M(ts)
		M_inv = np.linalg.inv(M)

		q0 = np.asarray(q0, dtype=float)
		qT = np.asarray(qT, dtype=float)

		if q0.shape == (3, 2) and qT.shape == (3, 2):
			q = np.vstack((q0, qT))
		elif q0.shape == (6, 2):
			q = q0
		else:
			raise ValueError("init_by_qT expects q0/qT as (3,2)+(3,2) or q0 as (6,2).")

		c = M_inv @ q
		return PolyTraj(c)

	def get_pos(self, t):
		beta = PolyTraj.construct_beta(t, 0)
		return beta @ self.coff

	def get_vel(self, t):
		beta = PolyTraj.construct_beta(t, 1)
		return beta @ self.coff

	def get_acc(self, t):
		beta = PolyTraj.construct_beta(t, 2)
		return beta @ self.coff

	def get_poly_coff(self):
		return self.coff


class MincoTrajFactory:
	def __init__(self):
		self.pJpC_ = np.zeros((POLY_DEGREE + 1, POLY_DEGREE + 1), dtype=float)

	def clearPJpC(self):
		self.pJpC_ = np.zeros((POLY_DEGREE + 1, POLY_DEGREE + 1), dtype=float)

	def get_horizon(self, ts):
		if isinstance(ts, (list, tuple, np.ndarray)) and len(ts) >= 2:
			return float(ts[1] - ts[0])
		return float(ts)

	def get_time_vec(self, ts):
		if isinstance(ts, (list, tuple, np.ndarray)) and len(ts) >= 2:
			return np.asarray(ts, dtype=float)
		horizon = float(ts)
		return np.array([0.0, horizon], dtype=float)

	def getMatrixMInv(self, ts, dof):
		ts = self.get_time_vec(ts)

		M = PolyTraj.construct_M(ts)
		M_inv = np.linalg.inv(M)

		# ~q是约束中的自由变量，如当不限制速度和加速度时 ~q=[0,0,0,0,vT,aT]
		# 对末端a（即[0,0,0,0,0,a]^T向量）求偏导:∂J/∂(~q) = ∂q^T/∂(~q) * ∂c^T/∂q * ∂J/∂c
		# pQptQ = ∂q^T/∂(~q) = diag([0,0,0, ... ,1])
		# pCpQ = ∂c^T/∂(q) = M^(-T)
		# pJpC = ∂J/∂c = ∫(ββ^T)dt * c + ...
		# pQptQ 是q对tilt q的偏导数

		pQptQ = np.zeros((POLY_DEGREE + 1, POLY_DEGREE + 1), dtype=float)
		if dof > 0:
			pQptQ[-dof:, -dof:] = np.eye(dof, dtype=float)

		pCpQ = M_inv.T
		pJptQ = pQptQ @ pCpQ @ self.pJpC_

		newM = np.array(M, copy=True)
		if dof > 0:
			newM[-dof:, :] = pJptQ[-dof:, :]

		newM_inv = np.linalg.inv(newM)
		return newM_inv

	def solveWithCostJ(self, q0, qT, ts, dof):
		q0 = np.asarray(q0, dtype=float)
		qT = np.asarray(qT, dtype=float)

		q = np.vstack((q0, qT))
		if q.shape[0] < 6:
			q = np.vstack((q, np.zeros((6 - q.shape[0], 2), dtype=float)))

		c = self.getMatrixMInv(ts, dof) @ q
		return PolyTraj(c)

	def solveEndPosEqu_cpp(self, ts, dof, t, rank, q0, velT, max_min_equ_vel):
		q0 = np.asarray(q0, dtype=float)
		velT = np.asarray(velT, dtype=float).reshape((2,))

		# 计算(Phi'(t)^T) * (M^-1)
		phi_rank = PolyTraj.construct_beta(t, rank)
		Minv = self.getMatrixMInv(ts, dof)
		bound_coff_A = (phi_rank.reshape((1, 6)) @ Minv).reshape((6,))

		# 构造系数向量[p0, v0, a0, 0, vT, 0]
		q = np.vstack((q0, np.zeros((3, 2), dtype=float)))

		# 末端速度受约束
		q[POLY_CTRL_EFFORT + 1, :] = velT

		# [c0*p0, c1*v0, c2*a0, c3*pT, c4*vT, c5*0]，取出pT对应的系数
		posT_coff_in_equ = float(bound_coff_A[POLY_CTRL_EFFORT])

		# 计算等式常数项(v_max - c0*p0+c1*v0+c2*a0+c4*vT)
		const_val_max_vec = np.full((POLY_DIM,), max_min_equ_vel, dtype=float)
		const_val_max = -(bound_coff_A @ q) + const_val_max_vec
		const_val_min = -(bound_coff_A @ q) - const_val_max_vec

		# 考虑变号
		if posT_coff_in_equ > 0:
			posUpper = const_val_max / posT_coff_in_equ   # 上界
			posLower = const_val_min / posT_coff_in_equ   # 下界
		else:
			posUpper = const_val_min / posT_coff_in_equ   # 上界
			posLower = const_val_max / posT_coff_in_equ   # 下界

		return posUpper.reshape((2, 1)), posLower.reshape((2, 1))

	def solvePosBoundKernel(self, p0, v0, a0, bound_coff, max_val, upper, lower):
		p0 = np.asarray(p0, dtype=float).reshape((2,))
		v0 = np.asarray(v0, dtype=float).reshape((2,))
		a0 = np.asarray(a0, dtype=float).reshape((2,))
		bound_coff = np.asarray(bound_coff, dtype=float).reshape((6,))
		upper = np.asarray(upper, dtype=float).reshape((2,))
		lower = np.asarray(lower, dtype=float).reshape((2,))

		# 当前版本仅适用于vT无约束的情况
		pT_coff = float(bound_coff[3])

		lft_const_coff = (
			p0 * bound_coff[0] +
			v0 * bound_coff[1] +
			a0 * bound_coff[2]
		)

		# !分母小于0，则不等式方向改变，需要交换上下界。
		# pT_u与pT_d相差正好是max_val的符号
		if pT_coff < 0:
			max_val = -max_val

		pT_u = ( max_val - lft_const_coff) / pT_coff
		pT_d = (-max_val - lft_const_coff) / pT_coff

		# 与当前界取交集
		upper[0] = min(upper[0], pT_u[0])
		upper[1] = min(upper[1], pT_u[1])
		lower[0] = max(lower[0], pT_d[0])
		lower[1] = max(lower[1], pT_d[1])

		return upper, lower

	def solveEndPosEqu(self, ts, dof, t, rank, q0, velT, max_min_equ_vel):
		if dof == 2:
			# dof=2: 根据 CUDA solvePosBoundKernel 逻辑
			q0 = np.asarray(q0, dtype=float)

			phi_rank = PolyTraj.construct_beta(t, rank)
			Minv = self.getMatrixMInv(ts, dof)
			bound_coff_A = (phi_rank.reshape((1, 6)) @ Minv).reshape((6,))

			p0 = q0[0, :]
			v0 = q0[1, :]
			a0 = q0[2, :]

			upper = np.array([1e3, 1e3], dtype=float)
			lower = np.array([-1e3, -1e3], dtype=float)

			upper, lower = self.solvePosBoundKernel(
				p0, v0, a0, bound_coff_A, max_min_equ_vel, upper, lower
			)
			return upper.reshape((2, 1)), lower.reshape((2, 1))

		# 其它 dof 保持原 .cpp 逻辑
		return self.solveEndPosEqu_cpp(ts, dof, t, rank, q0, velT, max_min_equ_vel)

	def solveEndPosBound(self, ts, q0, velT, rank, maxVel):
		assert rank == 1 or rank == 2

		ts = self.get_time_vec(ts)
		horizon = float(ts[1] - ts[0])

		# 区间划分的段数
		seg_count = 5
		den = horizon / (2.0 * seg_count)

		bound_u_results = np.zeros((seg_count, POLY_DIM), dtype=float)
		bound_d_results = np.zeros((seg_count, POLY_DIM), dtype=float)

		for i in range(seg_count):
			t = (2 * i + 1) * den

			# 当 dof=2 时，会自动走 solvePosBoundKernel 逻辑
			pos_u, pos_d = self.solveEndPosEqu(ts, 2, t, rank, q0, velT, maxVel)

			bound_u_results[i, :] = pos_u.reshape((2,))
			bound_d_results[i, :] = pos_d.reshape((2,))

		min_bound_u = np.min(bound_u_results, axis=0)
		max_bound_d = np.max(bound_d_results, axis=0)

		return min_bound_u.reshape((2, 1)), max_bound_d.reshape((2, 1))

	@staticmethod
	def betabetaT_int(t, rank):
		b = PolyTraj.construct_beta(t, rank)
		bbT = np.outer(b, b)
		int_coff = np.zeros((POLY_DEGREE + 1, POLY_DEGREE + 1), dtype=float)

		for i in range(POLY_DEGREE + 1):
			for j in range(POLY_DEGREE + 1):
				if i + j - 2 * rank + 1 > 0:
					int_coff[i, j] = 1.0 / (i + j - 2 * rank + 1)

		return t * (bbT * int_coff)

	@staticmethod
	def betabetaT(t, rank):
		b = PolyTraj.construct_beta(t, rank)
		return np.outer(b, b)

	def add_control_effort_cost(self, T):
		# pJpC = ∂J/∂c = ∫(ββ^T)dt * c
		# 构造末端时间的代价矩阵∫(ββ^T)dt，维度(M+1)*(M+1)
		bbT_int = self.betabetaT_int(T, POLY_CTRL_EFFORT)
		self.pJpC_ += bbT_int

	def add_acc_cost(self, T):
		# pJpC = ∂J/∂c = ∫(ββ^T)dt * c
		# 构造末端时间的代价矩阵∫(ββ^T)dt，维度(M+1)*(M+1)
		bbT_int = self.betabetaT_int(T, POLY_CTRL_EFFORT - 1)
		self.pJpC_ += bbT_int

	def solveEndPosBoundCombined(self, ts, q0, velT, maxVel, maxAcc, seg_count=5):
		
		#综合速度约束(rank=1)与加速度约束(rank=2)，计算终端位置pT的可行区间交集。

	
		vel_ub, vel_lb = self.solveEndPosBound(ts, q0, velT, rank=1, maxVel=maxVel)
		acc_ub, acc_lb = self.solveEndPosBound(ts, q0, velT, rank=2, maxVel=maxAcc)

		# 取交集：下界取更严格者（更大），上界取更严格者（更小）
		lb = np.maximum(vel_lb, acc_lb)
		ub = np.minimum(vel_ub, acc_ub)
		return lb, ub


class ClosedLoopMincoPlanner:
	def __init__(self, piece_dt, ratio):
		self.piece_dt = max(float(piece_dt), 1e-8)
		self.ratio = max(float(ratio), 1e-8)
		self.planning_t = self.piece_dt / self.ratio

		mat_m = self.construct_minco_m2(self.planning_t)
		mat_m_inv = np.linalg.inv(mat_m)
		mat_r = self.construct_mat_r(self.piece_dt)
		mat_s = np.diag([1, 1, 1, 1, 0, 0])
		mat_u = np.array([[0, 0, 0, 0, 1, 0]], dtype=float).T

		self.mat_F = mat_m_inv @ mat_s @ mat_r
		self.mat_G = (mat_m_inv @ mat_u).reshape(-1, 1)

		Q = self.construct_bbint(self.piece_dt, rank=POLY_CTRL_EFFORT)
		R = np.array([[1.0]], dtype=float)
		P = solve_discrete_are(self.mat_F, self.mat_G, Q, R)
		self.K = np.linalg.inv(R + self.mat_G.T @ P @ self.mat_G) @ (self.mat_G.T @ P @ self.mat_F)
		self.Kpp = pinv(self.mat_G) @ (np.identity(POLY_DEGREE + 1) - self.mat_F) + self.K

		self.mat_F_stab = self.mat_F - self.mat_G @ self.K
		self.mat_G_stab = self.mat_G @ self.Kpp @ np.array([[1, 0, 0, 0, 0, 0]], dtype=float).T

	@staticmethod
	def construct_bbint(pieceT, rank):
		return MincoTrajFactory.betabetaT_int(pieceT, rank)

	@staticmethod
	def construct_mat_r(pieceT):
		mat_r = np.zeros((POLY_DEGREE + 1, POLY_DEGREE + 1), dtype=float)
		for i in range(POLY_DEGREE + 1):
			mat_r[i, :] = PolyTraj.construct_beta(pieceT, i)
		return mat_r

	@staticmethod
	def construct_minco_m2(pieceT):
		mat_m = np.zeros((POLY_DEGREE + 1, POLY_DEGREE + 1), dtype=float)
		for i in range(POLY_DEGREE - 1):
			mat_m[i, :] = PolyTraj.construct_beta(0.0, i)

		mat_m[-2, :] = PolyTraj.construct_beta(pieceT, 0)
		mat_m[-1, :] = PolyTraj.construct_beta(pieceT, 1)

		mat_m_inv = np.linalg.inv(mat_m)
		mat_supp = np.array([[0, 0, 0, 0, 0, 1]], dtype=float) @ mat_m_inv @ ClosedLoopMincoPlanner.construct_bbint(pieceT, rank=POLY_CTRL_EFFORT)
		mat_m[-1, :] = mat_supp[-1, :]
		return mat_m

	def coeff_state_from_kinematics(self, position, velocity, acceleration):
		coeff = np.zeros((POLY_DEGREE + 1, POLY_DIM), dtype=float)
		coeff[0, :] = np.asarray(position, dtype=float).reshape((POLY_DIM,))
		coeff[1, :] = np.asarray(velocity, dtype=float).reshape((POLY_DIM,))
		coeff[2, :] = 0.5 * np.asarray(acceleration, dtype=float).reshape((POLY_DIM,))
		return coeff

	def iterate(self, target_pos, coeff_state):
		target_pos = np.asarray(target_pos, dtype=float).reshape((1, POLY_DIM))
		return self.mat_F_stab @ coeff_state + self.mat_G_stab @ target_pos

	def calc_bound(self, t, rank, val, coeff_state):
		const_coeff = PolyTraj.construct_beta(t, rank).reshape((1, POLY_DEGREE + 1)) @ self.mat_F_stab
		bound_coeff = PolyTraj.construct_beta(t, rank).reshape((1, POLY_DEGREE + 1)) @ self.mat_G_stab
		bound_coeff_scalar = float(bound_coeff[0, 0])
		if abs(bound_coeff_scalar) < 1e-8:
			return np.array([
				[[-np.inf, -np.inf]],
				[[np.inf, np.inf]],
			], dtype=float)
		if bound_coeff_scalar < 0:
			val = -val
		lower = (-val - const_coeff @ coeff_state) / bound_coeff_scalar
		upper = ( val - const_coeff @ coeff_state) / bound_coeff_scalar
		return np.array([lower, upper], dtype=float)

	def bound_all(self, coeff_state, nckpt, max_vel, max_acc):
		ts = [((i + 1) / (nckpt + 1) * self.planning_t) for i in range(nckpt)]
		vel_bounds = np.concatenate([self.calc_bound(t, 1, max_vel, coeff_state) for t in ts], axis=1)
		acc_bounds = np.concatenate([self.calc_bound(t, 2, max_acc, coeff_state) for t in ts], axis=1)
		bounds = np.concatenate((vel_bounds, acc_bounds), axis=1)
		lower = np.max(bounds[0, :, :], axis=0)
		upper = np.min(bounds[1, :, :], axis=0)
		return lower.reshape((POLY_DIM, 1)), upper.reshape((POLY_DIM, 1))

	def get_pos(self, coeff_state, t):
		return (PolyTraj.construct_beta(t, 0) @ coeff_state).reshape((POLY_DIM, 1))

	def get_vel(self, coeff_state, t):
		return (PolyTraj.construct_beta(t, 1) @ coeff_state).reshape((POLY_DIM, 1))

	def get_acc(self, coeff_state, t):
		return (PolyTraj.construct_beta(t, 2) @ coeff_state).reshape((POLY_DIM, 1))


# 2d pursuit-evasion with per-step relative goal commands and MINCO-style rollout
class Example8(Problem):

	# 初始化：配置问题参数、状态/动作维度、限幅、索引等
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
		self.low_level_controller_name = "relative_goal_to_closed_loop_minco"
		self.default_render_substeps = 10
		self.detailed_render_substeps = 30
		self.default_diagnostic_substeps = 50
		self.detailed_diagnostic_substeps = 200
		self.detailed_visualization_on = False
		self.render_substeps = self.default_render_substeps
		self.diagnostic_substeps = self.default_diagnostic_substeps
		self.desired_distance = 1.0
		self.init_min_dist = 2.0
		self.split_spawn_by_team = False
		self.evaders = [0, 1]
		self.pursuers = [2, 3]
		self.turn_groups = [np.array([0, 1]), np.array([2, 3])]
		self.time_idx = 8
		self.active_idxs = [9, 10, 11, 12]
		self.evader_speed_lim_range = (2.0, 2.0)
		self.pursuer_speed_lim_range = (2.0, 2.0)

		self.evader_acc_lim = self.evader_speed_lim_range[1] / self.dt
		self.pursuer_acc_lim = self.pursuer_speed_lim_range[1] / self.dt
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
		self.times = np.arange(self.t0,self.tf+self.dt,self.dt)
		self.lidar_num_beams = 64
		self.lidar_num_channels = 4
		self.lidar_self_feature_dim = 8
		self.policy_encoding_dim = self.lidar_self_feature_dim + self.lidar_num_beams * self.lidar_num_channels
		self.lidar_angles = 2.0 * np.pi * np.arange(self.lidar_num_beams, dtype=float) / self.lidar_num_beams
		self.lidar_directions = np.stack((np.cos(self.lidar_angles), np.sin(self.lidar_angles)), axis=1)
		self.lidar_half_beam_width = np.pi / self.lidar_num_beams
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
		self.approx_dist = (self.state_lims[0,1] - self.state_lims[0,0]) / 10

		self.obstacles = [
			# np.array([[-4.0, -2.0], [-3.5,-1.5]], dtype=float),  # 中间偏左
			# np.array([[2.0, 4.0], [1.5, 3.5]], dtype=float),    # 中间偏右
			# np.array([[-4.0, -2.0], [1.5, 3.5]], dtype=float),
			# np.array([[2.0, 4.0], [-3.5, -1.5]], dtype=float),
			np.array([[-4.0,-2.0], [-1.5,1.5]], dtype=float),  # 中间偏左
			np.array([[2.0,4.0], [-1.5,1.5]], dtype=float),    # 中间偏右
		]

		self.current_evader_speed_lim = 2.0
		self.current_pursuer_speed_lim = 2.0
		self.update_action_lims()
		self.update_init_lims()

		self.use_minco_dynamics = True
		self.closed_loop_piece_dt = 0.1
		self.closed_loop_ratio = 0.1
		self.closed_loop_checkpoints = 20
		self._closed_loop_planners = {}
		self._coeff_state_cache = {}
		self.Fc = np.array(((0,0), (0,0)))
		self.Bc = np.array(((1,0), (0,1)))

	# 参数配置：更新控制与动作的限幅（根据当前速度限制）
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
	# 参数配置：随机化并更新逃逸方与追捕方的速度限制
	def randomize_speed_limits(self):
		self.current_evader_speed_lim = np.random.uniform(*self.evader_speed_lim_range)
		self.current_pursuer_speed_lim = np.random.uniform(*self.pursuer_speed_lim_range)
		self.update_action_lims()

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

	def update_init_lims(self):
		evader_y_lims = (-9, 0) if self.split_spawn_by_team else (-9, 9)
		pursuer_y_lims = (0, 9) if self.split_spawn_by_team else (-9, 9)
		self.init_lims = np.array((
			(-9,9), evader_y_lims,
			(-9,9), evader_y_lims,
			(-9,9), pursuer_y_lims,
			(-9,9), pursuer_y_lims,
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
		), dtype=float)

	# 查询：判断指定机器人是否仍处于活跃状态
	def is_active(self, state, robot):
		return state[self.active_idxs[robot], 0] > 0.5

	def same_team(self, robot_a, robot_b):
		return (
			(robot_a in self.evaders and robot_b in self.evaders)
			or (robot_a in self.pursuers and robot_b in self.pursuers)
		)

	def normalize_state_value(self, value, state_idx):
		low = float(self.state_lims[state_idx, 0])
		high = float(self.state_lims[state_idx, 1])
		if abs(high - low) < 1e-8:
			return 0.0
		return 2.0 * (float(value) - low) / (high - low) - 1.0

	def lidar_max_range(self):
		x_extent = float(self.state_lims[0, 1] - self.state_lims[0, 0])
		y_extent = float(self.state_lims[1, 1] - self.state_lims[1, 0])
		return max(np.hypot(x_extent, y_extent), 1e-8)

	def self_lidar_features(self, state, robot):
		state = np.asarray(state, dtype=float).reshape((-1, 1))
		pos_idxs = self.state_idxs[robot]
		vel_idxs = self.vel_idxs[robot]
		acc_idxs = self.acc_idxs[robot]

		features = np.zeros((self.lidar_self_feature_dim, 1), dtype=float)
		features[0, 0] = 1.0 if self.is_active(state, robot) else 0.0
		features[1, 0] = self.normalize_state_value(state[pos_idxs[0], 0], pos_idxs[0])
		features[2, 0] = self.normalize_state_value(state[pos_idxs[1], 0], pos_idxs[1])
		features[3, 0] = self.normalize_state_value(state[vel_idxs[0], 0], vel_idxs[0])
		features[4, 0] = self.normalize_state_value(state[vel_idxs[1], 0], vel_idxs[1])
		features[5, 0] = self.normalize_state_value(state[acc_idxs[0], 0], acc_idxs[0])
		features[6, 0] = self.normalize_state_value(state[acc_idxs[1], 0], acc_idxs[1])
		features[7, 0] = self.normalize_state_value(state[self.time_idx, 0], self.time_idx)
		return features

	def ray_point_distance(self, origin, direction, point):
		rel = np.asarray(point, dtype=float).reshape((2,)) - origin
		distance = float(np.linalg.norm(rel))
		if distance <= 1e-8:
			return np.inf
		projection = float(np.dot(rel, direction))
		if projection <= 0.0:
			return np.inf
		cross = float(direction[0] * rel[1] - direction[1] * rel[0])
		angle = abs(np.arctan2(cross, projection))
		if angle > self.lidar_half_beam_width + 1e-8:
			return np.inf
		return distance

	def ray_box_distance(self, origin, direction, box):
		t_min = -np.inf
		t_max = np.inf
		for axis in range(2):
			low = float(box[axis, 0])
			high = float(box[axis, 1])
			if abs(direction[axis]) < 1e-8:
				if origin[axis] < low or origin[axis] > high:
					return np.inf
				continue

			t1 = (low - origin[axis]) / direction[axis]
			t2 = (high - origin[axis]) / direction[axis]
			if t1 > t2:
				t1, t2 = t2, t1
			t_min = max(t_min, t1)
			t_max = min(t_max, t2)
			if t_min > t_max:
				return np.inf

		if t_max < 0.0:
			return np.inf
		return t_min if t_min >= 0.0 else t_max

	def ray_box_distances(self, origin, directions, box):
		origin = np.asarray(origin, dtype=float).reshape((2,))
		directions = np.asarray(directions, dtype=float)
		distances = np.full((directions.shape[0],), np.inf, dtype=float)
		t_min = np.full((directions.shape[0],), -np.inf, dtype=float)
		t_max = np.full((directions.shape[0],), np.inf, dtype=float)
		valid = np.ones((directions.shape[0],), dtype=bool)

		for axis in range(2):
			low = float(box[axis, 0])
			high = float(box[axis, 1])
			dir_axis = directions[:, axis]
			parallel = np.abs(dir_axis) < 1e-8
			if origin[axis] < low or origin[axis] > high:
				valid[parallel] = False

			non_parallel = ~parallel
			t1 = np.empty_like(dir_axis)
			t2 = np.empty_like(dir_axis)
			t1[non_parallel] = (low - origin[axis]) / dir_axis[non_parallel]
			t2[non_parallel] = (high - origin[axis]) / dir_axis[non_parallel]
			axis_min = np.minimum(t1[non_parallel], t2[non_parallel])
			axis_max = np.maximum(t1[non_parallel], t2[non_parallel])
			t_min[non_parallel] = np.maximum(t_min[non_parallel], axis_min)
			t_max[non_parallel] = np.minimum(t_max[non_parallel], axis_max)
			valid[non_parallel] &= t_min[non_parallel] <= t_max[non_parallel]

		valid &= t_max >= 0.0
		hit_distances = np.where(t_min >= 0.0, t_min, t_max)
		distances[valid] = hit_distances[valid]
		return distances

	def semantic_lidar(self, state, robot):
		state = np.asarray(state, dtype=float).reshape((-1, 1))
		channel_distances = np.full(
			(self.lidar_num_beams, self.lidar_num_channels),
			self.lidar_max_range(),
			dtype=float,
		)
		if not self.is_active(state, robot):
			return np.ones((self.lidar_num_beams * self.lidar_num_channels, 1), dtype=float)

		max_range = float(channel_distances[0, 0])
		origin = state[self.state_idxs[robot], 0].astype(float)
		boundary = np.array([
			[self.state_lims[0, 0], self.state_lims[0, 1]],
			[self.state_lims[1, 0], self.state_lims[1, 1]],
		], dtype=float)

		for other_robot in range(self.num_robots):
			if other_robot == robot or not self.is_active(state, other_robot):
				continue
			rel = state[self.state_idxs[other_robot], 0].astype(float) - origin
			distance = float(np.linalg.norm(rel))
			if distance <= 1e-8 or distance > max_range:
				continue
			projection = self.lidar_directions @ rel
			cross = self.lidar_directions[:, 0] * rel[1] - self.lidar_directions[:, 1] * rel[0]
			angles = np.abs(np.arctan2(cross, projection))
			mask = (projection > 0.0) & (angles <= self.lidar_half_beam_width + 1e-8)
			if not np.any(mask):
				continue
			channel = 0 if self.same_team(robot, other_robot) else 1
			channel_distances[mask, channel] = np.minimum(channel_distances[mask, channel], distance)

		for obstacle in self.obstacles:
			distances = self.ray_box_distances(origin, self.lidar_directions, obstacle)
			mask = np.isfinite(distances) & (distances <= max_range)
			channel_distances[mask, 2] = np.minimum(channel_distances[mask, 2], distances[mask])

		distances = self.ray_box_distances(origin, self.lidar_directions, boundary)
		mask = np.isfinite(distances) & (distances <= max_range)
		channel_distances[mask, 3] = np.minimum(channel_distances[mask, 3], distances[mask])

		for higher_priority in range(self.lidar_num_channels):
			higher_distances = channel_distances[:, higher_priority]
			has_higher = higher_distances < max_range
			for lower_priority in range(higher_priority + 1, self.lidar_num_channels):
				tie_mask = has_higher & np.isclose(
					channel_distances[:, lower_priority],
					higher_distances,
					atol=1e-8,
					rtol=0.0,
				)
				channel_distances[tie_mask, lower_priority] = max_range

		return np.clip(channel_distances / max_range, 0.0, 1.0).reshape((-1, 1))

	# 查询：判断指定机器人是否与障碍物发生碰撞
	def check_obstacle_collision(self, state, robot):
		pos = state[self.state_idxs[robot], :]
		for obstacle in self.obstacles:
			if contains(pos, obstacle):
				return True
		return False

	def obstacle_boundary_distance(self, pos, obstacle):
		x = float(pos[0, 0])
		y = float(pos[1, 0])
		dx = max(obstacle[0, 0] - x, 0.0, x - obstacle[0, 1])
		dy = max(obstacle[1, 0] - y, 0.0, y - obstacle[1, 1])
		return np.hypot(dx, dy)

	def has_spawn_clearance(self, state, robot):
		pos = state[self.state_idxs[robot], :]
		min_clearance = 0.1 * self.desired_distance
		return all(
			self.obstacle_boundary_distance(pos, obstacle) > min_clearance
			for obstacle in self.obstacles
		)

	# 查询：获取指定机器人的当前速度上限
	def get_robot_speed_limit(self, robot):
		if robot in self.evaders:
			return self.current_evader_speed_lim
		return self.current_pursuer_speed_lim

	# 查询：获取指定机器人的控制限幅（速度限幅）
	def get_robot_control_lims(self, robot):
		return self.control_lims[self.action_idxs[robot], :]

	# 查询：在动作限幅内随机采样一个动作
	def sample_action(self, state=None):
		return sample_vector(self.action_lims)

	# 辅助：将给定速度裁剪到指定机器人的速度限幅内
	def apply_velocity_limits(self, velocity, robot):
		control_lims = self.get_robot_control_lims(robot)
		return np.clip(
			velocity,
			control_lims[:, 0].reshape((-1, 1)),
			control_lims[:, 1].reshape((-1, 1)),
		)

	# 辅助：将动作（相对位移）转换为受速度限幅约束的期望速度
	def action_to_velocity(self, action, robot, dt):
		delta_position = action[self.action_idxs[robot], :]
		dt = max(dt, 1e-8)
		desired_velocity = delta_position / dt
		return self.apply_velocity_limits(desired_velocity, robot)

	# 辅助：根据离散化动力学直接推进机器人位置
	def propagate_robot_state(self, state, velocity, dt, robot):
		Fd = np.eye(len(self.state_idxs[robot])) + dt * self.Fc
		Bd = dt * self.Bc
		return np.dot(Fd, state[self.state_idxs[robot], :]) + np.dot(Bd, velocity)

	# 辅助：获取MINCO轨迹规划的时间 horizon（避免过小）
	def get_minco_horizon(self, dt):
		return max(float(dt), 1e-8)

	def get_closed_loop_schedule(self, dt):
		remaining = max(float(dt), 1e-8)
		piece_dt = max(float(self.closed_loop_piece_dt), 1e-8)
		durations = []
		while remaining > 1e-9:
			duration = min(piece_dt, remaining)
			durations.append(duration)
			remaining -= duration
		return durations

	def get_closed_loop_planner(self, piece_dt):
		key = round(float(piece_dt), 8)
		if key not in self._closed_loop_planners:
			self._closed_loop_planners[key] = ClosedLoopMincoPlanner(piece_dt=piece_dt, ratio=self.closed_loop_ratio)
		return self._closed_loop_planners[key]

	def get_state_signature(self, state):
		return np.ascontiguousarray(np.asarray(state, dtype=np.float64)).tobytes()

	def clear_closed_loop_cache(self):
		self._coeff_state_cache = {}

	def get_robot_dynamic_limits(self, robot):
		vel_lims = self.get_robot_velocity_lims(robot)
		acc_lims = self.state_lims[self.acc_idxs[robot], :]
		max_vel = float(np.min(vel_lims[:, 1]))
		max_acc = float(np.min(acc_lims[:, 1]))
		return max_vel, max_acc

	def coeff_state_from_state(self, state, robot, planner=None):
		if planner is None:
			planner = self.get_closed_loop_planner(self.closed_loop_piece_dt)
		position = state[self.state_idxs[robot], :].reshape((POLY_DIM,))
		velocity = state[self.vel_idxs[robot], :].reshape((POLY_DIM,))
		acceleration = state[self.acc_idxs[robot], :].reshape((POLY_DIM,))
		return planner.coeff_state_from_kinematics(position, velocity, acceleration)

	def get_cached_coeff_state(self, state, robot, planner=None):
		signature = self.get_state_signature(state)
		robot_coeffs = self._coeff_state_cache.get(signature, {})
		if robot in robot_coeffs:
			return np.array(robot_coeffs[robot], copy=True)
		coeff_state = self.coeff_state_from_state(state, robot, planner=planner)
		self._coeff_state_cache.setdefault(signature, {})[robot] = np.array(coeff_state, copy=True)
		return coeff_state

	def cache_coeff_state_for_state(self, state, robot, coeff_state):
		signature = self.get_state_signature(state)
		self._coeff_state_cache.setdefault(signature, {})[robot] = np.array(coeff_state, copy=True)

	# 查询：获取指定机器人的动作限幅（位移限幅）
	def get_robot_action_lims(self, robot):
		return self.action_lims[self.action_idxs[robot], :]

	# 查询：获取指定机器人的速度限幅
	def get_robot_velocity_lims(self, robot):
		return self.get_robot_control_lims(robot)

	# 参数配置：设置可视化精细度（渲染/诊断采样步数）
	def set_visualization_detail(self, detailed_on=False):
		self.detailed_visualization_on = bool(detailed_on)
		if self.detailed_visualization_on:
			self.render_substeps = self.detailed_render_substeps
			self.diagnostic_substeps = self.detailed_diagnostic_substeps
		else:
			self.render_substeps = self.default_render_substeps
			self.diagnostic_substeps = self.default_diagnostic_substeps

	def compute_dynamic_position_bounds(self, state, robot, planner=None, coeff_state=None):
		if planner is None:
			planner = self.get_closed_loop_planner(self.closed_loop_piece_dt)
		if coeff_state is None:
			coeff_state = self.get_cached_coeff_state(state, robot, planner=planner)

		max_vel, max_acc = self.get_robot_dynamic_limits(robot)
		lower, upper = planner.bound_all(
			coeff_state=coeff_state,
			nckpt=self.closed_loop_checkpoints,
			max_vel=max_vel,
			max_acc=max_acc,
		)
		position_lims = self.state_lims[self.state_idxs[robot], :]
		lower = np.maximum(lower, position_lims[:, 0].reshape((-1, 1)))
		upper = np.minimum(upper, position_lims[:, 1].reshape((-1, 1)))

		p0 = state[self.state_idxs[robot], :]
		collapse_mask = lower > upper
		if np.any(collapse_mask):
			lower[collapse_mask] = p0[collapse_mask]
			upper[collapse_mask] = p0[collapse_mask]
		return lower, upper

	# 辅助：计算可行指令（experiment2 风格：先算未来窗口可行区间，再裁剪目标点）
	def compute_feasible_command(self, state, action, robot, planner=None, coeff_state=None):
		p0 = state[self.state_idxs[robot], :]
		delta = action[self.action_idxs[robot], :]
		p_cmd_raw = p0 + delta
		lower, upper = self.compute_dynamic_position_bounds(
			state,
			robot,
			planner=planner,
			coeff_state=coeff_state,
		)
		p_cmd_feasible = np.clip(p_cmd_raw, lower, upper)
		return p_cmd_raw, p_cmd_feasible

	# 辅助：根据可行指令计算末端参考速度（带限幅）
	def compute_terminal_velocity_ref(self, p0, p_cmd_feasible, robot, horizon):
		velocity_lims = self.get_robot_velocity_lims(robot)
		desired_velocity = (p_cmd_feasible - p0) / max(float(horizon), 1e-8)
		return np.clip(
			desired_velocity,
			velocity_lims[:, 0].reshape((-1, 1)),
			velocity_lims[:, 1].reshape((-1, 1)),
		)

	

	# 轨迹规划：构造给定时刻 t 与阶数 rank 的 beta 向量
	def construct_beta(self, t, rank):
		betaT = np.zeros(6, dtype=float)
		beta_coff = np.zeros(6, dtype=float)
		betaT[rank] = 1.0
		for i in range(rank + 1, 6):
			betaT[i] = betaT[i - 1] * t
		for i in range(rank, 6):
			coff = 1.0
			for j in range(rank):
				coff *= (i - j)
			beta_coff[i] = coff
		return betaT * beta_coff

	# 轨迹规划：由时间边界 ts 构造多项式约束矩阵 M
	def construct_M(self, ts):
		M = np.zeros((6, 6))
		row_idx = 0
		for i in range(len(ts)):
			for s in range(3):
				M[row_idx, :] = self.construct_beta(ts[i], s)
				row_idx += 1
		return M

	# 轨迹规划：根据初末端状态 q0/qT 与时间 ts 求解多项式系数 coff
	def init_by_qT(self, q0, qT, ts):
		M = self.construct_M(ts)
		M_inv = np.linalg.inv(M)
		q = np.vstack((q0, qT))
		c = M_inv @ q
		return c

	# 轨迹规划：由多项式系数 coff 计算时刻 t 的位置
	def get_pos(self, coff, t):
		beta = self.construct_beta(t, 0)
		return (beta @ coff).reshape((2, 1))

	# 轨迹规划：由多项式系数 coff 计算时刻 t 的速度
	def get_vel(self, coff, t):
		beta = self.construct_beta(t, 1)
		return (beta @ coff).reshape((2, 1))

	# 轨迹规划：由多项式系数 coff 计算时刻 t 的加速度
	def get_acc(self, coff, t):
		beta = self.construct_beta(t, 2)
		return (beta @ coff).reshape((2, 1))

	

	# 状态推演：基于多项式轨迹规划，计算机器人在 dt 后的新状态
	# 采用 dof=2（末端速度和加速度自由），加入最小 jerk + 加速度代价优化
	# 新增：循环更新机制——先计算综合速度/加速度约束下的终端位置可行区间，
	# 若上层目标越界则裁剪修正，再用修正后的目标重新生成轨迹，直至收敛。
	def rollout_robot_state(self, state, action, robot, dt):
		safe_dt = max(float(dt), 1e-8)
		target_world = state[self.state_idxs[robot], :] + action[self.action_idxs[robot], :]
		current_state = np.array(state, copy=True)
		p_cmd_raw = np.array(target_world, copy=True)
		p_cmd_feasible = np.array(target_world, copy=True)
		final_coeff_state = None
		segments = []

		for piece_dt in self.get_closed_loop_schedule(safe_dt):
			planner = self.get_closed_loop_planner(piece_dt)
			coeff_state = self.get_cached_coeff_state(current_state, robot, planner=planner)

			piece_action = np.zeros_like(action)
			piece_action[self.action_idxs[robot], :] = target_world - current_state[self.state_idxs[robot], :]
			p_cmd_raw, p_cmd_feasible = self.compute_feasible_command(
				current_state,
				piece_action,
				robot,
				planner=planner,
				coeff_state=coeff_state,
			)
			final_coeff_state = planner.iterate(p_cmd_feasible.reshape((POLY_DIM,)), coeff_state)
			position = planner.get_pos(final_coeff_state, piece_dt)
			velocity = planner.get_vel(final_coeff_state, piece_dt)
			acceleration = planner.get_acc(final_coeff_state, piece_dt)

			segments.append({
				"duration": piece_dt,
				"coeff_state": np.array(final_coeff_state, copy=True),
				"p_cmd_raw": np.array(p_cmd_raw, copy=True),
				"p_cmd_feasible": np.array(p_cmd_feasible, copy=True),
			})

			current_state[self.state_idxs[robot], :] = position
			current_state[self.vel_idxs[robot], :] = velocity
			current_state[self.acc_idxs[robot], :] = acceleration
			self.cache_coeff_state_for_state(current_state, robot, final_coeff_state)

		position = current_state[self.state_idxs[robot], :]
		velocity = current_state[self.vel_idxs[robot], :]
		acceleration = current_state[self.acc_idxs[robot], :]
		return p_cmd_raw, p_cmd_feasible, position, velocity, acceleration, final_coeff_state, segments

	# ------------------------------------------------------------------
	# Game logic
	# ------------------------------------------------------------------

	# 游戏逻辑：统计当前仍处于活跃的逃逸方机器人数量
	def active_evader_count(self, state):
		return sum(self.is_active(state, e) for e in self.evaders)
	# 游戏逻辑：统计当前仍处于活跃的追捕方机器人数量
	def active_pursuer_count(self, state):
		return sum(self.is_active(state, p) for p in self.pursuers)
	# 游戏逻辑：计算逃逸方与追捕方之间的最小跨队距离
	def min_cross_team_dist(self, state):
		min_dist = np.inf
		for e in self.evaders:
			for p in self.pursuers:
				d = np.linalg.norm(state[self.state_idxs[e], :] - state[self.state_idxs[p], :])
				min_dist = min(min_dist, d)
		return min_dist
	# 游戏逻辑：获取当前状态下所有满足捕获条件的（追捕方, 逃逸方）配对
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

	# 游戏逻辑：判断当前状态是否存在已发生的捕获
	def is_captured(self, s):
		return len(self.get_capture_pairs(s)) > 0

	# 游戏逻辑：计算给定状态-动作对的奖励值
	def reward(self,s,a):
		reward = self.normalized_reward(s,a)
		return reward

	@override
	# 游戏逻辑：随机生成一个合法的初始状态
	def initialize(self):
		valid = False
		self.clear_closed_loop_cache()
		self.update_init_lims()
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
			valid = (
				not self.is_terminal(state)
				and (self.min_cross_team_dist(state) > 2 * self.init_min_dist)
				and all(self.has_spawn_clearance(state, robot) for robot in range(self.num_robots))
			)
		for robot in range(self.num_robots):
			if self.is_active(state, robot):
				self.cache_coeff_state_for_state(state, robot, self.coeff_state_from_state(state, robot))
		return state

	# 游戏逻辑：计算归一化的单步奖励（捕获奖励 + 存活奖励 + 越界惩罚）
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
			if (not contains(s_next[self.state_idxs[robot],:],self.state_lims[self.state_idxs[robot],:])) or self.check_obstacle_collision(s_next, robot):
				if robot in self.evaders:
					reward[self.evaders, 0] = -1.0
				else:
					reward[self.pursuers, 0] = -1.0
		return reward

	# 游戏逻辑：执行一个仿真步，更新所有机器人状态并处理捕获判定
	def step(self,s,a,dt):
		s_tp1 = np.array(s,copy=True)
		next_coeff_states = {}
		for robot in range(self.num_robots):
			s_tp1[self.active_idxs[robot], 0] = s[self.active_idxs[robot], 0]
			if not self.is_active(s, robot):
				s_tp1[self.state_idxs[robot],:] = s[self.state_idxs[robot],:]
				s_tp1[self.vel_idxs[robot],:] = 0.0
				s_tp1[self.acc_idxs[robot],:] = 0.0
				next_coeff_states[robot] = self.coeff_state_from_state(s_tp1, robot)
				continue

			if self.use_minco_dynamics:
				_, _, position, velocity, acceleration, coeff_state, _ = self.rollout_robot_state(s, a, robot, dt)
				s_tp1[self.state_idxs[robot],:] = position
				s_tp1[self.vel_idxs[robot],:] = velocity
				s_tp1[self.acc_idxs[robot],:] = acceleration
				next_coeff_states[robot] = coeff_state
			else:
				velocity = self.action_to_velocity(a, robot, dt)
				s_tp1[self.state_idxs[robot],:] = self.propagate_robot_state(s, velocity, dt, robot)
				s_tp1[self.vel_idxs[robot],:] = 0.0
				s_tp1[self.acc_idxs[robot],:] = 0.0
				next_coeff_states[robot] = self.coeff_state_from_state(s_tp1, robot)

		s_tp1[self.time_idx, 0] = s[self.time_idx, 0] + dt
		for p, e in self.get_capture_pairs(s_tp1):
			s_tp1[self.active_idxs[p], 0] = 0.0
			s_tp1[self.active_idxs[e], 0] = 0.0
			s_tp1[self.vel_idxs[p], :] = 0.0
			s_tp1[self.vel_idxs[e], :] = 0.0
			s_tp1[self.acc_idxs[p], :] = 0.0
			s_tp1[self.acc_idxs[e], :] = 0.0
			next_coeff_states[p] = self.coeff_state_from_state(s_tp1, p)
			next_coeff_states[e] = self.coeff_state_from_state(s_tp1, e)

		for robot in range(self.num_robots):
			coeff_state = next_coeff_states.get(robot)
			if coeff_state is None:
				coeff_state = self.coeff_state_from_state(s_tp1, robot)
			self.cache_coeff_state_for_state(s_tp1, robot, coeff_state)
		return s_tp1

	# 游戏逻辑：判断当前状态是否为终止状态（越界/全捕获/超时）
	def is_terminal(self,state):
		# return not self.is_valid(state)
		#return (not self.is_valid(state)) or self.is_captured(state)
		return ((not self.is_valid(state)) or self.active_evader_count(state) == 0 or self.active_pursuer_count(state) == 0) or state[self.time_idx, 0] >= self.tf

	# 游戏逻辑：判断当前状态下所有机器人是否都在合法范围内
	def is_valid(self,state):
		for robot in range(self.num_robots):
			if (not contains(state[self.state_idxs[robot], :], self.state_lims[self.state_idxs[robot], :])) or self.check_obstacle_collision(state, robot):
				return False
		return True

	# 状态编码：生成策略网络输入编码
	def policy_encoding(self,state,robot):
		state = np.asarray(state, dtype=float).reshape((-1, 1))
		return np.vstack((
			self.self_lidar_features(state, robot),
			self.semantic_lidar(state, robot),
		))

	# 状态编码：生成价值网络输入编码
	def value_encoding(self,state):
		return np.asarray(state, dtype=float).reshape((-1, 1))

	# 数据分组：按非目标机器人状态将数据集分组（用于可视化）
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

	# 辅助：判断两个状态向量是否在近似距离阈值内
	def isApprox(self,s1,s2):
		return np.linalg.norm(s1-s2) < self.approx_dist

	# ------------------------------------------------------------------
	# Visualization
	# ------------------------------------------------------------------

	# 可视化：对单个机器人在状态序列间采样，优先按真实动作重放闭环 MINCO 轨迹
	def sample_render_trajectory(self, states, robot, actions=None, substeps=None):
		states = np.atleast_2d(np.asarray(states).squeeze())
		if actions is not None:
			actions = np.asarray(actions)
			if actions.size == 0:
				actions = None
			else:
				actions = np.atleast_2d(actions.squeeze())
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
			if actions is not None and step_idx < actions.shape[0] and self.use_minco_dynamics and self.is_active(s0, robot):
				action = actions[step_idx].reshape((-1, 1))
				_, _, _, _, _, _, segments = self.rollout_robot_state(s0, action, robot, horizon)
				elapsed = 0.0
				for segment in segments:
					segment_dt = float(segment["duration"])
					segment_substeps = max(1, int(np.ceil(substeps * segment_dt / horizon)))
					sample_times = np.linspace(0.0, segment_dt, segment_substeps + 1)[1:]
					planner = self.get_closed_loop_planner(segment_dt)
					coeff_state = segment["coeff_state"]
					for t in sample_times:
						position = planner.get_pos(coeff_state, t)
						velocity = planner.get_vel(coeff_state, t)
						acceleration = planner.get_acc(coeff_state, t)
						times.append(float(s0[self.time_idx, 0] + elapsed + t))
						positions.append(position[:, 0])
						velocities.append(velocity[:, 0])
						accelerations.append(acceleration[:, 0])
					elapsed += segment_dt
			else:
				q0 = np.vstack((
					s0[self.state_idxs[robot], :].T,
					s0[self.vel_idxs[robot], :].T,
					s0[self.acc_idxs[robot], :].T,
				))
				qT = np.vstack((
					s1[self.state_idxs[robot], :].T,
					s1[self.vel_idxs[robot], :].T,
					s1[self.acc_idxs[robot], :].T,
				))
				ts = np.array([0.0, horizon])
				c = self.init_by_qT(q0, qT, ts)
				sample_times = np.linspace(0.0, horizon, int(substeps) + 1)[1:]
				for t in sample_times:
					position = self.get_pos(c, t)
					velocity = self.get_vel(c, t)
					acceleration = self.get_acc(c, t)
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

	# 可视化：获取单个机器人在状态序列间的插值位置点（供渲染调用）
	def sample_render_positions(self, states, robot, actions=None):
		return self.sample_render_trajectory(states, robot, actions=actions)["positions"]

	def _draw_obstacles(self, ax):
		for obstacle in self.obstacles:
			rect = patches.Rectangle(
				(obstacle[0, 0], obstacle[1, 0]),
				(obstacle[0, 1] - obstacle[0, 0]),
				(obstacle[1, 1] - obstacle[1, 0]),
				facecolor='gray',
				alpha=0.7,
			)
			ax.add_patch(rect)

	# 可视化：渲染追逃场景（轨迹、起点、终点、捕获范围、图例）
	def render(self,states=None,actions=None,fig=None,ax=None):
		# states, np array in [nt x state_dim]

		states = np.atleast_2d(np.asarray(states).squeeze()) if states is not None else None
		if actions is not None:
			actions = np.asarray(actions)
			if actions.size == 0:
				actions = None
			else:
				actions = np.atleast_2d(actions.squeeze())

		if fig == None or ax == None:
			fig,ax = plotter.make_fig()

		self._draw_obstacles(ax)

		if states is not None:

			colors = plotter.get_n_colors(self.num_robots)
			for robot in range(self.num_robots):
				robot_state_idxs = self.state_idxs[robot]
				render_positions = self.sample_render_positions(states, robot, actions=actions)

				ax.plot(render_positions[:,0], render_positions[:,1], color=colors[robot])
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

	# 可视化：绘制运行诊断图（各机器人速度/加速度随时间变化，含限幅对比）
	def plot_run_diagnostics(self, sim_result):
		states = np.atleast_2d(np.asarray(sim_result["states"]).squeeze())
		if states.shape[0] == 0:
			return

		colors = plotter.get_n_colors(self.num_robots)
		labels = ["Evader1", "Evader2", "Pursuer1", "Pursuer2"]

		for robot in range(self.num_robots):
			trajectory = self.sample_render_trajectory(states, robot, actions=sim_result.get("actions"), substeps=self.diagnostic_substeps)
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

			for series_name, sampled_values, state_values, y_lims in series:
				fig, ax = plt.subplots(figsize=(10, 4))
				ax.plot(sampled_times, sampled_values, color=colors[robot], alpha=0.25, linewidth=1.0)
				ax.scatter(sampled_times, sampled_values, color=colors[robot], s=10, alpha=0.85, label="MINCO internal samples")
				ax.scatter(state_times, state_values, color="black", s=26, marker="x", linewidths=0.9, label="stored states")
				ax.set_title("{} {} vs Time".format(labels[robot], series_name))
				ax.set_xlabel("time [s]")
				ax.set_ylabel(series_name)
				ax.set_ylim(y_lims)
				ax.grid(True, alpha=0.25)
				ax.legend(loc="best")
				fig.tight_layout()

	# 可视化：绘制价值函数数据集（等高线或散点图）
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

	# 可视化：绘制策略函数数据集（箭头场/散点图）
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

	# 可视化：综合绘制价值函数等高线与策略箭头场（在初始状态周围采样）
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
