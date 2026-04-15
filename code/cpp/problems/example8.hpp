#pragma once 

#include <string>
#include <vector>
#include <random>
#include <eigen3/Eigen/Dense>
#include "problem.hpp"
#include <algorithm>
#include <cmath>
#include <utility>
#include <limits>

class Example8 : public Problem { 
	
	public:
		Eigen::Matrix<float,2,2> m_Fc;
		Eigen::Matrix<float,2,2> m_Bc;
		Eigen::Matrix<float,2,2> m_I; 
		Eigen::Matrix<float,2,2> m_Q;
		Eigen::Matrix<float,2,2> m_R;  
		float m_r_max; 
		float m_r_min; 
		float m_state_control_weight;
		float m_dist;
		float m_tf;
		int m_state_dim_per_robot; 
		int m_action_dim_per_robot;
		bool m_use_minco_rollout = true;
		int m_time_idx = 8;
		std::vector<int> m_active_idxs = {9, 10, 11, 12};
		std::vector<std::vector<int>> m_vel_idxs = {{13, 14}, {15, 16}, {17, 18}, {19, 20}};
		std::vector<std::vector<int>> m_acc_idxs = {{21, 22}, {23, 24}, {25, 26}, {27, 28}};
		std::vector<int> m_evaders = {0, 1};
		std::vector<int> m_pursuers = {2, 3};

	private:
		using Vec2f = Eigen::Matrix<float,2,1>;
		using Vec6f = Eigen::Matrix<float,6,1>;
		using Mat6f = Eigen::Matrix<float,6,6>;
		using Mat6x2f = Eigen::Matrix<float,6,2>;

		struct Rollout_Data {
			bool active = false;
			Mat6x2f coeffs = Mat6x2f::Zero();
			Vec2f position = Vec2f::Zero();
			Vec2f velocity = Vec2f::Zero();
			Vec2f acceleration = Vec2f::Zero();
		};

		float get_minco_horizon(float timestep) const {
			return std::max(timestep, 1e-8f);
		}

		std::pair<Vec2f, Vec2f> get_robot_action_bounds(int robot) const {
			Vec2f lower;
			Vec2f upper;
			for (int ii = 0; ii < 2; ++ii) {
				const int idx = m_action_idxs[robot][ii];
				lower(ii) = m_action_lims(idx,0);
				upper(ii) = m_action_lims(idx,1);
			}
			return {lower, upper};
		}

		std::pair<Vec2f, Vec2f> get_robot_position_bounds(int robot) const {
			Vec2f lower;
			Vec2f upper;
			for (int ii = 0; ii < 2; ++ii) {
				const int idx = m_state_idxs[robot][ii];
				lower(ii) = m_state_lims(idx,0);
				upper(ii) = m_state_lims(idx,1);
			}
			return {lower, upper};
		}

		std::pair<Vec2f, Vec2f> get_robot_velocity_bounds(int robot) const {
			const float inv_dt = 1.0f / std::max(m_timestep, 1e-8f);
			auto action_bounds = get_robot_action_bounds(robot);
			return {action_bounds.first * inv_dt, action_bounds.second * inv_dt};
		}

		std::pair<Vec2f, Vec2f> get_robot_acceleration_bounds(int robot) const {
			Vec2f lower;
			Vec2f upper;
			for (int ii = 0; ii < 2; ++ii) {
				const int idx = m_acc_idxs[robot][ii];
				lower(ii) = m_state_lims(idx,0);
				upper(ii) = m_state_lims(idx,1);
			}
			return {lower, upper};
		}

		Vec2f get_symmetric_abs_bounds(const std::pair<Vec2f, Vec2f> &bounds) const {
			Vec2f abs_bounds;
			for (int ii = 0; ii < 2; ++ii) {
				abs_bounds(ii) = std::min(-bounds.first(ii), bounds.second(ii));
			}
			return abs_bounds;
		}

		float project_scalar_to_bounds(float nominal, float lower, float upper, float eps = 1e-6f) const {
			const bool lower_finite = std::isfinite(lower);
			const bool upper_finite = std::isfinite(upper);
			if (lower_finite && upper_finite) {
				if (lower <= upper + eps) {
					return std::max(lower, std::min(nominal, upper));
				}
				return (std::fabs(nominal - lower) <= std::fabs(nominal - upper)) ? lower : upper;
			}
			if (lower_finite) {
				return lower;
			}
			if (upper_finite) {
				return upper;
			}
			return nominal;
		}
		Vec6f construct_beta(float t, int rank) const {
			Vec6f beta_t = Vec6f::Zero();
			Vec6f beta_coeff = Vec6f::Zero();
			beta_t(rank) = 1.0f;
			for (int i = rank + 1; i < beta_t.size(); ++i) {
				beta_t(i) = beta_t(i - 1) * t;
			}
			for (int i = rank; i < beta_coeff.size(); ++i) {
				float coeff = 1.0f;
				for (int j = 0; j < rank; ++j) {
					coeff *= static_cast<float>(i - j);
				}
				beta_coeff(i) = coeff;
			}
			return beta_t.cwiseProduct(beta_coeff);
		}

		Mat6f construct_M(float horizon) const {
			Mat6f M;
			int row = 0;
			for (int endpoint = 0; endpoint < 2; ++endpoint) {
				const float t = (endpoint == 0) ? 0.0f : horizon;
				for (int rank = 0; rank < 3; ++rank) {
					M.row(row++) = construct_beta(t, rank).transpose();
				}
			}
			return M;
		}

		Mat6f construct_quintic_boundary_inverse(float horizon) const {
			return construct_M(horizon).inverse();
		}

		Mat6f betabetaT_int(float t, int rank) const {
			const Vec6f beta = construct_beta(t, rank);
			const Mat6f bbT = beta * beta.transpose();
			Mat6f int_coeff = Mat6f::Zero();
			for (int i = 0; i < 6; ++i) {
				for (int j = 0; j < 6; ++j) {
					const int denom = i + j - 2*rank + 1;
					if (denom > 0) {
						int_coeff(i, j) = 1.0f / static_cast<float>(denom);
					}
				}
			}
			return t * bbT.cwiseProduct(int_coeff);
		}

		Mat6f get_minco_matrix_inv(float horizon) const {
			const Mat6f M = construct_M(horizon);
			const Mat6f M_inv = M.inverse();
			const Mat6f pJpC = betabetaT_int(horizon, 3);
			Mat6f pQptQ = Mat6f::Zero();
			pQptQ(5,5) = 1.0f;
			const Mat6f pCpQ = M_inv.transpose();
			const Mat6f pJptQ = pQptQ * pCpQ * pJpC;
			Mat6f new_M = M;
			new_M.row(5) = pJptQ.row(5);
			return new_M.inverse();
		}

		Mat6x2f solve_quintic_coeffs(
			const Vec2f &p0,
			const Vec2f &v0,
			const Vec2f &a0,
			const Vec2f &pT,
			const Vec2f &vT,
			const Vec2f &aT,
			const Mat6f &boundary_inv) const
		{
			Mat6x2f q;
			q.row(0) = p0.transpose();
			q.row(1) = v0.transpose();
			q.row(2) = a0.transpose();
			q.row(3) = pT.transpose();
			q.row(4) = vT.transpose();
			q.row(5) = aT.transpose();
			return boundary_inv * q;
		}

		Mat6x2f solve_minco_coeffs(
			const Vec2f &p0,
			const Vec2f &v0,
			const Vec2f &a0,
			const Vec2f &pT,
			const Vec2f &vT,
			const Mat6f &minco_inv) const
		{
			Mat6x2f q = Mat6x2f::Zero();
			q.row(0) = p0.transpose();
			q.row(1) = v0.transpose();
			q.row(2) = a0.transpose();
			q.row(3) = pT.transpose();
			q.row(4) = vT.transpose();
			return minco_inv * q;
		}

		std::pair<Vec2f, Vec2f> compute_feasible_command(
			const Eigen::Matrix<float,-1,1> &state,
			const Eigen::Matrix<float,-1,1> &action,
			int robot) const
		{
			const Vec2f p0 = state.block(m_state_idxs[robot][0],0,2,1);
			const Vec2f delta = action.block(m_action_idxs[robot][0],0,2,1);
			const Vec2f p_cmd_raw = p0 + delta;
			return {p_cmd_raw, p_cmd_raw};
		}

		Vec2f compute_terminal_velocity(
			const Vec2f &p0,
			const Vec2f &v0,
			const Vec2f &a0,
			const Vec2f &p_cmd,
			int robot,
			float horizon) const
		{
			const float T = std::max(horizon, 1e-8f);
			const Vec2f vT_nom =
				(15.0f * (p_cmd - p0) - 7.0f * T * v0 - (T * T) * a0) / (8.0f * T);
			const auto velocity_bounds = get_robot_velocity_bounds(robot);
			return vT_nom.cwiseMax(velocity_bounds.first).cwiseMin(velocity_bounds.second);
		}
		
		std::pair<Vec2f, Vec2f> solve_end_pos_equ(
			float horizon,
			float t,
			int rank,
			const Eigen::Matrix<float,3,2> &q0,
			const Vec2f &vT,
			const Vec2f &max_abs,
			const Mat6f &minco_inv) const
		{
			const Vec6f bound_coeff = minco_inv.transpose() * construct_beta(t, rank);
			Mat6x2f q = Mat6x2f::Zero();
			q.block<3,2>(0,0) = q0;
			q.row(4) = vT.transpose();
			const float posT_coeff = bound_coeff(3);
			const Vec2f const_center = -(q.transpose() * bound_coeff);
			const Vec2f const_val_max = const_center + max_abs;
			const Vec2f const_val_min = const_center - max_abs;
			Vec2f upper;
			Vec2f lower;
			const float eps = 1e-12f;
			if (posT_coeff > eps) {
				upper = const_val_max / posT_coeff;
				lower = const_val_min / posT_coeff;
			} else if (posT_coeff < -eps) {
				upper = const_val_min / posT_coeff;
				lower = const_val_max / posT_coeff;
			} else {
				upper = Vec2f::Constant(std::numeric_limits<float>::infinity());
				lower = Vec2f::Constant(-std::numeric_limits<float>::infinity());
			}
			return {upper, lower};
		}

		std::pair<Vec2f, Vec2f> solve_end_pos_bound(
			float horizon,
			const Eigen::Matrix<float,3,2> &q0,
			const Vec2f &vT,
			int rank,
			const Vec2f &max_abs,
			const Mat6f &minco_inv,
			int seg_count = 5) const
		{
			Vec2f upper = Vec2f::Constant(std::numeric_limits<float>::infinity());
			Vec2f lower = Vec2f::Constant(-std::numeric_limits<float>::infinity());
			const float den = horizon / (2.0f * static_cast<float>(seg_count));
			for (int i = 0; i < seg_count; ++i) {
				const float t = (2.0f * static_cast<float>(i) + 1.0f) * den;
				auto pos_bound = solve_end_pos_equ(horizon, t, rank, q0, vT, max_abs, minco_inv);
				upper = upper.cwiseMin(pos_bound.first);
				lower = lower.cwiseMax(pos_bound.second);
			}
			return {upper, lower};
		}

		void project_terminal_command(
			const Vec2f &p0,
			const Vec2f &v0,
			const Vec2f &a0,
			const Vec2f &p_cmd_nominal,
			const Vec2f &vT,
			int robot,
			float horizon,
			const Mat6f &minco_inv,
			Vec2f &pT_cmd) const
		{
			const auto velocity_bounds = get_robot_velocity_bounds(robot);
			const Vec2f vel_abs = get_symmetric_abs_bounds(velocity_bounds);
			Eigen::Matrix<float,3,2> q0;
			q0.row(0) = p0.transpose();
			q0.row(1) = v0.transpose();
			q0.row(2) = a0.transpose();
			auto vel_pos_bounds = solve_end_pos_bound(horizon, q0, vT, 1, vel_abs, minco_inv);
			pT_cmd = p_cmd_nominal;
			for (int ii = 0; ii < 2; ++ii) {
				pT_cmd(ii) = project_scalar_to_bounds(
					p_cmd_nominal(ii),
					vel_pos_bounds.second(ii),
					vel_pos_bounds.first(ii));
			}
		}

		void evaluate_quintic(
			const Mat6x2f &coeffs,
			float t,
			Vec2f &position,
			Vec2f &velocity,
			Vec2f &acceleration) const
		{
			position = coeffs.transpose() * construct_beta(t, 0);
			velocity = coeffs.transpose() * construct_beta(t, 1);
			acceleration = coeffs.transpose() * construct_beta(t, 2);
		}

		Rollout_Data rollout_robot_state(
			const Eigen::Matrix<float,-1,1> &state,
			const Eigen::Matrix<float,-1,1> &action,
			int robot,
			float dt) const
		{
			Rollout_Data rollout;
			rollout.active = true;
			const float safe_dt = std::max(dt, 1e-8f);
			const Vec2f p0 = state.block(m_state_idxs[robot][0],0,2,1);
			const Vec2f v0 = state.block(m_vel_idxs[robot][0],0,2,1);
			if (!m_use_minco_rollout) {
				const Vec2f delta = action.block(m_action_idxs[robot][0],0,2,1);
				const auto velocity_bounds = get_robot_velocity_bounds(robot);
				const auto acceleration_bounds = get_robot_acceleration_bounds(robot);
				rollout.velocity = (delta / safe_dt)
					.cwiseMax(velocity_bounds.first)
					.cwiseMin(velocity_bounds.second);
				rollout.position = p0 + safe_dt * rollout.velocity;
				rollout.acceleration = ((rollout.velocity - v0) / safe_dt)
					.cwiseMax(acceleration_bounds.first)
					.cwiseMin(acceleration_bounds.second);
				return rollout;
			}

			const auto command_pair = compute_feasible_command(state, action, robot);
			const Vec2f p_cmd_nominal = command_pair.second;
			const float horizon = get_minco_horizon(safe_dt);
			const Vec2f a0 = state.block(m_acc_idxs[robot][0],0,2,1);
			const Vec2f vT_cmd = compute_terminal_velocity(p0, v0, a0, p_cmd_nominal, robot, horizon);
			const Mat6f minco_inv = get_minco_matrix_inv(horizon);
			Vec2f pT_cmd;
			project_terminal_command(
				p0, v0, a0, p_cmd_nominal, vT_cmd, robot, horizon, minco_inv, pT_cmd);
			rollout.coeffs = solve_minco_coeffs(
				p0, v0, a0, pT_cmd, vT_cmd, minco_inv);
			evaluate_quintic(rollout.coeffs, safe_dt, rollout.position, rollout.velocity, rollout.acceleration);
			return rollout;
		}

	public:
		void set_params(Problem_Settings & problem_settings) override 
		{
            m_state_dim = problem_settings.state_dim;
            m_action_dim = problem_settings.action_dim;
            m_num_robots = problem_settings.num_robots;
            m_state_idxs = problem_settings.state_idxs;
            m_action_idxs = problem_settings.action_idxs;

            // problem_settings.state_lims.resize(m_state_dim,2);
            // problem_settings.action_lims.resize(m_action_dim,2);
            // problem_settings.init_lims.resize(m_state_dim,2);

			m_timestep = problem_settings.timestep;
			m_tf = problem_settings.tf;
			m_gamma = problem_settings.gamma;
			m_r_max = problem_settings.r_max;
			m_r_min = problem_settings.r_min;
			m_state_control_weight = problem_settings.state_control_weight;
			m_state_lims = problem_settings.state_lims; 
			m_action_lims = problem_settings.action_lims; 
			m_init_lims = problem_settings.init_lims;
			m_dist = problem_settings.desired_distance;
			m_use_minco_rollout = problem_settings.use_minco_rollout;

			std::uniform_real_distribution<double> dist(0,1.0f); 

			m_Fc.setZero();
			m_Bc.setIdentity();
			m_I.setIdentity();

			m_Q.setIdentity();
			m_R.setIdentity();
			m_R = m_R * m_state_control_weight;
		}
		bool is_active(const Eigen::Matrix<float,-1,1> &state, int robot) const {
			return state(m_active_idxs[robot], 0) > 0.5f;
		}

		int active_evader_count(const Eigen::Matrix<float,-1,1> &state) const {
			int cnt = 0;
			for (int e : m_evaders) {
				if (is_active(state, e)) cnt++;
			}
			return cnt;
		}

		int active_pursuer_count(const Eigen::Matrix<float,-1,1> &state) const {
			int cnt = 0;
			for (int p : m_pursuers) {
				if (is_active(state, p)) cnt++;
			}
			return cnt;
		}

		std::vector<std::pair<int,int>> get_capture_pairs(
			const Eigen::Matrix<float,-1,1> &state) const
		{
			struct Candidate {
				float d;
				int p;
				int e;
			};

			std::vector<Candidate> cand;
			for (int p : m_pursuers) {
				if (!is_active(state, p)) continue;
				for (int e : m_evaders) {
					if (!is_active(state, e)) continue;
					float d = (state.block(m_state_idxs[p][0],0,m_state_idxs[p].size(),1) -
							state.block(m_state_idxs[e][0],0,m_state_idxs[e].size(),1)).norm();
					if (d < m_dist) {
						cand.push_back({d, p, e});
					}
				}
			}

			std::sort(cand.begin(), cand.end(),
				[](const Candidate &a, const Candidate &b) {
					return a.d < b.d;
				}); //排序，距离近的优先

			std::vector<bool> used_p(m_num_robots, false);
			std::vector<bool> used_e(m_num_robots, false);
			std::vector<std::pair<int,int>> result;
// 贪心匹配，距离近的先匹配，且每个机器人只能匹配一次
			for (const auto &x : cand) {
				if (used_p[x.p] || used_e[x.e]) continue;
				used_p[x.p] = true;
				used_e[x.e] = true;
				result.push_back({x.p, x.e});
			}

			return result;
		}

		Eigen::Matrix<float,-1,1> step(
			Eigen::Matrix<float,-1,1> state,
			Eigen::Matrix<float,-1,1> action,
			float timestep) override
		{
			Eigen::Matrix<float,-1,1> next_state = state;
			const float safe_dt = std::max(timestep, 1e-8f);
            // dynamics 
			for (int ii = 0; ii < m_num_robots; ii++){
				next_state(m_active_idxs[ii], 0) = state(m_active_idxs[ii], 0);
				if (!is_active(state, ii)) {
					next_state.block(m_state_idxs[ii][0],0,m_state_idxs[ii].size(),1) =
						state.block(m_state_idxs[ii][0],0,m_state_idxs[ii].size(),1);
					next_state.block(m_vel_idxs[ii][0],0,m_vel_idxs[ii].size(),1).setZero();
					next_state.block(m_acc_idxs[ii][0],0,m_acc_idxs[ii].size(),1).setZero();
					continue;
					}

				const Rollout_Data rollout = rollout_robot_state(state, action, ii, safe_dt);
				next_state.block(m_state_idxs[ii][0],0,2,1) = rollout.position;
				next_state.block(m_vel_idxs[ii][0],0,2,1) = rollout.velocity;
				next_state.block(m_acc_idxs[ii][0],0,2,1) = rollout.acceleration;
            }   

            next_state(m_time_idx,0) = state(m_time_idx,0) + timestep;
			auto capture_pairs = get_capture_pairs(next_state);
			for(auto &pe : capture_pairs) {
				next_state(m_active_idxs[pe.first], 0) = 0.0f;
				next_state(m_active_idxs[pe.second], 0) = 0.0f;
				next_state.block(m_vel_idxs[pe.first][0],0,m_vel_idxs[pe.first].size(),1).setZero();
				next_state.block(m_vel_idxs[pe.second][0],0,m_vel_idxs[pe.second].size(),1).setZero();
				next_state.block(m_acc_idxs[pe.first][0],0,m_acc_idxs[pe.first].size(),1).setZero();
				next_state.block(m_acc_idxs[pe.second][0],0,m_acc_idxs[pe.second].size(),1).setZero();
			}
            return next_state;
		}


        Eigen::Matrix<float,-1,1> reward(
            Eigen::Matrix<float,-1,1> state,
            Eigen::Matrix<float,-1,1> action) override
        { 
            return normalized_reward(state,action);
        }


        Eigen::Matrix<float,-1,1> normalized_reward(
            Eigen::Matrix<float,-1,1> state,
            Eigen::Matrix<float,-1,1> action) override
        {
			auto next_state = step(state, action, m_timestep);
			Eigen::Matrix<float,-1,1> r(m_num_robots,1);
			r.setZero();
			float r1=0.0 , r2=0.0;
			int newly_captured = active_evader_count(state) - active_evader_count(next_state);
			if (newly_captured > 0) {
				float t = std::min(next_state(m_time_idx,0), m_tf);
				r1=0.5*newly_captured*(t / m_tf);		// 逃跑者，被抓住越晚越好
				r2=0.5*newly_captured*(1.0f - t / m_tf);	// 追捕者，抓住越早越好
				r(0,0)=r1; r(1,0)=r1; r(2,0)=r2; r(3,0)=r2;
			} 
            
            if ((state(m_time_idx,0) < m_tf) && (next_state(m_time_idx,0) >= m_tf)){
				int surviving_evaders = active_evader_count(next_state);
				for (int e : m_evaders) {
					r(e,0) += 0.5f * surviving_evaders;
				}
			}
        
            for (int j=0; j<m_num_robots; ++j){
                int s0 = m_state_idxs[j][0];
                int sd = (int)m_state_idxs[j].size();
                bool in_low  = (next_state.block(s0,0,sd,1).array() >= m_state_lims.block(s0,0,sd,1).array()).all();
                bool in_high = (next_state.block(s0,0,sd,1).array() <= m_state_lims.block(s0,1,sd,1).array()).all();
                if (!(in_low && in_high)) r(j,0) = -1.0f;
            }
		
			return r;
        }

        bool is_valid(Eigen::Matrix<float,-1,1> state) override
        {
            for (int j = 0; j < m_num_robots; ++j) {
                int s0 = m_state_idxs[j][0];
                int sd = static_cast<int>(m_state_idxs[j].size());
                bool in_low  = (state.block(s0,0,sd,1).array() >= m_state_lims.block(s0,0,sd,1).array()).all();
                bool in_high = (state.block(s0,0,sd,1).array() <= m_state_lims.block(s0,1,sd,1).array()).all();
                if (!(in_low && in_high)) {
                    return false;
                }
            }
            return true;
        }

        bool is_terminal(Eigen::Matrix<float,-1,1> state) override 
        {
            // return !is_valid(state);
            //return ( (!is_valid(state))) || is_captured(state);
			 return ((!is_valid(state))  || state(m_time_idx,0) >= m_tf ||
            active_evader_count(state) == 0 ||
            active_pursuer_count(state) == 0);
        }

        bool is_captured(Eigen::Matrix<float,-1,1> state) {
        return !get_capture_pairs(state).empty();
        }
		
};
