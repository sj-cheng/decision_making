
#pragma once 

#include <string>
#include <vector>
#include <random>
#include <eigen3/Eigen/Dense>
#include "problem.hpp"
#include <algorithm>
#include <cmath>
#include <limits>
#include <utility>

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

		std::pair<Vec2f, Vec2f> get_robot_velocity_bounds(int robot) const {
			const float inv_dt = 1.0f / std::max(m_timestep, 1e-8f);
			auto action_bounds = get_robot_action_bounds(robot);
			return {action_bounds.first * inv_dt, action_bounds.second * inv_dt};
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

		Mat6f construct_quintic_boundary_inverse(float horizon) const {
			Mat6f mat;
			int row = 0;
			for (int endpoint = 0; endpoint < 2; ++endpoint) {
				const float t = (endpoint == 0) ? 0.0f : horizon;
				for (int rank = 0; rank < 3; ++rank) {
					mat.row(row++) = construct_beta(t, rank).transpose();
				}
			}
			return mat.inverse();
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

		std::pair<Vec2f, Vec2f> compute_feasible_command(
			const Eigen::Matrix<float,-1,1> &state,
			const Eigen::Matrix<float,-1,1> &action,
			int robot) const
		{
			const Vec2f p0 = state.block(m_state_idxs[robot][0],0,2,1);
			const Vec2f delta = action.block(m_action_idxs[robot][0],0,2,1);
			const Vec2f p_cmd_raw = p0 + delta;
			auto action_bounds = get_robot_action_bounds(robot);
			const Vec2f delta_clipped = delta.cwiseMax(action_bounds.first).cwiseMin(action_bounds.second);
			const Vec2f p_cmd_bounded = p0 + delta_clipped;
			Vec2f position_lower;
			Vec2f position_upper;
			for (int ii = 0; ii < 2; ++ii) {
				const int idx = m_state_idxs[robot][ii];
				position_lower(ii) = m_state_lims(idx,0);
				position_upper(ii) = m_state_lims(idx,1);
			}
			const Vec2f p_cmd_feasible = p_cmd_bounded.cwiseMax(position_lower).cwiseMin(position_upper);
			return {p_cmd_raw, p_cmd_feasible};
		}

		Vec2f compute_terminal_velocity_ref(
			const Vec2f &p0,
			const Vec2f &p_cmd_feasible,
			int robot,
			float horizon) const
		{
			auto velocity_bounds = get_robot_velocity_bounds(robot);
			const Vec2f desired_velocity = (p_cmd_feasible - p0) / std::max(horizon, 1e-8f);
			return desired_velocity.cwiseMax(velocity_bounds.first).cwiseMin(velocity_bounds.second);
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
			const float horizon = get_minco_horizon(safe_dt);
			const Mat6f boundary_inv = construct_quintic_boundary_inverse(horizon);

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

				const Vec2f p0 = state.block(m_state_idxs[ii][0],0,2,1);
				const Vec2f v0 = state.block(m_vel_idxs[ii][0],0,2,1);
				const Vec2f a0 = state.block(m_acc_idxs[ii][0],0,2,1);
				const auto command_pair = compute_feasible_command(state, action, ii);
				const Vec2f p_cmd_raw = command_pair.first;
				const Vec2f p_cmd_feasible = command_pair.second;
				(void)p_cmd_raw;
				const Vec2f vT_ref = compute_terminal_velocity_ref(p0, p_cmd_feasible, ii, horizon);
				const Vec2f aT_ref = Vec2f::Zero();
				const Mat6x2f coeffs = solve_quintic_coeffs(
					p0,
					v0,
					a0,
					p_cmd_feasible,
					vT_ref,
					aT_ref,
					boundary_inv);
				Vec2f position;
				Vec2f velocity;
				Vec2f acceleration;
				evaluate_quintic(coeffs, safe_dt, position, velocity, acceleration);
				next_state.block(m_state_idxs[ii][0],0,2,1) = position;
				next_state.block(m_vel_idxs[ii][0],0,2,1) = velocity;
				next_state.block(m_acc_idxs[ii][0],0,2,1) = acceleration;
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
