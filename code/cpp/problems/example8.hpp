
#pragma once 

#include <string>
#include <vector>
#include <random>
#include <eigen3/Eigen/Dense>
#include "problem.hpp"
#include <algorithm>
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
		std::vector<int> m_evaders = {0, 1};
		std::vector<int> m_pursuers = {2, 3};
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
			Eigen::Matrix<float,-1,1> next_state(m_state_dim,1); 
			Eigen::Matrix<float,2,2> Fd = m_I + m_Fc * timestep;
			Eigen::Matrix<float,2,2> Bd = m_Bc * timestep; 

            // dynamics 
            for (int ii = 0; ii < m_num_robots; ii++){
				next_state(m_active_idxs[ii], 0) = state(m_active_idxs[ii], 0);
				if (!is_active(state, ii)) {
            	next_state.block(m_state_idxs[ii][0],0,m_state_idxs[ii].size(),1) =
            	    state.block(m_state_idxs[ii][0],0,m_state_idxs[ii].size(),1);
           		continue;
			}
                next_state.block(m_state_idxs[ii][0],0,m_state_idxs[ii].size(),1) = 
                    Fd * state.block(m_state_idxs[ii][0],0,m_state_idxs[ii].size(),1) + 
                    Bd * action.block(m_action_idxs[ii][0],0,m_action_idxs[ii].size(),1);
            }   

            next_state(8,0) = state(8,0) + timestep;
			auto capture_pairs = get_capture_pairs(next_state);
			for(auto &pe : capture_pairs) {
				next_state(m_active_idxs[pe.first], 0) = 0.0f;
				next_state(m_active_idxs[pe.second], 0) = 0.0f;
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
            
            if (next_state(m_time_idx,0) >= m_tf) {
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
            return (state.array() >= m_state_lims.col(0).array()).all() && (state.array() <= m_state_lims.col(1).array()).all();
        }

        bool is_terminal(Eigen::Matrix<float,-1,1> state) override 
        {
            // return !is_valid(state);
            //return ( (!is_valid(state))) || is_captured(state);
			 return ((!is_valid(state))  ||
            active_evader_count(state) == 0 ||
            active_pursuer_count(state) == 0);
        }

        bool is_captured(Eigen::Matrix<float,-1,1> state) {
        return !get_capture_pairs(state).empty();
        }
		
};

