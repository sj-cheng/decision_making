

#pragma once 
#include <cmath>
#include <eigen3/Eigen/Dense>
#include "policy_network.hpp"
#include "network.hpp"

class GaussianPolicyNetwork : public PolicyNetwork {

	public: 
		float scale_action(float lower, float upper, float raw_action) const {
			float center = 0.5f * (upper + lower);
			float half_range = 0.5f * (upper - lower);
			return center + half_range * std::tanh(raw_action);
		}

		Eigen::Matrix<float,-1,1> eval(
			Problem * problem, Eigen::Matrix<float,-1,1> encoding, int robot, std::default_random_engine & gen) override {
			
			int robot_action_dim = problem->m_action_idxs[robot].size();

			Eigen::Matrix<float,-1,1> action(robot_action_dim); 

			auto distribution = m_phi.eval(encoding);
			auto mu = distribution.block(0,0,robot_action_dim,1);
			std::normal_distribution<float> noise_dist(0.0f,1.0f);

			for (int i = 0; i < robot_action_dim; i++) {
				int action_idx = problem->m_action_idxs[robot][i];
				float sampled = mu(i,0) + 0.1f * noise_dist(gen);
				float lower = problem->m_action_lims(action_idx,0);
				float upper = problem->m_action_lims(action_idx,1);
				action(i,0) = scale_action(lower, upper, sampled);
			}

			return action; 
		}

};
