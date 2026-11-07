
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
		float m_evader_distance_reward_weight;
		float m_cross_team_distance_reward_weight;
		float m_evader_progress_reward_weight;
		float m_pursuer_progress_reward_weight;
		float m_tf;
		int m_state_dim_per_robot; 
		int m_action_dim_per_robot;
		int m_time_idx = 8;
		std::vector<int> m_active_idxs = {9, 10, 11, 12};
		std::vector<std::vector<int>> m_vel_idxs = {{13, 14}, {15, 16}, {17, 18}, {19, 20}};
		std::vector<std::vector<int>> m_acc_idxs = {{21, 22}, {23, 24}, {25, 26}, {27, 28}};
		std::vector<int> m_evaders = {0, 1};
		std::vector<int> m_pursuers = {2, 3};
		std::vector<Eigen::Matrix<float,2,2>> m_obstacles;
		bool m_rollout_mode = false;
		bool m_swept_collision = false;
		std::vector<std::vector<int>> m_memory_idxs = {
			{29, 30, 31, 32, 33, 34},
			{35, 36, 37, 38, 39, 40},
			{41, 42, 43, 44, 45, 46},
			{47, 48, 49, 50, 51, 52},
		};

	private:
		using Vec2f = Eigen::Matrix<float,2,1>;
		using Vec6f = Eigen::Matrix<float,6,1>;
		using Mat6f = Eigen::Matrix<float,6,6>;
		using Mat6x2f = Eigen::Matrix<float,6,2>;
		enum {
			kLidarNumBeams = 64,
			kLidarNumChannels = 4,
			kLidarSelfFeatureDim = 8,
			kPolicyMemoryFeatureDim = 6,
			kPolicyEncodingDim = kLidarSelfFeatureDim + kLidarNumBeams * kLidarNumChannels + kPolicyMemoryFeatureDim
		};
		struct MemoryObservation {
			bool visible = false;
			Vec2f relative_position = Vec2f::Zero();
			Vec2f relative_velocity = Vec2f::Zero();
		};
		std::vector<Vec2f> m_lidar_directions;

		static constexpr float lidar_pi() {
			return 3.14159265358979323846f;
		}

		bool same_team(int robot_a, int robot_b) const {
			const bool a_evader = std::find(m_evaders.begin(), m_evaders.end(), robot_a) != m_evaders.end();
			const bool b_evader = std::find(m_evaders.begin(), m_evaders.end(), robot_b) != m_evaders.end();
			const bool a_pursuer = std::find(m_pursuers.begin(), m_pursuers.end(), robot_a) != m_pursuers.end();
			const bool b_pursuer = std::find(m_pursuers.begin(), m_pursuers.end(), robot_b) != m_pursuers.end();
			return (a_evader && b_evader) || (a_pursuer && b_pursuer);
		}

		float normalize_state_value(float value, int state_idx) const {
			const float low = m_state_lims(state_idx, 0);
			const float high = m_state_lims(state_idx, 1);
			if (std::abs(high - low) < 1e-8f) {
				return 0.0f;
			}
			return 2.0f * (value - low) / (high - low) - 1.0f;
		}

		float lidar_max_range() const {
			const float x_extent = m_state_lims(0, 1) - m_state_lims(0, 0);
			const float y_extent = m_state_lims(1, 1) - m_state_lims(1, 0);
			return std::max(std::sqrt(x_extent * x_extent + y_extent * y_extent), 1e-8f);
		}

		float min_cross_team_dist(const Eigen::Matrix<float,-1,1> &state) const {
			float nearest = std::numeric_limits<float>::infinity();
			for (int e : m_evaders) {
				if (!is_active(state, e)) continue;
				const Vec2f epos = state.block(m_state_idxs[e][0], 0, 2, 1);
				for (int p : m_pursuers) {
					if (!is_active(state, p)) continue;
					const Vec2f ppos = state.block(m_state_idxs[p][0], 0, 2, 1);
					nearest = std::min(nearest, (epos - ppos).norm());
				}
			}
			return nearest;
		}

		Eigen::Matrix<float,-1,1> self_lidar_features(
			const Eigen::Matrix<float,-1,1> &state,
			int robot) const
		{
			Eigen::Matrix<float,-1,1> features(static_cast<int>(kLidarSelfFeatureDim), 1);
			features.setZero();
			features(0, 0) = is_active(state, robot) ? 1.0f : 0.0f;
			features(1, 0) = normalize_state_value(state(m_state_idxs[robot][0], 0), m_state_idxs[robot][0]);
			features(2, 0) = normalize_state_value(state(m_state_idxs[robot][1], 0), m_state_idxs[robot][1]);
			features(3, 0) = normalize_state_value(state(m_vel_idxs[robot][0], 0), m_vel_idxs[robot][0]);
			features(4, 0) = normalize_state_value(state(m_vel_idxs[robot][1], 0), m_vel_idxs[robot][1]);
			features(5, 0) = normalize_state_value(state(m_acc_idxs[robot][0], 0), m_acc_idxs[robot][0]);
			features(6, 0) = normalize_state_value(state(m_acc_idxs[robot][1], 0), m_acc_idxs[robot][1]);
			features(7, 0) = normalize_state_value(state(m_time_idx, 0), m_time_idx);
			return features;
		}

		float ray_point_distance(
			const Vec2f &origin,
			const Vec2f &direction,
			const Vec2f &point) const
		{
			if (m_swept_collision) {
				// The local protocol partitions the circle into disjoint bins.
				// Comparing float sin/cos angles independently leaves tiny gaps
				// at beam boundaries (and can duplicate one target in two bins).
				const Eigen::Vector2d relative = point.cast<double>() - origin.cast<double>();
				const double distance = relative.norm();
				if (distance <= 1e-8) return std::numeric_limits<float>::infinity();
				const double scale = kLidarNumBeams / (2.0 * 3.14159265358979323846);
				const int target = (int(std::nearbyint(std::atan2(relative(1), relative(0))*scale))%kLidarNumBeams+kLidarNumBeams)%kLidarNumBeams;
				const int beam = (int(std::nearbyint(std::atan2(double(direction(1)), double(direction(0)))*scale))%kLidarNumBeams+kLidarNumBeams)%kLidarNumBeams;
				return target == beam ? float(distance) : std::numeric_limits<float>::infinity();
			}
			const Vec2f rel = point - origin;
			const float distance = rel.norm();
			if (distance <= 1e-8f) {
				return std::numeric_limits<float>::infinity();
			}
			const float projection = rel.dot(direction);
			if (projection <= 0.0f) {
				return std::numeric_limits<float>::infinity();
			}
			const float cross = direction(0) * rel(1) - direction(1) * rel(0);
			const float angle = std::abs(std::atan2(cross, projection));
			const float half_beam_width = lidar_pi() / static_cast<float>(kLidarNumBeams);
			if (angle > half_beam_width + 1e-8f) {
				return std::numeric_limits<float>::infinity();
			}
			return distance;
		}

		float ray_box_distance(
			const Vec2f &origin,
			const Vec2f &direction,
			const Eigen::Matrix<float,2,2> &box) const
		{
			float t_min = -std::numeric_limits<float>::infinity();
			float t_max = std::numeric_limits<float>::infinity();
			for (int axis = 0; axis < 2; ++axis) {
				const float low = box(axis, 0);
				const float high = box(axis, 1);
				if (std::abs(direction(axis)) < 1e-8f) {
					if (origin(axis) < low || origin(axis) > high) {
						return std::numeric_limits<float>::infinity();
					}
					continue;
				}

				float t1 = (low - origin(axis)) / direction(axis);
				float t2 = (high - origin(axis)) / direction(axis);
				if (t1 > t2) {
					std::swap(t1, t2);
				}
				t_min = std::max(t_min, t1);
				t_max = std::min(t_max, t2);
				if (t_min > t_max) {
					return std::numeric_limits<float>::infinity();
				}
			}
			if (t_max < 0.0f) {
				return std::numeric_limits<float>::infinity();
			}
			return (t_min >= 0.0f) ? t_min : t_max;
		}

		Eigen::Matrix<float,-1,1> semantic_lidar(
			const Eigen::Matrix<float,-1,1> &state,
			int robot) const
		{
			Eigen::Matrix<float,-1,1> lidar(
				static_cast<int>(kLidarNumBeams * kLidarNumChannels), 1);
			lidar.setOnes();
			if (!is_active(state, robot)) {
				return lidar;
			}

			const float max_range = lidar_max_range();
			const Vec2f origin = state.block(m_state_idxs[robot][0], 0, 2, 1);
			Eigen::Matrix<float,2,2> boundary;
			boundary <<
				m_state_lims(0, 0), m_state_lims(0, 1),
				m_state_lims(1, 0), m_state_lims(1, 1);

			const float occlusion_eps = 1e-8f;
			for (int beam_idx = 0; beam_idx < kLidarNumBeams; ++beam_idx) {
				const Vec2f &direction = m_lidar_directions[beam_idx];

				float channel_distances[kLidarNumChannels] = {
					max_range, max_range, max_range, max_range
				};

				for (const auto &obs : m_obstacles) {
					const float d = ray_box_distance(origin, direction, obs);
					if (std::isfinite(d) && d <= max_range && d < channel_distances[2]) {
						channel_distances[2] = d;
					}
				}

				for (int other_robot = 0; other_robot < m_num_robots; ++other_robot) {
					if (other_robot == robot || !is_active(state, other_robot)) {
						continue;
					}
					const Vec2f other_pos = state.block(m_state_idxs[other_robot][0], 0, 2, 1);
					const float d = ray_point_distance(origin, direction, other_pos);
					if (!std::isfinite(d) || d > max_range ||
						d >= channel_distances[2] - occlusion_eps) {
						continue;
					}
					const int channel = same_team(robot, other_robot) ? 0 : 1;
					if (d < channel_distances[channel]) {
						channel_distances[channel] = d;
					}
				}

				const float boundary_d = ray_box_distance(origin, direction, boundary);
				if (std::isfinite(boundary_d) && boundary_d <= max_range &&
					boundary_d < channel_distances[3]) {
					channel_distances[3] = boundary_d;
				}

				for (int higher_priority = 0; higher_priority < kLidarNumChannels; ++higher_priority) {
					if (channel_distances[higher_priority] >= max_range) {
						continue;
					}
					for (int lower_priority = higher_priority + 1;
						lower_priority < kLidarNumChannels;
						++lower_priority) {
						if (std::abs(channel_distances[lower_priority] -
							channel_distances[higher_priority]) <= 1e-8f) {
							channel_distances[lower_priority] = max_range;
						}
					}
				}

				for (int channel = 0; channel < kLidarNumChannels; ++channel) {
					const float normalized = channel_distances[channel] / max_range;
					lidar(beam_idx * kLidarNumChannels + channel, 0) =
						std::max(0.0f, std::min(1.0f, normalized));
				}
			}

			return lidar;
		}

		MemoryObservation visible_cross_team_observation(
			const Eigen::Matrix<float,-1,1> &state, int robot) const
		{
			MemoryObservation result;
			if (!is_active(state, robot)) return result;
			const Vec2f origin = state.block(m_state_idxs[robot][0], 0, 2, 1);
			const float max_range = lidar_max_range();
			std::vector<float> nearest_obstacle(kLidarNumBeams, max_range);
			for (const auto &obstacle : m_obstacles) {
				for (int beam_idx = 0; beam_idx < kLidarNumBeams; ++beam_idx) {
					const float distance = ray_box_distance(
						origin, m_lidar_directions[beam_idx], obstacle);
					if (std::isfinite(distance) && distance < nearest_obstacle[beam_idx]) {
						nearest_obstacle[beam_idx] = distance;
					}
				}
			}

			float best_distance = std::numeric_limits<float>::infinity();
			for (int other_robot = 0; other_robot < m_num_robots; ++other_robot) {
				if (other_robot == robot || same_team(robot, other_robot) ||
					!is_active(state, other_robot)) continue;
				const Vec2f relative = state.block(m_state_idxs[other_robot][0], 0, 2, 1) - origin;
				const float distance = relative.norm();
				if (distance <= 1e-8f || distance > max_range) continue;
				bool visible = false;
				for (int beam_idx = 0; beam_idx < kLidarNumBeams; ++beam_idx) {
					const Vec2f &direction = m_lidar_directions[beam_idx];
					if (m_swept_collision) {
						const Vec2f point = state.block(m_state_idxs[other_robot][0], 0, 2, 1);
						if (std::isfinite(ray_point_distance(origin, direction, point)) && distance < nearest_obstacle[beam_idx] - 1e-8f) {visible = true; break;}
						continue;
					}
					const float projection = direction.dot(relative);
					if (projection <= 0.0f) continue;
					const float cross = direction(0) * relative(1) - direction(1) * relative(0);
					const float angle = std::abs(std::atan2(cross, projection));
					if (angle <= lidar_pi() / static_cast<float>(kLidarNumBeams) + 1e-8f &&
						distance < nearest_obstacle[beam_idx] - 1e-8f) {
						visible = true;
						break;
					}
				}
				if (visible && distance < best_distance) {
					best_distance = distance;
					result.visible = true;
					result.relative_position = relative;
					result.relative_velocity =
						state.block(m_vel_idxs[other_robot][0], 0, 2, 1) -
						state.block(m_vel_idxs[robot][0], 0, 2, 1);
				}
			}
			return result;
		}

		void update_observation_memory(
			const Eigen::Matrix<float,-1,1> &previous_state,
			Eigen::Matrix<float,-1,1> &next_state,
			float timestep) const
		{
			for (int robot = 0; robot < m_num_robots; ++robot) {
				const auto &idx = m_memory_idxs[robot];
				next_state(idx[0], 0) = previous_state(idx[0], 0);
				next_state(idx[1], 0) = previous_state(idx[1], 0);
				next_state(idx[2], 0) = previous_state(idx[2], 0);
				next_state(idx[3], 0) = previous_state(idx[3], 0);
				next_state(idx[4], 0) = std::min(
					m_tf, std::max(0.0f, previous_state(idx[4], 0) + std::max(timestep, 0.0f)));
				next_state(idx[5], 0) = 0.0f;
				const MemoryObservation observation = visible_cross_team_observation(next_state, robot);
				if (observation.visible) {
					next_state(idx[0], 0) = observation.relative_position(0);
					next_state(idx[1], 0) = observation.relative_position(1);
					Vec2f relative_velocity = observation.relative_velocity;
					if (timestep > 1e-8f && previous_state(idx[5], 0) > 0.5f) {
						relative_velocity = (observation.relative_position -
							previous_state.block(idx[0], 0, 2, 1)) / timestep;
					}
					relative_velocity = relative_velocity.cwiseMax(Vec2f::Constant(-2.4f)).cwiseMin(Vec2f::Constant(2.4f));
					next_state(idx[2], 0) = relative_velocity(0);
					next_state(idx[3], 0) = relative_velocity(1);
					next_state(idx[4], 0) = 0.0f;
					next_state(idx[5], 0) = 1.0f;
				}
			}
		}

		Eigen::Matrix<float,-1,1> policy_memory_features(
			const Eigen::Matrix<float,-1,1> &state, int robot) const
		{
			Eigen::Matrix<float,-1,1> features(static_cast<int>(kPolicyMemoryFeatureDim), 1);
			const auto &idx = m_memory_idxs[robot];
			features(0, 0) = state(idx[0], 0) / 20.0f;
			features(1, 0) = state(idx[1], 0) / 20.0f;
			features(2, 0) = state(idx[2], 0) / 2.4f;
			features(3, 0) = state(idx[3], 0) / 2.4f;
			features(4, 0) = std::max(0.0f, std::min(1.0f, state(idx[4], 0) / std::max(m_tf, 1e-8f)));
			features(5, 0) = std::max(0.0f, std::min(1.0f, state(idx[5], 0)));
			return features;
		}

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

		// Uniform random joint action whose displacement keeps every active
		// robot inside the arena and out of obstacles; otherwise random
		// rollouts end mostly by boundary exits instead of game outcomes.
		Eigen::Matrix<float,-1,1> sample_rollout_action(
			const Eigen::Matrix<float,-1,1> & state,
			std::default_random_engine & gen) override
		{
			Eigen::Matrix<float,-1,1> action = sample_action(gen);
			for (int robot = 0; robot < m_num_robots; ++robot) {
				if (!is_active(state, robot)) {
					continue;
				}
				const auto bounds = get_robot_action_bounds(robot);
				const int a0 = m_action_idxs[robot][0];
				const Vec2f pos = state.block(m_state_idxs[robot][0], 0, 2, 1);
				Eigen::Matrix<float,-1,1> probe = state;
				bool found = false;
				for (int attempt = 0; attempt < 20 && !found; ++attempt) {
					Vec2f a;
					for (int ii = 0; ii < 2; ++ii) {
						a(ii) = bounds.first(ii) + static_cast<float>(dist(gen)) * (bounds.second(ii) - bounds.first(ii));
					}
					probe.block(m_state_idxs[robot][0], 0, 2, 1) = pos + a;
					if (is_robot_state_valid(probe, robot)) {
						action.block(a0, 0, 2, 1) = a;
						found = true;
					}
				}
				if (!found) {
					action.block(a0, 0, 2, 1).setZero();
				}
			}
			return action;
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
			m_evader_distance_reward_weight = problem_settings.evader_distance_reward_weight;
			m_cross_team_distance_reward_weight = problem_settings.cross_team_distance_reward_weight;
			m_evader_progress_reward_weight = problem_settings.evader_progress_reward_weight;
			m_pursuer_progress_reward_weight = problem_settings.pursuer_progress_reward_weight;
			m_obstacles = problem_settings.obstacles;
			m_swept_collision = problem_settings.swept_collision;
			m_lidar_directions.resize(kLidarNumBeams);
			for (int beam_idx = 0; beam_idx < kLidarNumBeams; ++beam_idx) {
				const float theta = 2.0f * lidar_pi() * static_cast<float>(beam_idx) /
					static_cast<float>(kLidarNumBeams);
				m_lidar_directions[beam_idx] << std::cos(theta), std::sin(theta);
			}

			std::uniform_real_distribution<double> dist(0,1.0f); 

			m_Fc.setZero();
			m_Bc.setIdentity();
			m_I.setIdentity();

			m_Q.setIdentity();
			m_R.setIdentity();
			m_R = m_R * m_state_control_weight;
		}
		bool is_active(const Eigen::Matrix<float,-1,1> &state, int robot) const override {
			return state(m_active_idxs[robot], 0) > 0.5f;
		}

		void set_rollout_mode(bool enabled) override {
			m_rollout_mode = enabled;
		}

		bool check_obstacle_collision(const Eigen::Matrix<float,-1,1> &state, int robot) const {
			float x = state(m_state_idxs[robot][0], 0);
			float y = state(m_state_idxs[robot][1], 0);
			for (const auto &obs : m_obstacles) {
				if (x >= obs(0,0) && x <= obs(0,1) && y >= obs(1,0) && y <= obs(1,1)) {
					return true;
				}
			}
			return false;
		}

		float obstacle_boundary_distance(const Vec2f &pos, const Eigen::Matrix<float,2,2> &obs) const {
			const float x = pos(0);
			const float y = pos(1);
			const float dx = std::max({obs(0,0) - x, 0.0f, x - obs(0,1)});
			const float dy = std::max({obs(1,0) - y, 0.0f, y - obs(1,1)});
			return std::sqrt(dx * dx + dy * dy);
		}

		bool has_spawn_clearance(const Eigen::Matrix<float,-1,1> &state, int robot) const {
			const Vec2f pos = state.block(m_state_idxs[robot][0], 0, 2, 1);
			const float min_clearance = 1.0f * m_dist;
			for (const auto &obs : m_obstacles) {
				if (obstacle_boundary_distance(pos, obs) <= min_clearance) {
					return false;
				}
			}
			return true;
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

		float nearest_active_pursuer_distance(
			const Eigen::Matrix<float,-1,1> &state,
			int evader) const
		{
			if (!is_active(state, evader)) {
				return std::numeric_limits<float>::infinity();
			}
			float nearest = std::numeric_limits<float>::infinity();
			const Vec2f evader_pos = state.block(m_state_idxs[evader][0], 0, 2, 1);
			for (int p : m_pursuers) {
				if (!is_active(state, p)) {
					continue;
				}
				const Vec2f pursuer_pos = state.block(m_state_idxs[p][0], 0, 2, 1);
				nearest = std::min(nearest, (evader_pos - pursuer_pos).norm());
			}
			return nearest;
		}

		float nearest_active_evader_distance(
			const Eigen::Matrix<float,-1,1> &state,
			int pursuer) const
		{
			if (!is_active(state, pursuer)) {
				return std::numeric_limits<float>::infinity();
			}
			float nearest = std::numeric_limits<float>::infinity();
			const Vec2f pursuer_pos = state.block(m_state_idxs[pursuer][0], 0, 2, 1);
			for (int e : m_evaders) {
				if (!is_active(state, e)) {
					continue;
				}
				const Vec2f evader_pos = state.block(m_state_idxs[e][0], 0, 2, 1);
				nearest = std::min(nearest, (pursuer_pos - evader_pos).norm());
			}
			return nearest;
		}

		void add_distance_progress_reward(
			Eigen::Matrix<float,-1,1> &reward,
			const Eigen::Matrix<float,-1,1> &state,
			const Eigen::Matrix<float,-1,1> &next_state) const
		{
			const float scale = std::max(4.0f * m_dist, 1e-6f);
			for (int e : m_evaders) {
				const float d0 = nearest_active_pursuer_distance(state, e);
				const float d1 = nearest_active_pursuer_distance(next_state, e);
				if (!std::isfinite(d0) || !std::isfinite(d1) || !is_active(next_state, e)) {
					continue;
				}
				const float delta = std::max(-1.0f, std::min(1.0f, (d1 - d0) / scale));
				reward(e, 0) += m_evader_progress_reward_weight * delta;
			}
			for (int p : m_pursuers) {
				const float d0 = nearest_active_evader_distance(state, p);
				const float d1 = nearest_active_evader_distance(next_state, p);
				if (!std::isfinite(d0) || !std::isfinite(d1) || !is_active(next_state, p)) {
					continue;
				}
				const float delta = std::max(-1.0f, std::min(1.0f, (d0 - d1) / scale));
				reward(p, 0) += m_pursuer_progress_reward_weight * delta;
			}
		}

		void add_evader_distance_reward(
			Eigen::Matrix<float,-1,1> &reward,
			const Eigen::Matrix<float,-1,1> &state) const
		{
			const float max_reward_distance = std::max(lidar_max_range() - m_dist, 1e-8f);
			for (int e : m_evaders) {
				const float distance = nearest_active_pursuer_distance(state, e);
				if (!std::isfinite(distance)) {
					continue;
				}
				const float distance_fraction = std::max(
					0.0f,
					std::min(1.0f, (distance - m_dist) / max_reward_distance));
				reward(e, 0) += m_evader_distance_reward_weight * distance_fraction;
			}
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
				if (m_swept_collision && !is_robot_state_valid(state,p)) continue;
				for (int e : m_evaders) {
					if (!is_active(state, e)) continue;
					if (m_swept_collision && !is_robot_state_valid(state,e)) continue;
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

		bool is_robot_state_valid(const Eigen::Matrix<float,-1,1> &state, int robot) const {
			int s0 = m_state_idxs[robot][0];
			int sd = static_cast<int>(m_state_idxs[robot].size());
			bool in_low  = (state.block(s0,0,sd,1).array() >= m_state_lims.block(s0,0,sd,1).array()).all();
			bool in_high = (state.block(s0,0,sd,1).array() <= m_state_lims.block(s0,1,sd,1).array()).all();

			return in_low && in_high && !check_obstacle_collision(state, robot);
		}

		Eigen::Matrix<float,-1,1> step(
			Eigen::Matrix<float,-1,1> state,
			Eigen::Matrix<float,-1,1> action,
			float timestep) override
		{
			Eigen::Matrix<float,-1,1> next_state = state;
			Eigen::Matrix<float,2,2> Fd = m_I + m_Fc * timestep;
			Eigen::Matrix<float,2,2> Bd = m_Bc * timestep;

			for (int ii = 0; ii < m_num_robots; ii++){
				next_state(m_active_idxs[ii], 0) = state(m_active_idxs[ii], 0);
				if (!is_active(state, ii)) {
					next_state.block(m_state_idxs[ii][0],0,m_state_idxs[ii].size(),1) =
						state.block(m_state_idxs[ii][0],0,m_state_idxs[ii].size(),1);
					next_state.block(m_vel_idxs[ii][0],0,m_vel_idxs[ii].size(),1).setZero();
					next_state.block(m_acc_idxs[ii][0],0,m_acc_idxs[ii].size(),1).setZero();
					continue;
					}
				auto control = action.block(m_action_idxs[ii][0],0,2,1) / timestep;
				next_state.block(m_state_idxs[ii][0],0,m_state_idxs[ii].size(),1) =
					Fd * state.block(m_state_idxs[ii][0],0,m_state_idxs[ii].size(),1) +
					Bd * control;
				next_state.block(m_vel_idxs[ii][0],0,m_vel_idxs[ii].size(),1).setZero();
				next_state.block(m_acc_idxs[ii][0],0,m_acc_idxs[ii].size(),1).setZero();
			}
			if (m_swept_collision) {
				for (int ii = 0; ii < m_num_robots; ++ii) {
					if (!is_active(state, ii)) continue;
					Vec2f start = state.block(m_state_idxs[ii][0],0,2,1);
					Vec2f delta = next_state.block(m_state_idxs[ii][0],0,2,1) - start;
					float first = 2.f;
					for (const auto &box : m_obstacles) {
						float lo = 0.f, hi = 1.f;
						bool hit = true;
						for (int axis = 0; axis < 2; ++axis) {
							if (std::abs(delta(axis)) < 1e-12f) {
								if (start(axis) < box(axis,0) || start(axis) > box(axis,1)) hit = false;
							} else {
								float a = (box(axis,0)-start(axis))/delta(axis);
								float b = (box(axis,1)-start(axis))/delta(axis);
								lo = std::max(lo, std::min(a,b));
								hi = std::min(hi, std::max(a,b));
							}
						}
						if (hit && lo <= hi && lo < first) {
							first = lo;
							Vec2f point = start + delta * std::min(hi, lo + 1e-5f);
							for (int axis=0; axis<2; ++axis) point(axis) = std::max(box(axis,0), std::min(box(axis,1), point(axis)));
							next_state.block(m_state_idxs[ii][0],0,2,1) = point;
						}
					}
				}
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
			if (!m_rollout_mode) {
				update_observation_memory(state, next_state, timestep);
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
			return normalized_reward_from_next(state, action, next_state);
		}


		Eigen::Matrix<float,-1,1> normalized_reward_from_next(
			Eigen::Matrix<float,-1,1> state,
			Eigen::Matrix<float,-1,1> action,
			Eigen::Matrix<float,-1,1> next_state) override
		{
			Eigen::Matrix<float,-1,1> r(m_num_robots,1);
			r.setZero();
			float r1=0.0 , r2=0.0;
			int newly_captured = active_evader_count(state) - active_evader_count(next_state);
			if (newly_captured > 0) {
				float t = std::min(next_state(m_time_idx,0), m_tf);
				r1=-0.5f*newly_captured*(1.0f - t / m_tf);	// 逃跑者，被抓是惩罚，越晚越轻
				r2=0.5f*newly_captured*(1.0f - t / m_tf);	// 追捕者，抓住越早越好
				r(0,0)=r1; r(1,0)=r1; r(2,0)=r2; r(3,0)=r2;
			} 

			// Dense progress is assigned to each active robot separately.  The
			// terminal capture reward remains the dominant event signal.
			add_distance_progress_reward(r, state, next_state);

			// Optional compatibility term for the earlier global potential.
			const float d0 = min_cross_team_dist(state);
			const float d1 = min_cross_team_dist(next_state);
			if (m_cross_team_distance_reward_weight != 0.0f && std::isfinite(d0) && std::isfinite(d1)) {
				const float scale = std::max(4.0f * m_dist, 1e-6f);
				const float delta = std::max(-1.0f, std::min(1.0f, (d0 - d1) / scale));
				for (int p : m_pursuers) r(p,0) += m_cross_team_distance_reward_weight * delta;
				for (int e : m_evaders) r(e,0) -= m_cross_team_distance_reward_weight * delta;
			}
            
	            if ((state(m_time_idx,0) < m_tf) && (next_state(m_time_idx,0) >= m_tf)){
				int surviving_evaders = active_evader_count(next_state);
				for (int e : m_evaders) {
					r(e,0) += 0.5f * surviving_evaders;
				}
			}
			add_evader_distance_reward(r, next_state);
        
            for (int j=0; j<m_num_robots; ++j){
                if (!is_robot_state_valid(next_state, j)) {
                	if (std::find(m_evaders.begin(), m_evaders.end(), j) != m_evaders.end()) {
                		for (int e : m_evaders) {
                			r(e,0) = -1.0f;
                		}
                	} else {
                		for (int p : m_pursuers) {
                			r(p,0) = -1.0f;
                		}
                	}
                }
            }

			return r;
        }

        bool is_valid(Eigen::Matrix<float,-1,1> state) override
        {
            for (int j = 0; j < m_num_robots; ++j) {
                if (!is_robot_state_valid(state, j)) {
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

		Eigen::Matrix<float,-1,1> policy_encoding(
			Eigen::Matrix<float,-1,1> state,
			int robot) override
		{
			Eigen::Matrix<float,-1,1> encoding(static_cast<int>(kPolicyEncodingDim), 1);
			const auto self_features = self_lidar_features(state, robot);
			const auto lidar_features = semantic_lidar(state, robot);
			const auto memory_features = policy_memory_features(state, robot);
			encoding.block(0, 0, kLidarSelfFeatureDim, 1) = self_features;
			encoding.block(kLidarSelfFeatureDim, 0, kLidarNumBeams * kLidarNumChannels, 1) =
				lidar_features;
			encoding.block(kLidarSelfFeatureDim + kLidarNumBeams * kLidarNumChannels,
				0, kPolicyMemoryFeatureDim, 1) = memory_features;
			return encoding;
		}

		Eigen::Matrix<float,-1,1> value_encoding(Eigen::Matrix<float,-1,1> state) override
		{
			return state.head(29);
		}

		Eigen::Matrix<float,-1,1> initialize(std::default_random_engine & gen)
		{
			Eigen::Matrix<float,-1,1> state(m_state_dim,1);
			bool valid = false;
			while (!valid) {
				for (int ii = 0; ii < m_state_dim; ++ii) {
					float alpha = dist(gen);
					state(ii,0) = alpha * (m_init_lims(ii,1) - m_init_lims(ii,0)) + m_init_lims(ii,0);
				}
				valid = !is_terminal(state);
				if (!valid) {
					continue;
				}
				for (int robot = 0; robot < m_num_robots; ++robot) {
					if (!has_spawn_clearance(state, robot)) {
						valid = false;
						break;
					}
				}
			}
			update_observation_memory(state, state, 0.0f);
			return state;
		}
		
};
