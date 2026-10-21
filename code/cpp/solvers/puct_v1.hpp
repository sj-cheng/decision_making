#pragma once 
#include "solver.hpp"
#include <limits>
#include <cmath>
#include <algorithm>
#include <cstddef>

// Node payloads live in a few contiguous buffers instead of one heap
// allocation per node. The buffers are reserved for the worst-case node
// count but only the used prefix is written. Capacity is retained between
// searches so repeated self-play decisions do not reallocate the whole tree.
class PUCT_V1 : public Solver {

	public: 

		void set_params(Solver_Settings & solver_settings, 
				std::vector<Policy_Network_Wrapper> & policy_network_wrappers,
				Value_Network_Wrapper & value_network_wrapper) override {
			std::random_device dev;
			std::default_random_engine gen(dev());  
			g_gen = gen;
			m_num_simulations = solver_settings.num_simulations;
			m_search_depth = solver_settings.search_depth;
			m_C_exp = solver_settings.C_exp;
			m_alpha_exp = solver_settings.alpha_exp;
			m_C_pw = solver_settings.C_pw;
			m_alpha_pw = solver_settings.alpha_pw;
			m_beta_policy = solver_settings.beta_policy;
			m_beta_value = solver_settings.beta_value;
			m_export_tree = solver_settings.export_tree;
			m_policy_network_wrappers = policy_network_wrappers;
			m_value_network_wrapper = value_network_wrapper;
		}

		int tree_turn(Problem * problem, int turn, int depth) {
			// 对 2v2 的 example8：只在 0 和 2 两个“代表 turn”之间轮换
			if (problem->m_num_robots == 4) {
				int root_turn = turn;
				if (root_turn == 1) root_turn = 0;
				if (root_turn == 3) root_turn = 2;

				if (root_turn == 0 || root_turn == 2) {
					if (depth % 2 == 0) return root_turn;
					return (root_turn == 0) ? 2 : 0;
				}
			}

			return (depth + turn) % problem->m_num_robots;
		}

		// Example8 is a simultaneous 2v2 decision at the Python boundary, but
		// the search tree alternates the two teams.  A tree edge therefore must
		// contain an action for the acting team only.  Sampling all four robots
		// on every edge made the Q value for one team depend on an unrelated
		// random action from the other team and destroyed the pursuit signal.
		bool robot_is_on_turn(Problem * problem, int robot_turn, int robot) const {
			if (problem->m_num_robots == 4) {
				return robot_turn < 2 ? robot < 2 : robot >= 2;
			}
			return robot == robot_turn;
		}

		void keep_turn_action(Problem * problem,
			Eigen::Matrix<float,-1,1> & action,
			const Eigen::Matrix<float,-1,1> & state,
			int robot_turn) const {
			for (int robot = 0; robot < problem->m_num_robots; ++robot) {
				const bool active = problem->is_active(state, robot);
				if (!robot_is_on_turn(problem, robot_turn, robot) || !active) {
					action.block(problem->m_action_idxs[robot][0], 0,
						problem->m_action_idxs[robot].size(), 1).setZero();
				}
			}
		}

		float team_value(int node, int robot_turn) const {
			const float* value = m_values.data() + std::ptrdiff_t(node) * m_value_dim;
			if (m_two_teams) {
				int first = (robot_turn < 2) ? 0 : 2;
				return 0.5f * (value[first] + value[first + 1]);
			}
			return value[robot_turn];
		}

		Solver_Result search(Problem * problem, Eigen::Matrix<float,-1,1> root_state, int turn){

			Solver_Result solver_result;
			solver_result.success = false;
			m_two_teams = (problem->m_num_robots == 4);
			begin_tree(problem, root_state);

			if (problem->is_terminal(root_state)){
				return solver_result; 
			}
			for (int ii = 0; ii < m_num_simulations; ii++){

				int curr = 0;
				int max_depth = m_search_depth;
				for (int d = 0; d < m_search_depth; d++){
					int robot_turn = tree_turn(problem, turn, d);
					int child;
					bool expanded_leaf = false;
					if (is_expanded(curr)){
						child = best_child(problem, curr, robot_turn);
						if (child < 0){
        					return solver_result;
    					}
					} else {
						child = expand_node(problem, curr, robot_turn);
						expanded_leaf = true;
					}
					m_path[d] = curr;
					store_reward(d, edge_reward_of(child));
					
					curr = child;
					// A newly expanded node is the leaf for this simulation.  Continuing
					// to expand its zero-visit descendants creates long random chains and
					// leaves default_policy() with little or no work to do.
					if (expanded_leaf || problem->is_terminal(state_of(child))){
						max_depth = d+1;
						break;
					}
				}
				store_matrix(max_depth, default_policy(problem, state_of(curr), turn, max_depth));
				m_path[max_depth] = curr;

				for (int d = 0; d <= max_depth; d++){
					m_meta[m_path[d]].num_visits += 1;
					add_return(m_path[d], d, max_depth + 1, problem->m_gamma);
				}
			};

			solver_result.success = true;
			int root_eval_turn = tree_turn(problem, turn, 0);
			// Visit counts are dominated by the exploration bonus when the root
			// has many nearly tied children.  Use the backed-up Q value for the
			// executable action and fall back to visits only if no valid Q exists.
			int best_action_node = best_value_child(problem, 0, root_eval_turn);
			if (best_action_node < 0){
				best_action_node = most_visited(0, root_eval_turn);
			}
			if (best_action_node < 0){
				solver_result.success = false;
				return solver_result;
			}
			solver_result.best_action = action_of(best_action_node);
			// The executable action above is selected by team-average Q.  Policy
			// targets need individual credit: copying that joint action to both
			// teammates makes one robot inherit reward caused by the other and
			// produces contradictory, low-magnitude evader labels.
			solver_result.best_actions.resize(problem->m_num_robots, m_action_dim);
			solver_result.best_actions.setZero();
			for (int robot = 0; robot < problem->m_num_robots; ++robot) {
				int robot_node = best_value_child_for_robot(problem, 0, robot);
				if (robot_node < 0) {
					robot_node = most_visited(0, robot);
				}
				if (robot_node >= 0) {
					solver_result.best_actions.row(robot) = action_of(robot_node).transpose();
				}
			}
			solver_result.child_distribution = export_child_distribution(problem);
			if (m_export_tree) {
				solver_result.tree = export_tree(problem);
			}
			float visits = (float)m_meta[0].num_visits;
			solver_result.value = value_of(0) / visits;
			solver_result.num_visits = m_meta[0].num_visits;
			return solver_result;
		}


	private:
		struct Meta {
			int parent = -1;
			int num_visits = 0;
			int child_head = -1;
			int child_tail = -1;
			int child_count = 0;
		};
		struct Edge {
			int child = -1;
			int next = -1;
		};

		using VecMap = Eigen::Map<Eigen::Matrix<float, -1, 1>, Eigen::Unaligned>;

		void begin_tree(Problem* problem, const Eigen::Matrix<float,-1,1>& root_state) {
			m_state_dim = problem->m_state_dim;
			m_action_dim = problem->m_action_dim;
			m_value_dim = problem->m_num_robots;
			// Each simulation adds at most one node because search stops at the
			// newly expanded leaf.  Reserving simulations*depth here allocated
			// roughly 600k nodes for a 3000x200 search even though at most 3001
			// nodes can be created.
			const std::size_t max_nodes = std::size_t(m_num_simulations) + 1;
			m_meta.clear();
			m_edges.clear();
			m_states.clear();
			m_actions.clear();
			m_values.clear();
			m_edge_rewards.clear();
			m_meta.reserve(max_nodes);
			m_edges.reserve(max_nodes);
			m_states.reserve(max_nodes * std::size_t(m_state_dim));
			m_actions.reserve(max_nodes * std::size_t(m_action_dim));
			m_values.reserve(max_nodes * std::size_t(m_value_dim));
			m_edge_rewards.reserve(max_nodes * std::size_t(m_value_dim));
			m_reward_buf.assign(std::size_t(m_search_depth + 1) * std::size_t(m_value_dim), 0.f);
			m_path.assign(std::size_t(m_search_depth + 1), 0);
			const int root = add_node(-1);
			state_of(root) = root_state;
		}

		int add_node(int parent) {
			const int id = (int)m_meta.size();
			Meta meta;
			meta.parent = parent;
			m_meta.push_back(meta);
			m_states.resize(m_states.size() + std::size_t(m_state_dim), 0.f);
			m_actions.resize(m_actions.size() + std::size_t(m_action_dim), 0.f);
			m_values.resize(m_values.size() + std::size_t(m_value_dim), 0.f);
			m_edge_rewards.resize(m_edge_rewards.size() + std::size_t(m_value_dim), 0.f);
			return id;
		}

		VecMap row_at(float* data, int index, int width) {
			return VecMap(data + std::ptrdiff_t(index) * width, width);
		}

		VecMap matrix_at(float* data, int index) {
			return row_at(data, index, m_value_dim);
		}

		VecMap state_of(int node) { return row_at(m_states.data(), node, m_state_dim); }
		VecMap action_of(int node) { return row_at(m_actions.data(), node, m_action_dim); }
		VecMap value_of(int node) { return row_at(m_values.data(), node, m_value_dim); }
		VecMap edge_reward_of(int node) { return row_at(m_edge_rewards.data(), node, m_value_dim); }

		void store_matrix(int index, const Eigen::Matrix<float,-1,1>& value) {
			matrix_at(m_reward_buf.data(), index) = value;
		}

		void store_reward(int depth, const Eigen::Matrix<float,-1,1>& reward) {
			store_matrix(depth, reward);
		}

		void add_return(int node, int path_depth, int total_depth, float gamma) {
			VecMap value = value_of(node);
			for (int depth = path_depth; depth < total_depth; depth++) {
				value += matrix_at(m_reward_buf.data(), depth) *
					powf(gamma, (float)(depth - path_depth));
			}
		}

		void append_child(int parent, int child) {
			const int edge = (int)m_edges.size();
			m_edges.push_back(Edge{child, -1});
			Meta& meta = m_meta[parent];
			if (meta.child_head < 0) {
				meta.child_head = edge;
				meta.child_tail = edge;
			} else {
				m_edges[meta.child_tail].next = edge;
				meta.child_tail = edge;
			}
			meta.child_count += 1;
		}

		template <typename Fn>
		void for_each_child(int node, Fn&& fn) const {
			for (int edge = m_meta[node].child_head; edge >= 0; edge = m_edges[edge].next) {
				fn(m_edges[edge].child);
			}
		}

		float team_edge_reward(int node, int robot_turn) const {
			const float* reward = m_edge_rewards.data() + std::ptrdiff_t(node) * m_value_dim;
			if (m_two_teams) {
				int first = (robot_turn < 2) ? 0 : 2;
				return 0.5f * (reward[first] + reward[first + 1]);
			}
			return reward[robot_turn];
		}

		int best_child(Problem * problem, int node, int robot_turn){
			int result = -1;
			int first = -1;
			float bestValue = -std::numeric_limits<float>::infinity();

			for_each_child(node, [&](int child) {
				if (first < 0) first = child;
				if (m_meta[child].num_visits <= 0) return;

				float q = team_edge_reward(child, robot_turn) +
					problem->m_gamma * team_value(child, robot_turn) /
					(float)m_meta[child].num_visits;
				float u = m_C_exp * sqrtf(
					powf((float)std::max(1, m_meta[node].num_visits), m_alpha_exp) / (float)m_meta[child].num_visits
				);
				float value = q + u;

				if (!std::isfinite(value)) return;
				if (result < 0 || value > bestValue) {
					bestValue = value;
					result = child;
				}
			});

			if (result < 0) return first;
			return result;
		}

		int most_visited(int node, int robot_turn){
			(void)robot_turn;
			int result = -1;
			int mostVisits = 0;
			for_each_child(node, [&](int child) {
				if (m_meta[child].num_visits > mostVisits) {
					mostVisits = m_meta[child].num_visits;
					result = child;
				}
			});
			return result;
		}

		int best_value_child(Problem * problem, int node, int robot_turn){
			int result = -1;
			float bestValue = -std::numeric_limits<float>::infinity();
			for_each_child(node, [&](int child) {
				if (m_meta[child].num_visits <= 0) return;
				float q = team_edge_reward(child, robot_turn) +
					problem->m_gamma * team_value(child, robot_turn) /
					(float)m_meta[child].num_visits;
				if (!std::isfinite(q)) return;
				if (result < 0 || q > bestValue) {
					bestValue = q;
					result = child;
				}
			});
			return result;
		}

		int best_value_child_for_robot(Problem * problem, int node, int robot){
			int result = -1;
			float bestValue = -std::numeric_limits<float>::infinity();
			for_each_child(node, [&](int child) {
				if (m_meta[child].num_visits <= 0) return;
				const float* reward = m_edge_rewards.data() + std::ptrdiff_t(child) * m_value_dim;
				const float* value = m_values.data() + std::ptrdiff_t(child) * m_value_dim;
				float q = reward[robot] + problem->m_gamma * value[robot] /
					(float)m_meta[child].num_visits;
				if (!std::isfinite(q)) return;
				if (result < 0 || q > bestValue) {
					bestValue = q;
					result = child;
				}
			});
			return result;
		}


		bool is_expanded(int node){
			int max_children = std::max(
				1, (int)std::ceil(m_C_pw * powf((float)m_meta[node].num_visits, m_alpha_pw)));
			return m_meta[node].child_count >= max_children;
		}


		int expand_node(Problem * problem, int parent, int robot_turn){
			// Keep unconstrained random actions for tree expansion.  Invalid
			// successors must remain in the tree so normalized_reward() can
			// assign the boundary/obstacle penalty and teach the search which
			// directions are unsafe.  The validity-filtered sampler erased that
			// contrast and made the remaining pursuit signal nearly random.
			auto action = problem->sample_action(g_gen);
			keep_turn_action(problem, action, state_of(parent), robot_turn);
			if (problem->dist(g_gen) < m_beta_policy){
				for (int ii = 0; ii < problem->m_num_robots; ii++) {
					if (robot_is_on_turn(problem, robot_turn, ii) &&
						m_policy_network_wrappers[ii].valid && problem->is_active(state_of(parent), ii)){
						auto encoding = problem->policy_encoding(state_of(parent), ii);
						action.block(problem->m_action_idxs[ii][0],0,problem->m_action_idxs[ii].size(),1) =
							m_policy_network_wrappers[ii].policy_network->eval(problem, encoding, ii, g_gen);
					}
				} 
			} 

			auto next_state = problem->step(state_of(parent), action, problem->m_timestep);
			// Reuse the successor already computed for the edge instead of
			// propagating the same action once more inside normalized_reward().
			auto edge_reward = problem->normalized_reward_from_next(
				state_of(parent), action, next_state);
			const int child = add_node(parent);
			action_of(child) = action;
			state_of(child) = next_state;
			edge_reward_of(child) = edge_reward;
			append_child(parent, child);
			return child;
		}


		Eigen::Matrix<float,-1,1> default_policy(
			Problem * problem,
			Eigen::Matrix<float,-1,1> curr_state,
			int turn,
			int start_depth){
			Eigen::Matrix<float,-1,1> value = Eigen::Matrix<float,-1,1>::Zero(problem->m_num_robots,1);
			int depth = 0; 

			if (problem->is_terminal(curr_state)) {
				return value;
			}
			if (m_value_network_wrapper.valid && problem->dist(g_gen) < m_beta_value) {
				auto encoding = problem->value_encoding(curr_state);
				value = m_value_network_wrapper.value_network->eval(problem,encoding,g_gen);
			} else {
				problem->set_rollout_mode(true);
				while ((!problem->is_terminal(curr_state)) && (depth < m_search_depth)) {
					// Keep random rollouts inside the playable region so they reach
					// capture/survival outcomes instead of terminating immediately
					// on a boundary or obstacle. Tree expansion still uses the
					// unconstrained sampler above, preserving collision penalties as
					// a search signal.
					auto action = problem->sample_rollout_action(curr_state, g_gen);
					const int robot_turn = tree_turn(problem, turn, start_depth + depth);
					keep_turn_action(problem, action, curr_state, robot_turn);
					auto next_state = problem->step(curr_state,action,problem->m_timestep);
					value += powf(problem->m_gamma,depth) *
						problem->normalized_reward_from_next(curr_state, action, next_state);
					curr_state = next_state;
					depth += 1;
				}
				problem->set_rollout_mode(false);
			}

			return value; 
		}


		Eigen::MatrixXf export_tree(Problem * problem){
			Eigen::MatrixXf tree((int)m_meta.size(), problem->m_state_dim + 1);
			for (int ii = 0; ii < (int)m_meta.size(); ++ii) {
				tree.row(ii).head(problem->m_state_dim) = state_of(ii);
				tree(ii, problem->m_state_dim) = m_meta[ii].parent;
			}
			return tree; 
		}


		Eigen::MatrixXf export_child_distribution(Problem * problem){
			Eigen::MatrixXf child_distribution(m_meta[0].child_count, problem->m_action_dim + 1);
			int count = 0;
			for_each_child(0, [&](int child) {
				child_distribution.row(count).head(problem->m_action_dim) = action_of(child);
				child_distribution(count, problem->m_action_dim) = m_meta[child].num_visits;
				count += 1;
			});
			return child_distribution;
		}


		int m_num_simulations = 0;
		int m_search_depth = 0;
		float m_C_exp = 0.f;
		float m_alpha_exp = 0.f;
		float m_C_pw = 0.f;
		float m_alpha_pw = 0.f;
		float m_beta_policy = 0.f;
		float m_beta_value = 0.f;
		bool m_export_tree = false;
		bool m_two_teams = false;
		int m_state_dim = 0;
		int m_action_dim = 0;
		int m_value_dim = 0;
		std::vector<Meta> m_meta;
		std::vector<Edge> m_edges;
		std::vector<float> m_states;
		std::vector<float> m_actions;
		std::vector<float> m_values;
		std::vector<float> m_edge_rewards;
		std::vector<float> m_reward_buf;
		std::vector<int> m_path;

};
