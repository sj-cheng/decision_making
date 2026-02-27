
#pragma once 

// #include <string>
#include <vector>
#include <random>
#include <eigen3/Eigen/Dense>
#include "problem.hpp"

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
				// block(i,j,p,q): Block of size (p,q), starting at (i,j) 
                next_state.block(m_state_idxs[ii][0],0,m_state_idxs[ii].size(),1) = 
                    Fd * state.block(m_state_idxs[ii][0],0,m_state_idxs[ii].size(),1) + 
                    Bd * action.block(m_action_idxs[ii][0],0,m_action_idxs[ii].size(),1);
            }   
            
            next_state(8,0) = state(8,0) + timestep;
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
			auto next_statee = step(state, action, m_timestep);
			Eigen::Matrix<float,-1,1> r(m_num_robots,1);
			r.setZero();
				float r1=0.0 , r2=0.0;
			if (is_captured(next_statee) || next_statee(8,0) >= m_tf){
				r1 = next_statee(8,0) / m_tf;
				r2 = 1 - r1;
				printf("captured or terminal: r1=%f, r2=%f\n", r1, r2);
			} 
            r(0,0)=r1; r(1,0)=r1; r(2,0)=r2; r(3,0)=r2;
            
        
            for (int j=0; j<m_num_robots; ++j){
                int s0 = m_state_idxs[j][0];
                int sd = (int)m_state_idxs[j].size();
                bool in_low  = (next_statee.block(s0,0,sd,1).array() >= m_state_lims.block(s0,0,sd,1).array()).all();
                bool in_high = (next_statee.block(s0,0,sd,1).array() <= m_state_lims.block(s0,1,sd,1).array()).all();
                if (!(in_low && in_high)) r(j,0) = -1.0;
            }
		// 	if (is_captured(next_statee) || next_statee(4,0) >= m_tf){
		// 		float t = std::min(next_statee(4,0), m_tf);
		// 		r(0,0) = t / m_tf;
		// 		r(1,0) = 1.0 - r(0,0);
		// 	} else if ( !(
		// 		(next_statee.block(0,0,2,1).array() >= m_state_lims.block(0,0,2,1).array()).all() && 
		// 		(next_statee.block(0,0,2,1).array() <= m_state_lims.block(0,1,2,1).array()).all() )) {
		// 		r(0,0) = -1.0;
		// 	} else if ( !(
		// 		(next_statee.block(2,0,2,1).array() >= m_state_lims.block(2,0,2,1).array()).all() &&
		// 		(next_statee.block(2,0,2,1).array() <= m_state_lims.block(2,1,2,1).array()).all() )) {
		// 		r(1,0) = -1.0;
		// 	}
		// 	    else {
		// 	// ---- minimal progress shaping (dense signal, anti-hover) ----
		// 	// reward pursuer for reducing distance: shape = k*(d - dn)
		// 	const float k = 0.05f;  // 0.02~0.05 先试 0.03

		// 	const float d  = (state.block(0,0,2,1)      - state.block(2,0,2,1)).norm();
		// 	const float dn = (next_statee.block(0,0,2,1) - next_statee.block(2,0,2,1)).norm();

		// 	const float shape = k * (d - dn);  // >0 means pursuer got closer

		// 	r(1,0) += shape;  // pursuer +
		// 	r(0,0) -= shape;  // evader  -
		// }
			return r;
        }

		// Eigen::Matrix<float,-1,1> normalized_reward(
        //     Eigen::Matrix<float,-1,1> state,
        //     Eigen::Matrix<float,-1,1> action) override
        // {
		// 	Eigen::Matrix<float,-1,1> r(m_num_robots,1);
		// 	r.setZero();
		// 	if (is_captured(state) || state(4,0) > m_tf){
		// 		float t = std::min(state(4,0), m_tf);
		// 		r(0,0) = t / m_tf;
		// 		r(1,0) = 1.0f - r(0,0);
		// 	} else if ( !(
		// 		(state.block(0,0,2,1).array() >= m_state_lims.block(0,0,2,1).array()).all() && 
		// 		(state.block(0,0,2,1).array() <= m_state_lims.block(0,1,2,1).array()).all() )) {
		// 		r(0,0) = -1.0;
		// 	} else if ( !(
		// 		(state.block(2,0,2,1).array() >= m_state_lims.block(2,0,2,1).array()).all() &&
		// 		(state.block(2,0,2,1).array() <= m_state_lims.block(2,1,2,1).array()).all() )) {
		// 		r(1,0) = -1.0;
		// 	}
			
        //     return r;
        // }


        bool is_valid(Eigen::Matrix<float,-1,1> state) override
        {
            return (state.array() >= m_state_lims.col(0).array()).all() && (state.array() <= m_state_lims.col(1).array()).all();
        }

        bool is_terminal(Eigen::Matrix<float,-1,1> state) override 
        {
            // return !is_valid(state);
            return ( (!is_valid(state))) || is_captured(state);
        }

        bool is_captured(Eigen::Matrix<float,-1,1> state) {
        	//return (state.block(0,0,2,1) - state.block(2,0,2,1)).norm() < m_dist;
			float min_d=1e5;
			for (int e=0; e<2; ++e){
    			for (int p=2; p<4; ++p){
        			float d = (state.block(m_state_idxs[e][0],0,2,1) - state.block(m_state_idxs[p][0],0,2,1)).norm();
        			if (d < min_d) min_d = d;
    			}
			}
        	return min_d < m_dist;
        }
		
};


// #pragma once 

// // #include <string>
// #include <vector>
// #include <random>
// #include <eigen3/Eigen/Dense>
// #include "problem.hpp"

// class Example8 : public Problem { 
	
// 	public:
// 		Eigen::Matrix<float,2,2> m_Fc;
// 		Eigen::Matrix<float,2,2> m_Bc;
// 		Eigen::Matrix<float,2,2> m_I; 
// 		Eigen::Matrix<float,2,2> m_Q;
// 		Eigen::Matrix<float,2,2> m_R;  
// 		float m_r_max; 
// 		float m_r_min; 
// 		float m_state_control_weight;
// 		float m_dist;
// 		float m_tf;
// 		int m_state_dim_per_robot; 
// 		int m_action_dim_per_robot;

// 		void set_params(Problem_Settings & problem_settings) override 
// 		{
//             m_state_dim = problem_settings.state_dim;
//             m_action_dim = problem_settings.action_dim;
//             m_num_robots = problem_settings.num_robots;
//             m_state_idxs = problem_settings.state_idxs;
//             m_action_idxs = problem_settings.action_idxs;

//             // problem_settings.state_lims.resize(m_state_dim,2);
//             // problem_settings.action_lims.resize(m_action_dim,2);
//             // problem_settings.init_lims.resize(m_state_dim,2);

// 			m_timestep = problem_settings.timestep;
// 			m_tf = problem_settings.tf;
// 			m_gamma = problem_settings.gamma;
// 			m_r_max = problem_settings.r_max;
// 			m_r_min = problem_settings.r_min;
// 			m_state_control_weight = problem_settings.state_control_weight;
// 			m_state_lims = problem_settings.state_lims; 
// 			m_action_lims = problem_settings.action_lims; 
// 			m_init_lims = problem_settings.init_lims;
// 			m_dist = problem_settings.desired_distance;

// 			std::uniform_real_distribution<double> dist(0,1.0f); 

// 			m_Fc.setZero();
// 			m_Bc.setIdentity();
// 			m_I.setIdentity();

// 			m_Q.setIdentity();
// 			m_R.setIdentity();
// 			m_R = m_R * m_state_control_weight;
// 		}


// 		Eigen::Matrix<float,-1,1> step(
// 			Eigen::Matrix<float,-1,1> state,
// 			Eigen::Matrix<float,-1,1> action,
// 			float timestep) override
// 		{
// 			Eigen::Matrix<float,-1,1> next_state(m_state_dim,1); 
// 			Eigen::Matrix<float,2,2> Fd = m_I + m_Fc * timestep;
// 			Eigen::Matrix<float,2,2> Bd = m_Bc * timestep; 

//             // dynamics 
//             for (int ii = 0; ii < m_num_robots; ii++){
// 				// block(i,j,p,q): Block of size (p,q), starting at (i,j) 
//                 next_state.block(m_state_idxs[ii][0],0,m_state_idxs[ii].size(),1) = 
//                     Fd * state.block(m_state_idxs[ii][0],0,m_state_idxs[ii].size(),1) + 
//                     Bd * action.block(m_action_idxs[ii][0],0,m_action_idxs[ii].size(),1);
//             }   
            
//             next_state(8,0) = state(8,0) + timestep;
//             return next_state;
// 		}


//         Eigen::Matrix<float,-1,1> reward(
//             Eigen::Matrix<float,-1,1> state,
//             Eigen::Matrix<float,-1,1> action) override
//         { 
//             return normalized_reward(state,action);
//         }


//         Eigen::Matrix<float,-1,1> normalized_reward(
//             Eigen::Matrix<float,-1,1> state,
//             Eigen::Matrix<float,-1,1> action) override
//         {
// 			Eigen::Matrix<float,-1,1> r(m_num_robots,1);
// 			r(0,0) = 0.0;
// 			r(1,0) = 0.0;
// 			r(2,0) = 0.0;
// 			r(3,0) = 0.0;
// 			float r1=0.0 , r2=0.0;
// 			if (is_captured(state) || state(8,0) > m_tf){
// 				r1 = state(8,0) / m_tf;
// 				r2 = 1 - r1;
// 			} 
			
           
//             r(0,0)=r1; r(1,0)=r1; r(2,0)=r2; r(3,0)=r2;
            
        
//             for (int j=0; j<m_num_robots; ++j){
//                 int s0 = m_state_idxs[j][0];
//                 int sd = (int)m_state_idxs[j].size();
//                 bool in_low  = (state.block(s0,0,sd,1).array() >= m_state_lims.block(s0,0,sd,1).array()).all();
//                 bool in_high = (state.block(s0,0,sd,1).array() <= m_state_lims.block(s0,1,sd,1).array()).all();
//                 if (!(in_low && in_high)) r(j,0) = -1.0;
//             }
// 			// else if ( !(
// 			// 	(state.block(0,0,2,1).array() >= m_state_lims.block(0,0,2,1).array()).all() && 
// 			// 	(state.block(0,0,2,1).array() <= m_state_lims.block(0,1,2,1).array()).all() )) {
// 			// 	r(0,0) = -1;
// 			// } else if ( !(
// 			// 	(state.block(2,0,2,1).array() >= m_state_lims.block(2,0,2,1).array()).all() &&
// 			// 	(state.block(2,0,2,1).array() <= m_state_lims.block(2,1,2,1).array()).all() )) {
// 			// 	r(1,0) = -1;
// 			// }
//             return r;
//         }


//         bool is_valid(Eigen::Matrix<float,-1,1> state) override
//         {
//             return (state.array() >= m_state_lims.col(0).array()).all() && (state.array() <= m_state_lims.col(1).array()).all();
//         }

//         bool is_terminal(Eigen::Matrix<float,-1,1> state) override 
//         {
//             // return !is_valid(state);
//             return ( (!is_valid(state))) || is_captured(state);
//         }

//         bool is_captured(Eigen::Matrix<float,-1,1> state) {
// 			float min_d=1e5;
// 			for (int e=0; e<2; ++e){
//     			for (int p=2; p<4; ++p){
//         			float d = (state.block(m_state_idxs[e][0],0,2,1) - state.block(m_state_idxs[p][0],0,2,1)).norm();
//         			if (d < min_d) min_d = d;
//     			}
// 			}
//         	return min_d < m_dist;
//         }
		
// };
