
#include <iostream>
#include <eigen3/Eigen/Dense>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>
#include "problems/problem_wrapper.hpp"
#include "solvers/solver_wrapper.hpp"
#include "learning/policy_network_wrapper.hpp"
#include "learning/value_network_wrapper.hpp"
#include "local_geometry.hpp"


Solver_Result cpp_search(Problem_Wrapper& problem_wrapper, Solver_Wrapper& solver_wrapper, Eigen::Matrix<float,-1,1> state, int turn) {
    Solver_Result solver_result; 
    state.resize(problem_wrapper.problem->m_state_dim,1);
    solver_result = solver_wrapper.solver->search(problem_wrapper.problem,state,turn);
    return solver_result;
}


// python interface
PYBIND11_MODULE(bindings, m) {

    m.def("cpp_search", &cpp_search, "PUCT");
    m.def("local_geometry_legal",&local_geometry::legal);
    m.def("local_geometry_clear",&local_geometry::clear);
    m.def("local_geometry_visible",&local_geometry::visibility);
    m.def("local_geometry_negative",&local_geometry::negative);
    // Causal local feature encoder: accepts sensor packets and posterior
    // samples only, with no simulator state or opponent identity argument.
    m.def("local_belief_features", [](Eigen::MatrixXf base, Eigen::MatrixXf positions,
            Eigen::VectorXf active, Eigen::MatrixXf limits, Eigen::MatrixXf particles,
            Eigen::VectorXf age, Eigen::VectorXf visible, Eigen::VectorXf existence) {
        if (base.rows()!=2 || base.cols()!=270 || positions.rows()!=2 || positions.cols()!=2 ||
            active.size()!=2 || limits.rows()!=4 || limits.cols()!=2 || particles.rows()==0 ||
            particles.cols()!=8 || age.size()!=2 || visible.size()!=2 || existence.size()!=2)
            throw std::invalid_argument("invalid local observation feature dimensions");
        Eigen::MatrixXf result = Eigen::MatrixXf::Zero(2,293);
        Eigen::RowVectorXf mean = particles.colwise().mean();
        Eigen::RowVectorXf variance = (particles.rowwise()-mean).array().square().colwise().mean();
        for (int local=0; local<2; ++local) {
            result.block(local,0,1,270)=base.row(local);
            int nearest=-1; float distance=std::numeric_limits<float>::infinity();
            for (int slot=0; slot<2; ++slot) {
                const int offset=slot*4, out=270+slot*9;
                float dx=mean(offset)-positions(local,0), dy=mean(offset+1)-positions(local,1);
                result(local,out)=dx/20.f; result(local,out+1)=dy/20.f;
                result(local,out+2)=mean(offset+2)/2.4f; result(local,out+3)=mean(offset+3)/2.4f;
                result(local,out+4)=variance(offset)/400.f; result(local,out+5)=variance(offset+1)/400.f;
                result(local,out+6)=std::min(age(slot)/80.f,1.f);
                result(local,out+7)=visible(slot); result(local,out+8)=existence(slot);
                if (existence(slot)>.5f && dx*dx+dy*dy<distance) {nearest=slot;distance=dx*dx+dy*dy;}
            }
            if (nearest>=0) {
                int offset=nearest*4;
                result(local,264)=(mean(offset)-positions(local,0))/20.f;
                result(local,265)=(mean(offset+1)-positions(local,1))/20.f;
                result(local,266)=mean(offset+2)/2.4f;result(local,267)=mean(offset+3)/2.4f;
                result(local,268)=std::min(age(nearest)/80.f,1.f);result(local,269)=visible(nearest);
            }
            result(local,288)=(positions(1-local,0)-positions(local,0))/20.f;
            result(local,289)=(positions(1-local,1)-positions(local,1))/20.f;
            result(local,290)=active(1-local);
            result(local,291)=limits(local*2,1)/1.2f;result(local,292)=limits(local*2+1,1)/1.2f;
        }
        return result.cwiseMax(-1.f).cwiseMin(1.f).eval();
    });
    m.def("physical_gaussian_parameters", [](Eigen::MatrixXf raw, Eigen::MatrixXf limits) {
        if (raw.cols()!=4 || limits.rows()!=2 || limits.cols()!=2)
            throw std::invalid_argument("invalid physical Gaussian dimensions");
        Eigen::MatrixXf result(raw.rows(),4);
        for (int row=0;row<raw.rows();++row) for (int axis=0;axis<2;++axis) {
            result(row,axis)=.5f*(limits(axis,0)+limits(axis,1))+.5f*(limits(axis,1)-limits(axis,0))*std::tanh(raw(row,axis));
            result(row,axis+2)=std::exp(.5f*std::max(-8.f,std::min(2.f,raw(row,axis+2))));
        }
        return result;
    });

    pybind11::class_<Problem_Wrapper> (m, "Problem_Wrapper")
        .def(pybind11::init<std::string,Problem_Settings>())
        // Expose the exact C++ encoders for cross-language parity checks.
        .def("policy_encoding", [](Problem_Wrapper &wrapper,
                                    Eigen::Matrix<float,-1,1> state, int robot) {
            return wrapper.problem->policy_encoding(state, robot);
        })
        .def("value_encoding", [](Problem_Wrapper &wrapper,
                                   Eigen::Matrix<float,-1,1> state) {
            return wrapper.problem->value_encoding(state);
        })
        .def("step", [](Problem_Wrapper &wrapper,
                         Eigen::Matrix<float,-1,1> state,
                         Eigen::Matrix<float,-1,1> action,
                         float timestep) {
            return wrapper.problem->step(state, action, timestep);
        })
        .def("normalized_reward", [](Problem_Wrapper &wrapper,
                                      Eigen::Matrix<float,-1,1> state,
                                      Eigen::Matrix<float,-1,1> action) {
            return wrapper.problem->normalized_reward(state, action);
        })
        .def("transition_reward", [](Problem_Wrapper &wrapper,
                                      Eigen::Matrix<float,-1,1> state,
                                      Eigen::Matrix<float,-1,1> action,
                                      Eigen::Matrix<float,-1,1> next) {
            return wrapper.problem->normalized_reward_from_next(state, action, next);
        })
        .def("is_terminal", [](Problem_Wrapper &wrapper, Eigen::Matrix<float,-1,1> state) {
            return wrapper.problem->is_terminal(state);
        });

    pybind11::class_<Problem_Settings> (m, "Problem_Settings")
        .def(pybind11::init())
        .def_readwrite("state_dim", &Problem_Settings::state_dim)
        .def_readwrite("action_dim", &Problem_Settings::action_dim)
        .def_readwrite("num_robots", &Problem_Settings::num_robots)
        .def_readwrite("timestep", &Problem_Settings::timestep)
        .def_readwrite("tf", &Problem_Settings::tf)
        .def_readwrite("state_lims", &Problem_Settings::state_lims)
        .def_readwrite("action_lims", &Problem_Settings::action_lims)
        .def_readwrite("state_idxs", &Problem_Settings::state_idxs)
        .def_readwrite("action_idxs", &Problem_Settings::action_idxs)
        .def_readwrite("init_lims", &Problem_Settings::init_lims)
        .def_readwrite("desired_distance", &Problem_Settings::desired_distance)
        .def_readwrite("swept_collision", &Problem_Settings::swept_collision)
        .def_readwrite("evader_distance_reward_weight", &Problem_Settings::evader_distance_reward_weight)
        .def_readwrite("cross_team_distance_reward_weight", &Problem_Settings::cross_team_distance_reward_weight)
        .def_readwrite("evader_progress_reward_weight", &Problem_Settings::evader_progress_reward_weight)
        .def_readwrite("pursuer_progress_reward_weight", &Problem_Settings::pursuer_progress_reward_weight)
        .def_readwrite("desired_state", &Problem_Settings::desired_state)
        .def_readwrite("r_max", &Problem_Settings::r_max)
        .def_readwrite("r_min", &Problem_Settings::r_min)
        .def_readwrite("g", &Problem_Settings::g)
        .def_readwrite("mass", &Problem_Settings::mass)
        .def_readwrite("state_control_weight", &Problem_Settings::state_control_weight)
        .def_readwrite("obstacles", &Problem_Settings::obstacles)
        .def_readwrite("m1", &Problem_Settings::m1)
        .def_readwrite("m2", &Problem_Settings::m2)
        .def_readwrite("c1", &Problem_Settings::c1)
        .def_readwrite("c2", &Problem_Settings::c2)
        .def_readwrite("R", &Problem_Settings::R)
        .def_readwrite("gamma", &Problem_Settings::gamma);

    pybind11::class_<Solver_Wrapper> (m, "Solver_Wrapper")
        .def(pybind11::init<std::string,Solver_Settings,std::vector<Policy_Network_Wrapper>,Value_Network_Wrapper>())
        .def("set_seed",[](Solver_Wrapper &wrapper,unsigned int seed) {wrapper.solver->g_gen.seed(seed);});

    pybind11::class_<Solver_Settings> (m, "Solver_Settings")
        .def(pybind11::init())
        .def_readwrite("number_simulations", &Solver_Settings::num_simulations)
        .def_readwrite("search_depth", &Solver_Settings::search_depth)
        .def_readwrite("C_exp", &Solver_Settings::C_exp)
        .def_readwrite("alpha_exp", &Solver_Settings::alpha_exp)
        .def_readwrite("C_pw", &Solver_Settings::C_pw)
        .def_readwrite("alpha_pw", &Solver_Settings::alpha_pw)
        .def_readwrite("beta_policy", &Solver_Settings::beta_policy)
        .def_readwrite("beta_value", &Solver_Settings::beta_value)
        .def_readwrite("export_tree", &Solver_Settings::export_tree);

    pybind11::class_<Solver_Result> (m, "Solver_Result")
        .def(pybind11::init())
        .def_readwrite("best_action", &Solver_Result::best_action)
        .def_readwrite("best_actions", &Solver_Result::best_actions)
        .def_readwrite("child_distribution", &Solver_Result::child_distribution)
        .def_readwrite("tree", &Solver_Result::tree)
        .def_readwrite("value", &Solver_Result::value)
        .def_readwrite("success", &Solver_Result::success);

    pybind11::class_<Policy_Network_Wrapper> (m, "Policy_Network_Wrapper")
        .def(pybind11::init())
        .def("initialize", &Policy_Network_Wrapper::initialize)
        .def("addLayer", &Policy_Network_Wrapper::addLayer);

    pybind11::class_<Value_Network_Wrapper> (m, "Value_Network_Wrapper")
        .def(pybind11::init())
        .def("initialize", &Value_Network_Wrapper::initialize)
        .def("addLayer", &Value_Network_Wrapper::addLayer);

}
