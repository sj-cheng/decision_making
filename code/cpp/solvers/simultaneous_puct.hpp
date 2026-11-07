#pragma once
#include "solver.hpp"
#include <array>
#include <cmath>
#include <limits>
#include <map>

// Diagnostic baseline: decoupled UCT with progressive widening. Both arms
// are selected at the same node, before either action is revealed or applied.
// One joint edge advances one physical timestep. No equilibrium guarantee.
class SimultaneousPUCT : public Solver {
    using Vec = Eigen::VectorXf;
    struct Arm { Vec action; Vec sum; int visits=0; };
    struct Node {
        Vec state;
        int visits=0;
        std::array<std::vector<Arm>,2> arms;
        std::map<std::pair<int,int>,int> children;
    };
    struct Edge { int node, a, b; Vec reward; };
    Solver_Settings settings;
    std::vector<Node> nodes;
    long long policies=0, values=0;

    float mean_team(const Vec& x, int team) const {
        return .5f*(x(team*2)+x(team*2+1));
    }
    Vec sample(Problem* p, const Vec& s, int team) {
        Vec a=p->sample_action(g_gen);
        const bool neural=p->dist(g_gen)<settings.beta_policy;
        for (int r=0;r<4;++r) {
            if (r/2!=team || !p->is_active(s,r)) {
                a.segment(p->m_action_idxs[r][0],p->m_action_idxs[r].size()).setZero();
            } else if (neural && m_policy_network_wrappers[r].valid) {
                ++policies;
                a.segment(p->m_action_idxs[r][0],p->m_action_idxs[r].size())=
                    m_policy_network_wrappers[r].policy_network->eval(p,p->policy_encoding(s,r),r,g_gen);
            }
        }
        return a;
    }
    int choose(Problem* p, int n, int team) {
        const int width=std::max(1,(int)std::ceil(settings.C_pw*
            std::pow((float)nodes[n].visits,settings.alpha_pw)));
        if ((int)nodes[n].arms[team].size()<width) {
            Arm arm; arm.action=sample(p,nodes[n].state,team); arm.sum=Vec::Zero(4);
            nodes[n].arms[team].push_back(std::move(arm));
            return (int)nodes[n].arms[team].size()-1;
        }
        int best=0; float score=-std::numeric_limits<float>::infinity();
        for (int i=0;i<(int)nodes[n].arms[team].size();++i) {
            const Arm& a=nodes[n].arms[team][i];
            float u=a.visits ? mean_team(a.sum,team)/a.visits + settings.C_exp*
                std::sqrt(std::pow((float)std::max(1,nodes[n].visits),settings.alpha_exp)/a.visits)
                : std::numeric_limits<float>::infinity();
            if (u>score) {score=u;best=i;}
        }
        return best;
    }
    Vec rollout(Problem* p, Vec s, int remaining) {
        Vec total=Vec::Zero(4);
        if (p->is_terminal(s)) return total;
        if (m_value_network_wrapper.valid && p->dist(g_gen)<settings.beta_value) {
            ++values;
            return m_value_network_wrapper.value_network->eval(p,p->value_encoding(s),g_gen);
        }
        p->set_rollout_mode(true);
        try {
            float discount=1.f;
            for (int d=0;d<remaining && !p->is_terminal(s);++d) {
                Vec a=p->sample_rollout_action(s,g_gen);
                for (int r=0;r<4;++r) if (!p->is_active(s,r))
                    a.segment(p->m_action_idxs[r][0],p->m_action_idxs[r].size()).setZero();
                Vec next=p->step(s,a,p->m_timestep);
                total+=discount*p->normalized_reward_from_next(s,a,next);
                discount*=p->m_gamma;s=next;
            }
        } catch (...) {p->set_rollout_mode(false);throw;}
        p->set_rollout_mode(false);
        return total;
    }
public:
    void set_params(Solver_Settings& s,std::vector<Policy_Network_Wrapper>& pw,
                    Value_Network_Wrapper& vw) override {
        settings=s;m_policy_network_wrappers=pw;m_value_network_wrapper=vw;
    }
    Solver_Result search(Problem* p, Vec state, int turn) override {
        Solver_Result result;result.success=false;
        if (p->m_num_robots!=4 || p->is_terminal(state)) return result;
        const auto steps_before=p->diagnostic_step_calls;
        policies=values=0;nodes.clear();nodes.reserve(settings.num_simulations+1);
        Node root;root.state=state;nodes.push_back(std::move(root));
        Vec sum=Vec::Zero(4);
        for (int simulation=0;simulation<settings.num_simulations;++simulation) {
            int n=0,depth=0;std::vector<Edge> path;
            while (depth<settings.search_depth && !p->is_terminal(nodes[n].state)) {
                // Neither selection receives the other team's selected arm.
                const int a=choose(p,n,0),b=choose(p,n,1);
                const auto pair=std::make_pair(a,b);
                auto found=nodes[n].children.find(pair);
                int child;Vec reward;bool fresh=found==nodes[n].children.end();
                if (fresh) {
                    Vec joint=nodes[n].arms[0][a].action+nodes[n].arms[1][b].action;
                    Vec next=p->step(nodes[n].state,joint,p->m_timestep);
                    reward=p->normalized_reward_from_next(nodes[n].state,joint,next);
                    child=(int)nodes.size();Node node;node.state=next;
                    nodes.push_back(std::move(node));nodes[n].children.emplace(pair,child);
                } else {
                    child=found->second;
                    Vec joint=nodes[n].arms[0][a].action+nodes[n].arms[1][b].action;
                    reward=p->normalized_reward_from_next(nodes[n].state,joint,nodes[child].state);
                }
                path.push_back({n,a,b,reward});n=child;++depth;
                if (fresh) break;
            }
            Vec value=rollout(p,nodes[n].state,settings.search_depth-depth);
            ++nodes[n].visits;
            for (auto it=path.rbegin();it!=path.rend();++it) {
                value=it->reward+p->m_gamma*value;
                Node& parent=nodes[it->node];++parent.visits;
                Arm& a=parent.arms[0][it->a];Arm& b=parent.arms[1][it->b];
                ++a.visits;++b.visits;a.sum+=value;b.sum+=value;
            }
            sum+=value;
        }
        const int team=turn<2?0:1;
        int best=-1;float q=-std::numeric_limits<float>::infinity();
        auto& arms=nodes[0].arms[team];
        for (int i=0;i<(int)arms.size();++i) if (arms[i].visits) {
            float candidate=mean_team(arms[i].sum,team)/arms[i].visits;
            if (candidate>q) {q=candidate;best=i;}
        }
        if (best<0) return result;
        result.success=true;result.best_action=arms[best].action;
        result.best_actions=Eigen::MatrixXf::Zero(4,p->m_action_dim);
        result.child_distribution=Eigen::MatrixXf::Zero(arms.size(),p->m_action_dim+1);
        result.root_audit=Eigen::MatrixXf::Zero(arms.size(),p->m_action_dim+6);
        for (int i=0;i<(int)arms.size();++i) {
            result.child_distribution(i,0)=arms[i].visits;
            result.child_distribution.row(i).tail(p->m_action_dim)=arms[i].action.transpose();
            result.root_audit(i,0)=i;result.root_audit(i,1)=arms[i].visits;
            result.root_audit(i,3)=mean_team(arms[i].sum,team);
            result.root_audit(i,4)=mean_team(arms[i].sum,team)/std::max(1,arms[i].visits);
            result.root_audit(i,5)=i==best;
            result.root_audit.row(i).tail(p->m_action_dim)=arms[i].action.transpose();
        }
        result.num_visits=settings.num_simulations;result.value=sum/settings.num_simulations;
        result.step_calls=p->diagnostic_step_calls-steps_before;
        result.policy_evaluations=policies;result.value_evaluations=values;
        return result;
    }
};
