#pragma once
#include "interception_guidance.hpp"
#include "../learning/policy_network.hpp"

// F/D/G share P's physical action box, perturbation and endpoint validity rule.
struct RootCandidateAblation {
    using V=Eigen::Vector2f;
    static V policy_mean(PolicyNetwork* policy,Problem* p,const Eigen::VectorXf& state,int robot) {
        auto encoding=p->policy_encoding(state,robot);auto raw=policy->m_phi.eval(encoding);V mean;
        for(int i=0;i<2;++i) {
            int at=p->m_action_idxs[robot][i];float lo=p->m_action_lims(at,0),hi=p->m_action_lims(at,1);
            mean(i)=.5f*(hi+lo)+.5f*(hi-lo)*std::tanh(raw(i));
        }
        return mean;
    }
    static bool base(InterceptionGuide& guide,const Eigen::VectorXf& state,
                     int mode,const V& mean,V& action) {
        auto* p=guide.p;
        V at=state.segment(p->m_state_idxs[guide.pursuer][0],2);
        V target=state.segment(p->m_state_idxs[guide.evader][0],2);
        if(mode==1) {
            V upper;
            for(int i=0;i<2;++i)upper(i)=p->m_action_lims(p->m_action_idxs[guide.pursuer][i],1);
            float length=mean.cwiseAbs().cwiseQuotient(upper).maxCoeff();
            action=length>1e-8f?V(mean/length):V::Zero();
        } else if(mode==2) action=guide.move(at,target,guide.pursuer);
        else if(mode==3) return guide.step_to(at,guide.target(target,guide.pursuer),action);
        else return false;
        return action.allFinite()&&guide.legal(at+action);
    }
    static V proposal(InterceptionGuide& guide,const Eigen::VectorXf& state,int mode,
                      const V& mean,std::default_random_engine& gen,bool& ok) {
        // Match P's route lottery slot; controls do not use its route identity.
        if(!guide.routes.empty())guide.p->dist(gen);
        V action=V::Zero();ok=base(guide,state,mode,mean,action);
        ++guide.stats["control_proposals"];
        if(!ok){++guide.stats["proposal_fallback"];return action;}
        V noisy=action;std::normal_distribution<float> normal(0,1);
        for(int i=0;i<2;++i) {
            float upper=guide.p->m_action_lims(guide.p->m_action_idxs[guide.pursuer][i],1);
            noisy(i)=std::max(-upper,std::min(upper,action(i)+.1f*upper*normal(gen)));
        }
        V at=state.segment(guide.p->m_state_idxs[guide.pursuer][0],2);
        if(!guide.legal(at+noisy)){++guide.stats["noise_fallback"];return action;}
        return noisy;
    }
};
