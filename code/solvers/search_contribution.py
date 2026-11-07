"""Whole-policy deadlines and isolated post-capture contribution variants."""
import time
import types
import numpy as np
from solvers.c_puct import C_PUCT
from solvers.candidate_ablation import CandidateAblationPUCT
from cpp.build_search_contribution import bindings_search_contribution as cpp

def native(method):
    scope=dict(method.__globals__)
    for name in ('Problem_Settings','Problem_Wrapper','Policy_Network_Wrapper','Value_Network_Wrapper'):
        scope[name]=getattr(cpp,name)
    scope['cpp']=cpp
    return types.FunctionType(method.__code__,scope,method.__name__,method.__defaults__,method.__closure__)

class ContributionPUCT(CandidateAblationPUCT):
    _problem_wrapper_for=native(C_PUCT._problem_wrapper_for)
    create_cpp_policy_oracle=native(C_PUCT.create_cpp_policy_oracle)
    create_cpp_value_oracle=native(C_PUCT.create_cpp_value_oracle)

    def __init__(self,policies,value,budget,variant='G',deadline_ms=0.):
        self.calls=0;self.last_result=None;self.cost=dict(step_calls=0,policy_evaluations=0,value_evaluations=0)
        self.policy_oracle=self.create_cpp_policy_oracle(policies);self.value_oracle=self.create_cpp_value_oracle(value)
        self.variant=variant;self.deadline_ms=float(deadline_ms);self.policy_deadline=None;self.last_policy_ms=0.
        self.vis_on=False;self.solver_name='C_CONTRIBUTION_'+dict(S='S',F='F',G='G',N='N',Geometry='GEOMETRY')[variant]
        ss=self.solver_settings=cpp.Solver_Settings();ss.number_simulations=int(budget);ss.search_depth=100
        ss.C_pw,ss.alpha_pw,ss.C_exp,ss.alpha_exp=2.,.5,1.,.25;ss.beta_policy,ss.beta_value=.5,0.
        ss.export_tree,ss.export_root=False,True
        self.solver_wrapper=cpp.Solver_Wrapper(self.solver_name,ss,self.policy_oracle,self.value_oracle)

    def policy(self,problem,state,team=None):
        started=time.perf_counter()
        self.policy_deadline=started+self.deadline_ms/1000 if self.deadline_ms>0 else None
        try:return super().policy(problem,state,team=team)
        finally:self.last_policy_ms=(time.perf_counter()-started)*1000;self.policy_deadline=None

    def search(self,problem,root_state,turn=0):
        wrapper=self._problem_wrapper_for(problem)
        remaining=max(.000001,(self.policy_deadline-time.perf_counter())*1000) if self.policy_deadline is not None else 0.
        self.solver_wrapper.set_time_limit_ms(remaining)
        result=cpp.cpp_search(wrapper,self.solver_wrapper,root_state,turn)
        if not result.success or not np.isfinite(result.best_action).all():raise RuntimeError('Contribution search failed')
        self.calls+=1;self.last_result=result
        for key in self.cost:self.cost[key]+=int(getattr(result,key))
        return result
