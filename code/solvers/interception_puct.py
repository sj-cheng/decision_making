"""Interception extension; a root with two evaders follows frozen search exactly."""
import numpy as np
from solvers.deployment_puct import DeploymentPUCT
from solvers.c_puct import C_PUCT
from cpp.build_interception import bindings_interception as cpp
import types

def native_method(method):
    namespace=dict(method.__globals__)
    for name in ('Problem_Settings','Problem_Wrapper','Policy_Network_Wrapper','Value_Network_Wrapper'):
        namespace[name]=getattr(cpp,name)
    return types.FunctionType(method.__code__,namespace,method.__name__,method.__defaults__,method.__closure__)

class InterceptionPUCT(DeploymentPUCT):
    _problem_wrapper_for=native_method(C_PUCT._problem_wrapper_for)
    create_cpp_policy_oracle=native_method(C_PUCT.create_cpp_policy_oracle)
    create_cpp_value_oracle=native_method(C_PUCT.create_cpp_value_oracle)

    def __init__(self,policies,value,budget,beta_policy,beta_value,variant='S',depth=100):
        self.calls=0;self.last_result=None;self.cost=dict(step_calls=0,policy_evaluations=0,value_evaluations=0)
        self.policy_oracle=self.create_cpp_policy_oracle(policies)
        self.value_oracle=self.create_cpp_value_oracle(value)
        self.vis_on=False
        self.solver_name={'B':'C_PUCT_V1','S':'C_PUCT_SYNC','OFF':'C_PUCT_INTERCEPT_OFF',
            'P':'C_PUCT_INTERCEPT_P','R':'C_PUCT_INTERCEPT_R','PR':'C_PUCT_INTERCEPT_PR'}[variant]
        ss=self.solver_settings=cpp.Solver_Settings()
        ss.number_simulations=int(budget);ss.search_depth=int(depth)
        ss.C_pw,ss.alpha_pw,ss.C_exp,ss.alpha_exp=2.,.5,1.,.25
        ss.beta_policy,ss.beta_value,ss.export_tree,ss.export_root=beta_policy,beta_value,False,True
        self.solver_wrapper=cpp.Solver_Wrapper(self.solver_name,ss,self.policy_oracle,self.value_oracle)

    def search(self,problem,root_state,turn=0):
        result=cpp.cpp_search(self._problem_wrapper_for(problem),self.solver_wrapper,root_state,turn)
        if not result.success or not np.isfinite(result.best_action).all():raise RuntimeError('Interception search failed')
        self.calls+=1;self.last_result=result
        for key in self.cost:self.cost[key]+=int(getattr(result,key))
        return result
