"""Isolated legacy and simultaneous searches, with RNG and cost accounting."""
import types
import numpy as np
from solvers.c_puct import C_PUCT
from cpp.build_deployment import bindings_deployment as cpp


def private_method(method):
    namespace=dict(method.__globals__)
    for name in ('Problem_Settings','Problem_Wrapper','Policy_Network_Wrapper','Value_Network_Wrapper'):
        namespace[name]=getattr(cpp,name)
    return types.FunctionType(method.__code__,namespace,method.__name__,method.__defaults__,method.__closure__)


class DeploymentPUCT(C_PUCT):
    _problem_wrapper_for=private_method(C_PUCT._problem_wrapper_for)
    create_cpp_policy_oracle=private_method(C_PUCT.create_cpp_policy_oracle)
    create_cpp_value_oracle=private_method(C_PUCT.create_cpp_value_oracle)

    def __init__(self,policies,value,budget,beta_policy,beta_value,synchronous=False,depth=100):
        self.calls=0;self.last_result=None
        self.cost=dict(step_calls=0,policy_evaluations=0,value_evaluations=0)
        self.policy_oracle=self.create_cpp_policy_oracle(policies)
        self.value_oracle=self.create_cpp_value_oracle(value)
        self.vis_on=False;self.solver_name='C_PUCT_SYNC' if synchronous else 'C_PUCT_V1'
        ss=self.solver_settings=cpp.Solver_Settings()
        ss.number_simulations=int(budget);ss.search_depth=int(depth)
        ss.C_pw,ss.alpha_pw,ss.C_exp,ss.alpha_exp=2.,.5,1.,.25
        ss.beta_policy,ss.beta_value,ss.export_tree,ss.export_root=beta_policy,beta_value,False,True
        self.solver_wrapper=cpp.Solver_Wrapper(self.solver_name,ss,self.policy_oracle,self.value_oracle)

    def search(self,problem,root_state,turn=0):
        result=cpp.cpp_search(self._problem_wrapper_for(problem),self.solver_wrapper,root_state,turn)
        if not result.success or not np.isfinite(result.best_action).all():
            raise RuntimeError('Deployment search failed')
        self.calls+=1;self.last_result=result
        for key in self.cost:self.cost[key]+=int(getattr(result,key))
        return result

    def get_rng_state(self,problem):
        return dict(solver=self.solver_wrapper.get_rng_state(),problem=self._problem_wrapper_for(problem).get_rng_state())

    def set_rng_state(self,problem,record):
        self.solver_wrapper.set_rng_state(record['solver'])
        self._problem_wrapper_for(problem).set_rng_state(record['problem'])
