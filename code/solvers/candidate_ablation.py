"""Private, frozen-compatible interception solver with root F/D/G controls."""
import types
from solvers.interception_puct import InterceptionPUCT
from solvers.c_puct import C_PUCT
from cpp.build_candidate_ablation import bindings_candidate_ablation as cpp

def native(method):
    scope=dict(method.__globals__)
    for name in ('Problem_Settings','Problem_Wrapper','Policy_Network_Wrapper','Value_Network_Wrapper'):
        scope[name]=getattr(cpp,name)
    scope['cpp']=cpp
    return types.FunctionType(method.__code__,scope,method.__name__,method.__defaults__,method.__closure__)

class CandidateAblationPUCT(InterceptionPUCT):
    _problem_wrapper_for=native(C_PUCT._problem_wrapper_for)
    create_cpp_policy_oracle=native(C_PUCT.create_cpp_policy_oracle)
    create_cpp_value_oracle=native(C_PUCT.create_cpp_value_oracle)
    search=native(InterceptionPUCT.search)

    def __init__(self,policies,value,budget,beta_policy,beta_value,variant='S',depth=100):
        self.calls=0;self.last_result=None
        self.cost=dict(step_calls=0,policy_evaluations=0,value_evaluations=0)
        self.policy_oracle=self.create_cpp_policy_oracle(policies);self.value_oracle=self.create_cpp_value_oracle(value)
        self.vis_on=False
        self.solver_name={'S':'C_PUCT_SYNC','P':'C_PUCT_INTERCEPT_P','PR':'C_PUCT_INTERCEPT_PR',
            'F':'C_PUCT_ABLATION_F','D':'C_PUCT_ABLATION_D','G':'C_PUCT_ABLATION_G'}[variant]
        ss=self.solver_settings=cpp.Solver_Settings()
        ss.number_simulations=int(budget);ss.search_depth=int(depth)
        ss.C_pw,ss.alpha_pw,ss.C_exp,ss.alpha_exp=2.,.5,1.,.25
        ss.beta_policy,ss.beta_value,ss.export_tree,ss.export_root=beta_policy,beta_value,False,True
        self.solver_wrapper=cpp.Solver_Wrapper(self.solver_name,ss,self.policy_oracle,self.value_oracle)
