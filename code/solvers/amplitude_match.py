"""M/F/S in a private extension; only M omits F's action-mean rescaling."""
import types
from solvers.candidate_ablation import CandidateAblationPUCT
from solvers.c_puct import C_PUCT
from cpp.build_amplitude_match import bindings_amplitude_match as cpp

def native(method):
    scope=dict(method.__globals__)
    for name in ('Problem_Settings','Problem_Wrapper','Policy_Network_Wrapper','Value_Network_Wrapper'):
        scope[name]=getattr(cpp,name)
    scope['cpp']=cpp
    return types.FunctionType(method.__code__,scope,method.__name__,method.__defaults__,method.__closure__)

class AmplitudeMatchPUCT(CandidateAblationPUCT):
    _problem_wrapper_for=native(C_PUCT._problem_wrapper_for)
    create_cpp_policy_oracle=native(C_PUCT.create_cpp_policy_oracle)
    create_cpp_value_oracle=native(C_PUCT.create_cpp_value_oracle)
    search=native(CandidateAblationPUCT.search)

    def __init__(self,policies,value,budget,beta_policy,beta_value,variant='M',depth=100):
        self.calls=0;self.last_result=None
        self.cost=dict(step_calls=0,policy_evaluations=0,value_evaluations=0)
        self.policy_oracle=self.create_cpp_policy_oracle(policies);self.value_oracle=self.create_cpp_value_oracle(value)
        self.vis_on=False
        self.solver_name={'M':'C_PUCT_AMPLITUDE_M','F':'C_PUCT_ABLATION_F','S':'C_PUCT_SYNC'}[variant]
        ss=self.solver_settings=cpp.Solver_Settings()
        ss.number_simulations=int(budget);ss.search_depth=int(depth)
        ss.C_pw,ss.alpha_pw,ss.C_exp,ss.alpha_exp=2.,.5,1.,.25
        ss.beta_policy,ss.beta_value,ss.export_tree,ss.export_root=beta_policy,beta_value,False,True
        self.solver_wrapper=cpp.Solver_Wrapper(self.solver_name,ss,self.policy_oracle,self.value_oracle)
