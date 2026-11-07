"""Original MCTS with optional, non-consuming root diagnostics."""
import types
import numpy as np
from solvers.c_puct import C_PUCT
from cpp.build_root_audit import bindings_root_audit as cpp


def private_method(method):
    namespace = dict(method.__globals__)
    for name in ('Problem_Settings', 'Problem_Wrapper', 'Policy_Network_Wrapper', 'Value_Network_Wrapper'):
        namespace[name] = getattr(cpp, name)
    return types.FunctionType(method.__code__, namespace, method.__name__, method.__defaults__, method.__closure__)


class RootPUCT(C_PUCT):
    _problem_wrapper_for = private_method(C_PUCT._problem_wrapper_for)
    create_cpp_policy_oracle = private_method(C_PUCT.create_cpp_policy_oracle)
    create_cpp_value_oracle = private_method(C_PUCT.create_cpp_value_oracle)

    def __init__(self, policies, value, budget, beta_policy, beta_value, export_root=False):
        self.calls, self.last_result = 0, None
        self.policy_oracle = self.create_cpp_policy_oracle(policies)
        self.value_oracle = self.create_cpp_value_oracle(value)
        self.vis_on, self.solver_name = False, 'C_PUCT_V1'
        ss = self.solver_settings = cpp.Solver_Settings()
        ss.number_simulations, ss.search_depth = int(budget), 100
        ss.C_pw, ss.alpha_pw, ss.C_exp, ss.alpha_exp = 2., .5, 1., .25
        ss.beta_policy, ss.beta_value, ss.export_tree = beta_policy, beta_value, False
        ss.export_root = bool(export_root)
        self.solver_wrapper = cpp.Solver_Wrapper(self.solver_name, ss, self.policy_oracle, self.value_oracle)

    def search(self, problem, root_state, turn=0):
        result = cpp.cpp_search(self._problem_wrapper_for(problem), self.solver_wrapper, root_state, turn)
        if not result.success or not np.isfinite(result.best_action).all():
            raise RuntimeError('Original private search failed')
        self.calls += 1
        self.last_result = result
        return result

    def get_rng_state(self, problem):
        return dict(solver=self.solver_wrapper.get_rng_state(), problem=self._problem_wrapper_for(problem).get_rng_state())

    def set_rng_state(self, problem, state):
        self.solver_wrapper.set_rng_state(state['solver'])
        self._problem_wrapper_for(problem).set_rng_state(state['problem'])

    def seeded_state(self, problem, solver_seed, problem_seed):
        self.solver_wrapper.set_seed(int(solver_seed))
        record = self.get_rng_state(problem)
        # default_random_engine is fingerprinted; stream parsing accepts its
        # seeded state, and leaves the saved uniform distribution unchanged.
        auxiliary = int(problem_seed) % 2147483647 or 1
        distribution = record['problem'].split('\n', 1)[1]
        self._problem_wrapper_for(problem).set_rng_state(str(auxiliary)+'\n'+distribution)
        return self.get_rng_state(problem)
