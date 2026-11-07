"""Private MCTS adapter with solver-local completion utility and diagnostics."""
import types
import numpy as np
from solvers.c_puct import C_PUCT
from cpp.build_completion import bindings_completion as cpp


def private_method(method):
    namespace = dict(method.__globals__)
    for name in ('Problem_Settings', 'Problem_Wrapper', 'Policy_Network_Wrapper', 'Value_Network_Wrapper'):
        namespace[name] = getattr(cpp, name)
    return types.FunctionType(method.__code__, namespace, method.__name__, method.__defaults__, method.__closure__)


class CompletionPUCT(C_PUCT):
    _problem_wrapper_for = private_method(C_PUCT._problem_wrapper_for)
    create_cpp_policy_oracle = private_method(C_PUCT.create_cpp_policy_oracle)
    create_cpp_value_oracle = private_method(C_PUCT.create_cpp_value_oracle)

    def __init__(self, policy_oracle, value_oracle, budget=1000, beta_value=0., completion_bonus=0.):
        self.policy_oracle = self.create_cpp_policy_oracle(policy_oracle)
        self.value_oracle = self.create_cpp_value_oracle(value_oracle)
        self.vis_on = False
        self.solver_name = 'C_PUCT_V1'
        self.solver_settings = cpp.Solver_Settings()
        settings = self.solver_settings
        settings.number_simulations, settings.search_depth = int(budget), 100
        settings.C_pw, settings.alpha_pw, settings.C_exp, settings.alpha_exp = 2., .5, 1., .25
        settings.beta_policy, settings.beta_value = .5, float(beta_value)
        settings.export_tree = False
        settings.completion_bonus = float(completion_bonus)
        self.solver_wrapper = cpp.Solver_Wrapper(self.solver_name, settings, self.policy_oracle, self.value_oracle)
        self.diagnostics = []

    def search(self, problem, root_state, turn=0):
        result = cpp.cpp_search(self._problem_wrapper_for(problem), self.solver_wrapper, root_state, turn)
        if not result.success or not np.isfinite(result.best_action).all():
            raise RuntimeError('Private MCTS failed to produce a finite executable action')
        self.diagnostics.append(dict(before_first=bool(problem.active_evader_count(root_state) == 2),
            tree_events=int(result.completion_tree_events), rollout_events=int(result.completion_rollout_events),
            simulations=int(result.num_visits), turn=int(turn)))
        return result
