"""Original search with continuous RNG streams across a real-root phase switch."""
import types
import numpy as np
from solvers.c_puct import C_PUCT
from cpp.build_phase import bindings_phase as cpp


def private_method(method):
    namespace = dict(method.__globals__)
    for name in ('Problem_Settings', 'Problem_Wrapper', 'Policy_Network_Wrapper', 'Value_Network_Wrapper'):
        namespace[name] = getattr(cpp, name)
    return types.FunctionType(method.__code__, namespace, method.__name__, method.__defaults__, method.__closure__)


class PrivatePUCT(C_PUCT):
    _problem_wrapper_for = private_method(C_PUCT._problem_wrapper_for)
    create_cpp_policy_oracle = private_method(C_PUCT.create_cpp_policy_oracle)
    create_cpp_value_oracle = private_method(C_PUCT.create_cpp_value_oracle)

    def __init__(self, policies, value, budget, beta_policy, beta_value):
        self.policy_oracle = self.create_cpp_policy_oracle(policies)
        self.value_oracle = self.create_cpp_value_oracle(value)
        self.vis_on, self.solver_name = False, 'C_PUCT_V1'
        ss = self.solver_settings = cpp.Solver_Settings()
        ss.number_simulations, ss.search_depth = int(budget), 100
        ss.C_pw, ss.alpha_pw, ss.C_exp, ss.alpha_exp = 2., .5, 1., .25
        ss.beta_policy, ss.beta_value, ss.export_tree = beta_policy, beta_value, False
        self.solver_wrapper = cpp.Solver_Wrapper(self.solver_name, ss, self.policy_oracle, self.value_oracle)

    def search(self, problem, root_state, turn=0):
        result = cpp.cpp_search(self._problem_wrapper_for(problem), self.solver_wrapper, root_state, turn)
        if not result.success or not np.isfinite(result.best_action).all():
            raise RuntimeError('Original private search failed')
        return result

    def get_rng_state(self, problem):
        return dict(solver=self.solver_wrapper.get_rng_state(), problem=self._problem_wrapper_for(problem).get_rng_state())

    def set_rng_state(self, problem, state):
        self.solver_wrapper.set_rng_state(state['solver'])
        self._problem_wrapper_for(problem).set_rng_state(state['problem'])


class PhaseSolver:
    """Each complete search uses one model; no phase gate inside the tree."""
    def __init__(self, solvers, before, after, team):
        self.solvers, self.before, self.after, self.team = solvers, before, after, tuple(team)
        self.active = before
        self.switches = 0
        self.last_switched = False

    def policy(self, problem, state):
        self.last_switched = False
        if problem.active_evader_count(state) == 1 and self.active != self.after:
            rng = self.solvers[self.active].get_rng_state(problem)
            self.solvers[self.after].set_rng_state(problem, rng)
            self.active, self.last_switched = self.after, True
            self.switches += 1
        return self.solvers[self.active].policy(problem, state, team=self.team)

    def get_rng_state(self, problem):
        return dict(active=self.active, rng=self.solvers[self.active].get_rng_state(problem), switches=self.switches)

    def set_rng_state(self, problem, record):
        if record['active'] not in self.solvers:
            raise ValueError('Unknown restored phase')
        self.active, self.switches = record['active'], record['switches']
        self.solvers[self.active].set_rng_state(problem, record['rng'])
