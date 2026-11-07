"""One executable root-action override after the first real legal capture."""
import copy
import numpy as np


def select_aq(table, problem, state, team=(2, 3)):
    """Match the frozen root audit's AQ, including native-order deduplication."""
    table = np.asarray(table, dtype=np.float32)
    if table.ndim != 2 or table.shape[1] != 6 + problem.action_dim or not len(table):
        raise ValueError('Malformed root table')
    if not np.isfinite(table).all() or np.any(table[:, 1] <= 0):
        raise ValueError('Nonfinite or unvisited root entry')
    if not np.array_equal(table[:, 0], np.arange(len(table))):
        raise ValueError('Root traversal order changed')
    if not np.all(np.isin(table[:, 5], [0., 1.])) or table[:, 5].sum() != 1:
        raise ValueError('Expected exactly one original selected action')
    q = table[:, 2].astype(float) + float(np.float32(problem.gamma)) * table[:, 3].astype(float) / table[:, 1]
    if np.any(np.abs(q - table[:, 4]) > 1e-6 * np.maximum(1., np.abs(q))):
        raise ValueError('Original Q formula changed')
    actions = table[:, 6:].copy()
    for robot in range(problem.num_robots):
        if robot not in team or not problem.is_active(state, robot):
            actions[:, problem.action_idxs[robot]] = 0.
    keys = [tuple(float(x) for x in row) for row in actions]
    a0 = int(np.flatnonzero(table[:, 5])[0])
    seen, unique = {keys[a0]}, []
    for i, key in enumerate(keys):
        if key not in seen:
            seen.add(key)
            unique.append(i)
    chosen = max(unique, key=lambda i: (float(table[i, 4]), -i)) if unique else a0
    event = dict(a0_ordinal=a0, aq_ordinal=chosen, fallback=chosen == a0,
        a0_q=float(table[a0, 4]), aq_q=float(table[chosen, 4]),
        a0_visits=int(table[a0, 1]), aq_visits=int(table[chosen, 1]),
        a0_action=actions[a0].tolist(), aq_action=actions[chosen].tolist(), distinct_actions=1 + len(unique))
    return actions[chosen].astype(float).reshape(-1, 1), event


class AQOnceSolver:
    def __init__(self, solver, team, enabled):
        self.solver, self.team, self.enabled = solver, tuple(team), bool(enabled)
        self.consumed, self.switches, self.last_switched = False, 0, False
        self.events = []

    def policy(self, problem, state):
        self.last_switched = False
        if problem.is_terminal(state):
            raise ValueError('No decision is allowed after termination')
        action = self.solver.policy(problem, state, team=self.team)
        if self.enabled and not self.consumed and problem.active_evader_count(state) == 1:
            selected, event = select_aq(self.solver.last_result.root_audit, problem, state, self.team)
            if not np.array_equal(action.reshape(-1), event['a0_action']):
                raise ValueError('Root selected joint action differs from executable A0')
            event.update(time=float(state[problem.time_idx, 0]), state=np.asarray(state).tolist())
            self.consumed, self.last_switched, self.switches = True, True, 1
            self.events.append(event)
            return selected
        return action

    def get_rng_state(self, problem):
        return dict(enabled=self.enabled, consumed=self.consumed, switches=self.switches,
            events=copy.deepcopy(self.events), rng=self.solver.get_rng_state(problem))

    def set_rng_state(self, problem, record):
        if record['enabled'] != self.enabled or record['switches'] != int(record['consumed']) or \
                len(record['events']) != record['switches'] or (record['consumed'] and not self.enabled):
            raise ValueError('Incompatible AQ snapshot')
        before = self.solver.get_rng_state(problem)
        try:
            self.solver.set_rng_state(problem, record['rng'])
        except BaseException:
            self.solver.set_rng_state(problem, before)
            raise
        self.consumed, self.switches = record['consumed'], record['switches']
        self.events = copy.deepcopy(record['events'])
        self.last_switched = False
