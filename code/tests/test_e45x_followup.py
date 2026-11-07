import numpy as np
from test_e45x_event_diagnostic import tiny_tree
from run_e45x_event_followup import Rollout16
from learning.e45x_event_diagnostic import ETA,TIME_COST


def test_true_rollout_terminal_reward_is_not_repeated():
    c=tiny_tree([[.5,0],[3.5,0]],[[0,0],[3,0]])
    c.__class__=Rollout16;c.trajectory()
    assert len(c.traces[0])==1
    assert np.isclose(c.root.totals[0],1-TIME_COST-ETA*(-4.))


def test_rollout_reaches16_after_fresh_node_without_polluting_search_tree():
    c=tiny_tree([[.5,0],[-8,-8]],[[0,0],[9,9]])
    c.__class__=Rollout16
    c.propose=lambda node,direct=False:np.zeros((2,2))
    c.trajectory()
    assert len(c.traces[0])==16
    assert c.root.visits==[1]
    assert np.isclose(c.root.totals[0],-16*TIME_COST+ETA*(-3.-(-4.)))
    assert all(not child.children for child in c.root.children.values())
