import unittest
import numpy as np
from solvers.e45_team_adapter import TeamBelief,TeamController,optimal_assignment,public_observation,team_hidden_mask
from solvers.frozen_d_search_interface import sensor_visible
from run_e45_team_pilot import make_case,checkpoint
from corner_estimator_scenarios import original_model


class TeamTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.case=make_case('long_wall',0)

    def test_assignment_has_a_strong_nonredundant_baseline(self):
        cost=np.array([[1.,2.],[1.1,9.]])
        self.assertEqual(optimal_assignment(cost,[True,True],[True,True],True),{0:0,1:0})
        self.assertEqual(optimal_assignment(cost,[True,True],[True,True]),{0:1,1:0})

    def test_team_masks_and_public_replay(self):
        c=self.case;a=TeamBelief(c['map'],17);b=TeamBelief(c['map'],17)
        for obs in c['history']:
            a.update(obs);b.update(obs)
            self.assertEqual(a.signature(),b.signature())
        for e in a.estimators:
            self.assertTrue(team_hidden_mask(e.particles,np.array(obs['positions']),np.array(obs['active_p']),np.array(c['map'])).all())
            self.assertAlmostEqual(e.weights.sum(),1.)
        with self.assertRaises(ValueError):a.update(c['history'][-1])

    def test_simultaneous_sensor_does_not_reveal_hidden_targets(self):
        positions=np.array([[-4.,0.],[4.,0.]])
        targets=np.array([[-3.,0.],[3.,0.]])
        boxes=np.array([[[-1.,1.],[-5.,5.]]])
        o=public_observation(0,positions,targets,np.ones(2,bool),np.ones(2,bool),boxes)
        self.assertIsNone(o['reports'][0][1]);self.assertIsNone(o['reports'][1][0])
        self.assertEqual(o['reports'][0][0],[-3.,0.]);self.assertEqual(o['reports'][1][1],[3.,0.])

    def test_original_capture_retires_exactly_one_pair(self):
        p=original_model([],1.1);s=np.zeros((p.state_dim,1));s[p.active_idxs,0]=1
        s[:8,0]=[0.,0.,8.,8.,.5,0.,-8.,-8.]
        nxt=p.step(s,np.zeros((8,1)),1.)
        np.testing.assert_array_equal(nxt[p.active_idxs,0],[0,1,0,1])

    def test_factorized_nte_public_repeat_and_legal_joint_actions(self):
        c=self.case;answers=[]
        from run_frozen_d_closed_loop import legal_action
        for _ in range(2):
            own=TeamController(c['map'],c['speed'],17,'AFN17',checkpoint('AFN17'))
            for obs in c['history'][:-1]:own.observe(obs)
            a=own.act(c['history'][-1],fixed_simulations=4)
            for i in range(2):self.assertTrue(legal_action(np.array(c['history'][-1]['positions'][i]),a[i],c['speed'],np.array(c['map'])))
            self.assertFalse(own.last_stats['joint_action_search'])
            answers.append((a,own.belief.signature()))
        np.testing.assert_array_equal(answers[0][0],answers[1][0]);self.assertEqual(answers[0][1],answers[1][1])


if __name__=='__main__':unittest.main()
