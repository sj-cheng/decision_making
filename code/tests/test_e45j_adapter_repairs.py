import unittest
from types import SimpleNamespace
import numpy as np
from learning.e45j_adapter_repairs import (canonical_actor_inputs,representative,
    progress_move,route_apf_action)
from learning.e45i_official_mappo import apf_action
from solvers.frozen_d_search_interface import PublicMap,move
from learning.e45g_ppo import execution_legal_action


class AdapterRepairTests(unittest.TestCase):
    def own(self):
        q=np.array([4.,0.]);estimator=SimpleNamespace(particles=np.repeat(q[None],64,axis=0),particle_weights=np.full(64,1/64))
        return SimpleNamespace(public=PublicMap([[[-1.,1.],[-5.,5.]]],1.1),belief=SimpleNamespace(
            known=[q,None],estimators=[estimator,None],last_stats={}),last_stats={})

    def test_old_route_zero_hop_reproduced_and_candidate_leaves_vertex(self):
        own=self.own();at=np.array([-1.05,-5.05]);q=np.array([4.,0.]);p=own.public
        np.testing.assert_array_equal(move(at,q,p.speed,p.vertices,p.dist,p.boxes,np.zeros(2,np.int64)),[0.,0.])
        a=progress_move(p,at,q)
        self.assertGreater(np.max(np.abs(a)),.1)
        self.assertTrue(execution_legal_action(at,a,p.speed,p.boxes))

    def test_positive_gains_cannot_escape_old_apf_strip_route_candidate_can_capture(self):
        gains=[1.,1.5,.2,1.2]
        for corrected in (False,True):
            own=self.own();positions=np.array([[-3.,.2],[-8.,-8.]]);caught=False;maxy=0.
            for _ in range(100):
                obs=dict(positions=positions.tolist(),active_p=[True,False],active_e=[True,False])
                action=(route_apf_action if corrected else apf_action)(own,obs,coefficients=gains)
                self.assertTrue(execution_legal_action(positions[0],action[0],own.public.speed,own.public.boxes))
                positions+=action;maxy=max(maxy,abs(positions[0,1]))
                if np.linalg.norm(positions[0]-[4.,0.])<1: caught=True;break
            self.assertEqual(caught,corrected)
            if not corrected:self.assertLessEqual(maxy,1.1+1e-10)

    def test_representative_is_support_point_when_mean_inside_wall(self):
        estimator=SimpleNamespace(particles=np.array([[-3.,0.],[3.,0.]]),particle_weights=np.array([.5,.5]))
        q=representative(estimator,None)
        self.assertIn(tuple(q),[(-3.,0.),(3.,0.)])
        swapped=SimpleNamespace(particles=estimator.particles[::-1],particle_weights=estimator.particle_weights[::-1])
        np.testing.assert_array_equal(q,representative(swapped,None))

    def test_canonical_particle_adapter_is_exactly_permutation_invariant(self):
        rng=np.random.default_rng(17)
        x=dict(maps=np.zeros((8,4)),map_mask=np.zeros(8),agents=np.zeros((2,6)),
            particles=rng.normal(size=(2,64,3)),weights=rng.uniform(size=(2,64)),
            history=np.zeros((16,14)),history_mask=np.zeros(16),public=np.zeros(8))
        baseline=canonical_actor_inputs(x);copy={k:v.copy() for k,v in x.items()}
        for j in range(2):
            order=rng.permutation(64);copy['particles'][j]=copy['particles'][j,order];copy['weights'][j]=copy['weights'][j,order]
        np.testing.assert_array_equal(baseline,canonical_actor_inputs(copy))


if __name__=='__main__':unittest.main()
