"""Probability, information boundary and simulation checks for research gate."""
import sys
from pathlib import Path
import unittest
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from solvers.frozen_d_search_interface import (
    Controller, Observation, belief_view, sample_roots, loader, sensor_visible, safe)
from audit_frozen_d_search_interface import independent_traces, probability_audit


class WeightedInterfaceTests(unittest.TestCase):
    def controller(self,method='T+Geo'):
        c=Controller([[[-1.,1.],[-5.,5.]]],1.1,method,17)
        c.estimator.update(Observation(0,np.array([-3.,6.]),np.array([2.,6.]),80))
        return c

    def test_skewed_weights_do_not_become_uniform(self):
        weights=np.full(64,.1/63);weights[0]=.9
        modes=np.zeros((64,9));modes[:,3]=1.
        counts,joint=sample_roots(np.arange(64),weights,modes,200000,17)
        self.assertLess(abs(counts[0]/200000-.9),.004)
        self.assertEqual(int(joint[:,3].sum()),200000)
        self.assertEqual(int(joint.sum()),200000)

    def test_same_public_history_yields_same_posterior_and_fixed_replay(self):
        obs=Observation(1,np.array([-3.,5.]),None,79)
        signatures=[]
        for method in ('G','T','T+Geo'):
            c=self.controller(method);c.act(obs,fixed_simulations=16)
            signatures.append((c.estimator.particles,c.estimator.particle_weights,c.estimator.rng))
        for actual in signatures[1:]:
            for a,b in zip(actual,signatures[0]):np.testing.assert_array_equal(a,b)
        c=self.controller();action=c.act(obs,fixed_simulations=32)
        first=(action,c.rng.copy(),c.last_stats['root_counts'],c.last_stats['root_totals'])
        trace=np.concatenate(c.traces)
        np.testing.assert_array_equal(independent_traces(trace,c.public.boxes,1.1),np.zeros(4))
        self.assertEqual(sum(c.last_stats['terminations']),32)
        d=self.controller();other=d.act(obs,fixed_simulations=32)
        np.testing.assert_array_equal(first[0],other)
        np.testing.assert_array_equal(first[1],d.rng)
        self.assertEqual(first[2],d.last_stats['root_counts'])
        self.assertEqual(first[3],d.last_stats['root_totals'])

    def test_transition_factorization_and_negative_observation(self):
        c=self.controller('G');obs=Observation(1,np.array([-3.,5.]),None,79)
        c.act(obs)
        result=probability_audit(c.estimator,obs,17)
        for name in ('root_sampling_passed','joint_sampling_passed','transition_sampling_passed',
                     'filter_equivalent','frozen_probabilities_normalized'):
            self.assertTrue(result[name],(name,result))
        # This exact wall-contact fixture exposes D's conservative sensor guard.
        # Keep the discrepancy visible; it must not silently pass the exactness gate.
        self.assertFalse(result['sensor_consistent'])
        self.assertGreater(result['sensor_disagreement_mass'],0.)
        self.assertEqual(result['illegal_graph_edges'],0)
        self.assertLess(result['filter_max_error'],1e-12)

    def test_invalid_importance_weights_rejected_and_snapshot_immutable(self):
        c=self.controller('G');c.act(Observation(1,np.array([-3.,5.]),None,79))
        before=loader.verify()
        c.estimator.particle_weights[0]=-.1
        with self.assertRaises(ValueError):belief_view(c.estimator)
        self.assertEqual(before,loader.verify())
        # Controller has no hidden state argument, or environment reference.
        with self.assertRaises(TypeError):c.act(Observation(2,np.array([-3.,4.]),None,78),hidden_state=[2.,4.])

    def test_exact_corner_contact_regressions(self):
        boxes=np.array([[[-6.483153737652282,6.660551698936242],[-1.035961503016004,.964038496983996]]])
        at=np.array([7.860551698936242,.16403849698399586])
        action=np.array([-1.3,-1.3])
        guarded=boxes.copy();guarded[:,:,0]-=1e-6;guarded[:,:,1]+=1e-6
        self.assertFalse(safe(at,action,1.3,guarded,np.zeros(2,np.int64)))
        boxes=np.array([[[-6.39312374454356,6.198344433011194],[-.9543122981213618,1.0456877018786381]]])
        at=np.array([-9.393123744543558,.5456877018786388])
        target=np.array([-3.99312374454356,1.4456877018786383])
        self.assertFalse(sensor_visible(at,target,boxes))


if __name__=='__main__':unittest.main()
