"""Mechanism isolation, information boundary and budget reliability checks."""
from pathlib import Path
import sys,unittest,time
from unittest.mock import patch
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import solvers.rollout_replan as revised
from solvers.frozen_d_search_interface import Controller as Original,Observation,geometric_candidates,belief_view,safe
from audit_frozen_d_search_interface import independent_traces
from run_rollout_replan_experiment import fixed_signature

BOXES=[[[-1.,1.],[-5.,5.]]]
FIRST=Observation(0,np.array([-3.,6.]),np.array([2.,6.]),80)
HIDDEN=Observation(1,np.array([-3.,5.]),None,79)


def actor(cache=True):
    c=revised.Controller(BOXES,1.1,'T+Geo-R',17,cache_enabled=cache)
    c.estimator.update(FIRST)
    return c


def arrays(cap=32):
    return (np.full((cap+1,64),-1,np.int32),np.zeros((cap+1,2)),np.zeros(cap+1,np.int64),np.zeros(cap+1))


class RolloutReplanTests(unittest.TestCase):
    def test_disabled_replan_exactly_matches_original_kernel(self):
        old=Original(BOXES,1.1,'T+Geo',17,budget_ms=200.)
        old.estimator.update(FIRST);a=old.act(HIDDEN,fixed_simulations=32)
        e,p=old.estimator,old.public
        _,constant=geometric_candidates(HIDDEN.position,e.particles,e.particle_weights,e.regions[e.ids],p.speed,p.vertices,p.dist,p.boxes,old.motion_boxes)
        new=actor()
        def original_continuation(at):
            action=constant.copy()
            if not safe(at,action,p.speed,old.motion_boxes,np.zeros(2,np.int64)):
                action[:]=0.
            return action,True
        with patch.object(revised,'rollout_action',side_effect=original_continuation):
            b=new.act(HIDDEN,fixed_simulations=32)
        self.assertEqual(fixed_signature(old,a),fixed_signature(new,b))

    def test_exact_cache_direct_equality_and_fixed_seed_replay(self):
        outputs=[]
        for cache in (True,False,True):
            c=actor(cache);a=c.act(HIDDEN,fixed_simulations=16)
            outputs.append(fixed_signature(c,a))
            errors=independent_traces(np.concatenate(c.traces),c.public.boxes,c.public.speed)
            np.testing.assert_array_equal(errors,np.zeros(4))
        self.assertEqual(outputs[0],outputs[1]);self.assertEqual(outputs[0],outputs[2])

    def ready(self):
        c=actor();c.estimator.update(HIDDEN)
        e,p=c.estimator,c.public
        roots,base=geometric_candidates(HIDDEN.position,e.particles,e.particle_weights,e.regions[e.ids],p.speed,p.vertices,p.dist,p.boxes,c.motion_boxes)
        return c,roots,base

    def simulate(self,c,roots,base,ids=None):
        e,p=c.estimator,c.public
        points,links,lengths,probabilities,*_=e.graph
        view=list(belief_view(e))
        if ids is not None:view[0]=ids
        storage=arrays();rng=np.array([12345],np.int64)
        result=revised.one_simulation(HIDDEN.position,p.speed,8,points,links,lengths,probabilities,*view,roots,base,p.boxes,c.motion_boxes,rng,*storage,1)
        return result,storage

    def test_geometry_does_not_read_simulated_hidden_target(self):
        c,roots,base=self.ready();e=c.estimator
        revised._active=revised.make_context(c,time.perf_counter(),True)
        first,_=self.simulate(c,roots,base,np.full(64,e.ids[0],np.int64))
        revised._active=revised.make_context(c,time.perf_counter(),True)
        second,_=self.simulate(c,roots,base,np.full(64,e.ids[-1],np.int64))
        n=min(len(first[1]),len(second[1]))
        self.assertGreater(n,0)
        np.testing.assert_array_equal(first[1][:n,:4],second[1][:n,:4])

    def test_partial_budget_abort_never_backpropagates(self):
        c,roots,base=self.ready()
        revised._active=revised.make_context(c,time.perf_counter(),True)
        count=[0]
        def permitted():count[0]+=1;return count[0]==1
        with patch.object(revised,'step_allowed',side_effect=permitted):
            result,storage=self.simulate(c,roots,base)
        self.assertEqual(result[3],2);self.assertEqual(len(result[1]),1)
        np.testing.assert_array_equal(storage[2],np.zeros_like(storage[2]))
        np.testing.assert_array_equal(storage[3],np.zeros_like(storage[3]))


if __name__=='__main__':unittest.main()
