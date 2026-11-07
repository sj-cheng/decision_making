"""Exact execution optimization checks; no change of rollout policy."""
from pathlib import Path
import json,sys,unittest
from unittest.mock import patch
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import solvers.rollout_replan as slow
import solvers.rollout_geometry_exact as fast
from solvers.frozen_d_search_interface import geometric_candidates,Observation
from run_rollout_replan_experiment import SOURCE,development_entries,observation,fixed_signature
from estimator_audit_geometry import point_legal


class ExactGeometryTests(unittest.TestCase):
    def test_candidate_arrays_and_action_bits_equal_on_maps_and_positions(self):
        rng=np.random.default_rng(83420261026)
        entries=development_entries()
        checked=0
        for index in (0,1,4,5,8,9):
            entry=entries[index];packet=json.loads((SOURCE/'bank'/entry['public']).read_text())
            c=slow.Controller(packet['map'],packet['motion']['pursuer_axis_limit'],'T+Geo-R',17)
            for item in packet['history']:c.estimator.update(observation(item))
            e,p=c.estimator,c.public;labels=e.regions[e.ids]
            prepared=fast.prepare_geometry(e.particles,e.particle_weights,labels,p.vertices,p.dist,p.boxes,p.speed)
            positions=[np.asarray(packet['history'][-1]['position'])]
            while len(positions)<33:
                at=rng.uniform(-9.9,9.9,2)
                if point_legal(at,c.motion_boxes):positions.append(at)
            for at in positions:
                args=(at,e.particles,e.particle_weights,labels,p.speed,p.vertices,p.dist,p.boxes,c.motion_boxes)
                old=geometric_candidates(*args);new=fast.geometric_candidates_exact(*args,*prepared)
                self.assertEqual(old[0].tobytes(),new[0].tobytes())
                self.assertEqual(old[1].tobytes(),new[1].tobytes())
                checked+=1
        self.assertEqual(checked,198)

    def test_complete_search_rng_visits_returns_and_traces_exact(self):
        first=Observation(0,np.array([-3.,6.]),np.array([2.,6.]),80)
        hidden=Observation(1,np.array([-3.,5.]),None,79)
        outputs=[]
        for accelerated in (False,True):
            c=slow.Controller([[[-1.,1.],[-5.,5.]]],1.1,'T+Geo-R',17)
            c.estimator.update(first)
            if accelerated:
                with patch.object(slow,'make_context',fast.make_context),patch.object(slow,'rollout_action',fast.rollout_action):
                    action=c.act(hidden,fixed_simulations=64)
            else:action=c.act(hidden,fixed_simulations=64)
            outputs.append(fixed_signature(c,action))
        self.assertEqual(outputs[0],outputs[1])


if __name__=='__main__':unittest.main()
