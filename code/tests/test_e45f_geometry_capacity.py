import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
from solvers.frozen_d_search_interface import PublicMap,geometric_candidates
from solvers.e45f_geometry_capacity import geometric_candidates_capacity
from e45c_belief_diagnostics import RecordedTeamBelief,install

ROOT=Path(__file__).resolve().parents[2]
FOLDER=ROOT/'current/e45f_generalization_benchmark_20261029'


class GeometryCapacityTests(unittest.TestCase):
    def compare_case(self,case,seed,old_must_fail=False):
        with tempfile.TemporaryDirectory() as evidence:
            install(evidence);belief=RecordedTeamBelief(case['map'],seed)
            for observation in case['history']:belief.update(observation)
            public=PublicMap(case['map'],case['speed']);motion=public.boxes.copy()
            motion[:,:,0]-=1e-6;motion[:,:,1]+=1e-6
            failures=0
            for at in case['history'][-1]['positions']:
                for estimator in belief.estimators:
                    args=(np.array(at),estimator.particles,estimator.particle_weights,estimator.regions[estimator.ids],
                        case['speed'],public.vertices,public.dist,public.boxes,motion)
                    new=geometric_candidates_capacity(*args)
                    self.assertTrue(np.isfinite(new[0]).all())
                    try:old=geometric_candidates(*args)
                    except RuntimeError as exc:
                        self.assertIn('Candidate capacity exceeded',str(exc));failures+=1
                    else:
                        np.testing.assert_array_equal(old[0],new[0]);np.testing.assert_array_equal(old[1],new[1])
            if old_must_fail:self.assertGreater(failures,0)
            else:self.assertEqual(failures,0)

    def test_original_maps_have_exactly_identical_candidates_and_selected_action(self):
        manifest=json.loads((ROOT/'current/e45c_budget_scaling_20261028/bank/manifest.json').read_text())
        for family in ('long_wall','wall_corner','multi_route'):
            for entry in [e for e in manifest['entries'] if e['split']=='validation' and e['family']==family][:2]:
                self.compare_case(json.loads(Path(entry['path']).read_text()),317)

    def test_replays_recorded_dense_capacity_failure_and_returns_all_candidates(self):
        repair=FOLDER/'repair_capacity_v1/original_public_failure.json'
        proof=json.loads(repair.read_text())
        manifest=json.loads((FOLDER/'bank/manifest.json').read_text())
        case=next(json.loads(Path(e['path']).read_text()) for e in manifest['validation']
                  if json.loads(Path(e['path']).read_text())['map']==proof['boxes'])
        self.compare_case(case,proof['seed'],old_must_fail=True)


if __name__=='__main__':unittest.main()
