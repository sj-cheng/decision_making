import unittest
import tempfile
from unittest.mock import patch
from pathlib import Path
import numpy as np
from e45b_sensor_audit import exact_visible,visible_batch
from e45b_scenarios import audit_simulated,audit_simulated_legacy
from solvers.frozen_d_search_interface import sensor_visible


class SensorAuditTests(unittest.TestCase):
    def test_actual_failure_preserves_raw_simulation_evidence(self):
        import run_e45b_role_free as runner
        from e45b_scenarios import make_case
        case=make_case('long_wall',0,'validation',945220261027)
        with tempfile.TemporaryDirectory() as directory:
            output=Path(directory);runner.save(output/'case.json',case)
            task=dict(case=str(output/'case.json'),seed=17,method='U',model_round=0,
                opponent='fixed_D_kernel',world='AB',limit=1,fixed=2,round=1,
                key='fault_injection',budget_ms=200.,stage='engineering')
            with patch.object(runner,'OUT',output),patch.object(runner,'audit_simulated',return_value=np.array([0,0,0,1])):
                with self.assertRaisesRegex(RuntimeError,'evidence='):runner.episode(task)
            self.assertEqual(len(list((output/'audit_failures').glob('*.json'))),1)
            evidence=list((output/'audit_failures').glob('*.npz'));self.assertEqual(len(evidence),1)
            with np.load(evidence[0]) as data:
                self.assertEqual(data['trace'].shape[1],26)
                np.testing.assert_array_equal(data['errors'],[0,0,0,1])

    def test_actual_map_corner_regressions_certified_exactly(self):
        with np.load(Path(__file__).parent/'fixtures/e45b_sensor_boundary_v1.npz') as data:
            pairs=data['pairs'];boxes=data['boxes']
        frozen=tuple(tuple(tuple(axis) for axis in box) for box in boxes)
        actual=visible_batch(pairs[:,:2],pairs[:,2:4],boxes)
        exact=[exact_visible(tuple(p[:2]),tuple(p[2:4]),frozen) for p in pairs]
        production=[sensor_visible(p[:2],p[2:4],boxes) for p in pairs]
        np.testing.assert_array_equal(actual,exact)
        np.testing.assert_array_equal(actual,production)
        self.assertTrue(actual.all())
        # A standstill public simulation with one active observer and target.
        for pair in pairs:
            row=np.zeros(26);row[:2]=row[4:6]=pair[:2];row[2:4]=row[6:8]=pair[:2]
            row[8:10]=row[12:14]=pair[2:4];row[10:12]=row[14:16]=pair[2:4]
            row[[16,18,20,22,24]]=1
            self.assertEqual(audit_simulated_legacy(row[None],boxes,1.3)[3],1)
            np.testing.assert_array_equal(audit_simulated(row[None],boxes,1.3),[0,0,0,0])
            row[24]=0
            self.assertEqual(audit_simulated(row[None],boxes,1.3)[3],1)

    def test_blocked_clear_contact_range_and_random_pairs(self):
        boxes=np.array([[[0.,1.],[0.,1.]]]);starts=np.array([[-2.,.5],[-2.,2.],[-2.,0.],[0.,2.],[0.,0.]])
        ends=np.array([[2.,.5],[2.,2.],[2.,0.],[0.,3.],[7.,0.]])
        np.testing.assert_array_equal(visible_batch(starts,ends,boxes),[False,True,False,True,False])
        rng=np.random.default_rng(45);starts=rng.uniform(-5,5,(5000,2));ends=rng.uniform(-5,5,(5000,2))
        np.testing.assert_array_equal(visible_batch(starts,ends,boxes),[sensor_visible(a,q,boxes) for a,q in zip(starts,ends)])
        frozen=(((0.,1.),(0.,1.)),)
        np.testing.assert_array_equal(visible_batch(starts[:250],ends[:250],boxes),[exact_visible(tuple(a),tuple(q),frozen) for a,q in zip(starts[:250],ends[:250])])


if __name__=='__main__':unittest.main()
