import unittest
from pathlib import Path
import numpy as np
from e45c_capture_audit import audit_simulated,capture_errors,environment_pairing
from e45b_scenarios import audit_simulated_legacy
from corner_estimator_scenarios import original_model


class CaptureAuditTests(unittest.TestCase):
    def test_real_failure_is_norm_tie_matching_original_environment(self):
        with np.load(Path(__file__).parent/'fixtures/e45c_capture_pairing_v1.npz') as data:
            trace=data['trace'];boxes=data['boxes'];speed=float(data['speed'])
        row=trace[0];p=original_model(boxes,speed);s=np.zeros((p.state_dim,1))
        s[:4,0]=row[12:16];s[4:8,0]=row[4:8];s[p.active_idxs,0]=1
        self.assertEqual(p.get_capture_pairs(s),[(2,0)])
        next_state=p.step(s,np.zeros((8,1)),1.)
        np.testing.assert_array_equal(next_state[p.active_idxs[:2],0],row[22:24])
        np.testing.assert_array_equal(next_state[p.active_idxs[2:],0],row[20:22])
        self.assertEqual(audit_simulated_legacy(trace,boxes,speed)[2],1)
        np.testing.assert_array_equal(audit_simulated(trace,boxes,speed),[0,0,0,0])
        wrong=trace.copy();wrong[0,22:24]=[1,0]
        self.assertEqual(capture_errors(wrong),1)
        wrong=trace.copy();wrong[0,20:22]=[1,0]
        self.assertEqual(capture_errors(wrong),1)

    def test_random_and_near_radius_states_match_original_pairing(self):
        rng=np.random.default_rng(945420261028);p=original_model([],1.1);rows=[]
        for n in range(1000):
            pp=rng.uniform(-1.5,1.5,(2,2));ee=rng.uniform(-1.5,1.5,(2,2));ap=rng.random(2)>.2;ae=rng.random(2)>.2
            if n<100:
                pp[0]=0.;ee[0]=[.6,np.nextafter(.8,0.) if n%3==0 else .8 if n%3==1 else np.nextafter(.8,1.)]
                ee[1]=ee[0]+[0.,np.spacing(.8)*(n%7-3)]
            s=np.zeros((p.state_dim,1));s[:4,0]=ee.ravel();s[4:8,0]=pp.ravel();s[p.active_idxs[:2],0]=ae;s[p.active_idxs[2:],0]=ap
            expected_p=ap.copy();expected_e=ae.copy()
            for i,j in p.get_capture_pairs(s):expected_p[i-2]=False;expected_e[j]=False
            actual_p,actual_e=environment_pairing(pp,ee,ap,ae)
            np.testing.assert_array_equal(actual_p,expected_p);np.testing.assert_array_equal(actual_e,expected_e)
            row=np.zeros(26);row[:4]=row[4:8]=pp.ravel();row[8:12]=row[12:16]=ee.ravel()
            row[16:18]=ap;row[18:20]=ae;row[20:22]=expected_p;row[22:24]=expected_e;rows.append(row)
        self.assertEqual(capture_errors(np.array(rows)),0)
        self.assertEqual(capture_errors(np.zeros((0,26))),0)


if __name__=='__main__':unittest.main()
