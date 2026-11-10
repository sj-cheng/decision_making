"""Engineering checks only: E49 sampled capture, motion and private native parity."""
import unittest
import numpy as np
from e49_rules import create_environment, joint_transition, reference_joint_transition, make_opponent


def state(p, e=((5.,5.),(7.,7.)), pursuers=((-5.,-5.),(-7.,-7.)), active=(1,1,1,1), time=0.):
    s=np.zeros((p.state_dim,1));s[:8,0]=np.r_[np.ravel(e),np.ravel(pursuers)]
    s[p.active_idxs,0]=active;s[8,0]=time
    return s


class E49RulesTests(unittest.TestCase):
    def test_strict_radius_and_wall(self):
        p=create_environment([]);s=state(p,e=((1,0),(7,7)),pursuers=((0,0),(-7,-7)))
        self.assertEqual(p.get_capture_pairs(s),[])
        s[0,0]=.99;self.assertEqual(p.get_capture_pairs(s),[(2,0)])
        p=create_environment([[[.4,.5],[-1,1]]]);self.assertEqual(p.get_capture_pairs(s),[])

    def test_crossing_and_captured_stop(self):
        p=create_environment([]);s=state(p,e=((1.2,0),(7,7)),pursuers=((0,0),(-7,-7)))
        a=np.zeros((8,1));a[0,0]=-1.5;a[4,0]=1.
        n=p.step(s,a,1.)
        self.assertFalse(n[9,0]);self.assertTrue(n[11,0]);self.assertEqual(n[4,0],1.)
        t=p.last_capture_events[0]['time'];self.assertAlmostEqual(t,.09)
        self.assertAlmostEqual(n[0,0],1.2-1.5*t)
        # Uncaptured endpoints are both outside the capture radius.
        s=state(p,e=((.55,0),(7,7)),pursuers=((-.55,0),(-7,-7)))
        a=np.zeros((8,1));a[0,0]=-1.5;a[4,0]=1.
        n=p.step(s,a,1.);self.assertFalse(n[9,0])

    def test_same_pursuer_multiple_substeps_and_matching_tie(self):
        p=create_environment([]);s=state(p,e=((.5,0),(1.4,0)),pursuers=((0,0),(-7,-7)))
        a=np.zeros((8,1));a[4]=1
        n=p.step(s,a,1.)
        self.assertEqual(n[9:13,0].tolist(),[0,0,1,1])
        self.assertEqual([e['pursuer'] for e in p.last_capture_events],[0,0])
        self.assertEqual(p.last_capture_events[0]['time'],0.)
        self.assertGreater(p.last_capture_events[1]['time'],0.)
        s=state(p,e=((0,.2),(0,-.2)),pursuers=((-0.2,0),(.2,0)))
        self.assertEqual(p.get_capture_pairs(s),[(2,0),(3,1)])
        p.step(s,np.zeros((8,1)),1.)
        self.assertEqual([e['time'] for e in p.last_capture_events],[0.,0.])

    def test_axis_limits_inactive_and_horizon(self):
        for h in (40,80,160):
            p=create_environment([],h);s=state(p,time=h-1)
            n=p.step(s,np.full((8,1),100.),1.)
            np.testing.assert_allclose(n[:8,0]-s[:8,0],[1.5]*4+[1.]*4)
            self.assertEqual(n[8,0],h);self.assertTrue(p.is_terminal(n))
            np.testing.assert_array_equal(p.step(n,np.ones((8,1))),n)
        p=create_environment([]);s=state(p,active=(0,1,1,1));n=p.step(s,np.ones((8,1)))
        np.testing.assert_array_equal(n[:2],s[:2]);self.assertTrue(n[11,0])

    def test_original_collision_endpoint(self):
        from corner_estimator_scenarios import original_model
        boxes=[[[.4,.5],[-1,1]]]
        p=create_environment(boxes);old=original_model(boxes,1.)
        s=state(p,e=((5,5),(7,7)),pursuers=((0,0),(-7,-7)))
        a=np.zeros((8,1));a[4]=1.
        n=p.step(s,a);o=old.step(s,a,1.)
        np.testing.assert_allclose(n[:8],o[:8],atol=1e-10)
        self.assertFalse(p.is_valid(n))

    def test_compiled_reference_dense_parity(self):
        rng=np.random.default_rng(4949)
        boxes=np.array([[[float(-8+3*(i%6)),float(-7.6+3*(i%6))],
                         [float(-7+4*(i//6)),float(-6.6+4*(i//6))]] for i in range(24)])
        for delta in (.01,.005,.3):
            for _ in range(30):
                # Include nearby encounters, unsafe swept segments, and masks.
                p=rng.uniform(-9,9,(2,2));e=p+rng.uniform(-2,2,(2,2))
                pa=rng.uniform(-2,2,(2,2));ea=rng.uniform(-2,2,(2,2))
                ap=rng.integers(0,2,2).astype(bool);ae=rng.integers(0,2,2).astype(bool)
                args=(p,e,pa,ea,ap,ae,boxes,delta)
                actual=joint_transition(*args);expected=reference_joint_transition(*args)
                for a,b in zip(actual[:4],expected[:4]):np.testing.assert_allclose(a,b,atol=1e-12)
                self.assertEqual(actual[4],expected[4])

    def test_private_cpp_equivalence_and_rng(self):
        rng=np.random.default_rng(849)
        for boxes in ([],[[[0.,.4],[-2.,2.]]]):
            p=create_environment(boxes,40);other=make_opponent(p,19,simulations=3)
            wrapper=other._problem_wrapper_for(p)
            fixtures=[(state(p,e=((.5,0),(1.4,0)),pursuers=((0,0),(-7,-7))),np.r_[np.zeros(4),1.,0.,0.,0.].reshape(8,1))]
            s=state(p,e=((1.2,0),(7,7)),pursuers=((0,0),(-7,-7)))
            a=np.zeros((8,1));a[0]=-1.5;a[4]=1.
            fixtures.append((s,a))
            fixtures.append((state(p,e=((1.,0),(7,7)),pursuers=((0,0),(-7,-7))),np.zeros((8,1))))
            for _ in range(12):
                fixtures.append((state(p,e=rng.uniform(3,8,(2,2)),pursuers=rng.uniform(-8,-3,(2,2))),rng.uniform(-2,2,(8,1))))
            for s,a in fixtures:
                np.testing.assert_allclose(p.step(s,a),np.asarray(wrapper.step(s,a,1.)).reshape(-1,1),atol=3e-5)
                np.testing.assert_allclose(p.normalized_reward(s,a),np.asarray(wrapper.normalized_reward(s,a)).reshape(-1,1),atol=3e-5)
            s=state(p)
            saved=other.get_rng_state(p);a=other.search(p,s).best_action.copy()
            other.set_rng_state(p,saved);b=other.search(p,s).best_action.copy()
            np.testing.assert_array_equal(a,b)
            self.assertEqual(other.solver_settings.search_depth,100)


if __name__=='__main__':unittest.main()
