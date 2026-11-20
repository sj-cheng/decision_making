"""Engineering checks only: E49 sampled capture, motion and private native parity."""
import unittest
from unittest.mock import patch
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

    def test_geometric_evader_speed_reaches_simulator(self):
        from solvers.e49_geometric_evader import GeometricEvader

        env = create_environment([], evader_speed=2.)
        initial = state(env, e=((0., 0.), (7., 7.)), pursuers=((-2., 0.), (-8., -8.)),
                        active=(1, 0, 1, 0))
        plan = GeometricEvader(evader_speed=2.).search(env, initial).best_action.reshape(8, 1)
        self.assertEqual(np.max(np.abs(plan[:2])), 2.)
        advanced = env.step(initial, plan)
        np.testing.assert_array_equal(advanced[:2] - initial[:2], plan[:2])
        np.testing.assert_array_equal(advanced[2:8], initial[2:8])

    def test_sampled_curves_capture_at_crossing_and_freeze_target(self):
        from solvers.e50_interfaces import quintic_coefficients, sample_quintic

        env = create_environment([])
        initial = state(env, e=((2., 0.), (7., 7.)), pursuers=((-2., 0.), (-8., -8.)))
        evaders = initial[:4, 0].reshape(2, 2)
        pursuers = initial[4:8, 0].reshape(2, 2)
        e_path = sample_quintic(quintic_coefficients(evaders, [[-2., 0.], [7., 7.]], 1.), 1., .01)[0]
        p_path = sample_quintic(quintic_coefficients(pursuers, [[2., 0.], [-8., -8.]], 1.), 1., .01)[0]
        advanced = env.step_trajectories(initial, p_path, e_path)
        self.assertFalse(advanced[env.active_idxs[0], 0])
        self.assertTrue(advanced[env.active_idxs[1], 0])
        event, = env.last_capture_events
        self.assertGreater(event['sample'], 0)
        self.assertLess(event['sample'], 100)
        np.testing.assert_array_equal(advanced[:2, 0], e_path[event['sample'], 0])

    def test_evader_minco_keeps_safe_tail_and_derivative_continuity(self):
        from solvers.e50_evader_minco import EvaderMinco

        executor = EvaderMinco(np.array([[[.5, 1.5], [-.5, .5]]]), 2., 100)
        prefix, mode = executor.prefix(0, np.zeros(2), np.array([2., 0.]))
        self.assertEqual(mode, 'replan')
        self.assertGreater(np.linalg.norm(prefix['positions'][-1]), 0.)
        for point in prefix['positions']:
            self.assertFalse(.5 <= point[0] <= 1.5 and -.5 <= point[1] <= .5)
        with patch('solvers.e50_evader_minco.evaluate_minco_endpoint',
                   return_value={'feasible': False, 'positions': prefix['positions']}):
            remaining, mode = executor.prefix(0, prefix['positions'][-1], np.array([2., 0.]))
        self.assertEqual(mode, 'continue')
        for key in ('positions', 'velocities', 'accelerations'):
            np.testing.assert_array_equal(prefix[key][-1], remaining[key][0])
        np.testing.assert_allclose(remaining['velocities'][-1], 0., atol=1e-10)
        np.testing.assert_allclose(remaining['accelerations'][-1], 0., atol=1e-10)

    def test_shared_waypoint_preserves_velocity_and_can_finish_braking(self):
        from solvers.e50_evader_minco import EvaderMinco

        executor = EvaderMinco(np.empty((0, 2, 2)), 1.5, 100,
                               max_acceleration=1.5, continuous_speed=True)
        prefix, mode = executor.prefix(0, np.zeros(2), np.array([1.5, 0.]))
        self.assertEqual(mode, 'pass_through')
        self.assertGreater(prefix['velocities'][-1, 0], .1)
        with patch('solvers.e50_evader_minco.evaluate_minco_endpoint',
                   return_value={'feasible': False}):
            while 0 in executor.tail:
                remaining, mode = executor.prefix(0, prefix['positions'][-1], np.array([1.5, 0.]))
                self.assertEqual(mode, 'continue')
                for key in prefix:
                    self.assertEqual(remaining[key].shape, (101, 2))
                    np.testing.assert_allclose(remaining[key][0], prefix[key][-1], atol=1e-10)
                prefix = remaining
        np.testing.assert_allclose(prefix['velocities'][-1], 0., atol=1e-10)
        np.testing.assert_allclose(prefix['accelerations'][-1], 0., atol=1e-10)

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
