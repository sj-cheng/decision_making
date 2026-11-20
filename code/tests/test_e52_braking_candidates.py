"""Derivative-aware recovery must use ordinary feasible MINCO trajectories."""
from dataclasses import replace
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import numpy as np

from solvers.e50_distributed_geometry import evaluate_minco_endpoint
from solvers.e50_distributed_runtime import AgentObservation
from solvers.e52_hmcts import HierarchicalMCTS, LocalSnapshot


class _Map:
    def __init__(self,boxes=()): self.boxes=np.asarray(boxes,float).reshape(-1,2,2)
    def obstacle_boxes(self): return self.boxes
    def free_space_endpoints(self, **kwargs): return []


def observation(velocity=(1.4,0.), acceleration=(0.,1.), particles=(), boxes=()):
    particles=np.asarray(particles,float).reshape(-1,2)
    return AgentObservation(0,0,0.,np.zeros(2),np.asarray(velocity,float),0.,LocalSnapshot(_Map(boxes)),
        SimpleNamespace(particles=particles,weights=np.ones(len(particles))),
        acceleration=np.asarray(acceleration,float))


class BrakingCandidateTests(unittest.TestCase):
    def test_cruising_to_waypoint_preserves_speed_at_the_same_endpoint(self):
        start, goal = np.zeros(2), np.array([1.6, 1.2])
        velocity = (goal - start) / 2.
        args = dict(duration=2., start_velocity=velocity, max_speed=1.5,
                    max_acceleration=1.5, continuous_speed=True, continuous_acceleration=True)
        through = evaluate_minco_endpoint(start, goal, pass_through=True, **args)
        stop = evaluate_minco_endpoint(start, goal, **args)
        self.assertTrue(through['feasible'])
        np.testing.assert_allclose(through['positions'][-1], goal)
        np.testing.assert_allclose(through['velocities'][-1], velocity)
        np.testing.assert_allclose(through['accelerations'], 0., atol=1e-12)
        np.testing.assert_allclose(stop['velocities'][-1], 0., atol=1e-12)

    def test_speed_limit_rejects_diagonal_peak_between_samples(self):
        # From rest, only the endpoints (both stationary) appear on this grid.
        goal = np.array([1.2, 1.2])
        result = evaluate_minco_endpoint([0., 0.], goal, duration=2., sample_dt=2.,
            max_speed=1.5, continuous_speed=True)
        self.assertFalse(result['feasible'])
        self.assertAlmostEqual(result['max_speed'], 1.875*np.linalg.norm(goal)/2.)

    def test_continuous_acceleration_rejects_peak_between_samples(self):
        track = evaluate_minco_endpoint([0., 0.], [2., 0.], sample_dt=2.,
            max_acceleration=2., continuous_acceleration=True)
        np.testing.assert_allclose(track['accelerations'], 0., atol=1e-12)
        self.assertAlmostEqual(track['max_acceleration'], 5./np.sqrt(3.))
        self.assertFalse(track['feasible'])

    def test_lower_limit_requires_more_room_to_brake(self):
        obs = observation(velocity=(1.7, 0.), acceleration=(0., 0.),
                          boxes=[[[.31, .45], [-.2, .2]]])
        track = dict(positions=np.array([obs.pose]), velocities=np.array([obs.velocity]),
                     accelerations=np.array([obs.acceleration]))
        for limit, expected in ((10., True), (2., False)):
            controller = HierarchicalMCTS(max_acceleration=limit)
            controller.midpoint = 0
            self.assertEqual(controller._can_brake(obs.local_map, track), expected)

    def test_public_derivatives_generate_short_forward_candidates(self):
        controller=HierarchicalMCTS()
        obs=observation()
        actions=controller._braking_actions(obs)
        points=[p for p,_ in actions]
        self.assertEqual(len(points),4)
        self.assertTrue(all(np.dot(p-obs.pose,obs.velocity)>0 for p in points))
        self.assertTrue(all(np.linalg.norm(p-obs.pose)<=controller.max_distance for p in points))
        t=actions[0][1]
        np.testing.assert_allclose(points[0],.5*t*obs.velocity+t*t*obs.acceleration/12.)
        changed=controller._braking_actions(replace(obs,acceleration=np.zeros(2)))[0][0]
        self.assertFalse(np.allclose(points[0],changed[0]))
        self.assertEqual(controller._braking_actions(observation(velocity=(0.,0.),acceleration=(0.,0.))),[])

    def test_recovery_preserves_incoming_derivatives_and_finishes_at_rest(self):
        controller=HierarchicalMCTS()
        obs=observation()
        for endpoint, duration in controller._braking_actions(obs):
            track=evaluate_minco_endpoint(obs.pose,endpoint,duration=duration,
                start_velocity=obs.velocity,start_acceleration=obs.acceleration,
                max_speed=2.,max_acceleration=10.,proxy=obs.local_map)
            self.assertTrue(track['feasible'])
            np.testing.assert_allclose(track['velocities'][0],obs.velocity,atol=1e-12)
            np.testing.assert_allclose(track['accelerations'][0],obs.acceleration,atol=1e-12)
            np.testing.assert_allclose(track['velocities'][-1],np.zeros(2),atol=1e-12)
            np.testing.assert_allclose(track['accelerations'][-1],np.zeros(2),atol=1e-12)

    def test_braking_available_without_camera_target_and_passes_whole_curve_check(self):
        # The in-place endpoint turns back through this small observed obstacle;
        # a short forward stop continues beyond it on a different legal curve.
        center=np.array([.301056,.04608]); half=np.array([.005,.005])
        boxes=np.array([np.column_stack((center-half,center+half))])
        obs=observation(boxes=boxes)
        stationary=evaluate_minco_endpoint(obs.pose,obs.pose,duration=2.,start_velocity=obs.velocity,
            start_acceleration=obs.acceleration,proxy=obs.local_map)
        self.assertFalse(obs.local_map.path_safe(stationary['positions']))
        with patch('solvers.e52_hmcts.single_agent_endpoints',return_value=(np.zeros(2),)), \
             patch('solvers.e52_hmcts._local_tangent_candidates',return_value=[]):
            controller=HierarchicalMCTS(depth=1,high_simulations=32,time_budget_ms=1000.)
            plan=controller.act(obs)
        self.assertTrue(plan.minco_feasible)
        self.assertGreater(plan.endpoint[0],0.)
        self.assertTrue(controller.last_search['root'])
        self.assertTrue(all(c['option']=='recover' for c in controller.last_search['root']))
        chosen=evaluate_minco_endpoint(obs.pose,plan.endpoint,duration=plan.duration,start_velocity=obs.velocity,
            start_acceleration=obs.acceleration,max_speed=2.,max_acceleration=10.,proxy=obs.local_map)
        self.assertTrue(chosen['feasible'])
        self.assertTrue(obs.local_map.path_safe(chosen['positions']))

    def test_short_braking_duration_is_carried_out_of_search(self):
        obs = observation(velocity=(1.7, 0.), acceleration=(0., 0.),
                          boxes=[[[.31, .45], [-.05, .05]]])
        with patch('solvers.e52_hmcts.single_agent_endpoints', return_value=(np.zeros(2),)), \
             patch('solvers.e52_hmcts._local_tangent_candidates', return_value=[]):
            plan = HierarchicalMCTS(depth=1, high_simulations=32, time_budget_ms=1000.).act(obs)
        self.assertTrue(plan.minco_feasible)
        self.assertLess(plan.duration, .35)
        self.assertLess(plan.endpoint[0], .31)
        track = evaluate_minco_endpoint(obs.pose, plan.endpoint, duration=plan.duration,
            start_velocity=obs.velocity, start_acceleration=obs.acceleration,
            max_speed=2., max_acceleration=10.)
        self.assertTrue(track['feasible'])
        self.assertTrue(obs.local_map.path_safe(track['positions']))
        self.assertFalse(obs.local_map.path_safe(evaluate_minco_endpoint(
            obs.pose, plan.endpoint, duration=2., start_velocity=obs.velocity
        )['positions']))

    def test_recover_stays_within_existing_option_and_candidate_budget(self):
        controller=HierarchicalMCTS(depth=1,high_simulations=64,time_budget_ms=1000.)
        controller.act(observation(particles=[[5.,0.]]))
        recovery=[c for c in controller.last_search['root'] if c['option']=='recover']
        self.assertLessEqual(len(recovery),8)
        self.assertGreater(len(recovery),0)
        self.assertEqual(controller.OPTIONS,('chase','frontier','bypass','recover'))

    def test_no_feasibility_bypass_when_all_trajectories_blocked(self):
        obs=observation(boxes=[[[ -3.,3.],[-3.,3.]]])
        controller=HierarchicalMCTS(high_simulations=64,time_budget_ms=1000.)
        plan=controller.act(obs)
        self.assertFalse(plan.minco_feasible)
        self.assertEqual(controller.last_search['root'],[])


if __name__=='__main__': unittest.main()
