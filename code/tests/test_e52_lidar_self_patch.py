"""A connected hit cloud around ego must never erase observed obstacles."""
import unittest
import numpy as np
from solvers.e50_interfaces import LidarScan
from solvers.e50_lidar_map import LidarPlanningMap, points_to_obstacle_boxes


def ring_points():
    coordinates=np.linspace(-1.,1.,21)
    return np.concatenate([np.column_stack((coordinates,np.full(21,y))) for y in (-1.,1.)] +
                          [np.column_stack((np.full(21,x),coordinates)) for x in (-1.,1.)])


def covered(points, boxes):
    return ((points[:,None] >= boxes[None,:,:,0]) &
            (points[:,None] <= boxes[None,:,:,1])).all(axis=-1).any(axis=1)


class LidarSelfPatchTests(unittest.TestCase):
    def test_wrapping_real_returns_preserved_and_ego_free(self):
        points=ring_points()
        original=points_to_obstacle_boxes(points,inflation=.05,merge_distance=.25,max_boxes=24)
        self.assertEqual(len(original),1)
        self.assertTrue(np.all((0 >= original[0,:,0]) & (0 <= original[0,:,1])))
        result=points_to_obstacle_boxes(points,inflation=.05,merge_distance=.25,
                                        max_boxes=24,sensor_position=np.zeros(2))
        self.assertTrue(covered(points,result).all())
        self.assertFalse(((0 >= result[:,:,0]) & (0 <= result[:,:,1])).all(axis=1).any())
        self.assertLessEqual(len(result),24)
        # The complete front face is retained; planning through it is blocked.
        self.assertTrue(covered(np.array([[1.,0.]]),result).all())

    def test_capacity_pressure_never_drops_measured_hits(self):
        points=ring_points()
        result=points_to_obstacle_boxes(points,inflation=.05,merge_distance=.25,
                                        max_boxes=2,sensor_position=np.zeros(2))
        self.assertLessEqual(len(result),2)
        self.assertTrue(covered(points,result).all())

    def test_actual_zero_self_return_removed_but_real_hit_retained(self):
        m=LidarPlanningMap(inflation=.05)
        scan=LidarScan('map',0.,np.array([0.,np.pi/2]),np.array([0.,1.]),10.)
        m.update(np.zeros(2),scan)
        self.assertEqual(len(m.points),1)
        result=m.obstacle_boxes()
        self.assertTrue(covered(np.array([[0.,1.]]),result).all())
        self.assertFalse(((0 >= result[:,:,0]) & (0 <= result[:,:,1])).all(axis=1).any())

    def test_real_close_return_is_not_deleted_as_self_hit(self):
        result=points_to_obstacle_boxes([[.02,0.]],inflation=.05,merge_distance=.25,
                                       max_boxes=24,sensor_position=np.zeros(2))
        self.assertTrue(covered(np.array([[.02,0.]]),result).all())


if __name__=='__main__': unittest.main()
