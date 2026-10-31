"""Headless ROS message/transport checks; uses an isolated ROS domain."""
import math
import os
from pathlib import Path
import sys
import time
import unittest
from unittest import mock

import numpy as np
import rclpy
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from visualization_msgs.msg import Marker, MarkerArray

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rb2301_fp import dynamic_window_approach as dwa
from rb2301_fp.obstacle_course import ObstacleCourseNode


class RvizVisualizationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Override inherited robot domain; never connect the test to a robot.
        rclpy.init(args=[], domain_id=int(os.environ.get('FP_TEST_DOMAIN_ID', '167')))

    @classmethod
    def tearDownClass(cls):
        rclpy.shutdown()

    def setUp(self):
        self.node = ObstacleCourseNode(is_simulation=True, calibrate=True)
        self.node.timer.cancel()
        self.node._rviz_timer.cancel()
        self.reader = Node('fp_visualization_test_reader')
        self.executor = SingleThreadedExecutor()
        self.executor.add_node(self.node)
        self.executor.add_node(self.reader)
        self.qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE,
                              durability=DurabilityPolicy.TRANSIENT_LOCAL)

    def tearDown(self):
        self.executor.shutdown()
        self.reader.destroy_node()
        self.node.destroy_node()

    def receive_snapshot(self):
        """Join after publication: verify a real DDS late-joining subscriber."""
        self.node.publish_rviz_visualization()
        received = []
        subscription = self.reader.create_subscription(
            MarkerArray, '/obstaclecourse/visualization', received.append, self.qos)
        deadline = time.monotonic() + 5.0
        while not received and time.monotonic() < deadline:
            self.executor.spin_once(timeout_sec=0.05)
        self.reader.destroy_subscription(subscription)
        self.assertTrue(received, 'No transient-local snapshot received over DDS')
        return {marker.ns: marker for marker in received[-1].markers}

    def test_late_join_without_scan_or_prediction(self):
        self.node.pose = np.array([2.0, -1.0, 90.0])
        scene = self.receive_snapshot()
        self.assertEqual(scene['obstacles'].action, Marker.DELETE)
        self.assertEqual(scene['predicted_trajectory'].action, Marker.DELETE)
        for marker in scene.values():
            self.assertEqual(marker.header.frame_id, 'fp_visualization')
            self.assertEqual(marker.pose.orientation.w, 1.0)
            self.assertEqual(marker.lifetime.sec, 1)
        arrow = scene['robot_heading']
        np.testing.assert_allclose(
            [(p.x, p.y) for p in arrow.points], [[2.0, -1.0], [2.0, -0.5]])
        self.assertNotIn('matplotlib.backends.backend_qtagg', sys.modules)

    def test_timer_publishes_while_controller_has_no_scan(self):
        self.node.pose = np.array([1.0, 2.0, 0.0])
        received = []
        self.reader.create_subscription(
            MarkerArray, '/obstaclecourse/visualization', received.append, self.qos)
        self.node._rviz_timer.reset()
        deadline = time.monotonic() + 5.0
        while not received and time.monotonic() < deadline:
            self.executor.spin_once(timeout_sec=0.05)
        self.assertTrue(received)
        self.assertIsNone(self.node.actual_velocity)
        self.assertEqual(self.node._ob.size, 0)

    def test_empty_layers_and_completed_subgoals_delete_old_markers(self):
        self.node.calibrate = False
        self.node.pose = np.array([0.0, 0.0, 0.0])
        self.node._ob = np.array([[1.0, 2.0], [1.5, 2.0]])
        self.node._segmentation = [self.node._ob.copy()]
        self.node._rviz_predicted_trajectory = np.array([
            [0.0, 0.0, 0.0, 0.1, 0.0], [0.1, 0.0, 0.0, 0.1, 0.0]])
        scene = self.receive_snapshot()
        self.assertEqual(scene['predicted_trajectory'].action, Marker.ADD)
        self.assertEqual(scene['segmentation'].action, Marker.ADD)
        self.assertEqual(len(scene['segmentation'].colors), 2)
        self.node._ob = np.empty((0, 2))
        self.node._segmentation = []
        self.node._sub_goal = []
        scene = self.receive_snapshot()
        for name in ('obstacles', 'segmentation', 'predicted_trajectory', 'sub_goal_heading'):
            self.assertEqual(scene[name].action, Marker.DELETE)

    def test_footprint_rotation_matches_pose_and_dwa_dimensions(self):
        self.node.pose = np.array([2.0, 3.0, 90.0])
        with mock.patch.object(dwa.config, '_robot_type', dwa.RobotType.rectangle):
            footprint = self.receive_snapshot()['robot_footprint']
        point = footprint.points[0]
        self.assertAlmostEqual(point.x, 2.0 - dwa.config.robot_width / 2)
        self.assertAlmostEqual(point.y, 3.0 - dwa.config.robot_length / 2)
        self.assertEqual(footprint.points[0], footprint.points[-1])

    def test_circle_and_infeasible_prediction(self):
        self.node.calibrate = False
        self.node.pose = np.array([-2.0, 1.0, 45.0])
        self.node._ob = np.array([[5.0, 5.0]])
        self.node._rviz_predicted_trajectory = np.zeros((1, 5))
        with mock.patch.object(dwa.config, '_robot_type', dwa.RobotType.circle):
            scene = self.receive_snapshot()
        for point in scene['robot_footprint'].points:
            self.assertAlmostEqual(math.hypot(point.x + 2.0, point.y - 1.0),
                                   dwa.config.robot_radius)
        self.assertEqual(scene['predicted_trajectory'].action, Marker.DELETE)


if __name__ == '__main__':
    unittest.main()
