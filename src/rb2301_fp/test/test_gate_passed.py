"""Gate passed: planner publishes it once, controller switches DWA to its final-zone overrides."""
from pathlib import Path as FilePath
import sys
import time
import unittest
from unittest import mock

import numpy as np
import rclpy
from geometry_msgs.msg import PointStamped
from rclpy.executors import SingleThreadedExecutor
from rclpy.parameter import Parameter
from rclpy.qos import DurabilityPolicy, ReliabilityPolicy
from sensor_msgs.msg import LaserScan
from visualization_msgs.msg import Marker

sys.path.insert(0, str(FilePath(__file__).resolve().parents[1]))
sys.path.insert(0, str(FilePath(__file__).resolve().parent))
from rb2301_fp import dynamic_window_approach
from rb2301_fp.global_planner import GlobalPlannerNode
from rb2301_fp.obstacle_course import ObstacleCourseNode
from test_global_planner_core import FORWARD, GateScene


def gate_msg(x=4.0, y=0.3):
    msg = PointStamped()
    msg.header.frame_id = 'fp_visualization'
    msg.point.x, msg.point.y = x, y
    return msg


class PlannerGateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        rclpy.init(args=[], domain_id=171)

    @classmethod
    def tearDownClass(cls):
        rclpy.shutdown()

    def make_node(self, overrides=None):
        node = GlobalPlannerNode(is_simulation=True, parameter_overrides=overrides)
        for timer in (node.map_timer, node.plan_timer, node.log_timer, node.map_publish_timer):
            timer.cancel()
        self.addCleanup(node.destroy_node)
        node.path_publisher = mock.Mock()
        self._stamp = 0
        return node

    def tick(self, node, u, v, yaw_rate=10.0):
        '''New pose, then a map tick with a scan skipped for yaw rate, so the map written by the test stays.'''
        self._stamp += 1
        node.set_pose(u, v, 0.0, mock.Mock(sec=self._stamp, nanosec=0))
        node._scan = (LaserScan(), node.pose.copy(), yaw_rate)
        node.map_timer_callback()

    def drive_through_gate(self, node):
        self.tick(node, FORWARD[0], 0.3 - 1.0) # first pose, one leg to the side: arms the detector
        scene = GateScene()
        scene.grid = node.planner.grid
        scene.gate()
        for u in FORWARD:
            self.tick(node, u, 0.3)
        node.goal_callback(gate_msg(8.0, 0.3))

    def test_gate_passed_is_published_once_on_latched_topic(self):
        node = self.make_node()
        qos = node.gate_publisher.qos_profile
        self.assertEqual(qos.durability, DurabilityPolicy.TRANSIENT_LOCAL)
        self.assertEqual(qos.reliability, ReliabilityPolicy.RELIABLE)
        node.gate_publisher = mock.Mock()
        self.drive_through_gate(node)
        self.assertTrue(node.gate_passed)
        self.assertEqual(node.gate_publisher.publish.call_count, 1)
        msg = node.gate_publisher.publish.call_args.args[0]
        self.assertEqual(msg.header.frame_id, 'fp_visualization')
        np.testing.assert_allclose([msg.point.x, msg.point.y], [4.0, 0.3], atol=0.051)
        self.tick(node, 5.3, 0.3)
        self.assertEqual(node.gate_publisher.publish.call_count, 1)

    def test_direct_mode_publishes_empty_paths_after_the_gate(self):
        node = self.make_node()
        node.gate_publisher = mock.Mock()
        self.drive_through_gate(node)
        with mock.patch.object(node.planner, 'plan', wraps=node.planner.plan) as plan:
            node.plan_timer_callback()
            node.plan_timer_callback()
        plan.assert_not_called()
        for call in node.path_publisher.publish.call_args_list:
            self.assertEqual(len(call.args[0].poses), 0)
        self.assertEqual(node.path_publisher.publish.call_count, 2)

    def test_keep_mode_keeps_planning(self):
        node = self.make_node([Parameter('final_zone_mode', value='keep')])
        node.gate_publisher = mock.Mock()
        self.drive_through_gate(node)
        self.assertTrue(node.gate_passed)
        node.plan_timer_callback()
        self.assertGreater(len(node.path_publisher.publish.call_args.args[0].poses), 0)

    def test_disabled_detection_is_unchanged(self):
        node = self.make_node([Parameter('gate_detection', value=False)])
        self.assertFalse(hasattr(node, 'gate_publisher'))
        self.drive_through_gate(node)
        self.assertIsNone(node.gate_detector)
        self.assertFalse(node.gate_passed)
        node.plan_timer_callback()
        self.assertGreater(len(node.path_publisher.publish.call_args.args[0].poses), 0)

    def test_scan_ticks_are_still_mapped(self):
        node = self.make_node()
        node.gate_publisher = mock.Mock()
        with mock.patch.object(GlobalPlannerNode, 'scan_to_world',
                               return_value=(np.zeros(2), np.empty((0, 2)), np.empty(0, bool))) as to_world:
            self.tick(node, 0.0, 0.0, yaw_rate=0.0)
            self.tick(node, 0.0, 0.0, yaw_rate=10.0)
        self.assertEqual(to_world.call_count, 1)
        self.assertEqual(node._skipped_scans, 1)
        self.assertEqual(len(node.gate_detector._trail), 2) # the detector ran on both ticks

    def test_performance_log_reports_gate_passed(self):
        node = self.make_node()
        node.gate_publisher = mock.Mock()
        self.drive_through_gate(node)
        logger = mock.Mock()
        with mock.patch.object(node, 'get_logger', return_value=logger):
            node.log_timer_callback()
        self.assertIn('gate_passed=True', logger.info.call_args.args[0])

    def test_invalid_final_zone_mode_is_rejected(self):
        with self.assertRaises(ValueError):
            GlobalPlannerNode(is_simulation=True, parameter_overrides=[
                Parameter('final_zone_mode', value='fast')])


class _TypoConfig(dynamic_window_approach.Config):
    '''Config whose final_zone contains a name DWA does not have.'''
    def __setattr__(self, name, value):
        if name == 'final_zone':
            value = {'to_goal_cost_gain': 0.5}
        super().__setattr__(name, value)


class ControllerGateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        rclpy.init(args=[], domain_id=172)

    @classmethod
    def tearDownClass(cls):
        rclpy.shutdown()

    def make_node(self, overrides=None):
        node = ObstacleCourseNode(is_simulation=True, parameter_overrides=overrides)
        node.timer.cancel()
        node._rviz_timer.cancel()
        self.addCleanup(node.destroy_node)
        node.move_2D = mock.Mock()
        node.goal_publisher = mock.Mock()
        node._ob = np.array([[100., 100.]])
        node.actual_velocity = (0., 0.)
        node.pose = np.zeros(3)
        node._initialize_point_goal()
        return node

    def test_final_zone_is_empty_by_default(self):
        for is_simulation in (True, False):
            config = dynamic_window_approach.Config(is_simulation=is_simulation)
            self.assertEqual(config.final_zone, {})
            before = dict(vars(config))
            config.apply_final_zone()
            self.assertEqual(vars(config), before)

    def test_overrides_are_applied_exactly_once(self):
        node = self.make_node()
        self.assertEqual(node.sub_gate.qos_profile.durability, DurabilityPolicy.TRANSIENT_LOCAL)
        node.dwa_config.final_zone = {'max_speed': 0.123, 'predict_time': 1.1}
        with mock.patch.object(node.dwa_config, 'apply_final_zone',
                               wraps=node.dwa_config.apply_final_zone) as apply:
            node.gate_passed_callback(gate_msg())
            self.assertEqual(node.dwa_config.max_speed, 0.123)
            self.assertEqual(node.dwa_config.predict_time, 1.1)
            node.dwa_config.max_speed = 0.4
            node.gate_passed_callback(gate_msg(9.0, 9.0)) # repeat: ignored
        self.assertEqual(apply.call_count, 1)
        self.assertEqual(node.dwa_config.max_speed, 0.4)
        self.assertEqual(node.stage, 'final_zone')
        np.testing.assert_allclose(node.gate_center, [4.0, 0.3])

    def test_ignored_without_global_path(self):
        node = self.make_node([Parameter('use_global_path', value=False)])
        node.dwa_config.final_zone = {'max_speed': 0.123}
        speed = node.dwa_config.max_speed
        node.gate_passed_callback(gate_msg())
        self.assertEqual(node.dwa_config.max_speed, speed)
        self.assertEqual(node.stage, 'point_goal')
        self.assertIsNone(node.gate_center)

    def test_unknown_override_fails_at_construction(self):
        with self.assertRaises(AttributeError):
            _TypoConfig()
        with mock.patch('rb2301_fp.obstacle_course.dynamic_window_approach.Config', _TypoConfig):
            with self.assertRaises(AttributeError):
                ObstacleCourseNode(is_simulation=True)
        config = dynamic_window_approach.Config()
        config.final_zone = {'final_zone': {}}
        with self.assertRaises(AttributeError):
            config.apply_final_zone()

    def test_gate_marker_appears_after_gate_passed(self):
        node = self.make_node()
        node._rviz_publisher = mock.Mock()

        def scene():
            node.publish_rviz_visualization()
            return {m.ns: m for m in node._rviz_publisher.publish.call_args.args[0].markers}

        self.assertEqual(scene()['gate'].action, Marker.DELETE)
        node.gate_passed_callback(gate_msg())
        marker = scene()['gate']
        self.assertEqual(marker.action, Marker.ADD)
        self.assertAlmostEqual(marker.points[0].x, 4.0)
        self.assertAlmostEqual(marker.points[0].y, 0.3)

    def test_late_controller_receives_latched_gate_from_planner(self):
        planner = GlobalPlannerNode(is_simulation=True)
        for timer in (planner.map_timer, planner.plan_timer, planner.log_timer, planner.map_publish_timer):
            timer.cancel()
        self.addCleanup(planner.destroy_node)
        planner.gate_publisher.publish(gate_msg())
        node = self.make_node()
        executor = SingleThreadedExecutor()
        self.addCleanup(executor.shutdown)
        executor.add_node(planner)
        executor.add_node(node)
        deadline = time.monotonic() + 5.0
        while node.gate_center is None and time.monotonic() < deadline:
            executor.spin_once(timeout_sec=0.05)
        self.assertIsNotNone(node.gate_center, 'latched gate_passed did not reach a late subscriber')
        self.assertEqual(node.stage, 'final_zone')


if __name__ == '__main__':
    unittest.main()
