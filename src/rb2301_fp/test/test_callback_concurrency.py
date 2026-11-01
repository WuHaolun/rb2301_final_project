"""Headless concurrency checks on an isolated ROS domain; never drive a robot."""
import math
import os
from pathlib import Path
import sys
from threading import Event, Lock, Thread
import time
import unittest
from unittest import mock

import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from sensor_msgs.msg import LaserScan

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rb2301_fp import dynamic_window_approach as dwa
from rb2301_fp.obstacle_course import ObstacleCourseNode


def pose_message(nanosec, x, yaw=0.0):
    msg = PoseStamped()
    msg.header.stamp.sec = 1
    msg.header.stamp.nanosec = nanosec
    msg.pose.position.x = x
    msg.pose.orientation.z = math.sin(yaw / 2)
    msg.pose.orientation.w = math.cos(yaw / 2)
    return msg


class CallbackConcurrencyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        rclpy.init(args=[], domain_id=int(os.environ.get('FP_TEST_DOMAIN_ID', '168')))

    @classmethod
    def tearDownClass(cls):
        rclpy.shutdown()

    def setUp(self):
        self.node = ObstacleCourseNode(is_simulation=False)
        self.node.timer.cancel()
        self.node._rviz_timer.cancel()
        self.node.move_2D = mock.Mock()  # Suppress all command publication.
        self.node.frame = dict(origin_x=0.0, origin_y=0.0, rotation_deg=0.0,
                               heading_offset_deg=0.0)
        self.node.optitrack_callback(pose_message(0, 0.0))
        self.node.optitrack_callback(pose_message(100_000_000, 0.02))
        self.node._ob = np.array([[10.0, 10.0]])
        self.driver = Node('fp_concurrency_test_driver')
        self.executor = MultiThreadedExecutor(num_threads=2)
        self.executor.add_node(self.node)
        self.executor.add_node(self.driver)
        self.executor_thread = None
        self.errors = []
        self.release = Event()

    def tearDown(self):
        self.release.set()
        self.node.timer.cancel()
        self.executor.shutdown(timeout_sec=5.0)
        if self.executor_thread is not None:
            self.executor_thread.join(timeout=5.0)
            self.assertFalse(self.executor_thread.is_alive(), 'Executor did not stop')
        self.driver.destroy_node()
        self.node.destroy_node()
        self.assertEqual(self.errors, [])

    def record_errors(self, callback, *args):
        try:
            callback(*args)
        except BaseException as error:
            self.errors.append(error)

    def wait_for(self, predicate, message, timeout=5.0):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if predicate():
                return
            if self.errors:
                self.fail(str(self.errors))
            time.sleep(0.01)
        self.fail(message)

    def test_sensors_update_during_dwa_without_overlapping_planners(self):
        entered = Event()
        counter_lock = Lock()
        calls = []

        def blocked_dwa(**kwargs):
            with counter_lock:
                calls.append(time.monotonic())
            entered.set()
            if not self.release.wait(timeout=5.0):
                raise TimeoutError('Test did not release DWA')
            state = kwargs['x'].copy()
            return [0.1, 0.0], np.array([state, state])

        patcher = mock.patch.object(dwa, 'dwa_control', side_effect=blocked_dwa)
        patcher.start()
        self.addCleanup(patcher.stop)
        pose_writer = self.driver.create_publisher(
            PoseStamped, self.node.map_sub.topic_name, 2)
        scan_writer = self.driver.create_publisher(LaserScan, 'scan', 2)
        self.executor_thread = Thread(
            target=self.record_errors, args=(self.executor.spin,), daemon=True)
        self.executor_thread.start()
        self.wait_for(
            lambda: pose_writer.get_subscription_count() == 1
            and scan_writer.get_subscription_count() == 1,
            'Sensor subscriptions were not discovered')

        self.node.timer.reset()
        self.assertTrue(entered.wait(timeout=5.0), 'DWA did not start')
        pose_writer.publish(pose_message(200_000_000, 0.04, yaw=0.1))
        scan = LaserScan()
        scan.range_min, scan.range_max = 0.05, 10.0
        scan.angle_min, scan.angle_increment = -0.3, 0.2
        scan.ranges = [2.0] * 4
        scan_writer.publish(scan)
        self.wait_for(
            lambda: self.node._read_sensor_data('last_timestamp')[0] == 1.2,
            'Optitrack callback was blocked by DWA')
        self.wait_for(
            lambda: self.node._read_sensor_data('_ob')[0].shape == (4, 2),
            'LiDAR callback was blocked by DWA')
        velocity, = self.node._read_sensor_data('actual_velocity')
        self.assertAlmostEqual(velocity[1], 1.0)
        # Cross multiple 50 ms timer periods while DWA is still running.
        self.assertFalse(self.release.wait(timeout=0.12))
        with counter_lock:
            self.assertEqual(len(calls), 1, 'DWA overlapped itself')
        self.assertFalse(self.node.move_2D.called)
        self.node.timer.cancel()
        self.release.set()
        self.wait_for(lambda: self.node.move_2D.called, 'DWA did not finish')

    def test_pose_and_velocity_are_read_only_after_the_complete_update(self):
        estimating = Event()
        reading = Event()
        read_done = Event()
        observed = []
        update_velocity = self.node.update_actual_velocity

        def paused_estimate(stamp):
            estimating.set()
            if not self.release.wait(timeout=5.0):
                raise TimeoutError('Test did not release velocity estimation')
            update_velocity(stamp)

        def read_state():
            reading.set()
            observed.append(self.node._read_sensor_data(
                'pose', 'actual_velocity', 'last_timestamp'))
            read_done.set()

        with mock.patch.object(self.node, 'update_actual_velocity', paused_estimate):
            writer = Thread(target=self.record_errors, args=(
                self.node.optitrack_callback, pose_message(200_000_000, 0.05)))
            reader = Thread(target=self.record_errors, args=(read_state,))
            writer.start()
            try:
                self.assertTrue(estimating.wait(timeout=5.0))
                reader.start()
                self.assertTrue(reading.wait(timeout=5.0))
                self.assertFalse(read_done.wait(timeout=0.05),
                                 'Reader saw pose before velocity was committed')
            finally:
                self.release.set()
                writer.join(timeout=5.0)
                if reader.ident is not None:
                    reader.join(timeout=5.0)
            self.assertFalse(writer.is_alive())
            self.assertFalse(reader.is_alive())
        self.assertEqual(len(observed), 1)
        pose, velocity, stamp = observed[0]
        np.testing.assert_allclose(pose, [0.05, 0.0, 0.0])
        np.testing.assert_allclose(velocity, [0.3, 0.0])
        self.assertAlmostEqual(stamp, 1.2)
        # Readers own their arrays after the short lock has been released.
        pose[0] = 999.0
        self.assertAlmostEqual(self.node._read_sensor_data('pose')[0][0], 0.05)


if __name__ == '__main__':
    unittest.main()
