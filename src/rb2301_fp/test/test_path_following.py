"""Controller follows the global path through a lookahead waypoint, without driving a robot."""
from pathlib import Path as FilePath
import sys
import unittest
from unittest import mock

import numpy as np
import rclpy
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped
from rclpy.parameter import Parameter
from visualization_msgs.msg import Marker

sys.path.insert(0, str(FilePath(__file__).resolve().parents[1]))
from rb2301_fp.obstacle_course import ObstacleCourseNode


def path_msg(points):
    msg = Path()
    for x, y in points:
        pose = PoseStamped()
        pose.pose.position.x, pose.pose.position.y = float(x), float(y)
        msg.poses.append(pose)
    return msg


class PathFollowingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        rclpy.init(args=[], domain_id=170)

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

    def mock_dwa(self):
        patcher = mock.patch(
            'rb2301_fp.obstacle_course.dynamic_window_approach.dwa_control',
            return_value=([.1, .05], np.zeros((2, 5))))
        self.addCleanup(patcher.stop)
        return patcher.start()

    def straight_path(self, node, length=3.0):
        '''Path along x to length, then straight to the final goal.'''
        points = [(x, 0.0) for x in np.arange(0.0, length + 1e-9, 0.1)]
        points.append(tuple(node._goal.goal_node))
        node.path_callback(path_msg(points))

    def test_goal_is_published_once_frozen(self):
        node = self.make_node()
        msg = node.goal_publisher.publish.call_args.args[0]
        np.testing.assert_allclose([msg.point.x, msg.point.y], node._goal.goal_node)
        self.assertEqual(msg.header.frame_id, 'fp_visualization')
        node._initialize_point_goal() # already frozen: not published again
        self.assertEqual(node.goal_publisher.publish.call_count, 1)

    def test_dwa_gets_lookahead_waypoint_and_final_goal_is_kept(self):
        node = self.make_node()
        dwa = self.mock_dwa()
        final_goal = node._goal.goal_node.copy()
        self.straight_path(node)
        node.pose = np.array([0.5, 0.05, 0.0])
        node.timer_callback()
        np.testing.assert_allclose(dwa.call_args.kwargs['goal'], [0.5 + node.path_follower.lookahead, 0.0], atol=1e-9)
        np.testing.assert_array_equal(node._goal.goal_node, final_goal)
        # Near the end of the path the waypoint is the final goal.
        node.pose = np.array([*final_goal - [0.4, 0.0], 0.0])
        node.path_callback(path_msg([node.pose[:2], final_goal]))
        node.timer_callback()
        np.testing.assert_allclose(dwa.call_args.kwargs['goal'], final_goal)

    def test_missing_empty_or_stale_path_uses_final_goal(self):
        node = self.make_node()
        dwa = self.mock_dwa()
        node.timer_callback() # no path yet
        np.testing.assert_array_equal(dwa.call_args.kwargs['goal'], node._goal.goal_node)
        node.path_callback(path_msg([]))
        node.timer_callback()
        np.testing.assert_array_equal(dwa.call_args.kwargs['goal'], node._goal.goal_node)
        self.straight_path(node)
        node.path_follower.stamp -= node.path_follower.timeout + 0.1
        node.timer_callback()
        np.testing.assert_array_equal(dwa.call_args.kwargs['goal'], node._goal.goal_node)

    def test_disabled_global_path_leaves_dwa_call_unchanged(self):
        node = self.make_node(overrides=[Parameter('use_global_path', value=False)])
        dwa = self.mock_dwa()
        self.straight_path(node)
        node.timer_callback()
        self.assertIs(dwa.call_args.kwargs['goal'], node._goal.goal_node)
        self.assertIsNone(dwa.call_args.kwargs['goal_heading'])

    def test_lookahead_and_timeout_are_parameters(self):
        node = self.make_node(overrides=[Parameter('path_lookahead', value=0.5),
                                         Parameter('path_timeout', value=1.0)])
        self.assertEqual(node.path_follower.lookahead, 0.5)
        self.assertEqual(node.path_follower.timeout, 1.0)

    def test_visualization_shows_path_and_waypoint_only_while_following(self):
        node = self.make_node()
        self.mock_dwa()
        node._rviz_publisher = mock.Mock()

        def scene():
            node.publish_rviz_visualization()
            return {m.ns: m for m in node._rviz_publisher.publish.call_args.args[0].markers}

        self.assertEqual(scene()['global_path'].action, Marker.DELETE)
        self.straight_path(node)
        node.timer_callback()
        markers = scene()
        self.assertEqual(markers['global_path'].action, Marker.ADD)
        self.assertEqual(markers['path_waypoint'].action, Marker.ADD)
        self.assertAlmostEqual(markers['path_waypoint'].points[0].x, node.path_follower.lookahead)
        node.path_follower.stamp -= node.path_follower.timeout + 0.1
        self.assertEqual(scene()['global_path'].action, Marker.DELETE)
        self.assertEqual(scene()['path_waypoint'].action, Marker.DELETE)


if __name__ == '__main__':
    unittest.main()
