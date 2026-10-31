"""Collision regression checks and a ROS-free benchmark.

Run tests with pytest, or run this file with --benchmark on the target robot.
"""

import argparse
import importlib.util
from pathlib import Path
import sys
import time
import unittest
from unittest import mock

import numpy as np


MODULE_PATH = Path(__file__).resolve().parents[1] / "rb2301_fp" / "dynamic_window_approach.py"
SPEC = importlib.util.spec_from_file_location("dwa_collision_checks", str(MODULE_PATH))
dwa = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(dwa)
RobotType = dwa.RobotType


# Frozen pre-optimization implementation; independent of the new broad phase.
def reference_obstacle_cost(trajectory, ob, config):
    """
    calc obstacle cost inf: collision
    """
    if np.size(ob) == 0:
        return 0.0
    ox = ob[:, 0]
    oy = ob[:, 1]
    dx = trajectory[:, 0] - ox[:, None]
    dy = trajectory[:, 1] - oy[:, None]
    r = np.hypot(dx, dy)

    if config.robot_type == RobotType.rectangle:
        yaw = trajectory[:, 2]
        rot = np.array([[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]])
        rot = np.transpose(rot, [2, 0, 1])
        local_ob = ob[:, None] - trajectory[:, 0:2]
        # Rotate each relative point using the heading at the same time step.
        local_ob = np.einsum('oti,tij->otj', local_ob, rot)
        local_ob = local_ob.reshape(-1, local_ob.shape[-1])
        upper_check = local_ob[:, 0] <= config.robot_length / 2
        right_check = local_ob[:, 1] <= config.robot_width / 2
        bottom_check = local_ob[:, 0] >= -config.robot_length / 2
        left_check = local_ob[:, 1] >= -config.robot_width / 2
        if (np.logical_and(np.logical_and(upper_check, right_check),
                           np.logical_and(bottom_check, left_check))).any():
            return float("Inf")
    elif config.robot_type == RobotType.circle:
        if np.array(r <= config.robot_radius).any():
            return float("Inf")

    min_r = np.min(r)
    return 1.0 / min_r  # OK



def room_obstacles(count, half_width=1.0):
    """Raycast a rectangular room around the origin."""
    angles = np.linspace(-np.pi, np.pi, count, endpoint=False)
    cosine, sine = np.cos(angles), np.sin(angles)
    ranges = np.minimum(
        1.5 / np.maximum(np.abs(cosine), 1e-12),
        half_width / np.maximum(np.abs(sine), 1e-12),
    )
    return np.column_stack((ranges * cosine, ranges * sine))


class CollisionCheckingTests(unittest.TestCase):
    def setUp(self):
        self.config = dwa.Config()

    def assert_reference_cost(self, trajectory, obstacles):
        expected = reference_obstacle_cost(trajectory, obstacles, self.config)
        actual = dwa.calc_obstacle_cost(trajectory, obstacles, self.config)
        np.testing.assert_equal(actual, expected)

    def test_empty_obstacles(self):
        trajectory = np.zeros((1, 5))
        for obstacles in (np.empty((0, 2)), np.array([])):
            self.assertEqual(dwa.calc_obstacle_cost(trajectory, obstacles, self.config), 0.0)

    def test_rectangle_includes_edges_and_corners(self):
        trajectory = np.zeros((1, 5))
        half_length = self.config.robot_length / 2
        half_width = self.config.robot_width / 2
        for x in (-half_length, 0.0, half_length):
            for y in (-half_width, 0.0, half_width):
                obstacles = np.array([[x, y]])
                self.assertEqual(dwa.calc_obstacle_cost(trajectory, obstacles, self.config), np.inf)
        for x, y in (
                (np.nextafter(half_length, np.inf), 0.0),
                (0.0, np.nextafter(half_width, np.inf))):
            self.assertTrue(np.isfinite(dwa.calc_obstacle_cost(
                trajectory, np.array([[x, y]]), self.config)))

    def test_circle_broad_phase_does_not_enlarge_rectangle(self):
        trajectory = np.zeros((1, 5))
        obstacles = np.array([[0.12, 0.0]])
        self.assertAlmostEqual(dwa.calc_obstacle_cost(trajectory, obstacles, self.config), 1 / 0.12)

    def test_clearance_still_uses_distant_points(self):
        trajectory = np.zeros((2, 5))
        trajectory[1, 0] = 0.2
        obstacles = np.array([[4.0, 0.0], [0.0, 3.0]])
        self.assertEqual(dwa.calc_obstacle_cost(trajectory, obstacles, self.config), 1 / 3.0)
        self.assert_reference_cost(trajectory, obstacles)

    def test_collision_at_middle_pose_with_different_heading(self):
        trajectory = np.array([
            [0.0, 0.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, np.pi / 2, 0.0, 0.0],
            [2.0, 0.0, np.pi, 0.0, 0.0],
        ])
        obstacles = np.array([[1.0, 0.11], [10.0, 10.0]])
        self.assertEqual(dwa.calc_obstacle_cost(trajectory, obstacles, self.config), np.inf)
        self.assert_reference_cost(trajectory, obstacles)

    def test_rotated_boundary_points_match_reference(self):
        half_length = self.config.robot_length / 2
        half_width = self.config.robot_width / 2
        local_points = np.array([
            [half_length, half_width], [-half_length, half_width],
            [half_length, -half_width], [-half_length, -half_width],
            [half_length, 0.0], [-half_length, 0.0],
            [0.0, half_width], [0.0, -half_width],
            [half_length * (1 + 1e-5), 0.0],
            [0.0, half_width * (1 + 1e-5)],
        ])
        for dtype in (np.float32, np.float64):
            for yaw in (0.0, np.pi / 6, np.pi / 2, np.pi, -2.7):
                trajectory = np.array([[1.25, -0.75, yaw, 0.0, 0.0]], dtype=dtype)
                rotation = np.array([[np.cos(yaw), -np.sin(yaw)],
                                     [np.sin(yaw), np.cos(yaw)]], dtype=dtype)
                points = local_points.astype(dtype) @ rotation.T + trajectory[0, :2]
                for point in points:
                    with self.subTest(dtype=dtype, yaw=yaw, point=point):
                        self.assert_reference_cost(trajectory, point[None, :])

    def test_random_costs_match_reference(self):
        rng = np.random.RandomState(2301)
        for robot_type in (RobotType.rectangle, RobotType.circle):
            self.config.robot_type = robot_type
            for count in (1, 17, 360, 720, 1440):
                for _ in range(16):
                    trajectory = rng.normal(size=(26, 5))
                    obstacles = rng.normal(size=(count, 2)) * 5
                    self.assert_reference_cost(trajectory, obstacles)

    def test_circle_radius_boundary(self):
        self.config.robot_type = RobotType.circle
        trajectory = np.zeros((1, 5))
        radius = self.config.robot_radius
        self.assertEqual(dwa.calc_obstacle_cost(
            trajectory, np.array([[radius, 0.0]]), self.config), np.inf)
        self.assertTrue(np.isfinite(dwa.calc_obstacle_cost(
            trajectory, np.array([[np.nextafter(radius, np.inf), 0.0]]), self.config)))

    def test_inputs_are_unchanged(self):
        trajectory = dwa.predict_trajectory(np.zeros(5), 0.12, 0.1, self.config)
        obstacles = room_obstacles(360)
        original_trajectory, original_obstacles = trajectory.copy(), obstacles.copy()
        dwa.calc_obstacle_cost(trajectory, obstacles, self.config)
        np.testing.assert_array_equal(trajectory, original_trajectory)
        np.testing.assert_array_equal(obstacles, original_obstacles)

    def test_dwa_commands_and_trajectories_match_reference(self):
        state = np.array([0.0, 0.0, 0.0, 0.12, 0.05])
        for robot_type in (RobotType.rectangle, RobotType.circle):
            self.config.robot_type = robot_type
            for half_width in (1.0, 0.18):
                obstacles = room_obstacles(720, half_width)
                for heading in ("+x", "-x", "+y", "-y"):
                    with mock.patch.object(dwa, "calc_obstacle_cost", reference_obstacle_cost):
                        expected = dwa.dwa_control(state.copy(), obstacles, goal_heading=heading,
                                                   config=self.config)
                    actual = dwa.dwa_control(state.copy(), obstacles, goal_heading=heading,
                                             config=self.config)
                    np.testing.assert_array_equal(actual[0], expected[0])
                    np.testing.assert_array_equal(actual[1], expected[1])


def benchmark():
    parser = argparse.ArgumentParser(description="Compare original and optimized DWA without ROS.")
    parser.add_argument("--benchmark", action="store_true")
    parser.add_argument("--points", nargs="+", type=int, default=[360, 720, 1440])
    parser.add_argument("--repeats", type=int, default=10)
    args = parser.parse_args()
    if args.repeats < 1 or any(count < 1 for count in args.points):
        parser.error("points and repeats must be positive")

    config = dwa.Config()
    config.robot_type = RobotType.rectangle
    state = np.array([0.0, 0.0, 0.0, 0.12, 0.05])
    trajectory = dwa.predict_trajectory(state, 0.12, 0.1, config)
    optimized_cost = dwa.calc_obstacle_cost
    print("Python {}; NumPy {}; synthetic scans, no ROS or GUI".format(
        sys.version.split()[0], np.__version__))
    print("scene,points,cost_old_ms,cost_new_ms,dwa_old_ms,dwa_new_ms,dwa_speedup")
    for scene, half_width in (("room", 1.0), ("corridor", 0.18)):
        for count in args.points:
            obstacles = room_obstacles(count, half_width)
            timings = []
            outputs = []
            for function in (reference_obstacle_cost, optimized_cost):
                cost_samples, dwa_samples = [], []
                with mock.patch.object(dwa, "calc_obstacle_cost", function):
                    dwa.dwa_control(state.copy(), obstacles, goal_heading="+x", config=config)
                    for _ in range(args.repeats):
                        start = time.perf_counter()
                        function(trajectory, obstacles, config)
                        middle = time.perf_counter()
                        output = dwa.dwa_control(state.copy(), obstacles, goal_heading="+x", config=config)
                        end = time.perf_counter()
                        cost_samples.append((middle - start) * 1000)
                        dwa_samples.append((end - middle) * 1000)
                    outputs.append(output)
                timings.append((float(np.median(cost_samples)), float(np.median(dwa_samples))))
            np.testing.assert_array_equal(outputs[0][0], outputs[1][0])
            np.testing.assert_array_equal(outputs[0][1], outputs[1][1])
            print("{},{},{:.3f},{:.3f},{:.3f},{:.3f},{:.2f}x".format(
                scene, count, timings[0][0], timings[1][0],
                timings[0][1], timings[1][1], timings[0][1] / timings[1][1]))


if __name__ == "__main__":
    if "--benchmark" in sys.argv:
        benchmark()
    else:
        unittest.main()
