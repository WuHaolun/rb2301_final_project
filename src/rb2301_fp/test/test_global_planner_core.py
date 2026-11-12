"""Global planner core without ROS: map update, inflation, A* and path following."""
import ast
from pathlib import Path
import sys
import time
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rb2301_fp.global_planner_core import (GlobalPlanner, OccupancyGrid, PathFollower,
                                           resample_path)


def wall(start, end, spacing=0.02):
    '''Points along a straight wall, as a LiDAR would see it.'''
    start, end = np.asarray(start, float), np.asarray(end, float)
    n = max(int(np.ceil(np.linalg.norm(end - start) / spacing)), 1) + 1
    return start + np.linspace(0.0, 1.0, n)[:, None] * (end - start)


def add_walls(planner, walls, sensor=(0.0, 0.0), repeats=3):
    points = np.vstack(walls)
    for _ in range(repeats):
        planner.update_map(sensor, points, np.ones(len(points), bool))


def path_clearance(planner, path):
    '''Smallest distance from any path point to an occupied cell centre.'''
    cells = np.argwhere(planner.grid.occupied())[:, ::-1]
    occupied = planner.grid.cell_to_world(cells)
    return min(np.hypot(*(occupied - p).T).min() for p in path)


class OccupancyGridTests(unittest.TestCase):
    def test_hits_become_occupied_and_beams_clear_free_space(self):
        grid = OccupancyGrid((0.0, 0.0))
        grid.integrate_scan((0.0, 0.0), [[1.0, 0.0], [0.0, 1.5]], [True, True])
        occupied = grid.occupied()
        for point in ([1.0, 0.0], [0.0, 1.5]):
            ix, iy = grid.world_to_cell(point)[0]
            self.assertTrue(occupied[iy, ix])
        ix, iy = grid.world_to_cell([0.5, 0.0])[0]
        self.assertLess(grid.log_odds[iy, ix], 0.0)
        self.assertEqual(occupied.sum(), 2)

    def test_cut_beams_clear_but_do_not_mark(self):
        grid = OccupancyGrid((0.0, 0.0))
        grid.integrate_scan((0.0, 0.0), [[3.5, 0.0]], [False])
        self.assertFalse(grid.occupied().any())

    def test_obstacle_that_moved_away_is_cleared(self):
        grid = OccupancyGrid((0.0, 0.0))
        for _ in range(10): # saturated evidence
            grid.integrate_scan((0.0, 0.0), [[1.0, 0.0]], [True])
        ix, iy = grid.world_to_cell([1.0, 0.0])[0]
        self.assertTrue(grid.occupied()[iy, ix])
        for _ in range(4): # obstacle swung away: beams now hit the wall behind it
            grid.integrate_scan((0.0, 0.0), [[2.0, 0.0]], [True])
        self.assertFalse(grid.occupied()[iy, ix])


class MapValuesTests(unittest.TestCase):
    '''int8 values published as nav_msgs/OccupancyGrid for RViz.'''
    def test_occupancy_values_mark_unknown_free_and_occupied(self):
        grid = OccupancyGrid((0.0, 0.0))
        grid.integrate_scan((0.0, 0.0), [[1.0, 0.0]], [True])
        values = grid.to_map_values()
        self.assertEqual(values.dtype, np.int8)
        self.assertEqual(values.shape, grid.shape)
        ix, iy = grid.world_to_cell([1.0, 0.0])[0]
        fx, fy = grid.world_to_cell([0.5, 0.0])[0]
        ux, uy = grid.world_to_cell([0.0, 2.0])[0]
        self.assertGreater(values[iy, ix], 50) # hit, indexed [row=iy, col=ix]
        self.assertTrue(0 <= values[fy, fx] < 50) # beam passed through
        self.assertEqual(values[uy, ux], -1) # never seen
        grid.log_odds[:] = grid.max_evidence
        self.assertTrue((grid.to_map_values() == 100).all())

    def test_costmap_values_follow_rviz_costmap_scheme(self):
        planner = GlobalPlanner((0.0, 0.0))
        add_walls(planner, [wall((1.0, -1.0), (1.0, 1.0))])
        planner.update_costmap()
        occupied = planner.grid.occupied()
        values = planner.costmap.to_map_values(occupied)
        self.assertEqual(values.dtype, np.int8)
        self.assertTrue((values[occupied] == 100).all())
        self.assertTrue((values[planner.costmap.lethal & ~occupied] == 99).all())
        soft = (planner.costmap.soft_cost > 0) & ~planner.costmap.lethal
        self.assertTrue(soft.any())
        self.assertTrue(((values[soft] >= 1) & (values[soft] <= 98)).all())
        self.assertTrue((values[planner.costmap.soft_cost == 0] == 0).all())


class PlannerTests(unittest.TestCase):
    def test_goal_behind_u_wall_is_reached_around_it(self):
        planner = GlobalPlanner((0.0, 0.0))
        add_walls(planner, [wall((2.0, -1.0), (2.0, 1.0)),
                            wall((2.0, -1.0), (4.0, -1.0)), wall((2.0, 1.0), (4.0, 1.0))])
        path = planner.plan((0.0, 0.0), (3.0, 0.0))
        self.assertTrue(planner.goal_reached)
        np.testing.assert_allclose(path[0], [0.0, 0.0])
        np.testing.assert_allclose(path[-1], [3.0, 0.0])
        self.assertGreater(path[:, 0].max(), 4.0) # went round the open end
        self.assertGreater(path_clearance(planner, path), 0.16)
        steps = np.hypot(*np.diff(path, axis=0).T)
        self.assertLessEqual(steps.max(), 0.1 + 1e-9)

    def test_gap_of_0_6_m_is_crossed_with_default_inflation(self):
        planner = GlobalPlanner((0.0, 0.0))
        # Wall across x = 1 with a 0.6 m gate centred at y = 0.5; the rest is closed by side walls.
        add_walls(planner, [wall((1.0, -3.0), (1.0, 0.2)), wall((1.0, 0.8), (1.0, 3.0)),
                            wall((-1.0, -3.0), (1.0, -3.0)), wall((-1.0, 3.0), (1.0, 3.0)),
                            wall((-1.0, -3.0), (-1.0, 3.0))])
        path = planner.plan((0.0, 0.0), (2.0, 0.0))
        self.assertTrue(planner.goal_reached)
        crossing = path[np.argmin(np.abs(path[:, 0] - 1.0))]
        self.assertTrue(0.2 < crossing[1] < 0.8)
        self.assertGreater(path_clearance(planner, path), 0.16)

    def test_goal_inside_wall_ends_at_nearest_reachable_cell(self):
        planner = GlobalPlanner((0.0, 0.0))
        add_walls(planner, [wall((2.0, y), (2.6, y)) for y in np.arange(-1.0, 1.01, 0.04)])
        path = planner.plan((0.0, 0.0), (2.3, 0.0))
        self.assertFalse(planner.goal_reached)
        self.assertLess(np.hypot(*(path[-1] - [2.3, 0.0])), 0.5)
        ix, iy = planner.grid.world_to_cell(path[-1])[0]
        self.assertFalse(planner.costmap.lethal[iy, ix])

    def test_start_inside_inflation_plans_out(self):
        planner = GlobalPlanner((0.0, 0.0))
        add_walls(planner, [wall((0.1, -1.0), (0.1, 1.0))], sensor=(-0.5, 0.0))
        path = planner.plan((0.0, 0.0), (-2.0, 0.0))
        self.assertTrue(planner.goal_reached)
        np.testing.assert_allclose(path[-1], [-2.0, 0.0])

    def test_planning_time_on_400_by_400_grid(self):
        planner = GlobalPlanner((0.0, 0.0), resolution=0.05, half_size=10.0)
        self.assertEqual(planner.grid.shape, (400, 400))
        # Zig-zag corridor: alternate walls with openings on opposite sides.
        walls = [wall((-9.0, -1.0), (9.0, -1.0)), wall((-9.0, 9.0), (9.0, 9.0))]
        for i, x in enumerate(np.arange(-6.0, 8.0, 2.0)):
            walls.append(wall((x, -1.0), (x, 7.5)) if i % 2 == 0 else wall((x, 0.5), (x, 9.0)))
        points = np.vstack(walls)
        started = time.perf_counter()
        planner.update_map((0.0, 4.0), points, np.ones(len(points), bool))
        map_ms = (time.perf_counter() - started) * 1000.0
        started = time.perf_counter()
        planner.update_costmap()
        inflate_ms = (time.perf_counter() - started) * 1000.0
        started = time.perf_counter()
        path = planner.plan((-8.0, 4.0), (8.5, 4.0))
        plan_ms = (time.perf_counter() - started) * 1000.0
        self.assertTrue(planner.goal_reached)
        print(f'\n400x400 grid: map update {map_ms:.1f} ms ({len(points)} beams), '
              f'inflation {inflate_ms:.1f} ms, A* {plan_ms:.1f} ms '
              f'({planner.astar.expansions} expansions, path {len(path)} points)')


class PathFollowerTests(unittest.TestCase):
    def setUp(self):
        self.follower = PathFollower(lookahead=0.8, timeout=3.0)
        self.goal = np.array([2.0, 1.0])
        # L-shaped path: 2 m along x, then 1 m along y to the goal.
        self.follower.set_path(resample_path([[0.0, 0.0], [2.0, 0.0], [2.0, 1.0]], 0.1), stamp=10.0)

    def test_waypoint_is_lookahead_along_path(self):
        np.testing.assert_allclose(self.follower.select_goal([0.0, 0.05], self.goal, 10.5), [0.8, 0.0])
        np.testing.assert_allclose(self.follower.select_goal([1.6, 0.0], self.goal, 10.5), [2.0, 0.4], atol=1e-9)

    def test_end_of_path_gives_final_goal(self):
        np.testing.assert_allclose(self.follower.select_goal([2.0, 0.6], self.goal, 10.5), self.goal)

    def test_missing_empty_or_stale_path_falls_back_to_final_goal(self):
        self.assertIs(PathFollower().select_goal([0.0, 0.0], self.goal, 0.0), self.goal)
        self.assertIs(self.follower.select_goal([0.0, 0.0], self.goal, 13.5), self.goal)
        self.assertIsNone(self.follower.waypoint)
        self.follower.set_path([], stamp=20.0)
        self.assertIs(self.follower.select_goal([0.0, 0.0], self.goal, 20.0), self.goal)


class Python38Tests(unittest.TestCase):
    def test_new_modules_parse_with_foxy_python38(self):
        package = Path(__file__).resolve().parents[1] / 'rb2301_fp'
        for name in ('global_planner_core.py', 'global_planner.py', 'obstacle_course.py'):
            ast.parse((package / name).read_text(), feature_version=(3, 8))


if __name__ == '__main__':
    unittest.main()
