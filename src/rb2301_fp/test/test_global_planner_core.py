"""Global planner core without ROS: map update, inflation, A* and path following."""
import ast
from pathlib import Path
import sys
import time
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rb2301_fp import global_planner_core as core
from rb2301_fp.global_planner_core import (GateDetector, GlobalPlanner, OccupancyGrid, PathFollower,
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


class GateScene():
    '''Grid built by writing log-odds directly, in a frame (u along the final direction, v sideways).'''
    def __init__(self, direction='+x'):
        self.direction = direction
        self.grid = OccupancyGrid((0.0, 0.0))

    def world(self, u, v):
        return {'+x': (u, v), '-x': (-u, v), '+y': (v, u), '-y': (v, -u)}[self.direction]

    def fill(self, u0, u1, v0, v1, value):
        '''Set every cell of a (u, v) rectangle: -1 observed free, 0 unknown.'''
        corners = self.grid.world_to_cell([self.world(u0, v0), self.world(u1, v1)])
        (x0, y0), (x1, y1) = corners.min(axis=0), corners.max(axis=0)
        self.grid.log_odds[y0:y1 + 1, x0:x1 + 1] = value

    def wall(self, u0, v0, u1, v1):
        points = [self.world(u, v) for u, v in wall((u0, v0), (u1, v1), spacing=0.01)]
        cells = self.grid.world_to_cell(points)
        self.grid.log_odds[cells[:, 1], cells[:, 0]] = self.grid.max_evidence

    def gate(self, width=0.7, center=0.3, at=4.0, side=(-1.5, 1.8)):
        '''Corridor along u with a cross wall at u = at and a gap of width centred at v = center.'''
        self.fill(2.0, 6.0, side[0], side[1], -1.0)
        for v in side:
            self.wall(2.0, v, 6.0, v)
        self.wall(at, side[0], at, center - width / 2.0)
        self.wall(at, center + width / 2.0, at, side[1])
        return self


def drive(detector, scene, us, v, arm_offset=1.0):
    '''Arm at a sideways offset (like the earlier course legs), then drive along u at v.

    Returns the update results and the positions of the drive.
    '''
    detector.update(scene.grid, scene.world(us[0], v - arm_offset))
    positions = [np.array(scene.world(u, v)) for u in us]
    return [detector.update(scene.grid, p) for p in positions], positions


FORWARD = np.arange(3.0, 5.2, 0.06) # through the gate wall at u = 4.0


class GateDetectorTests(unittest.TestCase):
    def assert_latches_at_gate(self, scene, v=0.3, center=(4.0, 0.3)):
        detector = GateDetector(scene.direction)
        results, positions = drive(detector, scene, FORWARD, v)
        self.assertTrue(results[-1])
        # Latches on the GATE_CONFIRM-th positive check of the first run of positives.
        probe = GateDetector(scene.direction)
        positive = [probe._find_gate(scene.grid, p) is not None for p in positions]
        first = positive.index(True)
        self.assertTrue(all(positive[first:first + core.GATE_CONFIRM]))
        latch = results.index(True)
        self.assertEqual(latch, first + core.GATE_CONFIRM - 1)
        self.assertFalse(any(results[:latch]))
        robot_u = FORWARD[latch]
        self.assertTrue(4.0 < robot_u < 4.0 + core.GATE_BACK_MAX + 0.1, robot_u)
        np.testing.assert_allclose(detector.gate_center, scene.world(*center),
                                   atol=scene.grid.resolution + 1e-9)
        return detector

    def assert_never(self, scene, us=FORWARD, v=0.3, arm_offset=1.0):
        detector = GateDetector(scene.direction)
        results, _ = drive(detector, scene, us, v, arm_offset)
        self.assertFalse(any(results))
        self.assertFalse(detector.latched)
        self.assertIsNone(detector.gate_center)

    def test_gates_of_0_6_to_0_8_m_latch_after_confirm(self):
        for width in (0.6, 0.7, 0.8):
            with self.subTest(width=width):
                detector = self.assert_latches_at_gate(GateScene().gate(width=width), v=0.32)
                self.assertAlmostEqual(detector.gate_width, width, delta=0.1)

    def test_robot_still_in_front_of_the_wall(self):
        self.assert_never(GateScene().gate(), us=np.arange(2.5, 3.95, 0.06))

    def test_corridor_with_side_walls_only(self):
        scene = GateScene()
        scene.fill(2.0, 6.0, 0.0, 0.6, -1.0)
        scene.wall(2.0, 0.0, 6.0, 0.0)
        scene.wall(2.0, 0.6, 6.0, 0.6)
        self.assert_never(scene)

    def test_solid_wall_behind_robot_after_corner(self):
        scene = GateScene()
        scene.fill(2.0, 6.0, -1.5, 1.8, -1.0)
        scene.wall(3.8, -1.5, 3.8, 1.8)
        scene.wall(3.8, -1.5, 6.0, -1.5)
        scene.wall(3.8, 1.8, 6.0, 1.8)
        self.assert_never(scene, us=np.arange(3.9, 5.2, 0.06))

    def test_small_obstacle_next_to_side_wall(self):
        scene = GateScene()
        scene.fill(2.0, 6.0, -1.5, 1.8, -1.0)
        for v in (-1.5, 1.8):
            scene.wall(2.0, v, 6.0, v)
        o0, o1 = -0.9, -0.83 # 7 cm obstacle, 0.6 m from the side wall
        scene.wall(3.965, o0, 4.035, o0)
        scene.wall(3.965, o1, 4.035, o1)
        scene.wall(3.965, o0, 3.965, o1)
        scene.wall(4.035, o0, 4.035, o1)
        self.assert_never(scene, v=-1.2)

    def test_box_next_to_side_wall(self):
        scene = GateScene()
        scene.fill(2.0, 6.0, -1.5, 1.8, -1.0)
        for v in (-1.5, 1.8):
            scene.wall(2.0, v, 6.0, v)
        scene.fill(3.8, 4.2, -0.9, -0.5, 0.0) # inside of the box is never seen
        o0, o1 = -0.9, -0.5 # 0.4 m box, 0.6 m from the side wall
        scene.wall(3.8, o0, 4.2, o0)
        scene.wall(3.8, o1, 4.2, o1)
        scene.wall(3.8, o0, 3.8, o1)
        scene.wall(4.2, o0, 4.2, o1)
        self.assert_never(scene, v=-1.2)

    def test_unobserved_gap_is_not_a_gate(self):
        scene = GateScene().gate()
        scene.fill(3.85, 4.15, -0.05, 0.65, 0.0)
        self.assert_never(scene)

    def test_wide_opening_is_not_a_gate(self):
        self.assert_never(GateScene().gate(width=1.2, side=(-2.5, 2.5)))

    def test_thick_offset_and_holed_walls_are_still_detected(self):
        thick = GateScene().gate()
        thick.wall(4.05, -1.5, 4.05, -0.05)
        thick.wall(4.05, 0.65, 4.05, 1.8)
        self.assert_latches_at_gate(thick, center=(4.025, 0.3))

        offset = GateScene()
        offset.fill(2.0, 6.0, -1.5, 1.8, -1.0)
        for v in (-1.5, 1.8):
            offset.wall(2.0, v, 6.0, v)
        offset.wall(4.0, -1.5, 4.0, -0.05)
        offset.wall(4.05, 0.65, 4.05, 1.8) # one cell further along
        self.assert_latches_at_gate(offset, center=(4.025, 0.3))

        holed = GateScene().gate()
        for v in np.arange(-0.05 - 0.12, -1.5, -0.15).tolist() + np.arange(0.65 + 0.12, 1.8, 0.15).tolist():
            holed.fill(4.0, 4.0, v, v, -1.0) # single-cell holes every third cell of both flanks
        self.assert_latches_at_gate(holed)

    def test_all_four_directions(self):
        for direction in ('+x', '-x', '+y', '-y'):
            with self.subTest(direction=direction):
                self.assert_latches_at_gate(GateScene(direction).gate())

    def test_not_armed_without_sideways_travel(self):
        self.assert_never(GateScene().gate(), arm_offset=0.5)
        detector = GateDetector('+x', arm_lateral=0.0) # arming disabled
        self.assertTrue(any(drive(detector, GateScene().gate(), FORWARD, 0.3, arm_offset=0.0)[0]))

    def test_crossing_against_the_final_direction(self):
        self.assert_never(GateScene().gate(), us=FORWARD[::-1])
        self.assert_never(GateScene('-y').gate(), us=FORWARD[::-1])

    def test_stays_latched(self):
        scene = GateScene().gate()
        detector = self.assert_latches_at_gate(scene)
        center = detector.gate_center.copy()
        empty = OccupancyGrid((0.0, 0.0))
        for position in ([0.0, 0.0], [4.5, 0.3], [-3.0, 7.0], [float('nan'), 0.0]):
            self.assertTrue(detector.update(empty, position))
        np.testing.assert_array_equal(detector.gate_center, center)

    def test_rejects_unknown_direction(self):
        with self.assertRaises(ValueError):
            GateDetector('x')

    def test_update_time(self):
        scene = GateScene().gate()
        detector = GateDetector('+x', confirm=10 ** 9) # never latches: every call runs the full check
        detector.update(scene.grid, scene.world(3.0, -0.7))
        positions = [scene.world(u, 0.3) for u in np.arange(3.0, 5.2, 0.02)]
        times, peak_hits = [], 0
        for p in positions:
            started = time.perf_counter()
            detector.update(scene.grid, p)
            times.append(time.perf_counter() - started)
            peak_hits = max(peak_hits, detector.hits)
        self.assertGreater(peak_hits, core.GATE_CONFIRM) # the gate was found on many calls
        mean_ms, max_ms = np.mean(times) * 1000.0, np.max(times) * 1000.0
        print(f'\nGateDetector.update on {scene.grid.shape[0]}x{scene.grid.shape[1]} grid: '
              f'mean {mean_ms * 1000.0:.0f} us, max {max_ms * 1000.0:.0f} us over {len(positions)} calls')
        self.assertLess(mean_ms, 1.0)


class Python38Tests(unittest.TestCase):
    def test_new_modules_parse_with_foxy_python38(self):
        package = Path(__file__).resolve().parents[1] / 'rb2301_fp'
        for name in ('global_planner_core.py', 'global_planner.py', 'obstacle_course.py',
                     'dynamic_window_approach.py'):
            ast.parse((package / name).read_text(), feature_version=(3, 8))


if __name__ == '__main__':
    unittest.main()
