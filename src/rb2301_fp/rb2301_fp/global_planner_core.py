"""
Global planner core: occupancy grid from LiDAR, inflation, A* and path following.

No ROS imports here, so everything can be unit-tested without a ROS graph and
runs on the robot's Foxy / Python 3.8. global_planner.py wraps this in a node;
obstacle_course.py uses PathFollower to hand DWA a waypoint ("carrot") on the path.

All coordinates are in the pose frame (Gazebo odom in simulation, CA2 maze frame
on the real robot). Grid arrays are indexed [row, col] = [iy, ix].
"""
import heapq
import math
from typing import List, Optional, Tuple

import numpy as np

#=============================== Parameters: edit here =========================================
# Defaults for the planner node (./gp.sh) and the controller's path following (./fp.sh).
# The ones marked (ROS) can also be overridden at startup with --ros-args -p name:=value.

# Map
GRID_RESOLUTION = 0.05          # [m] cell size (ROS: grid_resolution)
GRID_HALF_SIZE = 10.0           # [m] grid is 2x this wide, centred on the first pose (ROS: grid_half_size)
MAPPING_RANGE = 3.5             # [m] beams are cut here; no hit is marked beyond it (ROS: mapping_range)
MAX_MAPPING_YAW_RATE = 0.5      # [rad/s] skip scans while turning faster, against pose/scan skew (ROS: max_mapping_yaw_rate)
SCAN_STEP = 2                   # keep every Nth LiDAR beam (ROS: scan_step)
HIT_EVIDENCE = 0.85             # log-odds added to a cell a beam ends in
MISS_EVIDENCE = 0.4             # log-odds removed from a cell a beam passes through
MAX_EVIDENCE = 2.0              # log-odds clamp; lower = moved obstacles clear faster
OCCUPIED_THRESHOLD = 0.6        # log-odds above this is an obstacle (unknown = 0 = free)

# Inflation
ROBOT_HALF_WIDTH = 0.12         # [m] (ROS: robot_half_width)
INFLATION_MARGIN = 0.04         # [m] half width + margin is impassable (ROS: inflation_margin)
COST_RADIUS = 0.30              # [m] soft cost decays to 0 here; keeps paths mid-corridor (ROS: cost_radius)

# A*
COST_WEIGHT = 3.0               # step cost = length * (1 + COST_WEIGHT * soft cost) (ROS: cost_weight)
MAX_EXPANSIONS = 400000         # search limit per plan
PATH_SPACING = 0.1              # [m] spacing of the published path points (ROS: path_spacing)

# Node timing
MAP_UPDATE_RATE = 5.0           # [Hz] (ROS: map_update_rate)
PLAN_RATE = 1.0                 # [Hz] (ROS: plan_rate)
LOG_PERIOD = 10.0               # [s] between timing log lines

# LiDAR mounting from nanocar_description.urdf, same as ObstacleCourseNode.update_ob
LIDAR_OFFSET = 0.11423          # [m] forward of the pose
LIDAR_YAW = 3.1416              # [rad]

# Path following (controller side)
PATH_LOOKAHEAD = 0.8            # [m] along the path to the DWA waypoint (ROS on fp.sh: path_lookahead)
PATH_TIMEOUT = 3.0              # [s] older paths are ignored, DWA gets the final goal (ROS on fp.sh: path_timeout)
PATH_SEARCH_WINDOW = 3.0        # [m] closest path point is only searched this far along, so a folded path cannot jump ahead
PATH_GOAL_TOLERANCE = 0.3       # [m] a path ending this close to the final goal counts as reaching it
#================================================================================================

_SQRT2 = math.sqrt(2.0)


def resample_path(points:np.ndarray, spacing:float) -> np.ndarray:
    '''Resample a polyline (N x 2) at a fixed arc-length spacing, always keeping the last point.'''
    points = np.asarray(points, dtype=float).reshape(-1, 2)
    if len(points) < 2:
        return points.copy()
    arc = np.concatenate(([0.0], np.cumsum(np.hypot(*np.diff(points, axis=0).T))))
    if arc[-1] <= 0.0:
        return points[:1].copy()
    s = np.arange(0.0, arc[-1], spacing)
    if arc[-1] - s[-1] < 1e-9:
        s = s[:-1]
    s = np.append(s, arc[-1])
    return np.column_stack((np.interp(s, arc, points[:, 0]), np.interp(s, arc, points[:, 1])))


#============================================map==================================================================================
class OccupancyGrid():
    '''Fixed log-odds grid centred on a world point (normally the first pose).

    Hits add evidence, beams passing through a cell remove it, so a swinging
    obstacle that has moved away is cleared again. Unknown cells (log-odds 0)
    count as free.
    '''
    def __init__(self, center, resolution:float=GRID_RESOLUTION, half_size:float=GRID_HALF_SIZE,
                 hit_evidence:float=HIT_EVIDENCE, miss_evidence:float=MISS_EVIDENCE,
                 max_evidence:float=MAX_EVIDENCE, occupied_threshold:float=OCCUPIED_THRESHOLD):
        if resolution <= 0 or half_size <= 0:
            raise ValueError('resolution and half_size must be positive')
        self.resolution = float(resolution)
        self.size = int(math.ceil(2.0 * half_size / resolution))
        self.origin = np.asarray(center, dtype=float)[:2] - half_size # world position of the corner of cell (0, 0)
        self.hit_evidence = hit_evidence
        self.miss_evidence = miss_evidence
        self.max_evidence = max_evidence
        self.occupied_threshold = occupied_threshold
        self.log_odds = np.zeros((self.size, self.size), dtype=np.float32)

    @property
    def shape(self) -> Tuple[int, int]:
        return self.log_odds.shape

    def world_to_cell(self, points) -> np.ndarray:
        '''World points (N x 2) -> integer cells (N x 2) as [ix, iy]; may be out of bounds.'''
        points = np.asarray(points, dtype=float).reshape(-1, 2)
        return np.floor((points - self.origin) / self.resolution).astype(np.int64)

    def cell_to_world(self, cells) -> np.ndarray:
        '''Cells (N x 2) as [ix, iy] -> world coordinates of the cell centres.'''
        cells = np.asarray(cells, dtype=float).reshape(-1, 2)
        return self.origin + (cells + 0.5) * self.resolution

    def in_bounds(self, cells) -> np.ndarray:
        cells = np.asarray(cells).reshape(-1, 2)
        return (cells >= 0).all(axis=1) & (cells < self.size).all(axis=1)

    def _flat_indices(self, cells) -> np.ndarray:
        '''Unique flat indices of the in-bounds cells.'''
        cells = cells[self.in_bounds(cells)]
        return np.unique(cells[:, 1] * self.size + cells[:, 0])

    def integrate_scan(self, sensor_xy, endpoints, is_hit) -> None:
        '''Add one scan.

        sensor_xy: LiDAR position in the world. endpoints (N x 2): end of each beam
        in the world. is_hit (N): True if the endpoint is a real return, False if
        the beam was cut at the mapping range (clear along it, but mark nothing).
        '''
        sensor_xy = np.asarray(sensor_xy, dtype=float)[:2]
        endpoints = np.asarray(endpoints, dtype=float).reshape(-1, 2)
        is_hit = np.asarray(is_hit, dtype=bool).reshape(-1)
        if len(endpoints) == 0:
            return
        delta = endpoints - sensor_xy
        length = np.hypot(delta[:, 0], delta[:, 1])

        # Free space: sample every beam at half-cell steps, stopping one cell short of its end.
        step = self.resolution / 2.0
        t = np.arange(int(math.ceil(length.max() / step)) + 1) * step
        along = t[None, :] < (length - self.resolution)[:, None]
        ratio = np.divide(t[None, :], length[:, None], out=np.zeros(along.shape), where=length[:, None] > 0)
        samples_x = sensor_xy[0] + delta[:, 0:1] * ratio
        samples_y = sensor_xy[1] + delta[:, 1:2] * ratio
        misses = self._flat_indices(self.world_to_cell(
            np.column_stack((samples_x[along], samples_y[along]))))
        hits = self._flat_indices(self.world_to_cell(endpoints[is_hit]))
        misses = np.setdiff1d(misses, hits, assume_unique=True)

        flat = self.log_odds.reshape(-1)
        flat[misses] -= self.miss_evidence
        flat[hits] += self.hit_evidence
        np.clip(self.log_odds, -self.max_evidence, self.max_evidence, out=self.log_odds)

    def occupied(self) -> np.ndarray:
        '''Boolean occupancy; unknown cells are free.'''
        return self.log_odds > self.occupied_threshold


class CostMap():
    '''Inflation of an occupancy grid without scipy.

    Cells within inflation_radius of an occupied cell are lethal. Between that and
    cost_radius the cost decays linearly from 1 to 0, which keeps paths in the
    middle of corridors.
    '''
    def __init__(self, resolution:float=GRID_RESOLUTION, inflation_radius:float=ROBOT_HALF_WIDTH + INFLATION_MARGIN,
                 cost_radius:float=COST_RADIUS):
        if cost_radius < inflation_radius:
            raise ValueError('cost_radius must be at least inflation_radius')
        self.resolution = resolution
        self.inflation_radius = inflation_radius
        self.cost_radius = cost_radius
        reach = int(math.floor(cost_radius / resolution))
        offsets = [(dx, dy, math.hypot(dx, dy) * resolution)
                   for dx in range(-reach, reach + 1) for dy in range(-reach, reach + 1)
                   if math.hypot(dx, dy) * resolution <= cost_radius]
        self._offsets = sorted(offsets, key=lambda o: o[2]) # nearest first: the first offset to reach a cell is its clearance
        self.clearance = None
        self.lethal = None
        self.soft_cost = None

    def update(self, occupied:np.ndarray) -> None:
        '''Recompute clearance (capped at cost_radius), lethal mask and soft cost.'''
        rows, cols = occupied.shape
        clearance = np.full(occupied.shape, np.inf)
        if occupied.any():
            # Only shift the bounding box of the occupied cells, grown by the reach.
            ys, xs = np.nonzero(occupied)
            reach = int(math.floor(self.cost_radius / self.resolution))
            y0, y1 = max(ys.min() - reach, 0), min(ys.max() + reach + 1, rows)
            x0, x1 = max(xs.min() - reach, 0), min(xs.max() + reach + 1, cols)
            source = occupied[y0:y1, x0:x1]
            window = clearance[y0:y1, x0:x1]
            h, w = source.shape
            for dx, dy, distance in self._offsets:
                # target[y, x] is set if source[y - dy, x - dx] is occupied
                ty0, ty1 = max(dy, 0), min(h + dy, h)
                tx0, tx1 = max(dx, 0), min(w + dx, w)
                if ty0 >= ty1 or tx0 >= tx1:
                    continue
                target = window[ty0:ty1, tx0:tx1]
                hit = source[ty0 - dy:ty1 - dy, tx0 - dx:tx1 - dx] & np.isinf(target)
                target[hit] = distance
        self.clearance = clearance
        self.lethal = clearance <= self.inflation_radius
        span = max(self.cost_radius - self.inflation_radius, 1e-9)
        self.soft_cost = np.where(np.isfinite(clearance),
                                  np.clip((self.cost_radius - clearance) / span, 0.0, 1.0), 0.0)


#============================================planner==============================================================================
class AStarPlanner():
    '''A* on an 8-connected grid with soft cost and an escape from inflation.

    Step cost = step length * (1 + cost_weight * soft_cost of the cell entered).
    '''
    def __init__(self, cost_weight:float=COST_WEIGHT, max_expansions:int=MAX_EXPANSIONS):
        self.cost_weight = cost_weight
        self.max_expansions = max_expansions
        self.expansions = 0

    def plan(self, lethal:np.ndarray, soft_cost:np.ndarray, occupied:np.ndarray,
             start:Tuple[int, int], goal:Tuple[int, int]) -> Tuple[List[Tuple[int, int]], bool]:
        '''Cells [ix, iy] from start towards goal, and whether the goal itself was reached.

        If start is inside inflation, first take the shortest way out of it. If goal
        is blocked or unreachable, return the path to the reachable cell closest to it.
        '''
        rows, cols = lethal.shape
        blocked = lethal.copy()
        # A blocked border ring means neighbour indices never wrap around a row.
        blocked[0, :] = blocked[-1, :] = blocked[:, 0] = blocked[:, -1] = True

        escape = []
        if blocked[start[1], start[0]]:
            escape = self._escape(blocked, occupied, start)
            if not escape:
                return [tuple(start)], False
            start = escape.pop()

        target = tuple(goal)
        if not (0 <= goal[0] < cols and 0 <= goal[1] < rows) or blocked[goal[1], goal[0]]:
            target = self._nearest_free(blocked, goal)
        cells, reached = self._search(blocked, soft_cost, start, target, goal)
        reached = reached and target == tuple(goal)
        return escape + cells, reached

    def _nearest_free(self, blocked:np.ndarray, cell) -> Optional[Tuple[int, int]]:
        free_y, free_x = np.nonzero(~blocked)
        if free_x.size == 0:
            return None
        i = int(np.argmin((free_x - cell[0]) ** 2 + (free_y - cell[1]) ** 2))
        return int(free_x[i]), int(free_y[i])

    def _escape(self, blocked:np.ndarray, occupied:np.ndarray, start) -> List[Tuple[int, int]]:
        '''Breadth-first search through non-occupied cells to the nearest unblocked cell.'''
        rows, cols = blocked.shape
        start = (int(start[0]), int(start[1]))
        parent = {start: None}
        frontier = [start]
        while frontier:
            next_frontier = []
            for x, y in frontier:
                for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1)):
                    cell = (x + dx, y + dy)
                    if cell in parent or not (1 <= cell[0] < cols - 1 and 1 <= cell[1] < rows - 1):
                        continue
                    if occupied[cell[1], cell[0]]:
                        continue
                    parent[cell] = (x, y)
                    if not blocked[cell[1], cell[0]]:
                        path = [cell]
                        while parent[path[-1]] is not None:
                            path.append(parent[path[-1]])
                        return path[::-1]
                    next_frontier.append(cell)
            frontier = next_frontier
        return []

    def _search(self, blocked, soft_cost, start, target, original_goal):
        rows, cols = blocked.shape
        start = (int(start[0]), int(start[1]))
        if target is None:
            return [start], False
        free = (~blocked).reshape(-1).tolist()
        base = (1.0 + self.cost_weight * soft_cost).reshape(-1).tolist()
        tx, ty = target
        gx, gy = original_goal
        start_index = start[1] * cols + start[0]
        target_index = ty * cols + tx
        neighbours = [(1, 1.0), (-1, 1.0), (cols, 1.0), (-cols, 1.0),
                      (cols + 1, _SQRT2), (cols - 1, _SQRT2), (-cols + 1, _SQRT2), (-cols - 1, _SQRT2)]

        def heuristic(index):
            dx = abs(index % cols - tx)
            dy = abs(index // cols - ty)
            return max(dx, dy) + (_SQRT2 - 1.0) * min(dx, dy)

        cost = {start_index: 0.0}
        parent = {start_index: -1}
        best_index = start_index
        best_distance = math.hypot(start[0] - gx, start[1] - gy)
        open_heap = [(heuristic(start_index), 0.0, start_index)]
        self.expansions = 0
        reached = False
        while open_heap and self.expansions < self.max_expansions:
            _, g, index = heapq.heappop(open_heap)
            if g > cost[index]:
                continue # stale heap entry
            self.expansions += 1
            distance = math.hypot(index % cols - gx, index // cols - gy)
            if distance < best_distance:
                best_distance, best_index = distance, index
            if index == target_index:
                reached = True
                best_index = index
                break
            for offset, length in neighbours:
                n = index + offset
                if not free[n]:
                    continue
                new_cost = g + length * base[n]
                if new_cost < cost.get(n, math.inf):
                    cost[n] = new_cost
                    parent[n] = index
                    heapq.heappush(open_heap, (new_cost + heuristic(n), new_cost, n))

        cells = []
        index = best_index
        while index != -1:
            cells.append((index % cols, index // cols))
            index = parent[index]
        return cells[::-1], reached


class GlobalPlanner():
    '''Map + inflation + A*: the whole planner behind one object (used by the ROS node).'''
    def __init__(self, center, resolution:float=GRID_RESOLUTION, half_size:float=GRID_HALF_SIZE,
                 robot_half_width:float=ROBOT_HALF_WIDTH, inflation_margin:float=INFLATION_MARGIN,
                 cost_radius:float=COST_RADIUS, cost_weight:float=COST_WEIGHT, path_spacing:float=PATH_SPACING):
        self.grid = OccupancyGrid(center, resolution, half_size)
        self.costmap = CostMap(resolution, robot_half_width + inflation_margin, cost_radius)
        self.astar = AStarPlanner(cost_weight)
        self.path_spacing = path_spacing
        self.goal_reached = False # whether the last path ends at the goal itself
        self._costmap_dirty = True

    def update_map(self, sensor_xy, endpoints, is_hit) -> None:
        self.grid.integrate_scan(sensor_xy, endpoints, is_hit)
        self._costmap_dirty = True

    def update_costmap(self) -> None:
        if self._costmap_dirty:
            self.costmap.update(self.grid.occupied())
            self._costmap_dirty = False

    def plan(self, start_xy, goal_xy) -> np.ndarray:
        '''World path (N x 2) at path_spacing from start towards goal; empty if the start is off the grid.'''
        self.update_costmap()
        start_xy = np.asarray(start_xy, dtype=float)[:2]
        goal_xy = np.asarray(goal_xy, dtype=float)[:2]
        start = self.grid.world_to_cell(start_xy)[0]
        if not self.grid.in_bounds(start)[0]:
            self.goal_reached = False
            return np.empty((0, 2))
        goal = self.grid.world_to_cell(goal_xy)[0]
        cells, self.goal_reached = self.astar.plan(
            self.costmap.lethal, self.costmap.soft_cost, self.grid.occupied(),
            (int(start[0]), int(start[1])), (int(goal[0]), int(goal[1])))
        points = self.grid.cell_to_world(cells)
        points[0] = start_xy # start exactly at the robot, not at its cell centre
        if self.goal_reached:
            points = np.vstack((points[:-1], goal_xy)) if len(points) > 1 else np.vstack((points, goal_xy))
        return resample_path(points, self.path_spacing)


#============================================path following=======================================================================
class PathFollower():
    '''Controller side: pick the DWA goal ("carrot") on the latest global path.

    Falls back to the final goal when there is no path, the path is empty, or it
    is older than timeout. Times are in seconds from any monotonic clock.
    '''
    def __init__(self, lookahead:float=PATH_LOOKAHEAD, timeout:float=PATH_TIMEOUT,
                 search_window:float=PATH_SEARCH_WINDOW, goal_tolerance:float=PATH_GOAL_TOLERANCE):
        self.lookahead = lookahead
        self.timeout = timeout
        self.search_window = search_window
        self.goal_tolerance = goal_tolerance
        self.path = np.empty((0, 2))
        self.stamp = None
        self.waypoint = None # last waypoint taken from the path, None when falling back

    def set_path(self, points, stamp:float) -> None:
        self.path = np.asarray(points, dtype=float).reshape(-1, 2)
        self.stamp = stamp

    def has_fresh_path(self, now:float) -> bool:
        return self.stamp is not None and len(self.path) > 0 and now - self.stamp <= self.timeout

    def select_goal(self, position, final_goal, now:float) -> np.ndarray:
        '''Point to pass to DWA as goal: lookahead along the path, else the final goal.'''
        if final_goal is None or not self.has_fresh_path(now):
            self.waypoint = None
            return final_goal
        position = np.asarray(position, dtype=float)[:2]
        path = self.path
        arc = np.concatenate(([0.0], np.cumsum(np.hypot(*np.diff(path, axis=0).T)))) if len(path) > 1 else np.zeros(1)
        window = max(int(np.searchsorted(arc, self.search_window, side='right')), 1)
        closest = int(np.argmin(np.hypot(*(path[:window] - position).T)))
        target = arc[closest] + self.lookahead
        if target >= arc[-1]:
            end = path[-1]
            near_goal = math.hypot(end[0] - final_goal[0], end[1] - final_goal[1]) <= self.goal_tolerance
            self.waypoint = np.array(final_goal, dtype=float) if near_goal else end.copy()
        else:
            self.waypoint = np.array([np.interp(target, arc, path[:, 0]), np.interp(target, arc, path[:, 1])])
        return self.waypoint
