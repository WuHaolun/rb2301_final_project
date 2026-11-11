"""
Global planner node: occupancy grid from LiDAR + pose, A* to the controller's goal.

Runs as its own process (./gp.sh) so it lands on a different core from the
controller. The planning itself lives in global_planner_core.py (no ROS).

Subscribes: scan, odom (simulation) or the Optitrack pose (real robot), global_goal
Publishes:  global_path (nav_msgs/Path, frame id = visualization_frame)
"""
import math
import os
import time
import numpy as np
import rclpy

from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, ReliabilityPolicy, QoSProfile, qos_profile_sensor_data
from geometry_msgs.msg import PointStamped, PoseStamped
from nav_msgs.msg import Odometry, Path
from sensor_msgs.msg import LaserScan

from rb2301_fp import global_planner_core as params # all defaults live at the top of global_planner_core.py
from rb2301_fp.global_planner_core import GlobalPlanner
from rb2301_fp.obstacle_course import (LidarNode, load_frame, optitrack_to_maze,
                                       robot_number_from_domain, yaw_from_quaternion)


class GlobalPlannerNode(Node):
    '''Builds the map at map_update_rate and replans at plan_rate.'''
    def __init__(self, is_simulation:bool=True, *, parameter_overrides=None):
        super().__init__('globalplanner', parameter_overrides=parameter_overrides)
        self.get_logger().info("Starting GlobalPlannerNode")
        self.is_simulation = is_simulation

        #======Parameters======#
        self.resolution = self.declare_parameter('grid_resolution', params.GRID_RESOLUTION).value
        self.half_size = self.declare_parameter('grid_half_size', params.GRID_HALF_SIZE).value
        self.mapping_range = self.declare_parameter('mapping_range', params.MAPPING_RANGE).value
        self.max_mapping_yaw_rate = self.declare_parameter('max_mapping_yaw_rate', params.MAX_MAPPING_YAW_RATE).value
        self.robot_half_width = self.declare_parameter('robot_half_width', params.ROBOT_HALF_WIDTH).value
        self.inflation_margin = self.declare_parameter('inflation_margin', params.INFLATION_MARGIN).value
        self.cost_radius = self.declare_parameter('cost_radius', params.COST_RADIUS).value
        self.cost_weight = self.declare_parameter('cost_weight', params.COST_WEIGHT).value
        self.path_spacing = self.declare_parameter('path_spacing', params.PATH_SPACING).value
        self.scan_step = self.declare_parameter('scan_step', params.SCAN_STEP).value
        map_update_rate = self.declare_parameter('map_update_rate', params.MAP_UPDATE_RATE).value
        plan_rate = self.declare_parameter('plan_rate', params.PLAN_RATE).value
        self.frame_id = self.declare_parameter('visualization_frame', 'fp_visualization').value
        self.log_period = params.LOG_PERIOD

        #======State======#
        self.planner = None # GlobalPlanner, created at the first pose so the grid is centred on it
        self.pose = None # [x, y, heading_deg], same frame as the controller
        self.yaw_rate = 0.0 # [rad/s] from consecutive poses
        self.goal = None # final goal from the controller
        self.frame = None
        self._last_pose = self._last_pose_time = None
        self._scan = None # (msg, pose, yaw_rate) of the latest scan not yet in the map
        self._stats = {'map': [], 'plan': []} # seconds, since the last log line
        self._skipped_scans = 0

        #======ROS======#
        self.sub_scan = self.create_subscription(LaserScan, "scan", self.sub_scan_callback, qos_profile_sensor_data)
        if self.is_simulation:
            self.sub_pose = self.create_subscription(Odometry, 'odom', self.odometer_callback, qos_profile_sensor_data)
        else:
            self.robot_number = robot_number_from_domain()
            self.frame, _ = load_frame(self.robot_number)
            topic = f'/vrpn_mocap/bingda_{self.robot_number:03d}/pose'
            self.get_logger().info(f"Reading pose from {topic}")
            self.sub_pose = self.create_subscription(
                PoseStamped, topic, self.optitrack_callback,
                QoSProfile(depth=2, reliability=ReliabilityPolicy.BEST_EFFORT))
        latched = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE,
                             durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.sub_goal = self.create_subscription(PointStamped, 'global_goal', self.goal_callback, latched)
        self.path_publisher = self.create_publisher(Path, 'global_path', 10)

        self.map_timer = self.create_timer(1.0 / map_update_rate, self.map_timer_callback)
        self.plan_timer = self.create_timer(1.0 / plan_rate, self.plan_timer_callback)
        self.log_timer = self.create_timer(self.log_period, self.log_timer_callback)

#========Callbacks=======#
    def odometer_callback(self, msg:Odometry):
        '''Gazebo odometry -> pose [x, y, heading_deg].'''
        p = msg.pose.pose
        self.set_pose(p.position.x, p.position.y, math.degrees(yaw_from_quaternion(p.orientation)), msg.header.stamp)

    def optitrack_callback(self, msg:PoseStamped):
        '''Optitrack pose -> CA2 maze-frame pose, exactly as the controller does.'''
        heading = math.degrees(yaw_from_quaternion(msg.pose.orientation))
        x, y, heading = optitrack_to_maze(msg.pose.position.x, msg.pose.position.y, heading, self.frame)
        self.set_pose(x, y, heading, msg.header.stamp)

    def set_pose(self, x:float, y:float, heading_deg:float, stamp):
        pose = np.array((x, y, heading_deg))
        if not np.isfinite(pose).all():
            return
        timestamp = stamp.sec + stamp.nanosec * 1e-9
        if self._last_pose is not None and timestamp > self._last_pose_time:
            delta = math.radians(heading_deg - self._last_pose[2])
            delta = math.atan2(math.sin(delta), math.cos(delta))
            self.yaw_rate = delta / (timestamp - self._last_pose_time)
        self._last_pose, self._last_pose_time = pose, timestamp
        self.pose = pose
        if self.planner is None:
            self.planner = GlobalPlanner(
                pose[:2], self.resolution, self.half_size, self.robot_half_width,
                self.inflation_margin, self.cost_radius, self.cost_weight, self.path_spacing)
            self.get_logger().info(f'Map: {self.planner.grid.size}x{self.planner.grid.size} cells centred on {pose[:2]}')

    def sub_scan_callback(self, msg:LaserScan):
        '''Keep only the latest scan with the pose it was taken at; the map timer consumes it.'''
        if self.pose is None:
            return
        self._scan = (msg, self.pose.copy(), self.yaw_rate)

    def goal_callback(self, msg:PointStamped):
        goal = np.array((msg.point.x, msg.point.y))
        if self.goal is None or not np.allclose(goal, self.goal):
            self.get_logger().info(f'Global goal: {goal}')
        self.goal = goal

#========Map and planning=======#
    def scan_to_world(self, msg:LaserScan, pose):
        '''Returns (sensor_xy, endpoints, is_hit) in the pose frame; invalid/infinite beams are dropped.'''
        r = np.asarray(msg.ranges, dtype=float)[::self.scan_step]
        yaw = math.radians(pose[2])
        a = msg.angle_min + np.arange(r.size) * msg.angle_increment * self.scan_step + params.LIDAR_YAW + yaw
        valid = np.isfinite(r) & (r >= msg.range_min) & (r <= msg.range_max)
        r, a = r[valid], a[valid]
        is_hit = r <= self.mapping_range
        r = np.minimum(r, self.mapping_range)
        sensor = np.array((pose[0] + params.LIDAR_OFFSET * math.cos(yaw), pose[1] + params.LIDAR_OFFSET * math.sin(yaw)))
        return sensor, sensor + np.column_stack((r * np.cos(a), r * np.sin(a))), is_hit

    def map_timer_callback(self):
        if self._scan is None or self.planner is None:
            return
        msg, pose, yaw_rate = self._scan
        self._scan = None
        if abs(yaw_rate) > self.max_mapping_yaw_rate:
            self._skipped_scans += 1 # pose/scan timing skew smears the map while turning fast
            return
        started = time.perf_counter()
        self.planner.update_map(*self.scan_to_world(msg, pose))
        self._stats['map'].append(time.perf_counter() - started)

    def plan_timer_callback(self):
        path = np.empty((0, 2))
        if self.planner is not None and self.goal is not None and self.pose is not None:
            started = time.perf_counter()
            path = self.planner.plan(self.pose[:2], self.goal)
            self._stats['plan'].append(time.perf_counter() - started)
        self.publish_path(path)

    def publish_path(self, points):
        '''Publish points (N x 2) as nav_msgs/Path; an empty array publishes an empty path.'''
        msg = Path()
        msg.header.frame_id = self.frame_id
        msg.header.stamp = self.get_clock().now().to_msg()
        for x, y in points:
            pose = PoseStamped()
            pose.header = msg.header
            pose.pose.position.x, pose.pose.position.y = float(x), float(y)
            pose.pose.orientation.w = 1.0
            msg.poses.append(pose)
        self.path_publisher.publish(msg)

    def log_timer_callback(self):
        def summary(values):
            if not values:
                return 'none'
            return f'mean={np.mean(values) * 1000.0:.1f} ms, max={np.max(values) * 1000.0:.1f} ms (n={len(values)})'
        reached = self.planner is not None and self.planner.goal_reached
        self.get_logger().info(
            f"Performance: map_update {summary(self._stats['map'])}, plan {summary(self._stats['plan'])}, "
            f"skipped_scans={self._skipped_scans}, goal_reachable={reached}")
        self._stats = {'map': [], 'plan': []}
        self._skipped_scans = 0


def main(args=None):
    print("Starting global planner")
    rclpy.init(args=args)

    # Use lidar scan length to determine if in simulation or not, as the controller does
    lidar = LidarNode()
    while lidar.is_simulation is None:
        rclpy.spin_once(lidar)
    is_simulation = lidar.is_simulation
    lidar.destroy_node()

    planner = GlobalPlannerNode(is_simulation)
    try:
        rclpy.spin(planner)
    except (SystemExit, KeyboardInterrupt):
        print("Shutting down")
    planner.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
