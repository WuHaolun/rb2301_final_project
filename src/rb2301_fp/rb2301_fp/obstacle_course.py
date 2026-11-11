import argparse
import configparser
import math
import os
from collections import deque
from time import monotonic

import numpy as np
import rclpy

from rclpy.node import Node
from rclpy.logging import set_logger_level, LoggingSeverity
from rclpy.qos import DurabilityPolicy, ReliabilityPolicy, QoSProfile, qos_profile_sensor_data
from geometry_msgs.msg import Point, PoseStamped, Twist
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan
from std_msgs.msg import ColorRGBA
from visualization_msgs.msg import Marker, MarkerArray

from rb2301_fp.lidar_route import lidar_reference
from rb2301_fp.time_elastic_band import LIDAR_X_OFFSET, TebplanSolver

set_logger_level("obstaclecourse", level=LoggingSeverity.INFO) # Configure to either LoggingSeverity.INFO or LoggingSeverity.DEBUG

_PACKAGE_DIR = os.path.dirname(os.path.realpath(__file__))

def wrap_deg(angle:float) -> float:
    '''Wrap an angle in degrees into [-180, 180).'''
    return (angle + 180.0) % 360.0 - 180.0

def optitrack_to_maze(x:float, y:float, heading_deg:float, frame:dict) -> tuple:
    '''Optitrack-frame pose (x, y, heading in degrees) -> CA2 maze-frame pose (see optitrack_variables.config).'''
    dx, dy = x - frame["origin_x"], y - frame["origin_y"]
    r = math.radians(frame["rotation_deg"])
    c, s = math.cos(r), math.sin(r)
    return (c * dx + s * dy, -s * dx + c * dy,
            wrap_deg(heading_deg - frame["rotation_deg"] - frame["heading_offset_deg"]))

def maze_to_course(x:float, y:float, heading_deg:float, course:dict) -> tuple:
    '''CA2 maze-frame pose -> final-project course frame (origin/rotation of the course in the maze frame, [course] in the config).'''
    dx, dy = x - course["origin_x"], y - course["origin_y"]
    r = math.radians(course["rotation_deg"])
    c, s = math.cos(r), math.sin(r)
    return (c * dx + s * dy, -s * dx + c * dy, wrap_deg(heading_deg - course["rotation_deg"]))

def load_frame(robot_number:int) -> tuple:
    '''Read optitrack_variables.config: returns (frame, course) for this robot.'''
    parser = configparser.ConfigParser()
    path = os.path.join(_PACKAGE_DIR, "optitrack_variables.config")
    if not parser.read(path):
        raise FileNotFoundError(f"Could not read {path}")
    frame = {k: parser.getfloat("frame", k) for k in ("origin_x", "origin_y", "rotation_deg")}
    raw = parser.get("frame", "heading_offset_deg", fallback="auto").strip().lower()
    # Rigid bodies are created facing +y of the maze frame, so Motive's zero heading is maze +y (90 deg)
    frame["heading_offset_deg"] = wrap_deg(-(frame["rotation_deg"] + 90.0)) if raw == "auto" else float(raw)
    if parser.has_section("heading_offsets") and parser.has_option("heading_offsets", str(robot_number)):
        frame["heading_offset_deg"] = parser.getfloat("heading_offsets", str(robot_number))
    course = {k: parser.getfloat("course", k, fallback=0.0) for k in ("origin_x", "origin_y", "rotation_deg")}
    return frame, course


class LidarNode(Node):
    def __init__(self):
        """Node constructor"""
        super().__init__("LidarNode")
        self.sub_scan = self.create_subscription(LaserScan, "scan", self.sub_scan_callback, qos_profile_sensor_data) # Subscribe to lidar
        self.is_simulation = None

    def sub_scan_callback(self, msg):
        """Scan subscriber"""
        if len(msg.ranges) <= 360:
            self.is_simulation = True
        else:
            self.is_simulation = False

class ObstacleCourseNode(Node):
    '''Node to navigate obstacle course, using pose from either gazebo odometer or optitrack and lidar scan data'''
    def __init__(self, is_simulation:bool=True, calibrate:bool=False, *, parameter_overrides=None):
        super().__init__('obstaclecourse', parameter_overrides=parameter_overrides)
        self.get_logger().info("Starting ObstacleCourseNode")

        self.is_simulation = is_simulation

        self.calibrate = calibrate # Real robot: only print poses, never drive
        self.raw_pose = None # Pose exactly as Optitrack reported it (real robot only)
        self.frame = self.course = None
        self._ticks = 0

        self.pose = None
        self._ob = np.empty((0, 2))
        self._scan_received_at = None
        self._odom_history = deque(maxlen=32)
        self._pending_sim_scan = None
        self._goal_position = None
        self._teb_detour = None
        self._teb_candidate_index = 0
        self._point_goal_reached = False
        self._course_yaw = 0.0 if is_simulation else -math.pi / 2

        self.point_goal_distance = self.declare_parameter('point_goal_distance', 8.0).value
        self.point_goal_left_angle_deg = self.declare_parameter(
            'point_goal_left_angle_deg', -35.0).value
        if (not math.isfinite(self.point_goal_distance) or self.point_goal_distance <= 0
                or not math.isfinite(self.point_goal_left_angle_deg)):
            raise ValueError('point_goal_distance must be positive and goal settings must be finite')

        self.max_translate_velocity = 0.5 if is_simulation else 0.3
        self.max_angular_velocity = 2 * self.max_translate_velocity
        self._teb_solver = TebplanSolver(
            x0=np.zeros(3), xf=np.zeros(3), n=9,
            v_max=self.max_translate_velocity, omega_max=self.max_angular_velocity,
            T_max=0.6, w_kin=50.0,
        )
        self.footprint = self._teb_solver.footprint
        self.control_period = 0.05
        self.scan_timeout = 0.5

        self.sub_scan = self.create_subscription(LaserScan, "scan", self.sub_scan_callback, qos_profile_sensor_data) # Subscribe to LiDAR scan data

        # The simulator publishes ground-truth pose through odometry.
        if self.is_simulation:
            self.subscription = self.create_subscription(Odometry, 'odom', self.odometer_callback, qos_profile_sensor_data)
        else:
            qos_profile = QoSProfile(depth=2, reliability=ReliabilityPolicy.BEST_EFFORT)
            # On the Bingda robots ROS_DOMAIN_ID is the robot's number, so the Optitrack topic is picked automatically
            domain = os.environ.get("ROS_DOMAIN_ID", "")
            self.robot_number = int(domain) if domain.isdigit() and 1 <= int(domain) <= 99 else 3
            self.frame, self.course = load_frame(self.robot_number)
            self.get_logger().info(f"Reading pose from /vrpn_mocap/bingda_{self.robot_number:03d}/pose")

            self.map_sub = self.create_subscription(
                PoseStamped,
                f'/vrpn_mocap/bingda_{self.robot_number:03d}/pose',
                self.optitrack_callback,
                qos_profile
                )

        self.publisher_ = self.create_publisher(Twist, 'cmd_vel', 10) # Publish to cmd_vel node
        self.timer = self.create_timer(self.control_period, self.timer_callback)
        self.init_rviz_visualization()

    def init_rviz_visualization(self):
        """Choose direct ROS visualization or JSON snapshots for SSH viewing."""
        self._rviz_predicted_trajectory = None
        self._rviz_server = None
        enabled = self.declare_parameter('visualization_enabled', True).value
        self._rviz_frame = self.declare_parameter(
            'visualization_frame', 'fp_visualization').value
        transport = self.declare_parameter('visualization_transport', 'ros').value
        if not enabled:
            return
        if transport == 'ssh':
            from rb2301_fp.visualization_bridge import VisualizationSnapshotServer
            port = self.declare_parameter('visualization_port', 8765).value
            self._rviz_server = VisualizationSnapshotServer(port)
            self.get_logger().info(
                'SSH visualization: http://127.0.0.1:{}/snapshot'.format(self._rviz_server.port))
        elif transport == 'ros':
            # Keep the latest complete scene for RViz instances that join later.
            qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE,
                             durability=DurabilityPolicy.TRANSIENT_LOCAL)
            self._rviz_publisher = self.create_publisher(
                MarkerArray, '~/visualization', qos)
        else:
            raise ValueError('visualization_transport must be ros or ssh')
        self._rviz_timer = self.create_timer(0.1, self.publish_rviz_visualization)

    def _rviz_marker(self, name, marker_type, points, color, size, stamp, z=0.03):
        """Build one marker in controller coordinates; empty points delete it."""
        marker = Marker()
        marker.header.frame_id = self._rviz_frame
        marker.header.stamp = stamp
        marker.ns = name
        marker.id = 0
        marker.type = marker_type
        marker.pose.orientation.w = 1.0
        marker.scale.x = marker.scale.y = marker.scale.z = float(size)
        marker.color = ColorRGBA(r=color[0], g=color[1], b=color[2], a=1.0)
        marker.lifetime.sec = 1
        marker.points = [Point(x=float(p[0]), y=float(p[1]), z=z) for p in points]
        marker.action = Marker.ADD if marker.points else Marker.DELETE
        return marker

    def publish_rviz_visualization(self):
        """Show the TEB trajectory, robot, scan points and fixed goal.

        All data use self.pose's coordinates (Gazebo odom or Optitrack maze).
        The standalone fp_visualization frame needs no TF when used as RViz's
        Fixed Frame. This timer also runs while calibration/control is waiting.
        """
        if self.pose is None:
            return
        stamp = self.get_clock().now().to_msg()
        position = np.asarray(self.pose[:2])
        yaw = math.radians(self.pose[2])
        scene = MarkerArray()

        predicted = self._rviz_predicted_trajectory
        scene.markers.append(self._rviz_marker(
            'predicted_trajectory', Marker.LINE_STRIP,
            [] if predicted is None else predicted[:, :2],
            (0.0, 0.7, 0.0), 0.025, stamp))
        scene.markers.append(self._rviz_marker(
            'robot_position', Marker.SPHERE_LIST, [position],
            (1.0, 0.0, 0.0), 0.07, stamp, z=0.08))
        scene.markers.append(self._rviz_marker(
            'obstacles', Marker.POINTS, np.asarray(self._ob).reshape(-1, 2),
            (0.1, 0.1, 0.1), 0.035, stamp))

        # Blue cross marks the active point goal.
        goal = [] if self._goal_position is None else self._goal_position + np.array(
            [[-0.1, -0.1], [0.1, 0.1], [-0.1, 0.1], [0.1, -0.1]])
        scene.markers.append(self._rviz_marker(
            'reference_goal', Marker.LINE_LIST, goal,
            (0.0, 0.2, 1.0), 0.035, stamp))

        xmin, xmax, ymin, ymax = self.footprint.base_bounds
        outline = np.array([[xmin, ymax], [xmax, ymax], [xmax, ymin],
                            [xmin, ymin], [xmin, ymax]])
        rotation = np.array([[math.cos(yaw), -math.sin(yaw)],
                             [math.sin(yaw), math.cos(yaw)]])
        scene.markers.append(self._rviz_marker(
            'robot_footprint', Marker.LINE_STRIP, outline @ rotation.T + position,
            (0.1, 0.1, 0.1), 0.02, stamp, z=0.06))
        heading = [position, position + 0.5 * np.array([math.cos(yaw), math.sin(yaw)])]
        arrow = self._rviz_marker('robot_heading', Marker.ARROW, heading,
                                  (1.0, 0.0, 0.0), 0.04, stamp, z=0.1)
        arrow.scale.y, arrow.scale.z = 0.08, 0.12
        scene.markers.append(arrow)

        if self._rviz_server is not None:
            try:
                self._rviz_server.publish(scene)
            except (ValueError, TypeError) as error:
                self.get_logger().warning('Visualization frame skipped: {}'.format(error))
        else:
            self._rviz_publisher.publish(scene)

    def destroy_node(self):
        if self._rviz_server is not None:
            self._rviz_server.close()
        return super().destroy_node()


    @staticmethod
    def _stamp_ns(msg):
        stamp = msg.header.stamp
        return stamp.sec * 1_000_000_000 + stamp.nanosec

    def sub_scan_callback(self, msg):
        received_at = monotonic()
        if self.is_simulation:
            self._pending_sim_scan = (self._stamp_ns(msg), received_at, msg)
            self._project_pending_sim_scan()
        else:
            self._project_scan(msg, self.pose, received_at)

    def _project_pending_sim_scan(self):
        if self._pending_sim_scan is None or not self._odom_history:
            return
        scan_stamp, received_at, msg = self._pending_sim_scan
        if monotonic() - received_at > self.scan_timeout:
            self._pending_sim_scan = None
            return
        if scan_stamp > self._odom_history[-1][0]:
            return  # The scan arrived first; wait for odometry to bracket it.

        self._pending_sim_scan = None
        previous = None
        for stamp, pose in self._odom_history:
            if stamp == scan_stamp:
                self._project_scan(msg, pose, received_at)
                return
            if stamp > scan_stamp:
                if previous is not None:
                    previous_stamp, previous_pose = previous
                    gap = stamp - previous_stamp
                    # Gazebo odometry is configured at 50 Hz. Drop scans across long gaps.
                    if gap <= 100_000_000:
                        fraction = (scan_stamp - previous_stamp) / gap
                        delta = pose - previous_pose
                        delta[2] = wrap_deg(delta[2])
                        self._project_scan(msg, previous_pose + fraction * delta, received_at)
                return
            previous = stamp, pose

    def _project_scan(self, msg, pose, received_at):
        if pose is None or not np.isfinite(pose).all() or len(msg.ranges) == 0:
            self._scan_received_at = None
            return
        ranges = np.asarray(msg.ranges)
        valid = np.isfinite(ranges) & (ranges >= msg.range_min) & (ranges <= msg.range_max)
        yaw = math.radians(pose[2])
        # LiDAR mounting from nanocar_description.urdf.
        angles = msg.angle_min + np.flatnonzero(valid) * msg.angle_increment + 3.1416 + yaw
        origin = pose[:2] + LIDAR_X_OFFSET * np.array([math.cos(yaw), math.sin(yaw)])
        self._ob = origin + ranges[valid, None] * np.column_stack(
            (np.cos(angles), np.sin(angles)))
        # +inf is a valid no-return reading. An entirely invalid scan cannot guide motion.
        self._scan_received_at = received_at if np.any(valid | (ranges > msg.range_max)) else None

    def yaw_from_quaternion(self, q):
        '''Returns yaw angle (in rad) for orientation based on given quaternion input q'''
        siny_cosp = 2 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1 - 2 * (q.y * q.y + q.z * q.z)
        return np.arctan2(siny_cosp, cosy_cosp)

    def optitrack_callback(self, msg:PoseStamped):
        '''Callback to calculate 2D pose info from Optitrack node. Pose info includes x and y coordinates, as well as heading in degrees.
        This callback will run everytime the rclpy executor spins'''
        x, y = msg.pose.position.x, msg.pose.position.y
        heading = np.rad2deg(self.yaw_from_quaternion(msg.pose.orientation))
        self.raw_pose = np.array((x,y,heading))
        x, y, heading = optitrack_to_maze(x, y, heading, self.frame) # CA2 maze frame, see optitrack_variables.config
        self.pose = np.array((x,y,heading))
        self._initialize_point_goal()
        return self.pose

    def to_course(self, pose=None):
        '''Convert a pose [x, y, heading] (default: self.pose) from the CA2 maze frame to the final-project course frame. Optional; see [course] in optitrack_variables.config'''
        pose = self.pose if pose is None else pose
        return np.array(maze_to_course(pose[0], pose[1], pose[2], self.course)) if self.course else np.array(pose)

    def odometer_callback(self, msg):
        '''Callback to calculate 2D pose info from Gazebo odomoter. Pose info includes x and y coordinates, as well as heading in degrees.
        This callback will run everytime the rclpy executor spins'''
        latest_pose_msg = msg.pose.pose
        heading = np.rad2deg(self.yaw_from_quaternion(latest_pose_msg.orientation))
        self.pose = np.array((latest_pose_msg.position.x, latest_pose_msg.position.y, heading))
        self._initialize_point_goal()
        stamp = self._stamp_ns(msg)
        if self._odom_history and stamp < self._odom_history[-1][0]:
            # A Gazebo clock reset invalidates observations from the old timeline.
            self._odom_history.clear()
            self._pending_sim_scan = None
            self._scan_received_at = None
            self._ob = np.empty((0, 2))
        if self._odom_history and stamp == self._odom_history[-1][0]:
            self._odom_history.pop()
        self._odom_history.append((stamp, self.pose))
        self._project_pending_sim_scan()
        return self.pose

    def move_2D(self, x: float = 0.0, y: float = 0.0, turn: float = 0.0):
        twist = Twist()
        twist.linear.x = float(np.clip(x, -self.max_translate_velocity, self.max_translate_velocity))
        twist.linear.y = float(np.clip(y, -self.max_translate_velocity, self.max_translate_velocity))
        twist.angular.z = float(np.clip(turn, -self.max_angular_velocity, self.max_angular_velocity))
        self.publisher_.publish(twist)

    def _initialize_point_goal(self):
        """Freeze one target relative to the first valid pose and course direction."""
        if self._goal_position is not None or not np.isfinite(self.pose).all():
            return
        angle = self._course_yaw + math.radians(self.point_goal_left_angle_deg)
        offset = self.point_goal_distance * np.array([math.cos(angle), math.sin(angle)])
        self._goal_position = self.pose[:2] + offset
        self.get_logger().info(f'Fixed point goal: {self._goal_position}')

    def _teb_obstacles(self, start, target):
        """Downsample nearby points for the soft TEB objective."""
        cell_size = 0.12
        reach = max(0.5, self.footprint.radius)
        lower = np.minimum(start, target) - reach
        upper = np.maximum(start, target) + reach
        nearby = self._ob[np.all((self._ob >= lower) & (self._ob <= upper), axis=1)]
        _, representatives = np.unique(np.floor(nearby / cell_size), axis=0, return_index=True)
        return nearby[representatives]

    def _first_teb_command(self, trajectory, time_steps):
        """Convert the first optimized segment to forward and yaw velocities."""
        if (trajectory.ndim != 2 or trajectory.shape[1] != 3 or len(trajectory) < 2
                or time_steps.shape != (len(trajectory) - 1,)
                or not np.isfinite(trajectory).all() or not np.isfinite(time_steps).all()
                or np.any(time_steps <= 0)):
            raise ValueError('TEB returned an invalid trajectory or time sequence')

        dx, dy, yaw_change = trajectory[1] - trajectory[0]
        yaw_change = math.atan2(math.sin(yaw_change), math.cos(yaw_change))
        heading = trajectory[0, 2] + yaw_change / 2
        speed = (dx * math.cos(heading) + dy * math.sin(heading)) / time_steps[0]
        turn = yaw_change / time_steps[0]
        speed = float(np.clip(speed, -self.max_translate_velocity, self.max_translate_velocity))
        turn = float(np.clip(turn, -self.max_angular_velocity, self.max_angular_velocity))

        duration = max(time_steps[0], self.control_period)
        # Try a slower command before discarding an otherwise usable band.
        for scale in (1.0, 0.5, 0.25):
            candidate_speed, candidate_turn = speed * scale, turn * scale
            if not self._footprint_blocked(trajectory[0], candidate_speed, candidate_turn, duration):
                return candidate_speed, candidate_turn
        raise ValueError('TEB command intersects the LiDAR footprint clearance')

    def _footprint_blocked(self, pose, speed=0.0, turn=0.0, duration=0.0):
        """Check the moving rectangular footprint against the raw LiDAR points."""
        times = np.linspace(0, duration, 9)
        half_turns = turn * times / 2
        directions = pose[2] + half_turns
        distances = speed * times * np.sinc(half_turns / math.pi)
        centers = pose[:2] + distances[:, None] * np.column_stack(
            (np.cos(directions), np.sin(directions)))
        headings = pose[2] + 2 * half_turns
        # Bound how far any corner can move between the samples.
        padding = (abs(speed) + self.footprint.radius * abs(turn)) * (times[1] - times[0]) / 2
        return self._footprints_hit_scan(centers, headings, padding)

    def _footprints_hit_scan(self, centers, headings, padding):
        cos_yaw, sin_yaw = np.cos(headings)[:, None], np.sin(headings)[:, None]
        dx = self._ob[None, :, 0] - centers[:, None, 0]
        dy = self._ob[None, :, 1] - centers[:, None, 1]
        local_x = dx * cos_yaw + dy * sin_yaw
        local_y = -dx * sin_yaw + dy * cos_yaw
        xmin, xmax, ymin, ymax = self.footprint.base_bounds
        return np.any((local_x >= xmin - padding) & (local_x <= xmax + padding)
                      & (local_y >= ymin - padding) & (local_y <= ymax + padding))

    def _band_blocked(self, trajectory):
        """Check the swept footprint along every displayed trajectory segment."""
        for start, end in zip(trajectory[:-1], trajectory[1:]):
            delta = end - start
            delta[2] = math.atan2(math.sin(delta[2]), math.cos(delta[2]))
            corner_travel = np.linalg.norm(delta[:2]) + self.footprint.radius * abs(delta[2])
            intervals = max(1, math.ceil(corner_travel / 0.02))
            poses = start + np.linspace(0, 1, intervals + 1)[:, None] * delta
            if self._footprints_hit_scan(poses[:, :2], poses[:, 2], corner_travel / (2 * intervals)):
                return True
        return False

    def _plan_teb_command(self, start, offset, distance, scan_received_at):
        if self._footprint_blocked(start):
            raise ValueError('LiDAR point inside the current robot footprint')

        target_yaw = math.atan2(offset[1], offset[0])
        segments = self._teb_solver.n + 1
        candidates = []
        direct = np.array([*(start[:2] + offset * (min(distance, 0.8) / distance)), target_yaw])
        if self._band_blocked(np.array([start, direct])):
            reference = lidar_reference(
                start, start[:2] + offset, self._ob, self.footprint.radius, segments)
            if reference is not None and not self._band_blocked(reference):
                candidates.append((reference[-1], False, reference))
        # Keep following a selected detour until close enough to resume toward the main goal.
        if self._teb_detour is not None and np.linalg.norm(self._teb_detour[:2] - start[:2]) > 0.18:
            candidates.append((self._teb_detour, True, None))
        for angle_deg, reach in (
            (0, 0.8), (75, 0.55), (-75, 0.55),
            (0, 0.45), (30, 0.45), (-30, 0.45), (55, 0.45), (-55, 0.45),
            (0, 0.30), (30, 0.30), (-30, 0.30), (75, 0.30), (-75, 0.30),
        ):
            yaw = target_yaw + math.radians(angle_deg)
            target = start[:2] + min(distance, reach) * np.array([math.cos(yaw), math.sin(yaw)])
            candidates.append((np.array([*target, yaw]), angle_deg != 0, None))

        clear = [candidate for candidate in candidates if not self._footprint_blocked(candidate[0])]
        if not clear:
            self._teb_detour = None
            self._teb_candidate_index = 0
            x, y, yaw = start
            raise ValueError(
                f'all {len(candidates)} local TEB goals overlap LiDAR points '
                f'at pose ({x:.2f}, {y:.2f}, {math.degrees(yaw):.0f} deg)'
            )

        index = self._teb_candidate_index % len(clear)
        deadline = scan_received_at + self.scan_timeout
        error = 'scan too old for another TEB solve'
        for _ in range(min(2, len(clear))):
            # Leave time for one solve and return to the executor so scans can be processed.
            if deadline - monotonic() < 0.18:
                break
            goal, is_detour, reference = clear[index]
            try:
                trajectory, time_steps = self._teb_solver.solve(
                    x0=start, xf=goal, obstacles=self._teb_obstacles(start[:2], goal[:2]),
                    reference=reference,
                )
                speed, turn = self._first_teb_command(trajectory, time_steps)
                if self._band_blocked(trajectory):
                    raise ValueError('TEB trajectory crosses the LiDAR footprint clearance')
            except (RuntimeError, ValueError) as failure:
                error = str(failure)
                index = (index + 1) % len(clear)
                self._teb_candidate_index = index
                continue
            if monotonic() >= deadline:
                self._teb_candidate_index = (index + 1) % len(clear)
                raise ValueError('scan expired during the TEB solve')
            self._teb_detour = goal if is_detour else None
            self._teb_candidate_index = 0
            return trajectory, speed, turn
        raise ValueError(f'no usable local TEB trajectory ({len(clear)} clear goals, {error})')

    def timer_callback(self):
        if self.calibrate:
            if self.raw_pose is not None:
                self._ticks += 1
                if self._ticks % 10 == 0:
                    self.get_logger().info(
                        f'Optitrack pose: {self.raw_pose}; maze pose: {self.pose}')
            return

        self._rviz_predicted_trajectory = None
        if self._goal_position is None or not np.isfinite(self.pose).all():
            self.move_2D()
            return

        offset = self._goal_position - self.pose[:2]
        distance = np.linalg.norm(offset)
        if self._point_goal_reached or distance <= 0.3:
            if not self._point_goal_reached:
                self.get_logger().info('Point goal reached')
            self._point_goal_reached = True
            self.move_2D()
            return

        scan_received_at = self._scan_received_at
        if scan_received_at is None or monotonic() - scan_received_at > self.scan_timeout:
            self.move_2D()
            return

        start = np.array([*self.pose[:2], math.radians(self.pose[2])])
        try:
            trajectory, speed, turn = self._plan_teb_command(start, offset, distance, scan_received_at)
        except (RuntimeError, ValueError) as error:
            self.move_2D()
            self.get_logger().warning(f'TEB stopped: {error}', throttle_duration_sec=1.0)
            return
        if monotonic() - scan_received_at > self.scan_timeout:
            self.move_2D()
            return
        self._rviz_predicted_trajectory = trajectory
        self.move_2D(speed, turn=turn)


def main(args=None):
    arg_parser = argparse.ArgumentParser(description="RB2301 final project")
    arg_parser.add_argument('--calibrate', action='store_true', help="Real robot: only print the Optitrack pose and the maze-frame pose twice a second, never drive")
    cli_args, ros_args = arg_parser.parse_known_args(args=args)
    print("Starting obstacle course")
    rclpy.init(args=ros_args)

    # Use lidar scan length to determine if in simulation or not
    lidar = LidarNode()
    while lidar.is_simulation is None:
        rclpy.spin_once(lidar)
    is_simulation = lidar.is_simulation
    lidar.destroy_node()

    obstacle = ObstacleCourseNode(is_simulation, cli_args.calibrate)
    try:
        rclpy.spin(obstacle)
    except (KeyboardInterrupt, SystemExit):
        pass
    finally:
        if rclpy.ok() and not obstacle.calibrate:
            obstacle.move_2D()
        obstacle.destroy_node()
        rclpy.try_shutdown()

if __name__ == '__main__':
    main()
