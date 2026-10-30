import argparse
import configparser
import math
import os
import time
import numpy as np
import rclpy
from typing import List, Tuple
from rb2301_fp import dynamic_window_approach

from rclpy.node import Node
from rclpy.logging import set_logger_level, LoggingSeverity
from rclpy.qos import ReliabilityPolicy, QoSProfile
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan

np.set_printoptions(2, suppress=True, threshold=np.inf) # Print numpy arrays to specified d.p., suppress scientific notation and do not truncate
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
        self.sub_scan = self.create_subscription(LaserScan, "scan", self.sub_scan_callback, 2) # Subscribe to lidar
        self.is_simulation = None

    def sub_scan_callback(self, msg):
        """Scan subscriber"""
        if len(msg.ranges) <= 360:
            self.is_simulation = True
        else:
            self.is_simulation = False

class ObstacleCourseNode(Node):
    '''Node to navigate obstacle course, using pose from either gazebo odometer or optitrack and lidar scan data'''
    def __init__(self, is_simulation:bool=True, calibrate:bool=False):
        super().__init__('obstaclecourse')
        self.get_logger().info("Starting ObstacleCourseNode")

        self.is_simulation = is_simulation
        self.calibrate = calibrate # Real robot: only print poses, never drive
        self.raw_pose = None # Pose exactly as Optitrack reported it (real robot only)
        self.frame = self.course = None
        self._ticks = 0

        self.actual_velocity = None
        self.last_timestamp = None

        #DWA
        self._state = np.array([0.0, 0.0, 0.0 , 0.0, 0.0])  #state representation:
        #                      [x(m), y(m), yaw(rad), v(m/s), omega(rad/s)]
        self._goal = np.array([5.0, 0.0])

        if is_simulation:
            self._sub_goal = ['-y','+x']# sub_goal_heading, in reverse order
        else:
            self._sub_goal = ['+x','+y']
        self._trajectory = np.array(self._state)
        self._ob = np.array([]) #obstacle array
        self._is_dwa_initialized = False

        self.density_threshold = 0.05#the obstacle dot density threshhold for detecting deadend
        self.check_list_threshold = 25 # how many dot needed to start determine if it is a wall
        self.outlier_dot_threshold = 0.05 # The distance from the point with the outlier dot to the fitted line
        self.tilt_threshold = 5 # in angular
        self.close_threshold = 0.7# how close to a deadend triggers transformation to the next _sub_goal
        self.curve_threshold = 20# 
        self.sub_goal_cd =1.0# sub_goal switch cooldown,in seconds
        self.timestamp = 0.0
        self._segmentation = []
        self.deadend_check_gap = 10# how many times the timercallback run required for deadend check once
        self.timer_callback_counter = 1

        if is_simulation:
            self.max_translate_velocity = 0.5
        else:
            self.max_translate_velocity = 0.3 # Please keep this in place; 0.3m/s is more than fast enough

        self.sub_scan = self.create_subscription(LaserScan, "scan", self.sub_scan_callback, 2) # Subscribe to LiDAR scan data

        # Subscribe to the dynamic_pose topic from Gazebo that publishes ground-truth pose data
        if self.is_simulation:
            self.subscription = self.create_subscription(Odometry, 'odom', self.odometer_callback, 2)
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
        self.timer = self.create_timer(0.05, self.timer_callback)  # Runs at 20Hz. Can be changed.

        self.pose = None

    def sub_scan_callback(self, msg):
        """Scan subscriber"""
        if len(msg.ranges) <= 720:#was 360,choose whether want downsampling
            self.last_scan = np.array(msg.ranges)
        else:
            self.last_scan = np.array(msg.ranges)[::2]

        self.update_ob(msg)

    def update_ob(self,msg):
        if self.pose is None:
            return
        r = self.last_scan
        step = 2 if len(msg.ranges) > 720 else 1
        yaw = np.deg2rad(self.pose[2])
        # LiDAR mounting from nanocar_description.urdf: x=0.11423 m, yaw=3.1416 rad.
        a = msg.angle_min + np.arange(r.size) * msg.angle_increment * step + 3.1416 + yaw
        valid = np.isfinite(r) & (r >= msg.range_min) & (r <= msg.range_max)
        x = self.pose[0] + 0.11423 * np.cos(yaw)
        y = self.pose[1] + 0.11423 * np.sin(yaw)
        self._ob = np.column_stack((x + r[valid] * np.cos(a[valid]), y + r[valid] * np.sin(a[valid])))

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
        self.update_actual_velocity(msg.header.stamp)
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
        self.update_actual_velocity(msg.header.stamp)
        return self.pose

    def move_2D(self, x:float=0.0, y:float=0.0, turn:float=0.0):
        '''Publishes a Twist message to ROS to move a robot. Inputs are x and y linear velocities, as well as turn (z-axis yaw) angular velocity.'''
        twist_msg = Twist()
        x = np.clip(x, -self.max_translate_velocity, self.max_translate_velocity)
        y = np.clip(y, -self.max_translate_velocity, self.max_translate_velocity)
        turn = np.clip(turn, -self.max_translate_velocity*2, self.max_translate_velocity*2)
        twist_msg.linear.x, twist_msg.linear.y, twist_msg.linear.z = float(x), float(y), 0.0
        twist_msg.angular.x, twist_msg.angular.y, twist_msg.angular.z = 0.0, 0.0, float(turn)
        self.publisher_.publish(twist_msg)

    def is_goal_reached(self,threshold = 0.3)->bool:
        dist_to_goal = math.hypot(self._state[0] - self._goal[0], self._state[1] - self._goal[1])
        if dist_to_goal <= threshold:
            print("Goal!!")
            return True
        return False

    def update_state(self)->np.ndarray:
        """update speed and orientation using pose[], keep vel and omega"""
        return np.array([self.pose[0], self.pose[1], np.deg2rad(self.pose[2]), self._state[3], self._state[4]])

    def update_actual_velocity(self, stamp):
        """use differentiation of position to estimate the forward speed(m/s) and angular speed(rad/s)"""
        timestamp = stamp.sec + stamp.nanosec * 1e-9
        if self.last_timestamp is not None:
            dt = timestamp - self.last_timestamp
            if dt == 0:
                return  
            if dt < 0:
                self.actual_velocity = None
                self.filtered_forward_velocity = None
            else:
                dx, dy = self.pose[:2] - self.last_pose[:2]
                yaw = math.radians(self.pose[2])
                previous_yaw = math.radians(self.last_pose[2])
                delta_yaw = math.atan2(math.sin(yaw - previous_yaw),
                                       math.cos(yaw - previous_yaw))
                midpoint_yaw = previous_yaw + delta_yaw / 2
                v = (dx * math.cos(midpoint_yaw) + dy * math.sin(midpoint_yaw)) / dt
                self.actual_velocity = (v, delta_yaw / dt)
        self.last_pose = self.pose.copy()
        self.last_timestamp = timestamp


    def timer_callback(self):
        """Controller loop. Insert path planning and PID control logic here"""
        if self.pose is None:
            print("No pose detected")
            return # Does not run if no pose received from Odom or Optitrack

        if self.calibrate and self.raw_pose is not None:
            self._ticks += 1
            if self._ticks % 10 == 0: # twice a second
                ox, oy, oh = self.raw_pose
                #self.get_logger().info(f"optitrack ({ox:+.3f}, {oy:+.3f}, {oh:+7.1f} deg)  ->  maze ({self.pose[0]:+.3f}, {self.pose[1]:+.3f}, {self.pose[2]:+7.1f} deg)")
            return # Calibration mode: look, don't drive

        ###### INSERT CODE HERE ######
        #self.get_logger().info(f"Pose: {self.pose}")

        #try DWA integration

        #TODO give sub goal

        #TODO
        if self.pose is None or self._ob.size == 0 or self.actual_velocity is None:
            return
        self._state = self.update_state()

        if not self._sub_goal:
            self.move_2D()
            return

        self._segmentation = []

        if self.timer_callback_counter % self.deadend_check_gap == 0:
            self.timer_callback_counter = 1
            #self.get_logger().info('deadend check!')
            #Detect deadend_wall and change subgoal
            deadend_wall = self.deadend_detector()
            if deadend_wall is not None:
                distance_to_deadend_wall = self.measure_distance_dot_and_line(self.pose,deadend_wall)
                self.get_logger().info(f'wall detected!distance={distance_to_deadend_wall}')
                if distance_to_deadend_wall <= self.close_threshold and time.monotonic()>=self.timestamp+self.sub_goal_cd:
                    self._sub_goal.pop()
                    self.timestamp = time.monotonic()
                    if not self._sub_goal:
                        self.move_2D()
                        self.get_logger().info('All sub-goals completed')
                        return
                    else:
                        self.get_logger().info(f'switch to the next _sub_goal!{self._sub_goal[-1]}')

        else:
            self.timer_callback_counter+=1


        self._trajectory = np.vstack((self._trajectory, self._state))
        u, predicted_trajectory = dynamic_window_approach.dwa_control(
                                                                    x=self._state,
                                                                    #goal=self._goal,
                                                                    goal_heading= self._sub_goal[-1],
                                                                    ob=self._ob
                                                                    )
        self.move_2D(np.clip(u[0],0,self.max_translate_velocity),0.0,u[1])
        self._state[3] , self._state[4] = self.actual_velocity[0], self.actual_velocity[1]    #TODO change this to real speed
        #dynamic_window_approach.animate(predicted_trajectory,self._state,self._goal,self._ob, self._segmentation)

        ###### INSERT CODE HERE ######

    @staticmethod
    def measure_distance_dot_and_line(dot,line):
        A = line[0]
        B = line[1]
        C = line[2]
        _dot = np.asarray(dot)[:2]
        norm = np.hypot(A, B)
        if norm == 0:
            raise ValueError('A and B cannot both be zero')
        return float(np.abs(_dot @ np.array([A, B]) + C) / norm)

    @staticmethod
    def measure_angle(vector_1,vector_2):
            norm_product = np.linalg.norm(vector_1) * np.linalg.norm(vector_2)
            cos_theta = np.dot(vector_1, vector_2) / norm_product
            angle_deg = np.degrees(np.arccos(np.clip(cos_theta, -1.0, 1.0)))
            return angle_deg
    def deadend_detector(self):
        """
        利用ob数组有序的性质,检测阻挡子目标的那堵墙
        """
        
        if self.pose is None or self._ob.size == 0 or not self._sub_goal:
            return None

        def check_segment(segment,farthest_dot):
            if len(segment) < self.check_list_threshold:
                return None

            points = np.asarray(segment, dtype=float)
            center = points.mean(axis=0)
            centered = points - center

            _, s, vh = np.linalg.svd(centered, full_matrices=False)
            # 第一主方向沿着墙，第二主方向垂直于墙
            # vh[0]点云延伸最明显的方向,vh[1]法线方向
            #s[0]点云沿第1个方向的分散程度
            if s[0] < 1e-8:
                return None

            A, B = vh[1]  # 单位法向量，A²+B²=1
            C = -np.dot([A, B], center)

            distance = self.measure_distance_dot_and_line(farthest_dot,(A,B,C))
            self._segmentation.append(segment)  #add valid segment for animation  

            if distance > self.outlier_dot_threshold:
                #self.get_logger().warn(f'infeasible wall!too short:distance:{distance}')
                return None

            goal_direction = {
                '+x': (1, 0), '-x': (-1, 0),
                '+y': (0, 1), '-y': (0, -1),
            }[self._sub_goal[-1]]
            normal = np.array([A, B])
            # Reject a wall behind the robot along the current goal direction.
            robot_side = normal @ self.pose[:2] + C
            if robot_side * np.dot(normal, goal_direction) > 0:
                #self.get_logger().warn(f'infeasible wall!it is behind the robot')
                return None

            vector_1 = vh[0]
            if self._sub_goal[-1] == '+x' or self._sub_goal[-1]=='-x':
                vector_2 = np.array([1,0])
            else:
                vector_2 = np.array([0,1])

            angle_deg = self.measure_angle(vector_1,vector_2)
            # A blocking wall is perpendicular to travel; SVD signs are arbitrary.
            if abs(angle_deg - 90.0) > self.tilt_threshold:
                #self.get_logger().warn(f'infeasible wall!it is not in good angle')
                return None
            
            return (A,B,C)

        # Find the global extreme before accepting any candidate segment.
        farthest_dot=self.pose[:2]
        last_dot = None
        double_ob = np.concatenate((self._ob, self._ob), axis=0)
        for dot in double_ob:
            if self._sub_goal[-1] == '+x':# update the farthest dot
                if dot[0]>farthest_dot[0]:
                    farthest_dot = dot
            elif self._sub_goal[-1] == '-x':
                if dot[0]<farthest_dot[0]:
                    farthest_dot = dot
            elif self._sub_goal[-1] == '+y':
                if dot[1]>farthest_dot[1]:
                    farthest_dot = dot
            elif self._sub_goal[-1] == '-y':
                if dot[1]<farthest_dot[1]:
                    farthest_dot = dot
            else:
                raise ValueError('_sub_goal should only contain "+x","-x","+y"or"-y"')

        continuous_list=[]
        aver_hist_vector = None
        fracture = False
        for dot in double_ob:
            if last_dot is not None:
                distance = abs(last_dot[0]-dot[0]) + abs(last_dot[1]-dot[1])

                new_vector = np.asarray(last_dot)-np.asarray(dot)
                if aver_hist_vector is None and last_dot is not None:
                    aver_hist_vector = np.asarray(last_dot)-np.asarray(dot)
                if aver_hist_vector is not None:
                    angle = self.measure_angle(new_vector,aver_hist_vector)
                    if angle < self.curve_threshold:
                        aver_hist_vector= (aver_hist_vector+new_vector)/2
                        fracture = False
                    else:
                        fracture = True
                        aver_hist_vector = new_vector
            if (last_dot is not None and distance > self.density_threshold) or (aver_hist_vector is not None and fracture == True):  #once the continuous list fracture or reaches the end
                result = check_segment(continuous_list,farthest_dot)
                if result is not None:
                    return result
                continuous_list=[dot]
            else:
                continuous_list.append(dot)
            last_dot = dot
        return check_segment(continuous_list,farthest_dot)

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

    # Start spinning the obstacle node and only stop once SystemExit error is raised within the node callback
    obstacle = ObstacleCourseNode(is_simulation, cli_args.calibrate)
    try:
        rclpy.spin(obstacle)
    except SystemExit:
        print("Shutting down")
    obstacle.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
