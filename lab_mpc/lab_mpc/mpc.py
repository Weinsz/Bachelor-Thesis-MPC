#!/usr/bin/env python3
import math
from dataclasses import dataclass, field
import numpy as np
from scipy.linalg import block_diag
from scipy.sparse import block_diag, csc_matrix, diags
from scipy.spatial import transform
import signal
import sys
import time
from datetime import timedelta
import pandas as pd

import cvxpy
from lab_mpc.utils import nearest_point

import rclpy
from rclpy.node import Node
from ackermann_msgs.msg import AckermannDriveStamped
from geometry_msgs.msg import PoseStamped, Point, PoseWithCovarianceStamped
from nav_msgs.msg import Odometry
from visualization_msgs.msg import Marker



@dataclass
class mpc_config:
    NXK: int = 4  # length of kinematic state vector: z = [x, y, v, yaw]
    NU: int = 2  # length of input vector: u = = [steering speed, acceleration]
    TK: int = 5  # finite time horizon length kinematic

    # ---------------------------------------------------
    # Penalties for tuning the matrices

    Rk: np.ndarray = field(default_factory=lambda: np.diag([0.001, 90.0]))  # input cost matrix, penalty for inputs: [accel, steering_speed]
    Rdk: np.ndarray = field(default_factory=lambda: np.diag([0.1, 600.0]))  # input difference cost matrix, penalty for change of inputs: [accel, steering_speed]
    # state error cost matrix, for the the next (T) prediction time steps [x, y, delta, v, yaw, yaw-rate, beta]
    # state error: [x, y, v, yaw]
    Qk: np.ndarray = field(default_factory=lambda: np.diag([70.0, 70.0, 30.0, 70.0]))  
    # final state error matrix, penalty  for the final state constraints: [x, y, delta, v, yaw, yaw-rate, beta]
    # final state error: [x, y, v, yaw]
    Qfk: np.ndarray = field(default_factory=lambda: np.diag([70.0, 70.0, 30.0, 70.0]))  # final state error matrix
    # ---------------------------------------------------
    #N_IND_SEARCH: int = 20  # Search index number
    DTK: float = 0.03  # time step [s] kinematic
    dlk: float = 0.03  # dist step [m] kinematic
    LENGTH: float = 0.58  # Length of the vehicle [m]
    WIDTH: float = 0.31  # Width of the vehicle [m]
    WB: float = 0.33  # Wheelbase [m]
    MIN_STEER: float = -0.4189  # maximum steering angle [rad]
    MAX_STEER: float = 0.4189  # maximum steering angle [rad]
    MAX_DSTEER: float = np.deg2rad(180.0)  # maximum steering speed [rad/s]
    MAX_SPEED: float = 15.0  # maximum speed [m/s]
    MIN_SPEED: float = 0.0  # minimum backward speed [m/s]
    MAX_ACCEL: float = 3.0  # maximum acceleration [m/ss]


@dataclass
class State:
    x: float = 0.0
    y: float = 0.0
    delta: float = 0.0
    v: float = 0.0
    yaw: float = 0.0
    yawrate: float = 0.0
    beta: float = 0.0

class MPC(Node):
    map_name: str
    profile: str
    def stop(self, x, y):
        self.log("STOPPING CAR")
        stop = AckermannDriveStamped()
        stop.drive.speed = 0.0
        self._drive.publish(stop)
        m = "monza"
        prof = "hp"
        if "spa" == self.map_name:
            m = "spa"
        if "fast" == self.profile:
            prof = "fast"
        elif "safe" == self.profile:
            prof = "safe"
        self.exp_df.to_csv("/home/weins/sim_ws/csv/mpc_" + m + "_" + prof + "_controls_out.csv", header=['x', 'y', 'theta', 'speed', 'accel'], index=False)
        print("exporting csv")
        sys.exit(0)

    def __init__(self, map_name, profile):
        """
        Initialize the MPC (Model Predictive Control) node for the F1TENTH racecar.

        Args:
            map_name (str): The name of the map/racetrack to be used.
            profile (str): The profile configuration for the MPC algorithm.
        
        This constructor sets up the ROS node, initializes parameters, creates subscribers and publishers,
        loads waypoints from a specified file and configures the MPC optimization problem.
        """
        super().__init__('mpc_node')
        # Create ROS subscribers and publishers
        # Use MPC as a path tracker
        self.log = self.get_logger().info
        signal.signal(signal.SIGINT, self.stop)
        signal.signal(signal.SIGSEGV, self.stop)

        # Config parameters
        self.declare_parameter('TK', 5)
        self.declare_parameter('DTK', 0.03)
        self.declare_parameter('MAX_DSTEER', np.deg2rad(180.0))
        self.declare_parameter('MAX_SPEED', 15.0)
        self.declare_parameter('MAX_ACCEL', 3.0)
        self.declare_parameter('Rk', [0.001, 90.0])
        self.declare_parameter('Rdk', [0.1, 600.0])
        self.declare_parameter('Qk', [70.0, 70.0, 30.0, 70.0])
        self.declare_parameter('Qfk', [70.0, 70.0, 30.0, 70.0])
        self.declare_parameter('map', '/home/weins/sim_ws/csv/traj_race_spa_v2.csv')
        self.declare_parameter('fast_start', True)

        self.is_real = False
        self.fast_start = self.get_parameter('fast_start').get_parameter_value().bool_value
        loc_topic = '/pf/viz/inferred_pose' if self.is_real else '/ego_racecar/odom'
        drive_topic = 'drive'
        self._drive = self.create_publisher(AckermannDriveStamped, drive_topic, 1)

        self.drive = AckermannDriveStamped()
        self._pose = self.create_subscription(
            PoseStamped if self.is_real else Odometry, loc_topic, self.pose_callback, 1)

        # Visualization for ROS & RViz 
        vis_ref_traj_topic = "/ref_traj_marker"
        vis_waypoints_topic = "/waypoints_marker"
        vis_pred_path_topic = "/pred_path_marker"
        self.vis_waypoints_pub = self.create_publisher(Marker, vis_waypoints_topic, 1)
        self.vis_waypoints_msg = Marker()
        self.vis_ref_traj_pub = self.create_publisher(Marker, vis_ref_traj_topic, 1)
        self.vis_ref_traj_msg = Marker()
        self.vis_pred_path_pub = self.create_publisher(Marker, vis_pred_path_topic, 1)
        self.vis_pred_path_msg = Marker()

        self.map_name = map_name
        self.profile = profile
        file = self.get_parameter('map').get_parameter_value().string_value
        #Levine:
        #self.wpts = np.loadtxt(file, delimiter=';', skiprows=0)
        #self.wpts[:, 3] += math.pi / 2
        self.wpts = np.loadtxt(file, delimiter=',', skiprows=0)
        self.wpts[:, 2] += math.pi / 2
        self.display_waypoints()

        # Start from the same position, just like in a race
        set_initial = self.create_publisher(PoseWithCovarianceStamped, '/initialpose', 1)
        initial_msg = PoseWithCovarianceStamped()
        initial_msg.header.frame_id = 'map'
        initial_msg.pose.pose.position.x = self.wpts[0, 0]
        initial_msg.pose.pose.position.y = self.wpts[0, 1]
        initial_msg.pose.pose.position.z = 0.0
        quat = transform.Rotation.from_euler('z', self.wpts[0, 2]).as_quat()
        initial_msg.pose.pose.orientation.x = quat[0]
        initial_msg.pose.pose.orientation.y = quat[1]
        initial_msg.pose.pose.orientation.z = quat[2]
        initial_msg.pose.pose.orientation.w = quat[3]
        set_initial.publish(initial_msg)

        # Load MPC Configuration
        self.config = mpc_config(
            TK=self.get_parameter('TK').get_parameter_value().integer_value,
            DTK=self.get_parameter('DTK').get_parameter_value().double_value,
            MAX_DSTEER=self.get_parameter('MAX_DSTEER').get_parameter_value().double_value,
            MAX_SPEED=self.get_parameter('MAX_SPEED').get_parameter_value().double_value,
            MAX_ACCEL=self.get_parameter('MAX_ACCEL').get_parameter_value().double_value,
            Rk=np.diag(self.get_parameter('Rk').get_parameter_value().double_array_value),
            Rdk=np.diag(self.get_parameter('Rdk').get_parameter_value().double_array_value),
            Qk=np.diag(self.get_parameter('Qk').get_parameter_value().double_array_value),
            Qfk=np.diag(self.get_parameter('Qfk').get_parameter_value().double_array_value),
        )
        self.odelta_v = None
        self.odelta = None
        self.oa = None
        self.init_flag = 0
        
        # Initialize MPC problem
        self.mpc_prob_init()

        self.rot_matrix = np.identity(3)
        self.last_lap = time.perf_counter_ns()
        self.td = 0
        self.timer_started = False

        # Create the dataframe to export registered data
        d = {'x': [0]*len(self.wpts), 'y': [0]*len(self.wpts), 'theta': [0]*len(self.wpts), 'speed': [0]*len(self.wpts), 'accel': [0]*len(self.wpts)}
        self.exp_df = pd.DataFrame(d)

    def find_nearest_waypoint(self, pose):
        """
        Finds the index of the nearest waypoint to the given pose. Similar to Pure Pursuit algorithm
        Args:
            pose (Pose): The current pose of the object.
        Returns:
            int: The index of the nearest waypoint in the `self.wpts` array.
        """
        def distance(x1, y1, x2, y2):
            return math.sqrt((x1 - x2)**2 + (y1-y2)**2)
        
        cx = pose.pose.position.x if self.is_real else pose.pose.pose.position.x
        cy = pose.pose.position.y if self.is_real else pose.pose.pose.position.y
        res = 0
        nearest = distance(cx, cy, self.wpts[0, 0], self.wpts[0, 1])
        for i in range(len(self.wpts)):
            x, y = self.wpts[i, 0], self.wpts[i, 1]
            curr = distance(cx, cy, x, y)
            if (curr < nearest):
                nearest = curr
                res = i
        return res

    def pose_callback(self, pose_msg):
        """
        Callback function to process pose messages and update the vehicle's state and control inputs.

        Args:
            pose_msg (PoseStamped or Odometry): The pose message containing the vehicle's current pose.

        This function mainly solves the Model Predictive Control (MPC) problem to calculate the optimal control inputs.

        Notes:
            - The function uses a linearized MPC approach to solve the control problem.
            - The function updates the vehicle's state and control inputs based on the calculated reference trajectory.
        """
        quat = pose_msg.pose.orientation if self.is_real else pose_msg.pose.pose.orientation
        quat = [quat.x, quat.y, quat.z, quat.w]
        self.rot_matrix = transform.Rotation.from_quat(quat).as_matrix()
        # Get vehicle state for the initial state
        vehicle_state = self.get_vehicle_state(pose_msg)

        # Calculate the next reference trajectory for the next T steps with current vehicle pose.
        # ref_x, ref_y, ref_yaw, ref_v are columns of self.wpts
        #ref_x, ref_y, ref_yaw, ref_v = self.wpts[:, 1], self.wpts[:, 2], \
        #    self.wpts[:, 3], self.wpts[:, 5]
        ref_x, ref_y, ref_yaw, ref_v = self.wpts[:, 0], self.wpts[:, 1], \
            self.wpts[:, 2], self.wpts[:, 3]
        
        ref_path = self.calc_ref_trajectory(vehicle_state, ref_x, ref_y, ref_yaw, ref_v)
        # Visualize reference trajectory
        #self.visualize_ref_trajectory(ref_path)
        self.display_reference_trajectory(ref_path)
        x0 = [vehicle_state.x, vehicle_state.y, vehicle_state.v, vehicle_state.yaw]

        # Solve the MPC control problem (linearized)
        (
            self.oa,
            self.odelta_v,
            ox,
            oy,
            oyaw,
            ov,
            state_predict,
        ) = self.linear_mpc_control(ref_path, x0, self.oa, self.odelta_v)

        # Set the control inputs
        steer_output = self.odelta_v[0]
        speed_output = vehicle_state.v + self.oa[0] * self.config.DTK
        self.drive.drive.steering_angle = steer_output
        self.drive.drive.speed = speed_output
        self.drive.drive.acceleration = self.oa[0]
        #self.drive.drive.steering_angle_velocity = self.odelta_v[0]

        self._drive.publish(self.drive)
        print("steering = {0: <6}speed = {1: <6}acc = {2: <6}".format(round(self.drive.drive.steering_angle, 2), round(self.drive.drive.speed, 2), round(self.drive.drive.acceleration, 2)))
        nearest_wpt_i = self.find_nearest_waypoint(pose_msg)
        print("waypoint: {}".format(nearest_wpt_i))
        print(f"LAP TIME: {self.td}")

        # Save the current data for the nearest waypoint from the current position
        self.exp_df.loc[nearest_wpt_i] = [ox[0], oy[0], steer_output, speed_output, self.oa[0]]
        
        # Lap time calculations
        if self.timer_started and nearest_wpt_i >= 0 and nearest_wpt_i <= 30:
            current_time = time.perf_counter_ns() # 10^-9
            lap_time = current_time - self.last_lap
            self.timer_started = False

            if lap_time > 1000000000:
                self.last_lap = current_time
                self.td = timedelta(microseconds=lap_time/1000)
        elif nearest_wpt_i > 30:
            self.timer_started = True
        self.vis_waypoints_pub.publish(self.vis_waypoints_msg)

    def get_vehicle_state(self, pose_msg: Odometry):
        """
        Extracts and returns the vehicle state from the given Odometry message.

        Args:
            pose_msg (Odometry): The odometry message containing the vehicle's pose and twist information.

        Returns:
            State: An object representing the vehicle's state, including its position (x, y), velocity (v), and yaw angle.
        """
        vehicle_state = State()
        vehicle_state.x = pose_msg.pose.position.x if self.is_real else pose_msg.pose.pose.position.x
        vehicle_state.y = pose_msg.pose.position.y if self.is_real else pose_msg.pose.pose.position.y
        vehicle_state.v = -1*self.drive.drive.speed if self.is_real else self.drive.drive.speed

        if self.fast_start == False:
            x, y = pose_msg.twist.twist.linear.x, pose_msg.twist.twist.linear.y
            vel = math.sqrt(x*x + y*y)

            vehicle_state.v = vel

        quat = pose_msg.pose.orientation if self.is_real else pose_msg.pose.pose.orientation
        quat = [quat.x, quat.y, quat.z, quat.w]
        # Conversion from Quaternion to Euler angles in intrinsic Tait–Bryan angles
        # Intrinsic rotations are elemental rotations that occur about the axes of a coordinate system XYZ attached to a moving body.
        vehicle_state.yaw = math.atan2(2 * (quat[3] * quat[2] + quat[0] * quat[1]), 1 - 2 * (quat[1]**2 + quat[2]**2))

        return vehicle_state
    

    def mpc_prob_init(self):
        """
        Create MPC quadratic optimization problem using CVXPY with OSQP solver.
        It'll be solved every iteration for control.
        More MPC problem information here: https://osqp.org/docs/examples/mpc.html
        More QP example in CVXPY here: https://www.cvxpy.org/examples/basic/quadratic_program.html
        """
        # Initialize and create vectors for the optimization problem
        # Vehicle State Vector
        self.xk = cvxpy.Variable(
            (self.config.NXK, self.config.TK + 1)
        )
        # Control Input vector
        self.uk = cvxpy.Variable(
            (self.config.NU, self.config.TK)
        )
        objective = 0.0  # Objective value of the optimization problem
        constraints = []  # Create constraints array

        # Initialize reference vectors
        self.x0k = cvxpy.Parameter((self.config.NXK,))
        self.x0k.value = np.zeros((self.config.NXK,))

        # Initialize reference trajectory parameter
        self.ref_traj_k = cvxpy.Parameter((self.config.NXK, self.config.TK + 1))
        self.ref_traj_k.value = np.zeros((self.config.NXK, self.config.TK + 1))

        # Initializes block diagonal form of R = [R, R, ..., R] (NU*T, NU*T)
        R_block = block_diag(tuple([self.config.Rk] * self.config.TK))

        # Initializes block diagonal form of Rd = [Rd, ..., Rd] (NU*(T-1), NU*(T-1))
        Rd_block = block_diag(tuple([self.config.Rdk] * (self.config.TK - 1)))

        # Initializes block diagonal form of Q = [Q, Q, ..., Qf] (NX*T, NX*T)
        Q_block = [self.config.Qk] * (self.config.TK)
        Q_block.append(self.config.Qfk)
        Q_block = block_diag(tuple(Q_block))

        # Formulate and create the finite-horizon optimal control problem (objective function), with the horizon of T timesteps.

        # --------------------------------------------------------
        # Objective Function:
        # Objective - Part 1: Influence of the control inputs. Inputs u multiplied by the penalty R
        # obj: uk^T * R * uk
        objective += cvxpy.quad_form(cvxpy.vec(self.uk), R_block)

        # Objective - Part 2: Deviation of the vehicle from the reference trajectory weighted by Q, including final timestep T weighted by Qf
        # obj: (xk - ref_traj_k)^T * Q * (xk - ref_traj_k)
        objective += cvxpy.quad_form(cvxpy.vec(self.xk - self.ref_traj_k), Q_block)

        # Objective - Part 3: Difference from one control input to the next control input weighted by Rd
        # obj: (uk+1 - uk)^T * Rd * (uk+1 - uk)
        # diff computes the difference between consecutive columns (control inputs at consecutive timesteps)
        # The objective computes the quadratic form of the differences in control inputs, effectively penalizing large changes in control inputs
        objective += cvxpy.quad_form(cvxpy.vec(cvxpy.diff(self.uk, axis=1)), Rd_block)
        # --------------------------------------------------------
        # Constraints:
        # Dynamics Constraints: Calculate the future vehicle behavior/states based on the vehicle dynamics model matrices.
        # Evaluate vehicle Dynamics for next T timesteps.
        A_block = []
        B_block = []
        C_block = []
        # init path to zeros
        path_predict = np.zeros((self.config.NXK, self.config.TK + 1))
        for t in range(self.config.TK):
            A, B, C = self.get_model_matrix(
                path_predict[2, t], path_predict[3, t], 0.0 # 0.0, 0.0, 0.0 # è uguale
            )
            A_block.append(A)
            B_block.append(B)
            C_block.extend(C)

        A_block = block_diag(tuple(A_block))
        B_block = block_diag(tuple(B_block))
        C_block = np.array(C_block)

        # [AA] Sparse matrix to CVX parameter for proper stuffing
        # Reference: https://github.com/cvxpy/cvxpy/issues/1159#issuecomment-718925710
        m, n = A_block.shape
        self.Annz_k = cvxpy.Parameter(A_block.nnz)
        data = np.ones(self.Annz_k.size)
        rows = A_block.row * n + A_block.col
        cols = np.arange(self.Annz_k.size)
        Indexer = csc_matrix((data, (rows, cols)), shape=(m * n, self.Annz_k.size))

        # Setting sparse matrix data
        self.Annz_k.value = A_block.data

        # Now we use this sparse version instead of the old A_ block matrix
        self.Ak_ = cvxpy.reshape(Indexer @ self.Annz_k, (m, n), order="C")

        # Same as A
        m, n = B_block.shape
        self.Bnnz_k = cvxpy.Parameter(B_block.nnz)
        data = np.ones(self.Bnnz_k.size)
        rows = B_block.row * n + B_block.col
        cols = np.arange(self.Bnnz_k.size)
        Indexer = csc_matrix((data, (rows, cols)), shape=(m * n, self.Bnnz_k.size))

        # Sparse version instead of the previous B_block
        self.Bk_ = cvxpy.reshape(Indexer @ self.Bnnz_k, (m, n), order="C")
        
        # Setting sparse matrix data
        self.Bnnz_k.value = B_block.data

        # No need for sparse matrices for C as most values are parameters
        self.Ck_ = cvxpy.Parameter(C_block.shape)
        self.Ck_.value = C_block

        # -------------------------------------------------------------
        # Constraint part 1: Add model dynamics constraints to the optimization problem
        
        flatten_prev_xk = cvxpy.vec(self.xk[:, :-1])
        flatten_next_xk = cvxpy.vec(self.xk[:, 1:])
        c1 = flatten_next_xk == self.Ak_ @ flatten_prev_xk + self.Bk_ @ cvxpy.vec(self.uk) + self.Ck_
        constraints.append(c1)
        
        # Constraint part 2: Add constraints on steering. 
        # Change in steering angle cannot exceed steering angle speed limit.
        
        dsteering = cvxpy.diff(self.uk[1, :])
        c2_lower = -self.config.MAX_DSTEER * self.config.DTK <= dsteering
        c2_upper = dsteering <= self.config.MAX_DSTEER * self.config.DTK
        constraints.append(c2_lower)
        constraints.append(c2_upper)
        
        # Constraint part 3: Add constraints on upper and lower bounds of states and inputs and initial state constraint.
             
        # init state constraint
        c3 = self.xk[:, 0] == self.x0k
        constraints.append(c3)

        # state constraints
        speed = self.xk[2, :]
        c4_lower = self.config.MIN_SPEED <= speed
        c4_upper = speed <= self.config.MAX_SPEED
        constraints.append(c4_lower)
        constraints.append(c4_upper)

        # input constraints
        steering = self.uk[1, :]
        c5_lower = self.config.MIN_STEER <= steering
        c5_upper = steering <= self.config.MAX_STEER
        constraints.append(c5_lower)
        constraints.append(c5_upper)

        acc = self.uk[0, :]
        c6 = acc <= self.config.MAX_ACCEL
        constraints.append(c6)
        # -------------------------------------------------------------
        # Create the MPC optimization problem in CVXPY and setup the workspace.
        # Optimization goal: minimize the objective function
        self.MPC_prob = cvxpy.Problem(cvxpy.Minimize(objective), constraints)

    def calc_ref_trajectory(self, state, cx, cy, cyaw, sp):
        """
        Calculate the reference trajectory ref_traj in T steps: [x, y, v, yaw]
        using the current velocity, calc the T points along the reference path
        :param cx: Course X-Position
        :param cy: Course y-Position
        :param cyaw: Course Heading
        :param sp: speed profile
        :dl: distance step
        :pind: Setpoint Index
        :return: reference trajectory ref_traj, reference steering angle
        """

        # Create placeholder Arrays for the reference trajectory for T steps
        ref_traj = np.zeros((self.config.NXK, self.config.TK + 1))
        ncourse = len(cx)

        # Find nearest index/setpoint from where the trajectories are calculated
        _, _, _, ind = nearest_point(np.array([state.x, state.y]), np.array([cx, cy]).T)

        # Load the initial parameters from the setpoint into the trajectory
        ref_traj[0, 0] = cx[ind]
        ref_traj[1, 0] = cy[ind]
        ref_traj[2, 0] = sp[ind]
        ref_traj[3, 0] = cyaw[ind]

        # based on current velocity, distance traveled on the ref line between time steps
        travel = abs(state.v) * self.config.DTK
        dind = travel / self.config.dlk
        dind = 2
        ind_list = int(ind) + np.insert(
            np.cumsum(np.repeat(dind, self.config.TK)), 0, 0
        ).astype(int)
        ind_list[ind_list >= ncourse] -= ncourse
        ref_traj[0, :] = cx[ind_list]
        ref_traj[1, :] = cy[ind_list]
        ref_traj[2, :] = sp[ind_list]

        angle_thres = 4.5

        for i in range(len(cyaw)):
            if cyaw[i] - state.yaw > angle_thres:
                cyaw[i] -= 2*np.pi
                # print(cyaw[i] - state.yaw)
            if state.yaw - cyaw[i] > angle_thres:
                cyaw[i] += 2*np.pi
                # print(cyaw[i] - state.yaw)

        ref_traj[3, :] = cyaw[ind_list]

        return ref_traj


    def predict_motion(self, x0, oa, od, xref):
        """
        Predicts the future motion of the vehicle based on the initial state, 
        control inputs and reference trajectory.

        Args:
            x0: initial state vector [x, y, v, yaw].
            oa: acceleration of T steps of last time
            od: delta of T steps of last time
            xref: Reference trajectory matrix.

        Returns:
            Predicted trajectory matrix.
        """
        path_predict = xref * 0.0
        for i, _ in enumerate(x0):
            path_predict[i, 0] = x0[i]

        state = State(x=x0[0], y=x0[1], yaw=x0[3], v=x0[2])
        for (ai, di, i) in zip(oa, od, range(1, self.config.TK + 1)):
            state = self.update_state(state, ai, di)
            path_predict[0, i] = state.x
            path_predict[1, i] = state.y
            path_predict[2, i] = state.v
            path_predict[3, i] = state.yaw

        return path_predict

    def update_state(self, state, a, delta):
        """
        Update the state of the vehicle based on the current state, acceleration, and steering angle.

        Args:
            state: The current state of the vehicle, which includes attributes x, y, yaw, and v.
            a: The acceleration input.
            delta: The steering angle input.

        Returns:
            The updated state of the vehicle.
        """

        # input check
        if delta >= self.config.MAX_STEER:
            delta = self.config.MAX_STEER
        elif delta <= -self.config.MAX_STEER:
            delta = -self.config.MAX_STEER

        state.x = state.x + state.v * math.cos(state.yaw) * self.config.DTK
        state.y = state.y + state.v * math.sin(state.yaw) * self.config.DTK
        state.yaw = (
            state.yaw + (state.v / self.config.WB) * math.tan(delta) * self.config.DTK
        )
        state.v = state.v + a * self.config.DTK

        if state.v > self.config.MAX_SPEED:
            state.v = self.config.MAX_SPEED
        elif state.v < self.config.MIN_SPEED:
            state.v = self.config.MIN_SPEED

        return state

    def get_model_matrix(self, v, phi, delta):
        """
        Calc linear and discrete time dynamic model -> Explicit discrete time-invariant
        Linear System: Xdot = Ax + Bu + C
        State vector: x = [x, y, v, yaw]
        :param v: speed
        :param phi: heading angle of the vehicle
        :param delta: steering angle: delta_bar
        :return: A, B, C
        """

        # State (or system) matrix A, 4x4
        A = np.zeros((self.config.NXK, self.config.NXK))
        A[0, 0] = 1.0
        A[1, 1] = 1.0
        A[2, 2] = 1.0
        A[3, 3] = 1.0
        A[0, 2] = self.config.DTK * math.cos(phi)
        A[0, 3] = -self.config.DTK * v * math.sin(phi)
        A[1, 2] = self.config.DTK * math.sin(phi)
        A[1, 3] = self.config.DTK * v * math.cos(phi)
        A[3, 2] = self.config.DTK * math.tan(delta) / self.config.WB

        # Input Matrix B; 4x2
        B = np.zeros((self.config.NXK, self.config.NU))
        B[2, 0] = self.config.DTK
        B[3, 1] = self.config.DTK * v / (self.config.WB * math.cos(delta) ** 2)

        C = np.zeros(self.config.NXK)
        C[0] = self.config.DTK * v * math.sin(phi) * phi
        C[1] = -self.config.DTK * v * math.cos(phi) * phi
        C[3] = -self.config.DTK * v * delta / (self.config.WB * math.cos(delta) ** 2)

        return A, B, C

    def mpc_prob_solve(self, ref_traj, path_predict, x0):
        """
        Solves the Model Predictive Control (MPC) problem for a given reference trajectory and predicted path.

        Parameters:
        ref_traj: The reference trajectory that the MPC should follow.
        path_predict: The predicted path containing state information.
        x0: The initial state vector, used for the warm start.

        Returns:
            A tuple containing the following elements:
            - oa: The optimal acceleration values.
            - odelta: The optimal steering angle values.
            - ox: The optimal x positions.
            - oy: The optimal y positions.
            - oyaw: The optimal yaw angles.
            - ov: The optimal velocities.

        Notes:
        - This function uses the CVXPY library to solve the optimization problem.
        - Solver options include cvxpy.OSQP and cvxpy.GUROBI, OSQP suggested.
        - If the problem is not solved optimally, the function returns None for all outputs.
        """
        self.x0k.value = x0

        A_block = []
        B_block = []
        C_block = []
        for t in range(self.config.TK):
            A, B, C = self.get_model_matrix(
                path_predict[2, t], path_predict[3, t], 0.0
            )
            A_block.append(A)
            B_block.append(B)
            C_block.extend(C)

        A_block = block_diag(tuple(A_block))
        B_block = block_diag(tuple(B_block))
        C_block = np.array(C_block)

        self.Annz_k.value = A_block.data
        self.Bnnz_k.value = B_block.data
        self.Ck_.value = C_block

        self.ref_traj_k.value = ref_traj

        # Solve the optimization problem in CVXPY
        # Solver selections: cvxpy.OSQP; cvxpy.GUROBI
        self.MPC_prob.solve(solver=cvxpy.OSQP, verbose=False, warm_start=True)

        if (
            self.MPC_prob.status == cvxpy.OPTIMAL
            or self.MPC_prob.status == cvxpy.OPTIMAL_INACCURATE
        ):
            ox = np.array(self.xk.value[0, :]).flatten()
            oy = np.array(self.xk.value[1, :]).flatten()
            ov = np.array(self.xk.value[2, :]).flatten()
            oyaw = np.array(self.xk.value[3, :]).flatten()
            oa = np.array(self.uk.value[0, :]).flatten()
            odelta = np.array(self.uk.value[1, :]).flatten()

        else:
            print("Error: Cannot solve mpc..")
            oa, odelta, ox, oy, oyaw, ov = None, None, None, None, None, None

        return oa, odelta, ox, oy, oyaw, ov

    def linear_mpc_control(self, ref_path, x0, oa, od):
        """
        MPC control with updating operational point iteratively
        :param ref_path: reference trajectory in T steps
        :param x0: initial state vector
        :param oa: acceleration of T steps of last time
        :param od: delta of T steps of last time
        """

        if oa is None or od is None:
            oa = [0.0] * self.config.TK
            od = [0.0] * self.config.TK

        # Call the Motion Prediction function: Predict the vehicle motion for x-steps
        path_predict = self.predict_motion(x0, oa, od, ref_path)
        # Visualize path predict
        #self.visualize_predicted_path(path_predict)
        self.display_predicted_path(path_predict)
        # Publish marker array 
        #self.pub_vis.publish(self.markerArray)

        poa, pod = oa[:], od[:]

        # Run the MPC optimization: Create and solve the optimization problem
        mpc_a, mpc_delta, mpc_x, mpc_y, mpc_yaw, mpc_v = self.mpc_prob_solve(
            ref_path, path_predict, x0
        )

        return mpc_a, mpc_delta, mpc_x, mpc_y, mpc_yaw, mpc_v, path_predict

    # Visualization for ROS & RViz
    def display_waypoints(self):
        self.vis_waypoints_msg.points = []
        self.vis_waypoints_msg.header.frame_id = '/map'
        self.vis_waypoints_msg.type = Marker.POINTS
        self.vis_waypoints_msg.color.g = 0.75
        self.vis_waypoints_msg.color.a = 1.0
        self.vis_waypoints_msg.scale.x = 0.05
        self.vis_waypoints_msg.scale.y = 0.05
        self.vis_waypoints_msg.id = 0
        for i in range(self.wpts.shape[0]):
            point = Point(x = self.wpts[i, 0], y = self.wpts[i, 1], z = 0.1)
            #point = Point(x = self.wpts[i, 1], y = self.wpts[i, 2], z = 0.1)
            self.vis_waypoints_msg.points.append(point)
        
        # self.vis_waypoints_pub.publish(self.vis_waypoints_msg)

    def display_reference_trajectory(self, ref_traj):
        # visualize the path data in the world frame
        self.vis_ref_traj_msg.points = []
        self.vis_ref_traj_msg.header.frame_id = '/map'
        self.vis_ref_traj_msg.type = Marker.LINE_STRIP
        self.vis_ref_traj_msg.color.b = 0.75
        self.vis_ref_traj_msg.color.a = 1.0
        self.vis_ref_traj_msg.scale.x = 0.08
        self.vis_ref_traj_msg.scale.y = 0.08
        self.vis_ref_traj_msg.id = 0
        for i in range(ref_traj.shape[1]):
            point = Point(x = ref_traj[0, i], y = ref_traj[1, i], z = 0.2)
            self.vis_ref_traj_msg.points.append(point)
        
        self.vis_ref_traj_pub.publish(self.vis_ref_traj_msg)

    def display_predicted_path(self, path_predict):
        # visualize the path data in the world frame
        self.vis_pred_path_msg.points = []
        self.vis_pred_path_msg.header.frame_id = '/map'
        self.vis_pred_path_msg.type = Marker.LINE_STRIP
        self.vis_pred_path_msg.color.r = 0.75
        self.vis_pred_path_msg.color.a = 1.0
        self.vis_pred_path_msg.scale.x = 0.08
        self.vis_pred_path_msg.scale.y = 0.08
        self.vis_pred_path_msg.id = 0
        for i in range(path_predict.shape[1]):
            point = Point(x = path_predict[0, i], y = path_predict[1, i], z = 0.2)
            self.vis_pred_path_msg.points.append(point)
        
        self.vis_pred_path_pub.publish(self.vis_pred_path_msg)

def main(args=None):
    rclpy.init(args=args)
    print("MPC Initialized")
    mpc_node = MPC(sys.argv[1], sys.argv[2])
    rclpy.spin(mpc_node)

    mpc_node.destroy_node()
    rclpy.shutdown()