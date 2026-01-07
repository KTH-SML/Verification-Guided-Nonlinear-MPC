#! /usr/bin/env python3

import casadi as ca
import numpy as np
import matplotlib.pyplot as plt
import os
import dynamics_car as dyn
import copy
import time as tm

from tqdm import tqdm
from matplotlib.animation import FuncAnimation

def get_value_function():
    # Import value function from file
    file_path = os.path.join(os.path.dirname(__file__), "out1_old.npy")
    return np.load(file_path)

class MPC_casadi:

    X0, XN = -2, 4
    Y0, YN = -2, 4

    def __init__(self, init_state, nr, vehicle_name='', config_ns='~mpc'):
        """
        This is the release 0 of a general-purpose Nonlinear Model Predictive Controller (NMPC) 
        designed for the SVEA platform. It offers the following features:

        - Nonlinear dynamics with state representation [x, y, theta, v, steering] 
        and control inputs [steering_rate, acceleration].
        - Accepts a reference trajectory as input (see `compute_control` method) 
        and optimizes the predicted trajectory to minimize the deviation of 
        the predicted points from the next reference point.
        - Functions as both an optimal path planner and path tracker:
            - For path planning and tracking: Reference points should be spaced apart to give 
                the MPC the freedom to determine the optimal path.
            - For tracking only: Adjust parameters such as prediction horizon to 
                follow the reference trajectory closely.

        Limitations:
        - Dynamic obstacles are not considered 
        in this implementation.

        Initialize the MPC controller with the given parameters:

        :param L: Wheelbase of the vehicle (unit [m])
        :param N: Prediction horizon steps
        :param dt: Sampling time
        :param min_steering: Minimum steering angle [rad]
        :param max_steering: Maximum steering angle [rad]
        :param min_acceleration: Minimum acceleration [m/s^2]
        :param max_acceleration: Maximum acceleration [m/s^2]
        :param min_velocity: Minimum velovity [m/s]
        :param max_velocity: Maximum velocity [m/s]
        :param Q1: State weight matrix (4x4)
        :param Q2: Control rate weight matrix (2x2)
        :param Q3: Control weight matrix (2x2)
        :param Qf: Final state weight matrix (4x4)
        :param Qv: Forward_speed_weight scalar
        :param Qs: steering weight scalar
        :param Qhj: HJ reachability weight scalar
        """

        ## Core Parameters

        # The prediction horizon steps for the mpc optimization problem.
        self.N = dyn.prediction_horizon

        self.current_horizon = self.N # Initially set to max horizon

        # The time step in which the optimization problem is divided (unit [s]).
        self.dt = dyn.time_step

        ## Initial state of the vehicle
        self.init_state = init_state
        
        ## Load weight matrices 
        # Note: we convert to dense matrix to allow symbolic operations.

        self.Q1_list = dyn.state_weight_matrix_5D
        self.Q1 = ca.DM(np.array(self.Q1_list).reshape((5, 5)))

        self.Q2_list = dyn.control_rate_weight_matrix
        self.Q2 = ca.DM(np.array(self.Q2_list).reshape((2, 2)))

        self.Q3_list = dyn.control_weight_matrix
        self.Q3 = ca.DM(np.array(self.Q3_list).reshape((2, 2)))

        self.Qf_list = dyn.final_state_weight_matrix_5D
        self.Qf = ca.DM(np.array(self.Qf_list).reshape((5, 5)))

        self.Qv = ca.DM(dyn.forward_speed_weight)

        self.Qs = ca.DM(dyn.steering_weight)

        self.Qhj = ca.DM(dyn.reachability_set_weight)
        
        ## Model Parameters

        self.min_steering = np.radians(dyn.steering_min)
        self.max_steering = np.radians(dyn.steering_max)

        self.min_steering_rate = np.radians(dyn.steering_rate_min)
        self.max_steering_rate = np.radians(dyn.steering_rate_max)

        self.min_velocity = dyn.velocity_min
        self.max_velocity = dyn.velocity_max

        self.min_acceleration = dyn.acceleration_min
        self.max_acceleration = dyn.acceleration_max

        # Wheelbase of the vehicle (unit [m]).
        self.L = ca.DM(dyn.wheelbase)
        
        ## Setup CasADi
        self.opti = ca.Opti()
        self.sol = None
        self.predicted_trajectory = None

        ## Import value function
        self.set_value_function()
        self.dims = self.value_function.shape
        self.value_func_wrapped = np.concatenate(
            (self.value_function,
             self.value_function,
             self.value_function),
            axis=3
        )

        ## Set up bounds for state and input
        time_max = self.dt * self.dims[0]
        self.min_bounds_state = np.array([       0,         self.X0,         self.Y0, -np.pi, self.min_velocity, self.min_steering])
        self.max_bounds_state = np.array([time_max, self.X0+self.XN, self.Y0+self.YN, +np.pi, self.max_velocity, self.max_steering])

        self.min_bounds_input = np.array([self.min_acceleration, self.min_steering_rate])
        self.max_bounds_input = np.array([self.max_acceleration, self.max_steering_rate])

        ## Setup grid
        self.grids = [np.linspace(self.min_bounds_state[i], self.max_bounds_state[i], self.dims[i]+1) for i in range(6)]
        self.d_states = [self.grids[i][1] - self.grids[i][0] for i in range(6)]
        self.local_state_cost = None
        self.local_input_cost = None
        self.HJ_time = None
        x = ca.MX.sym('x', 6)  # state vector [t, x, y, theta, v, delta]
        u = ca.MX.sym('u', 2)  # control input
        zero_func = ca.Function('zero_func', [u], [0])
        self.local_state_cost = ca.Function('local_state_cost', [x], [0])  # Initialize with zeros
        self.local_input_cost = [zero_func for _ in range(self.N)]  # Initialize with zeros

        ## Setup cost associated with value function
        self.define_state_and_control_variables()
        self.set_objective_function()
        self.set_state_constraints()
        self.set_control_input_constraints()
        self.set_solver_options()

        self.grids_segment_list = []

        self.iteration = 0
        self.computation_times = np.zeros((nr,3))

    def compute_control(self, state, HJ_time, reference_trajectory, it):
        """
        Compute the control actions (steering, acceleration) using MPC.
        
        :param state: Current state of the vehicle [x, y, theta, v, delta]
        :param HJ_time: Another state, used for interpolation of segment
        :return: acceleration, steering_rate
        """
        self.iteration = it

        comp_start = tm.time()
        # Bound the initial state to respect constraints
        bounded_state = self.bound_initial_state(state)

        # Set initial guess for self.x if reference_trajectory is long enough
        # if len(reference_trajectory) > self.N:
        x_guess = np.array(reference_trajectory[:self.N+1]).T[1:6]  # shape (5, N+1)
        # self.opti.set_initial(self.x, x_guess)

        self.trajectory = reference_trajectory

        solve_U_start = tm.time()
        self.solve_U_set([HJ_time] + state, self.N)
        solve_U_end = tm.time()

        self.computation_times[it,1] = solve_U_end - solve_U_start

        # self.solve_U_set([HJ_time] + state, self.current_horizon)
        self.opti = ca.Opti()
        self.sol = None
        self.define_state_and_control_variables()
        self.set_objective_function(x_ref=reference_trajectory)
        self.set_state_constraints(HJ_time)
        self.set_control_input_constraints()
        self.set_solver_options()

        # Set current state and reference trajectory for the active part of the horizon
        self.opti.set_value(self.x_init, bounded_state)


        solver_start = tm.time()
        # Extract control actions (acceleration and steering rate)
        try:
            # Solve the optimization problem
            
            # print("Attempting to solve MPC with current horizon:", self.current_horizon)
            self.sol = self.opti.solve()
            acceleration = self.sol.value(self.u[0, 0])
            steering_rate = self.sol.value(self.u[1, 0])

            self.predicted_trajectory = self.sol.value(self.x).T[1:]
            # print("Current:",state, "\nPredicted:",self.predicted_trajectory)
        except:
            # self.sol = None
            # print("MPC solve failed, using debug values")
            self.sol = self.opti.debug
            acceleration = self.sol.value(self.u[0, 0])
            steering_rate = self.sol.value(self.u[1, 0])

            self.predicted_trajectory = self.sol.value(self.x).T[1:]
            # print("Current:",state, "\nPredicted:",self.predicted_trajectory)

        solver_end = tm.time()
        self.computation_times[it][2] = solver_end - solver_start

        comp_end = tm.time()
        self.computation_times[it][0] = comp_end - comp_start

        return acceleration, steering_rate

    def get_optimal_control(self, all=True):
        """
        This method returns the optimal control computed by the mpc.
        :param all: Flag to indicate if all of the optimal control for the 
            whole prediction should	be returned or only the first step.
        :type all: bool
        :return: The optimal control computed by the mpc controller.
        :rtype: NumPy array
        """
        if self.sol is not None:
            u_opt = np.array(self.sol.value(self.u))
            if all:
                return u_opt
            else:
                return u_opt[:,0]
        else:
            return None
        
    def get_optimal_states(self):
        """
        This method returns the optimal states computed by the mpc controller
        for the whole prediction horizon.
        :return: The optimal states for the whole prediction horizon.
        :rtype: NumPy array
        """
        if self.sol is not None:
            return np.array(self.sol.value(self.x))   # Note: to get only optimized one: self.sol.value(self.x[:, :self.current_horizon])
        else: 
            return None

    def define_state_and_control_variables(self):
        # Define state and control variables
        self.x = self.opti.variable(5, self.N + 1)  # state = [x, y, theta, v, delta] / [x-pos, y-pos, yaw, velocity, steering]
        self.u = self.opti.variable(2, self.N)      # input = [a, omega]              / [acceleration, steering-rate]

        self.x_init = self.opti.parameter(5)             # Initial state

        self.M = self.opti.variable(self.N)            # Slack variable

    def set_objective_function(self, x_ref = [], HJ_time = 0, do_input_cost=True):
        # Define the objective function:
        # J = (x[k]-x_ref[k])^T Q1 (x[k]-x_ref[k]) + (u[k+1] - u[k])^T Q2 (u[k+1] - u[k]) + u[k]^T Q3 u[k] + Qv max(0,-x[3])^2
        self.objective = 0
        ref_len = len(x_ref)
        for k in range(self.current_horizon):

            # Q_hj = (k+1) / self.current_horizon
            Q_hj = self.Qhj if k == 0 else 0

            if k < ref_len:
                state_error = self.compute_state_error(self.x[:, k], x_ref[k][1:])
                self.objective += (ca.mtimes([state_error.T, self.Q1, state_error]))

            # Control input rate of change term (u[k+1] - u[k])
            if k < self.current_horizon - 1:
                input_cost = self.u[:, k+1] - self.u[:, k]
            else:
                input_cost = ca.DM.zeros(self.u.shape[0], 1)

            # Penalize for negative velocity (soft constraint)
            velocity_penalty = -self.x[3, k]  # Penalize if v < 0
            # velocity_penalty = ca.fmax(0, -self.x[3, k])  # Penalize if v < 0

            # Accumulate the terms into the objective
            self.objective += (
                                 ca.mtimes([input_cost.T, self.Q2, input_cost])
                               + ca.mtimes([self.u[:, k].T, self.Q3, self.u[:, k]])
                               + ca.mtimes([velocity_penalty.T, self.Qv, velocity_penalty])
                               )

            HJ_cost = self.local_input_cost[k](ca.vertcat(self.u[0, k],self.u[1, k]))
            self.objective += ca.mtimes([HJ_cost, Q_hj])
            # self.objective += (self.M[k] * Q_hj)

        if ref_len >= self.current_horizon:
            final_state_error = self.compute_state_error(self.x[:, self.current_horizon], x_ref[self.current_horizon][1:])
            self.objective += (ca.mtimes([final_state_error.T, self.Q1, final_state_error]))

        # Specify type of optimization problem
        self.opti.minimize(self.objective)

    def set_state_constraints(self, HJ_time=0):
        # Initial state constraint
        self.opti.subject_to(self.x[:, 0] == self.x_init)

        # Vehicle dynamics constraints - Simple kinematic bycicle model
        for k in range(self.N+1):
            if k != self.N:
                x_next = self.x[0, k] + self.dt * self.x[3, k] * ca.cos(self.x[2, k])                   # x_k+1 = x_k + dt * v_k * cos(theta_k)
                y_next = self.x[1, k] + self.dt * self.x[3, k] * ca.sin(self.x[2, k])                   # y_k+1 = y_k + dt * v_k * sin(theta_k)
                theta_next = self.x[2, k] + self.dt * (self.x[3, k] * ca.tan(self.x[4, k])/ self.L)     # theta_k+1 = theta_k + dt * v_k * tan(delta_k) / L
                v_next = self.x[3, k] + self.dt * self.u[0, k]                                          # v_k+1 = v_k + dt * a_k
                delta_next = self.x[4, k] + self.dt * self.u[1, k]                                      # delta_k+1 = delta_k + dt * omega_k

                self.opti.subject_to(self.x[0, k + 1] == x_next)
                self.opti.subject_to(self.x[1, k + 1] == y_next)
                self.opti.subject_to(self.x[2, k + 1] == theta_next)
                self.opti.subject_to(self.x[3, k + 1] == v_next)
                self.opti.subject_to(self.x[4, k + 1] == delta_next)

            # Velocity constraints
            self.opti.subject_to(self.x[3, k] <= self.max_velocity)
            self.opti.subject_to(self.x[3, k] >= self.min_velocity)

            # Steering angle constraints
            self.opti.subject_to(self.x[4, k] <= self.max_steering)
            self.opti.subject_to(self.x[4, k] >= self.min_steering)

    def set_control_input_constraints(self):
        # Input constraints (acceleration, steering angle)
        for k in range(self.N):
            # Acceleration constraints
            self.opti.subject_to(self.u[0, k] <= self.max_acceleration)
            self.opti.subject_to(self.u[0, k] >= self.min_acceleration)
            
            # Steering angle constraints
            self.opti.subject_to(self.u[1, k] <= self.max_steering_rate)
            self.opti.subject_to(self.u[1, k] >= self.min_steering_rate)

    def set_solver_options(self):
        # Set solver options
        opts = {
                "ipopt.print_level": 0, 
                "print_time": 0,
                "ipopt.max_iter": 100,
                "ipopt.acceptable_tol": 1e-2,               # relaxed tolerance for early stopping
                'ipopt.acceptable_obj_change_tol': 1e-4,    # must have met acceptable criteria for X iterations
                'ipopt.acceptable_iter': 5,
                'ipopt.acceptable_dual_inf_tol': 1e-1,
                }       
        self.opti.solver("ipopt", opts)

    def set_value_function(self):
        # Import value function from file
        file_path = os.path.join(os.path.dirname(__file__), "out1_old.npy")
        self.value_function = np.load(file_path)
        self.dims = self.value_function.shape
        print("Loaded value function of type",type(self.value_function),"of shape",self.dims)

    def compute_inital_trajectory(self):
        """
        Computes an initial warmstart for the optimization problem.
        This is done by computing a path from the reachability function V.
        """
        # Compute warmstart from the reachability function and returns a list of states
        self.local_state_cost = self.interpolate_V_segment(self.init_state, self.current_horizon, add_segment_to_list=False)
        self.set_objective_function(self.init_state[0], do_input_cost=False)
        self.opti.set_value(self.x_init, self.init_state[1:])

        # Solve the optimization problem
        try:    self.sol = self.opti.solve()
        except: self.sol = self.opti.debug

        self.trajectory = self.get_optimal_states().T[1:]

    def interpolate_V_segment(self, state, horizon, add_segment_to_list=True):
        """
        This intepolates a segment depending on the current state, prediction horizon and allowed marginal\n
        """
        delta_plus  = [horizon, horizon, horizon, horizon, 3, 3]
        delta_minus = [      1, horizon, horizon, horizon, 3, 3]
        index = list(self.coord_to_index(state))
        index[3] = index[3] + self.dims[3]
        indices = [[max(0,index[i]-delta_minus[i]), min(self.dims[i],index[i]+delta_plus[i])] for i in range(6)]
        indices[3] = [index[3]-delta_minus[3], index[3]+delta_plus[3]]    # Special case for theta since it wraps around

        slices = tuple(slice(indices[i][0], indices[i][1]+1) for i in range(6))

        value_func_segment = self.value_func_wrapped[slices]
        dims_segment = value_func_segment.shape
        # print(state)
        self.grids_segment = [np.linspace(self.d_states[i] * indices[i][0] + self.min_bounds_state[i], self.d_states[i] * indices[i][1] + self.min_bounds_state[i], dims_segment[i]) for i in range(6)]
        self.grids_segment[3] = np.linspace(self.d_states[3] * indices[3][0] + self.min_bounds_state[3] - 2*np.pi, self.d_states[3] * indices[3][1] + self.min_bounds_state[3] - 2*np.pi, dims_segment[3])

        if add_segment_to_list:
            self.grids_segment_list.append(copy.deepcopy(self.grids_segment))

        # Create CasADi interpolation function
        # print("Attempting interpolation for value function segment with shape", value_func_segment.shape)
        return ca.interpolant("local_state_cost", "linear", self.grids_segment, value_func_segment.ravel(order="F"))  # Column-major flattening

    def interpolate_U_segment(self, current_state, horizon):
        """
        Interpolates the Hamilton-Jacobi control input cost over a time horizon.

        Returns:
            List[casadi.Function]: A list of scalar-valued symbolic functions of the control inputs,
            one for each time step in the horizon.
        """

        HJ_time = current_state[0]
        predicted_states = np.array([[HJ_time + k*self.dt] + current_state[1:] for k in range(horizon)])
        
        U_locked_list = []

        ### Create function HJ_eq for how the input set is computed ###

        delta_plus  = [self.current_horizon, 3, 3, 3, 3, 3]
        delta_minus = [                   3, 3, 3, 3, 3, 3]               

        for k in range(horizon):

            ### Step 1: Compute gradient using local points
            index = list(self.coord_to_index(predicted_states[k]))
            
            index[3] = index[3] + self.dims[3]
            indices = [[max(0,index[i]-delta_minus[i]), min(self.dims[i],index[i]+delta_plus[i])] for i in range(6)]
            indices[3] = [index[3]-delta_minus[3], index[3]+delta_plus[3]]    # Special case for theta since it wraps around

            slices = tuple(slice(indices[i][0], indices[i][1]+1) for i in range(6))
            value_func_segment = self.value_func_wrapped[slices]
            dims_segment = value_func_segment.shape
            # print(state)
            self.grids_segment = [np.linspace(self.d_states[i] * indices[i][0] + self.min_bounds_state[i], self.d_states[i] * indices[i][1] + self.min_bounds_state[i], dims_segment[i]) for i in range(6)]
            self.grids_segment[3] = np.linspace(self.d_states[3] * indices[3][0] + self.min_bounds_state[3] - 2*np.pi, self.d_states[3] * indices[3][1] + self.min_bounds_state[3] - 2*np.pi, dims_segment[3])

            value_func_gradient = np.array(np.gradient(value_func_segment, *self.d_states, edge_order=2))  # Compute the gradient of the value function segment
            
            ### Step 2: Setup the formula for alpha+beta*u and compute some points

            t_c,x_c,y_c,theta_c,v_c,delta_c = predicted_states[k]

            f_1 = np.array([v_c * np.cos(theta_c),
                           v_c * np.sin(theta_c),
                           v_c * np.tan(delta_c)/float(self.L),
                           0,
                           0])  # State dynamics
            f_2 = np.array([[0,0],
                            [0,0],
                            [0,0],
                            [1,0],
                            [0,1]]) # Control dynamics
                        
            # Derivatives
            point_index = tuple(index[i] - indices[i][0] for i in range(6))

            V_Dt = value_func_gradient[0][point_index]
            V_Dx = [value_func_gradient[i][point_index] for i in range(1, 6)]

            alpha = V_Dt + V_Dx @ f_1
            beta = V_Dx @ f_2

            inputs = [[self.min_acceleration, self.min_steering_rate],
                      [self.min_acceleration, self.max_steering_rate],
                      [self.max_acceleration, self.min_steering_rate]]
            values = [alpha + beta @ inputs[i] for i in range(3)]

            ### Step 3: Use the points to setup the plane function

            # Build matrix A and vector b for solving Ax = b
            # Each row in A is [1, u1, u2] to represent the affine plane
            A = ca.DM([
                [1.0, *inputs[0]],
                [1.0, *inputs[1]],
                [1.0, *inputs[2]]
            ])  # Shape: (3, 3)

            b = ca.DM(values).reshape((3, 1))  # Shape: (3, 1)

            # Solve for [a, b1, b2] using CasADi
            coeffs = ca.solve(A, b)  # DM(3x1)

            u_sym = ca.MX.sym("u", 2)              # Symbolic input u = [u1, u2]
            u_aug = ca.vertcat(1, u_sym)           # Augment to [1, u1, u2]
            f_u = ca.mtimes(coeffs.T, u_aug)       # Shape: (1x1), scalar output

            # Make it a CasADi function
            f_interp = ca.Function(f"f_interp_{k}", [u_sym], [f_u])

            U_locked_list.append(copy.deepcopy(f_interp))

        return U_locked_list

    def interpolate_U_segment_old(self, current_state, horizon):
        """
        Returns an interpolated function for the control input cost.\n
        """

        ### Create function HJ_eq for how the input set is computed ###

        # delta_plus  = [self.current_horizon, 3, 3, 3, 3, 3]
        # delta_minus = [                   1, 3, 3, 3, 3, 3]
        delta_plus  = [self.current_horizon, 2, 2, 2, 2, 2]
        delta_minus = [                   1, 2, 2, 2, 2, 2]
        index = list(self.coord_to_index(current_state))
        index[3] = index[3] + self.dims[3]
        indices = [[max(0,index[i]-delta_minus[i]), min(self.dims[i],index[i]+delta_plus[i])] for i in range(6)]
        indices[3] = [index[3]-delta_minus[3], index[3]+delta_plus[3]]    # Special case for theta since it wraps around

        slices = tuple(slice(indices[i][0], indices[i][1]+1) for i in range(6))
        value_func_segment = self.value_func_wrapped[slices]
        dims_segment = value_func_segment.shape
        # print(state)
        self.grids_segment = [np.linspace(self.d_states[i] * indices[i][0] + self.min_bounds_state[i], self.d_states[i] * indices[i][1] + self.min_bounds_state[i], dims_segment[i]) for i in range(6)]
        self.grids_segment[3] = np.linspace(self.d_states[3] * indices[3][0] + self.min_bounds_state[3] - 2*np.pi, self.d_states[3] * indices[3][1] + self.min_bounds_state[3] - 2*np.pi, dims_segment[3])

        value_func_segment = self.value_func_wrapped[slices]
        value_func_gradient = np.array(np.gradient(value_func_segment, *self.d_states, edge_order=2))  # Compute the gradient of the value function segment

        ########################

        t, x, y, theta, v, delta = ca.MX.sym('t'), ca.MX.sym('x'), ca.MX.sym('y'), ca.MX.sym('theta'), ca.MX.sym('v'), ca.MX.sym('delta')
        a, omega = ca.MX.sym('a'), ca.MX.sym('omega')  # control input
        state = ca.vertcat(x, y, theta, v, delta)
        u = ca.vertcat(a, omega)

        # Solve for U_set
        # Partial derivatives
        # print(value_func_gradient[0].shape)  # Partial derivative with respect to time
        Dt_V_ip = ca.interpolant("Dt_V_ip", "linear", self.grids_segment, value_func_gradient[0].ravel(order="F"))
        Dx_V_ip = ca.interpolant("Dx_V_ip", "linear", self.grids_segment, value_func_gradient[1:].ravel(order="F"))

        Dt_V = ca.Function('Dt_V', [t, state], [Dt_V_ip(ca.vertcat(t, state))])(t,state)
        Dx_V = ca.Function('Dx_V', [t, state], [Dx_V_ip(ca.vertcat(t, state))])(t,state).T

        f_1 =ca.vertcat(v * ca.cos(theta),
                        v * ca.sin(theta),
                        v * ca.tan(delta)/self.L,
                        0,
                        0)  # State dynamics
        f_2 = ca.MX(5,2)    # Control input dynamics
        f_2[3,0] = 1        # Acceleration
        f_2[4,1] = 1        # Steering rate

        alpha = Dt_V + ca.mtimes(Dx_V, f_1)
        beta = ca.mtimes(Dx_V, f_2)

        # Hamilton-Jacobi equation
        U_live_MX = alpha + ca.mtimes(beta,u) # Set up the linear system (alpha + beta * u)
        U_live = ca.Function('U_live', [t, state, u], [U_live_MX])

        U_locked_list = []
        predicted_states = np.array(self.trajectory)[:self.N,1:]
        assert predicted_states.shape[1] == 5

        # Create a function for the HJ equation with locked state
        
        for k in range(self.current_horizon):
            time = min(current_state[0] + k*self.dt, self.dt * self.dims[0])
            # print("Time:", time, "k:", k, "State:", predicted_states[k,:])
            U_locked = ca.Function(f'U_locked_{k}', [u], [U_live(time,predicted_states[k,:],u)])
            U_locked_list.append(copy.deepcopy(U_locked))

        return U_locked_list

    def solve_U_set(self, current_state, horizon):
        """
        Sets up the interpolated function for the control input cost.\n
        """
        self.local_input_cost = self.interpolate_U_segment(current_state, horizon)

        return
    
    def bound_initial_state(self, state):
        """
        This method checks if the initial state provided to the mpc, which could come from the localization stack,
        is within the allowed bounds. If not, it clamps it to make the optimization problem feasible.
        """
        # Ensure the velocity is within the specified bounds
        # print(state)
        clamped_velocity = max(self.min_velocity, min(state[3], self.max_velocity))

        # Create a new state with the clamped velocity
        bounded_state = state.copy()
        bounded_state[3] = clamped_velocity
        
        return bounded_state
    
    def compute_state_error(self, x, x_ref):
        """
        Computes the state error between the current state and the reference state.
        Adjusts the yaw error to account for angle wrapping in the range [-π, π].
        """
        state_error = x - x_ref
        yaw_diff = x[2] - x_ref[2]
        state_error[2] = ca.atan2(ca.sin(yaw_diff), ca.cos(yaw_diff))  # Minimum yaw error 
        return state_error

    def update_step(self, state, control):
        '''
        Updates the state from the car dynamics. Useful for warm-start function
        NOT CORRECT
        '''
        L = self.L # Wheelbase of the vehicle
        tau = 0.2  # Time constant

        x = state[1]        # x-coordinate
        y = state[2]        # y-coordinate
        theta = state[3]    # Yaw
        v = state[4]        # Velocity
        delta = state[5]    # Steering angle
        
        a =     min(max(control[0], self.min_acceleration), self.max_acceleration)      # Acceleration
        omega = min(max(control[1], self.min_steering_rate), self.max_steering_rate)  # Steering angle rate

        # Compute the state derivatives
        dt = 1
        dx = v * ca.cos(theta)
        dy = v * ca.sin(theta)
        dtheta = v * np.tan(delta) / L
        dv = a
        ddelta = omega

        dstate = [dt,dx,dy,dtheta,dv,ddelta]
        new_state = [float(state[i] + self.dt*dstate[i]) for i in range(6)]
        bounded_state = [min(self.max_bounds_state[i], max(self.min_bounds_state[i], new_state[i])) for i in range(6)]
        bounded_state[3] = new_state[3]
        return bounded_state
    
    def coord_to_index(self, coords):
        '''
        Translates a state coordinate to its corresponding index in the value function array
        '''
        indices = []
        for i in range(6):
            norm_value = (coords[i] - self.min_bounds_state[i]) / (self.max_bounds_state[i] - self.min_bounds_state[i])
            index = np.round(norm_value * (self.dims[i] - 1))
            index = max(0, min(index, self.dims[i] - 1))
            indices.append(int(index))

        return tuple(indices)
    
    def find_edge_of_time(self, state):
        '''
        Returns the upper bound in time for which the value function is negative for a given state
        '''
        index = list(self.coord_to_index([0] + state))  # convert tuple to list

        while self.value_function[tuple(index)] <= 0 and index[0] < self.dims[0]:
            index[0] += 1

        return index[0] * self.dt

def plot_path(Car, states, some_trajectory=None):
    '''
    Plots the path that the system has taken. Also shows cost throughout the trajectory
    '''
    # Extract x and y coordinates from the states
    x_coords = [state[1] for state in states]
    y_coords = [state[2] for state in states]

    # Create a new figure
    fig, ax = plt.subplots()

    # Plot blue circles with lines connecting them
    ax.plot(x_coords, y_coords, 'o-', color='blue', label='Path')
    if some_trajectory is not None:
        # Plot the predicted trajectory in red
        pred_x = [state[1] for state in some_trajectory]
        pred_y = [state[2] for state in some_trajectory]
        ax.plot(pred_x, pred_y, '--', color='red', label='Reference trajectory')

    # Add labels and title
    ax.set_xlabel('X-axis')
    ax.set_ylabel('Y-axis')
    ax.set_title('Path Visualization')
    ax.legend(loc='upper right')

    # Set the aspect ratio of the plot to be equal
    ax.axis('equal')
    ax.set_xticks(np.arange(Car.X0, Car.X0 + Car.XN + 0.5, 0.5))
    ax.set_yticks(np.arange(Car.Y0, Car.Y0 + Car.YN + 0.5, 0.5))

    # Plot a square with black lines at the limits
    ax.plot([Car.X0, Car.X0 + Car.XN, Car.X0 + Car.XN, Car.X0, Car.X0],
            [Car.Y0, Car.Y0, Car.Y0 + Car.YN, Car.Y0 + Car.YN, Car.Y0], 
            color='black', label='Limits')
    ax.plot([Car.X0 + Car.XN * 3/8, Car.X0 + Car.XN * 3/8, Car.X0 + Car.XN * 4/8, Car.X0 + Car.XN * 4/8, Car.X0 + Car.XN * 8/8],
            [Car.Y0 + Car.YN * 8/8, Car.Y0 + Car.YN * 3.5/8, Car.Y0 + Car.YN * 3.5/8, Car.Y0 + Car.YN * 4/8, Car.Y0 + Car.YN * 4/8], 
            color='black', label='Limits')
    ax.plot([Car.X0 + Car.XN * 3/8, Car.X0 + Car.XN * 3/8, Car.X0 + Car.XN * 4/8, Car.X0 + Car.XN * 4/8],
            [Car.Y0 + Car.YN * 0/8, Car.Y0 + Car.YN * 0.5/8, Car.Y0 + Car.YN * 0.5/8, Car.Y0 + Car.YN * 0/8], 
            color='black', label='Limits')
        
    # Show the plot
    ax.grid()
    fig.savefig("a_path_5D_i.png", dpi=300)

def plot_input(Car, inp):
    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(8, 6))

    inp = np.array(inp)
    # Acceleration subplot
    ax1.plot(inp[:,0,0], label="Acceleration", color='tab:blue')
    ax1.axhline(0, color='black', linestyle='--', linewidth=1)
    ax1.set_ylim(-0.2, 0.2)
    ax1.set_ylabel("Acceleration", labelpad=10)
    ax1.set_title("Inputs")
    ax1.legend()
    ax1.grid(True)

    # Steering rate subplot
    ax2.plot(inp[:,1,0], label="Steering rate", color='tab:orange')
    ax2.axhline(0, color='black', linestyle='--', linewidth=1)
    ax2.set_ylim(-np.pi, np.pi)
    ax2.set_xlabel("Iteration (nr)")
    ax2.set_ylabel("Steering rate", labelpad=10)
    ax2.legend()
    ax2.grid(True)

    fig.align_ylabels([ax1, ax2])
    fig.tight_layout()
    fig.savefig("a_input_5D_i.png", dpi=300)

def plot_cost(Car, costs):
    fig, ax = plt.subplots()
    costs = np.array(costs).T
    time_axis = np.arange(len(costs)) * 0.2  # Each iteration is 0.2 seconds apart
    
    ax.plot(time_axis, costs, color='green', label='Cost over time')

    ax.axhline(0, color='black', linestyle='--', linewidth=1, label='Zero Cost')
    ax.set_xlabel("Time Step")
    ax.set_ylabel("Cost")
    ax.set_title("Cost Over Time")
    ax.set_ylim(-0.05,0.01)
    ax.legend()
    fig.savefig("a_cost_5D_i.png", dpi=300)

def plot_comp_time(Car):
    fig, ax = plt.subplots()
    ct = Car.computation_times

    ax.plot(ct[:,0], label="Total time", color="royalblue")
    ax.plot(ct[:,1], label = "Generating cost function", color = "orange")
    ax.plot(ct[:,2], label = "Solving MPC", color = "green")

    avg_total = np.cumsum(ct[:, 0]) / (np.arange(len(ct)) + 1)
    avg_cost = np.cumsum(ct[:, 1]) / (np.arange(len(ct)) + 1)
    avg_solve = np.cumsum(ct[:, 2]) / (np.arange(len(ct)) + 1)

    ax.plot(avg_total, color = "black", linewidth=5)
    ax.plot(avg_total, color = "royalblue", linewidth=3)
    ax.plot(avg_cost, color = "black", linewidth=5)
    ax.plot(avg_cost, color = "orange", linewidth=3)
    ax.plot(avg_solve, color = "black", linewidth=5)
    ax.plot(avg_solve, color = "green", linewidth=3)

    ax.axhline(0, color='black', linestyle='--', linewidth=1)
    ax.set_xlabel("Iteration (nr)")
    ax.set_ylabel("Average Time (s)")
    ax.set_title("Average Computation Times")
    ax.legend()
    fig.savefig("a_time_5D_i.png", dpi=300)

def animate_heat_map(Car, states, V, nr, predicted_path):
    '''
    Animates the states of the car in two separate heatmaps.\n
    Shows the state, predicted states and the bound between positive and negative values in value function
    '''
    print("Animating heat map for interpolation")

    print("Interpolating value function over the whole array (will take a couple of minutes)")
    # Interpolate V linearly over the whole array
    dims = V.shape
    grids = [np.linspace(Car.min_bounds_state[i], Car.max_bounds_state[i], dims[i]) for i in range(6)]
    V_interpolated = ca.interpolant("V_interp", "linear", grids, V.ravel(order="F"))

    # Create a figure for animation
    fig, (ax, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))

    print("Creating animation frames")
    progress = tqdm(total=nr)

    def update(index_t):
        
        ax.clear()
        ax2.clear()
        ax3.clear()

        t     = states[index_t][0]
        x     = states[index_t][1]
        y     = states[index_t][2]
        theta = states[index_t][3]
        v     = states[index_t][4]
        delta = states[index_t][5]

        # Heatmap for x and y
        x_vals = np.linspace(Car.X0, Car.X0 + Car.XN, 100)
        y_vals = np.linspace(Car.Y0, Car.Y0 + Car.YN, 100)
        Z_xy = np.zeros((len(x_vals), len(y_vals)))

        for i, x_f in enumerate(x_vals):
            for j, y_f in enumerate(y_vals):
                Z_xy[i, j] = V_interpolated(ca.vertcat(t, x_f, y_f, theta, v, delta))

        c = ax.contourf(x_vals, y_vals, Z_xy.T, levels=50, cmap="viridis")
        ax.contour(x_vals, y_vals, Z_xy.T, levels=10, colors="black", linewidths=0.5)
        if Z_xy.max() > 0 and Z_xy.min() < 0:
            ax.contour(x_vals, y_vals, Z_xy.T, levels=[0], colors="red", linewidths=2.0)
        ax.set_xlabel('X-axis')
        ax.set_ylabel('Y-axis')
        ax.set_title('Heatmap of Value Function (X vs Y)')
        ax.axis('equal')
        ax.plot(predicted_path[index_t][0], predicted_path[index_t][1], "yo")
        ax.plot(x, y, "ro")

        # Heatmap for theta and v
        theta_vals = np.linspace(-np.pi, np.pi, 100)
        v_vals = np.linspace(Car.min_velocity, Car.max_velocity, 100)
        Z_theta_v = np.zeros((len(theta_vals), len(v_vals)))

        for i, theta_f in enumerate(theta_vals):
            for j, v_f in enumerate(v_vals):
                Z_theta_v[i, j] = V_interpolated(ca.vertcat(t, x, y, theta_f, v_f, delta))

        c2 = ax2.contourf(theta_vals, v_vals, Z_theta_v.T, levels=50, cmap="viridis")
        ax2.contour(theta_vals, v_vals, Z_theta_v.T, levels=10, colors="black", linewidths=0.5)
        if Z_theta_v.max() > 0 and Z_theta_v.min() < 0:
            ax2.contour(theta_vals, v_vals, Z_theta_v.T, levels=[0], colors="red", linewidths=2.0)
        ax2.set_xlabel('Theta-axis')
        ax2.set_ylabel('Velocity-axis')
        ax2.set_title('Heatmap of Value Function (Theta vs Velocity)')
        ax2.plot(predicted_path[index_t][2], predicted_path[index_t][3], "yo")
        ax2.plot(theta, v, "ro")

        # Heatmap for theta and delta
        theta_vals = np.linspace(-np.pi, np.pi, 100)
        delta_vals = np.linspace(Car.min_steering, Car.max_steering, 100)
        Z_theta_delta = np.zeros((len(theta_vals), len(delta_vals)))

        for i, theta_f in enumerate(theta_vals):
            for j, delta_f in enumerate(delta_vals):
                Z_theta_v[i, j] = V_interpolated(ca.vertcat(t, x, y, theta_f, v, delta_f))

        c3 = ax3.contourf(theta_vals, delta_vals, Z_theta_delta.T, levels=50, cmap="viridis")
        ax3.contour(theta_vals, delta_vals, Z_theta_delta.T, levels=10, colors="black", linewidths=0.5)
        if Z_theta_delta.max() > 0 and Z_theta_delta.min() < 0:
            ax3.contour(theta_vals, delta_vals, Z_theta_delta.T, levels=[0], colors="red", linewidths=2.0)
        ax3.set_xlabel('Theta-axis')
        ax3.set_ylabel('Delta-axis')
        ax3.set_title('Heatmap of Value Function (Theta vs Delta)')
        ax3.plot(predicted_path[index_t][2], predicted_path[index_t][4], "yo")
        ax3.plot(theta, delta, "ro")

        progress.update(1)
        return c, c2, c3

    # Create animation
    ani = FuncAnimation(fig, update, frames=range(0, nr), interval=100, blit=False)
    
    # Save animation as a gif
    ani.save("animation_5D_i.gif", writer="pillow", fps=10)
    progress.close()

def animate_heat_map_segment(Car, states, nr, predicted_path):
    '''
    Animates the states of the car in two separate heatmaps.\n
    Shows the state, predicted states, the bound between positive and negative values in value function and the bounded square in which the interpolation takes place
    '''
    print("Animating heat map for interpolation segmentation")

    # Create a figure for animation
    fig, (ax, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))

    print("Creating animation frames")
    progress = tqdm(total=nr)

    def update(index_t):
        V_interpolated_seg = Car.interpolate_V_segment(states[index_t], dyn.prediction_horizon)

        ax.clear()
        ax2.clear()
        ax3.clear()

        t =     states[index_t][0]
        x =     states[index_t][1]
        y =     states[index_t][2]
        theta = states[index_t][3]
        v =     states[index_t][4]
        delta = states[index_t][5]

        # Heatmap for x and y
        x_vals = np.linspace(Car.X0, Car.X0 + Car.XN, 100)
        y_vals = np.linspace(Car.Y0, Car.Y0 + Car.YN, 100)
        Z_xy = np.zeros((len(x_vals), len(y_vals)))

        for i, x_f in enumerate(x_vals):
            for j, y_f in enumerate(y_vals):
                Z_xy[i, j] = V_interpolated_seg(ca.vertcat(t, x_f, y_f, theta, v, delta))

        c = ax.contourf(x_vals, y_vals, Z_xy.T, levels=50, cmap="viridis")
        ax.contour(x_vals, y_vals, Z_xy.T, levels=10, colors="black", linewidths=0.5)
        if Z_xy.max() > 0 and Z_xy.min() < 0:
            ax.contour(x_vals, y_vals, Z_xy.T, levels=[0], colors="red", linewidths=2.0)
        ax.set_xlabel('X-axis')
        ax.set_ylabel('Y-axis')
        ax.set_title('Heatmap of Value Function (X vs Y)')
        ax.axis('equal')
        ax.plot(predicted_path[index_t][0], predicted_path[index_t][1], "yo")
        ax.plot(x, y, "ro")
        

        # Heatmap for theta and v
        theta_vals = np.linspace(-np.pi, np.pi, 100)
        v_vals = np.linspace(Car.min_velocity, Car.max_velocity, 100)
        Z_theta_v = np.zeros((len(theta_vals), len(v_vals)))

        for i, theta_f in enumerate(theta_vals):
            for j, v_f in enumerate(v_vals):
                Z_theta_v[i, j] = V_interpolated_seg(ca.vertcat(t, x, y, theta_f, v_f, delta))

        c2 = ax2.contourf(theta_vals, v_vals, Z_theta_v.T, levels=50, cmap="viridis")
        ax2.contour(theta_vals, v_vals, Z_theta_v.T, levels=10, colors="black", linewidths=0.5)
        if Z_theta_v.max() > 0 and Z_theta_v.min() < 0:
            ax2.contour(theta_vals, v_vals, Z_theta_v.T, levels=[0], colors="red", linewidths=2.0)
        ax2.set_xlabel('Theta-axis')
        ax2.set_ylabel('Velocity-axis')
        ax2.set_title('Heatmap of Value Function (Theta vs Velocity)')
        ax2.plot(predicted_path[index_t][2], predicted_path[index_t][3], "yo")
        ax2.plot(theta, v, "ro")
        
        # Heatmap for theta and delta
        theta_vals = np.linspace(-np.pi, np.pi, 100)
        delta_vals = np.linspace(Car.min_steering, Car.max_steering, 100)
        Z_theta_delta = np.zeros((len(theta_vals), len(delta_vals)))

        for i, theta_f in enumerate(theta_vals):
            for j, delta_f in enumerate(delta_vals):
                Z_theta_delta[i, j] = V_interpolated_seg(ca.vertcat(t, x, y, theta_f, v, delta_f))

        c3 = ax3.contourf(theta_vals, delta_vals, Z_theta_delta.T, levels=50, cmap="viridis")
        ax3.contour(theta_vals, delta_vals, Z_theta_delta.T, levels=10, colors="black", linewidths=0.5)
        if Z_theta_delta.max() > 0 and Z_theta_delta.min() < 0:
            ax3.contour(theta_vals, delta_vals, Z_theta_delta.T, levels=[0], colors="red", linewidths=2.0)
        ax3.set_xlabel('Theta-axis')
        ax3.set_ylabel('Delta-axis')
        ax3.set_title('Heatmap of Value Function (Theta vs Delta)')
        ax3.plot(predicted_path[index_t][2], predicted_path[index_t][4], "yo")
        ax3.plot(theta, delta, "ro")

        seg = Car.grids_segment_list
        ax.plot( [seg[index_t][1][0], seg[index_t][1][-1],seg[index_t][1][-1], seg[index_t][1][0], seg[index_t][1][0]],
                 [seg[index_t][2][0], seg[index_t][2][0], seg[index_t][2][-1], seg[index_t][2][-1],seg[index_t][2][0]],"y")
        
        ax2.plot([seg[index_t][3][0], seg[index_t][3][-1],seg[index_t][3][-1], seg[index_t][3][0], seg[index_t][3][0]],
                 [seg[index_t][4][0], seg[index_t][4][0], seg[index_t][4][-1], seg[index_t][4][-1],seg[index_t][4][0]],"y")

        ax3.plot([seg[index_t][3][0], seg[index_t][3][-1],seg[index_t][3][-1], seg[index_t][3][0], seg[index_t][3][0]],
                 [seg[index_t][5][0], seg[index_t][5][0], seg[index_t][5][-1], seg[index_t][5][-1],seg[index_t][5][0]],"y")

        progress.update(1)
        return c, c2, c3

    # Create animation
    ani = FuncAnimation(fig, update, frames=range(0, nr), interval=100, blit=False)
    
    # Save animation as a gif
    ani.save("animation_seg_5D_i.gif", writer="pillow", fps=10)
    progress.close()

def animate_heat_map_input(Car, states, inputs, nr, predicted_inputs = None):
    '''
    Animates the inputs of the car in a heatmaps.\n
    Shows the input, predicted inputs and the bound between positive and negative values in value function
    '''
    print("Animating heat map for input interpolation")

    # Interpolate V linearly over the whole array

    # Create a figure for animation
    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    print("Creating animation frames")
    progress = tqdm(total=nr)

    def update(index_t):
        Car.solve_U_set(states[index_t], dyn.prediction_horizon)
        U_interpolated = Car.local_input_cost[0]

        ax.clear()

        a = inputs[index_t][1]
        omega = inputs[index_t][2]

        # Heatmap for theta and a
        a_vals = np.linspace(Car.min_acceleration, Car.max_acceleration, 100)
        omega_vals = np.linspace(Car.min_steering_rate, Car.max_steering_rate, 100)
        Z_aomega = np.zeros((len(a_vals), len(omega_vals)))

        for i, a_f in enumerate(a_vals):
            for j, omega_f in enumerate(omega_vals):
                Z_aomega[i, j] = U_interpolated(ca.vertcat(a_f, omega_f))

        min_val = np.nanmin(Z_aomega)
        max_val = np.nanmax(Z_aomega)

        levels = np.linspace(min_val, max_val + 1e-5, 50)

        c = ax.contourf(a_vals, omega_vals, Z_aomega.T, levels=levels, cmap="viridis")
        ax.contour(a_vals, omega_vals, Z_aomega.T, levels=10, colors="black", linewidths=0.5)
        if Z_aomega.max() > 0 and Z_aomega.min() < 0:
            ax.contour(a_vals, omega_vals, Z_aomega.T, levels=[0], colors="red", linewidths=2.0)
        ax.set_xlabel('a-axis')
        ax.set_ylabel('omega-axis')
        ax.set_title('Heatmap of Value Function for inputs (STEERING RATE vs ACCELERATION)')
        # ax.axis('equal')
        if predicted_inputs is not None:
            ax.plot(predicted_inputs[index_t][0], predicted_inputs[index_t][1], "yo")
        ax.plot(a, omega, "ro")

        progress.update(1)

        return c

    # Create animation
    # print(inputs)
    ani = FuncAnimation(fig, update, frames=range(0, nr-4), interval=100, blit=False)
    
    # Save animation as a gif
    ani.save("animation_5D_input_i.gif", writer="pillow", fps=10)

    progress.close()

def main():
    V = get_value_function()

    states = [[0, 1.25, -1, np.pi, 0.0, 0.0]]
    nr = 120
    tot_it = 120

    Car = MPC_casadi(states[0], tot_it)
    print("Starting value:",V[Car.coord_to_index(states[0])])
    inputs = []
    predicted_path = []
    predicted_input = []
    computation_times = []
    costs_list = []
    u_costs_list = []
    
    some_inputs = np.zeros([2,nr+2])  # Initial inputs

    ### Valid trajectory ###
    for i in range( 0, 75): some_inputs[0,i] = 0.2
    for i in range(40,45): some_inputs[1,i] = -0.5
    for i in range(60,64): some_inputs[1,i] = 0.5
    some_inputs[1,65] = 0.2
    
    ### Faulty trajectory for testing purposes ###
    # for i in range( 0, 75): some_inputs[0,i] = 0.2
    # for i in range(37,42): some_inputs[1,i] = -0.5
    # for i in range(57,60): some_inputs[1,i] = 0.5
    # some_inputs[1,61] = 0.2

    reference_trajectory = [states[0]]
    for k in range(nr+2):
        some_next = Car.update_step(reference_trajectory[-1], some_inputs[:,k])
        reference_trajectory.append(some_next)

    for it in tqdm(range(tot_it)):
        time = states[-1][0]
        n = int(time/Car.dt)
        Car.compute_control(states[-1][1:], time, reference_trajectory[n:],it)

        ns = Car.get_optimal_states()   #new_states
        nu = Car.get_optimal_control()  #new_control

        next_state = [ns[i][1] for i in range(5)]
        next_input = [nu[i][0] for i in range(2)]

        while next_state[2] < -np.pi: next_state[2] += 2*np.pi
        while next_state[2] >  np.pi: next_state[2] -= 2*np.pi

        predicted_path.append(ns)
        predicted_input.append(nu)

        u_costs = [Car.local_input_cost[i](ca.vertcat(nu[0][i], nu[1][i])).full().item() for i in range(Car.current_horizon)]
        u_costs_list.append(u_costs)

        # Possible to stop time when V<0 is violated
        # state_cost = V[Car.coord_to_index([time + 2*Car.dt] + next_state)]
        # if state_cost <= 0:
        #     states.append([time + Car.dt] + next_state)
        # else:
        #     states.append([time] + next_state)
        states.append([time + Car.dt] + next_state)

        inputs.append([time] + next_input)

    v_costs = []
    for state in states:
        index = Car.coord_to_index(state)
        v_costs.append(V[index])


    plot_path(Car, states, reference_trajectory)
    plot_input(Car, predicted_input)
    plot_cost(Car, v_costs)
    plot_comp_time(Car)
    # animate_heat_map(Car, states, V, nr, predicted_path)
    # animate_heat_map_segment(Car, states, nr, predicted_path)
    # animate_heat_map_input(Car, states, inputs, nr, predicted_inputs=predicted_input)

    # plt.legend(["x","y","theta","v","delta"])
    # plt.plot(np.array(reference_trajectory)[:nr-1, 1:],"--")
    plt.show()

main()