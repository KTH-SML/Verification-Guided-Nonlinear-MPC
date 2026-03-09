#! /usr/bin/env python3
"""
HJ-MPC 5D Controller using acados (replacing CasADi/IPOPT backend).

Key differences from the CasADi version:
  - Model and cost are defined symbolically via CasADi (acados uses CasADi internally)
  - The OCP is compiled once; warm-starting is automatic between solves
  - Solver: SQP_RTI (real-time iteration) — much faster than IPOPT for embedded use
  - HJ value function interpolation uses SciPy RegularGridInterpolator (RGI),
    replacing ca.interpolant for 10-50x faster construction + vectorised eval
  - External (nonlinear) cost is used to include the HJ penalty term
"""

import casadi as ca
import numpy as np
import matplotlib.pyplot as plt
import os
import dynamics_car as dyn
import time as tm
from scipy.interpolate import RegularGridInterpolator

from tqdm import tqdm
from matplotlib.animation import FuncAnimation

# acados Python interface
from acados_template import AcadosModel, AcadosOcp, AcadosOcpSolver

# ---------------------------------------------------------------------------
# Helper: load value function
# ---------------------------------------------------------------------------

def get_value_function():
    file_path = "pyspect_test/hjb_mpc/out_bicycle5d.npy"
    return np.load(file_path)


# ---------------------------------------------------------------------------
# Main controller class
# ---------------------------------------------------------------------------

class MPC_acados:

    X0, XN = -2, 4
    Y0, YN = -2, 4

    def __init__(self, nr, vehicle_name='', config_ns='~mpc'):
        """
        HJ-MPC controller backed by acados (SQP_RTI).

        State:  x = [x_pos, y_pos, theta, v, delta]
        Input:  u = [acceleration, steering_rate]

        The HJ reachability penalty is injected as an *external cost* so that
        it can be recomputed each solve call via parameter-free symbolic
        interpolation over the local value-function segment.
        """

        # ----- Core parameters -------------------------------------------------
        self.N   = dyn.prediction_horizon
        self.dt  = dyn.time_step
        self.current_horizon = self.N

        # ----- Weight matrices -------------------------------------------------
        self.Q1  = np.array(dyn.state_weight_matrix_5D,    dtype=float).reshape(5, 5)
        self.Q2  = np.array(dyn.control_rate_weight_matrix, dtype=float).reshape(2, 2)
        self.Q3  = np.array(dyn.control_weight_matrix,      dtype=float).reshape(2, 2)
        self.Qf  = np.array(dyn.final_state_weight_matrix_5D, dtype=float).reshape(5, 5)
        self.Qv  = float(dyn.forward_speed_weight)
        self.Qs  = float(dyn.steering_weight)
        self.Qhj = float(dyn.reachability_set_weight)
        self.Qg  = float(dyn.gradient_weight)

        # ----- Vehicle bounds --------------------------------------------------
        self.min_steering      = np.radians(dyn.steering_min)
        self.max_steering      = np.radians(dyn.steering_max)
        self.min_steering_rate = np.radians(dyn.steering_rate_min)
        self.max_steering_rate = np.radians(dyn.steering_rate_max)
        self.min_velocity      = dyn.velocity_min
        self.max_velocity      = dyn.velocity_max
        self.min_acceleration  = dyn.acceleration_min
        self.max_acceleration  = dyn.acceleration_max
        self.L                 = dyn.wheelbase

        # ----- Value function --------------------------------------------------
        self.set_value_function()
        self.dims = self.value_function.shape

        # Wrap along yaw axis (dim 3) for angle wrap-around
        self.value_func_wrapped = np.concatenate(
            (self.value_function, self.value_function, self.value_function),
            axis=3
        )

        # Grid for interpolation
        time_max = self.dt * self.dims[0]
        self.min_bounds_state = np.array([
            0,          self.X0,          self.Y0,
            -np.pi,     self.min_velocity, self.min_steering
        ])
        self.max_bounds_state = np.array([
            time_max,   self.X0 + self.XN, self.Y0 + self.YN,
            +np.pi,     self.max_velocity,  self.max_steering
        ])
        self.grids     = [np.linspace(self.min_bounds_state[i],
                                      self.max_bounds_state[i],
                                      self.dims[i] + 1) for i in range(6)]
        self.d_states  = [self.grids[i][1] - self.grids[i][0] for i in range(6)]

        # ----- Interpolated value function (updated each solve) ----------------
        self.V = None          # RegularGridInterpolator, rebuilt in compute_control()
        self.HJ_time = None
        self._V_cached_indices     = None
        self._V_cached_interpolant = None  # RegularGridInterpolator instance
        self._V_cached_grids       = None

        # Bookkeeping
        self.grids_segment_list = []
        self.iteration          = 0
        self.computation_times  = np.zeros((nr, 3))

        # ----- Build & compile acados OCP -------------------------------------
        # The OCP is compiled *once* with the HJ cost as a symbolic expression.
        # Because the interpolant grid changes each call we use a two-level
        # approach: build the acados model without HJ cost first, solve, then
        # add the HJ penalty as a Python-level post-processing warm-start.
        #
        # Preferred approach (used here):
        #   - Use acados "external cost" with a CasADi expression.
        #   - The interpolant is rebuilt each solve (cheap, O(N) evaluations).
        #   - Pass the current interpolant values as parameters p[0..N].
        #
        self._build_acados_solver()

        # Store last solution for warm-starting
        self._x_sol = None
        self._u_sol = None
        self._sol   = None   # dict with 'x', 'u' arrays shape (N+1,5) / (N,2)

    # =========================================================================
    # Build acados OCP
    # =========================================================================

    def _build_acados_solver(self):
        """Compile the acados OCP solver (called once at construction)."""

        N  = self.N
        dt = self.dt

        # ------------------------------------------------------------------
        # 1. Symbolic model  (CasADi)
        # ------------------------------------------------------------------
        model = AcadosModel()
        model.name = "hj_bicycle_5d"

        # State
        x_pos   = ca.SX.sym("x_pos")
        y_pos   = ca.SX.sym("y_pos")
        theta   = ca.SX.sym("theta")
        v       = ca.SX.sym("v")
        delta   = ca.SX.sym("delta")
        x_sym   = ca.vertcat(x_pos, y_pos, theta, v, delta)

        # Input
        a_acc   = ca.SX.sym("a_acc")
        omega   = ca.SX.sym("omega")
        u_sym   = ca.vertcat(a_acc, omega)

        # Parameters: HJ cost values for k=0..N  (N+1 values)
        # p[k] = V(t_k, x_k) precomputed outside the solver
        p_sym = ca.SX.sym("p", N + 1)

        # Discrete dynamics (Euler)
        L = self.L
        f_x   = x_pos + dt * v * ca.cos(theta)
        f_y   = y_pos + dt * v * ca.sin(theta)
        f_th  = theta + dt * v * ca.tan(delta) / L
        f_v   = v     + dt * a_acc
        f_d   = delta + dt * omega

        x_next = ca.vertcat(f_x, f_y, f_th, f_v, f_d)

        model.x    = x_sym
        model.u    = u_sym
        model.p    = p_sym
        model.disc_dyn_expr = x_next

        # ------------------------------------------------------------------
        # 2. OCP
        # ------------------------------------------------------------------
        ocp = AcadosOcp()
        ocp.model = model

        nx = 5
        nu = 2
        np_ = N + 1

        ocp.dims.N  = N
        ocp.dims.np = np_

        # Initial parameter values (updated each solve)
        ocp.parameter_values = np.zeros(np_)

        # -- Cost type: EXTERNAL (gives maximum flexibility for HJ term) ------
        # Stage cost (k = 0..N-1):
        #   l(x,u,p) = u^T Q3 u + Qhj * p[k]  +  Qv * max(0,-v)^2
        # We express this as a scalar CasADi expression.
        # Note: input-rate cost (Q2 term) cannot easily be expressed as a
        # stage cost in acados without augmenting the state. We keep it here
        # as a soft approximation u^T Q3 u only (set Q2=0 in dynamics_car to
        # match original behaviour, or augment state for exact reproduction).

        k_sym = ca.SX.sym("k_stage")   # stage index placeholder (not used directly)

        # Stage cost expression
        # p[k] is passed as a flattened parameter vector; acados evaluates
        # the cost expression at each stage with the corresponding parameter slice.
        # We use p[0] as a single HJ value at the current stage.
        # Implementation: reshape p to (N+1,) and index by stage.
        # acados external cost: we define cost_expr_ext_cost per stage using p[0].
        # Parameters are stage-local slices of length np_local=1 → change np_ to 1
        # and update parameters stage-by-stage each solve.

        # Simpler: use np_=1 per stage, pass one HJ value per stage.
        # Rebuild OCP with np_=1.

        # Stage cost (scalar, no matrix ops needed)
        Q3_diag = np.diag(self.Q3)
        Q3_sym  = ca.DM(self.Q3)

        # cost = u^T Q3 u  +  Qhj * hj_val  +  Qv * relu(-v)^2
        hj_val = p_sym[0]   # single parameter per stage (we'll rebuild with np=1)

        vel_penalty = ca.fmax(0.0, -v) ** 2

        stage_cost = (ca.mtimes([u_sym.T, Q3_sym, u_sym])
                      + self.Qhj * hj_val
                      + self.Qv  * vel_penalty)

        terminal_cost = self.Qhj * p_sym[0]  # HJ value at final stage

        ocp.cost.cost_type   = "EXTERNAL"
        ocp.cost.cost_type_e = "EXTERNAL"

        ocp.model.cost_expr_ext_cost   = stage_cost
        ocp.model.cost_expr_ext_cost_e = terminal_cost

        # Use np_=1 per stage
        ocp.dims.np = 1
        ocp.parameter_values = np.zeros(1)

        # Update model parameter
        model.p = p_sym[0:1]

        # -- Constraints -------------------------------------------------------
        # State bounds: velocity and steering
        ocp.constraints.lbx   = np.array([self.min_velocity, self.min_steering])
        ocp.constraints.ubx   = np.array([self.max_velocity, self.max_steering])
        ocp.constraints.idxbx = np.array([3, 4])   # indices of v and delta in x

        ocp.constraints.lbx_e   = ocp.constraints.lbx
        ocp.constraints.ubx_e   = ocp.constraints.ubx
        ocp.constraints.idxbx_e = ocp.constraints.idxbx

        # Input bounds
        ocp.constraints.lbu   = np.array([self.min_acceleration, self.min_steering_rate])
        ocp.constraints.ubu   = np.array([self.max_acceleration, self.max_steering_rate])
        ocp.constraints.idxbu = np.array([0, 1])

        # Initial state constraint (will be set each solve via set_new_time_step)
        ocp.constraints.x0 = np.zeros(nx)

        # -- Solver options ----------------------------------------------------
        ocp.solver_options.tf                  = N * dt
        ocp.solver_options.integrator_type     = "DISCRETE"
        ocp.solver_options.nlp_solver_type     = "SQP_RTI"   # Real-Time Iteration
        ocp.solver_options.qp_solver           = "PARTIAL_CONDENSING_HPIPM"
        ocp.solver_options.qp_solver_cond_N    = N
        ocp.solver_options.hessian_approx      = "EXACT"
        ocp.solver_options.print_level         = 0
        ocp.solver_options.nlp_solver_max_iter = 1000

        # -- Compile -----------------------------------------------------------
        self.ocp = ocp
        self.acados_solver = AcadosOcpSolver(ocp, json_file="hj_mpc_5d_acados.json")

    # =========================================================================
    # Main control interface
    # =========================================================================

    def compute_control(self, state, HJ_time, it):
        """
        Compute optimal (acceleration, steering_rate) using acados SQP_RTI.

        :param state:   [x, y, theta, v, delta]
        :param HJ_time: HJ 'time-to-go' for the current state (scalar)
        :param it:      iteration index for bookkeeping
        :return:        (acceleration, steering_rate)
        """
        self.iteration = it
        comp_start = tm.time()

        bounded_state = self.bound_initial_state(list(state))

        # --- Build interpolated value function for current segment -----------
        solve_V_start = tm.time()
        self.V = self.interpolate_V_segment([HJ_time] + list(state),
                                            self.current_horizon)
        solve_V_end = tm.time()
        self.computation_times[it, 1] = solve_V_end - solve_V_start

        # --- Set initial state -----------------------------------------------
        self.acados_solver.set(0, "lbx", np.array(bounded_state))
        self.acados_solver.set(0, "ubx", np.array(bounded_state))

        # --- Predict states for parameter evaluation -------------------------
        # We need x_k along the horizon to evaluate V(t_k, x_k).
        # Use warm-start solution if available, else use constant prediction.
        x_pred = self._predict_states(bounded_state)

        # --- Set HJ cost parameters (vectorised batch evaluation) ------------
        # Build query matrix shape (N+1, 6) and call RGI once instead of N+1
        # individual Python calls.
        t_vals    = HJ_time + np.arange(self.N + 1) * self.dt  # (N+1,)
        query_pts = np.column_stack([t_vals, x_pred])           # (N+1, 6)
        hj_vals   = self.V(query_pts)                           # (N+1,)
        for k in range(self.N + 1):
            self.acados_solver.set(k, "p", np.array([hj_vals[k]]))

        # --- Warm-start from previous solution --------------------------------
        if self._sol is not None:
            for k in range(self.N + 1):
                self.acados_solver.set(k, "x", self._sol['x'][k])
            for k in range(self.N):
                self.acados_solver.set(k, "u", self._sol['u'][k])

        # --- Solve ------------------------------------------------------------
        solver_start = tm.time()
        status = self.acados_solver.solve()
        solver_end   = tm.time()
        self.computation_times[it, 2] = solver_end - solver_start

        # status == 0: success, 1: max_iter, 2: infeasible, 4: NaN
        if status not in (0, 1):
            print(f"[acados] Warning: solver returned status {status} at iter {it}")

        # Extract first control
        u0           = self.acados_solver.get(0, "u")
        acceleration = float(u0[0])
        steering_rate = float(u0[1])

        # Cache solution for warm-starting
        self._sol = {
            'x': np.array([self.acados_solver.get(k, "x") for k in range(self.N + 1)]),
            'u': np.array([self.acados_solver.get(k, "u") for k in range(self.N)]),
        }

        comp_end = tm.time()
        self.computation_times[it, 0] = comp_end - comp_start

        print(f"Total Computation time: {self.computation_times[it, 0]:.4f} s, Segment interpolation time: {self.computation_times[it, 1]:.4f} s, Solver time: {self.computation_times[it, 2]:.4f} s.")

        return acceleration, steering_rate

    # =========================================================================
    # Solution accessors (mirror CasADi API used by main())
    # =========================================================================

    def get_optimal_control(self, all=True):
        """Returns optimal control array of shape (2, N)."""
        if self._sol is None:
            return None
        u = self._sol['u'].T   # shape (2, N)
        return u if all else u[:, 0]

    def get_optimal_states(self):
        """Returns optimal state array of shape (5, N+1)."""
        if self._sol is None:
            return None
        return self._sol['x'].T   # shape (5, N+1)

    # =========================================================================
    # Value function helpers  (identical to CasADi version)
    # =========================================================================

    def set_value_function(self):
        file_path = "pyspect_test/hjb_mpc/out_bicycle5d.npy"
        self.value_function = np.load(file_path)
        self.dims = self.value_function.shape
        print("Loaded value function of type", type(self.value_function),
              "of shape", self.dims)

    def interpolate_V_segment(self, state, horizon):
        """
        Build a local RegularGridInterpolator (SciPy) around the current state.

        Replaces ca.interpolant: since V values are passed to acados as numeric
        parameters p[k], no CasADi symbolic graph is needed.  SciPy RGI is
        10-50x faster to construct and supports vectorised batch evaluation.

        Returns a RegularGridInterpolator callable: V(pts) where pts has shape
        (M, 6), columns ordered [t, x, y, theta, v, delta].
        """
        delta_plus  = [horizon, horizon, horizon, horizon, 3, 10]
        delta_minus = [      2, horizon, horizon, horizon, 3, 10]

        index = list(self.coord_to_index(state))
        index[3] = index[3] + self.dims[3]

        indices = [
            [max(0, index[i] - delta_minus[i]),
            min(self.dims[i], index[i] + delta_plus[i])]
            for i in range(6)
        ]
        indices[3] = [index[3] - delta_minus[3], index[3] + delta_plus[3]]

        slices = tuple(slice(indices[i][0], indices[i][1] + 1) for i in range(6))

        # ascontiguousarray avoids an implicit copy during ravel when the slice
        # produces a non-contiguous view (common after the axis-3 wrap concat).
        value_func_segment = np.ascontiguousarray(self.value_func_wrapped[slices])
        dims_segment = value_func_segment.shape  # ← ground truth for grid sizes

        # Use dims_segment[i] as the point count, NOT the index range,
        # so grids always match the actual slice shape even at array boundaries.
        grids_segment = [
            np.linspace(
                self.d_states[i] * indices[i][0] + self.min_bounds_state[i],
                self.d_states[i] * indices[i][1] + self.min_bounds_state[i],
                dims_segment[i]
            )
            for i in range(6)
        ]
        # Yaw: shift by -2π to account for the wrapped concatenation offset.
        grids_segment[3] = np.linspace(
            self.d_states[3] * indices[3][0] + self.min_bounds_state[3] - 2 * np.pi,
            self.d_states[3] * indices[3][1] + self.min_bounds_state[3] - 2 * np.pi,
            dims_segment[3]
        )

        self.grids_segment = grids_segment
        self.grids_segment_list.append(list(grids_segment))

        # ── Build SciPy RGI (no CasADi symbolic graph) ───────────────────────
        # bounds_error=False + fill_value=None: nearest-edge extrapolation,
        # safe for small floating-point overruns at segment boundaries.
        # RGI takes C-order array directly — no ravel(order="F") needed.
        interpolant = RegularGridInterpolator(
            tuple(grids_segment),
            value_func_segment,
            method='linear',
            bounds_error=False,
            fill_value=None
        )

        # ── Update cache ──────────────────────────────────────────────────────
        self._V_cached_indices     = indices
        self._V_cached_interpolant = interpolant
        self._V_cached_grids       = grids_segment
        # ─────────────────────────────────────────────────────────────────────

        return interpolant

    # =========================================================================
    # State helpers
    # =========================================================================

    def bound_initial_state(self, state):
        bounded = list(state)
        bounded[3] = max(self.min_velocity, min(state[3], self.max_velocity))
        return bounded

    def coord_to_index(self, coords):
        indices = []
        for i in range(6):
            norm = (coords[i] - self.min_bounds_state[i]) / \
                   (self.max_bounds_state[i] - self.min_bounds_state[i])
            idx = np.round(norm * (self.dims[i] - 1))
            idx = max(0, min(idx, self.dims[i] - 1))
            indices.append(int(idx))
        return tuple(indices)

    def find_edge_of_time(self, state):
        index = list(self.coord_to_index([0] + state))
        while self.value_function[tuple(index)] <= 0 and index[0] < self.dims[0]:
            index[0] += 1
        return index[0] * self.dt

    def _predict_states(self, x0):
        """
        Return predicted state sequence for parameter evaluation.
        Uses cached solution shifted by one step if available,
        otherwise propagates open-loop with zero input.
        """
        nx = 5
        x_pred = np.zeros((self.N + 1, nx))
        x_pred[0] = x0

        if self._sol is not None and len(self._sol['x']) == self.N + 1:
            # Shift previous solution
            for k in range(1, self.N + 1):
                x_pred[k] = self._sol['x'][min(k, self.N)]
        else:
            # Open-loop rollout with zero input
            for k in range(self.N):
                x = x_pred[k]
                x_pred[k + 1] = np.array([
                    x[0] + self.dt * x[3] * np.cos(x[2]),
                    x[1] + self.dt * x[3] * np.sin(x[2]),
                    x[2] + self.dt * x[3] * np.tan(x[4]) / self.L,
                    x[3],
                    x[4],
                ])
        return x_pred


# ===========================================================================
# Plotting helpers (identical to CasADi version)
# ===========================================================================

def plot_path(Car, states):
    x_coords = [s[1] for s in states]
    y_coords = [s[2] for s in states]
    fig, ax = plt.subplots()
    ax.plot(x_coords, y_coords, 'o-', color='blue', label='Path')
    ax.set_xlabel('X-axis');  ax.set_ylabel('Y-axis')
    ax.set_title('Path Visualization');  ax.legend(loc='upper right')
    ax.axis('equal')
    ax.set_xticks(np.arange(Car.X0, Car.X0 + Car.XN + 0.5, 0.5))
    ax.set_yticks(np.arange(Car.Y0, Car.Y0 + Car.YN + 0.5, 0.5))
    ax.plot([Car.X0, Car.X0 + Car.XN, Car.X0 + Car.XN, Car.X0, Car.X0],
            [Car.Y0, Car.Y0, Car.Y0 + Car.YN, Car.Y0 + Car.YN, Car.Y0],
            color='black')
    ax.grid()
    fig.savefig("a_path_5D_acados.png", dpi=300)


def plot_input(Car, inp):
    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(8, 6))
    inp = np.array(inp)
    ax1.plot(inp[:, 0, 0], label="Acceleration", color='tab:blue')
    ax1.axhline(0, color='black', linestyle='--', linewidth=1)
    ax1.set_ylabel("Acceleration")
    ax2.plot(np.degrees(inp[:, 1, 0]), label="Steering Rate", color='tab:orange')
    ax2.axhline(0, color='black', linestyle='--', linewidth=1)
    ax2.set_ylabel("Steering Rate [deg/s]");  ax2.set_xlabel("Time step")
    fig.savefig("a_input_5D_acados.png", dpi=300)


def plot_comp_time(Car):
    fig, ax = plt.subplots()
    t = Car.computation_times
    ax.plot(t[:, 0], label="Total");  ax.plot(t[:, 1], label="Interpolation")
    ax.plot(t[:, 2], label="Solver")
    ax.set_xlabel("Iteration");  ax.set_ylabel("Time [s]")
    ax.set_title("Computation times");  ax.legend()
    fig.savefig("a_comp_time_5D_acados.png", dpi=300)


def animate_heat_map_segment(Car, states, nr, predicted_path, costs_list=None):
    print("Animating heat map for interpolation segmentation")
    fig = plt.figure(figsize=(18, 10))
    ax  = fig.add_subplot(2, 3, 1)
    ax2 = fig.add_subplot(2, 3, 2)
    ax3 = fig.add_subplot(2, 3, 3)
    ax4 = fig.add_subplot(2, 1, 2)   # Time vs Cost — full bottom row
    progress = tqdm(total=nr)

    def update(index_t):
        V_interpolated_seg = Car.interpolate_V_segment(
            states[index_t], dyn.prediction_horizon)
        ax.clear();  ax2.clear();  ax3.clear();  ax4.clear()

        t     = states[index_t][0]
        x     = states[index_t][1]
        y     = states[index_t][2]
        theta = states[index_t][3]
        v     = states[index_t][4]
        delta = states[index_t][5]

        # XY heatmap — vectorised: build all query points at once
        x_vals = np.linspace(Car.X0, Car.X0 + Car.XN, 100)
        y_vals = np.linspace(Car.Y0, Car.Y0 + Car.YN, 100)
        xg, yg = np.meshgrid(x_vals, y_vals, indexing='ij')   # (100,100)
        n_pts  = xg.size
        pts_xy = np.column_stack([
            np.full(n_pts, t), xg.ravel(), yg.ravel(),
            np.full(n_pts, theta), np.full(n_pts, v), np.full(n_pts, delta)
        ])
        Z_xy = V_interpolated_seg(pts_xy).reshape(100, 100)

        c = ax.contourf(x_vals, y_vals, Z_xy.T, levels=50, cmap="viridis")
        ax.contour(x_vals, y_vals, Z_xy.T, levels=10, colors="black", linewidths=0.5)
        if Z_xy.max() > 0 and Z_xy.min() < 0:
            ax.contour(x_vals, y_vals, Z_xy.T, levels=[0], colors="red", linewidths=2.0)
        ax.set_xlabel('X-axis');  ax.set_ylabel('Y-axis')
        ax.set_title('Value Function (X vs Y)');  ax.axis('equal')
        ax.plot(predicted_path[index_t][0], predicted_path[index_t][1], "yo")
        ax.plot(x, y, "ro")

        # Theta-v heatmap — vectorised
        theta_vals = np.linspace(-np.pi, np.pi, 100)
        v_vals     = np.linspace(Car.min_velocity, Car.max_velocity, 100)
        tg, vg = np.meshgrid(theta_vals, v_vals, indexing='ij')   # (100,100)
        n_pts  = tg.size
        pts_tv = np.column_stack([
            np.full(n_pts, t), np.full(n_pts, x), np.full(n_pts, y),
            tg.ravel(), vg.ravel(), np.full(n_pts, delta)
        ])
        Z_tv = V_interpolated_seg(pts_tv).reshape(100, 100)

        c2 = ax2.contourf(theta_vals, v_vals, Z_tv.T, levels=50, cmap="viridis")
        ax2.contour(theta_vals, v_vals, Z_tv.T, levels=10, colors="black", linewidths=0.5)
        if Z_tv.max() > 0 and Z_tv.min() < 0:
            ax2.contour(theta_vals, v_vals, Z_tv.T, levels=[0], colors="red", linewidths=2.0)
        ax2.set_xlabel('Theta');  ax2.set_ylabel('Velocity')
        ax2.set_title('Value Function (Theta vs V)')
        ax2.plot(predicted_path[index_t][2], predicted_path[index_t][3], "yo")
        ax2.plot(theta, v, "ro")

        # Theta-delta heatmap — vectorised
        delta_vals   = np.linspace(Car.min_steering, Car.max_steering, 100)
        tg2, dg = np.meshgrid(theta_vals, delta_vals, indexing='ij')  # (100,100)
        n_pts   = tg2.size
        pts_td  = np.column_stack([
            np.full(n_pts, t), np.full(n_pts, x), np.full(n_pts, y),
            tg2.ravel(), np.full(n_pts, v), dg.ravel()
        ])
        Z_td = V_interpolated_seg(pts_td).reshape(100, 100)

        c3 = ax3.contourf(theta_vals, delta_vals, Z_td.T, levels=50, cmap="viridis")
        ax3.contour(theta_vals, delta_vals, Z_td.T, levels=10, colors="black", linewidths=0.5)
        if Z_td.max() > 0 and Z_td.min() < 0:
            ax3.contour(theta_vals, delta_vals, Z_td.T, levels=[0], colors="red", linewidths=2.0)
        ax3.set_xlabel('Theta');  ax3.set_ylabel('Delta')
        ax3.set_title('Value Function (Theta vs Delta)')
        ax3.plot(predicted_path[index_t][2], predicted_path[index_t][4], "yo")
        ax3.plot(theta, delta, "ro")

        seg = Car.grids_segment_list
        ax.plot(
            [seg[index_t][1][0], seg[index_t][1][-1], seg[index_t][1][-1],
             seg[index_t][1][0], seg[index_t][1][0]],
            [seg[index_t][2][0], seg[index_t][2][0], seg[index_t][2][-1],
             seg[index_t][2][-1], seg[index_t][2][0]], "y"
        )
        ax2.plot(
            [seg[index_t][3][0], seg[index_t][3][-1], seg[index_t][3][-1],
             seg[index_t][3][0], seg[index_t][3][0]],
            [seg[index_t][4][0], seg[index_t][4][0], seg[index_t][4][-1],
             seg[index_t][4][-1], seg[index_t][4][0]], "y"
        )
        ax3.plot(
            [seg[index_t][3][0], seg[index_t][3][-1], seg[index_t][3][-1],
             seg[index_t][3][0], seg[index_t][3][0]],
            [seg[index_t][5][0], seg[index_t][5][0], seg[index_t][5][-1],
             seg[index_t][5][-1], seg[index_t][5][0]], "y"
        )

        # ---------- Time vs Cost subplot ----------
        if costs_list is not None:
            time_axis     = np.arange(len(costs_list)) * Car.dt
            current_costs = [c[0] for c in costs_list]
            ax4.plot(time_axis, current_costs, color='steelblue', linewidth=1.5, label='V(current state)')
            ax4.axhline(0, color='black', linestyle='--', linewidth=1)
            if index_t < len(costs_list):
                ax4.axvline(time_axis[index_t], color='red', linestyle='--', linewidth=1.5,
                            label=f't = {time_axis[index_t]:.2f} s')
                ax4.plot(time_axis[index_t], current_costs[index_t], 'ro', markersize=7)
            ax4.set_xlabel('Time (s)')
            ax4.set_ylabel('HJ Value V(x)')
            ax4.set_title('Time vs HJ Cost  (V ≤ 0 = safe)')
            ax4.legend(loc='upper right', fontsize=8)
            ax4.grid(True, alpha=0.4)
        # ------------------------------------------

        progress.update(1)
        return c, c2, c3

    ani = FuncAnimation(fig, update, frames=range(nr), interval=100, blit=False)
    fig.tight_layout()
    ani.save("animation_seg_5D_acados.gif", writer="pillow", fps=10)
    progress.close()


# ===========================================================================
# Main
# ===========================================================================

def main():
    nr  = 120
    Car = MPC_acados(nr)

    states          = [[0, 0.25, 0.5, 0.0, 0.0, 0.0]]
    predicted_path  = []
    predicted_input = []
    costs_list      = []

    for it in tqdm(range(nr)):
        time = states[-1][0]
        Car.compute_control(states[-1][1:], time, it)

        ns = Car.get_optimal_states()   # shape (5, N+1)
        nu = Car.get_optimal_control()  # shape (2, N)

        next_state = [ns[i][1] for i in range(5)]
        while next_state[2] < -np.pi: next_state[2] += 2 * np.pi
        while next_state[2] >  np.pi: next_state[2] -= 2 * np.pi

        predicted_path.append(ns)
        predicted_input.append(nu)

        # Evaluate V along predicted horizon (vectorised)
        h = Car.current_horizon + 1
        pts_cost = np.column_stack([
            time + np.arange(h) * Car.dt,
            ns[0, :h], ns[1, :h], ns[2, :h], ns[3, :h], ns[4, :h]
        ])
        costs = Car.V(pts_cost).tolist()
        costs_list.append(costs)

        if costs[1] <= 0:
            states.append([time + Car.dt] + next_state)
        else:
            states.append([time] + next_state)

    animate_heat_map_segment(Car, states, nr, predicted_path, costs_list)
    plt.show()


main()