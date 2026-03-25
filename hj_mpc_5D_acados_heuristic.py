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
# Heuristic mode switches (set exactly one to True)
# ---------------------------------------------------------------------------
USE_HJ_LINEAR = False
USE_HJ_DIAG_QUADRATIC = True
USE_HJ_FULL_QUADRATIC = False

assert sum([
    USE_HJ_LINEAR,
    USE_HJ_DIAG_QUADRATIC,
    USE_HJ_FULL_QUADRATIC,
]) == 1, "Set exactly one HJ surrogate mode to True."

HJ_STAGE_PARAM_DIM = 32

# ---------------------------------------------------------------------------
# Main controller class
# ---------------------------------------------------------------------------

class MPC_acados:

    X0, XN = 0, 4
    Y0, YN = 0, 4

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
        self.hj_lambda = 1.0

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
        start_load = tm.time()
        self.set_value_function()
        self.dims = self.value_function.shape
        end_load = tm.time()

        # Theta wrap-around is handled lazily in interpolate_V_segment via
        # np.take + modulo, so the full 3x concatenation along axis=3 is no
        # longer needed here.  This eliminates ~1 s of memory allocation every
        # time the value function is reloaded (every ~5 s in deployment).
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
        #   - Rebuild a local interpolant each solve.
        #   - Fit local HJ surrogate coefficients and pass them via stage params.
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

        # Parameters per stage:
        # p = [lam, zbar(5), a, g(5), hdiag(5), H_upper(15)]
        p_sym = ca.SX.sym("p", HJ_STAGE_PARAM_DIM)

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

        ocp.dims.N = N
        ocp.dims.np = HJ_STAGE_PARAM_DIM
        ocp.parameter_values = np.zeros(HJ_STAGE_PARAM_DIM)

        # -- Cost type: EXTERNAL (gives maximum flexibility for HJ term) ------
        # Note: input-rate cost (Q2 term) is not represented in this model.
        Q3_sym = ca.DM(self.Q3)

        lam = p_sym[0]
        zbar = p_sym[1:6]
        a_hj = p_sym[6]
        g_hj = p_sym[7:12]
        hdiag = p_sym[12:17]
        Hvec = p_sym[17:32]
        dz = x_sym - zbar

        def unpack_sym_5(Hvec_sym):
            H = ca.SX.zeros(5, 5)
            idx = 0
            for i in range(5):
                for j in range(i, 5):
                    H[i, j] = Hvec_sym[idx]
                    H[j, i] = Hvec_sym[idx]
                    idx += 1
            return H

        if USE_HJ_LINEAR:
            hj_surrogate = a_hj + ca.dot(g_hj, dz)
        elif USE_HJ_DIAG_QUADRATIC:
            hj_surrogate = a_hj + ca.dot(g_hj, dz) + 0.5 * ca.dot(hdiag, dz * dz)
        elif USE_HJ_FULL_QUADRATIC:
            H_hj = unpack_sym_5(Hvec)
            hj_surrogate = a_hj + ca.dot(g_hj, dz) + 0.5 * ca.mtimes([dz.T, H_hj, dz])
        else:
            raise RuntimeError("No HJ surrogate mode selected.")

        vel_penalty = ca.fmax(0.0, -v) ** 2
        hj_term = lam * hj_surrogate

        stage_cost = (
            ca.mtimes([u_sym.T, Q3_sym, u_sym])
            + self.Qhj * hj_term
            + self.Qv * vel_penalty
        )
        terminal_cost = self.Qhj * hj_term

        ocp.cost.cost_type   = "EXTERNAL"
        ocp.cost.cost_type_e = "EXTERNAL"

        ocp.model.cost_expr_ext_cost   = stage_cost
        ocp.model.cost_expr_ext_cost_e = terminal_cost

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
        # GAUSS_NEWTON: constructs the Hessian from first-order cost information
        # only, guaranteeing positive-semidefiniteness and eliminating HPIPM
        # ACADOS_MINSTEP failures caused by the indefinite second-order HJ terms
        # (Method B). Slightly slower convergence per SQP iteration than EXACT,
        # but far more robust for SQP_RTI in embedded / real-time use.
        ocp.solver_options.hessian_approx      = "GAUSS_NEWTON"
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

        # --- Set HJ surrogate parameters (batched RGI call) ------------------
        # _build_all_stage_params_batched assembles all finite-difference query
        # points for every stage into a single array and issues one self.V() call,
        # reducing O(N * n_pts) individual Python->C RGI invocations to O(1).
        all_params = self._build_all_stage_params_batched(x_pred, HJ_time)
        for k in range(self.N + 1):
            self.acados_solver.set(k, "p", all_params[k])

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

        # status == 0: success, 1: max_iter, 2: infeasible, 4: NaN/numerical failure
        if status == 4:
            # Numerical failure: warm-start state is likely corrupted with NaN.
            # Reset solver and warm-start cache, then apply a safe zero-input fallback
            # so the vehicle decelerates rather than receiving a garbage control command.
            print(f"[acados] Warning: solver returned status {status} (NaN) at iter {it} — "
                  f"resetting warm-start and applying zero-input fallback.")
            self._sol = None
            self.acados_solver.reset()
            acceleration  = 0.0
            steering_rate = 0.0
        else:
            if status not in (0, 1):
                print(f"[acados] Warning: solver returned status {status} at iter {it}")

            # Extract first control
            u0            = self.acados_solver.get(0, "u")
            acceleration  = float(u0[0])
            steering_rate = float(u0[1])

            # Cache solution for warm-starting
            self._sol = {
                'x': np.array([self.acados_solver.get(k, "x") for k in range(self.N + 1)]),
                'u': np.array([self.acados_solver.get(k, "u") for k in range(self.N)]),
            }

        comp_end = tm.time()
        self.computation_times[it, 0] = comp_end - comp_start

        # print(f"Total Computation time: {self.computation_times[it, 0]:.4f} s, HJ surrogate time: {surrogate_end - surrogate_start:.4f} s")

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

    def set_value_function(self, path = "out_bicycle5d.npy"):
        file_path = path
        self.value_function = np.load(file_path, mmap_mode='r').copy()
        self.dims = self.value_function.shape
        print("Loaded value function of type", type(self.value_function),
              "of shape", self.dims)

    def interpolate_V_segment(self, state, horizon):
        """
        Build a local RegularGridInterpolator (SciPy) around the current state.

        Replaces ca.interpolant: no CasADi symbolic graph is needed for table
        interpolation. SciPy RGI is
        10-50x faster to construct and supports vectorised batch evaluation.

        Returns a RegularGridInterpolator callable: V(pts) where pts has shape
        (M, 6), columns ordered [t, x, y, theta, v, delta].

        Theta wrap-around is handled lazily: instead of pre-concatenating the
        full array three times along axis=3 (expensive when the value function
        is reloaded every ~5 s), we compute the required theta indices with
        modulo arithmetic and use np.take to extract only the needed slice.
        The resulting segment and grid are mathematically identical to the
        original approach.
        """

        delta_plus  = [horizon, horizon, horizon, horizon, 5, 12]
        delta_minus = [      2, horizon, horizon, horizon, 5, 12]

        index = list(self.coord_to_index(state))
        index[3] = index[3] + self.dims[3]

        indices = [
            [max(0, index[i] - delta_minus[i]),
            min(self.dims[i], index[i] + delta_plus[i])]
            for i in range(6)
        ]
        indices[3] = [index[3] - delta_minus[3], index[3] + delta_plus[3]]

        # ── Lazy theta wrap via np.take + modulo ─────────────────────────────
        # Original approach: pre-concatenate value_func_wrapped = [V, V, V]
        # along axis=3, then slice normally.  Cost: O(dims[3] * rest) copy.
        #
        # New approach: slice the five non-theta axes directly, then use
        # np.take with wrapped indices along axis=3.  Only the segment's theta
        # extent (~2*horizon+1 entries) is ever copied, not the whole array.
        sl_t = slice(indices[0][0], indices[0][1] + 1)
        sl_x = slice(indices[1][0], indices[1][1] + 1)
        sl_y = slice(indices[2][0], indices[2][1] + 1)
        sl_v = slice(indices[4][0], indices[4][1] + 1)
        sl_d = slice(indices[5][0], indices[5][1] + 1)

        theta_idx = np.arange(indices[3][0], indices[3][1] + 1) % self.dims[3]

        tmp = self.value_function[sl_t, sl_x, sl_y, :, sl_v, sl_d]
        value_func_segment = np.ascontiguousarray(tmp[:, :, :, theta_idx, :, :])
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
        # Yaw: shift by -2π to account for the virtual wrap offset applied to
        # index[3] (index[3] += dims[3] → raw grid values exceed [−π, π] by
        # exactly 2π, so subtract 2π to restore physical coordinates).
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

        return interpolant

    def _pack_upper_triangle(self, H):
        vals = []
        for i in range(5):
            for j in range(i, 5):
                vals.append(H[i, j])
        return np.array(vals, dtype=float)

    def _build_all_stage_params_batched(self, x_pred, HJ_time):
        """
        Batched surrogate parameter construction for all N+1 stages.

        Key idea
        --------
        The original loop calls self.V(...) once per finite-difference point
        per stage, resulting in O(N * n_pts_per_stage) separate Python→C RGI
        calls.  SciPy's RegularGridInterpolator has nearly constant overhead
        per *call*, so batching all query points into a single ndarray and
        issuing one self.V(all_pts) call amortises that overhead to O(1).

        Query point layout (rows of all_pts)
        -------------------------------------
        Block 0 : centers           shape (S, 6)     S = N+1
        Block 1 : zbar + eps_0 * e0 shape (S, 6)   ─┐
        Block 2 : zbar - eps_0 * e0 shape (S, 6)    │  5 dims × 2 = 10 blocks
                ...                                  │  used for grad + hdiag
        Block 10: zbar + eps_4 * e4 shape (S, 6)   ─┘
        Block 11: zbar - eps_4 * e4 shape (S, 6)   ─┘
        [FULL_QUADRATIC only]
        Block 11+: cross-term blocks  4 pts × 10 pairs × S rows

        Total rows (DIAG_QUADRATIC): S × (1 + 10) = 11 S
        Total rows (FULL_QUADRATIC): S × (1 + 10 + 40) = 51 S
        Total rows (LINEAR):         S × (1 + 10) = 11 S  (hdiag reuses ±blocks)
        """
        S   = self.N + 1          # number of stages
        eps = np.array([
            max(1e-3, 0.5 * self.d_states[1]),  # x
            max(1e-3, 0.5 * self.d_states[2]),  # y
            max(1e-3, 0.5 * self.d_states[3]),  # theta
            max(1e-3, 0.5 * self.d_states[4]),  # v
            max(1e-3, 0.5 * self.d_states[5]),  # delta
        ], dtype=float)

        zbars = np.asarray(x_pred, dtype=float)           # (S, 5)
        t_ks  = HJ_time + np.arange(S) * self.dt          # (S,)

        # ------------------------------------------------------------------
        # 1. Assemble all query points in one array
        # ------------------------------------------------------------------
        pts_list = []

        # Block 0: center points
        pts_list.append(np.column_stack([t_ks, zbars]))   # (S, 6)

        # Blocks 1..10: ± perturbations along each of 5 state dims
        # (reused for both gradient and diagonal Hessian)
        for i in range(5):
            dz        = np.zeros((S, 5), dtype=float)
            dz[:, i]  = eps[i]
            pts_list.append(np.column_stack([t_ks, zbars + dz]))  # +
            pts_list.append(np.column_stack([t_ks, zbars - dz]))  # −

        # Blocks 11..: cross-term perturbations (FULL_QUADRATIC only)
        cross_pairs = []
        if USE_HJ_FULL_QUADRATIC:
            for i in range(5):
                for j in range(i + 1, 5):
                    dzi        = np.zeros((S, 5), dtype=float)
                    dzj        = np.zeros((S, 5), dtype=float)
                    dzi[:, i]  = eps[i]
                    dzj[:, j]  = eps[j]
                    pts_list.append(np.column_stack([t_ks, zbars + dzi + dzj]))
                    pts_list.append(np.column_stack([t_ks, zbars + dzi - dzj]))
                    pts_list.append(np.column_stack([t_ks, zbars - dzi + dzj]))
                    pts_list.append(np.column_stack([t_ks, zbars - dzi - dzj]))
                    cross_pairs.append((i, j))

        all_pts  = np.vstack(pts_list)      # (M_total, 6)

        # ------------------------------------------------------------------
        # 2. Single batched RGI call  ← the key optimisation
        # ------------------------------------------------------------------
        all_vals = self.V(all_pts)          # (M_total,)

        # ------------------------------------------------------------------
        # 3. Slice back into per-stage components
        # ------------------------------------------------------------------
        idx      = 0
        v_center = all_vals[idx : idx + S]; idx += S   # (S,)

        # v_pm[i] = (v_plus[i], v_minus[i]),  each shape (S,)
        v_pm = []
        for i in range(5):
            vp = all_vals[idx : idx + S]; idx += S
            vm = all_vals[idx : idx + S]; idx += S
            v_pm.append((vp, vm))

        # cross-term blocks (only populated when USE_HJ_FULL_QUADRATIC)
        cross_vals = []
        if USE_HJ_FULL_QUADRATIC:
            for _ in cross_pairs:
                vpp = all_vals[idx : idx + S]; idx += S
                vpm = all_vals[idx : idx + S]; idx += S
                vmp = all_vals[idx : idx + S]; idx += S
                vmm = all_vals[idx : idx + S]; idx += S
                cross_vals.append((vpp, vpm, vmp, vmm))

        # ------------------------------------------------------------------
        # 4. Compute surrogate coefficients and pack parameter vectors
        # ------------------------------------------------------------------
        # Pre-allocate output array: (S, HJ_STAGE_PARAM_DIM)
        params_all = np.zeros((S, HJ_STAGE_PARAM_DIM), dtype=float)
        params_all[:, 0]   = self.hj_lambda          # lam
        params_all[:, 1:6] = zbars                   # zbar

        a_hj_all = v_center                          # (S,)
        params_all[:, 6] = a_hj_all

        # Gradient  g[i] = (v+ - v-) / (2 eps[i])
        g_hj_all = np.column_stack([
            (v_pm[i][0] - v_pm[i][1]) / (2.0 * eps[i])
            for i in range(5)
        ])                                           # (S, 5)
        params_all[:, 7:12] = g_hj_all

        if USE_HJ_DIAG_QUADRATIC or USE_HJ_FULL_QUADRATIC:
            # Diagonal Hessian  h[i] = (v+ - 2*a + v-) / eps[i]^2
            hdiag_all = np.column_stack([
                (v_pm[i][0] - 2.0 * a_hj_all + v_pm[i][1]) / (eps[i] ** 2)
                for i in range(5)
            ])                                       # (S, 5)
            # PSD regularisation: clamp negative diagonal curvature
            hdiag_all = np.maximum(hdiag_all, 0.0)
            params_all[:, 12:17] = hdiag_all

        if USE_HJ_FULL_QUADRATIC:
            # Off-diagonal Hessian entries + PSD regularisation (Gill-Murray shift)
            # H_upper packing index offset within p: starts at 17
            tri_offset = 17
            for pair_idx, (i, j) in enumerate(cross_pairs):
                vpp, vpm_c, vmp, vmm = cross_vals[pair_idx]
                hij = (vpp - vpm_c - vmp + vmm) / (4.0 * eps[i] * eps[j])  # (S,)

                # Compute column index in upper-triangle packing:
                # upper triangle (row, col) with row<=col, packed row-major
                col_in_tri = 0
                for r in range(5):
                    for c in range(r, 5):
                        if r == i and c == j:
                            break
                        col_in_tri += 1
                params_all[:, tri_offset + col_in_tri] = hij

            # Per-stage PSD Gill-Murray shift on H_full
            # Rebuild H_full per stage only for eigenvalue check — unavoidable
            # but accounts for a small fraction of overall work.
            for k in range(S):
                hdiag_k = hdiag_all[k]
                H_full_k = np.diag(hdiag_k).astype(float)
                col_in_tri = 0
                for r in range(5):
                    for c in range(r, 5):
                        if r != c:
                            hij_k = params_all[k, tri_offset + col_in_tri]
                            H_full_k[r, c] = hij_k
                            H_full_k[c, r] = hij_k
                        col_in_tri += 1
                H_full_k = 0.5 * (H_full_k + H_full_k.T)
                eigvals   = np.linalg.eigvalsh(H_full_k)
                min_eig   = eigvals.min()
                if min_eig < 0.0:
                    shift = -min_eig + 1e-4
                    for r in range(5):
                        # diagonal entries in upper-triangle packing
                        diag_col = sum(5 - rr for rr in range(r))  # cumulative
                        # simpler: recompute flat index for (r,r)
                        flat = 0
                        for rr in range(5):
                            for cc in range(rr, 5):
                                if rr == r and cc == r:
                                    break
                                flat += 1
                        params_all[k, tri_offset + flat] += shift

        return params_all  # (S, HJ_STAGE_PARAM_DIM)

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

        _zmin_xy, _zmax_xy = Z_xy.min(), Z_xy.max()
        if _zmax_xy - _zmin_xy > 1e-10:
            c = ax.contourf(x_vals, y_vals, Z_xy.T, levels=np.linspace(_zmin_xy, _zmax_xy, 50), cmap="viridis")
            ax.contour(x_vals, y_vals, Z_xy.T, levels=np.linspace(_zmin_xy, _zmax_xy, 10), colors="black", linewidths=0.5)
        else:
            c = ax.contourf(x_vals, y_vals, Z_xy.T, levels=1, cmap="viridis")
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

        _zmin_tv, _zmax_tv = Z_tv.min(), Z_tv.max()
        if _zmax_tv - _zmin_tv > 1e-10:
            c2 = ax2.contourf(theta_vals, v_vals, Z_tv.T, levels=np.linspace(_zmin_tv, _zmax_tv, 50), cmap="viridis")
            ax2.contour(theta_vals, v_vals, Z_tv.T, levels=np.linspace(_zmin_tv, _zmax_tv, 10), colors="black", linewidths=0.5)
        else:
            c2 = ax2.contourf(theta_vals, v_vals, Z_tv.T, levels=1, cmap="viridis")
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

        _zmin_td, _zmax_td = Z_td.min(), Z_td.max()
        if _zmax_td - _zmin_td > 1e-10:
            c3 = ax3.contourf(theta_vals, delta_vals, Z_td.T, levels=np.linspace(_zmin_td, _zmax_td, 50), cmap="viridis")
            ax3.contour(theta_vals, delta_vals, Z_td.T, levels=np.linspace(_zmin_td, _zmax_td, 10), colors="black", linewidths=0.5)
        else:
            c3 = ax3.contourf(theta_vals, delta_vals, Z_td.T, levels=1, cmap="viridis")
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
    if USE_HJ_LINEAR:
        ani.save("animation_seg_5D_acados_heuristic_linear.gif", writer="pillow", fps=10)
    elif USE_HJ_DIAG_QUADRATIC:
        ani.save("animation_seg_5D_acados_heuristic_diagonal.gif", writer="pillow", fps=10)
    elif USE_HJ_FULL_QUADRATIC:
        ani.save("animation_seg_5D_acados_heuristic_full.gif", writer="pillow", fps=10)
    else:
        raise RuntimeError("No HJ surrogate mode selected.")
    progress.close()


# ===========================================================================
# Main
# ===========================================================================

def main():
    nr  = 100
    Car = MPC_acados(nr)
    Car.set_value_function("/home/tokisaki_kurumi/Projects/svea/verification_guided_mpc/hj_sets/hj_sets_20260323_192206/sets/slice3.npy")

    states          = [[0, 3.0, 0.5, np.pi/2, 0.0, 0.0]]
    predicted_path  = []
    predicted_input = []
    costs_list      = []
    
    # import json

    # data = json.load(open("hj_set_update.json"))
    # update_time = data["last_update"]

    for it in tqdm(range(nr)):
        start_time = tm.time()
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

        end_time = tm.time()

        # tm.sleep(dyn.time_step - (end_time - start_time))

    animate_heat_map_segment(Car, states, nr, predicted_path, costs_list)
    plt.show()


main()