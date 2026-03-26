
from tqdm import tqdm

import time as tm
import numpy as np
import casadi as ca
from acados_template import AcadosModel, AcadosOcp, AcadosOcpSolver
from scipy.interpolate import RegularGridInterpolator

def accum(params):
    i = 0
    for p in params:
        name, size = (p, 1) if isinstance(p, str) else p
        yield name, slice(i, i+size)
        i += size

# Controller parameters
N = 5
TIMESTEP = 0.1

# Model parameters
L = 0.32   # Wheelbase length (m)

STATE = {name: s for name, s in accum(["x", "y", "yaw", "vel"])}
NX = max(s.stop for s in STATE.values())

def get_state(name, seq=None):
    return STATE[name] if seq is None else seq[STATE[name]]


INPUT = {name: s for name, s in accum(["accel", "steer"])}
NU = max(s.stop for s in INPUT.values())

def get_input(name, seq=None):
    return INPUT[name] if seq is None else seq[INPUT[name]]


PARAM = {name: s for name, s in accum([
    ("xbar", NX),           # Linearization point
    ("a", 1), ("b", NX),    # Interpolated value function coefficients
])}
NP = max(s.stop for s in PARAM.values())

def get_param(name, seq=None):
    return PARAM[name] if seq is None else seq[PARAM[name]]

HJ_TIMESTEP = 20/100
HJ_GRID_MIN = (0.0, 0.0, -np.pi, 0.0)
HJ_GRID_MAX = (4.0, 4.0, np.pi, 0.8)
HJ_GRID_STP = (25, 25, 15, 11)
HJ_GRID_DX = [(HJ_GRID_MAX[i] - HJ_GRID_MIN[i]) / (HJ_GRID_STP[i] - 1) for i in range(4)]

class Controller:

    min_velocity = 0.0
    max_velocity = 0.8
    min_acceleration = -0.8
    max_acceleration = 0.8
    min_steering = -np.deg2rad(40)
    max_steering = np.deg2rad(40)

    Qhj = 10.0

    def __init__(self, registry, state_queue, stop_event):

        # Runtime communication primitives
        self.registry = registry
        self.state_queue = state_queue
        self.stop_event = stop_event
        
        self.solver = self._init_acados_solver()
        self._sol = {
            'x': np.zeros((N + 1, NX)),
            'u': np.zeros((N, NU)),
            'healthy': False,  # Flag to indicate if the solver has found a healthy solution before
        }

        self.latest_info = {'interval': (-np.inf, -np.inf), 'name': 'bar'}

    def _init_acados_solver(self):

        model = self._init_acados_model()

        ocp = self._init_acados_ocp(model)

        # Solver options
        ocp.solver_options.tf                  = N * TIMESTEP
        ocp.solver_options.N_horizon           = N
        ocp.solver_options.integrator_type     = "DISCRETE"
        ocp.solver_options.nlp_solver_type     = "SQP_RTI"   # Real-Time Iteration
        ocp.solver_options.qp_solver           = "PARTIAL_CONDENSING_HPIPM"
        ocp.solver_options.qp_solver_cond_N    = N
        # GAUSS_NEWTON: constructs the Hessian from first-order cost information
        # only, guaranteeing positive-semidefiniteness and eliminating HPIPM
        # ACADOS_MINSTEP failures caused by the indefinite second-order HJ terms
        # (Method B). Slightly slower convergence per SQP iteration than EXACT,
        # but far more robust for SQP_RTI in embedded / real-time use.
        # ocp.solver_options.hessian_approx      = "EXACT"
        ocp.solver_options.print_level         = 0
        ocp.solver_options.nlp_solver_max_iter = 1000

        return AcadosOcpSolver(ocp, json_file="acados_ocp.json")

    def _init_acados_model(self):
  
        model = AcadosModel()
        model.name = "Bicycle4D"

        # State
        model.x = ca.vertcat(*[ca.SX.sym(state, s.stop - s.start)
                               for state, s in STATE.items()])
        _state = lambda name: get_state(name, model.x)

        # Input
        model.u = ca.vertcat(*[ca.SX.sym(input, s.stop - s.start)
                               for input, s in INPUT.items()])
        _input = lambda name: get_input(name, model.u)

        # Parameters
        # ???
        model.p = ca.vertcat(*[ca.SX.sym(param, s.stop - s.start)
                               for param, s in PARAM.items()])
        _param = lambda name: get_param(name, model.p)

        # Dynamics
        dt = TIMESTEP    
        f_x     = _state("x") + dt * _state("vel") * ca.cos(_state("yaw"))
        f_y     = _state("y") + dt * _state("vel") * ca.sin(_state("yaw"))
        f_th    = _state("yaw") + dt * _state("vel") * ca.tan(_input("steer")) / L
        f_v     = _state("vel") + dt * _input("accel")
        model.disc_dyn_expr = ca.vertcat(f_x, f_y, f_th, f_v)

        # Interpolated value function
        # hat(V)_k(x) = a_k + b_k^T (x_k - bar(x)_k)
        vf_interp = _param("a") + ca.dot(_param("b"), model.x - _param("xbar"))

        # Stage Cost
        R = ca.DM(np.diag([1e-2, 1e-2]))
        model.cost_expr_ext_cost = (
            # ca.mtimes([model.u.T, R, model.u]) +
            self.Qhj * vf_interp
        )

        # Terminal Cost
        model.cost_expr_ext_cost_e = (
            self.Qhj * vf_interp
        )

        return model

    def _init_acados_ocp(self, model):
        
        ocp = AcadosOcp()
        ocp.model = model

        ocp.dims.np = NP
        ocp.parameter_values = np.zeros(NP)

        ocp.cost.cost_type      = "EXTERNAL"
        ocp.cost.cost_type_e    = "EXTERNAL"

        # Constraints
        # State bounds: velocity only, for now
        ocp.constraints.lbx   = np.array([self.min_velocity])
        ocp.constraints.ubx   = np.array([self.max_velocity])
        ocp.constraints.idxbx = np.array([3])   # indices of v and delta in x

        ocp.constraints.lbx_e   = ocp.constraints.lbx
        ocp.constraints.ubx_e   = ocp.constraints.ubx
        ocp.constraints.idxbx_e = ocp.constraints.idxbx

        # Input bounds
        ocp.constraints.lbu   = np.array([self.min_acceleration, self.min_steering])
        ocp.constraints.ubu   = np.array([self.max_acceleration, self.max_steering])
        ocp.constraints.idxbu = np.array([0, 1])

        # Initial state constraint (will be set each solve via set_new_time_step)
        ocp.constraints.x0 = np.zeros(NX)

        return ocp
    
    def update_current_valuefunc(self, time, state):

        self.registry['time'] = time
        self.registry['state'] = state
        
        info = self.registry['latest.info']

        if info['name'] != self.latest_info['name'] and info['interval'][0] <= time <= info['interval'][1]:
            print(f"Controller: Detected new info '{info['name']}' from verifier, updating value function interpolator.")
            
            self.latest_info = info
            self.latest_vf = self.registry[info['name'] + '.vf']
            
            self.update_vf_interpolator(self.latest_vf, info)
    
    def update_vf_interpolator(self, vf, info):
        """
        vf: ndarray, shape (HJ_NT, HJ_NX_1, HJ_NX_2, ..., HJ_NX_n)
        info:
        - interval = (t0, tf)
        - step = dt
        - grids = [x1_grid, x2_grid, ..., xn_grid]   # required
            OR equivalent per-axis bounds from which grids can be built
        """
        dt = HJ_TIMESTEP
        t0, tf = info["interval"]

        HJ_NT = vf.shape[0]
        time_grid = t0 + np.arange(HJ_NT) * dt

        # Preferred: explicit spatial grids
        spatial_grids = [
            np.linspace(HJ_GRID_MIN[i], HJ_GRID_MAX[i], HJ_GRID_STP[i])
            for i in range(NX)
        ]
        points = (time_grid, *spatial_grids)

        self.vf_interp = RegularGridInterpolator(
            points,
            vf,
            method="linear",
            bounds_error=False,
            fill_value=None,   # extrapolate if slightly outside
        )

    def get_optimal_states(self):
        xs = np.array([self.solver.get(k, "x") for k in range(N+1)]) # shape (N+1, NX)
        xs[:, 2] = (xs[:, 2] + np.pi) % (2 * np.pi) - np.pi  # Wrap yaw to [-pi, pi]
        return xs

    def get_optimal_inputs(self):
        return np.array([self.solver.get(k, "u") for k in range(N)]) # shape (N, NU)

    def evaluate_real_cost(self, time, state):
        pt = np.array([[time, *state]], dtype=float)
        return float(self.vf_interp(pt)[0])
    
    def compute_control(self, time, state):

        state = np.asarray(state, dtype=float).reshape(NX)
        state = self.bound_initial_state(state)

        # 0) Update the current value function
        self.update_current_valuefunc(time, state)

        # 1) Fix initial state at stage 0
        self.solver.set(0, "lbx", state)
        self.solver.set(0, "ubx", state)

        # 2) Build stage-wise linear HJ surrogates:
        #    p_k = [xbar_k, a_k, b_k]
        x_pred = self._predict_states(state)
        stage_params = self._build_stage_params(time, x_pred)

        for k in range(N + 1):
            self.solver.set(k, "p", stage_params[k])

        # 3) Warm-start from previous solution if available
        if self._sol['healthy']:
            for k in range(N + 1):
                self.solver.set(k, "x", self._sol["x"][k])
            for k in range(N):
                self.solver.set(k, "u", self._sol["u"][k])

        # 4) Solve OCP
        status = self.solver.solve()

        # 5) Handle solver status — must come BEFORE caching
        if status == 4:
            # Numerical failure: warm-start state is likely corrupted with NaN.
            # Reset solver and warm-start cache, then apply a safe zero-input fallback.
            print("[acados] Warning: solver returned status 4 (NaN), resetting warm start.")
            self._sol['healthy'] = False
            self.solver.reset()
            return [0.0] * NU
        else:
            if status not in (0, 1):
                print(f"[acados] Warning: solver returned status {status}")
            # 6) Cache solution for warm-starting next time
            x_cached = np.array([self.solver.get(k, "x") for k in range(N + 1)])
            x_cached[:, 2] = (x_cached[:, 2] + np.pi) % (2 * np.pi) - np.pi  # Wrap yaw to [-pi, pi]
            self._sol.update(
                x=x_cached,
                u=np.array([self.solver.get(k, "u") for k in range(N)]),
                healthy=(status in (0, 1)),
            )

        # 7) Return first control input
        return list(map(float, self.solver.get(0, "u")))

    def bound_initial_state(self, state):
        state = np.array(state, dtype=float, copy=True)
        state[get_state("vel")] = np.clip(
            get_state("vel", state),
            self.min_velocity,
            self.max_velocity,
        )
        return state
    
    def _predict_states(self, x0):
        x_pred = np.zeros((N+1, NX))
        x_pred[0] = x0

        if self._sol['healthy']:
            prev_x = self._sol['x']
            for k in range(1, N+1):
                x_pred[k] = prev_x[min(k, N)]
        else:
            for k in range(N):
                x = x_pred[k]
                x_pred[k + 1] = np.array([
                    x[0] + TIMESTEP * x[3] * np.cos(x[2]),
                    x[1] + TIMESTEP * x[3] * np.sin(x[2]),
                    x[2],
                    x[3],
                ])

        return x_pred

    def _build_stage_params(self, time, x_pred):
        """
        Build per-stage linear HJ surrogate parameters.

        For each stage k:
            t_k   = time + k * TIMESTEP
            xbar_k = x_pred[k]
            a_k    = V(t_k, xbar_k)
            b_k    = grad_x V(t_k, xbar_k)   (central finite differences)

        Returns
        -------
        params_all : ndarray, shape (N+1, NP)
            Each row is:
                p_k = [xbar_k(4), a_k(1), b_k(4)]
        """
        S = N + 1

        x_pred = np.asarray(x_pred, dtype=float).reshape(S, NX)
        x_pred[:, 2] = (x_pred[:, 2] + np.pi) % (2 * np.pi) - np.pi  # Wrap yaw to [-pi, pi]
        t_ks = time + np.arange(S) * TIMESTEP

        # Finite-difference step per state dimension
        eps = np.array(HJ_GRID_DX) * 0.5

        # ------------------------------------------------------------
        # 1) Assemble all query points for one batched V(...) call
        # ------------------------------------------------------------
        pts_list = []

        # center points: (S, NX+1) = [t, x, y, yaw, vel]
        pts_list.append(np.column_stack([t_ks, x_pred]))

        # +/- perturbations for each state dimension
        for i in range(NX):
            dx = np.zeros((S, NX), dtype=float)
            dx[:, i] = eps[i]
            
            # plus
            pts_list.append(np.column_stack([t_ks, x_pred + dx]))
            pts_list[-1][:, 3] = (pts_list[-1][:, 3] + np.pi) % (2 * np.pi) - np.pi  # Wrap yaw to [-pi, pi]
            
            # minus
            pts_list.append(np.column_stack([t_ks, x_pred - dx]))
            pts_list[-1][:, 3] = (pts_list[-1][:, 3] + np.pi) % (2 * np.pi) - np.pi  # Wrap yaw to [-pi, pi]

        all_pts = np.vstack(pts_list)       # shape ((1 + 2*NX)*S, NX+1)

        # Single batched interpolator evaluation
        all_vals = self.vf_interp(all_pts)  # shape ((1 + 2*NX)*S,)

        # ------------------------------------------------------------
        # 2) Unpack values
        # ------------------------------------------------------------
        idx = 0
        a_all = all_vals[idx:idx + S]
        idx += S

        v_pm = []
        for i in range(NX):
            v_plus  = all_vals[idx:idx + S]
            idx += S
            v_minus = all_vals[idx:idx + S]
            idx += S
            v_pm.append((v_plus, v_minus))

        # ------------------------------------------------------------
        # 3) Gradient by central finite differences
        # ------------------------------------------------------------
        b_all = np.column_stack([
            (v_pm[i][0] - v_pm[i][1]) / (2.0 * eps[i])
            for i in range(NX)
        ])  # shape (S, NX)

        # ------------------------------------------------------------
        # 4) Pack stage parameters
        # PARAM = [xbar(4), a(1), b(4)]  => NP = 9
        # ------------------------------------------------------------
        params_all = np.zeros((S, NP), dtype=float)
        params_all[:, get_param("xbar")] = x_pred
        params_all[:, get_param("a")]    = a_all[:, None]
        params_all[:, get_param("b")]    = b_all

        return params_all

    def run(self, t0, x0):

        real_path   = [[t0, *x0]]
        real_cost   = []
        pred_path   = []
        input_hist = []

        while self.registry.get("latest.info") is None:
            if self.stop_event.is_set():
                return
            print("Waiting for latest.info from verifier...")
            tm.sleep(0.1)

        for k in tqdm(range(200)):
            if self.stop_event.is_set():
                return

            t = real_path[-1][0]
            x = real_path[-1][1:]

            u0 = self.compute_control(t, x)
            input_hist.append(u0)

            pred_traj = self.get_optimal_states()
            pred_path.append(pred_traj)
            
            # First predicted state at next time step
            next_state = pred_traj[1]

            real_cost.append(self.evaluate_real_cost(t, x))
            real_path.append([t + TIMESTEP] + list(next_state))

        # plot_vf_xy_slice(
        #     controller=self,
        #     time=real_path[-1][0],
        #     state=real_path[-1][1:],
        #     save_path="vf_xy.png",
        #     obstacles=OBSTACLES,
        # )

        # plot_vf_yaw_vel_slice(
        #     controller=self,
        #     time=real_path[-1][0],
        #     state=real_path[-1][1:],
        #     save_path="vf_yaw_vel.png",
        # )

        # # keep a reference to anim
        animate_vf_xy(
            controller=self,
            real_path=real_path,
            pred_path=pred_path,
            save_path="vf_xy_anim.gif",
            obstacles=OBSTACLES,
        )




    def close(self):
        pass


#######################

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter


OBSTACLES = [
    (0.0, 0.0, 2.0, 2.0),
    (1.2, 2.8, 2.8, 0.4),
]


def _wrap_yaw(yaw):
    return (yaw + np.pi) % (2 * np.pi) - np.pi


def _draw_obstacles(ax, obstacles=None):
    if obstacles is None:
        obstacles = []
    for ox, oy, ow, oh in obstacles:
        ax.add_patch(
            plt.Rectangle(
                (ox, oy), ow, oh,
                edgecolor="black",
                facecolor="gray",
                alpha=0.4,
            )
        )


def _hj_axes():
    x_grid = np.linspace(HJ_GRID_MIN[0], HJ_GRID_MAX[0], HJ_GRID_STP[0])
    y_grid = np.linspace(HJ_GRID_MIN[1], HJ_GRID_MAX[1], HJ_GRID_STP[1])
    yaw_grid = np.linspace(HJ_GRID_MIN[2], HJ_GRID_MAX[2], HJ_GRID_STP[2])
    vel_grid = np.linspace(HJ_GRID_MIN[3], HJ_GRID_MAX[3], HJ_GRID_STP[3])
    return x_grid, y_grid, yaw_grid, vel_grid


def _nearest_index(grid, value):
    return int(np.argmin(np.abs(grid - value)))


def plot_vf_xy_slice(controller, time, state, save_path=None, obstacles=None):
    """
    Plot XY slice of V at fixed (time, yaw, vel).
    state = [x, y, yaw, vel]
    """
    x, y, yaw, vel = map(float, state)
    yaw = _wrap_yaw(yaw)

    x_grid, y_grid, _, _ = _hj_axes()
    xg, yg = np.meshgrid(x_grid, y_grid, indexing="ij")

    pts = np.column_stack([
        np.full(xg.size, time),
        xg.ravel(),
        yg.ravel(),
        np.full(xg.size, yaw),
        np.full(xg.size, vel),
    ])
    Z = controller.vf_interp(pts).reshape(len(x_grid), len(y_grid))

    fig, ax = plt.subplots(figsize=(7, 6))
    levels = 30 if np.max(Z) > np.min(Z) else 1
    cf = ax.contourf(x_grid, y_grid, Z.T, levels=levels)
    ax.contour(x_grid, y_grid, Z.T, levels=10, colors="black", linewidths=0.4)

    if np.min(Z) < 0 < np.max(Z):
        ax.contour(x_grid, y_grid, Z.T, levels=[0.0], colors="red", linewidths=2.0)

    _draw_obstacles(ax, obstacles)
    ax.plot(x, y, "ro", label="current state")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(f"VF XY slice at t={time:.2f}, yaw={yaw:.2f}, vel={vel:.2f}")
    ax.axis("equal")
    ax.grid(True)
    ax.legend()
    fig.colorbar(cf, ax=ax, label="V")

    if save_path is not None:
        fig.savefig(save_path, dpi=200, bbox_inches="tight")

    return fig, ax


def plot_vf_yaw_vel_slice(controller, time, state, save_path=None):
    """
    Plot yaw-vel slice of V at fixed (time, x, y).
    state = [x, y, yaw, vel]
    """
    x, y, yaw, vel = map(float, state)

    _, _, yaw_grid, vel_grid = _hj_axes()
    yg, vg = np.meshgrid(yaw_grid, vel_grid, indexing="ij")

    pts = np.column_stack([
        np.full(yg.size, time),
        np.full(yg.size, x),
        np.full(yg.size, y),
        yg.ravel(),
        vg.ravel(),
    ])
    Z = controller.vf_interp(pts).reshape(len(yaw_grid), len(vel_grid))

    fig, ax = plt.subplots(figsize=(7, 5))
    levels = 30 if np.max(Z) > np.min(Z) else 1
    cf = ax.contourf(yaw_grid, vel_grid, Z.T, levels=levels)
    ax.contour(yaw_grid, vel_grid, Z.T, levels=10, colors="black", linewidths=0.4)

    if np.min(Z) < 0 < np.max(Z):
        ax.contour(yaw_grid, vel_grid, Z.T, levels=[0.0], colors="red", linewidths=2.0)

    ax.plot(_wrap_yaw(yaw), vel, "ro", label="current state")
    ax.set_xlabel("yaw")
    ax.set_ylabel("velocity")
    ax.set_title(f"VF yaw-vel slice at t={time:.2f}, x={x:.2f}, y={y:.2f}")
    ax.grid(True)
    ax.legend()
    fig.colorbar(cf, ax=ax, label="V")

    if save_path is not None:
        fig.savefig(save_path, dpi=200, bbox_inches="tight")

    return fig, ax


def animate_vf_xy(controller, real_path, pred_path, save_path=None, obstacles=None, interval_ms=100):
    """
    3-panel debug animation:
      1) real VF slice in (x, y)
      2) interpolated VF slice in (x, y)
      3) real VF slice in (yaw, vel)

    real_path: list of [t, x, y, yaw, vel]
    pred_path: list of predicted trajectories, each shape (N+1, NX)
    """
    real_path = np.asarray(real_path, dtype=float)
    pred_path = [np.asarray(p, dtype=float) for p in pred_path]

    x_grid, y_grid, yaw_grid, vel_grid = _hj_axes()
    xg, yg = np.meshgrid(x_grid, y_grid, indexing="ij")
    yawg, velg = np.meshgrid(yaw_grid, vel_grid, indexing="ij")

    # time axis for real VF tensor
    vf = controller.latest_vf
    vf_time_grid = np.arange(vf.shape[0]) * 0.2   # or use your actual VF time step

    fig = plt.figure(figsize=(14, 10))
    ax_xy_real   = fig.add_subplot(2, 2, 1)
    ax_xy_interp = fig.add_subplot(2, 2, 2)
    ax_yv_real   = fig.add_subplot(2, 1, 2)

    x_min, x_max = HJ_GRID_MIN[0], HJ_GRID_MAX[0]
    y_min, y_max = HJ_GRID_MIN[1], HJ_GRID_MAX[1]
    yaw_min, yaw_max = HJ_GRID_MIN[2], HJ_GRID_MAX[2]
    vel_min, vel_max = HJ_GRID_MIN[3], HJ_GRID_MAX[3]

    def _draw_heading_arrow(ax, x, y, yaw, length=0.25, color="white", lw=2.5):
        dx = length * np.cos(yaw)
        dy = length * np.sin(yaw)
        ax.arrow(
            x, y, dx, dy,
            head_width=0.08,
            head_length=0.10,
            fc=color,
            ec=color,
            linewidth=lw,
            length_includes_head=True,
            zorder=8,
        )

    def update(i):
        for ax in (ax_xy_real, ax_xy_interp, ax_yv_real):
            ax.clear()

        # current state
        t, x, y, yaw, vel = real_path[i]
        yaw = _wrap_yaw(yaw)

        # nearest indices into real VF tensor
        it = _nearest_index(vf_time_grid, t)
        iyaw = _nearest_index(yaw_grid, yaw)
        ivel = _nearest_index(vel_grid, vel)
        ix = _nearest_index(x_grid, x)
        iy = _nearest_index(y_grid, y)

        # ------------------------------------------------------------
        # 1) REAL VF: XY slice at fixed (t, yaw, vel)
        # ------------------------------------------------------------
        Z_xy_real = vf[it, :, :, iyaw, ivel]

        ax_xy_real.contourf(x_grid, y_grid, Z_xy_real.T, levels=30, alpha=0.9)
        ax_xy_real.contour(x_grid, y_grid, Z_xy_real.T, levels=10,
                           colors="black", linewidths=0.4)

        if np.min(Z_xy_real) < 0 < np.max(Z_xy_real):
            ax_xy_real.contour(x_grid, y_grid, Z_xy_real.T, levels=[0.0],
                               colors="red", linewidths=2.5)

        _draw_obstacles(ax_xy_real, obstacles)
        ax_xy_real.plot(real_path[:i+1, 1], real_path[:i+1, 2],
                        color="white", linewidth=2.5, zorder=6, label="real path")
        ax_xy_real.plot(x, y, "ro", markersize=8, zorder=7, label="current")
        _draw_heading_arrow(ax_xy_real, x, y, yaw, color="white")

        if i < len(pred_path):
            pred = pred_path[i]
            ax_xy_real.plot(pred[:, 0], pred[:, 1],
                            color="orange", linestyle="--", linewidth=2.0,
                            zorder=6, label="prediction")

        ax_xy_real.set_xlim(x_min, x_max)
        ax_xy_real.set_ylim(y_min, y_max)
        ax_xy_real.set_aspect("equal", adjustable="box")
        ax_xy_real.set_title("Real VF slice: (x, y)")
        ax_xy_real.set_xlabel("x")
        ax_xy_real.set_ylabel("y")
        ax_xy_real.grid(True)
        ax_xy_real.legend(loc="upper right")

        # ------------------------------------------------------------
        # 2) INTERP VF: XY slice at fixed (t, yaw, vel)
        # ------------------------------------------------------------
        pts_xy = np.column_stack([
            np.full(xg.size, t),
            xg.ravel(),
            yg.ravel(),
            np.full(xg.size, yaw),
            np.full(xg.size, vel),
        ])
        pts_xy[:, 1] = np.clip(pts_xy[:, 1], x_min, x_max)
        pts_xy[:, 2] = np.clip(pts_xy[:, 2], y_min, y_max)
        pts_xy[:, 3] = _wrap_yaw(pts_xy[:, 3])
        pts_xy[:, 4] = np.clip(pts_xy[:, 4], vel_min, vel_max)

        Z_xy_interp = controller.vf_interp(pts_xy).reshape(len(x_grid), len(y_grid))

        ax_xy_interp.contourf(x_grid, y_grid, Z_xy_interp.T, levels=30, alpha=0.9)
        ax_xy_interp.contour(x_grid, y_grid, Z_xy_interp.T, levels=10,
                             colors="black", linewidths=0.4)

        if np.min(Z_xy_interp) < 0 < np.max(Z_xy_interp):
            ax_xy_interp.contour(x_grid, y_grid, Z_xy_interp.T, levels=[0.0],
                                 colors="red", linewidths=2.5)

        _draw_obstacles(ax_xy_interp, obstacles)
        ax_xy_interp.plot(real_path[:i+1, 1], real_path[:i+1, 2],
                          color="white", linewidth=2.5, zorder=6, label="real path")
        ax_xy_interp.plot(x, y, "ro", markersize=8, zorder=7, label="current")
        _draw_heading_arrow(ax_xy_interp, x, y, yaw, color="white")

        if i < len(pred_path):
            pred = pred_path[i]
            ax_xy_interp.plot(pred[:, 0], pred[:, 1],
                              color="orange", linestyle="--", linewidth=2.0,
                              zorder=6, label="prediction")

        ax_xy_interp.set_xlim(x_min, x_max)
        ax_xy_interp.set_ylim(y_min, y_max)
        ax_xy_interp.set_aspect("equal", adjustable="box")
        ax_xy_interp.set_title("Interpolated VF slice: (x, y)")
        ax_xy_interp.set_xlabel("x")
        ax_xy_interp.set_ylabel("y")
        ax_xy_interp.grid(True)
        ax_xy_interp.legend(loc="upper right")

        # ------------------------------------------------------------
        # 3) REAL VF: (yaw, vel) slice at fixed (t, x, y)
        # ------------------------------------------------------------
        Z_yv_real = vf[it, ix, iy, :, :]

        ax_yv_real.contourf(yaw_grid, vel_grid, Z_yv_real.T, levels=30, alpha=0.9)
        ax_yv_real.contour(yaw_grid, vel_grid, Z_yv_real.T, levels=10,
                           colors="black", linewidths=0.4)

        if np.min(Z_yv_real) < 0 < np.max(Z_yv_real):
            ax_yv_real.contour(yaw_grid, vel_grid, Z_yv_real.T, levels=[0.0],
                               colors="red", linewidths=2.5)

        ax_yv_real.plot(yaw, vel, "ro", markersize=8, label="current")
        if i < len(pred_path):
            pred = pred_path[i]
            ax_yv_real.plot(pred[:, 2], pred[:, 3],
                            color="orange", linestyle="--", linewidth=2.0,
                            label="prediction")

        ax_yv_real.set_xlim(yaw_min, yaw_max)
        ax_yv_real.set_ylim(vel_min, vel_max)
        ax_yv_real.set_title("Real VF slice: (yaw, vel)")
        ax_yv_real.set_xlabel("yaw")
        ax_yv_real.set_ylabel("velocity")
        ax_yv_real.grid(True)
        ax_yv_real.legend(loc="upper right")

        fig.suptitle(f"t = {t:.2f}", fontsize=14)
        return []

    anim = FuncAnimation(
        fig,
        update,
        frames=len(real_path),
        interval=interval_ms,
        blit=False,
        repeat=False,
    )

    if save_path is not None:
        anim.save(save_path, writer=PillowWriter(fps=max(1, int(1000 / interval_ms))))

    return fig, (ax_xy_real, ax_xy_interp, ax_yv_real), anim


def animate_heat_map_segment(Car, states, nr, predicted_path, costs_list=None, save_path="./",
                             used_sets=None, file_path=None):
    """
    逐帧渲染并保存为 PNG（存入 save_path/frames/），
    所有帧保存完毕后用 Pillow 合并成 GIF 动画。
 
    优点：
      - 每帧独立 figure，渲染后立即 plt.close()，内存峰值低
      - 可中断续跑（已存在的帧直接跳过）
      - 不依赖 ffmpeg，仅需 Pillow
    used_sets : list[int], 长度 >= nr，每步对应使用的 dataset 编号
    file_path : list[str]，dataset 编号 → npy 文件路径的映射
    """
    import glob
    from PIL import Image as PILImage
 
    print("Animating heat map for interpolation segmentation")
 
    # 跟踪当前已加载的 dataset，按需切换，避免重复 IO
    _active_dataset = [None]
 
    def _ensure_dataset(index_t):
        if used_sets is None or file_path is None:
            return
        ds = used_sets[index_t]
        if ds != _active_dataset[0]:
            Car.set_value_function(file_path[ds])
            _active_dataset[0] = ds
 
    # ── 确定输出 GIF 文件名 ────────────────────────────────────────────────
    if USE_HJ_LINEAR:
        gif_name = "animation_seg_4D_acados_heuristic_linear.gif"
    elif USE_HJ_DIAG_QUADRATIC:
        gif_name = "animation_seg_4D_acados_heuristic_diagonal.gif"
    elif USE_HJ_FULL_QUADRATIC:
        gif_name = "animation_seg_4D_acados_heuristic_full.gif"
    else:
        raise RuntimeError("No HJ surrogate mode selected.")
 
    frames_dir = os.path.join(save_path, "frames")
    os.makedirs(frames_dir, exist_ok=True)
 
    # ── 逐帧渲染并保存 PNG ────────────────────────────────────────────────
    progress = tqdm(total=nr, desc="Rendering frames")
    for index_t in range(nr):
        frame_path = os.path.join(frames_dir, f"frame_{index_t:04d}.png")
        if os.path.exists(frame_path):
            # 支持断点续跑：已有帧直接跳过
            progress.update(1)
            continue
 
        _ensure_dataset(index_t)
 
        V_interpolated_seg = Car.interpolate_V_segment(states[index_t], dyn.prediction_horizon)
        V_seg = Car.value_function # not interpolated
 
        fig = plt.figure(figsize=(18, 10))
        ax  = fig.add_subplot(2, 3, 1)
        ax2 = fig.add_subplot(2, 3, 2)
        ax3 = fig.add_subplot(2, 3, 3)
        ax4 = fig.add_subplot(2, 1, 2)   # Time vs Cost — full bottom row
 
        t     = states[index_t][0]
        x     = states[index_t][1]
        y     = states[index_t][2]
        theta = states[index_t][3]
        v     = states[index_t][4]
        # delta = states[index_t][5]
 
        _it, _ix, _iy, _ith, _iv = Car.coord_to_index([t, x, y, theta, v])
 
        # XY heatmap — vectorised
        x_vals = np.linspace(Car.X0, Car.X0 + Car.XN, 25)
        y_vals = np.linspace(Car.Y0, Car.Y0 + Car.YN, 25)
        xg, yg = np.meshgrid(x_vals, y_vals, indexing='ij')   # (100,100)
        n_pts  = xg.size
        pts_xy = np.column_stack([
            np.full(n_pts, t), xg.ravel(), yg.ravel(),
            np.full(n_pts, theta), np.full(n_pts, v)
        ])
        # Z_xy = V_interpolated_seg(pts_xy).reshape(100, 100)
        Z_xy = V_seg[_it, :, :, _ith, _iv]
 
        _zmin_xy, _zmax_xy = Z_xy.min(), Z_xy.max()
        if _zmax_xy - _zmin_xy > 1e-10:
            ax.contourf(x_vals, y_vals, Z_xy.T, levels=np.linspace(_zmin_xy, _zmax_xy, 50), cmap="viridis")
            ax.contour(x_vals, y_vals, Z_xy.T, levels=np.linspace(_zmin_xy, _zmax_xy, 10), colors="black", linewidths=0.5)
        else:
            ax.contourf(x_vals, y_vals, Z_xy.T, levels=1, cmap="viridis")
        if Z_xy.max() > 0 and Z_xy.min() < 0:
            ax.contour(x_vals, y_vals, Z_xy.T, levels=[0], colors="red", linewidths=2.0)
        ax.set_xlabel('X-axis');  ax.set_ylabel('Y-axis')
        ax.set_title('Value Function (X vs Y)');  ax.axis('equal')
        ax.plot(predicted_path[index_t][0], predicted_path[index_t][1], "yo")
        ax.plot(x, y, "ro")
        for _ox, _oy, _ow, _oh in OBSTACLES:
            ax.add_patch(plt.Rectangle((_ox, _oy), _ow, _oh,
                                       linewidth=1.5, edgecolor='white',
                                       facecolor='gray', alpha=0.6, zorder=5))
        for _w, _h, _x0, _y0, _x1, _y1, _t0, _t1 in MOVING_OBSTACLES:
            if not (_t0 <= t <= _t1):
                continue
            _alpha = (t - _t0) / (_t1 - _t0)
            _x = _x0 + _alpha * (_x1 - _x0)
            _y = _y0 + _alpha * (_y1 - _y0)
            _xlo = max(_x, 0.0); _ylo = max(_y, 0.0)
            _xhi = min(_x + _w, 4.0); _yhi = min(_y + _h, 4.0)
            if _xhi > _xlo and _yhi > _ylo:
                ax.add_patch(plt.Rectangle((_xlo, _ylo), _xhi - _xlo, _yhi - _ylo,
                                        linewidth=1.5, edgecolor='white',
                                        facecolor='white', alpha=0.5, zorder=6))
 
 
        # XY heatmap — vectorised: build all query points at once
        x_vals = np.linspace(Car.X0, Car.X0 + Car.XN, 100)
        y_vals = np.linspace(Car.Y0, Car.Y0 + Car.YN, 100)
        xg, yg = np.meshgrid(x_vals, y_vals, indexing='ij')   # (100,100)
        n_pts  = xg.size
        pts_xy = np.column_stack([
            np.full(n_pts, t), xg.ravel(), yg.ravel(),
            np.full(n_pts, theta), np.full(n_pts, v)
        ])
        Z_xy = V_interpolated_seg(pts_xy).reshape(100, 100)
 
        _zmin_xy, _zmax_xy = Z_xy.min(), Z_xy.max()
        if _zmax_xy - _zmin_xy > 1e-10:
            c = ax2.contourf(x_vals, y_vals, Z_xy.T, levels=np.linspace(_zmin_xy, _zmax_xy, 50), cmap="viridis")
            ax2.contour(x_vals, y_vals, Z_xy.T, levels=np.linspace(_zmin_xy, _zmax_xy, 10), colors="black", linewidths=0.5)
        else:
            c = ax2.contourf(x_vals, y_vals, Z_xy.T, levels=1, cmap="viridis")
        if Z_xy.max() > 0 and Z_xy.min() < 0:
            ax2.contour(x_vals, y_vals, Z_xy.T, levels=[0], colors="red", linewidths=2.0)
        ax2.set_xlabel('X-axis');  ax2.set_ylabel('Y-axis')
        ax2.set_title('Value Function (X vs Y)');  ax2.axis('equal')
        ax2.plot(predicted_path[index_t][0], predicted_path[index_t][1], "yo")
        ax2.plot(x, y, "ro")
        for _ox, _oy, _ow, _oh in OBSTACLES:
            ax2.add_patch(plt.Rectangle((_ox, _oy), _ow, _oh,
                                        linewidth=1.5, edgecolor='white',
                                        facecolor='gray', alpha=0.6, zorder=5))
        for _w, _h, _x0, _y0, _x1, _y1, _t0, _t1 in MOVING_OBSTACLES:
            if not (_t0 <= t <= _t1):
                continue
            _alpha = (t - _t0) / (_t1 - _t0)
            _x = _x0 + _alpha * (_x1 - _x0)
            _y = _y0 + _alpha * (_y1 - _y0)
            _xlo = max(_x, 0.0); _ylo = max(_y, 0.0)
            _xhi = min(_x + _w, 4.0); _yhi = min(_y + _h, 4.0)
            if _xhi > _xlo and _yhi > _ylo:
                ax2.add_patch(plt.Rectangle((_xlo, _ylo), _xhi - _xlo, _yhi - _ylo,
                                        linewidth=1.5, edgecolor='white',
                                        facecolor='white', alpha=0.5, zorder=6))
 
        # Theta-v heatmap — vectorised
        theta_vals = np.linspace(-np.pi, np.pi, 15)
        v_vals     = np.linspace(Car.min_velocity, Car.max_velocity, 11)
        tg, vg = np.meshgrid(theta_vals, v_vals, indexing='ij')   # (100,100)
        n_pts  = tg.size
        pts_tv = np.column_stack([
            np.full(n_pts, t), np.full(n_pts, x), np.full(n_pts, y),
            tg.ravel(), vg.ravel()
        ])
        # Z_tv = V_interpolated_seg(pts_tv).reshape(100, 100)
        Z_tv = V_seg[_it, _ix, _iy, :, :]
 
        _zmin_tv, _zmax_tv = Z_tv.min(), Z_tv.max()
        if _zmax_tv - _zmin_tv > 1e-10:
            ax3.contourf(theta_vals, v_vals, Z_tv.T, levels=np.linspace(_zmin_tv, _zmax_tv, 50), cmap="viridis")
            ax3.contour(theta_vals, v_vals, Z_tv.T, levels=np.linspace(_zmin_tv, _zmax_tv, 10), colors="black", linewidths=0.5)
        else:
            ax3.contourf(theta_vals, v_vals, Z_tv.T, levels=1, cmap="viridis")
        if Z_tv.max() > 0 and Z_tv.min() < 0:
            ax3.contour(theta_vals, v_vals, Z_tv.T, levels=[0], colors="red", linewidths=2.0)
        ax3.set_xlabel('Theta');  ax3.set_ylabel('Velocity')
        ax3.set_title('Value Function (Theta vs V)')
        ax3.plot(predicted_path[index_t][2], predicted_path[index_t][3], "yo")
        ax3.plot(theta, v, "ro")
 
        # 分割框
        seg = Car.grids_segment_list
        ax.plot(
            [seg[index_t][1][0], seg[index_t][1][-1], seg[index_t][1][-1],
             seg[index_t][1][0], seg[index_t][1][0]],
            [seg[index_t][2][0], seg[index_t][2][0], seg[index_t][2][-1],
             seg[index_t][2][-1], seg[index_t][2][0]], "y"
        )
        ax2.plot(
            [seg[index_t][1][0], seg[index_t][1][-1], seg[index_t][1][-1],
             seg[index_t][1][0], seg[index_t][1][0]],
            [seg[index_t][2][0], seg[index_t][2][0], seg[index_t][2][-1],
             seg[index_t][2][-1], seg[index_t][2][0]], "y"
        )
        ax3.plot(
            [seg[index_t][3][0], seg[index_t][3][-1], seg[index_t][3][-1],
             seg[index_t][3][0], seg[index_t][3][0]],
            [seg[index_t][4][0], seg[index_t][4][0], seg[index_t][4][-1],
             seg[index_t][4][-1], seg[index_t][4][0]], "y"
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
 
        fig.tight_layout()
        fig.savefig(frame_path, dpi=100)
        plt.close(fig)
        progress.update(1)
 
    progress.close()
 
    print("Assembling GIF from frames...")
    frame_paths = sorted(glob.glob(os.path.join(frames_dir, "frame_*.png")))
    if not frame_paths:
        raise RuntimeError(f"No frames found in {frames_dir}")
 
    pil_frames = [PILImage.open(fp).convert("RGB") for fp in frame_paths]
    gif_path = os.path.join(save_path, gif_name)
    pil_frames[0].save(
        gif_path,
        save_all=True,
        append_images=pil_frames[1:],
        loop=0,
        duration=100,   # 100 ms / frame → 10 fps
        optimize=False,
    )
    for img in pil_frames:
        img.close()
    print(f"GIF saved → {gif_path}  ({len(pil_frames)} frames)")

###################

if __name__ == "__main__":
    
    from math import pi 

    t0 = 0.0
    x0 = [3.0, 0.5, pi/2, 0.0]

    Controller().run(t0, x0)
