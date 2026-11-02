from tqdm import tqdm

import os
import shutil
import time as tm
import numpy as np
import casadi as ca
from acados_template import AcadosModel, AcadosOcp, AcadosOcpSolver

def accum(params):
    i = 0
    for p in params:
        name, size = (p, 1) if isinstance(p, str) else p
        yield name, slice(i, i+size)
        i += size

STATE = {name: s for name, s in accum(["x", "y", "yaw", "vel"])}
NX = max(s.stop for s in STATE.values())

def get_state(name, seq=None):
    return STATE[name] if seq is None else seq[STATE[name]]


INPUT = {name: s for name, s in accum(["accel", "steer"])}
NU = max(s.stop for s in INPUT.values())

def get_input(name, seq=None):
    return INPUT[name] if seq is None else seq[INPUT[name]]


# Diagonal quadratic surrogates around xbar_k, for two grid functions:
#   hat(F)_k(x) = a_k + g_k^T (x - xbar_k) + 0.5 * hdiag_k ⊙ (x - xbar_k)^2
#   - progress: time-to-reach map TTR(x)        -> cost
#   - safety:   local HJ value function V(t, x) -> constraint V <= -margin
#
# Parameter layout per stage:
#   [xbar(4), a(1), g(4), hdiag(4), sa(1), sg(4), shdiag(4)]  =>  NP = 22
PARAM = {name: s for name, s in accum([
    ("xbar",   NX),   # Linearization point
    ("a",      1),    # TTR(xbar)
    ("g",      NX),   # grad TTR(xbar)        — linear term
    ("hdiag",  NX),   # diag Hessian TTR(xbar) — quadratic term (clamped >= 0)
    ("sa",     1),    # V(xbar)
    ("sg",     NX),   # grad V(xbar)
    ("shdiag", NX),   # diag Hessian V(xbar) (clamped >= 0)
])}
NP = max(s.stop for s in PARAM.values())

def get_param(name, seq=None):
    return PARAM[name] if seq is None else seq[PARAM[name]]


class Controller:

    def __init__(self, registry, state_queue, stop_event):

        # Runtime communication primitives
        self.registry = registry
        self.state_queue = state_queue
        self.stop_event = stop_event

        self.config = self.registry['config']

        # Controller parameters
        self.N = self.config['controller']['n']
        self.TIMESTEP = self.config['controller']['timestep']
        self.Qhj = self.config['controller']['qhj']
        self.safety_margin = self.config['controller']['safety_margin']
        self.safety_slack_l1 = self.config['controller']['safety_slack_l1']
        self.safety_slack_l2 = self.config['controller']['safety_slack_l2']

        # Model parameters
        self.L = 0.32   # Wheelbase length (m)
        self.min_velocity = 0.0
        self.max_velocity = self.config['controller']['max_vel']
        self.min_acceleration = self.config['controller']['max_accel'] * -1
        self.max_acceleration = self.config['controller']['max_accel']
        self.min_steering = self.config['controller']['max_steer'] * -1
        self.max_steering = self.config['controller']['max_steer']

        self.HJ_TIMESTEP = self.config['verifier']['timestep']
        self.HJ_GRID_MIN = [axis['bounds'][0] for axis in self.config['verifier']['axes'] if axis['name'] != 't']
        self.HJ_GRID_MAX = [axis['bounds'][1] for axis in self.config['verifier']['axes'] if axis['name'] != 't']
        self.HJ_GRID_STP = [axis['points'] for axis in self.config['verifier']['axes'] if axis['name'] != 't']
        self.HJ_GRID_DX = [(self.HJ_GRID_MAX[i] - self.HJ_GRID_MIN[i]) / (self.HJ_GRID_STP[i] - 1) for i in range(4)]

        # Precompute spatial grids for direct indexing
        self.spatial_grids = [
            np.linspace(self.HJ_GRID_MIN[i], self.HJ_GRID_MAX[i], self.HJ_GRID_STP[i])
            for i in range(NX)
        ]

        self.solver = self._init_acados_solver()
        self._sol = {
            'x': np.zeros((self.N + 1, NX)),
            'u': np.zeros((self.N, NU)),
            'healthy': False,  # Flag to indicate if the solver has found a healthy solution before
        }

        self.latest_info = {'interval': (-np.inf, -np.inf), 'name': 'bar'}

    def _init_acados_solver(self):

        model = self._init_acados_model()

        ocp = self._init_acados_ocp(model)

        # Solver options
        ocp.solver_options.tf                  = self.N * self.TIMESTEP
        ocp.solver_options.N_horizon           = self.N
        ocp.solver_options.integrator_type     = "DISCRETE"
        ocp.solver_options.nlp_solver_type     = "SQP_RTI"   # Real-Time Iteration
        ocp.solver_options.qp_solver           = "PARTIAL_CONDENSING_HPIPM"
        ocp.solver_options.qp_solver_cond_N    = self.N
        # GAUSS_NEWTON: required for diagonal quadratic surrogate — the hdiag
        # second-order term makes the cost Hessian indefinite when hdiag < 0,
        # but we clamp hdiag >= 0 in _build_stage_params so GN is both safe
        # and faster than EXACT for SQP_RTI.
        ocp.solver_options.hessian_approx      = "GAUSS_NEWTON"
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
        model.p = ca.vertcat(*[ca.SX.sym(param, s.stop - s.start)
                               for param, s in PARAM.items()])
        _param = lambda name: get_param(name, model.p)

        # Dynamics
        dt = self.TIMESTEP    
        f_x     = _state("x") + dt * _state("vel") * ca.cos(_state("yaw"))
        f_y     = _state("y") + dt * _state("vel") * ca.sin(_state("yaw"))
        f_th    = _state("yaw") + dt * _state("vel") * ca.tan(_input("steer")) / self.L
        f_v     = _state("vel") + dt * _input("accel")
        model.disc_dyn_expr = ca.vertcat(f_x, f_y, f_th, f_v)

        # Diagonal quadratic surrogates:
        #   hat(F)(x) = a + g^T dz + 0.5 * hdiag ⊙ dz^2
        # where dz = x - xbar
        dz = model.x - _param("xbar")
        ttr_interp = (
            _param("a")
            + ca.dot(_param("g"), dz)
            + 0.5 * ca.dot(_param("hdiag"), dz * dz)
        )
        vf_interp = (
            _param("sa")
            + ca.dot(_param("sg"), dz)
            + 0.5 * ca.dot(_param("shdiag"), dz * dz)
        )

        # Stage and terminal cost: go towards the goal as fast as possible
        model.cost_expr_ext_cost = self.Qhj * ttr_interp
        model.cost_expr_ext_cost_e = self.Qhj * ttr_interp

        # Safety constraint on stages 1..N (stage 0 is the fixed measured state)
        model.con_h_expr = vf_interp
        model.con_h_expr_e = vf_interp

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

        # Safety: V(t_k, x_k) <= -margin, softened with a large slack penalty
        v_lower = np.array([-1e6])
        v_upper = np.array([-self.safety_margin])
        l1 = np.array([self.safety_slack_l1])
        l2 = np.array([self.safety_slack_l2])

        ocp.constraints.lh    = v_lower
        ocp.constraints.uh    = v_upper
        ocp.constraints.idxsh = np.array([0])
        ocp.cost.zl, ocp.cost.zu, ocp.cost.Zl, ocp.cost.Zu = l1, l1, l2, l2

        ocp.constraints.lh_e    = v_lower
        ocp.constraints.uh_e    = v_upper
        ocp.constraints.idxsh_e = np.array([0])
        ocp.cost.zl_e, ocp.cost.zu_e, ocp.cost.Zl_e, ocp.cost.Zu_e = l1, l1, l2, l2

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
        
        info = self.registry.get('latest.info', None)
        if info is None:
            return

        current_set_name = str(info['name'])
        t_start, t_end = float(info['interval'][0]), float(info['interval'][1])

        if not hasattr(self, 'current_loaded_set_name'):
            self.current_loaded_set_name = None

        if current_set_name != self.current_loaded_set_name:
            if t_start <= time <= t_end:
                print(f"[Controller] New received set: '{current_set_name}' valid in [{t_start:.2f}, {t_end:.2f}] at t_sim={time:.2f}s")
                
                self.current_loaded_set_name = current_set_name
                self.latest_info = dict(info)
                self.latest_vf = self.registry[current_set_name + '.vf']

                HJ_NT = self.latest_vf.shape[0]
                self.vf_time_grid = t_start + np.arange(HJ_NT) * self.HJ_TIMESTEP

    def _current_set_idx(self) -> int:
        """Return the set_idx of the currently active verification set, or -1 if unknown."""
        tl = self.registry.get('verifier.timeline', [])
        for entry in reversed(tl):
            if list(self.latest_info.get('interval', [])) == entry.get('interval'):
                return entry['set_idx']
        return -1

    def _query_vf(self, t_pts, x_pts, vf=None, time_grid=None):
        """
        5D local linear interpolation on the value function ndarray.
        yaw (spatial dim 2) is treated as periodic on [-π, π].

        Parameters
        ----------
        t_pts : ndarray, shape (M,)
        x_pts : ndarray, shape (M, NX)
        vf, time_grid : grid function to interpolate (default: latest safety set)

        Returns
        -------
        vals : ndarray, shape (M,)
        """
        if vf is None:
            vf, time_grid = self.latest_vf, self.vf_time_grid   # shape (NT, N0, N1, N2, N3)
        M  = len(t_pts)

        # All 5 grids: [time, x, y, yaw, vel]
        grids = [time_grid, *self.spatial_grids]
        shape = vf.shape   # (NT, N0, N1, N2, N3)

        # Wrap yaw to [-π, π] before interpolation
        x_pts = x_pts.copy()
        # x_pts[:, 2] = (x_pts[:, 2] + np.pi) % (2 * np.pi) - np.pi

        all_pts = np.column_stack([t_pts, x_pts])  # (M, 5)

        idx0    = np.zeros((5, M), dtype=int)
        idx1    = np.zeros((5, M), dtype=int)
        w1      = np.zeros((5, M), dtype=float)   # weight toward right neighbor

        for d in range(5):
            g  = grids[d]
            dx = g[1] - g[0]
            i0 = np.floor((all_pts[:, d] - g[0]) / dx).astype(int)
            i0 = np.clip(i0, 0, len(g) - 2)
            alpha = np.clip((all_pts[:, d] - g[i0]) / dx, 0.0, 1.0)

            if d == 3:  # yaw: periodic — right neighbor wraps to index 0 at boundary
                i1 = np.where(i0 == len(g) - 2,
                            np.zeros(M, dtype=int),   # wrap: last cell right → index 0
                            i0 + 1)
            else:
                i1 = i0 + 1

            idx0[d] = i0
            idx1[d] = i1
            w1[d]   = alpha

        # 2^5 = 32 corner combinations
        result = np.zeros(M, dtype=float)
        for bits in range(32):
            weight = np.ones(M, dtype=float)
            idxs   = []
            for d in range(5):
                if (bits >> d) & 1:
                    weight *= w1[d]
                    idxs.append(idx1[d])
                else:
                    weight *= (1.0 - w1[d])
                    idxs.append(idx0[d])
            result += weight * vf[idxs[0], idxs[1], idxs[2], idxs[3], idxs[4]]

        return result

    def get_optimal_states(self):
        xs = np.array([self.solver.get(k, "x") for k in range(self.N+1)]) # shape (N+1, NX)
        return xs

    def get_optimal_inputs(self):
        return np.array([self.solver.get(k, "u") for k in range(self.N)]) # shape (N, NU)

    def evaluate_real_cost(self, time, state):
        t_pts = np.array([time])
        x_pts = np.array(state, dtype=float).reshape(1, NX)
        return float(self._query_vf(t_pts, x_pts)[0])

    def evaluate_interp_cost(self, time, state):
        return self.evaluate_real_cost(time, state)

    def compute_control(self, time, state):

        state = np.asarray(state, dtype=float).reshape(NX)
        state = self.bound_initial_state(state)

        # 0) Update the current value function
        self.update_current_valuefunc(time, state)

        # 1) Fix initial state at stage 0
        self.solver.set(0, "lbx", state)
        self.solver.set(0, "ubx", state)

        # 2) Build stage-wise diagonal quadratic HJ surrogates:
        #    p_k = [xbar_k, a_k, g_k, hdiag_k]
        x_pred = self._predict_states(state)
        stage_params = self._build_stage_params(time, x_pred)

        for k in range(self.N + 1):
            self.solver.set(k, "p", stage_params[k])

        # 3) Warm-start from previous solution if available
        if self._sol['healthy']:
            for k in range(self.N + 1):
                self.solver.set(k, "x", self._sol["x"][k])
            for k in range(self.N):
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
            x_cached = np.array([self.solver.get(k, "x") for k in range(self.N + 1)])
            self._sol.update(
                x=x_cached,
                u=np.array([self.solver.get(k, "u") for k in range(self.N)]),
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
        x_pred = np.zeros((self.N+1, NX))
        x_pred[0] = x0

        if self._sol['healthy']:
            prev_x = self._sol['x']
            for k in range(1, self.N+1):
                x_pred[k] = prev_x[min(k, self.N)]
        else:
            for k in range(self.N):
                x = x_pred[k]
                x_pred[k + 1] = np.array([
                    x[0] + self.TIMESTEP * x[3] * np.cos(x[2]),
                    x[1] + self.TIMESTEP * x[3] * np.sin(x[2]),
                    x[2],
                    x[3],
                ])

        return x_pred

    def _build_stage_params(self, time, x_pred):
        """
        Build per-stage parameters: progress (TTR) and safety (V) surrogates.

        Returns
        -------
        params_all : ndarray, shape (N+1, NP)
            Each row is:
                p_k = [xbar_k(4), a_k(1), g_k(4), hdiag_k(4), sa_k(1), sg_k(4), shdiag_k(4)]   NP = 22
        """
        S = self.N + 1

        x_pred = np.asarray(x_pred, dtype=float).reshape(S, NX)
        t_ks = time + np.arange(S) * self.TIMESTEP

        a_all, g_all, hdiag_all = self._surrogate(t_ks, x_pred, self.ttr_vf, self.ttr_time_grid)
        sa_all, sg_all, shdiag_all = self._surrogate(t_ks, x_pred, self.latest_vf, self.vf_time_grid)

        params_all = np.zeros((S, NP), dtype=float)
        params_all[:, get_param("xbar")]   = x_pred
        params_all[:, get_param("a")]      = a_all[:, None]
        params_all[:, get_param("g")]      = g_all
        params_all[:, get_param("hdiag")]  = hdiag_all
        params_all[:, get_param("sa")]     = sa_all[:, None]
        params_all[:, get_param("sg")]     = sg_all
        params_all[:, get_param("shdiag")] = shdiag_all

        return params_all

    def _surrogate(self, t_ks, x_pred, vf, time_grid):
        """
        Diagonal quadratic surrogate of a grid function F around each predicted state.

        For each stage k:
            xbar_k  = x_pred[k]
            a_k     = F(t_k, xbar_k)
            g_k     = grad_x F(t_k, xbar_k)      (central finite differences)
            hdiag_k = diag Hessian F(t_k, xbar_k) (second differences, clamped >= 0)

        Returns
        -------
        a_all : (S,), g_all : (S, NX), hdiag_all : (S, NX)
        """
        S = len(t_ks)

        # Finite-difference step per state dimension
        eps = np.array(self.HJ_GRID_DX) * 0.5

        # ------------------------------------------------------------
        # 1) Assemble all query points for one batched _query_vf call
        #    Layout: [center(S), +dim0(S), -dim0(S), ..., +dim3(S), -dim3(S)]
        #    Total: (1 + 2*NX) * S rows
        # ------------------------------------------------------------
        pts_t = []
        pts_x = []

        # center points
        pts_t.append(t_ks)
        pts_x.append(x_pred)

        # +/- perturbations for each state dimension
        for i in range(NX):
            dx = np.zeros((S, NX), dtype=float)
            dx[:, i] = eps[i]
            pts_t.append(t_ks)
            pts_x.append(x_pred + dx)
            pts_t.append(t_ks)
            pts_x.append(x_pred - dx)

        all_t = np.concatenate(pts_t)           # shape ((1 + 2*NX)*S,)
        all_x = np.vstack(pts_x)                # shape ((1 + 2*NX)*S, NX)

        # Single batched query
        all_vals = self._query_vf(all_t, all_x, vf, time_grid)

        # ------------------------------------------------------------
        # 2) Unpack values
        # ------------------------------------------------------------
        idx = 0
        a_all = all_vals[idx:idx + S];  idx += S

        v_pm = []
        for i in range(NX):
            v_plus  = all_vals[idx:idx + S];  idx += S
            v_minus = all_vals[idx:idx + S];  idx += S
            v_pm.append((v_plus, v_minus))

        # ------------------------------------------------------------
        # 3) Gradient by central finite differences
        #    g[k, i] = (V+ - V-) / (2 * eps[i])
        # ------------------------------------------------------------
        g_all = np.column_stack([
            (v_pm[i][0] - v_pm[i][1]) / (2.0 * eps[i])
            for i in range(NX)
        ])  # shape (S, NX)

        # ------------------------------------------------------------
        # 4) Diagonal Hessian by second finite differences
        #    hdiag[k, i] = (V+ - 2*a + V-) / eps[i]^2
        #    Clamped to >= 0 to guarantee PSD surrogate Hessian,
        #    which keeps HPIPM stable under GAUSS_NEWTON.
        # ------------------------------------------------------------
        hdiag_all = np.column_stack([
            (v_pm[i][0] - 2.0 * a_all + v_pm[i][1]) / (eps[i] ** 2)
            for i in range(NX)
        ])  # shape (S, NX)
        hdiag_all = np.maximum(hdiag_all, 0.0)   # PSD clamp

        return a_all, g_all, hdiag_all

    def run(self, t0, x0):

        real_path   = [[t0, *x0]]
        pred_path   = []
        input_hist = []

        real_cost   = []
        interp_cost = []

        # Timeline: list of set_idx used at each MPC step (index k → used set)
        used_sets: list = []

        while self.registry.get("latest.info") is None:
            if self.stop_event.is_set():
                return
            tm.sleep(0.1)

        hj_horizon = self.config['verifier']['horizon']

        # Time-to-reach map (published by the verifier before the first set) is
        # time-invariant: two identical time slices let _query_vf interpolate it.
        ttr = np.asarray(self.registry['progress.ttr'])
        self.ttr_vf = np.stack([ttr, ttr])
        self.ttr_time_grid = np.array([0.0, hj_horizon])
        number_of_steps = int(hj_horizon / self.TIMESTEP)

        for k in tqdm(range(number_of_steps), desc="Controller loop"):
            start_time = tm.time()
            if self.stop_event.is_set():
                return
            
            t = real_path[-1][0]
            x = real_path[-1][1:]

            u0 = self.compute_control(t, x)
            input_hist.append(u0)

            # Record which verification set was active this step
            used_sets.append(self._current_set_idx())
            self.registry['controller.timeline'] = {
                'mpc_dt':    self.TIMESTEP,
                'used_sets': list(used_sets),
            }

            pred_traj = self.get_optimal_states()
            pred_path.append(pred_traj)
            
            # First predicted state at next time step
            next_state = pred_traj[1]

            real_cost.append(self.evaluate_real_cost(t, x))
            interp_cost.append(self.evaluate_interp_cost(t, x))

            real_path.append([t + self.TIMESTEP] + list(next_state))

            end_time = tm.time()

            set_usage = []
            for k, sid in enumerate(used_sets):
                t_k = real_path[k][0]
                if set_usage and set_usage[-1]['set_idx'] == sid:
                    set_usage[-1]['t_end'] = t_k + self.TIMESTEP
                else:
                    set_usage.append({'set_idx': sid, 't_start': t_k, 't_end': t_k + self.TIMESTEP})

            tm.sleep(max(0.0, 0.1 - (end_time-start_time)))

        
        self.registry['results'] = {
            'real_path': real_path,
            'pred_path': pred_path,
            'input_hist': input_hist,
            'real_cost': real_cost,
            'interp_cost': interp_cost,
            'latest_vf': self.latest_vf,
            'latest_info': self.latest_info,
            'vf_interp': VFInterp(
                self.latest_vf,
                self.vf_time_grid,
                self.HJ_GRID_MIN,
                self.HJ_GRID_DX,
                self.HJ_GRID_STP,
            )
        }
        

    def close(self):
        pass

class VFInterp:
    """Pickle-able wrapper around _query_vf for animation compatibility."""
    def __init__(self, vf, time_grid, hj_grid_min, hj_grid_dx, hj_grid_stp):
        self.vf = vf
        self.time_grid = time_grid
        self.hj_grid_min = hj_grid_min
        self.hj_grid_dx = hj_grid_dx
        self.hj_grid_stp = hj_grid_stp

    def __call__(self, pts):
        # pts: shape (M, 5) = [t, x, y, yaw, vel]
        t_pts = pts[:, 0]
        x_pts = pts[:, 1:]
        vf, tg = self.vf, self.time_grid
        NX = x_pts.shape[1]

        it = np.clip(
            np.round((t_pts - tg[0]) / (tg[1] - tg[0])).astype(int),
            0, vf.shape[0] - 1,
        )
        idxs = []
        for i in range(NX):
            if i == 2:
                wrapped = (x_pts[:, i] - self.hj_grid_min[i]) % (
                    self.hj_grid_min[i + 1] - self.hj_grid_min[i]
                    if False else (self.hj_grid_dx[i] * (self.hj_grid_stp[i] - 1))
                )
                ix = np.clip(np.round(wrapped / self.hj_grid_dx[i]).astype(int), 0, self.hj_grid_stp[i] - 1)
            else:
                ix = np.clip(
                    np.round((x_pts[:, i] - self.hj_grid_min[i]) / self.hj_grid_dx[i]).astype(int),
                    0, self.hj_grid_stp[i] - 1,
                )
            idxs.append(ix)
        return vf[it, idxs[0], idxs[1], idxs[2], idxs[3]]


if __name__ == "__main__":
    
    from math import pi 

    t0 = 0.0
    x0 = [3.0, 0.5, pi/2, 0.0]

    Controller().run(t0, x0)