from functools import reduce
from math import pi

import csv
import shutil
import os
import threading
import time as tm

import numpy as np
import jax.numpy as jnp
import hj_reachability as hj

from pyspect import *
from pyspect.impls.hj_reachability import TVHJImpl
from pyspect.systems.hj_reachability.bicycle4d import Bicycle4D

import jax

from obstacles import crossing_events, footprint_extents
from log import log, rule


class ImplicitSet(SetBuilder):
    """
    Set given directly by its implicit function fn(impl) (<= 0 inside), with the same
    convention as the pyspect primitives (halfspace = signed offset, intersection = max,
    union = min). fn may return a broadcastable array (e.g. no vel axis): it is only
    expanded to the full grid here, so obstacles inflated per yaw node stay cheap.
    """

    def __init__(self, fn):
        self.fn = fn

    def __call__(self, impl, **m):
        return jnp.asarray(self.fn(impl)) + jnp.zeros(impl.shape)


@primitive(ALWAYS('_1'))
def Always_rci(_1: 'TLTLike') -> tuple[SetBuilder, APPROXDIR]:
    b1, a1 = _1._builder, _1._approx

    ao = APPROXDIR.UNDER
    return (
        AppliedSet('rci', b1),
        ao + a1 if ao * a1 == APPROXDIR.EXACT else 
        a1      if ao == a1 else
        APPROXDIR.INVALID,
    )

TLT.select(ContLTL | Always_rci)

class Verifier:

    save_name = "sets"
    results_dir = "results"

    # Columns of results/crop_<method>.csv, one row per short-horizon set
    STATS_FIELDS = [
        'method', 'set', 't_start', 't_end', 'long_version',
        'n_x', 'n_y', 'n_yaw', 'n_vel', 'cells', 'cells_pct',
        'empty', 'compiled', 't_box', 't_pde', 't_total',
        'sim_start', 'sim_end', 'late', 'v_state',
    ]

    load_from_file = False

    slice_time = [[0.0, 10.0], [5.0, 15.0], [10.0, 20.0], [15.0, 20.0]]

    slice_data = False

    def __init__(self, registry, state_queue, stop_event):

        # Runtime communication primitives
        self.registry = registry
        self.state_queue = state_queue
        self.stop_event = stop_event

        cfg = self.registry['config']

        self.TIMESTEP = cfg['verifier']['timestep']
        self.HORIZON = cfg['verifier']['horizon']
        self.MISSION_TIME = cfg['mission_time']

        self.WHEELBASE = cfg['wheelbase']
        
        self.MAX_VEL = cfg['verifier']['max_vel']
        self.MAX_STEER = cfg['verifier']['max_steer']
        self.MAX_ACCEL = cfg['verifier']['max_accel']

        self.GOAL = cfg['goal']

        self.X0 = cfg['X0']
        self.Y0 = cfg['Y0']
        self.XW = cfg['XW']
        self.YW = cfg['YW']

        self.OBSTACLES = cfg['obstacles']
        self.DYN_OBS = cfg['dynamic_obstacles']
        # Body of the vehicle (None length: point vehicle, obstacles not inflated)
        self.VEHICLE = cfg.get('vehicle')

        self.AXES = cfg['verifier']['axes']

        dynamics = dict(cls=Bicycle4D,
                        min_steer=-self.MAX_STEER, max_steer=self.MAX_STEER,
                        min_accel=-self.MAX_ACCEL, max_accel=self.MAX_ACCEL,
                        wheelbase=self.WHEELBASE)
        # self.impl_long: long-horizon PDE, solved in a background thread (see long_loop).
        # self.impl: grid of the long-horizon window in use by the short-horizon loop (no PDE).
        self.impl_long = TVHJImpl(dynamics.copy(), self.AXES)
        self.impl = TVHJImpl(dynamics.copy(), self.AXES)

        # Receding long-horizon window: self.impl and self.impl_short (short-horizon sets before
        # cropping) share the grid of the long-horizon window in use, moved by activate_long.
        self.WINDOW_SIZE = tuple(cfg['verifier']['window_size'])
        self.WINDOW_BACK = cfg['verifier']['window_back']
        self.LONG_UPDATE = cfg['verifier']['long_update']
        self.LONG_TARGET_DEPTH = cfg['verifier']['long_target_depth']
        self.LONG_TARGET_Y = cfg['verifier'].get('long_target_y')   # (y_lo, y_hi) or None
        self.window = None
        self.long_version = -1
        self.long_t0 = None
        self.long_timeline = []
        self.LONG_LEAD = cfg['verifier'].get('long_lead', 1.0)
        # Long-horizon sets solved by the background thread, not yet used: version -> solution
        self.long_ready = {}
        self.long_cond = threading.Condition()
        self.long_error = None

        self.T_PRED = cfg['verifier']['t_prediction']
        self.SHORT_AXES = []
        for ax in self.AXES:
            if ax['name'] == 't':
                self.SHORT_AXES.append(dict(name='t', bounds=[0.0, self.T_PRED], step=self.TIMESTEP))
            else:
                self.SHORT_AXES.append(dict(ax))

        # Full-space short-horizon impl: only used to build the sets (cheap array ops), no PDE
        self.impl_short = TVHJImpl(dynamics.copy(), self.SHORT_AXES)

        # Cropped short-horizon impl: the PDE is solved only on a box of the window (see crop_box)
        self.T_DELAY = cfg['verifier'].get('t_delay', self.T_PRED)
        self.SHORT_PERIOD = cfg['verifier'].get('short_period', 1.0)
        self.CROP_METHOD = cfg['verifier'].get('crop_method', 'zero_level')
        self.CROP_AXES = {name.lstrip('*') for name in cfg['verifier'].get('crop_axes', ())}
        self.CROP_MARGIN = cfg['verifier'].get('crop_margin', 3)
        self.setup_crop(dynamics.copy())

        if not self.load_from_file:
            # Remove existing directory if it exists, to ensure a clean save
            if os.path.exists(self.save_name):
                shutil.rmtree(self.save_name)
            os.mkdir(self.save_name)

        # Per-set statistics of the short-horizon PDE, one file per crop method (to compare methods)
        os.makedirs(self.results_dir, exist_ok=True)
        self.stats_path = os.path.join(self.results_dir, f"crop_{self.CROP_METHOD}.csv")
        self.stats = []
        with open(self.stats_path, 'w', newline='') as f:
            csv.writer(f).writerow(self.STATS_FIELDS)

        n_full = [len(c) for c in self.coords]
        axes = ', '.join(n for n in (ax['name'].lstrip('*') for ax in self.AXES) if n in self.CROP_AXES) or 'none'
        log("Verifier", f"JAX backend: {jax.default_backend()} ({', '.join(map(str, jax.devices()))})")
        log("Verifier", f"Window grid {'x'.join(map(str, n_full))} = {int(np.prod(n_full)):,} cells, "
                        f"dt = {self.TIMESTEP} s")
        log("Verifier", f"Long-horizon window {self.HORIZON} s, one every {self.LONG_UPDATE} s, "
                        f"computed {self.LONG_LEAD} s ahead (background thread)")
        log("Verifier", f"Short-horizon window {self.T_PRED} s, one every {self.SHORT_PERIOD} s, "
                        f"computed {self.T_DELAY} s ahead")
        log("Verifier", f"Crop method: {self.CROP_METHOD} (axes: {axes}, margin: {self.CROP_MARGIN} cells)"
                        + (f", fixed box {'x'.join(map(str, self.reach_n))}" if self.CROP_METHOD == 'reach' else ""))
        log("Verifier", f"Short-horizon PDE statistics -> {self.stats_path}")

    def run(self):

        data_set_no = 0
        # path = "/home/tokisaki_kurumi/Projects/svea/verification_guided_mpc/kaj/sets/data_1.npy"
        path = os.path.join(self.save_name, "data_0.npy")
        if self.load_from_file:
            log("Verifier", f"Loading set from {path}")

        # Timeline records: list of dicts, one per verification set published.
        # Each entry: {set_idx, interval, compute_start, compute_end, loop_end}
        # compute_start / compute_end / loop_end are wall-clock times (absolute).
        timeline: list = []

        # Schedule origin (simulation time): long-horizon set j covers [t_ref + j*T_U, ... + HORIZON],
        # short-horizon set k starts at t_ref + k*T_L.
        t_ref = float(self.registry.get('time', 0.0))

        # Long-horizon set 0 is solved here (the MPC waits for the first short-horizon set), the next ones in
        # a background thread, each LONG_LEAD seconds before its window starts.
        self.activate_long(self.solve_long(0, t_ref, self.registry.get('state', self.registry['x0'])))
        if not self.load_from_file and not self.slice_data:
            threading.Thread(target=self.long_loop, args=(t_ref,), daemon=True).start()
        T_DELAY = self.T_DELAY

        while self.load_from_file:

            # if self.registry.get('time', 0) >= 11.0:
            #     path = "/home/tokisaki_kurumi/Projects/svea/verification_guided_mpc/hj_sets/hj_sets_20260323_192206/sets/slice3.npy"
            #     raw = np.load(path, mmap_mode='r').copy()
            #     info = {'interval': (10.0, 20.0), 'name': 'foo'}
            #     print("slice 3")
            # elif self.registry.get('time', 0) >= 6.0:
            #     path = "/home/tokisaki_kurumi/Projects/svea/verification_guided_mpc/hj_sets/hj_sets_20260323_192206/sets/slice2.npy"
            #     raw = np.load(path, mmap_mode='r').copy()
            #     info = {'interval': (5.0, 15.0), 'name': 'foo'}
            #     print("slice 2")
            # else:
            #     path = "/home/tokisaki_kurumi/Projects/svea/verification_guided_mpc/hj_sets/hj_sets_20260323_192206/sets/slice1.npy"
            #     raw = np.load(path, mmap_mode='r').copy()
            #     info = {'interval': (0.0, 10.0), 'name': 'foo'}
            #     print("slice 1")

            compute_start = tm.time()

            path = "/home/tokisaki_kurumi/Projects/svea/verification_guided_mpc/kaj/sets/data_3.npy"
            raw = np.load(path, mmap_mode='r').copy()
            info = {'interval': (0.0, 20.0), 'name': 'foo'}

            vf = np.concatenate([raw, raw, raw], axis=3)   # periodic yaw, see solve_long

            compute_end = tm.time()

            self.registry['foo.vf'] = vf
            self.registry['latest.info'] = info

            # Record timeline entry (loop_end == compute_end: no sleep in file-load mode)
            timeline.append({
                'set_idx':       data_set_no,
                'interval':      list(info['interval']),
                'compute_start': compute_start,
                'compute_end':   compute_end,
                'loop_end':      compute_end,
            })
            self.registry['verifier.timeline'] = list(timeline)

            # File-load mode publishes once and exits (break prevents infinite re-load)
            break

        # Slow loop
        # Fixed schedule in simulation time: short-horizon window k starts at t_ref + k * T_L,
        # independently of the computation time. Its computation starts T_DELAY earlier
        # (set 0 starts at t_ref with no delay: the MPC waits for it).
        T_L = self.SHORT_PERIOD
        last_plan_start = -np.inf
        while not self.load_from_file and not self.slice_data and self.registry.get('time', 0.0) <= self.MISSION_TIME - 1.0:

            # Every window lasts exactly T_PRED (the length of the time axis of the short-horizon sets):
            # near the end of the mission the window is moved earlier, not shortened.
            t_plan_start = min(t_ref + data_set_no * T_L, self.MISSION_TIME - self.T_PRED)
            t_plan_end = t_plan_start + self.T_PRED
            t_tick = t_plan_start - (T_DELAY if data_set_no > 0 else 0.0)

            # The last window [MISSION_TIME - T_PRED, MISSION_TIME] is already published
            if t_plan_start <= last_plan_start + 1e-9:
                log("Verifier", f"Last window [{t_plan_start:.2f}, {t_plan_end:.2f}] s already computed: stop.",
                    float(self.registry.get('time', 0.0)))
                break
            last_plan_start = t_plan_start

            # Wait for the tick of this set (simulation time)
            while float(self.registry.get('time', 0.0)) < t_tick - 1e-6 and not self.stop_event.is_set():
                tm.sleep(0.005)
            if self.stop_event.is_set():
                break
            current_time = float(self.registry.get('time', 0.0))
            if current_time > t_tick + 1e-6:
                log("Verifier", f"WARNING: s{data_set_no} computation starts {current_time - t_tick:4.2f} s "
                                f"after its tick ({t_tick:.2f} s): previous computation longer than T_L",
                    current_time)

            # Long-horizon solution for this window: the latest one starting no later than the
            # window (version floor((t_plan_start - t_ref) / T_U)), solved in the background.
            self.select_long(int(np.floor((t_plan_start - t_ref) / self.LONG_UPDATE + 1e-9)),
                               t_plan_end)
            if self.stop_event.is_set():
                break
            if t_plan_end > self.long_t0 + self.HORIZON + 1e-9:
                log("Verifier", f"WARNING: short-horizon window ends at {t_plan_end:.2f} s, past the horizon of "
                                f"long-horizon set v{self.long_version} ({self.long_t0 + self.HORIZON:.2f} s)",
                    current_time)

            # The window starts T_DELAY in the future, so the current state is
            # used as an approximation of the state at t_plan_start.
            state_now = self.registry.get('state', self.registry['x0'])

            set_tag = f"Short s{data_set_no}"
            loop_start = tm.time()

            # Short-horizon task: ALWAYS(Phi_long(t) and not dynamic obstacles(t)) over the window.
            # Phi_long already excludes the static obstacles and contains the long-horizon target,
            # so staying in it keeps the goal reachable (liveness is inherited from the long-horizon set).
            phi_short = self.build_short_phi(t_plan_start)

            # Spatial crop: the sets are built on the full short-horizon grid (cheap), then
            # restricted to the bounding box of the long-horizon zero sublevel set over
            # [t_plan_start, t_plan_end], where the PDE is solved.
            box_start = tm.time()
            box = self.crop_box(t_plan_start, t_plan_end, state_now)
            box_time = tm.time() - box_start
            n_full = [len(c) for c in self.coords]
            if box is None:
                crop_bounds, compiled, shape = None, False, (0,) * len(n_full)
            else:
                crop_bounds, compiled = self.update_crop(box)
                shape = tuple(n for _, n in box)
            crop_pct = float(100 * np.prod(shape) / np.prod(n_full))

            log(set_tag, f"window [{t_plan_start:5.2f}, {t_plan_end:5.2f}] s, long-horizon set v{self.long_version}, "
                         f"box {'x'.join(map(str, shape))} ({crop_pct:5.1f} % of the window)"
                         + (", new shape: JIT compile" if compiled else ""), current_time)

            pde_start = tm.time()
            if box is None:
                # No state reaches the long-horizon target from this window: the short-horizon set is empty
                log(set_tag, "long-horizon zero sublevel set empty over the window: short-horizon set empty, PDE skipped",
                    current_time)
                out2 = np.ones((len(self.impl_short.timeline), *n_full), dtype=np.float32)
                pde_end = tm.time()
            else:
                phi_crop = self.crop(phi_short(self.impl_short))
                task_short = ALWAYS(Set(phi_crop))

                out2_crop = np.asarray(TLT.construct(task_short).realize(self.impl_crop))
                pde_end = tm.time()

                # Back to the full grid (controller and animation are unchanged). States outside
                # the box are outside the short-horizon set: they are marked unsafe.
                out2 = self.uncrop(out2_crop, fill=max(float(out2_crop.max()), 1.0))

            raw = np.asarray(out2)
            vf = np.concatenate([raw, raw, raw], axis=3)   # periodic yaw, see solve_long

            set_name = f'set_{data_set_no}'
            info = {'interval': (t_plan_start, t_plan_end), 'name': set_name, 'crop': crop_bounds,
                    'window': self.window, 'long_version': self.long_version}

            self.registry[f'{set_name}.vf'] = vf
            self.registry['latest.info'] = info

            path = self.save_set(out2, data_set_no)

            # Diagnostic only: is the current state inside the safety set?
            v_short = self.short_value(out2, state_now)
            safe = v_short <= 0.0

            loop_end = tm.time()

            compute_duration = loop_end - loop_start
            pde_duration = pde_end - pde_start
            sim_time_after = float(self.registry.get('time', current_time))

            log(set_tag, f"PDE {pde_duration:5.2f} s{' (incl. JIT)' if compiled else ''}, "
                         f"total {compute_duration:5.2f} s, valid for {t_plan_end - sim_time_after:4.2f} s more, "
                         f"V(state) = {v_short:+.3f} ({'inside' if safe else 'OUTSIDE'})", sim_time_after)
            late = max(0.0, sim_time_after - t_plan_start)
            if late > 1e-6:
                log(set_tag, f"WARNING: published {late:4.2f} s after its window start "
                             f"(computation longer than T_DELAY = {T_DELAY} s)", sim_time_after)

            self.record_stats(dict(
                method=self.CROP_METHOD, set=data_set_no, t_start=t_plan_start, t_end=t_plan_end,
                long_version=self.long_version,
                n_x=shape[0], n_y=shape[1], n_yaw=shape[2], n_vel=shape[3],
                cells=int(np.prod(shape)), cells_pct=crop_pct,
                empty=box is None, compiled=compiled,
                t_box=box_time, t_pde=pde_duration, t_total=compute_duration,
                sim_start=current_time, sim_end=sim_time_after, late=late, v_state=v_short,
            ))

            timeline.append({
                'set_idx':       data_set_no,
                'interval':      list(info['interval']),
                'compute_start': loop_start,
                'compute_end':   loop_end,
                'loop_end':      loop_end,
                'path':          path,
                'safe':          safe,
                'window':        self.window,
                'sim_start':     current_time,     # simulation time when the computation started
                'sim_end':       sim_time_after,   # simulation time when the set was published
                'long_version': self.long_version,
                'crop':          crop_bounds,      # PDE box [(lo, hi)] per axis x, y, yaw, vel (None: empty)
                'crop_pct':      crop_pct,         # % of the window cells in the PDE box
            })
            self.registry['verifier.timeline'] = list(timeline)

            data_set_no += 1

        if self.stats:
            self.print_summary()
        
        # Slow loop
        while not self.load_from_file and self.slice_data:

            compute_start = tm.time()

            log("Verifier", f"Computing set {data_set_no}", self.registry.get('time', 0.0))

            env = self.build_env(self.window)

            goal_x, goal_y = self.GOAL[0], self.GOAL[1]
            goal_w, goal_h = self.GOAL[2], self.GOAL[3]

            target_set = BoundedSet(x=(goal_x, goal_x + goal_w), y=(goal_y, goal_y + goal_h), vel=(..., 1e-3))   # stopped

            task = UNTIL(env['safe_region'], target_set)

            info = {'interval': (0.0, self.HORIZON), 'name': 'foo'}

            # task = self.create_task()

            out1 = TLT.construct(task).realize(self.impl)

            raw = np.asarray(out1)
            vf = np.concatenate([raw, raw, raw], axis=3)   # periodic yaw, see solve_long

            self.registry['foo.vf'] = vf
            self.registry['latest.info'] = info

            path = self.save_set(out1, data_set_no)

            compute_end = tm.time()

            sleep_dur = max(0.0, 5.0 - (compute_end - compute_start))
            tm.sleep(sleep_dur)

            loop_end = tm.time()

            # Record timeline entry
            timeline.append({
                'set_idx':       data_set_no,
                'interval':      list(info['interval']),
                'compute_start': compute_start,
                'compute_end':   compute_end,
                'loop_end':      loop_end,
                'path':          path,
            })
            self.registry['verifier.timeline'] = list(timeline)

            data_set_no += 1

    def solve_long(self, version, t_g, state):
        """
        Solve long-horizon problem `version` on a window placed around `state`, on self.impl_long
        (it does not touch the state used by the short-horizon loop: see activate_long).

        The window covers [t_g, t_g + HORIZON] in time and WINDOW_W x WINDOW_H in space;
        its target is the band on the right edge (or the final goal once inside).
        Publishes:
            progress.ttr.<version>    time-to-reach map on the window grid (yaw extended)
            progress.info             {version, t0, window}  (written after the map)
            verifier.long_timeline  one entry per long-horizon window
        Returns {version, t0, window, env, out1}.
        """
        compute_start = tm.time()
        # 'time' is written by the MPC from its first step on; before that the simulation is
        # at its start (0), not at t_g: long-horizon sets solved while the MPC waits for the first
        # short-horizon set (v0, and v1 if LONG_LEAD >= LONG_UPDATE) are computed at time 0
        sim_start = float(self.registry.get('time', 0.0))
        window = self.place_window(state)
        self._move_window(self.impl_long, window)

        env = self.build_env(window)
        task_long = UNTIL(env['safe_region'], env['target_loc'])
        out1 = np.asarray(TLT.construct(task_long).realize(self.impl_long))

        # Progress map for the MPC cost: minimum time to reach the current target
        # from every state of the window (static obstacles only), minimised over
        # yaw and vel, i.e. it only depends on the position (x, y).
        ttr = self.time_to_reach(out1)
        ttr = ttr.min(axis=(2, 3), keepdims=True) * np.ones_like(ttr)
        # One key per version, so the map is always read with its own window. The controller
        # uses the version its short-horizon set was derived from, which may lag one or two behind.
        # Yaw is periodic on the hj grid (N nodes at -pi + k*2pi/N, +pi excluded), so
        # three plain copies extend it to [-3pi, 3pi) like the short-horizon sets.
        self.registry[f'progress.ttr.{version}'] = np.concatenate([ttr, ttr, ttr], axis=2)
        self.registry.pop(f'progress.ttr.{version - 3}', None)

        info = dict(version=version, t0=float(t_g), window=window)
        self.registry['progress.info'] = info

        # XY projection of the long-horizon set (min over yaw, vel), shape (NT, nx, ny):
        # only for the animation, which overlays its zero level on the short-horizon set
        os.makedirs(self.save_name, exist_ok=True)
        xy_path = os.path.join(self.save_name, f"long_{version}_xy.npy")
        np.save(xy_path, out1.min(axis=(3, 4)))

        compute_end = tm.time()
        sim_end = float(self.registry.get('time', 0.0))
        self.long_timeline.append(dict(
            info,
            xy_path=xy_path,
            target=env['target_rect'],
            target_is_goal=env['target_is_goal'],
            compute_start=compute_start,
            compute_end=compute_end,
            sim_time_start=sim_start,    # simulation time when the computation started
            sim_time_end=sim_end,        # simulation time when it was published
        ))
        self.registry['verifier.long_timeline'] = list(self.long_timeline)

        (x_lo, x_hi), (y_lo, y_hi) = window
        coords = [np.asarray(c) for c in self.impl_long.grid.coordinate_vectors]
        log(f"Long v{version}",
            f"window [{t_g:5.2f}, {t_g + self.HORIZON:5.2f}] s, x [{x_lo:5.2f}, {x_hi:5.2f}], "
            f"y [{y_lo:5.2f}, {y_hi:5.2f}] -> {'GOAL' if env['target_is_goal'] else 'band'}, "
            f"solved in {compute_end - compute_start:4.2f} s, "
            f"time-to-reach from vehicle {ttr[self.state_indices(state, coords)]:4.2f} s", sim_end)

        return dict(version=version, t0=float(t_g), window=window, env=env, out1=out1)

    def activate_long(self, g):
        """Use long-horizon solution `g` (from solve_long) for the next short-horizon sets: moves the
        short-horizon grids to its window and sets self.window, env, target, out1, long_t0, version."""
        self.long_version = g['version']
        self.long_t0 = g['t0']
        self.window = g['window']
        self.env = g['env']
        self.target = self.env['target_loc']
        self.out1 = g['out1']
        for impl in (self.impl, self.impl_short):
            self._move_window(impl, self.window)
        self.coords = [np.asarray(c) for c in self.impl.grid.coordinate_vectors]

    def long_loop(self, t_ref):
        """
        Background thread: long-horizon set j >= 1 covers [t_g, t_g + HORIZON] with t_g = t_ref + j*T_U.
        Its computation starts at t_g - LONG_LEAD (simulation time), on the state at that
        moment; it must be ready by t_g - T_DELAY, when the first short-horizon set that uses it
        is computed. Long-horizon sets are only computed while a short-horizon window can still start after t_g.
        """
        j = 1
        try:
            # The schedule runs on simulation time, which starts with the first MPC step
            # (after long-horizon set 0 and the first short-horizon set). Wait for it, so that a long-horizon set whose
            # tick is at the start (t_g - LONG_LEAD <= t_ref) is computed from the start of
            # the run, not during the initialisation (where it would also slow down set 0).
            while self.registry.get('time') is None:
                if self.stop_event.is_set():
                    return
                tm.sleep(0.005)
            while not self.stop_event.is_set():
                t_g = t_ref + j * self.LONG_UPDATE
                if t_g > self.MISSION_TIME - self.T_PRED + 1e-9:
                    break
                t_start = t_g - self.LONG_LEAD
                while float(self.registry.get('time', 0.0)) < t_start - 1e-6:
                    if self.stop_event.is_set():
                        return
                    tm.sleep(0.005)
                now = float(self.registry.get('time', 0.0))
                if now > t_start + 1e-6:
                    log(f"Long v{j}", f"WARNING: computation starts {now - t_start:4.2f} s after its "
                                        f"tick ({t_start:.2f} s): previous long-horizon set longer than T_U", now)
                g = self.solve_long(j, t_g, self.registry.get('state', self.registry['x0']))
                with self.long_cond:
                    self.long_ready[j] = g
                    self.long_cond.notify_all()
                j += 1
        except Exception as e:
            log("Long", f"ERROR: {e!r}: long-horizon thread stopped")
            with self.long_cond:
                self.long_error = e
                self.long_cond.notify_all()
            raise

    def select_long(self, version, t_end):
        """
        Make long-horizon set `version` the one in use by the short-horizon loop. If it is not ready yet,
        keep the newest available one as long as its horizon covers the short-horizon window
        (ends after t_end); otherwise wait for it.
        """
        if version <= self.long_version:
            return
        with self.long_cond:
            def newest():
                ready = [v for v in self.long_ready if v <= version]
                return max(ready) if ready else None
            v = newest()
            if v != version:
                covered = (self.long_t0 if v is None else self.long_ready[v]['t0']) + self.HORIZON
                now = float(self.registry.get('time', 0.0))
                if t_end <= covered + 1e-9:
                    log("Verifier", f"WARNING: long-horizon set v{version} not ready, using "
                                    f"v{self.long_version if v is None else v}", now)
                else:
                    log("Verifier", f"WARNING: long-horizon set v{version} not ready, waiting for it", now)
                    while (version not in self.long_ready and self.long_error is None
                           and not self.stop_event.is_set()):
                        self.long_cond.wait(timeout=0.05)
                    if self.long_error is not None:
                        raise RuntimeError("long-horizon thread failed") from self.long_error
                    v = newest()
            if v is None:
                return
            g = self.long_ready[v]
            # drop the solutions that will never be used
            for old in [u for u in self.long_ready if u <= v]:
                del self.long_ready[old]
        self.activate_long(g)

    def record_stats(self, row):
        """Store the statistics of one short-horizon set and append them to results/crop_<method>.csv."""
        self.stats.append(row)
        with open(self.stats_path, 'a', newline='') as f:
            csv.DictWriter(f, fieldnames=self.STATS_FIELDS).writerow(
                {k: (f"{v:.4f}" if isinstance(v, float) else v) for k, v in row.items()})

    def print_summary(self):
        """Summary of the short-horizon PDE over the run (the numbers to compare between crop methods)."""
        rows = self.stats
        pct = np.array([r['cells_pct'] for r in rows])
        pde = np.array([r['t_pde'] for r in rows])
        tot = np.array([r['t_total'] for r in rows])
        jit = np.array([r['compiled'] for r in rows])
        empty = sum(r['empty'] for r in rows)
        outside = sum(r['v_state'] > 0 for r in rows)
        g_times = [g['compute_end'] - g['compute_start'] for g in self.long_timeline]

        def stats(a):
            return f"mean {a.mean():.3f}, min {a.min():.3f}, max {a.max():.3f}" if len(a) else "-"

        rule(f"Short-horizon PDE summary - crop method: {self.CROP_METHOD}")
        print(f"  short-horizon sets           : {len(rows)} ({empty} empty, state outside the set in {outside})")
        print(f"  box size [% cells]           : {stats(pct)}")
        print(f"  PDE time [s]                 : {stats(pde[~jit])}  (without JIT compilation)")
        print(f"  JIT compilations             : {int(jit.sum())} (PDE time with compilation: {stats(pde[jit])})")
        print(f"  short-horizon total time [s] : {stats(tot)}")
        late = np.array([r['late'] for r in rows])
        print(f"  published late               : {int((late > 1e-6).sum())} of {len(rows)} sets "
              f"(max {late.max() if len(late) else 0.0:.2f} s after window start)")
        print(f"  long-horizon solves          : {len(g_times)}, {stats(np.array(g_times))} s")
        g_late = [g for g in self.long_timeline[1:] if g['sim_time_end'] > g['t0'] - self.T_DELAY + 1e-6]
        print(f"  long-horizon sets late       : {len(g_late)} of {len(self.long_timeline) - 1} "
              f"(ready after t0 - T_DELAY, when the first short-horizon set using them is computed)")
        print(f"  per-set statistics           : {self.stats_path}")
        rule()

    # ------------------------------------------------------------------
    # Crop of the short-horizon PDE domain
    #
    # Each method returns a box as (first index, number of cells) per spatial axis
    # on the window grid (the first index may be negative on a periodic axis: the
    # box wraps around), or None if the short-horizon set is known to be empty. The rest
    # (grid, solver, crop / uncrop of the arrays) is shared: see update_crop.
    # ------------------------------------------------------------------

    CROP_METHODS = ('none', 'reach', 'zero_level')

    def setup_crop(self, dynamics):
        """
        State for the crop of the short-horizon problem (see update_crop).

        One impl is kept per box shape (self.crop_impls): the jitted solver is compiled
        the first time a shape appears and reused afterwards. 'reach' always has the same
        shape (one compilation), 'zero_level' keeps its size and only grows (see _keep_size).
        """
        if self.CROP_METHOD not in self.CROP_METHODS:
            raise ValueError(f"crop_method must be one of {self.CROP_METHODS}, got {self.CROP_METHOD!r}")
        self.crop_dynamics = dynamics
        self.crop_impls = {}     # box shape (n_x, n_y, n_yaw, n_vel) -> TVHJImpl
        self.impl_crop = None
        self.crop_shape = None   # 'zero_level': box size kept between iterations (see _keep_size)
        self.coords = [np.asarray(c) for c in self.impl.grid.coordinate_vectors]
        self.crop_idx = [np.arange(len(c)) for c in self.coords]

        # 'reach': fixed number of cells per axis, from the worst-case motion in
        # T = T_DELAY + T_PRED (independent of the state):
        #   x, y: vmax * T     yaw: vmax * tan(steer_max) / L * T     vel: accel_max * T
        T = self.T_DELAY + self.T_PRED
        vmax = max(abs(b) for ax in self.AXES if ax['name'] == 'vel' for b in ax['bounds'])
        reach = {
            'x':   vmax * T,
            'y':   vmax * T,
            'yaw': vmax * np.tan(self.MAX_STEER) / self.WHEELBASE * T,
            'vel': self.MAX_ACCEL * T,
        }
        self.reach_n = []
        for k, ax in enumerate(ax for ax in self.AXES if ax['name'] != 't'):
            name, N = ax['name'].lstrip('*'), len(self.coords[k])
            dx = self.coords[k][1] - self.coords[k][0]
            n = N
            if name in self.CROP_AXES:
                n = min(N, 2 * int(np.ceil(reach[name] / dx)) + 1 + 2 * self.CROP_MARGIN)
            self.reach_n.append(n)

    def crop_box(self, t_start, t_end, state):
        """Box of the short-horizon PDE for the window [t_start, t_end], with the configured method."""
        if self.CROP_METHOD == 'reach':
            return self._box_reach(state)
        if self.CROP_METHOD == 'zero_level':
            box = self._box_zero_level(t_start, t_end)
            return None if box is None else self._keep_size(box)
        return [(0, len(c)) for c in self.coords]          # 'none': whole window

    def _keep_size(self, box):
        """
        'zero_level' with a size that never shrinks: the first box sets the size; later
        boxes keep it along every axis where the needed box fits, and grow only the axes
        where it does not (size = max(previous, needed)). The box of that size is centred
        on the needed one and shifted to stay inside the window, so it always contains it.
        The shape changes (and the solver is JIT-compiled) only when an axis grows.
        """
        if self.crop_shape is None:
            self.crop_shape = [n for _, n in box]
            return box
        out = []
        for k, ((j0, m), n_prev) in enumerate(zip(box, self.crop_shape)):
            N = len(self.coords[k])
            n = min(N, max(m, n_prev))
            if n == N:
                out.append((0, N))
            elif self.impl.axis_is_periodic(k + 1):
                out.append((j0 - (n - m) // 2, n))      # may wrap around (see update_crop)
            else:
                out.append((int(np.clip(j0 - (n - m) // 2, 0, N - n)), n))
        self.crop_shape = [n for _, n in out]
        return out

    def _box_reach(self, state):
        """
        'reach': box of fixed size (self.reach_n) centred on the current state: the
        states the vehicle can reach in T_DELAY + T_PRED with worst-case speeds.
        Near the window edge the box is shifted inside, not shrunk.
        """
        center = self.state_indices(state)
        box = []
        for k, n in enumerate(self.reach_n):
            N = len(self.coords[k])
            if n == N:
                box.append((0, N))
            elif self.impl.axis_is_periodic(k + 1):
                box.append((center[k] - n // 2, n))
            else:
                box.append((int(np.clip(center[k] - n // 2, 0, N - n)), n))
        return box

    def _box_zero_level(self, t_start, t_end):
        """
        'zero_level': bounding box of the long-horizon zero sublevel set over the short-horizon
        window [t_start, t_end] (absolute times).

        The short-horizon problem has a smaller safe set (moving obstacle) and its terminal target is
        the long-horizon set at t_end (plus the current long-horizon target), so at every time tau of the window
            V_short(tau, x) <= 0   =>   V_long(t_start + tau, x) <= 0
        i.e. the short-horizon set is contained in the union over [t_start, t_end] of the long-horizon zero
        sublevel sets. The PDE is then only solved on a box around that union, plus
        CROP_MARGIN cells per side (spatial derivative stencil, numerical error on the level).

        Returns None if the union is empty (then the short-horizon set is empty too).
        """
        time_grid = np.array(self.impl.timeline)
        t0 = float(np.clip(t_start - self.long_t0, 0.0, self.HORIZON))
        t1 = float(np.clip(t_end - self.long_t0, 0.0, self.HORIZON))
        # long-horizon slices covering [t0, t1]: from the node at or before t0 to the node at or after t1
        i0 = max(0, int(np.searchsorted(time_grid, t0 + 1e-6, side='right')) - 1)
        i1 = min(len(time_grid) - 1, int(np.searchsorted(time_grid, t1 - 1e-6, side='left')))

        inside = (np.asarray(self.out1)[i0:i1 + 1] <= 0.0).any(axis=0)
        if not inside.any():
            return None

        m = self.CROP_MARGIN
        spatial_axes = [ax for ax in self.AXES if ax['name'] != 't']
        box = []
        for k, ax in enumerate(spatial_axes):
            N = inside.shape[k]
            proj = inside.any(axis=tuple(a for a in range(inside.ndim) if a != k))
            nz = np.flatnonzero(proj)
            if ax['name'].lstrip('*') not in self.CROP_AXES:
                box.append((0, N))
            elif self.impl.axis_is_periodic(k + 1):
                # smallest arc covering the set: complement of the largest gap between occupied cells
                gaps = np.diff(np.r_[nz, nz[0] + N])
                j = int(np.argmax(gaps))
                start, length = int(nz[(j + 1) % len(nz)]), int(N - (gaps[j] - 1))
                n = min(N, length + 2 * m)
                box.append((0, N) if n == N else (start - m, n))
            else:
                lo, hi = max(0, int(nz[0]) - m), min(N - 1, int(nz[-1]) + m)
                box.append((lo, hi - lo + 1))
        return box

    def update_crop(self, box):
        """
        Prepare the short-horizon PDE on `box` (see crop_box): select (or create) self.impl_crop
        with the box shape, move its grid onto the box and set self.crop_idx, the window
        indices of the box along each axis.

        Returns (bounds, compiled): the box bounds [(lo, hi), ...] per spatial axis, and
        whether a new solver had to be created for this shape (its first solve includes
        the JIT compilation).
        """
        spatial_axes = [ax for ax in self.AXES if ax['name'] != 't']
        lo, hi, self.crop_idx = [], [], []
        for k, (i0, n) in enumerate(box):
            c = self.coords[k]
            N, dx = len(c), c[1] - c[0]
            if n == N:
                # axis not cropped: keep the long-horizon domain (periodic hi is excluded from the nodes)
                idx = np.arange(N)
                lo.append(float(self.impl.grid.domain.lo[k]))
                hi.append(float(self.impl.grid.domain.hi[k]))
            elif self.impl.axis_is_periodic(k + 1):
                # periodic axis: the window may wrap around, coordinates continue past +-pi
                idx = (i0 + np.arange(n)) % N
                lo.append(float(c[0] + i0 * dx))
                hi.append(float(c[0] + (i0 + n - 1) * dx))
            else:
                idx = np.arange(i0, i0 + n)
                lo.append(float(c[i0]))
                hi.append(float(c[i0 + n - 1]))
            self.crop_idx.append(idx)

        shape = tuple(n for _, n in box)
        impl = self.crop_impls.get(shape)
        new = impl is None
        if new:
            specs = [dict(name='t', bounds=[0.0, self.T_PRED], step=self.TIMESTEP)]
            for k, ax in enumerate(spatial_axes):
                spec = dict(ax, points=shape[k], bounds=[lo[k], hi[k]])
                if shape[k] < len(self.coords[k]):
                    spec['name'] = ax['name'].lstrip('*')   # a cut periodic axis is no longer periodic
                specs.append(spec)
            impl = TVHJImpl(self.crop_dynamics.copy(), specs)
            self.crop_impls[shape] = impl
        self._set_grid(impl, lo, hi)
        self.impl_crop = impl
        return list(zip(lo, hi)), new

    @staticmethod
    def _set_grid(impl, lo, hi):
        """
        Move the spatial grid of `impl` to the box [lo, hi], keeping its shape and
        boundary conditions (same shape -> the jitted solver is not recompiled).
        """
        impl.grid = hj.Grid.from_lattice_parameters_and_boundary_conditions(
            hj.sets.Box(jnp.array(lo), jnp.array(hi)),
            tuple(impl.grid.shape),
            boundary_conditions=impl.grid.boundary_conditions,
        )
        impl._dx = impl.grid.spacings
        impl._min_bounds = (impl._min_bounds[0], *lo)
        impl._max_bounds = (impl._max_bounds[0], *hi)

    def place_window(self, state):
        """
        Long-horizon window around `state`: WINDOW_BACK metres behind it along x (the vehicle
        travels towards +x), centred on it along y, kept inside the map.

        The window is snapped to a lattice anchored at (X0, Y0) with the window
        resolution, so the grid nodes of consecutive windows coincide.

        Returns ((x_lo, x_hi), (y_lo, y_hi)).
        """
        def place(start, origin, end, n, step):
            # first node index of the window, clipped so that the window stays in [origin, end]
            i_max = max(0, int(np.floor((end - origin) / step + 1e-9)) - (n - 1))
            i0 = int(np.clip(round((start - origin) / step), 0, i_max))
            return origin + i0 * step, origin + (i0 + n - 1) * step

        nx, ny = self.impl_long.grid.shape[:2]
        W, H = self.WINDOW_SIZE
        dx, dy = W / (nx - 1), H / (ny - 1)   # exact (the jax grid spacings are float32)
        x_lo, x_hi = place(float(state[0]) - self.WINDOW_BACK, self.X0, self.X0 + self.XW, nx, dx)
        y_lo, y_hi = place(float(state[1]) - H / 2,            self.Y0, self.Y0 + self.YW, ny, dy)
        return ((x_lo, x_hi), (y_lo, y_hi))

    def _move_window(self, impl, window):
        """Move the x, y bounds of `impl` to `window` in place (same shape: no recompilation)."""
        (x_lo, x_hi), (y_lo, y_hi) = window
        lo = [float(v) for v in impl.grid.domain.lo]
        hi = [float(v) for v in impl.grid.domain.hi]
        lo[0], hi[0], lo[1], hi[1] = x_lo, x_hi, y_lo, y_hi
        self._set_grid(impl, lo, hi)

    def crop(self, arr):
        """Restrict an array on the full short-horizon grid to the current crop window."""
        arr = np.asarray(arr)
        return arr[(slice(None),) * (arr.ndim - 4) + np.ix_(*self.crop_idx)]

    def uncrop(self, arr_crop, fill):
        """Put a cropped array back on the full grid; outside the window the value is `fill`."""
        arr_crop = np.asarray(arr_crop)
        full = np.full((*arr_crop.shape[:-4], *(len(c) for c in self.coords)), fill, dtype=arr_crop.dtype)
        full[(slice(None),) * (arr_crop.ndim - 4) + np.ix_(*self.crop_idx)] = arr_crop
        return full

    def time_to_reach(self, out1):
        """
        Minimum time to reach the goal from every grid state, shape (x, y, yaw, vel).

        out1 is a backward reachable set: at time t_g of the long-horizon set the remaining budget
        is HORIZON - t_g, and the sets grow backward in time. The last time index
        where a state is inside the set gives its minimum time-to-go (same idea as
        the old crop_to_t_max, applied to the whole grid at once).
        States that never reach the goal (e.g. inside obstacles) get the full horizon.
        """
        vals = np.asarray(out1)
        time_grid = np.array(self.impl.timeline)

        inside = vals <= 0.0
        last_idx = vals.shape[0] - 1 - np.argmax(inside[::-1], axis=0)

        ttr = time_grid[-1] - time_grid[last_idx]
        ttr[~inside.any(axis=0)] = time_grid[-1] - time_grid[0]
        return ttr

    def state_indices(self, state, coords=None):
        """Nearest (ix, iy, iyaw, ivel) grid indices of a state [x, y, yaw, vel]
        (default grid: the long-horizon window in use by the short-horizon loop)."""
        x_val, y_val, yaw_val, vel_val = state[:4]

        # wrap yaw into [-pi, pi) to match the periodic grid
        yaw_val = (yaw_val + pi) % (2 * pi) - pi

        if coords is None:
            coords = [np.array(c) for c in self.impl.grid.coordinate_vectors]
        dist = [np.abs(c - v) for c, v in zip(coords, (x_val, y_val, yaw_val, vel_val))]
        dist[2] = np.minimum(dist[2], 2 * pi - dist[2])   # yaw is periodic: -pi is next to +pi
        return tuple(int(np.argmin(d)) for d in dist)

    def short_value(self, out2, state):
        """Value of the short-horizon set at the start of its window (t = 0 of the window) for the given state.
        <= 0: the state can avoid all obstacles during the window and still reach the goal."""
        return float(np.asarray(out2)[(0, *self.state_indices(state))])

    def get_long_tube_short(self, out1_long, t_start: float):
        """Long-horizon set over the short-horizon window [t_start, t_start + T_PRED], on the time axis of the window:
        index i <- nearest long-horizon slice at t_start + t_i (clipped to the long horizon).
        Shape (nt_short, x, y, yaw, vel)."""
        t_glob = np.asarray(self.impl.timeline)
        t_loc = np.clip(t_start - self.long_t0 + np.asarray(self.impl_short.timeline), 0.0, self.HORIZON)
        idx = np.abs(t_glob[None, :] - t_loc[:, None]).argmin(axis=1)
        return np.asarray(out1_long)[idx]

    def build_short_phi(self, t_start: float):
        """Phi_short(t) = Phi_long(t) minus the moving obstacles, over [t_start, t_start + T_PRED]."""
        tube = self.get_long_tube_short(self.out1, t_start)
        moving = self.build_moving_object_short(t_start, self.T_PRED)
        return Inter(ImplicitSet(lambda impl: tube), Compl(moving))

    def get_short_terminal_target(self, out1_long, t_sim: float):
        """Terminal target for the short-horizon problem: long-horizon set sliced at simulation time t_sim,
        united with the current long-horizon target (band on the window edge or final goal)."""
        # out1 covers [long_t0, long_t0 + HORIZON]; past its end, use the last slice
        t_target = float(np.clip(t_sim - self.long_t0, 0.0, self.HORIZON))

        time_grid = np.array(self.impl.timeline)
        idx_target = int(np.argmin(np.abs(time_grid - t_target)))

        v_terminal_4d = np.asarray(out1_long)[idx_target, ...]

        sdf_goal = self.target(self.impl_short)

        v_combined = np.minimum(v_terminal_4d, sdf_goal)

        return Set(v_combined)

    def close(self):
        pass

    def build_env(self, window):
        """
        Static environment restricted to the long-horizon window `window`.

        safe_region: inside the window and outside the static obstacles that intersect it.
            Every window edge is unsafe: the real map edges are walls, the others are the
            limit of the map portion considered now.
        target_loc: the final GOAL if it lies inside the window, otherwise a band of depth
            LONG_TARGET_DEPTH along the right edge of the window (direction of travel),
            outside the obstacles, restricted to LONG_TARGET_Y (e.g. the right lane). If a
            static obstacle covers that band, the band uses the whole window height.
        """
        (x_lo, x_hi), (y_lo, y_hi) = window

        # pyspect wants set bounds inside the grid: stay slightly inside the window
        eps = 1e-2
        bx, by = (x_lo + eps, x_hi - eps), (y_lo + eps, y_hi - eps)
        window_box = BoundedSet(x=bx, y=by)

        # static obstacles inflated by the vehicle body (per yaw node) and clipped to
        # the window (those that do not reach it are ignored)
        obstacle_union = ImplicitSet(lambda impl: self._static_obstacles_value(impl, bx, by))

        safe_region = Inter(window_box, Compl(obstacle_union))

        gx, gy, gw, gh = self.GOAL
        target_is_goal = bx[0] <= gx and gx + gw <= bx[1] and by[0] <= gy and gy + gh <= by[1]
        if target_is_goal:
            # final goal: inside the box and stopped (vel <= 1e-3 keeps only the vel = 0 grid nodes)
            target_loc = BoundedSet(x=(gx, gx + gw), y=(gy, gy + gh), vel=(..., 1e-3))
            target_rect = tuple(self.GOAL)
        else:
            band_x = (max(x_hi - self.LONG_TARGET_DEPTH, bx[0]), bx[1])
            band_y = by
            if self.LONG_TARGET_Y is not None:
                lane = (max(self.LONG_TARGET_Y[0], by[0]), min(self.LONG_TARGET_Y[1], by[1]))
                # a blocked lane band (e.g. broken-down car at the window edge) would leave the
                # long-horizon problem without target: then use the whole height
                if lane[1] > lane[0] and not self._band_blocked(band_x, lane):
                    band_y = lane
            target_loc = Inter(BoundedSet(x=band_x, y=band_y), safe_region)
            target_rect = (band_x[0], band_y[0], band_x[1] - band_x[0], band_y[1] - band_y[0])

        # target_rect / target_is_goal: only for logging and animation
        return dict(safe_region=safe_region, target_loc=target_loc,
                    target_rect=target_rect, target_is_goal=target_is_goal)

    def _band_blocked(self, band_x, band_y, min_free=1.0):
        """True if static obstacles, enlarged by the vehicle length in x (the body around the
        rear axle) and half its width in y, leave less than `min_free` metres of the target
        band free in y."""
        veh = self.VEHICLE or {}
        mx = veh.get('length', 0.0)
        my = 0.5 * veh.get('width', 0.0)
        cuts = sorted((max(oy - my, band_y[0]), min(oy + oh + my, band_y[1]))
                      for (ox, oy, ow, oh) in self.OBSTACLES
                      if ox - mx < band_x[1] and band_x[0] < ox + ow + mx)
        free, y = 0.0, band_y[0]
        for lo, hi in cuts:
            if hi <= lo:
                continue
            free = max(free, lo - y)
            y = max(y, hi)
        free = max(free, band_y[1] - y)
        return free < min_free

    def build_static_object(
            self,
            width=0.4,
            length=0.5,
            center_x=2.0,
            between_y=(3.0, 3.5),
        ):

        _w = width # static obstacle width
        _x = center_x # static obstacle center x
        _l = length # static obstacle length
        _y0 = between_y[0] # static obstacle initial y position
        return BoundedSet(x=(_x-_w/2, _x+_w/2), y=(_y0-_l/2, _y0+_l/2))

    def _footprint(self, impl):
        """
        Extents (fx_min, fx_max, fy_min, fy_max) of the vehicle body around the rear axle,
        one per yaw node of `impl` (each node covers +- half a yaw cell), shaped like
        impl.axis_vec('yaw'). Obstacle O becomes [o_lo - f_max, o_hi - f_min] on each axis.
        Zeros for a point vehicle.
        """
        yaw = np.asarray(impl.axis_vec('yaw'))
        half = float(impl.axis_step('yaw')) / 2
        return footprint_extents(self.VEHICLE, yaw - half, yaw + half)

    def _static_obstacles_value(self, impl, bx, by):
        """Implicit function of the union of the static obstacles inflated by the vehicle
        body (per yaw node), clipped to the box bx x by. Shape (1, nx, ny, nyaw, 1)."""
        x, y = np.asarray(impl.axis_vec('x')), np.asarray(impl.axis_vec('y'))
        fx_min, fx_max, fy_min, fy_max = self._footprint(impl)
        value = np.inf
        for (ox, oy, ow, oh) in self.OBSTACLES:
            x_lo = np.maximum(ox - fx_max, bx[0])
            x_hi = np.minimum(ox + ow - fx_min, bx[1])
            y_lo = np.maximum(oy - fy_max, by[0])
            y_hi = np.minimum(oy + oh - fy_min, by[1])
            if not np.any((x_lo < x_hi) & (y_lo < y_hi)):
                continue   # does not reach the window for any heading
            value = np.minimum(value, reduce(np.maximum, [x_lo - x, x - x_hi, y_lo - y, y - y_hi]))
        return value

    def build_moving_object_short(self, t_start: float, duration: float):
        """
        Union of the moving obstacles over the short-horizon window [t_start, t_start + duration]
        (time 0 of the window = t_start), inflated by the vehicle body per yaw node.
        Only the crossings that overlap the window are included.
        """
        events = []
        for p in self.DYN_OBS:
            for i, v, y_lo0, y_hi0, t_enter, t_exit in crossing_events(p):
                if t_exit < t_start or t_enter > t_start + duration:
                    continue
                events.append((p, v, y_lo0, y_hi0, t_enter - t_start))
        if not events:
            return EMPTY
        window = self.window
        return ImplicitSet(lambda impl: self._moving_obstacles_value(impl, events, window))

    def _moving_obstacles_value(self, impl, events, window):
        """
        Implicit function of the moving obstacles in (t, x, y, yaw). For each crossing:
            x  in [cx - w/2 - fx_max, cx + w/2 - fx_min]        (clipped to the window)
            y  in [y_bot - fy_max, y_top - fy_min]              (band, open past the window)
            y >= y_lo0 - fy_max + v (t - t_loc)                  (behind its front ...)
            y <= y_hi0 - fy_min + v (t - t_loc)                  (... and ahead of its back edge)
        Shape (nt, nx, ny, nyaw, 1).
        """
        (wx_lo, wx_hi), (wy_lo, wy_hi) = window
        eps = 1e-2
        t = np.asarray(impl.axis_vec('t'))
        x, y = np.asarray(impl.axis_vec('x')), np.asarray(impl.axis_vec('y'))
        fx_min, fx_max, fy_min, fy_max = self._footprint(impl)
        value = np.inf
        for p, v, y_lo0, y_hi0, t_loc in events:
            cx, w = p['center_x'], p['width']
            y_bot, y_top = p['between_y']
            x_lo = np.maximum(cx - w / 2 - fx_max, wx_lo + eps)
            x_hi = np.minimum(cx + w / 2 - fx_min, wx_hi - eps)
            band_lo = np.where(y_bot - fy_max <= wy_lo, -np.inf, y_bot - fy_max)
            band_hi = np.where(y_top - fy_min >= wy_hi, np.inf, y_top - fy_min)
            back = y_lo0 - fy_max + v * (t - t_loc)
            front = y_hi0 - fy_min + v * (t - t_loc)
            value = np.minimum(value, reduce(np.maximum, [
                x_lo - x, x - x_hi,
                band_lo - y, y - band_hi,
                back - y, y - front,
            ]))
        return value

    def save_set(self, data, number):
        os.makedirs(self.save_name, exist_ok=True)
        path = os.path.join(self.save_name, f"data_{number}.npy")
        np.save(path, np.asarray(data))
        return path