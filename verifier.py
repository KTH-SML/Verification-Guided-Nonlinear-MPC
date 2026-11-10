from functools import reduce
from math import pi

import shutil
import os
import time as tm

import numpy as np
import jax.numpy as jnp
import hj_reachability as hj

from pyspect import *
from pyspect.impls.hj_reachability import TVHJImpl
from pyspect.systems.hj_reachability.bicycle4d import Bicycle4D

from obstacles import crossing_events
import jax, sys; print(sys.executable, jax.default_backend(), jax.devices())


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
        self.DYN_OBS = cfg['dynamic_obstacle']

        self.AXES = cfg['verifier']['axes']

        dynamics = dict(cls=Bicycle4D,
                        min_steer=-self.MAX_STEER, max_steer=self.MAX_STEER,
                        min_accel=-self.MAX_ACCEL, max_accel=self.MAX_ACCEL,
                        wheelbase=self.WHEELBASE)
        self.impl = TVHJImpl(dynamics.copy(), self.AXES)

        # Receding global window: self.impl (global problem) and self.impl_local (local
        # sets before cropping) share the window grid, moved by update_window.
        self.WINDOW_SIZE = tuple(cfg['verifier']['window_size'])
        self.WINDOW_BACK = cfg['verifier']['window_back']
        self.GLOBAL_UPDATE = cfg['verifier']['global_update']
        self.GLOBAL_TARGET_DEPTH = cfg['verifier']['global_target_depth']
        self.window = None
        self.global_version = -1
        self.global_t0 = None
        self.global_timeline = []

        self.T_PRED = cfg['verifier']['t_prediction']
        self.LOCAL_AXES = []
        for ax in self.AXES:
            if ax['name'] == 't':
                self.LOCAL_AXES.append(dict(name='t', bounds=[0.0, self.T_PRED], step=self.TIMESTEP))
            else:
                self.LOCAL_AXES.append(dict(ax))

        # Full-space local impl: only used to build the sets (cheap array ops), no PDE
        self.impl_local = TVHJImpl(dynamics.copy(), self.LOCAL_AXES)

        # Cropped local impl: the PDE is solved here, on a window that follows the vehicle
        self.T_DELAY = cfg['verifier'].get('t_delay', self.T_PRED)
        self.CROP_AXES = {name.lstrip('*') for name in cfg['verifier'].get('crop_axes', ())}
        self.CROP_MARGIN = cfg['verifier'].get('crop_margin', 3)
        self.setup_crop(dynamics.copy())

        if not self.load_from_file:
            # Remove existing directory if it exists, to ensure a clean save
            if os.path.exists(self.save_name):
                shutil.rmtree(self.save_name)
            os.mkdir(self.save_name)

    def run(self):

        data_set_no = 0
        # path = "/home/tokisaki_kurumi/Projects/svea/verification_guided_mpc/kaj/sets/data_1.npy"
        path = os.path.join(self.save_name, "data_0.npy")
        if self.load_from_file:
            print(f"Loading set from {path}...")
        else:
            print(f"Starting verification loop for {self.HORIZON}...")

        # Timeline records: list of dicts, one per verification set published.
        # Each entry: {set_idx, interval, compute_start, compute_end, loop_end}
        # compute_start / compute_end / loop_end are wall-clock times (absolute).
        timeline: list = []

        # First global window, placed on the initial state
        self.update_global(float(self.registry.get('time', 0.0)),
                           self.registry.get('state', self.registry['x0']))
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

            vf = np.concatenate([raw, raw, raw], axis=3)   # periodic yaw, see update_global

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
        last_plan_start = -np.inf
        while not self.load_from_file and not self.slice_data and self.registry.get('time', 0.0) <= self.MISSION_TIME - 1.0:

            current_time = float(self.registry.get('time', 0.0))

            def plan_window(now):
                # Every window lasts exactly T_PRED (the length of the local time axis):
                # near the end of the mission the window is moved earlier, not shortened.
                start = now if data_set_no == 0 else now + T_DELAY
                start = min(start, self.MISSION_TIME - self.T_PRED)
                return start, start + self.T_PRED

            # Receding global window: re-solved every GLOBAL_UPDATE seconds around the vehicle,
            # or earlier if the next local window would end past the global horizon (its
            # terminal target, the global set at t_plan_end, would not exist).
            t_plan_start, t_plan_end = plan_window(current_time)
            if (current_time - self.global_t0 >= self.GLOBAL_UPDATE - 1e-9
                    or t_plan_end > self.global_t0 + self.HORIZON + 1e-9):
                self.update_global(current_time, self.registry.get('state', self.registry['x0']))
                current_time = float(self.registry.get('time', current_time))
                t_plan_start, t_plan_end = plan_window(current_time)
                if t_plan_end > self.global_t0 + self.HORIZON + 1e-9:
                    print(f"[Verifier] WARNING: local window ends at {t_plan_end:.2f} s, past the global "
                          f"horizon {self.global_t0 + self.HORIZON:.2f} s: HJ_HORIZON is too short "
                          f"for global compute time + T_DELAY + T_PRED.")
            window_duration = self.T_PRED

            # The last window [HORIZON - T_PRED, HORIZON] is already published
            if t_plan_start <= last_plan_start + 1e-9:
                print(f"[Verifier] Stop: last window [{t_plan_start:.2f} s -> {t_plan_end:.2f} s] already computed.")
                break
            last_plan_start = t_plan_start

            # The window starts T_DELAY in the future, so the current state is
            # used as an approximation of the state at t_plan_start.
            state_now = self.registry.get('state', self.registry['x0'])

            print(f"\n" + "=" * 65)
            print(f"[Verifier] Start computing set {data_set_no}")
            print(f"  • Current simulation time: {current_time:.2f} s")
            print(f"  • Planned horizon:      [{t_plan_start:.2f} s -> {t_plan_end:.2f} s] (Duration: {window_duration:.2f} s)")
            print(f"=" * 65)
            loop_start = tm.time()

            moving_obj_local = self.build_moving_object_local(t_plan_start, self.T_PRED)

            safe_region_local = Inter(self.env['safe_region'], Compl(moving_obj_local))
            # safe_region_local = env['safe_region']

            # Safety set (used as a constraint by the MPC): avoid all obstacles during the
            # window and end in a state from which the goal is still reachable at all.
            # Global slice at t = 0 = largest time budget, so waiting is never "unsafe".
            target_local = self.get_local_terminal_target(self.out1, t_plan_end)

            # Spatial crop: the sets are built on the full local grid (cheap), then
            # restricted to the bounding box of the global zero sublevel set over
            # [t_plan_start, t_plan_end], where the PDE is solved.
            crop_bounds = self.update_crop(t_plan_start, t_plan_end)
            crop_pct = 0.0 if crop_bounds is None else float(
                100 * np.prod(self.impl_crop.grid.shape) / np.prod([len(c) for c in self.coords]))
            pde_start = tm.time()
            if crop_bounds is None:
                # No state reaches the global target from this window: the local set is empty
                print("[Verifier] Global zero sublevel set empty over the window: local set empty, PDE skipped.")
                out2 = np.ones((len(self.impl_local.timeline), *(len(c) for c in self.coords)), dtype=np.float32)
                pde_end = tm.time()
            else:
                safe_crop = self.crop(safe_region_local(self.impl_local))
                target_crop = self.crop(target_local(self.impl_local))
                task_local = UNTIL(Set(safe_crop), Set(target_crop))

                out2_crop = np.asarray(TLT.construct(task_local).realize(self.impl_crop))
                pde_end = tm.time()

                # Back to the full grid (controller and animation are unchanged). States outside
                # the box are outside the local set: they are marked unsafe.
                out2 = self.uncrop(out2_crop, fill=max(float(out2_crop.max()), 1.0))

            raw = np.asarray(out2)
            vf = np.concatenate([raw, raw, raw], axis=3)   # periodic yaw, see update_global

            set_name = f'set_{data_set_no}'
            info = {'interval': (t_plan_start, t_plan_end), 'name': set_name, 'crop': crop_bounds,
                    'window': self.window, 'global_version': self.global_version}

            self.registry[f'{set_name}.vf'] = vf
            self.registry['latest.info'] = info

            path = self.save_set(out2, data_set_no)

            # Diagnostic only: is the current state inside the safety set?
            v_local = self.local_value(out2, state_now)
            safe = v_local <= 0.0

            loop_end = tm.time()

            compute_duration = loop_end - loop_start
            pde_duration = pde_end - pde_start
            sim_time_after = float(self.registry.get('time', current_time))

            print(f"[Verifier] Computed set {data_set_no}")
            print(f"  • Total computation time (CPU): {compute_duration:.2f} s (PDE: {pde_duration:.2f} s)")
            print(f"  • Simulation time after computation: {sim_time_after:.2f} s")
            print(f"  • Remaining time in the set:     {t_plan_end - sim_time_after:.2f} s")
            print(f"  • Safe at current state:         {'YES' if safe else 'NO'} (V = {v_local:+.3f})")
            print(f"=" * 65 + "\n")

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
                'global_version': self.global_version,
                'crop':          crop_bounds,      # PDE box [(lo, hi)] per axis x, y, yaw, vel (None: empty)
                'crop_pct':      crop_pct,         # % of the window cells in the PDE box
            })
            self.registry['verifier.timeline'] = list(timeline)

            data_set_no += 1

            tm.sleep(1.0)
        
        # Slow loop
        while not self.load_from_file and self.slice_data:

            compute_start = tm.time()

            print(f"Computing Set {data_set_no}, at time {self.registry.get('time', 0)}")

            env = self.build_env()

            goal_x, goal_y = self.GOAL[0], self.GOAL[1]
            goal_w, goal_h = self.GOAL[2], self.GOAL[3]

            target_set = BoundedSet(x=(goal_x, goal_x + goal_w), y=(goal_y, goal_y + goal_h), vel=(..., 1e-3))   # stopped

            task = UNTIL(env['safe_region'], target_set)

            info = {'interval': (0.0, self.HORIZON), 'name': 'foo'}

            # task = self.create_task()

            out1 = TLT.construct(task).realize(self.impl)

            raw = np.asarray(out1)
            vf = np.concatenate([raw, raw, raw], axis=3)   # periodic yaw, see update_global

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

    def update_global(self, t_now, state):
        """
        Re-solve the global problem on a window placed around `state`.

        The window covers [t_now, t_now + HORIZON] in time and WINDOW_W x WINDOW_H in
        space; its target is the band on the right edge (or the final goal once inside).
        Sets self.window, self.env, self.target, self.out1, self.global_t0 and publishes:
            progress.ttr.<version>  time-to-reach map on the window grid (yaw extended)
            progress.info           {version, t0, window}  (written after the map: when
                                    the version changes, the matching map is there)
            verifier.global_timeline  one entry per global window
        """
        compute_start = tm.time()
        self.global_version += 1
        self.global_t0 = float(t_now)
        self.window = self.update_window(state)

        self.env = self.build_env()
        self.target = self.env['target_loc']
        task_global = UNTIL(self.env['safe_region'], self.target)
        self.out1 = np.asarray(TLT.construct(task_global).realize(self.impl))

        # Progress map for the MPC cost: minimum time to reach the current target
        # from every state of the window (static obstacles only), minimised over
        # yaw and vel, i.e. it only depends on the position (x, y).
        ttr = self.time_to_reach(self.out1)
        ttr = ttr.min(axis=(2, 3), keepdims=True) * np.ones_like(ttr)
        # One key per version, so the map is always read with its own window. The controller
        # uses the version its local set was derived from, which may lag one or two behind.
        # Yaw is periodic on the hj grid (N nodes at -pi + k*2pi/N, +pi excluded), so
        # three plain copies extend it to [-3pi, 3pi) like the local sets.
        self.registry[f'progress.ttr.{self.global_version}'] = np.concatenate([ttr, ttr, ttr], axis=2)
        self.registry.pop(f'progress.ttr.{self.global_version - 3}', None)

        info = dict(version=self.global_version, t0=self.global_t0, window=self.window)
        self.registry['progress.info'] = info

        # XY projection of the global set (min over yaw, vel), shape (NT, nx, ny):
        # only for the animation, which overlays its zero level on the local set
        os.makedirs(self.save_name, exist_ok=True)
        xy_path = os.path.join(self.save_name, f"global_{self.global_version}_xy.npy")
        np.save(xy_path, self.out1.min(axis=(3, 4)))

        compute_end = tm.time()
        self.global_timeline.append(dict(
            info,
            xy_path=xy_path,
            target=self.env['target_rect'],
            target_is_goal=self.env['target_is_goal'],
            compute_start=compute_start,
            compute_end=compute_end,
            sim_time_end=float(self.registry.get('time', t_now)),
        ))
        self.registry['verifier.global_timeline'] = list(self.global_timeline)

        (x_lo, x_hi), (y_lo, y_hi) = self.window
        print(f"\n[Verifier] Global window {self.global_version} at t = {self.global_t0:.2f} s: "
              f"x = [{x_lo:.2f}, {x_hi:.2f}], y = [{y_lo:.2f}, {y_hi:.2f}], "
              f"target = {'GOAL' if self.env['target_is_goal'] else 'band'}")
        print(f"  • Computed in {compute_end - compute_start:.2f} s "
              f"(sim time now {self.global_timeline[-1]['sim_time_end']:.2f} s), "
              f"TTR at vehicle {ttr[self.state_indices(state)]:.2f} s")

    def setup_crop(self, dynamics):
        """
        State for the crop of the local problem (see update_crop).

        The crop window is recomputed at every iteration, so its shape changes: one
        impl is kept per crop shape (self.crop_impls), and the jitted solver is only
        recompiled the first time a shape appears.
        """
        self.crop_dynamics = dynamics
        self.crop_impls = {}     # crop shape (n_x, n_y, n_yaw, n_vel) -> TVHJImpl
        self.impl_crop = None
        self.coords = [np.asarray(c) for c in self.impl.grid.coordinate_vectors]
        self.crop_idx = [np.arange(len(c)) for c in self.coords]

    def zero_level_box(self, t_start, t_end):
        """
        Bounding box of the global zero sublevel set over the local window [t_start, t_end]
        (absolute times), as global grid indices along each spatial axis.

        The local problem has a smaller safe set (moving obstacle) and its terminal target is
        the global set at t_end (plus the current global target), so at every local time tau
            V_local(tau, x) <= 0   =>   V_global(t_start + tau, x) <= 0
        i.e. the local set is contained in the union over [t_start, t_end] of the global zero
        sublevel sets. The PDE is then only solved on a box around that union, plus
        CROP_MARGIN cells per side (spatial derivative stencil, numerical error on the level).

        Returns (lo_idx, n) per spatial axis (lo_idx may be negative on a periodic axis:
        the box wraps around), or None if the union is empty.
        """
        time_grid = np.array(self.impl.timeline)
        t0 = float(np.clip(t_start - self.global_t0, 0.0, self.HORIZON))
        t1 = float(np.clip(t_end - self.global_t0, 0.0, self.HORIZON))
        # global slices covering [t0, t1]: from the node at or before t0 to the node at or after t1
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

    def update_crop(self, t_start, t_end):
        """
        Set the crop window for the local window [t_start, t_end] (see zero_level_box):
        selects (or creates) self.impl_crop with the window shape, moves its grid onto the
        window and sets self.crop_idx, the global indices of the window along each axis.

        Returns the window bounds [(lo, hi), ...] per spatial axis, or None if the global
        zero sublevel set is empty over the window (then the local set is empty too).
        """
        box = self.zero_level_box(t_start, t_end)
        if box is None:
            return None

        spatial_axes = [ax for ax in self.AXES if ax['name'] != 't']
        lo, hi, self.crop_idx = [], [], []
        for k, (i0, n) in enumerate(box):
            c = self.coords[k]
            N, dx = len(c), c[1] - c[0]
            if n == N:
                # axis not cropped: keep the global domain (periodic hi is excluded from the nodes)
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

        full = np.prod([len(c) for c in self.coords])
        print(f"[Verifier] Crop window (zero sublevel set over [{t_start:.2f}, {t_end:.2f}] s): "
              f"{shape} of {tuple(len(c) for c in self.coords)} cells "
              f"({100 * np.prod(shape) / full:.1f} %){' [new shape: compiling]' if new else ''}")
        return list(zip(lo, hi))

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

    def update_window(self, state):
        """
        Place the global window around `state`: WINDOW_BACK metres behind it along x
        (the vehicle travels towards +x), centred on it along y, kept inside the map.

        The window is snapped to a lattice anchored at (X0, Y0) with the window
        resolution, so the grid nodes of consecutive windows coincide. The x, y bounds
        of self.impl and self.impl_local are moved in place (same shape), yaw and vel
        are untouched, and self.coords is refreshed for the crop.

        Returns ((x_lo, x_hi), (y_lo, y_hi)).
        """
        def place(start, origin, end, n, step):
            # first node index of the window, clipped so that the window stays in [origin, end]
            i_max = max(0, int(np.floor((end - origin) / step + 1e-9)) - (n - 1))
            i0 = int(np.clip(round((start - origin) / step), 0, i_max))
            return origin + i0 * step, origin + (i0 + n - 1) * step

        nx, ny = self.impl.grid.shape[:2]
        W, H = self.WINDOW_SIZE
        dx, dy = W / (nx - 1), H / (ny - 1)   # exact (the jax grid spacings are float32)
        x_lo, x_hi = place(float(state[0]) - self.WINDOW_BACK, self.X0, self.X0 + self.XW, nx, dx)
        y_lo, y_hi = place(float(state[1]) - H / 2,            self.Y0, self.Y0 + self.YW, ny, dy)

        for impl in (self.impl, self.impl_local):
            lo = [float(v) for v in impl.grid.domain.lo]
            hi = [float(v) for v in impl.grid.domain.hi]
            lo[0], hi[0], lo[1], hi[1] = x_lo, x_hi, y_lo, y_hi
            self._set_grid(impl, lo, hi)

        self.coords = [np.asarray(c) for c in self.impl.grid.coordinate_vectors]
        return ((x_lo, x_hi), (y_lo, y_hi))

    def crop(self, arr):
        """Restrict an array on the full local grid to the current crop window."""
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

        out1 is a backward reachable set: at global time t_g the remaining budget
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

    def state_indices(self, state):
        """Nearest (ix, iy, iyaw, ivel) grid indices of a state [x, y, yaw, vel]."""
        x_val, y_val, yaw_val, vel_val = state[:4]

        # wrap yaw into [-pi, pi) to match the periodic grid
        yaw_val = (yaw_val + pi) % (2 * pi) - pi

        coords = [np.array(c) for c in self.impl.grid.coordinate_vectors]
        dist = [np.abs(c - v) for c, v in zip(coords, (x_val, y_val, yaw_val, vel_val))]
        dist[2] = np.minimum(dist[2], 2 * pi - dist[2])   # yaw is periodic: -pi is next to +pi
        return tuple(int(np.argmin(d)) for d in dist)

    def local_value(self, out2, state):
        """Value of the local set at the start of its window (local t = 0) for the given state.
        <= 0: the state can avoid all obstacles during the window and still reach the goal."""
        return float(np.asarray(out2)[(0, *self.state_indices(state))])

    def get_local_terminal_target(self, out1_global, t_global: float):
        """Terminal target for the local problem: global set sliced at global time t_global,
        united with the current global target (band on the window edge or final goal)."""
        # out1 covers [global_t0, global_t0 + HORIZON]; past its end, use the last slice
        t_target = float(np.clip(t_global - self.global_t0, 0.0, self.HORIZON))

        time_grid = np.array(self.impl.timeline)
        idx_target = int(np.argmin(np.abs(time_grid - t_target)))

        v_terminal_4d = np.asarray(out1_global)[idx_target, ...]

        sdf_goal = self.target(self.impl_local)

        v_combined = np.minimum(v_terminal_4d, sdf_goal)

        return Set(v_combined)

    def close(self):
        pass

    def build_env(self):
        """
        Static environment restricted to the current global window (self.window).

        safe_region: inside the window and outside the static obstacles that intersect it.
            Every window edge is unsafe: the real map edges are walls, the others are the
            limit of the map portion considered now.
        target_loc: the final GOAL if it lies inside the window, otherwise a band of depth
            GLOBAL_TARGET_DEPTH along the right edge of the window (direction of travel),
            outside the obstacles.
        """
        (x_lo, x_hi), (y_lo, y_hi) = self.window

        # pyspect wants set bounds inside the grid: stay slightly inside the window
        eps = 1e-2
        bx, by = (x_lo + eps, x_hi - eps), (y_lo + eps, y_hi - eps)
        window_box = BoundedSet(x=bx, y=by)

        # static obstacles clipped to the window (those outside it are ignored)
        obs_sets = []
        for (ox, oy, ow, oh) in self.OBSTACLES:
            cx = (max(ox, bx[0]), min(ox + ow, bx[1]))
            cy = (max(oy, by[0]), min(oy + oh, by[1]))
            if cx[0] < cx[1] and cy[0] < cy[1]:
                obs_sets.append(BoundedSet(x=cx, y=cy))
        obstacle_union = reduce(Union, obs_sets, EMPTY)

        safe_region = Inter(window_box, Compl(obstacle_union))

        gx, gy, gw, gh = self.GOAL
        target_is_goal = bx[0] <= gx and gx + gw <= bx[1] and by[0] <= gy and gy + gh <= by[1]
        if target_is_goal:
            # final goal: inside the box and stopped (vel <= 1e-3 keeps only the vel = 0 grid nodes)
            target_loc = BoundedSet(x=(gx, gx + gw), y=(gy, gy + gh), vel=(..., 1e-3))
            target_rect = tuple(self.GOAL)
        else:
            band_x = (max(x_hi - self.GLOBAL_TARGET_DEPTH, bx[0]), bx[1])
            target_loc = Inter(BoundedSet(x=band_x, y=by), safe_region)
            target_rect = (band_x[0], by[0], band_x[1] - band_x[0], by[1] - by[0])

        # target_rect / target_is_goal: only for logging and animation
        return dict(safe_region=safe_region, target_loc=target_loc,
                    target_rect=target_rect, target_is_goal=target_is_goal)

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

    def _above_line(self, y0: float, t0: float, v: float, duration: float):
        """
        Set {y >= y0 + v * (t - t0)} on the local grid t in [0, duration].

        pyspect requires the half-space offset to lie inside the axis bounds,
        so the anchor is slid *along* the line into the grid box instead of
        clamping t0 (which would shift the line in time). If the line misses
        the box entirely the set is either empty or everything.
        """
        y_min, y_max = self.window[1]   # y extent of the current window grid
        y_start = y0 + v * (0.0 - t0)
        y_end = y0 + v * (duration - t0)
        if min(y_start, y_end) > y_max:
            return EMPTY
        if max(y_start, y_end) < y_min:
            return Compl(EMPTY)
        y_a = float(np.clip(y_start, y_min, y_max))
        t_a = float(np.clip(t0 + (y_a - y0) / v, 0.0, duration))
        y_a = float(np.clip(y0 + v * (t_a - t0), y_min, y_max))
        return HalfSpaceSet(normal=[+1, -v], offset=[y_a, t_a], axes=['y', 't'])

    def build_moving_object_local(self, t_start: float, duration: float):

        p = self.DYN_OBS
        _w = p['width']
        _x = p['center_x']

        # The obstacle lives in map coordinates: clip its x extent and its band to the
        # current window (pyspect wants bounds inside the grid; Ellipsis = up to the edge).
        (x_lo, x_hi), (y_lo, y_hi) = self.window
        eps = 1e-2
        obs_x = (max(_x - _w / 2, x_lo + eps), min(_x + _w / 2, x_hi - eps))
        y_bot, y_top = p['between_y']
        if obs_x[0] >= obs_x[1] or y_top <= y_lo or y_bot >= y_hi:
            return EMPTY   # the obstacle never enters this window
        band_y = (... if y_bot <= y_lo else y_bot, ... if y_top >= y_hi else y_top)
        if band_y == (..., ...):
            boundary = BoundedSet(x=obs_x)   # band covers the whole window height: no y bound
        else:
            boundary = BoundedSet(x=obs_x, y=band_y)

        local_obj = EMPTY
        for i, v, y_lo0, y_hi0, t_enter, t_exit in crossing_events(p):
            # skip crossings that do not overlap this window
            if t_exit < t_start or t_enter > t_start + duration:
                continue
            t_loc = t_enter - t_start

            above_back = self._above_line(y_lo0, t_loc, v, duration)
            below_front = Compl(self._above_line(y_hi0, t_loc, v, duration))

            local_obj = Union(local_obj, reduce(Inter, [
                above_back,
                below_front,
                boundary,
            ]))

        return local_obj

    def save_set(self, data, number):
        os.makedirs(self.save_name, exist_ok=True)
        path = os.path.join(self.save_name, f"data_{number}.npy")
        np.save(path, np.asarray(data))
        return path