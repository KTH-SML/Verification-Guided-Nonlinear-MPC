from functools import reduce
from math import pi

import shutil
import os
import time as tm

import numpy as np

from pyspect import *
from pyspect.impls.hj_reachability import TVHJImpl
from pyspect.systems.hj_reachability.bicycle4d import Bicycle4D

from obstacles import crossing_events


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

        self.T_PRED = cfg['verifier']['t_prediction']
        self.LOCAL_AXES = []
        for ax in self.AXES:
            if ax['name'] == 't':
                self.LOCAL_AXES.append(dict(name='t', bounds=[0.0, self.T_PRED], step=self.TIMESTEP))
            else:
                self.LOCAL_AXES.append(dict(ax))

        self.impl_local = TVHJImpl(dynamics.copy(), self.LOCAL_AXES)

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
        print(f"Computing Global Set (out1) for {self.HORIZON}s...")
        compute_start = tm.time()

        env = self.build_env()
        task_global = UNTIL(env['safe_region'], env['target_loc'])
        out1 = TLT.construct(task_global).realize(self.impl)

        compute_end = tm.time()
        print(f"Global Set computed in {compute_end - compute_start:.2f}s.")
        T_DELAY = self.T_PRED

        # Progress map for the MPC cost, computed once: minimum time to reach the goal
        # from every state (static obstacles only). Yaw is extended like the local sets.
        ttr = self.time_to_reach(out1)
        ttr = ttr.min(axis=(2, 3), keepdims=True) * np.ones_like(ttr)
        self.registry['progress.ttr'] = np.concatenate([ttr, ttr[:, :, 1:], ttr[:, :, 1:]], axis=2)
        print(f"[Verifier] Time-to-reach map published (max {ttr.max():.2f} s).")

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

            vf = np.concatenate([raw[:,:,:,:-1], raw, raw[:,:,:,1:]], axis=3)

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
        while not self.load_from_file and not self.slice_data and self.registry.get('time', 0.0) <= self.HORIZON - 1.0:

            current_time = float(self.registry.get('time', 0.0))
            
            if data_set_no == 0:
                t_plan_start = current_time
            else:
                t_plan_start = current_time + T_DELAY

            t_plan_end = min(t_plan_start + self.T_PRED, self.HORIZON)

            window_duration = t_plan_end - t_plan_start

            # Window starts at/after the horizon: nothing left to verify
            if window_duration < self.TIMESTEP:
                print(f"[Verifier] Stop: window [{t_plan_start:.2f} s -> {t_plan_end:.2f} s] is empty.")
                break

            # The window starts T_DELAY in the future, so the current state is
            # used as an approximation of the state at t_plan_start.
            state_now = self.registry.get('state', self.registry['x0'])

            print(f"\n" + "=" * 65)
            print(f"[Verifier] Start computing set {data_set_no}")
            print(f"  • Current simulation time: {current_time:.2f} s")
            print(f"  • Planned horizon:      [{t_plan_start:.2f} s -> {t_plan_end:.2f} s] (Duration: {window_duration:.2f} s)")
            print(f"=" * 65)
            loop_start = tm.time()

            moving_obj_local = self.build_moving_object_local(t_plan_start, min(self.T_PRED, t_plan_end - t_plan_start))

            safe_region_local = Inter(env['safe_region'], Compl(moving_obj_local))
            # safe_region_local = env['safe_region']

            # Safety set (used as a constraint by the MPC): avoid all obstacles during the
            # window and end in a state from which the goal is still reachable at all.
            # Global slice at t = 0 = largest time budget, so waiting is never "unsafe".
            target_local = self.get_local_terminal_target(out1, t_plan_end)
            task_local = UNTIL(safe_region_local, target_local)

            pde_start = tm.time()
            out2 = TLT.construct(task_local).realize(self.impl_local)
            pde_end = tm.time()

            raw = np.asarray(out2)
            vf = np.concatenate([raw, raw[:, :, :, 1:], raw[:, :, :, 1:]], axis=3)

            set_name = f'set_{data_set_no}'
            info = {'interval': (t_plan_start, t_plan_end), 'name': set_name}

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
            })
            self.registry['verifier.timeline'] = list(timeline)

            data_set_no += 1

            tm.sleep(0.05)
        
        # Slow loop
        while not self.load_from_file and self.slice_data:

            compute_start = tm.time()

            print(f"Computing Set {data_set_no}, at time {self.registry.get('time', 0)}")

            env = self.build_env()

            goal_x, goal_y = self.GOAL[0], self.GOAL[1]
            goal_w, goal_h = self.GOAL[2], self.GOAL[3]

            target_set = BoundedSet(x=(goal_x, goal_x + goal_w), y=(goal_y, goal_y + goal_h))

            task = UNTIL(env['safe_region'], target_set)

            info = {'interval': (0.0, self.HORIZON), 'name': 'foo'}

            # task = self.create_task()

            out1 = TLT.construct(task).realize(self.impl)

            raw = np.asarray(out1)
            vf = np.concatenate([raw, raw[:,:,:,1:], raw[:,:,:,1:]], axis=3)

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
        return tuple(int(np.argmin(np.abs(c - v)))
                     for c, v in zip(coords, (x_val, y_val, yaw_val, vel_val)))

    def local_value(self, out2, state):
        """Value of the local set at the start of its window (local t = 0) for the given state.
        <= 0: the state can avoid all obstacles during the window and still reach the goal."""
        return float(np.asarray(out2)[(0, *self.state_indices(state))])

    def get_local_terminal_target(self, out1_global, t_global: float):
        """Terminal target for the local problem: global set sliced at global time t_global."""
        t_target = min(t_global, self.HORIZON)

        time_grid = np.array(self.impl.timeline)
        idx_target = int(np.argmin(np.abs(time_grid - t_target)))

        v_terminal_4d = np.asarray(out1_global)[idx_target, ...]

        goal_x, goal_y = self.GOAL[0], self.GOAL[1]
        goal_w, goal_h = self.GOAL[2], self.GOAL[3]
        target_set = BoundedSet(x=(goal_x, goal_x + goal_w), y=(goal_y, goal_y + goal_h))
        sdf_goal = target_set(self.impl_local)

        v_combined = np.minimum(v_terminal_4d, sdf_goal)

        return Set(v_combined)

    def close(self):
        pass

    def build_env(self):

        map_box = BoundedSet(x=(self.X0 + 0.01, self.X0 + self.XW - 0.01),
                         y=(self.Y0 + 0.01, self.Y0 + self.YW - 0.01))

        obs_sets = []
        eps = 1e-3
        for (ox, oy, ow, oh) in self.OBSTACLES:
            obs_sets.append(Inter(
                map_box,
                BoundedSet(
                    x=(max(ox, self.X0 + eps), min(ox + ow, self.X0 + self.XW - eps)),
                    y=(max(oy, self.Y0 + eps), min(oy + oh, self.Y0 + self.YW - eps)),
                ),
            ))
        obstacle_union = reduce(Union, obs_sets)

        safe_region = Inter(map_box, Compl(obstacle_union))

        gx, gy, gw, gh = self.GOAL
        target_loc = BoundedSet(x=(gx, gx + gw), y=(gy, gy + gh))

        return dict(safe_region=safe_region, target_loc=target_loc)

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
        y_start = y0 + v * (0.0 - t0)
        y_end = y0 + v * (duration - t0)
        if min(y_start, y_end) > self.YW:
            return EMPTY
        if max(y_start, y_end) < self.Y0:
            return Compl(EMPTY)
        y_a = float(np.clip(y_start, self.Y0, self.YW))
        t_a = float(np.clip(t0 + (y_a - y0) / v, 0.0, duration))
        y_a = float(np.clip(y0 + v * (t_a - t0), self.Y0, self.YW))
        return HalfSpaceSet(normal=[+1, -v], offset=[y_a, t_a], axes=['y', 't'])

    def build_moving_object_local(self, t_start: float, duration: float):

        p = self.DYN_OBS
        _w = p['width']
        _x = p['center_x']

        # pyspect wants bounds strictly inside the grid; Ellipsis = up to the map edge
        y_bot, y_top = p['between_y']
        band_y = (... if y_bot <= self.Y0 else y_bot, ... if y_top >= self.YW else y_top)
        boundary = BoundedSet(x=(_x - _w / 2, _x + _w / 2), y=band_y)

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