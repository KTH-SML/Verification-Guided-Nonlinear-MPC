from functools import reduce
from math import pi

import shutil
import os
import time as tm

import numpy as np

from pyspect import *
from pyspect.impls.hj_reachability import TVHJImpl
from pyspect.systems.hj_reachability.bicycle4d import Bicycle4D

from btree import NodeStatus, ActionNode, Sequence, Fallback

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

    slice_data = True

    slice_time = [[0.0, 4.0], [3.0, 7.0], [6.0, 10.0], [9.0, 13.0], [12.0, 16.0], [15.0, 19.0], [18.0, 20.0]]

    timeline: list = []

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

        self.AXES = cfg['verifier']['axes']

        dynamics = dict(cls=Bicycle4D,
                        min_steer=-self.MAX_STEER, max_steer=self.MAX_STEER,
                        min_accel=-self.MAX_ACCEL, max_accel=self.MAX_ACCEL,
                        wheelbase=self.WHEELBASE)
        self.impl = TVHJImpl(dynamics, self.AXES)

        if not self.load_from_file:
            # Remove existing directory if it exists, to ensure a clean save
            if os.path.exists(self.save_name):
                shutil.rmtree(self.save_name)
            os.mkdir(self.save_name)

    def run(self):
        tree = Sequence(
            *[self.make_slice_branch(i, t0, t1) for i, (t0, t1) in enumerate(self.slice_time)],
            name="VerifierRoot"
        )
        tree.run()

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

    def build_moving_object(
            self, 
            speed=0.3,
            width=0.4,
            length=1.0,
            center_x=2.0,
            between_y=(3.0, 3.5),
        ):

        # Forward being in direction of y
        _v = speed # moving obstacle speed
        _w = width # moving obstacle width
        _x = center_x # moving obstacle center x
        _l = length # moving obstacle length
        _y = between_y[0] # moving obstacle initial y position
        boundary = BoundedSet(x=(_x-_w/2, _x+_w/2), y=between_y)
        obj = EMPTY
        for i, t in enumerate([0.1, 5.1, 10.1, 15.1, 19.1]):
            __v = +_v if i % 2 == 0 else -_v
            __t = t + 0.5 if i % 2 == 0 else t - 0.5
            infront_obj = HalfSpaceSet(normal=[     +1, -__v],
                                       offset=[_y+_l/2,    t],
                                       axes = [    'y',  't'])

            behind_obj = HalfSpaceSet(normal=[     -1, +__v],
                                      offset=[_y-_l/2,  __t],
                                      axes = [    'y',  't'])
            obj = Union(obj, reduce(Inter, [
                Compl(infront_obj), 
                Compl(behind_obj),
                boundary,
            ]))

        return obj

    def save_set(self, data, number):
        path = f"{self.save_name}/data_{number}.npy"
        np.save(path, np.asarray(data))
        return path

    def slice_and_save(self, raw: np.ndarray, t_start: float, t_end: float, idx: int) -> str:
        i_start = int(t_start / self.TIMESTEP)
        i_end   = int(t_end   / self.TIMESTEP)
        sliced  = raw[i_start:i_end]          # shape: [i_end-i_start, 25, 25, 37, 17]
        path    = f"{self.save_name}/slice_{idx}.npy"
        np.save(path, sliced)
        return path, [t_start, t_end]

    def make_slice_branch(self, slice_idx, t_start, t_end):
        def _branch():
            compute_start = tm.time()
            print(f"Computing set {slice_idx}: slice [{t_start}, {t_end}]")

            env = self.build_env()
            goal_x, goal_y = self.GOAL[0], self.GOAL[1]
            goal_w, goal_h = self.GOAL[2], self.GOAL[3]
            target_set = BoundedSet(x=(goal_x, goal_x + goal_w), y=(goal_y, goal_y + goal_h))
            task = UNTIL(env['safe_region'], target_set)
            out = TLT.construct(task).realize(self.impl)
            raw = np.asarray(out)

            path, interval = self.slice_and_save(raw, t_start, t_end, slice_idx)

            compute_end = tm.time()
            sleep_dur = max(0.0, 3.0 - (compute_end - compute_start))
            tm.sleep(sleep_dur)
            loop_end = tm.time()

            self.timeline.append({
                'set_idx':       slice_idx,
                'interval':      interval,
                'compute_start': compute_start,
                'compute_end':   compute_end,
                'loop_end':      loop_end,
                'path':          path,
                })
            self.registry['verifier.timeline'] = list(self.timeline)
            return NodeStatus.SUCCESS
        return ActionNode(_branch, f"Slice {slice_idx} [{t_start},{t_end}]")