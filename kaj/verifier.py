
from functools import reduce
from math import pi

import numpy as np

from pyspect import *
from pyspect.impls.hj_reachability import TVHJImpl
from pyspect.systems.hj_reachability.bicycle4d import Bicycle4D


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

    TIMESTEP = 0.2
    HORIZON = 20.0

    WHEELBASE = 0.32

    X0, XW = 0.0, 4.0
    Y0, YW = 0.0, 4.0
    MAX_VEL = 0.8

    MAX_STEER = 0.4
    MAX_ACCEL = 0.8

    AXES = [dict(name='t',      bounds=[  0, HORIZON],  step=TIMESTEP),
            dict(name='x',      bounds=[ X0, X0 + XW],  points=25),
            dict(name='y',      bounds=[ Y0, Y0 + YW],  points=25),
            dict(name='*yaw',   bounds=[-pi,      pi],  points=15),
            dict(name='vel',    bounds=[0.0, MAX_VEL],  points=11)]

    def __init__(self, registry, state_queue, stop_event):

        # Runtime communication primitives
        self.registry = registry
        self.state_queue = state_queue
        self.stop_event = stop_event

        dynamics = dict(cls=Bicycle4D,
                        min_steer=-self.MAX_STEER, max_steer=self.MAX_STEER,
                        min_accel=-self.MAX_ACCEL, max_accel=self.MAX_ACCEL,
                        wheelbase=self.WHEELBASE)
        self.impl = TVHJImpl(dynamics, self.AXES)

    def run(self):
        
        raw = np.load('../vf_data.npy', mmap_mode='r').copy()
        self.vf = np.concatenate([raw, raw, raw], axis=3)
        info = {'interval': (0.0, 20.0), 'name': 'foo'}

        self.registry['latest.info'] = info
        self.registry['foo.vf'] = self.vf

        # Slow loop
        while True:

            env = self.build_env()
            static_obj = self.build_static_object()
            moving_obj = self.build_moving_object()

            goal_x, goal_y = 3.58, 3.46
            goal_w, goal_h = 0.30, 0.36

            target_set = BoundedSet(x=(goal_x, goal_x + goal_w), y=(goal_y, goal_y + goal_h))

            task = UNTIL(Compl(env['static_env']), target_set)

            task = self.create_task()

            out1 = TLT(task).realize(self.impl)



        

    def close(self):
        pass

    def build_env(self):

        box_area = BoundedSet.relative(axes=['x', 'y'], 
                                          anchor=[0.0, 0.0],
                                          size=[2.0, 2.0])
        
        thin_wall = BoundedSet.relative(axes=['x', 'y'],
                                        anchor=[1.2, 2.8],
                                        size=[2.8, 0.4])

        static_env = Union(box_area, thin_wall)

        checkpoint = BoundedSet.relative()

        target_loc = BoundedSet.relative(axes=['x', 'y'],
                                         anchor=[3.58, 3.46],
                                         size=[0.30, 0.36])

        return dict(static_env=static_env,
                    checkpoint=checkpoint,
                    target_loc=target_loc)

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
    
