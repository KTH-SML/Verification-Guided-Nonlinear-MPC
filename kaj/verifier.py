
from pyspect import *
from pyspect.impls.hj_reachability import TVHJImpl
from pyspect.systems.hj_reachability.bicycle4d import Bicycle4D

import numpy as np

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
    
    def __init__(self, registry, state_queue, stop_event):

        # Runtime communication primitives
        self.registry = registry
        self.state_queue = state_queue
        self.stop_event = stop_event

        self.vf = np.load('/home/tokisaki_kurumi/Projects/svea/verification_guided_mpc/out_bicycle4d_with_obs.npy', mmap_mode='r').copy()
        info = {'interval': (0.0, 20.0), 'name': 'foo'}

        self.registry['latest.info'] = info
        self.registry['foo.vf'] = self.vf

    
    def run(self):
        pass

    def close(self):
        pass