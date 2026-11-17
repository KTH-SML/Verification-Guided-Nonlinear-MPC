import os
from math import pi


# ------------------------------------------------------------------
# Scenario
#   'corridor'   : original small-scale corridor (SVEA 1:10, point vehicle)
#   'urban_road' : full-size passenger car on a two-lane urban road, with
#                  static blockages and several road users crossing it
# Select it here or with the environment variable SCENARIO=... at launch.
# Everything that depends on the scenario (vehicle, map, obstacles, window,
# grid, initial state) is defined in its block below.
# ------------------------------------------------------------------

SCENARIO = os.environ.get('SCENARIO', 'urban_road')
if SCENARIO not in ('corridor', 'urban_road'):
    raise ValueError(f"Unknown SCENARIO '{SCENARIO}' (expected 'corridor' or 'urban_road')")

# ------------------------------------------------------------------
# Verifier (HJ reachability)
# ------------------------------------------------------------------

HJ_TIMESTEP = 0.2    # s
# HJ_HORIZON (s, time length of each global window, receding horizon): see the scenarios below
HJ_PREDICTION = 2.0  # s
HJ_DELAY      = 0.8   # s, how far in the future each local window starts (T_DELAY)
HJ_LOCAL_PERIOD = 0.8  # s, fixed time between the starts of consecutive local windows (T_L)
# Local schedule (simulation time): window k = [k*T_L, k*T_L + HJ_PREDICTION], computed
# at k*T_L - HJ_DELAY. Requirements:
#   local compute time <= HJ_DELAY        (set ready before its window starts)
#   local compute time <= HJ_LOCAL_PERIOD (the schedule does not drift)
#   HJ_LOCAL_PERIOD    <  HJ_PREDICTION   (consecutive local sets overlap)

# Spatial crop of the local grid: the local PDE is solved only on a box of the window.
#   'none'       : whole window (reference, no crop)
#   'reach'      : box reachable from the current state in HJ_DELAY + HJ_PREDICTION
#                  with worst-case speeds (fixed size, centred on the vehicle)
#   'zero_level' : bounding box of the global zero sublevel set over the local window
#                  (the local set is contained in it). The size is set by the first set and
#                  kept afterwards: an axis only grows when the needed box does not fit
#                  (fewer shapes -> fewer JIT compilations)
# Per-set statistics are written to results/crop_<method>.csv.
HJ_CROP_METHOD = 'zero_level'
HJ_CROP_AXES   = ('x', 'y', '*yaw', 'vel')   # axes to crop (names as in AXES)
HJ_CROP_MARGIN = 2                            # extra cells per side (numerical boundary effects)

# Receding global window: every GLOBAL_UPDATE seconds the global problem is
# re-solved on a WINDOW_W x WINDOW_H spatial window placed around the vehicle
# (WINDOW_BACK metres behind it, the rest ahead). The target is a band of depth
# GLOBAL_TARGET_DEPTH on the right edge of the window, or GOAL once it is inside.
# GLOBAL_TARGET_Y = (y_lo, y_hi) restricts the band to a range of y (e.g. one lane);
# None: the whole window height.
# Global schedule (simulation time): global j covers [j*T_U, j*T_U + HJ_HORIZON], is
# computed from j*T_U - T_G on the state at that moment, and is used by the local
# windows starting in [j*T_U, (j+1)*T_U). Requirements:
#   global compute time <= T_G - HJ_DELAY   (ready before the first local set using it is computed)
#   global compute time <= T_U              (the schedule does not drift)
#   T_U + HJ_PREDICTION <= HJ_HORIZON       (every local window ends inside its global horizon)
#   MAX_VEL * (T_G + T_U + HJ_PREDICTION) <= WINDOW_W - WINDOW_BACK  (worst case: the vehicle does
#                                            not pass the target band before the next global is used)
#   (WINDOW_W - WINDOW_BACK) / MAX_VEL + MAX_VEL / (2 MAX_ACCEL) < HJ_HORIZON  (from standstill the
#                                            target band is reachable within the global horizon)
GLOBAL_UPDATE       = 2.6          # s, T_U: global j covers [j*T_U, j*T_U + HJ_HORIZON]
GLOBAL_LEAD         = 2.6          # s, T_G: global j is computed from j*T_U - T_G (background thread)
# WINDOW_W, WINDOW_H, WINDOW_BACK, GLOBAL_TARGET_DEPTH, MISSION_TIME: see the scenarios below

# ------------------------------------------------------------------
# MPC (controller)
# ------------------------------------------------------------------

MPC_TIMESTEP = 0.1  # s
MPC_N        = 6    # horizon steps  →  total horizon = N * TIMESTEP = 0.6 s
Qhj          = 1.0        # cost weight for the time-to-reach (progress) map

# Safety constraint V_safe(x_k) <= -SAFETY_MARGIN at every MPC stage.
# Soft (slack) with a large penalty: identical to a hard constraint whenever a
# safe solution exists, but the solver does not break on small violations.
# SAFETY_MARGIN (extra margin on the value function, m): see the scenarios below
SAFETY_SLACK_L1 = 1e3     # linear penalty on the slack
SAFETY_SLACK_L2 = 1e2     # quadratic penalty on the slack

# Local value function used by the safety constraint:
#   'state' : V_l(t, x, y, yaw, vel) at the predicted state (sufficient for safety)
#   'min'   : min over (yaw, vel) of V_l, i.e. it only depends on (t, x, y), like the
#             TTR map. Only a NECESSARY condition: the constraint is satisfied if SOME
#             heading/speed is safe at that position, not necessarily the vehicle's own
#             (e.g. too fast to stop before a crossing pedestrian), and it gives no
#             gradient in yaw and vel.
SAFETY_V_MODE   = 'state'

# ------------------------------------------------------------------
# Scenario definitions
#
# Vehicle: the state (x, y) is the centre of the rear axle (kinematic bicycle).
#   VEHICLE_LENGTH = None -> point vehicle (obstacles are not inflated).
#   Otherwise the body is [-REAR_OVERHANG, LENGTH - REAR_OVERHANG] x [-WIDTH/2, WIDTH/2]
#   in the vehicle frame, and the verifier inflates every obstacle (static and moving)
#   by the body, separately for each yaw node of the grid (see obstacles.py).
#
# Map
#   X0, Y0: lower-left corner of the map, XW, YW: its size. The whole map (the
#   "city") is never gridded: only the global window is.
#   OBSTACLES, GOAL: (x_min, y_min, width, height). OBSTACLES matches the
#   geometry in verifier.build_env() and the animations.
#
# Dynamic obstacles: list of road users, each crossing the band between_y along y at
#   center_x. width = x-extent, length = y-extent (along its motion).
#   crossings: list of entry times t (direction alternates, + first) or (t, dir)
#   with dir = +1 (towards +y) / -1 (towards -y). One crossing lasts
#   (band + length) / speed. Shared by verifier.py and animation.py (obstacles.py).
#
# Grid: GRID_POINTS = number of nodes for (x, y, yaw, vel) of the global WINDOW.
# ------------------------------------------------------------------

if SCENARIO == 'corridor':

    # Vehicle (SVEA 1:10), point model
    WHEELBASE      = 0.32   # m
    VEHICLE_LENGTH = None
    VEHICLE_WIDTH  = None
    REAR_OVERHANG  = None

    # Verifier (HJ reachability) limits
    MAX_VEL_VERIFIER   = 0.7    # m/s
    MAX_ACCEL_VERIFIER = 0.5    # m/s^2
    MAX_STEER_VERIFIER = 0.6    # rad
    # MPC (controller) limits
    MAX_VEL_MPC        = MAX_VEL_VERIFIER    # m/s, must not exceed the HJ velocity grid
    MAX_ACCEL_MPC      = 0.8    # m/s^2
    MAX_STEER_MPC      = 0.7    # rad

    HJ_HORIZON          = 7.0          # s, (5 - 1.5) / 0.7 + 0.7 / 1.0 = 5.7 s from standstill
    WINDOW_W, WINDOW_H  = 5.0, 4.0     # m
    WINDOW_BACK         = 1.5          # m
    GLOBAL_TARGET_DEPTH = 0.3          # m
    GLOBAL_TARGET_Y     = None         # band over the whole window height
    MISSION_TIME        = 20.0         # s
    SAFETY_MARGIN       = 0.2

    X0, Y0 = 0.0, 0.0
    XW, YW = 12.0, 4.0

    # Horizontal corridor with a slalom of blocks
    OBSTACLES = [
        (2.5, 0.0, 0.5, 2.2),   # bottom block
        (5.0, 1.8, 0.5, 2.2),   # top block
        (8.0, 0.0, 0.5, 2.0),   # bottom block
    ]
    GOAL = (11.3, 1.86, 0.40, 0.4)

    DYN_OBSTACLES = [
        dict(name='box', speed=0.25, width=0.4, length=0.5,
             center_x=6.5, between_y=(0.0, 4.0), crossings=[5.0]),
    ]

    GRID_POINTS = (25, 25, 37, 17)
    X_INIT = [0.5, 2.0, 0.0, 0.0]   # left end of the corridor, heading right

elif SCENARIO == 'urban_road':

    # Vehicle: compact passenger car (e.g. VW Golf class)
    WHEELBASE      = 2.70   # m
    VEHICLE_LENGTH = 4.50   # m, bumper to bumper
    VEHICLE_WIDTH  = 1.80   # m, without mirrors
    REAR_OVERHANG  = 0.90   # m, rear bumper to rear axle (front overhang = 0.90 m)

    # Urban driving (30 km/h zone). The verifier assumes less control authority
    # than the MPC uses, as in the corridor scenario.
    MAX_VEL_VERIFIER   = 6.0    # m/s (28.8 km/h)
    MAX_ACCEL_VERIFIER = 4.0    # m/s^2
    MAX_STEER_VERIFIER = 0.5     # rad (~29 deg at the wheels)
    MAX_VEL_MPC        = 6.0    # m/s, must not exceed the HJ velocity grid
    MAX_ACCEL_MPC      = 4.0    # m/s^2 (comfortable braking / acceleration)
    MAX_STEER_MPC      = 0.5    # rad

    # Window: MAX_VEL * (T_G + T_U + HJ_PREDICTION) = 8 * 5 = 40 m <= WINDOW_W - WINDOW_BACK = 42 m.
    # The window covers the whole road width.
    # From standstill: 42 / 8 + 8 / (2 * 2) = 7.25 s to the target band -> horizon 9 s.
    HJ_HORIZON          = 10.0          # s
    WINDOW_W            = 48.0         # m (WINDOW_H = road width, set below)
    WINDOW_BACK         = 6.0          # m
    GLOBAL_TARGET_DEPTH = 2.0          # m (2 cells in x)
    MISSION_TIME        = 30.0         # s
    SAFETY_MARGIN       = 0.2

    # Straight two-lane road, 120 m long, lanes of 3.5 m with a shoulder on each side
    # (more room between the blockages and the road edge):
    #   right shoulder y in [0, 1], right lane (direction of travel) [1, 4.5],
    #   left lane [4.5, 8], left shoulder [8, 9].
    LANE_WIDTH = 3.5
    SHOULDER   = 1.0
    X0, Y0 = 0.0, 0.0
    XW, YW = 120.0, 2 * LANE_WIDTH + 2 * SHOULDER
    Y_RIGHT  = Y0 + SHOULDER                 # right edge of the right lane
    Y_CENTRE = Y_RIGHT + LANE_WIDTH          # centre line
    WINDOW_H = YW                            # the window covers the whole road width

    # Curbs: just outside the road, so only their inflation (half the body plus
    # the heading allowance) enters the road. They keep the body on the asphalt.
    ROAD_CURBS = [
        (X0, Y0 - 1.0, XW, 1.0),          # right curb (y < 0)
        (X0, Y0 + YW,  XW, 1.0),          # left curb  (y > YW)
    ]
    # Blockages: the car has to change lane to pass them (slalom, as in the corridor).
    OBSTACLES = ROAD_CURBS + [
        (34.0, Y_RIGHT + 0.3, 6.5, 2.2),                           # double-parked delivery van, right lane
        (62.0, Y_CENTRE + 0.7, 8.0, LANE_WIDTH - 0.7 + SHOULDER),  # road works, left lane + shoulder
        (96.0, Y_RIGHT + 0.3, 4.5, 1.8),                           # broken-down car, right lane
    ]
    # Stop in the right lane past the last crosswalk (rear axle in the box, vel = 0)
    GOAL = (110.0, Y_RIGHT, 5.0, LANE_WIDTH)          # whole right lane
    # Band of the global target: right lane only (direction of travel). None: whole road
    GLOBAL_TARGET_Y = (Y_RIGHT, Y_CENTRE)

    # Road users crossing the whole road (y in [0, YW]) at different places and times
    ROAD = (Y0, Y0 + YW)
    # Sequence along the road: intersection (car), zebra crossing (pedestrian),
    # intersection (tram), intersection (car). One crossing blocks the road for
    # (YW + length) / speed: car 2.7 s, pedestrian 6.4 s, tram 6.5 s.
    DYN_OBSTACLES = [
        dict(name='crossing car 1', speed=5.0, width=1.8, length=4.5,
             center_x=22.0, between_y=ROAD, crossings=[(1.0, +1)]),    # intersection, right to left
        dict(name='pedestrian',     speed=1.0, width=0.6, length=0.6,
             center_x=50.0, between_y=ROAD, crossings=[(3.0, -1)]),    # zebra crossing, left to right
        dict(name='tram',           speed=6.0, width=2.65, length=30.0,
             center_x=84.0, between_y=ROAD, crossings=[(8.0, +1)]),    # tram line, right to left
        dict(name='crossing car 2', speed=5.0, width=1.8, length=4.5,
             center_x=106.0, between_y=ROAD, crossings=[(15.5, -1)]),  # intersection, left to right
    ]

    # 1.0 m in x, 0.25 m in y (9 m road), 2pi/37 = 9.7 deg in yaw, 0.5 m/s in vel
    GRID_POINTS = (49, 37, 37, 17)
    # Right lane centre, heading along the road, standing still
    X_INIT = [5.0, Y_RIGHT + LANE_WIDTH / 2, 0.0, 0.0]

# Axes of the global WINDOW grid (not of the whole map). x, y bounds are those
# of the initial window: the verifier moves them at every global update,
# keeping the same shape and resolution (snapped to a lattice anchored at X0, Y0).
# '*' prefix means periodic (pyspect convention);
# consumers triple-concat yaw to extend from [-pi, pi] to [-3pi, 3pi]
points = GRID_POINTS  # number of grid points for each axis (x,y,yaw,vel)
AXES = [
    dict(name='t',    bounds=[0.0, HJ_HORIZON],     step=HJ_TIMESTEP),
    dict(name='x',    bounds=[X0, X0 + WINDOW_W],   points=points[0]),
    dict(name='y',    bounds=[Y0, Y0 + WINDOW_H],   points=points[1]),
    dict(name='*yaw', bounds=[-pi, pi],         points=points[2]),
    dict(name='vel',  bounds=[0.0, MAX_VEL_VERIFIER],    points=points[3]),
]

# ------------------------------------------------------------------
# Registry interface
# ------------------------------------------------------------------

def run(registry):
    """Write all shared configuration into the registry.
    Called by Runtime before any subprocess is started.
    """
    registry['config'] = dict(
        scenario     = SCENARIO,
        # Vehicle
        wheelbase    = WHEELBASE,
        vehicle      = dict(
            wheelbase     = WHEELBASE,
            length        = VEHICLE_LENGTH,
            width         = VEHICLE_WIDTH,
            rear_overhang = REAR_OVERHANG,
        ),
        # Map
        X0           = X0,
        Y0           = Y0,
        XW           = XW,
        YW           = YW,
        mission_time = MISSION_TIME,
        x_init       = list(X_INIT),
        # Verifier
        verifier = dict(
            # Vehicle
            max_steer    = MAX_STEER_VERIFIER,
            max_accel    = MAX_ACCEL_VERIFIER,
            max_vel      = MAX_VEL_VERIFIER,
            # HJ reachability
            timestep  = HJ_TIMESTEP,
            horizon   = HJ_HORIZON,
            t_prediction = HJ_PREDICTION,
            t_delay      = HJ_DELAY,
            local_period = HJ_LOCAL_PERIOD,
            crop_method  = HJ_CROP_METHOD,
            crop_axes    = HJ_CROP_AXES,
            crop_margin  = HJ_CROP_MARGIN,
            axes         = AXES,
            # Receding global window
            global_update       = GLOBAL_UPDATE,
            global_lead         = GLOBAL_LEAD,
            window_size         = (WINDOW_W, WINDOW_H),
            window_back         = WINDOW_BACK,
            global_target_depth = GLOBAL_TARGET_DEPTH,
            global_target_y     = GLOBAL_TARGET_Y,
        ),
        # MPC
        controller = dict(
            # Vehicle
            max_vel      = MAX_VEL_MPC,
            max_accel    = MAX_ACCEL_MPC,
            max_steer    = MAX_STEER_MPC,
            # MPC parameters
            timestep = MPC_TIMESTEP,
            n        = MPC_N,
            qhj      = Qhj,
            safety_margin   = SAFETY_MARGIN,
            safety_slack_l1 = SAFETY_SLACK_L1,
            safety_slack_l2 = SAFETY_SLACK_L2,
            safety_v_mode   = SAFETY_V_MODE,
        ),
        # Environment
        obstacles    = OBSTACLES,
        goal         = GOAL,
        dynamic_obstacles = [dict(d) for d in DYN_OBSTACLES],
    )
