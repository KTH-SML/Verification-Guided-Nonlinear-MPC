from math import pi


# ------------------------------------------------------------------
# Vehicle
# ------------------------------------------------------------------

WHEELBASE = 0.32   # m

# Verifier (HJ reachability) limits
MAX_VEL_VERIFIER   = 0.7    # m/s
MAX_ACCEL_VERIFIER = 0.5    # m/s^2
MAX_STEER_VERIFIER = 0.6    # rad

# MPC (controller) limits
MAX_VEL_MPC        = MAX_VEL_VERIFIER    # m/s, must not exceed the HJ velocity grid
MAX_ACCEL_MPC      = 0.8    # m/s^2
MAX_STEER_MPC      = 0.7    # rad

# ------------------------------------------------------------------
# Verifier (HJ reachability)
# ------------------------------------------------------------------

HJ_TIMESTEP = 0.2    # s
HJ_HORIZON  = 7.0   # s, time length of each global window (receding horizon)
HJ_PREDICTION = 2.0  # s
HJ_DELAY      = 0.2   # s, how far in the future each local window starts (T_DELAY)

# Spatial crop of the local grid: the local set over [t_s, t_e] is contained in the
# global zero sublevel set over the same interval, so the local PDE is solved only on
# the bounding box of that set (recomputed at every local iteration).
HJ_CROP_AXES   = ('x', 'y', '*yaw', 'vel')   # axes to crop (names as in AXES)
HJ_CROP_MARGIN = 2                            # extra cells per side (numerical boundary effects)

# Receding global window: every GLOBAL_UPDATE seconds the global problem is
# re-solved on a WINDOW_W x WINDOW_H spatial window placed around the vehicle
# (WINDOW_BACK metres behind it, the rest ahead). The target is a band of depth
# GLOBAL_TARGET_DEPTH on the right edge of the window, or GOAL once it is inside.
# Requirements:
#   HJ_HORIZON >= GLOBAL_UPDATE + compute time + HJ_DELAY + HJ_PREDICTION
#   MAX_VEL * (GLOBAL_UPDATE + compute time) < WINDOW_W - WINDOW_BACK
GLOBAL_UPDATE       = 2.0          # s
WINDOW_W, WINDOW_H  = 5.0, 4.0     # m
WINDOW_BACK         = 1.5          # m
GLOBAL_TARGET_DEPTH = 0.3          # m

# Total duration of the simulation (independent of the global window length)
MISSION_TIME = 20.0  # s

# ------------------------------------------------------------------
# MPC (controller)
# ------------------------------------------------------------------

MPC_TIMESTEP = 0.1  # s
MPC_N        = 5    # horizon steps  →  total horizon = N * TIMESTEP = 0.6 s
Qhj          = 3.0        # cost weight for the time-to-reach (progress) map

# Safety constraint V_safe(x_k) <= -SAFETY_MARGIN at every MPC stage.
# Soft (slack) with a large penalty: identical to a hard constraint whenever a
# safe solution exists, but the solver does not break on small violations.
SAFETY_MARGIN   = 0.07    # extra margin on the value function
SAFETY_SLACK_L1 = 1e3     # linear penalty on the slack
SAFETY_SLACK_L2 = 1e2     # quadratic penalty on the slack

# ------------------------------------------------------------------
# Map
# X0, Y0: Lower-left corner of the map
# XW, YW: Upper-right corner of the map is (XW, YW)
# Obstacles
# Each entry: (x_min, y_min, width, height)
# Matches the BoundedSet geometry in verifier.build_env() and
# the _draw_obstacles() format used in controller animations.
# ------------------------------------------------------------------

# The whole map (the "city") is never gridded: only the global window is.
X0, Y0 = 0.0, 0.0
XW, YW = 12.0, 4.0

# Horizontal corridor with a slalom of blocks
OBSTACLES = [
    (2.5, 0.0, 0.5, 2.2),   # bottom block
    (5.0, 1.8, 0.5, 2.2),   # top block
    (8.0, 0.0, 0.5, 2.0),   # bottom block
]

# Final goal region: (x_min, y_min, width, height)
GOAL = (11.3, 1.86, 0.40, 0.4)

# ------------------------------------------------------------------
# Dynamic (moving) obstacle
# Crosses the whole band between_y, entering it at each time in `crossings`
# and leaving it through the far side (alternating direction: +speed on even
# index, -speed on odd). One crossing lasts (band + length) / speed.
# Shared by verifier.py (HJ reachability) and animation.py (visualization)
# so both use the exact same obstacle model.
# ------------------------------------------------------------------

DYN_OBS_SPEED     = 0.25                 # m/s
DYN_OBS_WIDTH     = 0.4                  # m (x-extent)
DYN_OBS_LENGTH    = 0.5                  # m (y-extent)
DYN_OBS_CENTER_X  = 6.5
DYN_OBS_BETWEEN_Y = (0.0, 4.0)
DYN_OBS_CROSSINGS = [5.0]                # s

# Axes of the global WINDOW grid (not of the whole map). x, y bounds are those
# of the initial window: the verifier moves them at every global update,
# keeping the same shape and resolution (snapped to a lattice anchored at X0, Y0).
# '*' prefix means periodic (pyspect convention);
# consumers triple-concat yaw to extend from [-pi, pi] to [-3pi, 3pi]
points = (25, 25, 37, 17)  # number of grid points for each axis (x,y,yaw,vel)
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
        # Vehicle
        wheelbase    = WHEELBASE,
        # Map
        X0           = X0,
        Y0           = Y0,
        XW           = XW,
        YW           = YW,
        mission_time = MISSION_TIME,
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
            crop_axes    = HJ_CROP_AXES,
            crop_margin  = HJ_CROP_MARGIN,
            axes         = AXES,
            # Receding global window
            global_update       = GLOBAL_UPDATE,
            window_size         = (WINDOW_W, WINDOW_H),
            window_back         = WINDOW_BACK,
            global_target_depth = GLOBAL_TARGET_DEPTH,
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
        ),
        # Environment
        obstacles    = OBSTACLES,
        goal         = GOAL,
        dynamic_obstacle = dict(
            speed     = DYN_OBS_SPEED,
            width     = DYN_OBS_WIDTH,
            length    = DYN_OBS_LENGTH,
            center_x  = DYN_OBS_CENTER_X,
            between_y = DYN_OBS_BETWEEN_Y,
            crossings = DYN_OBS_CROSSINGS,
        ),
    )