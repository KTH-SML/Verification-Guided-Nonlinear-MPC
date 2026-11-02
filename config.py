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
HJ_HORIZON  = 35.0   # s
HJ_PREDICTION = 2.0  # s

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

X0, Y0 = 0.0, 0.0
XW, YW = 4.0, 4.0

OBSTACLES = [
    (0.0, 0.0, 2.0, 2.0),   # box_area:  x=[0.01,2.0], y=[0.01,2.0]
    (1.2, 2.8, 2.8, 0.4),   # thin_wall: x=[1.2, ...], y=[2.8, 3.2]
    (3.5, 0.0, 0.4, 3.0),
    (0.0, 3.8, 4.0, 0.2),
]

# Goal region: (x_min, y_min, width, height)
GOAL = (3.58, 3.46, 0.30, 0.36)

# ------------------------------------------------------------------
# Dynamic (moving) obstacle
# Crosses the whole band between_y, entering it at each time in `crossings`
# and leaving it through the far side (alternating direction: +speed on even
# index, -speed on odd). One crossing lasts (band + length) / speed.
# Shared by verifier.py (HJ reachability) and animation.py (visualization)
# so both use the exact same obstacle model.
# ------------------------------------------------------------------

DYN_OBS_SPEED     = 0.09                  # m/s
DYN_OBS_WIDTH     = 0.4                  # m (x-extent)
DYN_OBS_LENGTH    = 0.5                  # m (y-extent)
DYN_OBS_CENTER_X  = 2.0
DYN_OBS_BETWEEN_Y = (3.0, 4.0)
DYN_OBS_CROSSINGS = [0.4, 5.5, 10.5, 15.5]   # s

# State-space axes (spatial only, no time)
# '*' prefix means periodic (pyspect convention);
# consumers triple-concat yaw to extend from [-pi, pi] to [-3pi, 3pi]
points = (25, 25, 37, 17)  # number of grid points for each axis (x,y,yaw,vel)
AXES = [
    dict(name='t',    bounds=[0.0, HJ_HORIZON], step=HJ_TIMESTEP),
    dict(name='x',    bounds=[X0, XW],          points=points[0]),
    dict(name='y',    bounds=[Y0, YW],          points=points[1]),
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
            axes         = AXES,
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