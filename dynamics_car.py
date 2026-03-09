##
## Configuration file for MPC Controller
##

## Core Parameters
# The time step in which the optimization problem is divided (unit [s])
time_step = 0.2

# The prediction horizon steps for the mpc optimization problem
prediction_horizon = 3

## Objective Function J

# Q1
state_weight_matrix = [1, 0, 0, 0,
                       0, 1, 0, 0,
                       0, 0, 1, 0,
                       0, 0, 0, 1]

# Q1 5D
state_weight_matrix_5D =   [0, 0, 0, 0, 0,
                            0, 0, 0, 0, 0,
                            0, 0, 0, 0, 0,
                            0, 0, 0, 0, 0,
                            0, 0, 0, 0, 0]
# state_weight_matrix_5D =   [1, 0, 0, 0, 0,
#                             0, 1, 0, 0, 0,
#                             0, 0, 1, 0, 0,
#                             0, 0, 0, 1, 0,
#                             0, 0, 0, 0, 1]

# Q2
control_rate_weight_matrix = [0, 0,
                              0, 0]

# Q3
control_weight_matrix = [0.0, 0,
                        0, 0.0]

# Qf
final_state_weight_matrix = [1, 0, 0, 0,
                             0, 1, 0, 0,
                             0, 0, 1, 0,
                             0, 0, 0, 1]

# Qf 5D
final_state_weight_matrix_5D = [0, 0, 0, 0, 0,
                                0, 0, 0, 0, 0,
                                0, 0, 0, 0, 0,
                                0, 0, 0, 0, 0,
                                0, 0, 0, 0, 0]
# final_state_weight_matrix_5D = [1, 0, 0, 0, 0,
#                                 0, 1, 0, 0, 0,
#                                 0, 0, 1, 0, 0,
#                                 0, 0, 0, 1, 0,
#                                 0, 0, 0, 0, 1]

# Qv
# Note = A big value encourage forward movements rather than backwards.
#       FEATURE DOES NOT WORK SO FAR.
forward_speed_weight = 0

# Qs
steering_weight = 0.0

# Qhj
reachability_set_weight = 1.0

#Qg
gradient_weight = 0.0

## Model Parameters

# The min/max steering angle allowed for SVEA (unit [degrees])
steering_min = -40
steering_max = 40

# The min/max steering rate allowed for SVEA (unit [degrees/s])
steering_rate_min = -180 
steering_rate_max = 180

# The min/max velocity allowed for SVEA (unit [m/s^2])
velocity_min = 0.0
velocity_max = 1.5

# The min/max acceleration allowed for SVEA (unit [m/s^2])
acceleration_min = -0.2
acceleration_max = 0.2

# The wheelbase of the vehicle (unit [m]).
wheelbase = 0.32