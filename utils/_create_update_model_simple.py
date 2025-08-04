import casadi as ca
import os 
import sys
# Add the parent directory to the path to allow imports from other directories
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# Now import from other directories
try:
    from ur5e import ur_custom_casadi as ur_kin  # Commented out - not used
except ImportError:
    print("Warning: Could not import ur_custom_casadi")
    ur_kin = None


# from ..ur5e import ur_custom_casadi as ur_kin


def make_spring_force(k_s, l):
    """Create spring force function for rope dynamics."""
    pos_i = ca.SX.sym("pos_i", 3)
    pos_i1 = ca.SX.sym("pos_i1", 3)
    pos_rel = pos_i1 - pos_i
    norm = ca.norm_2(pos_rel)
    force = k_s * (norm - l) * pos_rel / norm  # Hooke's law

    F = ca.Function("F", [pos_i, pos_i1], [force], ["pos_i", "pos_i1"], ["F"])
    return F


def make_kalman_acceleration_estimator():
    """
    Create a CasADi function that uses Kalman filter to estimate acceleration.

    Returns:
        casadi.Function: Function that estimates acceleration using Kalman filter
    """
    # Define symbolic variables for Kalman filter inputs
    position = ca.SX.sym("position", 3)  # Current position measurement
    velocity = ca.SX.sym("velocity", 3)  # Current velocity
    dt = ca.SX.sym("dt", 1)  # Time step

    # Kalman filter state: [x, y, z, vx, vy, vz, ax, ay, az]
    kf_state = ca.SX.sym("kf_state", 9)

    # Create symbolic Kalman filter matrices
    # State transition matrix A (9x9) for constant acceleration model
    A = ca.SX.zeros(9, 9)

    # Position rows: x = x + vx*dt + 0.5*ax*dt^2
    A[0, 0] = 1.0  # x
    A[0, 3] = dt  # vx
    A[0, 6] = 0.5 * dt * dt  # ax

    A[1, 1] = 1.0  # y
    A[1, 4] = dt  # vy
    A[1, 7] = 0.5 * dt * dt  # ay

    A[2, 2] = 1.0  # z
    A[2, 5] = dt  # vz
    A[2, 8] = 0.5 * dt * dt  # az

    # Velocity rows: vx = vx + ax*dt
    A[3, 3] = 1.0  # vx
    A[3, 6] = dt  # ax

    A[4, 4] = 1.0  # vy
    A[4, 7] = dt  # ay

    A[5, 5] = 1.0  # vz
    A[5, 8] = dt  # az

    # Acceleration rows: ax, ay, az remain constant
    A[6, 6] = 1.0  # ax
    A[7, 7] = 1.0  # ay
    A[8, 8] = 1.0  # az

    # Measurement matrix H (3x9) - we measure position only
    H = ca.SX.zeros(3, 9)
    H[0, 0] = 1.0  # measure x
    H[1, 1] = 1.0  # measure y
    H[2, 2] = 1.0  # measure z

    # Process noise covariance (9x9)
    Q = ca.SX.eye(9)
    Q[0, 0] = 1e-4  # position noise
    Q[1, 1] = 1e-4
    Q[2, 2] = 1e-4
    Q[3, 3] = 1e-4  # velocity noise
    Q[4, 4] = 1e-4
    Q[5, 5] = 1e-4
    Q[6, 6] = 1e-4  # acceleration noise
    Q[7, 7] = 1e-4
    Q[8, 8] = 1e-4

    # Measurement noise covariance (3x3)
    R = ca.SX.eye(3) * 1e4

    # Prediction step
    kf_state_pred = ca.mtimes(A, kf_state)

    # Update step (simplified - using current velocity as measurement)
    # measurement is the current position.
    # Innovation: measurement - predicted measurement
    innovation = position - ca.mtimes(H, kf_state_pred)

    # Kalman gain (simplified calculation)
    S = ca.mtimes(ca.mtimes(H, Q), H.T) + R
    K = ca.mtimes(ca.mtimes(Q, H.T), ca.inv(S))

    # Update state
    kf_state_updated = kf_state_pred + ca.mtimes(K, innovation)

    # Extract acceleration from updated state
    acceleration = kf_state_updated[6:9]  # [ax, ay, az]

    # Create the function
    kalman_acc_fn = ca.Function(
        "kalman_acceleration",
        [kf_state, position, velocity, dt],
        [acceleration, kf_state_updated],
        ["kf_state", "position", "velocity", "dt"],
        ["acceleration", "kf_state_updated"],
    )

    return kalman_acc_fn


"""Create an update model for the rope dynamics.
rope_dynamics =  spring_force
robot = joint angular acceleration (input)
human = Kalman filter acceleration
"""


def create_update_model(
    k_s=2,  # spring constant
    l=0.1375,  # natural length
    m=0.1125,  # mass
    G=ca.DM([0, 0, 9.81]),  # gravity
    ns=5,  # number of springs
    nx=24,  # state dimension
    nu=3,
    dt=1.0 / 30.0,  # time step for Kalman filter
):  # control dimension
    """
    kf = KalmanFilter3D(
            initial_x=initial_pos[0],
            initial_y=initial_pos[1],
            initial_z=initial_pos[2],
            initial_vx=0,  # initial_vel[0],
            initial_vy=0,  # initial_vel[1],
            initial_vz=0,  # initial_vel[2],
            initial_ax=0,  # initial_acc[0],
            initial_ay=0,  # initial_acc[1],
            initial_az=initial_acc[2],
            dt=dt,
        )

    States : Casadi SX
        x= (
        x_rob,x_2,,,x_N-1,x_hum, #N point
        v_rob,v_2,,,v_N-1,v_hum, #N point
        )
        x_dot = (
        v_rob,v_2,,,v_N-1,v_hum, #N point
        a_rob,a_2,,,a_N-1,a_hum, #N point
        )

        a_rob = u #control input
        a_2 -- a_N-1 <- Spring force.
        a_hum <- Kalman filter acceleration. in MPC human motion is constant velocity model
    """

    states = ca.SX.sym("states", nx)  # state vector
    ctrls = ca.SX.sym("ctrls", nu)  # robot's control input
    a_human = ca.SX.sym("a_human", 3)  # human acceleration

    #ur_kin = ur_casadi.UrCustomCasadi()
    # forward kinematics
    # p_eef, R_eef = ur_kin.fk_func(states[0:6])
    # jacobian
    # J = ur_kin.jacobian_func(states[0:6])

    # Rope transition function
    F = make_spring_force(k_s, l)

    # Number of rope states
    offset = 3 * 3  # robot, middle, human
    #position
    x_robot = states[:3]
    x_middle = states[3:6]
    x_human = states[6:9]
    #x_dot
    x_dot_robot = states[offset:offset+3] + ctrls
    x_dot_middle = states[offset+3:offset+6]
    x_dot_human = states[offset+6:offset+9]
    #x_ddot
    x_ddot_robot = ca.SX.zeros(3)
    x_ddot_middle = (F(pos_i=x_middle, pos_i1=x_human)["F"]-F(pos_i=x_robot, pos_i1=x_middle)["F"])/m - G
    x_ddot_human = a_human

    # robot's end-effector acceleration estimation.
    states_dot = ca.vertcat(
        x_dot_robot,
        x_dot_middle,
        x_dot_human,
        x_ddot_robot,
        x_ddot_middle,
        x_ddot_human,
    ) 

    f = ca.Function("f", [states,ctrls,a_human], [states_dot], ["x", "u","a_human"], ["x_dot"])

    return f
    
def create_update_model_4springs(
    k_s=2,  # spring constant
    l=0.1375,  # natural length
    m=0.1125,  # mass
    G=ca.DM([0, 0, 9.81]),  # gravity
    ns=5,  # number of springs
    nx=24,  # state dimension
    nu=3,
    dt=1.0 / 30.0,  # time step for Kalman filter
):  # control dimension
    """
    kf = KalmanFilter3D(
            initial_x=initial_pos[0],
            initial_y=initial_pos[1],
            initial_z=initial_pos[2],
            initial_vx=0,  # initial_vel[0],
            initial_vy=0,  # initial_vel[1],
            initial_vz=0,  # initial_vel[2],
            initial_ax=0,  # initial_acc[0],
            initial_ay=0,  # initial_acc[1],
            initial_az=initial_acc[2],
            dt=dt,
        )

    States : Casadi SX
        x= (
        x_rob,x_2,,,x_N-1,x_hum, #N point
        v_rob,v_2,,,v_N-1,v_hum, #N point
        )
        x_dot = (
        v_rob,v_2,,,v_N-1,v_hum, #N point
        a_rob,a_2,,,a_N-1,a_hum, #N point
        )

        a_rob = u #control input
        a_2 -- a_N-1 <- Spring force.
        a_hum <- Kalman filter acceleration. in MPC human motion is constant velocity model
    """

    states = ca.SX.sym("states", nx)  # state vector
    ctrls = ca.SX.sym("ctrls", nu)  # robot's control input
    a_human = ca.SX.sym("a_human", 3)  # human acceleration

    #ur_kin = ur_casadi.UrCustomCasadi()
    # forward kinematics
    # p_eef, R_eef = ur_kin.fk_func(states[0:6])
    # jacobian
    # J = ur_kin.jacobian_func(states[0:6])

    # Rope transition function
    F = make_spring_force(k_s, l)

    # Number of rope states
    offset = 3 * 5  # robot, middle, human
    #position
    x_robot = states[:3]
    x_1 = states[3:6]
    x_m = states[6:9]
    x_3 = states[9:12]
    x_human = states[12:15]
    #x_dot
    x_dot_robot = states[offset:offset+3] + ctrls
    x_dot_1 = states[offset+3:offset+6]
    x_dot_m = states[offset+6:offset+9]
    x_dot_3 = states[offset+9:offset+12]
    x_dot_human = states[offset+12:offset+15]
    #x_ddot
    x_ddot_robot = ca.SX.zeros(3)
    x_ddot_1 = (F(pos_i=x_1, pos_i1=x_m)["F"]-F(pos_i=x_robot, pos_i1=x_1)["F"])/m - G
    x_ddot_m = (F(pos_i=x_m, pos_i1=x_3)["F"]-F(pos_i=x_1, pos_i1=x_m)["F"])/m - G
    x_ddot_3 = (F(pos_i=x_3, pos_i1=x_human)["F"]-F(pos_i=x_m, pos_i1=x_3)["F"])/m - G
    x_ddot_human = a_human

    # robot's end-effector acceleration estimation.
    states_dot = ca.vertcat(
        x_dot_robot,
        x_dot_1,
        x_dot_m,
        x_dot_3,
        x_dot_human,
        x_ddot_robot,
        x_ddot_1,
        x_ddot_m,
        x_ddot_3,
        x_ddot_human,
    ) 

    f = ca.Function("f", [states,ctrls,a_human], [states_dot], ["x", "u","a_human"], ["x_dot"])

    return f

    # robot state estimation.


def create_update_model_with_kf(
    kf_instance,  # Your constructed KalmanFilter3D instance
    k_s=2,  # spring constant
    l=0.1375,  # natural length
    m=0.1125,  # mass
    G=ca.DM([0, 0, 9.81]),  # gravity
    ns=5,  # number of springs
    nx=24,  # state dimension
    nu=3,
    dt=1.0 / 30.0,  # time step for Kalman filter
):
    """
    Create update model with a pre-constructed KalmanFilter3D instance.

    Args:
        kf_instance: Pre-constructed KalmanFilter3D instance
        ... other parameters same as create_update_model
    """

    def kalman_wrapper(pos, vel):
        """Wrapper to use the KalmanFilter3D instance"""
        # Update the Kalman filter with current position
        kf_instance.update(pos)
        # Get acceleration from the filter
        acc = kf_instance.get_acceleration()
        return acc

    # Create the main update model
    f = create_update_model(k_s=k_s, l=l, m=m, G=G, ns=ns, nx=nx, nu=nu, dt=dt)

    # Return both the CasADi function and the Kalman filter wrapper
    return f, kalman_wrapper
