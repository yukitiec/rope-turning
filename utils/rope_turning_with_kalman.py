import casadi as ca
import numpy as np
from kalman_filter_3d import KalmanFilter3D


def make_spring_force():
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


def make_f_with_kalman():
    """
    Create the state transition function with Kalman filter acceleration estimation.

    Returns:
        casadi.Function: State transition function
    """
    # Constants (should match the original rope_turning.ipynb)
    global k_s, l, m, G, ns, nx, nu

    # Initialize constants if not defined
    if "k_s" not in globals():
        k_s = 2  # spring constant
        l = 0.1375  # natural length
        m = 0.1125  # mass
        G = ca.DM([0, 0, 9.81])  # gravity
        ns = 5  # number of springs
        nx = 24  # state dimension
        nu = 3  # control dimension

    states = ca.SX.sym("states", nx)  # state vector
    ctrls = ca.SX.sym("ctrls", nu)  # robot's control input
    ctrl_hum = ca.SX.sym("ctrl_hum", 3)  # human's control input

    # Rope transition function
    F = make_spring_force()

    # Kalman filter acceleration estimator
    kalman_acc_fn = make_kalman_acceleration_estimator()

    # Number of rope states
    offset = 3 * ns  # rope position

    X_dot = []
    V_dot = []

    # Initialize Kalman filter states for each rope point
    kf_states = []
    for k in range(ns):
        # Initialize with current position and velocity
        pos = states[3 * k : 3 * (k + 1)]
        vel = states[offset + 3 * k : offset + 3 * (k + 1)] if k < ns - 1 else ctrl_hum
        kf_state = ca.vertcat(pos, vel, ca.SX.zeros(3))  # [pos, vel, acc]
        kf_states.append(kf_state)

    # Velocity from robot end-effector to human's turning point
    for k in range(0, ns - 2):
        vel = states[offset + 3 * k : offset + 3 * (k + 1)]
        X_dot.append(vel)

    # Acceleration using Kalman filter estimation
    for k in range(1, ns - 1):  # except robot and human
        pos = states[3 * k : 3 * (k + 1)]  # current position
        vel = states[offset + 3 * k : offset + 3 * (k + 1)]  # current velocity

        # Use Kalman filter to estimate acceleration
        dt = 0.1  # time step (should be passed as parameter)
        acc_est, kf_state_updated = kalman_acc_fn(
            kf_states[k], pos, vel, dt
        )  # This can be used for human acceleration estimation.

        # Update Kalman filter state
        kf_states[k] = kf_state_updated

        # Use estimated acceleration instead of spring force calculation
        vel_dot = acc_est - G  # acceleration minus gravity

        V_dot.append(vel_dot)

    states_dot = ca.vertcat(ctrls, *X_dot, ctrl_hum, *V_dot)

    f = ca.Function(
        "f_with_kalman",
        [states, ctrls, ctrl_hum],
        [states_dot],
        ["x", "u", "u_hum"],
        ["x_dot"],
    )

    return f


def make_f_hybrid():
    """
    Create a hybrid approach that uses both spring forces and Kalman filter.

    Returns:
        casadi.Function: State transition function
    """
    # Constants
    global k_s, l, m, G, ns, nx, nu

    if "k_s" not in globals():
        k_s = 2
        l = 0.1375
        m = 0.1125
        G = ca.DM([0, 0, 9.81])
        ns = 5
        nx = 24
        nu = 3

    states = ca.SX.sym("states", nx)
    ctrls = ca.SX.sym("ctrls", nu)
    ctrl_hum = ca.SX.sym("ctrl_hum", 3)
    # Rope transition function
    F = make_spring_force()

    # Kalman filter acceleration estimator
    kalman_acc_fn = make_kalman_acceleration_estimator()

    offset = 3 * ns

    X_dot = []
    V_dot = []

    # Initialize Kalman filter states
    kf_states = []
    for k in range(ns):
        pos = states[3 * k : 3 * (k + 1)]
        vel = states[offset + 3 * k : offset + 3 * (k + 1)] if k < ns - 1 else ctrl_hum
        kf_state = ca.vertcat(pos, vel, ca.SX.zeros(3))
        kf_states.append(kf_state)

    # Velocity
    for k in range(0, ns - 2):
        vel = states[offset + 3 * k : offset + 3 * (k + 1)]
        X_dot.append(vel)

    # Hybrid acceleration: combine spring forces with Kalman estimation
    for k in range(1, ns - 1):
        pos = states[3 * k : 3 * (k + 1)]
        pos_n = states[3 * (k + 1) : 3 * (k + 2)]
        vel = states[offset + 3 * k : offset + 3 * (k + 1)]

        if k != 0:
            pos_p = states[3 * (k - 1) : 3 * k]
        else:
            pos_p = states[0:3]

        # Spring force calculation
        # This can be used for rope points acceleration estimation.
        spring_acc = (
            F(pos_i=pos, pos_i1=pos_n)["F"] - F(pos_i=pos_p, pos_i1=pos)["F"]
        ) / m

        # Kalman filter estimation
        dt = 0.1
        kalman_acc, kf_state_updated = kalman_acc_fn(kf_states[k], pos, vel, dt)

        # Update Kalman filter state
        kf_states[k] = kf_state_updated

        # Combine spring forces with Kalman estimation (weighted average)
        alpha = 0.7  # weight for spring forces
        vel_dot = alpha * spring_acc + (1 - alpha) * kalman_acc - G

        V_dot.append(vel_dot)

    states_dot = ca.vertcat(ctrls, *X_dot, ctrl_hum, *V_dot)

    f = ca.Function(
        "f_hybrid",
        [states, ctrls, ctrl_hum],
        [states_dot],
        ["x", "u", "u_hum"],
        ["x_dot"],
    )

    return f


# Example usage function
def example_kalman_integration():
    """
    Example of how to use the Kalman filter integration in rope turning.
    """
    print("=== Kalman Filter Integration in Rope Turning ===")

    # Create the functions
    f_kalman = make_f_with_kalman()
    f_hybrid = make_f_hybrid()

    print("Created Kalman filter integration functions:")
    print(f"- f_kalman: {f_kalman}")
    print(f"- f_hybrid: {f_hybrid}")

    # Test with sample data
    states = np.random.rand(24)
    ctrls = np.random.rand(3)
    ctrl_hum = np.random.rand(3)

    # Convert to CasADi DM
    states_dm = ca.DM(states)
    ctrls_dm = ca.DM(ctrls)
    ctrl_hum_dm = ca.DM(ctrl_hum)

    # Test Kalman filter approach
    result_kalman = f_kalman(x=states_dm, u=ctrls_dm, u_hum=ctrl_hum_dm)
    print(f"\nKalman filter result shape: {result_kalman['x_dot'].shape}")

    # Test hybrid approach
    result_hybrid = f_hybrid(x=states_dm, u=ctrls_dm, u_hum=ctrl_hum_dm)
    print(f"Hybrid approach result shape: {result_hybrid['x_dot'].shape}")

    return f_kalman, f_hybrid


if __name__ == "__main__":
    # Run example
    f_kalman, f_hybrid = example_kalman_integration()
