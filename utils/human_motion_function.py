import casadi
import numpy as np


def make_human_motion():
    """
    Create a CasADi function for human motion update.

    This function encapsulates the human motion update logic from the original code,
    converting the numpy-based implementation to a CasADi symbolic function.

    Returns:
    --------
    casadi.Function
        Function that updates human position and velocity based on current index and time step.
        Inputs: [idx_hum, dt, p_hum_prev, samples_human, vel_samples_turn_human]
        Outputs: [pos_human, vel_human, u_hum, idx_hum_new, p_hum_prev_new]

    Usage:
    ------
    human_motion_fn = make_human_motion()

    # Convert numpy arrays to CasADi DM
    samples_human_dm = casadi.DM(samples_human)
    vel_samples_turn_human_dm = casadi.DM(vel_samples_turn_human)

    # Call the function
    result = human_motion_fn(
        idx_hum=casadi.DM([idx_hum]),
        dt=casadi.DM([dt]),
        p_hum_prev=casadi.DM(p_hum_prev),
        samples_human=samples_human_dm,
        vel_samples_turn_human=vel_samples_turn_human_dm
    )

    pos_human, vel_human, u_hum, idx_hum_new, p_hum_prev_new = result
    """
    # Define symbolic variables
    idx_hum = casadi.SX.sym("idx_hum", 1)  # current human index
    dt = casadi.SX.sym("dt", 1)  # time step
    p_hum_prev = casadi.SX.sym("p_hum_prev", 3)  # previous human position

    # Create symbolic arrays for samples_human and vel_samples_turn_human
    # These will be passed as parameters to the function
    samples_human_param = casadi.SX.sym("samples_human", 360, 3)
    vel_samples_turn_human_param = casadi.SX.sym("vel_samples_turn_human", 360)

    # Calculate degree step based on velocity (equivalent to the original code)
    # Original: deg_step = max(1, int(vel_samples_turn_human[idx_hum] * dt))
    deg_step = casadi.fmax(
        1, casadi.floor(vel_samples_turn_human_param[casadi.floor(idx_hum)] * dt)
    )

    # Update human index (equivalent to the original code)
    # Original: idx_hum += deg_step; idx_hum = idx_hum % 360
    idx_hum_new = casadi.fmod(idx_hum + deg_step, 360)

    # Get current human position (equivalent to the original code)
    # Original: pos_human = samples_human[idx_hum]
    idx_int = casadi.floor(idx_hum_new)
    pos_human = samples_human_param[idx_int, :]

    # Calculate velocity (equivalent to the original code)
    # Original: vel_human = (pos_human - p_hum_prev) / dt
    vel_human = (pos_human - p_hum_prev) / dt

    # Control input (same as velocity, equivalent to the original code)
    # Original: u_hum = vel_human.copy()
    u_hum = vel_human

    # Update previous position (equivalent to the original code)
    # Original: p_hum_prev = pos_human.copy()
    p_hum_prev_new = pos_human

    # Create the function
    human_motion_fn = casadi.Function(
        "human_motion",
        [idx_hum, dt, p_hum_prev, samples_human_param, vel_samples_turn_human_param],
        [pos_human, vel_human, u_hum, idx_hum_new, p_hum_prev_new],
        ["idx_hum", "dt", "p_hum_prev", "samples_human", "vel_samples_turn_human"],
        ["pos_human", "vel_human", "u_hum", "idx_hum_new", "p_hum_prev_new"],
    )

    return human_motion_fn


def make_human_motion_simple():
    """
    Create a simplified CasADi function for human motion update.

    This version uses a simpler approach that directly maps the numpy logic
    to CasADi symbolic operations.

    Returns:
    --------
    casadi.Function
        Function that updates human position and velocity.
        Inputs: [idx_hum, dt, p_hum_prev, samples_human, vel_samples_turn_human]
        Outputs: [pos_human, vel_human, u_hum, idx_hum_new, p_hum_prev_new]
    """
    # Define symbolic variables
    idx_hum = casadi.SX.sym("idx_hum", 1)
    dt = casadi.SX.sym("dt", 1)
    p_hum_prev = casadi.SX.sym("p_hum_prev", 3)
    samples_human = casadi.SX.sym("samples_human", 360, 3)
    vel_samples_turn_human = casadi.SX.sym("vel_samples_turn_human", 360)

    # Calculate step size
    vel_at_idx = vel_samples_turn_human[casadi.floor(idx_hum)]
    deg_step = casadi.fmax(1, casadi.floor(vel_at_idx * dt))

    # Update index with modulo operation
    idx_hum_new = casadi.fmod(idx_hum + deg_step, 360)

    # Get position at new index
    idx_int = casadi.floor(idx_hum_new)
    pos_human = samples_human[idx_int, :]

    # Calculate velocity and control input
    vel_human = (pos_human - p_hum_prev) / dt
    u_hum = vel_human

    # Update previous position
    p_hum_prev_new = pos_human

    # Create function
    fn = casadi.Function(
        "human_motion_simple",
        [idx_hum, dt, p_hum_prev, samples_human, vel_samples_turn_human],
        [pos_human, vel_human, u_hum, idx_hum_new, p_hum_prev_new],
    )

    return fn


# Example usage function
def example_usage():
    """
    Example of how to use the make_human_motion function.
    """
    # Create the function
    human_motion_fn = make_human_motion()

    # Example data (you would get this from your actual simulation)
    samples_human = np.random.rand(360, 3)  # 360 positions, 3D coordinates
    vel_samples_turn_human = np.random.uniform(10, 20, 360)  # velocity at each position
    idx_hum = 0  # current index
    dt = 0.1  # time step
    p_hum_prev = samples_human[0]  # previous position

    # Convert to CasADi DM
    samples_human_dm = casadi.DM(samples_human)
    vel_samples_turn_human_dm = casadi.DM(vel_samples_turn_human)

    # Call the function
    result = human_motion_fn(
        idx_hum=casadi.DM([idx_hum]),
        dt=casadi.DM([dt]),
        p_hum_prev=casadi.DM(p_hum_prev),
        samples_human=samples_human_dm,
        vel_samples_turn_human=vel_samples_turn_human_dm,
    )

    pos_human, vel_human, u_hum, idx_hum_new, p_hum_prev_new = result

    print("Original index:", idx_hum)
    print("New index:", idx_hum_new)
    print("Position:", pos_human)
    print("Velocity:", vel_human)
    print("Control input:", u_hum)

    return result


if __name__ == "__main__":
    # Run example
    example_usage()
