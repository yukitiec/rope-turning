import casadi as ca
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


class KalmanFilter3D:
    """
    3D Kalman Filter with constant acceleration model using CasADi.

    State vector: [x, y, z, vx, vy, vz, ax, ay, az]
    Measurement: [x, y, z] (position only) or
        [x, y, z, vx, vy, vz] (position and velocity)
    """

    # Noise constants (matching the C++ constants)
    NOISE_POS = 1e-4
    NOISE_VEL = 1e-4
    NOISE_ACC = 1e-4
    NOISE_SENSOR = 1e4
    NOISE_POS_3D = 1e-4
    NOISE_VEL_3D = 1e-4
    NOISE_ACC_3D = 1e-4
    NOISE_SENSOR_3D = 1e4

    def __init__(
        self,
        initial_x=0.0,
        initial_y=0.0,
        initial_z=0.0,
        initial_vx=0.0,
        initial_vy=0.0,
        initial_vz=0.0,
        initial_ax=0.0,
        initial_ay=0.0,
        initial_az=0.0,
        process_noise_pos=None,
        process_noise_vel=None,
        process_noise_acc=None,
        measurement_noise=None,
        dt=1.0 / 30.0,
        fps=30.0,
        measure_velocity=False,  # NEW: whether to measure velocity as well
    ):
        """
        Initialize the 3D Kalman Filter.

        Args:
            initial_x, initial_y, initial_z: Initial position
            initial_vx, initial_vy, initial_vz: Initial velocity
            initial_ax, initial_ay, initial_az: Initial acceleration
            process_noise_pos: Process noise for position
            process_noise_vel: Process noise for velocity
            process_noise_acc: Process noise for acceleration
            measurement_noise: Measurement noise (applies to all measured states)
            dt: Time step
            fps: Frames per second
            measure_velocity: If True, use both position and velocity as measurements
        """

        self.dt = dt
        self.fps = fps
        self.measure_velocity = measure_velocity  # NEW

        # Set default noise values if not provided
        if process_noise_pos is None:
            process_noise_pos = self.NOISE_POS_3D
        if process_noise_vel is None:
            process_noise_vel = self.NOISE_VEL_3D
        if process_noise_acc is None:
            process_noise_acc = self.NOISE_ACC_3D
        if measurement_noise is None:
            measurement_noise = self.NOISE_SENSOR_3D

        # Initialize state vector [x, y, z, vx, vy, vz, ax, ay, az]
        self.state = np.array(
            [
                initial_x,
                initial_y,
                initial_z,
                initial_vx,
                initial_vy,
                initial_vz,
                initial_ax,
                initial_ay,
                initial_az,
            ]
        )

        # Initialize estimate error covariance matrix (9x9)
        self.P = np.eye(9)

        # Process noise covariance matrix (9x9)
        self.Q = np.diag(
            [
                process_noise_pos,
                process_noise_pos,
                process_noise_pos,
                process_noise_vel,
                process_noise_vel,
                process_noise_vel,
                process_noise_acc,
                process_noise_acc,
                process_noise_acc,
            ]
        )

        # Measurement matrix H
        if self.measure_velocity:
            # Measure both position and velocity
            self.H = np.zeros((6, 9))
            self.H[0, 0] = 1.0  # x
            self.H[1, 1] = 1.0  # y
            self.H[2, 2] = 1.0  # z
            self.H[3, 3] = 1.0  # vx
            self.H[4, 4] = 1.0  # vy
            self.H[5, 5] = 1.0  # vz
            self.R = np.eye(6) * measurement_noise
        else:
            # Measure only position
            self.H = np.array(
                [
                    [1.0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 1.0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 1.0, 0, 0, 0, 0, 0, 0],
                ]
            )
            self.R = np.eye(3) * measurement_noise

        # Counters
        self.counter_update = 0
        self.counter_notUpdate = 0
        self.frame_last = -0.1

        # Create CasADi functions for prediction and update
        self._create_casadi_functions()

    def _create_casadi_functions(self):
        """Create CasADi functions for prediction and update steps."""

        # State transition matrix A (9x9) for constant acceleration model
        dt_sym = ca.SX.sym("dt", 1)

        # Create A matrix symbolically
        A = ca.SX.zeros(9, 9)

        # Position rows: x = x + vx*dt + 0.5*ax*dt^2
        A[0, 0] = 1.0  # x
        A[0, 3] = dt_sym  # vx
        A[0, 6] = 0.5 * dt_sym * dt_sym  # ax

        A[1, 1] = 1.0  # y
        A[1, 4] = dt_sym  # vy
        A[1, 7] = 0.5 * dt_sym * dt_sym  # ay

        A[2, 2] = 1.0  # z
        A[2, 5] = dt_sym  # vz
        A[2, 8] = 0.5 * dt_sym * dt_sym  # az

        # Velocity rows: vx = vx + ax*dt
        A[3, 3] = 1.0  # vx
        A[3, 6] = dt_sym  # ax

        A[4, 4] = 1.0  # vy
        A[4, 7] = dt_sym  # ay

        A[5, 5] = 1.0  # vz
        A[5, 8] = dt_sym  # az

        # Acceleration rows: ax, ay, az remain constant
        A[6, 6] = 1.0  # ax
        A[7, 7] = 1.0  # ay
        A[8, 8] = 1.0  # az

        # Create prediction function
        state_sym = ca.SX.sym("state", 9)
        P_sym = ca.SX.sym("P", 9, 9)
        Q_sym = ca.SX.sym("Q", 9, 9)

        # Predict next state
        state_pred = ca.mtimes(A, state_sym)

        # Predict next covariance
        P_pred = ca.mtimes(ca.mtimes(A, P_sym), A.T) + Q_sym

        self.predict_fn = ca.Function(
            "predict",
            [state_sym, P_sym, Q_sym, dt_sym],
            [state_pred, P_pred],
            ["state", "P", "Q", "dt"],
            ["state_pred", "P_pred"],
        )

        # Create update function
        # For update, allow H and R to be variable size (3x9 or 6x9, 3x3 or 6x6)
        measurement_sym = ca.SX.sym("measurement", 6)  # max size
        R_sym = ca.SX.sym("R", 6, 6)
        H_sym = ca.SX.sym("H", 6, 9)
        # Kalman gain calculation
        S = ca.mtimes(ca.mtimes(H_sym, P_sym), H_sym.T) + R_sym
        K = ca.mtimes(ca.mtimes(P_sym, H_sym.T), ca.inv(S))
        # Update state
        innovation = measurement_sym - ca.mtimes(H_sym, state_sym)
        state_updated = state_sym + ca.mtimes(K, innovation)
        # Update covariance
        identity_matrix = ca.SX.eye(9)
        P_updated = ca.mtimes(
            ca.mtimes(identity_matrix - ca.mtimes(K, H_sym), P_sym),
            identity_matrix.T,
        ) + ca.mtimes(ca.mtimes(K, R_sym), K.T)
        self.update_fn = ca.Function(
            "update",
            [state_sym, P_sym, H_sym, R_sym, measurement_sym],
            [state_updated, P_updated, K],
            ["state", "P", "H", "R", "measurement"],
            ["state_updated", "P_updated", "K"],
        )

    def predict(self, dframe=None):
        """
        Prediction step of the Kalman filter.

        Args:
            dframe: Frame difference (optional, uses self.dt if None)
        """
        if dframe is None:
            dt = self.dt
        else:
            dt = dframe / self.fps * 1000.0  # Convert to milliseconds

        # Convert to CasADi DM
        state_dm = ca.DM(self.state)
        P_dm = ca.DM(self.P)
        Q_dm = ca.DM(self.Q)
        dt_dm = ca.DM([dt])

        # Call CasADi function
        result = self.predict_fn(state=state_dm, P=P_dm, Q=Q_dm, dt=dt_dm)
        state_pred, P_pred = result["state_pred"], result["P_pred"]

        # Update state and covariance
        self.state = np.array(state_pred).flatten()
        self.P = np.array(P_pred)

        self.counter_notUpdate += 1

    def predict_only(self, dframe=None):
        """
        Prediction step without updating the filter state.

        Args:
            dframe: Frame difference (optional, uses self.dt if None)

        Returns:
            prediction: Predicted state vector
        """
        if dframe is None:
            dt = self.dt
        else:
            dt = dframe / self.fps * 1000.0  # Convert to milliseconds

        # Create state transition matrix A
        A = np.array(
            [
                [1.0, 0, 0, dt, 0, 0, 0.5 * dt * dt, 0, 0],
                [0, 1.0, 0, 0, dt, 0, 0, 0.5 * dt * dt, 0],
                [0, 0, 1.0, 0, 0, dt, 0, 0, 0.5 * dt * dt],
                [0, 0, 0, 1.0, 0, 0, dt, 0, 0],
                [0, 0, 0, 0, 1.0, 0, 0, dt, 0],
                [0, 0, 0, 0, 0, 1.0, 0, 0, dt],
                [0, 0, 0, 0, 0, 0, 1.0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 1.0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 1.0],
            ]
        )

        # Predict next state
        prediction = A @ self.state
        return prediction

    def update(self, measurement):
        """
        Update step of the Kalman filter.

        Args:
            measurement: Measurement vector ([x, y, z] or [x, y, z, vx, vy, vz])
        """
        # Convert to CasADi DM
        state_dm = ca.DM(self.state)
        P_dm = ca.DM(self.P)
        H_dm = ca.DM(self.H)
        R_dm = ca.DM(self.R)
        if self.measure_velocity:
            # Use first 6 elements
            measurement_dm = ca.DM(measurement[:6])
        else:
            measurement_dm = ca.DM(measurement[:3])
        # Call CasADi function (pass full-size measurement, H, R)
        result = self.update_fn(
            state=state_dm,
            P=P_dm,
            H=H_dm,
            R=R_dm,
            measurement=measurement_dm,
        )
        state_updated, P_updated, _ = (
            result["state_updated"],
            result["P_updated"],
            result["K"],
        )
        # Update state and covariance
        self.state = np.array(state_updated).flatten()
        self.P = np.array(P_updated)
        self.counter_notUpdate = 0
        self.counter_update += 1

    def get_state(self):
        """
        Get the current state estimate.

        Returns:
            state: Current state vector [x, y, z, vx, vy, vz, ax, ay, az]
        """
        return self.state.copy()

    def get_position(self):
        """Get current position estimate."""
        return self.state[:3]

    def get_velocity(self):
        """Get current velocity estimate."""
        return self.state[3:6]

    def get_acceleration(self):
        """Get current acceleration estimate."""
        return self.state[6:9]

    def reset(self, new_state=None):
        """
        Reset the filter with a new state.

        Args:
            new_state: New state vector (optional)
        """
        if new_state is not None:
            self.state = np.array(new_state)
        self.P = np.eye(9)
        self.counter_update = 0
        self.counter_notUpdate = 0


def plot_results(
    t_values,
    true_positions,
    estimated_positions,
    true_velocities,
    estimated_velocities,
    true_accelerations,
    estimated_accelerations,
    measurements,
    pos_error,
    vel_error,
    acc_error,
):
    """
    Create comprehensive plots for Kalman filter results.

    Args:
        t_values: Time array
        true_positions: True position data
        estimated_positions: Estimated position data
        true_velocities: True velocity data
        estimated_velocities: Estimated velocity data
        true_accelerations: True acceleration data
        estimated_accelerations: Estimated acceleration data
        measurements: Measurement data
        pos_error: Position error array
        vel_error: Velocity error array
        acc_error: Acceleration error array
    """
    # Create figure with subplots
    fig = plt.figure(figsize=(20, 15))

    # 3D trajectory plot
    ax1 = fig.add_subplot(3, 4, 1, projection="3d")
    ax1.plot(
        true_positions[:, 0],
        true_positions[:, 1],
        true_positions[:, 2],
        "b-",
        label="True Trajectory",
        linewidth=2,
    )
    ax1.plot(
        estimated_positions[:, 0],
        estimated_positions[:, 1],
        estimated_positions[:, 2],
        "r--",
        label="Estimated Trajectory",
        linewidth=2,
    )
    ax1.scatter(
        measurements[:, 0],
        measurements[:, 1],
        measurements[:, 2],
        c="g",
        alpha=0.5,
        s=10,
        label="Measurements",
    )
    ax1.set_xlabel("X")
    ax1.set_ylabel("Y")
    ax1.set_zlabel("Z")
    ax1.set_title("3D Trajectory")
    ax1.legend()

    # Position components vs time
    ax2 = fig.add_subplot(3, 4, 2)
    ax2.plot(t_values, true_positions[:, 0], "b-", label="True X", linewidth=2)
    ax2.plot(
        t_values, estimated_positions[:, 0], "r--", label="Estimated X", linewidth=2
    )
    ax2.scatter(
        t_values, measurements[:, 0], c="g", alpha=0.5, s=10, label="Measurements X"
    )
    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("X Position")
    ax2.set_title("X Position vs Time")
    ax2.legend()

    ax3 = fig.add_subplot(3, 4, 3)
    ax3.plot(t_values, true_positions[:, 1], "b-", label="True Y", linewidth=2)
    ax3.plot(
        t_values, estimated_positions[:, 1], "r--", label="Estimated Y", linewidth=2
    )
    ax3.scatter(
        t_values, measurements[:, 1], c="g", alpha=0.5, s=10, label="Measurements Y"
    )
    ax3.set_xlabel("Time (s)")
    ax3.set_ylabel("Y Position")
    ax3.set_title("Y Position vs Time")
    ax3.legend()

    ax4 = fig.add_subplot(3, 4, 4)
    ax4.plot(t_values, true_positions[:, 2], "b-", label="True Z", linewidth=2)
    ax4.plot(
        t_values, estimated_positions[:, 2], "r--", label="Estimated Z", linewidth=2
    )
    ax4.scatter(
        t_values, measurements[:, 2], c="g", alpha=0.5, s=10, label="Measurements Z"
    )
    ax4.set_xlabel("Time (s)")
    ax4.set_ylabel("Z Position")
    ax4.set_title("Z Position vs Time")
    ax4.legend()

    # Velocity components vs time
    ax5 = fig.add_subplot(3, 4, 5)
    ax5.plot(t_values, true_velocities[:, 0], "b-", label="True Vx", linewidth=2)
    ax5.plot(
        t_values, estimated_velocities[:, 0], "r--", label="Estimated Vx", linewidth=2
    )
    ax5.set_xlabel("Time (s)")
    ax5.set_ylabel("X Velocity")
    ax5.set_title("X Velocity vs Time")
    ax5.legend()

    ax6 = fig.add_subplot(3, 4, 6)
    ax6.plot(t_values, true_velocities[:, 1], "b-", label="True Vy", linewidth=2)
    ax6.plot(
        t_values, estimated_velocities[:, 1], "r--", label="Estimated Vy", linewidth=2
    )
    ax6.set_xlabel("Time (s)")
    ax6.set_ylabel("Y Velocity")
    ax6.set_title("Y Velocity vs Time")
    ax6.legend()

    ax7 = fig.add_subplot(3, 4, 7)
    ax7.plot(t_values, true_velocities[:, 2], "b-", label="True Vz", linewidth=2)
    ax7.plot(
        t_values, estimated_velocities[:, 2], "r--", label="Estimated Vz", linewidth=2
    )
    ax7.set_xlabel("Time (s)")
    ax7.set_ylabel("Z Velocity")
    ax7.set_title("Z Velocity vs Time")
    ax7.legend()

    # Acceleration components vs time
    ax8 = fig.add_subplot(3, 4, 8)
    ax8.plot(t_values, true_accelerations[:, 0], "b-", label="True Ax", linewidth=2)
    ax8.plot(
        t_values,
        estimated_accelerations[:, 0],
        "r--",
        label="Estimated Ax",
        linewidth=2,
    )
    ax8.set_xlabel("Time (s)")
    ax8.set_ylabel("X Acceleration")
    ax8.set_title("X Acceleration vs Time")
    ax8.legend()

    ax9 = fig.add_subplot(3, 4, 9)
    ax9.plot(t_values, true_accelerations[:, 1], "b-", label="True Ay", linewidth=2)
    ax9.plot(
        t_values,
        estimated_accelerations[:, 1],
        "r--",
        label="Estimated Ay",
        linewidth=2,
    )
    ax9.set_xlabel("Time (s)")
    ax9.set_ylabel("Y Acceleration")
    ax9.set_title("Y Acceleration vs Time")
    ax9.legend()

    ax10 = fig.add_subplot(3, 4, 10)
    ax10.plot(t_values, true_accelerations[:, 2], "b-", label="True Az", linewidth=2)
    ax10.plot(
        t_values,
        estimated_accelerations[:, 2],
        "r--",
        label="Estimated Az",
        linewidth=2,
    )
    ax10.set_xlabel("Time (s)")
    ax10.set_ylabel("Z Acceleration")
    ax10.set_title("Z Acceleration vs Time")
    ax10.legend()

    # Error plots
    ax11 = fig.add_subplot(3, 4, 11)
    ax11.plot(t_values, pos_error, "r-", label="Position Error", linewidth=2)
    ax11.set_xlabel("Time (s)")
    ax11.set_ylabel("Position Error")
    ax11.set_title("Position Error vs Time")
    ax11.legend()

    ax12 = fig.add_subplot(3, 4, 12)
    ax12.plot(t_values, vel_error, "g-", label="Velocity Error", linewidth=2)
    ax12.plot(t_values, acc_error, "b-", label="Acceleration Error", linewidth=2)
    ax12.set_xlabel("Time (s)")
    ax12.set_ylabel("Error")
    ax12.set_title("Velocity and Acceleration Error vs Time")
    ax12.legend()

    plt.tight_layout()
    plt.savefig("kalman_filter_results.png", dpi=300, bbox_inches="tight")
    plt.show()


def create_kalman_filter_3d(initial_state=None, **kwargs):
    """
    Factory function to create a 3D Kalman filter.

    Args:
        initial_state: Initial state vector [x, y, z, vx, vy, vz, ax, ay, az]
        **kwargs: Additional arguments for KalmanFilter3D constructor

    Returns:
        KalmanFilter3D instance
    """
    if initial_state is not None:
        return KalmanFilter3D(*initial_state, **kwargs)
    else:
        return KalmanFilter3D(**kwargs)


# Example usage and testing
def example_usage():
    """Example of how to use the 3D Kalman filter."""

    # Debug function with specified trajectory
    def debug_kalman_filter():
        """
        Debug function with trajectory:
        x(t) = 0.1*t
        y(t) = 1.0*t
        z(t) = -4.9*t*(t-5)
        """
        print("=== Kalman Filter Debug with Specified Trajectory ===")

        # Time parameters
        dt = 0.1  # Time step
        t_max = 10.0  # Maximum time
        t_values = np.arange(0, t_max + dt, dt)

        # True trajectory functions
        def true_position(t):
            x = 0.1 * t
            y = 1.0 * t
            z = -4.9 * t * (t - 5)
            return np.array([x, y, z])

        def true_velocity(t):
            vx = 0.1
            vy = 1.0
            vz = -4.9 * (2 * t - 5)
            return np.array([vx, vy, vz])

        def true_acceleration(t):
            ax = 0.0
            ay = 0.0
            az = -9.8  # Constant acceleration due to gravity-like effect
            return np.array([ax, ay, az])

        # Initialize Kalman filter with true initial conditions
        initial_pos = true_position(0)
        initial_vel = true_velocity(0)
        initial_acc = true_acceleration(0)

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

        print(f"Initial true position: {initial_pos}")
        print(f"Initial true velocity: {initial_vel}")
        print(f"Initial true acceleration: {initial_acc}")
        print(f"Initial Kalman state: {kf.get_state()}")
        print()

        # Storage for results
        true_positions = []
        estimated_positions = []
        true_velocities = []
        estimated_velocities = []
        true_accelerations = []
        estimated_accelerations = []
        measurements = []

        # Add noise to measurements
        measurement_noise_std = 0.1

        for i, t in enumerate(t_values):
            # True values
            true_pos = true_position(t)
            true_vel = true_velocity(t)
            true_acc = true_acceleration(t)

            # Add noise to measurement
            noisy_measurement = true_pos + np.random.normal(0, measurement_noise_std, 3)

            # Kalman filter prediction and update
            kf.predict()
            kf.update(noisy_measurement)

            # Store results
            true_positions.append(true_pos)
            estimated_positions.append(kf.get_position())
            true_velocities.append(true_vel)
            estimated_velocities.append(kf.get_velocity())
            true_accelerations.append(true_acc)
            estimated_accelerations.append(kf.get_acceleration())
            measurements.append(noisy_measurement)

            # Print every 10th step
            if i % 10 == 0:
                print(f"Time: {t:.1f}s")
                print(f"  True pos: {true_pos}")
                print(f"  Measured: {noisy_measurement}")
                print(f"  Estimated pos: {kf.get_position()}")
                print(f"  True vel: {true_vel}")
                print(f"  Estimated vel: {kf.get_velocity()}")
                print(f"  True acc: {true_acc}")
                print(f"  Estimated acc: {kf.get_acceleration()}")
                print()

        # Convert to numpy arrays
        true_positions = np.array(true_positions)
        estimated_positions = np.array(estimated_positions)
        true_velocities = np.array(true_velocities)
        estimated_velocities = np.array(estimated_velocities)
        true_accelerations = np.array(true_accelerations)
        estimated_accelerations = np.array(estimated_accelerations)
        measurements = np.array(measurements)

        # Calculate errors
        pos_error = np.linalg.norm(true_positions - estimated_positions, axis=1)
        vel_error = np.linalg.norm(true_velocities - estimated_velocities, axis=1)
        acc_error = np.linalg.norm(true_accelerations - estimated_accelerations, axis=1)

        print("=== Summary ===")
        print(f"Average position error: {np.mean(pos_error):.4f}")
        print(f"Average velocity error: {np.mean(vel_error):.4f}")
        print(f"Average acceleration error: {np.mean(acc_error):.4f}")
        print(f"Max position error: {np.max(pos_error):.4f}")
        print(f"Max velocity error: {np.max(vel_error):.4f}")
        print(f"Max acceleration error: {np.max(acc_error):.4f}")

        # Create plots
        plot_results(
            t_values,
            true_positions,
            estimated_positions,
            true_velocities,
            estimated_velocities,
            true_accelerations,
            estimated_accelerations,
            measurements,
            pos_error,
            vel_error,
            acc_error,
        )

        return {
            "t_values": t_values,
            "true_positions": true_positions,
            "estimated_positions": estimated_positions,
            "true_velocities": true_velocities,
            "estimated_velocities": estimated_velocities,
            "true_accelerations": true_accelerations,
            "estimated_accelerations": estimated_accelerations,
            "measurements": measurements,
            "pos_error": pos_error,
            "vel_error": vel_error,
            "acc_error": acc_error,
        }

    # Run debug function
    results = debug_kalman_filter()

    # Simple test case
    print("\n=== Simple Test Case ===")
    kf = KalmanFilter3D(
        initial_x=0.0,
        initial_y=0.0,
        initial_z=0.0,
        initial_vx=1.0,
        initial_vy=0.5,
        initial_vz=0.0,
        initial_ax=0.1,
        initial_ay=0.0,
        initial_az=0.0,
    )

    print("Initial state:", kf.get_state())

    # Prediction step
    kf.predict()
    print("After prediction:", kf.get_state())

    # Update step with measurement
    measurement = np.array([1.1, 0.6, 0.0])  # Measured position
    kf.update(measurement)
    print("After update:", kf.get_state())

    # Get specific components
    print("Position:", kf.get_position())
    print("Velocity:", kf.get_velocity())
    print("Acceleration:", kf.get_acceleration())

    return kf, results


# if __name__ == "__main__":
#    # Run example
#    kf = example_usage()
