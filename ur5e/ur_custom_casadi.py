import numpy as np
import casadi as ca


class UrCustomCasadi:
    def __init__(self, robot_type="ur5e"):
        """
        URロボットのCasADi版カスタムクラス

        Parameters
        ----------
        robot_type : str
            ロボットの種類

        Attributes
        ----------
        _a : np.array
            リンクの長さ
        _d : np.array
            リンクのオフセット
        """

        if robot_type == "ur5e":
            self._a = np.array([0, -0.4250, -0.3922, 0, 0, 0])
            self._d = np.array([0.1625, 0, 0, 0.1333, 0.0997, 0.0996])
        elif robot_type == "ur10e":
            self._a = np.array([0, -0.6127, -0.57155, 0, 0, 0])
            self._d = np.array([0.1807, 0, 0, 0.17415, 0.11985, 0.11655])
        elif robot_type == "fr5":
            self._a = np.array([0, -0.425, -0.395, 0, 0, 0])
            self._d = np.array([0.152, 0, 0, 0.102, 0.102, 0.100])
        else:
            print("robot type is not defined")
            exit(-1)

        # Create symbolic variables for joints
        self.q = ca.SX.sym("q", 6)  # joint angles

        # Create symbolic forward kinematics
        self._create_symbolic_fk()

        # Create symbolic Jacobian
        self._create_symbolic_jacobian()

    def _create_symbolic_fk(self):
        """Create symbolic forward kinematics using DH parameters"""

        # Extract joint angles
        q1, q2, q3, q4, q5, q6 = (
            self.q[0],
            self.q[1],
            self.q[2],
            self.q[3],
            self.q[4],
            self.q[5],
        )

        # DH transformation matrices (standard DH convention)
        # T01: base to joint 1
        c1, s1 = ca.cos(q1), ca.sin(q1)
        T01 = ca.vertcat(
            ca.horzcat(c1, 0, s1, 0),
            ca.horzcat(s1, 0, -c1, 0),
            ca.horzcat(0, 1, 0, self._d[0]),
            ca.horzcat(0, 0, 0, 1),
        )

        # T12: joint 1 to joint 2
        c2, s2 = ca.cos(q2), ca.sin(q2)
        T12 = ca.vertcat(
            ca.horzcat(c2, -s2, 0, self._a[1] * c2),
            ca.horzcat(s2, c2, 0, self._a[1] * s2),
            ca.horzcat(0, 0, 1, 0),
            ca.horzcat(0, 0, 0, 1),
        )

        # T23: joint 2 to joint 3
        c3, s3 = ca.cos(q3), ca.sin(q3)
        T23 = ca.vertcat(
            ca.horzcat(c3, -s3, 0, self._a[2] * c3),
            ca.horzcat(s3, c3, 0, self._a[2] * s3),
            ca.horzcat(0, 0, 1, 0),
            ca.horzcat(0, 0, 0, 1),
        )

        # T34: joint 3 to joint 4
        c4, s4 = ca.cos(q4), ca.sin(q4)
        T34 = ca.vertcat(
            ca.horzcat(c4, 0, s4, 0),
            ca.horzcat(s4, 0, -c4, 0),
            ca.horzcat(0, 1, 0, self._d[3]),
            ca.horzcat(0, 0, 0, 1),
        )

        # T45: joint 4 to joint 5
        c5, s5 = ca.cos(q5), ca.sin(q5)
        T45 = ca.vertcat(
            ca.horzcat(c5, 0, -s5, 0),
            ca.horzcat(s5, 0, c5, 0),
            ca.horzcat(0, -1, 0, self._d[4]),
            ca.horzcat(0, 0, 0, 1),
        )

        # T56: joint 5 to joint 6
        c6, s6 = ca.cos(q6), ca.sin(q6)
        T56 = ca.vertcat(
            ca.horzcat(c6, -s6, 0, 0),
            ca.horzcat(s6, c6, 0, 0),
            ca.horzcat(0, 0, 1, self._d[5]),
            ca.horzcat(0, 0, 0, 1),
        )

        # Calculate cumulative transformations
        T02 = ca.mtimes(T01, T12)
        T03 = ca.mtimes(T02, T23)
        T04 = ca.mtimes(T03, T34)
        T05 = ca.mtimes(T04, T45)
        T06 = ca.mtimes(T05, T56)

        # Store transformation matrices
        self.T01 = T01
        self.T02 = T02
        self.T03 = T03
        self.T04 = T04
        self.T05 = T05
        self.T06 = T06

        # Extract end-effector position and orientation
        self.p_eef = T06[:3, 3]  # position
        self.R_eef = T06[:3, :3]  # rotation matrix

        # Create function for forward kinematics
        self.fk_func = ca.Function("fk", [self.q], [self.p_eef, self.R_eef])

    def _create_symbolic_jacobian(self):
        """Create symbolic Jacobian matrix"""

        # Initialize Jacobian matrix
        J = ca.SX.zeros(6, 6)

        # Get end-effector position
        p_eef = self.p_eef

        # Calculate each column of the Jacobian
        for i in range(6):
            # Get transformation matrix for joint i
            if i == 0:
                T_i = ca.SX.eye(4)  # self.T01
            elif i == 1:
                T_i = self.T01  # self.T02
            elif i == 2:
                T_i = self.T02  # self.T03
            elif i == 3:
                T_i = self.T03  # self.T04
            elif i == 4:
                T_i = self.T04  # self.T05
            elif i == 5:
                T_i = self.T05  # self.T06
            else:
                raise ValueError(f"Invalid joint index: {i}")

            # Get joint axis (Z-axis of transformation matrix)
            z_axis = T_i[:3, 2]

            # Get joint position
            p_i = T_i[:3, 3]

            # Calculate vector from joint to end-effector
            r = p_eef - p_i

            # Linear part: cross product of joint axis and r
            J_l = ca.cross(z_axis, r)

            # Angular part: joint axis itself
            J_a = z_axis

            # Combine linear and angular parts
            J_col = ca.vertcat(J_l, J_a)

            # Set column in Jacobian matrix
            J[:, i] = J_col

        self.J = J

        # Create function for Jacobian
        self.jacobian_func = ca.Function("jacobian", [self.q], [self.J])

    def forward_kinematics(self, joints):
        """
        Calculate forward kinematics

        Parameters
        ----------
        joints : np.array
            関節角度

        Returns
        -------
        position : np.array
            エンドエフェクタの位置
        rotation : np.array
            エンドエフェクタの回転行列
        """
        position, rotation = self.fk_func(joints)
        return np.array(position), np.array(rotation)

    def jacobian(self, joints):
        """
        Calculate Jacobian matrix

        Parameters
        ----------
        joints : np.array
            関節角度

        Returns
        -------
        J : np.array
            ヤコビアン行列
        """
        J = self.jacobian_func(joints)
        return np.array(J)

    def inverse_kinematics(
        self, target_position, target_rotation=None, initial_guess=None
    ):
        """
        Calculate inverse kinematics using optimization

        Parameters
        ----------
        target_position : np.array
            目標位置
        target_rotation : np.array, optional
            目標回転行列
        initial_guess : np.array, optional
            初期推定値

        Returns
        -------
        joints : np.array
            関節角度
        """
        # Create optimization problem
        opti = ca.Opti()

        # Decision variables
        q = opti.variable(6)

        # Objective function
        position, rotation = self.fk_func(q)

        # Position error
        pos_error = ca.sumsqr(position - target_position)

        # Rotation error (if specified)
        if target_rotation is not None:
            rot_error = ca.sumsqr(rotation - target_rotation)
            objective = pos_error + 0.1 * rot_error
        else:
            objective = pos_error

        opti.minimize(objective)

        # Joint limits (optional)
        opti.subject_to(q >= -np.pi)
        opti.subject_to(q <= np.pi)

        # Set initial guess
        if initial_guess is not None:
            opti.set_initial(q, initial_guess)

        # Solve
        try:
            sol = opti.solve()
            return sol.value(q)
        except:
            print("Inverse kinematics failed")
            return None

    def get_joint_axes_info(self):
        """Print information about joint axes for verification"""
        print("UR5e Joint Axes Information (CasADi):")
        print("All joints use standard DH convention with Z-axis rotation")
        print("Joint 1: Z-axis rotation (base frame)")
        print("Joint 2: Z-axis rotation (frame 1)")
        print("Joint 3: Z-axis rotation (frame 2)")
        print("Joint 4: Z-axis rotation (frame 3)")
        print("Joint 5: Z-axis rotation (frame 4)")
        print("Joint 6: Z-axis rotation (frame 5)")
        print("\nDH Parameters for UR5e:")
        print("a (link lengths):", self._a)
        print("d (link offsets):", self._d)


if __name__ == "__main__":
    # Test the CasADi implementation
    ur_casadi = UrCustomCasadi()

    # Test joint configuration
    joints = np.array([0.1, 0.3, 0.5, 0.7, 0.9, 1.1])

    # Print joint axes information
    ur_casadi.get_joint_axes_info()

    # Test forward kinematics
    position, rotation = ur_casadi.forward_kinematics(joints)
    print(f"\nEnd-effector position: {position}")
    print(f"End-effector rotation:\n{rotation}")

    # Test Jacobian
    J = ur_casadi.jacobian(joints)
    print(f"\nJacobian matrix:\n{J}")

    # Test inverse kinematics
    target_pos = np.array([-0.5, -0.2, -0.3])
    ik_solution = ur_casadi.inverse_kinematics(target_pos, initial_guess=joints)
    if ik_solution is not None:
        print(f"\nInverse kinematics solution: {ik_solution}")
        # Verify the solution
        pos_check, _ = ur_casadi.forward_kinematics(ik_solution)
        print(f"Position error: {np.linalg.norm(pos_check - target_pos)}")
