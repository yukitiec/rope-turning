import numpy as np
import cv2


class UrCustom:
    def __init__(self, robot_type="ur5e"):
        """
        URロボットのカスタムクラス

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
            self._alpha = np.array([np.pi / 2.0, 0, 0, np.pi / 2, -np.pi / 2, 0])
        elif robot_type == "ur10e":
            self._a = np.array([0, -0.6127, -0.57155, 0, 0, 0])
            self._d = np.array([0.1807, 0, 0, 0.17415, 0.11985, 0.11655])
            self._alpha = np.array([np.pi / 2.0, 0, 0, np.pi / 2, -np.pi / 2, 0])
        else:
            print("robot type is not defined")
            exit(-1)

        # Initialize
        self._T01 = np.array([])
        self._T02 = np.array([])
        self._T03 = np.array([])
        self._T04 = np.array([])
        self._T05 = np.array([])
        self._T06 = np.array([])
        # link transformation matrix
        self._T12 = np.array([])
        self._T23 = np.array([])
        self._T34 = np.array([])
        self._T45 = np.array([])
        self._T56 = np.array([])

        # jacobian
        self.J = np.array([])

    def dh(self, theta, a, d, alpha):
        """DH parameter-based transformation matrix"""
        t00 = np.cos(theta)
        t01 = -1.0 * np.sin(theta) * np.cos(alpha)
        t02 = np.sin(theta) * np.sin(alpha)
        t03 = a * np.cos(theta)

        t10 = np.sin(theta)
        t11 = np.cos(theta) * np.cos(alpha)
        t12 = -1.0 * np.cos(theta) * np.sin(alpha)
        t13 = a * np.sin(theta)

        t20 = 0
        t21 = np.sin(alpha)
        t22 = np.cos(alpha)
        t23 = d

        return np.array(
            [
                [t00, t01, t02, t03],
                [t10, t11, t12, t13],
                [t20, t21, t22, t23],
                [0, 0, 0, 1],
            ]
        )

    def _mat01(self, theta1):
        self._T01 = self.dh(
            theta=theta1, a=self._a[0], d=self._d[0], alpha=self._alpha[0]
        )

    def _mat12(self, theta2):
        self._T12 = self.dh(
            theta=theta2, a=self._a[1], d=self._d[1], alpha=self._alpha[1]
        )

    def _mat23(self, theta3):
        self._T23 = self.dh(
            theta=theta3, a=self._a[2], d=self._d[2], alpha=self._alpha[2]
        )

    def _mat34(self, theta4):
        self._T34 = self.dh(
            theta=theta4, a=self._a[3], d=self._d[3], alpha=self._alpha[3]
        )

    def _mat45(self, theta5):
        self._T45 = self.dh(
            theta=theta5, a=self._a[4], d=self._d[4], alpha=self._alpha[4]
        )

    def _mat56(self, theta6):
        self._T56 = self.dh(
            theta=theta6, a=self._a[5], d=self._d[5], alpha=self._alpha[5]
        )

    def mat(self, joints):
        self._mat01(joints[0])
        self._mat12(joints[1])
        self._mat23(joints[2])
        self._mat34(joints[3])
        self._mat45(joints[4])
        self._mat56(joints[5])

    def _cal_pose1(self, joints, bool_mat):
        if bool_mat:
            self.mat(joints)

        n1, _ = cv2.Rodrigues(
            self._T01[:3, :3]
        )  # convert rotation matrix to roattion vector with Rodrigues formula.

        angle = self._norm_rotVec(n1)

        self.pose1 = np.array(
            [
                self._T01[0, 3],
                self._T01[1, 3],
                self._T01[2, 3],
                n1[0][0] / angle,
                n1[1][0] / angle,
                n1[2][0] / angle,
            ]
        )  # (x,y,z,rx,ry,rz)

    def _cal_pose2(self, joints, bool_mat):
        if bool_mat:
            self.mat(joints)

        self._T02 = self._T01 @ self._T12
        n2, _ = cv2.Rodrigues(
            self._T02[:3, :3]
        )  # convert rotation matrix to roattion vector with Rodrigues formula.

        angle = self._norm_rotVec(n2)

        self.pose2 = np.array(
            [
                self._T02[0, 3],
                self._T02[1, 3],
                self._T02[2, 3],
                n2[0][0] / angle,
                n2[1][0] / angle,
                n2[2][0] / angle,
            ]
        )

    def _cal_pose3(self, joints, bool_mat):
        if bool_mat:
            self.mat(joints)

        self._T03 = self._T01 @ self._T12 @ self._T23
        n3, _ = cv2.Rodrigues(
            self._T03[:3, :3]
        )  # convert rotation matrix to roattion vector with Rodrigues formula.

        angle = self._norm_rotVec(n3)

        self.pose3 = np.array(
            [
                self._T03[0, 3],
                self._T03[1, 3],
                self._T03[2, 3],
                n3[0][0] / angle,
                n3[1][0] / angle,
                n3[2][0] / angle,
            ]
        )

    def _cal_pose4(self, joints, bool_mat):
        if bool_mat:
            self.mat(joints)

        self._T04 = self._T01 @ self._T12 @ self._T23 @ self._T34
        n4, _ = cv2.Rodrigues(
            self._T04[:3, :3]
        )  # convert rotation matrix to roattion vector with Rodrigues formula.

        angle = self._norm_rotVec(n4)

        self.pose4 = np.array(
            [
                self._T04[0, 3],
                self._T04[1, 3],
                self._T04[2, 3],
                n4[0][0] / angle,
                n4[1][0] / angle,
                n4[2][0] / angle,
            ]
        )

    def _cal_pose5(self, joints, bool_mat):
        if bool_mat:
            self.mat(joints)

        self._T05 = self._T01 @ self._T12 @ self._T23 @ self._T34 @ self._T45
        n5, _ = cv2.Rodrigues(
            self._T05[:3, :3]
        )  # convert rotation matrix to roattion vector with Rodrigues formula.

        angle = self._norm_rotVec(n5)
        self.pose5 = np.array(
            [
                self._T05[0, 3],
                self._T05[1, 3],
                self._T05[2, 3],
                n5[0][0] / angle,
                n5[1][0] / angle,
                n5[2][0] / angle,
            ]
        )

    def _cal_pose6(self, joints, bool_mat):
        if bool_mat:
            self.mat(joints)

        self._T06 = (
            self._T01 @ self._T12 @ self._T23 @ self._T34 @ self._T45 @ self._T56
        )
        n6, _ = cv2.Rodrigues(
            self._T06[:3, :3]
        )  # convert rotation matrix to roattion vector with Rodrigues formula.

        angle = self._norm_rotVec(n6)

        self.pose6 = np.array(
            [
                self._T06[0, 3],
                self._T06[1, 3],
                self._T06[2, 3],
                n6[0][0] / angle,
                n6[1][0] / angle,
                n6[2][0] / angle,
            ]
        )

    def _norm_rotVec(self, n):
        return (n[0][0] ** 2 + n[1][0] ** 2 + n[2][0] ** 2 + 1e-16) ** (0.5)

    def cal_pose_all(self, joints):
        self.mat(joints)

        self._cal_pose1(joints, False)
        self._cal_pose2(joints, False)
        self._cal_pose3(joints, False)
        self._cal_pose4(joints, False)
        self._cal_pose5(joints, False)
        self._cal_pose6(joints, False)

    def cal_J(self, joints):
        """
        ヤコビアンを求める

        Parameters
        ----------
        joints : np.array
            関節角度

        Attributes
        ----------
        J : np.array
            ヤコビアン
        """

        # calculate forward kinematics first.
        self.cal_pose_all(joints)
        print(f"Estimated end-effector pose : {self.pose6}")

        # Get end-effector position
        p_eef = self._T06[:3, 3]

        # Calculate each column of the Jacobian using the correct joint axes
        J0 = self.cal_jacobian_column_fixed(np.eye(4), p_eef, 0)  # 1st column
        J1 = self.cal_jacobian_column_fixed(self._T01, p_eef, 1)  # 1st column
        J2 = self.cal_jacobian_column_fixed(self._T02, p_eef, 2)  # 2nd column
        J3 = self.cal_jacobian_column_fixed(self._T03, p_eef, 3)  # 3rd column
        J4 = self.cal_jacobian_column_fixed(self._T04, p_eef, 4)  # 4th column
        J5 = self.cal_jacobian_column_fixed(self._T05, p_eef, 5)  # 5th column
        # J6 = self.cal_jacobian_column_fixed(self._T06, p_eef, 5)  # 6th column

        J = np.column_stack((J0, J1, J2, J3, J4, J5))  # Shape: (6,6)
        # J = np.column_stack((J1, J2, J3, J4, J5, J6))  # Shape: (6,6)
        self.J = J.copy()

    def cal_jacobian_column_fixed(self, T_i, p_eef, joint_idx):
        """Calculate the i-th column of the jacobian using the correct joint axis in base frame

        Parameters:
            T_i : (numpy.ndarray)
                transformation matrix from base to joint i
            p_eef : (numpy.ndarray)
                end-effector position
            joint_idx : (int)
                joint index (0-5)
        """
        # For standard DH convention, all joints rotate around Z-axis
        # The joint axis is the Z-axis of the transformation matrix from base to that joint

        # Get the joint axis in the base frame (Z-axis of the transformation matrix)
        z_axis = T_i[:3, 2]  # Z-axis of the transformation matrix

        # Get the position of joint i
        p_i = T_i[:3, 3]

        # Calculate the vector from joint i to end-effector
        r = p_eef - p_i

        # Linear part: cross product of joint axis and r
        J_l = np.cross(z_axis, r)

        # Angular part: joint axis itself
        J_a = z_axis.copy()

        # Combine linear and angular parts
        J_col = np.hstack((J_l, J_a))  # (6,)

        return J_col

    def cal_jacobian_column(self, P):
        """Calculate the col-th column of the jacobian using b an r

        Parameters:
            P : (numpy.ndarray)
                transformation matrix from base to joint i
        """
        # Get the joint axis in the base frame (Z-axis of the transformation matrix)
        b = P[:3, 2]  # z-th column of rotation matrix

        # Get the position of joint i
        p_i = P[:3, 3]  # translation element

        # Use consistent end-effector position from transformation matrix
        p_eef = self._T06[:3, 3]  # end-effector position from T06

        # Calculate the vector from joint i to end-effector
        r = p_eef - p_i

        # Linear part: cross product of joint axis and r
        J_l = np.cross(b, r)

        # Angular part: joint axis itself
        J_a = b.copy()

        # Combine linear and angular parts
        J_col = np.hstack((J_l, J_a))  # (6,)

        return J_col

    def get_joint_axes_info(self):
        """Print information about joint axes for verification"""
        print("UR5e Joint Axes Information:")
        print("Joint 1: Z-axis rotation (base frame)")
        print("Joint 2: Y-axis rotation (frame 1)")
        print("Joint 3: Y-axis rotation (frame 2)")
        print("Joint 4: Z-axis rotation (frame 3)")
        print("Joint 5: Y-axis rotation (frame 4)")
        print("Joint 6: Z-axis rotation (frame 5)")
        print("\nDH Parameters for UR5e:")
        print("a (link lengths):", self._a)
        print("d (link offsets):", self._d)

    def cal_jacobian01(self, joints):
        px = self.pose6[0]
        py = self.pose6[1]
        pz = self.pose6[2]

        self.J10 = self._jacobian(
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, px, py, pz, 1
        )

    def cal_jacobian02(self, joints):
        s1 = np.sin(joints[0])
        c1 = np.cos(joints[0])
        px = self.pose6[0]
        py = self.pose6[1]
        pz = self.pose6[2]

        self.J02 = self._jacobian(
            s1, c1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, px, py, pz, 2
        )

    def cal_jacobian03(self, joints):
        s1 = np.sin(joints[0])
        c1 = np.cos(joints[0])
        s23 = np.sin(joints[1] + joints[2])
        c23 = np.cos(joints[1] + joints[2])
        s234 = np.sin(joints[1] + joints[2] + joints[3])
        c234 = np.cos(joints[1] + joints[2] + joints[3])
        s5 = np.sin(joints[4])
        c5 = np.cos(joints[4])
        px = self.pose6[0]
        py = self.pose6[1]
        pz = self.pose6[2]

        self.J03 = self._jacobian(
            s1, c1, s23, c23, s234, c234, s5, c5, 0.0, 0.0, 0.0, px, py, pz, 3
        )

    def cal_jacobian04(self, joints):
        s1 = np.sin(joints[0])
        c1 = np.cos(joints[0])
        s23 = np.sin(joints[1] + joints[2])
        c23 = np.cos(joints[1] + joints[2])
        s234 = np.sin(joints[1] + joints[2] + joints[3])
        c234 = np.cos(joints[1] + joints[2] + joints[3])
        s5 = np.sin(joints[4])
        c5 = np.cos(joints[4])
        px = self._T06[0, 3]
        py = self._T06[1, 3]
        pz = self._T06[2, 3]

        self.J04 = self._jacobian(
            s1, c1, s23, c23, s234, c234, s5, c5, 0.0, 0.0, 0.0, px, py, pz, 4
        )

    def cal_jacobian05(self, joints):
        s1 = np.sin(joints[0])
        c1 = np.cos(joints[0])
        s23 = np.sin(joints[1] + joints[2])
        c23 = np.cos(joints[1] + joints[2])
        s234 = np.sin(joints[1] + joints[2] + joints[3])
        c234 = np.cos(joints[1] + joints[2] + joints[3])
        s5 = np.sin(joints[4])
        c5 = np.cos(joints[4])
        px = self.pose6[0]
        py = self.pose6[1]
        pz = self.pose6[2]

        self.J05 = self._jacobian(
            s1, c1, s23, c23, s234, c234, s5, c5, 0.0, 0.0, 0.0, px, py, pz, 5
        )

    def cal_jacobian(self, joints):
        """
        ヤコビアンを求める

        Parameters
        ----------
        joints : np.array
            関節角度

        Attributes
        ----------
        J : np.array
            ヤコビアン
        """

        s1 = np.sin(joints[0])
        c1 = np.cos(joints[0])
        s23 = np.sin(joints[1] + joints[2])
        c23 = np.cos(joints[1] + joints[2])
        s234 = np.sin(joints[1] + joints[2] + joints[3])
        c234 = np.cos(joints[1] + joints[2] + joints[3])
        s5 = np.sin(joints[4])
        c5 = np.cos(joints[4])
        r13 = self._T06[0, 2]
        r23 = self._T06[1, 2]
        r33 = self._T06[2, 2]
        px = self._T06[0, 3]
        py = self._T06[1, 3]
        pz = self._T06[2, 3]

        self.J = self._jacobian(
            s1, c1, s23, c23, s234, c234, s5, c5, r13, r23, r33, px, py, pz, 6
        )

    def _jacobian(
        self, s1, c1, s23, c23, s234, c234, s5, c5, r13, r23, r33, px, py, pz, index
    ):
        j00 = -py
        j10 = px
        j20 = 0.0
        j30 = 0.0
        j40 = 0.0
        j50 = 1.0
        j01 = -c1 * (pz - self._d[0])
        j11 = -s1 * (pz - self._d[0])
        j21 = s1 * py + c1 * px
        j31 = s1
        j41 = -c1
        j51 = 0.0
        j02 = c1 * (s234 * s5 * self._d[5] + c234 * self._d[4] - s23 * self._a[2])
        j12 = s1 * (s234 * s5 * self._d[5] + c234 * self._d[4] - s23 * self._a[2])
        j22 = -c234 * s5 * self._d[5] + s234 * self._d[4] + c23 * self._a[2]
        j32 = s1
        j42 = -c1
        j52 = 0.0
        j03 = c1 * (s234 * s5 * self._d[5] + c234 * self._d[4])
        j13 = s1 * (s234 * s5 * self._d[5] + c234 * self._d[4])
        j23 = -c234 * s5 * self._d[5] + s234 * self._d[4]
        j33 = s1
        j43 = -c1
        j53 = 0.0
        j04 = -self._d[5] * (s1 * s5 + c1 * c234 * c5)
        j14 = self._d[5] * (c1 * s5 - s1 * c234 * c5)
        j24 = -c5 * s234 * self._d[5]
        j34 = c1 * s234
        j44 = s1 * s234
        j54 = -c234
        j05 = 0.0
        j15 = 0.0
        j25 = 0.0
        j35 = r13
        j45 = r23
        j55 = r33

        j = np.array(
            [
                [j00, j01, j02, j03, j04, j05],
                [j10, j11, j12, j13, j14, j15],
                [j20, j21, j22, j23, j24, j25],
                [j30, j31, j32, j33, j34, j35],
                [j40, j41, j42, j43, j44, j45],
                [j50, j51, j52, j53, j54, j55],
            ]
        )

        return j[:, :index]

    def determinant(self, j):
        """
        ヤコビアンの行列式を求める

        Parameters
        ----------
        j : np.array
            ヤコビアン

        Returns
        -------
        det : float
            行列式
        """

        if j.shape[0] == j.shape[1]:
            return np.linalg.det(j)

        jT = np.transpose(j)
        jTj = jT @ j

        return np.sqrt(np.linalg.det(jTj))


if __name__ == "__main__":
    ur = UrCustom()
    joints = np.array([0.1, 0.3, 0.5, 0.7, 0.9, 1.1])

    # Print joint axes information
    ur.get_joint_axes_info()

    # Calculate forward kinematics first
    ur.cal_pose_all(joints)

    # Debug: Print transformation matrices and joint axes
    # print(f"\nT01:\n{ur._T01}")
    # print(f"\nT02:\n{ur._T02}")
    # print(f"\nT06:\n{ur._T06}")

    # Debug: Print joint axes
    z1 = np.array([0, 0, 1])
    z2 = ur._T01[:3, :3] @ np.array([0, 0, 1])
    z3 = ur._T02[:3, :3] @ np.array([0, 0, 1])
    # print(f"\nJoint 1 axis (base): {z1}")
    # print(f"Joint 2 axis (base): {z2}")
    # print(f"Joint 3 axis (base): {z3}")

    # Test original column-based Jacobian
    ur.cal_J(joints)
    J_column_based = ur.J.copy()
    print(f"\nColumn-based Jacobian:\n{ur.J}")

    # Test analytical Jacobian
    ur.cal_jacobian(joints)
    J_analytical = ur.J.copy()
    print(f"\nAnalytical Jacobian:\n{ur.J}")

    # Calculate difference
    diff_J = J_column_based - J_analytical
    diff_J = diff_J < 1e-3
    print(f"\nDifference between column-based and analytical Jacobian:\n{diff_J}")
    # print(f"\nMax difference: {np.max(np.abs(diff_J))}")
    # print(f"Mean difference: {np.mean(np.abs(diff_J))}")
