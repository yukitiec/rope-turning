import numpy as np


class Human:
    def __init__(self,rope_length=None):
        # Rope setting.
        # mass. general long rope mass : 5.67e-4 kg/cm -> 30m-1.7 kg
        # 30m-0.6 kg~3.6 kg.
        self._m_rope_lb = 2.0e-4  # Upper bound for the link mass
        self._m_rope_ub = 1.2e-3  # upper bound for the link mass
        # length
        self._l_rope_lb = 0.8  # Upper bound for the rope length
        self._l_rope_ub = 3.0  # upper bound for the rope length

        # Human motion setting.
        self._rate_h2r_lb = 0.6
        self._rate_h2r_ub = 0.8
        self._z_lb = 0.4  # lower bound for the human's initial position
        self._z_ub = 0.9  # upper bound for the human's initial position
        self._r_min_turn = 0.1  # lower bound for the human's turning radius.
        self._r_max_turn = 0.75  # upper bound for the human's turning radius.
        self._zmin_rob = (
            self._z_lb - self._l_rope_ub * self._r_min_turn - 0.1
        )  # minimum Z for the robot.
        self._zmax_rob = (
            self._z_ub + self._l_rope_ub * self._r_min_turn + 0.1
        )  # maximum z for the robot.
        self._v_hum_ub = 10.0  # upper bound for the human speed. 15 m/s -> 48 km/h
        self._v_hum_lb = 0.5  # lower bound for the human speed. 0.5 m/s -> 1.6 km/h
        self._omega_hum_ub = (
            4.0 * np.pi
        )  # upper bound for the human's angular velocity.
        self._omega_hum_lb = (
            2.0 * np.pi
        )  # lower bound for the human's angular velocity.
        self._n_max_turn = (
            1.5  # max turning circuits per second. 5 circuits per second.
        )
        # change human motion
        self._n_circuit_update = 10  # 10 circuits. How often do we update the turning conditions in rope turning?

        # human wrist&s motion.
        self._idx_human_motion = 1
        # 1. epsilon-greedy method.
        self._epsilon_motion = 0.1  # 0.9 : ordinary speed. 0.1 : abnormal speed.
        self._variance_motion = 0.1  # +- 0.1*velocity.

        # 2. downward : accelerate, upward : deccelerate.
        self._variance_acc = 0.1  # acceleration variaon. +-0.1

    def create_motion(self, pose_eef, rope_length=None):
        """
        Create the human's motion.

        Args:
        ----------
        pose_eef : np.ndarray (6,)
            The initial pose of the end-effector.

        Return:
        ----------
        self.moving_point_center,self.samples_turn, self.deg_per_sec, self.vel_samples_turn
            The center position of the human's turning point.
        samples_turn : np.ndarray (360,3)
            The samples of the human's turning point.
        deg_per_sec : int
            The degree per second of the human's turning speed. [deg/s]
        vel_samples_turn : np.ndarray (360,)
            The velocity of the human's turning speed. [deg/s]
        """

        #### Rope settings.
        # 1. Rope mass
        self.link_mass = np.random.uniform(self._m_rope_lb, self._m_rope_ub)  # kg/link

        # 2. Rope length
        if rope_length is None:
            self.rope_length = np.random.uniform(self._l_rope_lb, self._l_rope_ub)
        else:
            self.rope_length = rope_length
        self.link_length = 0.01  # each link's length is 1 cm.
        self.num_links = int(self.rope_length / self.link_length)
        ##################

        ## Human motion settings.
        # step1. Get the robot's end-effector position.
        pos_end_effector = pose_eef[:3].copy()

        # step2 calculate the distance between human and robot.
        low = self._rate_h2r_lb * self.rope_length
        high = self._rate_h2r_ub * self.rope_length
        self.R_r2h = np.random.uniform(low, high)
        self.dist_r2h_current = self.R_r2h

        # step3. decide the human's turning center
        self.moving_point_center = [0, 0, 0]  # 0.6
        vec_r2h = np.ones(3)
        while (
            vec_r2h[0] >= 0.0
            or self.moving_point_center[2] < self._z_lb
            or self._z_ub < self.moving_point_center[2]
        ):
            vec_r2h = np.array(
                [
                    np.random.uniform(-1, 0.0),
                    np.random.uniform(-1, 1),
                    np.random.uniform(-0.1, 0.1),
                ]
            )
            vec_r2h = vec_r2h / np.linalg.norm(vec_r2h + 1e-10)  # normalize the vector.
            self.moving_point_center = pos_end_effector + self.R_r2h * vec_r2h

        # step4. Determine the human movement radius.
        zmin = self._zmin_rob
        zmax = self._zmax_rob
        self.moving_point_radius = np.random.uniform(
            self._r_min_turn * self.rope_length, self._r_max_turn * self.rope_length
        )  # get the randomized turning radius.
        while zmin <= self._zmin_rob or self._zmax_rob <= zmax:  # not meet the criteia.
            self.moving_point_radius = np.random.uniform(
                self._r_min_turn * self.rope_length, self._r_max_turn * self.rope_length
            )  # get the randomized turning radius.
            zmin = self.moving_point_center[2] - self.moving_point_radius
            zmax = self.moving_point_center[2] + self.moving_point_radius

        # step5. Human's turning point sampling.
        # calculate the unit vectors in the turning plane.
        vec_r2h_plane = vec_r2h.copy()  # remove z-axis element.
        vec_r2h_plane[2] = 0.0
        vec_r2h_plane /= np.linalg.norm(vec_r2h_plane)
        u1, u2 = self.compute_orthogonal_vectors(vec_r2h_plane)

        # prepare all samples in the circular trajectory. (360,3(x,y,z))
        self.samples_turn = np.array(
            [
                [
                    self.moving_point_center[j]
                    + self.moving_point_radius
                    * (
                        np.cos(i / 180 * np.pi) * u1[j]
                        + np.sin(i / 180 * np.pi) * u2[j]
                    )
                    for j in range(3)
                ]
                for i in range(360)
            ]
        )  # sample from the every angle.
        self.target_position = self.samples_turn[0]
        self.target_angle = 0.0
        self.counter_step = 1

        # step6. human's turning speed
        omega_h = np.random.uniform(self._omega_hum_lb, self._omega_hum_ub)
        self.deg_per_sec = min(
            int(360 * self._n_max_turn), omega_h * 180 / np.pi
        )  # [degree/s]

        # for motion-policy 2, make the corresponding velocity list.
        # 1. check the position when the sign for the change in z-ccordinate switches.
        idx_switch = 0
        for i in range(1, self.samples_turn.shape[0]):  # for each samples.
            dz_past = (
                self.samples_turn[i][2] - self.samples_turn[i - 1][2]
            )  # z-coordinate
            dz_next = (
                self.samples_turn[(i + 1) % self.samples_turn.shape[0]][2]
                - self.samples_turn[i][2]
            )
            if dz_past >= 0.0:  # change in deltaz. positive
                if dz_next <= 0.0:  # non-positive
                    idx_switch = i
                    break

        # idx_switch is the baseline to determine the velocity.
        idx_st_accelerate = idx_switch  # + 90  # the slowes point
        idx_end_accelerate = (idx_st_accelerate + 179) % 360  # the fastest point.
        omega_max = self.deg_per_sec * (1.0 + self._variance_acc)  # [degree/s]
        omega_min = self.deg_per_sec * (1.0 - self._variance_acc)  # [degree/s]
        self.vel_samples_turn = omega_min * np.ones(
            360
        )  # [deg/s]. make an array with (360,)
        acc_per_angle = (omega_max - omega_min) / 180.0
        bool_accelerate = True
        counter = 0
        idx = idx_st_accelerate

        # make self.vel_samples_turn. start from the slowest point.
        samples_turn_adjusted = (
            self.samples_turn.copy()
        )  # 1st element is the beginning of the slowest point.
        while counter < self.samples_turn.shape[0]:  # for each samples.
            # iterator : counter, idx.
            if counter == 180:
                bool_accelerate = False
            if bool_accelerate:  # accelerating zone
                samples_turn_adjusted[counter % self.samples_turn.shape[0]] = (
                    self.samples_turn[idx]
                )
                self.vel_samples_turn[counter % self.samples_turn.shape[0]] = (
                    self.vel_samples_turn[max(0, counter - 1)] + acc_per_angle
                )  # increment by acc_per_angle.
            else:  # deccelerating zone.
                samples_turn_adjusted[counter % self.samples_turn.shape[0]] = (
                    self.samples_turn[idx]
                )
                self.vel_samples_turn[counter % self.samples_turn.shape[0]] = (
                    self.vel_samples_turn[max(0, counter - 1)] - acc_per_angle
                )  # increment by acc_per_angle.
            # increment counter
            counter += 1
            idx = (idx + 1) % 360

        # adjust the start point of the samples_turn.
        self.samples_turn = samples_turn_adjusted.copy()
        self.target_position = self.samples_turn[0]

        return (
            self.moving_point_center,
            self.samples_turn,
            self.deg_per_sec,
            self.vel_samples_turn,
        )

    def compute_orthogonal_vectors(self, v):
        """Calculate the orghogonal two vectors from the norm vecotr.

        Args:
            v (numpy.ndarray): norm vector

        Returns:
            u1, u2: two orthogonal vecotrs in the perpendicular plane to v.
        """
        v = np.asarray(v, dtype=float)
        v = v / np.linalg.norm(v)  # Normalize input vector

        # Choose a vector not parallel to v
        if abs(v[0]) < abs(v[1]) and abs(v[0]) < abs(v[2]):
            temp = np.array([1.0, 0.0, 0.0])
        elif abs(v[1]) < abs(v[2]):
            temp = np.array([0.0, 1.0, 0.0])
        else:
            temp = np.array([0.0, 0.0, 1.0])

        # First orthogonal vector (cross product)
        u1 = np.cross(v, temp)
        u1 /= np.linalg.norm(u1)  # Normalize

        # Second orthogonal vector (cross product)
        u2 = np.cross(v, u1)
        u2 /= np.linalg.norm(u2)  # Normalize

        return u1, u2
