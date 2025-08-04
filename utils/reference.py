import numpy as np


# make the referential data by approximating the rope as isoceles triangle.
def create_referential_data(length_rope, theta_deg, center_human, pose_eef, ns):
    """
    Create the referential data for each rope point's position and velocity.
    Args:
        length_rope (float): length of the rope. [m]
        theta_deg (float): angle of the isoceless triangle approximatingthe rope. [deg]
        center_human (np.ndarray): center position of the human's turning point.
        pose_eef (np.ndarray): pose of the end-effector.
        ns (int): number of rope points.
    Returns:
        x_ref (np.array): referential data for each rope point's position and velocity. (ns,3)
        vec_human_to_robot (np.array): vector from the human's turning point to the robot's end-effector. rotation axis of the rope. (3,)
    """
    # vector from the human's turning point to the robot's end-effector.
    vec_human_to_robot = pose_eef[:3] - center_human
    vec_human_to_robot = vec_human_to_robot / np.linalg.norm(
        vec_human_to_robot
    )  # unit vector.
    vec_human_to_robot[2] = 0.0  # horizontal vector.

    # 1. Calculate the angle of the rope.
    theta_rad = np.deg2rad(theta_deg)

    # 2. calculate the horizontal rope point's position.
    length_horizon = length_rope * np.sin(theta_rad / 2.0)

    # 3. calculate the vertical rope point's position.
    length_vertical = (length_rope / 2.0) * np.cos(theta_rad / 2.0)

    # 3. vertical point.
    x_ref = np.zeros(
        (ns, 3)
    )  # the middle point of the rope. The order is from human to robot.
    unit_length_horizon = length_horizon / ns  # horizontal unit length of the link.
    unit_length_vertical = length_vertical / (
        ns / 2
    )  # vertical unit length of the link.

    if ns % 2 == 0:  # even number of rope links.
        for i in range(ns):  # for each link.
            # horizontal point.
            x_ref[-i - 1] = (
                center_human + vec_human_to_robot * (i + 0.5) * unit_length_horizon
            )
            # vertical point.
            if i < ns / 2:
                x_ref[-i - 1][2] = center_human[2] - (i + 0.5) * unit_length_vertical
            else:
                x_ref[-i - 1][2] = (
                    center_human[2]
                    - length_vertical
                    + (i - ns / 2 + 0.5) * unit_length_vertical
                )
    else:  # odd number of rope links.
        for i in range(ns):  # for each link.
            # horizontal point.
            x_ref[-i - 1] = (
                center_human + vec_human_to_robot * (i + 0.5) * unit_length_horizon
            )
            # vertical point.
            if i < (ns - 1) / 2:
                x_ref[-i - 1][2] = center_human[2] - (i + 0.5) * unit_length_vertical
            else:
                x_ref[-i - 1][2] = (
                    center_human[2]
                    - length_vertical
                    + (i - ns / 2 + 0.5) * unit_length_vertical
                )

    return x_ref, vec_human_to_robot


def rotate_referential_data(theta_deg, x_ref, center_human, rot_axis):
    """
    Rotate the referential data around the rotation axis.
    Args:
        theta_deg (float): angle of the rotation. [deg]
        x_ref (np.array): referential data for each rope point's position and velocity. (ns,3)
        center_human (np.array): center position of the human's turning point. (3,)
        rot_axis (np.array): rotation axis. (3,)
    Returns:
        x_ref_rotated (np.array): rotated referential data by theta_deg. (ns,3)
    """
    # ensure the rotation axis is a unit vector.
    rot_axis[2] = 0.0  # horizontal vector.
    rot_axis = rot_axis / (np.linalg.norm(rot_axis) + 1e-10)
    # convert the angle to radian.
    theta_rad = np.deg2rad(theta_deg)
    x_ref_rotated = x_ref.copy()
    for i in range(
        1, x_ref.shape[0] - 1
    ):  # for each rope point except the first (robot) and last (human).
        vec_human_to_ref = x_ref[i] - center_human
        vec_human_to_ref = vec_human_to_ref / np.linalg.norm(vec_human_to_ref)
        # using the Rodrigues' rotation formula is appropriate for rotating a vector around an arbitrary axis.
        # The formula is: v_rot = v*cos(theta) + (k x v)*sin(theta) + k*(k·v)*(1-cos(theta))
        # where v is the vector to rotate, k is the unit rotation axis, and theta is the rotation angle.
        # Your code below implements this correctly.
        # inverse rotation.
        vec_human_to_ref = (
            vec_human_to_ref * np.cos(-theta_rad)
            + np.cross(rot_axis, vec_human_to_ref) * np.sin(-theta_rad)
            + rot_axis * np.dot(rot_axis, vec_human_to_ref) * (1 - np.cos(-theta_rad))
        )
        x_ref_rotated[i] = center_human + vec_human_to_ref
    return x_ref_rotated
