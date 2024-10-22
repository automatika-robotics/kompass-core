import math
from typing import List, Union

import numpy as np
import quaternion
from quaternion import quaternion as quat

from ..datatypes.pose import PoseData
from ..datatypes.laserscan import LaserScanData


def distance(obj_1_x: float, obj_2_x: float, obj_1_y: float, obj_2_y: float) -> float:
    """
    Distance between two points

    :param obj_1_x: x coordinates of object 1
    :type obj_1_x: float
    :param obj_2_x: x coordinates of object 2
    :type obj_2_x: float
    :param obj_1_y: y coordinates of object 1
    :type obj_1_y: float
    :param obj_2_y: y coordinates of object 2
    :type obj_2_y: float

    :return: Distance (m)
    :rtype: float
    """
    return math.sqrt((obj_1_x - obj_2_x) ** 2 + (obj_1_y - obj_2_y) ** 2)


def probability_of_collision(
    obj_1_x: float,
    obj_1_y: float,
    obj_2_x: float,
    obj_2_y: float,
    obj_1_radius: float,
    obj_2_radius: float,
) -> float:
    """
    Computes the probability of collision between two circular objects

    :param obj_1_x: x coordinates of object 1
    :type obj_1_x: float
    :param obj_1_y: y coordinates of object 1
    :type obj_1_y: float
    :param obj_2_x: x coordinates of object 2
    :type obj_2_x: float
    :param obj_2_y: y coordinates of object 2
    :type obj_2_y: float
    :param obj_1_radius: Radius of object 1
    :type obj_1_radius: float
    :param obj_2_radius: Radius of object 2
    :type obj_2_radius: float

    :return: Probability of collision in [0,1]
    :rtype: float
    """
    dist = distance(obj_1_x, obj_2_x, obj_1_y, obj_2_y)
    if dist >= obj_1_radius + obj_2_radius:
        return 0
    if dist < abs(
        obj_1_radius - obj_1_radius
    ):  # one object contained in the other -> sure collision
        return 1
    else:  # compute the intersection space
        area_1 = obj_1_radius**2 * math.acos(
            (dist**2 + obj_1_radius**2 - obj_2_radius**2) / (2 * dist * obj_1_radius)
        )
        area_2 = obj_2_radius**2 * math.acos(
            (dist**2 + obj_2_radius**2 - obj_1_radius**2) / (2 * dist * obj_2_radius)
        )
        area_3 = -0.5 * math.sqrt(
            (dist + obj_1_radius + obj_2_radius)
            * (dist - obj_1_radius + obj_2_radius)
            * (dist + obj_1_radius - obj_2_radius)
            * (-dist + obj_1_radius + obj_2_radius)
        )
        intersection_area = area_1 + area_2 + area_3
        obj_1_area = math.pi * obj_1_radius**2
        prop_col = intersection_area / obj_1_area
        return prop_col


def _rotate_vector_by_quaternion(q: quaternion.quaternion, v: List) -> List:
    """
    rotate a vector v by a rotation quaternion q

    :param      q: the rotation to perform
    :type       q: quaternion.quaternion
    :param      v: the vector to be rotated
    :type       v: List

    :return:    the rotated position of the vector
    :rtype:     List
    """
    vq = quat(0, 0, 0, 0)
    vq.imag = v
    return (q * vq * q.inverse()).imag


def get_pose_target_in_reference_frame(
    reference_pose: PoseData, target_pose: PoseData
) -> PoseData:
    """
    Computes a target pose with respect to the frame (coordinate system) defined by a reference.
    target and reference should be given first in a common frame

    :param      reference_pose:     Pose of reference frame in a common frame
    :type       reference_pose:     PoseData
    :param      target_pose:        Pose of target frame in a common frame
    :type       target_pose:        PoseData

    :return:    The pose of target in the reference frame
    :rtype:     PoseData
    """
    position_target = target_pose.get_position()
    orientation_target = target_pose.get_orientation()

    reference_position = reference_pose.get_position()
    reference_rotation = reference_pose.get_orientation()

    orientation_target_in_ref = reference_rotation.inverse() * orientation_target

    position_target_in_ref = _rotate_vector_by_quaternion(
        reference_rotation.inverse(), (position_target - reference_position).tolist()
    )

    target_pose_in_ref = PoseData()
    target_pose_in_ref.set_pose(
        x=position_target_in_ref[0],
        y=position_target_in_ref[1],
        z=position_target_in_ref[2],
        qw=orientation_target_in_ref.w,
        qx=orientation_target_in_ref.x,
        qy=orientation_target_in_ref.y,
        qz=orientation_target_in_ref.z,
    )

    return target_pose_in_ref


def from_frame1_to_frame2(
    pose_1_in_2: PoseData, pose_target_in_1: PoseData
) -> PoseData:
    """
    get the pose of a target in frame 2 instead of frame 1

    :param      pose_1_in_2:        pose of frame 1 in frame 2
    :type       pose_1_in_2:        PoseData
    :param      pose_target_in_1:   pose of target in frame 1
    :type       pose_target_in_1:   PoseData

    :return:    pose of target in frame 2
    :rtype:     PoseData
    """
    pose_2_origin = PoseData()
    pose_2_in_1 = get_pose_target_in_reference_frame(
        reference_pose=pose_1_in_2, target_pose=pose_2_origin
    )

    pose_target_in_2 = get_pose_target_in_reference_frame(
        reference_pose=pose_2_in_1, target_pose=pose_target_in_1
    )

    return pose_target_in_2


def from_euler_to_quaternion(yaw: float, pitch: float, roll: float) -> np.ndarray:
    """
    Converts Euler angles to a quaternion

    :param yaw: Yaw angle in radians (z rotation)
    :type yaw: float
    :param pitch: Pitch angle in radians
    :type pitch: float
    :param roll: Roll angle in radians
    :type roll: float

    :return: Rotation quaternion
    :rtype: numpy array of quaternion coordinates
    """
    cy = math.cos(yaw / 2)
    sy = math.sin(yaw / 2)
    cp = math.cos(pitch / 2)
    sp = math.sin(pitch / 2)
    cr = math.cos(roll / 2)
    sr = math.sin(roll / 2)

    qw = cy * cp * cr + sy * sp * sr
    qx = cy * cp * sr - sy * sp * cr
    qy = sy * cp * sr + cy * sp * cr
    qz = sy * cp * cr - cy * sp * sr

    return np.array([qw, qx, qy, qz])


def from_2d_to_PoseData(x: float, y: float, heading: float) -> PoseData:
    """
    Converts 2d pose information to PoseData

    :param x: x-coordinates
    :type x: float
    :param y: y-coordinates
    :type y: float
    :param heading: Heading in 2d space (Yaw)
    :type heading: float

    :return: Pose as PoseData
    :rtype: PoseData
    """
    pose_data = PoseData()
    quat_data: quat = from_euler_to_quaternion(heading, 0.0, 0.0)
    pose_data.set_position(x, y, 0.0)
    pose_data.set_orientation(
        qw=quat_data[0], qx=quat_data[1], qy=quat_data[2], qz=quat_data[3]
    )
    return pose_data


def from_frame1_to_frame2_2d(
    position_x_1_in_2: float,
    position_y_1_in_2: float,
    heading_1_in_2: float,
    position_target_x_in_1: float,
    position_target_y_in_1: float,
    heading_target_in_1: float,
) -> PoseData:
    """
    get the pose of a target in frame 2 instead of frame 1

    :param position_x_1_in_2: x-coordinates of frame 1 in frame 2
    :type position_x_1_in_2: float
    :param position_y_1_in_2: y-coordinates of frame 1 in frame 2
    :type position_y_1_in_2: float
    :param heading_1_in_2: Heading of frame 1 in frame 2
    :type heading_1_in_2: float
    :param position_target_x_in_1: x-coordinates of target in frame 1
    :type position_target_x_in_1: float
    :param position_target_y_in_1: y-coordinates of target in frame 1
    :type position_target_y_in_1: float
    :param heading_target_in_1: Heading of target in frame 1
    :type heading_target_in_1: float

    :return:    pose of target in frame 2
    :rtype:     PoseData
    """
    # Parse 2D input pose to PoseDate
    pose_1_in_2: PoseData = from_2d_to_PoseData(
        position_x_1_in_2, position_y_1_in_2, heading_1_in_2
    )

    pose_target_in_1: PoseData = from_2d_to_PoseData(
        position_target_x_in_1, position_target_y_in_1, heading_target_in_1
    )

    # Get transformd pose
    pose_target_in_2: PoseData = from_frame1_to_frame2(pose_1_in_2, pose_target_in_1)

    return pose_target_in_2


def convert_to_plus_minus_pi(ang: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """
    Return angle between [-pi,pi]
    """
    if isinstance(ang, float):
        normalized_angle = (ang + math.pi) % (2 * math.pi) - math.pi
    elif isinstance(ang, np.ndarray):
        normalized_angle = np.vectorize(
            lambda ang_i: (ang_i + math.pi) % (2 * math.pi) - math.pi
        )(ang)

    else:
        raise TypeError("Angle must be float or np.ndarray")
    return normalized_angle


def convert_to_0_2pi(value: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """
    Converts an angle or array of angles to [0,2pi]

    :param value: Input Angle(s) (rad)
    :type value: float | np.ndarray
    :return: Converted Angle(s) (rad)
    :rtype: float | np.ndarray
    """
    value = value % (2 * math.pi)

    if isinstance(value, np.ndarray):
        return _convert_to_0_2pi_array(value)

    if value < 0:
        value += 2 * math.pi
    return value


def _convert_to_0_2pi_array(value_array: np.ndarray) -> np.ndarray:
    """
    Converts an array of angles to [0,2pi]

    :param value_array: Input Angles (rad)
    :type value_array: np.ndarray
    :return:  Converted Angles (rad)
    :rtype: np.ndarray
    """
    for idx, val in enumerate(value_array):
        if val < 0:
            value_array[idx] += 2 * np.pi
    return value_array


def add_angle(angle1: float, angle2: float) -> float:
    """
    Adds two angles

    :param angle1: First input angle (rad)
    :type angle1: float
    :param angle2: Second input angle (rad)
    :type angle2: float
    :return: angles sum (rad)
    :rtype: float
    """
    result = angle1 + angle2
    if result > math.pi:
        result -= 2 * math.pi
    if result <= -math.pi:
        result += 2 * math.pi
    return result


def get_polar_transformation_vector(translation_x: float, translation_y: float) -> list:
    """
    Get a transformation vector in polar coordinates

    :param translation_x: Translation on x-axis
    :type translation_x: float
    :param translation_y: Translation on y-axis
    :type translation_y: float

    :return: Polar transformation [radius, angle]
    :rtype: list
    """
    r_tr = np.sqrt(translation_x**2 + translation_y**2)
    if r_tr > 0:
        ang_tr = np.arccos(translation_x / r_tr)
        return [r_tr, ang_tr]
    return [0.0, 0.0]


def get_transform_polar_coordinates(
    radius: Union[float, np.ndarray],
    angle: Union[float, np.ndarray],
    transf_vec: List[float],
    rotation_angle: float,
) -> tuple[Union[float, np.ndarray], Union[float, np.ndarray]]:
    """
    Get radius transformation in polar coordinates

    :param radius: Given radius in polar coordinates
    :type radius: float
    :param angle: Given angle in polar coordinates
    :type angle: float
    :param transf_vec: transformation vector in polar coordinates [radius_trans, angle_trans]
    :type transf_vec: list

    :return: Transformed radius
    :rtype: float
    """
    radius_transformed_sq = (
        radius**2
        + transf_vec[0] ** 2
        - 2 * radius * transf_vec[0] * np.cos(angle - transf_vec[1])
    )
    radius_new = np.sqrt(radius_transformed_sq)

    angle_new = convert_to_0_2pi(
        convert_to_0_2pi(angle) + convert_to_0_2pi(rotation_angle)
    )

    return (radius_new, angle_new)


def get_laserscan_transformed_polar_coordinates(
    angle_min: float,
    angle_max: float,
    angle_increment: float,
    laser_scan_ranges: np.ndarray,
    max_scan_range: float,
    translation: List[float],
    rotation: List[float],
) -> LaserScanData:
    """
    Transform list of angles and ranges to laserscan data using a given polar transformation

    :param angle_min: Scan min angle (rad)
    :type angle_min: float
    :param angle_max: Scan max angle (rad)
    :type angle_max: float
    :param angle_increment: Scan angle step (rad)
    :type angle_increment: float
    :param laser_scan_ranges: Values of the laser scan along the angles range (m)
    :type laser_scan_ranges: list[float]
    :param max_scan_range: Max range for the scan (m)
    :type max_scan_range: float
    :param trans_vec: Polar translation vector [x, y]
    :type trans_vec: list[float]
    :param rotation_angle: Polar rotation angle (rad)
    :type rotation_angle: float

    :return: Transformed laser scan data
    :rtype: LaserScanData
    """
    angles: np.ndarray = np.arange(
        angle_min, angle_max + angle_increment, angle_increment
    )  # create list of angles

    if len(angles) < len(laser_scan_ranges):
        raise ValueError(
            f"Missing laser scan ranges for angles in [{angle_min}, {angle_max}], got length {len(laser_scan_ranges)} of ranges for {len(angles)} angles"
        )

    angles = angles[: len(laser_scan_ranges)]

    ranges_transformed = np.empty_like(angles)
    angles_transformed = np.empty_like(angles)

    # Limit ranges by max value (to remove inf values)
    r_max = max_scan_range
    ranges: np.ndarray = np.where(
        laser_scan_ranges != np.inf, np.minimum(laser_scan_ranges, r_max), r_max
    )

    trans_vec = get_polar_transformation_vector(
        translation_x=translation[0], translation_y=translation[1]
    )
    rotation_angle = 2 * math.atan2(rotation[2], rotation[3])

    ranges_transformed, angles_transformed = get_transform_polar_coordinates(
        radius=ranges, angle=angles, transf_vec=trans_vec, rotation_angle=rotation_angle
    )

    # Sort values to be compatible with laserscan format
    sorted_indices = np.argsort(angles_transformed)
    if not isinstance(angles_transformed, np.ndarray) or not isinstance(
        ranges_transformed, np.ndarray
    ):
        raise TypeError("Cannot create laser scan data with one value")
    sorted_angles = angles_transformed[sorted_indices]
    sorted_ranges = ranges_transformed[sorted_indices]

    laserscan_transformed = LaserScanData(
        angle_min=min(sorted_angles),
        angle_max=max(sorted_angles),
        angle_increment=angle_increment,
        angles=sorted_angles,
        range_min=min(sorted_ranges),
        range_max=max(sorted_ranges),
        ranges=sorted_ranges,
    )
    return laserscan_transformed
