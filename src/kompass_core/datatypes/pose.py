from typing import Union

import numpy as np


def _equal_approx(
    v1: np.ndarray,
    v2: np.ndarray,
    is_quaternion: bool = False,
    absolute_tolerance: float = 0.01,
) -> Union[bool, np.bool_]:
    """
    Check if two vectors or quaternions are approximately equal within a tolerance.

    :param v1: First vector or quaternion [w, x, y, z]
    :param v2: Second vector or quaternion
    :param is_quaternion: Set to True if comparing quaternions
    :param absolute_tolerance: Absolute tolerance for comparison
    :return: True if approximately equal
    """
    if is_quaternion:
        # q and -q represent the same rotation
        return np.allclose(v1, v2, rtol=0.0, atol=absolute_tolerance) or np.allclose(
            v1, -v2, rtol=0.0, atol=absolute_tolerance
        )
    else:
        return np.allclose(v1, v2, rtol=0.0, atol=absolute_tolerance)


class PoseData:
    """
    3D representation of a point in space
    attributes:
    x   float
    y   float
    z   float
    qx  float
    qy  float
    qz  float
    qz  float

    position and orientation can be presented as list and quaternion respectively.
    """

    def __init__(self):
        """
        initialize a PoseData instance
        """
        self.x = 0.0
        self.y = 0.0
        self.z = 0.0
        self.qx = 0.0
        self.qy = 0.0
        self.qz = 0.0
        self.qw = 1.0

    def set_position(self, x: float, y: float, z: float):
        """
        set position in 3D

        :param      x:      x-direction
        :type       x:      float
        :param      y:      y-direction
        :type       y:      float
        :param      z:      z-direction
        :type       z:      float
        """
        self.x = x
        self.y = y
        self.z = z

    def set_orientation(self, qw: float, qx: float, qy: float, qz: float):
        """
        set orientation of the 3D pose from quaternion components

        :param      qw:     w component
        :type       qw:     float
        :param      qx:     x component
        :type       qx:     float
        :param      qy:     y component
        :type       qy:     float
        :param      qz:     z component
        :type       qz:     float
        """
        self.qw = qw
        self.qx = qx
        self.qy = qy
        self.qz = qz

    def set_pose(
        self, x: float, y: float, z: float, qw: float, qx: float, qy: float, qz: float
    ):
        """
        set the pose using 3D coordinates and quaternion components
        wrapper for set_position and set_orientation

        :param      x:      x-direction
        :type       x:      float
        :param      y:      y-direction
        :type       y:      float
        :param      z:      z-direction
        :type       z:      float
        :param      qw:     w component
        :type       qw:     float
        :param      qx:     x component
        :type       qx:     float
        :param      qy:     y component
        :type       qy:     float
        :param      qz:     z component
        :type       qz:     float
        """
        self.set_position(x, y, z)
        self.set_orientation(qw, qx, qy, qz)

    def __str__(self):
        """
        string representation of the class.
        """
        representation = f"""position: (x={self.x}, y={self.y}, z={self.z}) - \
orientation: (qw={self.qw}, qx={self.qx}, qy={self.qy}, qz={self.qz})"""
        return representation

    def get_position(self) -> np.ndarray:
        """
        return 3D position as a list

        :return: List of 3D coordinates for the position
        :rtype: List
        """
        return np.array([self.x, self.y, self.z], dtype=np.float32)

    def get_orientation(self) -> np.ndarray:
        """
        Get the orientation represented as quaternion

        :return: orientation represented as quaternion
        :rtype: np.ndarray
        """
        return np.array([self.qw, self.qx, self.qy, self.qz])

    def get_yaw(self) -> np.float64:
        """
        Gets Yaw angle, the orientation in 2D plane (rotation around z-axis)

        :return: Yaw angle
        :rtype: float
        """
        return 2 * np.arctan(self.qz / self.qw)

    def __eq__(self, other) -> bool:
        """
        equivalent to '=='. It compares if self and other PoseData are approximately
        equal

        :param      other:      Pose to compare with
        :type       other:      PoseData

        :return:    are they approximately equal?
        :rtype:     bool
        """
        check_position = _equal_approx(
            v1=self.get_position(),
            v2=other.get_position(),
            is_quaternion=False,
            absolute_tolerance=0.0,
        )
        check_orientation = _equal_approx(
            v1=self.get_orientation(),
            v2=other.get_orientation(),
            is_quaternion=True,
            absolute_tolerance=0.0,
        )

        return bool(check_position and check_orientation)

    def check_approximate_equivalence(self, other, absolute_tolerance=0.01) -> bool:
        """
        equivalent to '=='. It compares if self and other PoseData are approximately
        equal

        :param      other:      Pose to compare with
        :type       other:      PoseData

        :return:    are they approximately equal?
        :rtype:     bool
        """
        check_position = _equal_approx(
            v1=self.get_position(),
            v2=other.get_position(),
            is_quaternion=False,
            absolute_tolerance=absolute_tolerance,
        )
        check_orientation = _equal_approx(
            v1=self.get_orientation(),
            v2=other.get_orientation(),
            is_quaternion=True,
            absolute_tolerance=absolute_tolerance,
        )

        return bool(check_position and check_orientation)
