import math
from typing import Union
import numpy as np
from attrs import define, field

from ..utils.common import BaseAttrs, base_validators


@define
class LaserScanData(BaseAttrs):
    """
    Single scan from a planar laser range-finder (LiDAR)

    attributes:
    angle_min           float32         start angle of the scan [rad]
    angle_max           float32         end angle of the scan [rad]
    angle_increment     float32         angular distance between measurements [rad]

    time_increment      float32         time between measurements [seconds] - if your scanner
                                        is moving, this will be used in interpolating position of 3d points

    scan_time           float32         time between scans [seconds]
    range_min           float32         minimum range value [m]
    range_max           float32         maximum range value [m]

    ranges              List[float32]   range data [m] (Note: values < range_min or > range_max should be discarded)
    intensities         float32[]       intensity data [device-specific units].
                                        If your device does not provide intensities, please leave the array empty.
    """

    angle_min: float = field(
        default=0.0,
        validator=base_validators.in_range(
            min_value=-2 * math.pi, max_value=2 * math.pi
        ),
    )
    angle_max: float = field(
        default=2 * math.pi,
        validator=base_validators.in_range(
            min_value=-2 * math.pi, max_value=2 * math.pi
        ),
    )
    angle_increment: float = field(
        default=0.01 * math.pi,
        validator=base_validators.in_range(min_value=-math.pi, max_value=math.pi),
    )
    time_increment: float = field(
        default=1e-3, validator=base_validators.in_range(min_value=0.0, max_value=1e3)
    )
    scan_time: float = field(
        default=1e-3, validator=base_validators.in_range(min_value=0.0, max_value=1e3)
    )
    range_min: float = field(
        default=0.0, validator=base_validators.in_range(min_value=0.0, max_value=1e3)
    )
    range_max: float = field(
        default=20.0, validator=base_validators.in_range(min_value=1e-3, max_value=1e3)
    )
    ranges: np.ndarray = field(default=np.empty(0))
    angles: np.ndarray = field(default=np.empty(0))
    intensities: np.ndarray = field(default=np.empty(0))

    def __attrs_post_init__(self):
        if self.angles.size == 0:
            self.angles = np.arange(
                self.angle_min,
                self.angle_max + self.angle_increment,
                self.angle_increment,
            )

        if self.ranges.size == 0:
            # default to max range
            self.ranges = np.full(self.angles.size, self.range_max)

        if self.angles.size != self.ranges.size:
            minimum_size = min(self.angles.size, self.ranges.size)
            self.angles = self.angles[:minimum_size]
            self.ranges = self.ranges[:minimum_size]

    def __convert_to_0_2pi(
        cls, value: Union[float, np.ndarray]
    ) -> Union[float, np.ndarray]:
        """
        Helper method to convert an angle or array of angles to [0,2pi]

        :param value: Input Angle(s) (rad)
        :type value: float | np.ndarray
        :return: Converted Angle(s) (rad)
        :rtype: float | np.ndarray
        """
        value = value % (2 * math.pi)

        if isinstance(value, np.ndarray):
            for idx, val in enumerate(value):
                if val < 0:
                    value[idx] += 2 * np.pi
            return value

        if value < 0:
            value += 2 * math.pi
        return value

    def get_ranges(
        self,
        right_angle: float,
        left_angle: float,
    ) -> np.ndarray:
        """Get ranges values in a defined zone between a left angle and a right angle

        :param right_angle: Value of the angle on the right of the ranges (rad)
        :type right_angle: float
        :param left_angle: Value of the angle on the left of the ranges (rad)
        :type left_angle: float

        :return: Ranges values in the specified zone
        :rtype: np.ndarray
        """
        angles: np.ndarray = self.__convert_to_0_2pi(self.angles)

        left_angle = self.__convert_to_0_2pi(left_angle)
        right_angle = self.__convert_to_0_2pi(right_angle)

        # Get indices of angles in the specified zone
        if right_angle > left_angle:
            angles_in_zone = (angles <= left_angle) | (angles >= right_angle)
        else:
            angles_in_zone = (angles <= left_angle) & (angles >= right_angle)

        return self.ranges[angles_in_zone]

    def get_angles(
        self,
        right_angle: float,
        left_angle: float,
    ) -> np.ndarray:
        """Get ranges values in a defined zone between a left angle and a right angle

        :param right_angle: Value of the angle on the right of the ranges (rad)
        :type right_angle: float
        :param left_angle: Value of the angle on the left of the ranges (rad)
        :type left_angle: float

        :return: Ranges values in the specified zone
        :rtype: np.ndarray
        """
        angles: np.ndarray = self.__convert_to_0_2pi(self.angles)

        left_angle = self.__convert_to_0_2pi(left_angle)
        right_angle = self.__convert_to_0_2pi(right_angle)

        # Get indices of angles in the specified zone
        if right_angle > left_angle:
            angles_in_zone = (angles <= left_angle) | (angles >= right_angle)
        else:
            angles_in_zone = (angles <= left_angle) & (angles >= right_angle)

        return self.angles[angles_in_zone]
