import math
from typing import Union
import numpy as np
from attrs import define, field

from ..utils.common import BaseAttrs, in_range


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
        default=0.0, validator=in_range(min_value=-2 * math.pi, max_value=2 * math.pi)
    )
    angle_max: float = field(
        default=2 * math.pi,
        validator=in_range(min_value=-2 * math.pi, max_value=2 * math.pi),
    )
    angle_increment: float = field(
        default=0.01 * math.pi,
        validator=in_range(min_value=-math.pi, max_value=math.pi),
    )
    time_increment: float = field(
        default=1e-3, validator=in_range(min_value=0.0, max_value=1e3)
    )
    scan_time: float = field(
        default=1e-3, validator=in_range(min_value=0.0, max_value=1e3)
    )
    range_min: float = field(
        default=0.0, validator=in_range(min_value=0.0, max_value=1e3)
    )
    range_max: float = field(
        default=20.0, validator=in_range(min_value=1e-3, max_value=1e3)
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

    def __sub__(self, __other):
        """
        Substract the scan values of two scans

        :param __other: _description_
        :type __other: _type_
        :raises TypeError: If given value is not a LaserScan Data
        :raises ValueError: If given LaserScanData is incompatible (different min, max or increment)
        :return: _description_
        :rtype: LaserScanData
        """
        if not isinstance(__other, LaserScanData):
            raise TypeError(
                f"Cannot substract LaserScanData and type '{type(__other)}'"
            )

        if (
            self.angle_min != __other.angle_min
            or self.angle_max != __other.angle_max
            or self.angle_increment != __other.angle_increment
        ):
            raise ValueError("Cannot substract incompatible LaserScanData")
        # TODO: Align angles and values
        # def normalize_angle(ang):
        #     value = ang % (2 * math.pi)
        #     if value < 0:
        #         value += 2 * math.pi
        #     return value
        # ang_min = normalize_angle(self.angle_min)
        # ang_max = normalize_angle(self.angle_max)
        # other_min = normalize_angle(__other.angle_min)
        # other_max = normalize_angle(__other.angle_max)

        result = self
        result.ranges = self.ranges - __other.ranges
        result.intensities = self.intensities - __other.intensities
        return result

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

    def get_ranges(self, angle_min: float, angle_max: float) -> np.ndarray:
        """Get laserscan ranges values between two given angles

        :param angle_min: Minimum angle (rad)
        :type angle_min: float
        :param angle_max:  Maximum angle (rad)
        :type angle_max: float

        :return: Laserscan values in angles range
        :rtype: np.ndarray
        """
        angles_converted = self.__convert_to_0_2pi(self.angles)
        if self.__convert_to_0_2pi(angle_min) < self.__convert_to_0_2pi(angle_max):
            idx_mask = (angles_converted >= self.__convert_to_0_2pi(angle_min)) & (
                angles_converted <= self.__convert_to_0_2pi(angle_max)
            )
        else:
            idx_mask = (angles_converted >= self.__convert_to_0_2pi(angle_min)) | (
                angles_converted <= self.__convert_to_0_2pi(angle_max)
            )
        return self.ranges[idx_mask]
