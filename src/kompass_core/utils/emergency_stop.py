from typing import Optional, List
from ..models import (
    Robot,
    RobotGeometry,
)
import numpy as np
from ..datatypes import LaserScanData
from kompass_cpp.utils import CollisionChecker


class EmergencyChecker:
    """Emergency stop checker class using a minimum safety distance, a critical zone angle and 2D LaserScan data"""

    def __init__(
        self,
        robot: Robot,
        emergency_distance: float,
        emergency_angle: float,
        sensor_position_robot: Optional[List[float]] = None,
        sensor_rotation_robot: Optional[List[float]] = None,
    ) -> None:
        self._collision_checker = CollisionChecker(
            robot_shape=RobotGeometry.Type.to_kompass_cpp_lib(robot.geometry_type),
            robot_dimensions=robot.geometry_params,
            sensor_position_body=sensor_position_robot or [0.0, 0.0, 0.0],
            sensor_rotation_body=sensor_rotation_robot or [0.0, 0.0, 0.0, 1.0],
        )
        self.__min_dist = emergency_distance
        self.__critical_zone = {
            "left_angle": np.radians(emergency_angle) / 2,
            "right_angle": (2 * np.pi) - (np.radians(emergency_angle) / 2),
        }

    def run(self, *_, scan: LaserScanData, forward: bool = True) -> bool:
        """Runs emergency checking on new incoming laser scan data

        :param scan: 2D Laserscan data (ranges/angles)
        :type scan: LaserScanData
        :param forward: If the robot is moving forward or not, defaults to True
        :type forward: bool, optional
        :return: If an obstacle is within the safety zone
        :rtype: bool
        """
        if forward:
            # Check in front
            ranges_to_check = scan.get_ranges(
                right_angle=self.__critical_zone["right_angle"],
                left_angle=self.__critical_zone["left_angle"],
            )
            angles_to_check = scan.get_angles(
                right_angle=self.__critical_zone["right_angle"],
                left_angle=self.__critical_zone["left_angle"],
            )
        else:
            # Moving backwards -> Check behind
            ranges_to_check = scan.get_ranges(
                right_angle=self.critical_zone["right_angle"] + np.pi,
                left_angle=self.critical_zone["left_angle"] + np.pi,
            )
            angles_to_check = scan.get_angles(
                right_angle=self.__critical_zone["right_angle"] + np.pi,
                left_angle=self.__critical_zone["left_angle"] + np.pi,
            )
        min_dist: float = self._collision_checker.get_min_distance_laserscan(
            ranges=ranges_to_check, angles=angles_to_check
        )
        return min_dist <= self.__min_dist
