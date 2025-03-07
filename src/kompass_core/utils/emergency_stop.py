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
        self.__critical_angle = emergency_angle

    def run(self, *_, scan: LaserScanData, forward: bool = True) -> bool:
        """Runs emergency checking on new incoming laser scan data

        :param scan: 2D Laserscan data (ranges/angles)
        :type scan: LaserScanData
        :param forward: If the robot is moving forward or not, defaults to True
        :type forward: bool, optional
        :return: If an obstacle is within the safety zone
        :rtype: bool
        """
        return self._collision_checker.check_critical_zone(
            ranges=scan.ranges,
            angles=scan.angles,
            forward=forward,
            critical_angle=self.__critical_angle,
            critical_distance=self.__min_dist,
        )
