from typing import Optional, List
from logging import Logger
from ..models import (
    Robot,
    RobotGeometry,
)
from ..datatypes import LaserScanData


class EmergencyChecker:
    """Emergency stop checker class using a minimum safety distance, a critical zone angle and 2D LaserScan data"""

    def __init__(
        self,
        robot: Robot,
        emergency_distance: float,
        emergency_angle: float,
        sensor_position_robot: Optional[List[float]] = None,
        sensor_rotation_robot: Optional[List[float]] = None,
        use_gpu: bool = False,
    ) -> None:
        self.__emergency_distance = emergency_distance
        self.__emergency_angle = emergency_angle
        self.__sensor_position_robot = sensor_position_robot or [0.0, 0.0, 0.0]
        self.__sensor_rotation_robot = sensor_rotation_robot or [0.0, 0.0, 0.0, 1.0]
        self.__robot_shape = RobotGeometry.Type.to_kompass_cpp_lib(
            self.__robot.geometry_type
        )
        self.__robot_dimensions = robot.geometry_params
        self.__use_gpu = use_gpu

    def run(self, *_, scan: LaserScanData, forward: bool = True) -> bool:
        """Runs emergency checking on new incoming laser scan data

        :param scan: 2D Laserscan data (ranges/angles)
        :type scan: LaserScanData
        :param forward: If the robot is moving forward or not, defaults to True
        :type forward: bool, optional
        :return: If an obstacle is within the safety zone
        :rtype: bool
        """
        if self.__use_gpu:
            try:
                from kompass_cpp.utils import CriticalZoneCheckerGPU

                self._critical_zone_checker = CriticalZoneCheckerGPU(
                    robot_shape=self.__robot_shape,
                    robot_dimensions=self.__robot_dimensions,
                    sensor_position_body=self.__sensor_position_robot
                    or [0.0, 0.0, 0.0],
                    sensor_rotation_body=self.__sensor_rotation_robot
                    or [0.0, 0.0, 0.0, 1.0],
                    critical_angle=self.__emergency_angle,
                    critical_distance=self.__emergency_distance,
                    scan_size=len(scan.angles),
                )
                # Checker will be initialized on the first incoming laserscan data to get the scan size
                self._critical_zone_checker = None
            except (ImportError, ModuleNotFoundError):
                Logger(name="EmergencyChecker").error(
                    "GPU use is enabled but GPU implementation is found -> Using CPU implementation"
                )
                use_gpu = False
        if not use_gpu:
            from kompass_cpp.utils import CriticalZoneChecker

            self._critical_zone_checker = CriticalZoneChecker(
                robot_shape=self.__robot_shape,
                robot_dimensions=self.__robot_dimensions,
                sensor_position_body=self.__sensor_position_robot or [0.0, 0.0, 0.0],
                sensor_rotation_body=self.__sensor_rotation_robot
                or [0.0, 0.0, 0.0, 1.0],
                critical_angle=self.__emergency_angle,
                critical_distance=self.__emergency_distance,
            )
        if not self._critical_zone_checker:
            self._critical_zone_checker = self.__CriticalZoneCheckerClass(
                robot_shape=RobotGeometry.Type.to_kompass_cpp_lib(
                    self.__robot.geometry_type
                ),
                robot_dimensions=self.__robot.geometry_params,
                sensor_position_body=self.__sensor_position_robot or [0.0, 0.0, 0.0],
                sensor_rotation_body=self.__sensor_rotation_robot
                or [0.0, 0.0, 0.0, 1.0],
                critical_angle=self.__emergency_angle,
                critical_distance=self.__emergency_distance,
                scan_angles=scan.angles,
            )
        return self._critical_zone_checker.check(
            ranges=scan.ranges, angles=scan.angles, forward=forward
        )
