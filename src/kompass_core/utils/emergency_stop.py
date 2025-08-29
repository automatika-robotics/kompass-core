from typing import Optional, Union
from logging import Logger
from ..models import (
    Robot,
    RobotGeometry,
)
import numpy as np
from ..datatypes import LaserScanData, PointCloudData, ScanModelConfig


class EmergencyChecker:
    """Emergency stop checker class using a minimum safety distance, a critical zone angle and 2D LaserScan data"""

    def __init__(
        self,
        robot: Robot,
        emergency_distance: float,
        slowdown_distance: float,
        emergency_angle: float,
        scan_model: Optional[ScanModelConfig] = None,
        sensor_position_robot: Optional[np.ndarray] = None,
        sensor_rotation_robot: Optional[np.ndarray] = None,
        use_gpu: bool = False,
    ) -> None:
        self.__scan_model = scan_model or ScanModelConfig()
        self.__emergency_distance = emergency_distance
        self.__slowdown_distance = slowdown_distance
        self.__emergency_angle = emergency_angle
        self.__sensor_position_robot = (
            sensor_position_robot
            if sensor_position_robot is not None
            else np.array([0.0, 0.0, 0.0], dtype=np.float32)
        )
        self.__sensor_rotation_robot = (
            sensor_rotation_robot
            if sensor_rotation_robot is not None
            else np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        )
        self.__robot_shape = RobotGeometry.Type.to_kompass_cpp_lib(robot.geometry_type)
        self.__robot_dimensions = robot.geometry_params
        self.__use_gpu = use_gpu
        self.__initialized = False

    def _init_checker(self, scan: Union[LaserScanData, PointCloudData]) -> None:
        angles = (
            scan.angles
            if isinstance(scan, LaserScanData)
            else np.arange(
                0.0,
                2 * np.pi,
                self.__scan_model.angle_step,
            )
        )
        if self.__use_gpu:
            try:
                from kompass_cpp.utils import CriticalZoneCheckerGPU

                self._critical_zone_checker = CriticalZoneCheckerGPU(
                    robot_shape=self.__robot_shape,
                    robot_dimensions=self.__robot_dimensions,
                    sensor_position_body=self.__sensor_position_robot,
                    sensor_rotation_body=self.__sensor_rotation_robot,
                    critical_angle=self.__emergency_angle,
                    critical_distance=self.__emergency_distance,
                    slowdown_distance=self.__slowdown_distance,
                    scan_angles=angles,
                    min_height=self.__scan_model.min_height,
                    max_height=self.__scan_model.max_height,
                    range_max=self.__scan_model.range_max,
                )
            except (ImportError, ModuleNotFoundError):
                Logger(name="EmergencyChecker").error(
                    "GPU use is enabled but GPU implementation is found -> Using CPU implementation"
                )
                self.__use_gpu = False

        if not self.__use_gpu:
            from kompass_cpp.utils import CriticalZoneChecker

            self._critical_zone_checker = CriticalZoneChecker(
                robot_shape=self.__robot_shape,
                robot_dimensions=self.__robot_dimensions,
                sensor_position_body=self.__sensor_position_robot,
                sensor_rotation_body=self.__sensor_rotation_robot,
                critical_angle=self.__emergency_angle,
                critical_distance=self.__emergency_distance,
                slowdown_distance=self.__slowdown_distance,
                scan_angles=angles,
                min_height=self.__scan_model.min_height,
                max_height=self.__scan_model.max_height,
                range_max=self.__scan_model.range_max,
            )

    def run(
        self, *_, scan: Union[LaserScanData, PointCloudData], forward: bool = True
    ) -> float:
        """Runs emergency checking on new incoming laser scan data

        :param scan: 2D Laserscan data (ranges/angles)
        :type scan: LaserScanData
        :param forward: If the robot is moving forward or not, defaults to True
        :type forward: bool, optional
        :return: Slowdown factor if an obstacle is within the safety zone
        :rtype: float
        """
        if not self.__initialized:
            self._init_checker(scan)
            self.__initialized = True
        if isinstance(scan, LaserScanData):
            return self._critical_zone_checker.check(
                ranges=scan.ranges, forward=forward
            )
        else:
            return self._critical_zone_checker.check(**scan.asdict())
