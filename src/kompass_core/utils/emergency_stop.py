from typing import Optional
from logging import Logger
from ..models import (
    Robot,
    RobotGeometry,
)
import numpy as np
from ..datatypes import ScanModelConfig
from kompass_cpp.types import SensorInputType, PointFieldType


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
        cloud_field_type: PointFieldType = PointFieldType.FLOAT32,
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
        self.__robot_height = robot.height
        self.__use_gpu = use_gpu
        self.__initialized = False
        # NOTE: cloud_field_type is only used in point cloud and gotten from PointCloudCallback in Kompass
        self.__cloud_field_type = cloud_field_type

    def _init_checker(self, scan_angles: Optional[np.ndarray] = None) -> None:
        """Build the underlying checker for the sensor it is about to be fed

        :param scan_angles: Angles of a laser scan, or None for a point cloud,
            whose angles are binned from the scan model instead
        :type scan_angles: Optional[np.ndarray]
        """
        if scan_angles is not None:
            kwargs = {
                "input_type": SensorInputType.LASERSCAN,
                "scan_angles": scan_angles,
            }
        else:
            kwargs = {
                "input_type": SensorInputType.POINTCLOUD,
                "scan_angles": np.arange(
                    0.0,
                    2 * np.pi,
                    self.__scan_model.angle_step,
                ),
            }
            if self.__use_gpu:
                # this parameter is only used in the GPU kernel
                kwargs["cloud_field_type"] = self.__cloud_field_type

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
                    min_height=-self.__robot_height,
                    max_height=self.__robot_height,
                    range_max=self.__scan_model.range_max,
                    **kwargs,
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
                min_height=-self.__robot_height,
                max_height=self.__robot_height,
                range_max=self.__scan_model.range_max,
                **kwargs,
            )

    def run_on_laserscan(
        self, *, ranges: np.ndarray, angles: np.ndarray, forward: bool = True
    ) -> float:
        """Runs emergency checking on new incoming laser scan data

        :param ranges: Measured range along each angle (m)
        :type ranges: np.ndarray
        :param angles: Angle of each range measurement (rad)
        :type angles: np.ndarray
        :param forward: If the robot is moving forward or not, defaults to True
        :type forward: bool, optional
        :return: Slowdown factor if an obstacle is within the safety zone
        :rtype: float
        """
        if not self.__initialized:
            self._init_checker(scan_angles=angles)
            self.__initialized = True

        return self._critical_zone_checker.check(ranges=ranges, forward=forward)

    def run_on_pointcloud(
        self,
        *,
        data: np.ndarray,
        point_step: int,
        row_step: int,
        height: int,
        width: int,
        x_offset: int,
        y_offset: int,
        z_offset: int,
        forward: bool = True,
        **_,
    ) -> float:
        """Runs emergency checking on new incoming point cloud data

        The parameters mirror the sensor_msgs/PointCloud2 layout: the raw buffer
        is handed to the checker as-is rather than decoded in Python. Any
        further fields a caller's cloud container carries are ignored, so a
        whole container can be splatted in.

        :param data: Raw point buffer as a flat byte array
        :type data: np.ndarray
        :param point_step: Length of a single point in bytes
        :type point_step: int
        :param row_step: Length of a single row in bytes
        :type row_step: int
        :param height: Number of rows (1 for unorganized clouds)
        :type height: int
        :param width: Number of points per row
        :type width: int
        :param x_offset: Byte offset of the 'x' field within a point
        :type x_offset: int
        :param y_offset: Byte offset of the 'y' field within a point
        :type y_offset: int
        :param z_offset: Byte offset of the 'z' field within a point
        :type z_offset: int
        :param forward: If the robot is moving forward or not, defaults to True
        :type forward: bool, optional
        :return: Slowdown factor if an obstacle is within the safety zone
        :rtype: float
        """
        if not self.__initialized:
            self._init_checker()
            self.__initialized = True

        return self._critical_zone_checker.check(
            data=data,
            point_step=point_step,
            row_step=row_step,
            height=height,
            width=width,
            x_offset=x_offset,
            y_offset=y_offset,
            z_offset=z_offset,
            forward=forward,
        )
