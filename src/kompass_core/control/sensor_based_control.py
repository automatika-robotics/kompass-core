import numpy as np
from typing import Optional, Union
from ..datatypes.laserscan import LaserScanData
from ..utils.geometry import convert_to_0_2pi

from ..algorithms import DeformableVirtualZone, DeformableVirtualZoneParams
from ..models import Robot, RobotCtrlLimits


class LaserScanBasedAction:
    """LaserScanBasedAction."""

    def __init__(self) -> None:
        """__init__.

        :rtype: None
        """
        pass

    def set_new_scan(self, laser_scan_data: LaserScanData):
        """
        Update class with new scan data

        :param laser_scan_data: 2D laser scan
        :type laser_scan_data: LaserScanData
        """
        self.ranges = np.array(laser_scan_data.ranges)

        if laser_scan_data.angles.any():
            self.angles = laser_scan_data.angles
        else:
            angles = np.arange(
                laser_scan_data.angle_min,
                laser_scan_data.angle_max,
                laser_scan_data.angle_increment,
            )
            self.angles: np.ndarray = convert_to_0_2pi(angles)


class LaserScanBasedStop(LaserScanBasedAction):
    """
    Checks for emergency stop based on 2D laser scan
    """

    def __init__(
        self,
        robot_radius: float,
        critical_zone_distance: float,
        critical_zone_angle: float = 45.0,
    ):
        """
        intialize an instance

        :param critical_zone_angle: Radius of the robot where the sensor is mounted (m)
        :type critical_zone_angle: float
        :param critical_zone_angle: Angle field for the critical zone (deg)
        :type critical_zone_angle: float
        :param critical_zone_distance: Max distance for the critical zone
        :type critical_zone_distance: float
        """
        super().__init__()
        self.critical_zone = {
            "left_angle": np.radians(critical_zone_angle) / 2,
            "right_angle": (2 * np.pi) - (np.radians(critical_zone_angle) / 2),
            "distance": critical_zone_distance + robot_radius,
        }
        self.robot_radius = robot_radius

    def obstacle_check(self, laser_scan_data: Union[LaserScanData, float]) -> bool:
        """
        Checks if there's any obstacle in the critical zone of the robot (described by angle and distance)

        :param      laser_scan_data: 2D LiDAR data
        :type       laser_scan_data: LaserScanData

        :return:    Emergency stop indicator
        :rtype:     np.bool_
        """
        if isinstance(laser_scan_data, float):
            return laser_scan_data < self.critical_zone["distance"] - self.robot_radius

        self.set_new_scan(laser_scan_data)
        angles_in_critical_indices = (
            self.angles <= self.critical_zone["left_angle"]
        ) | (self.angles >= self.critical_zone["right_angle"])

        emergency_stop = np.any(
            self.ranges[angles_in_critical_indices] <= self.critical_zone["distance"]
        )

        return bool(emergency_stop)


class ScanBasedDVZControl(LaserScanBasedAction):
    """
    Execute control based on LaserScan data (2D lidar)
    """

    def __init__(
        self,
        robot: Robot,
        config: Optional[DeformableVirtualZoneParams],
        ctrl_limits: RobotCtrlLimits,
        config_file: Optional[str],
    ) -> None:
        """__init__.

        :rtype: None
        """
        super().__init__()

        """Init a DVZ controller

        :param robot: Robot using the controller
        :type robot: Robot
        :param config: DVZ config params
        :type config: Optional[DeformableVirtualZoneParams]
        :param config_file: YAML file
        :type config_file: Optional[str]
        """
        if not config:
            config = DeformableVirtualZoneParams()
        self.dvz_controller = DeformableVirtualZone(
            robot=robot, ctrl_limits=ctrl_limits, config=config
        )

        if config_file:
            self.dvz_controller.set_from_yaml(config_file)

        self.dvz_linear: float = 0.0
        self.dvz_angular: float = 0.0

    def _get_dvz_deformation(self, laser_scan_data: LaserScanData, debug: bool = False):
        """
        Update DVZ deformation with new scan

        :param laser_scan_data: 2D LiDAR scan
        :type laser_scan_data: LaserScanData
        """
        self.set_new_scan(laser_scan_data)
        self.dvz_controller.update_zone_size(self.dvz_linear)
        self.dvz_controller.set_scan_values(
            scan_values=self.ranges, scan_angles=self.angles
        )

        self.dvz_controller.get_total_deformation(compute_deformation_plot=debug)

    def get_new_dvz_ctr(
        self,
        laser_scan_data: LaserScanData,
        time_step: float,
        ref_linear: float = 0.0,
        ref_angular: float = 0.0,
        debug: bool = False,
    ):
        """
        Get new DVZ control values

        :param laser_scan_data: 2D LiDAR scan
        :type laser_scan_data: LaserScanData
        :param ref_linear: Reference linear control (m/s)
        :type ref_linear: float
        :param ref_angular: Reference angular control (rad/s)
        :type ref_angular: float
        """
        self._get_dvz_deformation(laser_scan_data, debug)
        self.dvz_linear = self.dvz_controller.compute_linear_control(
            ref_linear, self.dvz_linear, time_step
        )
        self.dvz_angular = self.dvz_controller.compute_angular_control(ref_angular)
