from typing import Optional, List
import logging
import numpy as np
from ..datatypes.laserscan import LaserScanData
from nav_msgs.msg import Path

import kompass_cpp
from ..algorithms import DeformableVirtualZoneParams
from ..models import Robot, RobotCtrlLimits, RobotState, RobotType

from ._base_ import FollowerTemplate
from .sensor_based_control import ScanBasedDVZControl
from .stanley import Stanley, StanleyConfig
from attrs import define, field
from ..utils.common import in_range


@define
class DVZConfig(DeformableVirtualZoneParams):
    heading_gain: float = field(
        default=1.0, validator=in_range(min_value=0.0, max_value=1e2)
    )

    cross_track_gain: float = field(
        default=0.1, validator=in_range(min_value=0.0, max_value=1e2)
    )


class DVZ(FollowerTemplate):
    """DVZ Controller."""

    def __init__(
        self,
        robot: Robot,
        ctrl_limits: RobotCtrlLimits,
        control_time_step: float,
        config_file: Optional[str] = None,
        config: Optional[DVZConfig] = None,
        config_yaml_root_name: Optional[str] = None,
        **_,
    ):
        """Setup DVZ Local Planner

        :param robot: Robot that will use the controller
        :type robot: Robot
        :param ctrl_limits: Robot velocity control limits
        :type ctrl_limits: RobotCtrlLimits
        :param control_time_step: Time step (s)
        :type control_time_step: float
        :param config_file: Path to YAML config file, defaults to None
        :type config_file: Optional[str], optional
        :param config: DVZ configuration, defaults to None
        :type config: Optional[DVZConfig], optional
        :param config_yaml_root_name: Root name for the config in the YAML file, defaults to None
        :type config_yaml_root_name: Optional[str], optional
        """
        # Init the controller
        self._robot = robot
        self._control_time_step = control_time_step

        if not config:
            config = DVZConfig()

        self._path_controller = ScanBasedDVZControl(
            robot=robot, config=config, ctrl_limits=ctrl_limits, config_file=config_file
        )

        generator_config = StanleyConfig(
            heading_gain=config.heading_gain, cross_track_gain=config.cross_track_gain
        )
        # Setup a stanley follower to generate the reference commands
        self.__refrence_cmd_generator = Stanley(
            robot=robot,
            ctrl_limits=ctrl_limits,
            config=generator_config,
            config_file=config_file,
            config_yaml_root_name=config_yaml_root_name,
            generate_reference=True,
        )
        logging.info("DVZ Controller is ready")
        self.rotating_in_place: bool = False

    def reached_end(self) -> bool:
        """Check if current goal is reached

        :return: If goal is reached
        :rtype: bool
        """
        return self.__refrence_cmd_generator.reached_end()

    def interpolated_path(self, msg_header) -> kompass_cpp.types.Path:
        """
        Getter of the interpolated path

        :return: Interpolated path
        :rtype: kompass_cpp.types.Path
        """
        return self.__refrence_cmd_generator.interpolated_path(msg_header)

    @property
    def tracked_state(self) -> RobotState:
        """
        Tracked state on the path

        :return: _description_
        :rtype: RobotState
        """
        return self.__refrence_cmd_generator.tracked_state

    def set_path(self, global_path: Path, **_) -> None:
        """
        Set the reference path to the controller

        :param global_path: Global (reference) path
        :type global_path: Path
        """
        self.__refrence_cmd_generator.set_path(global_path=global_path)

    def loop_step(
        self,
        *,
        laser_scan: LaserScanData,
        current_state: RobotState,
        initial_control_seq: Optional[np.ndarray] = None,
        **_,
    ) -> bool:
        """
        Implements a loop iteration of the controller

        :param laser_scan_callback: 2D laserscan handler
        :type laser_scan_callback: LaserScanCallback
        :param initial_control_seq: Initial (reference) control sequence
        :type initial_control_seq: np.ndarray
        """
        if initial_control_seq:
            # Get path tracking reference commands
            _ref_linear_x_cmd = initial_control_seq[0, 0]
            # _ref_linear_y_cmd = initial_control_seq[0, 1] TODO update dvz to take omni motion
            _ref_angular_cmd = initial_control_seq[0, 2]
        else:
            # Generate the reference
            ref_found: bool = self.__refrence_cmd_generator.loop_step(
                current_state=current_state
            )

            # If reference cannot be computed set to zero and compute DVZ to insure reactive behavior
            if not ref_found:
                _ref_linear_x_cmd = 0.0
                _ref_angular_cmd = 0.0
            else:
                _ref_linear_x_cmd = self.__refrence_cmd_generator.linear_x_control[0]
                _ref_angular_cmd = self.__refrence_cmd_generator.angular_control[0]

        # Get new dvz control
        self._path_controller.get_new_dvz_ctr(
            laser_scan_data=laser_scan,
            time_step=self._control_time_step,
            ref_linear=_ref_linear_x_cmd,
            ref_angular=_ref_angular_cmd,
        )
        return True

    @property
    def planner(self) -> kompass_cpp.control.Follower:
        return self.__refrence_cmd_generator.planner

    def logging_info(self) -> str:
        """logging_info.

        :rtype: str
        """
        return f"total deformation : {self._path_controller.dvz_controller.total_deformation}"

    @property
    def linear_x_control(self) -> List[float]:
        """
        Gettter of the last linear forward velocity control computed by the controller

        :return: Linear Velocity Control (m/s)
        :rtype: float
        """
        if (
            self._robot.robot_type != RobotType.ACKERMANN
            and abs(self._path_controller.dvz_angular)
            > self.__refrence_cmd_generator.config.min_angular_vel
        ):
            if (
                abs(self.orientation_error)
                > self.__refrence_cmd_generator.config.max_angle_error
                and abs(self.distance_error)
                < self.__refrence_cmd_generator.config.max_distance_error
            ):
                # Rotate in place
                return [0.0]
            # Apply angular first (Set linear to zero): Rotate then move
            return [0.0, self._path_controller.dvz_linear]

        return [self._path_controller.dvz_linear]

    @property
    def linear_y_control(self) -> List[float]:
        """
        Getter the last linear velocity lateral control computed by the controller

        :return: Linear Velocity Control (m/s)
        :rtype: float
        """
        if (
            self._robot.robot_type != RobotType.ACKERMANN
            and abs(self._path_controller.dvz_angular)
            > self.__refrence_cmd_generator.config.min_angular_vel
        ):
            if (
                abs(self.orientation_error)
                > self.__refrence_cmd_generator.config.max_angle_error
                and abs(self.distance_error)
                < self.__refrence_cmd_generator.config.max_distance_error
            ):
                return [0.0]
            # Apply angular first (Set linear to zero): Rotate then move
            return [0.0, 0.0]

        return [0.0]

    @property
    def angular_control(self) -> List[float]:
        """
        Getter of the last angular velocity control computed by the controller

        :return: Angular Velocity Control (rad/s)
        :rtype: float
        """
        if (
            self._robot.robot_type != RobotType.ACKERMANN
            and abs(self._path_controller.dvz_angular)
            > self.__refrence_cmd_generator.config.min_angular_vel
        ):
            if (
                abs(self.orientation_error)
                > self.__refrence_cmd_generator.config.max_angle_error
                and abs(self.distance_error)
                < self.__refrence_cmd_generator.config.max_distance_error
            ):
                self.rotating_in_place = True
                return [self.__refrence_cmd_generator.in_place_rotation()]
            self.rotating_in_place = False
            # Apply angular first (Set linear to zero): Rotate then move
            return [self._path_controller.dvz_angular, 0.0]
        return [self._path_controller.dvz_angular]