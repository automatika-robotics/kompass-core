from typing import Optional, List
import logging
from attrs import define, field
from ..utils.common import base_validators

import kompass_cpp
from ..models import Robot, RobotCtrlLimits, RobotState, RobotType
from ._base_ import FollowerTemplate, FollowerConfig
import numpy as np


@define
class StanleyConfig(FollowerConfig):
    """
    Stanley follower parameters

    ```{list-table}
    :widths: 10 10 10 70
    :header-rows: 1

    * - Name
      - Type
      - Default
      - Description
    * - control_time_step
      - `float`
      - `0.1`
      - Time interval between control actions. Must be between `1e-6` and `1e3`.
    * - wheel_base
      - `float`
      - `0.266`
      - Distance between the front and rear axles of the robot. Must be between `1e-3` and `1e3`.
    * - heading_gain
      - `float`
      - `0.7`
      - Gain for heading control. Must be between `0.0` and `1e2`.
    * - cross_track_min_linear_vel
      - `float`
      - `0.05`
      - Minimum linear velocity for cross-track control. Must be between `1e-4` and `1e2`.
    * - cross_track_gain
      - `float`
      - `1.5`
      - Gain for cross-track control. Must be between `0.0` and `1e2`.
    * - max_angle_error
      - `float`
      - `np.pi / 16`
      - Maximum allowable angular error in radians. Must be between `1e-9` and `π`.
    * - max_distance_error
      - `float`
      - `0.1`
      - Maximum allowable distance error. Must be between `1e-9` and `1e9`.
    * - min_angular_vel
      - `float`
      - `0.01`
      - Minimum allowable angular velocity. Must be between `0.0` and `1e9`.

    ```
    """

    control_time_step: float = field(
        default=0.1, validator=base_validators.in_range(min_value=1e-6, max_value=1e3)
    )

    wheel_base: float = field(
        default=0.266, validator=base_validators.in_range(min_value=1e-3, max_value=1e3)
    )

    heading_gain: float = field(
        default=0.7, validator=base_validators.in_range(min_value=0.0, max_value=1e2)
    )

    cross_track_min_linear_vel: float = field(
        default=0.05, validator=base_validators.in_range(min_value=1e-4, max_value=1e2)
    )

    cross_track_gain: float = field(
        default=1.5, validator=base_validators.in_range(min_value=0.0, max_value=1e2)
    )

    max_angle_error: float = field(
        default=np.pi / 16,
        validator=base_validators.in_range(min_value=1e-9, max_value=np.pi),
    )

    max_distance_error: float = field(
        default=0.1, validator=base_validators.in_range(min_value=1e-9, max_value=1e9)
    )

    min_angular_vel: float = field(
        default=0.01, validator=base_validators.in_range(min_value=0.0, max_value=1e9)
    )

    def to_kompass_cpp(self) -> kompass_cpp.control.StanleyParameters:
        """
        Convert to kompass_cpp lib config format

        :return: _description_
        :rtype: kompass_cpp.control.StanleyParameters
        """
        stanley_config = kompass_cpp.control.StanleyParameters()
        stanley_config.from_dict(self.asdict())
        return stanley_config


class Stanley(FollowerTemplate):
    def __init__(
        self,
        robot: Robot,
        ctrl_limits: RobotCtrlLimits,
        config: Optional[StanleyConfig] = None,
        config_file: Optional[str] = None,
        config_yaml_root_name: Optional[str] = None,
        generate_reference: bool = False,
        **_,
    ):
        """
        Setup the controller

        :param robot: Robot using the controller
        :type robot: Robot
        :param params_file: Yaml file containing the parameters of the controller under 'dvz_controller'
        :type params_file: str
        """
        self.__generate_reference = generate_reference
        self._robot = robot

        # Init and configure the follower
        if not config:
            config = StanleyConfig(wheel_base=robot.wheelbase)

        if config_file:
            config.from_yaml(
                file_path=config_file, nested_root_name=config_yaml_root_name
            )

        self._config = config
        self._control_time_step = config.control_time_step

        self._planner = kompass_cpp.control.Stanley(config.to_kompass_cpp())

        # Set the control limits
        self._planner.set_linear_ctr_limits(
            ctrl_limits.linear_to_kompass_cpp_lib(ctrl_limits.vx_limits),
            ctrl_limits.linear_to_kompass_cpp_lib(ctrl_limits.vy_limits),
        )
        self._planner.set_angular_ctr_limits(ctrl_limits.angular_to_kompass_cpp_lib())

        self.__max_angular = ctrl_limits.omega_limits.max_vel

        # Init the following result
        self._result = kompass_cpp.control.FollowingResult()
        logging.info("STANLEY PATH CONTROLLER IS READY")

    @property
    def planner(self) -> kompass_cpp.control.Follower:
        return self._planner

    def loop_step(self, *, current_state: RobotState, **_) -> bool:
        """
        Implements a loop iteration of the controller

        :param laser_scan_callback: 2D laserscan handler
        :type laser_scan_callback: LaserScanCallback
        :param initial_control_seq: Initial (reference) control sequence
        :type initial_control_seq: np.ndarray
        """
        self._planner.set_current_state(
            current_state.x, current_state.y, current_state.yaw, current_state.speed
        )
        # If end point is reached -> no need to compute a new control
        if self.reached_end():
            return True

        self._result = self._planner.compute_velocity_commands(self._control_time_step)
        return self._result.status == kompass_cpp.control.FollowingStatus.COMMAND_FOUND

    def logging_info(self) -> str:
        """Get logging information

        :return: Information
        :rtype: str
        """
        return f"Follower current status: {self._result.status}, Velocity command: {self._result.velocity_command}"

    @property
    def linear_x_control(self) -> List[float]:
        """
        Getter of the last linear forward velocity control computed by the controller

        :return: Linear Velocity Control (m/s)
        :rtype: List[float]
        """
        if self.__generate_reference:
            return [self._planner.get_vx_cmd()] if not self.reached_end() else [0.0]

        elif (
            self._robot.robot_type != RobotType.ACKERMANN.value
            and abs(self._planner.get_omega_cmd()) > self._config.min_angular_vel
        ):
            if (
                abs(self.orientation_error) > self._config.max_angle_error
                and abs(self.distance_error) < self._config.max_distance_error
            ):
                return [0.0]
            # Apply angular first (Set linear to zero): Rotate then move
            return [0.0, self._planner.get_vx_cmd()]

        return [self._planner.get_vx_cmd()]

    @property
    def linear_y_control(self) -> List[float]:
        """
        Getter the last linear velocity lateral control computed by the controller

        :return: Linear Velocity Control (m/s)
        :rtype: List[float]
        """
        if self.__generate_reference:
            return [self._planner.get_vy_cmd()] if not self.reached_end() else [0.0]

        elif (
            self._robot.robot_type != RobotType.ACKERMANN.value
            and abs(self._planner.get_omega_cmd()) > self._config.min_angular_vel
        ):
            if (
                abs(self.orientation_error) > self._config.max_angle_error
                and abs(self.distance_error) < self._config.max_distance_error
            ):
                return [0.0]
            # Apply angular first (Set linear to zero): Rotate then move
            return [0.0, self._planner.get_vy_cmd()]

        return [self._planner.get_vy_cmd()]

    @property
    def angular_control(self) -> List[float]:
        """
        Getter of the last angular velocity control computed by the controller

        :return: Angular Velocity Control (rad/s)
        :rtype: List[float]
        """
        if self.__generate_reference:
            return [self._planner.get_omega_cmd()] if not self.reached_end() else [0.0]

        if (
            self._robot.robot_type != RobotType.ACKERMANN.value
            and abs(self._planner.get_omega_cmd()) > self._config.min_angular_vel
        ):
            if (
                abs(self.orientation_error) > self._config.max_angle_error
                and abs(self.distance_error) < self._config.max_distance_error
            ):
                return [self.in_place_rotation()]

            # Apply angular first (Set linear to zero): Rotate then move
            return [self._planner.get_omega_cmd(), 0.0]
        return [self._planner.get_omega_cmd()]

    def in_place_rotation(self) -> float:
        rotation_val = (
            self.__max_angular
            * self.orientation_error
            / (self._control_time_step * 2 * np.pi)
        )
        return min(max(rotation_val, -self.__max_angular), self.__max_angular)
