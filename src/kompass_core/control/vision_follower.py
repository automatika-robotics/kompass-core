from ._base_ import ControllerTemplate
from typing import Optional
from attrs import define, field
from ..utils.common import BaseAttrs, in_range
from ..models import Robot, RobotCtrlLimits, RobotType
import numpy as np


@define
class VisionFollowerConfig(BaseAttrs):
    control_time_step: float = field(
        default=0.1, validator=in_range(min_value=1e-4, max_value=1e6)
    )
    control_horizon: float = field(
        default=0.2, validator=in_range(min_value=1e-4, max_value=1e6)
    )
    tolerance: float = field(default=0.05, validator=in_range(min_value=1e-6, max_value=1e3))
    target_distance: Optional[float] = field(default=None)
    alpha : float = field(
        default=1.0, validator=in_range(min_value=1e-9, max_value=1e9)
    )
    beta : float = field(
        default=1.0, validator=in_range(min_value=1e-9, max_value=1e9)
    )
    gamma : float = field(
        default=1.0, validator=in_range(min_value=1e-9, max_value=1e9)
    )
    min_vel: float = field(
        default=0.01, validator=in_range(min_value=1e-9, max_value=1e9)
    )


class VisionFollower(ControllerTemplate):
    def __init__(
        self,
        robot: Robot,
        ctrl_limits: RobotCtrlLimits,
        config: Optional[VisionFollowerConfig] = None,
        config_file: Optional[str] = None,
        config_yaml_root_name: Optional[str] = None,
        **_,
    ):
        self._ctrl_limits = ctrl_limits

        self.config = config or VisionFollowerConfig()

        if config_file:
            self.config.from_yaml(config_file, config_yaml_root_name, get_common=False)

        # Vx, Vy, Omega
        self.__vx_ctrl = []
        self.__vy_ctrl = []
        self.__omega_ctrl = []

        self.__target_ref_size = None

        self.__rotate_in_place: bool = robot.robot_type != RobotType.ACKERMANN.value

        self._time_steps = np.arange(
            0.0, self.config.control_horizon, self.config.control_time_step
        )

    def reset_target(self,
        target_size_w:  float,
        target_size_h: float,
        target_depth: Optional[float]):
        self.__target_ref_size = target_size_w * target_size_h
        if target_depth and not self.config.target_distance:
            self.config.target_distance = target_depth

    def loop_step(
        self,
        *,
        target_image_x: float,
        target_image_y: float,
        target_size_w: float,
        target_size_h: float,
        target_depth: Optional[float] = None,
        **_,
    ):
        # If depth is provided but no reference distance is available -> set current depth as reference distance to keep
        if target_depth and not self.config.target_distance:
            self.config.target_distance = target_depth
        # Otherwise set a reference size (bounding box size)
        elif not self.__target_ref_size:
            # Assume dummy distance of 1 meter at the targeted box size if it is not provided
            self.config.target_distance = self.config.target_distance or 1.0
            self.__target_ref_size = target_size_w * target_size_h

        # Get distance (depth) error
        target_depth = target_depth or self.__target_ref_size / (
            target_size_w * target_size_h
        )
        distance_error = target_depth - self.config.target_distance

        self.__omega_ctrl = len(self._time_steps) * [0.0]
        self.__vx_ctrl = len(self._time_steps) * [0.0]
        self.__vy_ctrl = len(self._time_steps) * [0.0]

        if (
            abs(distance_error) > self.config.tolerance
            or abs(target_image_x) > self.config.tolerance
            or abs(target_image_y) > self.config.tolerance
        ):
            simulated_depth = target_depth
            # Simulate over the control horizon
            for i, t in enumerate(self._time_steps):
                if self.__rotate_in_place and i % 2 != 0:
                    continue
                distance_error = simulated_depth - self.config.target_distance
                dist_speed = (
                    np.sign(distance_error) * self._ctrl_limits.vx_limits.max_vel
                    if abs(distance_error) > 0.5
                    else distance_error * self._ctrl_limits.vx_limits.max_vel
                    + np.sign(distance_error) * self.config.min_vel
                )
                omega = - self.config.alpha * target_image_x
                v = - self.config.beta * target_image_y + self.config.gamma * dist_speed
                simulated_depth += - v * t
                if self.__rotate_in_place:
                    self.__vx_ctrl[i:i + 1] = [0.0, v]
                    self.__omega_ctrl[i : i + 1] = [omega, 0.0]
                else:
                    self.__vx_ctrl[i] = v
                    self.__omega_ctrl[i] = omega
            return

    def logging_info(self) -> str:
        """
        Returns controller progress info for the Node to log

        :return: Controller Info
        :rtype: str
        """
        return f"Vision Object Follower found control: {self.linear_x_control}, {self.angular_control}"

    @property
    def linear_x_control(self):
        """
        Getter of the last linear forward velocity control computed by the controller

        :return: Linear Velocity Control (m/s)
        :rtype: List[float]
        """
        return self.__vx_ctrl

    @property
    def linear_y_control(self):
        """
        Getter the last linear velocity lateral control computed by the controller

        :return: Linear Velocity Control (m/s)
        :rtype: List[float]
        """
        return self.__vy_ctrl

    @property
    def angular_control(self):
        """
        Getter of the last angular velocity control computed by the controller

        :return: Angular Velocity Control (rad/s)
        :rtype: List[float]
        """
        return self.__omega_ctrl
