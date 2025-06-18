from ._base_ import ControllerTemplate
import logging
from typing import Optional, List
from attrs import define, field
from ..utils.common import BaseAttrs, base_validators
from ..models import Robot, RobotCtrlLimits, RobotType
from kompass_cpp.control import RGBFollower, RGBFollowerParameters
from kompass_cpp.types import Bbox2D
import numpy as np


@define
class VisionRGBFollowerConfig(BaseAttrs):
    control_time_step: float = field(
        default=0.1, validator=base_validators.in_range(min_value=1e-4, max_value=1e6)
    )
    control_horizon: int = field(
        default=2, validator=base_validators.in_range(min_value=1, max_value=1000)
    )
    buffer_size: int = field(
        default=1, validator=base_validators.in_range(min_value=1, max_value=10)
    )
    tolerance: float = field(
        default=0.1, validator=base_validators.in_range(min_value=1e-6, max_value=1.0)
    )
    target_distance: Optional[float] = field(default=None)
    target_wait_timeout: float = field(
        default=30.0, validator=base_validators.in_range(min_value=0.0, max_value=1e3)
    )  # wait for target to appear again timeout (seconds), used if search is disabled
    target_search_timeout: float = field(
        default=30.0, validator=base_validators.in_range(min_value=0.0, max_value=1e3)
    )  # search timeout in seconds
    target_search_pause: float = field(
        default=2.0, validator=base_validators.in_range(min_value=0.0, max_value=1e3)
    )  # pause between search actions to find target (seconds)
    target_search_radius: float = field(
        default=0.5, validator=base_validators.in_range(min_value=1e-4, max_value=1e4)
    )
    rotation_gain: float = field(
        default=1.0, validator=base_validators.in_range(min_value=1e-9, max_value=1.0)
    )
    speed_gain: float = field(
        default=0.7, validator=base_validators.in_range(min_value=1e-9, max_value=10.0)
    )
    min_vel: float = field(
        default=0.1, validator=base_validators.in_range(min_value=1e-9, max_value=1e9)
    )
    enable_search: bool = field(default=True)

    camera_position_to_robot: np.ndarray = field(
        default=np.array([0.0, 0.0, 0.0], dtype=np.float32)
    )

    camera_rotation_to_robot: np.ndarray = field(
        default=np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
    )

    def to_kompass_cpp(self) -> RGBFollowerParameters:
        """
        Convert to kompass_cpp lib config format

        :return: _description_
        :rtype: kompass_cpp.control.VisionFollowerParameters
        """
        vision_config = RGBFollowerParameters()
        vision_config.from_dict(self.asdict())
        return vision_config


class VisionRGBFollower(ControllerTemplate):
    def __init__(
        self,
        robot: Robot,
        ctrl_limits: RobotCtrlLimits,
        config: Optional[VisionRGBFollowerConfig] = None,
        config_file: Optional[str] = None,
        config_root_name: Optional[str] = None,
        **_,
    ):
        """Init Vision RGB (Image) Follower Controller

        :param robot: Robot object to be controlled
        :type robot: Robot
        :param ctrl_limits: Robot control limits
        :type ctrl_limits: RobotCtrlLimits
        :param config: Controller configuration, defaults to None
        :type config: Optional[VisionRGBDFollowerConfig], optional
        :param config_file: Path to config file (yaml, json, toml), defaults to None
        :type config_file: Optional[str], optional
        :param config_root_name: Root name for the controller config in the file, defaults to None
        :type config_root_name: Optional[str], optional
        """
        self._config = config or VisionRGBFollowerConfig()

        if config_file:
            config.from_file(config_file, config_root_name, get_common=False)
        self.__controller = RGBFollower(
            control_type=RobotType.to_kompass_cpp_lib(robot.robot_type),
            control_limits=ctrl_limits.to_kompass_cpp_lib(),
            config=self._config.to_kompass_cpp(),
        )

        self._found_ctrl = False
        logging.info("VISION TARGET FOLLOWING CONTROLLER IS READY")

    def set_initial_tracking_2d_target(self, target_box: Bbox2D, **_) -> bool:
        self.__controller.reset_target(target_box)
        return True

    def loop_step(
        self,
        *,
        detections_2d: Optional[List[Bbox2D]],
        **_,
    ) -> bool:
        self._found_ctrl = self.__controller.run(detections_2d[0] if detections_2d else None)
        if self._found_ctrl:
            self._ctrl = self.__controller.get_ctrl()
        return self._found_ctrl

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
        return self._ctrl.vx if self._found_ctrl else None

    @property
    def linear_y_control(self):
        """
        Getter the last linear velocity lateral control computed by the controller

        :return: Linear Velocity Control (m/s)
        :rtype: List[float]
        """
        return self._ctrl.vy if self._found_ctrl else None

    @property
    def angular_control(self):
        """
        Getter of the last angular velocity control computed by the controller

        :return: Angular Velocity Control (rad/s)
        :rtype: List[float]
        """
        return self._ctrl.omega if self._found_ctrl else None
