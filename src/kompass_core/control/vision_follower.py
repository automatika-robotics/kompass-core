from ._base_ import ControllerTemplate
import logging
from typing import Optional
from attrs import define, field
from ..utils.common import BaseAttrs, base_validators
from ..models import Robot, RobotCtrlLimits, RobotType
from ..datatypes import TrackingData
import kompass_cpp


@define
class VisionFollowerConfig(BaseAttrs):
    control_time_step: float = field(
        default=0.1, validator=base_validators.in_range(min_value=1e-4, max_value=1e6)
    )
    control_horizon: int = field(
        default=2, validator=base_validators.in_range(min_value=1, max_value=1000)
    )
    tolerance: float = field(
        default=0.1, validator=base_validators.in_range(min_value=1e-6, max_value=1e3)
    )
    target_distance: Optional[float] = field(default=None)
    target_search_timeout: int = field(
        default=30, validator=base_validators.in_range(min_value=0, max_value=1000)
    )  # number of steps to timeout
    target_search_pause: int = field(
        default=2, validator=base_validators.in_range(min_value=0, max_value=1000)
    )  # number of steps to pause
    target_search_radius: float = field(
        default=0.5, validator=base_validators.in_range(min_value=1e-4, max_value=1e4)
    )
    rotation_multiple: float = field(
        default=1.0, validator=base_validators.in_range(min_value=1e-9, max_value=1.0)
    )
    speed_depth_multiple: float = field(
        default=0.7, validator=base_validators.in_range(min_value=1e-9, max_value=1.0)
    )
    min_vel: float = field(
        default=0.01, validator=base_validators.in_range(min_value=1e-9, max_value=1e9)
    )
    enable_search: bool = field(default=True)

    def to_kompass_cpp(self) -> kompass_cpp.control.VisionFollowerParameters:
        """
        Convert to kompass_cpp lib config format

        :return: _description_
        :rtype: kompass_cpp.control.VisionFollowerParameters
        """
        vision_config = kompass_cpp.control.VisionFollowerParameters()
        vision_config.from_dict(self.asdict())
        return vision_config


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
        config = config or VisionFollowerConfig()

        if config_file:
            config.from_yaml(config_file, config_yaml_root_name, get_common=False)
        self.__controller = kompass_cpp.control.VisionFollower(
            control_type=RobotType.to_kompass_cpp_lib(robot.robot_type),
            control_limits=ctrl_limits.to_kompass_cpp_lib(),
            config=config.to_kompass_cpp(),
        )

        self._found_ctrl = False
        logging.info("VISION TARGET FOLLOWING CONTROLLER IS READY")

    def reset_target(self, tracking: TrackingData):
        self.__controller.reset_target(tracking.to_kompass_cpp())

    def loop_step(
        self,
        *,
        tracking: TrackingData,
        **_,
    ) -> bool:
        tracking_cpp = tracking.to_kompass_cpp() if tracking else None
        self._found_ctrl = self.__controller.run(tracking_cpp)
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
