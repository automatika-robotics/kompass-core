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
    """
    Configuration class for an RGB-based vision target follower.

    This class defines configuration parameters for controlling a robot that follows a target using RGB vision.
    It provides settings for control behavior, target tracking, search strategies, velocity and tolerance tuning,
    and camera-to-robot coordinate transformations.

    Attributes:
        control_time_step (float): Time interval between control updates (s).
        control_horizon (int): Number of time steps in the control planning horizon.
        buffer_size (int): Number of buffered detections to maintain.
        tolerance (float): Acceptable error when tracking the target.
        target_distance (Optional[float]): Desired distance to maintain from the target (m).
        target_wait_timeout (float): Maximum time to wait for a target to reappear if lost (s).
        target_search_timeout (float): Maximum duration to perform a search when target is lost (s).
        target_search_pause (float): Delay between successive search attempts (s).
        target_search_radius (float): Radius used for searching the target (m).
        rotation_gain (float): Proportional gain for angular control.
        speed_gain (float): Proportional gain for linear speed control.
        min_vel (float): Minimum linear velocity allowed during target following (m/s).
        enable_search (bool): Whether to activate search behavior when the target is lost.
    """

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
    """
    VisionRGBFollower is a controller for vision-based target following using RGB image data.

    This controller processes 2D object detections (e.g., bounding boxes) and generates velocity commands
    to follow a visual target using a proportional control law. It supports configuration via Python or external
    configuration files and allows integration into Kompass-style robotic systems.

    - Usage Example:

    ```python
    import numpy as np
    from kompass_core.control import VisionRGBFollower, VisionRGBFollowerConfig
    from kompass_core.models import (
        Robot,
        RobotType,
        RobotCtrlLimits,
        LinearCtrlLimits,
        AngularCtrlLimits,
        RobotGeometry,
    )
    from kompass_core.datatypes import Bbox2D

    # Define robot
    my_robot = Robot(
        robot_type=RobotType.ACKERMANN,
        geometry_type=RobotGeometry.Type.CYLINDER,
        geometry_params=np.array([0.2, 0.5])
    )

    # Define control limits
    ctrl_limits = RobotCtrlLimits(
        vx_limits=LinearCtrlLimits(max_vel=1.0, max_acc=2.0, max_decel=4.0),
        omega_limits=AngularCtrlLimits(
            max_vel=2.0, max_acc=3.0, max_decel=3.0, max_steer=np.pi
        )
    )

    # Create the controller
    config = VisionRGBFollowerConfig(
        target_search_timeout=20.0,
        speed_gain=0.8,
        rotation_gain=0.9,
        enable_search=True,
    )
    controller = VisionRGBFollower(robot=my_robot, ctrl_limits=ctrl_limits, config=config)

    # 2D detection
    detection = Bbox2D(
                top_left_corner=np.array(
                    [30, 40], dtype=np.int32
                ),
                size=np.array(
                    [
                        100,
                        200,
                    ],
                    dtype=np.int32,
                ),
                timestamp=0.0,
                label="person",
            )
    detection.set_img_size(np.array([640, 480], dtype=np.int32))

    # Set initial target
    controller.set_initial_tracking_2d_target(detection)

    # Perform a control loop step
    success = controller.loop_step(detections_2d=[detection])

    # Access control outputs
    vx = controller.linear_x_control
    omega = controller.angular_control
    ```
    """

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
            self._config.from_file(config_file, config_root_name, get_common=False)
        self.__controller = RGBFollower(
            control_type=RobotType.to_kompass_cpp_lib(robot.robot_type),
            control_limits=ctrl_limits.to_kompass_cpp_lib(),
            config=self._config.to_kompass_cpp(),
        )

        self._found_ctrl = False
        logging.info("VISION TARGET FOLLOWING CONTROLLER IS READY")

    def set_initial_tracking_2d_target(self, target_box: Bbox2D, **_) -> bool:
        """Sets the initial target for the controller to track

        :param target_box: 2D bounding box of the target
        :type target_box: Bbox2D
        :return: True if the target was set successfully, False otherwise
        :rtype: bool
        """
        self.__controller.reset_target(target_box)
        return True

    @property
    def dist_error(self) -> float:
        """Getter of the last distance error computed by the controller

        :return: Last distance error (m)
        :rtype: float
        """
        return self.__controller.get_errors()[0]

    @property
    def orientation_error(self) -> float:
        """Getter of the last orientation error computed by the controller (radians)

        :return: Last orientation error (radians)
        :rtype: float
        """
        return self.__controller.get_errors()[1]

    def loop_step(
        self,
        *,
        detections_2d: Optional[List[Bbox2D]],
        **_,
    ) -> bool:
        self._found_ctrl = self.__controller.run(
            detections_2d[0] if detections_2d else None
        )
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
