from typing import Optional, List
import logging
import numpy as np
from attrs import define, field
from ..utils.common import base_validators

import kompass_cpp
from ..models import Robot, RobotCtrlLimits, RobotState, RobotType, RobotGeometry
from ._base_ import FollowerTemplate, FollowerConfig


@define
class PurePursuitConfig(FollowerConfig):
    """
    Pure Pursuit follower parameters

    ```{list-table}
    :widths: 10 10 10 70
    :header-rows: 1

    * - Name
      - Type
      - Default
      - Description
    * - wheel_base
      - `float`
      - `0.34`
      - Distance between the front and rear axles of the robot. Must be between `0.0` and `100.0`.
    * - lookahead_gain_forward
      - `float`
      - `0.8`
      - Gain for lookahead distance calculation (k * v). Must be between `0.1` and `5.0`.
    * - prediction_horizon
      - `int`
      - `10`
      - Number of future steps for collision prediction. Must be between `0` and `100`.
    * - path_search_step
      - `float`
      - `0.2`
      - Offset step to search for a new path when doing obstacle avoidance.
    * - max_search_candidates
      - `int`
      - `10`
      - Number of search candidates to try for obstacle avoidance.
    ```
    """

    # Pure Pursuit specific params
    wheel_base: float = field(
        default=0.34, validator=base_validators.in_range(min_value=0.0, max_value=100.0)
    )
    lookahead_gain_forward: float = field(
        default=0.8, validator=base_validators.in_range(min_value=0.1, max_value=5.0)
    )
    # Collision avoidance params
    prediction_horizon: int = field(
        default=10, validator=base_validators.in_range(min_value=0, max_value=100)
    )
    path_search_step: float = field(
        default=0.2,
        validator=base_validators.in_range(min_value=0.001, max_value=1000.0),
    )
    max_search_candidates: int = field(
        default=10, validator=base_validators.in_range(min_value=2, max_value=1000)
    )

    proximity_sensor_position_to_robot: np.ndarray = field(
        default=np.array([0.0, 0.0, 0.0], dtype=np.float32)
    )

    proximity_sensor_rotation_to_robot: np.ndarray = field(
        default=np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
    )

    def to_kompass_cpp(self) -> kompass_cpp.control.PurePursuitConfig:
        """
        Convert to kompass_cpp lib config format

        :return: C++ Config Object
        :rtype: kompass_cpp.control.PurePursuitConfig
        """
        pp_config = kompass_cpp.control.PurePursuitConfig()

        # Map attributes to C++ config
        pp_config.from_dict(self.asdict())

        return pp_config


class PurePursuit(FollowerTemplate):
    def __init__(
        self,
        robot: Robot,
        ctrl_limits: RobotCtrlLimits,
        config: Optional[PurePursuitConfig] = None,
        config_file: Optional[str] = None,
        config_root_name: Optional[str] = None,
        control_time_step: float = 0.1,
        sensor_position: Optional[List[float]] = None,
        sensor_rotation: Optional[List[float]] = None,
        octree_res: float = 0.1,
        **_,
    ):
        """
        Init Pure Pursuit controller

        :param robot: Robot object to be controlled
        :type robot: Robot
        :param ctrl_limits: Robot control limits
        :type ctrl_limits: RobotCtrlLimits
        :param config: Controller configuration, defaults to None
        :type config: Optional[PurePursuitConfig], optional
        :param config_file: Path to config file (yaml, json, toml), defaults to None
        :type config_file: Optional[str], optional
        :param config_root_name: Root name for the controller config in the file, defaults to None
        :type config_root_name: Optional[str], optional
        :param generate_reference: Use to generate reference commands, defaults to False
        :type generate_reference: bool, optional
        """
        self._robot = robot

        # Defaults for sensor transform if not provided
        if sensor_position is None:
            sensor_position = [0.0, 0.0, 0.0]
        if sensor_rotation is None:
            sensor_rotation = [0.0, 0.0, 0.0, 1.0]

        # Init and configure the follower
        if not config:
            config = PurePursuitConfig(wheel_base=robot.wheelbase)

        if config_file:
            config.from_file(file_path=config_file, nested_root_name=config_root_name)

        self._config = config
        self._control_time_step = control_time_step

        self._planner = kompass_cpp.control.PurePursuit(
            control_type=RobotType.to_kompass_cpp_lib(robot.robot_type),
            control_limits=ctrl_limits.to_kompass_cpp_lib(),
            robot_shape_type=RobotGeometry.Type.to_kompass_cpp_lib(robot.geometry_type),
            robot_dimensions=robot.geometry_params.tolist(),
            sensor_position_robot=config.proximity_sensor_position_to_robot,
            sensor_rotation_robot=config.proximity_sensor_rotation_to_robot,
            octree_res=octree_res,
            config=config.to_kompass_cpp(),
        )

        # Init the following result
        self._result = None
        logging.info("PURE PURSUIT CONTROLLER IS READY")

    @property
    def planner(self) -> kompass_cpp.control.Follower:
        return self._planner

    def loop_step(self, *, current_state: RobotState, **kwargs) -> bool:
        """
        Implements a loop iteration of the controller

        :param current_state: Robot current state
        """
        self._planner.set_current_state(
            current_state.x, current_state.y, current_state.yaw, current_state.speed
        )

        current_velocity = kompass_cpp.types.ControlCmd(
            vx=current_state.vx, vy=current_state.vy, omega=current_state.omega
        )

        self._planner.set_current_velocity(current_velocity)

        # Execute controller
        # Check for sensor data to determine which execute overload to call
        if "local_map" in kwargs:
            # Execute with PointCloud
            self._result = self._planner.execute(
                self._control_time_step, kwargs["local_map"]
            )
        if "laser_scan" in kwargs:
            # Execute with LaserScan
            sensor_data = kompass_cpp.types.LaserScan(
                ranges=kwargs["laser_scan"].ranges, angles=kwargs["laser_scan"].angles
            )
            self._result = self._planner.execute(self._control_time_step, sensor_data)
        elif "point_cloud" in kwargs:
            # Execute with PointCloud
            sensor_data = kwargs["point_cloud"].data
            self._result = self._planner.execute(self._control_time_step, sensor_data)
        else:
            # Execute Nominal (State Update + Control)
            self._result = self._planner.execute(self._control_time_step)

        return self._result.status in [
            kompass_cpp.control.FollowingStatus.COMMAND_FOUND,
            kompass_cpp.control.FollowingStatus.GOAL_REACHED,
        ]

    def logging_info(self) -> str:
        """Get logging information

        :return: Information
        :rtype: str
        """
        if self._result:
            vel = self._result.velocity_command
            return f"Follower status: {self._result.status}, Cmd: vx={vel.vx:.2f}, vy={vel.vy:.2f}, w={vel.omega:.2f}"
        return "Follower not started"

    @property
    def linear_x_control(self) -> List[float]:
        """
        Getter of the last linear forward velocity control computed by the controller

        :return: Linear Velocity Control (m/s)
        :rtype: List[float]
        """
        if not self._result:
            return [0.0]

        vx = self._result.velocity_command.vx

        return [vx]

    @property
    def linear_y_control(self) -> List[float]:
        """
        Getter the last linear velocity lateral control computed by the controller

        :return: Linear Velocity Control (m/s)
        :rtype: List[float]
        """
        if not self._result:
            return [0.0]

        vy = self._result.velocity_command.vy

        return [vy]

    @property
    def angular_control(self) -> List[float]:
        """
        Getter of the last angular velocity control computed by the controller

        :return: Angular Velocity Control (rad/s)
        :rtype: List[float]
        """
        if not self._result:
            return [0.0]

        omega = self._result.velocity_command.omega

        return [omega]
