import logging
from typing import Optional, Union, List
import numpy as np
from attrs import Factory, define, field
from ..utils.common import base_validators

import kompass_cpp
from ..models import (
    Robot,
    RobotCtrlLimits,
    RobotGeometry,
    RobotState,
    RobotType,
)

from ._base_ import FollowerTemplate, FollowerConfig
from ._trajectory_ import TrajectoryCostsWeights


@define
class DWAConfig(FollowerConfig):
    """
    DWA Parameters

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
      - Time interval between control actions (sec). Must be between `1e-4` and `1e6`.

    * - prediction_horizon
      - `float`
      - `1.0`
      - Duration over which predictions are made (sec). Must be between `1e-4` and `1e6`.

    * - control_horizon
      - `float`
      - `0.2`
      - Duration over which control actions are planned (sec). Must be between `1e-4` and `1e6`.

    * - max_linear_samples
      - `int`
      - `20`
      - Maximum number of linear control samples. Must be between `1` and `1e3`.

    * - max_angular_samples
      - `int`
      - `20`
      - Maximum number of angular control samples. Must be between `1` and `1e3`.

    * - sensor_position_to_robot
      - `List[float]`
      - `[0.0, 0.0, 0.0]`
      - Position of the sensor relative to the robot in 3D space (x, y, z) coordinates.

    * - sensor_rotation_to_robot
      - `List[float]`
      - `[0.0, 0.0, 0.0, 1.0]`
      - Orientation of the sensor relative to the robot as a quaternion (x, y, z, w).

    * - octree_resolution
        - `float`
        - `0.1`
        - Resolution of the Octree used for collision checking. Must be between `1e-9` and `1e3`.

    * - costs_weights
      - `TrajectoryCostsWeights`
      -
      - Weights for trajectory cost evaluation.

    * - max_num_threads
      - `int`
      - `1`
      - Maximum number of threads used when running the controller. Must be between `1` and `1e2`.

    * - drop_samples
      - `bool`
      - `True`
      - To drop the entire sample once a collision is detected (True), or maintain the first collision-free segment of the sample (if False)

    ```
    """

    control_time_step: float = field(
        default=0.1, validator=base_validators.in_range(min_value=1e-4, max_value=1e6)
    )

    control_horizon: int = field(
        default=2, validator=base_validators.in_range(min_value=1, max_value=1000)
    )
    # Number of steps for applying the control

    prediction_horizon: int = field(
        default=10, validator=base_validators.in_range(min_value=1, max_value=1000)
    )
    # Number of steps for future prediction

    max_linear_samples: int = field(
        default=20, validator=base_validators.in_range(min_value=1, max_value=1e3)
    )

    max_angular_samples: int = field(
        default=20, validator=base_validators.in_range(min_value=1, max_value=1e3)
    )

    proximity_sensor_position_to_robot: np.ndarray = field(
        default=np.array([0.0, 0.0, 0.0], dtype=np.float32)
    )

    proximity_sensor_rotation_to_robot: np.ndarray = field(
        default=np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
    )

    octree_resolution: float = field(
        default=0.1, validator=base_validators.in_range(min_value=1e-9, max_value=1e3)
    )

    costs_weights: TrajectoryCostsWeights = Factory(TrajectoryCostsWeights)

    max_num_threads: int = field(
        default=1, validator=base_validators.in_range(min_value=1, max_value=1e2)
    )

    drop_samples: bool = field(default=True)

    def __attrs_post_init__(self):
        """Attrs post init"""
        if self.control_horizon > self.prediction_horizon:
            logging.error(
                f"Control horizon (={self.control_horizon}) cannot be greater than the future prediction horizon (={self.prediction_horizon}). Setting the control horizon to max available (control_horizon={self.prediction_horizon})"
            )
            self.control_horizon = self.prediction_horizon


class DWA(FollowerTemplate):
    """
    DWA is a popular local planning method developed since the 90s. DWA is a sampling-method consists of sampling a set of constant velocity trajectories within a window of admissible reachable velocities. This window of reachable velocities will change based on the current velocity and the acceleration limits, i.e. a Dynamic Window.

    ```python
    from kompass_core.control import DWAConfig, DWA
    from kompass_core.models import (
        AngularCtrlLimits,
        LinearCtrlLimits,
        Robot,
        RobotCtrlLimits,
        RobotGeometry,
        RobotType,
    )

    # Configure the robot
    my_robot = Robot(
            robot_type=RobotType.ACKERMANN,
            geometry_type=RobotGeometry.Type.CYLINDER,
            geometry_params=np.array([0.1, 0.4]),
        )

    # Configure the control limits (used to compute the dynamic window)
    robot_ctr_limits = RobotCtrlLimits(
        vx_limits=LinearCtrlLimits(max_vel=1.0, max_acc=5.0, max_decel=10.0),
        omega_limits=AngularCtrlLimits(
            max_omega=2.0, max_acc=3.0, max_decel=3.0, max_ang=np.pi
        ),
    )

    # Configure the controller
    config = DWAConfig(
            max_linear_samples=20,
            max_angular_samples=20,
            octree_resolution=0.1,
            prediction_horizon=1.0,
            control_horizon=0.2,
            control_time_step=0.1,
        )

    controller = DWA(robot=my_robot, ctrl_limits=robot_ctr_limits, config=config)
    ```

    """

    def __init__(
        self,
        robot: Robot,
        ctrl_limits: RobotCtrlLimits,
        config: Optional[DWAConfig] = None,
        config_file: Optional[str] = None,
        config_root_name: Optional[str] = None,
        control_time_step: Optional[float] = None,
        **_,
    ):
        """Init DWA (Dynamic Window Approach) Controller

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
        :param control_time_step: Control time step (sec), defaults to None
        :type control_time_step: Optional[float], optional
        """
        # Init and configure the planner
        self._config = config or DWAConfig()

        if config_file:
            self._config.from_file(
                file_path=config_file, nested_root_name=config_root_name
            )

        if control_time_step:
            self._config.control_time_step = control_time_step

        self._got_path = False

        self._planner = kompass_cpp.control.DWA(
            control_limits=ctrl_limits.to_kompass_cpp_lib(),
            control_type=robot.robot_type,
            time_step=self._config.control_time_step,
            prediction_horizon=self._config.prediction_horizon
            * self._config.control_time_step,
            control_horizon=self._config.control_horizon
            * self._config.control_time_step,
            max_linear_samples=self._config.max_linear_samples,
            max_angular_samples=self._config.max_angular_samples,
            robot_shape_type=robot.geometry_type,
            robot_dimensions=robot.geometry_params,
            sensor_position_robot=self._config.proximity_sensor_position_to_robot,
            sensor_rotation_robot=self._config.proximity_sensor_rotation_to_robot,
            octree_resolution=self._config.octree_resolution,
            cost_weights=self._config.costs_weights.to_kompass_cpp(),
            max_num_threads=self._config.max_num_threads,
        )

        # Init the following result
        self._result = kompass_cpp.control.SamplingControlResult()
        self._end_of_ctrl_horizon: int = max(self._config.control_horizon, 1)
        logging.info("DWA PATH CONTROLLER IS READY")

    @property
    def planner(self) -> kompass_cpp.control.Follower:
        return self._planner

    def loop_step(
        self,
        *,
        current_state: RobotState,
        ranges: Optional[np.ndarray] = None,
        angles: Optional[np.ndarray] = None,
        points: Optional[np.ndarray] = None,
        local_map: Optional[np.ndarray] = None,
        local_map_resolution: Optional[float] = None,
        debug: bool = False,
        **_,
    ) -> bool:
        """
        One iteration of the DWA planner

        Takes obstacles from exactly one source, in order of precedence:
        ``local_map``, then a laser scan (``ranges`` with ``angles``), then a
        cartesian point cloud (``points``).

        Frame contract, set by the input type:

        - A laser scan (``ranges`` with ``angles``) is in the **sensor
          frame**. The planner applies the sensor-to-robot mount pose and the
          robot's world pose itself.
        - ``points`` and ``local_map`` are Cartesian obstacle points in 3D
          space, already in the **world frame** -- the same frame the
          trajectory samples and the reference path live in. They are handed
          to the planner untouched, so any sensor-to-world transform must
          already have been applied upstream (in the ROS callbacks that
          decode the point cloud or the occupancy grid).

        :param current_state: Current robot state (position and velocity)
        :type current_state: RobotState
        :param ranges: Measured range along each angle of a laser scan (m),
            in the sensor frame
        :type ranges: Optional[np.ndarray]
        :param points: Cartesian obstacle points as an Nx3 array (m), in the
            world frame
        :type points: Optional[np.ndarray]
        :param angles: Angle of each range measurement (rad)
        :type angles: Optional[np.ndarray]
        :param local_map: Occupied cells as an Nx3 array (m), in the world
            frame
        :type local_map: Optional[np.ndarray]

        :return: If planner found a valid solution
        :rtype: bool
        """
        if not self._got_path:
            logging.error("Path is not available to DWA controller")
            return False

        self._planner.set_current_state(
            current_state.x, current_state.y, current_state.yaw, current_state.speed
        )

        if local_map_resolution:
            self._planner.set_resolution(local_map_resolution)

        # If end point is reached -> no need to compute a new control
        if self.reached_end():
            logging.info("End is reached")
            self._result.is_found = False
            return False

        current_velocity = kompass_cpp.types.Velocity2D(
            vx=current_state.vx, vy=current_state.vy, omega=current_state.omega
        )

        # float32 C-contiguous is the zero-copy fast path at the binding
        # (no-op for ROS-native data)
        if local_map is not None:
            sensor_args = (np.ascontiguousarray(local_map, dtype=np.float32),)
        elif ranges is not None and angles is not None:
            if len(angles) != len(ranges):
                logging.error(
                    "Received incompatible LaserScan data -> Cannot compute control"
                )
                return False
            # Passed as raw arrays. The scan is built C++-side after the GIL
            # release
            sensor_args = (
                np.asarray(ranges, dtype=np.float32),
                np.asarray(angles, dtype=np.float32),
            )
        elif points is not None:
            sensor_args = (np.ascontiguousarray(points, dtype=np.float32),)
        else:
            logging.error(
                "Cannot compute control without sensor data. Provide 'ranges' and 'angles', 'points' or 'local_map' input"
            )
            return False

        try:
            if debug:
                self._planner.debug_velocity_search(
                    current_velocity, *sensor_args, self._config.drop_samples
                )
            self._result = self._planner.compute_velocity_commands(
                current_velocity, *sensor_args
            )

        except Exception as e:
            logging.error(f"Could not find velocity command: {e}")
            return False

        return True

    def has_result(self) -> None:
        """
        Set global path to be tracked by the planner

        :param global_path: Global reference path
        :type global_path: Path
        """
        return self._result.is_found

    def set_path(self, global_path, **_) -> None:
        """
        Set global path to be tracked by the planner

        :param global_path: Global reference path
        :type global_path: Path
        """
        super().set_path(global_path)
        self._got_path = True

    def logging_info(self) -> str:
        """logging_info."""
        if self._result.is_found:
            return f"DWA Controller found trajectory with cost: {self._result.cost}"
        else:
            return "DWA Controller Failed to find a valid trajectory"

    @property
    def control_till_horizon(
        self,
    ) -> Optional[kompass_cpp.types.TrajectoryVelocities2D]:
        """
        Getter of the planner control result until the control horizon

        :return: Velocity commands of the minimal cost path
        :rtype: List[kompass_cpp.types.TrajectoryVelocities2D]
        """
        if self._result.is_found:
            return self._result.trajectory.velocities
        return None

    def optimal_path(self) -> Optional[kompass_cpp.types.TrajectoryPath]:
        """Get optimal (local) plan."""
        if not self._result.is_found:
            return None
        return self._result.trajectory.path

    @property
    def result_cost(self) -> Optional[float]:
        """
        Getter of the planner optimal path

        :return: Path found with the least cost
        :rtype: Optional[float]
        """
        if self._result.is_found:
            return self._result.cost
        return None

    @property
    def linear_x_control(self) -> Union[List[float], np.ndarray]:
        """
        Getter of the last linear forward velocity control computed by the controller

        :return: Linear Velocity Control (m/s)
        :rtype: List[float]
        """
        if self._result.is_found:
            return self.control_till_horizon.vx[: self._end_of_ctrl_horizon]
        return [0.0]

    @property
    def linear_y_control(self) -> Union[List[float], np.ndarray]:
        """
        Getter the last linear velocity lateral control computed by the controller

        :return: Linear Velocity Control (m/s)
        :rtype: List[float]
        """
        if self._result.is_found:
            return self.control_till_horizon.vy[: self._end_of_ctrl_horizon]
        return [0.0]

    @property
    def angular_control(self) -> Union[List[float], np.ndarray]:
        """
        Getter of the last angular velocity control computed by the controller

        :return: Angular Velocity Control (rad/s)
        :rtype: List[float]
        """
        if self._result.is_found:
            return self.control_till_horizon.omega[: self._end_of_ctrl_horizon]
        return [0.0]
