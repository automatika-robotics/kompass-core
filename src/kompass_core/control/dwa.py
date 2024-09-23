import logging
from typing import List, Optional
import numpy as np
from attrs import Factory, define, field
from ..datatypes.laserscan import LaserScanData
from ..datatypes.pointcloud import PointCloudData
from ..utils.common import in_range
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped

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
    DWA Local Planner Parameters
    """

    control_time_step: float = field(
        default=0.1, validator=in_range(min_value=1e-4, max_value=1e6)
    )

    prediction_horizon: float = field(
        default=1.0, validator=in_range(min_value=1e-4, max_value=1e6)
    )

    control_horizon: float = field(
        default=0.2, validator=in_range(min_value=1e-4, max_value=1e6)
    )

    max_linear_samples: int = field(
        default=20, validator=in_range(min_value=1, max_value=1e3)
    )

    max_angular_samples: int = field(
        default=20, validator=in_range(min_value=1, max_value=1e3)
    )

    sensor_position_to_robot: List[float] = field(default=[0.0, 0.0, 0.0])

    sensor_rotation_to_robot: List[float] = field(default=[0.0, 0.0, 0.0, 1.0])

    octree_resolution: float = field(
        default=0.1, validator=in_range(min_value=1e-9, max_value=1e3)
    )

    costs_weights: TrajectoryCostsWeights = Factory(TrajectoryCostsWeights)

    def __attrs_post_init__(self):
        """Attrs post init"""
        if self.control_horizon > self.prediction_horizon:
            logging.error(
                f"Control horizon (={self.control_horizon}) cannot be greater than the future prediction horizon (={self.prediction_horizon}). Setting the control horizon to max available (control_horizon={self.prediction_horizon})"
            )
            self.control_horizon = self.prediction_horizon


class DWA(FollowerTemplate):
    """DWA."""

    def __init__(
        self,
        robot: Robot,
        ctrl_limits: RobotCtrlLimits,
        config: Optional[DWAConfig] = None,
        config_file: Optional[str] = None,
        config_yaml_root_name: Optional[str] = None,
        control_time_step: Optional[float] = None,
        prediction_horizon: Optional[float] = None,
        **_,
    ):
        """
        Setup the controller

        :param robot: Robot using the controller
        :type robot: Robot
        :param params_file: Yaml file containing the parameters of the controller under 'dvz_controller'
        :type params_file: str
        """
        # Init and configure the planner
        if not config:
            # Default config
            config = DWAConfig()

        if config_file:
            config.from_yaml(
                file_path=config_file, nested_root_name=config_yaml_root_name
            )

        if control_time_step:
            config.control_time_step = control_time_step

        if prediction_horizon:
            config.prediction_horizon = prediction_horizon

        self._got_path = False

        self._config = config

        self._planner = kompass_cpp.control.DWA(
            control_limits=ctrl_limits.to_kompass_cpp_lib(),
            control_type=RobotType.to_kompass_cpp_lib(robot.robot_type),
            time_step=config.control_time_step,
            prediction_horizon=config.prediction_horizon,
            control_horizon=config.control_horizon,
            max_linear_samples=config.max_linear_samples,
            max_angular_samples=config.max_angular_samples,
            robot_shape_type=RobotGeometry.Type.to_kompass_cpp_lib(robot.geometry_type),
            robot_dimensions=robot.geometry_params,
            sensor_position_robot=config.sensor_position_to_robot,
            sensor_rotation_robot=config.sensor_rotation_to_robot,
            octree_resolution=config.octree_resolution,
            cost_weights=config.costs_weights.to_kompass_cpp(),
        )

        # Init the following result
        self._result = kompass_cpp.control.SamplingControlResult()
        logging.info("DWA PLANNER IS READY")

    @property
    def planner(self) -> kompass_cpp.control.Follower:
        return self._planner

    def loop_step(
        self,
        *,
        current_state: RobotState,
        laser_scan: Optional[LaserScanData] = None,
        point_cloud: Optional[PointCloudData] = None,
        local_map: Optional[np.ndarray] = None,
        **_,
    ) -> bool:
        """
        One iteration of the DWA planner

        :param current_state: Current robot state (position and velocity)
        :type current_state: RobotState
        :param laser_scan: Current laser scan value
        :type laser_scan: LaserScanData

        :return: If planner found a valid solution
        :rtype: bool
        """
        if not self._got_path:
            logging.error("Path is not available to DWA controller")
            return False

        self._planner.set_current_state(
            current_state.x, current_state.y, current_state.yaw, current_state.speed
        )

        # If end point is reached -> no need to compute a new control
        if self.reached_end():
            logging.info("End is reached")
            self._result.is_found = False
            return False

        current_velocity = kompass_cpp.types.ControlCmd(
            vx=current_state.vx, vy=current_state.vy, omega=current_state.omega
        )

        if local_map is not None:
            sensor_data = PointCloudData.numpy_to_kompass_cpp(local_map)
        elif laser_scan:
            if len(laser_scan.angles) != len(laser_scan.ranges):
                logging.error(
                    "Received incompatible LaserScan data -> Cannot compute control"
                )
                return False
            sensor_data = kompass_cpp.types.LaserScan(
                ranges=laser_scan.ranges, angles=laser_scan.angles
            )
        elif point_cloud:
            sensor_data = point_cloud.to_kompass_cpp()
        else:
            logging.error(
                "Cannot compute control without sensor data. Provide 'laser_scan' or 'point_cloud' input"
            )
            return False

        try:
            self._result = self._planner.compute_velocity_commands(
                current_velocity, sensor_data
            )

        except Exception as e:
            logging.error(f"Could not find velocity command: {e}")
            return False

        return self._result.is_found

    def set_path(self, global_path: Path, **_) -> None:
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
            return f"Controller found trajectory with cost: {self._result.cost}, Velocity command: {self._result.trajectory.velocity[0]}"
        else:
            return "Controller Failed"

    @property
    def control_till_horizon(self) -> Optional[List[kompass_cpp.types.ControlCmd]]:
        """
        Getter of the planner control result until the control horizon

        :return: Velocity commands of the minimal cost path
        :rtype: List[kompass_cpp.types.ControlCmd]
        """
        if self._result.is_found:
            end_of_ctrl_horizon: int = max(
                int(self._config.control_horizon / self._config.control_time_step), 1
            )
            return self._result.trajectory.velocity[:end_of_ctrl_horizon]
        return None

    def optimal_path(self, msg_header) -> Optional[Path]:
        """Get optimal (local) plan."""
        if not self._result.is_found:
            return None
        kompass_cpp_path: kompass_cpp.types.Path = self._result.trajectory.path
        ros_path = Path()
        ros_path.header = msg_header
        parsed_points = []
        for point in kompass_cpp_path.points:
            ros_point = PoseStamped()
            ros_point.header = msg_header
            ros_point.pose.position.x = point.x
            ros_point.pose.position.y = point.y
            parsed_points.append(ros_point)

        ros_path.poses = parsed_points
        return ros_path

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
    def linear_x_control(self) -> List[float]:
        """
        Getter of the last linear forward velocity control computed by the controller

        :return: Linear Velocity Control (m/s)
        :rtype: float
        """
        if self._result.is_found:
            return [vel.vx for vel in self.control_till_horizon]
        return [0.0]

    @property
    def linear_y_control(self) -> List[float]:
        """
        Getter the last linear velocity lateral control computed by the controller

        :return: Linear Velocity Control (m/s)
        :rtype: float
        """
        if self._result.is_found:
            return [vel.vy for vel in self.control_till_horizon]
        return [0.0]

    @property
    def angular_control(self) -> List[float]:
        """
        Getter of the last angular velocity control computed by the controller

        :return: Angular Velocity Control (rad/s)
        :rtype: float
        """
        if self._result.is_found:
            return [vel.omega for vel in self.control_till_horizon]
        return [0.0]
