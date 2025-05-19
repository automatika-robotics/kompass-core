from attrs import define, field
from ..utils.common import base_validators
from kompass_cpp.control import (
    VisionDWA as VisionDWACpp,
    VisionDWAParameters,
    SamplingControlResult,
)
from kompass_cpp.types import (
    Bbox3D,
    Bbox2D,
    ControlCmd,
    LaserScan,
    TrajectoryVelocities2D,
    TrajectoryPath,
    TrackedPose2D
)
from typing import Optional, List
import numpy as np
import logging
from ._base_ import ControllerTemplate
from ..models import Robot, RobotState, RobotCtrlLimits, RobotGeometry, RobotType
from ..datatypes.laserscan import LaserScanData
from ..datatypes.pointcloud import PointCloudData
from .dwa import DWAConfig


@define
class VisionDWAConfig(DWAConfig):

    tolerance: float = field(
        default=0.01, validator=base_validators.in_range(min_value=1e-6, max_value=1e3)
    )
    # Tolerance value for distance and angle following errors

    target_distance: Optional[float] = field(
        default=0.1,
        validator=base_validators.in_range(min_value=1e-9, max_value=1e9),
    )  # Target distance to maintain with the target (m)

    target_orientation: float = field(
        default=0.0,
        validator=base_validators.in_range(min_value=-np.pi, max_value=np.pi),
    )  # Bearing angle to maintain with the target (rad)

    target_wait_timeout: float = field(
        default=30.0, validator=base_validators.in_range(min_value=0.0, max_value=1e3)
    )  # Wait for target to appear again timeout (seconds), used if search is disabled

    target_search_timeout: float = field(
        default=30.0, validator=base_validators.in_range(min_value=0.0, max_value=1e3)
    )  # Search timeout in seconds

    target_search_radius: float = field(
        default=0.5, validator=base_validators.in_range(min_value=1e-4, max_value=1e4)
    )  # Search radius for finding the target (m)

    target_search_pause: float = field(
        default=1.0, validator=base_validators.in_range(min_value=0.0, max_value=1e3)
    )  # Pause between search actions to find target (seconds)

    rotation_gain: float = field(
        default=0.5, validator=base_validators.in_range(min_value=1e-2, max_value=10.0)
    )  # Gain for the rotation control law

    speed_gain: float = field(
        default=1.0, validator=base_validators.in_range(min_value=1e-2, max_value=10.0)
    )  # Gain for the speed control law

    min_vel: float = field(
        default=0.1, validator=base_validators.in_range(min_value=1e-9, max_value=1e9)
    )  # Minimum velocity to apply (m/s)

    enable_search: bool = field(default=False)  # Enable or disable the search mechanism

    error_pose: float = field(
        default=0.05, validator=base_validators.in_range(min_value=1e-9, max_value=1e9)
    )  # Error in pose estimation (m)

    error_vel: float = field(
        default=0.05, validator=base_validators.in_range(min_value=1e-9, max_value=1e9)
    )  # Error in velocity estimation (m/s)

    error_acc: float = field(
        default=0.05, validator=base_validators.in_range(min_value=1e-9, max_value=1e9)
    )  # Error in acceleration estimation (m/s^2)

    depth_conversion_factor: float = field(
        default=1e-3, validator=base_validators.in_range(min_value=1e-9, max_value=1e9)
    )  # Factor to convert depth image values to meters

    min_depth: float = field(
        default=0.0, validator=base_validators.in_range(min_value=0.0, max_value=1e3)
    )  # Range of interest minimum depth value (m)

    max_depth: float = field(
        default=1e3, validator=base_validators.in_range(min_value=1e-3, max_value=1e9)
    )  # Range of interest maximum depth value (m)

    camera_position_to_robot: np.ndarray = field(
        default=np.array([0.0, 0.0, 0.0], dtype=np.float32)
    )

    camera_rotation_to_robot: np.ndarray = field(
        default=np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
    )

    def to_kompass_cpp(self) -> VisionDWAParameters:
        """
        Convert to kompass_cpp lib config format

        :return: _description_
        :rtype: kompass_cpp.control.VisionDWAParameters
        """
        vision_dwa_params = VisionDWAParameters()

        # Special handling for None values that are represented by -1 in C++
        params_dict = self.asdict()

        if params_dict["target_distance"] is None:
            params_dict["target_distance"] = -1.0

        vision_dwa_params.from_dict(params_dict)
        return vision_dwa_params


class VisionDWA(ControllerTemplate):
    def __init__(
        self,
        robot: Robot,
        ctrl_limits: RobotCtrlLimits,
        config: Optional[VisionDWAConfig] = None,
        config_file: Optional[str] = None,
        config_yaml_root_name: Optional[str] = None,
        control_time_step: Optional[float] = None,
        camera_focal_length: Optional[List[float]] = None,
        camera_principal_point: Optional[List[float]] = None,
        **_,
    ):
        """
        Setup the controller

        :param robot: Robot using the controller
        :type robot: Robot
        :param params_file: Yaml file containing the parameters of the controller under 'dvz_controller'
        :type params_file: str
        """
        self._config = config or VisionDWAConfig()

        if config_file:
            self._config.from_yaml(
                file_path=config_file, nested_root_name=config_yaml_root_name
            )

        if control_time_step:
            self._config.control_time_step = control_time_step

        self._planner = VisionDWACpp(
            control_type=RobotType.to_kompass_cpp_lib(robot.robot_type),
            control_limits=ctrl_limits.to_kompass_cpp_lib(),
            max_linear_samples=self._config.max_linear_samples,
            max_angular_samples=self._config.max_angular_samples,
            robot_shape_type=RobotGeometry.Type.to_kompass_cpp_lib(robot.geometry_type),
            robot_dimensions=robot.geometry_params,
            proximity_sensor_position_wrt_body=self._config.proximity_sensor_position_to_robot,
            proximity_sensor_rotation_wrt_body=self._config.proximity_sensor_rotation_to_robot,
            vision_sensor_position_wrt_body=self._config.camera_position_to_robot,
            vision_sensor_rotation_wrt_body=self._config.camera_rotation_to_robot,
            octree_res=self._config.octree_resolution,
            cost_weights=self._config.costs_weights.to_kompass_cpp(),
            max_num_threads=self._config.max_num_threads,
            config=self._config.to_kompass_cpp(),
        )
        if camera_focal_length is not None and camera_principal_point is not None:
            self._planner.set_camera_intrinsics(camera_focal_length[0], camera_focal_length[1], camera_principal_point[0], camera_principal_point[1])

        # Init the following result
        self._result = SamplingControlResult()
        self._end_of_ctrl_horizon: int = max(
            self._config.control_horizon, 1
        )
        logging.info("VisionDWA CONTROLLER IS READY")

    def set_camera_intrinsics(self, fx: float, fy: float, cx: float, cy: float) -> None:
        self._planner.set_camera_intrinsics(fx, fy, cx, cy)

    def set_initial_tracking_3d(
        self,
        pose_x_img: int,
        pose_y_img: int,
        detected_boxes: List[Bbox3D],
    ) -> bool:
        """
        Set initial tracking state

        :param detected_boxes: Detected boxes
        :type detected_boxes: List[Bbox3D]
        """
        try:
            if any(detected_boxes):
                return self._planner.set_initial_tracking(pose_x_img, pose_y_img, detected_boxes)

            logging.error(f"Could not set initial tracking state: No detections are provided")
            return False
        except Exception as e:
            logging.error(f"Could not set initial tracking state: {e}")
            return False

    def set_initial_tracking_depth(
        self,
        current_state: RobotState,
        pose_x_img: int,
        pose_y_img: int,
        detected_boxes: List[Bbox2D],
        aligned_depth_image: np.ndarray,
    ) -> bool:
        """
        Set initial tracking state

        :param detected_boxes: Detected boxes
        :type detected_boxes: List[Bbox3D]
        """
        try:
            self._planner.set_current_state(
            current_state.x, current_state.y, current_state.yaw, current_state.speed)
            if any(detected_boxes):
                return self._planner.set_initial_tracking(
                    pose_x_img, pose_y_img, aligned_depth_image, detected_boxes, current_state.yaw
                )
            logging.error(f"Could not set initial tracking state: No detections are provided")
            return False

        except Exception as e:
            logging.error(f"Could not set initial tracking state: {e}")
            return False

    def loop_step(
        self,
        *,
        current_state: RobotState,
        detections_3d: Optional[List[Bbox3D]] = None,
        detections_2d: Optional[List[Bbox2D]] = None,
        depth_image: Optional[np.ndarray] = None,
        tracked_pose: Optional[TrackedPose2D] = None,
        laser_scan: Optional[LaserScanData] = None,
        point_cloud: Optional[List[np.ndarray]] = None,
        local_map: Optional[np.ndarray] = None,
        local_map_resolution: Optional[float] = None,
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
        self._planner.set_current_state(
            current_state.x, current_state.y, current_state.yaw, current_state.speed
        )

        if local_map_resolution:
            self._planner.set_resolution(local_map_resolution)

        current_velocity = ControlCmd(
            vx=current_state.vx, vy=current_state.vy, omega=current_state.omega
        )

        if local_map is not None:
            sensor_data = PointCloudData.numpy_to_kompass_cpp(local_map)
        elif laser_scan:
            sensor_data = LaserScan(
                ranges=laser_scan.ranges, angles=laser_scan.angles
            )
        elif point_cloud is not None:
            sensor_data = point_cloud
        else:
            logging.error(
                "Cannot compute control without sensor data. Provide 'laser_scan' or 'point_cloud' input"
            )
            return False

        try:
            if (detections_3d or tracked_pose) is not None:
                self._result = self._planner.get_tracking_ctrl(
                    tracked_pose or detections_3d, current_velocity, sensor_data
                )
            else:
                self._result = self._planner.get_tracking_ctrl(
                    depth_image, detections_2d, current_velocity, sensor_data
                )

        except Exception as e:
            logging.error(f"Could not find velocity command: {e}")
            return False

        return self._result.is_found

    def has_result(self) -> None:
        """
        Set global path to be tracked by the planner

        :param global_path: Global reference path
        :type global_path: Path
        """
        return self._result.is_found

    def logging_info(self) -> str:
        """logging_info."""
        if self._result.is_found:
            return f"VisionDWA Controller found trajectory with cost: {self._result.cost}"
        else:
            return "VisionDWA Controller Failed to find a valid trajectory"

    @property
    def control_till_horizon(
        self,
    ) -> Optional[TrajectoryVelocities2D]:
        """
        Getter of the planner control result until the control horizon

        :return: Velocity commands of the minimal cost path
        :rtype: List[kompass_cpp.types.TrajectoryVelocities2D]
        """
        if self._result.is_found:
            return self._result.trajectory.velocities
        return None

    def optimal_path(self) -> Optional[TrajectoryPath]:
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
    def linear_x_control(self) -> np.ndarray:
        """
        Getter of the last linear forward velocity control computed by the controller

        :return: Linear Velocity Control (m/s)
        :rtype: List[float]
        """
        if self._result.is_found:
            return self.control_till_horizon.vx[: self._end_of_ctrl_horizon]
        return [0.0]

    @property
    def linear_y_control(self) -> np.ndarray:
        """
        Getter the last linear velocity lateral control computed by the controller

        :return: Linear Velocity Control (m/s)
        :rtype: List[float]
        """
        if self._result.is_found:
            return self.control_till_horizon.vy[: self._end_of_ctrl_horizon]
        return [0.0]

    @property
    def angular_control(self) -> np.ndarray:
        """
        Getter of the last angular velocity control computed by the controller

        :return: Angular Velocity Control (rad/s)
        :rtype: List[float]
        """
        if self._result.is_found:
            return self.control_till_horizon.omega[: self._end_of_ctrl_horizon]
        return [0.0]
