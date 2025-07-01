from attrs import define, field
from ..utils.common import base_validators
from kompass_cpp.control import (
    VisionDWA as VisionDWACpp,
    VisionDWAParameters,
    SamplingControlResult,
)
from kompass_cpp.types import (
    Bbox2D,
    ControlCmd,
    LaserScan,
    TrajectoryVelocities2D,
    TrajectoryPath,
)
from typing import Optional, List, Union
import numpy as np
import logging
from ._base_ import ControllerTemplate
from ..models import Robot, RobotState, RobotCtrlLimits, RobotGeometry, RobotType
from ..datatypes.laserscan import LaserScanData
from ..datatypes.pointcloud import PointCloudData
from .dwa import DWAConfig


@define
class VisionRGBDFollowerConfig(DWAConfig):
    """
    Configuration class for a vision-based RGB-D target follower using Dynamic Window Approach (DWA) planning.

    Attributes:
        control_time_step (float): Time interval between control updates (s).
        control_horizon (int): Number of steps in the control horizon.
        prediction_horizon (int): Number of steps in the prediction horizon.
        buffer_size (int): Size of the trajectory buffer.
        target_distance (Optional[float]): Desired distance to maintain from the target (m).
        target_wait_timeout (float): Time to wait for the target to reappear before giving up (s).
        target_search_timeout (float): Time limit for the search process if the target is lost (s).
        target_search_pause (float): Pause between successive search attempts (s).
        target_search_radius (float): Radius within which to search for the lost target (m).
        rotation_gain (float): Gain applied to rotational control (unitless).
        speed_gain (float): Gain applied to speed control (unitless).
        enable_search (bool): Enables or disables the target search behavior.
        distance_tolerance (float): Acceptable deviation in target distance (m).
        angle_tolerance (float): Acceptable deviation in target bearing (rad).
        target_orientation (float): Desired orientation relative to the target (rad).
        use_local_coordinates (bool): Whether to use robot-local coordinates for tracking.
        error_pose (float): Estimated error in pose measurements (m).
        error_vel (float): Estimated error in velocity measurements (m/s).
        error_acc (float): Estimated error in acceleration measurements (m/s²).
        depth_conversion_factor (float): Factor to convert raw depth image values to meters.
        min_depth (float): Minimum depth value considered valid (m).
        max_depth (float): Maximum depth value considered valid (m).
        camera_position_to_robot (np.ndarray): 3D translation vector from the camera frame to the robot base (m).
        camera_rotation_to_robot (np.ndarray): Quaternion representing camera-to-robot rotation.
    """
    control_time_step: float = field(
        default=0.1, validator=base_validators.in_range(min_value=1e-4, max_value=1e6)
    )
    control_horizon: int = field(
        default=2, validator=base_validators.in_range(min_value=1, max_value=1000)
    )
    prediction_horizon: int = field(
        default=10, validator=base_validators.in_range(min_value=1, max_value=1000)
    )
    buffer_size: int = field(
        default=1, validator=base_validators.in_range(min_value=1, max_value=10)
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
    enable_search: bool = field(default=True)
    distance_tolerance: float = field(
        default=0.05, validator=base_validators.in_range(min_value=1e-6, max_value=1e3)
    )
    angle_tolerance: float = field(
        default=0.1, validator=base_validators.in_range(min_value=1e-6, max_value=1e3)
    )
    # Tolerance value for distance and angle following errors

    target_orientation: float = field(
        default=0.0,
        validator=base_validators.in_range(min_value=-np.pi, max_value=np.pi),
    )  # Bearing angle to maintain with the target (rad)

    rotation_gain: float = field(
        default=0.5, validator=base_validators.in_range(min_value=1e-2, max_value=10.0)
    )  # Gain for the rotation control law

    speed_gain: float = field(
        default=1.0, validator=base_validators.in_range(min_value=1e-2, max_value=10.0)
    )  # Gain for the speed control law

    use_local_coordinates: bool = field(
        default=True
    )  # Track the target using robot local coordinates (no need for robot location at lop step)

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


class VisionRGBDFollower(ControllerTemplate):
    """
    VisionRGBDFollower is a controller for vision-based target tracking using RGB-D (color + depth) image data.

    This controller combines image-based detections (2D bounding boxes) and depth data to estimate 3D positions
    of visual targets and uses a sampling-based planner (similar to DWA) to compute optimal local motion commands.
    It integrates camera intrinsics, robot geometry, and multiple sensor modalities (e.g., point clouds, laser scans,
    local maps) to generate robust and feasible trajectories for following dynamic or static targets in the environment.

    - Usage Example:

    ```python
    import numpy as np
    from kompass_core.control import VisionRGBDFollower, VisionRGBDFollowerConfig
    from kompass_core.models import (
        Robot,
        RobotType,
        RobotCtrlLimits,
        LinearCtrlLimits,
        AngularCtrlLimits,
        RobotGeometry,
        RobotState,
    )
    from kompass_core.datatypes import Bbox2D, LaserScanData

    # Define robot
    my_robot = Robot(
        robot_type=RobotType.ACKERMANN,
        geometry_type=RobotGeometry.Type.CYLINDER,
        geometry_params=np.array([0.3, 0.6])
    )

    # Define control limits
    ctrl_limits = RobotCtrlLimits(
        vx_limits=LinearCtrlLimits(max_vel=1.5, max_acc=3.0, max_decel=3.0),
        omega_limits=AngularCtrlLimits(
            max_vel=2.5, max_acc=2.5, max_decel=2.5, max_steer=np.pi / 2
        )
    )

    # Configure controller
    config = VisionRGBDFollowerConfig(
        max_linear_samples=15,
        max_angular_samples=15,
        control_horizon=10,
        enable_obstacle_avoidance=True,
    )
    controller = VisionRGBDFollower(
        robot=my_robot,
        ctrl_limits=ctrl_limits,
        config=config,
        camera_focal_length=[525.0, 525.0],
        camera_principal_point=[319.5, 239.5],
    )

    # Prepare sensor inputs
    bbox = Bbox2D(top_left_corner=np.array([200, 150]), size=np.array([50, 100]))
    bbox.set_img_size(np.array([640, 480]))

    aligned_depth_image = np.random.rand(480, 640).astype(np.int32)  # Fake depth
    robot_state = RobotState(x=0.0, y=0.0, yaw=0.0, speed=0.0)

    # Initialize target tracking
    controller.set_initial_tracking_2d_target(
        current_state=robot_state,
        target_box=bbox,
        aligned_depth_image=aligned_depth_image,
    )

    # Run control loop step
    success = controller.loop_step(
        current_state=robot_state,
        detections_2d=[bbox],
        depth_image=aligned_depth_image,
        local_map=np.random.rand(100, 100),  # Fake local map
        local_map_resolution=0.05
    )

    # Access control outputs
    vx_cmd = controller.linear_x_control
    omega_cmd = controller.angular_control
    ```
    """
    def __init__(
        self,
        robot: Robot,
        ctrl_limits: RobotCtrlLimits,
        config: Optional[VisionRGBDFollowerConfig] = None,
        config_file: Optional[str] = None,
        config_root_name: Optional[str] = None,
        control_time_step: Optional[float] = None,
        camera_focal_length: Optional[List[float]] = None,
        camera_principal_point: Optional[List[float]] = None,
        **_,
    ):
        """Init Vision RGBD (Depth) Follower Controller

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
        :param camera_focal_length: Depth camera focal length, defaults to None
        :type camera_focal_length: Optional[List[float]], optional
        :param camera_principal_point: Depth camera principal point, defaults to None
        :type camera_principal_point: Optional[List[float]], optional
        """
        self._config = config or VisionRGBDFollowerConfig()

        if config_file:
            self._config.from_file(
                file_path=config_file, nested_root_name=config_root_name
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
            self._planner.set_camera_intrinsics(
                camera_focal_length[0],
                camera_focal_length[1],
                camera_principal_point[0],
                camera_principal_point[1],
            )

        # Init the following result
        self._result = SamplingControlResult()
        self._end_of_ctrl_horizon: int = max(self._config.control_horizon, 1)
        logging.info("VisionDWA CONTROLLER IS READY")

    def set_camera_intrinsics(self, fx: float, fy: float, cx: float, cy: float) -> None:
        """Set depth camera intrinsics for the planner

        :param fx: Focal length in x direction
        :type fx: float
        :param fy: Focal length in y direction
        :type fy: float
        :param cx: Principal point x coordinate
        :type cx: float
        :param cy: Principal point y coordinate
        :type cy: float
        """
        self._planner.set_camera_intrinsics(fx, fy, cx, cy)

    def set_initial_tracking_2d_target(
        self,
        current_state: RobotState,
        target_box: Bbox2D,
        aligned_depth_image: np.ndarray,
    ) -> bool:
        """
        Set initial tracking state

        :param detected_boxes: Detected boxes
        :type detected_boxes: List[Bbox3D]
        """
        try:
            if current_state:
                self._planner.set_current_state(
                    current_state.x,
                    current_state.y,
                    current_state.yaw,
                    current_state.speed,
                )
            return self._planner.set_initial_tracking(
                aligned_depth_image,
                target_box,
                current_state.yaw,
            )

        except Exception as e:
            logging.error(f"Could not set initial tracking state: {e}")
            return False

    @property
    def dist_error(self) -> float:
        """Getter of the last distance error computed by the controller

        :return: Last distance error (m)
        :rtype: float
        """
        return self._planner.get_errors()[0]

    @property
    def orientation_error(self) -> float:
        """Getter of the last orientation error computed by the controller (radians)

        :return: Last orientation error (radians)
        :rtype: float
        """
        return self._planner.get_errors()[1]

    def set_initial_tracking_image(
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
            if self._config.use_local_coordinates:
                self._planner.set_current_state(
                    current_state.x,
                    current_state.y,
                    current_state.yaw,
                    current_state.speed,
                )
            if any(detected_boxes):
                return self._planner.set_initial_tracking(
                    pose_x_img,
                    pose_y_img,
                    aligned_depth_image,
                    detected_boxes,
                    current_state.yaw,
                )
            logging.error(
                "Could not set initial tracking state: No detections are provided"
            )
            return False

        except Exception as e:
            logging.error(f"Could not set initial tracking state: {e}")
            return False

    def loop_step(
        self,
        *,
        current_state: Optional[RobotState] = None,
        detections_2d: Optional[List[Bbox2D]] = None,
        depth_image: Optional[np.ndarray] = None,
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
        robot_cmd = None
        if self._config.use_local_coordinates and current_state is not None:
            self._planner.set_current_state(
                current_state.x, current_state.y, current_state.yaw, current_state.speed
            )
            robot_cmd = ControlCmd(
                vx=current_state.vx, vy=current_state.vy, omega=current_state.omega
            )

        if local_map_resolution:
            self._planner.set_resolution(local_map_resolution)

        if local_map is not None:
            sensor_data = PointCloudData.numpy_to_kompass_cpp(local_map)
        elif laser_scan:
            sensor_data = LaserScan(ranges=laser_scan.ranges, angles=laser_scan.angles)
        elif point_cloud is not None:
            sensor_data = point_cloud
        else:
            logging.error(
                "Cannot compute control without sensor data. Provide 'laser_scan' or 'point_cloud' input"
            )
            return False

        try:
            self._result = self._planner.get_tracking_ctrl(
                depth_image, detections_2d, robot_cmd or self._last_cmd, sensor_data
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
            return (
                f"VisionDWA Controller found trajectory with cost: {self._result.cost}"
            )
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

    @property
    def _last_cmd(self) -> ControlCmd:
        """
        Getter of the last command sent to the controller

        :return: Last command sent to the controller
        :rtype: ControlCmd
        """
        return ControlCmd(
            vx=self.linear_x_control[-1],
            vy=self.linear_y_control[-1],
            omega=self.angular_control[-1],
        )
