from attrs import define, field
from ..utils.common import base_validators
from kompass_cpp.control import (
    RGBDFollower as RGBDFollowerCpp,
    RGBDFollowerParameters,
    SamplingControlResult,
)
from kompass_cpp.types import (
    Bbox2D,
    Velocity2D,
    TrajectoryVelocities2D,
    TrajectoryPath,
)
from typing import Optional, List, Union
import numpy as np
import logging
from ._base_ import ControllerTemplate, FollowerConfig
from ..models import Robot, RobotState, RobotCtrlLimits, RobotGeometry, RobotType


@define
class VisionRGBDFollowerConfig(FollowerConfig):
    """
    VisionRGBDFollower Configuration Parameters

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
      - Time interval between control updates (sec). Must be between `1e-4` and `1e6`.

    * - control_horizon
      - `int`
      - `2`
      - Number of steps in the control horizon. Must be between `1` and `1000`.

    * - prediction_horizon
      - `int`
      - `10`
      - Number of steps in the prediction horizon. Must be between `1` and `1000`.

    * - buffer_size
      - `int`
      - `1`
      - Size of the trajectory buffer. Must be between `1` and `10`.

    * - target_distance
      - `Optional[float]`
      - `None`
      - Edge-to-edge distance to maintain from the target (m). `None` lets the controller decide.

    * - target_wait_timeout
      - `float`
      - `30.0`
      - Time to wait for the target to reappear before giving up (sec). Used when search is disabled. Must be between `0.0` and `1e3`.

    * - target_search_timeout
      - `float`
      - `30.0`
      - Time limit for the search process when the target is lost (sec). Must be between `0.0` and `1e3`.

    * - target_search_pause
      - `float`
      - `2.0`
      - Pause between successive search actions (sec). Must be between `0.0` and `1e3`.

    * - target_search_radius
      - `float`
      - `0.5`
      - Radius around the last known target position used during search (m). Must be between `1e-4` and `1e4`.

    * - enable_search
      - `bool`
      - `True`
      - Enables target search behavior when the target is lost.

    * - distance_tolerance
      - `float`
      - `0.05`
      - Acceptable deviation in target distance (m). Must be between `1e-6` and `1e3`.

    * - angle_tolerance
      - `float`
      - `0.1`
      - Acceptable deviation in target bearing (rad). Must be between `1e-6` and `1e3`.

    * - target_orientation
      - `float`
      - `0.0`
      - Bearing-to-target to maintain in the robot frame (rad). Must be between `-π` and `π`.

    * - rotation_gain
      - `float`
      - `0.5`
      - Gain applied in the rotational control law. Must be between `1e-2` and `10.0`.

    * - speed_gain
      - `float`
      - `1.0`
      - Gain applied in the speed control law. Must be between `1e-2` and `10.0`.

    * - _use_local_coordinates
      - `bool`
      - `True`
      - Track the target in the robot's local frame (no world pose required). Set to `False` to track in the world frame, in which case `current_state` becomes mandatory in `loop_step`. Underscore-prefixed because it is plumbed through to the C++ planner rather than being a typical user knob.

    * - error_pose
      - `float`
      - `0.05`
      - Error in pose estimation (m). Must be between `1e-9` and `1e9`.

    * - error_vel
      - `float`
      - `0.05`
      - Error in velocity estimation (m/s). Must be between `1e-9` and `1e9`.

    * - error_acc
      - `float`
      - `0.05`
      - Error in acceleration estimation (m/s²). Must be between `1e-9` and `1e9`.

    * - depth_conversion_factor
      - `float`
      - `1e-3`
      - Factor to convert raw depth image values to meters. Must be between `1e-9` and `1e9`.

    * - min_depth
      - `float`
      - `0.0`
      - Minimum depth value considered valid (m). Must be between `0.0` and `1e3`.

    * - max_depth
      - `float`
      - `1e3`
      - Maximum depth value considered valid (m). Must be between `1e-3` and `1e9`.

    * - camera_position_to_robot
      - `np.ndarray`
      - `[0.0, 0.0, 0.0]`
      - Translation vector from the robot base to the camera frame (m).

    * - camera_rotation_to_robot
      - `np.ndarray`
      - `[0.0, 0.0, 0.0, 1.0]`
      - Quaternion `(x, y, z, w)` from the robot base to the camera frame.

    ```
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

    _use_local_coordinates: bool = field(
        default=True, alias="_use_local_coordinates"
    )  # Track in local frame (default) or world frame (when False)

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

    def to_kompass_cpp(self) -> RGBDFollowerParameters:
        """
        Convert to kompass_cpp lib config format

        :return: C++ parameter object populated from this config
        :rtype: kompass_cpp.control.RGBDFollowerParameters
        """
        vision_dwa_params = RGBDFollowerParameters()

        # Special handling for None values that are represented by -1 in C++
        params_dict = self.asdict()

        if params_dict["target_distance"] is None:
            params_dict["target_distance"] = -1.0

        vision_dwa_params.from_dict(params_dict)
        return vision_dwa_params


class VisionRGBDFollower(ControllerTemplate):
    """
    Vision-based target follower driven by RGB-D (color + depth) input.

    The controller takes 2D bounding-box detections plus an aligned depth
    image, projects the selected target into 3D using the camera intrinsics
    and the body-to-camera transform, tracks it across frames, and emits a
    pure-pursuit-style velocity command toward it. When the target is lost,
    the controller falls back to a configurable wait/search behavior before
    giving up.

    Tracking can run in either the robot's local frame (default) or the world
    frame; toggle via `_use_local_coordinates` on the config. World-frame
    tracking requires `current_state` on every `loop_step` call.

    ```python
    import numpy as np
    from kompass_core.control import VisionRGBDFollower, VisionRGBDFollowerConfig
    from kompass_core.models import (
        AngularCtrlLimits,
        LinearCtrlLimits,
        Robot,
        RobotCtrlLimits,
        RobotGeometry,
        RobotState,
        RobotType,
    )
    from kompass_core.datatypes import Bbox2D

    # Define robot
    my_robot = Robot(
        robot_type=RobotType.DIFFERENTIAL_DRIVE,
        geometry_type=RobotGeometry.Type.CYLINDER,
        geometry_params=np.array([0.3, 0.6]),
    )

    # Define control limits
    ctrl_limits = RobotCtrlLimits(
        vx_limits=LinearCtrlLimits(max_vel=1.5, max_acc=3.0, max_decel=3.0),
        omega_limits=AngularCtrlLimits(
            max_vel=2.5, max_acc=2.5, max_decel=2.5, max_steer=np.pi / 2
        ),
    )

    # Configure controller
    config = VisionRGBDFollowerConfig(
        control_horizon=2,
        prediction_horizon=10,
        target_distance=0.5,
    )
    controller = VisionRGBDFollower(
        robot=my_robot,
        ctrl_limits=ctrl_limits,
        config=config,
        camera_focal_length=[525.0, 525.0],
        camera_principal_point=[319.5, 239.5],
    )

    # Prepare detections + depth image
    bbox = Bbox2D(top_left_corner=np.array([200, 150]), size=np.array([50, 100]))
    bbox.set_img_size(np.array([640, 480]))
    aligned_depth_image = np.zeros((480, 640), dtype=np.uint16)  # mm depth
    robot_state = RobotState(x=0.0, y=0.0, yaw=0.0, speed=0.0)

    # Initialize target tracking
    controller.set_initial_tracking_2d_target(
        current_state=robot_state,
        target_box=bbox,
        aligned_depth_image=aligned_depth_image,
    )

    # Run one control step
    success = controller.loop_step(
        current_state=robot_state,
        detections_2d=[bbox],
        depth_image=aligned_depth_image,
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

        self._planner = RGBDFollowerCpp(
            control_type=RobotType.to_kompass_cpp_lib(robot.robot_type),
            control_limits=ctrl_limits.to_kompass_cpp_lib(),
            robot_shape_type=RobotGeometry.Type.to_kompass_cpp_lib(robot.geometry_type),
            robot_dimensions=robot.geometry_params,
            vision_sensor_position_wrt_body=self._config.camera_position_to_robot,
            vision_sensor_rotation_wrt_body=self._config.camera_rotation_to_robot,
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
        logging.info("RGBDFollower CONTROLLER IS READY")

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
        """Set initial tracking from a single 2D target box and depth image.

        In global mode the robot state is used by the depth detector to project
        the 3D box into the world frame. In local mode the state is ignored
        and the box is computed relative to the robot.

        :param current_state: Current robot state
        :type current_state: RobotState
        :param target_box: 2D bounding box of the target
        :type target_box: Bbox2D
        :param aligned_depth_image: Aligned depth image
        :type aligned_depth_image: np.ndarray
        :return: Whether tracking was successfully initialized
        :rtype: bool
        """
        try:
            if not self._config._use_local_coordinates:
                # Global mode: detector needs the robot pose for world-frame projection
                self._planner.set_current_state(
                    current_state.x,
                    current_state.y,
                    current_state.yaw,
                    current_state.speed,
                )
            return self._planner.set_initial_tracking(
                aligned_depth_image,
                target_box,
                current_state.yaw if current_state else 0.0,
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
        """Set initial tracking by selecting a pixel inside one of the 2D
        detection boxes.

        In global mode the robot state is used by the depth detector to project
        the 3D box into the world frame. In local mode the state is ignored
        and the box is computed relative to the robot.

        :param current_state: Current robot state
        :type current_state: RobotState
        :param pose_x_img: X pixel coordinate of the target in the image
        :type pose_x_img: int
        :param pose_y_img: Y pixel coordinate of the target in the image
        :type pose_y_img: int
        :param detected_boxes: List of 2D detection bounding boxes
        :type detected_boxes: List[Bbox2D]
        :param aligned_depth_image: Aligned depth image
        :type aligned_depth_image: np.ndarray
        :return: Whether tracking was successfully initialized
        :rtype: bool
        """
        try:
            if not self._config._use_local_coordinates:
                # Global mode: detector needs the robot pose for world-frame projection
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
                    current_state.yaw if current_state else 0.0,
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
        **_,
    ) -> bool:
        """Run one iteration of the vision follower.

        In global mode (``_use_local_coordinates=False``) ``current_state`` is
        **mandatory** — it is used by the depth detector (world-frame
        projection) and the control law (distance and bearing computation). In
        local mode ``current_state`` is optional; if provided, only its
        velocity component is used as the seed for the next command.

        Extra keyword arguments are accepted but ignored, for compatibility
        with callers that drive multiple controllers from a single dispatch.

        :param current_state: Current robot state. Required in global mode.
        :type current_state: Optional[RobotState]
        :param detections_2d: 2D detection bounding boxes from the vision pipeline
        :type detections_2d: Optional[List[Bbox2D]]
        :param depth_image: Aligned depth image
        :type depth_image: Optional[np.ndarray]
        :return: Whether the planner found a valid solution
        :rtype: bool
        """
        robot_cmd = None
        if not self._config._use_local_coordinates:
            # Global mode: state is mandatory — detector and control law need it
            if current_state is None:
                logging.error(
                    "Global mode (use_local_coordinates=False) requires "
                    "current_state in loop_step"
                )
                return False
            self._planner.set_current_state(
                current_state.x, current_state.y, current_state.yaw, current_state.speed
            )
            robot_cmd = Velocity2D(
                vx=current_state.vx, vy=current_state.vy, omega=current_state.omega
            )
        elif current_state is not None:
            # Local mode: ignore position, but extract velocity for DWA fallback
            robot_cmd = Velocity2D(
                vx=current_state.vx, vy=current_state.vy, omega=current_state.omega
            )

        try:
            self._result = self._planner.get_tracking_ctrl(
                depth_image, detections_2d, robot_cmd or self._last_cmd
            )

        except Exception as e:
            logging.error(f"Could not find velocity command: {e}")
            return False

        return self._result.is_found

    def has_result(self) -> bool:
        """
        Check whether the last `loop_step` produced a valid trajectory.

        :return: True if a valid trajectory was found on the last iteration
        :rtype: bool
        """
        return self._result.is_found

    def logging_info(self) -> str:
        """logging_info."""
        if self._result.is_found:
            return f"RGBDFollower Controller found trajectory with cost: {self._result.cost}"
        else:
            return "RGBDFollower Controller Failed to find a valid trajectory"

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
    def _last_cmd(self) -> Velocity2D:
        """
        Getter of the last command sent to the controller

        :return: Last command sent to the controller
        :rtype: Velocity2D
        """
        return Velocity2D(
            vx=self.linear_x_control[-1],
            vy=self.linear_y_control[-1],
            omega=self.angular_control[-1],
        )
