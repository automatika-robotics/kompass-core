import json
import logging
import os
import cv2
from typing import Union, List

import numpy as np
import pytest
from attrs import define, field, Factory

from kompass_cpp.types import PathInterpolationType, Path as PathCpp

from kompass_core.datatypes import Bbox2D
from kompass_core.control import (
    DVZ,
    DWAConfig,
    TrajectoryCostsWeights,
    DWA,
    StanleyConfig,
    Stanley,
    PurePursuit,
    PurePursuitConfig,
    VisionRGBDFollower,
    VisionRGBDFollowerConfig,
    VisionRGBFollower,
    VisionRGBFollowerConfig,
)
from kompass_core.models import (
    AngularCtrlLimits,
    LinearCtrlLimits,
    Robot,
    RobotCtrlLimits,
    RobotGeometry,
    RobotType,
)

logger = logging.getLogger(__name__)
os.makedirs("logs", exist_ok=True)

dir_name = os.path.dirname(os.path.abspath(__file__))
control_resources = os.path.join(dir_name, "resources/control")
EPSILON = 1e-3


# Data Classes similar to ROS geometry_msgs.msg.PoseStamped and nav_msgs.msg.Path for testing
@define
class Vector4:
    """Class that replaces ROS Point and Quaternion classes for testing"""

    x: float = field(default=0.0)
    y: float = field(default=0.0)
    z: float = field(default=0.0)
    w: float = field(default=1.0)


@define
class Pose:
    """Class that replaces ROS geometry_msgs/Pose class for testing"""

    position: Vector4 = field(default=Factory(Vector4))
    orientation: Vector4 = field(default=Factory(Vector4))


@define
class PoseStamped:
    """Class that replaces ROS geometry_msgs/PoseStamped class for testing
    Discards the 'Header' part as it is not required for testing
    """

    pose: Pose = field(default=Factory(Pose))


@define
class Path:
    """Class that replaces ROS nav_msgs/Path class for testing
    Discards the 'Header' part as it is not required for testing
    """

    poses: List[PoseStamped] = field()


def plot_path(
    path: Path,
    x_robot,
    y_robot,
    tracked_point_x,
    tracked_point_y,
    interpolation_x,
    interpolation_y,
    figure_name: str,
    figure_tag: str,
):
    """Plot Test Results"""
    try:
        import matplotlib
        import matplotlib.pyplot as plt

        matplotlib.use("Agg")  # avoid Qt errors, no GUI
    except ImportError:
        logger.warning(
            "Matplotlib is required for visualization. Figures will not be generated. To generate test figures, install it using 'pip install matplotlib'."
        )
        return
    # Extract x and y coordinates from the Path message
    x_coords = [pose.pose.position.x for pose in path.poses]
    y_coords = [pose.pose.position.y for pose in path.poses]
    # Plot the path
    plt.figure()
    plt.plot(
        x_coords, y_coords, marker="o", linestyle="-", color="b", label="Reference Path"
    )
    plt.plot(
        interpolation_x,
        interpolation_y,
        label="Interpolated Path",
        linestyle="-",
        color="g",
    )
    plt.plot(x_robot, y_robot, color="r", label="Robot Path")
    plt.scatter(tracked_point_x, tracked_point_y, label="Tracked Point")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title(figure_tag)
    plt.grid(True)
    plt.legend()
    plt.savefig(f"logs/{figure_name}.png")


def json_to_ros_path(json_file: str) -> Union[Path, None]:
    """
    Reads a given json file and parse a ROS nav_msgs.msg.Path if exists

    :param json_file: Path to the json file
    :type json_file: str
    :return: ROS path message
    :rtype: Union[PathMsg, None]
    """
    try:
        with open(json_file, "r") as f:
            path_dict = json.load(f)

        poses: List[PoseStamped] = []

        for pose_dict in path_dict["poses"]:
            pose_stamped = PoseStamped()

            pose_stamped.pose.position.x = pose_dict["pose"]["position"]["x"]
            pose_stamped.pose.position.y = pose_dict["pose"]["position"]["y"]
            pose_stamped.pose.position.z = pose_dict["pose"]["position"]["z"]

            pose_stamped.pose.orientation.x = pose_dict["pose"]["orientation"]["x"]
            pose_stamped.pose.orientation.y = pose_dict["pose"]["orientation"]["y"]
            pose_stamped.pose.orientation.z = pose_dict["pose"]["orientation"]["z"]
            pose_stamped.pose.orientation.w = pose_dict["pose"]["orientation"]["w"]

            poses.append(pose_stamped)

        path = Path(poses=poses)

        return path
    # File not found or format is not compatible
    except Exception as e:
        print(f"{e}")
        return None


def run_control(
    controller,
    global_path: Path,
    robot: Robot,
    control_time_step: float,
    plot_results: bool,
    figure_name: str = "test",
    figure_tag: str = "Trajectory following & Control",
) -> bool:
    """Run the control loop until end of given path is reached by the robot

    :param controller: _description_
    :type controller: _type_
    :param global_path: _description_
    :type global_path: Path
    :param robot: _description_
    :type robot: Robot
    :param plot_results: _description_
    :type plot_results: bool
    :return: _description_
    :rtype: bool
    """
    end_reached = False

    controller.set_path(global_path)

    # Interpolated path for visualization
    interpolated_path = controller.interpolated_path()
    interpolation_x = interpolated_path.x()
    interpolation_y = interpolated_path.y()

    i = 0
    x_robot = []
    y_robot = []
    tracked_point_x = []
    tracked_point_y = []
    robot.state.x = -0.51731912
    robot.state.y = 0.0
    robot.state.yaw = np.pi / 2

    # A default all-max scan: the controller only needs a well-formed
    # ranges/angles pair here, the path is what is under test
    angles = np.arange(0.0, 2 * np.pi, 0.01 * np.pi)
    ranges = np.full(angles.size, 20.0)

    while not end_reached and i < 100:
        ctrl_found = controller.loop_step(
            current_state=robot.state, ranges=ranges, angles=angles
        )
        if not ctrl_found or not controller.path:
            end_reached = controller.reached_end()
            break

        tracked_state = controller.tracked_state
        tracked_point_x.append(tracked_state.x)
        tracked_point_y.append(tracked_state.y)

        for vx, vy, omega in zip(
            controller.linear_x_control,
            controller.linear_y_control,
            controller.angular_control,
        ):
            x_robot.append(robot.state.x)
            y_robot.append(robot.state.y)
            robot.set_control(
                velocity_x=vx,
                velocity_y=vy,
                omega=omega,
            )
            robot.get_state(dt=control_time_step)
            i += 1
            end_reached = controller.reached_end()

    print(f"End reached in: {i}")

    if plot_results:
        plot_path(
            global_path,
            x_robot,
            y_robot,
            tracked_point_x,
            tracked_point_y,
            interpolation_x,
            interpolation_y,
            figure_name=figure_name,
            figure_tag=figure_tag,
        )
    return end_reached


def test_path_interpolation(plot: bool = False):
    """Test path interpolation in followers

    :param plot: Generate a figure plot of the interpolation results, defaults to True
    :type plot: bool, optional
    :raises ValueError: If the reference path file is not found
    """
    global my_robot, robot_ctr_limits

    ref_path = json_to_ros_path(f"{control_resources}/global_path.json")

    if not global_path:
        raise ValueError("Global path file not found")

    # Create a follower to access the interpolation
    follower = Stanley(robot=my_robot, ctrl_limits=robot_ctr_limits)

    print("testing LINEAR path interpolation")
    follower.set_interpolation_type(PathInterpolationType.LINEAR)
    follower.set_path(ref_path)
    linear_interpolation = follower.interpolated_path()

    print("testing HERMITE_SPLINE path interpolation")
    follower.set_interpolation_type(PathInterpolationType.HERMITE_SPLINE)
    follower.set_path(ref_path)
    hermite_spline_interpolation = follower.interpolated_path()

    print("testing CUBIC_SPLINE path interpolation")
    follower.set_interpolation_type(PathInterpolationType.CUBIC_SPLINE)
    follower.set_path(ref_path)
    cubic_spline_interpolation = follower.interpolated_path()

    if plot:
        print("Plotting...")
        # Extract x and y coordinates from the Path message
        x_ref = [pose.pose.position.x for pose in ref_path.poses]
        y_ref = [pose.pose.position.y for pose in ref_path.poses]

        x_inter_lin = linear_interpolation.x()
        y_inter_lin = linear_interpolation.y()

        x_inter_her = hermite_spline_interpolation.x()
        y_inter_her = hermite_spline_interpolation.y()

        x_inter_cub = cubic_spline_interpolation.x()
        y_inter_cub = cubic_spline_interpolation.y()

        try:
            import matplotlib
            import matplotlib.pyplot as plt

            matplotlib.use("Agg")  # avoid Qt errors, no GUI
        except ImportError:
            logger.warning(
                "Matplotlib is required for visualization. Figures will not be generated. To generate test figures, install it using 'pip install matplotlib'."
            )
            return
        # Plot the path
        plt.figure()
        plt.plot(
            x_ref, y_ref, marker="o", linestyle="-", color="b", label="Reference Path"
        )
        plt.plot(x_inter_lin, y_inter_lin, color="g", label="Interpolated Path: Linear")
        plt.plot(
            x_inter_her,
            y_inter_her,
            color="r",
            label="Interpolated Path: Hermite Spline",
        )
        plt.plot(
            x_inter_cub,
            y_inter_cub,
            color="m",
            label="Interpolated Path: Cubic Spline",
        )

        plt.xlabel("X")
        plt.ylabel("Y")
        plt.grid(True)
        plt.legend()
        plt.savefig("logs/interpolation_test.png")

    def path_length(path: Union[Path, PathCpp]) -> float:
        """Computes the length of a path

        :param path: Path
        :type path: Path
        :return: Path length
        :rtype: float
        """
        length = 0.0
        if isinstance(path, Path):
            for idx in range(len(path.poses) - 1):
                d_x = (
                    path.poses[idx + 1].pose.position.x
                    - path.poses[idx].pose.position.x
                )
                d_y = (
                    path.poses[idx + 1].pose.position.y
                    - path.poses[idx].pose.position.y
                )
                length += np.sqrt(d_x**2 + d_y**2)
        elif isinstance(path, PathCpp):
            for idx in range(path.size() - 1):
                d_x = path.getIndex(idx + 1)[0] - path.getIndex(idx)[0]
                d_y = path.getIndex(idx + 1)[1] - path.getIndex(idx)[1]
                length += np.sqrt(d_x**2 + d_y**2)
        return length

    # length_diff = path_length(ref_path) - path_length(linear_interpolation)
    print(f"Original path length: {path_length(ref_path)}")
    print(f"linear_interpolation path length: {path_length(linear_interpolation)}")
    print(
        f"hermite_spline_interpolation path length: {path_length(hermite_spline_interpolation)}"
    )
    print(
        f"cubic_spline_interpolation path length: {path_length(cubic_spline_interpolation)}"
    )

    # assert abs(length_diff) <= EPSILON


def test_stanley(
    plot: bool = False, figure_name: str = "stanley", figure_tag: str = "stanley"
):
    """Run Stanley pytest and assert reaching end"""
    global global_path, my_robot, robot_ctr_limits, control_time_step

    config = StanleyConfig(cross_track_gain=1.5, heading_gain=2.0)

    stanley = Stanley(
        robot=my_robot,
        ctrl_limits=robot_ctr_limits,
        config=config,
        control_time_step=control_time_step,
    )
    reached_end = run_control(
        stanley,
        global_path,
        my_robot,
        control_time_step,
        plot_results=plot,
        figure_name=figure_name,
        figure_tag=figure_tag,
    )

    assert reached_end is True


def test_dvz(plot: bool = False, figure_name: str = "dvz", figure_tag: str = "dvz"):
    """Run DVZ pytest and assert reaching end"""
    global global_path, my_robot, robot_ctr_limits, control_time_step

    dvz = DVZ(
        robot=my_robot,
        ctrl_limits=robot_ctr_limits,
        control_time_step=control_time_step,
    )
    dvz.set_path(global_path)

    reached_end = run_control(
        dvz,
        global_path,
        my_robot,
        control_time_step,
        plot_results=plot,
        figure_name=figure_name,
        figure_tag=figure_tag,
    )

    assert reached_end is True


def test_dwa(plot: bool = False, figure_name: str = "dwa", figure_tag: str = "dwa"):
    """Run DWA pytest and assert reaching end"""
    global global_path, my_robot, robot_ctr_limits, control_time_step

    cost_weights = TrajectoryCostsWeights(
        reference_path_distance_weight=3.0,
        goal_distance_weight=1.0,
        smoothness_weight=0.0,
        jerk_weight=0.0,
        obstacles_distance_weight=0.0,
    )
    config = DWAConfig(
        max_linear_samples=4,
        max_angular_samples=4,
        octree_resolution=0.1,
        costs_weights=cost_weights,
        prediction_horizon=10,
        control_horizon=2,
        control_time_step=control_time_step,
        max_num_threads=1,
    )

    dwa = DWA(robot=my_robot, ctrl_limits=robot_ctr_limits, config=config)

    reached_end = run_control(
        dwa,
        global_path,
        my_robot,
        control_time_step,
        plot_results=plot,
        figure_name=figure_name,
        figure_tag=figure_tag,
    )

    assert reached_end is True


def test_dwa_accepts_cartesian_points():
    """DWA must take obstacles as an Nx3 cartesian array.

    Regression test: the point cloud used to be handed over as its raw
    PointCloud2 byte buffer, which matches no `compute_velocity_commands`
    overload, so every step failed inside the try/except and only surfaced
    as a 'Could not find velocity command' log line.
    """
    global global_path, my_robot, robot_ctr_limits, control_time_step

    dwa = DWA(
        robot=my_robot,
        ctrl_limits=robot_ctr_limits,
        config=DWAConfig(
            max_linear_samples=4,
            max_angular_samples=4,
            octree_resolution=0.1,
            control_time_step=control_time_step,
        ),
    )
    dwa.set_path(global_path=global_path)

    # Ring of obstacles well clear of the robot, so a control must be found
    theta = np.linspace(0.0, 2 * np.pi, 360, endpoint=False)
    points = np.column_stack([
        5.0 * np.cos(theta),
        5.0 * np.sin(theta),
        np.zeros(theta.size),
    ]).astype(np.float32)

    assert dwa.loop_step(current_state=my_robot.state, points=points) is True


def test_linear_ctrl_limits_min_vel():
    """The creep speed is the minimum speed of the linear control limits,
    with the sampler's former default when not given."""
    limits = LinearCtrlLimits(max_vel=1.0, max_acc=5.0, max_decel=10.0)
    assert limits.min_vel == 0.05
    assert (
        LinearCtrlLimits(max_vel=1.0, max_acc=5.0, max_decel=10.0, min_vel=0.1).min_vel
        == 0.1
    )
    limits.min_vel = 0.2
    assert limits.min_vel == 0.2


def test_dwa_allow_reverse():
    """Reversing samples are generated unless the DWA config turns them off,
    in which case no command is ever negative."""
    assert DWAConfig().allow_reverse is True

    # The goal lies behind the robot: with reversing allowed the planner
    # backs up, forward-only it must turn around instead
    def first_command(allow_reverse: bool):
        robot = Robot(
            robot_type=RobotType.DIFFERENTIAL_DRIVE,
            geometry_type=RobotGeometry.Type.CYLINDER,
            geometry_params=np.array([0.2, 0.4]),
        )
        limits = RobotCtrlLimits(
            vx_limits=LinearCtrlLimits(max_vel=0.8, max_acc=5.0, max_decel=10.0),
            omega_limits=AngularCtrlLimits(
                max_omega=1.5, max_acc=3.0, max_decel=3.0, max_ang=np.pi
            ),
        )
        dwa = DWA(
            robot=robot,
            ctrl_limits=limits,
            config=DWAConfig(
                control_time_step=0.1,
                prediction_horizon=10,
                control_horizon=2,
                max_linear_samples=9,
                max_angular_samples=10,
                max_num_threads=1,
                allow_reverse=allow_reverse,
            ),
        )
        poses = []
        for x in (0.0, -1.5, -3.0):
            pose = PoseStamped()
            pose.pose.position.x = x
            poses.append(pose)
        dwa.set_path(Path(poses=poses))
        robot.state.x, robot.state.y, robot.state.yaw = 0.0, 0.0, 0.0
        angles = np.arange(0.0, 2 * np.pi, 0.1 * np.pi)
        ranges = np.full(angles.size, 20.0)
        assert dwa.loop_step(current_state=robot.state, ranges=ranges, angles=angles)
        return dwa.linear_x_control[0]

    assert first_command(True) < 0.0
    assert first_command(False) >= 0.0


def test_angular_ctrl_limits_min_omega():
    """The angular deadband of the controllers is the minimum angular velocity
    of the angular control limits, 0.05 rad/s when not given."""
    limits = AngularCtrlLimits(max_omega=1.0, max_acc=3.0, max_decel=3.0, max_ang=np.pi)
    assert limits.min_omega == 0.05
    assert (
        AngularCtrlLimits(
            max_omega=1.0, max_acc=3.0, max_decel=3.0, max_ang=np.pi, min_omega=0.2
        ).min_omega
        == 0.2
    )
    limits.min_omega = 0.3
    assert limits.min_omega == 0.3


def test_dwa_creeps_onto_a_close_goal():
    """A goal closer than the smallest grid speed covers within the horizon:
    0.5 s steps, a 5 s rollout and a speed grid whose smallest speed (0.2 m/s)
    travels 0.9 m, twice the distance to the goal. Every straight grid sample
    overshoots, so only the creep speed can end near the
    goal. The controller must reach it with forward motion only: no reverse,
    no turning, no overshoot.
    """
    robot = Robot(
        robot_type=RobotType.DIFFERENTIAL_DRIVE,
        geometry_type=RobotGeometry.Type.CYLINDER,
        geometry_params=np.array([0.2, 0.4]),
    )
    # The creep speed is the minimum speed of the x-axis limits
    limits = RobotCtrlLimits(
        vx_limits=LinearCtrlLimits(
            max_vel=0.8, max_acc=5.0, max_decel=10.0, min_vel=0.05
        ),
        omega_limits=AngularCtrlLimits(
            max_omega=1.5, max_acc=3.0, max_decel=3.0, max_ang=np.pi
        ),
    )
    config = DWAConfig(
        control_time_step=0.5,
        prediction_horizon=10,
        control_horizon=5,
        max_linear_samples=9,
        max_angular_samples=10,
        octree_resolution=0.1,
        max_num_threads=1,
        costs_weights=TrajectoryCostsWeights(
            reference_path_distance_weight=1.0,
            goal_distance_weight=1.0,
            smoothness_weight=0.0,
            jerk_weight=0.0,
            obstacles_distance_weight=0.0,
        ),
    )
    dwa = DWA(robot=robot, ctrl_limits=limits, config=config)

    # A straight 3 m path along x
    poses = []
    for x in (0.0, 1.5, 3.0):
        pose = PoseStamped()
        pose.pose.position.x = x
        poses.append(pose)
    dwa.set_path(Path(poses=poses))

    # 0.45 m short of the goal, facing it, at rest
    robot.state.x = 2.55
    robot.state.y = 0.0
    robot.state.yaw = 0.0
    start_distance = 0.45

    angles = np.arange(0.0, 2 * np.pi, 0.1 * np.pi)
    ranges = np.full(angles.size, 20.0)

    steps = 0
    while steps < 60:
        steps += 1
        if not dwa.loop_step(
            current_state=robot.state, ranges=ranges, angles=angles
        ):
            # No command once the given state is at the goal
            break
        vx = dwa.linear_x_control[0]
        omega = dwa.angular_control[0]
        assert vx >= 0.0, f"reverse command {vx} at step {steps}"
        assert abs(omega) < 0.3, f"turning command {omega} at step {steps}"
        robot.set_control(velocity_x=vx, omega=omega)
        robot.get_state(dt=config.control_time_step)
        distance = np.hypot(3.0 - robot.state.x, robot.state.y)
        assert distance <= start_distance + 0.05, f"moved away at step {steps}"

    assert dwa.reached_end(), f"goal not reached in {steps} steps"


def test_pure_pursuit(
    plot: bool = False,
    figure_name: str = "pure_pursuit",
    figure_tag: str = "pure_pursuit",
):
    """Run Pure Pursuit pytest and assert reaching end"""
    global global_path, my_robot, robot_ctr_limits, control_time_step

    config = PurePursuitConfig(
        wheel_base=my_robot.wheelbase,
        lookahead_gain_forward=1.0,
    )

    controller = PurePursuit(
        robot=my_robot,
        ctrl_limits=robot_ctr_limits,
        config=config,
        control_time_step=control_time_step,
    )

    reached_end = run_control(
        controller,
        global_path,
        my_robot,
        control_time_step,
        plot_results=plot,
        figure_name=figure_name,
        figure_tag=figure_tag,
    )

    assert reached_end is True


def test_pure_pursuit_consumes_sensor_data():
    """PurePursuit must dispatch on the ``ranges``/``angles``/``points`` kwargs.

    Regression test: the sensor branches used to read ``.ranges``/``.angles``
    off a LaserScanData object and ``.data`` off a PointCloudData object. Once
    those classes were dropped, a caller on the new convention matched no
    branch and fell through to the sensor-less ``execute(dt)`` overload, so
    collision avoidance was silently disabled while path tracking kept
    reporting success.
    """
    global global_path, my_robot, robot_ctr_limits, control_time_step

    def _make_controller() -> PurePursuit:
        controller = PurePursuit(
            robot=my_robot,
            ctrl_limits=robot_ctr_limits,
            config=PurePursuitConfig(
                wheel_base=my_robot.wheelbase, lookahead_gain_forward=1.0
            ),
            control_time_step=control_time_step,
        )
        controller.set_path(global_path)
        return controller

    def _step(**sensor_kwargs) -> tuple:
        """Take one step from a fixed pose and return the command issued."""
        controller = _make_controller()
        my_robot.state.x = 0.0
        my_robot.state.y = 0.0
        my_robot.state.yaw = np.pi / 2
        found = controller.loop_step(current_state=my_robot.state, **sensor_kwargs)
        return found, controller.linear_x_control[0], controller.angular_control[0]

    # Obstacles pressed right up against the robot from every side. There is
    # no way out, so a controller that sees them must command a full stop.
    blocking_distance = 0.15
    angles = np.arange(-np.pi, np.pi, 0.01 * np.pi)
    blocked_ranges = np.full(angles.size, blocking_distance)

    theta = np.linspace(0.0, 2 * np.pi, 720, endpoint=False)
    blocking_points = np.column_stack([
        blocking_distance * np.cos(theta),
        blocking_distance * np.sin(theta),
        np.zeros(theta.size),
    ]).astype(np.float32)

    # Baseline: with no sensor data the controller just tracks the path and
    # drives off. This is exactly what the broken dispatch silently did with
    # every one of the sensor kwargs below.
    _, nominal_vx, nominal_omega = _step()
    assert nominal_vx != 0.0 or nominal_omega != 0.0, (
        "sensor-less tracking is expected to move; the obstacle assertions "
        "below cannot discriminate otherwise"
    )

    for sensor_kwargs in (
        {"ranges": blocked_ranges, "angles": angles},
        {"points": blocking_points},
        {"local_map": blocking_points},
    ):
        found, vx, omega = _step(**sensor_kwargs)
        # NOTE: loop_step still reports success here -- a full stop is a valid
        # command -- so the return value alone proves nothing about dispatch.
        assert found is True
        assert (vx, omega) == (0.0, 0.0), (
            f"obstacle avoidance did not engage for {list(sensor_kwargs)}: "
            f"got vx={vx}, omega={omega}"
        )

    # A mismatched scan can only be rejected if the laser scan branch is
    # actually entered -- the unmigrated code returned True here instead.
    assert (
        _make_controller().loop_step(
            current_state=my_robot.state,
            ranges=blocked_ranges[:-1],
            angles=angles,
        )
        is False
    )


def test_vision_rgb_follower():
    """Run VisionRGBFollower pytest and assert reaching end"""
    global global_path, my_robot, robot_ctr_limits, control_time_step

    from kompass_core import set_logging_level

    set_logging_level("DEBUG")

    my_robot.state.x = -0.5

    box = Bbox2D(top_left_corner=np.array([410, 0]), size=np.array([410, 390]))
    box.set_img_size(np.array([640, 480], dtype=np.int32))
    detections = [box]

    config = VisionRGBFollowerConfig(
        control_time_step=control_time_step, speed_gain=1.0, rotation_gain=1.0
    )

    controller = VisionRGBFollower(
        robot=my_robot,
        ctrl_limits=robot_ctr_limits,
        config=config,
    )

    found_target = controller.set_initial_tracking_2d_target(box)

    if not found_target:
        print("Point not found on image")
        return
    else:
        print("Point found on image ...")

    res = controller.loop_step(
        detections_2d=detections,
    )
    if not res:
        print("No control found")

    assert res

    (vx, vy, omega) = (
        controller.linear_x_control,
        controller.linear_y_control,
        controller.angular_control,
    )
    print(f"Found Control {vx}, {vy}, {omega}")


@pytest.fixture(autouse=True)
def run_before_and_after_tests():
    """Fixture to execute asserts before and after a test is run"""

    global global_path, my_robot, robot_ctr_limits, control_time_step

    global_path = json_to_ros_path(f"{control_resources}/global_path.json")

    if not global_path:
        raise ValueError("Global path file not found")

    my_robot = Robot(
        robot_type=RobotType.ACKERMANN,
        geometry_type=RobotGeometry.Type.CYLINDER,
        geometry_params=np.array([0.1, 0.4]),
    )

    robot_ctr_limits = RobotCtrlLimits(
        vx_limits=LinearCtrlLimits(max_vel=1.0, max_acc=5.0, max_decel=10.0),
        omega_limits=AngularCtrlLimits(
            max_omega=4.0, max_acc=3.0, max_decel=3.0, max_ang=np.pi
        ),
    )

    control_time_step = 0.1

    yield


def main():
    global global_path, my_robot, robot_ctr_limits, control_time_step

    global_path = json_to_ros_path(f"{control_resources}/global_path.json")

    if not global_path:
        raise ValueError("Global path file not found")

    my_robot = Robot(
        robot_type=RobotType.ACKERMANN,
        geometry_type=RobotGeometry.Type.CYLINDER,
        geometry_params=np.array([0.1, 0.4]),
    )

    robot_ctr_limits = RobotCtrlLimits(
        vx_limits=LinearCtrlLimits(max_vel=1.0, max_acc=5.0, max_decel=10.0),
        omega_limits=AngularCtrlLimits(
            max_omega=4.0, max_acc=3.0, max_decel=3.0, max_ang=np.pi
        ),
    )

    control_time_step = 0.1

    test_path_interpolation(plot=True)

    # print("RUNNING PATH INTERPOLATION TEST")
    # test_path_interpolation(plot=True)

    # ## TESTING STANLEY ##
    # print("RUNNING STANLEY CONTROLLER TEST")
    # test_stanley(
    #     plot=True, figure_name="stanley", figure_tag="Stanley Controller Test Results"
    # )

    ## TESTING PURE PURSUIT ##
    print("RUNNING PURE PURSUIT CONTROLLER TEST")
    test_pure_pursuit(
        plot=True,
        figure_name="pure_pursuit",
        figure_tag="Pure Pursuit Controller Test Results",
    )

    # ## TESTING DVZ ##
    # print("RUNNING DVZ CONTROLLER TEST")
    # test_dvz(plot=True, figure_name="dvz", figure_tag="DVZ Controller Test Results")

    # ## TESTING DWA DEBUG MODE ##
    # print("RUNNING ONE DWA CONTROLLER DEBUG STEP TEST")
    # test_dwa_debug()

    ## TESTING DWA ##
    print("RUNNING DWA CONTROLLER TEST")
    test_dwa(plot=True, figure_name="dwa", figure_tag="DWA Controller Test Results")

    # ## TESTING VISION RGB Follower ##
    # print("RUNNING VISION RGB FOLLOWER TEST")
    # test_vision_rgb_follower()


if __name__ == "__main__":
    main()


def test_laserscan_type_exposes_numpy_views():
    """LaserScan accepts float32 (zero-copy) and float64 (converted) input
    and exposes ranges/angles as float32 numpy views (was: Python lists)."""
    from kompass_cpp.types import LaserScan as LaserScanCpp

    ranges = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    angles = np.array([0.0, 0.1, 0.2], dtype=np.float32)

    scan = LaserScanCpp(ranges=ranges, angles=angles)
    assert isinstance(scan.ranges, np.ndarray)
    assert scan.ranges.dtype == np.float32
    assert np.allclose(scan.ranges, ranges)
    assert np.allclose(scan.angles, angles)

    scan64 = LaserScanCpp(
        ranges=ranges.astype(np.float64), angles=angles.astype(np.float64)
    )
    assert np.allclose(np.asarray(scan64.ranges), ranges)


def test_dwa_dtype_equivalence():
    """float32 and float64 sensor inputs must yield identical commands for
    every input kind (the wrappers coerce both onto the same float32 path)."""
    global global_path, my_robot, robot_ctr_limits, control_time_step

    def make_dwa():
        dwa = DWA(
            robot=my_robot,
            ctrl_limits=robot_ctr_limits,
            config=DWAConfig(
                max_linear_samples=4,
                max_angular_samples=4,
                octree_resolution=0.1,
                control_time_step=control_time_step,
            ),
        )
        dwa.set_path(global_path=global_path)
        return dwa

    my_robot.state.x = 0.0
    my_robot.state.y = 0.0
    my_robot.state.yaw = np.pi / 2

    angles64 = np.arange(0.0, 2 * np.pi, 0.01 * np.pi)
    ranges64 = np.full(angles64.size, 20.0)
    theta = np.linspace(0.0, 2 * np.pi, 360, endpoint=False)
    points64 = np.column_stack([
        5.0 * np.cos(theta),
        5.0 * np.sin(theta),
        np.zeros(theta.size),
    ])

    cases = (
        (
            {"ranges": ranges64, "angles": angles64},
            {
                "ranges": ranges64.astype(np.float32),
                "angles": angles64.astype(np.float32),
            },
        ),
        ({"points": points64}, {"points": points64.astype(np.float32)}),
        ({"local_map": points64}, {"local_map": points64.astype(np.float32)}),
    )
    for kwargs64, kwargs32 in cases:
        dwa64 = make_dwa()
        assert dwa64.loop_step(current_state=my_robot.state, **kwargs64)
        cmd64 = (dwa64.linear_x_control[0], dwa64.angular_control[0])

        dwa32 = make_dwa()
        assert dwa32.loop_step(current_state=my_robot.state, **kwargs32)
        cmd32 = (dwa32.linear_x_control[0], dwa32.angular_control[0])

        assert cmd64 == pytest.approx(cmd32, abs=1e-4), (
            f"dtype divergence for {list(kwargs64)}: {cmd64} vs {cmd32}"
        )


def test_dwa_custom_cost():
    """Custom Python cost functions on the DWA planner.

    Covers three contracts in one setup:

    1. A cost registered after construction is invoked at all — regression
       for the GPU evaluator's temp-costs buffer, which was only allocated
       at construction time when custom costs already existed, so a cost
       added from Python dereferenced a null device pointer on first solve.
    2. The values flow into the search: a constant cost must shift the
       winning trajectory's reported cost by exactly weight * constant
       (compared against an identically configured planner without it).
    3. The solve releases the GIL: another Python thread must make progress
       during a single long native solve call — with the GIL held no other
       thread can run during a native call, so the tick delta would be 0.
    """
    global global_path, my_robot, robot_ctr_limits, control_time_step
    import threading
    import time
    from kompass_cpp.types import Velocity2D

    def _make_dwa() -> DWA:
        cost_weights = TrajectoryCostsWeights(
            reference_path_distance_weight=3.0,
            goal_distance_weight=1.0,
            smoothness_weight=0.0,
            jerk_weight=0.0,
            obstacles_distance_weight=0.0,
        )
        config = DWAConfig(
            max_linear_samples=4,
            max_angular_samples=4,
            octree_resolution=0.1,
            costs_weights=cost_weights,
            prediction_horizon=10,
            control_horizon=2,
            control_time_step=control_time_step,
            max_num_threads=1,
        )
        dwa = DWA(robot=my_robot, ctrl_limits=robot_ctr_limits, config=config)
        dwa.set_path(global_path)
        return dwa

    plain = _make_dwa()
    with_cost = _make_dwa()

    calls = {"n": 0}
    CONSTANT = 7.5
    WEIGHT = 2.0

    def constant_cost(trajectory, reference_path) -> float:
        calls["n"] += 1
        return CONSTANT

    with_cost._planner.add_custom_cost(WEIGHT, constant_cost)

    my_robot.state.x = -0.51731912
    my_robot.state.y = 0.0
    my_robot.state.yaw = np.pi / 2

    n = 360
    angles = np.linspace(0.0, 2 * np.pi, n, endpoint=False, dtype=np.float32)
    ranges = np.full(n, 10.0, dtype=np.float32)

    # 1. + 2. — same state, same scan, only the custom cost differs
    assert plain.loop_step(
        current_state=my_robot.state, ranges=ranges, angles=angles
    ), "baseline planner found no command"
    assert with_cost.loop_step(
        current_state=my_robot.state, ranges=ranges, angles=angles
    ), "planner with custom cost found no command"
    assert calls["n"] > 0, "custom Python cost was never invoked"
    assert with_cost._result.cost == pytest.approx(
        plain._result.cost + WEIGHT * CONSTANT, rel=1e-4
    ), "custom cost value did not flow into the trajectory costs"

    # 3. — one long native solve (100k-point cloud), ticker thread must run
    # during it. Counter deltas are read immediately around the single call,
    # so with a held GIL the delta is 0.
    rng = np.random.default_rng(11)
    cloud = np.ascontiguousarray(
        rng.uniform(-5.0, 5.0, (100_000, 3)).astype(np.float32)
    )
    vel = Velocity2D(vx=0.0, vy=0.0, omega=0.0)

    ticks = {"n": 0, "stop": False}

    def _ticker():
        while not ticks["stop"]:
            ticks["n"] += 1
            time.sleep(0)

    ticker = threading.Thread(target=_ticker)
    ticker.start()
    try:
        before = ticks["n"]
        plain._planner.compute_velocity_commands(vel, cloud)
        delta = ticks["n"] - before
    finally:
        ticks["stop"] = True
        ticker.join()

    assert delta > 0, (
        "no other Python thread ran during the solve -> the GIL was held"
    )
