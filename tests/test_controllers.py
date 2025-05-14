import json
import logging
import os
import cv2
from typing import Union, List

import matplotlib.pyplot as plt
import numpy as np
import pytest
from attrs import define, field, Factory

import kompass_cpp
from kompass_cpp.types import PathInterpolationType, Path as PathCpp

from kompass_core.datatypes.laserscan import LaserScanData
from kompass_core.datatypes import TrackedPose2D, Bbox3D, Bbox2D
from kompass_core.control import (
    DVZ,
    DWAConfig,
    TrajectoryCostsWeights,
    DWA,
    StanleyConfig,
    Stanley,
    VisionDWA,
    VisionDWAConfig,
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

    laser_scan = LaserScanData()
    # laser_scan.angles = np.array([0.0, 0.1])
    # laser_scan.ranges = np.array([0.4, 0.3])

    while not end_reached and i < 100:
        ctrl_found = controller.loop_step(
            current_state=robot.state, laser_scan=laser_scan
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

    follower.set_interpolation_type(PathInterpolationType.LINEAR)
    follower.set_path(ref_path)
    linear_interpolation = follower.interpolated_path()

    follower.set_interpolation_type(PathInterpolationType.HERMITE_SPLINE)
    follower.set_path(ref_path)
    hermite_spline_interpolation = follower.interpolated_path()

    follower.set_interpolation_type(PathInterpolationType.CUBIC_SPLINE)
    follower.set_path(ref_path)
    cubic_spline_interpolation = follower.interpolated_path()

    if plot:
        # Extract x and y coordinates from the Path message
        x_ref = [pose.pose.position.x for pose in ref_path.poses]
        y_ref = [pose.pose.position.y for pose in ref_path.poses]

        x_inter_lin = linear_interpolation.x()
        y_inter_lin = linear_interpolation.y()

        x_inter_her = hermite_spline_interpolation.x()
        y_inter_her = hermite_spline_interpolation.y()

        x_inter_cub = cubic_spline_interpolation.x()
        y_inter_cub = cubic_spline_interpolation.y()

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

    length_diff = path_length(ref_path) - path_length(linear_interpolation)

    assert abs(length_diff) <= EPSILON


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
        obstacles_distance_weight=1.0,
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


def test_vision_dwa_with_depth_img():
    """Run DWA pytest and assert reaching end"""
    global global_path, my_robot, robot_ctr_limits, control_time_step

    from kompass_core import set_logging_level

    set_logging_level("DEBUG")

    my_robot.state.x = -0.5

    image_file_path = f"{control_resources}/bag_image_depth.tif"
    # clicked point on image to track target
    clicked_point = [610, 200]
    # Detected 2D box
    box = Bbox2D(top_left_corner=np.array([410, 0]), size=np.array([410, 390]))
    detections = [box]

    point_cloud = [np.array([10.3, 10.5, 0.2], dtype=np.float32)]

    control_horizon = 2
    prediction_horizon = 10

    focal_length = [911.71, 910.288]
    principal_point = [643.06, 366.72]

    cost_weights = TrajectoryCostsWeights(
        reference_path_distance_weight=1.0,
        goal_distance_weight=0.0,
        smoothness_weight=0.0,
        jerk_weight=0.0,
        obstacles_distance_weight=0.0,
    )
    config = VisionDWAConfig(
        control_time_step=control_time_step,
        control_horizon=control_horizon,
        prediction_horizon=prediction_horizon,
        speed_gain=1.0,
        rotation_gain=1.0,
        costs_weights=cost_weights,
        max_linear_samples=20,
        max_angular_samples=20,
        octree_resolution=0.1,
        max_num_threads=1,
        min_vel=0.05,
        min_depth=0.001,
        max_depth=5.0,
        depth_conversion_factor=0.001,
        camera_position_to_robot=np.array([0.32, 0.02089, 0.2999]),
        camera_rotation_to_robot=np.array([0.385 , -0.5846, 0.595, -0.395]),
    )

    controller = VisionDWA(
        robot=my_robot,
        ctrl_limits=robot_ctr_limits,
        config=config,
        camera_focal_length=focal_length,
        camera_principal_point=principal_point,
    )

    # Get image
    cv_img = cv2.imread(image_file_path, cv2.IMREAD_UNCHANGED)

    if cv_img is None:
        logging.error("Could not open or find the image")
        return

    # Create a NumPy array from the OpenCV Mat
    depth_image = np.array(cv_img, dtype=np.ushort, order="F")

    found_target = controller.set_initial_tracking_depth(
        my_robot.state, clicked_point[0], clicked_point[1], detections, depth_image
    )

    if not found_target:
        print("Point not found on image")
        return
    else:
        print("Point found on image ...")

    res = controller.loop_step(
        current_state=my_robot.state,
        detections_2d=detections,
        depth_image=depth_image,
        point_cloud=point_cloud,
    )
    if not res:
        print("No control found")

    assert res

    velocities = controller.control_till_horizon
    (vx, vy, omega) = velocities.vx[0], velocities.vy[0], velocities.omega[0]
    print(f"Found Control {vx}, {vy}, {omega}")


@pytest.mark.parametrize("use_tracker", [False, True])
@pytest.mark.parametrize(
    "point_cloud", [[np.array([10.3, 10.5, 0.2])], [np.array([0.3, 0.27, 0.1])]]
)
def test_vision_dwa(use_tracker: bool, point_cloud: List[np.ndarray]):
    """Run DWA pytest and assert reaching end"""
    global global_path, my_robot, robot_ctr_limits, control_time_step

    control_horizon = 2
    prediction_horizon = 10

    # Simulate tracked pose and vel
    tracked_vel_lin = 0.1
    tracked_vel_ang = 0.3
    tracked_pose = TrackedPose2D(0.0, 0.0, 0.0, tracked_vel_lin, 0.0, tracked_vel_ang)

    # To plot
    robot_x = []
    robot_y = []
    tracked_x = []
    tracked_y = []
    my_robot.state.x = -0.51731912
    my_robot.state.y = 0.0
    my_robot.state.yaw = 0.0

    cost_weights = TrajectoryCostsWeights(
        reference_path_distance_weight=1.0,
        goal_distance_weight=0.0,
        smoothness_weight=0.0,
        jerk_weight=0.0,
        obstacles_distance_weight=0.0,
    )
    config = VisionDWAConfig(
        control_time_step=control_time_step,
        control_horizon=control_horizon,
        prediction_horizon=prediction_horizon,
        speed_gain=1.0,
        rotation_gain=1.0,
        costs_weights=cost_weights,
        max_linear_samples=20,
        max_angular_samples=20,
        octree_resolution=0.1,
        max_num_threads=1,
        min_vel=0.05,
    )

    controller = VisionDWA(robot=my_robot, ctrl_limits=robot_ctr_limits, config=config)

    steps = 0

    # Detected Boxes
    ref_point_img = [150, 150]
    detected_boxes = []
    boxes_ori = 0.0
    for i in range(3):
        new_box = Bbox3D(
            center=np.array([0.0, 0.0, 0.0], dtype=np.float32),
            size=np.array([0.5, 0.5, 1.0], dtype=np.float32),
            center_img_frame=np.array([150, 150], dtype=np.int32),
            size_img_frame=np.array([25, 25], dtype=np.int32),
        )
        new_box_shift = np.array([0.7 * i, 0.7 * i, 0.0], dtype=np.float32)
        img_frame_shift = np.array([50 * i, 50 * i], dtype=np.int32)
        new_box.center = new_box_shift
        new_box.center_img_frame = img_frame_shift + ref_point_img
        detected_boxes.append(new_box)

    def moveBoxes(boxes_orientation, boxes) -> float:
        """Move boxes in the direction of the tracked pose"""
        for box in boxes:
            box.center[0] += (
                tracked_vel_lin * np.cos(boxes_orientation) * control_time_step
            )
            box.center[1] += (
                tracked_vel_lin * np.sin(boxes_orientation) * control_time_step
            )
        boxes_orientation += tracked_vel_ang * control_time_step
        return boxes_orientation

    if use_tracker:
        controller.set_initial_tracking_3d(150, 150, detected_boxes)

    while steps < 100:
        robot_x.append(my_robot.state.x)
        robot_y.append(my_robot.state.y)
        tracked_x.append(tracked_pose.x())
        tracked_y.append(tracked_pose.y())

        if use_tracker:
            res = controller.loop_step(
                current_state=my_robot.state,
                detections_3d=detected_boxes,
                point_cloud=point_cloud,
            )
        else:
            res = controller.loop_step(
                current_state=my_robot.state,
                tracked_pose=tracked_pose,
                point_cloud=point_cloud,
            )
        if not res:
            print("No control found")
            break
        # print(f"Found trajectory with cost {controller.result_cost}")
        velocities = controller.control_till_horizon
        (vx, vy, omega) = velocities.vx[0], velocities.vy[0], velocities.omega[0]
        print(f"Found Control {vx}, {vy}, {omega}")
        my_robot.set_control(
            velocity_x=vx,
            velocity_y=vy,
            omega=omega,
        )
        my_robot.get_state(dt=control_time_step)
        tracked_pose.update(control_time_step)
        if use_tracker:
            boxes_ori = moveBoxes(boxes_ori, detected_boxes)
        steps += 1

    figure_name = "vision_dwa"
    if use_tracker:
        figure_name += "_with_tracker"
    else:
        figure_name += "_no_tracker"

    figure_name += f"{point_cloud[0]}"
    plt.figure()
    plt.plot(robot_x, robot_y, marker="o", linestyle="-", color="b", label="Robot Path")
    plt.plot(
        tracked_x,
        tracked_y,
        label="Tracked Object",
        marker="*",
        linestyle="-",
        color="g",
    )
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title(figure_name)
    plt.grid(True)
    plt.legend()
    plt.savefig(f"logs/{figure_name}.png")

    assert steps == 100


def test_dwa_debug():
    global global_path, my_robot, robot_ctr_limits, control_time_step

    cost_weights = TrajectoryCostsWeights(
        reference_path_distance_weight=3.0,
        goal_distance_weight=1.0,
        smoothness_weight=0.0,
        jerk_weight=0.0,
        obstacles_distance_weight=1.0,
    )
    config = DWAConfig(
        max_linear_samples=21,
        max_angular_samples=21,
        octree_resolution=0.1,
        costs_weights=cost_weights,
        prediction_horizon=10,
        control_horizon=2,
        control_time_step=control_time_step,
        max_num_threads=1,
    )

    dwa = DWA(robot=my_robot, ctrl_limits=robot_ctr_limits, config=config)

    dwa.set_path(global_path)

    my_robot.state.x = -0.51731912
    my_robot.state.y = 0.0
    my_robot.state.yaw = 0.0

    laser_scan = LaserScanData()
    laser_scan.angles = np.array([0.0, 0.1])
    laser_scan.ranges = np.array([0.4, 0.3])
    dwa.planner.set_current_state(
        my_robot.state.x, my_robot.state.y, my_robot.state.yaw, my_robot.state.speed
    )

    dwa.planner.set_resolution(0.1)

    current_velocity = kompass_cpp.types.ControlCmd(
        vx=my_robot.state.vx, vy=my_robot.state.vy, omega=my_robot.state.omega
    )

    sensor_data = kompass_cpp.types.LaserScan(
        ranges=laser_scan.ranges, angles=laser_scan.angles
    )

    dwa.planner.debug_velocity_search(current_velocity, sensor_data, False)
    (paths_x, paths_y) = dwa.planner.get_debugging_samples()
    _, ax = plt.subplots()
    # Add the two laserscan obstacles
    obstacle_1 = plt.Circle((-0.117319, 0), 0.1, color="black", label="obstacle")
    obstacle_2 = plt.Circle((-0.218818, 0.02995), 0.1, color="black")
    for idx in range(paths_x.shape[0] - 1):
        path_i_x = paths_x[idx, :]
        path_i_y = paths_y[idx, :]
        plt.plot(path_i_x, path_i_y)
    ax.add_patch(obstacle_1)
    ax.add_patch(obstacle_2)
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title("Trajectory Samples")
    plt.grid(True)
    plt.legend()
    plt.savefig("logs/trajectory_samples_debug.png")


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
            max_vel=4.0, max_acc=3.0, max_decel=3.0, max_steer=np.pi
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
            max_vel=4.0, max_acc=3.0, max_decel=3.0, max_steer=np.pi
        ),
    )

    control_time_step = 0.1

    # print("RUNNING PATH INTERPOLATION TEST")
    # test_path_interpolation(plot=True)

    # ## TESTING STANLEY ##
    # print("RUNNING STANLEY CONTROLLER TEST")
    # test_stanley(
    #     plot=True, figure_name="stanley", figure_tag="Stanley Controller Test Results"
    # )

    # ## TESTING DVZ ##
    # print("RUNNING DVZ CONTROLLER TEST")
    # test_dvz(plot=True, figure_name="dvz", figure_tag="DVZ Controller Test Results")

    # ## TESTING DWA DEBUG MODE ##
    # print("RUNNING ONE DWA CONTROLLER DEBUG STEP TEST")
    # test_dwa_debug()

    # ## TESTING DWA ##
    # print("RUNNING DWA CONTROLLER TEST")
    # test_dwa(plot=True, figure_name="dwa", figure_tag="DWA Controller Test Results")

    # ## TESTING VISION DWA ##
    # print("RUNNING VISION DWA CONTROLLER TEST")
    # test_vision_dwa(use_tracker=True, point_cloud=[np.array([0.3, 0.27, 0.1])])
    test_vision_dwa_with_depth_img()


if __name__ == "__main__":
    main()
