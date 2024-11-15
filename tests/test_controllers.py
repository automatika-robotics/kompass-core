import json
import logging
import os
from typing import Union

import matplotlib.pyplot as plt
import numpy as np
import pytest
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Path
from kompass_cpp.types import PathInterpolationType

from kompass_core.datatypes.laserscan import LaserScanData
from kompass_core.control import (
    DVZ,
    DWAConfig,
    TrajectoryCostsWeights,
    DWA,
    StanleyConfig,
    Stanley,
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

dir_name = os.path.dirname(os.path.abspath(__file__))
control_resources = os.path.join(dir_name, "resources/control")
EPSILON = 1e-3


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
    plt.savefig(f"{control_resources}/{figure_name}.png")


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

        path = Path()
        path.header.stamp.sec = path_dict["header"]["stamp"]["sec"]
        path.header.stamp.nanosec = path_dict["header"]["stamp"]["nanosec"]
        path.header.frame_id = path_dict["header"]["frame_id"]

        for pose_dict in path_dict["poses"]:
            pose_stamped = PoseStamped()
            pose_stamped.header.stamp.sec = pose_dict["header"]["stamp"]["sec"]
            pose_stamped.header.stamp.nanosec = pose_dict["header"]["stamp"]["nanosec"]
            pose_stamped.header.frame_id = pose_dict["header"]["frame_id"]

            pose_stamped.pose.position.x = pose_dict["pose"]["position"]["x"]
            pose_stamped.pose.position.y = pose_dict["pose"]["position"]["y"]
            pose_stamped.pose.position.z = pose_dict["pose"]["position"]["z"]

            pose_stamped.pose.orientation.x = pose_dict["pose"]["orientation"]["x"]
            pose_stamped.pose.orientation.y = pose_dict["pose"]["orientation"]["y"]
            pose_stamped.pose.orientation.z = pose_dict["pose"]["orientation"]["z"]
            pose_stamped.pose.orientation.w = pose_dict["pose"]["orientation"]["w"]

            path.poses.append(pose_stamped)

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
    interpolated_path = controller.interpolated_path(msg_header=None)
    interpolation_x = []
    interpolation_y = []
    for point in interpolated_path.poses:
        interpolation_x.append(point.pose.position.x)
        interpolation_y.append(point.pose.position.y)

    i = 0
    x_robot = []
    y_robot = []
    tracked_point_x = []
    tracked_point_y = []
    robot.state.x = -0.51731912
    robot.state.y = 0.0
    robot.state.yaw = np.pi / 2

    laser_scan = LaserScanData()
    # laser_scan.angles = np.array([4.0, 4.1])
    # laser_scan.ranges = np.array([0.4, 0.3])

    while not end_reached and i < 100:
        controller.loop_step(current_state=robot.state, laser_scan=laser_scan)
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
    linear_interpolation = follower.interpolated_path(msg_header=ref_path.header)

    follower.set_interpolation_type(PathInterpolationType.HERMITE_SPLINE)
    follower.set_path(ref_path)
    hermite_spline_interpolation = follower.interpolated_path(
        msg_header=ref_path.header
    )

    follower.set_interpolation_type(PathInterpolationType.CUBIC_SPLINE)
    follower.set_path(ref_path)
    cubic_spline_interpolation = follower.interpolated_path(msg_header=ref_path.header)

    if plot:
        # Extract x and y coordinates from the Path message
        x_ref = [pose.pose.position.x for pose in ref_path.poses]
        y_ref = [pose.pose.position.y for pose in ref_path.poses]

        x_inter_lin = [pose.pose.position.x for pose in linear_interpolation.poses]
        y_inter_lin = [pose.pose.position.y for pose in linear_interpolation.poses]

        x_inter_her = [
            pose.pose.position.x for pose in hermite_spline_interpolation.poses
        ]
        y_inter_her = [
            pose.pose.position.y for pose in hermite_spline_interpolation.poses
        ]

        x_inter_cub = [
            pose.pose.position.x for pose in cubic_spline_interpolation.poses
        ]
        y_inter_cub = [
            pose.pose.position.y for pose in cubic_spline_interpolation.poses
        ]
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
        plt.savefig(f"{control_resources}/interpolation_test.png")

    def path_length(path: Path) -> float:
        """Computes the length of a path

        :param path: Path
        :type path: Path
        :return: Path length
        :rtype: float
        """
        length = 0.0
        for idx in range(len(path.poses) - 1):
            d_x = path.poses[idx + 1].pose.position.x - path.poses[idx].pose.position.x
            d_y = path.poses[idx + 1].pose.position.y - path.poses[idx].pose.position.y
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
        max_linear_samples=20,
        max_angular_samples=20,
        octree_resolution=0.1,
        costs_weights=cost_weights,
        prediction_horizon=1.0,
        control_horizon=0.2,
        control_time_step=control_time_step,
        max_num_threads=10,
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


@pytest.fixture(autouse=True)
def run_before_and_after_tests():
    """Fixture to execute asserts before and after a test is run"""

    global global_path, my_robot, robot_ctr_limits, control_time_step

    global_path = json_to_ros_path(f"{control_resources}/global_path.json")

    if not global_path:
        raise ValueError("Global path file not found")

    my_robot = Robot(
        robot_type=RobotType.DIFFERENTIAL_DRIVE,
        geometry_type=RobotGeometry.Type.CYLINDER,
        geometry_params=np.array([0.1, 0.4]),
    )

    robot_ctr_limits = RobotCtrlLimits(
        vx_limits=LinearCtrlLimits(max_vel=1.0, max_acc=5.0, max_decel=10.0),
        omega_limits=AngularCtrlLimits(
            max_vel=10.0, max_acc=30.0, max_decel=20.0, max_steer=np.pi
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
            max_vel=2.0, max_acc=3.0, max_decel=3.0, max_steer=np.pi
        ),
    )

    control_time_step = 0.1

    print("RUNNING PATH INTERPOLATION TEST")
    test_path_interpolation(plot=True)

    ## TESTING STANLEY ##
    print("RUNNING STANLEY CONTROLLER TEST")
    test_stanley(
        plot=True, figure_name="stanley", figure_tag="Stanley Controller Test Results"
    )

    # TESTING DVZ ##
    print("RUNNING DVZ CONTROLLER TEST")
    test_dvz(plot=True, figure_name="dvz", figure_tag="DVZ Controller Test Results")

    ## TESTING DWA ##
    print("RUNNING DWA CONTROLLER TEST")
    test_dwa(plot=True, figure_name="dwa", figure_tag="DWA Controller Test Results")


if __name__ == "__main__":
    main()
