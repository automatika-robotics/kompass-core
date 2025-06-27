import os
import json
import numpy as np
import logging
import pytest
from kompass_core.datatypes import LaserScanData
from kompass_core.utils.geometry import get_laserscan_transformed_polar_coordinates
from kompass_core.utils.emergency_stop import EmergencyChecker
from kompass_core.models import Robot, RobotType, RobotGeometry


def laser_scan_data_fixed() -> LaserScanData:
    """
    fixed laser scan data

    :return:    laser scan data created
    :rtype:     LaserScanData
    """
    dir_name = os.path.dirname(os.path.abspath(__file__))
    json_file_path = os.path.join(dir_name, "resources/mapping/laserscan_data.json")
    data = json.load(open(json_file_path))

    return LaserScanData(
        angle_min=data["angle_min"],
        angle_max=data["angle_max"],
        angle_increment=data["angle_increment"],
        time_increment=data["time_increment"],
        scan_time=data["scan_time"],
        ranges=np.array(data["ranges"]),
        range_min=data["range_min"],
        range_max=data["range_max"],
    )


@pytest.fixture
def laser_scan_data() -> LaserScanData:
    return laser_scan_data_fixed()


def test_laserscan_polar_tf(laser_scan_data: LaserScanData, plot: bool = False):
    """Test transforming laserscan (rotation test)

    :param laser_scan_data: Laser scan data
    :type laser_scan_data: LaserScanData
    :param plot: To generate and save a plot of the results, defaults to False
    :type plot: bool, optional
    """
    # 90 Deg rotation around z
    translation = [0.0, 0.0, 0.173]
    rotation = [0.0, 0.0, 0.7071068, 0.7071068]

    transformed_scan = get_laserscan_transformed_polar_coordinates(
        angle_min=laser_scan_data.angle_min,
        angle_max=laser_scan_data.angle_max,
        angle_increment=laser_scan_data.angle_increment,
        laser_scan_ranges=laser_scan_data.ranges,
        max_scan_range=laser_scan_data.range_max,
        translation=translation,
        rotation=rotation,
    )
    if plot:
        try:
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots(subplot_kw={"projection": "polar"})
            ax.plot(
                laser_scan_data.angles,
                laser_scan_data.ranges,
                label="Original LaserScan",
            )
            ax.plot(
                transformed_scan.angles,
                transformed_scan.ranges,
                label="Transformed LaserScan",
            )
            fig.legend()
            dir_name = os.path.dirname(os.path.abspath(__file__))
            plt.savefig(os.path.join(dir_name, "laserscan_tf_test.png"))
        except ImportError:
            logging.warning(
                "Matplotlib is required for visualization. Figures will not be generated. To generate test figures, install it using 'pip install matplotlib'."
            )

    old_range = laser_scan_data.ranges[
        laser_scan_data.angles == laser_scan_data.angle_min
    ]
    new_range = transformed_scan.ranges[
        transformed_scan.angles == laser_scan_data.angle_min + np.pi / 2
    ]
    try:
        assert old_range == new_range
    except AssertionError:
        logging.error(
            f"Original range at 0 {old_range} is not equal to transformed range at 90 {new_range} after 90 degree transformation"
        )
        raise


def test_laserscan_partial_data(laser_scan_data: LaserScanData, plot: bool = False):
    """Test getting partial data from laserscan (data between two angles)

    :param laser_scan_data: Laser scan data
    :type laser_scan_data: LaserScanData
    :param plot: To generate and save a plot of the results, defaults to False
    :type plot: bool, optional
    """
    right_angle = -np.pi / 2
    left_angle = np.pi / 2
    partial_ranges = laser_scan_data.get_ranges(
        right_angle=right_angle, left_angle=left_angle
    )

    partial_angles = laser_scan_data.get_angles(
        right_angle=right_angle, left_angle=left_angle
    )

    assert len(partial_angles) <= laser_scan_data.angles.size
    assert len(partial_ranges) <= laser_scan_data.ranges.size

    if plot:
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            logging.warning(
                "Matplotlib is required for visualization. Figures will not be generated. To generate test figures, install it using 'pip install matplotlib'."
            )
            return
        fig, ax = plt.subplots(subplot_kw={"projection": "polar"})
        ax.plot(
            laser_scan_data.angles, laser_scan_data.ranges, label="Original LaserScan"
        )
        ax.plot(partial_angles, partial_ranges, label="Partial LaserScan")
        fig.legend()
        dir_name = os.path.dirname(os.path.abspath(__file__))
        plt.savefig(os.path.join(dir_name, "laserscan_partial_test.png"))


@pytest.mark.parametrize("use_gpu", [False, True])
def test_emergency_stop(laser_scan_data: LaserScanData, use_gpu):
    """Test emergency stop

    :param laser_scan_data: Laser scan data
    :type laser_scan_data: LaserScanData
    """
    robot_radius = 0.1
    robot = Robot(
        robot_type=RobotType.ACKERMANN,
        geometry_type=RobotGeometry.Type.CYLINDER,
        geometry_params=np.array([robot_radius, 0.4]),
    )
    emergency_distance = 0.5
    slowdown_distance = 1.0
    emergency_angle = 90.0

    large_range = 10.0
    emergency_value = robot_radius + emergency_distance / 2

    emergency_stop = EmergencyChecker(
        robot=robot,
        emergency_distance=emergency_distance,
        slowdown_distance=slowdown_distance,
        emergency_angle=emergency_angle,
        sensor_position_robot=np.array([0.0, 0.0, 0.173], dtype=np.float32),
        sensor_rotation_robot=np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
        use_gpu=use_gpu,
    )
    angles_size = np.arange(
        laser_scan_data.angle_min,
        laser_scan_data.angle_max,
        laser_scan_data.angle_increment,
    ).shape[0]
    laser_scan_data.ranges = np.array([large_range] * angles_size)

    assert emergency_stop.run(scan=laser_scan_data, forward=True) == 1.0

    # Add an obstacle in the critical zone in front of the robot
    laser_scan_data.ranges[0] = emergency_value
    assert emergency_stop.run(scan=laser_scan_data, forward=True) == 0.0
    assert emergency_stop.run(scan=laser_scan_data, forward=False) == 1.0


if __name__ == "__main__":
    laser_scan = laser_scan_data_fixed()
    test_laserscan_polar_tf(laser_scan, plot=True)
    test_laserscan_partial_data(laser_scan, plot=True)
    test_emergency_stop(laser_scan, True)
