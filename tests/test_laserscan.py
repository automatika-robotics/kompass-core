import os
import json
import numpy as np
import logging
import pytest
import matplotlib.pyplot as plt
from kompass_core.datatypes import LaserScanData
from kompass_core.utils.geometry import get_laserscan_transformed_polar_coordinates


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
    # rotation = [0.0, 0.0, 0.0, 1.0]

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
        fig, ax = plt.subplots(subplot_kw={"projection": "polar"})
        ax.plot(
            laser_scan_data.angles, laser_scan_data.ranges, label="Original LaserScan"
        )
        ax.plot(
            transformed_scan.angles,
            transformed_scan.ranges,
            label="Transformed LaserScan",
        )
        fig.legend()
        dir_name = os.path.dirname(os.path.abspath(__file__))
        plt.savefig(os.path.join(dir_name, "laserscan_tf_test.png"))

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

    if plot:
        fig, ax = plt.subplots(subplot_kw={"projection": "polar"})
        ax.plot(
            laser_scan_data.angles, laser_scan_data.ranges, label="Original LaserScan"
        )
        ax.plot(partial_angles, partial_ranges, label="Partial LaserScan")
        fig.legend()
        dir_name = os.path.dirname(os.path.abspath(__file__))
        plt.savefig(os.path.join(dir_name, "laserscan_partial_test.png"))

    assert len(partial_angles) <= laser_scan_data.angles.size
    assert len(partial_ranges) <= laser_scan_data.ranges.size


if __name__ == "__main__":
    laser_scan = laser_scan_data_fixed()
    test_laserscan_polar_tf(laser_scan, plot=True)
    test_laserscan_partial_data(laser_scan, plot=True)
