import os
import json
import numpy as np
import pytest
from typing import Dict
from kompass_core.utils.emergency_stop import EmergencyChecker
from kompass_core.models import Robot, RobotType, RobotGeometry


def laser_scan_data_fixed() -> Dict[str, np.ndarray]:
    """
    fixed laser scan data

    :return:    Scan as the ranges/angles pair the checker takes
    :rtype:     Dict[str, np.ndarray]
    """
    dir_name = os.path.dirname(os.path.abspath(__file__))
    json_file_path = os.path.join(dir_name, "resources/mapping/laserscan_data.json")
    data = json.load(open(json_file_path))

    angles = np.arange(
        data["angle_min"], data["angle_max"], data["angle_increment"]
    )
    ranges = np.array(data["ranges"])[: angles.size]

    return {"ranges": ranges, "angles": angles}


@pytest.fixture
def laser_scan_data() -> Dict[str, np.ndarray]:
    return laser_scan_data_fixed()


@pytest.mark.parametrize("use_gpu", [False, True])
def test_emergency_stop(laser_scan_data: Dict[str, np.ndarray], use_gpu):
    """Test emergency stop

    :param laser_scan_data: Laser scan ranges and angles
    :type laser_scan_data: Dict[str, np.ndarray]
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
    angles = laser_scan_data["angles"]
    ranges = np.array([large_range] * angles.size)

    assert (
        emergency_stop.run_on_laserscan(ranges=ranges, angles=angles, forward=True)
        == 1.0
    )

    # Add an obstacle in the critical zone in front of the robot
    ranges[0] = emergency_value
    assert (
        emergency_stop.run_on_laserscan(ranges=ranges, angles=angles, forward=True)
        == 0.0
    )
    assert (
        emergency_stop.run_on_laserscan(ranges=ranges, angles=angles, forward=False)
        == 1.0
    )


if __name__ == "__main__":
    test_emergency_stop(laser_scan_data_fixed(), True)
