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


# Matches the PointCloud2 layout the mapper tests use: 16 bytes per point,
# x/y/z as float32 at offsets 0/4/8, with 4 bytes of padding.
_PC_STRIDE = 16


def _make_pointcloud(points_xyz: np.ndarray) -> Dict[str, object]:
    """Pack an Nx3 float32 array into a PointCloud2-style byte buffer.

    :param points_xyz: Points in the sensor frame (m)
    :type points_xyz: np.ndarray
    :return: The layout kwargs ``run_on_pointcloud`` takes
    :rtype: Dict[str, object]
    """
    n = points_xyz.shape[0]
    buffer = np.zeros((n, 4), dtype=np.float32)
    buffer[:, :3] = points_xyz.astype(np.float32)
    return dict(
        data=np.frombuffer(buffer.tobytes(), dtype=np.uint8),
        point_step=_PC_STRIDE,
        row_step=n * _PC_STRIDE,
        height=1,
        width=n,
        x_offset=0,
        y_offset=4,
        z_offset=8,
    )


@pytest.mark.parametrize("use_gpu", [False, True])
def test_emergency_stop_on_pointcloud(use_gpu):
    """Emergency checking on a raw point cloud buffer.

    Mirrors ``test_emergency_stop``: a cloud clear of the robot lets it run,
    a point inside the forward critical zone stops it, and reversing away
    from that point is allowed again.
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

    emergency_stop = EmergencyChecker(
        robot=robot,
        emergency_distance=emergency_distance,
        slowdown_distance=slowdown_distance,
        emergency_angle=emergency_angle,
        # Sensor at the robot origin, so cloud points are already body frame
        sensor_position_robot=np.array([0.0, 0.0, 0.0], dtype=np.float32),
        sensor_rotation_robot=np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32),
        use_gpu=use_gpu,
    )

    # A ring of points well outside the slowdown zone
    theta = np.linspace(0.0, 2 * np.pi, 360, endpoint=False)
    far_ring = np.column_stack([
        10.0 * np.cos(theta),
        10.0 * np.sin(theta),
        np.zeros(theta.size),
    ])

    assert (
        emergency_stop.run_on_pointcloud(**_make_pointcloud(far_ring), forward=True)
        == 1.0
    )

    # Add an obstacle in the critical zone straight ahead (+x)
    emergency_value = robot_radius + emergency_distance / 2
    with_obstacle = np.vstack([far_ring, [emergency_value, 0.0, 0.0]])

    assert (
        emergency_stop.run_on_pointcloud(
            **_make_pointcloud(with_obstacle), forward=True
        )
        == 0.0
    )
    assert (
        emergency_stop.run_on_pointcloud(
            **_make_pointcloud(with_obstacle), forward=False
        )
        == 1.0
    )


if __name__ == "__main__":
    test_emergency_stop(laser_scan_data_fixed(), True)
