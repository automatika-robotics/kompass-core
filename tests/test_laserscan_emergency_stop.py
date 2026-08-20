import os
import json
from typing import Dict, Optional

import numpy as np
import pytest
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

    angles = np.arange(data["angle_min"], data["angle_max"], data["angle_increment"])
    ranges = np.array(data["ranges"])[: angles.size]

    return {"ranges": ranges, "angles": angles}


@pytest.fixture
def laser_scan_data() -> Dict[str, np.ndarray]:
    return laser_scan_data_fixed()



def _make_checker(
    use_gpu: bool,
    scan_angles: Optional[np.ndarray] = None,
    sensor_position: Optional[np.ndarray] = None,
    sensor_rotation: Optional[np.ndarray] = None,
):
    """Build the raw kompass_cpp critical zone checker.

    Constructed the same way Kompass' drive manager does it in production —
    the raw class is the only supported entry point. A laser scan checker
    takes the real scan angles; a pointcloud checker (``scan_angles=None``)
    gates every point directly in the body frame instead.

    :param use_gpu: Use the GPU implementation; the test is skipped when the
        build does not include it
    :type use_gpu: bool
    :param scan_angles: Angles of the laser scan, or None for a point cloud
    :type scan_angles: Optional[np.ndarray]
    :param sensor_position: Sensor position in the body frame [x, y, z] (m)
    :type sensor_position: Optional[np.ndarray]
    :param sensor_rotation: Sensor rotation in the body frame (quaternion)
    :type sensor_rotation: Optional[np.ndarray]
    """
    from kompass_cpp.types import SensorConfig, SensorInputType

    if use_gpu:
        try:
            from kompass_cpp.utils import CriticalZoneCheckerGPU as Checker
        except ImportError:
            pytest.skip("Build has no GPU implementation")
    else:
        from kompass_cpp.utils import CriticalZoneChecker as Checker

    robot_radius = 0.1
    robot = Robot(
        robot_type=RobotType.ACKERMANN,
        geometry_type=RobotGeometry.Type.CYLINDER,
        geometry_params=np.array([robot_radius, 0.4]),
    )

    sensor = SensorConfig(
        position=sensor_position
        if sensor_position is not None
        else np.array([0.0, 0.0, 0.0], dtype=np.float32),
        rotation=sensor_rotation
        if sensor_rotation is not None
        else np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32),
    )

    if scan_angles is not None:
        input_type = SensorInputType.LASERSCAN
        angles_kwargs = {"scan_angles": scan_angles}
    else:
        # Pointcloud checkers take no scan angles: points are gated per
        # point in the body frame
        input_type = SensorInputType.POINTCLOUD
        angles_kwargs = {}

    return Checker(
        robot_shape=robot.geometry_type,
        robot_dimensions=robot.geometry_params,
        sensor_configs=[sensor],
        critical_angle=90.0,
        critical_distance=0.5,
        slowdown_distance=1.0,
        min_height=-robot.height,
        max_height=robot.height,
        range_max=20.0,
        input_type=input_type,
        **angles_kwargs,
    )


@pytest.mark.parametrize("use_gpu", [False, True])
def test_emergency_stop(laser_scan_data: Dict[str, np.ndarray], use_gpu):
    """Test emergency stop

    :param laser_scan_data: Laser scan ranges and angles
    :type laser_scan_data: Dict[str, np.ndarray]
    """
    robot_radius = 0.1
    emergency_distance = 0.5

    large_range = 10.0
    emergency_value = robot_radius + emergency_distance / 2

    angles = laser_scan_data["angles"]
    checker = _make_checker(
        use_gpu,
        scan_angles=angles,
        sensor_position=np.array([0.0, 0.0, 0.173], dtype=np.float32),
        sensor_rotation=np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
    )
    ranges = np.array([large_range] * angles.size)

    assert checker.check(ranges=ranges, forward=True) == 1.0

    # Add an obstacle in the critical zone in front of the robot
    ranges[0] = emergency_value
    assert checker.check(ranges=ranges, forward=True) == 0.0
    assert checker.check(ranges=ranges, forward=False) == 1.0


# Matches the PointCloud2 layout the mapper tests use: 16 bytes per point,
# x/y/z as float32 at offsets 0/4/8, with 4 bytes of padding.
_PC_STRIDE = 16


def _make_pointcloud(points_xyz: np.ndarray) -> Dict[str, object]:
    """Pack an Nx3 float32 array into a PointCloud2-style byte buffer.

    :param points_xyz: Points in the sensor frame (m)
    :type points_xyz: np.ndarray
    :return: The layout kwargs ``check`` takes for a cloud
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
    emergency_distance = 0.5

    # Sensor at the robot origin, so cloud points are already body frame
    checker = _make_checker(use_gpu)

    # A ring of points well outside the slowdown zone
    theta = np.linspace(0.0, 2 * np.pi, 360, endpoint=False)
    far_ring = np.column_stack([
        10.0 * np.cos(theta),
        10.0 * np.sin(theta),
        np.zeros(theta.size),
    ])

    assert checker.check(**_make_pointcloud(far_ring), forward=True) == 1.0

    # Add an obstacle in the critical zone straight ahead (+x)
    emergency_value = robot_radius + emergency_distance / 2
    with_obstacle = np.vstack([far_ring, [emergency_value, 0.0, 0.0]])

    assert checker.check(**_make_pointcloud(with_obstacle), forward=True) == 0.0
    assert checker.check(**_make_pointcloud(with_obstacle), forward=False) == 1.0


@pytest.mark.parametrize("use_gpu", [False, True])
def test_emergency_stop_pointcloud_accepts_bytes(use_gpu):
    """Raw `bytes` input (the natural type of PointCloud2.data) must work
    and agree with the uint8-array result."""
    checker = _make_checker(use_gpu)

    theta = np.linspace(0.0, 2 * np.pi, 360, endpoint=False)
    ring = np.column_stack([
        10.0 * np.cos(theta),
        10.0 * np.sin(theta),
        np.zeros(theta.size),
    ])
    cloud = _make_pointcloud(ring)
    as_bytes = dict(cloud, data=np.asarray(cloud["data"]).tobytes())

    assert checker.check(**cloud, forward=True) == checker.check(
        **as_bytes, forward=True
    )


@pytest.mark.parametrize("use_gpu", [False, True])
def test_emergency_stop_pointcloud_ignores_nan_points(use_gpu):
    """NaN padding points must not affect the safety verdict (and must not
    corrupt memory — the conversion used to produce UB bin indices)."""
    checker = _make_checker(use_gpu)

    theta = np.linspace(0.0, 2 * np.pi, 360, endpoint=False)
    ring = np.column_stack([
        10.0 * np.cos(theta),
        10.0 * np.sin(theta),
        np.zeros(theta.size),
    ])
    ring_with_nan = np.vstack([ring, np.full((50, 3), np.nan)])

    clean = _make_pointcloud(ring)
    padded = _make_pointcloud(ring_with_nan)

    assert checker.check(**clean, forward=True) == checker.check(
        **padded, forward=True
    )


if __name__ == "__main__":
    test_emergency_stop(laser_scan_data_fixed(), True)


@pytest.mark.parametrize("use_gpu", [False, True])
def test_emergency_stop_pointcloud_rejects_negative_offsets(use_gpu):
    """Malformed metadata (negative field offsets) must raise, never read
    out of bounds and never come back as an 'all clear' verdict. The ROS
    layer maps this error to an emergency stop."""
    checker = _make_checker(use_gpu)

    ring = np.column_stack([
        np.ones(8), np.zeros(8), np.zeros(8)
    ]).astype(np.float32)
    cloud = _make_pointcloud(ring)
    cloud["y_offset"] = -4

    with pytest.raises(ValueError, match="non-negative"):
        checker.check(**cloud, forward=True)
