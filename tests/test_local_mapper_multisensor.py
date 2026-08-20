"""Multi-sensor point cloud fusion tests: raw kompass_cpp bindings and the
kompass_core.mapping.LocalMapper wrapper."""

import types

import numpy as np
import pytest

from kompass_cpp.mapping import OCCUPANCY_TYPE
from kompass_cpp.mapping import LocalMapper as LocalMapperCpp
from kompass_cpp.types import SensorConfig

try:
    from kompass_cpp.mapping import LocalMapperGPU

    _HAS_GPU = True
except ImportError:
    _HAS_GPU = False

from kompass_core.datatypes.pose import PoseData
from kompass_core.datatypes.scan_model import ScanModelConfig
from kompass_core.mapping import LocalMapper, MapConfig

OCCUPIED = OCCUPANCY_TYPE.OCCUPIED.value
UNEXPLORED = OCCUPANCY_TYPE.UNEXPLORED.value

_PC_STRIDE = 16  # 16B points (x, y, z, pad), matching the other test files


def _pack_points(points) -> np.ndarray:
    """Packs (x, y, z) tuples into a PointCloud2-style uint8 buffer."""
    buffer = np.zeros((len(points), 4), dtype=np.float32)
    for row, point in enumerate(points):
        buffer[row, :3] = point
    return buffer.reshape(-1).view(np.uint8)


def _cloud_dict(points, extra_keys: bool = True) -> dict:
    data = _pack_points(points)
    cloud = {
        "data": data,
        "point_step": _PC_STRIDE,
        "row_step": _PC_STRIDE * len(points),
        "height": 1,
        "width": len(points),
        "x_offset": 0,
        "y_offset": 4,
        "z_offset": 8,
    }
    if extra_keys:
        # A whole container gets splatted in practice; extras must be ignored
        cloud["frame_id"] = "livox_frame"
        cloud["timestamp"] = 0.0
    return cloud


def _quat_yaw(yaw: float) -> np.ndarray:
    return np.array(
        [0.0, 0.0, np.sin(yaw / 2.0), np.cos(yaw / 2.0)], dtype=np.float32
    )


def _front_back_sensors():
    return [
        SensorConfig(position=np.array([0.3, 0.0, 0.2], dtype=np.float32)),
        SensorConfig(
            position=np.array([-0.3, 0.0, 0.2], dtype=np.float32),
            rotation=_quat_yaw(np.pi),
        ),
    ]


def _make_mapper(sensor_configs, use_gpu: bool):
    """Raw-binding mapper: 20x20 grid at 0.1 m/cell, 360 bins,
    body-frame band [0, 0.5]."""
    kwargs = dict(
        grid_height=20,
        grid_width=20,
        resolution=0.1,
        sensor_configs=sensor_configs,
        is_pointcloud=True,
        scan_size=360,
        max_height=0.5,
        min_height=0.0,
        range_max=5.0,
        max_points_per_line=43,
    )
    if use_gpu:
        if not _HAS_GPU:
            pytest.skip("kompass_cpp built without GPU support")
        return LocalMapperGPU(**kwargs)
    return LocalMapperCpp(**kwargs, max_num_threads=1)


_BACKENDS = [False, True]


# ---------------------------------------------------------------------------
# Raw bindings
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("use_gpu", _BACKENDS)
def test_multisensor_fusion_front_back(use_gpu):
    """One obstacle ahead of each sensor in its own frame: the fused grid
    holds OCCUPIED cells both ahead of and behind the robot center."""
    mapper = _make_mapper(_front_back_sensors(), use_gpu)

    front = _cloud_dict([(0.4, 0.0, -0.1)])  # body (0.7, 0, 0.1)
    back = _cloud_dict([(0.4, 0.0, -0.1)])  # body (-0.7, 0, 0.1)

    grid = np.asarray(mapper.scan_to_grid(clouds=[front, back]))
    assert (grid[14:18, :] == OCCUPIED).any(), "no OCCUPIED cell ahead"
    assert (grid[1:5, :] == OCCUPIED).any(), "no OCCUPIED cell behind"


@pytest.mark.parametrize("use_gpu", _BACKENDS)
def test_multisensor_none_means_no_data(use_gpu):
    """None entries are skipped; an all-None batch yields an untouched grid."""
    mapper = _make_mapper(_front_back_sensors(), use_gpu)

    front = _cloud_dict([(0.4, 0.0, -0.1)])
    grid = np.asarray(mapper.scan_to_grid(clouds=[front, None]))
    assert (grid[14:18, :] == OCCUPIED).any()
    assert not (grid[0:9, :] == OCCUPIED).any(), (
        "front-only batch put OCCUPIED cells behind the robot"
    )

    grid = np.asarray(mapper.scan_to_grid(clouds=[None, None]))
    assert (grid == UNEXPLORED).all()


@pytest.mark.parametrize("use_gpu", _BACKENDS)
def test_multisensor_elements_must_be_dicts(use_gpu):
    """The contract is tight: an element is a dict (asdict()-style) or None.
    Attribute-carrying objects are rejected with the offending index."""
    mapper = _make_mapper(_front_back_sensors(), use_gpu)

    front = _cloud_dict([(0.4, 0.0, -0.1)])
    back_obj = types.SimpleNamespace(**_cloud_dict([(0.4, 0.0, -0.1)]))
    with pytest.raises(ValueError, match=r"clouds\[1\].*dict"):
        mapper.scan_to_grid(clouds=[front, back_obj])


@pytest.mark.parametrize("use_gpu", _BACKENDS)
def test_multisensor_bytes_data(use_gpu):
    """`bytes` buffers (PointCloud2.data) cross the batched entry too."""
    mapper = _make_mapper(_front_back_sensors(), use_gpu)

    front = _cloud_dict([(0.4, 0.0, -0.1)])
    as_array = np.asarray(mapper.scan_to_grid(clouds=[front, None])).copy()

    front_bytes = dict(front)
    front_bytes["data"] = front["data"].tobytes()
    as_bytes = np.asarray(mapper.scan_to_grid(clouds=[front_bytes, None]))
    assert (as_array == as_bytes).all()


@pytest.mark.parametrize("use_gpu", _BACKENDS)
def test_multisensor_error_paths(use_gpu):
    mapper = _make_mapper(_front_back_sensors(), use_gpu)
    front = _cloud_dict([(0.4, 0.0, -0.1)])

    # Cloud count must match the configured sensor count
    with pytest.raises(ValueError):
        mapper.scan_to_grid(clouds=[front])

    # Negative offsets are named per cloud
    bad = _cloud_dict([(0.4, 0.0, -0.1)])
    bad["y_offset"] = -4
    with pytest.raises(ValueError, match=r"clouds\[1\].*non-negative"):
        mapper.scan_to_grid(clouds=[front, bad])

    # Missing layout fields are named per cloud
    incomplete = {"data": front["data"]}
    with pytest.raises(ValueError, match=r"clouds\[0\].*point_step"):
        mapper.scan_to_grid(clouds=[incomplete, front])


# ---------------------------------------------------------------------------
# kompass_core wrapper
# ---------------------------------------------------------------------------


def _wrapper_configs():
    map_config = MapConfig(width=2.0, height=2.0, resolution=0.1)
    scan_model = ScanModelConfig(
        angle_step=float(2.0 * np.pi / 360.0),
        max_height=0.5,
        min_height=0.0,
        range_max=5.0,
    )
    return map_config, scan_model


def test_wrapper_multisensor_fusion():
    map_config, scan_model = _wrapper_configs()
    mapper = LocalMapper(
        config=map_config,
        scan_model_config=scan_model,
        sensors=_front_back_sensors(),
    )
    assert mapper.num_sensors == 2

    mapper.update_from_pointclouds(
        PoseData(),
        clouds=[_cloud_dict([(0.4, 0.0, -0.1)]), _cloud_dict([(0.4, 0.0, -0.1)])],
    )
    occupancy = mapper.occupancy
    assert (occupancy[14:18, :] == OCCUPIED).any(), "no OCCUPIED cell ahead"
    assert (occupancy[1:5, :] == OCCUPIED).any(), "no OCCUPIED cell behind"


def test_wrapper_multisensor_accepts_pose_data_mounts():
    map_config, scan_model = _wrapper_configs()
    mapper = LocalMapper(
        config=map_config,
        scan_model_config=scan_model,
        sensors=[PoseData()],
    )
    mapper.update_from_pointclouds(
        PoseData(), clouds=[_cloud_dict([(0.5, 0.0, 0.1)])]
    )
    assert (mapper.occupancy == OCCUPIED).any()


def test_wrapper_multisensor_guards():
    map_config, scan_model = _wrapper_configs()

    # Mutually exclusive mount arguments
    with pytest.raises(ValueError, match="not both"):
        LocalMapper(
            config=map_config,
            scan_model_config=scan_model,
            pose_laser_scanner_in_robot=PoseData(),
            sensors=[SensorConfig()],
        )

    # At least one sensor
    with pytest.raises(ValueError, match="at least one"):
        LocalMapper(config=map_config, scan_model_config=scan_model, sensors=[])

    # The multi-sensor pipeline is non-Bayesian by contract, rejected at
    # construction for ANY sensor count (config errors must not surface at
    # runtime)
    bayes_config = MapConfig(
        width=2.0, height=2.0, resolution=0.1, bayesian_update=True
    )
    for sensor_set in (_front_back_sensors(), [SensorConfig()]):
        with pytest.raises(ValueError, match="multi-sensor pipeline"):
            LocalMapper(
                config=bayes_config,
                scan_model_config=scan_model,
                sensors=sensor_set,
            )

    # The plural entry requires multi-sensor construction
    legacy_mapper = LocalMapper(config=map_config, scan_model_config=scan_model)
    with pytest.raises(RuntimeError, match="sensors="):
        legacy_mapper.update_from_pointclouds(
            PoseData(), clouds=[_cloud_dict([(0.4, 0.0, -0.1)])]
        )

    mapper = LocalMapper(
        config=map_config,
        scan_model_config=scan_model,
        sensors=_front_back_sensors(),
    )

    # Single-cloud and laserscan entries reject the multi-sensor mode
    cloud = _cloud_dict([(0.4, 0.0, -0.1)])
    with pytest.raises(RuntimeError, match="update_from_pointclouds"):
        mapper.update_from_pointcloud(PoseData(), **cloud)
    with pytest.raises(NotImplementedError):
        mapper.update_from_laserscan(
            PoseData(),
            ranges=np.ones(4, dtype=np.float32),
            angles=np.zeros(4, dtype=np.float32),
        )

    # Cloud count must match the configured sensor count
    with pytest.raises(ValueError, match="Expected 2 clouds"):
        mapper.update_from_pointclouds(PoseData(), clouds=[cloud])


def test_cpu_mapper_honors_float64_field_type():
    """A FLOAT64 cloud through the CPU mapper must produce the exact grid of
    its FLOAT32 twin when SensorConfig.cloud_field_type says FLOAT64."""
    from kompass_cpp.types import PointFieldType

    points = [(0.4, 0.0, -0.1), (0.0, 0.5, -0.1), (-0.3, -0.3, -0.1)]

    mapper32 = _make_mapper(
        [SensorConfig(position=np.array([0.3, 0.0, 0.2], dtype=np.float32))],
        use_gpu=False,
    )
    grid32 = np.asarray(mapper32.scan_to_grid(clouds=[_cloud_dict(points)])).copy()

    buffer64 = np.zeros((len(points), 3), dtype=np.float64)
    for row, point in enumerate(points):
        buffer64[row] = point
    cloud64 = {
        "data": buffer64.reshape(-1).view(np.uint8),
        "point_step": 24,
        "row_step": 24 * len(points),
        "height": 1,
        "width": len(points),
        "x_offset": 0,
        "y_offset": 8,
        "z_offset": 16,
    }
    mapper64 = _make_mapper(
        [
            SensorConfig(
                position=np.array([0.3, 0.0, 0.2], dtype=np.float32),
                cloud_field_type=PointFieldType.FLOAT64,
            )
        ],
        use_gpu=False,
    )
    grid64 = np.asarray(mapper64.scan_to_grid(clouds=[cloud64]))

    assert (grid32 == OCCUPIED).sum() > 0
    np.testing.assert_array_equal(grid32, grid64)
