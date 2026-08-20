"""Multi-sensor (multi point cloud) emergency stop tests, on the raw
kompass_cpp checkers — mirrors how Kompass' drive manager will construct
them for a robot with front + back lidars."""

import numpy as np
import pytest

from kompass_cpp.types import RobotGeometry, SensorConfig, SensorInputType
from kompass_cpp.utils import CriticalZoneChecker

try:
    from kompass_cpp.utils import CriticalZoneCheckerGPU

    _HAS_GPU = True
except ImportError:
    _HAS_GPU = False


_PC_STRIDE = 16  # 16B points (x, y, z, pad)

# Robot: cylinder radius 0.5; critical 0.3 (stop inside 0.8 m from center),
# slowdown 0.6 (ramp up to 1.1 m)
_ROBOT_RADIUS = 0.5
_CRIT = 0.3
_SLOW = 0.6


def _pack_points(points) -> dict:
    buffer = np.zeros((len(points), 4), dtype=np.float32)
    for row, point in enumerate(points):
        buffer[row, :3] = point
    return {
        "data": buffer.reshape(-1).view(np.uint8),
        "point_step": _PC_STRIDE,
        "row_step": _PC_STRIDE * len(points),
        "height": 1,
        "width": len(points),
        "x_offset": 0,
        "y_offset": 4,
        "z_offset": 8,
    }


def _quat_yaw(yaw: float) -> np.ndarray:
    return np.array(
        [0.0, 0.0, np.sin(yaw / 2.0), np.cos(yaw / 2.0)], dtype=np.float32
    )


def _front_back_sensors():
    """Front and back lidars at x = +/-0.2 m, mounted 0.2 m high."""
    return [
        SensorConfig(position=np.array([0.2, 0.0, 0.2], dtype=np.float32)),
        SensorConfig(
            position=np.array([-0.2, 0.0, 0.2], dtype=np.float32),
            rotation=_quat_yaw(np.pi),
        ),
    ]


def _make_checker(use_gpu: bool, sensors=None):
    if use_gpu:
        if not _HAS_GPU:
            pytest.skip("Build has no GPU implementation")
        checker_class = CriticalZoneCheckerGPU
    else:
        checker_class = CriticalZoneChecker
    return checker_class(
        input_type=SensorInputType.POINTCLOUD,
        robot_shape=RobotGeometry.CYLINDER,
        robot_dimensions=np.array([_ROBOT_RADIUS, 1.0], dtype=np.float32),
        sensor_configs=sensors if sensors is not None else _front_back_sensors(),
        critical_angle=90.0,
        critical_distance=_CRIT,
        slowdown_distance=_SLOW,
        min_height=0.0,  # body frame: ground ...
        max_height=1.0,  # ... to robot top
        range_max=20.0,
    )


_BACKENDS = [False, True]


@pytest.mark.parametrize("use_gpu", _BACKENDS)
def test_back_obstacle_stops_reverse_only(use_gpu):
    """The headline two-Livox case: an obstacle seen ONLY by the back sensor
    stops reverse motion and leaves forward motion untouched.
    Back-sensor frame (0.5, 0, 0.1) -> body (-0.7, 0, 0.3):
    0.7 - 0.5 (radius) = 0.2 < 0.3 (critical)."""
    checker = _make_checker(use_gpu)
    back = _pack_points([(0.5, 0.0, 0.1)])

    assert checker.check(clouds=[None, back], forward=False) == 0.0
    assert checker.check(clouds=[None, back], forward=True) == 1.0


@pytest.mark.parametrize("use_gpu", _BACKENDS)
def test_min_factor_across_clouds(use_gpu):
    """Fused result is the minimum over both sensors' factors."""
    checker = _make_checker(use_gpu)
    # Front-sensor frame (0.75, 0, 0.1) -> body (0.95, 0, 0.3):
    # factor = (0.95 - 0.5 - 0.3) / 0.3 = 0.5
    front_slow = _pack_points([(0.75, 0.0, 0.1)])
    back_far = _pack_points([(5.0, 0.0, 0.1)])

    factor = checker.check(clouds=[front_slow, back_far], forward=True)
    assert 0.4 < factor < 0.6

    # Adding a critical point to the FRONT cloud drives the min to 0
    front_mixed = _pack_points([(0.75, 0.0, 0.1), (0.55, 0.0, 0.1)])
    assert checker.check(clouds=[front_mixed, back_far], forward=True) == 0.0


@pytest.mark.parametrize("use_gpu", _BACKENDS)
def test_repeat_calls_and_empty_batches(use_gpu):
    """A stop must not leak into the next call, and an all-None batch is
    unconstrained (nothing observed)."""
    checker = _make_checker(use_gpu)
    danger = _pack_points([(0.5, 0.0, 0.1)])
    safe = _pack_points([(5.0, 0.0, 0.1)])

    assert checker.check(clouds=[danger, None], forward=True) == 0.0
    assert checker.check(clouds=[safe, safe], forward=True) == 1.0
    assert checker.check(clouds=[None, None], forward=True) == 1.0


@pytest.mark.parametrize("use_gpu", _BACKENDS)
def test_body_frame_height_band(use_gpu):
    """The band gates BODY-frame z: with sensors mounted at 0.2 m, a point at
    sensor z = -0.1 sits at body z = 0.1 (inside [0, 1]) and must count; a
    point at sensor z = 0.9 sits at body z = 1.1 (above the band) and must
    not."""
    checker = _make_checker(use_gpu)

    in_band = _pack_points([(0.5, 0.0, -0.1)])  # body z = 0.1
    assert checker.check(clouds=[in_band, None], forward=True) == 0.0

    above_band = _pack_points([(0.5, 0.0, 0.9)])  # body z = 1.1
    assert checker.check(clouds=[above_band, None], forward=True) == 1.0


@pytest.mark.parametrize("use_gpu", _BACKENDS)
def test_num_sensors_and_errors(use_gpu):
    checker = _make_checker(use_gpu)
    assert checker.num_sensors == 2

    cloud = _pack_points([(5.0, 0.0, 0.1)])

    with pytest.raises(ValueError):
        checker.check(clouds=[cloud], forward=True)

    bad = _pack_points([(5.0, 0.0, 0.1)])
    bad["y_offset"] = -4
    with pytest.raises(ValueError, match=r"clouds\[1\].*non-negative"):
        checker.check(clouds=[cloud, bad], forward=True)

    # Laserscan input on a pointcloud checker is rejected loudly
    with pytest.raises(RuntimeError):
        checker.check(
            ranges=np.ones(4, dtype=np.float32).reshape(-1), forward=True
        )


@pytest.mark.parametrize("use_gpu", _BACKENDS)
def test_cpu_gpu_agree(use_gpu):
    """Both backends compute the same factor for the same batch (the CPU
    per-point loop mirrors the GPU kernel)."""
    if not use_gpu:
        pytest.skip("comparison runs once, from the GPU parametrization")
    cpu = _make_checker(False)
    gpu = _make_checker(True)

    front = _pack_points([(0.75, 0.0, 0.1), (2.0, 1.0, 0.1)])
    back = _pack_points([(0.9, 0.2, 0.1), (5.0, 0.0, 0.1)])

    for forward in (True, False):
        cpu_factor = cpu.check(clouds=[front, back], forward=forward)
        gpu_factor = gpu.check(clouds=[front, back], forward=forward)
        assert cpu_factor == pytest.approx(gpu_factor, abs=1e-5)
