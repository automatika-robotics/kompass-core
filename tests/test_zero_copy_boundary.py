"""Zero-copy guard for the batched cloud entries.

The batched ``clouds=[...]`` bindings extract METADATA ONLY under the GIL
(8 ints + a buffer view per cloud); the point bytes must never be copied
or converted at the Python boundary. This guard times the batched N=1
entry against the legacy single-cloud entry on the same 100k-point buffer:
they run the identical C++ path, so any per-point boundary work (which
would be 10-100x the metadata cost) blows the ratio immediately.

CPU classes are used on purpose: no JIT warm-up noise, and the binding
boundary under test is the same code for the GPU classes.
"""

import time

import numpy as np

from kompass_cpp.mapping import LocalMapper as LocalMapperCpp
from kompass_cpp.types import RobotGeometry, SensorConfig, SensorInputType
from kompass_cpp.utils import CriticalZoneChecker

_PC_STRIDE = 16  # 16B points (x, y, z, pad)
_N_POINTS = 100_000
_REPS = 30
# A metadata-only boundary measures ~1.0; per-point work measures 10x+.
# 1.5 tolerates scheduler noise without ever masking a regression.
_MAX_RATIO = 1.5


def _far_cloud(n: int = _N_POINTS) -> dict:
    """n points uniformly in a 2-10 m ring around the sensor: outside the
    checker's slowdown ring (no early-exit, every point is swept) and
    inside the mapper's range_max (every point reaches the binning)."""
    rng = np.random.default_rng(1234)
    buffer = np.zeros((n, 4), dtype=np.float32)
    radius = rng.uniform(2.0, 10.0, size=n).astype(np.float32)
    theta = rng.uniform(-np.pi, np.pi, size=n).astype(np.float32)
    buffer[:, 0] = radius * np.cos(theta)
    buffer[:, 1] = radius * np.sin(theta)
    buffer[:, 2] = rng.uniform(0.1, 0.9, size=n).astype(np.float32)
    return {
        "data": buffer.reshape(-1).view(np.uint8),
        "point_step": _PC_STRIDE,
        "row_step": _PC_STRIDE * n,
        "height": 1,
        "width": n,
        "x_offset": 0,
        "y_offset": 4,
        "z_offset": 8,
    }


def _median_ratio(single_call, batched_call, reps: int = _REPS) -> float:
    """Interleaved A/B timing -> median(batched) / median(single)."""
    for _ in range(3):
        single_call()
        batched_call()
    singles, batched = [], []
    for _ in range(reps):
        t0 = time.perf_counter()
        single_call()
        t1 = time.perf_counter()
        batched_call()
        t2 = time.perf_counter()
        singles.append(t1 - t0)
        batched.append(t2 - t1)
    return float(np.median(batched) / np.median(singles))


def test_checker_batched_n1_within_noise_of_single_cloud():
    checker = CriticalZoneChecker(
        input_type=SensorInputType.POINTCLOUD,
        robot_shape=RobotGeometry.CYLINDER,
        robot_dimensions=np.array([0.5, 1.0], dtype=np.float32),
        sensor_configs=[SensorConfig()],
        critical_angle=90.0,
        critical_distance=0.3,
        slowdown_distance=0.6,
        min_height=0.0,
        max_height=1.0,
        range_max=20.0,
    )
    cloud = _far_cloud()

    # Same C++ path -> identical result before we bother timing anything
    assert checker.check(**cloud, forward=True) == checker.check(
        clouds=[cloud], forward=True
    )

    ratio = _median_ratio(
        lambda: checker.check(**cloud, forward=True),
        lambda: checker.check(clouds=[cloud], forward=True),
    )
    assert ratio < _MAX_RATIO, (
        f"batched N=1 check() costs {ratio:.2f}x the single-cloud entry: "
        "the binding boundary is doing per-point work"
    )


def test_mapper_batched_n1_within_noise_of_single_cloud():
    mapper = LocalMapperCpp(
        grid_height=100,
        grid_width=100,
        resolution=0.05,
        sensor_configs=[SensorConfig()],
        is_pointcloud=True,
        scan_size=360,
        max_height=1.0,
        min_height=0.0,
        range_max=20.0,
    )
    cloud = _far_cloud()

    single_grid = np.asarray(mapper.scan_to_grid(**cloud)).copy()
    batched_grid = np.asarray(mapper.scan_to_grid(clouds=[cloud]))
    np.testing.assert_array_equal(single_grid, batched_grid)

    ratio = _median_ratio(
        lambda: mapper.scan_to_grid(**cloud),
        lambda: mapper.scan_to_grid(clouds=[cloud]),
    )
    assert ratio < _MAX_RATIO, (
        f"batched N=1 scan_to_grid() costs {ratio:.2f}x the single-cloud "
        "entry: the binding boundary is doing per-point work"
    )
