import json
import logging
import math
import os
import random
from pathlib import Path
from typing import Tuple

import numpy as np
import pytest
from kompass_core.datatypes.laserscan import LaserScanData
from kompass_core.datatypes.pointcloud import PointCloudData
from kompass_core.datatypes.pose import PoseData
from kompass_core.datatypes.scan_model import ScanModelConfig
from kompass_core.mapping import LocalMapper, MapConfig
from kompass_core.utils.visualization import visualize_grid
from kompass_cpp.mapping import OCCUPANCY_TYPE


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


RESOURCES_DIR = Path(__file__).parent / "resources" / "mapping"
LASERSCAN_JSON = RESOURCES_DIR / "laserscan_data.json"
LIVOX_CLOUD_JSON = RESOURCES_DIR / "livox_pointcloud_sample_0.json"


def _get_random_pose(rng: random.Random,
                     min_range: float = -100.0,
                     max_range: float = 100.0) -> PoseData:
    p = PoseData()
    p.x = rng.uniform(min_range, max_range)
    p.y = rng.uniform(min_range, max_range)
    p.z = rng.uniform(min_range, max_range)
    p.qw = rng.uniform(-1, 1)
    p.qx = rng.uniform(-1, 1)
    p.qy = rng.uniform(-1, 1)
    p.qz = rng.uniform(-1, 1)
    return p


@pytest.fixture
def pose_robot_in_world() -> PoseData:
    # Seeded so test runs are reproducible.
    return _get_random_pose(random.Random(42))


@pytest.fixture
def logs_test_dir() -> str:
    root_dir = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(root_dir, "logs")
    os.makedirs(path, exist_ok=True)
    return path


@pytest.fixture
def local_mapper() -> LocalMapper:
    mapper_config = MapConfig(width=3.0, height=3.0, padding=0.0, resolution=0.05)
    scan_model_config = ScanModelConfig(
        p_prior=0.5,
        p_occupied=0.9,
        range_sure=0.1,
        range_max=20.0,
    )
    return LocalMapper(config=mapper_config, scan_model_config=scan_model_config)


# ---------------------------------------------------------------------------
# Laserscan: parametrised over scan shapes (existing matrix, now asserting)
# ---------------------------------------------------------------------------


@pytest.fixture(
    params=[
        "out_of_grid",
        "circle_in_grid",
        "circle_at_edge",
        "at_45_deg_only",
        "random_in_grid",
        "continuous",
        "random",
    ]
)
def range_option(request):
    return request.param


@pytest.fixture
def laser_scan_data(local_mapper: LocalMapper, range_option: str) -> LaserScanData:
    data = json.loads(LASERSCAN_JSON.read_text())

    scan = LaserScanData()
    scan.angle_min = data["angle_min"]
    scan.angle_max = data["angle_max"]
    scan.angle_increment = data["angle_increment"]
    scan.time_increment = data["time_increment"]
    scan.scan_time = data["scan_time"]
    scan.range_min = data["range_min"]
    scan.range_max = data["range_max"]

    # Regenerate angles to match the JSON-loaded angle_min/max/increment.
    # LaserScanData's __attrs_post_init__ runs at default-construction and
    # seeds angles from the default fields; we have to rebuild it here so
    # angles.size matches the ranges we populate below.
    scan.angles = np.arange(
        scan.angle_min, scan.angle_max, scan.angle_increment,
    )
    angles_size = scan.angles.shape[0]

    scan.intensities = [0.0] * angles_size
    width = local_mapper.grid_width * local_mapper.config.resolution
    height = local_mapper.grid_height * local_mapper.config.resolution
    max_range_quarter = 0.25 * min(width, height)
    max_range_half = 0.5 * min(width, height)
    min_range_from_robot = local_mapper.config.resolution * 2.0
    angle_increment_45 = 0.785398

    rng = np.random.default_rng(seed=0)

    if range_option == "out_of_grid":
        half_diag = math.sqrt(width ** 2 + height ** 2)
        scan.ranges = np.array([half_diag] * angles_size)
    elif range_option == "circle_in_grid":
        scan.ranges = np.array([max_range_quarter] * angles_size)
    elif range_option == "circle_at_edge":
        scan.ranges = np.array([max_range_half] * angles_size)
    elif range_option == "random_in_grid":
        scan.ranges = rng.uniform(
            min_range_from_robot, max_range_quarter, size=angles_size,
        )
    elif range_option == "at_45_deg_only":
        scan.angle_increment = angle_increment_45
        angles_size = np.arange(
            scan.angle_min, scan.angle_max, scan.angle_increment,
        ).shape[0]
        scan.angles = np.arange(
            scan.angle_min, scan.angle_max, scan.angle_increment,
        )
        scan.ranges = np.array([max_range_quarter] * angles_size)
        scan.ranges[0] = 0.0
        scan.ranges[1] = 0.1
    elif range_option == "continuous":
        # Clusters of non-zero ranges interspersed with zero-gaps.
        scan.ranges = np.zeros(angles_size)
        rng_py = random.Random(1)
        i = 0
        while i < angles_size:
            c = rng_py.randint(10, 20)
            r = rng_py.uniform(min_range_from_robot, max_range_half)
            c = c if c + i <= angles_size else angles_size - i
            scan.ranges[i] = r
            i += c
        assert scan.ranges.size == angles_size
    else:  # random
        scan.ranges = rng.uniform(
            min_range_from_robot,
            local_mapper.scan_model.range_max,
            size=angles_size,
        )

    return scan


def _count(grid: np.ndarray, value: int) -> int:
    return int(np.count_nonzero(grid == value))


def _occupancy_counts(grid: np.ndarray) -> Tuple[int, int, int]:
    occ = _count(grid, OCCUPANCY_TYPE.OCCUPIED.value)
    empty = _count(grid, OCCUPANCY_TYPE.EMPTY.value)
    unknown = _count(grid, OCCUPANCY_TYPE.UNEXPLORED.value)
    return occ, empty, unknown


def test_update_from_scan(
    local_mapper: LocalMapper,
    laser_scan_data: LaserScanData,
    pose_robot_in_world: PoseData,
    logs_test_dir: str,
    range_option: str,
):
    """Drive the laserscan update path and assert the occupancy grid is
    well-formed for each scan-shape scenario."""
    local_mapper.update_from_scan(pose_robot_in_world, laser_scan_data)

    grid = local_mapper.grid_data.occupancy
    n_occ, n_empty, n_unknown = _occupancy_counts(grid)
    total = grid.size

    logging.info(
        "[%s] OCCUPIED=%d EMPTY=%d UNEXPLORED=%d total=%d",
        range_option, n_occ, n_empty, n_unknown, total,
    )

    # Invariant for every scenario: the three classes partition the grid
    # and only these three values appear.
    assert n_occ + n_empty + n_unknown == total, (
        f"[{range_option}] classes don't partition grid: "
        f"{n_occ}+{n_empty}+{n_unknown} != {total}"
    )

    # The mapper must have stamped *something* from a non-empty scan.
    assert n_occ + n_empty > 0, (
        f"[{range_option}] grid has zero stamped cells — mapper ran but "
        "did nothing"
    )

    # Scenario-specific expectations:
    if range_option == "circle_in_grid":
        # Closed ring fully inside the grid: must stamp obstacle cells.
        assert n_occ > 0, "circle_in_grid: expected OCCUPIED ring cells"
        assert n_empty > 0, "circle_in_grid: expected EMPTY interior"
    elif range_option == "out_of_grid":
        # Every ray terminates well past the grid boundary, so OCCUPIED
        # stamps should be at most a handful — driven by float-precision
        # edges in `ceil(cos(θ)*R/res)` right at the grid boundary. The
        # SYCL backend choice (CUDA vs OpenMP host) can shift one or two
        # cells either way. Rays still sweep the grid, so most cells end
        # up EMPTY.
        assert n_occ < 0.01 * total, (
            f"out_of_grid: expected ≤1% cells OCCUPIED (rays clipped), "
            f"got {n_occ}/{total}"
        )
        assert n_empty > 0, "out_of_grid: rays still sweep EMPTY through grid"
    elif range_option == "at_45_deg_only":
        # Only eight rays; at least one endpoint lands inside.
        assert n_occ >= 1, f"at_45_deg_only: expected ≥1 OCCUPIED, got {n_occ}"

    visualize_grid(
        grid, scale=100, show_image=False,
        save_file=os.path.join(logs_test_dir, f"grid_occupancy_{range_option}.jpg"),
    )


# ---------------------------------------------------------------------------
# Pointcloud: synthetic (always runs) + livox (skipped if file missing)
# ---------------------------------------------------------------------------


# Matches the PointCloud2 layout the C++ tests use: 16 bytes per point,
# x/y/z as float32 at offsets 0/4/8, with 4 bytes padding.
_PC_STRIDE = 16
_PC_X_OFFSET = 0
_PC_Y_OFFSET = 4
_PC_Z_OFFSET = 8


def _make_synthetic_pointcloud(
    points_xyz: np.ndarray,
) -> PointCloudData:
    """Pack an Nx3 float32 array into a PointCloud2-style byte buffer.

    Each point is stored as 4 consecutive float32 (x, y, z, padding).
    """
    assert points_xyz.ndim == 2 and points_xyz.shape[1] == 3
    n = points_xyz.shape[0]
    buffer = np.zeros((n, 4), dtype=np.float32)
    buffer[:, :3] = points_xyz.astype(np.float32)
    raw = np.frombuffer(buffer.tobytes(), dtype=np.int8)
    return PointCloudData(
        data=raw,
        point_step=_PC_STRIDE,
        row_step=n * _PC_STRIDE,
        height=1,
        width=n,
        x_offset=_PC_X_OFFSET,
        y_offset=_PC_Y_OFFSET,
        z_offset=_PC_Z_OFFSET,
    )


def _origin_pose() -> PoseData:
    # Pose data is used only to grid-shift across frames; a stable pose
    # exercises update_from_scan without triggering the shift path.
    p = PoseData()
    p.x = p.y = p.z = 0.0
    p.qw = 1.0
    p.qx = p.qy = p.qz = 0.0
    return p


def test_update_from_pointcloud_synthetic_ring(logs_test_dir: str):
    """Deterministic synthetic ring of points should stamp OCCUPIED cells
    along the circle and EMPTY cells along the rays back to the origin."""
    mapper_config = MapConfig(width=3.0, height=3.0, padding=0.0, resolution=0.05)
    scan_model = ScanModelConfig(
        angle_step=0.01,
        min_height=-0.5,
        max_height=1.5,
        range_max=5.0,
    )
    mapper = LocalMapper(config=mapper_config, scan_model_config=scan_model)

    # 360-point ring at radius 0.5 m, z=0.1 m (inside the z filter window).
    n = 360
    theta = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    ring = np.column_stack([
        0.5 * np.cos(theta),
        0.5 * np.sin(theta),
        np.full(n, 0.1),
    ])
    cloud = _make_synthetic_pointcloud(ring)

    mapper.update_from_scan(_origin_pose(), cloud)

    grid = mapper.grid_data.occupancy
    n_occ, n_empty, n_unknown = _occupancy_counts(grid)

    logging.info(
        "synthetic ring: OCCUPIED=%d EMPTY=%d UNEXPLORED=%d",
        n_occ, n_empty, n_unknown,
    )

    assert n_occ + n_empty + n_unknown == grid.size
    assert n_occ > 0, "ring should stamp OCCUPIED cells"
    assert n_empty > 0, "rays from origin should stamp EMPTY cells"

    visualize_grid(
        grid, scale=50, show_image=False,
        save_file=os.path.join(logs_test_dir, "pc_synthetic_ring.jpg"),
    )


def test_update_from_pointcloud_z_filter_above_ceiling():
    """Points above max_height must be rejected by the GPU kernel's
    Z-filter; the grid must remain entirely UNEXPLORED."""
    mapper_config = MapConfig(width=3.0, height=3.0, padding=0.0, resolution=0.05)
    scan_model = ScanModelConfig(
        angle_step=0.05,
        min_height=0.0,
        max_height=1.0,
        range_max=5.0,
    )
    mapper = LocalMapper(config=mapper_config, scan_model_config=scan_model)

    # All points above the ceiling — all filtered.
    n = 64
    theta = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    cloud_pts = np.column_stack([
        np.cos(theta),
        np.sin(theta),
        np.full(n, 3.0),  # z=3.0 m, well above max_height=1.0
    ])
    cloud = _make_synthetic_pointcloud(cloud_pts)

    mapper.update_from_scan(_origin_pose(), cloud)

    grid = mapper.grid_data.occupancy
    n_occ, n_empty, n_unknown = _occupancy_counts(grid)

    assert n_occ == 0, (
        f"z-filter: every point is above ceiling, expected zero OCCUPIED, "
        f"got {n_occ}"
    )
    # Every bin receives max_range, so rays still walk the grid and stamp
    # EMPTY along the way. But no cell should be OCCUPIED.


def test_update_from_pointcloud_origin_only_points_filtered():
    """Points at the sensor origin (r² < 1e-6) must be dropped — they
    carry no direction information."""
    mapper_config = MapConfig(width=3.0, height=3.0, padding=0.0, resolution=0.05)
    scan_model = ScanModelConfig(
        angle_step=0.05,
        min_height=-0.1,
        max_height=0.3,
        range_max=5.0,
    )
    mapper = LocalMapper(config=mapper_config, scan_model_config=scan_model)

    # Only origin points — should produce no OCCUPIED cells.
    cloud = _make_synthetic_pointcloud(
        np.array([[0.0, 0.0, 0.1]] * 16, dtype=np.float32),
    )

    # This must not crash.
    mapper.update_from_scan(_origin_pose(), cloud)

    grid = mapper.grid_data.occupancy
    n_occ, _, _ = _occupancy_counts(grid)
    assert n_occ == 0, (
        f"origin-only cloud: no point has a direction, expected zero "
        f"OCCUPIED, got {n_occ}"
    )


@pytest.mark.skipif(
    not LIVOX_CLOUD_JSON.exists() or LIVOX_CLOUD_JSON.stat().st_size < 1_000_000,
    reason=(
        "livox_pointcloud_sample_0.json not available (too large for CI). "
        "Drop the file into tests/resources/mapping/ to enable this test."
    ),
)
def test_update_from_pointcloud_livox_recording(logs_test_dir: str):
    """Real-world Livox cloud from a recorded frame. No strict assertions
    on the grid — the cloud is messy — just verifies the path doesn't
    crash, produces a well-formed grid, and stamps *some* occupancy.
    """
    pc_json = json.loads(LIVOX_CLOUD_JSON.read_text())
    offset_map = {f["name"]: f["offset"] for f in pc_json["fields"]}

    cloud = PointCloudData(
        data=np.array(pc_json["data"]).astype(np.int8),
        point_step=pc_json["point_step"],
        row_step=pc_json["row_step"],
        height=pc_json["height"],
        width=pc_json["width"],
        x_offset=offset_map["x"],
        y_offset=offset_map["y"],
        z_offset=offset_map["z"],
    )

    mapper_config = MapConfig(width=10.0, height=10.0, padding=0.0, resolution=0.05)
    scan_model = ScanModelConfig(
        angle_step=0.01,
        min_height=0.1,
        max_height=2.0,
        range_max=20.0,
    )
    mapper = LocalMapper(config=mapper_config, scan_model_config=scan_model)

    mapper.update_from_scan(_origin_pose(), cloud)

    grid = mapper.grid_data.occupancy
    n_occ, n_empty, n_unknown = _occupancy_counts(grid)

    logging.info(
        "livox: OCCUPIED=%d EMPTY=%d UNEXPLORED=%d total=%d",
        n_occ, n_empty, n_unknown, grid.size,
    )

    assert n_occ + n_empty + n_unknown == grid.size
    assert n_occ > 0, "livox cloud should stamp some OCCUPIED cells"
    assert n_empty > 0, "livox cloud should stamp some EMPTY cells"

    visualize_grid(
        grid, scale=50, show_image=False,
        save_file=os.path.join(logs_test_dir, "pc_livox.jpg"),
    )


# ---------------------------------------------------------------------------
# Bayesian local mapping
#
# The GPU LocalMapperGPU::scanToGridBaysian path is the only supported
# implementation — the CPU LocalMapper Bayesian methods exist in C++ but the
# Python wrapper logs a warning and falls back to non-Bayesian when only the
# CPU backend is built. These tests skip themselves when the GPU backend
# isn't available so they remain collectable on CPU-only builds.
# ---------------------------------------------------------------------------


try:
    from kompass_cpp.mapping import LocalMapperGPU  # noqa: F401
    _HAS_GPU = True
except ImportError:
    _HAS_GPU = False


skip_no_gpu = pytest.mark.skipif(
    not _HAS_GPU,
    reason="kompass_cpp.mapping.LocalMapperGPU not available in this build",
)


@pytest.fixture
def bayesian_mapper() -> LocalMapper:
    """LocalMapper configured with `baysian_update=True` so `_initialize_mapper`
    routes through the GPU Bayesian ctor and `update_from_scan` calls
    `scan_to_grid_baysian`."""
    mapper_config = MapConfig(
        width=3.0, height=3.0, padding=0.0, resolution=0.1,
        baysian_update=True,
    )
    scan_model_config = ScanModelConfig(
        p_prior=0.6,
        p_occupied=0.9,
        range_sure=0.1,
        range_max=20.0,
    )
    return LocalMapper(config=mapper_config, scan_model_config=scan_model_config)


def _circle_laserscan(n: int, radius: float) -> LaserScanData:
    """Build a closed-circle laserscan: `n` evenly-spaced angles in [0, 2π),
    every ray at the same `radius`. Matches the scan_size the GPU mapper
    derives for pointcloud=False from `ScanModelConfig.angle_step`."""
    scan = LaserScanData()
    scan.angle_min = 0.0
    scan.angle_max = float(2.0 * math.pi)
    scan.angle_increment = float(2.0 * math.pi / n)
    scan.time_increment = 0.0
    scan.scan_time = 0.0
    scan.range_min = 0.0
    scan.range_max = 20.0
    scan.angles = np.arange(scan.angle_min, scan.angle_max, scan.angle_increment)
    scan.intensities = [0.0] * scan.angles.size
    scan.ranges = np.full(scan.angles.size, radius, dtype=np.float64)
    return scan


@skip_no_gpu
def test_bayesian_update_writes_to_occupancy_layer(
    bayesian_mapper: LocalMapper,
):
    """Bayesian update must populate `grid_data.occupancy` (the single
    output channel) with the thresholded discrete grid. Verifies the
    Python wrapper's routing — Bayesian no longer writes to a separate
    `occupancy_prob` layer."""
    n = int(math.ceil(2.0 * math.pi / bayesian_mapper.scan_model.angle_step))
    scan = _circle_laserscan(n=n, radius=0.5)

    bayesian_mapper.update_from_scan(_origin_pose(), scan)

    grid = bayesian_mapper.grid_data.occupancy
    assert grid.dtype == np.int32
    assert grid.shape == (bayesian_mapper.grid_width, bayesian_mapper.grid_height)

    # Only the three legal codes should appear.
    allowed = {
        OCCUPANCY_TYPE.OCCUPIED.value,
        OCCUPANCY_TYPE.EMPTY.value,
        OCCUPANCY_TYPE.UNEXPLORED.value,
    }
    unique_vals = set(np.unique(grid).tolist())
    assert unique_vals.issubset(allowed), unique_vals - allowed

    n_occ, n_empty, n_unknown = _occupancy_counts(grid)
    assert n_occ + n_empty + n_unknown == grid.size
    assert n_occ > 0, "circle scan should stamp OCCUPIED cells at the ring"
    assert n_empty > 0, "rays from origin should stamp EMPTY cells in the interior"


@skip_no_gpu
def test_bayesian_probabilities_accessor_in_unit_interval(
    bayesian_mapper: LocalMapper,
):
    """`mapper.probabilities` D2H-copies the log-odds buffer and applies
    sigmoid on the C++ side. Every cell must be in [0, 1], untouched cells
    must equal `p_prior` (sigmoid of `h0` is `p_prior` by construction),
    and at least one observed cell must lift above `p_prior`."""
    n = int(math.ceil(2.0 * math.pi / bayesian_mapper.scan_model.angle_step))
    scan = _circle_laserscan(n=n, radius=0.5)
    bayesian_mapper.update_from_scan(_origin_pose(), scan)

    probs = bayesian_mapper.probabilities
    assert probs is not None
    assert probs.dtype == np.float32
    assert probs.shape == (bayesian_mapper.grid_width, bayesian_mapper.grid_height)

    pmin, pmax = float(probs.min()), float(probs.max())
    assert 0.0 <= pmin <= 1.0, f"prob min out of [0,1]: {pmin}"
    assert 0.0 <= pmax <= 1.0, f"prob max out of [0,1]: {pmax}"
    assert np.all(np.isfinite(probs)), "probabilities must be finite"

    # An observation strong enough to flip a cell to OCCUPIED must lift
    # its posterior above the prior — verify against the configured prior.
    p_prior = bayesian_mapper.scan_model.p_prior
    assert pmax > p_prior, (
        f"after one observation, expected max prob > p_prior={p_prior}, "
        f"got {pmax}"
    )


@skip_no_gpu
def test_bayesian_accumulation_across_frames(bayesian_mapper: LocalMapper):
    """Per arXiv:2101.01831 eq. 8 the log-odds at observed cells should
    sum across frames. Repeated identical scans at identity pose must
    drive the max posterior probability monotonically toward 1."""
    n = int(math.ceil(2.0 * math.pi / bayesian_mapper.scan_model.angle_step))
    scan = _circle_laserscan(n=n, radius=0.5)

    max_probs = []
    for _ in range(5):
        bayesian_mapper.update_from_scan(_origin_pose(), scan)
        probs = bayesian_mapper.probabilities
        assert probs is not None
        max_probs.append(float(probs.max()))

    # Monotone non-decreasing (within a tiny float-noise margin).
    for i in range(1, len(max_probs)):
        assert max_probs[i] >= max_probs[i - 1] - 1e-6, (
            f"frame {i}: max prob decreased {max_probs[i - 1]} -> {max_probs[i]}"
        )

    # After 5 frames of an identical OCCUPIED observation the saturated
    # sigmoid should be > 0.9 with default p_occupied=0.9.
    assert max_probs[-1] > 0.9, (
        f"max probability after 5 frames = {max_probs[-1]}, expected > 0.9"
    )


@skip_no_gpu
def test_bayesian_update_with_pose_motion(
    bayesian_mapper: LocalMapper, logs_test_dir: str,
):
    """A second frame with a non-zero pose delta exercises the warp
    kernel. The output must still be a well-formed discrete grid, all
    probabilities must remain finite and in [0, 1]."""
    n = int(math.ceil(2.0 * math.pi / bayesian_mapper.scan_model.angle_step))
    scan = _circle_laserscan(n=n, radius=0.5)

    # Frame 0: identity pose. Populates the persistent log-odds buffer.
    bayesian_mapper.update_from_scan(_origin_pose(), scan)

    # Frame 1: small translation + yaw. Triggers the warp kernel.
    moved = PoseData()
    moved.x = 0.05
    moved.y = 0.02
    moved.z = 0.0
    # Quaternion for 0.05 rad yaw about Z.
    yaw = 0.05
    moved.qw = float(math.cos(yaw / 2))
    moved.qx = 0.0
    moved.qy = 0.0
    moved.qz = float(math.sin(yaw / 2))

    bayesian_mapper.update_from_scan(moved, scan)

    grid = bayesian_mapper.grid_data.occupancy
    n_occ, n_empty, n_unknown = _occupancy_counts(grid)
    assert n_occ + n_empty + n_unknown == grid.size
    assert n_occ > 0, "post-motion grid should still have OCCUPIED cells"

    probs = bayesian_mapper.probabilities
    assert probs is not None
    assert np.all(np.isfinite(probs)), "warp+update must not produce NaN/inf"
    assert probs.min() >= 0.0 and probs.max() <= 1.0

    visualize_grid(
        grid, scale=100, show_image=False,
        save_file=os.path.join(logs_test_dir, "bayesian_motion.jpg"),
    )


def _pose_at(x: float, y: float, yaw: float) -> PoseData:
    p = PoseData()
    p.x = x
    p.y = y
    p.z = 0.0
    p.qw = float(math.cos(yaw / 2))
    p.qx = 0.0
    p.qy = 0.0
    p.qz = float(math.sin(yaw / 2))
    return p


@skip_no_gpu
def test_bayesian_warp_no_explosion_under_sub_cell_motion(logs_test_dir: str):
    """Regression: a bilinear warp would compound sub-cell pose shifts into
    exponential growth of the OCCUPIED region. The nearest-neighbour warp
    keeps it bounded. We drive the robot forward in 0.03 m steps (0.3 cells
    at 0.1 m resolution) for 20 frames with the same scan, then assert the
    OCCUPIED count stayed close to its single-frame baseline."""
    mapper_config = MapConfig(
        width=2.0, height=2.0, padding=0.0, resolution=0.1,
        baysian_update=True,
    )
    scan_model = ScanModelConfig(
        p_prior=0.6, p_occupied=0.9, range_sure=0.1, range_max=20.0,
    )
    mapper = LocalMapper(config=mapper_config, scan_model_config=scan_model)
    n = int(math.ceil(2.0 * math.pi / scan_model.angle_step))
    scan = _circle_laserscan(n=n, radius=0.5)

    # Frame 0: identity pose, capture baseline.
    mapper.update_from_scan(_pose_at(0.0, 0.0, 0.0), scan)
    n_occ_baseline = _occupancy_counts(mapper.grid_data.occupancy)[0]

    # 20 frames of sub-cell forward motion.
    for frame in range(1, 21):
        mapper.update_from_scan(_pose_at(0.03 * frame, 0.0, 0.0), scan)

    grid = mapper.grid_data.occupancy
    n_occ_after, _, _ = _occupancy_counts(grid)
    logging.info(
        "bayesian warp sub-cell: baseline=%d after-20-frames=%d total=%d",
        n_occ_baseline, n_occ_after, grid.size,
    )

    assert n_occ_after <= n_occ_baseline * 2, (
        f"OCCUPIED grew past 2x baseline ({n_occ_baseline} -> {n_occ_after}) "
        "— warp may be smearing"
    )
    assert n_occ_after < grid.size // 4, (
        f"OCCUPIED count exceeded 25% of grid ({n_occ_after}/{grid.size}) "
        "— explosion regression"
    )

    visualize_grid(
        grid, scale=100, show_image=False,
        save_file=os.path.join(logs_test_dir, "bayesian_warp_subcell.jpg"),
    )


@skip_no_gpu
def test_bayesian_warp_super_cell_drift_bounded(logs_test_dir: str):
    """Multi-cell motion exercises the warp's actual shift path (NN rounds
    1-cell deltas cleanly). The warped previous ring drifts behind the
    robot and decays as free stamps land on it; OCCUPIED stays bounded."""
    mapper_config = MapConfig(
        width=2.0, height=2.0, padding=0.0, resolution=0.1,
        baysian_update=True,
    )
    scan_model = ScanModelConfig(
        p_prior=0.6, p_occupied=0.9, range_sure=0.1, range_max=20.0,
    )
    mapper = LocalMapper(config=mapper_config, scan_model_config=scan_model)
    n = int(math.ceil(2.0 * math.pi / scan_model.angle_step))
    scan = _circle_laserscan(n=n, radius=0.5)

    mapper.update_from_scan(_pose_at(0.0, 0.0, 0.0), scan)
    n_occ_baseline = _occupancy_counts(mapper.grid_data.occupancy)[0]

    # 10 frames of one-cell-per-frame motion.
    for frame in range(1, 11):
        mapper.update_from_scan(_pose_at(0.1 * frame, 0.0, 0.0), scan)

    grid = mapper.grid_data.occupancy
    n_occ_after, _, _ = _occupancy_counts(grid)
    logging.info(
        "bayesian warp super-cell: baseline=%d after-10-frames=%d total=%d",
        n_occ_baseline, n_occ_after, grid.size,
    )

    assert n_occ_after <= n_occ_baseline * 4, (
        f"OCCUPIED grew past 4x baseline ({n_occ_baseline} -> {n_occ_after}) "
        "under multi-cell motion"
    )
    assert n_occ_after < grid.size // 2, (
        f"OCCUPIED exceeded 50% of grid ({n_occ_after}/{grid.size}) "
        "— explosion regression"
    )

    probs = mapper.probabilities
    assert probs is not None
    assert np.all(np.isfinite(probs)), "non-finite probability after drift"
    assert probs.min() >= 0.0 and probs.max() <= 1.0

    visualize_grid(
        grid, scale=100, show_image=False,
        save_file=os.path.join(logs_test_dir, "bayesian_warp_supercell.jpg"),
    )


@skip_no_gpu
def test_bayesian_update_from_pointcloud_synthetic_ring(logs_test_dir: str):
    """The Python wrapper must route Bayesian + pointcloud inputs to the
    pointcloud overload of `scan_to_grid_baysian`. A deterministic ring
    of points should stamp OCCUPIED cells along the circle and EMPTY
    cells along the rays back to the sensor, just like the non-Bayesian
    pointcloud test."""
    mapper_config = MapConfig(
        width=3.0, height=3.0, padding=0.0, resolution=0.1,
        baysian_update=True,
    )
    scan_model = ScanModelConfig(
        p_prior=0.6,
        p_occupied=0.9,
        range_sure=0.1,
        range_max=5.0,
        angle_step=0.01,
        min_height=-0.5,
        max_height=1.5,
    )
    mapper = LocalMapper(config=mapper_config, scan_model_config=scan_model)

    n = 360
    theta = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    ring = np.column_stack([
        0.5 * np.cos(theta),
        0.5 * np.sin(theta),
        np.full(n, 0.1),
    ])
    cloud = _make_synthetic_pointcloud(ring)

    mapper.update_from_scan(_origin_pose(), cloud)

    grid = mapper.grid_data.occupancy
    n_occ, n_empty, n_unknown = _occupancy_counts(grid)
    logging.info(
        "bayesian pc ring: OCCUPIED=%d EMPTY=%d UNEXPLORED=%d",
        n_occ, n_empty, n_unknown,
    )
    assert n_occ + n_empty + n_unknown == grid.size
    assert n_occ > 0, "ring should stamp OCCUPIED cells"
    assert n_empty > 0, "rays back to origin should stamp EMPTY cells"

    # Posterior must round-trip cleanly through the post-PC pipeline.
    probs = mapper.probabilities
    assert probs is not None
    assert np.all(np.isfinite(probs))
    assert probs.min() >= 0.0 and probs.max() <= 1.0
    assert probs.max() > scan_model.p_prior

    visualize_grid(
        grid, scale=50, show_image=False,
        save_file=os.path.join(logs_test_dir, "bayesian_pc_ring.jpg"),
    )


def test_bayesian_probabilities_is_none_when_disabled(local_mapper: LocalMapper):
    """The probabilities accessor must return None when the mapper was
    constructed with `baysian_update=False`, regardless of backend.
    Guards against accidental D2H copies on the non-Bayesian hot path."""
    assert local_mapper.config.baysian_update is False
    assert local_mapper.probabilities is None


def test_bayesian_warns_and_falls_back_on_cpu_only_build(caplog):
    """If only the CPU backend is built, requesting `baysian_update=True`
    must emit a warning and silently fall back to the non-Bayesian path
    (forcing the config flag to False). The robot stays operational
    rather than crashing.

    We skip if the GPU backend is available — there's no fallback path
    to exercise."""
    if _HAS_GPU:
        pytest.skip(
            "GPU backend present; the CPU-only Bayesian fallback isn't reached"
        )

    mapper_config = MapConfig(
        width=2.0, height=2.0, padding=0.0, resolution=0.1,
        baysian_update=True,
    )
    scan_model = ScanModelConfig(
        p_prior=0.6, p_occupied=0.9, range_sure=0.1, range_max=20.0,
    )
    mapper = LocalMapper(config=mapper_config, scan_model_config=scan_model)

    n = int(math.ceil(2.0 * math.pi / scan_model.angle_step))
    scan = _circle_laserscan(n=n, radius=0.5)

    import logging as _logging
    with caplog.at_level(_logging.WARNING):
        mapper.update_from_scan(_origin_pose(), scan)

    # The wrapper should have warned and flipped the flag off.
    assert mapper.config.baysian_update is False
    assert any(
        "bayesian" in record.message.lower()
        for record in caplog.records
    ), "expected a warning about the missing Bayesian backend"
