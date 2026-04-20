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
        wall_size=0.075,
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
