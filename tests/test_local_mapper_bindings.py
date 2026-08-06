import numpy as np
import pytest

from kompass_cpp.mapping import OCCUPANCY_TYPE


# LocalMapperGPU is only exported when the build has SYCL. Import guarded
# so CPU-only builds still collect the file.
try:
    from kompass_cpp.mapping import LocalMapperGPU
    _HAS_GPU = True
except ImportError:
    _HAS_GPU = False

from kompass_cpp.mapping import LocalMapper as LocalMapperCpp


# ---------------------------------------------------------------------------
# Helpers — deterministic inputs that don't need any resource files.
# ---------------------------------------------------------------------------


# Matches tests/test_local_mapper_pytest.py — 16B points (x, y, z, pad).
_PC_STRIDE = 16


def _ring_cloud(n: int = 200, radius: float = 0.5, z: float = 0.1) -> np.ndarray:
    """Build a PointCloud2-style byte buffer holding `n` points
    evenly spaced on a circle of `radius` at height `z`."""
    theta = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    buf = np.zeros((n, 4), dtype=np.float32)
    buf[:, 0] = radius * np.cos(theta)
    buf[:, 1] = radius * np.sin(theta)
    buf[:, 2] = z
    return np.frombuffer(buf.tobytes(), dtype=np.uint8)


def _occupancy_counts(grid: np.ndarray):
    occ = int(np.count_nonzero(grid == OCCUPANCY_TYPE.OCCUPIED.value))
    empty = int(np.count_nonzero(grid == OCCUPANCY_TYPE.EMPTY.value))
    unknown = int(np.count_nonzero(grid == OCCUPANCY_TYPE.UNEXPLORED.value))
    return occ, empty, unknown


# ---------------------------------------------------------------------------
# GPU bindings: LocalMapperGPU
# ---------------------------------------------------------------------------


pytestmark_gpu = pytest.mark.skipif(
    not _HAS_GPU,
    reason="kompass_cpp.mapping.LocalMapperGPU not available in this build",
)


@pytestmark_gpu
def test_gpu_binding_signature_laserscan_ctor():
    """Construction with laserscan-mode kwargs must accept every argument
    the Python wrapper passes in _initialize_mapper."""
    mapper = LocalMapperGPU(
        grid_height=40,
        grid_width=40,
        resolution=0.05,
        laserscan_position=np.array([0.0, 0.0, 0.0], dtype=np.float32),
        laserscan_orientation=0.0,
        is_pointcloud=False,
        scan_size=360,
        angle_step=0.01,
        max_height=2.0,
        min_height=0.0,
        range_max=10.0,
        max_points_per_line=32,
    )
    assert mapper is not None


@pytestmark_gpu
def test_gpu_binding_laserscan_scan_to_grid_basic():
    """A full-circle laserscan at a radius that fits inside the grid must
    produce a valid occupancy grid (correct dtype, shape, partition)."""
    grid_height = 40
    grid_width = 40
    n = 360
    mapper = LocalMapperGPU(
        grid_height=grid_height,
        grid_width=grid_width,
        resolution=0.05,
        laserscan_position=np.array([0.0, 0.0, 0.0], dtype=np.float32),
        laserscan_orientation=0.0,
        is_pointcloud=False,
        scan_size=n,
        angle_step=float(2.0 * np.pi / n),
        max_height=2.0,
        min_height=0.0,
        range_max=10.0,
    )

    angles = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    ranges = np.full(n, 0.5, dtype=np.float64)

    grid = mapper.scan_to_grid(angles=angles, ranges=ranges)
    grid_np = np.asarray(grid)

    assert grid_np.shape == (grid_height, grid_width), grid_np.shape
    assert grid_np.dtype == np.int32, grid_np.dtype

    # Only the three known codes should appear.
    allowed = {
        OCCUPANCY_TYPE.OCCUPIED.value,
        OCCUPANCY_TYPE.EMPTY.value,
        OCCUPANCY_TYPE.UNEXPLORED.value,
    }
    unique_vals = set(np.unique(grid_np).tolist())
    assert unique_vals.issubset(allowed), (
        f"unexpected cell value: {unique_vals - allowed}"
    )

    n_occ, n_empty, n_unknown = _occupancy_counts(grid_np)
    assert n_occ + n_empty + n_unknown == grid_np.size
    assert n_occ > 0, "ring scan should stamp OCCUPIED cells"


@pytestmark_gpu
def test_gpu_binding_pointcloud_scan_to_grid_basic():
    """The pointcloud overload should accept raw bytes + offsets and
    produce a valid occupancy grid with no CPU-side laserscan intermediate."""
    grid_height = 40
    grid_width = 40
    n_rays = 360
    mapper = LocalMapperGPU(
        grid_height=grid_height,
        grid_width=grid_width,
        resolution=0.05,
        laserscan_position=np.array([0.0, 0.0, 0.0], dtype=np.float32),
        laserscan_orientation=0.0,
        is_pointcloud=True,
        scan_size=n_rays,
        angle_step=float(2.0 * np.pi / n_rays),
        max_height=1.5,
        min_height=-0.5,
        range_max=5.0,
    )

    cloud_bytes = _ring_cloud(n=200, radius=0.5, z=0.1)
    num_points = cloud_bytes.size // _PC_STRIDE

    grid = mapper.scan_to_grid(
        data=cloud_bytes,
        point_step=_PC_STRIDE,
        row_step=num_points * _PC_STRIDE,
        height=1,
        width=num_points,
        x_offset=0,
        y_offset=4,
        z_offset=8,
    )
    grid_np = np.asarray(grid)

    assert grid_np.shape == (grid_height, grid_width), grid_np.shape
    assert grid_np.dtype == np.int32, grid_np.dtype

    n_occ, n_empty, n_unknown = _occupancy_counts(grid_np)
    assert n_occ + n_empty + n_unknown == grid_np.size
    assert n_occ > 0, "ring pointcloud should stamp OCCUPIED cells"
    assert n_empty > 0, "rays back to origin should stamp EMPTY cells"


@pytestmark_gpu
def test_gpu_binding_pointcloud_z_filter_above_ceiling():
    """Every point with z > max_height must be rejected by the GPU
    conversion kernel — no OCCUPIED cells should appear."""
    grid_height = 40
    grid_width = 40
    n_rays = 360
    mapper = LocalMapperGPU(
        grid_height=grid_height,
        grid_width=grid_width,
        resolution=0.05,
        laserscan_position=np.array([0.0, 0.0, 0.0], dtype=np.float32),
        laserscan_orientation=0.0,
        is_pointcloud=True,
        scan_size=n_rays,
        angle_step=float(2.0 * np.pi / n_rays),
        max_height=1.0,
        min_height=0.0,
        range_max=5.0,
    )

    # All points at z=3.0 — above the ceiling.
    above = _ring_cloud(n=200, radius=0.5, z=3.0)
    num_points = above.size // _PC_STRIDE

    grid = mapper.scan_to_grid(
        data=above,
        point_step=_PC_STRIDE,
        row_step=num_points * _PC_STRIDE,
        height=1,
        width=num_points,
        x_offset=0,
        y_offset=4,
        z_offset=8,
    )
    grid_np = np.asarray(grid)

    n_occ, _, _ = _occupancy_counts(grid_np)
    assert n_occ == 0, f"z-filter: expected zero OCCUPIED, got {n_occ}"


@pytestmark_gpu
def test_gpu_binding_pointcloud_empty_cloud_does_not_crash():
    """An empty pointcloud must be handled cleanly — no SIGSEGV, grid
    stays UNEXPLORED everywhere."""
    grid_height = 20
    grid_width = 20
    n_rays = 90
    mapper = LocalMapperGPU(
        grid_height=grid_height,
        grid_width=grid_width,
        resolution=0.1,
        laserscan_position=np.array([0.0, 0.0, 0.0], dtype=np.float32),
        laserscan_orientation=0.0,
        is_pointcloud=True,
        scan_size=n_rays,
        angle_step=float(2.0 * np.pi / n_rays),
        max_height=1.0,
        min_height=0.0,
        range_max=5.0,
    )

    empty = np.zeros(0, dtype=np.uint8)

    # width=0 signals zero points. The pointcloud overload short-circuits
    # on this and returns an all-UNEXPLORED grid; it must NOT stamp EMPTY
    # out to max_range, which would mask a dropped sensor frame.
    grid = mapper.scan_to_grid(
        data=empty,
        point_step=_PC_STRIDE,
        row_step=0,
        height=0,
        width=0,
        x_offset=0,
        y_offset=4,
        z_offset=8,
    )
    grid_np = np.asarray(grid)

    assert grid_np.shape == (grid_height, grid_width)
    n_occ, n_empty, n_unknown = _occupancy_counts(grid_np)
    assert n_occ == 0, f"empty cloud: expected zero OCCUPIED, got {n_occ}"
    assert n_empty == 0, (
        f"empty cloud: expected zero EMPTY (should not mask sensor "
        f"dropouts), got {n_empty}"
    )
    assert n_unknown == grid_np.size, (
        f"empty cloud: expected all UNEXPLORED, got {n_unknown}/{grid_np.size}"
    )


# ---------------------------------------------------------------------------
# CPU bindings: LocalMapper
# ---------------------------------------------------------------------------


def test_cpu_binding_laserscan_scan_to_grid_basic():
    """Laserscan overload on the CPU mapper; same shape / dtype invariants
    as the GPU test so signature drift between the two is caught.
    Uses the basic (non-Bayesian) LocalMapper ctor exposed in
    bindings_mapping.cpp:23-30."""
    grid_height = 40
    grid_width = 40
    n = 180
    mapper = LocalMapperCpp(
        grid_height=grid_height,
        grid_width=grid_width,
        resolution=0.05,
        laserscan_position=np.array([0.0, 0.0, 0.0], dtype=np.float32),
        laserscan_orientation=0.0,
        is_pointcloud=False,
        scan_size=n,
        angle_step=float(2.0 * np.pi / n),
        max_height=2.0,
        min_height=0.0,
        range_max=10.0,
    )

    angles = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    ranges = np.full(n, 0.5, dtype=np.float64)

    grid = mapper.scan_to_grid(angles=angles, ranges=ranges)
    grid_np = np.asarray(grid)

    assert grid_np.shape == (grid_height, grid_width)
    assert grid_np.dtype == np.int32

    n_occ, n_empty, n_unknown = _occupancy_counts(grid_np)
    assert n_occ + n_empty + n_unknown == grid_np.size
    assert n_occ > 0


def test_cpu_binding_pointcloud_scan_to_grid_basic():
    """The pointcloud overload on the CPU mapper: raw bytes + offsets in,
    valid occupancy grid out. Mirrors the GPU test above — this overload
    previously had no CPU-side coverage."""
    grid_height = 40
    grid_width = 40
    n_rays = 360
    mapper = LocalMapperCpp(
        grid_height=grid_height,
        grid_width=grid_width,
        resolution=0.05,
        laserscan_position=np.array([0.0, 0.0, 0.0], dtype=np.float32),
        laserscan_orientation=0.0,
        is_pointcloud=True,
        scan_size=n_rays,
        angle_step=float(2.0 * np.pi / n_rays),
        max_height=1.5,
        min_height=-0.5,
        range_max=5.0,
    )

    cloud_bytes = _ring_cloud(n=200, radius=0.5, z=0.1)
    num_points = cloud_bytes.size // _PC_STRIDE

    grid = mapper.scan_to_grid(
        data=cloud_bytes,
        point_step=_PC_STRIDE,
        row_step=num_points * _PC_STRIDE,
        height=1,
        width=num_points,
        x_offset=0,
        y_offset=4,
        z_offset=8,
    )
    grid_np = np.asarray(grid)

    assert grid_np.shape == (grid_height, grid_width)
    assert grid_np.dtype == np.int32

    n_occ, n_empty, n_unknown = _occupancy_counts(grid_np)
    assert n_occ + n_empty + n_unknown == grid_np.size
    assert n_occ > 0, "ring obstacles must mark occupied cells"


def test_cpu_binding_laserscan_scan_to_grid_baysian_returns_tuple():
    """Regression test: the laserscan `scan_to_grid_baysian` binding used to
    point at plain `scanToGrid` (same argument list, wrong overload_cast
    target), so it returned a single matrix instead of the
    (occupancy, probability) tuple the Python wrapper unpacks."""
    grid_height = 40
    grid_width = 40
    n = 180
    mapper = LocalMapperCpp(
        grid_height=grid_height,
        grid_width=grid_width,
        resolution=0.05,
        laserscan_position=np.array([0.0, 0.0, 0.0], dtype=np.float32),
        laserscan_orientation=0.0,
        is_pointcloud=False,
        scan_size=n,
        p_prior=0.6,
        p_occupied=0.9,
        p_empty=0.1,
        range_sure=0.1,
        range_max=10.0,
        wall_size=0.1,
        angle_step=float(2.0 * np.pi / n),
        max_height=2.0,
        min_height=0.0,
        max_points_per_line=32,
    )

    angles = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    ranges = np.full(n, 0.5, dtype=np.float64)

    result = mapper.scan_to_grid_baysian(angles=angles, ranges=ranges)

    assert isinstance(result, tuple) and len(result) == 2
    occupancy, probability = result
    occupancy = np.asarray(occupancy)
    probability = np.asarray(probability)

    assert occupancy.shape == (grid_height, grid_width)
    assert occupancy.dtype == np.int32
    assert probability.shape == (grid_height, grid_width)
    assert np.issubdtype(probability.dtype, np.floating)
    # Probabilities must be a valid distribution over cells
    assert float(probability.min()) >= 0.0
    assert float(probability.max()) <= 1.0


def test_cpu_binding_laserscan_dtypes_and_noncontiguous_agree():
    """float32 (zero-copy), float64 (convert-copy), and a non-contiguous
    slice must all be accepted and produce identical grids."""
    n = 180
    mapper = LocalMapperCpp(
        grid_height=40,
        grid_width=40,
        resolution=0.05,
        laserscan_position=np.array([0.0, 0.0, 0.0], dtype=np.float32),
        laserscan_orientation=0.0,
        is_pointcloud=False,
        scan_size=n,
        angle_step=float(2.0 * np.pi / n),
        max_height=2.0,
        min_height=0.0,
        range_max=10.0,
    )

    angles64 = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    ranges64 = np.full(n, 0.5, dtype=np.float64)
    angles32 = angles64.astype(np.float32)
    ranges32 = ranges64.astype(np.float32)
    # Non-contiguous views with the same values
    angles_nc = np.repeat(angles64, 2)[::2]
    ranges_nc = np.repeat(ranges64, 2)[::2]
    assert not angles_nc.flags["C_CONTIGUOUS"]

    grid32 = np.array(mapper.scan_to_grid(angles=angles32, ranges=ranges32))
    grid64 = np.array(mapper.scan_to_grid(angles=angles64, ranges=ranges64))
    grid_nc = np.array(mapper.scan_to_grid(angles=angles_nc, ranges=ranges_nc))

    assert np.array_equal(grid32, grid64)
    assert np.array_equal(grid32, grid_nc)


def test_cpu_binding_pointcloud_accepts_bytes():
    """A raw Python `bytes` buffer (e.g. sensor_msgs/PointCloud2.data) must
    be accepted zero-copy and produce the same grid as the uint8 array."""
    grid_height = 40
    grid_width = 40
    n_rays = 360
    mapper = LocalMapperCpp(
        grid_height=grid_height,
        grid_width=grid_width,
        resolution=0.05,
        laserscan_position=np.array([0.0, 0.0, 0.0], dtype=np.float32),
        laserscan_orientation=0.0,
        is_pointcloud=True,
        scan_size=n_rays,
        angle_step=float(2.0 * np.pi / n_rays),
        max_height=1.5,
        min_height=-0.5,
        range_max=5.0,
    )

    cloud_u8 = _ring_cloud(n=200, radius=0.5, z=0.1)
    num_points = cloud_u8.size // _PC_STRIDE
    kwargs = dict(
        point_step=_PC_STRIDE,
        row_step=num_points * _PC_STRIDE,
        height=1,
        width=num_points,
        x_offset=0,
        y_offset=4,
        z_offset=8,
    )

    grid_u8 = np.array(mapper.scan_to_grid(data=cloud_u8, **kwargs))
    grid_bytes = np.array(mapper.scan_to_grid(data=cloud_u8.tobytes(), **kwargs))

    assert np.array_equal(grid_u8, grid_bytes)
    n_occ, _, _ = _occupancy_counts(grid_u8)
    assert n_occ > 0


def _nan_padded(cloud_u8: np.ndarray, n_pad: int = 50) -> np.ndarray:
    """Append `n_pad` NaN-coordinate points (organized-cloud padding) to a
    PointCloud2-style buffer."""
    pad = np.full((n_pad, 4), np.nan, dtype=np.float32)
    return np.concatenate([cloud_u8, np.frombuffer(pad.tobytes(), dtype=np.uint8)])


def test_cpu_binding_pointcloud_nan_points_are_ignored():
    """Regression test: NaN padding points (RGBD organized clouds) used to
    slip past every filter and corrupt the bin index (UB int cast ->
    out-of-bounds write). They must be skipped, leaving the grid identical
    to the same cloud without padding."""
    def make_mapper():
        return LocalMapperCpp(
            grid_height=40,
            grid_width=40,
            resolution=0.05,
            laserscan_position=np.array([0.0, 0.0, 0.0], dtype=np.float32),
            laserscan_orientation=0.0,
            is_pointcloud=True,
            scan_size=360,
            angle_step=float(2.0 * np.pi / 360),
            max_height=1.5,
            min_height=-0.5,
            range_max=5.0,
        )

    clean = _ring_cloud(n=200, radius=0.5, z=0.1)
    padded = _nan_padded(clean)

    def kwargs(buf):
        n = buf.size // _PC_STRIDE
        return dict(
            data=buf,
            point_step=_PC_STRIDE,
            row_step=n * _PC_STRIDE,
            height=1,
            width=n,
            x_offset=0,
            y_offset=4,
            z_offset=8,
        )

    grid_clean = np.array(make_mapper().scan_to_grid(**kwargs(clean)))
    grid_padded = np.array(make_mapper().scan_to_grid(**kwargs(padded)))

    assert np.array_equal(grid_clean, grid_padded)


@pytestmark_gpu
def test_gpu_binding_pointcloud_nan_points_are_ignored():
    """Same regression as the CPU test, for the GPU conversion kernel."""
    def make_mapper():
        return LocalMapperGPU(
            grid_height=40,
            grid_width=40,
            resolution=0.05,
            laserscan_position=np.array([0.0, 0.0, 0.0], dtype=np.float32),
            laserscan_orientation=0.0,
            is_pointcloud=True,
            scan_size=360,
            angle_step=float(2.0 * np.pi / 360),
            max_height=1.5,
            min_height=-0.5,
            range_max=5.0,
        )

    clean = _ring_cloud(n=200, radius=0.5, z=0.1)
    padded = _nan_padded(clean)

    def kwargs(buf):
        n = buf.size // _PC_STRIDE
        return dict(
            data=buf,
            point_step=_PC_STRIDE,
            row_step=n * _PC_STRIDE,
            height=1,
            width=n,
            x_offset=0,
            y_offset=4,
            z_offset=8,
        )

    grid_clean = np.array(make_mapper().scan_to_grid(**kwargs(clean)))
    grid_padded = np.array(make_mapper().scan_to_grid(**kwargs(padded)))

    assert np.array_equal(grid_clean, grid_padded)
