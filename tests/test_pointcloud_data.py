import json
from pathlib import Path

import numpy as np
import pytest
from kompass_cpp.utils import pointcloud_to_laserscan_from_raw


RESOURCES_DIR = Path(__file__).parent / "resources" / "mapping"
LIVOX_CLOUD_JSON = RESOURCES_DIR / "livox_pointcloud_sample_0.json"


# ---------------------------------------------------------------------------
# Plotting helpers — kept for ad-hoc local debugging; not used by CI tests.
# ---------------------------------------------------------------------------


def plot_ranges_angles(angles: list, ranges: list, output_image_path: str):
    """Polar plot of a laserscan. Silently skipped if matplotlib is missing."""
    try:
        import matplotlib
        import matplotlib.pyplot as plt

        matplotlib.use("Agg")
    except ImportError:
        print(
            "Matplotlib is not installed. Test figures will not be generated. "
            "To generate figures run 'pip install matplotlib'"
        )
        return

    if len(ranges) != len(angles):
        raise ValueError("'ranges' and 'angles' must have the same length.")

    fig, ax = plt.subplots(subplot_kw={"projection": "polar"})
    ax.plot(angles, ranges, marker="o")
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    ax.set_title("Ranges vs Angles", va="bottom")
    plt.savefig(output_image_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_pointcloud_from_json(file_path: str, output_image_path: str) -> None:
    """3D scatter render of a recorded pointcloud. Silently skipped if
    plotly is missing. Used during manual debugging; CI tests build clouds
    inline so they don't hit plotly."""
    try:
        import plotly.graph_objects as go
    except ImportError:
        print(
            "Plotly is not installed. 3D pointcloud figures will not be "
            "generated. To generate figures run 'pip install plotly'"
        )
        return

    with open(file_path, "r") as f:
        pc_json = json.load(f)

    data = pc_json["data"]
    point_step = pc_json["point_step"]
    fields = pc_json["fields"]
    width = pc_json["width"]
    height = pc_json["height"]

    offset_map = {f["name"]: f["offset"] for f in fields}
    x_offset = offset_map.get("x")
    y_offset = offset_map.get("y")
    z_offset = offset_map.get("z")
    if x_offset is None or y_offset is None or z_offset is None:
        raise ValueError("JSON missing x, y, or z fields")

    buffer = np.array(data, dtype=np.uint8)
    num_points = width * height

    points = []
    for i in range(num_points):
        base = i * point_step
        x = np.frombuffer(
            buffer[base + x_offset : x_offset + base + 4], dtype=np.float32,
        )[0]
        y = np.frombuffer(
            buffer[base + y_offset : y_offset + base + 4], dtype=np.float32,
        )[0]
        z = np.frombuffer(
            buffer[base + z_offset : z_offset + base + 4], dtype=np.float32,
        )[0]
        points.append((x, y, z))

    points = np.array(points)

    fig = go.Figure(
        data=[
            go.Scatter3d(
                x=points[:, 0],
                y=points[:, 1],
                z=points[:, 2],
                mode="markers",
                marker={
                    "size": 2,
                    "color": points[:, 2],
                    "colorscale": "Viridis",
                    "opacity": 0.8,
                },
            )
        ]
    )
    fig.update_layout(
        scene={
            "xaxis_title": "X",
            "yaxis_title": "Y",
            "zaxis_title": "Z",
            "aspectmode": "data",
        },
        title="PointCloud2 3D Plot",
    )
    fig.write_html(output_image_path, include_plotlyjs="cdn")
    return pc


# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------


# PointCloud2 layout used by the tests: 16 bytes per point, x/y/z as
# float32 at offsets 0/4/8 with 4 bytes padding.
_PC_STRIDE = 16


def _make_cloud_bytes(points_xyz: np.ndarray) -> bytes:
    """Pack an Nx3 float32 array into a PointCloud2-style byte buffer."""
    assert points_xyz.ndim == 2 and points_xyz.shape[1] == 3
    n = points_xyz.shape[0]
    buf = np.zeros((n, 4), dtype=np.float32)
    buf[:, :3] = points_xyz.astype(np.float32)
    return np.frombuffer(buf.tobytes(), dtype=np.uint8)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_conversion_synthetic_ring_populates_bins():
    """A ring of 100 points at 1.0 m, all at in-range z, should populate
    nearly every angular bin with distance ≈ 1.0."""
    n = 100
    theta = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    ring = np.column_stack([np.cos(theta), np.sin(theta), np.full(n, 0.5)])
    cloud = _make_cloud_bytes(ring)

    max_range = 10.0
    angle_step = 0.05  # ~126 bins over 2π

    ranges, angles = pointcloud_to_laserscan_from_raw(
        data=cloud,
        point_step=_PC_STRIDE,
        row_step=n * _PC_STRIDE,
        height=1,
        width=n,
        x_offset=0,
        y_offset=4,
        z_offset=8,
        max_range=max_range,
        min_z=0.0,
        max_z=1.0,
        angle_step=angle_step,
    )
    # Zero-copy binding returns float32 numpy arrays (was: Python lists)
    assert isinstance(ranges, np.ndarray) and ranges.dtype == np.float32
    assert isinstance(angles, np.ndarray) and angles.dtype == np.float32
    ranges = np.asarray(ranges)
    angles = np.asarray(angles)

    # Bin count matches the CPU helper's formula.
    expected_bins = int(np.ceil(2.0 * np.pi / angle_step))
    assert ranges.shape == (expected_bins,)
    assert angles.shape == (expected_bins,)

    # Most bins should carry ~1.0 (ring radius). Allow a few gaps where two
    # points land in the same bin and leave adjacent bins empty.
    populated = int(np.count_nonzero(ranges < max_range))
    assert populated > 0.4 * expected_bins, (
        f"expected >40% bins populated, got {populated}/{expected_bins}"
    )

    # All populated bins should carry the ring radius.
    hit_ranges = ranges[ranges < max_range]
    assert np.all(np.abs(hit_ranges - 1.0) < 1e-3), (
        f"populated bins should carry ~1.0 m, got min={hit_ranges.min()} "
        f"max={hit_ranges.max()}"
    )


def test_conversion_origin_points_are_filtered():
    """Points at r < 1e-3 m (origin ± ε) must be dropped — they carry no
    direction information. Grid stays at max_range everywhere."""
    n = 50
    cloud = _make_cloud_bytes(np.zeros((n, 3), dtype=np.float32))

    max_range = 5.0
    ranges, _ = pointcloud_to_laserscan_from_raw(
        data=cloud,
        point_step=_PC_STRIDE,
        row_step=n * _PC_STRIDE,
        height=1,
        width=n,
        x_offset=0,
        y_offset=4,
        z_offset=8,
        max_range=max_range,
        min_z=-1.0,
        max_z=1.0,
        angle_step=0.1,
    )
    ranges = np.asarray(ranges)

    assert np.all(ranges == max_range), (
        "origin points should leave every bin at max_range"
    )


def test_conversion_z_filter_rejects_above_ceiling():
    """Points above max_z must be filtered out — every bin stays at
    max_range."""
    n = 40
    theta = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    # All at z=3.0, above max_z=1.0.
    above = np.column_stack([np.cos(theta), np.sin(theta), np.full(n, 3.0)])
    cloud = _make_cloud_bytes(above)

    max_range = 10.0
    ranges, _ = pointcloud_to_laserscan_from_raw(
        data=cloud,
        point_step=_PC_STRIDE,
        row_step=n * _PC_STRIDE,
        height=1,
        width=n,
        x_offset=0,
        y_offset=4,
        z_offset=8,
        max_range=max_range,
        min_z=0.0,
        max_z=1.0,
        angle_step=0.1,
    )
    ranges = np.asarray(ranges)

    assert np.all(ranges == max_range), (
        "every point is above the ceiling, expected every bin at max_range"
    )


def test_conversion_z_band_entirely_below_the_sensor_is_honoured():
    """A wholly negative band is a real band, not a request to disable the
    gate.

    A sensor mounted above the volume it guards -- a mast lidar over a shorter
    robot -- yields a band like [-1.2, -0.2] in its own frame. Reading the sign
    of max_z as "no ceiling" would let everything overhead through, which on
    the safety-stop path means braking for door frames.
    """
    n = 40
    theta = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    # Ceiling returns, above the band's upper edge of -0.2
    overhead = np.column_stack([np.cos(theta), np.sin(theta), np.full(n, 0.5)])
    # Returns inside the band, at the robot's midpoint
    in_band = np.column_stack([np.cos(theta), np.sin(theta), np.full(n, -0.7)])

    max_range = 10.0
    kwargs = dict(
        point_step=_PC_STRIDE,
        height=1,
        x_offset=0,
        y_offset=4,
        z_offset=8,
        max_range=max_range,
        min_z=-1.2,
        max_z=-0.2,
        angle_step=0.1,
    )

    ranges, _ = pointcloud_to_laserscan_from_raw(
        data=_make_cloud_bytes(overhead),
        row_step=n * _PC_STRIDE,
        width=n,
        **kwargs,
    )
    assert np.all(np.asarray(ranges) == max_range), (
        "points above the band's upper edge leaked through a negative max_z"
    )

    ranges, _ = pointcloud_to_laserscan_from_raw(
        data=_make_cloud_bytes(in_band),
        row_step=n * _PC_STRIDE,
        width=n,
        **kwargs,
    )
    assert np.any(np.asarray(ranges) < max_range), (
        "points inside the band were dropped"
    )


@pytest.mark.skipif(
    not LIVOX_CLOUD_JSON.exists() or LIVOX_CLOUD_JSON.stat().st_size < 1_000_000,
    reason=(
        "livox_pointcloud_sample_0.json not committed (too large for CI). "
        "Drop the file into tests/resources/mapping/ to enable this test."
    ),
)
def test_conversion_livox_recording_produces_nontrivial_output():
    """A real Livox frame should land distances in a reasonable fraction
    of the angular bins (exact shape is data-dependent; we check that
    the call doesn't silently return all-max_range)."""
    pc_json = json.loads(LIVOX_CLOUD_JSON.read_text())
    offset_map = {f["name"]: f["offset"] for f in pc_json["fields"]}
    data = np.array(pc_json["data"]).astype(np.uint8)

    max_range = 20.0
    ranges, angles = pointcloud_to_laserscan_from_raw(
        data=data,
        point_step=pc_json["point_step"],
        row_step=pc_json["row_step"],
        height=pc_json["height"],
        width=pc_json["width"],
        x_offset=offset_map["x"],
        y_offset=offset_map["y"],
        z_offset=offset_map["z"],
        max_range=max_range,
        min_z=1.6,
        max_z=1.8,
        angle_step=0.05,
    )
    ranges = np.asarray(ranges)
    angles = np.asarray(angles)

    expected_bins = int(np.ceil(2.0 * np.pi / 0.05))
    assert ranges.shape == (expected_bins,)
    assert angles.shape == (expected_bins,)

    populated = int(np.count_nonzero(ranges < max_range))
    assert populated > 10, (
        f"livox frame at z∈[1.6, 1.8] m should populate >10 bins, got "
        f"{populated}"
    )


def test_conversion_row_padding_is_not_decoded_as_points():
    """Trailing row padding in an organized cloud must never become points.

    The walk over a row is bounded by ``width`` points, not ``row_step``
    (mirroring the GPU kernel): padded rows occur in real organized clouds,
    and padding bytes are arbitrary — here they are deliberately crafted to
    decode as a finite, in-band obstacle at 1 m straight ahead, which the
    old row_step-driven loop would have turned into a phantom obstacle.
    """
    max_range = 10.0
    angle_step = 0.1
    width, height = 3, 2

    # Real points: row 0 at bearing pi/2, row 1 at bearing pi, both at 5 m
    row0 = np.array([[0.0, 5.0, 0.0]] * width, dtype=np.float32)
    row1 = np.array([[-5.0, 0.0, 0.0]] * width, dtype=np.float32)

    # Padding: one extra point-slot per row, decoding as (1, 0, 0) — finite,
    # non-zero, inside the z band. Must be skipped, not read as a point.
    phantom = np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32).tobytes()

    def _row(points: np.ndarray) -> bytes:
        buf = np.zeros((width, 4), dtype=np.float32)
        buf[:, :3] = points
        return buf.tobytes() + phantom

    data = _row(row0) + _row(row1)
    row_step = width * _PC_STRIDE + len(phantom)
    assert len(data) == height * row_step

    ranges, _ = pointcloud_to_laserscan_from_raw(
        data=np.frombuffer(data, dtype=np.uint8),
        point_step=_PC_STRIDE,
        row_step=row_step,
        height=height,
        width=width,
        x_offset=0,
        y_offset=4,
        z_offset=8,
        max_range=max_range,
        min_z=-1.0,
        max_z=1.0,
        angle_step=angle_step,
    )
    ranges = np.asarray(ranges)

    phantom_bin = 0  # bearing 0 -> first bin
    assert ranges[phantom_bin] == max_range, (
        "row padding bytes were decoded as a phantom obstacle"
    )

    # Real points from BOTH rows must still land (row addressing intact)
    bin_row0 = int((np.pi / 2) / angle_step)
    bin_row1 = int(np.pi / angle_step)
    assert ranges[bin_row0] < max_range, "row 0 points were dropped"
    assert ranges[bin_row1] < max_range, "row 1 points were dropped"


def test_conversion_rejects_negative_offsets():
    """Negative field offsets are malformed metadata and must raise.

    They used to slip the per-point bounds check (which takes the max of the
    three offsets) and wrap the unsigned index math into an out-of-bounds
    read. Both overloads carry the guard.
    """
    n = 8
    ring = np.column_stack([
        np.ones(n), np.zeros(n), np.zeros(n)
    ]).astype(np.float32)
    kwargs = dict(
        data=_make_cloud_bytes(ring),
        point_step=_PC_STRIDE,
        row_step=n * _PC_STRIDE,
        height=1,
        width=n,
        y_offset=4,
        z_offset=8,
        max_range=10.0,
        min_z=-1.0,
        max_z=1.0,
    )
    with pytest.raises(ValueError, match="non-negative"):
        pointcloud_to_laserscan_from_raw(x_offset=-4, angle_step=0.1, **kwargs)
    with pytest.raises(ValueError, match="non-negative"):
        pointcloud_to_laserscan_from_raw(x_offset=-4, num_bins=64, **kwargs)
