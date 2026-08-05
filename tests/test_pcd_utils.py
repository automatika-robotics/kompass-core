"""Tests for the PCD reading utilities (`kompass_core.utils.pointcloud`).

The PCD files are generated on the fly in ``tmp_path``, so the suite commits
no binary resources for this coverage.
"""

import struct

import numpy as np
import pytest
from kompass_core.utils import get_occupancy_grid_from_pcd, get_points_from_pcd
from kompass_cpp.mapping import OCCUPANCY_TYPE

XYZ = np.array(
    [
        [0.0, 0.0, 1.0],
        [2.5, 0.0, 0.1],
        [0.0, 2.5, 2.0],
        [2.5, 2.5, 0.5],
    ],
    dtype=np.float32,
)


def _header(fields: str, n: int, data_format: str) -> str:
    per_field = {"x y z": 3, "intensity x y z": 4}[fields]
    return (
        "\n".join(
            [
                "# .PCD v0.7 - Point Cloud Data file format",
                "VERSION 0.7",
                f"FIELDS {fields}",
                "SIZE " + " ".join(["4"] * per_field),
                "TYPE " + " ".join(["F"] * per_field),
                "COUNT " + " ".join(["1"] * per_field),
                f"WIDTH {n}",
                "HEIGHT 1",
                "VIEWPOINT 0 0 0 1 0 0 0",
                f"POINTS {n}",
                f"DATA {data_format}",
            ]
        )
        + "\n"
    )


def _write_ascii_pcd(path, points: np.ndarray) -> None:
    body = "".join(f"{x} {y} {z}\n" for x, y, z in points)
    path.write_text(_header("x y z", len(points), "ascii") + body)


def _write_binary_pcd(path, points: np.ndarray) -> None:
    """Binary PCD with a leading intensity field, so the reader's per-point
    byte offsets for x/y/z are actually exercised."""
    blob = b"".join(struct.pack("<ffff", 42.0, x, y, z) for x, y, z in points)
    path.write_bytes(
        _header("intensity x y z", len(points), "binary").encode("ascii") + blob
    )


def test_read_pcd_ascii(tmp_path):
    pcd = tmp_path / "cloud_ascii.pcd"
    _write_ascii_pcd(pcd, XYZ)

    points = get_points_from_pcd(str(pcd))

    assert points.shape == (len(XYZ), 3)
    assert np.allclose(points, XYZ)


def test_read_pcd_binary_extracts_xyz_at_offsets(tmp_path):
    pcd = tmp_path / "cloud_binary.pcd"
    _write_binary_pcd(pcd, XYZ)

    points = get_points_from_pcd(str(pcd))

    assert points.shape == (len(XYZ), 3)
    # Exact match: the same float32 bytes must come back, not the intensity
    assert np.array_equal(points, XYZ)


def test_read_pcd_missing_file_raises(tmp_path):
    with pytest.raises(RuntimeError):
        get_points_from_pcd(str(tmp_path / "does_not_exist.pcd"))


def test_occupancy_grid_from_pcd(tmp_path):
    occupied = OCCUPANCY_TYPE.OCCUPIED.value
    empty = OCCUPANCY_TYPE.EMPTY.value
    unknown = OCCUPANCY_TYPE.UNEXPLORED.value

    # A free-band point sharing cell (0, 0) with an occupied one: the cell
    # combines by max, so occupied must win
    cloud = np.vstack([XYZ, [[0.1, 0.1, 0.0]]]).astype(np.float32)
    pcd = tmp_path / "cloud_grid.pcd"
    _write_ascii_pcd(pcd, cloud)

    grid, origin = get_occupancy_grid_from_pcd(
        str(pcd), grid_resolution=1.0, z_ground_limit=0.2, robot_height=1.5
    )

    # Bounding box spans [0, 2.5] on both axes -> ceil(2.5) = 3 cells each way,
    # origin at the min corner
    assert grid.shape == (3, 3)
    assert np.allclose(np.asarray(origin), [0.0, 0.0, 0.0])

    assert grid[0, 0] == occupied  # z=1.0 in band; free point in same cell loses
    assert grid[2, 0] == empty  # z=0.1 <= ground limit
    assert grid[0, 2] == unknown  # z=2.0 above robot height
    assert grid[2, 2] == occupied  # z=0.5 in band
    # Cells no point landed in stay unknown
    assert grid[1, 1] == unknown
