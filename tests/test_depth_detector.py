import pytest
import numpy as np
from kompass_core.vision import DepthDetector
from kompass_core.datatypes import Bbox2D, PointsOfInterest

# -----------------------------------------------------------------------------
# Fixtures
# -----------------------------------------------------------------------------


@pytest.fixture
def camera_params():
    return {
        "depth_range": np.array([0.1, 10.0], dtype=np.float32),
        "focal_length": np.array([500.0, 500.0], dtype=np.float32),
        "principal_point": np.array([320.0, 240.0], dtype=np.float32),
        "img_shape": (480, 640),
    }


@pytest.fixture
def identity_transform():
    return {
        "translation": np.array([0.0, 0.0, 0.0], dtype=np.float32),
        "rotation": np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32),
    }


@pytest.fixture
def detector(camera_params, identity_transform):
    return DepthDetector(
        camera_params["depth_range"],
        identity_transform["translation"],
        identity_transform["rotation"],
        camera_params["focal_length"],
        camera_params["principal_point"],
        1e-3,
    )


@pytest.fixture
def center_bbox_2d(camera_params):
    cx, cy = camera_params["principal_point"]
    box_w, box_h = 100, 100
    img_h, img_w = camera_params["img_shape"]

    box = Bbox2D()
    box.top_left_corner = np.array(
        [int(cx - box_w / 2), int(cy - box_h / 2)], dtype=np.int32
    )
    box.size = np.array([box_w, box_h], dtype=np.int32)
    box.img_size = np.array([img_w, img_h], dtype=np.int32)

    return box


@pytest.fixture
def synthetic_depth_image(camera_params, center_bbox_2d):
    """
    Creates a depth image where the background is 0 (invalid depth)
    and the area exactly inside the 2D bounding box is 3000mm (3.0 meters).
    """
    h, w = camera_params["img_shape"]
    # Force column-major layout to match Eigen's default MatrixX layout
    img = np.zeros((h, w), dtype=np.uint16, order="F")

    tl_x, tl_y = center_bbox_2d.top_left_corner
    w_box, h_box = center_bbox_2d.size

    # Fill the exact region of the box with 3000mm (3 meters)
    img[tl_y : tl_y + h_box, tl_x : tl_x + w_box] = 3000

    return img


# -----------------------------------------------------------------------------
# Tests
# -----------------------------------------------------------------------------


def test_compute_3d_robot_frame(detector, synthetic_depth_image, center_bbox_2d):
    """
    Test detection effectively in Robot Frame by passing 0.0 state.
    """
    results = detector.compute_3d_detections(
        synthetic_depth_image, [center_bbox_2d], 0.0, 0.0, 0.0, 0.0
    )

    assert results is not None
    assert len(results) == 1

    box3d = results[0]

    # Body frame is FLU: depth (3.0m) is forward (X), box centered on the
    # principal point -> Y, Z ~ 0.0
    assert box3d.center[0] == pytest.approx(3.0, abs=0.05)
    assert box3d.center[1] == pytest.approx(0.0, abs=0.05)
    assert box3d.center[2] == pytest.approx(0.0, abs=0.05)


def test_compute_3d_world_frame(detector, synthetic_depth_image, center_bbox_2d):
    """
    Test detection in World Frame using explicit float inputs.
    """
    # Robot at X=10.0, Y=5.0
    rx, ry, ryaw, rspeed = 10.0, 5.0, 0.0, 0.0

    results = detector.compute_3d_detections(
        synthetic_depth_image, [center_bbox_2d], rx, ry, ryaw, rspeed
    )

    assert results is not None
    box3d = results[0]

    # Robot at (10, 5) with yaw=0; target 3m straight ahead in body frame
    # -> world position (13, 5, 0).
    assert box3d.center[0] == pytest.approx(13.0, abs=0.1)
    assert box3d.center[1] == pytest.approx(5.0, abs=0.1)
    assert box3d.center[2] == pytest.approx(0.0, abs=0.1)


def test_empty_input(detector, synthetic_depth_image):
    results = detector.compute_3d_detections(
        synthetic_depth_image, [], 0.0, 0.0, 0.0, 0.0
    )
    if results is not None:
        assert len(results) == 0


# -----------------------------------------------------------------------------
# PointsOfInterest Tests
# -----------------------------------------------------------------------------


@pytest.fixture
def center_poi(camera_params):
    cx, cy = camera_params["principal_point"]
    img_h, img_w = camera_params["img_shape"]
    center = np.array([int(cx), int(cy)], dtype=np.int32)
    return PointsOfInterest(
        points=[center],
        img_size=np.array([img_w, img_h], dtype=np.int32),
    )


@pytest.fixture
def synthetic_depth_image_poi(camera_params, center_poi):
    """
    Creates a depth image where a region around the POI points is filled
    with 3000mm (3.0 meters). The region is large enough to cover the
    bounding box that Bbox2D(PointsOfInterest) computes.
    """
    h, w = camera_params["img_shape"]
    img = np.zeros((h, w), dtype=np.uint16, order="F")

    # Use the first point as reference for the fill region
    px, py_ = center_poi.points_2d[0]
    margin = max(w, h) // 4
    x0 = max(0, px - margin)
    y0 = max(0, py_ - margin)
    x1 = min(w, px + margin)
    y1 = min(h, py_ + margin)
    img[y0:y1, x0:x1] = 3000

    return img


def test_poi_compute_3d_robot_frame(detector, synthetic_depth_image_poi, center_poi):
    """
    Test detection from PointsOfInterest in Robot Frame.
    """
    results = detector.compute_3d_detections(
        synthetic_depth_image_poi, center_poi, 0.0, 0.0, 0.0, 0.0
    )

    assert results is not None
    assert len(results) == 1

    box3d = results[0]

    # Body frame is FLU: depth (3.0m) is forward (X), POI on principal point.
    assert box3d.center[0] == pytest.approx(3.0, abs=0.05)
    assert box3d.center[1] == pytest.approx(0.0, abs=0.05)
    assert box3d.center[2] == pytest.approx(0.0, abs=0.05)


def test_poi_compute_3d_world_frame(detector, synthetic_depth_image_poi, center_poi):
    """
    Test detection from PointsOfInterest in World Frame.
    """
    rx, ry, ryaw, rspeed = 10.0, 5.0, 0.0, 0.0

    results = detector.compute_3d_detections(
        synthetic_depth_image_poi, center_poi, rx, ry, ryaw, rspeed
    )

    assert results is not None
    box3d = results[0]

    # Robot at (10, 5) with yaw=0; target 3m forward in body frame
    # -> world position (13, 5, 0).
    assert box3d.center[0] == pytest.approx(13.0, abs=0.1)
    assert box3d.center[1] == pytest.approx(5.0, abs=0.1)
    assert box3d.center[2] == pytest.approx(0.0, abs=0.1)


def test_poi_multipoint_robot_frame(detector, camera_params):
    """
    Test detection from PointsOfInterest with multiple scattered points.
    The MAD-based bounding box should encompass the spread and the 3D
    center should correspond to the median of the cluster.
    """
    cx, cy = int(camera_params["principal_point"][0]), int(
        camera_params["principal_point"][1]
    )
    img_h, img_w = camera_params["img_shape"]

    # Spread points around the principal point
    points = [
        np.array([cx - 30, cy - 20], dtype=np.int32),
        np.array([cx - 10, cy - 10], dtype=np.int32),
        np.array([cx, cy], dtype=np.int32),
        np.array([cx + 10, cy + 10], dtype=np.int32),
        np.array([cx + 30, cy + 20], dtype=np.int32),
    ]

    poi = PointsOfInterest(
        points=points,
        img_size=np.array([img_w, img_h], dtype=np.int32),
    )

    # Build depth image filling a generous region around the cluster
    img = np.zeros((img_h, img_w), dtype=np.uint16, order="F")
    margin = 80
    img[cy - margin : cy + margin, cx - margin : cx + margin] = 3000

    results = detector.compute_3d_detections(img, poi, 0.0, 0.0, 0.0, 0.0)

    assert results is not None
    assert len(results) == 1

    box3d = results[0]

    # Body frame is FLU: 3.0m forward (X), median of symmetric cluster on the
    # principal point -> Y, Z ~ 0.0.
    assert box3d.center[0] == pytest.approx(3.0, abs=0.05)
    assert box3d.center[1] == pytest.approx(0.0, abs=0.1)
    assert box3d.center[2] == pytest.approx(0.0, abs=0.1)
