import pytest
import numpy as np
from kompass_core.vision import CameraFrameConvention, DepthDetector
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


#: REP 103 rotation from optical axes (x right, y down, z forward) to body
#: ones (x forward, y left, z up), as (x, y, z, w).
OPTICAL_TO_BODY = np.array([-0.5, 0.5, -0.5, 0.5], dtype=np.float32)


@pytest.fixture
def forward_facing_camera():
    """A camera at the body origin looking straight ahead.

    Poses are read in the optical convention by default -- the frame a ROS
    Image or CameraInfo names in its header -- so a camera whose view direction
    is the body's forward axis carries the fixed REP 103 turn as its rotation.
    """
    return {
        "translation": np.array([0.0, 0.0, 0.0], dtype=np.float32),
        "rotation": OPTICAL_TO_BODY,
    }


@pytest.fixture
def detector(camera_params, forward_facing_camera):
    return DepthDetector(
        camera_params["depth_range"],
        forward_facing_camera["translation"],
        forward_facing_camera["rotation"],
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
    # C-contiguous, the native layout of every camera driver: this is the
    # zero-copy fast path through the bindings
    img = np.zeros((h, w), dtype=np.uint16)

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
    img = np.zeros((h, w), dtype=np.uint16)

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
    img = np.zeros((img_h, img_w), dtype=np.uint16)
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


def test_compute_3d_float32_metres_matches_uint16_mm(
    detector, synthetic_depth_image, center_bbox_2d
):
    """A float32 depth image in METRES must produce the same 3D detection as
    its uint16-millimetres twin: the binding dispatches on the dtype and no
    conversion pass is needed anywhere (zero-copy for both encodings)."""
    img_f32 = synthetic_depth_image.astype(np.float32) * 1e-3  # mm -> m
    assert img_f32.flags.c_contiguous

    result_u16 = detector.compute_3d_detections(
        synthetic_depth_image, [center_bbox_2d], 0.0, 0.0, 0.0, 0.0
    )
    result_f32 = detector.compute_3d_detections(
        img_f32, [center_bbox_2d], 0.0, 0.0, 0.0, 0.0
    )

    assert len(result_u16) == 1 and len(result_f32) == 1
    for axis in range(3):
        assert result_f32[0].center[axis] == pytest.approx(
            result_u16[0].center[axis], abs=1e-5
        )
        assert result_f32[0].size[axis] == pytest.approx(
            result_u16[0].size[axis], abs=1e-5
        )


def test_compute_3d_float32_nan_pixels_are_rejected(
    detector, camera_params, center_bbox_2d
):
    """NaN padding in float depth images dies in the min/max range gate: a
    box whose region is all-NaN yields no detection instead of garbage."""
    h, w = camera_params["img_shape"]
    img = np.full((h, w), np.nan, dtype=np.float32)

    results = detector.compute_3d_detections(
        img, [center_bbox_2d], 0.0, 0.0, 0.0, 0.0
    )
    assert len(results) == 0


# -----------------------------------------------------------------------------
# Camera frame convention
# -----------------------------------------------------------------------------


def test_optical_pose_places_a_target_dead_ahead_on_the_forward_axis(
    camera_params, center_bbox_2d, synthetic_depth_image
):
    """A camera pose taken straight from a ROS TF lookup must put a target
    centred in the image in front of the robot, not off to one side.

    Regression: the pose was read as body-aligned while the projection had
    already turned the optical axes into body ones, so the REP 103 quarter
    rotation landed twice. A person standing dead ahead came back ~90 degrees
    to the right, which drove the vision follower to turn away and lose them.
    """
    # A real mount: 0.32 m ahead of base_link, 0.30 m up, tilted ~23 deg down.
    # This is the optical frame's pose, which is what TF resolves for the
    # frame_id a depth CameraInfo carries.
    translation = np.array([0.32, 0.0, 0.30], dtype=np.float32)
    rotation = np.array(
        [-0.5839669, 0.5936764, -0.38588133, 0.3970222], dtype=np.float32
    )

    detector = DepthDetector(
        camera_params["depth_range"],
        translation,
        rotation,
        camera_params["focal_length"],
        camera_params["principal_point"],
        1e-3,
        CameraFrameConvention.OPTICAL,
    )
    box = detector.compute_3d_detections(
        synthetic_depth_image, [center_bbox_2d], 0.0, 0.0, 0.0, 0.0
    )[0]

    bearing = np.arctan2(box.center[1], box.center[0])
    assert abs(bearing) < np.deg2rad(2.0), (
        f"target centred in the image came back at {np.rad2deg(bearing):.1f} deg"
    )
    # Forward of the robot, and further than the camera itself
    assert box.center[0] > translation[0]


def test_body_aligned_pose_is_still_honoured(
    camera_params, center_bbox_2d, synthetic_depth_image
):
    """Callers that already turned the pose into robot axes can say so and
    keep the previous behaviour."""
    detector = DepthDetector(
        camera_params["depth_range"],
        np.array([0.0, 0.0, 0.0], dtype=np.float32),
        np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32),  # identity: FLU mount
        camera_params["focal_length"],
        camera_params["principal_point"],
        1e-3,
        CameraFrameConvention.BODY_ALIGNED,
    )
    box = detector.compute_3d_detections(
        synthetic_depth_image, [center_bbox_2d], 0.0, 0.0, 0.0, 0.0
    )[0]

    assert box.center[0] == pytest.approx(3.0, abs=0.05)
    assert box.center[1] == pytest.approx(0.0, abs=0.05)
    assert box.center[2] == pytest.approx(0.0, abs=0.05)


def test_the_two_conventions_differ_by_the_rep103_turn(
    camera_params, center_bbox_2d, synthetic_depth_image
):
    """Reading an optical pose as body-aligned is exactly the bug: the result
    is the correct one rotated by the REP 103 quarter turn."""
    translation = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    rotation = OPTICAL_TO_BODY

    def centre(convention):
        detector = DepthDetector(
            camera_params["depth_range"],
            translation,
            rotation,
            camera_params["focal_length"],
            camera_params["principal_point"],
            1e-3,
            convention,
        )
        return np.asarray(
            detector.compute_3d_detections(
                synthetic_depth_image, [center_bbox_2d], 0.0, 0.0, 0.0, 0.0
            )[0].center
        )

    # Optical: the turn is applied once -> depth lands on forward
    assert centre(CameraFrameConvention.OPTICAL)[0] == pytest.approx(3.0, abs=0.05)
    # Body-aligned: applied twice -> depth lands on -y, a clean 90 deg off
    doubled = centre(CameraFrameConvention.BODY_ALIGNED)
    assert doubled[1] == pytest.approx(-3.0, abs=0.05)
