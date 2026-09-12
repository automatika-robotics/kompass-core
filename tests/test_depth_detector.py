import pytest
import numpy as np
from kompass_core.vision import CameraFrameConvention, DepthDetector
from kompass_core.datatypes import Bbox2D, PointsOfInterest
from kompass_cpp.types import SensorConfig

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
    cx, cy = (
        int(camera_params["principal_point"][0]),
        int(camera_params["principal_point"][1]),
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

    results = detector.compute_3d_detections(img, [center_bbox_2d], 0.0, 0.0, 0.0, 0.0)
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


# -----------------------------------------------------------------------------
# Point cloud input
# -----------------------------------------------------------------------------


def _cloud_dict(points_xyz: np.ndarray) -> dict:
    """Packs Nx3 float32 points as the PointCloud2 layout dict the mapper
    takes: 16-byte records (x, y, z at offsets 0/4/8 plus 4 bytes of padding),
    the way LiDAR drivers publish."""
    n = len(points_xyz)
    records = np.zeros((n, 4), dtype=np.float32)
    records[:, :3] = points_xyz
    return {
        "data": records.reshape(-1).view(np.uint8),
        "point_step": 16,
        "row_step": 16 * n,
        "height": 1,
        "width": n,
        "x_offset": 0,
        "y_offset": 4,
        "z_offset": 8,
    }


def _optical_points_from_depth(depth_mm: np.ndarray, camera_params: dict) -> np.ndarray:
    """Every valid pixel back-projected at its centre into the camera's
    optical frame (x right, y down, z forward), Nx3 float32."""
    fx, fy = camera_params["focal_length"]
    cx, cy = camera_params["principal_point"]
    rows, cols = np.nonzero(depth_mm)
    d = depth_mm[rows, cols].astype(np.float32) * 1e-3
    return np.stack(
        [(cols + 0.5 - cx) * d / fx, (rows + 0.5 - cy) * d / fy, d], axis=1
    ).astype(np.float32)


def _optical_to_body(points_opt: np.ndarray) -> np.ndarray:
    """REP 103: optical (x right, y down, z forward) -> body (x forward,
    y left, z up) for a camera at the body origin."""
    return np.stack([points_opt[:, 2], -points_opt[:, 0], -points_opt[:, 1]], axis=1)


@pytest.fixture
def camera_as_cloud_sensor(forward_facing_camera):
    """A cloud expressed in the camera's optical frame has the camera itself
    as its sensor."""
    return SensorConfig(
        position=forward_facing_camera["translation"],
        rotation=forward_facing_camera["rotation"],
    )


def _assert_same_boxes(from_depth, from_cloud, tol=1e-3):
    assert len(from_depth) == len(from_cloud) == 1
    assert np.allclose(from_depth[0].center, from_cloud[0].center, atol=tol)
    assert np.allclose(from_depth[0].size, from_cloud[0].size, atol=tol)


@pytest.mark.parametrize(
    "robot_state", [(0.0, 0.0, 0.0, 0.0), (10.0, 5.0, 0.7, 0.0)], ids=["body", "world"]
)
def test_point_cloud_matches_depth_image(
    detector,
    camera_params,
    synthetic_depth_image,
    center_bbox_2d,
    camera_as_cloud_sensor,
    robot_state,
):
    """The same scene as a point cloud in the camera's optical frame lifts to
    the same 3D box as the depth image."""
    from_depth = detector.compute_3d_detections(
        synthetic_depth_image, [center_bbox_2d], *robot_state
    )
    detector.set_point_cloud_sensor(camera_as_cloud_sensor)
    cloud = _cloud_dict(
        _optical_points_from_depth(synthetic_depth_image, camera_params)
    )
    from_cloud = detector.compute_3d_detections(
        **cloud,
        input=[center_bbox_2d],
        robot_x=robot_state[0],
        robot_y=robot_state[1],
        robot_yaw=robot_state[2],
        robot_speed=robot_state[3],
    )
    _assert_same_boxes(from_depth, from_cloud)


def test_point_cloud_default_sensor_is_the_body_frame(
    detector, camera_params, synthetic_depth_image, center_bbox_2d
):
    """Without set_point_cloud_sensor the cloud is taken in the robot body
    frame: the optical cloud turned into body axes lifts to the same box."""
    from_depth = detector.compute_3d_detections(
        synthetic_depth_image, [center_bbox_2d], 0.0, 0.0, 0.0, 0.0
    )
    body_points = _optical_to_body(
        _optical_points_from_depth(synthetic_depth_image, camera_params)
    )
    from_cloud = detector.compute_3d_detections(
        **_cloud_dict(body_points),
        input=[center_bbox_2d],
        robot_x=0.0,
        robot_y=0.0,
        robot_yaw=0.0,
        robot_speed=0.0,
    )
    _assert_same_boxes(from_depth, from_cloud)


def test_point_cloud_ignores_invalid_points(
    detector,
    camera_params,
    synthetic_depth_image,
    center_bbox_2d,
    camera_as_cloud_sensor,
):
    """Non-finite points, points behind the camera and points beyond the
    depth range (a wall behind the target that projects into the same box)
    do not change the result."""
    detector.set_point_cloud_sensor(camera_as_cloud_sensor)
    clean = _optical_points_from_depth(synthetic_depth_image, camera_params)
    args = {
        "input": [center_bbox_2d],
        "robot_x": 0.0,
        "robot_y": 0.0,
        "robot_yaw": 0.0,
        "robot_speed": 0.0,
    }
    from_clean = detector.compute_3d_detections(**_cloud_dict(clean), **args)

    # A wall at 20 m (beyond the 10 m range) covering the whole box, more
    # points than the target itself, plus a point behind the camera and NaNs
    wall = clean.copy()
    wall[:, 2] = 20.0
    wall[:, :2] *= 20.0 / 3.0
    behind = np.array([[0.0, 0.0, -3.0]], dtype=np.float32)
    nans = np.full((10, 3), np.nan, dtype=np.float32)
    polluted = np.concatenate([wall, behind, nans, clean, wall])
    from_polluted = detector.compute_3d_detections(**_cloud_dict(polluted), **args)
    _assert_same_boxes(from_clean, from_polluted)


def test_point_cloud_pois_match_depth_image(
    detector,
    camera_params,
    synthetic_depth_image_poi,
    center_poi,
    camera_as_cloud_sensor,
):
    from_depth = detector.compute_3d_detections(
        synthetic_depth_image_poi, center_poi, 0.0, 0.0, 0.0, 0.0
    )
    detector.set_point_cloud_sensor(camera_as_cloud_sensor)
    cloud = _cloud_dict(
        _optical_points_from_depth(synthetic_depth_image_poi, camera_params)
    )
    from_cloud = detector.compute_3d_detections(
        **cloud,
        input=center_poi,
        robot_x=0.0,
        robot_y=0.0,
        robot_yaw=0.0,
        robot_speed=0.0,
    )
    _assert_same_boxes(from_depth, from_cloud)


def test_point_cloud_lifts_through_lidar_mount_pose(detector, camera_params):
    """A LiDAR mounted away from the camera, yawed, in its own body-aligned
    frame: a cluster placed at a known body-frame location comes back there,
    with a larger out-of-range wall behind it ignored."""
    yaw = 0.2
    lidar = SensorConfig(
        position=np.array([-0.1, 0.0, 0.8], dtype=np.float32),
        rotation=np.array(
            [0.0, 0.0, np.sin(yaw / 2), np.cos(yaw / 2)], dtype=np.float32
        ),
    )
    detector.set_point_cloud_sensor(lidar)
    rot = np.array(
        [
            [np.cos(yaw), -np.sin(yaw), 0.0],
            [np.sin(yaw), np.cos(yaw), 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )

    def in_lidar(points_body: np.ndarray) -> np.ndarray:
        return (points_body - lidar.position) @ rot  # rot^T applied to each row

    cluster_center = np.array([3.0, 0.5, 0.4], dtype=np.float32)
    offsets = np.array(
        [
            [0.05 * k, 0.04 * i, 0.04 * j]
            for i in range(-5, 6)
            for j in range(-5, 6)
            for k in (-1, 0, 1)
        ],
        dtype=np.float32,
    )
    cluster = cluster_center + offsets
    wall = np.array(
        [
            [15.0, 0.5 + 0.05 * i, 0.4 + 0.05 * j]
            for i in range(-10, 11)
            for j in range(-10, 11)
        ],
        dtype=np.float32,
    )
    assert len(wall) > len(cluster)
    points = in_lidar(np.concatenate([cluster, wall]))

    # The cluster projects to u in [203, 270] and v in [140, 207] for this
    # camera (fx = fy = 500 at the body origin)
    box = Bbox2D(
        top_left_corner=np.array([200, 137], dtype=np.int32),
        size=np.array([74, 73], dtype=np.int32),
    )
    box.img_size = np.array([640, 480], dtype=np.int32)
    boxes = detector.compute_3d_detections(
        **_cloud_dict(points),
        input=[box],
        robot_x=0.0,
        robot_y=0.0,
        robot_yaw=0.0,
        robot_speed=0.0,
    )
    assert len(boxes) == 1
    assert np.allclose(boxes[0].center, cluster_center, atol=0.02)
    assert 0.0 < boxes[0].size[0] < 0.2


def test_point_cloud_accepts_bytes_and_readonly_buffers(
    detector,
    camera_params,
    synthetic_depth_image,
    center_bbox_2d,
    camera_as_cloud_sensor,
):
    """The buffer crosses as delivered: a read-only np.frombuffer view (what a
    ROS callback hands out) and plain bytes are both accepted; a dict missing
    a layout field is rejected."""
    detector.set_point_cloud_sensor(camera_as_cloud_sensor)
    cloud = _cloud_dict(
        _optical_points_from_depth(synthetic_depth_image, camera_params)
    )
    args = {
        "input": [center_bbox_2d],
        "robot_x": 0.0,
        "robot_y": 0.0,
        "robot_yaw": 0.0,
        "robot_speed": 0.0,
    }
    reference = detector.compute_3d_detections(**cloud, **args)

    raw = cloud["data"].tobytes()
    readonly = np.frombuffer(raw, dtype=np.uint8)
    assert not readonly.flags.writeable
    _assert_same_boxes(
        reference,
        detector.compute_3d_detections(**{**cloud, "data": readonly}, **args),
    )
    _assert_same_boxes(
        reference,
        detector.compute_3d_detections(**{**cloud, "data": raw}, **args),
    )

    broken = {k: v for k, v in cloud.items() if k != "z_offset"}
    with pytest.raises(TypeError):
        detector.compute_3d_detections(**broken, **args)


# -----------------------------------------------------------------------------
# Provenance of a lifted box: which input it came from, how much depth it rests on
# -----------------------------------------------------------------------------


def _box_over_the_void(center_bbox_2d):
    """A box on pixels that carry no depth, which the detector drops."""
    box = Bbox2D()
    box.top_left_corner = np.array([0, 0], dtype=np.int32)
    box.size = np.array([50, 50], dtype=np.int32)
    box.img_size = center_bbox_2d.img_size
    return box


def test_boxes_keep_their_source_index_when_others_are_dropped(
    detector, synthetic_depth_image, center_bbox_2d
):
    """The output is not positional, so a survivor must say which input it
    was lifted from."""
    results = detector.compute_3d_detections(
        synthetic_depth_image,
        [_box_over_the_void(center_bbox_2d), center_bbox_2d],
        0.0, 0.0, 0.0, 0.0,
    )
    assert [box.source_index for box in results] == [1]


def test_sample_count_is_the_usable_depth_inside_the_box(
    detector, synthetic_depth_image, center_bbox_2d
):
    """The synthetic scene carries depth exactly inside the box, so every
    pixel counts; a box half over the void counts half."""
    (box,) = detector.compute_3d_detections(
        synthetic_depth_image, [center_bbox_2d], 0.0, 0.0, 0.0, 0.0
    )
    w, h = center_bbox_2d.size
    assert box.sample_count == w * h

    half = Bbox2D()
    half.top_left_corner = center_bbox_2d.top_left_corner - np.array(
        [w // 2, 0], dtype=np.int32
    )
    half.size = center_bbox_2d.size
    half.img_size = center_bbox_2d.img_size
    (box,) = detector.compute_3d_detections(
        synthetic_depth_image, [half], 0.0, 0.0, 0.0, 0.0
    )
    # the box limits are inclusive, which may add one column
    assert box.sample_count == pytest.approx(w * h / 2, abs=h)


def test_poi_boxes_carry_provenance(detector, synthetic_depth_image_poi, center_poi):
    (box,) = detector.compute_3d_detections(
        synthetic_depth_image_poi, center_poi, 0.0, 0.0, 0.0, 0.0
    )
    assert box.source_index == 0
    assert box.sample_count > 1


def test_point_cloud_boxes_carry_provenance(
    detector, camera_params, camera_as_cloud_sensor, synthetic_depth_image, center_bbox_2d
):
    """Every back-projected point of the scene lands in the box, so the
    count is the point count, and the index survives a dropped box."""
    detector.set_point_cloud_sensor(camera_as_cloud_sensor)
    points = _optical_points_from_depth(synthetic_depth_image, camera_params)
    (box,) = detector.compute_3d_detections(
        **_cloud_dict(points),
        input=[_box_over_the_void(center_bbox_2d), center_bbox_2d],
        robot_x=0.0, robot_y=0.0, robot_yaw=0.0, robot_speed=0.0,
    )
    assert box.source_index == 1
    assert box.sample_count == len(points)
