"""Zero-copy guard for the batched cloud entries.

The batched ``clouds=[...]`` bindings extract METADATA ONLY under the GIL
(8 ints + a buffer view per cloud); the point bytes must never be copied
or converted at the Python boundary. This guard times the batched N=1
entry against the legacy single-cloud entry on the same 100k-point buffer:
they run the identical C++ path, so any per-point boundary work (which
would be 10-100x the metadata cost) blows the ratio immediately.

CPU classes are used on purpose: no JIT warm-up noise, and the binding
boundary under test is the same code for the GPU classes.

The vision guards use image-size scaling instead: the depth detector and
follower only read pixels inside the (fixed-size) target box, so a
zero-copy boundary costs the same on a VGA image and on one with 4x the
pixels. Any hidden full-image copy or dtype/layout conversion scales with
the image area and blows the ratio.
"""

import itertools
import time

import numpy as np
import pytest

from kompass_cpp.mapping import LocalMapper as LocalMapperCpp
from kompass_cpp.types import RobotGeometry, SensorConfig, SensorInputType
from kompass_cpp.utils import CriticalZoneChecker

from kompass_core.control import VisionRGBDFollower, VisionRGBDFollowerConfig
from kompass_core.datatypes import Bbox2D
from kompass_core.models import (
    AngularCtrlLimits,
    LinearCtrlLimits,
    Robot,
    RobotCtrlLimits,
    RobotGeometry as CoreRobotGeometry,
    RobotState,
    RobotType,
)
from kompass_core.vision import DepthDetector

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


# ---------------------------------------------------------------------------
# Vision depth boundary
# ---------------------------------------------------------------------------

_BOX_SIZE = 40  # px, target box shared by both image sizes
_BOX_CORNER = (300, 220)  # inside VGA and the 4x image alike


def _depth_image(rows: int, cols: int) -> np.ndarray:
    """C-contiguous uint16 depth (the zero-copy fast path): background 0
    (invalid), the target box filled with 3000mm."""
    img = np.zeros((rows, cols), dtype=np.uint16)
    x0, y0 = _BOX_CORNER
    img[y0 : y0 + _BOX_SIZE, x0 : x0 + _BOX_SIZE] = 3000
    return img


def _target_box(img_w: int, img_h: int) -> Bbox2D:
    box = Bbox2D(
        top_left_corner=np.array(_BOX_CORNER, dtype=np.int32),
        size=np.array([_BOX_SIZE, _BOX_SIZE], dtype=np.int32),
    )
    box.set_img_size(np.array([img_w, img_h], dtype=np.int32))
    return box


def test_depth_detector_boundary_does_not_scale_with_image_size():
    detector = DepthDetector(
        np.array([0.1, 10.0], dtype=np.float32),  # depth range
        np.array([0.0, 0.0, 0.0], dtype=np.float32),  # camera translation
        np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32),  # camera rotation
        np.array([500.0, 500.0], dtype=np.float32),  # focal length
        np.array([320.0, 240.0], dtype=np.float32),  # principal point
        1e-3,  # mm -> m
    )
    small = _depth_image(480, 640)
    big = _depth_image(960, 1280)  # 4x the pixels, same box content

    # Same box pixels -> identical 3D result regardless of image size, and
    # a read-only buffer (np.frombuffer-style input) must be accepted as-is
    res_small = detector.compute_3d_detections(
        small, [_target_box(640, 480)], 0.0, 0.0, 0.0, 0.0
    )
    res_big = detector.compute_3d_detections(
        big, [_target_box(1280, 960)], 0.0, 0.0, 0.0, 0.0
    )
    np.testing.assert_allclose(res_small[0].center, res_big[0].center)
    read_only = small.copy()
    read_only.setflags(write=False)
    res_ro = detector.compute_3d_detections(
        read_only, [_target_box(640, 480)], 0.0, 0.0, 0.0, 0.0
    )
    np.testing.assert_allclose(res_ro[0].center, res_small[0].center)

    small_box, big_box = _target_box(640, 480), _target_box(1280, 960)
    ratio = _median_ratio(
        lambda: detector.compute_3d_detections(small, [small_box], 0.0, 0.0, 0.0, 0.0),
        lambda: detector.compute_3d_detections(big, [big_box], 0.0, 0.0, 0.0, 0.0),
    )
    assert ratio < _MAX_RATIO, (
        f"compute_3d_detections on a 4x-pixel image costs {ratio:.2f}x the "
        "VGA call: the depth boundary is doing per-pixel (full image) work"
    )


def _tracking_depth_image(rows: int, cols: int) -> np.ndarray:
    """Materialized (non-lazy) pages so a copy regression costs a real
    memcpy: 60m background (rejected by the 10m range gate), 3m target box."""
    img = np.full((rows, cols), 60000, dtype=np.uint16)
    x0, y0 = _BOX_CORNER
    img[y0 : y0 + _BOX_SIZE, x0 : x0 + _BOX_SIZE] = 3000
    return img


def test_rgbd_follower_boundary_does_not_scale_with_image_size():
    # NOTE: the timed calls run the REAL tracking path (a detection per
    # call, advancing timestamps). The no-detections wait path is not a
    # usable timing subject: its internal state machine alternates between
    # a cheap and an expensive tick, which interleaved A/B timing folds
    # into a bogus ratio.
    robot = Robot(
        robot_type=RobotType.DIFFERENTIAL_DRIVE,
        geometry_type=CoreRobotGeometry.Type.CYLINDER,
        geometry_params=np.array([0.1, 0.4]),
    )
    ctrl_limits = RobotCtrlLimits(
        vx_limits=LinearCtrlLimits(max_vel=1.5, max_acc=3.0, max_decel=3.0),
        omega_limits=AngularCtrlLimits(
            max_omega=2.5, max_acc=2.5, max_decel=2.5, max_ang=np.pi / 2
        ),
    )
    follower = VisionRGBDFollower(
        robot=robot,
        ctrl_limits=ctrl_limits,
        config=VisionRGBDFollowerConfig(
            control_time_step=0.1,
            _use_local_coordinates=True,
            depth_conversion_factor=1e-3,
            min_depth=0.1,
            max_depth=10.0,
        ),
        camera_focal_length=[500.0, 500.0],
        camera_principal_point=[320.0, 240.0],
    )
    small = _tracking_depth_image(480, 640)
    big = _tracking_depth_image(960, 1280)  # 4x the pixels, same box
    state = RobotState(x=0.0, y=0.0, yaw=0.0, speed=0.0)

    x0, y0 = _BOX_CORNER
    assert follower.set_initial_tracking_image(
        current_state=state,
        pose_x_img=x0 + _BOX_SIZE // 2,
        pose_y_img=y0 + _BOX_SIZE // 2,
        detected_boxes=[_target_box(640, 480)],
        depth_image=small,
    )

    clock = itertools.count(1)

    def detection(img_w: int, img_h: int) -> list:
        # Fresh timestamp per call: the tracker needs dt > 0
        box = _target_box(img_w, img_h)
        box.timestamp = next(clock) * 0.1
        return [box]

    ratio = _median_ratio(
        lambda: follower.loop_step(
            current_state=state,
            detections_2d=detection(640, 480),
            depth_image=small,
        ),
        lambda: follower.loop_step(
            current_state=state,
            detections_2d=detection(1280, 960),
            depth_image=big,
        ),
    )
    assert ratio < _MAX_RATIO, (
        f"loop_step with a 4x-pixel depth image costs {ratio:.2f}x the VGA "
        "call: the depth boundary is doing per-pixel (full image) work"
    )


def test_rgbd_follower_point_cloud_accepts_readonly_buffer():
    # NOTE: there is deliberately no timing-ratio guard for the point-cloud
    # path. Its cost is O(points) by design (every point is projected into
    # the image), so a copy of the buffer is not separable from the work
    # itself by scaling the input the way the depth-image guards do. What the
    # boundary must guarantee is that the sensor's buffer crosses as
    # delivered: a ROS callback hands out a read-only np.frombuffer view.
    follower = VisionRGBDFollower(
        robot=Robot(
            robot_type=RobotType.DIFFERENTIAL_DRIVE,
            geometry_type=CoreRobotGeometry.Type.CYLINDER,
            geometry_params=np.array([0.1, 0.4]),
        ),
        ctrl_limits=RobotCtrlLimits(
            vx_limits=LinearCtrlLimits(max_vel=1.5, max_acc=3.0, max_decel=3.0),
            omega_limits=AngularCtrlLimits(
                max_omega=2.5, max_acc=2.5, max_decel=2.5, max_ang=np.pi / 2
            ),
        ),
        config=VisionRGBDFollowerConfig(
            control_time_step=0.1,
            _use_local_coordinates=True,
            min_depth=0.1,
            max_depth=10.0,
        ),
        camera_focal_length=[500.0, 500.0],
        camera_principal_point=[320.0, 240.0],
    )
    # The default sensor (identity mount) takes the cloud in the body frame:
    # a 3 m target, 1 m wide and tall, dense enough to fill its box
    ys, zs = np.meshgrid(np.linspace(-0.5, 0.5, 60), np.linspace(-0.5, 0.5, 60))
    target = np.zeros((ys.size, 4), dtype=np.float32)
    target[:, 0] = 3.0
    target[:, 1] = ys.ravel()
    target[:, 2] = zs.ravel()
    readonly = np.frombuffer(target.tobytes(), dtype=np.uint8)
    assert not readonly.flags.writeable
    cloud = {
        "data": readonly,
        "point_step": _PC_STRIDE,
        "row_step": _PC_STRIDE * len(target),
        "height": 1,
        "width": len(target),
        "x_offset": 0,
        "y_offset": 4,
        "z_offset": 8,
    }
    # The target projects to a ~167 px square around the principal point
    box = Bbox2D(
        top_left_corner=np.array([230, 150], dtype=np.int32),
        size=np.array([180, 180], dtype=np.int32),
    )
    box.set_img_size(np.array([640, 480], dtype=np.int32))
    state = RobotState(x=0.0, y=0.0, yaw=0.0, speed=0.0)
    assert follower.set_initial_tracking_image(
        current_state=state,
        pose_x_img=320,
        pose_y_img=240,
        detected_boxes=[box],
        **cloud,
    )
    box.timestamp = 0.1
    assert follower.loop_step(current_state=state, detections_2d=[box], **cloud)


def test_depth_image_dtype_binds_its_own_overload_and_is_never_cast():
    """The uint16 and float32 depth-image overloads differ only by dtype, so
    the depth argument is bound noconvert. Without it, any call that needs an
    implicit conversion elsewhere (an int for a float argument, say) drops to
    nanobind's converting pass, where the uint16 overload -- registered first
    -- accepts a float32 image by casting it: metres truncated to integers,
    then read as millimetres. So a float32 image must lift the same whatever
    the other arguments look like, and a dtype or layout matching neither
    overload must be refused rather than silently copied."""
    detector = DepthDetector(
        np.array([0.1, 10.0], dtype=np.float32),  # depth range
        np.array([0.0, 0.0, 0.0], dtype=np.float32),  # camera translation
        np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32),  # camera rotation
        np.array([500.0, 500.0], dtype=np.float32),  # focal length
        np.array([320.0, 240.0], dtype=np.float32),  # principal point
        1e-3,  # mm -> m, applies to the uint16 path only
    )
    box = _target_box(640, 480)
    metres = np.zeros((480, 640), dtype=np.float32)
    x0, y0 = _BOX_CORNER
    metres[y0 : y0 + _BOX_SIZE, x0 : x0 + _BOX_SIZE] = 2.5

    exact = detector.compute_3d_detections(metres, [box], 0.0, 0.0, 0.0, 0.0)
    # robot_x as an int forces the converting pass. A cast to uint16 would
    # read the box as 2 mm, below the depth range, and lift nothing at all.
    converted_elsewhere = detector.compute_3d_detections(
        metres, [box], 0, 0.0, 0.0, 0.0
    )
    assert exact and converted_elsewhere, "the float32 image was not lifted"
    np.testing.assert_allclose(converted_elsewhere[0].center, exact[0].center)
    assert np.linalg.norm(exact[0].center) == pytest.approx(2.5, abs=0.05)

    # Neither overload matches: refuse instead of copying into one of them
    for unsupported in (
        metres.astype(np.float64),
        metres.astype(np.int32),
        np.repeat(metres, 2, axis=1)[:, ::2],  # right shape, not contiguous
    ):
        with pytest.raises(TypeError):
            detector.compute_3d_detections(unsupported, [box], 0.0, 0.0, 0.0, 0.0)
