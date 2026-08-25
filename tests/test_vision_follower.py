"""Parametrized tests for VisionRGBDFollower against fixture cases.

Each fixture is a directory under tests/resources/vision_follower/ containing a
depth.png (16-bit, mm) and a case.json. See generate_fixtures.py for the
schema and how to add new cases (synthetic or recorded from the robot).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List

import cv2
import numpy as np
import pytest

from kompass_core.control import VisionRGBDFollower, VisionRGBDFollowerConfig
from kompass_core.datatypes import Bbox2D
from kompass_core.models import (
    AngularCtrlLimits,
    LinearCtrlLimits,
    Robot,
    RobotCtrlLimits,
    RobotGeometry,
    RobotState,
    RobotType,
)


FIXTURE_ROOT = Path(__file__).parent / "resources" / "vision_follower"


def _discover_fixtures() -> List[Path]:
    return sorted(p for p in FIXTURE_ROOT.iterdir() if (p / "case.json").exists())


def _load_case(case_dir: Path) -> dict:
    with open(case_dir / "case.json") as f:
        return json.load(f)


def _build_detections(case: dict) -> List[Bbox2D]:
    out: List[Bbox2D] = []
    img_w = int(case["camera"]["img_w"])
    img_h = int(case["camera"]["img_h"])
    for det in case["detections"]:
        box = Bbox2D(
            top_left_corner=np.array(det["top_left"], dtype=np.int32),
            size=np.array(det["size"], dtype=np.int32),
            timestamp=float(det.get("timestamp", 0.0)),
            label=str(det.get("label", "target")),
        )
        box.set_img_size(np.array([img_w, img_h], dtype=np.int32))
        out.append(box)
    return out


def _make_follower(case: dict) -> VisionRGBDFollower:
    cam = case["camera"]
    robot = Robot(
        robot_type=RobotType.DIFFERENTIAL_DRIVE,
        geometry_type=RobotGeometry.Type.CYLINDER,
        geometry_params=np.array([0.1, 0.4]),
    )
    ctrl_limits = RobotCtrlLimits(
        vx_limits=LinearCtrlLimits(max_vel=1.5, max_acc=3.0, max_decel=3.0),
        omega_limits=AngularCtrlLimits(
            max_omega=2.5, max_acc=2.5, max_decel=2.5, max_ang=np.pi / 2
        ),
    )
    config = VisionRGBDFollowerConfig(
        control_time_step=0.1,
        control_horizon=2,
        prediction_horizon=6,
        target_distance=0.5,
        distance_tolerance=0.1,
        _use_local_coordinates=True,
        depth_conversion_factor=float(cam["depth_conversion_factor"]),
        min_depth=float(cam["min_depth"]),
        max_depth=float(cam["max_depth"]),
    )
    return VisionRGBDFollower(
        robot=robot,
        ctrl_limits=ctrl_limits,
        config=config,
        camera_focal_length=[float(cam["fx"]), float(cam["fy"])],
        camera_principal_point=[float(cam["cx"]), float(cam["cy"])],
    )


@pytest.mark.parametrize(
    "case_dir", _discover_fixtures(), ids=lambda p: p.name
)
def test_vision_follower_fixture(case_dir: Path) -> None:
    case = _load_case(case_dir)
    depth = cv2.imread(str(case_dir / "depth.png"), cv2.IMREAD_UNCHANGED)
    assert depth is not None and depth.dtype == np.uint16, (
        f"Could not load 16-bit depth.png for {case_dir.name}"
    )

    follower = _make_follower(case)
    detections = _build_detections(case)
    state = RobotState(
        x=case["robot"]["x"], y=case["robot"]["y"],
        yaw=case["robot"]["yaw"], speed=case["robot"]["speed"],
    )

    init_ok = follower.set_initial_tracking_image(
        current_state=state,
        pose_x_img=int(case["click"]["x"]),
        pose_y_img=int(case["click"]["y"]),
        detected_boxes=detections,
        aligned_depth_image=depth,
    )
    assert init_ok == case["expected"]["init_success"], (
        f"{case_dir.name}: setInitialTracking returned {init_ok}, "
        f"expected {case['expected']['init_success']}"
    )
    if not init_ok:
        return

    # Run one control step. Sensor data is an empty point cloud (no obstacles).
    found = follower.loop_step(
        current_state=state,
        detections_2d=detections,
        depth_image=depth,
    )
    assert found, f"{case_dir.name}: planner failed to find a control"

    vx = follower.linear_x_control[0]
    omega = follower.angular_control[0]
    exp = case["expected"]
    assert exp["vx_min"] <= vx <= exp["vx_max"], (
        f"{case_dir.name}: vx={vx} outside [{exp['vx_min']}, {exp['vx_max']}]"
    )
    assert exp["omega_min"] <= omega <= exp["omega_max"], (
        f"{case_dir.name}: omega={omega} outside "
        f"[{exp['omega_min']}, {exp['omega_max']}]"
    )


def test_rgbd_follower_exposes_rgb_follower_interface() -> None:
    """The C++ RGBDFollower also inherits RGBFollower; nanobind registers a
    single base (Follower), so the RGBFollower interface is re-bound on the
    RGBDFollower binding and must stay reachable from Python."""
    case = _load_case(_discover_fixtures()[0])
    follower = _make_follower(case)
    planner = follower._planner

    box = Bbox2D(
        top_left_corner=np.array([300, 220], dtype=np.int32),
        size=np.array([40, 40], dtype=np.int32),
    )
    box.set_img_size(np.array([640, 480], dtype=np.int32))

    planner.reset_target(box)
    assert isinstance(planner.run(box), bool)
    assert isinstance(planner.run(None), bool)
    # get_ctrl returns the velocities the RGB control path produced
    assert planner.get_ctrl() is not None
    assert planner.get_errors() is not None


# ---------------------------------------------------------------------------
# Close-target regression: the bearing-hold feedforward must not run away
# ---------------------------------------------------------------------------

#: Depth camera on the front of a box-shaped robot, tilted ~23 degrees down.
#: The tilt matters: it mixes the target's visible height into the body-x
#: extent, which is what inflates the target radius.
_TILTED_MOUNT = (-0.5839669, 0.5936764, -0.38588133, 0.3970222)


def _close_target_follower() -> VisionRGBDFollower:
    robot = Robot(
        robot_type=RobotType.DIFFERENTIAL_DRIVE,
        # A real chassis: the circumradius is 0.357 m, so a target closer than
        # that is inside the robot's own footprint circle
        geometry_type=RobotGeometry.Type.BOX,
        geometry_params=np.array([0.61, 0.37, 0.4]),
    )
    ctrl_limits = RobotCtrlLimits(
        vx_limits=LinearCtrlLimits(max_vel=1.5, max_acc=3.0, max_decel=3.0),
        omega_limits=AngularCtrlLimits(
            max_omega=2.5, max_acc=2.5, max_decel=2.5, max_ang=np.pi / 2
        ),
    )
    config = VisionRGBDFollowerConfig(
        control_time_step=0.1,
        target_distance=0.5,
        _use_local_coordinates=True,
        camera_position_to_robot=np.array([0.0, 0.0, 0.3], dtype=np.float32),
        camera_rotation_to_robot=np.array(_TILTED_MOUNT, dtype=np.float32),
    )
    return VisionRGBDFollower(
        robot=robot,
        ctrl_limits=ctrl_limits,
        config=config,
        camera_focal_length=[500.0, 500.0],
        camera_principal_point=[320.0, 240.0],
    )


def _close_target_frame():
    """A person 0.45 m away, filling the frame and a little to the left.

    At that range the target radius plus the robot's own radius exceeds the
    range, so the surface-to-surface gap collapses to its floor -- the state
    that used to detonate the omega feedforward.
    """
    depth = np.zeros((480, 640), dtype=np.uint16)
    x0, y0, w, h = 60, 0, 400, 480
    depth[y0 : y0 + h, x0 : x0 + w] = 450  # mm

    box = Bbox2D()
    box.top_left_corner = np.array([x0, y0], dtype=np.int32)
    box.size = np.array([w, h], dtype=np.int32)
    box.img_size = np.array([640, 480], dtype=np.int32)
    return depth, box, (x0 + w // 2, y0 + h // 2)


def test_close_target_does_not_saturate_omega() -> None:
    """A target inside the robot's own radius must not rail the rotation.

    Regression: the bearing-hold feedforward divided by the surface-to-surface
    gap rather than the range to the target. That gap floors at 1 mm as soon as
    the target is within the combined radii, turning the term into a ~1000x
    gain that pinned omega to the limit and drove the robot round until it lost
    the person. `rotation_gain` cannot damp it -- the gain scales only the
    feedback term, so turning it down makes matters worse by removing the one
    term pulling back.
    """
    follower = _close_target_follower()
    depth, box, (click_x, click_y) = _close_target_frame()
    state = RobotState(x=0.0, y=0.0, yaw=0.0, speed=0.0)

    assert follower.set_initial_tracking_image(
        current_state=state,
        pose_x_img=click_x,
        pose_y_img=click_y,
        detected_boxes=[box],
        aligned_depth_image=depth,
    )
    assert follower.loop_step(
        current_state=state, detections_2d=[box], depth_image=depth
    )

    omega = follower.angular_control[0]
    max_omega = 2.5
    assert abs(omega) < 0.5 * max_omega, (
        f"omega={omega} on a close target: the feedforward is running away "
        f"(limit is {max_omega})"
    )
