"""Parametrized tests for VisionRGBDFollower against fixture cases.

Each fixture is a directory under tests/resources/vision_dwa/ containing a
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


FIXTURE_ROOT = Path(__file__).parent / "resources" / "vision_dwa"


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
            max_vel=2.5, max_acc=2.5, max_decel=2.5, max_steer=np.pi / 2
        ),
    )
    config = VisionRGBDFollowerConfig(
        control_time_step=0.1,
        control_horizon=2,
        prediction_horizon=6,
        max_linear_samples=15,
        max_angular_samples=15,
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
def test_vision_dwa_fixture(case_dir: Path) -> None:
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
        point_cloud=[],
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
