"""Generate synthetic RGBDFollower test fixtures.

Each fixture is a directory under this folder containing:
  - depth.png: 16-bit single-channel depth image (millimeters)
  - case.json: camera intrinsics, robot state, 2D detections, click point,
               and loose expected bounds for the resulting control command.

The fixtures are committed to git so both the C++ Boost test and the Python
pytest run against identical data. Re-run this script to regenerate them, or
add new cases by appending to FIXTURES below.

To capture a fixture from the real robot instead:
  1. Save the aligned depth image as a 16-bit PNG (mm).
  2. Record the detection bounding boxes as Bbox2D-style entries.
  3. Record the camera_info (fx, fy, cx, cy) and robot odom at the same stamp.
  4. Hand-pick the click pixel inside the target detection.
  5. Write a case.json mirroring the structure produced here. Pick LOOSE
     expected bounds — exact values drift with tracker/detector tuning.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np


HERE = Path(__file__).parent

# Default camera (matches the kompass test fixtures style)
FX, FY = 525.0, 525.0
CX, CY = 320.0, 240.0
IMG_W, IMG_H = 640, 480
DEPTH_CONV = 1e-3  # mm -> m
MIN_DEPTH = 0.1
MAX_DEPTH = 5.0


@dataclass
class Detection:
    top_left: Tuple[int, int]
    size: Tuple[int, int]
    label: str = "target"
    timestamp: float = 0.0


@dataclass
class Expected:
    init_success: bool = True
    # Loose bounds on the FIRST control command produced by getTrackingCtrl
    vx_min: float = -1e3
    vx_max: float = 1e3
    omega_min: float = -1e3
    omega_max: float = 1e3
    # Loose bounds on the recovered 3D target distance from the robot (meters)
    target_distance_min_m: float = 0.0
    target_distance_max_m: float = 1e3


@dataclass
class Fixture:
    name: str
    description: str
    detections: List[Detection]
    click: Tuple[int, int]
    target_depth_m: float
    expected: Expected
    robot: Tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)  # x, y, yaw, speed
    fill_pad: int = 0  # extra mm pixels around each detection box

    def render_depth(self) -> np.ndarray:
        img = np.zeros((IMG_H, IMG_W), dtype=np.uint16)
        depth_mm = int(round(self.target_depth_m * 1000.0))
        for det in self.detections:
            x0, y0 = det.top_left
            w, h = det.size
            x0p = max(0, x0 - self.fill_pad)
            y0p = max(0, y0 - self.fill_pad)
            x1p = min(IMG_W, x0 + w + self.fill_pad)
            y1p = min(IMG_H, y0 + h + self.fill_pad)
            img[y0p:y1p, x0p:x1p] = depth_mm
        return img

    def to_case_json(self) -> dict:
        return {
            "name": self.name,
            "description": self.description,
            "camera": {
                "fx": FX,
                "fy": FY,
                "cx": CX,
                "cy": CY,
                "img_w": IMG_W,
                "img_h": IMG_H,
                "depth_conversion_factor": DEPTH_CONV,
                "min_depth": MIN_DEPTH,
                "max_depth": MAX_DEPTH,
            },
            "robot": {
                "x": self.robot[0],
                "y": self.robot[1],
                "yaw": self.robot[2],
                "speed": self.robot[3],
            },
            "click": {"x": self.click[0], "y": self.click[1]},
            "detections": [asdict(d) for d in self.detections],
            "target_depth_m": self.target_depth_m,
            "expected": asdict(self.expected),
        }


FIXTURES: List[Fixture] = [
    Fixture(
        name="static_centered_2m",
        description=(
            "Target box centered on principal point, 2.0m ahead. Far from "
            "target_distance -> forward; centered -> no turn."
        ),
        detections=[Detection(top_left=(270, 190), size=(100, 100))],
        click=(320, 240),
        target_depth_m=2.0,
        expected=Expected(
            vx_min=0.05,
            vx_max=2.0,
            omega_min=-0.1,
            omega_max=0.1,
            target_distance_min_m=0.0,
            target_distance_max_m=1000.0,
        ),
    ),
    Fixture(
        name="far_centered_4m",
        description=(
            "Target box centered on principal point, 4.0m ahead. Far from "
            "target_distance -> strong forward; centered -> no turn."
        ),
        detections=[Detection(top_left=(295, 215), size=(50, 50))],
        click=(320, 240),
        target_depth_m=4.0,
        expected=Expected(
            vx_min=0.05,
            vx_max=2.5,
            omega_min=-0.1,
            omega_max=0.1,
        ),
    ),
    Fixture(
        name="close_centered_0p3m",
        description=(
            "Target box centered on principal point, 0.3m ahead. Near "
            "target_distance -> small/zero forward command; centered -> no turn."
        ),
        detections=[Detection(top_left=(170, 90), size=(300, 300))],
        click=(320, 240),
        target_depth_m=0.3,
        expected=Expected(
            vx_min=-1.5,
            vx_max=0.5,
            omega_min=-0.3,
            omega_max=0.3,
        ),
    ),
    Fixture(
        name="offcenter_left_2m",
        description=(
            "Target box offset to the left at 2.0m. Robot should turn "
            "left (positive omega)."
        ),
        detections=[Detection(top_left=(120, 190), size=(100, 100))],
        click=(170, 240),
        target_depth_m=2.0,
        expected=Expected(
            vx_min=-0.5,
            vx_max=2.0,
            omega_min=0.05,
            omega_max=3.0,
        ),
    ),
    Fixture(
        name="offcenter_right_2m",
        description=(
            "Target box offset to the right at 2.0m. Robot should turn "
            "right (negative omega)."
        ),
        detections=[Detection(top_left=(420, 190), size=(100, 100))],
        click=(470, 240),
        target_depth_m=2.0,
        expected=Expected(
            vx_min=-0.5,
            vx_max=2.0,
            omega_min=-3.0,
            omega_max=-0.05,
        ),
    ),
]


def main() -> None:
    for fx in FIXTURES:
        out_dir = HERE / fx.name
        out_dir.mkdir(parents=True, exist_ok=True)
        depth = fx.render_depth()
        cv2.imwrite(str(out_dir / "depth.png"), depth)
        with open(out_dir / "case.json", "w") as f:
            json.dump(fx.to_case_json(), f, indent=2)
        print(f"wrote {fx.name}")


if __name__ == "__main__":
    main()
