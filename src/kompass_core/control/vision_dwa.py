from attrs import define, field
from ..utils.common import BaseAttrs, base_validators
import kompass_cpp
from typing import Optional
import numpy as np


@define
class VisionDWAConfig(BaseAttrs):
    control_time_step: float = field(
        default=0.1, validator=base_validators.in_range(min_value=1e-4, max_value=1e6)
    )
    # Control time step (s)

    control_horizon: int = field(
        default=2, validator=base_validators.in_range(min_value=1, max_value=1000)
    )
    # Number of steps for applying the control

    prediction_horizon: int = field(
        default=10, validator=base_validators.in_range(min_value=1, max_value=1000)
    )
    # Number of steps for future prediction

    tolerance: float = field(
        default=0.01, validator=base_validators.in_range(min_value=1e-6, max_value=1e3)
    )
    # Tolerance value for distance and angle following errors

    target_distance: Optional[float] = field(
        default=0.1,
        validator=base_validators.in_range(min_value=1e-9, max_value=1e9),
    )  # Target distance to maintain with the target (m)

    target_orientation: float = field(
        default=0.0,
        validator=base_validators.in_range(min_value=-np.pi, max_value=np.pi),
    )  # Bearing angle to maintain with the target (rad)

    target_wait_timeout: float = field(
        default=30.0, validator=base_validators.in_range(min_value=0.0, max_value=1e3)
    )  # Wait for target to appear again timeout (seconds), used if search is disabled

    target_search_timeout: float = field(
        default=30.0, validator=base_validators.in_range(min_value=0.0, max_value=1e3)
    )  # Search timeout in seconds

    target_search_radius: float = field(
        default=0.5, validator=base_validators.in_range(min_value=1e-4, max_value=1e4)
    )  # Search radius for finding the target (m)

    target_search_pause: float = field(
        default=1.0, validator=base_validators.in_range(min_value=0.0, max_value=1e3)
    )  # Pause between search actions to find target (seconds)

    rotation_gain: float = field(
        default=0.5, validator=base_validators.in_range(min_value=1e-2, max_value=10.0)
    )  # Gain for the rotation control law

    speed_gain: float = field(
        default=1.0, validator=base_validators.in_range(min_value=1e-2, max_value=10.0)
    )  # Gain for the speed control law

    min_vel: float = field(
        default=0.1, validator=base_validators.in_range(min_value=1e-9, max_value=1e9)
    )  # Minimum velocity to apply (m/s)

    enable_search: bool = field(default=False)  # Enable or disable the search mechanism

    error_pose: float = field(
        default=0.05, validator=base_validators.in_range(min_value=1e-9, max_value=1e9)
    )  # Error in pose estimation (m)

    error_vel: float = field(
        default=0.05, validator=base_validators.in_range(min_value=1e-9, max_value=1e9)
    )  # Error in velocity estimation (m/s)

    error_acc: float = field(
        default=0.05, validator=base_validators.in_range(min_value=1e-9, max_value=1e9)
    )  # Error in acceleration estimation (m/s^2)

    def to_kompass_cpp(self) -> kompass_cpp.control.VisionDWAParameters:
        """
        Convert to kompass_cpp lib config format

        :return: _description_
        :rtype: kompass_cpp.control.VisionDWAParameters
        """
        vision_dwa_params = kompass_cpp.control.VisionDWAParameters()

        # Special handling for None values that are represented by -1 in C++
        params_dict = self.asdict()
        if params_dict["target_distance"] is None:
            params_dict["target_distance"] = -1.0

        vision_dwa_params.from_dict(params_dict)
        return vision_dwa_params
