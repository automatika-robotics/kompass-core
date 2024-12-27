from abc import abstractmethod
from typing import Optional, List
import numpy as np
import kompass_cpp
from ..models import RobotState
from attrs import define, field
from ..utils.common import BaseAttrs, base_validators
from ..utils.geometry import convert_to_plus_minus_pi
from ..datatypes.laserscan import LaserScanData
from kompass_cpp.types import PathInterpolationType


@define
class FollowerConfig(BaseAttrs):
    """
    General path follower parameters
    """

    max_point_interpolation_distance: float = field(
        default=0.01, validator=base_validators.in_range(min_value=1e-4, max_value=1e2)
    )

    lookahead_distance: float = field(
        default=1.0, validator=base_validators.in_range(min_value=1e-4, max_value=1e2)
    )

    goal_dist_tolerance: float = field(
        default=0.1, validator=base_validators.in_range(min_value=1e-4, max_value=1e2)
    )

    goal_orientation_tolerance: float = field(
        default=0.1, validator=base_validators.in_range(min_value=1e-4, max_value=np.pi)
    )

    path_segment_length: float = field(
        default=1.0, validator=base_validators.in_range(min_value=1e-4, max_value=1e2)
    )

    loosing_goal_distance: float = field(
        default=0.2, validator=base_validators.in_range(min_value=1e-4, max_value=1e2)
    )


class ControllerTemplate:
    """
    Any controller defined in KOMPASS should inherit this class
    """

    @abstractmethod
    def __init__(
        self,
        config_file: Optional[str] = None,
        config_yaml_root_name: Optional[str] = None,
        **_,
    ) -> None:
        """
        Sets up the controller and any required objects
        """
        raise NotImplementedError

    @abstractmethod
    def loop_step(
        self,
        *,
        laser_scan: LaserScanData,
        initial_control_seq: np.ndarray,
        current_state: RobotState,
        goal_state: RobotState,
        **kwargs,
    ):
        """
        Implements one loop iteration of the controller
        Contains the main controller logic - should be reimplemented by child

        :param laser_scan: 2d laser scan data, defaults to None
        :type laser_scan: LaserScanData | None, optional
        :param initial_control_seq: Reference control sequence normally provided by a pure follower, defaults to None
        :type initial_control_seq: np.ndarray | None, optional
        :param current_state: Robot current state, defaults to None
        :type current_state: RobotState | None, optional

        :raises NotImplementedError: If method is not implemented in child
        """
        raise NotImplementedError

    @abstractmethod
    def logging_info(self) -> str:
        """
        Returns controller progress info for the Node to log

        :return: Controller Info
        :rtype: str
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def linear_x_control(self):
        """
        Getter of the last linear forward velocity control computed by the controller

        :return: Linear Velocity Control (m/s)
        :rtype: float
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def linear_y_control(self):
        """
        Getter the last linear velocity lateral control computed by the controller

        :return: Linear Velocity Control (m/s)
        :rtype: float
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def angular_control(self):
        """
        Getter of the last angular velocity control computed by the controller

        :return: Angular Velocity Control (rad/s)
        :rtype: float
        """
        raise NotImplementedError


class FollowerTemplate:
    """
    Any Follower defined in KOMPASS should inherit this class
    """

    @abstractmethod
    def __init__(
        self,
        config: Optional[FollowerConfig] = None,
        config_file: Optional[str] = None,
        config_yaml_root_name: Optional[str] = None,
        **kwargs,
    ) -> None:
        """
        Sets up the controller and any required objects
        """
        self._config = config or FollowerConfig()
        raise NotImplementedError

    @property
    @abstractmethod
    def planner(self) -> kompass_cpp.control.Follower:
        raise NotImplementedError

    def reached_end(self) -> bool:
        """Check if current goal is reached

        :return: If goal is reached
        :rtype: bool
        """
        return self.planner.is_goal_reached()

    def set_path(self, global_path, **_) -> None:
        """
        Set global path to be tracked by the planner

        :param global_path: Global reference path
        :type global_path: Path
        """
        parsed_points = []
        for point in global_path.poses:
            parsed_point = kompass_cpp.types.Point(
                point.pose.position.x, point.pose.position.y
            )
            parsed_points.append(parsed_point)

        self.planner.set_current_path(kompass_cpp.types.Path(points=parsed_points))
        self._got_path = True

    @property
    def path(self) -> bool:
        """
        Checks if the follower path
        """
        return self.planner.has_path()

    @path.setter
    def path(self, global_path) -> None:
        """
        Getter of the follower path

        :raises NotImplementedError: If method is not implemented in child
        """
        self.set_path(global_path=global_path)

    @abstractmethod
    def loop_step(self, *, current_state: RobotState, **kwargs) -> bool:
        """
        Implements one loop iteration of the follower
        Contains the main controller logic - should be reimplemented by child

        :param current_state: Robot current state, defaults to None
        :type current_state: RobotState | None, optional

        :raises NotImplementedError: If method is not implemented in child
        """
        raise NotImplementedError

    @abstractmethod
    def logging_info(self) -> str:
        """
        Returns controller progress info for the Node to log

        :return: Controller Info
        :rtype: str
        """
        raise NotImplementedError

    def optimal_path(self) -> Optional[kompass_cpp.types.Path]:
        """Get optimal (local) plan."""
        pass

    def interpolated_path(self) -> Optional[kompass_cpp.types.Path]:
        """Get path interpolation."""
        return self.planner.get_current_path()

    def set_interpolation_type(self, interpolation_type: PathInterpolationType):
        """Set the follower path interpolation type

        :param interpolation_type: Used interpolation (linear, hermit, etc.)
        :type interpolation_type: PathInterpolationType
        """
        self._planner.set_interpolation_type(interpolation_type)

    @property
    def tracked_state(self) -> Optional[RobotState]:
        """
        Tracked state on the path

        :return: _description_
        :rtype: RobotState
        """
        if not self.path:
            return None
        target: kompass_cpp.control.FollowingTarget = self.planner.get_tracked_target()
        if not target:
            return None
        return RobotState(
            x=target.movement.x, y=target.movement.y, yaw=target.movement.yaw
        )

    @property
    @abstractmethod
    def linear_x_control(self) -> List[float]:
        """
        Getter of the last linear forward velocity control computed by the controller

        :return: Linear Velocity Control (m/s)
        :rtype: float
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def linear_y_control(self) -> List[float]:
        """
        Getter the last linear velocity lateral control computed by the controller

        :return: Linear Velocity Control (m/s)
        :rtype: float
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def angular_control(self) -> List[float]:
        """
        Getter of the last angular velocity control computed by the controller

        :return: Angular Velocity Control (rad/s)
        :rtype: float
        """
        raise NotImplementedError

    @property
    def distance_error(self) -> float:
        """
        Getter of the path tracking lateral distance error (m)

        :return: Lateral distance to tracked path (m)
        :rtype: float
        """
        target: kompass_cpp.control.FollowingTarget = self.planner.get_tracked_target()
        return target.crosstrack_error

    @property
    def orientation_error(self) -> float:
        """
        Getter of the path tracking orientation error (rad)

        :return: Orientation error to tracked path (rad)
        :rtype: float
        """
        target: kompass_cpp.control.FollowingTarget = self.planner.get_tracked_target()
        return convert_to_plus_minus_pi(target.heading_error)
