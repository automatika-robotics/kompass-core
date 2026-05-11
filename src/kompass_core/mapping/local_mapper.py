from typing import Optional, Union
from attrs import define, field, validators
import math
import numpy as np
from ..datatypes.pose import PoseData
from ..datatypes.laserscan import LaserScanData
from ..datatypes.pointcloud import PointCloudData

from kompass_cpp.mapping import (
    OCCUPANCY_TYPE,
)

from ..utils.geometry import transform_point_from_local_to_global, get_relative_pose

from ..datatypes.scan_model import ScanModelConfig
from ..utils.common import BaseAttrs, base_validators, logging


@define
class GridData(BaseAttrs):
    """Grid Data class with layers for:
    - Scan Occupancy
    - Scan Probabilistic Occupancy
    - Total Occupancy
    - Total Probabilistic occupancy

    :param BaseAttrs: _description_
    :type BaseAttrs: _type_
    :return: _description_
    :rtype: _type_
    """

    width: int = field()
    height: int = field()
    p_prior: float = field(default=0.5)
    occupancy: np.ndarray = field(init=False)
    # TODO: Add semantic occupancy
    # semantic_occupancy : np.ndarray = field(init=False)
    # semantic : np.ndarray = field(init=False)

    def __attrs_post_init__(self):
        self.occupancy = self.get_initial_grid_data()

    def get_initial_grid_data(self) -> np.ndarray:
        """
        get an initial empty grid with every cell assumed to be unexplored yet

        :return:    2D array filled with unexplored occupancy
        :rtype:     np.ndarray
        """
        data = np.full(
            (self.width, self.height),
            OCCUPANCY_TYPE.UNEXPLORED.value,
            dtype=np.int32,
            order="F",
        )
        return data


@define(kw_only=True)
class MapConfig(BaseAttrs):
    """
    Local mapper configuration parameters
    """

    width: float = field(
        default=3.0, validator=base_validators.in_range(min_value=0.1, max_value=1e2)
    )
    height: float = field(
        default=3.0, validator=base_validators.in_range(min_value=0.1, max_value=1e2)
    )
    resolution: float = field(
        default=0.1, validator=base_validators.in_range(min_value=1e-9, max_value=1e2)
    )
    padding: float = field(
        default=0.0, validator=base_validators.in_range(min_value=0.0, max_value=10.0)
    )
    baysian_update: bool = field(default=False)
    max_num_threads: int = field(default=1, validator=validators.ge(1))

    filter_limit: float = field(
        validator=base_validators.in_range(min_value=0.1, max_value=1e2)
    )

    max_points_per_line: int = field(
        validator=base_validators.in_range(min_value=1, max_value=1e3)
    )

    @filter_limit.default
    def _set_filter_limit(self) -> float:
        # calculate scan limit for filtering - diameter of circle inscribing rectangle
        return (
            self.width * math.sqrt(2)
            if self.width >= self.height
            else self.height * math.sqrt(2)
        )

    @max_points_per_line.default
    def _set_max_points_per_line(self) -> float:
        # estimate max number of points drawn per scan line
        # at average 1.5 points per setup
        return round((self.filter_limit / self.resolution) * 1.5)


class LocalMapper:
    """
    LocalMapper class produces a grid map around the current robot position using LaserScanData

    Supported layers:
    - Occupancy
    - Probabilistic Occupancy
    """

    def __init__(
        self,
        config: MapConfig,
        scan_model_config: ScanModelConfig,
        pose_laser_scanner_in_robot: Optional[PoseData] = None,
    ):
        """Initialize a LocalMapper

        :param config: Mapper config
        :type config: MapConfig
        :param scan_model_config: LaserScan or PointCloud model config
        :type scan_model_config: ScanModelConfig
        """

        self.config = config

        self.grid_width = int(self.config.width / self.config.resolution)
        self.grid_height = int(self.config.height / self.config.resolution)

        self._local_lower_right_corner_point = PoseData()
        self._local_lower_right_corner_point.set_position(
            x=-1 * config.width / 2, y=-1 * config.height / 2, z=0
        )

        # TODO: Add robot point to track robot footprint
        # self.grid_robot_point = [
        #     int(self.grid_width / 2) - 1,
        #     int(self.grid_height / 2) - 1,
        # ]

        # current obstacles and grid data
        self._pose_robot_in_world = PoseData()
        self.lower_right_corner_pose = PoseData()

        self.scan_model = scan_model_config

        self.pose_laserscanner_in_robot = (
            pose_laser_scanner_in_robot if pose_laser_scanner_in_robot else PoseData()
        )

        self.laserscan_orientation_in_robot = 2 * np.arctan(
            self.pose_laserscanner_in_robot.qz / self.pose_laserscanner_in_robot.qw
        )

        self.grid_data = GridData(
            width=self.grid_width,
            height=self.grid_height,
            p_prior=self.scan_model.p_prior,
        )

        # flag for pointcloud
        self.is_pointcloud = False
        # turned to true after the first map update is done
        self.processed = False

    @property
    def occupancy(self) -> np.ndarray:
        """Getter of current grid occupancy

        :return: Grid occupancy layer
        :rtype: np.ndarray
        """
        return self.grid_data.occupancy

    @property
    def probabilities(self) -> Optional[np.ndarray]:
        """Debug view of the current Bayesian probability grid.

        Copies the log-odds GPU buffer and applies sigmoid on the C++
        side; values are in [0, 1]. Only valid when `config.baysian_update`
        is enabled AND the GPU backend is available.

        Call after `update_from_scan` to inspect the posterior produced by
        the current scan-model parameters. Useful for tuning `p_occupied`,
        `range_sure`, `range_max`, etc.

        :return: Probability grid (float32) or None on the CPU backend.
        :rtype: Optional[np.ndarray]
        """
        if not self.config.baysian_update:
            return None
        getter = getattr(self.local_mapper, "get_probabilities", None)
        if getter is None:
            return None
        return np.copy(getter())

    def _initialize_mapper(self, scan_size: int) -> None:
        """Initialize cpp local mapper.

        If `config.baysian_update` is true, attempts to construct the GPU mapper
        via its Bayesian ctor.

        The CPU Bayesian path is not exposed and will be removed. Switches to
        non-Bayesian when GPU unavailable.
        """
        try:
            from kompass_cpp.mapping import LocalMapperGPU

            if self.config.baysian_update:
                self.local_mapper = LocalMapperGPU(
                    grid_height=self.grid_height,
                    grid_width=self.grid_width,
                    resolution=self.config.resolution,
                    laserscan_position=self.pose_laserscanner_in_robot.get_position(),
                    laserscan_orientation=self.laserscan_orientation_in_robot,
                    is_pointcloud=self.is_pointcloud,
                    scan_size=scan_size,
                    p_prior=self.scan_model.p_prior,
                    p_occupied=self.scan_model.p_occupied,
                    p_empty=self.scan_model.p_empty,
                    range_sure=self.scan_model.range_sure,
                    range_max=self.scan_model.range_max,
                    angle_step=self.scan_model.angle_step,
                    max_height=self.scan_model.max_height,
                    min_height=self.scan_model.min_height,
                    max_points_per_line=self.config.max_points_per_line,
                )
            else:
                self.local_mapper = LocalMapperGPU(
                    grid_height=self.grid_height,
                    grid_width=self.grid_width,
                    resolution=self.config.resolution,
                    laserscan_position=self.pose_laserscanner_in_robot.get_position(),
                    laserscan_orientation=self.laserscan_orientation_in_robot,
                    is_pointcloud=self.is_pointcloud,
                    scan_size=scan_size,
                    angle_step=self.scan_model.angle_step,
                    max_height=self.scan_model.max_height,
                    min_height=self.scan_model.min_height,
                    range_max=self.scan_model.range_max,
                    max_points_per_line=self.config.max_points_per_line,
                )
        except ImportError:
            from kompass_cpp.mapping import LocalMapper as LocalMapperCpp

            if self.config.baysian_update:
                logging.warning(
                    "Bayesian mapping is not available on the CPU backend in "
                    "this build; falling back to non-Bayesian scan_to_grid for "
                    "this session."
                )
                self.config.baysian_update = False

            self.local_mapper = LocalMapperCpp(
                grid_height=self.grid_height,
                grid_width=self.grid_width,
                resolution=self.config.resolution,
                laserscan_position=self.pose_laserscanner_in_robot.get_position(),
                laserscan_orientation=self.laserscan_orientation_in_robot,
                is_pointcloud=self.is_pointcloud,
                scan_size=scan_size,
                **self.scan_model.asdict(),
                max_points_per_line=self.config.max_points_per_line,
                max_num_threads=self.config.max_num_threads,
            )

    def update_from_scan(
        self,
        robot_pose: PoseData,
        scan: Union[LaserScanData, PointCloudData],
    ):
        """
        Update the local map using new LaserScan data

        :param robot_pose: Current robot position
        :type robot_pose: PoseData
        :param scan: Scan data, laserscan or pointcloud
        :type scan: LaserScanData or PointCloudData
        """
        # Get transformation between the previous robot state (pose and grid) w.r.t the current state.

        if not self.processed:
            self.is_pointcloud = isinstance(scan, PointCloudData)
            if self.is_pointcloud:
                # NOTE: `angle_step` is the canonical knob on the Python side; the
                # bin count is derived from it with a ceil so every angle in
                # [0, 2π) lands in a valid bin. The C++ ctor then enforces
                # the inverse relation `angle_step = 2π / scan_size` so the
                # conversion kernel (which bins by `scan_size`) and the
                # ray-cast kernel (which reads `initializedAngles[i] =
                # i * angle_step`) can't drift apart.
                self._initialize_mapper(
                    math.ceil(2 * np.pi / self.scan_model.angle_step)
                )
            else:
                self._initialize_mapper(scan.ranges.size)  # type: ignore

        # Capture the previous robot pose BEFORE overwriting it. The Bayesian
        # path needs (prev_pose, current_pose) to compute the warp delta.
        previous_robot_pose = self._pose_robot_in_world

        # Calculate new grid pose
        self._pose_robot_in_world = robot_pose
        self.lower_right_corner_pose = transform_point_from_local_to_global(
            self._local_lower_right_corner_point, robot_pose
        )

        if self.config.baysian_update:
            # Compute odometry delta vs the previous frame. On the first
            # frame there's no previous pose, so we pass zeros and the
            # kernel skips the warp.
            if self.processed:
                pose_delta = get_relative_pose(
                    pose_1_in_ref=previous_robot_pose,
                    pose_2_in_ref=robot_pose,
                )
                _pos = pose_delta.get_position()
                position_in_previous_pose = np.asarray(_pos[:2], dtype=np.float32)
                orientation_in_previous_pose = float(pose_delta.get_yaw())
            else:
                position_in_previous_pose = np.zeros(2, dtype=np.float32)
                orientation_in_previous_pose = 0.0

            if self.is_pointcloud:
                # Pointcloud Bayesian path
                scan_occupancy = self.local_mapper.scan_to_grid_baysian(
                    **scan.asdict(),
                    position_in_previous_pose=position_in_previous_pose,
                    orientation_in_previous_pose=orientation_in_previous_pose,
                )
            else:
                # filter out negative range and points outside grid limit
                filtered_ranges = np.minimum(
                    self.config.filter_limit,
                    np.maximum(0.0, scan.ranges),  # type: ignore
                )
                scan_occupancy = self.local_mapper.scan_to_grid_baysian(
                    angles=scan.angles,  # type: ignore
                    ranges=filtered_ranges,
                    position_in_previous_pose=position_in_previous_pose,
                    orientation_in_previous_pose=orientation_in_previous_pose,
                )

        else:
            if self.is_pointcloud:
                scan_occupancy = self.local_mapper.scan_to_grid(
                    **scan.asdict(),
                )
            else:
                # filter out negative range and points outside grid limit
                filtered_ranges = np.minimum(
                    self.config.filter_limit,
                    np.maximum(0.0, scan.ranges),  # type: ignore
                )
                scan_occupancy = self.local_mapper.scan_to_grid(
                    angles=scan.angles,  # type:ignore
                    ranges=filtered_ranges,
                )

        # Update grid (both Bayesian and non-Bayesian return a discreet grid)
        self.grid_data.occupancy = np.copy(scan_occupancy)

        # flag to enable fetching the mapping data
        self.processed = True

        # TODO: robot occupied zone: make it in a separate function and proportional to the actual robot size\
        # self.grid_data["scan_occupancy"][
        #     self.grid_robot_point[0] : self.grid_robot_point[0] + 2,
        #     self.grid_robot_point[1] : self.grid_robot_point[1] + 2,
        # ] = OCCUPANCY_TYPE.OCCUPIED.value
