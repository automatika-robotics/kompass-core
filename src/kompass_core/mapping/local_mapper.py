from typing import Optional
from attrs import define, field, validators
import math
import numpy as np
from ..datatypes.pose import PoseData
from ..datatypes.laserscan import LaserScanData

from kompass_cpp.mapping import (
    OCCUPANCY_TYPE,
)

from ..utils.geometry import from_frame1_to_frame2, get_pose_target_in_reference_frame

from .laserscan_model import LaserScanModelConfig
from ..utils.common import BaseAttrs, base_validators


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
    occupancy_prob: np.ndarray = field(init=False)
    scan_occupancy: np.ndarray = field(init=False)
    scan_occupancy_prob: np.ndarray = field(init=False)
    # TODO: Add semantic occupancy
    # semantic_occupancy : np.ndarray = field(init=False)
    # semantic : np.ndarray = field(init=False)

    def __attrs_post_init__(self):
        self.init_scan_data()
        self.occupancy = self.get_initial_grid_data()
        self.occupancy_prob = self.get_initial_grid_data()

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

    def init_scan_data(self, baysian: bool = False) -> None:
        """Initialize Scan Occupancy Layers"""
        self.scan_occupancy = self.get_initial_grid_data()
        if baysian:
            self.scan_occupancy_prob = np.full(
                (self.width, self.height), self.p_prior, dtype=np.float32, order="F"
            )


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
        scan_model_config: LaserScanModelConfig,
        pose_laser_scanner_in_robot: Optional[PoseData] = None,
    ):
        """Initialize a LocalMapper

        :param config: Mapper config
        :type config: MapConfig
        :param scan_model_config: LaserScan model config
        :type scan_model_config: LaserScanModelConfig
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

        self.scan_update_model = scan_model_config

        self.pose_laserscanner_in_robot = pose_laser_scanner_in_robot if pose_laser_scanner_in_robot else PoseData()

        self.laserscan_orientation_in_robot = 2 * np.arctan(
            self.pose_laserscanner_in_robot.qz / self.pose_laserscanner_in_robot.qw
        )

        self.grid_data = GridData(
            width=self.grid_width,
            height=self.grid_height,
            p_prior=self.scan_update_model.p_prior,
        )

        # for bayesian update
        if self.config.baysian_update:
            self.previous_grid_prob_transformed = np.full(
                (self.grid_data.width, self.grid_data.height),
                self.grid_data.p_prior,
                dtype=np.float32,
                order="F",
            )
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
    def probabilistic_occupancy(self) -> np.ndarray:
        """Getter of current grid probabilistic occupancy

        :return: Grid probabilistic layer
        :rtype: np.ndarray
        """
        return self.grid_data.occupancy_prob

    def _initialize_mapper(self, scan_size: int) -> None:
        """Initialize cpp local mapper"""
        try:
            from kompass_cpp.mapping import LocalMapperGPU
            self.local_mapper = LocalMapperGPU(
                grid_height=self.grid_height,
                grid_width=self.grid_width,
                resolution=self.config.resolution,
                laserscan_position=self.pose_laserscanner_in_robot.get_position(),
                laserscan_orientation=self.laserscan_orientation_in_robot,
                scan_size=scan_size,
                max_points_per_line=self.config.max_points_per_line,
            )
        except ImportError:
            from kompass_cpp.mapping import LocalMapper as LocalMapperCpp
            self.local_mapper = LocalMapperCpp(
                grid_height=self.grid_height,
                grid_width=self.grid_width,
                resolution=self.config.resolution,
                laserscan_position=self.pose_laserscanner_in_robot.get_position(),
                laserscan_orientation=self.laserscan_orientation_in_robot,
                **self.scan_update_model.asdict(),
                max_points_per_line=self.config.max_points_per_line,
                max_num_threads=self.config.max_num_threads,
            )

    def _calculate_poses(self, current_robot_pose: PoseData):
        """Calculates 3D global poses of the 4 corners of the grid based on the curren robot position

        :param current_robot_pose: Current robot position in global frame
        :type current_robot_pose: PoseData
        """
        if self.processed:
            # self._pose_robot_in_world has been set already at least once
            # i.e. we have a t+1 state
            # get current shift in translation and orientation of the new center
            # with respect to the previous old center
            pose_current_robot_in_previous_robot = get_pose_target_in_reference_frame(
                reference_pose=self._pose_robot_in_world, target_pose=current_robot_pose
            )
            # new position and orientation with respect to the previous pose
            _position_in_previous_pose = (
                pose_current_robot_in_previous_robot.get_position()
            )
            _orientation_in_previous_pose = (
                pose_current_robot_in_previous_robot.get_yaw()
            )

            self.previous_grid_prob_transformed = (
                self.local_mapper.get_previous_grid_in_current_pose(
                    current_position_in_previous_pose=_position_in_previous_pose[:2],
                    current_orientation_in_previous_pose=_orientation_in_previous_pose,
                    previous_grid_data=self.grid_data.scan_occupancy_prob,
                    unknown_value=self.scan_update_model.p_prior,
                )
            )

        self._pose_robot_in_world = current_robot_pose
        self.lower_right_corner_pose = from_frame1_to_frame2(
            current_robot_pose, self._local_lower_right_corner_point
        )

    def update_from_scan(
        self,
        robot_pose: PoseData,
        laser_scan: LaserScanData,
    ):
        """
        Update the local map using new LaserScan data

        :param robot_pose: Current robot position
        :type robot_pose: PoseData
        :param laser_scan: LaserScan data
        :type laser_scan: LaserScanData
        :param pose_laser_scanner_in_robot: Pose of the sensor w.r.t the robot, defaults to None
        :type pose_laser_scanner_in_robot: Optional[PoseData], optional
        """
        # Get transformation between the previous robot state (pose and grid) w.r.t the current state.

        if not self.processed:
            self._initialize_mapper(laser_scan.ranges.size)

        self._calculate_poses(robot_pose)

        # filter out negative range and points outside grid limit
        filtered_ranges = np.minimum(
            self.config.filter_limit,
            np.maximum(0.0, laser_scan.ranges),
        )

        self.grid_data.init_scan_data(self.config.baysian_update)

        if self.config.baysian_update:
            self.local_mapper.scan_to_grid_baysian(
                angles=laser_scan.angles,
                ranges=filtered_ranges,
                grid_data=self.grid_data.scan_occupancy,
                grid_data_prob=self.grid_data.scan_occupancy_prob,
            )

            # flag to enable fetching the mapping data
            self.processed = True

            # Update grid
            self.grid_data.occupancy = np.copy(self.grid_data.scan_occupancy)

            self.grid_data.occupancy_prob[
                self.grid_data.scan_occupancy_prob > self.scan_update_model.p_prior
            ] = OCCUPANCY_TYPE.OCCUPIED.value
            self.grid_data.occupancy_prob[
                self.grid_data.scan_occupancy_prob == self.scan_update_model.p_prior
            ] = OCCUPANCY_TYPE.UNEXPLORED.value
            self.grid_data.occupancy_prob[
                self.grid_data.scan_occupancy_prob < self.scan_update_model.p_prior
            ] = OCCUPANCY_TYPE.EMPTY.value

        else:
            self.local_mapper.scan_to_grid(
                angles=laser_scan.angles,
                ranges=filtered_ranges,
                grid_data=self.grid_data.scan_occupancy,
            )

            # Update grid
            self.grid_data.occupancy = np.copy(self.grid_data.scan_occupancy)

        # robot occupied zone - TODO: make it in a separate function and proportional to the actual robot size\
        # self.grid_data["scan_occupancy"][
        #     self.grid_robot_point[0] : self.grid_robot_point[0] + 2,
        #     self.grid_robot_point[1] : self.grid_robot_point[1] + 2,
        # ] = OCCUPANCY_TYPE.OCCUPIED.value
