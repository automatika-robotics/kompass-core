from typing import Optional
from attrs import define, field, Factory
import numpy as np
from ..datatypes.pose import PoseData
from ..datatypes.laserscan import LaserScanData
from ..datatypes.obstacles import OCCUPANCY_TYPE, ObstaclesData

from ..utils.geometry import from_frame1_to_frame2, get_pose_target_in_reference_frame

from .grid import get_previous_grid_in_current_pose
from .laserscan_model import LaserScanModelConfig
from .bresenham import laserscan_to_grid
from ..utils.common import BaseAttrs, in_range


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
    odd_log_p_prior: float = field()
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
            OCCUPANCY_TYPE.UNEXPLORED,
            dtype=np.int8,
        )
        return data

    def init_scan_data(self) -> None:
        """Initialize Scan Occupancy Layers"""
        self.scan_occupancy = self.get_initial_grid_data()
        self.scan_occupancy_prob = np.full(
            (self.width, self.height), self.odd_log_p_prior, dtype=np.float64
        )


@define
class GridObstacles(BaseAttrs):
    """Grid Obstacles class"""

    scan: ObstaclesData = field(default=Factory(ObstaclesData))
    semantic: ObstaclesData = field(default=Factory(ObstaclesData))
    total: ObstaclesData = field(default=Factory(ObstaclesData))


@define
class MapConfig(BaseAttrs):
    """
    Local mapper configuration parameters
    """

    width: float = field(default=3.0, validator=in_range(min_value=0.1, max_value=1e2))
    height: float = field(default=3.0, validator=in_range(min_value=0.1, max_value=1e2))
    resolution: float = field(
        default=0.1, validator=in_range(min_value=1e-9, max_value=1e2)
    )
    padding: float = field(
        default=0.0, validator=in_range(min_value=0.0, max_value=10.0)
    )


class LocalMapper:
    """
    LocalMapper class produces a grid map around the current robot position using LaserScanData

    Supported layers:
    - Occupancy
    - Probabilistic Occupancy
    """

    def __init__(self, config: MapConfig, scan_model_config: LaserScanModelConfig):
        """Initialize a LocalMapper

        :param config: Mapper config
        :type config: MapConfig
        :param scan_model_config: LaserScan model config
        :type scan_model_config: LaserScanModelConfig
        """
        self.resolution = config.resolution
        self._scan_occupied_radius = config.resolution

        # turned to true after the first map update is done
        self.processed = False

        self.scan_update_model = scan_model_config

        self._local_lower_right_corner_point = PoseData()
        self._local_lower_right_corner_point.set_position(
            x=-1 * config.width / 2, y=-1 * config.height / 2, z=0
        )

        self.grid_width = int(config.width / self.resolution)
        self.grid_height = int(config.height / self.resolution)

        # TODO: Add robot point to track robot footprint
        # self.grid_robot_point = [
        #     int(self.grid_width / 2) - 1,
        #     int(self.grid_height / 2) - 1,
        # ]

        self._point_central_in_grid = np.array([
            round(self.grid_width / 2) - 1,
            round(self.grid_height / 2) - 1,
        ])

        # current obstacles and grid data
        self._pose_robot_in_world = PoseData()
        # self._origin_pose = PoseData()

        self.odd_log_p_prior = self.scan_update_model.odd_log_p_prior

        self.lower_right_corner_pose = PoseData()
        self.grid_data = GridData(
            width=self.grid_width,
            height=self.grid_height,
            odd_log_p_prior=self.odd_log_p_prior,
        )

        # for bayesian update
        self.previous_grid_prob_transformed = np.copy(
            self.grid_data.scan_occupancy_prob
        )

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

    def _merge_data(self):
        """
        Merge grid occupancy data
        """
        self.grid_data.occupancy = np.maximum(self.grid_data.scan_occupancy, -1)

        UNEXPLORED_THRESHOLD = self.odd_log_p_prior
        self.grid_data.occupancy_prob[
            self.grid_data.scan_occupancy_prob > UNEXPLORED_THRESHOLD
        ] = OCCUPANCY_TYPE.OCCUPIED
        self.grid_data.occupancy_prob[
            self.grid_data.scan_occupancy_prob == UNEXPLORED_THRESHOLD
        ] = OCCUPANCY_TYPE.UNEXPLORED
        self.grid_data.occupancy_prob[
            self.grid_data.scan_occupancy_prob < UNEXPLORED_THRESHOLD
        ] = OCCUPANCY_TYPE.EMPTY

    def _calculate_poses(self, current_robot_pose: PoseData):
        """Calculates 3D global poses of the 4 corners of the grid based on the curren robot position

        :param current_robot_pose: Current robot position in global frame
        :type current_robot_pose: PoseData
        """
        if self.processed:
            # self._pose_robot_in_world has been set already at least once (= we have a t-1 state)
            # get current shift in translation and orientation of the new center
            # with respect to the previous old center
            pose_current_robot_in_previous_robot = get_pose_target_in_reference_frame(
                reference_pose=self._pose_robot_in_world, target_pose=current_robot_pose
            )
            # current position and orientation with respect to the previous pose
            current_position = pose_current_robot_in_previous_robot.get_position()
            current_yaw_orientation = pose_current_robot_in_previous_robot.get_yaw()

            self.previous_grid_prob_transformed = get_previous_grid_in_current_pose(
                current_position=current_position,
                current_2d_orientation=current_yaw_orientation,
                previous_grid_data=self.grid_data.scan_occupancy_prob,
                central_point=self._point_central_in_grid,
                grid_width=self.grid_width,
                grid_height=self.grid_height,
                resolution=self.resolution,
                unknown_value=self.odd_log_p_prior,
            )

        self._pose_robot_in_world = current_robot_pose
        self.lower_right_corner_pose = from_frame1_to_frame2(
            current_robot_pose, self._local_lower_right_corner_point
        )

    def update_from_scan(
        self,
        robot_pose: PoseData,
        laser_scan: LaserScanData,
        pose_laser_scanner_in_robot: Optional[PoseData] = None,
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
        # it's important to recalculate the current poses before doing any update
        # In order to get relationship between previous robot state (pose and grid)
        # with respect to the current state.

        self._calculate_poses(robot_pose)

        if not pose_laser_scanner_in_robot:
            pose_laser_scanner_in_robot = PoseData()

        laserscan_orientation = (
            2
            * np.arctan(pose_laser_scanner_in_robot.qz / pose_laser_scanner_in_robot.qw)
            if pose_laser_scanner_in_robot
            else 0.0
        )

        self.grid_data.init_scan_data()
        # filter out infinity range and negative range
        filtered_ranges = np.minimum(
            self.scan_update_model.range_max, np.maximum(0.0, laser_scan.ranges)
        )

        laserscan_to_grid(
            angles=laser_scan.angles,
            ranges=filtered_ranges,
            grid_data=self.grid_data.scan_occupancy,
            grid_data_prob=self.grid_data.scan_occupancy_prob,
            central_point=self._point_central_in_grid,
            resolution=self.resolution,
            laser_scan_pose=pose_laser_scanner_in_robot.get_position(),
            laser_scan_orientation=laserscan_orientation,
            previous_grid_data_prob=self.previous_grid_prob_transformed,
            **self.scan_update_model.asdict(),
        )

        # robot occupied zone - TODO: make it in a separate function and proportional to the actual robot size\
        # self.grid_data["scan_occupancy"][
        #     self.grid_robot_point[0] : self.grid_robot_point[0] + 2,
        #     self.grid_robot_point[1] : self.grid_robot_point[1] + 2,
        # ] = OCCUPANCY_TYPE.OCCUPIED

        # flag to enable fetching the mapping data
        self.processed = True

        self._merge_data()
