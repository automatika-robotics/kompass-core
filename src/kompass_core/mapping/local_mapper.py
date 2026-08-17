from typing import List, Optional, Sequence, Union
from attrs import define, field, validators
import math
import numpy as np
from ..datatypes.pose import PoseData

from kompass_cpp.mapping import (
    OCCUPANCY_TYPE,
)
from kompass_cpp.types import SensorConfig

from ..utils.geometry import transform_point_from_local_to_global, get_relative_pose

from ..datatypes.scan_model import ScanModelConfig
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
    # TODO: Add semantic occupancy
    # semantic_occupancy : np.ndarray = field(init=False)
    # semantic : np.ndarray = field(init=False)

    def __attrs_post_init__(self):
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
    bayesian_update: bool = field(default=False)
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
    LocalMapper class produces a grid map around the current robot position using
    laser scan or point cloud data

    Supported layers:
    - Occupancy
    - Probabilistic Occupancy
    """

    def __init__(
        self,
        config: MapConfig,
        scan_model_config: ScanModelConfig,
        pose_laser_scanner_in_robot: Optional[PoseData] = None,
        *,
        sensors: Optional[Sequence[Union[PoseData, SensorConfig]]] = None,
    ):
        """Initialize a LocalMapper

        :param config: Mapper config
        :type config: MapConfig
        :param scan_model_config: LaserScan or PointCloud model config
        :type scan_model_config: ScanModelConfig
        :param pose_laser_scanner_in_robot: Single-sensor mount pose
            (mutually exclusive with ``sensors``)
        :type pose_laser_scanner_in_robot: Optional[PoseData]
        :param sensors: Multi-sensor mode: one mount per sensor, as PoseData
            or kompass_cpp.types.SensorConfig (the latter also carries the
            cloud field encoding). Point clouds are then fused with
            ``update_from_pointclouds``, one cloud per sensor, positional
            pairing
        :type sensors: Optional[Sequence[Union[PoseData, SensorConfig]]]
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

        # multi sensor config (pointclouds)
        if sensors is not None:
            if pose_laser_scanner_in_robot is not None:
                raise ValueError(
                    "Pass either 'pose_laser_scanner_in_robot' (single sensor) "
                    "or 'sensors' (multi-sensor), not both"
                )
            if len(sensors) == 0:
                raise ValueError("'sensors' requires at least one entry")
            if config.bayesian_update:
                raise ValueError(
                    "bayesian_update is not supported with the multi-sensor "
                    "pipeline ('sensors='); use the single-sensor "
                    "'pose_laser_scanner_in_robot' argument instead"
                )
        self._sensors = list(sensors) if sensors is not None else None

        self.pose_laserscanner_in_robot = (
            pose_laser_scanner_in_robot if pose_laser_scanner_in_robot else PoseData()
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
    def probabilistic_occupancy(self) -> np.ndarray:
        """Getter of current grid probabilistic occupancy

        :return: Grid probabilistic layer
        :rtype: np.ndarray
        """
        return self.grid_data.occupancy_prob

    @property
    def num_sensors(self) -> int:
        """Number of configured sensors"""
        return len(self._sensors) if self._sensors is not None else 1

    def _sensor_configs(self) -> List[SensorConfig]:
        """Builds the per-sensor mount configs passed to the cpp mapper.

        The full mount quaternion is forwarded (roll/pitch honored on the
        pointcloud path; the laserscan path consumes the mount as planar).
        """

        def _to_config(sensor) -> SensorConfig:
            if isinstance(sensor, SensorConfig):
                return sensor
            return SensorConfig(
                position=sensor.get_position(),
                rotation=np.array(
                    [sensor.qx, sensor.qy, sensor.qz, sensor.qw],
                    dtype=np.float32,
                ),
            )

        if self._sensors is not None:
            return [_to_config(sensor) for sensor in self._sensors]
        return [_to_config(self.pose_laserscanner_in_robot)]

    def _initialize_mapper(self, scan_size: int) -> None:
        """Initialize cpp local mapper"""
        try:
            from kompass_cpp.mapping import LocalMapperGPU

            self.local_mapper = LocalMapperGPU(
                grid_height=self.grid_height,
                grid_width=self.grid_width,
                resolution=self.config.resolution,
                sensor_configs=self._sensor_configs(),
                is_pointcloud=self.is_pointcloud,
                scan_size=scan_size,
                max_height=self.scan_model.max_height,
                min_height=self.scan_model.min_height,
                range_max=self.scan_model.range_max,
                max_points_per_line=self.config.max_points_per_line,
            )
        except ImportError:
            from kompass_cpp.mapping import LocalMapper as LocalMapperCpp

            # angle_step only used to derive scan_size for pointclouds, not used
            # in cpp ctor
            scan_model_params = self.scan_model.asdict()
            scan_model_params.pop("angle_step", None)
            self.local_mapper = LocalMapperCpp(
                grid_height=self.grid_height,
                grid_width=self.grid_width,
                resolution=self.config.resolution,
                sensor_configs=self._sensor_configs(),
                is_pointcloud=self.is_pointcloud,
                scan_size=scan_size,
                **scan_model_params,
                max_points_per_line=self.config.max_points_per_line,
                max_num_threads=self.config.max_num_threads,
            )

    def _calculate_grid_shift(self, current_robot_pose: PoseData):
        """Calculates 3D global pose shift of the last step probability grid based on the current robot position

        :param current_robot_pose: Current robot position in global frame
        :type current_robot_pose: PoseData
        """
        # self._pose_robot_in_world has been set already at least once
        # i.e. we have a t+1 state
        # get current shift in translation and orientation of the new center
        # with respect to the previous old center
        pose_current_robot_in_previous_robot = get_relative_pose(
            pose_1_in_ref=self._pose_robot_in_world, pose_2_in_ref=current_robot_pose
        )
        # new position and orientation with respect to the previous pose
        _position_in_previous_pose = pose_current_robot_in_previous_robot.get_position()
        _orientation_in_previous_pose = pose_current_robot_in_previous_robot.get_yaw()

        # Shifts the C++ side previous probability grid in place
        # (unknown cells fill with p_prior)
        self.local_mapper.get_previous_grid_in_current_pose(
            current_position_in_previous_pose=_position_in_previous_pose[:2],
            current_orientation_in_previous_pose=_orientation_in_previous_pose,
        )

    def update_from_laserscan(
        self,
        robot_pose: PoseData,
        *,
        ranges: np.ndarray,
        angles: np.ndarray,
    ):
        """
        Update the local map using new 2D laser scan data

        :param robot_pose: Current robot position
        :type robot_pose: PoseData
        :param ranges: Measured range along each angle (m)
        :type ranges: np.ndarray
        :param angles: Angle of each range measurement (rad)
        :type angles: np.ndarray
        """
        if self.num_sensors > 1:
            raise NotImplementedError(
                "Multi-sensor LaserScan fusion is not supported; configure "
                "multiple sensors with point cloud input"
            )
        if not self.processed:
            self.is_pointcloud = False
            self._initialize_mapper(ranges.size)

        self._move_grid_to(robot_pose)

        # filter out negative range and points outside grid limit; float32 is
        # the zero-copy fast path at the binding
        filtered_ranges = np.clip(
            np.asarray(ranges, dtype=np.float32), 0.0, self.config.filter_limit
        )

        self._update_grid(
            angles=np.asarray(angles, dtype=np.float32), ranges=filtered_ranges
        )

    def update_from_pointcloud(
        self,
        robot_pose: PoseData,
        *,
        data: np.ndarray,
        point_step: int,
        row_step: int,
        height: int,
        width: int,
        x_offset: int,
        y_offset: int,
        z_offset: int,
        **_,
    ):
        """
        Update the local map using new point cloud data

        The parameters mirror the sensor_msgs/PointCloud2 layout: the raw buffer
        is handed to the conversion kernel as-is rather than decoded in Python.
        Any further fields a caller's cloud container carries are ignored, so a
        whole container can be splatted in.

        :param robot_pose: Current robot position
        :type robot_pose: PoseData
        :param data: Raw point buffer as a flat byte sequence; a uint8
            numpy array or ``bytes`` (e.g. PointCloud2 ``data``) both cross
            zero-copy
        :type data: Union[np.ndarray, bytes]
        :param point_step: Length of a single point in bytes
        :type point_step: int
        :param row_step: Length of a single row in bytes
        :type row_step: int
        :param height: Number of rows (1 for unorganized clouds)
        :type height: int
        :param width: Number of points per row
        :type width: int
        :param x_offset: Byte offset of the 'x' field within a point
        :type x_offset: int
        :param y_offset: Byte offset of the 'y' field within a point
        :type y_offset: int
        :param z_offset: Byte offset of the 'z' field within a point
        :type z_offset: int
        """
        if self.num_sensors > 1:
            raise RuntimeError(
                "This mapper is configured with multiple sensors; use "
                "update_from_pointclouds instead"
            )
        if not self.processed:
            self.is_pointcloud = True
            # NOTE: `angle_step` is the canonical knob on the Python side; the
            # bin count is derived from it with a ceil so every angle in
            # [0, 2π) lands in a valid bin. The cpp mapping kernels derive their
            # step as `2π / scan_size`, so the two can't drift apart by construction.
            # The effective step is therefore marginally finer than the requested
            # one whenever 2π isn't an exact multiple of it.
            self._initialize_mapper(math.ceil(2 * np.pi / self.scan_model.angle_step))

        self._move_grid_to(robot_pose)

        self._update_grid(
            data=data,
            point_step=point_step,
            row_step=row_step,
            height=height,
            width=width,
            x_offset=x_offset,
            y_offset=y_offset,
            z_offset=z_offset,
        )

    def update_from_pointclouds(
        self,
        robot_pose: PoseData,
        *,
        clouds: Sequence,
    ):
        """
        Update the local map by fusing one point cloud per configured sensor

        ``clouds[i]`` pairs with the i-th configured sensor (positional
        pairing). Each element is a dict and ``None`` means the sensor
        contributed no data this tick.

        :param robot_pose: Current robot position
        :type robot_pose: PoseData
        :param clouds: One cloud (or None) per configured sensor
        :type clouds: Sequence
        """
        if self._sensors is None:
            raise RuntimeError(
                "update_from_pointclouds requires the mapper to be "
                "constructed with 'sensors='"
            )
        if len(clouds) != self.num_sensors:
            raise ValueError(
                f"Expected {self.num_sensors} clouds (one per configured "
                f"sensor), got {len(clouds)}"
            )
        if not self.processed:
            self.is_pointcloud = True
            # NOTE: (see update_from_pointcloud), cpp kernels derive their bin
            # step from scan_size
            scan_size = math.ceil(2 * np.pi / self.scan_model.angle_step)
            self._initialize_mapper(scan_size)

        self._move_grid_to(robot_pose)
        self._update_grid(clouds=list(clouds))

    def _move_grid_to(self, robot_pose: PoseData) -> None:
        """Re-center the grid around a new robot pose

        :param robot_pose: Current robot position
        :type robot_pose: PoseData
        """
        self._pose_robot_in_world = robot_pose
        self.lower_right_corner_pose = transform_point_from_local_to_global(
            self._local_lower_right_corner_point, robot_pose
        )

        # Get transformation between the previous robot state (pose and grid)
        # w.r.t the current state.
        if self.config.bayesian_update and self.processed:
            self._calculate_grid_shift(robot_pose)

    def _update_grid(self, **scan) -> None:
        """Run one scan through the mapper and store the resulting layers

        :param scan: Sensor fields forwarded as-is to the matching kompass_cpp
            overload: ``angles``/``ranges`` for a laser scan, or the raw buffer
            and its layout for a point cloud
        """
        if self.config.bayesian_update:
            scan_occupancy, scan_occupancy_prob = (
                self.local_mapper.scan_to_grid_bayesian(**scan)
            )

            # Update grids in place, copy into preallocated storage
            np.copyto(self.grid_data.occupancy, scan_occupancy)

            # Classify probabilities into occupancy codes
            # (== p_prior is the default)
            np.copyto(
                self.grid_data.occupancy_prob,
                np.select(
                    [
                        scan_occupancy_prob > self.scan_model.p_prior,
                        scan_occupancy_prob < self.scan_model.p_prior,
                    ],
                    [OCCUPANCY_TYPE.OCCUPIED.value, OCCUPANCY_TYPE.EMPTY.value],
                    default=OCCUPANCY_TYPE.UNEXPLORED.value,
                ),
            )

        else:
            # Update grid in place
            np.copyto(self.grid_data.occupancy, self.local_mapper.scan_to_grid(**scan))

        # flag to enable fetching the mapping data
        self.processed = True

        # robot occupied zone - TODO: make it in a separate function and proportional to the actual robot size\
        # self.grid_data["scan_occupancy"][
        #     self.grid_robot_point[0] : self.grid_robot_point[0] + 2,
        #     self.grid_robot_point[1] : self.grid_robot_point[1] + 2,
        # ] = OCCUPANCY_TYPE.OCCUPIED.value
