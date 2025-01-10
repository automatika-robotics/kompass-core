import json
import logging
import math
import os
import random
import numpy as np
import pytest
from kompass_core.datatypes.pose import PoseData
from kompass_core.datatypes.laserscan import LaserScanData
from kompass_cpp.mapping import OCCUPANCY_TYPE

from kompass_core.mapping import LocalMapper, MapConfig, LaserScanModelConfig
from kompass_core.utils.visualization import visualize_grid


def get_random_pose(min_range: float = -100.0, max_range: float = 100.0) -> PoseData:
    """
    Get a random pose in space
    :return: pose described with position and orientation
    :rtype: PoseData
    """
    p = PoseData()
    p.x = random.uniform(min_range, max_range)
    p.y = random.uniform(min_range, max_range)
    p.z = random.uniform(min_range, max_range)
    p.qw = random.uniform(-1, 1)
    p.qx = random.uniform(-1, 1)
    p.qy = random.uniform(-1, 1)
    p.qz = random.uniform(-1, 1)

    return p


@pytest.fixture
def pose_robot_in_world() -> PoseData:
    """
    get random pose of the robot in the 3D world

    :return: robot pose in the 3D world
    :rtype: PoseData
    """
    pose_robot_in_world = get_random_pose()
    return pose_robot_in_world


@pytest.fixture
def logs_test_dir() -> str:
    """
    get root test directory

    :return:    log direcotry
    :rtype:     str
    """
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    logs_test_relative_path = "tests/logs/"
    log_test_absolute_path = os.path.join(root_dir, logs_test_relative_path)
    os.makedirs(log_test_absolute_path, exist_ok=True)

    return log_test_absolute_path


@pytest.fixture
def local_mapper() -> LocalMapper:
    """
    get a local mapper instance

    :return: local mapper instance
    :rtype: LocalMapper
    """

    mapper_config = MapConfig(width=3.0, height=3.0, padding=0.0, resolution=0.05)

    scan_model_config = LaserScanModelConfig(
        p_prior=0.5, p_occupied=0.9, range_sure=0.1, range_max=20.0, wall_size=0.075
    )

    local_mapper = LocalMapper(
        config=mapper_config, scan_model_config=scan_model_config
    )

    return local_mapper


@pytest.fixture(
    params=[
        "out_of_grid",
        "circle_in_grid",
        "circle_at_edge",
        "at_45_deg_only",
        "random_in_grid",
        "continuous",
        "random",
    ]
)
def range_option(request):
    """
    option for the range of laser scan
    """
    return request.param


@pytest.fixture
def laser_scan_data(local_mapper: LocalMapper, range_option: str) -> LaserScanData:
    """
    Different scenarios for laserscan data read by the LiDAR.

    :param      local_mapper: local mapper to build the map
    :type       local_mapper: LocalMapper
    :param      range_option: range scenarios
    :type       range_option: str

    :return:    laser scan data created
    :rtype:     LaserScanData
    """
    dir_name = os.path.dirname(os.path.abspath(__file__))
    json_file_path = os.path.join(dir_name, "resources/mapping/laserscan_data.json")
    data = json.load(open(json_file_path))

    laser_scan_data = LaserScanData()
    laser_scan_data.angle_min = data["angle_min"]
    laser_scan_data.angle_max = data["angle_max"]
    laser_scan_data.angle_increment = data["angle_increment"]
    laser_scan_data.time_increment = data["time_increment"]
    laser_scan_data.scan_time = data["scan_time"]
    laser_scan_data.range_min = data["range_min"]
    laser_scan_data.range_max = data["range_max"]

    angles_size = np.arange(
        laser_scan_data.angle_min,
        laser_scan_data.angle_max,
        laser_scan_data.angle_increment,
    ).shape[0]

    laser_scan_data.intensities = [0.0] * angles_size
    width = local_mapper.grid_width * local_mapper.config.resolution
    height = local_mapper.grid_height * local_mapper.config.resolution
    max_range_quarter = 0.25 * min(width, height)
    max_range_half = 0.5 * min(width, height)
    min_range_from_robot = local_mapper.config.resolution * 2.0
    angle_increment_45 = 0.785398

    if range_option == "out_of_grid":
        map_half_diagonal_in_meter = math.sqrt(math.pow(width, 2) + math.pow(height, 2))

        laser_scan_data.ranges = np.array([map_half_diagonal_in_meter] * angles_size)
    elif range_option == "circle_in_grid":
        laser_scan_data.ranges = np.array([max_range_quarter] * angles_size)

    elif range_option == "circle_at_edge":
        laser_scan_data.ranges = np.array([max_range_half] * angles_size)

    elif range_option == "random_in_grid":
        laser_scan_data.ranges = np.random.uniform(
            low=min_range_from_robot, high=max_range_quarter, size=angles_size
        )
    elif range_option == "at_45_deg_only":
        laser_scan_data.angle_increment = angle_increment_45  # 45 deg angles only
        angles_size = np.arange(
            laser_scan_data.angle_min,
            laser_scan_data.angle_max,
            laser_scan_data.angle_increment,
        ).shape[0]
        laser_scan_data.angles = np.arange(
            laser_scan_data.angle_min,
            laser_scan_data.angle_max,
            laser_scan_data.angle_increment,
        )
        laser_scan_data.ranges = np.array([max_range_quarter] * angles_size)
        laser_scan_data.ranges[0] = 0.0
        laser_scan_data.ranges[1] = 0.1

    elif range_option == "continuous":
        min_obstacle_radius = 10
        max_obstacle_radius = 20
        laser_scan_data.ranges = []
        i = 0
        laser_scan_data.ranges = np.zeros(angles_size)
        while i < angles_size:
            c = random.randint(min_obstacle_radius, max_obstacle_radius)
            r = random.uniform(min_range_from_robot, max_range_half)
            c = c if c + i <= angles_size else angles_size - i
            laser_scan_data.ranges[i] = r
            i += c

        assert laser_scan_data.ranges.size == angles_size
    else:  # random
        laser_scan_data.ranges = np.random.uniform(
            low=min_range_from_robot,
            high=local_mapper.scan_update_model.range_max,
            size=angles_size,
        )

    return laser_scan_data


def test_update_from_scan(
    local_mapper: LocalMapper,
    laser_scan_data: LaserScanData,
    pose_robot_in_world: PoseData,
    logs_test_dir: str,
    range_option: str,
):
    """
    given laser scan data, get the obstacles detected and compare if they are
    equals to the filled grid cells.
    """
    local_mapper.update_from_scan(
        pose_robot_in_world,
        laser_scan_data,
    )

    number_occupied_cells = np.count_nonzero(
        local_mapper.grid_data.occupancy == OCCUPANCY_TYPE.OCCUPIED.value
    )
    # log visualization for grid
    visualize_grid(
        local_mapper.grid_data.occupancy,
        scale=100,
        show_image=False,
        save_file=os.path.join(logs_test_dir, f"grid_occupancy_{range_option}.jpg"),
    )

    logging.info(f"number_occupied_cells: {number_occupied_cells}")


@pytest.fixture
def laser_scan_data_fixed() -> LaserScanData:
    """
    fixed laser scan data

    :return:    laser scan data created
    :rtype:     LaserScanData
    """
    dir_name = os.path.dirname(os.path.abspath(__file__))
    json_file_path = os.path.join(dir_name, "resources/mapping.laserscan_data.json")
    data = json.load(open(json_file_path))

    laser_scan_data = LaserScanData()
    laser_scan_data.angle_min = data["angle_min"]
    laser_scan_data.angle_max = data["angle_max"]
    laser_scan_data.angle_increment = data["angle_increment"]
    laser_scan_data.time_increment = data["time_increment"]
    laser_scan_data.scan_time = data["scan_time"]
    laser_scan_data.range_min = data["range_min"]
    laser_scan_data.range_max = data["range_max"]

    angles_size = np.arange(
        laser_scan_data.angle_min,
        laser_scan_data.angle_max,
        laser_scan_data.angle_increment,
    ).shape[0]

    laser_scan_data.intensities = [0.0] * angles_size

    laser_scan_data.ranges = np.array([1.0] * angles_size)

    return laser_scan_data
