from typing import Tuple
import numpy as np
from kompass_cpp.utils import read_pcd, read_pcd_to_occupancy_grid


def get_points_from_pcd(file_path: str) -> np.ndarray:
    """Read point cloud data from a pcd file.
    :param file_path: Path to pcd file
    :type str
    """
    return read_pcd(file_path)


def get_occupancy_grid_from_pcd(
    file_path: str,
    grid_resolution: float,
    z_ground_limit: float,
    robot_height: float,
) -> Tuple[np.ndarray, list]:
    """Read occupancy grid directly from a pcd file.
    :param file_path: Path to pcd file
    :type str
    :param grid_resolution: Resolution of the grid
    :type float
    :param z_ground_limit: Height limit to consider
    :type float
    :param robot_height: Height of the robot
    :type float
    """
    return read_pcd_to_occupancy_grid(
        file_path, grid_resolution, z_ground_limit, robot_height
    )
