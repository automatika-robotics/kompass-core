import sys
import cv2
import numpy as np
import numba
from ..datatypes.pose import PoseData
from ..utils.geometry import get_pose_target_in_reference_frame

np.set_printoptions(threshold=sys.maxsize)


@numba.jit(nopython=True)
def fill_around_point(
    grid_data: np.ndarray,
    point: np.ndarray,
    value: np.ndarray,
    padding: np.ndarray,
) -> np.ndarray:
    """
    Fills a region around a point with a given value on a grid for a given padding

    :param grid_data: Grid data to be updated
    :type grid_data: np.ndarray
    :param point: Point coordinates on the grid
    :type point: np.ndarray
    :param value: Value to be assigned to the filled area
    :type value: np.ndarray
    :param padding: Padding size for filling
    :type padding: np.ndarray

    :return: Updated grid
    :rtype: np.ndarray
    """
    least = point - padding
    most = point + padding
    start_index_i = max(0, least[0])
    end_index_i = min(grid_data.shape[0], most[0])

    start_index_j = max(0, least[1])
    end_index_j = min(grid_data.shape[1], most[1])

    grid_data[start_index_i:end_index_i, start_index_j:end_index_j] = value

    return grid_data


@numba.jit(nopython=True)
def check_in_grid(i_j: np.ndarray, grid_width: int, grid_height: int) -> bool:
    """
    Checks if a given pair of indices are within limits

    :param i_j: Indices pair
    :type i_j: np.ndarray
    :param grid_width: Grid width
    :type grid_width: int
    :param grid_height: Grid height
    :type grid_height: int
    :return: If indices are within limits
    :rtype: bool
    """
    return -1 < i_j[0] and i_j[0] < grid_width and -1 < i_j[1] and i_j[1] < grid_height


@numba.jit(nopython=True)
def get_value_in_grid(
    grid_data: np.ndarray,
    point_target_in_grid: np.ndarray,
    default_value: float = 0.0,
) -> float:
    """
    Gets the value of a point on a grid

    :param grid_data: Grid
    :type grid_data: np.ndarray
    :param point_target_in_grid: Target point indices on the grid
    :type point_target_in_grid: np.ndarray
    :param default_value: Default value to return if point is not found on the grid, defaults to 0.0
    :type default_value: float, optional

    :return: Point value on the grid
    :rtype: float
    """
    value = default_value
    if check_in_grid(point_target_in_grid, grid_data.shape[0], grid_data.shape[1]):
        value = grid_data[point_target_in_grid[0], point_target_in_grid[1]]

    return value


@numba.jit(nopython=True)
def grid_to_local(
    point_target_in_grid: np.ndarray,
    central_point: np.ndarray,
    resolution: float,
    height: float = 0,
) -> np.ndarray:
    """
    Transforms a point from grid coordinate (i,j) to the local coordinates frame of the grid (around the central cell) (x,y)

    :param point_target_in_grid: Point indices in the grid
    :type point_target_in_grid: np.ndarray
    :param central_point: Central grid cell coordinates (x,y)
    :type central_point: np.ndarray
    :param resolution: Grid resolution (meter/cell)
    :type resolution: float
    :param height: Default 3D position height, defaults to 0
    :type height: float, optional

    :return: Point coordinates in the local frame (x, y, z)
    :rtype: np.ndarray
    """
    pose_b = np.zeros(3)
    pose_b[0] = central_point[0] - point_target_in_grid[0] * resolution
    pose_b[1] = central_point[1] - point_target_in_grid[1] * resolution
    pose_b[2] = height

    return pose_b


@numba.jit(nopython=True)
def local_to_grid(
    pose_target_in_central: np.ndarray, central_point: np.ndarray, resolution: float
) -> np.ndarray:
    """
     Transforms a point from the local coordinates frame of the grid (around the central cell) (x,y) to the grid indices (i,j)

    :param pose_target_in_central: Point coordinates in the local frame (x, y)
    :type pose_target_in_central: np.ndarray
    :param central_point: Indices of the grid central point (i,j)
    :type central_point: np.ndarray
    :param resolution: Grid resolution (meter/cell)
    :type resolution: float

    :return: Point indices in the grid (i,j)
    :rtype: np.ndarray
    """
    grid_point = np.zeros(2)
    grid_point[0] = central_point[0] + round(pose_target_in_central[0] / resolution)
    grid_point[1] = central_point[1] + round(pose_target_in_central[1] / resolution)

    return grid_point


@numba.jit(nopython=True)
def fill_grid_around_point(
    grid_data: np.ndarray, grid_point: np.ndarray, grid_padding: int, indicator: int
):
    """
    Fill an area around a point on the grid with given padding

    :param grid_data: Grid to be filled
    :type grid_data: np.ndarray
    :param grid_point: Grid point indices (i,j)
    :type grid_point: np.ndarray
    :param grid_padding: Padding to be filled (number of cells)
    :type grid_padding: int
    :param indicator: Value to be assign to filled cells
    :type indicator: int
    """
    # NOTE:  This function should be replaced with slicing directly after checking the non-violation of indexing limits that can be caused by the padding area
    occupied_zone = []
    for i in range(grid_point[0] - grid_padding, grid_point[0] + grid_padding):
        if -1 < i < grid_data.shape[0]:
            for j in range(grid_point[1] - grid_padding, grid_point[1] + grid_padding):
                if -1 < j < grid_data.shape[1]:
                    # grid_index = i + (self.grid_width * j)
                    occupied_zone.append((i, j))
                    grid_data[i, j] = indicator

    if (-1 < grid_point[0] < grid_data.shape[0]) and (
        -1 < grid_point[1] < grid_data.shape[1]
    ):
        grid_data[grid_point[0], grid_point[1]] = indicator


def from_current_to_previous_grid_cell(
    point_target_in_current_grid: np.ndarray,
    pose_previous_in_current: PoseData,
    central_point: np.ndarray,
    resolution: float,
    height: float = 0,
) -> np.ndarray:
    """
    Gets the original coordinates of a point on a transformed grid given its current coordinates

    :param point_target_in_current_grid: Current coordinates of the point on the grid in the current grid position
    :type point_target_in_current_grid: np.ndarray
    :param pose_previous_in_current: Transformation from previous to current
    :type pose_previous_in_current: PoseData
    :param central_point: Coordinates of the current central grid point
    :type central_point: np.ndarray
    :param resolution: Grid resolution (meter/cell)
    :type resolution: float
    :param height: Default 3D position height, defaults to 0
    :type height: float, optional

    :return: Point coordinates in the previous grid position
    :rtype: np.ndarray
    """
    pose_target_in_current_robot = PoseData()
    pose_target_in_current_robot.set_position(
        *grid_to_local(point_target_in_current_grid, central_point, resolution, height)
    )
    pose_target_in_previous_robot = get_pose_target_in_reference_frame(
        reference_pose=pose_previous_in_current,
        target_pose=pose_target_in_current_robot,
    )

    point_target_in_previous_grid = local_to_grid(
        pose_target_in_previous_robot.get_position(), central_point, resolution
    )

    return point_target_in_previous_grid


def get_previous_grid_in_current_pose(
    current_position: np.ndarray,
    current_2d_orientation: np.float64,
    previous_grid_data: np.ndarray,
    central_point: np.ndarray,
    grid_width: int,
    grid_height: int,
    resolution: float,
    unknown_value: float,
) -> np.ndarray:
    """
    Transform a grid to be centered in egocentric view of the current position given its previous position

    :param current_position: Current egocentric position for the transformation
    :type current_position: np.ndarray
    :param current_2d_orientation: Current egocentric orientation for the transformation
    :type current_2d_orientation: np.float64
    :param previous_grid_data: Previous grid data (pre-transformation)
    :type previous_grid_data: np.ndarray
    :param central_point: Coordinates of the central grid point
    :type central_point: np.ndarray
    :param grid_width: Grid size (width)
    :type grid_width: int
    :param grid_height: Grid size (height)
    :type grid_height: int
    :param resolution: Grid resolution (meter/cell)
    :type resolution: float
    :param unknown_value: Value of unknown occupancy (prior value for grid cells)
    :type unknown_value: float

    :return: Transformed grid
    :rtype: np.ndarray
    """
    # the new center on the previous map
    # the previous map needs to move to this center
    current_center = local_to_grid(current_position, central_point, resolution)
    # getting the angle from the difference in quaternion vector
    current_orientation_angle = np.degrees(current_2d_orientation)

    # create transformation matrix to translate and rotate the center of the grid
    # from previous robot pose to the current robot pose
    transformation_matrix = cv2.getRotationMatrix2D(
        (current_center[1], current_center[0]), -1 * current_orientation_angle, 1.0
    )
    transformation_matrix[0, 2] += 0.5 * grid_height - current_center[1]
    transformation_matrix[1, 2] += 0.5 * grid_width - current_center[0]

    # Apply the affine transformation using cv2.warpAffine()

    unknown_value = (
        int(unknown_value) if "int" in str(previous_grid_data.dtype) else unknown_value
    )

    previous_grid_data_not_transformed = np.copy(previous_grid_data)
    previous_grid_data_transformed = cv2.warpAffine(
        previous_grid_data_not_transformed,
        transformation_matrix,
        (grid_height, grid_width),
        borderValue=unknown_value,
    )

    return previous_grid_data_transformed
