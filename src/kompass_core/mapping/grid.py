import cv2
import numpy as np
from kompass_cpp.mapping import local_to_grid


def get_previous_grid_in_current_pose(
    current_position_in_previous_pose: np.ndarray,
    current_orientation_in_previous_pose: np.float64,
    previous_grid_data: np.ndarray,
    central_point: np.ndarray,
    grid_width: int,
    grid_height: int,
    resolution: float,
    unknown_value: float,
) -> np.ndarray:
    """
    Transform a grid to be centered in egocentric view of the current position given its previous position

    :param current_position_in_previous_pose: Current egocentric position for the transformation
    :type current_position_in_previous_pose: np.ndarray
    :param current_yaw_orientation_in_previous_pose: Current egocentric orientation for the transformation
    :type current_yaw_orientation_in_previous_pose: np.float64
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
    current_center = local_to_grid(
            current_position_in_previous_pose[0:2], central_point, resolution
    )
    # getting the angle from the difference in quaternion vector
    current_orientation_angle = np.degrees(current_orientation_in_previous_pose)

    # create transformation matrix to translate and rotate the center of the grid
    # from previous robot pose to the current robot pose
    transformation_matrix = cv2.getRotationMatrix2D(
        (current_center[1], current_center[0]), -1 * current_orientation_angle, 1.0
    )
    transformation_matrix[0, 2] += 0.5 * grid_height - current_center[1]
    transformation_matrix[1, 2] += 0.5 * grid_width - current_center[0]

    previous_grid_data_not_transformed = np.copy(previous_grid_data)

    # Apply the affine transformation using cv2.warpAffine()
    return cv2.warpAffine(
        previous_grid_data_not_transformed,
        transformation_matrix,
        (grid_height, grid_width),
        borderValue=unknown_value,
    )
