import numba
import numpy as np
from ..datatypes.obstacles import OCCUPANCY_TYPE

from .grid import local_to_grid, get_value_in_grid, fill_grid_around_point
from .laserscan_model import update_grid_cell_probability


numba.config.THREADING_LAYER = "omp"


@numba.jit(nopython=True, parallel=True)
def bresenham_line_drawing(pt1: np.ndarray, pt2: np.ndarray) -> np.ndarray:  # noqa: C901
    """
    Bresenham line drawing algorithm to generate a set of points on a line connecting two provided points
    based on http://eugen.dedu.free.fr/projects/bresenham/
    Copyright (c) Meta Platforms, Inc. and its affiliates.

    :param pt1: First point coordinates on 2D grid
    :type pt1: np.ndarray
    :param pt2: Second point coordinates on 2D grid
    :type pt2: np.ndarray

    :return: List of points on discretized line between pt1 to pt2
    :rtype: np.ndarray
    """
    y_step, x_step = 1, 1

    x, y = pt1
    dx, dy = pt2 - pt1

    if dy < 0:
        y_step *= -1
        dy *= -1

    if dx < 0:
        x_step *= -1
        dx *= -1

    line_pts = [[x, y]]

    ddx, ddy = 2 * dx, 2 * dy
    if ddx > ddy:
        errorprev = dx
        error = dx
        for _ in range(int(dx)):
            x += x_step
            error += ddy

            if error > ddx:
                y += y_step
                error -= ddx
                if error + errorprev < ddx:
                    line_pts.append([x, y - y_step])
                elif error + errorprev > ddx:
                    line_pts.append([x - x_step, y])
                else:
                    line_pts.append([x - x_step, y])
                    line_pts.append([x, y - y_step])

            line_pts.append([x, y])

            errorprev = error
    else:
        errorprev = dx
        error = dx
        for _ in range(int(dy)):
            y += y_step
            error += ddx

            if error > ddy:
                x += x_step
                error -= ddy
                if error + errorprev < ddy:
                    line_pts.append([x - x_step, y])
                elif error + errorprev > ddy:
                    line_pts.append([x, y - y_step])
                else:
                    line_pts.append([x - x_step, y])
                    line_pts.append([x, y - y_step])

            line_pts.append([x, y])

            errorprev = error

    return np.array(line_pts).astype(np.int32)


@numba.jit(nopython=True)
def laserscan_to_grid(
    angles: np.ndarray,
    ranges: np.ndarray,
    grid_data: np.ndarray,
    grid_data_prob: np.ndarray,
    central_point: np.ndarray,
    resolution: float,
    laser_scan_pose: np.ndarray,
    laser_scan_orientation: float,
    previous_grid_data_prob: np.ndarray,
    p_prior: float,
    p_empty: float,
    p_occupied: float,
    range_sure: float,
    range_max: float,
    wall_size: float,
    odd_log_p_prior: float,
):
    """
    Processes LaserScan data (angles and ranges) to project on a 2D grid using Bresenham line drawing for each LaserScan beam

    :param angles: LaserScan angles in radians
    :type angles: np.ndarray
    :param ranges: LaserScan ranges in meters
    :type ranges: np.ndarray
    :param robot_orientation: Current robot orientation in the map frame (rad)
    :type robot_orientation: float
    :param grid_data: Current grid data
    :type grid_data: np.ndarray
    :param grid_data_prob: Current probabilistic grid data
    :type grid_data_prob: np.ndarray
    :param central_point: Coordinates of the central point of the grid
    :type central_point: np.ndarray
    :param resolution: Grid resolution
    :type resolution: float
    :param laser_scan_pose: Pose of the LaserScan sensor w.r.t the robot
    :type laser_scan_pose: np.ndarray
    :param previous_grid_data_prob: Previous value of the probabilistic grid data
    :type previous_grid_data_prob: np.ndarray
    :param p_prior: LaserScan model's prior probability value
    :type p_prior: float
    :param p_empty: LaserScan model's probability value of empty cell
    :type p_empty: float
    :param p_occupied: LaserScan model's probability value of occupied cell
    :type p_occupied: float
    :param range_sure: LaserScan model's certain range (m)
    :type range_sure: float
    :param range_max: LaserScan model's max range (m)
    :type range_max: float
    :param wall_size: LaserScan model's padding size when hitting an obstacle (m)
    :type wall_size: float
    :param odd_log_p_prior: Log Odds of the LaserScan model's prior probability value
    :type odd_log_p_prior: float

    :return: List of occupied grid cells
    :rtype: List
    """
    starting_point = local_to_grid(
        laser_scan_pose,
        central_point,
        resolution,
    )

    for angle, current_range in zip(angles, ranges):
        # calculate hit point
        x = laser_scan_pose[0] + current_range * np.cos(laser_scan_orientation + angle)
        y = laser_scan_pose[1] + current_range * np.sin(laser_scan_orientation + angle)
        to_point = local_to_grid(np.array([x, y]), central_point, resolution)

        line_pts = bresenham_line_drawing(starting_point, to_point)
        ray_stopped = True

        last_grid_point = line_pts[0]
        for pt in line_pts:
            x, y = pt

            if -1 < x < grid_data.shape[0] and -1 < y < grid_data.shape[1]:
                ### non-bayesian update ###
                grid_data[x, y] = max(grid_data[x, y], OCCUPANCY_TYPE.EMPTY.value)
                ### bayesian update ###
                # NOTE: when previous grid is transformed to align with the current grid
                # it's not perfectly aligned at the nearest grid cell.
                # Experimentally (empirically), a shift of 1 grid cell reverse the
                # mis-alignment effect. -> check test_local_grid_mapper.py for more info

                SHIFT = 1
                x_prev = x + SHIFT
                y_prev = y + SHIFT
                previous_value = get_value_in_grid(
                    previous_grid_data_prob,
                    np.array([x_prev, y_prev]),
                    default_value=odd_log_p_prior,
                )

                d = pt - starting_point
                distance = np.sqrt(d[0] ** 2 + d[1] ** 2)

                new_value = update_grid_cell_probability(
                    distance=distance,
                    current_range=current_range,
                    odd_log_p_prev=previous_value,
                    resolution=resolution,
                    p_prior=p_prior,
                    p_empty=p_empty,
                    p_occupied=p_occupied,
                    range_sure=range_sure,
                    range_max=range_max,
                    wall_size=wall_size,
                    odd_log_p_prior=odd_log_p_prior,
                )

                grid_data_prob[x, y] = max(grid_data_prob[x, y], new_value)

                # save last point drawn
                last_grid_point[0] = x
                last_grid_point[1] = y
            else:
                ray_stopped = False

        if ray_stopped:
            # Force non-bayesian map to set ending_point to occupied.
            # this guarantee that when visualizing the map
            # the obstacles detected by the laser scan can be visualized instead
            # of having a map that visualize empty and unknown zones only.
            fill_grid_around_point(
                grid_data,
                last_grid_point,
                0,
                OCCUPANCY_TYPE.OCCUPIED,
            )
