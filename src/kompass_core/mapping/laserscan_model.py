import numpy as np
from ..utils.common import BaseAttrs, in_range
from attrs import define, field
import numba


@define
class LaserScanModelConfig(BaseAttrs):
    """
    Implements a basic scan sensor probability model
    To be used for bayesian update of the occupancy map

    based on:

        Fast Processing of Grid Maps using Graphical Multiprocessors - https://www.sciencedirect.com/science/article/pii/S1474667016350807

    and

        Active Bayesian Multi-class Mapping from Range and Semantic Segmentation Observation - https://arxiv.org/abs/2101.01831

    Attributes:
    p_prior         float       prior probability of the model
    range_sure      float       laser scan range where the information coming from sensor is almost certain
    p_occupied      float       probability of area to be occupied
    p_empty         float       probability of area to be empty
    wall_size       float       ending range of the laser scan where its considered occupied (hit by an obstacle)
    """

    p_prior: float = field(
        default=0.6, validator=in_range(min_value=0.0, max_value=1.0)
    )

    p_empty: float = field(init=False)

    p_occupied: float = field(
        default=0.9, validator=in_range(min_value=0.0, max_value=1.0)
    )

    range_sure: float = field(
        default=0.1, validator=in_range(min_value=1e-4, max_value=1e6)
    )

    range_max: float = field(
        default=20.0, validator=in_range(min_value=1e-4, max_value=1e6)
    )

    wall_size: float = field(
        default=0.075, validator=in_range(min_value=1e-4, max_value=1e6)
    )

    odd_log_p_prior: float = field(init=False)

    def __attrs_post_init__(self):
        self.p_empty = 1 - self.p_occupied
        self.odd_log_p_prior = float(np.log(self.p_prior / (1 - self.p_prior)))


@numba.jit(nopython=True)
def update_grid_cell_probability(
    distance: float,
    current_range: float,
    odd_log_p_prev: float,
    resolution: float,
    p_prior: float,
    p_empty: float,
    p_occupied: float,
    range_sure: float,
    range_max: float,
    wall_size: float,
    odd_log_p_prior: float,
) -> float:
    """
    Updates a grid cell occupancy probability using the LaserScanModel

    :param distance: Hit pint distance from the sensor (m)
    :type distance: float
    :param current_range: Scan ray max range (m)
    :type current_range: float
    :param odd_log_p_prev: Log Odds of the previous probability of the grid cell occupancy
    :type odd_log_p_prev: float
    :param resolution: Grid resolution (meter/cell)
    :type resolution: float
    :param p_prior: Prior probability of the model
    :type p_prior: float
    :param p_empty: Empty probability of the model
    :type p_empty: float
    :param p_occupied: Occupied probability of the model
    :type p_occupied: float
    :param range_sure: Certainty range of the model (m)
    :type range_sure: float
    :param range_max: Max range of the sensor (m)
    :type range_max: float
    :param wall_size: Padding size of the model (m)
    :type wall_size: float
    :param odd_log_p_prior: Log Odds of the prior probability
    :type odd_log_p_prior: float

    :return: Current occupancy probability
    :rtype: float
    """
    # get the current sensor probability of being occupied for an area in a given distance from the scanner
    distance = distance * resolution
    range_sure = range_sure
    current_range = current_range - wall_size

    p_f = p_empty if distance < current_range else p_occupied
    delta = 0.0 if distance < range_sure else 1.0

    p_sensor = p_f + (delta * ((distance - range_sure) / range_max) * (p_prior - p_f))

    # get the current bayesian updated probability given its previous probability and sensor probability
    p_curr = odd_log_p_prev + np.log(p_sensor / (1 - p_sensor)) - odd_log_p_prior

    return p_curr
