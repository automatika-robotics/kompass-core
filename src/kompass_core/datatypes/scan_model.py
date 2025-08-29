from ..utils.common import BaseAttrs, base_validators
from attrs import define, field
import numpy as np


@define
class ScanModelConfig(BaseAttrs):
    """
    Configuration for a probabilistic scan sensor model used in occupancy mapping.

    This model supports Bayesian update of occupancy grids using a simplified
    sensor model inspired by:

    - *Fast Processing of Grid Maps using Graphical Multiprocessors*
      (https://www.sciencedirect.com/science/article/pii/S1474667016350807)

    - *Active Bayesian Multi-class Mapping from Range and Semantic Segmentation Observation*
      (https://arxiv.org/abs/2101.01831)

    Attributes
    ----------
    p_prior : float
        Prior probability that any given cell is occupied, before incorporating any sensor measurements.
    p_occupied : float
        Probability used to update a cell when the sensor detects it as occupied.
    p_empty : float
        Probability used to update a cell when the sensor detects it as free. This is automatically computed as `1 - p_occupied`.
    range_sure : float
        Distance from the sensor within which measurements are considered almost certain.
    range_max : float
        Maximum range of the sensor. Measurements beyond this range are ignored.
    wall_size : float
        Thickness (in meters) beyond the measured range where a cell is assumed to be occupied (e.g., for modeling walls or obstacles).
    max_height : float
        Maximum Z-axis height (in meters) of a point to be considered valid for occupancy updates.
    min_height : float
        Minimum Z-axis height (in meters) of a point to be considered valid for occupancy updates.
    """

    p_prior: float = field(
        default=0.6, validator=base_validators.in_range(min_value=0.0, max_value=1.0)
    )

    p_empty: float = field(init=False)

    p_occupied: float = field(
        default=0.9, validator=base_validators.in_range(min_value=0.0, max_value=1.0)
    )

    range_sure: float = field(
        default=0.1, validator=base_validators.in_range(min_value=1e-4, max_value=1e6)
    )

    range_max: float = field(
        default=20.0, validator=base_validators.in_range(min_value=1e-4, max_value=1e6)
    )

    wall_size: float = field(
        default=0.1, validator=base_validators.in_range(min_value=1e-4, max_value=1e6)
    )

    angle_step: float = field(
        default=0.01,
        validator=base_validators.in_range(min_value=1e-6, max_value=np.pi / 4),
    )

    max_height: float = field(
        default=10.0, validator=base_validators.in_range(min_value=-1e2, max_value=1e2)
    )

    min_height: float = field(
        default=-10.0, validator=base_validators.in_range(min_value=-1e2, max_value=1e2)
    )

    def __attrs_post_init__(self):
        self.p_empty = 1 - self.p_occupied
