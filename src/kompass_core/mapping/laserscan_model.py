from ..utils.common import BaseAttrs, base_validators
from attrs import define, field


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

    def __attrs_post_init__(self):
        self.p_empty = 1 - self.p_occupied
