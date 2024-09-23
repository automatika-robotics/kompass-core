from attrs import field, define
from ..utils.common import BaseAttrs, in_range
import kompass_cpp


@define
class TrajectoryCostsWeights(BaseAttrs):
    """
    Values for the weights of the costs used to evaluate the trajectory samples in the DWA Planner
    If a weight is set to 0.0 then that cost criteria is not taken into consideration

    """

    reference_path_distance_weight: float = field(
        default=3.0, validator=in_range(min_value=0.0, max_value=1e3)
    )

    goal_distance_weight: float = field(
        default=3.0, validator=in_range(min_value=0.0, max_value=1e3)
    )

    obstacles_distance_weight: float = field(
        default=1.0, validator=in_range(min_value=0.0, max_value=1e3)
    )

    smoothness_weight: float = field(
        default=0.0, validator=in_range(min_value=0.0, max_value=1e3)
    )

    jerk_weight: float = field(
        default=0.0, validator=in_range(min_value=0.0, max_value=1e3)
    )

    def to_kompass_cpp(self) -> kompass_cpp.control.TrajectoryCostWeights:
        """
        Convert to kompass_cpp lib config format

        :return: DWA Planner costs weights parameters
        :rtype: kompass_cpp.control.DWACostWeights
        """
        costs_config = kompass_cpp.control.TrajectoryCostWeights()
        costs_config.from_dict(self.asdict())
        return costs_config
