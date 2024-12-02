from attrs import field, define
from ..utils.common import BaseAttrs, base_validators
import kompass_cpp


@define
class TrajectoryCostsWeights(BaseAttrs):
    """
    Values for the weights of the costs used to evaluate a set of trajectory samples
    If a weight is set to 0.0 then that cost criteria is not taken into consideration

    ```{list-table}
    :widths: 10 10 10 70
    :header-rows: 1
    * - Name
      - Type
      - Default
      - Description

    * - reference_path_distance_weight
      - `float`
      - `3.0`
      - Weight of the reference path cost. Must be between `0.0` and `1e3`.

    * - goal_distance_weight
      - `float`
      - `3.0`
      - Weight of the goal position cost. Must be between `0.0` and `1e3`.
    * - obstacles_distance_weight
      - `float`
      - `1.0`
      - Weight of the obstacles distance cost. Must be between `0.0` and `1e3`.
    * - smoothness_weight
      - `float`
      - `0.0`
      - Weight of the trajectory smoothness cost. Must be between `0.0` and `1e3`.
    * - jerk_weight
      - `float`
      - `0.0`
      - Weight of the trajectory jerk cost. Must be between `0.0` and `1e3`.

    ```

    """

    reference_path_distance_weight: float = field(
        default=3.0, validator=base_validators.in_range(min_value=0.0, max_value=1e3)
    )

    goal_distance_weight: float = field(
        default=3.0, validator=base_validators.in_range(min_value=0.0, max_value=1e3)
    )

    obstacles_distance_weight: float = field(
        default=1.0, validator=base_validators.in_range(min_value=0.0, max_value=1e3)
    )

    smoothness_weight: float = field(
        default=0.0, validator=base_validators.in_range(min_value=0.0, max_value=1e3)
    )

    jerk_weight: float = field(
        default=0.0, validator=base_validators.in_range(min_value=0.0, max_value=1e3)
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
