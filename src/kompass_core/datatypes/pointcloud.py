from typing import Optional, Union
from attrs import define, field
from ..utils.common import BaseAttrs, base_validators
import numpy as np


@define
class PointCloudData(BaseAttrs):
    """PointCloud data class"""

    data: np.ndarray = field()
    point_step: int = field(validator=base_validators.gt(0))
    row_step: int = field(validator=base_validators.gt(0))
    height: int = field(validator=base_validators.gt(0))
    width: int = field(validator=base_validators.gt(0))
    x_offset: Optional[Union[int, float]] = field(default=None)
    y_offset: Optional[Union[int, float]] = field(default=None)
    z_offset: Optional[Union[int, float]] = field(default=None)
