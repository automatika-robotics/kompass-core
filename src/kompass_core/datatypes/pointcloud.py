from typing import Optional, Union
from attrs import define, field
from ..utils.common import BaseAttrs, base_validators
import numpy as np
from kompass_cpp.utils import read_pcd


def read_points_from_pcd(file_path: str) -> np.ndarray:
    """Read point cloud data from a pcd file.
    :param file_path: Path to pcd file
    :type str
    """
    return read_pcd(file_path)


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
