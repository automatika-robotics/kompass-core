from attrs import define, field
from typing import List
from ..utils.common import BaseAttrs
from kompass_cpp.types import Point3D
import numpy as np


@define
class PointCloudData(BaseAttrs):
    """PointCloud data class"""

    x_points: List[float] = field(default=[])
    y_points: List[float] = field(default=[])
    z_points: List[float] = field(default=[])

    def to_kompass_cpp(self) -> List[Point3D]:
        """Convert to kompass_cpp PointCloud structure

        :return:
        :rtype: List[Point3D]
        """
        return [
            Point3D(x, y, z)
            for x, y, z in zip(self.x_points, self.y_points, self.z_points)
        ]

    @classmethod
    def numpy_to_kompass_cpp(cls, data: np.ndarray) -> List[Point3D]:
        """Convert to kompass_cpp PointCloud structure

        :return:
        :rtype: List[Point3D]
        """
        if data.ndim != 2 or data.shape[1] not in [2, 3]:
            raise ValueError(
                f"Invalid data points of dimension {data.ndim}. Can only process 2D and 3D arrays into PointCloud data"
            )
        if data.shape[1] == 3:
            return [
                Point3D(x, y, 0.05)
                for x, y, _ in data.reshape(
                    -1, 3
                )  # Flatten the array and group values by 3
            ]

        return [Point3D(x, y, 0.05) for x, y in data.reshape(-1, 2)]

    def add(self, x: float, y: float, z: float):
        """Adds new point to the PointCloud data

        :param x: X value
        :type x: float
        :param y: Y value
        :type y: float
        :param z: Z value
        :type z: float
        """
        self.x_points.append(x)
        self.y_points.append(y)
        self.z_points.append(z)
