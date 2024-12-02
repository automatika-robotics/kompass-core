from typing import Union

import fcl
from attrs import define, field
from ...utils.common import BaseAttrs, base_validators

from ...models import RobotGeometry

# Mapping from RobotGeometry to FCL Geometry
fcl_object_geometry = {
    RobotGeometry.Type.BOX: fcl.Box,  # (x, y, z) Axis-aligned box with given side lengths
    RobotGeometry.Type.SPHERE: fcl.Sphere,  # (rad) Sphere with given radius
    RobotGeometry.Type.ELLIPSOID: fcl.Ellipsoid,  # (x, y, z) Axis-aligned ellipsoid with given radis
    RobotGeometry.Type.CAPSULE: fcl.Capsule,  # (rad, lz) Capsule with given radius and height along z-axis
    RobotGeometry.Type.CONE: fcl.Cone,  # (rad, lz) Cone with given radius and cylinder height along z-axis
    RobotGeometry.Type.CYLINDER: fcl.Cylinder,  # (rad, lz) Cylinder with given radius and height along z-axis
}


@define
class FCLConfig(BaseAttrs):
    """
    FCL parameters
    """

    map_resolution: float = field(
        default=0.01, validator=base_validators.in_range(min_value=1e-9, max_value=1e9)
    )

    robot_geometry_type: Union[str, RobotGeometry.Type] = field(
        default=RobotGeometry.Type.BOX,
        converter=lambda value: RobotGeometry.Type.from_str(value),
    )
    robot_geometry_params: tuple[float] = field(default=(1.0, 1.0, 1.0))
