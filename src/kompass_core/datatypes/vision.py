from ..utils.common import BaseAttrs, in_range
from attrs import define, field
from typing import Optional, List
import numpy as np


def _xy_validator(_, attribute, value):
    if len(value) != 2:
        raise ValueError(f"Attribute {attribute} should be a list of length 2: [x, y]")  # noqa: F821
    return


def _xy_optional_validator(_, attribute, value):
    if value and len(value) != 2:
        raise ValueError(
            f"Attribute {attribute} should be None or a list of length 2: [x, y]"
        )  # noqa: F821
    return


@define
class ImageMetaData(BaseAttrs):
    width: int = field()
    height: int = field()
    encoding: str = field(default="rgb8")


@define
class TrackingData(BaseAttrs):
    label: str = field()
    center_xy: List = field(validator=_xy_validator)
    size_xy: List = field(validator=_xy_validator)
    id: int = field()
    img_meta: ImageMetaData = field()
    velocity_xy: Optional[List] = field(default=None, validator=_xy_optional_validator)
    depth: Optional[float] = field(default=None)
