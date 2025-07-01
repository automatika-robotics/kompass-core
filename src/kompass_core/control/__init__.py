from __future__ import annotations

from enum import Enum
from typing import List, Optional, TypeVar, Type

from kompass_cpp.control import FollowingStatus

from ._base_ import FollowerTemplate
from ._trajectory_ import TrajectoryCostsWeights
from .dvz import DVZ, DVZConfig
from .dwa import DWA, DWAConfig
from .stanley import StanleyConfig, Stanley
from .rgb_follower import VisionRGBFollower, VisionRGBFollowerConfig
from .rgbd_follower import VisionRGBDFollower, VisionRGBDFollowerConfig

ControllerType = FollowerTemplate

T = TypeVar("T", bound="StrEnum")


class StrEnum(Enum):
    """
    Extends Enum class with methods to get all  values and get string value corresponding to given enum value
    """

    @classmethod
    def get_enum(cls: Type[T], __value: str) -> Optional[T]:
        """get_enum.

        :param __value:
        :type __value: str
        :rtype: Optional[StrEnum]
        """
        for enum_member in cls:
            if enum_member.value == __value:
                return enum_member
        return None

    @classmethod
    def values(cls) -> List:
        """values.

        :rtype: List
        """
        return [member.value for member in cls]

    def __str__(self) -> str:
        """
        Gets value of enum

        :return: Enum value
        :rtype: str
        """
        return self.value

    def __repr__(self) -> str:
        """
        Gets value of enum

        :return: Enum value
        :rtype: str
        """
        return self.value


class ControllersID(StrEnum):
    """
    Local planners compute a local plan along with the commands to follow it (work in standalone fashion)
    """

    STANLEY = "Stanley"
    DWA = "DWA"
    DVZ = "DVZ"
    VISION_IMG = "VisionRGBFollower"
    VISION_DEPTH = "VisionRGBDFollower"


ControlClasses = {
    ControllersID.STANLEY: Stanley,
    ControllersID.DVZ: DVZ,
    ControllersID.DWA: DWA,
    ControllersID.VISION_IMG: VisionRGBFollower,
    ControllersID.VISION_DEPTH: VisionRGBDFollower,
}

ControlConfigClasses = {
    ControllersID.STANLEY: StanleyConfig,
    ControllersID.DVZ: DVZConfig,
    ControllersID.DWA: DWAConfig,
    ControllersID.VISION_IMG: VisionRGBFollowerConfig,
    ControllersID.VISION_DEPTH: VisionRGBDFollowerConfig,
}


__all__ = [
    "StrEnum",
    "ControllerType",
    "DVZ",
    "DVZConfig",
    "ControllersID",
    "ControlClasses",
    "ControlConfigClasses",
    "Stanley",
    "StanleyConfig",
    "FollowingStatus",
    "DWA",
    "DWAConfig",
    "TrajectoryCostsWeights",
    "VisionRGBFollower",
    "VisionRGBFollowerConfig",
    "VisionRGBDFollower",
    "VisionRGBDFollowerConfig",
]
