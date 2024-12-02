from enum import Enum
from typing import List, Optional, Union

from .utils import common as CommonUtils
from .utils.common import BaseAttrs, base_validators, set_params_from_yaml
from .utils import geometry as GeometryUtils

import numpy as np
from attrs import Factory, define, field, validators
from .datatypes.path import Point2D

import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle

import kompass_cpp


@define
class MotionModel2DParams(BaseAttrs):
    """MotionModel2DParams."""

    # Motion in x direction
    x_dot_prop_vx: float = field(
        default=1.0,
        validator=base_validators.in_range(min_value=0.0, max_value=1.5),
    )

    x_dot_prop_vy: float = field(
        default=0.0,
        validator=base_validators.in_range(min_value=0.0, max_value=1.5),
    )  # default 0 for non-holonomic motion
    # Motion in y direction
    y_dot_prop_vx: float = field(
        default=1.0,
        validator=base_validators.in_range(min_value=0.0, max_value=1.5),
    )

    y_dot_prop_vy: float = field(
        default=0.0,
        validator=base_validators.in_range(min_value=0.0, max_value=1.5),
    )  # default 0 for non-holonomic motion
    # Rotational motion
    yaw_dot_prop: float = field(
        default=1.0,
        validator=base_validators.in_range(min_value=0.0, max_value=1.5),
    )


class MotionModel2D:
    """MotionModel2D."""

    def __init__(self, params: Optional[MotionModel2DParams] = None) -> None:
        """
        Inits 2D kinematic motion model with calibration parameters

        :param params: _description_, defaults to None
        :type params: MotionModel2DParams | None, optional
        """
        if not params:
            params = MotionModel2DParams()
        self.params = params

    @classmethod
    def x_model(cls, X: tuple, x_dot_prop_vx: float, x_dot_prop_vy: float) -> float:
        """
        Motion model x-axis

        :param X: Model inputs (x_old, yaw_old, v, dt)
        :type X: tuple
        :param a_vx: Multiplicative model parameter
        :type a_vx: float
        :param b_vx: Additive model parameter
        :type b_vx: float

        :return: Output - new x-axis location
        :rtype: float
        """
        x_old, yaw_old, v_x, v_y, dt = X
        x_new = (
            x_old
            + (
                x_dot_prop_vx * v_x * np.cos(yaw_old)
                - x_dot_prop_vy * v_y * np.sin(yaw_old)
            )
            * dt
        )
        return x_new

    @classmethod
    def y_model(cls, X: tuple, y_dot_prop_vx: float, y_dot_prop_vy: float) -> float:
        """
        Motion model y-axis

        :param X: Model inputs (y_old, yaw_old, v, dt)
        :type X: tuple
        :param a_vy: Multiplicative model parameter
        :type a_vy: float
        :param b_vy: Additive model parameter
        :type b_vy: float

        :return: Output - new y-axis location
        :rtype: float
        """
        y_old, yaw_old, v_x, v_y, dt = X
        y_new = (
            y_old
            + (
                y_dot_prop_vx * v_x * np.sin(yaw_old)
                + y_dot_prop_vy * v_y * np.cos(yaw_old)
            )
            * dt
        )
        return y_new

    @classmethod
    def heading_model(cls, X: tuple, yaw_dot_prop: float) -> float:
        """
        Motion model heading

        :param X: Model inputs (yaw_old, omega, dt)
        :type X: tuple
        :param a_omega: Multiplicative model parameter
        :type a_omega: float
        :param b_omega: Additive model parameter
        :type b_omega: float

        :return: Output - new heading
        :rtype: float
        """
        yaw_old, omega, dt = X
        yaw_new: float = GeometryUtils.convert_to_0_2pi(
            yaw_old + (yaw_dot_prop * omega) * dt
        )
        return yaw_new

    def apply(
        self,
        input_state: np.ndarray,
        v_x: float,
        omega: float,
        dt: float,
        v_y: float = 0.0,
    ) -> np.ndarray:
        """
        Applies a linear and angular velocity on a given input state according to the motion model

        :param input_state: Input state [x, y, heading]
        :type input_state: np.ndarray
        :param v_x: Linear velocity in x direction (m/s)
        :type v_x: float
        :param omega: Angular velocity (rad/s)
        :type omega: float
        :param dt: Time step (s)
        :type dt: float
        :param v_y: Linear velocity in y direction (m/s), defaults to 0.0 for non-holonomic robots
        :type v_y: float, Optional

        :return: New state after applying the control [x, y, heading]
        :rtype: np.ndarray
        """
        params = self.params
        output_state = np.zeros(3)
        output_state[0] = self.x_model(
            (input_state[0], input_state[2], v_x, v_y, dt),
            params.x_dot_prop_vx,
            params.x_dot_prop_vy,
        )
        output_state[1] = self.y_model(
            (input_state[1], input_state[2], v_x, v_y, dt),
            params.y_dot_prop_vx,
            params.y_dot_prop_vy,
        )
        output_state[2] = self.heading_model(
            (input_state[2], omega, dt), params.yaw_dot_prop
        )
        return output_state

    def set_params_from_yaml(self, path_to_file: str) -> None:
        """
        Sets the robot testing parameters values from a given yaml file under 'robot'

        :param path_to_file: Path to YAML file
        :type path_to_file: str
        """
        self.params.from_yaml(path_to_file)

    def set_linear_x_params(self, params: List[float]) -> None:
        """
        Sets the parameters of the X direction motion model

        :param params: [x_dot_prop_vx, x_dot_prop_vy] : X(t+1) = X(t) + x_dot_prop_vx * Vx * cos(yaw) + x_dot_prop_vy * Vy * sin(yaw)
        :type params: List[float]
        """
        self.params.x_dot_prop_vx = params[0]
        self.params.x_dot_prop_vy = params[1]

    def set_linear_y_params(self, params: List[float]) -> None:
        """
        Sets the parameters of the Y direction motion model

        :param params: [y_dot_prop_vx, y_dot_prop_vy] : Y(t+1) = Y(t) + y_dot_prop_vx * Vx * cos(yaw) + y_dot_prop_vy * Vy * sin(yaw)
        :type params: List[float]
        """
        self.params.y_dot_prop_vx = params[0]
        self.params.y_dot_prop_vy = params[1]

    def set_angular_params(self, params: List[float]) -> None:
        """
        Sets the parameters of the angular motion model

        :param params: [yaw_dot_prop] : Yaw(t+1) = Yaw(t) + yaw_dot_prop * Omega
        :type params: List[float]
        """
        self.params.yaw_dot_prop = params[0]

    def __str__(self) -> str:
        """__str__.

        :rtype: str
        """
        return f"""Kinematic Model:
        dx/at = {self.params.x_dot_prop_vx:.3f} * V_x * cos(yaw) - {self.params.x_dot_prop_vy:.3f} * V_y * sin(yaw)
        dy/at = {self.params.y_dot_prop_vx:.3f} * V_x * sin(yaw) + {self.params.y_dot_prop_vy:.3f} * V_y * cos(yaw)
        dyaw/at = {self.params.yaw_dot_prop:.3f} * Omega"""


@define(kw_only=True)
class RobotState:
    """
    Robot state class
    """

    model: MotionModel2D = field(default=Factory(MotionModel2D))
    x: float = field(default=0.0)
    y: float = field(default=0.0)
    yaw: float = field(default=0.0)
    speed: float = field(default=0.0)
    vx: float = field(default=0.0)
    vy: float = field(default=0.0)
    omega: float = field(default=0.0)

    def simulate(self, v_x: float, omega: float, dt: float, v_y: float = 0.0) -> None:
        """
        Applies the robot kinematic model and updates the state

        :param v_x: Linear velocity (m/s)
        :type v_x: float
        :param omega: Angular velocity (rad/s)
        :type omega: float
        :param dt: Time step (s)
        :type dt: float
        :param v_y: Linear velocity in y direction (m/s), defaults to 0.0 for non-holonomic robots
        :type v_y: float, Optional
        """
        robot_new_state = self.model.apply(
            np.array([self.x, self.y, self.yaw]), v_x=v_x, v_y=v_y, omega=omega, dt=dt
        )
        # Get motion direction
        dir_robot = np.sign(GeometryUtils.convert_to_plus_minus_pi(robot_new_state[2]))
        # Motion direction on x-axis
        dir_x = np.sign((robot_new_state[0] - self.x)) * dir_robot
        # Motion direction on y-axis
        dir_y = np.sign((robot_new_state[1] - self.y)) * dir_robot
        # Total motion direction
        dir_speed = -1 if (dir_x < 0 and dir_y < 0) else +1
        # Update state
        self.speed = dir_speed * np.sqrt(
            (robot_new_state[0] - self.x) ** 2 + (robot_new_state[1] - self.y) ** 2
        )
        self.x = robot_new_state[0]
        self.y = robot_new_state[1]
        self.yaw = robot_new_state[2]

    def set_from_yaml(self, path_to_file: str) -> None:
        """
        Sets the values from a given yaml file under 'robot'

        :param path_to_file: Path to YAML file
        :type path_to_file: str
        """
        params_map = [
            ("robot_initial_x", "x"),
            ("robot_initial_y", "y"),
            ("robot_initial_heading", "yaw"),
            ("robot_initial_speed", "speed"),
        ]
        set_params_from_yaml(
            self,
            path_to_file,
            param_names=params_map,
            root_name="robot",
        )
        return

    def __str__(self) -> str:
        """__str__.

        :rtype: str
        """
        return f"[x={self.x:.2f}, y={self.y:.2f}, heading={self.yaw:.2f}]"

    def __sub__(self, other_state):
        """
        Difference between two states

        :param other_state: Another robot state
        :type other_state: RobotState

        :return: State difference
        :rtype: RobotState
        """
        return RobotState(
            x=self.x - other_state.x,
            y=self.y - other_state.y,
            yaw=self.yaw - other_state.yaw,
            speed=self.speed - other_state.speed,
        )

    def __sum__(self, other_state):
        """
        Difference between two states

        :param other_state: Another robot state
        :type other_state: RobotState

        :return: State difference
        :rtype: RobotState
        """
        return RobotState(
            x=self.x + other_state.x,
            y=self.y + other_state.y,
            yaw=self.yaw + other_state.yaw,
            speed=self.speed + other_state.speed,
        )

    def __abs__(self) -> float:
        """
        Returns the state distance from origin

        :return: State Distance
        :rtype: float
        """
        return np.sqrt(self.x**2 + self.y**2)

    def __eq__(self, other: object) -> bool:
        """
        Checks if two robot states are identical

        :param other: Another robot state
        :type other: object

        :return: If two states are equal
        :rtype: bool
        """
        if isinstance(other, RobotState):
            return self.x == other.x and self.y == other.y and self.yaw == other.yaw
        else:
            return False

    def __lt__(self, value) -> bool:
        """
        Less than - compares with a RobotState or int or float

        :param value: Another robot state
        :type value: RobotState | int | float

        :raises TypeError: If compared value is not a RobotState or int or float

        :return: If absolute value of the state is less than value
        :rtype: bool
        """
        if isinstance(value, Union[float, int]):
            return self.__abs__() < value
        elif isinstance(value, RobotState):
            return self.__abs__() < value.__abs__()
        else:
            raise TypeError(f"Cannot compare RobotState with type {type(value)}")

    def __gt__(self, value) -> bool:
        """__gt__.

        :param value:
        :rtype: bool
        """
        if isinstance(value, Union[float, int]):
            return self.__abs__() > value
        elif isinstance(value, RobotState):
            return self.__abs__() > value.__abs__()
        else:
            raise TypeError(f"Cannot compare RobotState with type {type(value)}")

    def distance(self, other_state) -> float:
        """
        Returns Euclidean distance between two states

        :param other_state: Another robot state
        :type other_state: RobotState

        :raises TypeError: If other_state is not a RobotState

        :return: Distance between two states
        :rtype: float
        """
        if not isinstance(other_state, RobotState):
            raise TypeError(
                f"Cannot get distance between robot state and type '{type(other_state)}'"
            )
        return abs(self - other_state)

    def front_state_from_center_state(self, robot_radius: float):
        """
        Gets the state of the robot front point using the state of the robot center

        :param center_state: Center point state
        :type center_state: RobotState
        :param robot_radius: Robot radius (m)
        :type robot_radius: float

        :return: Robot front point state
        :rtype: RobotState
        """
        robot_front_pose_data = GeometryUtils.from_frame1_to_frame2_2d(
            position_x_1_in_2=self.x,
            position_y_1_in_2=self.y,
            heading_1_in_2=self.yaw,
            position_target_x_in_1=robot_radius / 2,
            position_target_y_in_1=0.0,
            heading_target_in_1=0.0,
        )

        front_heading = 2 * np.arctan2(
            robot_front_pose_data.qz, robot_front_pose_data.qw
        )

        front_state: RobotState = self
        front_state.x = robot_front_pose_data.x
        front_state.y = robot_front_pose_data.y
        front_state.yaw = front_heading

        return front_state


class CircularFootprint:
    """
    Circular footprint
    """

    def __init__(self, rad=1.0, robot_state: Optional[RobotState] = None):
        """
        Inits a circular robot footprint

        :param R: Radius (m)
        :type R: float
        :param robotState: Current robot state
        :type robotState: RobotState
        """
        self.radius = rad
        self.wheel_base = rad
        state = robot_state if robot_state else RobotState()
        self.center = Point2D(state.x, state.y)

    def set_from_yaml(self, path_to_file: str) -> None:
        """
        Sets the values from a given yaml file under 'robot'

        :param path_to_file: Path to YAML file
        :type path_to_file: str
        """
        set_params_from_yaml(
            self,
            path_to_file,
            param_names=[("robot_radius", "radius")],
            root_name="robot",
        )

    def plt_robot(
        self, x: float, y: float, heading: float, color="blue", ax=None
    ) -> None:
        """
        Plot the robot footprint at given location

        :param x: Robot center x-coordinates (m)
        :type x: float
        :param y: Robot center y-coordinates (m)
        :type y: float
        :param heading: Robot heading (rad)
        :type heading: float
        :param color: Plot color, defaults to 'blue'
        :type color: str, optional
        :param ax: Plot figure axis, defaults to None
        :type ax: _type_, optional
        """
        if not ax:
            ax = plt.gca()

        # Robot direction vector
        dx = np.cos(heading)
        dy = np.sin(heading)

        ax.add_patch(Circle((x, y), self.radius, color=color, alpha=0.5))

        # Calculate the positions of the front wheels
        front_left_wheel_x = x + (self.radius / 2) * dx - (2 * self.radius / 3) * dy
        front_left_wheel_y = y + (self.radius / 2) * dy + (2 * self.radius / 3) * dx

        front_right_wheel_x = x + (self.radius / 2) * dx + (2 * self.radius / 3) * dy
        front_right_wheel_y = y + (self.radius / 2) * dy - (2 * self.radius / 3) * dx

        # Plot the front wheels (circles)
        ax.add_patch(
            Circle(
                (front_left_wheel_x, front_left_wheel_y), self.radius / 4, color="black"
            )
        )
        ax.add_patch(
            Circle(
                (front_right_wheel_x, front_right_wheel_y),
                self.radius / 4,
                color="black",
            )
        )
        ax.plot(x, y, "b+")

    def get_radius(self) -> float:
        """
        Get the radius of a circle containing the robot

        :return: Robot radius
        :rtype: float
        """
        return self.radius


class RectangleFootprint:
    """
    Rectangular footprint
    """

    def __init__(self, width=1.0, length=2.0):
        """
        Inits a regtangular robot footprint

        :param width: Robot width (m), defaults to 2.0
        :type width: float, optional
        :param length: Robot length (m), defaults to 1.0
        :type length: float, optional
        :param robotState: Current robot state, defaults to RobotState()
        :type robotState: RobotState, optional
        """
        self.width = width
        self.length = length
        self.wheel_base = width

    def set_from_yaml(self, path_to_file: str) -> None:
        """
        Sets the values from a given yaml file under 'robot'

        :param path_to_file: Path to YAML file
        :type path_to_file: str
        """
        set_params_from_yaml(
            self,
            path_to_file,
            param_names=[("robot_width", "width"), ("robot_length", "length")],
            root_name="robot",
        )

    def plt_robot(
        self, x: float, y: float, heading: float, color="blue", ax=None
    ) -> None:
        """
        Plot the robot footprint at given location

        :param x: Robot center x-coordinates (m)
        :type x: float
        :param y: Robot center y-coordinates (m)
        :type y: float
        :param heading: Robot heading (rad)
        :type heading: float
        :param color: Plot color, defaults to 'blue'
        :type color: str, optional
        :param ax: Plot figure axis, defaults to None
        :type ax: _type_, optional
        """
        if not ax:
            ax = plt.gca()

        # Robot direction vector
        dx = np.cos(heading)
        dy = np.sin(heading)

        rect_start_x = x - (self.length / 2) * dx + (self.width / 2) * dy
        rect_start_y = y - (self.length / 2) * dy - (self.width / 2) * dx

        # Plot the robot body
        heading_deg = heading * 180 / np.pi
        body_rect = Rectangle(
            (rect_start_x, rect_start_y),
            self.length,
            self.width,
            angle=heading_deg,
            color=color,
            alpha=0.5,
        )
        ax.add_patch(body_rect)

        # Calculate the positions of the front wheels
        front_left_wheel_x = x + (self.length / 3) * dx - (self.width / 2) * dy
        front_left_wheel_y = y + (self.length / 3) * dy + (self.width / 2) * dx

        front_right_wheel_x = x + (self.length / 3) * dx + (self.width / 2) * dy
        front_right_wheel_y = y + (self.length / 3) * dy - (self.width / 2) * dx

        # Plot the front wheels (circles)
        ax.add_patch(
            Circle(
                (front_left_wheel_x, front_left_wheel_y), self.width / 4, color="black"
            )
        )
        ax.add_patch(
            Circle(
                (front_right_wheel_x, front_right_wheel_y),
                self.width / 4,
                color="black",
            )
        )

        ax.plot(x, y, "b+")

    def get_radius(self) -> float:
        """
        Get the radius of a circle containing the robot

        :return: Robot radius
        :rtype: float
        """
        # TODO get the limit taking into account the orientation of the robot along the path
        return GeometryUtils.distance(self.length / 2, 0, self.width / 2, 0)


footprint_types = [CircularFootprint, RectangleFootprint]


class RobotGeometry:
    """Robot Geometry types and parameters"""

    class Type(Enum):
        """Robot Geometry types"""

        BOX = "BOX"
        CYLINDER = "CYLINDER"
        SPHERE = "SPHERE"
        ELLIPSOID = "ELLIPSOID"
        CAPSULE = "CAPSULE"
        CONE = "CONE"

        @classmethod
        def to_kompass_cpp_lib(cls, value) -> kompass_cpp.types.RobotGeometry:
            """to_kompass_cpp_lib.

            :param value:
            :rtype: kompass_cpp.types.RobotGeometry
            """
            return kompass_cpp.types.RobotGeometry.get(value.value)

        @classmethod
        def values(cls) -> List[str]:
            """values.

            :rtype: List[str]
            """
            return [member.value for member in cls]

        @classmethod
        def to_str(cls, enum_value) -> str:
            """
            Return string value corresponding to enum value if exists

            :param enum_value: _description_
            :type enum_value: RobotType | str
            :raises ValueError: If the enum value is not from this class

            :return: String value
            :rtype: str
            """
            if isinstance(enum_value, cls):
                return enum_value.value
            # If the value is already given as a string check if it valid and return it
            elif isinstance(enum_value, str):
                if enum_value in cls.values():
                    return enum_value
            raise ValueError(f"{enum_value} is not a valid RobotGeometry.Type value")

        @classmethod
        def from_str(cls, value: str):
            """
            Return string value corresponding to enum value if exists

            :param enum_value: _description_
            :type enum_value: RobotType | str
            :raises ValueError: If the enum value is not from this class

            :return: String value
            :rtype: str
            """
            if isinstance(value, cls):
                return value
            # If the value is already given as a string check if it valid and return it
            elif isinstance(value, str):
                for enum_value in cls:
                    if value == enum_value.value or value == str(enum_value):
                        return enum_value
                raise ValueError(f"{value} is not a valid RobotGeometry.Type value")

    class ParamsLength(Enum):
        """Robot Geometry parameters length for each type"""

        BOX = 3  # (x, y, z) Axis-aligned box with given side lengths
        CYLINDER = 2  # (rad, lz) Cylinder with given radius and height along z-axis
        SPHERE = 1  # (rad) Sphere with given radius
        ELLIPSOID = 3  # (x, y, z) Axis-aligned ellipsoid with given radis
        CAPSULE = 2  # (rad, lz) Capsule with given radius and height along z-axis
        CONE = 2  # (rad, lz) Cone with given radius and height along z-axis

    @classmethod
    def is_valid_parameters(cls, geometry_type: Type, parameters: np.ndarray) -> bool:
        """
        Verifies that a given set of parameters are valid and suitable for a given geometry type

        :param geometry_type: Robot Geometry Type
        :type geometry_type: Type
        :param parameters: Robot Geometry Parameters
        :type parameters: np.ndarray

        :return: If parameters are valid
        :rtype: bool
        """
        required_length: int = cls.ParamsLength[geometry_type.value].value
        return len(parameters) == required_length and all(
            param > 0 for param in parameters
        )

    @classmethod
    def get_wheelbase(cls, geometry_type: Type, parameters: np.ndarray) -> float:
        """
        Gets the robot wheelbase (distance between the two wheels) from the geometry

        :param geometry_type: Robot Geometry Type
        :type geometry_type: Type
        :param parameters: Robot Geometry Parameters
        :type parameters: np.ndarray

        :return: Robot wheel base if the parameters are valid, else None
        :rtype: Optional[float]
        """
        if not cls.is_valid_parameters(geometry_type, parameters):
            raise ValueError("Invalid parameters for the robot geometry")
        if geometry_type in [
            cls.Type.CONE,
            cls.Type.CYLINDER,
            cls.Type.SPHERE,
            cls.Type.CAPSULE,
        ]:
            # First parameter is the radius -> equivilant to wheelbase
            return parameters[0]
        else:
            # Wheelbase is the distance on robot lateral axis (y-axis)
            return parameters[1]

    @classmethod
    def get_radius(cls, geometry_type: Type, parameters: np.ndarray) -> float:
        """
        Gets the robot radius (distance between the two wheels) from the geometry

        :param geometry_type: Robot Geometry Type
        :type geometry_type: Type
        :param parameters: Robot Geometry Parameters
        :type parameters: np.ndarray

        :return: Robot radius if the parameters are valid, else None
        :rtype: Optional[float]
        """
        if not cls.is_valid_parameters(geometry_type, parameters):
            raise ValueError("Invalid parameters for the robot geometry")
        if geometry_type in [
            cls.Type.CONE,
            cls.Type.CYLINDER,
            cls.Type.SPHERE,
            cls.Type.CAPSULE,
        ]:
            # First parameter is the radius -> equivilant to wheelbase
            return parameters[0]
        else:
            return np.sqrt(parameters[1] + parameters[0]) / 2

    @classmethod
    def get_length(cls, geometry_type: Type, parameters: np.ndarray) -> Optional[float]:
        """
        Gets the robot base length  from the geometry

        :param geometry_type: Robot Geometry Type
        :type geometry_type: Type
        :param parameters: Robot Geometry Parameters
        :type parameters: np.ndarray

        :return: Robot base width if the parameters are valid, else None
        :rtype: Optional[float]
        """
        if not cls.is_valid_parameters(geometry_type, parameters):
            return None
        return parameters[0]

    @classmethod
    def get_footprint(
        cls, geometry_type: Type, parameters: np.ndarray
    ) -> Union[CircularFootprint, RectangleFootprint]:
        """
        Gets a 2D footprint from the geometry

        :param geometry_type: Robot Geometry Type
        :type geometry_type: Type
        :param parameters: Robot Geometry Parameters
        :type parameters: np.ndarray

        :return: 2D footprint
        :rtype: Union[CircularFootprint, RectangleFootprint]
        """
        if geometry_type in [
            cls.Type.CONE,
            cls.Type.CYLINDER,
            cls.Type.SPHERE,
            cls.Type.CAPSULE,
        ]:
            # First parameter is the radius -> equivilant to wheelbase
            return CircularFootprint(rad=parameters[0])
        else:
            # Wheelbase is the distance on robot lateral axis (y-axis)
            return RectangleFootprint(width=parameters[1], length=parameters[0])


class MotionControl:
    """MotionControl."""

    def __init__(
        self, velocity_x: float, velocity_y: float, omega: float, wheel_base: float
    ):
        """__init__.

        :param velocity_x:
        :type velocity_x: float
        :param velocity_y:
        :type velocity_y: float
        :param omega:
        :type omega: float
        :param wheel_base:
        :type wheel_base: float
        """
        self.__v_x = velocity_x
        self.__v_y = velocity_y
        self.__omega = omega
        self.__robot_wheel_base = wheel_base

    def update_ctr(self, *, omega: float, velocity_x: float, velocity_y: float) -> None:
        """
        Update control values

        :param velocity: _description_
        :type velocity: float
        :param steering_angle: _description_
        :type steering_angle: float
        """
        self.__v_x = velocity_x
        self.__v_y = velocity_y
        self.__omega = omega

    @property
    def linear_velocity_x(self) -> float:
        """
        Getter for the linear velocity control for forward motion (x-axis)

        :return: Linear velocity V_x (m/s)
        :rtype: float
        """
        return self.__v_x

    @property
    def linear_velocity_y(self) -> float:
        """
        Getter for the linear velocity control for holonomic motion (y-axis)

        :return: Linear velocity V_y (m/s)
        :rtype: float
        """
        return self.__v_y

    @linear_velocity_x.setter
    def linear_velocity_x(self, __value) -> None:
        """linear_velocity_x.

        :param __value:
        :rtype: None
        """
        self.__v_x = __value

    @linear_velocity_y.setter
    def linear_velocity_y(self, __value) -> None:
        """linear_velocity_y.

        :param __value:
        :rtype: None
        """
        self.__v_y = __value

    @property
    def angular_velocity(self) -> float:
        """
        Getter of the angular velocity control

        :return: Omega (rad/s)
        :rtype: float
        """
        return self.__omega

    @angular_velocity.setter
    def angular_velocity(self, __value) -> None:
        """angular_velocity.

        :param __value:
        :rtype: None
        """
        self.__omega = __value

    @property
    def steering_angle(self) -> float:
        """
        Getter of the angular velocity control

        :return: _description_
        :rtype: _type_
        """
        return np.arctan(self.angular_velocity * self.__robot_wheel_base)

    @steering_angle.setter
    def steering_angle(self, value) -> None:
        """steering_angle.

        :param value:
        :rtype: None
        """
        self.angular_velocity = np.tan(value) / self.__robot_wheel_base


class DifferentialDriveControl(MotionControl):
    """
    Differential drive robot control commands
    """

    def __init__(self, velocity_x: float, omega: float, wheel_base: float):
        """__init__.

        :param velocity_x:
        :type velocity_x: float
        :param omega:
        :type omega: float
        :param wheel_base:
        :type wheel_base: float
        """
        super().__init__(velocity_x, 0.0, omega, wheel_base)

    @classmethod
    def init_zero(cls, wheel_base: float):
        """init_zero.

        :param wheel_base:
        :type wheel_base: float
        :rtype: DifferentialDriveControl
        """
        return DifferentialDriveControl(0.0, 0.0, wheel_base=wheel_base)

    def update_ctr(self, *, omega: float, velocity_x: float, **_) -> None:
        """
        Update control values

        :param velocity: _description_
        :type velocity: float
        :param steering_angle: _description_
        :type steering_angle: float
        """
        super().update_ctr(omega=omega, velocity_x=velocity_x, velocity_y=0.0)

    @property
    def v_right(self) -> float:
        """
        Getter for the linear velocity control for forward motion (x-axis)

        :return: Linear velocity V_x (m/s)
        :rtype: float
        """
        return (
            self.linear_velocity_x
            + (self.__robot_wheel_base) * self.angular_velocity / 2
        )

    @property
    def v_left(self) -> float:
        """
        Getter of the angular velocity control

        :return: _description_
        :rtype: _type_
        """
        return (
            self.linear_velocity_x
            - (self.__robot_wheel_base) * self.angular_velocity / 2
        )


class AckermannControl(MotionControl):
    """
    Ackermann robot control commands
    """

    def __init__(self, velocity_x: float, omega: float, wheel_base: float):
        """__init__.

        :param velocity_x:
        :type velocity_x: float
        :param omega:
        :type omega: float
        :param wheel_base:
        :type wheel_base: float
        """
        super().__init__(velocity_x, 0.0, omega, wheel_base)

    @classmethod
    def init_zero(cls, wheel_base: float):
        """init_zero.

        :param wheel_base:
        :type wheel_base: float
        :rtype: AckermannControl
        """
        return AckermannControl(0.0, 0.0, wheel_base=wheel_base)

    def update_ctr(self, *, omega: float, velocity_x: float, **_) -> None:
        """
        Update control values

        :param velocity: _description_
        :type velocity: float
        :param steering_angle: _description_
        :type steering_angle: float
        """
        super().update_ctr(omega=omega, velocity_x=velocity_x, velocity_y=0.0)

    @property
    def linear_vel(self) -> float:
        """
        Getter for the linear velocity control for forward motion (x-axis)

        :return: Linear velocity V_x (m/s)
        :rtype: float
        """
        return self.linear_velocity_x


class OmniDirectionalControl(MotionControl):
    """OmniDirectionalControl."""

    def __init__(
        self, velocity_x: float, velocity_y: float, omega: float, wheel_base: float
    ):
        """__init__.

        :param velocity_x:
        :type velocity_x: float
        :param velocity_y:
        :type velocity_y: float
        :param omega:
        :type omega: float
        :param wheel_base:
        :type wheel_base: float
        """
        super().__init__(velocity_x, velocity_y, omega, wheel_base)

    @classmethod
    def init_zero(cls, **kwargs):
        """init_zero.

        :param kwargs:
        :rtype: OmniDirectionalControl
        """
        return OmniDirectionalControl(0.0, 0.0, 0.0, **kwargs)


class RobotType(Enum):
    """RobotType."""

    ACKERMANN = "ACKERMANN_ROBOT"
    DIFFERENTIAL_DRIVE = "DIFFERENTIAL_DRIVE_ROBOT"
    OMNI = "OMNI_ROBOT"

    @classmethod
    def values(cls) -> List[str]:
        """values.

        :rtype: List[str]
        """
        return [member.value for member in cls]

    @classmethod
    def to_str(cls, enum_value) -> str:
        """
        Return string value corresponding to enum value if exists

        :param enum_value: _description_
        :type enum_value: RobotType | str
        :raises ValueError: If the enum value is not from this class

        :return: String value
        :rtype: str
        """
        if isinstance(enum_value, RobotType):
            return enum_value.value
        # If the value is already given as a string check if it valid and return it
        elif isinstance(enum_value, str):
            if enum_value in cls.values():
                return enum_value
        raise ValueError(f"{enum_value} is not a valid RobotType value")

    @classmethod
    def to_kompass_cpp_lib(cls, value: str) -> kompass_cpp.control.ControlType:
        """
        Parse to kompass_cpp control type

        :return: Robot Type
        :rtype: kompass_cpp.control_types
        """
        if value == "ACKERMANN_ROBOT":
            return kompass_cpp.control.ControlType.ACKERMANN
        if value == "DIFFERENTIAL_DRIVE_ROBOT":
            return kompass_cpp.control.ControlType.DIFFERENTIAL_DRIVE
        return kompass_cpp.control.ControlType.OMNI


control_types = {
    "ACKERMANN_ROBOT": AckermannControl,
    "DIFFERENTIAL_DRIVE_ROBOT": DifferentialDriveControl,
    "OMNI_ROBOT": OmniDirectionalControl,
}


@define(kw_only=True)
class LinearCtrlLimits(BaseAttrs):
    """Limitations of a linear control (Velocity, Acceleration and Deceleration)
    Deceleration is provided separately as many application requires higher deceleration values: emergency stopping for example. However, a default value if provided equal to the acceleration limit

    max_vel: Maximum velocity [m/s]
    max_acc: Maximum acceleration  [m/s^2]
    max_decel: Maximum deceleration, added separately to allow the robot to have a faster stopping [m/s^2]
    """

    max_vel: float = field(validator=validators.ge(0.0))  # m/s
    max_acc: float = field(validator=validators.ge(0.0))  # m/s^2
    max_decel: float = field(validator=validators.ge(0.0))  # m/s^2


@define(kw_only=True)
class AngularCtrlLimits(BaseAttrs):
    """Limitations of angular control (Velocity, Angle, Acceleration and Deceleration)"""

    max_vel: float = field(validator=validators.ge(0.0))
    max_steer: float = field(validator=validators.ge(0.0))
    max_acc: float = field(validator=validators.ge(0.0))
    max_decel: float = field(validator=validators.ge(0.0))


@define(kw_only=True)
class RobotCtrlLimits(BaseAttrs):
    """Robot 2D movement control limits
    Defaults to an Ackermann model with lateral control limits equal zero
    """

    vx_limits: LinearCtrlLimits = field()
    omega_limits: AngularCtrlLimits = field()
    vy_limits: LinearCtrlLimits = field(
        default=LinearCtrlLimits(max_vel=0.0, max_acc=0.0, max_decel=0.0)
    )

    def to_kompass_cpp_lib(self) -> kompass_cpp.control.ControlLimitsParams:
        """
        Get the control limits parameters transferred to Robotctrl library format

        :return: 2D control limits
        :rtype: kompass_cpp.control.ctr_limits_params
        """
        return kompass_cpp.control.ControlLimitsParams(
            vel_x_ctr_params=self.linear_to_kompass_cpp_lib(self.vx_limits),
            vel_y_ctr_params=self.linear_to_kompass_cpp_lib(self.vy_limits),
            omega_ctr_params=self.angular_to_kompass_cpp_lib(),
        )

    def linear_to_kompass_cpp_lib(
        self, linear_limits: LinearCtrlLimits
    ) -> kompass_cpp.control.LinearVelocityControlParams:
        """
        Get linear velocity control limits parameters transfered to Robotctrl library format

        :return: Linear forward velocity Vx parameters
        :rtype: kompass_cpp.control.linear_vel_x_params
        """
        return kompass_cpp.control.LinearVelocityControlParams(
            max_vel=linear_limits.max_vel,
            max_acc=linear_limits.max_acc,
            max_decel=linear_limits.max_decel,
        )

    def angular_to_kompass_cpp_lib(
        self,
    ) -> kompass_cpp.control.AngularVelocityControlParams:
        """
        Get Omega control limits parameters transfered to Robotctrl library format

        :return: Angular velocity Omega parameters
        :rtype: kompass_cpp.control.angular_vel_params
        """
        return kompass_cpp.control.AngularVelocityControlParams(
            max_omega=self.omega_limits.max_vel,
            max_ang=self.omega_limits.max_steer,
            max_acc=self.omega_limits.max_acc,
            max_decel=self.omega_limits.max_decel,
        )


@define(kw_only=True)
class Robot:
    """
    Robot Class
    """

    robot_type: Union[RobotType, str] = field(
        converter=lambda value: RobotType.to_str(value)
    )
    geometry_type: RobotGeometry.Type = field()
    geometry_params: np.ndarray = field()
    state: RobotState = field(default=Factory(RobotState))
    control: MotionControl = field(init=False)

    def __attrs_post_init__(self):
        """
        Setup control post init

        :raises ValueError: If given robot geometry type and params are incompatible
        """
        # Check that geometry type and params are compatible
        if not RobotGeometry.is_valid_parameters(
            self.geometry_type, self.geometry_params
        ):
            raise ValueError(
                f"Robot geometry parameters '{self.geometry_params}' are incompatible with given robot geometry {self.geometry_type}. Requires '{RobotGeometry.ParamsLength[self.geometry_type.value].value}' strictly positive parameters"
            )
        # Set inital zero control
        self.control = control_types[self.robot_type].init_zero(
            wheel_base=RobotGeometry.get_wheelbase(
                self.geometry_type, self.geometry_params
            )
        )

    @property
    def radius(self) -> float:
        """
        Gets the robot radius

        :return: Radius (meters)
        :rtype: float
        """
        return RobotGeometry.get_radius(self.geometry_type, self.geometry_params)

    @property
    def wheelbase(self) -> float:
        """
        Gets the robot wheelbase

        :return: Wheelbase (meters)
        :rtype: float
        """
        return RobotGeometry.get_wheelbase(self.geometry_type, self.geometry_params)

    @property
    def footprint(self) -> Union[CircularFootprint, RectangleFootprint]:
        """
        Gets the robot footprint

        :return: Footprint
        :rtype: Union[CircularFootprint, RectangleFootprint]
        """
        return RobotGeometry.get_footprint(self.geometry_type, self.geometry_params)

    def set_state(self, x: float, y: float, yaw: float, speed: float) -> None:
        """
        Set the current robot state

         :param x: Robot x-coordinates (m)
        :type x: float
        :param y: Robot y-coordinates (m)
        :type y: float
        :param yaw: Robot orientation (rad)
        :type yaw: float
        """
        self.state = RobotState(x=x, y=y, yaw=yaw, speed=speed)

    def set_control(
        self, velocity_x: float = 0.0, velocity_y: float = 0.0, omega: float = 0.0
    ) -> None:
        """
        Sets the robot control command based on the robot type

        :param control_1: linear velocity if ACKERMANN_ROBOT, linear velocity for left wheel if DIFFERENTIAL_DRIVE_ROBOT
        :type control_1: float
        :param control_2: steering angle if ACKERMANN_ROBOT, linear velocity for right wheel if DIFFERENTIAL_DRIVE_ROBOT
        :type control_2: float
        """
        self.control.update_ctr(
            velocity_x=velocity_x, velocity_y=velocity_y, omega=omega
        )

    def set_model(self, motion_model: MotionModel2D) -> None:
        """
        Set the robot 2D motion model

        :param motion_model: Motion model
        :type motion_model: MotionModel2D
        """
        self.state.model = motion_model

    def get_state(self, dt: float) -> RobotState:
        """
        Applies the sat control to the robot model and get the updated state

        :param dt: Time step (s)
        :type dt: float
        :return: Updated state
        :rtype: RobotState
        """
        # Update state using kinematics equations
        self.state.simulate(
            v_x=self.control.linear_velocity_x,
            v_y=self.control.linear_velocity_y,
            omega=self.control.angular_velocity,
            dt=dt,
        )

        return self.state
