import logging
from typing import Any, List

from .utils.common import set_params_from_yaml
from .utils.geometry import (
    convert_to_0_2pi,
    convert_to_plus_minus_pi,
    distance,
    probability_of_collision,
)
import numpy as np
from .datatypes.obstacles import ObstaclesData
from .datatypes.path import PathPoint, PathSample

from .py_path_tools.executor import PathExecutor
from .models import footprint_types


class MotionCost:
    """
    Basic motion cost class, used for any type of cost
    """

    def __init__(self, weight: float, margin: float):
        """
        Init a motion cost

        :param weight: Weight of the cost in a total cost computation
        :type weight: float
        :param margin: Margin used in the cost computation (if applicable)
        :type margin: float
        """
        self.value = 0.0
        self.weight = weight
        self.margin = margin

    def set_params(self, weight: float, margin: float):
        """
        Sets the motion cost parameters

        :param weight: Weight of the cost in a total cost computation
        :type weight: float
        :param margin: Margin used in the cost computation (if applicable)
        :type margin: float
        """
        self.weight = weight
        self.margin = margin

    def __call__(self) -> float:
        """
        Gets the weighted cost value

        :return: weighted cost
        :rtype: float
        """
        return self.weight * self.value


class CollisionCost(MotionCost):
    """
    Collision Cost class, used for both a static object collision and a dynamic object collision
    """

    def __init__(self, weight: float, margin: float, robot_footprint: Any):
        """
        Init the collision cost

        :param weight: Cost weight
        :type weight: float
        :param margin: Cost margin
        :type margin: float
        :param robot_footprint: Robot footprint
        :type robot_footprint: Any
        """
        try:
            assert any(isinstance(robot_footprint, t) for t in footprint_types)
            self.robot_footprint = robot_footprint
            super().__init__(weight, margin)
        except AssertionError as e:
            logging.error(f"{e}. Provided footprint is not valid")

    def _get_collision_limit_radius(self) -> float:
        """
        Returns the limit of the closest distance between the robot center and obstacle to consider a collision

        :return: The limit radius (m)
        :rtype: float
        """
        return self.robot_footprint.get_radius() + self.margin

    def call(self) -> float:
        return super().__call__()


class StaticCollisionCost(CollisionCost):
    """
    Static objects collision cost computer
    """

    _MAX_POINT_COLLISION_COST = 1.0

    def __init__(self, weight: float, margin: float, robot_footprint: Any):
        super().__init__(weight, margin, robot_footprint)
        self.closest_distance: float = None

    def __call__(
        self, point_x: float, point_y: float, local_map: ObstaclesData
    ) -> float:
        """
        Compute the cost for a given path sample point and a current static map

        :param point_x: X-coordinates of the path sample point under evaluation
        :type point_x: float
        :param point_y: Y-coordinates of the path sample point under evaluation
        :type point_y: float
        :param local_map: Local map
        :type local_map: ObstaclesData

        :return: Static collision value
        :rtype: float
        """
        collision_limit_radius = self._get_collision_limit_radius()
        dist_sqr = (local_map.x_global - point_x) ** 2 + (
            local_map.y_global - point_y
        ) ** 2
        self.closest_distance = min(dist_sqr)
        distances = np.where(
            dist_sqr <= collision_limit_radius**2,
            self._MAX_POINT_COLLISION_COST,
            0,
        )
        return np.sum(distances)

    def update(
        self, path_sample: PathSample, idx: int, local_map: ObstaclesData
    ) -> float:
        """
        Updates the static collision cost for a path sample under evaluation and returns the weighted cost

        :param path_sample: Path sample under evaluation
        :type path_sample: PathSample
        :param idx: Index of the last considered point on the path sample
        :type idx: int
        :param local_map: Local map
        :type local_map: ObstaclesData

        :return: _description_
        :rtype: float
        """
        point_x = path_sample.x_points[idx]
        point_y = path_sample.y_points[idx]
        self.value += self.__call__(point_x, point_y, local_map)
        return self.call()


class DynamicCollisionProbabilityCost(CollisionCost):
    """
    Dynamic objects probability of collision cost computer
    """

    _MAX_POINT_UNCERTAINTY = 0.5

    def __init__(self, weight: float, margin: float, robot_footprint: Any):
        super().__init__(weight, margin, robot_footprint)

    def __call__(
        self,
        point_x: float,
        point_y: float,
        idx: int,
        future_map,
        time_step: float,
        prediction_horizon: float,
    ) -> float:
        """
        Gets a the probability of collision using the predicted local map

        :param point_x: X-coordinates of the path sample point under evaluation
        :type point_x: float
        :param point_y: Y-coordinates of the path sample point under evaluation
        :type point_y: float
        :param idx: Prediction future index
        :type idx: int
        :param predicted_maps: Predicted local map
        :type predicted_maps: ObstaclesData
        :param time_step: Prediction time step (s)
        :type time_step: float
        :param prediction_horizon: Prediction future time horizon (s)
        :type prediction_horizon: float

        :return: Probability of collision with the obstacles on the predicted maps: in [0,1]
        :rtype: float
        """

        pose_uncertainty = self._MAX_POINT_UNCERTAINTY * (
            idx * time_step / prediction_horizon
        )

        robot_inflated_radius = self._get_collision_limit_radius() * (
            1.0 + pose_uncertainty
        )

        prob_of_col = 0.0
        for obs_index in range(len(future_map.obstacle_type)):
            object_inflated_radius = (
                future_map.occupied_zone[obs_index] + self.margin
            ) * (1 + pose_uncertainty)
            prob_of_col = max(
                prob_of_col,
                probability_of_collision(
                    point_x,
                    point_y,
                    future_map.x_global[obs_index],
                    future_map.y_global[obs_index],
                    robot_inflated_radius,
                    object_inflated_radius,
                ),
            )
        return prob_of_col

    def update(
        self,
        path_sample: PathSample,
        idx: int,
        predicted_maps,
        time_step: float,
        prediction_horizon: float,
    ) -> float:
        """
        Updates the dynamic probability of collision cost value and returns the weighted cost

        :param path_sample: Path sample under evaluation
        :type path_sample: PathSample
        :param idx: Index of the last considered point on the path sample
        :type idx: int
        :param predicted_maps: Predicted local maps
        :type predicted_maps: ObstaclesData
        :param time_step: Prediction time step (s)
        :type time_step: float
        :param prediction_horizon: Prediction future time horizon (s)
        :type prediction_horizon: float

        :return: Weighted dynamic probability of collision cost
        :rtype: float
        """
        future_map = predicted_maps[idx]
        point_x = path_sample.x_points[idx]
        point_y = path_sample.y_points[idx]
        self.value = max(
            self.__call__(
                point_x, point_y, idx, future_map, time_step, prediction_horizon
            ),
            self.value,
        )
        return self.call()


class ReferenceCost:
    """
    Reference path tracking cost computer
    """

    PATH_TRACKING_COST = 0
    GOAL_POINT_COST = 1

    _types = {0: List[PathPoint], 1: PathPoint}

    def __init__(
        self,
        displacement_weight: float,
        heading_weight: float,
        reference_type: int = PATH_TRACKING_COST,
    ):
        """
        nit the reference path related costs: displacement cost and heading error cost

        :param displacement_weight: Weight used for the displacement MotionCost
        :type displacement_weight: float
        :param heading_weight: Weight used for the heading MotionCost
        :type heading_weight: float
        :param type: Path tracking cost or end destination cost, defaults to PATH_TRACKING_COST
        :type type: int, optional
        """
        self.reference_type = reference_type
        self.displacement = MotionCost(displacement_weight, 0.0)
        self.heading_error = MotionCost(heading_weight, 0.0)

    def __call__(
        self,
        point_x: float,
        point_y: float,
        point_heading: float,
        reference,
    ) -> tuple[float, float]:
        """
        Returns the cost of diviation from a given reference path or goal point

        :param point_x: X-coordinates of the path sample point under evaluation
        :type point_x: float
        :param point_y: Y-coordinates of the path sample point under evaluation
        :type point_y: float
        :param point_heading:Heading at the path sample point under evaluation
        :type point_heading: float
        :param reference: Reference Path or Reference Goal Point
        :type reference: List[PathPoint] | PathPoint

        :raises AssertionError: If the provided reference does not match the class reference type

        :return: Displacement error, heading error
        :rtype: tuple[float, float]
        """
        try:
            if self.reference_type == self.PATH_TRACKING_COST:
                assert isinstance(reference, List)
                path_point: PathPoint = PathExecutor.get_closest_point(
                    reference, point_x, point_y
                )
            else:
                assert isinstance(reference, PathPoint)
                path_point = reference

            displacement = distance(point_x, path_point.x, point_y, path_point.y)
            heading_error = convert_to_plus_minus_pi(
                point_heading - convert_to_0_2pi(path_point.heading)
            )
            return (displacement, heading_error)

        except AssertionError as e:
            logging.error(
                f"Error: {e}. For reference path cost the input should be of type {self._types[self.reference_type]}"
            )
            raise

    def update(
        self, path_sample: PathSample, idx: int, ref_path: List[PathPoint]
    ) -> tuple[float, float]:
        """
        Updates the reference path following cost value and returns the weighted cost

        :param path_sample: Path sample under evaluation
        :type path_sample: PathSample
        :param idx: Index of the last considered point on the path sample
        :type idx: int
        :param ref_path: Reference path
        :type ref_path: List[PathPoint]

        :return: Weighted reference path following cost
        :rtype: float
        """
        path_point_x = path_sample.x_points[idx]
        path_point_y = path_sample.y_points[idx]
        path_point_heading = path_sample.heading_points[idx]

        displacement, heading_error = self.__call__(
            path_point_x, path_point_y, path_point_heading, ref_path
        )
        self.displacement.value += displacement / 2
        self.heading_error.value = abs(heading_error) / 2
        return self.displacement(), self.heading_error()


class MotionCostsParams:
    """
    Motion costs parameters
    """

    _COLLISION_MARGIN_METERS = 0.1
    _COST_DEFAULT_WEIGHT = 1.0
    _STATIC_COST_DEFAULT_WEIGHT = 5.0
    _ORIENTATION_COST_DEFAULT_WEIGHT = 0.1

    def __init__(self):
        """
        Motion cost parameters data class
        """
        self.static_collision_margin = self._COLLISION_MARGIN_METERS
        self.static_collision_weight = self._STATIC_COST_DEFAULT_WEIGHT
        self.dynamic_collision_margin = self._COLLISION_MARGIN_METERS
        self.dynamic_collision_weight = self._COST_DEFAULT_WEIGHT
        self.goal_lat_err_weight = self._COST_DEFAULT_WEIGHT
        self.goal_heading_err_weight = self._ORIENTATION_COST_DEFAULT_WEIGHT

    def set(
        self,
        static_collision_margin: float,
        static_collision_weight: float,
        dynamic_collision_margin: float,
        dynamic_collision_weight: float,
        goal_lat_err_weight: float,
        goal_heading_err_weight: float,
    ):
        """
        Sets the cost parameters values

        :param static_collision_margin: Safety distance margin with static obstacles by considering a collision cost starting from this distance (m)
        :type static_collision_margin: float
        :param static_collision_weight: Weight of the static collision cost in the overall motion cost
        :type static_collision_weight: float
        :param dynamic_collision_margin: Safety distance margin with dynamic obstacles by considering a collision cost starting from this distance (m)
        :type dynamic_collision_margin: float
        :param dynamic_collision_weight: Weight of the dynamic collision cost in the overall motion cost
        :type dynamic_collision_weight: float
        :param goal_lat_err_weight: Weight of the displacement cost from end destination in the overall motion cost
        :type goal_lat_err_weight: float
        :param goal_heading_err_weight: Weight of the heading error cost from end destination heading in the overall motion cost
        :type goal_heading_err_weight: float
        """
        self.static_collision_margin = static_collision_margin
        self.static_collision_weight = static_collision_weight
        self.dynamic_collision_margin = dynamic_collision_margin
        self.dynamic_collision_weight = dynamic_collision_weight
        self.goal_lat_err_weight = goal_lat_err_weight
        self.goal_heading_err_weight = goal_heading_err_weight

    def set_from_yaml(self, path_to_file: str):
        """
        Sets the values from a given yaml file under 'motion_costs_params'

        :param path_to_file: Path to YAML file
        :type path_to_file: str
        """
        params_map = [
            "static_collision_margin",
            "static_collision_weight",
            "dynamic_collision_margin",
            "dynamic_collision_weight",
            "goal_lat_err_weight",
            "goal_heading_err_weight",
        ]

        set_params_from_yaml(
            self,
            path_to_file,
            param_names=params_map,
            root_name="motion_costs",
            yaml_key_equal_attribute_name=True,
        )
        return
