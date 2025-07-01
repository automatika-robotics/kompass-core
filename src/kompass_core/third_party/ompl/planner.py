from typing import Callable, Optional, Dict

import numpy as np
from attrs import asdict, define, field
from ...utils.common import BaseAttrs, base_validators

import omplpy as ompl
from ...models import Robot, RobotGeometry
from omplpy import base, geometric
from kompass_cpp.planning import OMPL2DGeometricPlanner

from .config import create_config_class, initializePlanners, optimization_objectives


@define
class OMPLGeometricConfig(BaseAttrs):
    """OMPL Geometric Setup Config."""

    planning_timeout: float = field(
        default=5.0, validator=base_validators.in_range(min_value=1e-6, max_value=1e6)
    )  # seconds
    simplification_timeout: float = field(
        default=1.0, validator=base_validators.in_range(min_value=1e-6, max_value=1e6)
    )  # seconds
    goal_tolerance: float = field(
        default=1e-3, validator=base_validators.in_range(min_value=1e-9, max_value=1e3)
    )

    optimization_objective: str = field(
        default=optimization_objectives["length"],
        validator=base_validators.in_(list(optimization_objectives.values())),
    )

    planner_id: str = field(default="ompl.geometric.TRRT")

    optimization_objective_threshold: float = field(
        default=1.0, validator=base_validators.in_range(min_value=1e-9, max_value=1e3)
    )

    map_resolution: float = field(
        default=0.01, validator=base_validators.in_range(min_value=1e-9, max_value=1e9)
    )


class OMPLGeometric:
    """OMPLGeometric."""

    def __init__(
        self,
        robot: Robot,
        log_level: str = "ERROR",
        use_fcl: bool = True,
        config: Optional[OMPLGeometricConfig] = None,
        config_file: Optional[str] = None,
    ):
        """
        Init Open Motion Planning Lib geometric handler

        :param use_fcl: To use collision checking with FCL, defaults to False
        :type use_fcl: bool, optional
        :param config: OMPL configuration
        :type config: OMPLGeometricConfig
        :param fcl_config: Collision checker configuration, defaults to None
        :type fcl_config: FCLConfig | None, optional
        :param map_3d: Planning space map (3D Point Cloud data), defaults to None
        :type map_3d: np.ndarray | None, optional
        """
        # Initialize available planners lists in ompl.geometric and ompl.control
        initializePlanners(log_level)

        self.solution = None

        self._robot = robot

        # Set default config if config is not provided
        if config:
            self._config = config
        else:
            self._config = OMPLGeometricConfig()

        self._use_fcl = use_fcl

        # Create SE2 state space for 2D planning
        self.state_space = base.SE2StateSpace()

        # Validity heck methods dict
        self._custom_validity_check: dict[str, Callable] = {}

        # Setup ompl
        self.available_planners = ompl.geometric.planners.getPlanners()

        # create a simple setup object
        self._ompl_setup = geometric.SimpleSetup(self.state_space)

        # Set the selected planner
        self._set_planner()

        # Setup Path Optimization Objective
        optimization_objective = getattr(base, self._config.optimization_objective)(
            self._ompl_setup.getSpaceInformation()
        )

        self._ompl_setup.setOptimizationObjective(optimization_objective)

        if config_file:
            self.configure(config_file)

        self._cpp_planner: Optional[OMPL2DGeometricPlanner] = OMPL2DGeometricPlanner(
            robot_shape=RobotGeometry.Type.to_kompass_cpp_lib(
                self._robot.geometry_type
            ),
            robot_dimensions=self._robot.geometry_params,
            ompl_setup=self._ompl_setup,
            map_resolution=self._config.map_resolution,
        )

    def configure(
        self,
        config_file: str,
        root_name: Optional[str] = None,
        planner_id: Optional[str] = None,
    ):
        """
        Load config from a configuration file

        :param config_file: Path to file (yaml, json, toml)
        :type config_file: str
        :param root_name: Parent root name of the config in the file 'Parent.ompl' - config must be under 'ompl', defaults to None
        :type root_name: str | None, optional
        """
        if root_name:
            nested_root_name = root_name + ".ompl"
        else:
            nested_root_name = "ompl"
        self._config.from_file(config_file, nested_root_name=nested_root_name)

        if not planner_id:
            planner_id = self._config.planner_id

        elif planner_id in self.available_planners:
            self._config.planner_id = planner_id
        else:
            raise ValueError(
                f"Selected planner is invalid. Available supported planners are: {self.available_planners.keys()}"
            )

        # configure the planner from file
        planner_name = planner_id.split(".")[-1]
        planner_config = self.available_planners[planner_id]
        planner_params = create_config_class(name=planner_name, conf=planner_config)()

        planner_params.from_file(config_file, nested_root_name + "." + planner_name)

        self._set_planner(planner_params, planner_id)

        self.start = False

    @property
    def path_cost(self) -> float:
        """
        Getter of solution path cost using the configured optimization objective

        :return: Path cost
        :rtype: float | None
        """
        return self._cpp_planner.get_cost()

    def setup_problem(
        self,
        map_meta_data: Dict,
        start_x: float,
        start_y: float,
        start_yaw: float,
        goal_x: float,
        goal_y: float,
        goal_yaw: float,
        map_3d: Optional[np.ndarray] = None,
    ):
        """
        Setup a new planning problem with a map, start and goal information

        :param map_meta_data: Global map meta data as a dictionary
        :type map_meta_data: Dict
        :param start_x: X-coordinates for start point on the map (m)
        :type start_x: float
        :param start_y: Y-coordinates for start point on the map (m)
        :type start_y: float
        :param start_yaw: Yaw-coordinates (rotation around z) for start point on the map (rad)
        :type start_yaw: float
        :param goal_x: X-coordinates for goal point on the map (m)
        :type goal_x: float
        :param goal_y: Y-coordinates for goal point on the map (m)
        :type goal_y: float
        :param goal_yaw: Yaw-coordinates (rotation around z) for goal point on the map (rad)
        :type goal_yaw: float
        :param map_3d: 3D array for map PointCloud data, defaults to None
        :type map_3d: np.ndarray | None, optional
        """
        self._set_space_bounds(map_meta_data)
        self._cpp_planner.setup_problem(
            start_x=start_x,
            start_y=start_y,
            start_yaw=start_yaw,
            goal_x=goal_x,
            goal_y=goal_y,
            goal_yaw=goal_yaw,
            map_3d=map_3d,
        )

    @property
    def planner_params(self) -> Optional[ompl.base.ParamSet]:
        """
        Get the selected planner parameters

        :return: _description_
        :rtype: _type_
        """
        if hasattr(self, "planner"):
            return self.planner.params()
        return None

    @planner_params.setter
    def planner_params(self, config: BaseAttrs) -> None:
        """
        Set planner params from config class

        :param config: Planner parameters config
        :type config: BaseAttrs

        :raises AttributeError: If the planner is not selected yet
        """
        if not hasattr(self, "planner"):
            raise AttributeError(
                "Planner is not set yet. select a planner using set_planner before setting parameters"
            )
        inst_dict = asdict(config)
        for key, value in inst_dict.items():
            self.planner.params().setParam(key, str(value))

    @property
    def planner_id(self) -> str:
        """Get Planner Id.

        :rtype: str
        """
        return self._config.planner_id

    def _set_planner(
        self,
        planner_config: Optional[BaseAttrs] = None,
        planner_id: Optional[str] = None,
    ):
        """
        Set planning method to solve the problem

        :param planner_id: Planner ID name
        :type planner_id: str

        :raises ValueError: If planner_id is invalid
        :raises Exception: Any
        """
        try:
            if not planner_id:
                planner_id = self._config.planner_id

            else:
                self._config.planner_id = planner_id

            self.planner = eval(
                "%s(self._ompl_setup.getSpaceInformation())" % planner_id
            )

            if planner_config:
                self.planner_params = planner_config

            self._ompl_setup.setPlanner(self.planner)

        except Exception:
            raise

    def _set_space_bounds(self, map_meta_data: Dict):
        """
        Set planning space bounds from map data

        :param map_meta_data: Map meta data
        :type map_meta_data: Dict
        """
        self._cpp_planner.set_space_bounds_from_map(
            origin_x=map_meta_data["origin_x"],
            origin_y=map_meta_data["origin_y"],
            width=map_meta_data["width"],
            height=map_meta_data["height"],
            resolution=map_meta_data["resolution"],
        )

    def solve(self) -> ompl.geometric.PathGeometric:
        """
        Solve the problem using a given planner

        :return: Solution (Path points) or None if no solution is found
        :rtype: ompl.geometric.PathGeometric
        """
        solved = self._cpp_planner.solve(self._config.planning_timeout)

        if solved:
            self.solution = self._cpp_planner.get_solution()
            return self.solution
        return None
