from typing import Callable, Optional, Dict

import numpy as np
from attrs import asdict, define, field
from ...utils.common import BaseAttrs, base_validators
from ...utils.geometry import convert_to_plus_minus_pi

import ompl
from ...models import Robot, RobotState
from ompl import base, geometric

from ..fcl.collisions import FCL
from ..fcl.config import FCLConfig
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

    log_level: str = field(
        default="WARN",
        validator=base_validators.in_(["DEBUG", "INFO", "WARN", "ERROR"]),
    )

    map_resolution: float = field(
        default=0.01, validator=base_validators.in_range(min_value=1e-9, max_value=1e9)
    )


class OMPLGeometric:
    """OMPLGeometric."""

    def __init__(
        self,
        robot: Robot,
        use_fcl: bool = True,
        config: Optional[OMPLGeometricConfig] = None,
        config_file: Optional[str] = None,
        map_3d: Optional[np.ndarray] = None,
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
        initializePlanners()

        self.solution = None

        self._robot = robot

        # Set default config if config is not provided
        if config:
            self._config = config
        else:
            self._config = OMPLGeometricConfig()

        # Set log level
        ompl.util.setLogLevel(
            getattr(ompl.util.LogLevel, f"LOG_{self._config.log_level}")
        )

        self._use_fcl = use_fcl

        # Create SE2 state space for 2D planning
        self.state_space = base.SE2StateSpace()

        # Validity heck methods dict
        self._custom_validity_check: dict[str, Callable] = {}

        # Setup ompl
        self.available_planners = ompl.geometric.planners.getPlanners()

        # create a simple setup object
        self.setup = geometric.SimpleSetup(self.state_space)

        # Configure FCL
        if use_fcl:
            self.__init_fcl(map_3d)
            is_state_valid = self._validity_checker_with_fcl

        else:
            is_state_valid = self._validity_checker

        # Add validity method to ompl setup without custom validity checkers
        self.setup.setStateValidityChecker(is_state_valid)

        # Set the selected planner
        self.set_planner()

        if config_file:
            self.configure(config_file)

    def __init_fcl(self, map_3d: Optional[np.ndarray] = None):
        """
        Setup FCL

        :param map_3d: Map data
        :type map_3d: np.ndarray
        """
        fcl_config = FCLConfig(
            map_resolution=self._config.map_resolution,
            robot_geometry_type=self._robot.geometry_type,
            robot_geometry_params=self._robot.geometry_params.tolist(),
        )
        if not hasattr(self, "fcl"):
            # setup collision check
            if map_3d is not None:
                self.fcl = FCL(fcl_config, map_3d)
            else:
                self.fcl = FCL(fcl_config)
        else:
            self.fcl._setup_from_config()

    def configure(
        self,
        yaml_file: str,
        root_name: Optional[str] = None,
        planner_id: Optional[str] = None,
    ):
        """
        Load config from a yaml file

        :param yaml_file: Path to .yaml fila
        :type yaml_file: str
        :param root_name: Parent root name of the config in the file 'Parent.ompl' - config must be under 'ompl', defaults to None
        :type root_name: str | None, optional
        """
        if root_name:
            nested_root_name = root_name + ".ompl"
        else:
            nested_root_name = "ompl"
        self._config.from_yaml(yaml_file, nested_root_name=nested_root_name)

        # Set LOG level
        ompl.util.setLogLevel(
            getattr(ompl.util.LogLevel, f"LOG_{self._config.log_level}")
        )

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

        planner_params.from_yaml(yaml_file, nested_root_name + "." + planner_name)

        self.set_planner(planner_params, planner_id)

        self.start = False

        # configure FCL if it is used
        if self._use_fcl:
            self.__init_fcl()

    def clear(self):
        """
        Clears planning setup
        """
        self.setup.clear()

    @property
    def path_cost(self) -> float | None:
        """
        Getter of solution path cost using the configured optimization objective

        :return: Path cost
        :rtype: float | None
        """
        if self.solution:
            optimization_objective = getattr(base, self._config.optimization_objective)(
                self.setup.getSpaceInformation()
            )
            cost = self.solution.cost(optimization_objective)
            return cost.value()
        return None

    def get_cost_using_objective(self, objective_key: str) -> float | None:
        """
        Get solution cost using a specific objective
        This is used to get the cost using an objective other than the configured objective
        To get the cost using the default objective use self.path_cost directly

        :param objective_key: _description_
        :type objective_key: str
        :raises KeyError: Unknown objective name
        :return: _description_
        :rtype: _type_
        """
        if not self.solution:
            return None
        if objective_key in optimization_objectives:
            optimization_objective = getattr(base, objective_key)(
                self.setup.getSpaceInformation()
            )
            cost = self.solution.cost(optimization_objective)
            return cost.value()
        raise KeyError(
            f"Unknown optimization objective. Available optimization objectives are: {optimization_objectives.keys()}"
        )

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
        Setup a new planning problem with a map, start and goal infromation

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
        # Clear previous setup
        self.setup.clear()

        self.set_space_bounds(map_meta_data)

        scoped_start_state = base.ScopedState(self.state_space)
        start_state = scoped_start_state.get()
        start_state.setX(start_x)
        start_state.setY(start_y)

        # SE2 takes angle in [-pi,+pi]
        yaw = convert_to_plus_minus_pi(start_yaw)
        start_state.setYaw(yaw)

        scoped_goal_state = base.ScopedState(self.state_space)
        goal_state = scoped_goal_state.get()
        goal_state.setX(goal_x)
        goal_state.setY(goal_y)

        # SE2 takes angle in [-pi,+pi]
        yaw = convert_to_plus_minus_pi(goal_yaw)
        goal_state.setYaw(yaw)

        self.setup.setStartAndGoalStates(
            start=scoped_start_state, goal=scoped_goal_state
        )

        # Setup Path Optimization Objective
        optimization_objective = getattr(base, self._config.optimization_objective)(
            self.setup.getSpaceInformation()
        )

        self.setup.setOptimizationObjective(optimization_objective)

        if self._use_fcl and map_3d is None:
            raise ValueError(
                "OMPL is started with collision check -> Map should be provided"
            )
        elif self._use_fcl:
            # TODO: Add an option to update the map periodically or just at the start, or after a time interval
            if not self.start:
                self.fcl.set_map(map_3d)
                self.start = True
            self.fcl.update_state(
                robot_state=RobotState(x=start_x, y=start_y, yaw=start_yaw)
            )

    def add_validity_check(self, name: str, validity_function: Callable) -> None:
        """
        Add method for state validity check during planning

        :param name: Method key name
        :type name: str
        :param validity_function: Validity check method
        :type validity_function: Callable

        :raises TypeError: If validity check is not callable
        :raises TypeError: If validity check method does not return a boolean
        """
        # Check that validity function is callable
        if callable(validity_function) and callable(validity_function):
            # Check if the function returns a boolean value
            args = (None,) * validity_function.__code__.co_argcount
            if isinstance(validity_function(*args), bool):
                self._custom_validity_check[name] = validity_function
            else:
                raise TypeError("Validity check function needs to return a boolean")
        else:
            raise TypeError("Validity check function must be callable")

    def remove_validity_check(self, name: str) -> bool:
        """
        Removes an added validity chec!= Nonek

        :param name: Validity check name key
        :type name: str

        :raises ValueError: If given key does not correspond to an added validity check

        :return: If validity check is removed
        :rtype: bool
        """
        deleted_method = self._custom_validity_check.pop(name, None)
        if deleted_method is not None:
            return True
        else:
            raise ValueError(
                f"Cannot remove validity check titled {name} as it does not exist"
            )

    def _validity_checker(self, state, **_) -> bool:
        """
        State validity checker method

        :param state: Robot state
        :type state: SE2State

        :return: If state is valid
        :rtype: bool
        """
        # Run bounds and state check
        return self.setup.getSpaceInformation().satisfiesBounds(state)

    def _validity_checker_with_fcl(self, state, **_) -> bool:
        """
        State validity checker method

        :param state: Robot state
        :type state: SE2State

        :return: If state is valid
        :rtype: bool
        """
        # Run bounds and state check
        state_space_valid: bool = self.setup.getSpaceInformation().satisfiesBounds(
            state
        )
        if self.fcl.got_map:
            self.fcl.update_state(
                RobotState(x=state.getX(), y=state.getY(), yaw=state.getYaw())
            )
            is_collision = self.fcl.check_collision()
        else:
            is_collision = False

        return state_space_valid and not is_collision

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

    def set_planner(
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

            self.planner = eval("%s(self.setup.getSpaceInformation())" % planner_id)

            if planner_config:
                self.planner_params = planner_config

            self.setup.setPlanner(self.planner)

        except Exception:
            raise

    def set_space_bounds(self, map_meta_data: Dict):
        """
        Set planning space bounds from map data

        :param map_meta_data: Map meta data
        :type map_meta_data: Dict
        """
        # X-axis bounds
        x_lower = map_meta_data["origin_x"]
        x_upper = x_lower + map_meta_data["resolution"] * map_meta_data["width"]

        # Y-axis bounds
        y_lower = map_meta_data["origin_y"]
        y_upper = y_lower + map_meta_data["resolution"] * map_meta_data["height"]

        # set lower and upper bounds
        bounds = base.RealVectorBounds(2)

        bounds.setLow(index=0, value=x_lower)
        bounds.setLow(index=1, value=y_lower)
        bounds.setHigh(index=0, value=x_upper)
        bounds.setHigh(index=1, value=y_upper)

        self.state_space.setBounds(bounds)

    def solve(self) -> ompl.geometric.PathGeometric:
        """
        Solve the problem using a given planner

        :return: Solution (Path points) or None if no solution is found
        :rtype: ompl.geometric.PathGeometric
        """
        solved = self.setup.solve(self._config.planning_timeout)

        if solved:
            self.solution = self.setup.getSolutionPath()

            return self.solution
        return None

    def simplify_solution(self) -> ompl.geometric.PathGeometric:
        """
        Simplify the path

        :return: Simplified path
        :rtype: ompl.geometric.PathGeometric
        """
        if self.solution:
            self.setup.simplifySolution(self._config.simplification_timeout)
            self.solution = self.setup.getSolutionPath()
            return self.solution
        return None
