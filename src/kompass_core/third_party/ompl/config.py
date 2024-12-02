import inspect
from typing import Any

from attrs import field, make_class
from ...utils.common import BaseAttrs, base_validators

import ompl


class PlanningAlgorithms(object):
    """PlanningAlgorithms."""

    UNKNOWN = 0
    BOOL = 1
    ENUM = 2
    INT = 3
    DOUBLE = 4

    def __init__(self, module):
        """Follows the pattern specified in ompl python bindings
        for compatibility
        :param module: ompl.geometric
        """
        self.available_planners = {
            f"{module.__name__}.{name}": self.get_param_map(planner)
            for name, planner in inspect.getmembers(module, predicate=inspect.isclass)
            if issubclass(planner, ompl.base.Planner)
        }

    def get_param_map(self, planner):
        """Returns planner param map.
        :param planner
        """
        try:
            state_space = ompl.base.SE2StateSpace()
            setup = ompl.geometric.SimpleSetup(state_space)
            space_info = setup.getSpaceInformation()
            planner_obj = planner(space_info)
            params = planner_obj.params()
        except Exception as e:
            raise Exception(f"Could not get params for {planner}") from e
        pnames = params.getParamNames()
        paramMap = {}
        for pname in pnames:
            p = params[pname]
            rangeSuggestion = p.getRangeSuggestion()
            if rangeSuggestion == "":
                continue
            rangeSuggestion = rangeSuggestion.split(":")
            defaultValue = p.getValue()
            if len(rangeSuggestion) == 1:
                rangeSuggestion = rangeSuggestion[0].split(",")
                if len(rangeSuggestion) == 1:
                    raise Exception("Cannot parse range suggestion")
                elif len(rangeSuggestion) == 2:
                    rangeType = self.BOOL
                    defaultValue = 0 if defaultValue == rangeSuggestion[0] else 1
                    rangeSuggestion = ""
                else:
                    rangeType = self.ENUM
                    defaultValue = (
                        0 if defaultValue == "" else rangeSuggestion.index(defaultValue)
                    )
            else:
                if "." in rangeSuggestion[0] or "." in rangeSuggestion[-1]:
                    rangeType = self.DOUBLE
                    rangeSuggestion = [float(r) for r in rangeSuggestion]
                    defaultValue = 0.0 if defaultValue == "" else float(defaultValue)
                else:
                    rangeType = self.INT
                    rangeSuggestion = [int(r) for r in rangeSuggestion]
                    defaultValue = 0 if defaultValue == "" else int(defaultValue)
                if len(rangeSuggestion) == 2:
                    rangeSuggestion = [rangeSuggestion[0], 1, rangeSuggestion[1]]
            name = p.getName()
            displayName = name.replace("_", " ").capitalize()
            paramMap[p.getName()] = (
                displayName,
                rangeType,
                rangeSuggestion,
                defaultValue,
            )
        return paramMap

    def getPlanners(self):
        """Returns the available planners map."""
        return self.available_planners


def initializePlanners():
    """Initialize planner map, similar to ompl python bindings."""
    logLevel = ompl.util.getLogLevel()
    # TODO: make log_level an input to initializePlanners to set it when using ompl
    ompl.util.setLogLevel(ompl.util.LogLevel.LOG_ERROR)
    if not hasattr(ompl.geometric, "planners"):
        ompl.geometric.planners = PlanningAlgorithms(ompl.geometric)
    ompl.util.setLogLevel(logLevel)


optimization_objectives = {
    "length": "PathLengthOptimizationObjective",
    "max_min_clearance": "MaximizeMinClearanceObjective",
    "mechanical_work": "MechanicalWorkOptimizationObjective",
}

map_types = {0: Any, 1: bool, 2: "enum", 3: int, 4: float}


def create_field(input_tuple: tuple):
    """
    Create one attrs field with validators from OMPL param tuple

    :param input_tuple: OMPL parameter tuple (name, type, rangeSuggestion, default_value)
    :type input_tuple: tuple

    :return: Attrs class field
    :rtype: field
    """
    field_type = map_types[input_tuple[1]]
    if field_type is bool:
        default = bool(input_tuple[3])
    else:
        default = input_tuple[3]
    if input_tuple[2]:
        if field_type == "enum":
            return field(
                type=type(default),
                default=default,
                validator=base_validators.in_(input_tuple[2]),
            )
        min_val, step, max_val = input_tuple[2]
        return field(
            type=field_type,
            default=default,
            validator=base_validators.in_range_discretized(
                step=step, min_value=min_val, max_value=max_val
            ),
        )
    return field(type=field_type, default=default)


def create_config_class(name: str, conf: dict) -> type:
    """
    Create a BaseAttrs config class from ompl planner parameters

    :param name: Name to be assigned to the class
    :type name: str
    :param conf: Planner parameters map from OMPL
    :type conf: dict

    :return: BaseAttrs config class
    :rtype: BaseAttrs
    """
    # Manual fix of error in default value : suggested range os set to [8,1,1000] and default value was set to 0
    if name in ["LazyPRM", "PRM"]:
        conf["max_nearest_neighbors"] = ("Max nearest neighbors", 3, [8, 1, 1000], 8)
    if name == "RRTstar":
        # to fix error RRTstar: OrderedSampling requires either informed sampling or rejection sampling
        conf["informed_sampling"] = ("Informed sampling", 1, "", 1)

    conf = {key: create_field(val) for key, val in conf.items()}
    return make_class(name, conf, bases=(BaseAttrs,))
