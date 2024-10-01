import json
import logging
import os
import timeit
from typing import Dict
import numpy as np
import pandas as pd
from nav_msgs.msg import MapMetaData

import ompl
from kompass_core.models import Robot, RobotGeometry, RobotType
from kompass_core.third_party.ompl.config import create_config_class, initializePlanners
from kompass_core.third_party.ompl.planner import OMPLGeometric

logger = logging.getLogger(__name__)

dir_name = os.path.dirname(os.path.abspath(__file__))
ompl_resources = os.path.join(dir_name, "resources/ompl")

SOLUTION_TOLERANCE_TIME = 0.3
SOLUTION_TOLERANCE_LENGTH = 0.1

REF_RESULTS = pd.read_csv(
    os.path.join(ompl_resources, "test_results_geometric_ref.csv")
)


def generate_all_geometric_planners_configs():
    """
    To run a generation check on all geometric planners in ompl
    """
    initializePlanners()
    # Get planners {planner_name: planner_params}
    planners = ompl.geometric.planners.getPlanners()
    for planner_name, planner_params in planners.items():
        reduced_name = planner_name.split(".")[-1]
        # Print raw parameters
        print(f"{reduced_name} RAW PARAMETERS:\n {planner_params}\n")
        # Generate attrs config class
        planner_config_class = create_config_class(reduced_name, planner_params)
        print(f"{reduced_name} PARSED CONFIG CLASS: \n  {planner_config_class()}")
        print("-----------------------------\n")


def ompl_solve_once(
    ompl_planner: OMPLGeometric, map_data: MapMetaData, map_numpy: np.ndarray
):
    """
    Setup and solve OMPL planning problem with given map and map metadata

    :param ompl_planner: _description_
    :type ompl_planner: OMPLGeometric
    :param map_data: _description_
    :type map_data: MapMetaData
    :param map_numpy: _description_
    :type map_numpy: np.ndarray
    :return: OMPL path solution
    :rtype: _type_
    """
    # Start and Goal points selected for turtlebot_3 world example
    start_x = -1.88
    start_y = -0.38
    start_yaw = 0.3

    goal_x = 0.59
    goal_y = 0.73
    goal_yaw = 0.0

    print(
        f"Setting planning problem with from [{start_x},{start_y}] to [{goal_x}, {goal_y}] and map data {map_data}"
    )

    ompl_planner.setup_problem(
        map_data,
        start_x,
        start_y,
        start_yaw,
        goal_x,
        goal_y,
        goal_yaw,
        map_numpy,
    )
    return ompl_planner.solve()


def load_map_meta(map_file: str) -> Dict:
    """
    Load map meta data from json file

    :param map_file: _description_
    :type map_file: str
    :raises Exception: _description_

    :return: _description_
    :rtype: MapMetaData
    """
    try:
        with open(map_file, "r") as f:
            map_meta_dict = json.load(f)

        map_meta = {}
        map_meta["resolution"] = map_meta_dict["resolution"]
        map_meta["width"] = map_meta_dict["width"]
        map_meta["height"] = map_meta_dict["height"]
        map_meta["origin_x"] = map_meta_dict["origin"]["position"]["x"]
        map_meta["origin_y"] = map_meta_dict["origin"]["position"]["y"]
        map_meta["origin_yaw"] = 2 * np.arctan2(
            map_meta_dict["origin"]["orientation"]["z"],
            map_meta_dict["origin"]["orientation"]["w"],
        )

        logging.info(f"Loaded map metadata: {map_meta}")

        return map_meta
    except Exception as e:
        raise Exception(f"Failed to load map metadata: {str(e)}") from e


def ompl_geometric_testing(test_repetitions: int = 1):
    """
    Test all OMPL geometric planners

    :param test_repetitions: _description_, defaults to 7
    :type test_repetitions: int, optional
    """
    ompl_df = pd.DataFrame(
        columns=(
            "method",
            "solved",
            "solution_time",
            "solution_len",
            "simplification_time",
            "time_convert_2_ros",
        )
    )

    robot = Robot(
        robot_type=RobotType.ACKERMANN,
        geometry_type=RobotGeometry.Type.BOX,
        geometry_params=np.array([0.2, 0.2, 0.2]),
    )
    ompl_planner = OMPLGeometric(robot=robot)

    map_numpy = np.load(os.path.join(ompl_resources, "turtlebot_map.npy"))

    map_file = os.path.join(ompl_resources, "map_meta.json")

    map_data: MapMetaData = load_map_meta(map_file)

    config_file = os.path.join(ompl_resources, "testing_params.yaml")

    for planner_id in ompl_planner.available_planners.keys():
        solution_time: float = 0.0
        simplify_time: float = 0.0
        sol_len: float = 0.0
        if planner_id in ["ompl.geometric.AITstar", "ompl.geometric.LazyLBTRRT"]:
            continue
        logging.info(
            f"Testing planning with {planner_id}, running {test_repetitions} trials ..."
        )

        for i in range(test_repetitions):
            logging.info(f"Testing planning with {planner_id}, trial {i}")
            try:
                ompl_planner.configure(
                    config_file, root_name="ompl_tester", planner_id=planner_id
                )

                start_time = timeit.default_timer()
                path = ompl_solve_once(ompl_planner, map_data, map_numpy)

                end_time = timeit.default_timer()

                solution_time += end_time - start_time

                solved = True
                if path:
                    start_time = timeit.default_timer()

                    path = ompl_planner.simplify_solution()

                    end_time = timeit.default_timer()

                    simplify_time += end_time - start_time

                    solved = True

                    sol_len += ompl_planner.solution.length()
                else:
                    solved = False

            except Exception as e:
                logging.error(f"{e}")

        ompl_df.loc[len(ompl_df)] = {
            "method": planner_id,
            "solved": solved,
            "solution_time": solution_time / test_repetitions,
            "solution_len": sol_len / test_repetitions,
            "simplification_time": simplify_time / test_repetitions,
        }

    ompl_df.to_csv(f"{ompl_resources}/test_results.csv", index=False)
    return ompl_df


def ompl_test_all():
    logging.info("Running all OMPL planners tests")
    results_df = ompl_geometric_testing(test_repetitions=20)
    logging.info("Done all OMPL planners tests")
    logging.info("----------------------------")

    planners = ompl.geometric.planners.getPlanners()
    for method in planners.keys():
        row_df = results_df[results_df["method"] == method].iloc[0]
        row_csv_df = REF_RESULTS[REF_RESULTS["method"] == method].iloc[0]

        # Assert that 'solved' values are equal
        assert (
            row_df["solved"] == row_csv_df["solved"]
        ), f"Solved values for method {method} do not match"

        # Assert that 'solution_time' values are within the tolerance
        assert (
            abs(row_df["solution_time"] - row_csv_df["solution_time"])
            <= SOLUTION_TOLERANCE_TIME
        ), f"Solution time for method {method} differs more than the tolerance of {SOLUTION_TOLERANCE_TIME}"

        # Assert that 'solution_time' values are within the tolerance
        assert (
            abs(row_df["solution_len"] - row_csv_df["solution_len"])
            <= SOLUTION_TOLERANCE_LENGTH
        ), f"Solution time for method {method} differs more than the tolerance of {SOLUTION_TOLERANCE_TIME}"


def test():
    # test config generation for all planners
    generate_all_geometric_planners_configs()

    # Test planning
    ompl_geometric_testing()


if __name__ == "__main__":
    test()
