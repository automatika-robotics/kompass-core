from typing import Any, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from .datatypes.obstacles import ObstaclesData
from .datatypes.path import (
    PathPoint,
    PathSample,
    PathTrackingError,
    TrajectorySample,
)
from .utils import visualization

from .motion_cost import MotionCostsParams, ReferenceCost, StaticCollisionCost
from .simulation import RobotSim


class MotionResult:
    def __init__(self) -> None:
        self.time = []
        self.steer_cmds: List[float] = []
        self.robot_path: PathSample = None
        self.speed_cmd: List[float] = []
        self.ori_error: List[float] = []
        self.lat_error: List[float] = []
        self.success = False
        self.time_to_goal: float = 0.0
        self.end_error = PathTrackingError()

    def vis_result(
        self,
        test: List[PathPoint],
        robot_footprint: Any,
        figure_title="Figure 0",
    ):
        """
        Plot the results of a test: resulting control commands / follower errors / robot path

        :param test: Testing reference path
        :type test: List[PathPoint]
        :param robot_footprint: Testing robot footprint
        :type robot_footprint: Any
        :param figure_title: Title of the generated figure, defaults to 'Figure 0'
        :type figure_title: str, optional
        """
        _fig_margin = 0.5

        fig, [ax0, ax1, ax2, ax3] = plt.subplots(nrows=4, ncols=1, figsize=(8, 8))
        fig.suptitle(figure_title)
        fig.tight_layout(pad=3.0)

        # Plot steering control
        ax0.plot(self.time, self.steer_cmds, label="control")
        ax0.plot(self.time, self.ori_error, color="red", label="error")
        ax0.legend()
        ax0.set_ylim(
            min(self.steer_cmds) - _fig_margin,
            max(self.steer_cmds) + _fig_margin,
        )
        ax0.set_title("Steering Control and Orientation Error (rad)")
        ax0.set_xlabel("time (s)")
        ax0.set_ylabel("Angle (rad)")

        # Plot orientation error
        ax1.plot(self.time, self.speed_cmd)
        ax1.set_title("Linear Velocity Control (m/s)")
        ax1.set_ylim(
            min(self.speed_cmd) - _fig_margin, max(self.speed_cmd) + _fig_margin
        )
        ax1.set_xlabel("time (s)")
        ax1.set_ylabel("Speed (m/s)")

        # Plot lateral error
        ax2.plot(self.time, self.lat_error)
        ax2.set_title("Signed Lateral Distance Error (m)")
        ax2.set_ylim(
            min(self.lat_error) - _fig_margin, max(self.lat_error) + _fig_margin
        )
        ax2.set_xlabel("time (s)")
        ax2.set_ylabel("Distance (m)")

        # Plot resulting path
        ax3.axis("equal")
        visualization.plt_path_sample(self.robot_path, ax=ax3)
        ax3.set_ylim(
            min(self.robot_path.y_points) - robot_footprint.wheel_base,
            max(self.robot_path.y_points) + robot_footprint.wheel_base,
        )

        # Plot test path
        visualization.plt_path_points_List(test, color="red", ax=ax3)

        # Plot robot initial state
        robot_footprint.plt_robot(
            self.robot_path.x_points[0],
            self.robot_path.y_points[0],
            self.robot_path.heading_points[0],
            ax=ax3,
            color="gray",
        )

        # Plot robot final state
        robot_footprint.plt_robot(
            self.robot_path.x_points[-1],
            self.robot_path.y_points[-1],
            self.robot_path.heading_points[-1],
            ax=ax3,
        )

        ax3.set_title("Robot path")
        ax3.set_xlabel("X (m)")
        ax3.set_ylabel("Y (m)")


class TestBase(RobotSim):
    def __init__(self, params_file: str) -> None:
        # Init robot model & footprint
        super().__init__(params_file)
        print("init test base")
        self.reset()

    def reset(self):
        """
        Reset test and result
        """
        self.test: List[PathPoint] = []
        self.result = MotionResult()


class TestAvgResults:
    def __init__(self) -> None:
        columns = [
            "test_id",
            "test_type",
            "avg_linear_vel",
            "avg_angular_vel",
            "avg_lin_acc",
            "avg_ang_acc",
            "max_linear_vel",
            "max_ang_vel",
            "max_lin_acc",
            "max_ang_acc",
            "collision_cost",
            "reference_cost",
        ]
        self.df = pd.DataFrame(columns=columns)

    def add_test(
        self,
        test_id: int,
        test_type: str,
        robot_traj: TrajectorySample,
        collision_cost: float,
        ref_path_cost: float,
    ):
        """
        Compute and add new test values to dataframe

        :param test_id: Test ID number
        :type test_id: int
        :param test_type: Test type
        :type test_type: str
        :param robot_traj: Robot recorded trajectory along the test
        :type robot_traj: TrajectorySample
        :param collision_cost: Static collisions cost (Ideally 0)
        :type collision_cost: float
        :param ref_path_cost: Path following error cost (Ideally -> 0)
        :type ref_path_cost: float
        """
        robot_path: PathSample = robot_traj.path_sample
        dx = np.diff(robot_path.x_points)
        dy = np.diff(robot_path.y_points)
        d_heading = np.diff(robot_path.heading_points)
        d_time = np.diff(robot_traj.time)

        # Linear velocity stats
        linear_vel: np.ndarray = np.sqrt(dx**2 + dy**2) / d_time
        avg_linear_vel: float = linear_vel.mean()
        max_linear_vel: float = linear_vel.max()

        # Linear acceleration stata
        linear_acc: np.ndarray = np.diff(linear_vel) / d_time[:-1]
        avg_linear_acc: float = linear_acc.mean()
        max_linear_acc: float = linear_acc.max()

        # Angular velocity stats
        angular_vel: np.ndarray = d_heading / d_time
        avg_ang_vel: float = angular_vel.mean()
        max_ang_vel: float = angular_vel.max()

        # Angular acceleration stats
        angular_acc: np.ndarray = np.diff(angular_vel) / d_time[:-1]
        avg_ang_acc: float = angular_acc.mean()
        max_ang_acc: float = angular_acc.max()

        new_raw = {
            "test_id": test_id,
            "test_type": test_type,
            "avg_linear_vel": avg_linear_vel,
            "avg_angular_vel": avg_ang_vel,
            "avg_lin_acc": avg_linear_acc,
            "avg_ang_acc": avg_ang_acc,
            "max_linear_vel": max_linear_vel,
            "max_ang_vel": max_ang_vel,
            "max_lin_acc": max_linear_acc,
            "max_ang_acc": max_ang_acc,
            "collision_cost": collision_cost,
            "reference_cost": ref_path_cost,
        }
        self.df.loc[test_id, :] = new_raw


class MotionEvaluation(TestBase):
    def __init__(self, params_file: str) -> None:
        super().__init__(params_file)
        # Init tests variables
        self.robot_traj: TrajectorySample
        self.local_maps: List[ObstaclesData]
        self.cost_params = MotionCostsParams()

        # set params from file
        if params_file:
            self.cost_params.set_from_yaml(params_file)

        self.collision_cost = StaticCollisionCost(
            self.cost_params.static_collision_weight,
            self.cost_params.static_collision_margin,
            self.robot.footprint,
        )

        self.end_goal_cost = ReferenceCost(
            self.cost_params.goal_lat_err_weight,
            self.cost_params.goal_heading_err_weight,
        )
        self.results = TestAvgResults()

    def compute_motion_cost(self, ref_path: List[PathPoint]):
        """
        Computes the motion cost along a reference path

        :param ref_path: List of reference path points
        :type ref_path: List[PathPoint]
        """
        for idx in range(len(self.robot_traj.path_sample.x_points)):
            # self.collision_cost.update(self.robot_traj.path_sample, idx, self.local_maps[idx])
            self.end_goal_cost.update(self.robot_traj.path_sample, idx, ref_path)

    def add_test(
        self,
        test_id: int,
        test_type: str,
        ref_path: List[PathPoint],
        robot_traj: TrajectorySample,
        local_maps: List[ObstaclesData],
    ):
        """
         Adds new test analysis to results

        :param test_id: Test ID number
         :type test_id: int
         :param test_type: Test type
         :type test_type: str
         :param ref_path: List of reference path points
         :type ref_path: List[PathPoint]
         :param robot_traj: Robot recorded trajectory along the test
         :type robot_traj: TrajectorySample
         :param local_maps: List of robot local maps along the trajectory
         :type local_maps: List[ObstaclesData]
        """
        self.robot_traj = robot_traj
        self.local_maps = local_maps
        self.compute_motion_cost(ref_path)
        self.results.add_test(
            test_id,
            test_type,
            self.robot_traj,
            self.collision_cost.value,
            self.end_goal_cost.displacement.value,
        )

    def export(self, file_dir: str):
        """
        Export results dataframe to csv

        :param file_dir: Results csv file
        :type file_dir: str
        """
        self.results.df.to_csv(file_dir, index=False)
