import math
from typing import List

from .utils.common import set_params_from_yaml
import numpy as np
from .datatypes.path import PathPoint, PathSample

from .models import Robot, RobotState


class RobotSim:
    """
    Class used to implement a robot for simulation/testing
    """

    def __init__(self, params_file: str) -> None:
        self.init_robot(params_file=params_file)

    def set_robot_params(self, path_to_file: str):
        """
        Sets the robot testing parameters values from a given yaml file under 'robot'

        :param path_to_file: Path to YAML file
        :type path_to_file: str
        """
        set_params_from_yaml(
            self,
            path_to_file,
            param_names=[
                "robot_model_type",
                "robot_geometry_type",
                "robot_geometry_params",
                "robot_max_speed",
                "robot_max_steering_angle",
            ],
            root_name="robot",
            yaml_key_equal_attribute_name=True,
        )

    def init_robot(self, params_file: str):
        """
        Setup the testing robot footprint, motion model and initial state

        :param params_file: Path to config file
        :type params_file: str
        """
        # robot params in yaml
        self.robot_model_type: str
        self.robot_geometry_type: str
        self.robot_geometry_params: List
        self.robot_max_speed: float
        self.robot_max_steering_angle: float

        # Get robot params
        self.set_robot_params(params_file)

        # Init testing robot
        self.robot = Robot(
            robot_type=self.robot_model_type,
            geometry_type=self.robot_geometry_type,
            geometry_params=self.robot_geometry_params,
        )

        # Set robot initial state from config file
        self.robot.state.set_from_yaml(params_file)

        # Set robot motion model from config
        self.robot.state.model.set_params_from_yaml(params_file)

    @classmethod
    def simulate_motion(
        cls,
        time_step: float,
        number_of_steps: int,
        control_seq: np.ndarray,
        robot: Robot,
    ) -> PathSample:
        """
        Simulates the robot motion resulting from applied control and returns the robot path points

        :param time_step: Time step (s)
        :type time_step: float
        :param number_of_steps: Number of simulation time steps
        :type number_of_steps: int
        :param control_seq: Applied control
        :type control_seq: np.ndarray

        :return: Resulting robot path
        :rtype: PathSample
        """
        # Init robot path sample
        robot_path = PathSample(length=number_of_steps)

        # Set first point equal to current robot state
        initial_speed = robot.state.speed
        robot_path.set_point(
            x=robot.state.x, y=robot.state.y, yaw=robot.state.yaw, idx=0
        )

        # Start simulation
        for idx in range(number_of_steps):
            linear_velocity = control_seq[idx][0]
            angular_velocity = control_seq[idx][1]

            # apply control and update robot state
            robot.set_control(linear_velocity, angular_velocity)
            robot.get_state(time_step)

            # update the actual path
            robot_path.set_point(
                x=robot.state.x, y=robot.state.y, yaw=robot.state.yaw, idx=idx
            )

        # Reset robot state after the end of the simulation
        robot.set_state(
            x=robot_path.x_points[0],
            y=robot_path.y_points[0],
            yaw=robot_path.heading_points[0],
            speed=initial_speed,
        )
        return robot_path


class MotionPaths:
    basic_tests = {
        0: "step_response_path",
        1: "line_path",
        2: "open_circular_path",
        3: "u_turn_path",
    }

    def __init__(self) -> None:
        pass

    @classmethod
    def generate_circle_path(
        cls, initial_state: RobotState, N: int, circle_radius=1.0
    ) -> List[PathPoint]:
        """
        Generates a List of circular path points (open circle)

        :param initial_state: Starting state (start of the open circle path)
        :type initial_state: RobotState
        :param N: Number of path points
        :type N: int
        :param circle_radius: Radius of the circular path (m), defaults to 1.0
        :type circle_radius: float, optional

        :return: List of the circular path points
        :rtype: List[PathPoint]
        """
        circle_start_x = initial_state.x - circle_radius
        circle_start_y = initial_state.y
        return [
            PathPoint(
                idx,
                circle_start_x + math.cos(2 * math.pi / N * idx) * circle_radius,
                circle_start_y + math.sin(2 * math.pi / N * idx) * circle_radius,
            )
            for idx in range(0, int(0.9 * N))
        ]

    @classmethod
    def generate_line_path(
        cls, initial_state: RobotState, N: int, path_length=1.0
    ) -> List[PathPoint]:
        """
        Generates a List of straight line path points

        :param initial_state: Starting state (start of the line)
        :type initial_state: RobotState
        :param N: Number of path points
        :type N: int
        :param path_length: Length of the path (m), defaults to 1.0
        :type path_length: float, optional

        :return: List of the line path points
        :rtype: _type_
        """
        points_step_dist = path_length / N
        return [
            PathPoint(
                idx,
                initial_state.x + idx * points_step_dist * math.cos(initial_state.yaw),
                initial_state.y + idx * points_step_dist * math.sin(initial_state.yaw),
                initial_state.yaw,
            )
            for idx in range(0, N + 1)
        ]

    @classmethod
    def generate_step_path(
        cls, initial_state: RobotState, N: int, step_distance=0.5, path_length=1.0
    ) -> List[PathPoint]:
        """
        Generates a List of step response path points

        :param initial_state: Starting state (start of the step)
        :type initial_state: RobotState
        :param N: Number of path points
        :type N: int
        :param step_distance: Distance of the step (m), defaults to 0.5
        :type step_distance: float, optional
        :param path_length: Length of the path (m), defaults to 1.0
        :type path_length: float, optional

        :return: List of the line path points
        :rtype: _type_
        """
        # Get the step line initial point
        step_initial_x = initial_state.x - step_distance * math.sin(initial_state.yaw)
        step_initial_y = initial_state.y + step_distance * math.cos(initial_state.yaw)
        step_initial = RobotState(
            x=step_initial_x,
            y=step_initial_y,
            yaw=initial_state.yaw,
            speed=initial_state.speed,
        )

        # Return a line starting at a step distance from the robot
        return cls.generate_line_path(step_initial, N, path_length)

    @classmethod
    def generate_oval_path(
        cls,
        initial_state: RobotState,
        N: int,
        oval_major_length=1.0,
        oval_minor_length=0.5,
    ) -> List[PathPoint]:
        """
        Generates a List of oval path points

        :param initial_state: Starting state (start of the step)
        :type initial_state: RobotState
        :param N: Number of path points
        :type N: int
        :param oval_major_length: Oval major axis (m), defaults to 1.0
        :type oval_major_length: float, optional
        :param oval_minor_length: Oval minor axis (m), defaults to 0.5
        :type oval_minor_length: float, optional

        :return: List of the oval path points
        :rtype: List[PathPoint]
        """
        oval_path: List[PathPoint] = []

        # Get start point so the oval passes by the robot initial point
        oval_initial_x = initial_state.x - oval_minor_length * math.sin(
            initial_state.yaw
        )
        oval_initial_y = initial_state.y + oval_minor_length * math.cos(
            initial_state.yaw
        )

        # Generate oval points
        for idx, yaw in enumerate(
            np.linspace(3 * np.pi / 2, 2 * np.pi + 3 * np.pi / 2, N)
        ):
            x = (
                oval_initial_x
                + oval_major_length * np.cos(initial_state.yaw) * np.cos(yaw)
                - oval_minor_length * np.sin(initial_state.yaw) * np.sin(yaw)
            )
            y = (
                oval_initial_y
                + oval_major_length * np.sin(initial_state.yaw) * np.cos(yaw)
                + oval_minor_length * np.cos(initial_state.yaw) * np.sin(yaw)
            )
            oval_path.append(
                PathPoint(idx, x, y, initial_state.yaw, initial_state.speed)
            )
        return oval_path

    @classmethod
    def generate_u_turn_path(
        self, initial_state: RobotState, N: int, path_length=1.0, u_turn_radius=0.3
    ) -> List[PathPoint]:
        """
        Generates a List of u-turn path points

        :param initial_state: Starting state (start of the step)
        :type initial_state: RobotState
        :param N: Number of path points
        :type N: int
        :param path_length: Length of the straight line in the u-turn (m), defaults to 1.0
        :type path_length: float, optional
        :param u_turn_radius: Radius of the u-turn circle (m), defaults to 0.5
        :type u_turn_radius: float, optional

        :return: List of the u-turn path points
        :rtype: List[PathPoint]
        """
        # Part 1: line
        u_turn_start_line = self.generate_line_path(
            initial_state, int(N / 3), path_length
        )

        # Part 2: half Oval
        u_turn_oval_init = RobotState(
            u_turn_start_line[-1].x,
            u_turn_start_line[-1].y,
            initial_state.yaw,
            initial_state.speed,
        )
        u_turn_oval = self.generate_oval_path(
            u_turn_oval_init,
            int(N / 3),
            oval_major_length=u_turn_radius * 2,
            oval_minor_length=u_turn_radius,
        )
        u_turn_half_oval = u_turn_oval[: int(N / 6)]

        # Part 3: line
        u_turn_end_line_init = RobotState(
            u_turn_half_oval[-1].x,
            u_turn_half_oval[-1].y,
            initial_state.yaw - math.pi,
            initial_state.speed,
        )
        remaining_points: int = N - len(u_turn_half_oval) - len(u_turn_start_line)
        u_turn_end_line = self.generate_line_path(
            u_turn_end_line_init, remaining_points, path_length
        )

        # Complete u-turn
        return u_turn_start_line + u_turn_half_oval + u_turn_end_line
