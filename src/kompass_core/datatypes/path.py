import logging
import os
from typing import List, Union

import numpy as np
import pandas as pd
from attrs import define, field


class PathPoint:
    """
    Global path point data class
    """

    def __init__(self, idx=0, x=0.0, y=0.0, heading=0.0, speed=0.0):
        """
        Init path point at 0,0
        """
        self.idx: int = idx
        self.x: float = x
        self.y: float = y
        self.heading: float = heading
        self.speed: float = speed

    def __sub__(self, other: object):
        if not isinstance(other, PathPoint):
            raise TypeError(f"Cannot subtract PathPoint and type {type(other)}")
        else:
            return PathPoint(
                x=self.x - other.x,
                y=self.y - other.y,
                heading=self.heading - other.heading,
            )

    def __add__(self, other: object):
        if not isinstance(other, PathPoint):
            raise TypeError(f"Cannot substract PathPoint and type {type(other)}")
        else:
            return PathPoint(
                x=self.x + other.x,
                y=self.y + other.y,
                heading=self.heading + other.heading,
            )

    def __truediv__(self, value: Union[int, float]):
        return PathPoint(x=self.x / value, y=self.y / value, heading=self.heading)


class PathSample:
    """
    Path sample
    """

    def __init__(self, length: int, frame_id="map"):
        """
        Init a an empty path of given length

        :param length: Number of points in the path
        :type length: int
        :param frame_id: Path points coordinates frame, defaults to 'map'
        :type frame_id: str, optional
        """
        self.x_points = np.zeros(length, dtype=float)
        self.y_points = np.zeros(length, dtype=float)
        self.heading_points = np.zeros(length, dtype=float)
        self.frame_id: str = frame_id

    def set_path(
        self, x_points: np.ndarray, y_points: np.ndarray, heading_points: np.ndarray
    ):
        """
        Sets a path with a list of points

        :param x_points: X-coordinates of the path points
        :type x_points: np.ndarray
        :param y_points: Y-coordinates of the path points
        :type y_points: np.ndarray
        :param heading_points: Heading of the path points
        :type heading_points: np.ndarray
        """
        self.x_points = x_points
        self.y_points = y_points
        self.heading_points = heading_points

    def set_point(self, x: float, y: float, pitch: float, idx: int):
        """
        Sets a path point at given index

        :param x: X-coordinate of the path point
        :type x: float
        :param y: Y-coordinate of the path point
        :type y: float
        :param pitch: Heading of the path point
        :type pitch: float
        :param idx: Index of the path point
        :type idx: int
        """
        self.x_points[idx] = x
        self.y_points[idx] = y
        self.heading_points[idx] = pitch

    def set_points(
        self, x: List[float], y: List[float], pitch: List[float], idx_start: int
    ):
        """
        Sets a number of path point

        :param x: X-coordinate of the path points
        :type x: List[float]
        :param y: Y-coordinate of the path points
        :type y: List[float]
        :param pitch: Heading of the path points
        :type pitch: List[float]
        :param idx_start: Start index of the points along the sample
        :type idx_start: int
        """
        idx_end = idx_start + len(x)
        try:
            assert idx_end < len(self.x_points)
            self.x_points[idx_start:idx_end] = x
            self.y_points[idx_start:idx_end] = y
            self.heading_points[idx_start:idx_end] = pitch
        except AssertionError as e:
            logging.error(f"{e}. Cannot set points for longer than the sample lenth")


class TrajectorySample:
    def __init__(self, length: int, frame_id="map"):
        """
        Init a trajectory with a given length and trajectory points coordinates frame

        :param length: Length of the trajectory points
        :type length: int
        :param frame_id: Coordinates frame of the points, defaults to "map"
        :type frame_id: str, optional
        """
        self.frame_id = frame_id
        self.set_traj_length(length=length)

    def set_traj_length(self, length: int):
        """
        Re-init the trajectory sample with a given length

        :param length: Sample length (number of points)
        :type length: int
        """
        self.path_sample = PathSample(length, self.frame_id)
        self.time = np.empty(length, dtype=float)

    def set_traj(
        self,
        x_points: np.ndarray,
        y_points: np.ndarray,
        heading_points: np.ndarray,
        time_points: np.ndarray,
    ):
        """
        Sets a trajectory with a list of points

        :param x_points: X-coordinates of the trajectory points
        :type x_points: np.ndarray
        :param y_points: Y-coordinates of the trajectory points
        :type y_points: np.ndarray
        :param heading_points: Heading of the trajectory points
        :type heading_points: np.ndarray
        :param time_points: Trajectory time points (sec)
        :type time_points: np.ndarray
        """
        self.path_sample.set_path(x_points, y_points, heading_points)
        self.time = time_points

    def set_traj_from_path(self, path_sample: PathSample, time: np.ndarray):
        """
        Set the trajectory sample using a path sample

        :param path_sample: Path sample
        :type path_sample: PathSample
        :param time: Time vector
        :type time: np.ndarray
        """
        self.path_sample = path_sample
        self.time = time

    def set_traj_point(self, x: float, y: float, heading: float, time: float, idx: int):
        """
        Sets a trajectory point at given index

        :param x: X-coordinate of the trajectory point
        :type x: float
        :param y: Y-coordinate of the trajectory point
        :type y: float
        :param heading: Heading of the trajectory point
        :type heading: float
        :param time: Time of the trajectory point
        :type time: float
        :param idx: Index of the trajectory point
        :type idx: int
        """
        self.path_sample.set_point(x, y, heading, idx)
        self.time[idx] = time

    def set_traj_points(
        self,
        x: List[float],
        y: List[float],
        heading: List[float],
        time: List[float],
        idx_start: int,
    ):
        """
        Sets a trajectory point at given index

        :param x: X-coordinate of the trajectory point
        :type x: float
        :param y: Y-coordinate of the trajectory point
        :type y: float
        :param heading: Heading of the trajectory point
        :type heading: float
        :param time: Time of the trajectory point
        :type time: float
        :param idx: Index of the trajectory point
        :type idx: int
        """
        self.path_sample.set_points(x, y, heading, idx_start)
        idx_end = idx_start + len(time)
        self.time[idx_start:idx_end] = time


class MotionSample(TrajectorySample):
    """
    Motion sample is a trajectory sample along with control
    """

    CSV_NAMES = [
        "time",
        "frame_id",
        "x",
        "y",
        "heading",
        "lateral_control_x",
        "lateral_control_y",
        "angular_control",
    ]

    def __init__(self, length: int, frame_id="map"):
        super().__init__(length, frame_id)
        self.length = length
        self.control = np.zeros([length, 3], dtype=float)

    def set_length(self, length: int):
        """
        Re-init motion sample with new given length

        :param length: Sample length (number of points)
        :type length: int
        """
        self.set_traj_length(length=length)
        self.length = length
        self.control = np.zeros([length, 3], dtype=float)

    def set_control(
        self,
        linear_control_x: np.ndarray,
        linear_control_y: np.ndarray,
        angular_control: np.ndarray,
    ):
        """
        Set motion control sequence

        :param linear_control: Linear velocity control commands
        :type linear_control: np.ndarray
        :param angular_control: Angular Velocity control commands
        :type angular_control: np.ndarray
        """
        self.set_control_points(
            linear_control_x.tolist(),
            linear_control_y.tolist(),
            angular_control.tolist(),
            idx_start=0,
        )

    def set_control_points(
        self,
        linear_control_x: List[float],
        linear_control_y: List[float],
        angular_control: List[float],
        idx_start: int,
    ):
        """
        Set motion control point

        :param linear_control: Linear velocity control command
        :type linear_control: List[float]
        :param angular_control: Angular Velocity control command
        :type angular_control: List[float]
        :param idx_start: Control points start index
        :type idx_start: int
        """
        idx_end = idx_start + len(linear_control_x)
        try:
            assert (idx_end <= self.length) and (idx_start >= 0)
            self.control[idx_start:idx_end, 0] = linear_control_x
            self.control[idx_start:idx_end, 1] = linear_control_y
            self.control[idx_start:idx_end, 2] = angular_control

        except AssertionError as e:
            logging.error(f"{e}. Given control indeces should be in [0, {self.length}]")

    def set_control_point(
        self,
        linear_control_x: float,
        linear_control_y: float,
        angular_control: float,
        idx: int,
    ):
        """
        Set motion control point

        :param linear_control: Linear velocity control command
        :type linear_control: float
        :param angular_control: Angular Velocity control command
        :type angular_control: float
        :param idx: Control point index
        :type idx: int
        """
        try:
            assert idx <= self.length
            self.control[idx, 0] = linear_control_x
            self.control[idx, 1] = linear_control_y
            self.control[idx, 2] = angular_control

        except AssertionError as e:
            logging.error(f"{e}. Given control indices should be in [0, {self.length}]")

    def set_motion_point(
        self,
        x: float,
        y: float,
        heading: float,
        time: float,
        linear_control_x: float,
        linear_control_y: float,
        angular_control: float,
        idx: int,
    ):
        """
        Set motion point

        :param x: X-coordinate of the motion point
        :type x: float
        :param y: Y-coordinate of the motion point
        :type y: float
        :param heading: Heading of the motion point
        :type heading: float
        :param time: Time of the motion point
        :type time: float
        :param idx: Index of the motion point
        :type idx: int
        :param linear_control: Linear velocity control command
        :type linear_control: float
        :param angular_control: Angular Velocity control command
        :type angular_control: float
        """
        self.set_traj_point(x, y, heading, time, idx)
        self.set_control_points(
            [linear_control_x], [linear_control_y], [angular_control], idx
        )

    def set_motion_points(
        self,
        x: List[float],
        y: List[float],
        heading: List[float],
        time: List[float],
        linear_control_x: List[float],
        linear_control_y: List[float],
        angular_control: List[float],
        idx_start: int,
    ):
        """
        Sets motion points from a given index along the sample

        :param x: X-coordinate of the motion points
        :type x: List[float]
        :param y: Y-coordinate of the motion points
        :type y: List[float]
        :param heading: Heading of the motion points
        :type heading: List[float]
        :param time: Time of the motion points
        :type time: List[float]
        :param linear_control: Linear velocity control commands
        :type linear_control: List[float]
        :param angular_control: Angular Velocity control commands
        :type angular_control: List[float]
        :param idx_start: Start index
        :type idx_start: int
        """
        self.set_traj_points(x, y, heading, time, idx_start)
        self.set_control_points(
            linear_control_x, linear_control_y, angular_control, idx_start
        )

    def _csv_mapping(self):
        return {
            self.CSV_NAMES[0]: self.time,
            self.CSV_NAMES[1]: self.path_sample.frame_id,
            self.CSV_NAMES[2]: self.path_sample.x_points,
            self.CSV_NAMES[3]: self.path_sample.y_points,
            self.CSV_NAMES[4]: self.path_sample.heading_points,
            self.CSV_NAMES[5]: self.control[:, 0],
            self.CSV_NAMES[6]: self.control[:, 1],
            self.CSV_NAMES[7]: self.control[:, 2],
        }

    def save_to_csv(self, file_location: str, file_name: str) -> bool:
        """
        Saves motion data to a csv file

        :param file_location: File location
        :type file_location: str
        :param file_name: File name
        :type file_name: str

        :raises FileExistsError: IF the given file location does not exist

        :return: File is saved
        :rtype: bool
        """
        try:
            csv_mapping = self._csv_mapping()

            # Create a DataFrame using pandas
            motion_df = pd.DataFrame(csv_mapping)
            # Check if the directory exists, if not, create it
            if os.path.exists(file_location):
                if not file_name.lower().endswith(".csv"):
                    file_name += ".csv"

                # Save the DataFrame to a CSV file
                file_path = os.path.join(file_location, file_name)

                # If the file exists remove it to overwrite the data
                if os.path.exists(file_path):
                    os.remove(file_path)

                motion_df.to_csv(file_path, index=False)

                return True
            return False

        except FileNotFoundError as e:
            logging.error(f"{e}")
            raise

    def get_from_csv(self, file_location: str, file_name: str) -> bool:
        """
        Gets motion sample data from a given csv file

        :param file_location: CSV file location
        :type file_location: str
        :param file_name: CSV file name
        :type file_name: str

        :raises ValueError: If given file_name is not a csv file
        :raises AssertionError: If given csv file data is not a valid motion sample data

        :return: Motion sample loaded from file
        :rtype: bool
        """
        if os.path.exists(file_location):
            _, extension = os.path.splitext(file_name)

            if extension == "":
                # If there's no extension, add '.csv'
                file_name += ".csv"

            elif extension.lower() != ".csv":
                # If extension is not '.pdf', raise an error
                logging.error("Given file must be a csv file")
                raise ValueError

            file_path = os.path.join(file_location, file_name)

            # Get dataframe from csv file
            motion_df = pd.read_csv(file_path)

            matched_names = [
                name for name in self.CSV_NAMES if name in motion_df.columns
            ]

            try:
                assert len(matched_names) == len(self.CSV_NAMES)
                time = motion_df[self.CSV_NAMES[0]].values
                self.set_length(length=len(time))
                self.time = time

                path_sample = PathSample(length=len(time))
                path_sample.x_points = motion_df[self.CSV_NAMES[2]].values
                path_sample.y_points = motion_df[self.CSV_NAMES[3]].values
                path_sample.heading_points = motion_df[self.CSV_NAMES[4]].values
                self.path_sample = path_sample

                self.control[:, 0] = motion_df[self.CSV_NAMES[5]].values
                self.control[:, 1] = motion_df[self.CSV_NAMES[6]].values

                return True

            except AssertionError as e:
                logging.error(
                    f"{e} Please provide a valid csv file containing MotionSample data"
                )
                raise

        else:
            logging.error(f"No such folder {file_location}")
            return False


class InterpolationPoint:
    """
    Interpolated path point data class
    """

    def __init__(self, s, x, y, pitch):
        self.s = s  # arc length
        self.x = x
        self.y = y
        self.pitch = pitch


@define
class TrackedPoint:
    """
    Extended interpolated path point data class used for path following
    """

    s: float = field(default=0.0)
    x: float = field(default=0.0)
    y: float = field(default=0.0)
    tangent_ori: float = field(default=0.0)
    lat_dist: float = field(default=0.0)
    curv: float = field(default=0.0)
    ori_err: float = field(default=0.0)
    s_dot: float = field(default=0.0)
    lat_vel: float = field(default=0.0)
    pitch: float = field(default=0.0)
    forward_dist: float = field(default=0.0)


@define
class Point2D:
    """
    2D point class
    """

    x: float = field(default=0.0)
    y: float = field(default=0.0)


@define
class Range2D:
    """
    2D range limits data class
    """

    min_val: float = field(default=0.0)
    max_val: float = field(default=0.0)


class PathTrackingError:
    """
    Path tracking errors data class
    """

    def __init__(self):
        self.orientation_error = 0.0
        self.lateral_distance_error = 0.0

    def set(self, ori_error: float, lat_error: float):
        self.orientation_error = ori_error
        self.lateral_distance_error = lat_error


class Odom2D:
    """
    2D odometry data class
    """

    def __init__(self):
        self.x = 0.0
        self.y = 0.0
        self.ori = 0.0
        self.speed = 0.0

    def set(self, x, y, ori, speed):
        self.x = x
        self.y = y
        self.ori = ori
        self.speed = speed
