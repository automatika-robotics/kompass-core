import logging
import math
import os
import xml.etree.ElementTree as ET
from typing import List, Optional
from xml.dom import minidom

from ..utils.common import BaseAttrs, base_validators
from ..utils.geometry import convert_to_plus_minus_pi
import numpy as np
from attrs import define, field
from ..datatypes.path import (
    InterpolationPoint,
    MotionSample,
    PathPoint,
    TrackedPoint,
    TrajectorySample,
)

from .interpolation import SplineInterpolatedPath

follower_types = {"STANLEY_FOLLOWER": 1}


@define
class PathExecutorParams(BaseAttrs):
    max_end_ori_error: float = field(
        default=1.0,
        validator=base_validators.in_range(min_value=0.0, max_value=2 * math.pi),
    )  # Maximum allowed orientation error at the end of the path (rad)

    max_end_dist_error: float = field(
        default=0.3, validator=base_validators.in_range(min_value=0.0, max_value=1e6)
    )  # Maximum allowed displacement at the end of the path (m)

    min_interpolation_dist: float = field(
        default=2.0, validator=base_validators.in_range(min_value=1e-3, max_value=1e6)
    )  # Minimum path interpolation distance (m)

    follower_type: int = field(
        default=follower_types["STANLEY_FOLLOWER"],
        validator=base_validators.in_(list(follower_types.values())),
    )

    spline_segment_length: float = field(
        default=0.1, validator=base_validators.in_range(min_value=1e-6, max_value=1e6)
    )  # Segment length of the interpolated spline (m)

    min_segment_length: float = field(
        default=0.05, validator=base_validators.in_range(min_value=1e-6, max_value=1e3)
    )  # Spline segment step length (m)

    frame_id: str = field(default="map")

    def __str__(self) -> str:
        return f"""
        Executor Params:
        min_interpolation_dist: {self.min_interpolation_dist}
        max_end_ori_error: {self.max_end_ori_error}
        spline_segment_length: {self.spline_segment_length}
        min_segment_length: {self.min_segment_length}"""


class PathExecutor:
    """
    Path Executor class
    """

    def __init__(self, params: Optional[PathExecutorParams] = None):
        """
        Init a path executor

        :param params: Path executor parameters
        :type params: PathExecutorParams
        """
        if not params:
            params = PathExecutorParams()

        self.params: PathExecutorParams = params
        self.ref_path: List[PathPoint] = []  # a List of path pointers PathPt
        self.closest_point = TrackedPoint()
        self.execution_index: int = 0
        self.execution_s: float = 0.0
        self.total_length: float = 0.0  # total length of a recorded path
        self.interpolation = SplineInterpolatedPath(
            seg_len_init=self.params.min_segment_length,
            seg_len_max=self.params.min_interpolation_dist,
        )  # the interpolated path
        self.interpolation_xpoints: List[
            float
        ] = []  # the interpolated path x coordinates
        self.interpolation_ypoints: List[
            float
        ] = []  # the interpolated path y coordinates

        # Trajectory recording
        self.ref_traj = None
        self.traj_recording_idx: int = 0

        # Motion Recording
        self.rec_motion = None
        self.motion_recording_idx: int = 0

    def configure(self, config_file: str, nested_root_name: Optional[str] = None):
        """
        Configure the executor using a yaml file

        :param config_file: _description_
        :type config_file: str
        :param nested_root_name: _description_, defaults to None
        :type nested_root_name: Optional[str], optional
        """
        self.params.from_yaml(config_file, nested_root_name)

    def record_path_point(self, x: float, y: float, heading: float, vel: float) -> bool:
        """
        Adds a new point into the reference path of the executor

        :param x: Path point x-coordinates (m)
        :type x: float
        :param y: Path point y-coordinates (m)
        :type y: float
        :param heading: Path point heading (rad)
        :type heading: float
        :param vel: Linear velocity at this path point (m/s)
        :type vel: float

        :return: If the provided point is recorded
        :rtype: bool

        """
        path_point = PathPoint()
        delta_x: float = 0.0
        delta_y: float = 0.0

        # If the reference path is not empty get distance to the last point
        if self.ref_path:
            delta_x = x - self.ref_path[-1].x
            delta_y = y - self.ref_path[-1].y

        _delta_dist: float = math.sqrt(delta_x**2 + delta_y**2)

        # Record new point f the path is empty OR if the distance to the last point t=is more than the minimum path segment length
        if not self.ref_path or (_delta_dist > self.params.min_segment_length):
            path_point.idx = len(self.ref_path)
            path_point.x = x
            path_point.y = y
            path_point.speed = vel
            path_point.heading = heading

            self.ref_path.append(path_point)
            self.total_length += _delta_dist
            return True
        return False

    def record_trajectory_point(
        self, x: float, y: float, heading: float, time: float
    ) -> bool:
        """
        Record a new trajectory point

        :param x: Path point x-coordinates (m)
        :type x: float
        :param y: Path point y-coordinates (m)
        :type y: float
        :param heading: Path point heading (rad)
        :type heading: float
        :param time: Point recording time (s)
        :type time: float

        :return: If the provided point is recorded
        :rtype: bool
        """
        if not self.ref_traj:
            return False
        if self.traj_recording_idx < len(self.ref_traj.time):
            self.ref_traj.set_traj_point(
                x=x, y=y, heading=heading, time=time, idx=self.traj_recording_idx
            )
            self.traj_recording_idx += 1
            return True
        return False

    def record_motion_point(
        self,
        x: float,
        y: float,
        heading: float,
        time: float,
        linear_ctr_x: float,
        linear_ctr_y: float,
        angular_ctr: float,
    ) -> bool:
        """
        Record a new motion point

        :param x: Path point x-coordinates (m)
        :type x: float
        :param y: Path point y-coordinates (m)
        :type y: float
        :param heading: Path point heading (rad)
        :type heading: float
        :param time: Point recording time (s)
        :type time: float
        :param linear_ctr: Linear velocity control (m/s)
        :type linear_ctr: float
        :param angular_ctr: Angular velocity control (rad/s)
        :type angular_ctr: float


        :return: If the provided point is recorded
        :rtype: bool
        """
        if not self.rec_motion:
            return False
        if self.motion_recording_idx < len(self.rec_motion.time):
            self.rec_motion.set_traj_point(
                x=x, y=y, heading=heading, time=time, idx=self.motion_recording_idx
            )
            self.rec_motion.set_control_point(
                linear_control_x=linear_ctr_x,
                linear_control_y=linear_ctr_y,
                angular_control=angular_ctr,
                idx=self.motion_recording_idx,
            )
            self.motion_recording_idx += 1
            return True
        return False

    def save_path_to_xml(self, file_dir: str, file_name: str, frame_id: str) -> bool:
        """
        Saves the current path in the controller to an xml file

        :param file_dir: Path to saved file
        :type file_dir: str
        :param file_name: Saved file name
        :type file_name: str
        :param frame_id: The coordinates frame of the saved path points
        :type frame_id: str

        :raises OSError: Could not write to requested directory

        :return: If the path is saved to file
        :rtype: bool
        """
        # check if path is not available
        if not self.ref_path:
            return False
        # Generate the XML content
        root = ET.Element("Points", length=str(self.total_length))
        frame_name = ET.SubElement(root, "Frame")
        frame_name.set("id", str(frame_id))

        for i, path_point in enumerate(self.ref_path):
            path_point_element = ET.SubElement(root, "Point")
            path_point_element.set("idx", str(i))
            path_point_element.set("x", str(path_point.x))
            path_point_element.set("y", str(path_point.y))
            path_point_element.set("heading", str(path_point.heading))
            path_point_element.set("speed", str(path_point.speed))

        xml_content = minidom.parseString(ET.tostring(root)).toprettyxml(indent="  ")

        # Save the XML content to a file
        if not os.path.exists(file_dir):
            try:
                os.makedirs(file_dir)
            except OSError:
                raise
        try:
            file_path = os.path.join(file_dir, file_name + ".xml")
            with open(file_path, "w") as file:
                file.write(xml_content)
                return True
        except (FileNotFoundError, OSError):
            logging.exception(
                f"Could not save to requested file {file_dir}/{file_name}"
            )
            return False

    def read_path_from_xml(self, file_dir: str, file_name: str) -> bool:
        """
        Gets a reference path from xml file

        :param file_dir: Path to saved file
        :type file_dir: str
        :param file_name: Saved file name
        :type file_name: str

        :return: Is file found
        :rtype: bool
        """
        self.ref_path = []  # clear current path
        self.total_length = 0.0  # clear path length
        if os.path.exists(os.path.join(file_dir, file_name + ".xml")):
            try:
                doc = ET.parse(os.path.join(file_dir, file_name + ".xml"))
                root_element = doc.getroot()

                if root_element.tag == "Points":
                    for path_point in root_element.findall("Point"):
                        p = PathPoint()
                        p.idx = int(path_point.get("idx"))
                        p.x = float(path_point.get("x"))
                        p.y = float(path_point.get("y"))
                        p.heading = float(path_point.get("heading", 0.00))
                        p.speed = float(path_point.get("speed"))
                        self.ref_path.append(p)
                    self.total_length = float(root_element.get("length"))
                    self.params.frame_id = root_element.findall("Frame")[0].get("id")
                    return True
                else:
                    logging.error("File does not contain any points")
                    return False

            except Exception as e:
                logging.error(f"XML file parsing error: {str(e)}")
                return False

        logging.error(f"File not found at {file_dir}/{file_name}")
        return False

    def start_path_recording(self) -> None:
        """
        Sets the executor to start recording a new path
        """
        self.ref_path = []  # clear current path
        self.total_length = 0.0  # clear path length

    def start_trajectory_recording(
        self, recording_period: float, recording_step: float, frameid: str = "map"
    ) -> None:
        """
        Starts a new trajectory recording

        :param recording_period: Trajectory recording period (s)
        :type recording_period: float
        :param recording_step: Trajectory recording step (s)
        :type recording_step: float
        :param frameid: Trajectory points coordinates frame, defaults to 'map'
        :type frameid: str, optional
        """
        _traj_len = int(recording_period / recording_step) - 1
        self.ref_traj = TrajectorySample(length=_traj_len, frame_id=frameid)
        self.traj_recording_idx = 0

    def start_motion_recording(
        self, recording_period: float, recording_step: float, frameid: str = "map"
    ) -> None:
        """
        Starts a new motion recording

        :param recording_period: Motion recording period (s)
        :type recording_period: float
        :param recording_step: Motion recording step (s)
        :type recording_step: float
        :param frameid: Motion points coordinates frame, defaults to 'map'
        :type frameid: str, optional
        """
        _traj_len = int(recording_period / recording_step) - 1
        self.rec_motion = MotionSample(length=_traj_len, frame_id=frameid)
        self.motion_recording_idx = 0

    def save_motion_to_csv(self, file_location: str, file_name: str) -> bool:
        """
        Save recorded motion sample to a csv file

        :param file_location: File location
        :type file_location: str
        :param file_name: File name
        :type file_name: str

        :return: File is saved
        :rtype: bool
        """
        if not self.rec_motion:
            return False
        return self.rec_motion.save_to_csv(file_location, file_name)

    def _init_new_interpolation(self, x: float, y: float) -> None:
        """
        Interpolate a new path segment

        :param x: Robot x-coordinates
        :type x: float
        :param y: Robot y-coordinates
        :type y: float
        """
        self.interpolation = SplineInterpolatedPath(
            seg_len_init=self.params.min_segment_length,
            seg_len_max=self.params.min_interpolation_dist,
        )
        self.interpolation_xpoints = []
        self.interpolation_ypoints = []

        # Check closest point and update the execution index
        closest_idx = self.get_closest_point_index(self.ref_path, x, y)
        if not closest_idx:
            self.execution_index = 0
        else:
            self.execution_index = closest_idx - 1 if closest_idx > 0 else 0

    def _check_interpolation_distance(self) -> bool:
        """
        If the remaining reference path length is less than the minimum interpolation distance -> set the segment length to the total path length

        :return: If interpolation distance is within path length
        :rtype: bool
        """
        # If the remaining reference path length is less than the minimum interpolation distance -> take as one
        if self.params.min_interpolation_dist >= self.total_length:
            logging.info(
                f"Kompass Path Executor: Interpolating the remaining path as one segment with length {self.total_length}"
            )
            self.params.min_interpolation_dist = self.total_length
        return self.params.min_interpolation_dist < self.total_length

    def _get_interpolation_start_index(self) -> int:
        """
        Get the start index of the interpolation point

        :return: Interpolation start index
        :rtype: int
        """
        spline_start_index = self.execution_index - 1

        # Check if more that three points are available (spline function requires minimum three points)
        while (spline_start_index > 0) and (
            (self.execution_index - spline_start_index) <= 3
        ):
            spline_start_index -= 1

        # Augmenting path with more intermediate points
        if spline_start_index < 0:
            logging.debug("Augmenting path points to interpolate")
            _got_augmented = self._augment_path_points()
            if _got_augmented:
                return self._get_interpolation_start_index()

        return spline_start_index

    def _augment_path_points(self) -> bool:
        """
        Augments reference path with more intermediate points to interpolate

        :return: If path was augmented
        :rtype: bool
        """
        if len(self.ref_path) > 1:
            new_path = []
            for i in range(len(self.ref_path) - 1):
                new_path.append(self.ref_path[i])
                mean_point = (self.ref_path[i] + self.ref_path[i + 1]) / 2
                new_path.append(mean_point)
            new_path.append(self.ref_path[-1])
            self.ref_path = new_path
            return True
        return False

    def interpolate_path_spline(self, x: float, y: float, frame_id="map") -> bool:
        """
        Gets an interpolated path segment (self.interpolation) from the reference path (self.ref_path)

        :param x: Robot x-coordinates (m)
        :type x: float
        :param y: Robot y-coordinates (m)
        :type y: float
        :param frame_id: Path points frame, defaults to 'map'
        :type frame_id: str, optional

        :return: If interpolation is successful
        :rtype: bool
        """
        # Check if a previous interpolation is available and closest point is still before the end of the interpolation
        if self.closest_point:
            if (
                self.closest_point.s <= self.params.min_interpolation_dist
                and self.closest_point.s > 0.0
            ):
                # No need for new interpolation -> Return
                return True

        # Init empty interpolation spline
        self._init_new_interpolation(x, y)

        # Check if the remaining reference path length is less than the minimum interpolation distance
        self._check_interpolation_distance()

        interpolation_points = []

        spline_start_index: int = self._get_interpolation_start_index()
        logging.warn(f"Got index {spline_start_index}")

        if spline_start_index > len(self.ref_path) - 3:
            logging.warn(
                "Kompass Path Executor: Need more points to interpolate the path"
            )
            return False

        _arc_dist: float = 0.0  # total arc length of the segment
        _delta_dist: float = 0.0  # segment iteration distance

        start_index: int = spline_start_index if spline_start_index > 0 else 0

        for i in range(len(self.ref_path)):
            # Check if the current arc distance length is still within the spline length and the iteration index did not reach the end of the reference path
            if (_arc_dist < self.params.min_interpolation_dist) and (
                start_index + i < len(self.ref_path)
            ):
                _seg_length = 0.0

                if start_index + i > 0:
                    # Set _seg_length to the distance between two path points
                    _seg_length = math.sqrt(
                        (
                            self.ref_path[start_index + i].x
                            - self.ref_path[start_index + i - 1].x
                        )
                        ** 2
                        + (
                            self.ref_path[start_index + i].y
                            - self.ref_path[start_index + i - 1].y
                        )
                        ** 2
                    )
                    # Increase the iteration distance by the new _seg_length
                    _delta_dist += _seg_length

                # Increase the current arc distance
                _arc_dist += _seg_length

                # Check if the iteration distance is still within the spline length or the iteration point is at the start
                if (_delta_dist > self.params.spline_segment_length) or (
                    i == start_index
                ):
                    # Add new interpolation point
                    pp = InterpolationPoint(
                        _arc_dist,
                        self.ref_path[start_index + i].x,
                        self.ref_path[start_index + i].y,
                        self.ref_path[start_index + i].heading,
                    )
                    # Add to List of interpolation points
                    interpolation_points.append(pp)
                    # rest iteration distance
                    _delta_dist = 0.0

        # turn into array
        pp_array = np.array(interpolation_points, dtype=object)

        # sort interpolation points (spline needs increasing order)
        sorted_indices = np.argsort([p.x for p in pp_array])
        sorted_points = [interpolation_points[i] for i in sorted_indices]

        if len(sorted_points) > 3:
            # set the interpolation using the sorted interpolation points
            self.interpolation.set_path_points(sorted_points, frame_id)
            s = 0.0
            while s <= _arc_dist:
                x, y = self.interpolation(s)
                self.interpolation_xpoints.append(x)
                self.interpolation_ypoints.append(y)
                s += self.params.spline_segment_length

            return True
        else:
            logging.warn(
                "Kompass Path Executor: Need more points to interpolate the path"
            )
            return False

    def reached_end(self, x: float, y: float, ori: float) -> tuple[bool, float, float]:
        """
        Checks if the robot reached the end of the reference path and evaluate the tracking errors

        :param x: x-coordinates of the robot
        :type x: float
        :param y: y-coordinates of the robot
        :type y: float
        :param ori: _description_
        :type ori: float
        :return: Reached end, lateral distance error, orientation error
        :rtype: bool, float, float
        """
        reached = False

        try:
            path_end = self.ref_path[-1]

        except IndexError:
            logging.exception("No path points are available -> End is already reached")
            return True, 0.0, 0.0

        lat_dist = math.sqrt((x - path_end.x) ** 2 + (y - path_end.y) ** 2)
        ori_error: float = convert_to_plus_minus_pi(abs(path_end.heading - ori))
        if (
            abs(lat_dist) < self.params.max_end_dist_error
            and abs(ori_error) < self.params.max_end_ori_error
        ):
            reached = True
        return reached, lat_dist, ori_error

    @classmethod
    def get_closest_point_index(
        cls, ref_path: List[PathPoint], x: float, y: float
    ) -> Optional[int]:
        """
        Gets the index of the closest point on the given path to the robot

        :param ref_path: Current path
        :type ref_path: List[PathPt]
        :param x: Robot x-coordinates
        :type x: float
        :param y: Robot y-coordinates
        :type y: float
        :return: Closest point index or None if path is empty
        :rtype: int | None
        """
        lat_dists = [
            math.sqrt((x - path_pt.x) ** 2 + (y - path_pt.y) ** 2)
            for path_pt in ref_path
        ]
        if lat_dists:
            idx_closest_point = lat_dists.index(min(lat_dists))
            return idx_closest_point
        else:
            return None

    @classmethod
    def get_closest_point(
        cls, ref_path: List[PathPoint], x: float, y: float
    ) -> Optional[PathPoint]:
        """
        Gets the closest point on the given path to the robot

        :param ref_path: Current path
        :type ref_path: List[PathPt]
        :param x: Robot x-coordinates
        :type x: float
        :param y: Robot y-coordinates
        :type y: float

        :return: Closest point or None (if path is empty)
        :rtype: PathPoint
        """
        idx_closest_point = cls.get_closest_point_index(ref_path, x, y)
        if not idx_closest_point:
            return None
        if idx_closest_point >= 0:
            return ref_path[idx_closest_point]
        else:
            return None

    @classmethod
    def get_path_heading(cls, ref_path: List[PathPoint], idx: int) -> float:
        """
        Returns path heading at given index

        :param ref_path: Reference path
        :type ref_path: List[PathPoint]
        :param idx: Path point index
        :type idx: int

        :raises IndexError: Requested heading at invalid path point index

        :return: Path heading (rad)
        :rtype: float
        """
        if idx == len(ref_path) - 1:
            # Get heading from previous point
            _heading: float = math.atan2(
                ref_path[idx].y - ref_path[idx - 1].y,
                ref_path[idx].x - ref_path[idx - 1].x,
            )
        try:
            _heading: float = math.atan2(
                ref_path[idx + 1].y - ref_path[idx].y,
                ref_path[idx + 1].x - ref_path[idx].x,
            )
            return _heading
        except IndexError:
            logging.exception("Requested index is out of path points range")
            raise
