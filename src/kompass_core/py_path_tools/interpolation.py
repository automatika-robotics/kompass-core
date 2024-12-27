import logging
import math
from typing import List, Optional

from ..utils import geometry as GeometryUtils
import numpy as np
from ..datatypes.path import InterpolationPoint, Point2D, Range2D, TrackedPoint
from scipy.interpolate import CubicSpline


class Spline:
    def __init__(self):
        self.x_points: np.ndarray = np.array([])
        self.y_points: np.ndarray = np.array([])
        self.func: Optional[CubicSpline] = None

    def set_points(self, x_points: np.ndarray, y_points: np.ndarray) -> None:
        """
        Sets a spline with given set of points (x, y)

        :param x_points: X coordinates of the spline points
        :type x_points: np.ndarray
        :param y_points: Y coordinates of the spline points
        :type y_points: np.ndarray
        """
        # arrange x points in increasing order
        sorted_indices = np.argsort(x_points)
        self.x_points = np.array(x_points[sorted_indices])
        self.y_points = np.array(y_points[sorted_indices])
        self.func = CubicSpline(self.x_points, self.y_points)

    def __call__(self, x: float) -> float:
        """
        Gets the CubicSpline interpolation at point x as a float

        :param x: Interpolation point
        :type x: float

        :raises TypeError: If the requested point in out of the spline limits

        :return: Interpolation result
        :rtype: float
        """
        try:
            if not self.func:
                raise ValueError("No points are set for interpolation.")
            _output = self.func(x)
        except TypeError as e:
            raise TypeError(
                "Input point to spline is out of interpolation limits"
            ) from e
        return _output.item().real

    def get_points_len(self) -> int:
        """
        Gets the number of interpolated points

        :return: Number of points
        :rtype: int
        """
        return self.x_points.size

    def limit_check(self, x: float) -> bool:
        """
        Checks if a value is within the spline points limits

        :param x: Value under evaluation
        :type x: float

        :return: If the value is within the spline limits
        :rtype: bool
        """
        if self.x_points.any():
            return x > self.x_points.min() and x < self.x_points.max()
        return False


class SplineInterpolatedPath:
    """
    Interpolated path data class
    """

    # Max path tracking error when checking if end is reached
    LAT_DIST_ERROR_MAX_TRACKING = 0.1

    def __init__(self, seg_len_init: float, seg_len_max: float):
        self.spline_x = Spline()
        self.spline_y = Spline()
        self.spline_yaw = Spline()

        self.length = 0.0  # total length of the path
        self.range_x = Range2D(0, 0)
        self.range_y = Range2D(0, 0)
        self.end_point = Point2D(0, 0)  # coordinates of the last path point
        self.frame_id = "map"  # name of the points coordinates frame
        self.seg_len_init = seg_len_init  # Initial arc step length
        self.seg_len_max = seg_len_max  # Max arc length for the interpolation

    def set_path_points(self, path_pts: List[InterpolationPoint], frame_id):
        """
        Sets the interpolated path x,y splines from a set of interpolation points

        :param path_pts: List of interpolation points
        :type path_pts: _type_
        :param frame_id: Frame of the interpolated path
        :type frame_id: _type_
        """
        # Init three spline functions to interpolate point coordinates x,y and path distance s
        spline_s_points = np.array([p.s for p in path_pts])
        spline_x_points = np.array([p.x for p in path_pts])
        spline_y_points = np.array([p.y for p in path_pts])

        # Set the spline points spline_x(x)=s, spline_y(y)=s
        self.spline_x.set_points(spline_s_points, spline_x_points)
        self.spline_y.set_points(spline_s_points, spline_y_points)

        # Point at maximum distance
        max_point = max(path_pts, key=lambda p: p.s)

        self.length = max_point.s

        # Coordinates frame for the points
        self.frame_id = frame_id

        # Coordinates at maximum distance
        self.end_point.x = max_point.x
        self.end_point.y = max_point.y

        self.range_x.min_val, self.range_x.max_val = (
            min(p.x for p in path_pts),
            max(p.x for p in path_pts),
        )
        self.range_y.min_val, self.range_y.max_val = (
            min(p.y for p in path_pts),
            max(p.y for p in path_pts),
        )
        return

    def __call__(self, s: float) -> tuple[float, float]:
        """
        Returns the path coordinates corresponding to curvilinear distance s

        :param s: curvilinear distance (m)
        :type s: float
        :return: (x,y) coordinates at s
        :rtype: tuple[float]
        """
        return self.spline_x(s), self.spline_y(s)

    def get_interpolated_pose(self, s: float) -> tuple[float, float, float]:
        """
        Returns the path pose corresponding to curvilinear distance s

        :param s: curvilinear distance (m)
        :type s: float
        :return: (x, y, yaw) pose at s
        :rtype: tuple[float]
        """
        return self.spline_x(s), self.spline_y(s), self.spline_yaw(s)

    def set_path_points_yaw(self, path_pts, frame_id):
        """
        Sets the interpolated path x,y,yaw splines from a set of interpolation points

        :param path_pts: List of interpolation points
        :type path_pts: _type_
        :param frame_id: Frame of the interpolated path
        :type frame_id: _type_
        """
        self.set_path_points(path_pts, frame_id)

        spline_s_points = np.array([p.s for p in path_pts])
        spline_yaw_points = np.array([p.yaw for p in path_pts])

        self.spline_yaw.set_points(spline_s_points, spline_yaw_points)
        return

    def get_path_segment(self, s_max: float):
        """
        Returns a segment of the total path where s < s_max

        :param s_max: maximum curvilinear distance (m)
        :type s_max: float
        :return: Interpolated path segment
        :rtype: SplineInterpolatedPath
        """
        # Get the points where the distance is less than s_max
        sub_s_points = self.spline_x.x_points[self.spline_x.x_points < s_max]
        sub_y_points = self.spline_y.y_points[: sub_s_points.shape[0]]
        sub_x_points = self.spline_x.y_points[: sub_s_points.shape[0]]

        # define new spline interpolation
        sub_path = SplineInterpolatedPath(
            seg_len_init=self.seg_len_init, seg_len_max=s_max
        )
        # set the new spline using all the points < s_max
        sub_path.spline_x.set_points(sub_s_points, sub_x_points)
        sub_path.spline_y.set_points(sub_s_points, sub_y_points)
        sub_path.length = max(sub_s_points)

        return sub_path

    def get_closest_path_point(
        self, x: float, y: float, yaw: float, speed: float
    ) -> Optional[TrackedPoint]:
        """
        Returns information about the closest point of the path to the provided pose (x,y,yaw)

        :param seg_len_init: Incremental arc length (m)
        :type seg_len_init: float
        :param x: x-coordinates of the robot (m)
        :type x: float
        :param y: y-coordinates of the robot (m)
        :type y: float
        :param yaw: orientation of the robot (rad)
        :type yaw: float
        :param speed: speed of the robot in m/s
        :type speed: float
        :return: _description_
        :rtype: TrackedPoint
        """
        min_dist = 1e6
        dist = 1e6
        forward_dist = 0.0
        pp_x, pp_y, pp_s = None, None, None
        s = self.seg_len_init
        tracked_point = TrackedPoint()
        s_incr = self.seg_len_init

        while s_incr >= self.seg_len_init:
            # Iterate around the initial curvilinear distance seg_len_init in the positive direction
            while dist <= min_dist and s <= self.seg_len_max:
                # check if the curvilinear distance s is within the spline limits
                if self.spline_x.limit_check(s) and self.spline_y.limit_check(s):
                    # get path points at curvilinear distance s
                    pp_x, pp_y = self.spline_x(s), self.spline_y(s)
                    pp_s = s
                    # update the min distance
                    min_dist = dist
                    # get new distance to path point
                    dist = np.sqrt((pp_x - x) ** 2 + (pp_y - y) ** 2)
                    forward_dist = abs(pp_x - x)
                s += s_incr

            # update tracked point if a spline interpolation is available:
            if pp_x and pp_y:
                tracked_point.x = pp_x
                tracked_point.y = pp_y

            # Move to a finer iteration
            s_incr /= 2.0

            # Iterate around the initial curvilinear distance seg_len_init in the negative direction
            while dist <= min_dist and s >= -self.seg_len_max:
                # check if the curvilinear distance s is within the spline limits
                if self.spline_x.limit_check(s) and self.spline_y.limit_check(s):
                    # get path points at curvilinear distance s
                    pp_x, pp_y = self.spline_x(s), self.spline_y(s)
                    pp_s = s

                    # update the min distance
                    min_dist = dist
                    # get new distance to path point
                    dist = np.sqrt((pp_x - x) ** 2 + (pp_y - y) ** 2)
                    forward_dist = abs(pp_x - x)

                s -= s_incr
            # update tracked point if a spline interpolation is available:
            if pp_x and pp_y:
                tracked_point.x = pp_x
                tracked_point.y = pp_y
            s_incr /= 2.0

        # If no interpolation was available arount the input point
        if pp_x is None or pp_y is None or pp_s is None:
            logging.error(
                "No interpolation is available around the requested point -> Close compute closest point"
            )
            return None

        tracked_point.s = pp_s
        minimum_distance = min_dist

        s_incr = 0.5  # 0.5m tangent lookahead distance

        # to compute lateral distance, curviture, tangent_orientation
        x1, x2, x3, x4, x5 = (
            self.spline_x(s - 2 * s_incr),
            self.spline_x(s - s_incr),
            self.spline_x(s),
            self.spline_x(s + s_incr),
            self.spline_x(s + 2 * s_incr),
        )
        y1, y2, y3, y4, y5 = (
            self.spline_y(s - 2 * s_incr),
            self.spline_y(s - s_incr),
            self.spline_y(s),
            self.spline_y(s + s_incr),
            self.spline_y(s + 2 * s_incr),
        )
        alpha1, alpha2, alpha3, alpha4 = (
            math.atan2(y2 - y1, x2 - x1),
            math.atan2(y3 - y2, x3 - x2),
            math.atan2(y4 - y3, x4 - x3),
            math.atan2(y5 - y4, x5 - x4),
        )
        ori2, _, ori4 = (
            GeometryUtils.add_angle(
                alpha1, GeometryUtils.add_angle(alpha2, -alpha1) / 2.0
            ),
            GeometryUtils.add_angle(
                alpha2, GeometryUtils.add_angle(alpha3, -alpha2) / 2.0
            ),
            GeometryUtils.add_angle(
                alpha3, GeometryUtils.add_angle(alpha4, -alpha3) / 2.0
            ),
        )

        tracked_point.tangent_ori = GeometryUtils.convert_to_0_2pi(alpha3)

        beta = math.atan2(y - tracked_point.y, x - tracked_point.x)

        gamma = tracked_point.tangent_ori - GeometryUtils.convert_to_0_2pi(beta)

        tracked_point.lat_dist = min_dist * np.sin(gamma)

        tracked_point.forward_dist = forward_dist

        deltaOri42 = ori4 - ori2

        tracked_point.curv = deltaOri42 / (2.0 * s_incr)

        if speed < 0.0:
            tracked_point.ori_err = GeometryUtils.convert_to_0_2pi(
                tracked_point.tangent_ori - np.pi
            ) - GeometryUtils.convert_to_0_2pi(yaw)
            tracked_point.s_dot = (
                -speed
                * np.cos(tracked_point.ori_err)
                / (1 - tracked_point.curv * minimum_distance)
            )
            tracked_point.lat_vel = (
                -tracked_point.curv * speed * np.sin(tracked_point.ori_err)
            )
        else:
            tracked_point.ori_err = (
                tracked_point.tangent_ori - GeometryUtils.convert_to_0_2pi(yaw)
            )
            tracked_point.s_dot = (
                speed
                * np.cos(tracked_point.ori_err)
                / (1 - tracked_point.curv * minimum_distance)
            )
            tracked_point.lat_vel = (
                tracked_point.curv * speed * np.sin(tracked_point.ori_err)
            )

        tracked_point.ori_err = GeometryUtils.convert_to_plus_minus_pi(
            tracked_point.ori_err
        )

        return tracked_point

    def reached_end(self, x: float, y: float) -> bool:
        """
        Checks if the end of path is reached by the robot

        :param path: Provided path
        :type path: autonav_path_follower.PathClasses.Path
        :param x: Robot x-coordinates
        :param x: Robot x-coordinates (m)
        :type x: float
        :param y: Robot y-coordinates (m)
        :type y: float
        :return: Reached End
        :rtype: bool
        """
        # TODO Consider orientation at the end of path
        path_end = self.end_point
        lat_dist = math.sqrt((x - path_end.x) ** 2 + (y - path_end.y) ** 2)
        if abs(lat_dist) < self.LAT_DIST_ERROR_MAX_TRACKING:
            return True
        return False
