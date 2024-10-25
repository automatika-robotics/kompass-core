#pragma once

#include <cmath>
#include <cstddef>
#include <math.h>
#include <vector>
#include "utils/spline.h"


namespace Path {

enum class InterpolationType { LINEAR, CUBIC_SPLINE, HERMITE_SPLINE };

// Structure for Position
struct State {
  double x; // Speed on x-asix (m/s)
  double y;
  double yaw; // angular velocity (rad/s)
  double speed;

  State(double poseX = 0.0, double poseY = 0.0, double PoseYaw = 0.0,
        double speedValue = 0.0)
      : x(poseX), y(poseY), yaw(PoseYaw), speed(speedValue) {}
};

// Structure for a point in 2D space
struct Point {
  double x; // X coordinate
  double y; // Y coordinate

  Point(double xCoord = 0.0, double yCoord = 0.0) : x(xCoord), y(yCoord) {}
};

// Structure for Path Control parameters
struct Path {
  std::vector<Point> points;  // List of points defining the path
  std::vector<Path> segments; // List of path segments
  tk::spline* _spline;

  Path(const std::vector<Point> &points = {});

  size_t getMaxNumSegments();

  bool endReached(State currentState, double minDist);

  Point getEnd() const;

  Point getStart() const;

  double getEndOrientation() const;

  double getStartOrientation() const;

  double getOrientation(const size_t index) const;

  static double distance(const Point &p1, const Point &p2);

  double minDist(const std::vector<Point> &others) const;

  // Function to compute the total path length
  double totalPathLength() const;

  Point getPointAtLength(const double length) const;

  size_t getNumberPointsInLength(double length) const;

  void interpolate(double max_interpolation_point_dist, InterpolationType type);

  // Segment the path by a given segment path length [m]
  void segment(double pathSegmentLength);

  // Segment using a number of segments
  void segmentBySegmentNumber(int numSegments);

  // Segment using a segment points number
  void segmentByPointsNumber(int segmentLength);

};

struct PathPosition {
  size_t index{0};         // Index of the point in the segment
  size_t segment_index{0}; // Index of the segment in the Path
  double segment_length{-1.0};
  double parallel_distance{0.0};
  double normal_distance{0.0};
  State state;
};

} // namespace Path
