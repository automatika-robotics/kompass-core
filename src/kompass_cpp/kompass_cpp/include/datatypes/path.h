#pragma once

#include "utils/spline.h"
#include <Eigen/Dense>
#include <cmath>
#include <cstddef>
#include <math.h>
#include <vector>

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

// TODO change to floats!!!
// Point in 3D space
class Point : public Eigen::Vector3d {
public:
  // Default constructor
  Point() : Eigen::Vector3d(0.0, 0.0, 0.0) {}
  Point(double x, double y, double z = 0.0) : Eigen::Vector3d(x, y, z) {}

  Point(Eigen::Vector3d &ref) : Eigen::Vector3d(ref) {}

  // Accessors
  double x() const { return (*this)(0); }
  double y() const { return (*this)(1); }
  double z() const { return (*this)(2); }

  // Setters
  void setX(double const value) { (*this)(0) = value; }
  void setY(double const value) { (*this)(1) = value; }
  void setZ(double const value) { (*this)(2) = value; }
};

// Structure for Path Control parameters
struct Path {
  std::vector<Point> points;  // List of points defining the path
  std::vector<Path> segments; // List of path segments
  tk::spline *_spline;

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
