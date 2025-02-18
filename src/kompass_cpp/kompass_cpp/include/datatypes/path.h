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
  double x; // Speed on x-axis (m/s)
  double y;
  double yaw; // angular velocity (rad/s)
  double speed;

  State(double poseX = 0.0, double poseY = 0.0, double PoseYaw = 0.0,
        double speedValue = 0.0)
      : x(poseX), y(poseY), yaw(PoseYaw), speed(speedValue) {}
};

// Point in 3D space
class Point : public Eigen::Vector3f {
public:
  // Default constructor
  Point() : Eigen::Vector3f(0.0, 0.0, 0.0) {}
  Point(float x, float y, float z = 0.0) : Eigen::Vector3f(x, y, z) {}

  Point(Eigen::Vector3f &ref) : Eigen::Vector3f(ref) {}

  // Accessors
  float x() const { return (*this)(0); }
  float y() const { return (*this)(1); }
  float z() const { return (*this)(2); }

  // Setters
  void setX(float const value) { (*this)(0) = value; }
  void setY(float const value) { (*this)(1) = value; }
  void setZ(float const value) { (*this)(2) = value; }
};

// Structure for Path Control parameters
struct Path {
  std::vector<Point> points;  // List of points defining the path
  std::vector<Path> segments; // List of path segments
  tk::spline *_spline;
  // Max interpolation distance and total path distance are updated from user
  // config
  double _max_interpolation_dist{0.0}, _max_path_length{10.0};
  // Max segment size and max total path points size is calculated after
  // interpolation
  int max_segment_size{10}, max_size{10};

  Path(const std::vector<Point> &points = {});

  size_t getMaxNumSegments();

  void setMaxLength(double max_length);

  bool endReached(State currentState, double minDist);

  Point getEnd() const;

  Point getStart() const;

  float getEndOrientation() const;

  float getStartOrientation() const;

  float getOrientation(const size_t index) const;

  static float distance(const Point &p1, const Point &p2);

  float minDist(const std::vector<Point> &others) const;

  // Function to compute the total path length
  float totalPathLength() const;

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
