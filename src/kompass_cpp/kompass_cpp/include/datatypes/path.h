#pragma once

#include "datatypes/control.h"
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
      : x(poseX), y(poseY), yaw(PoseYaw), speed(speedValue){};

  void update(const Kompass::Control::Velocity2D &vel, const float timeStep) {
    this->x +=
        (vel.vx() * cos(this->yaw) - vel.vy() * sin(this->yaw)) * timeStep;
    this->y +=
        (vel.vx() * sin(this->yaw) + vel.vy() * cos(this->yaw)) * timeStep;
    this->yaw += vel.omega() * timeStep;
  };
};

// Point in 3D space
typedef Eigen::Vector3f Point;

// Structure for Path Control parameters
struct Path {
  std::vector<Point> points;  // Vector of points defining the path
  std::vector<Path> segments; // List of path segments
  tk::spline *_spline;
  // get all x values from points
  const std::vector<float>& getX() const;
  // get all y values from points
  const std::vector<float>& getY() const;
  // get all z values from points
  const std::vector<float>& getZ() const;
  // Max interpolation distance and total path distance are updated from user
  // config
  double _max_interpolation_dist{0.0}, _max_path_length{10.0};
  // Max segment size and max total path points size is calculated after
  // interpolation
  int max_segment_size{10};
  size_t max_size{10};
  size_t max_interpolation_iterations{500}; // Max number of iterations for interpolation between two path points

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

private:
  std::vector<float> X_; // Vector of X coordinates
  std::vector<float> Y_; // Vector of Y coordinates
  std::vector<float> Z_; // Vector of Z coordinates
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
