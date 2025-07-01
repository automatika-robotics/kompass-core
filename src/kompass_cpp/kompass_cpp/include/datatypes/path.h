#pragma once

#include "datatypes/control.h"
#include "utils/spline.h"
#include <Eigen/Dense>
#include <cmath>
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
      : x(poseX), y(poseY), yaw(PoseYaw), speed(speedValue) {};

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
  std::vector<Path> segments; // List of path segments
  // get all x values from points
  const Eigen::VectorXf getX() const;
  // get all y values from points
  const Eigen::VectorXf getY() const;
  // get all z values from points
  const Eigen::VectorXf getZ() const;
  // Max segment size and max total path points size is calculated after
  // interpolation
  int max_segment_size{10};

  Path(const Path &other) = default;

  Path(const std::vector<Point> &points = {}, const size_t new_max_size = 10);

  Path(const Eigen::VectorXf &x_points, const Eigen::VectorXf &y_points,
       const Eigen::VectorXf &z_points, const size_t new_max_size = 10);

  size_t getMaxNumSegments();

  void setMaxLength(double max_length);

  void resize(const size_t max_new_size);

  bool endReached(State currentState, double minDist);

  Point getEnd() const;

  Point getStart() const;

  Point getIndex(const size_t index) const;

  Path getPart(const size_t start, const size_t end,
               const size_t max_part_size = 0) const;

  void pushPoint(const Point &point);

  size_t getSize() const;

  size_t getMaxSize() const;

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

  struct Iterator {
    using iterator_category = std::forward_iterator_tag;
    using value_type = Point;
    using difference_type = std::ptrdiff_t;

    Iterator(const Path &p, size_t idx) : path(p), index(idx) {}

    Point operator*() const { return path.getIndex(index); }

    Iterator &operator++() {
      ++index;
      return *this;
    }

    bool operator!=(const Iterator &other) const {
      return index != other.index;
    }

  private:
    const Path &path;
    size_t index;
  };

  Iterator begin() const { return Iterator(*this, 0); }
  Iterator end() const { return Iterator(*this, current_size_); }

private:
  Eigen::VectorXf X_;                   // Vector of X coordinates
  Eigen::VectorXf Y_;                   // Vector of Y coordinates
  Eigen::VectorXf Z_;                   // Vector of Z coordinates
  size_t current_size_{0};              // Current size of the path
  size_t max_interpolation_iterations_; // Max number of iterations for
                                        // interpolation between two path points
  // Max interpolation distance and total path distance are updated from user
  // config
  float max_path_length_{10.0}, max_interpolation_dist_{0.0};
  tk::spline *spline_;
  size_t max_size_{10};
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
