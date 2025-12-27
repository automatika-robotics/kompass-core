#pragma once

#include "datatypes/control.h"
#include <Eigen/Dense>
#include <cmath>
#include <cstddef>
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

  struct View {
    // Explicitly set columns to 1 so Eigen knows this is a Vector, not a
    // Matrix
    using ConstSegment = Eigen::Block<const Eigen::VectorXf, Eigen::Dynamic, 1>;

    ConstSegment X;
    ConstSegment Y;
    ConstSegment Z;
    ConstSegment Curvature;

    // Constructor
    View(const Path &parent, size_t start, size_t length)
        : X(parent.X_.segment(start, length)),
          Y(parent.Y_.segment(start, length)),
          Z(parent.Z_.segment(start, length)),
          Curvature(parent.Curvature_.segment(start, length)) {}

    // --- Accessors ---
    size_t getSize() const { return X.size(); }

    const float *getXPointer() const { return X.data(); }
    const float *getYPointer() const { return Y.data(); }
    const float *getZPointer() const { return Z.data(); }
    const float *getCurvaturePointer() const { return Curvature.data(); }

    Point getIndex(size_t index) const {
      // Return a point constructed from the view's data at 'index'
      return Point(X(index), Y(index), Z(index));
    }

    double getCurvature(size_t index) const { return Curvature(index); }

    float totalSegmentLength() const {
      float length = 0.0f;
      for (size_t i = 0; i < getSize() - 1; ++i) {
        length += Path::distance(getIndex(i), getIndex(i + 1));
      }
      return length;
    }

    // --- Iterator Implementation ---
    struct Iterator {
      using iterator_category = std::forward_iterator_tag;
      using value_type = Point;
      using difference_type = std::ptrdiff_t;
      using pointer = const Point *;
      using reference = const Point;

      // The iterator holds a reference to the view and the current index
      Iterator(const View &v, size_t idx) : view(v), index(idx) {}

      // Dereference operator returns a Point
      Point operator*() const { return view.getIndex(index); }

      // Pre-increment
      Iterator &operator++() {
        ++index;
        return *this;
      }

      // Post-increment
      Iterator operator++(int) {
        Iterator temp = *this;
        ++index;
        return temp;
      }

      // Equality comparison
      bool operator==(const Iterator &other) const {
        // We compare indices. (Assuming they belong to the same View)
        return index == other.index;
      }

      bool operator!=(const Iterator &other) const {
        return index != other.index;
      }

    private:
      const View &view;
      size_t index;
    };

    // --- Begin/End Support ---

    Iterator begin() const { return Iterator(*this, 0); }
    Iterator end() const { return Iterator(*this, getSize()); }
  };

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

  void setMaxLength(double max_length);

  void resize(const size_t max_new_size);

  bool endReached(State currentState, double minDist);

  Point getEnd() const;

  Point getStart() const;

  Point getIndex(const size_t index) const;

  Path::View getPart(const size_t start, const size_t end) const;

  void pushPoint(const Point &point);

  size_t getSize() const;

  size_t getMaxSize() const;

  float getEndOrientation() const;

  float getStartOrientation() const;

  float getOrientation(const size_t index) const;

  // Get curvature at index
  double getCurvature(const size_t index) const;

  // distance between two points
  static inline float distance(const Point &p1, const Point &p2) {
    return (p1 - p2).norm();
  }

  // Squared distance between two points (faster for comparison)
  static inline float distanceSquared(const Point &p1, const Point &p2) {
    return (p1 - p2).squaredNorm();
  }

  static inline float distanceSquared(const State &state,
                                      const Point &point) {

    return (Point(state.x, state.y, 0.0) - point).squaredNorm();
  }

  // Function to compute the total path length
  float totalPathLength() const;

  Path::View getSegment(size_t segment_index) const;

  size_t getSegmentSize(size_t segment_index) const;

  size_t getSegmentStartIndex(size_t segment_index) const;

  size_t getSegmentEndIndex(size_t segment_index) const;

  size_t getNumSegments() const;

  void interpolate(double max_interpolation_point_dist, InterpolationType type);

  // Segment the path by a given segment path length [m]
  void segment(double pathSegmentLength);

  Point getSegmentStart(size_t segment_index) const;

  Point getSegmentEnd(size_t segment_index) const;

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
  Eigen::VectorXf X_;         // Vector of X coordinates
  Eigen::VectorXf Y_;         // Vector of Y coordinates
  Eigen::VectorXf Z_;         // Vector of Z coordinates
  Eigen::VectorXf Curvature_; // Curvature values
  // NOTE: segment_indices_ are used to store the starting index of each segment
  // on the path If segment_indices_ = {i, j , k} -> the path would contain
  // three segments [i, j-1], [j, k-1], [k, end_of_path_index]
  std::vector<size_t> segment_indices_;
  size_t current_size_{0}; // Current size of the path
  // Max interpolation distance and total path distance are updated from user
  // config
  float max_path_length_{10.0}, max_interpolation_dist_{0.0};
  float current_total_length_{0.0};
  bool interpolated_ = false;
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
