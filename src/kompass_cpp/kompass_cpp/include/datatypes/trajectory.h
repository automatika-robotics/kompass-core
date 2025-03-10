#pragma once

#include "control.h"
#include "datatypes/path.h"
#include <cmath>
#include <vector>

namespace Kompass {

namespace Control {

typedef Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
    MatrixXdR;

// Data structure to store 2D velocities of a single trajectory
struct TrajectoryVelocities2D {
  Eigen::VectorXd vx;
  Eigen::VectorXd vy;
  Eigen::VectorXd omega;
  size_t numPointsPerTrajectory_;

  // default constructor
  TrajectoryVelocities2D() = default;

  // empty initialization
  explicit TrajectoryVelocities2D(size_t numPointsPerTrajectory)
      : numPointsPerTrajectory_(numPointsPerTrajectory) {
    // velocities start from one points after the starting point on the
    // trajectory
    vx = Eigen::VectorXd(numPointsPerTrajectory - 1);
    vy = Eigen::VectorXd(numPointsPerTrajectory - 1);
    omega = Eigen::VectorXd(numPointsPerTrajectory - 1);
  };

  // initialize from a vector of velocities
  explicit TrajectoryVelocities2D(const std::vector<Velocity2D> &velocities) {
    vx = Eigen::VectorXd(velocities.size());
    vy = Eigen::VectorXd(velocities.size());
    omega = Eigen::VectorXd(velocities.size());
    numPointsPerTrajectory_ = velocities.size() + 1;
    for (size_t i = 0; i < velocities.size(); ++i) {
      vx(i) = velocities[i].vx();
      vy(i) = velocities[i].vy();
      omega(i) = velocities[i].omega();
    }
  };

  // initialize from eigen vectors
  TrajectoryVelocities2D(const Eigen::VectorXd &vx_, const Eigen::VectorXd &vy_,
                         const Eigen::VectorXd &omega_)
      : vx(vx_), vy(vy_), omega(omega_),
        numPointsPerTrajectory_(vx_.size() + 1) {};

  // add velocity to specified index in TrajectoryVelocities2D
  void add(size_t idx, const Velocity2D &velocity) {

    if (idx >= numPointsPerTrajectory_ - 1) {
      throw std::out_of_range("Vector index out of bounds");
    }

    vx(idx) = velocity.vx();
    vy(idx) = velocity.vy();
    omega(idx) = velocity.omega();
  };

  // get point on index
  Velocity2D getIndex(size_t idx) const {
    if (idx >= numPointsPerTrajectory_ - 1) {
      throw std::out_of_range("Vector index out of bounds");
    }
    return Velocity2D(vx(idx), vy(idx), omega(idx));
  };
  // get the first element
  Velocity2D getFront() const { return Velocity2D(vx(0), vy(0), omega(0)); };
  // get the last element
  Velocity2D getEnd() const {
    return Velocity2D(vx(numPointsPerTrajectory_ - 2),
                      vy(numPointsPerTrajectory_ - 2),
                      omega(numPointsPerTrajectory_ - 2));
  };
  struct Iterator {
    using iterator_category = std::forward_iterator_tag;
    using value_type = Velocity2D;
    using difference_type = std::ptrdiff_t;

    Iterator(const TrajectoryVelocities2D &v, size_t idx)
        : velocities(v), index(idx) {}

    Velocity2D operator*() const {
      return Velocity2D(velocities.vx(index), velocities.vy(index),
                        velocities.omega(index));
    }

    Iterator &operator++() {
      ++index;
      return *this;
    }

    bool operator!=(const Iterator &other) const {
      return index != other.index;
    }

  private:
    const TrajectoryVelocities2D &velocities;
    size_t index;
  };
  Iterator begin() const { return Iterator(*this, 0); }
  Iterator end() const { return Iterator(*this, numPointsPerTrajectory_ - 1); }
};

// Data structure to store Trajectory Path
struct TrajectoryPath {
  Eigen::VectorXd x;
  Eigen::VectorXd y;
  Eigen::VectorXd z;
  size_t numPointsPerTrajectory_;

  // default constructor
  TrajectoryPath() = default;

  // empty initialization
  explicit TrajectoryPath(size_t numPointsPerTrajectory)
      : numPointsPerTrajectory_(numPointsPerTrajectory) {
    x = Eigen::VectorXd(numPointsPerTrajectory);
    y = Eigen::VectorXd(numPointsPerTrajectory);
    z = Eigen::VectorXd(numPointsPerTrajectory);
  };

  // initialize from a  Path
  explicit TrajectoryPath(const Path::Path &path) {
    x = Eigen::VectorXd(path.points.size());
    y = Eigen::VectorXd(path.points.size());
    z = Eigen::VectorXd(path.points.size());
    numPointsPerTrajectory_ = path.points.size();
    for (size_t i = 0; i < path.points.size(); ++i) {
      x(i) = path.points[i].x();
      y(i) = path.points[i].y();
      z(i) = path.points[i].z();
    }
  };

  // initialize from eigen vectors
  TrajectoryPath(const Eigen::VectorXd &x_, const Eigen::VectorXd &y_,
                 const Eigen::VectorXd &z_)
      : x(x_), y(y_), z(z_), numPointsPerTrajectory_(x.size()) {};

  // add point to specified index in path
  void add(size_t idx, const Path::Point &point) {
    if (idx >= numPointsPerTrajectory_) {
      throw std::out_of_range("Vector index out of bounds");
    }
    x(idx) = point.x();
    y(idx) = point.y();
    z(idx) = point.z();
  };

  // add point using values
  void add(size_t idx, const double x, const double y, const double z = 0) {
    if (idx >= numPointsPerTrajectory_) {
      throw std::out_of_range("Vector index out of bounds");
    }
    this->x(idx) = x;
    this->y(idx) = y;
    this->z(idx) = z;
  };

  // calculate minimum distance with a given vector of points
  float minDist2D(const std::vector<Path::Point> &others) const {
    size_t s = others.size();
    if (s <= 0) {
      return 0.0f;
    }
    float minDist = std::numeric_limits<double>::max();
    float dist;
    for (size_t i = 0; i < s; ++i) {
      for (size_t j = 0; j < numPointsPerTrajectory_; ++j) {
        dist = pow((others[i].x() - x(j)), 2) + pow((others[i].y() - y(j)), 2);
        if (dist < minDist) {
          minDist = dist;
        }
      }
    }
    return sqrt(minDist);
  }

  // get point on index
  Path::Point getIndex(size_t idx) const {
    if (idx >= numPointsPerTrajectory_) {
      throw std::out_of_range("Vector index out of bounds");
    }
    return Path::Point(x(idx), y(idx), z(idx));
  };
  // get the first element
  Path::Point getFront() const { return Path::Point(x(0), y(0), z(0)); };
  // get the last element
  Path::Point getEnd() const {
    return Path::Point(x(numPointsPerTrajectory_ - 1),
                       y(numPointsPerTrajectory_ - 1),
                       z(numPointsPerTrajectory_ - 1));
  };
  struct Iterator {
    using iterator_category = std::forward_iterator_tag;
    using value_type = Path::Point;
    using difference_type = std::ptrdiff_t;

    Iterator(const TrajectoryPath &p, size_t idx) : path(p), index(idx) {}

    Path::Point operator*() const {
      return Path::Point(path.x(index), path.y(index), path.z(index));
    }

    Iterator &operator++() {
      ++index;
      return *this;
    }

    bool operator!=(const Iterator &other) const {
      return index != other.index;
    }

  private:
    const TrajectoryPath &path;
    size_t index;
  };

  Iterator begin() const { return Iterator(*this, 0); }
  Iterator end() const { return Iterator(*this, numPointsPerTrajectory_); }
};

// A data structure to hold a single 2D trajectory (path and corresponding
// velocities)
struct Trajectory2D {
  TrajectoryVelocities2D velocities;
  TrajectoryPath path;
  size_t numPointsPerTrajectory_;

  // default constructor
  Trajectory2D() = default;

  // empty initialization
  explicit Trajectory2D(size_t numPointsPerTrajectory)
      : numPointsPerTrajectory_(numPointsPerTrajectory) {
    velocities = TrajectoryVelocities2D(numPointsPerTrajectory);
    path = TrajectoryPath(numPointsPerTrajectory);
  };

  // initialize with paths and velocities
  explicit Trajectory2D(const TrajectoryVelocities2D &velocities_,
                        const TrajectoryPath &path_) {
    if (velocities_.numPointsPerTrajectory_ != path_.numPointsPerTrajectory_) {
      throw std::invalid_argument("TrajectoryVelocities2D and TrajectoryPath "
                                  "must have the same numPointsPerTrajectory");
    }
    this->velocities = velocities_;
    this->path = path_;
    numPointsPerTrajectory_ = velocities_.numPointsPerTrajectory_;
  };

  // initialize with path object and velocity vector
  explicit Trajectory2D(std::vector<Velocity2D> &velocities, Path::Path &path) {
    if (velocities.size() != path.points.size()) {
      throw std::invalid_argument(
          "Velocity2D vector and path points vector should have the same size "
          "must have the same numPointsPerTrajectory");
    }
    this->velocities = TrajectoryVelocities2D(velocities);
    this->path = TrajectoryPath(path);
    numPointsPerTrajectory_ = velocities.size();
  };
};

// Data structure to store velocities per trajectory for a set of trajectories
struct TrajectoryVelocitySamples2D {
  MatrixXdR vx;    // Speed on x-axis (m/s)
  MatrixXdR vy;    // Speed on y-axis (m/s)
  MatrixXdR omega; // Angular velocity (rad/s)
  size_t maxNumTrajectories_, numPointsPerTrajectory_;
  size_t velocitiesIndex_; // keep track of actual trajectories added

  // default constructor
  TrajectoryVelocitySamples2D() = default;

  // Constructor that pre-reserves capacity.
  explicit TrajectoryVelocitySamples2D(size_t maxNumTrajectories,
                                       size_t numPointsPerTrajectory)
      : maxNumTrajectories_(maxNumTrajectories),
        numPointsPerTrajectory_(numPointsPerTrajectory), velocitiesIndex_(-1),
        vx(MatrixXdR(maxNumTrajectories, numPointsPerTrajectory - 1)),
        vy(MatrixXdR(maxNumTrajectories, numPointsPerTrajectory - 1)),
        omega(MatrixXdR(maxNumTrajectories, numPointsPerTrajectory - 1)) {}

  // Add a new set of velocity values from a velocity vector.
  void push_back(const std::vector<Velocity2D> &velocities) {
    if (velocities.size() != numPointsPerTrajectory_ - 1) {
      throw std::invalid_argument("Velocity vector must have size equivalent "
                                  "to numPointsPerTrajectory");
    }

    velocitiesIndex_++;
    for (size_t i = 0; i < numPointsPerTrajectory_ - 1; ++i) {
      vx(velocitiesIndex_, i) = velocities[i].vx();
      vy(velocitiesIndex_, i) = velocities[i].vy();
      omega(velocitiesIndex_, i) = velocities[i].omega();
    }
  }

  // Add a new set of velocity values from a TrajectoryVelocities2D struct
  void push_back(const TrajectoryVelocities2D &velocities) {
    if (velocities.numPointsPerTrajectory_ != numPointsPerTrajectory_) {
      throw std::invalid_argument("TrajectoryVelocities2D must have "
                                  "numPointsPerTrajectory equivalent to "
                                  "numPointsPerTrajectory");
    }

    velocitiesIndex_++;
    vx.row(velocitiesIndex_) = velocities.vx;
    vy.row(velocitiesIndex_) = velocities.vy;
    omega.row(velocitiesIndex_) = velocities.omega;
  }

  // Iterator class to loop over rows and return Velocity struct.
  class Iterator {
  public:
    Iterator(const TrajectoryVelocitySamples2D &velocities, size_t index)
        : velocities_(velocities), index_(index) {}

    TrajectoryVelocities2D operator*() const {
      return TrajectoryVelocities2D(velocities_.vx.row(index_),
                                    velocities_.vy.row(index_),
                                    velocities_.omega.row(index_));
    }

    // Prefix increment
    Iterator &operator++() {
      ++index_;
      return *this;
    }

    // Postfix increment
    Iterator operator++(int) {
      Iterator tmp(*this);
      ++(*this);
      return tmp;
    }

    // Comparison operators
    bool operator!=(const Iterator &other) const {
      return index_ != other.index_;
    }

    bool operator==(const Iterator &other) const { return !(*this != other); }

  private:
    const TrajectoryVelocitySamples2D &velocities_;
    size_t index_;
  };

  Iterator begin() const { return Iterator(*this, 0); }
  Iterator end() const {
    return Iterator(*this, velocitiesIndex_ + 1);
  } // to ensure last value is captured
};

// Data structure to store path per trajectory for a set of trajectories
struct TrajectoryPathSamples {
  MatrixXdR x;
  MatrixXdR y;
  MatrixXdR z;
  size_t maxNumTrajectories_, numPointsPerTrajectory_;
  size_t pathIndex_;

  // default constructor
  TrajectoryPathSamples() = default;

  // Constructor that pre-reserves capacity.
  explicit TrajectoryPathSamples(size_t maxNumTrajectories,
                                 size_t numPointsPerTrajectory)
      : maxNumTrajectories_(maxNumTrajectories),
        numPointsPerTrajectory_(numPointsPerTrajectory), pathIndex_(-1),
        x(MatrixXdR(maxNumTrajectories, numPointsPerTrajectory)),
        y(MatrixXdR(maxNumTrajectories, numPointsPerTrajectory)),
        z(MatrixXdR(maxNumTrajectories, numPointsPerTrajectory)) {}

  // Add a new path from a Path struct.
  void push_back(const Path::Path &path) {
    if (path.points.size() != numPointsPerTrajectory_) {
      throw std::invalid_argument(
          "Path points vector must have size equivalent "
          "to numPointsPerTrajectory");
    }

    pathIndex_++;
    for (size_t i = 0; i < numPointsPerTrajectory_; ++i) {
      x(pathIndex_, i) = path.points[i].x();
      y(pathIndex_, i) = path.points[i].y();
      z(pathIndex_, i) = path.points[i].z();
    }
  }

  // Add a new path from a TrajectoryPath struct.
  void push_back(const TrajectoryPath &path) {
    if (path.numPointsPerTrajectory_ != numPointsPerTrajectory_) {
      throw std::invalid_argument(
          "TrajectoryPath must have numPointsPerTrajectory equivalent to "
          "numPointsPerTrajectory");
    }

    pathIndex_++;
    x.row(pathIndex_) = path.x;
    y.row(pathIndex_) = path.y;
    z.row(pathIndex_) = path.z;
  }

  class Iterator {
    using iterator_category = std::forward_iterator_tag;
    using value_type = TrajectoryPath;
    using difference_type = std::ptrdiff_t;

  public:
    Iterator(const TrajectoryPathSamples &paths, size_t index)
        : paths_(paths), index_(index) {}

    TrajectoryPath operator*() const {
      return TrajectoryPath(paths_.x.row(index_), paths_.y.row(index_),
                            paths_.z.row(index_));
    }

    // Prefix increment
    Iterator &operator++() {
      ++index_;
      return *this;
    }

    // Postfix increment
    Iterator operator++(int) {
      Iterator tmp(*this);
      ++(*this);
      return tmp;
    }

    // Comparison operators
    bool operator!=(const Iterator &other) const {
      return index_ != other.index_;
    }

    bool operator==(const Iterator &other) const { return !(*this != other); }

  private:
    const TrajectoryPathSamples &paths_;
    size_t index_;
  };

  Iterator begin() const { return Iterator(*this, 0); }
  Iterator end() const {
    return Iterator(*this, pathIndex_ + 1);
  } // to ensure last value is captured
};

// Data structure to store multiple 2D trajectory samples
struct TrajectorySamples2D {
  TrajectoryVelocitySamples2D velocities;
  TrajectoryPathSamples paths;
  size_t maxNumTrajectories_, numPointsPerTrajectory_;

  // default constructor
  TrajectorySamples2D() = default;

  // empty initialization
  explicit TrajectorySamples2D(size_t maxNumTrajectories,
                               size_t numPointsPerTrajectory)
      : maxNumTrajectories_(maxNumTrajectories),
        numPointsPerTrajectory_(numPointsPerTrajectory) {
    velocities =
        TrajectoryVelocitySamples2D(maxNumTrajectories, numPointsPerTrajectory);
    paths = TrajectoryPathSamples(maxNumTrajectories, numPointsPerTrajectory);
  };

  // initialize with paths and velocities
  explicit TrajectorySamples2D(TrajectoryVelocitySamples2D &velocities_,
                               TrajectoryPathSamples &paths_) {
    if (velocities_.maxNumTrajectories_ != paths_.maxNumTrajectories_) {
      throw std::invalid_argument(
          "TrajectoryVelocitySamples2D and TrajectoryPathSamples "
          "must have the same numTrajectories");
    }
    if (velocities_.numPointsPerTrajectory_ != paths_.numPointsPerTrajectory_) {
      throw std::invalid_argument(
          "TrajectoryVelocitySamples2D and TrajectoryPathSamples "
          "must have the same numPointsPerTrajectory");
    }
    this->velocities = velocities_;
    this->paths = paths_;
    maxNumTrajectories_ = velocities_.maxNumTrajectories_;
    numPointsPerTrajectory_ = velocities_.numPointsPerTrajectory_;
  };

  // Copy constructor
  TrajectorySamples2D(const TrajectorySamples2D &other)
      : velocities(other.velocities), paths(other.paths),
        maxNumTrajectories_(other.maxNumTrajectories_),
        numPointsPerTrajectory_(other.numPointsPerTrajectory_) {}

  // Add a new path and velocity (can be structs or vectors).
  template <typename T1, typename T2> void push_back(T1 &velocities, T2 &path) {
    this->velocities.push_back(velocities);
    this->paths.push_back(path);
  }

  // return  a Trajectory2D struct based on given index of sample
  Trajectory2D getIndex(size_t idx) const {
    if (idx > velocities.vx.size()) {
      throw std::out_of_range("Vector index out of bounds");
    }
    return Trajectory2D(
        TrajectoryVelocities2D(velocities.vx.row(idx), velocities.vy.row(idx),
                               velocities.omega.row(idx)),
        TrajectoryPath(paths.x.row(idx), paths.y.row(idx), paths.z.row(idx)));
  }

  struct Iterator {
    using iterator_category = std::forward_iterator_tag;
    using value_type = Trajectory2D;
    using difference_type = std::ptrdiff_t;

    Iterator(TrajectoryVelocitySamples2D::Iterator velIt,
             TrajectoryPathSamples::Iterator pathIt)
        : velIt(velIt), pathIt(pathIt) {}

    value_type operator*() const { return Trajectory2D(*velIt, *pathIt); }

    Iterator &operator++() {
      ++velIt;
      ++pathIt;
      return *this;
    }

    Iterator operator++(int) {
      Iterator tmp = *this;
      ++(*this);
      return tmp;
    }

    bool operator==(const Iterator &other) const {
      return velIt == other.velIt && pathIt == other.pathIt;
    }

    bool operator!=(const Iterator &other) const { return !(*this == other); }

  private:
    TrajectoryVelocitySamples2D::Iterator velIt;
    TrajectoryPathSamples::Iterator pathIt;
  };

  Iterator begin() const { return Iterator(velocities.begin(), paths.begin()); }

  Iterator end() const { return Iterator(velocities.end(), paths.end()); }
};

/**
 * @brief Trajectory control result (Local planners result), contains a
 * boolean indicating if the trajectory is found, the resulting trajectory and
 * its const
 *
 */
struct TrajSearchResult {
  bool isTrajFound;
  double trajCost;
  Trajectory2D trajectory;
};

} // namespace Control
} // namespace Kompass
