#pragma once

#include "control.h"
#include "datatypes/path.h"
#include <cmath>
#include <vector>

namespace Kompass {

namespace Control {

typedef Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
    MatrixXfR;

// Data structure to store 2D velocities of a single trajectory
struct TrajectoryVelocities2D {
  Eigen::VectorXf vx;
  Eigen::VectorXf vy;
  Eigen::VectorXf omega;
  size_t _numPointsPerTrajectory;

  // empty initialization
  explicit TrajectoryVelocities2D(size_t numPointsPerTrajectory) {
    vx = Eigen::VectorXf(numPointsPerTrajectory);
    vy = Eigen::VectorXf(numPointsPerTrajectory);
    omega = Eigen::VectorXf(numPointsPerTrajectory);
    _numPointsPerTrajectory = numPointsPerTrajectory;
  };

  // initialize from a vector of velocities
  explicit TrajectoryVelocities2D(const std::vector<Velocity2D> &velocities) {
    vx = Eigen::VectorXf(velocities.size());
    vy = Eigen::VectorXf(velocities.size());
    omega = Eigen::VectorXf(velocities.size());
    _numPointsPerTrajectory = velocities.size();
    for (size_t i = 0; i < velocities.size(); ++i) {
      vx(i) = velocities[i].vx();
      vy(i) = velocities[i].vy();
      omega(i) = velocities[i].omega();
    }
  };

  // add velocity to specified index in TrajectoryVelocities2D
  void add(size_t idx, const Velocity2D &velocity) {

    if (idx >= _numPointsPerTrajectory) {
      throw std::out_of_range("Vector index out of bounds");
    }

    vx(idx) = velocity.vx();
    vy(idx) = velocity.vy();
    omega(idx) = velocity.omega();
  };
};

// Data structure to store velocities per trajectory for a set of trajectories
struct TrajectoryVelocitySamples2D {
  MatrixXfR vx;    // Speed on x-axis (m/s)
  MatrixXfR vy;    // Speed on y-axis (m/s)
  MatrixXfR omega; // Angular velocity (rad/s)
  size_t _numTrajectories, _numPointsPerTrajectory;

  // Constructor that pre-reserves capacity.
  explicit TrajectoryVelocitySamples2D(size_t numTrajectories,
                                       size_t numPointsPerTrajectory) {
    vx = MatrixXfR(numTrajectories, numPointsPerTrajectory);
    vy = MatrixXfR(numTrajectories, numPointsPerTrajectory);
    omega = MatrixXfR(numTrajectories, numPointsPerTrajectory);
    _numTrajectories = numTrajectories;
    _numPointsPerTrajectory = numPointsPerTrajectory;
  }

  // Add a new set of velocity values from a velocity vector.
  void add(size_t row_idx, const std::vector<Velocity2D> &velocities) {
    if (velocities.size() != _numPointsPerTrajectory) {
      throw std::invalid_argument("Velocity vector must have size equivalent "
                                  "to numPointsPerTrajectory");
    }

    if (row_idx >= _numTrajectories) {
      throw std::out_of_range("Row index out of bounds");
    }

    for (size_t i = 0; i < _numPointsPerTrajectory; ++i) {
      vx(row_idx, i) = velocities[i].vx();
      vy(row_idx, i) = velocities[i].vy();
      omega(row_idx, i) = velocities[i].omega();
    }
  }

  // Add a new set of velocity values from a TrajectoryVelocities2D struct
  void add(size_t row_idx, const TrajectoryVelocities2D &velocities) {
    if (velocities._numPointsPerTrajectory != _numPointsPerTrajectory) {
      throw std::invalid_argument(
          "TrajectoryVelocities2D must have numPointsPerTrajectory equivalent to "
          "numPointsPerTrajectory");
    }

    vx.row(row_idx) = velocities.vx;
    vy.row(row_idx) = velocities.vy;
    omega.row(row_idx) = velocities.omega;
  }
  // Iterator class to loop over rows and return Velocity struct.
  class Iterator {
  public:
    Iterator(const TrajectoryVelocitySamples2D &velocities, size_t index)
        : velocities_(velocities), index_(index) {}

    TrajectoryVelocities2D operator*() const {
      TrajectoryVelocities2D v(velocities_._numPointsPerTrajectory);
      v.vx = velocities_.vx.row(index_);
      v.vy = velocities_.vy.row(index_);
      v.omega = velocities_.omega.row(index_);
      return v;
    }
    Iterator &operator++() {
      ++index_;
      return *this;
    }

    bool operator!=(const Iterator &other) const {
      return index_ != other.index_;
    }

  private:
    const TrajectoryVelocitySamples2D &velocities_;
    size_t index_;
  };

  Iterator begin() const { return Iterator(*this, 0); }
  Iterator end() const { return Iterator(*this, _numTrajectories); }
};

// Data structure to store Trajectory Path
struct TrajectoryPath {
  Eigen::VectorXf x;
  Eigen::VectorXf y;
  Eigen::VectorXf z;
  size_t _numPointsPerTrajectory;

  // empty initialization
  explicit TrajectoryPath(size_t numPointsPerTrajectory) {
    x = Eigen::VectorXf(numPointsPerTrajectory);
    y = Eigen::VectorXf(numPointsPerTrajectory);
    z = Eigen::VectorXf(numPointsPerTrajectory);
    _numPointsPerTrajectory = numPointsPerTrajectory;
  };

  // initialize from a  Path
  explicit TrajectoryPath(const Path::Path &path) {
    x = Eigen::VectorXf(path.points.size());
    y = Eigen::VectorXf(path.points.size());
    z = Eigen::VectorXf(path.points.size());
    _numPointsPerTrajectory = path.points.size();
    for (size_t i = 0; i < path.points.size(); ++i) {
      x(i) = path.points[i].x();
      y(i) = path.points[i].y();
      z(i) = path.points[i].z();
    }
  };

  // add point to specified index in path
  void add(size_t idx, const Path::Point &point) {
    if (idx >= _numPointsPerTrajectory) {
      throw std::out_of_range("Vector index out of bounds");
    }
    x(idx) = point.x();
    y(idx) = point.y();
    z(idx) = point.z();
  };
};

// Data structure to store path per trajectory for a set of trajectories
struct TrajectoryPathSamples {
  MatrixXfR x;
  MatrixXfR y;
  MatrixXfR z;
  size_t _numTrajectories, _numPointsPerTrajectory;

  // Constructor that pre-reserves capacity.
  explicit TrajectoryPathSamples(size_t numTrajectories,
                                       size_t numPointsPerTrajectory) {
    x = MatrixXfR(numTrajectories, numPointsPerTrajectory);
    y = MatrixXfR(numTrajectories, numPointsPerTrajectory);
    z = MatrixXfR(numTrajectories, numPointsPerTrajectory);
    _numTrajectories = numTrajectories;
    _numPointsPerTrajectory = numPointsPerTrajectory;
  }

  // Add a new path from a Path struct.
  void add(size_t row_idx, const Path::Path &path) {
    if (path.points.size() != _numPointsPerTrajectory) {
      throw std::invalid_argument("Path points vector must have size equivalent "
                                  "to numPointsPerTrajectory");
    }

    if (row_idx >= _numTrajectories) {
      throw std::out_of_range("Row index out of bounds");
    }

    for (size_t i = 0; i < _numPointsPerTrajectory; ++i) {
      x(row_idx, i) = path.points[i].x();
      y(row_idx, i) = path.points[i].y();
      z(row_idx, i) = path.points[i].z();
    }
  }

  // Add a new path from a TrajectoryPath struct.
  void add(size_t row_idx, const TrajectoryPath &path) {
    if (path._numPointsPerTrajectory != _numPointsPerTrajectory) {
      throw std::invalid_argument(
          "TrajectoryPath must have numPointsPerTrajectory equivalent to "
          "numPointsPerTrajectory");
    }

    x.row(row_idx) = path.x;
    y.row(row_idx) = path.y;
    z.row(row_idx) = path.z;
  }
  // Iterator class to loop over rows and return Velocity struct.
  class Iterator {
  public:
    Iterator(const TrajectoryPathSamples &paths, size_t index)
        : paths_(paths), index_(index) {}

    TrajectoryPath operator*() const {
      TrajectoryPath p(paths_._numPointsPerTrajectory);
      p.x = paths_.x.row(index_);
      p.y = paths_.y.row(index_);
      p.z = paths_.z.row(index_);
      return p;
    }
    Iterator &operator++() {
      ++index_;
      return *this;
    }

    bool operator!=(const Iterator &other) const {
      return index_ != other.index_;
    }

  private:
    const TrajectoryPathSamples &paths_;
    size_t index_;
  };

  Iterator begin() const { return Iterator(*this, 0); }
  Iterator end() const { return Iterator(*this, _numTrajectories); }
};

// Data structure to store multiple trajectory samples
struct TrajectorySamples {
  TrajectoryVelocitySamples2D velocities;
  TrajectoryPathSamples paths;
};

/**
 * @brief Trajectory information: Path + Velocity
 *
 */
struct Trajectory {
  std::vector<Velocity2D> velocity;
  Path::Path path;
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
  Trajectory trajectory;
};

} // namespace Control
} // namespace Kompass
