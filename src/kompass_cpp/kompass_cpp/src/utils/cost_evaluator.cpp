#include "utils/cost_evaluator.h"
#include "datatypes/path.h"
#include "datatypes/trajectory.h"
#include "utils/transformation.h"
#include <Eigen/src/Core/Matrix.h>
#include <Eigen/src/Geometry/Transform.h>
#include <cstddef>
#include <cstdlib>
#include <vector>

namespace Kompass {

namespace Control {
CostEvaluator::CostEvaluator(TrajectoryCostsWeights costsWeights) {

  this->costWeights = costsWeights;
}

CostEvaluator::CostEvaluator(TrajectoryCostsWeights costsWeights,
                             const std::array<float, 3> &sensor_position_body,
                             const std::array<float, 4> &sensor_rotation_body) {
  sensor_tf_body_ =
      getTransformation(Eigen::Quaternionf(sensor_rotation_body.data()),
                        Eigen::Vector3f(sensor_position_body.data()));
  this->costWeights = costsWeights;
}

void CostEvaluator::updateDefaultCostWeights(
    TrajectoryCostsWeights costsWeights) {
  this->costWeights = costsWeights;
}

CostEvaluator::~CostEvaluator() {

  // delete and clear custom cost pointers
  for (auto ptr : customTrajCostsPtrs_) {
    delete ptr;
  }
  customTrajCostsPtrs_.clear();
}

void CostEvaluator::setPointScan(const LaserScan &scan,
                                 const Path::State &current_state) {

  obstaclePoints.clear();
  Eigen::Isometry3f body_tf_world_ = getTransformation(current_state);

  for (size_t i = 0; i < scan.ranges.size(); i++) {
    // Convert polar to Cartesian (assuming scan data in the XY plane)
    double point_x = scan.ranges[i] * std::cos(scan.angles[i]);
    double point_y = scan.ranges[i] * std::sin(scan.angles[i]);

    Eigen::Vector3f pose_trans =
        transformPosition(Eigen::Vector3f(point_x, point_y, 0.0),
                          sensor_tf_body_ * body_tf_world_);
    obstaclePoints.push_back({pose_trans[0], pose_trans[1]});
  }
}

void CostEvaluator::setPointScan(const std::vector<Path::Point> &cloud,
                                 const Path::State &current_state) {

  obstaclePoints.clear();
  Eigen::Isometry3f body_tf_world_ = getTransformation(current_state);

  for (auto &point : cloud) {
    // TODO: fix
    Eigen::Vector3f pose_trans =
        transformPosition(Eigen::Vector3f(point.x(), point.y(), point.z()),
                          sensor_tf_body_ * body_tf_world_);
    obstaclePoints.push_back({pose_trans[0], pose_trans[1]});
  }
}

TrajSearchResult CostEvaluator::getMinTrajectoryCost(
    const TrajectorySamples2D &trajs, const Path::Path &reference_path,
    const Path::Path &tracked_segment, const size_t closest_segment_index) {
  double weight;
  double total_cost;
  double minCost = std::numeric_limits<double>::max();
  Trajectory2D minCostTraj(trajs.numPointsPerTrajectory_);
  bool traj_found = false;

  for (const auto &traj : trajs) {
    total_cost = 0.0;
    if (reference_path.totalPathLength() > 0.0) {
      if ((weight = costWeights.getParameter<double>("goal_distance_weight") >
                    0.0)) {
        double goalCost = goalCostFunc(traj, reference_path);
        total_cost += weight * goalCost;
      }
      if ((weight = costWeights.getParameter<double>(
                        "reference_path_distance_weight") > 0.0)) {
        double refPathCost = pathCostFunc(traj, reference_path);
        total_cost += weight * refPathCost;
      }
    }

    if (obstaclePoints.size() > 0 and
        (weight = costWeights.getParameter<double>(
             "obstacles_distance_weight")) > 0.0) {

      double objCost = obstaclesDistCostFunc(traj, obstaclePoints);
      total_cost += weight * objCost;
    }

    std::array<double, 3> accLimits{ctrlimits.velXParams.maxAcceleration,
                                    ctrlimits.velYParams.maxAcceleration,
                                    ctrlimits.omegaParams.maxAcceleration};

    if ((weight =
             costWeights.getParameter<double>("smoothness_weight") > 0.0)) {
      double smoothCost = smoothnessCostFunc(traj, accLimits);
      total_cost += weight * smoothCost;
    }

    if ((weight = costWeights.getParameter<double>("jerk_weight") > 0.0)) {
      double jerCost = jerkCostFunc(traj, accLimits);
      total_cost += weight * jerCost;
    }

    // Evaluate custom cost functions
    for (const auto &custom_cost : customTrajCostsPtrs_) {
      // custom cost functions takes in the trajectory and the reference path
      total_cost +=
          custom_cost->weight * custom_cost->evaluator_(traj, reference_path);
    }

    if (total_cost < minCost) {
      minCost = total_cost;
      minCostTraj = traj;
      traj_found = true;
    }
  }
  return {traj_found, minCost, minCostTraj};
}

void CostEvaluator::addCustomCost(double weight,
                                  CustomCostFunction custom_cost_function) {
  CustomTrajectoryCost *newCost =
      new CustomTrajectoryCost(weight, custom_cost_function);
  customTrajCostsPtrs_.push_back(newCost);
}

double CostEvaluator::pathCostFunc(const Trajectory2D &trajectory,
                                   const Path::Path &reference_path) {
  double total_cost = 0.0;

  double distError, dist;

  for (size_t i = 0; i < trajectory.path.x.size(); ++i) {
    // Set min distance between trajectory sample point i and the reference to
    // infinity
    distError = std::numeric_limits<double>::max();
    // Get minimum distance to the reference
    for (size_t j = 0; j < reference_path.points.size(); ++j) {
      dist = Path::Path::distance(reference_path.points[j],
                                  trajectory.path.getIndex(i));
      if (dist < distError) {
        distError = dist;
      }
    }
    // Total min distance to each point
    total_cost += distError;
  }

  // end point distance
  double end_dist_error =
      Path::Path::distance(trajectory.path.getEnd(), reference_path.getEnd());

  // Divide by number of points to get average distance

  return total_cost / trajectory.path.x.size() + end_dist_error;
}

// Compute the cost of a trajectory based on distance to a given reference path
double CostEvaluator::goalCostFunc(const Trajectory2D &trajectory,
                                   const Path::Path &reference_path) {
  // end point distance
  return Path::Path::distance(trajectory.path.getEnd(),
                              reference_path.getEnd()) /
         reference_path.totalPathLength();
}

double CostEvaluator::obstaclesDistCostFunc(
    const Trajectory2D &trajectory,
    const std::vector<Path::Point> &obstaclePoints) {
  return trajectory.path.minDist2D(obstaclePoints);
}

// Compute the cost of trajectory based on smoothness in velocity commands
double
CostEvaluator::smoothnessCostFunc(const Trajectory2D &trajectory,
                                  const std::array<double, 3> accLimits) {
  double smoothness_cost = 0.0;
  double delta_vx, delta_vy, delta_omega;
  for (size_t i = 1; i < trajectory.velocities.vx.size(); ++i) {
    if (accLimits[0] > 0) {
      delta_vx = trajectory.velocities.vx[i] - trajectory.velocities.vx[i - 1];
      smoothness_cost += std::pow(delta_vx, 2) / accLimits[0];
    }
    if (accLimits[1] > 0) {
      delta_vy = trajectory.velocities.vy[i] - trajectory.velocities.vy[i - 1];
      smoothness_cost += std::pow(delta_vy, 2) / accLimits[1];
    }
    if (accLimits[2] > 0) {
      delta_omega =
          trajectory.velocities.omega[i] - trajectory.velocities.omega[i - 1];
      smoothness_cost += std::pow(delta_omega, 2) / accLimits[2];
    }
  }
  return smoothness_cost / (3 * trajectory.velocities.vx.size());
}

// Compute the cost of trajectory based on jerk in velocity commands
double CostEvaluator::jerkCostFunc(const Trajectory2D &trajectory,
                                   const std::array<double, 3> accLimits) {
  double jerk_cost = 0.0;
  double jerk_vx, jerk_vy, jerk_omega;
  for (size_t i = 2; i < trajectory.velocities.vx.size(); ++i) {
    if (accLimits[0] > 0) {
      jerk_vx = trajectory.velocities.vx[i] -
                2 * trajectory.velocities.vx[i - 1] +
                trajectory.velocities.vx[i - 2];
      jerk_cost += std::pow(jerk_vx, 2) / accLimits[0];
    }
    if (accLimits[1] > 0) {
      jerk_vy = trajectory.velocities.vy[i] -
                2 * trajectory.velocities.vy[i - 1] +
                trajectory.velocities.vy[i - 2];
      jerk_cost += std::pow(jerk_vy, 2) / accLimits[1];
    }
    if (accLimits[2] > 0) {
      jerk_omega = trajectory.velocities.omega[i] -
                   2 * trajectory.velocities.omega[i - 1] +
                   trajectory.velocities.omega[i - 2];
      jerk_cost += std::pow(jerk_omega, 2) / accLimits[2];
    }
  }
  return jerk_cost / (3 * trajectory.velocities.vx.size());
}

}; // namespace Control
} // namespace Kompass
