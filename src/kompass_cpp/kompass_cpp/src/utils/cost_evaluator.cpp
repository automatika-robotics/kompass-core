#ifndef GPU
#include "utils/cost_evaluator.h"
#include "datatypes/path.h"
#include "datatypes/trajectory.h"
#include <Eigen/Dense>
#include <vector>

namespace Kompass {

namespace Control {
CostEvaluator::CostEvaluator(TrajectoryCostsWeights &costsWeights,
                             ControlLimitsParams ctrLimits,
                             size_t maxNumTrajectories,
                             size_t numPointsPerTrajectory,
                             size_t maxRefPathSegmentSize) {

  this->costWeights = std::make_unique<TrajectoryCostsWeights>(costsWeights);
  accLimits_ = {static_cast<float>(ctrLimits.velXParams.maxAcceleration),
                static_cast<float>(ctrLimits.velYParams.maxAcceleration),
                static_cast<float>(ctrLimits.omegaParams.maxAcceleration)};
}

CostEvaluator::CostEvaluator(TrajectoryCostsWeights &costsWeights,
                             const Eigen::Vector3f &sensor_position_body,
                             const Eigen::Quaternionf &sensor_rotation_body,
                             ControlLimitsParams ctrLimits,
                             size_t maxNumTrajectories,
                             size_t numPointsPerTrajectory,
                             size_t maxRefPathSegmentSize) {

  sensor_tf_body_ =
      getTransformation(sensor_rotation_body, sensor_position_body);
  this->costWeights = std::make_unique<TrajectoryCostsWeights>(costsWeights);
  accLimits_ = {static_cast<float>(ctrLimits.velXParams.maxAcceleration),
                static_cast<float>(ctrLimits.velYParams.maxAcceleration),
                static_cast<float>(ctrLimits.omegaParams.maxAcceleration)};
}

void CostEvaluator::updateCostWeights(TrajectoryCostsWeights &costsWeights) {
  this->costWeights = std::make_unique<TrajectoryCostsWeights>(costsWeights);
}

CostEvaluator::~CostEvaluator() {

  // Clear custom cost pointers
  customTrajCostsPtrs_.clear();
};

TrajSearchResult CostEvaluator::getMinTrajectoryCost(
    const std::unique_ptr<TrajectorySamples2D> &trajs,
    const Path::Path *reference_path, const Path::Path &tracked_segment) {
  double weight;
  float total_cost;
  float ref_path_length;
  float minCost = DEFAULT_MIN_DIST;
  Trajectory2D minCostTraj(trajs->numPointsPerTrajectory_);
  bool traj_found = false;

  for (const auto &traj : *trajs) {
    total_cost = 0.0;
    if ((ref_path_length = reference_path->totalPathLength()) > 0.0) {
      if ((weight = costWeights->getParameter<double>("goal_distance_weight")) >
          0.0) {
        float goalCost =
            goalCostFunc(traj, reference_path->getEnd(), ref_path_length);
        total_cost += weight * goalCost;
      }
      if ((weight = costWeights->getParameter<double>(
               "reference_path_distance_weight")) > 0.0) {
        float refPathCost = pathCostFunc(traj, tracked_segment,
                                         tracked_segment.totalPathLength());
        total_cost += weight * refPathCost;
      }
    }

    if (obstaclePointsX.size() > 0 and
        (weight = costWeights->getParameter<double>(
             "obstacles_distance_weight")) > 0.0) {

      float objCost = obstaclesDistCostFunc(traj);
      total_cost += weight * objCost;
    }

    if ((weight = costWeights->getParameter<double>("smoothness_weight")) >
        0.0) {
      float smoothCost = smoothnessCostFunc(traj);
      total_cost += weight * smoothCost;
    }

    if ((weight = costWeights->getParameter<double>("jerk_weight")) > 0.0) {
      float jerCost = jerkCostFunc(traj);
      total_cost += weight * jerCost;
    }

    // Evaluate custom cost functions
    for (const auto &custom_cost : customTrajCostsPtrs_) {
      // custom cost functions takes in the trajectory and the reference path
      total_cost +=
          custom_cost->weight * custom_cost->evaluator_(traj, *reference_path);
    }

    if (total_cost < minCost) {
      minCostTraj = traj;
      traj_found = true;
      minCost = total_cost;
    }
  }
  return {minCostTraj, traj_found, minCost};
}

float CostEvaluator::pathCostFunc(const Trajectory2D &trajectory,
                                  const Path::Path &tracked_segment,
                                  const float tracked_segment_length) {
  float total_cost = 0.0;

  float distError, dist;

  for (Eigen::Index i = 0; i < trajectory.path.x.size(); ++i) {
    // Set min distance between trajectory sample point i and the reference to
    // infinity
    distError = DEFAULT_MIN_DIST;
    // Get minimum distance to the reference
    for (size_t j = 0; j < tracked_segment.getSize(); ++j) {
      dist = Path::Path::distance(tracked_segment.getIndex(j),
                                  trajectory.path.getIndex(i));
      if (dist < distError) {
        distError = dist;
      }
    }
    // Total min distance to each point
    total_cost += distError;
  }

  // end point distance
  float end_dist_error =
      Path::Path::distance(trajectory.path.getEnd(), tracked_segment.getEnd()) /
      tracked_segment_length;

  // Divide by number of points to get average distance
  // and normalize the total cost
  return (total_cost / trajectory.path.x.size() + end_dist_error) / 2;
}

// Compute the cost of a trajectory based on distance to a given reference path
float CostEvaluator::goalCostFunc(const Trajectory2D &trajectory,
                                  const Path::Point &reference_path_end_point,
                                  const float ref_path_length) {
  // end point distance normalized to range [0, 1]
  return Path::Path::distance(trajectory.path.getEnd(),
                              reference_path_end_point) /
         ref_path_length;
}

float CostEvaluator::obstaclesDistCostFunc(const Trajectory2D &trajectory) {
  auto dist = trajectory.path.minDist2D(obstaclePointsX, obstaclePointsY);
  // Normalize the cost to [0, 1] based on the robot max local range for the
  // obstacles Minimum cost is assigned at distance value maxObstaclesDist
  return std::max(maxObstaclesDist - dist, 0.0f) / (maxObstaclesDist);
}

// Compute the cost of trajectory based on smoothness in velocity commands
float CostEvaluator::smoothnessCostFunc(const Trajectory2D &trajectory) {
  float smoothness_cost = 0.0;
  float delta_vx, delta_vy, delta_omega;
  for (Eigen::Index i = 1; i < trajectory.velocities.vx.size(); ++i) {
    if (accLimits_[0] > 0) {
      delta_vx = trajectory.velocities.vx[i] - trajectory.velocities.vx[i - 1];
      smoothness_cost += std::pow(delta_vx, 2) / accLimits_[0];
    }
    if (accLimits_[1] > 0) {
      delta_vy = trajectory.velocities.vy[i] - trajectory.velocities.vy[i - 1];
      smoothness_cost += std::pow(delta_vy, 2) / accLimits_[1];
    }
    if (accLimits_[2] > 0) {
      delta_omega =
          trajectory.velocities.omega[i] - trajectory.velocities.omega[i - 1];
      smoothness_cost += std::pow(delta_omega, 2) / accLimits_[2];
    }
  }
  return smoothness_cost / (3 * trajectory.velocities.vx.size());
}

// Compute the cost of trajectory based on jerk in velocity commands
float CostEvaluator::jerkCostFunc(const Trajectory2D &trajectory) {
  float jerk_cost = 0.0;
  float jerk_vx, jerk_vy, jerk_omega;
  for (Eigen::Index i = 2; i < trajectory.velocities.vx.size(); ++i) {
    if (accLimits_[0] > 0) {
      jerk_vx = trajectory.velocities.vx[i] -
                2 * trajectory.velocities.vx[i - 1] +
                trajectory.velocities.vx[i - 2];
      jerk_cost += std::pow(jerk_vx, 2) / accLimits_[0];
    }
    if (accLimits_[1] > 0) {
      jerk_vy = trajectory.velocities.vy[i] -
                2 * trajectory.velocities.vy[i - 1] +
                trajectory.velocities.vy[i - 2];
      jerk_cost += std::pow(jerk_vy, 2) / accLimits_[1];
    }
    if (accLimits_[2] > 0) {
      jerk_omega = trajectory.velocities.omega[i] -
                   2 * trajectory.velocities.omega[i - 1] +
                   trajectory.velocities.omega[i - 2];
      jerk_cost += std::pow(jerk_omega, 2) / accLimits_[2];
    }
  }
  return jerk_cost / (3 * trajectory.velocities.vx.size());
}

}; // namespace Control
} // namespace Kompass
#endif // GPU
