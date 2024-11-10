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

  referencePathDistCost->weight =
      costsWeights.getParameter<double>("reference_path_distance_weight");
  goalPointDistCost->weight =
      costsWeights.getParameter<double>("goal_distance_weight");
  smoothnessCost->weight =
      costsWeights.getParameter<double>("smoothness_weight");
  jerkCost->weight = costsWeights.getParameter<double>("jerk_weight");
  obstaclesDistCost->weight =
      costsWeights.getParameter<double>("obstacles_distance_weight");
}

CostEvaluator::CostEvaluator(TrajectoryCostsWeights costsWeights,
                             const std::array<float, 3> &sensor_position_body,
                             const std::array<float, 4> &sensor_rotation_body) {
  sensor_tf_body_ =
      getTransformation(Eigen::Quaternionf(sensor_rotation_body.data()),
                        Eigen::Vector3f(sensor_position_body.data()));
  referencePathDistCost->weight =
      costsWeights.getParameter<double>("reference_path_distance_weight");
  goalPointDistCost->weight =
      costsWeights.getParameter<double>("goal_distance_weight");
  smoothnessCost->weight =
      costsWeights.getParameter<double>("smoothness_weight");
  jerkCost->weight = costsWeights.getParameter<double>("jerk_weight");
  obstaclesDistCost->weight =
      costsWeights.getParameter<double>("obstacles_distance_weight");
}

CostEvaluator::~CostEvaluator() {
  delete referencePathDistCost;
  delete goalPointDistCost;
  delete smoothnessCost;
  delete jerkCost;

  // delete and clear custom cost pointers
  for (auto ptr : customTrajCostsPtrs_) {
    delete ptr;
  }
  customTrajCostsPtrs_.clear();
}

void CostEvaluator::setPointScan(const LaserScan &scan,
                                 const Path::State &current_state) {

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

void CostEvaluator::setPointScan(const std::vector<Point3D> &cloud,
                                  const Path::State &current_state) {

  Eigen::Isometry3f body_tf_world_ = getTransformation(current_state);

  for (auto &point : cloud) {

    Eigen::Vector3f pose_trans =
        transformPosition(Eigen::Vector3f(point.x, point.y, point.z),
                          sensor_tf_body_ * body_tf_world_);
    obstaclePoints.push_back({pose_trans[0], pose_trans[1]});
  }
}

double CostEvaluator::getTrajectoryCost(const Trajectory &traj,
                                        const Path::Path &reference_path,
                                        const Path::Path &tracked_segment,
                                        const size_t closest_segment_index) {
  double total_cost = 0.0;

  if (reference_path.totalPathLength() > 0.0) {

    if (goalPointDistCost->weight > 0.0) {
      CostFunctionArguments args = std::make_pair(traj, reference_path);
      double goalCost = goalPointDistCost->evaluator_(args);
      total_cost += goalPointDistCost->weight * goalCost;
    }
    if (referencePathDistCost->weight > 0.0) {
      CostFunctionArguments args =
          std::make_tuple(traj, tracked_segment, ctrType);
      double refPathCost = referencePathDistCost->evaluator_(args);
      total_cost += referencePathDistCost->weight * refPathCost;
    }
  }

  if (obstaclePoints.size() > 0 and obstaclesDistCost->weight > 0) {
    CostFunctionArguments args =
        std::make_pair(traj, Path::Path(obstaclePoints));
    total_cost +=
        obstaclesDistCost->weight * obstaclesDistCost->evaluator_(args);
  }

  std::array<double, 3> accLimits{ctrlimits.velXParams.maxAcceleration,
                                  ctrlimits.velYParams.maxAcceleration,
                                  ctrlimits.omegaParams.maxAcceleration};

  if (smoothnessCost->weight > 0.0) {
    CostFunctionArguments args = std::make_pair(traj, accLimits);
    double smoothCost = smoothnessCost->evaluator_(args);
    total_cost += smoothnessCost->weight * smoothCost;
  }

  if (jerkCost->weight > 0.0) {
    CostFunctionArguments args = std::make_pair(traj, accLimits);
    double jCost = jerkCost->evaluator_(args);
    total_cost += jerkCost->weight * jCost;
  }

  // Evaluate custom cost functions
  for (const auto &custom_cost : customTrajCostsPtrs_) {
    // custom cost functions takes in the trajectory and the reference path
    CostFunctionArguments args = std::make_pair(traj, reference_path);
    total_cost += custom_cost->weight * custom_cost->evaluator_(args);
  }

  return total_cost;
}

void CostEvaluator::addCustomCost(double weight,
                                  CustomCostFunction custom_cost_function) {
  TrajectoryCost *newCost = new TrajectoryCost(
      weight,
      costFunctionWrapper<decltype(custom_cost_function),
                          std::pair<const Trajectory, const Path::Path>>(
          custom_cost_function));
  customTrajCostsPtrs_.push_back(newCost);
}

// TODO: Remove controlTYpe from this function
double CostEvaluator::pathCostFunc(const Trajectory &trajectory,
                                   const Path::Path &reference_path,
                                   const ControlType controlType) {
  double total_cost = 0.0;

  double distError;

  for (size_t i = 0; i < trajectory.path.points.size(); ++i) {
    // Set min distance between trajectory sample point i and the reference to
    // infinity
    distError = std::numeric_limits<double>::max();
    // Get minimum distance to the reference
    for (size_t j = 0; j < reference_path.points.size(); ++j) {
      double dist = std::sqrt(
          std::pow(reference_path.points[j].x - trajectory.path.points[i].x,
                   2) +
          std::pow(reference_path.points[j].y - trajectory.path.points[i].y,
                   2));
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

  return total_cost / trajectory.path.points.size() + end_dist_error;
}

// Compute the cost of a trajectory based on distance to a given reference path
double CostEvaluator::goalCostFunc(const Trajectory &trajectory,
                                   const Path::Path &reference_path) {
  Path::Point ref_end = reference_path.getEnd();
  Path::Point traj_end = trajectory.path.getEnd();

  double dist = std::sqrt(std::pow(traj_end.x - ref_end.x, 2) +
                          std::pow(traj_end.y - ref_end.y, 2)) /
                reference_path.totalPathLength();
  return dist;
}

double CostEvaluator::obstaclesDistCostFunc(const Trajectory &trajectory,
                                            const Path::Path &obstaclePoints) {
  return trajectory.path.minDist(obstaclePoints.points);
}

// Compute the cost of trajectory based on smoothness in velocity commands
double
CostEvaluator::smoothnessCostFunc(const Trajectory &trajectory,
                                  const std::array<double, 3> accLimits) {
  double smoothness_cost = 0.0;
  double delta_vx, delta_vy, delta_omega;
  for (size_t i = 1; i < trajectory.velocity.size(); ++i) {
    if (accLimits[0] > 0) {
      delta_vx = trajectory.velocity[i].vx - trajectory.velocity[i - 1].vx;
      smoothness_cost += std::pow(delta_vx, 2) / accLimits[0];
    }
    if (accLimits[1] > 0) {
      delta_vy = trajectory.velocity[i].vy - trajectory.velocity[i - 1].vy;
      smoothness_cost += std::pow(delta_vy, 2) / accLimits[1];
    }
    if (accLimits[2] > 0) {
      delta_omega =
          trajectory.velocity[i].omega - trajectory.velocity[i - 1].omega;
      smoothness_cost += std::pow(delta_omega, 2) / accLimits[2];
    }
  }
  return smoothness_cost / (3 * trajectory.velocity.size());
}

// Compute the cost of trajectory based on jerk in velocity commands
double CostEvaluator::jerkCostFunc(const Trajectory &trajectory,
                                   const std::array<double, 3> accLimits) {
  double jerk_cost = 0.0;
  double jerk_vx, jerk_vy, jerk_omega;
  for (size_t i = 2; i < trajectory.velocity.size(); ++i) {
    if (accLimits[0] > 0) {
      jerk_vx = trajectory.velocity[i].vx - 2 * trajectory.velocity[i - 1].vx +
                trajectory.velocity[i - 2].vx;
      jerk_cost += std::pow(jerk_vx, 2) / accLimits[0];
    }
    if (accLimits[1] > 0) {
      jerk_vy = trajectory.velocity[i].vy - 2 * trajectory.velocity[i - 1].vy +
                trajectory.velocity[i - 2].vy;
      jerk_cost += std::pow(jerk_vy, 2) / accLimits[1];
    }
    if (accLimits[2] > 0) {
      jerk_omega = trajectory.velocity[i].omega -
                   2 * trajectory.velocity[i - 1].omega +
                   trajectory.velocity[i - 2].omega;
      jerk_cost += std::pow(jerk_omega, 2) / accLimits[2];
    }
  }
  return jerk_cost / (3 * trajectory.velocity.size());
}

}; // namespace Control
} // namespace Kompass
