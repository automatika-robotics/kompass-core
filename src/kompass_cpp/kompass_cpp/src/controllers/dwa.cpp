#include "controllers/dwa.h"
#include "controllers/follower.h"
#include "datatypes/control.h"
#include "datatypes/path.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>

namespace Kompass {
namespace Control {
DWA::DWA(ControlLimitsParams controlLimits, ControlType controlType,
         double timeStep, double predictionHorizon, double controlHorizon,
         int maxLinearSamples, int maxAngularSamples,
         const CollisionChecker::ShapeType robotShapeType,
         const std::vector<float> robotDimensions,
         const std::array<float, 3> &sensor_position_body,
         const std::array<float, 4> &sensor_rotation_body,
         const double octreeRes,
         CostEvaluator::TrajectoryCostsWeights costWeights, const int maxNumThreads)
    : Follower() {
  // Setup the trajectory sampler
  trajSampler = new TrajectorySampler(
      controlLimits, controlType, timeStep, predictionHorizon, controlHorizon,
      maxLinearSamples, maxAngularSamples, robotShapeType, robotDimensions,
      sensor_position_body, sensor_rotation_body, octreeRes, maxNumThreads);

  trajCostEvaluator = new CostEvaluator(costWeights, sensor_position_body,
                                        sensor_rotation_body);

  // Update the max forward distance the robot can make
  if (controlType == ControlType::OMNI) {
    max_forward_distance_ = std::max(controlLimits.velXParams.maxVel,
                                     controlLimits.velYParams.maxVel) *
                            predictionHorizon;
  } else {
    max_forward_distance_ = controlLimits.velXParams.maxVel * predictionHorizon;
  }
  this->maxNumThreads = maxNumThreads;
}

DWA::DWA(TrajectorySampler::TrajectorySamplerParameters config,
         ControlLimitsParams controlLimits, ControlType controlType,
         const CollisionChecker::ShapeType robotShapeType,
         const std::vector<float> robotDimensions,
         const std::array<float, 3> &sensor_position_body,
         const std::array<float, 4> &sensor_rotation_body,
         CostEvaluator::TrajectoryCostsWeights costWeights, const int maxNumThreads)
    : Follower() {
  // Setup the trajectory sampler
  trajSampler = new TrajectorySampler(
      config, controlLimits, controlType, robotShapeType, robotDimensions,
      sensor_position_body, sensor_rotation_body, maxNumThreads);

  trajCostEvaluator = new CostEvaluator(costWeights, sensor_position_body,
                                        sensor_rotation_body);

  // Update the max forward distance the robot can make
  double timeHorizon = config.getParameter<double>("time_horizon");
  if (controlType == ControlType::OMNI) {
    max_forward_distance_ = std::max(controlLimits.velXParams.maxVel,
                                     controlLimits.velYParams.maxVel) *
                            timeHorizon;
  } else {
    max_forward_distance_ = controlLimits.velXParams.maxVel * timeHorizon;
  }
  this->maxNumThreads = maxNumThreads;
}

DWA::~DWA() {
  delete trajSampler;
  delete trajCostEvaluator;
}

void DWA::reconfigure(ControlLimitsParams controlLimits,
                      ControlType controlType, double timeStep,
                      double predictionHorizon, double controlHorizon,
                      int maxLinearSamples, int maxAngularSamples,
                      const CollisionChecker::ShapeType robotShapeType,
                      const std::vector<float> robotDimensions,
                      const std::array<float, 3> &sensor_position_body,
                      const std::array<float, 4> &sensor_rotation_body,
                      const double octreeRes,
                      CostEvaluator::TrajectoryCostsWeights costWeights, const int maxNumThreads) {
  delete trajSampler;
  trajSampler = new TrajectorySampler(
      controlLimits, controlType, timeStep, predictionHorizon, controlHorizon,
      maxLinearSamples, maxAngularSamples, robotShapeType, robotDimensions,
      sensor_position_body, sensor_rotation_body, octreeRes, maxNumThreads);

  delete trajCostEvaluator;
  trajCostEvaluator = new CostEvaluator(costWeights);
  this->maxNumThreads = maxNumThreads;
}

void DWA::reconfigure(TrajectorySampler::TrajectorySamplerParameters config,
                      ControlLimitsParams controlLimits,
                      ControlType controlType,
                      const CollisionChecker::ShapeType robotShapeType,
                      const std::vector<float> robotDimensions,
                      const std::array<float, 3> &sensor_position_body,
                      const std::array<float, 4> &sensor_rotation_body,
                      CostEvaluator::TrajectoryCostsWeights costWeights, const int maxNumThreads) {
  delete trajSampler;
  trajSampler = new TrajectorySampler(
      config, controlLimits, controlType, robotShapeType, robotDimensions,
      sensor_position_body, sensor_rotation_body, maxNumThreads);

  delete trajCostEvaluator;
  trajCostEvaluator = new CostEvaluator(costWeights);
  this->maxNumThreads = maxNumThreads;
}

void DWA::addCustomCost(
    double weight, CostEvaluator::CustomCostFunction custom_cost_function) {
  trajCostEvaluator->addCustomCost(weight, custom_cost_function);
}

Path::Path DWA::findTrackedPathSegment() {
  Path::Path trackedPathSegment;
  size_t segment_index{current_segment_index_ + 1};
  Path::Path currentSegment = currentPath->segments[current_segment_index_];

  // If we reached end of the current segment and a new segment is available ->
  // take the next segment
  if (closestPosition->index > currentSegment.points.size() - 1 and
      current_segment_index_ < max_segment_index_) {
    trackedPathSegment = currentPath->segments[current_segment_index_ + 1];
    segment_index = current_segment_index_ + 1;
  }
  // Else take the segment points from the current point onwards
  else {
    trackedPathSegment.points = {currentSegment.points.begin() +
                                     closestPosition->index,
                                 currentSegment.points.end()};
  }
  size_t point_index{0};

  double segment_length = trackedPathSegment.totalPathLength();

  // If the segment does not have the required number of points add more points
  // from next path segment
  while (segment_length <= max_forward_distance_ and
         segment_index <= max_segment_index_) {
    if (point_index >= currentPath->segments[segment_index].points.size()) {
      point_index = 0;
      segment_index++;
      if (segment_index > max_segment_index_) {
        break;
      }
    }
    // Add distance between last point and new point
    Path::Point back_point = trackedPathSegment.points.back();
    segment_length += calculateDistance(
        back_point, currentPath->segments[segment_index].points[point_index]);
    trackedPathSegment.points.push_back(
        currentPath->segments[segment_index].points[point_index]);
    point_index++;
  }

  return trackedPathSegment;
}

TrajSearchResult DWA::findBestSegment(const std::vector<Trajectory> &samples) {
  double minCost = std::numeric_limits<double>::max();
  Trajectory minCostTraj;
  bool traj_found = false;
  Path::Path trackedRefPathSegment = findTrackedPathSegment();

  if (maxNumThreads > 1) {
    ThreadPool pool(maxNumThreads);
    // Mutex for locking updates to minCost and minCostTraj
    std::mutex mtx;

    for (const auto &sample : samples) {
      pool.enqueue([this, &sample, &trackedRefPathSegment, &minCost,
                    &minCostTraj, &traj_found, &mtx](void) {
        double traj_cost = trajCostEvaluator->getTrajectoryCost(
            sample, *currentPath, trackedRefPathSegment,
            current_segment_index_);
        // Lock to update the minimum cost and trajectory
        {
          std::lock_guard<std::mutex> lock(mtx);
          if (traj_cost < minCost) {
            minCost = traj_cost;
            minCostTraj = sample;
            traj_found = true;
          }
        }
      });
    }

  } else {
    // Evaluate the samples and get the sample with the minimum cost
    for (const auto &sample : samples) {
      double traj_cost = trajCostEvaluator->getTrajectoryCost(
          sample, *currentPath, trackedRefPathSegment, current_segment_index_);

      if (traj_cost < minCost) {
        minCost = traj_cost;
        minCostTraj = sample;
        traj_found = true;
      }
    }
  }

  return {traj_found, minCost, minCostTraj};
}

template <typename T>
TrajSearchResult DWA::findBestPath(const Velocity &global_vel,
                                   const T &scan_points) {
  // Throw an error if the global path is not set
  if (!currentPath) {
    throw std::invalid_argument(
        "Pointer to global path is NULL. Cannot use DWA local planner without "
        "setting a global path");
  }
  // find closest segment to use in cost computation
  determineTarget();

  // Generate set of valid trajectories in the DW
  std::vector<Trajectory> samples_ =
      trajSampler->generateTrajectories(global_vel, currentState, scan_points);

  trajCostEvaluator->setPointScan(scan_points, currentState);

  return findBestSegment(samples_);
}

template <typename T>
Controller::Result DWA::computeVelocityCommand(const Velocity &global_vel,
                                               const T &scan_points) {
  TrajSearchResult searchRes = findBestPath(global_vel, scan_points);
  Controller::Result finalResult;
  if (searchRes.isTrajFound) {
    finalResult.status = Controller::Result::Status::COMMAND_FOUND;
    // Get the first command to be applied
    finalResult.velocity_command = searchRes.trajectory.velocity.front();
    latest_velocity_command_ = finalResult.velocity_command;
  } else {
    finalResult.status = Controller::Result::Status::NO_COMMAND_POSSIBLE;
  }
  return finalResult;
}

TrajSearchResult DWA::computeVelocityCommandsSet(const Velocity &global_vel,
                                                 const LaserScan &scan) {
  TrajSearchResult searchRes = findBestPath(global_vel, scan);
  // Update latest velocity command
  if (searchRes.isTrajFound) {
    latest_velocity_command_ = searchRes.trajectory.velocity.front();
  }
  return searchRes;
}

TrajSearchResult
DWA::computeVelocityCommandsSet(const Velocity &global_vel,
                                const std::vector<Point3D> &cloud) {

  TrajSearchResult searchRes = findBestPath(global_vel, cloud);
  // Update latest velocity command
  if (searchRes.isTrajFound) {
    latest_velocity_command_ = searchRes.trajectory.velocity.front();
  }
  return searchRes;
}

}; // namespace Control
} // namespace Kompass
