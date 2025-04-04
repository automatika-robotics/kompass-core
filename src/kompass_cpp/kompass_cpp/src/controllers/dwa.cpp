#include "controllers/dwa.h"
#include "controllers/follower.h"
#include "datatypes/control.h"
#include "datatypes/path.h"
#include "datatypes/trajectory.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <memory>
#include <stdexcept>
#include <tuple>

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
         CostEvaluator::TrajectoryCostsWeights costWeights,
         const int maxNumThreads)
    : Follower() {
  // Setup the trajectory sampler and cost evaluator
  configure(controlLimits, controlType, timeStep, predictionHorizon,
            controlHorizon, maxLinearSamples, maxAngularSamples,
            robotShapeType, robotDimensions, sensor_position_body,
            sensor_rotation_body, octreeRes, costWeights, maxNumThreads);

  // Update the max forward distance the robot can make
  if (controlType == ControlType::OMNI) {
    max_forward_distance_ = std::max(controlLimits.velXParams.maxVel,
                                     controlLimits.velYParams.maxVel) *
                            predictionHorizon;
  } else {
    max_forward_distance_ = controlLimits.velXParams.maxVel * predictionHorizon;
  }
}

DWA::DWA(TrajectorySampler::TrajectorySamplerParameters config,
         ControlLimitsParams controlLimits, ControlType controlType,
         const CollisionChecker::ShapeType robotShapeType,
         const std::vector<float> robotDimensions,
         const std::array<float, 3> &sensor_position_body,
         const std::array<float, 4> &sensor_rotation_body,
         CostEvaluator::TrajectoryCostsWeights costWeights,
         const int maxNumThreads)
    : Follower() {
  // Setup the trajectory sampler and cost evaluator
  configure(config, controlLimits, controlType, robotShapeType,
            robotDimensions, sensor_position_body, sensor_rotation_body,
            costWeights, maxNumThreads);

  // Update the max forward distance the robot can make
  double timeHorizon = config.getParameter<double>("control_horizon");
  if (controlType == ControlType::OMNI) {
    max_forward_distance_ = std::max(controlLimits.velXParams.maxVel,
                                     controlLimits.velYParams.maxVel) *
                            timeHorizon;
  } else {
    max_forward_distance_ = controlLimits.velXParams.maxVel * timeHorizon;
  }
}

void DWA::configure(ControlLimitsParams controlLimits,
                      ControlType controlType, double timeStep,
                      double predictionHorizon, double controlHorizon,
                      int maxLinearSamples, int maxAngularSamples,
                      const CollisionChecker::ShapeType robotShapeType,
                      const std::vector<float> robotDimensions,
                      const std::array<float, 3> &sensor_position_body,
                      const std::array<float, 4> &sensor_rotation_body,
                      const double octreeRes,
                      CostEvaluator::TrajectoryCostsWeights costWeights,
                      const int maxNumThreads) {
  trajSampler = std::make_unique<TrajectorySampler>(
      controlLimits, controlType, timeStep, predictionHorizon, controlHorizon,
      maxLinearSamples, maxAngularSamples, robotShapeType, robotDimensions,
      sensor_position_body, sensor_rotation_body, octreeRes, maxNumThreads));

  trajCostEvaluator = std::make_unique<CostEvaluator>(
      costWeights, sensor_position_body, sensor_rotation_body, controlLimits,
      trajSampler->numTrajectories, trajSampler->numPointsPerTrajectory,
      maxSegmentSize);
  this->maxNumThreads = maxNumThreads;
}

void DWA::configure(TrajectorySampler::TrajectorySamplerParameters config,
                      ControlLimitsParams controlLimits,
                      ControlType controlType,
                      const CollisionChecker::ShapeType robotShapeType,
                      const std::vector<float> robotDimensions,
                      const std::array<float, 3> &sensor_position_body,
                      const std::array<float, 4> &sensor_rotation_body,
                      CostEvaluator::TrajectoryCostsWeights costWeights,
                      const int maxNumThreads) {
  trajSampler = std::make_unique<TrajectorySampler>(
      config, controlLimits, controlType, robotShapeType, robotDimensions,
      sensor_position_body, sensor_rotation_body, maxNumThreads));

  trajCostEvaluator = std::make_unique<CostEvaluator>(
      costWeights, sensor_position_body, sensor_rotation_body, controlLimits,
      trajSampler->numTrajectories, trajSampler->numPointsPerTrajectory,
      maxSegmentSize);
  this->maxNumThreads = maxNumThreads;
}

void DWA::resetOctreeResolution(const double octreeRes) {
  trajSampler->resetOctreeResolution(octreeRes);
}

void DWA::setSensorMaxRange(const float max_range) {
  maxLocalRange_ = max_range;
}

void DWA::addCustomCost(
    double weight, CostEvaluator::CustomCostFunction custom_cost_function) {
  trajCostEvaluator->addCustomCost(weight, custom_cost_function);
}

void DWA::setCurrentState(const Path::State &position) {
  this->currentState = position;
  this->trajSampler->updateState(position);
}

Path::Path DWA::findTrackedPathSegment() {
  std::vector<Path::Point> trackedPoints;
  size_t segment_index{current_segment_index_ + 1};
  Path::Path currentSegment = currentPath->segments[current_segment_index_];

  // If we reached end of the current segment and a new segment is available ->
  // take the next segment
  if (closestPosition->index > currentSegment.points.size() - 1 and
      current_segment_index_ < max_segment_index_) {
    trackedPoints = currentPath->segments[current_segment_index_ + 1].points;
    segment_index = current_segment_index_ + 1;
  }
  // Else take the segment points from the current point onwards
  else {
    trackedPoints = {currentSegment.points.begin() + closestPosition->index,
                     currentSegment.points.end()};
  }
  size_t point_index{0};

  float segment_length = 0.0;
  for (size_t i = 1; i < trackedPoints.size(); ++i) {
    segment_length +=
        Path::Path::distance(trackedPoints[i - 1], trackedPoints[i]);
  }

  // If the segment does not have the required number of points add more points
  // from next path segment
  while (segment_length < max_forward_distance_ and
         segment_index <= max_segment_index_ and
         trackedPoints.size() < maxSegmentSize) {
    if (point_index >= currentPath->segments[segment_index].points.size()) {
      point_index = 0;
      segment_index++;
      if (segment_index > max_segment_index_) {
        break;
      }
    }
    // Add distance between last point and new point
    Path::Point back_point = trackedPoints.back();
    segment_length += calculateDistance(
        back_point, currentPath->segments[segment_index].points[point_index]);
    trackedPoints.push_back(
        currentPath->segments[segment_index].points[point_index]);
    point_index++;
  }

  return Path::Path(trackedPoints);
}

template <typename T>
TrajSearchResult DWA::findBestPath(const Velocity2D &global_vel,
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
  std::unique_ptr<TrajectorySamples2D> samples_ =
      trajSampler->generateTrajectories(global_vel, currentState, scan_points);
  if (samples_->size() == 0) {
    return TrajSearchResult();
  }

  trajCostEvaluator->setPointScan(scan_points, currentState, maxLocalRange_);

  Path::Path trackedRefPathSegment = findTrackedPathSegment();

  // Evaluate the samples and get the sample with the minimum cost
  return trajCostEvaluator->getMinTrajectoryCost(samples_, *currentPath,
                                                 trackedRefPathSegment);
}

template <typename T>
Controller::Result DWA::computeVelocityCommand(const Velocity2D &global_vel,
                                               const T &scan_points) {
  TrajSearchResult searchRes = findBestPath(global_vel, scan_points);
  Controller::Result finalResult;
  if (searchRes.isTrajFound) {
    finalResult.status = Controller::Result::Status::COMMAND_FOUND;
    // Get the first command to be applied
    finalResult.velocity_command = searchRes.trajectory.velocities.getFront();
    latest_velocity_command_ = finalResult.velocity_command;
  } else {
    finalResult.status = Controller::Result::Status::NO_COMMAND_POSSIBLE;
  }
  return finalResult;
}

TrajSearchResult DWA::computeVelocityCommandsSet(const Velocity2D &global_vel,
                                                 const LaserScan &scan) {
  TrajSearchResult searchRes = findBestPath(global_vel, scan);
  // Update latest velocity command
  if (searchRes.isTrajFound) {
    latest_velocity_command_ = searchRes.trajectory.velocities.getFront();
  }
  return searchRes;
}

TrajSearchResult
DWA::computeVelocityCommandsSet(const Velocity2D &global_vel,
                                const std::vector<Path::Point> &cloud) {

  TrajSearchResult searchRes = findBestPath(global_vel, cloud);
  // Update latest velocity command
  if (searchRes.isTrajFound) {
    latest_velocity_command_ = searchRes.trajectory.velocities.getFront();
  }
  return searchRes;
}

std::tuple<MatrixXfR, MatrixXfR> DWA::getDebuggingSamples() const {
  if (debuggingSamples_ == nullptr) {
    throw std::invalid_argument("No debugging samples are available");
  }
  size_t paths_size = debuggingSamples_->paths.size();
  auto paths_x = debuggingSamples_->paths.x.topRows(paths_size).eval();
  auto paths_y = debuggingSamples_->paths.y.topRows(paths_size).eval();
  return std::tie(paths_x, paths_y);
}

Control::TrajectorySamples2D DWA::getDebuggingSamplesPure() const {
  if (debuggingSamples_ == nullptr) {
    throw std::invalid_argument("No debugging samples are available");
  }
  return *debuggingSamples_;
}

}; // namespace Control
} // namespace Kompass
