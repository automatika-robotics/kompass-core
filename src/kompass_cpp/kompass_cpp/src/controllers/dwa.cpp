#include "controllers/dwa.h"
#include "controllers/follower.h"
#include "datatypes/control.h"
#include "datatypes/path.h"
#include "datatypes/trajectory.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <ostream>
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
  // Setup the trajectory sampler
  trajSampler = new TrajectorySampler(
      controlLimits, controlType, timeStep, predictionHorizon, controlHorizon,
      maxLinearSamples, maxAngularSamples, robotShapeType, robotDimensions,
      sensor_position_body, sensor_rotation_body, octreeRes, maxNumThreads);

  trajCostEvaluator =
      new CostEvaluator(costWeights, sensor_position_body, sensor_rotation_body,
                        controlLimits, trajSampler->numTrajectories,
                        trajSampler->numPointsPerTrajectory, getMaxPathLength());

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
         CostEvaluator::TrajectoryCostsWeights costWeights,
         const int maxNumThreads)
    : Follower() {
  // Setup the trajectory sampler
  trajSampler = new TrajectorySampler(
      config, controlLimits, controlType, robotShapeType, robotDimensions,
      sensor_position_body, sensor_rotation_body, maxNumThreads);

  trajCostEvaluator =
      new CostEvaluator(costWeights, sensor_position_body, sensor_rotation_body,
                        controlLimits, trajSampler->numTrajectories,
                        trajSampler->numPointsPerTrajectory, getMaxPathLength());

  // Update the max forward distance the robot can make
  double timeHorizon = config.getParameter<double>("control_horizon");
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
  delete debuggingSamples_;
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
                      CostEvaluator::TrajectoryCostsWeights costWeights,
                      const int maxNumThreads) {
  delete trajSampler;
  trajSampler = new TrajectorySampler(
      controlLimits, controlType, timeStep, predictionHorizon, controlHorizon,
      maxLinearSamples, maxAngularSamples, robotShapeType, robotDimensions,
      sensor_position_body, sensor_rotation_body, octreeRes, maxNumThreads);

  delete trajCostEvaluator;
  trajCostEvaluator =
      new CostEvaluator(costWeights, sensor_position_body, sensor_rotation_body,
                        controlLimits, trajSampler->numTrajectories,
                        trajSampler->numPointsPerTrajectory, getMaxPathLength());
  this->maxNumThreads = maxNumThreads;
}

void DWA::reconfigure(TrajectorySampler::TrajectorySamplerParameters config,
                      ControlLimitsParams controlLimits,
                      ControlType controlType,
                      const CollisionChecker::ShapeType robotShapeType,
                      const std::vector<float> robotDimensions,
                      const std::array<float, 3> &sensor_position_body,
                      const std::array<float, 4> &sensor_rotation_body,
                      CostEvaluator::TrajectoryCostsWeights costWeights,
                      const int maxNumThreads) {
  delete trajSampler;
  trajSampler = new TrajectorySampler(
      config, controlLimits, controlType, robotShapeType, robotDimensions,
      sensor_position_body, sensor_rotation_body, maxNumThreads);

  delete trajCostEvaluator;

  trajCostEvaluator =
      new CostEvaluator(costWeights, sensor_position_body, sensor_rotation_body,
                        controlLimits, trajSampler->numTrajectories,
                        trajSampler->numPointsPerTrajectory, getMaxPathLength());
  this->maxNumThreads = maxNumThreads;
}

size_t DWA::getMaxPathLength() {
  return this->config.getParameter<double>("max_path_length") /
         this->config.getParameter<double>("max_point_interpolation_distance");
}

void DWA::resetOctreeResolution(const double octreeRes) {
  trajSampler->resetOctreeResolution(octreeRes);
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
  TrajectorySamples2D samples_ =
      trajSampler->generateTrajectories(global_vel, currentState, scan_points);

  if (samples_.size() == 0) {
    return TrajSearchResult();
  }

  trajCostEvaluator->setPointScan(scan_points, currentState);

  Path::Path trackedRefPathSegment = findTrackedPathSegment();

  // Evaluate the samples and get the sample with the minimum cost
  return trajCostEvaluator->getMinTrajectoryCost(
      samples_, *currentPath, trackedRefPathSegment, current_segment_index_);
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
