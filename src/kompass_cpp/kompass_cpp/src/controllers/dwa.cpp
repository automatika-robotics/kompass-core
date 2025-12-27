#include "controllers/dwa.h"
#include "controllers/follower.h"
#include "datatypes/control.h"
#include "datatypes/path.h"
#include "datatypes/trajectory.h"
#include <cmath>
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
         const Eigen::Vector3f &sensor_position_body,
         const Eigen::Vector4f &sensor_rotation_body, const double octreeRes,
         CostEvaluator::TrajectoryCostsWeights costWeights,
         const int maxNumThreads)
    : Follower() {
  // Setup the trajectory sampler and cost evaluator
  configure(controlLimits, controlType, timeStep, predictionHorizon,
            controlHorizon, maxLinearSamples, maxAngularSamples, robotShapeType,
            robotDimensions, sensor_position_body, sensor_rotation_body,
            octreeRes, costWeights, maxNumThreads);

  // Update the max forward distance the robot can make
  if (controlType == ControlType::OMNI) {
    max_forward_distance_ = std::max(controlLimits.velXParams.maxVel,
                                     controlLimits.velYParams.maxVel) *
                            predictionHorizon;
  } else {
    max_forward_distance_ = controlLimits.velXParams.maxVel * predictionHorizon;
  }

  initJitCompile();
}

DWA::DWA(TrajectorySampler::TrajectorySamplerParameters config,
         ControlLimitsParams controlLimits, ControlType controlType,
         const CollisionChecker::ShapeType robotShapeType,
         const std::vector<float> robotDimensions,
         const Eigen::Vector3f &sensor_position_body,
         const Eigen::Vector4f &sensor_rotation_body,
         CostEvaluator::TrajectoryCostsWeights costWeights,
         const int maxNumThreads)
    : Follower() {
  // Setup the trajectory sampler and cost evaluator
  configure(config, controlLimits, controlType, robotShapeType, robotDimensions,
            sensor_position_body, sensor_rotation_body, costWeights,
            maxNumThreads);

  // Update the max forward distance the robot can make
  double timeHorizon = config.getParameter<double>("control_horizon");
  if (controlType == ControlType::OMNI) {
    max_forward_distance_ = std::max(controlLimits.velXParams.maxVel,
                                     controlLimits.velYParams.maxVel) *
                            timeHorizon;
  } else {
    max_forward_distance_ = controlLimits.velXParams.maxVel * timeHorizon;
  }

  initJitCompile();
}

void DWA::initJitCompile() {
  const int dummyNumSamples = 1;
  const int dummyNumPoints = 2;
  TrajectoryVelocitySamples2D velocities(dummyNumSamples, dummyNumPoints);
  TrajectoryPathSamples paths(dummyNumSamples, dummyNumPoints);
  velocities.push_back({Velocity2D(1.0, 0.0, 0.0)});
  auto dummyPath = Path::Path(
      {Path::Point(0.0f, 0.0f, 0.0f), Path::Point(1.0f, 1.0f, 0.0f)});
  auto dummyPathView = dummyPath.getPart(0, 1);
  paths.push_back(dummyPath);
  std::unique_ptr<TrajectorySamples2D> dummySamples =
      std::make_unique<TrajectorySamples2D>(velocities, paths);
  trajCostEvaluator->getMinTrajectoryCost(dummySamples, &dummyPath,
                                          dummyPathView);
}

void DWA::configure(ControlLimitsParams controlLimits, ControlType controlType,
                    double timeStep, double predictionHorizon,
                    double controlHorizon, int maxLinearSamples,
                    int maxAngularSamples,
                    const CollisionChecker::ShapeType robotShapeType,
                    const std::vector<float> robotDimensions,
                    const Eigen::Vector3f &sensor_position_body,
                    const Eigen::Vector4f &sensor_rotation_body,
                    const double octreeRes,
                    CostEvaluator::TrajectoryCostsWeights costWeights,
                    const int maxNumThreads) {
  trajSampler = std::make_unique<TrajectorySampler>(
      controlLimits, controlType, timeStep, predictionHorizon, controlHorizon,
      maxLinearSamples, maxAngularSamples, robotShapeType, robotDimensions,
      sensor_position_body, Eigen::Quaternionf(sensor_rotation_body), octreeRes,
      maxNumThreads);

  trajCostEvaluator = std::make_unique<CostEvaluator>(
      costWeights, sensor_position_body,
      Eigen::Quaternionf(sensor_rotation_body), controlLimits,
      trajSampler->numTrajectories, trajSampler->numPointsPerTrajectory,
      max_segment_size_);
  this->maxNumThreads = maxNumThreads;
}

void DWA::configure(TrajectorySampler::TrajectorySamplerParameters config,
                    ControlLimitsParams controlLimits, ControlType controlType,
                    const CollisionChecker::ShapeType robotShapeType,
                    const std::vector<float> robotDimensions,
                    const Eigen::Vector3f &sensor_position_body,
                    const Eigen::Vector4f &sensor_rotation_body,
                    CostEvaluator::TrajectoryCostsWeights costWeights,
                    const int maxNumThreads) {
  trajSampler = std::make_unique<TrajectorySampler>(
      config, controlLimits, controlType, robotShapeType, robotDimensions,
      sensor_position_body, Eigen::Quaternionf(sensor_rotation_body),
      maxNumThreads);

  trajCostEvaluator = std::make_unique<CostEvaluator>(
      costWeights, sensor_position_body,
      Eigen::Quaternionf(sensor_rotation_body), controlLimits,
      trajSampler->numTrajectories, trajSampler->numPointsPerTrajectory,
      max_segment_size_);
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

Path::Path::View DWA::findTrackedPathSegment() {
  // Start the segment from the closest point index
  size_t global_start_index = closestPosition->index;

  if (global_start_index >= currentPath->getSize()) {
    global_start_index = currentPath->getSize() - 1;
  }

  // Find the End (Lookahead) Index based on max_forward_distance_ and
  // maxSegmentSize
  size_t global_end_index = global_start_index;
  float accumulated_distance = 0.0f;
  size_t points_count = 1;

  // Loop conditions:
  // - Don't go past the end of the entire path
  // - Don't exceed max points allowed in a local plan
  // - Don't exceed the lookahead distance (max_forward_distance_)
  while ((global_end_index + 1) < currentPath->getSize() &&
         points_count < this->max_segment_size_) {

    // Calculate distance to the NEXT point
    float dist =
        Path::Path::distance(currentPath->getIndex(global_end_index),
                             currentPath->getIndex(global_end_index + 1));

    if (accumulated_distance + dist > max_forward_distance_) {
      break;
    }

    accumulated_distance += dist;
    global_end_index++;
    points_count++;
  }

  // Create and Return the View
  return currentPath->getPart(global_start_index, global_end_index);
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
