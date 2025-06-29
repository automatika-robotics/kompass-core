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
  paths.push_back(dummyPath);
  std::unique_ptr<TrajectorySamples2D> dummySamples =
      std::make_unique<TrajectorySamples2D>(velocities, paths);
  trajCostEvaluator->getMinTrajectoryCost(dummySamples, &dummyPath, dummyPath);
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
      maxSegmentSize);
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

  size_t segment_index{current_segment_index_ + 1};
  Path::Path currentSegment = currentPath->segments[current_segment_index_];

  // If we reached end of the current segment and a new segment is available ->
  // take the next segment
  if (closestPosition->index >= currentSegment.getSize() - 1 and
      current_segment_index_ < max_segment_index_) {
    segment_index = current_segment_index_ + 1;
    return currentPath->segments[current_segment_index_ + 1];
  } else if (closestPosition->index >= currentSegment.getSize() - 1) {
    // Return current segment directly (last segment)
    return currentPath->segments[current_segment_index_];
  }
  // Else take the segment points from the current point onwards
  else {
    auto trackedPath = currentSegment.getPart(closestPosition->index,
                                              currentSegment.getSize() - 1,
                                              this->maxSegmentSize);
    size_t point_index{0};

    float segment_length = trackedPath.totalPathLength();

    // If the segment does not have the required number of points add more
    // points from next path segment
    while (segment_length < max_forward_distance_ and
           segment_index <= max_segment_index_ and
           trackedPath.getSize() < maxSegmentSize - 1) {
      if (point_index >= currentPath->segments[segment_index].getSize()) {
        point_index = 0;
        segment_index++;
        if (segment_index > max_segment_index_) {
          break;
        }
      }
      // Add distance between last point and new point
      Path::Point back_point = trackedPath.getEnd();
      segment_length += calculateDistance(
          back_point,
          currentPath->segments[segment_index].getIndex(point_index));
      trackedPath.pushPoint(
          currentPath->segments[segment_index].getIndex(point_index));
      point_index++;
    }

    return trackedPath;
  }
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
