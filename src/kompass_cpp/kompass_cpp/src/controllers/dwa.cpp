#include "controllers/dwa.h"
#include "controllers/follower.h"
#include "datatypes/control.h"
#include "datatypes/path.h"
#include "datatypes/trajectory.h"
#include "utils/logger.h"
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

  // warmup
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

  // warmup
  initJitCompile();
}

// Warmup kernel launch: drives every CostEvaluator GPU kernel through a
// single throwaway getMinTrajectoryCost() call at construction time so the
// SYCL runtime performs JIT compilation here, not during the first real
// planner tick (where it would blow the control-cycle deadline).
void DWA::initJitCompile() {
  const int dummyNumSamples = 1;
  const int dummyNumPoints = 2;
  TrajectoryVelocitySamples2D velocities(dummyNumSamples, dummyNumPoints);
  TrajectoryPathSamples paths(dummyNumSamples, dummyNumPoints);
  velocities.push_back({Velocity2D(1.0, 0.0, 0.0)});
  auto dummyPath = Path::Path(
      {Path::Point(0.0f, 0.0f, 0.0f), Path::Point(1.0f, 1.0f, 0.0f)});
  dummyPath.interpolate(0.5, Path::InterpolationType::LINEAR);
  dummyPath.segment(1.0, 100);
  auto dummyPathView = dummyPath.getSegment(0);
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

void DWA::adaptPredictionHorizonToCurvature() {
  const double base_horizon = trajSampler->getBasePredictionHorizon();
  const double v_max = ctrlimitsParams.velXParams.maxVel;
  if (!currentPath || v_max < 1e-3 || max_point_interpolation_distance_ <= 0.0) {
    trajSampler->setPredictionHorizon(base_horizon);
    max_forward_distance_ = base_horizon * v_max;
    return;
  }

  // Scan curvature over the full reach the sampler could ever produce so we
  // see approaching tight curves before the rollout runs into them.
  const size_t start_idx = std::min(closestPosition->index,
                                    currentPath->getSize() - 1);
  const size_t peek_points = static_cast<size_t>(
      std::ceil(base_horizon * v_max / max_point_interpolation_distance_));
  const size_t end_idx =
      std::min(start_idx + peek_points, currentPath->getSize() - 1);

  float kappa_max = 0.0f;
  for (size_t i = start_idx; i <= end_idx; ++i) {
    kappa_max = std::max(
        kappa_max, std::abs(static_cast<float>(currentPath->getCurvature(i))));
  }

  // Shrink the prediction horizon based on the max curvature ahead on the reference path.

  // NOTE: This chord-arc bound is the same one used
  // for clothoid path smoothing [1] and curvature-adaptive pure-pursuit
  // lookahead [2], applied here to keep straight-tangent rollouts
  // cost-competitive with stationary on tight-curvature paths.
  //
  // Small-angle sagitta (kappa * L << 1):  s ≈ L^2 * kappa / 8. Solving
  // s ≤ tol for T gives  T ≤ sqrt(8 * tol / kappa) / v_max.
  //
  // [1] Brezak & Petrović, "Real-time approximation of clothoids with
  //     bounded error for path planning applications", IEEE TRO 30(2),
  //     507-515, 2014. doi:10.1109/TRO.2013.2294061
  // [2] Snider, "Automatic steering methods for autonomous automobile path
  //     tracking", CMU-RI-TR-09-08, 2009.
  double adaptive_horizon = base_horizon;
  if (kappa_max > curvature_horizon_tolerance_) {
    const double horizon_cap =
        std::sqrt(8.0 * curvature_horizon_tolerance_ / kappa_max) / v_max;
    adaptive_horizon = std::min(base_horizon, horizon_cap);
    LOG_DEBUG("Using Adaptive Horizon: ", adaptive_horizon);
  }

  trajSampler->setPredictionHorizon(adaptive_horizon);
  max_forward_distance_ = adaptive_horizon * v_max;
}

Path::Path::View DWA::findTrackedPathSegment() {
  // Start the segment from the closest point index
  size_t global_start_index = closestPosition->index;

  if (global_start_index >= currentPath->getSize()) {
    global_start_index = currentPath->getSize() - 1;
  }

  // Extend the lookahead to at least the furthest point the trajectory can
  // reach over the prediction horizon (max_forward_distance_ = maxVel *
  // predictionHorizon).
  // NOTE: Without this, any min-distance cost will read longitudinal
  // overshoot as lateral error and penalizes fast trajectories.
  size_t dynamic_lookahead = max_segment_size_;
  if (max_point_interpolation_distance_ > 0.0) {
    dynamic_lookahead = std::max(
        max_segment_size_,
        static_cast<size_t>(std::ceil(max_forward_distance_ /
                                      max_point_interpolation_distance_)) + 1);
  }

  size_t global_end_index = std::min(global_start_index + dynamic_lookahead,
                                     currentPath->getSize() - 1);
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
