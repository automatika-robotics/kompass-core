#pragma once

#include "controllers/controller.h"
#include "datatypes/control.h"
#include "datatypes/path.h"
#include "datatypes/trajectory.h"
#include "follower.h"
#include "utils/cost_evaluator.h"
#include "utils/logger.h"
#include "utils/trajectory_sampler.h"
#include <Eigen/Dense>
#include <memory>
#include <vector>

namespace Kompass {
namespace Control {
/**
 * @class DWA
 * @brief A class implementing a local planner using the Dynamic Window Approach
 */
class DWA : public Follower {
public:
  DWA(ControlLimitsParams controlLimits, ControlType controlType,
      double timeStep, double predictionHorizon, double controlHorizon,
      int maxLinearSamples, int maxAngularSamples,
      const CollisionChecker::ShapeType robotShapeType,
      const std::vector<float> robotDimensions,
      const Eigen::Vector3f &sensor_position_body,
      const Eigen::Vector4f &sensor_rotation_body, const double octreeRes,
      CostEvaluator::TrajectoryCostsWeights costWeights,
      const int maxNumThreads = 1);

  DWA(TrajectorySampler::TrajectorySamplerParameters config,
      ControlLimitsParams controlLimits, ControlType controlType,
      const CollisionChecker::ShapeType robotShapeType,
      const std::vector<float> robotDimensions,
      const Eigen::Vector3f &sensor_position_body,
      const Eigen::Vector4f &sensor_rotation_body,
      CostEvaluator::TrajectoryCostsWeights costWeights,
      const int maxNumThreads = 1);

  // DWA(DWAConfig &cfg);

  /**
   * @brief  Destructor
   */
  ~DWA() = default;

  /**
   * @brief Reconfigures the trajectory planner
   */
  // void reconfigure(DWAConfig &cfg);
  void configure(ControlLimitsParams controlLimits, ControlType controlType,
                 double timeStep, double predictionHorizon,
                 double controlHorizon, int maxLinearSamples,
                 int maxAngularSamples,
                 const CollisionChecker::ShapeType robotShapeType,
                 const std::vector<float> robotDimensions,
                 const Eigen::Vector3f &sensor_position_body,
                 const Eigen::Vector4f &sensor_rotation_body,
                 const double octreeRes,
                 CostEvaluator::TrajectoryCostsWeights costWeights,
                 const int maxNumThreads = 1);

  void configure(TrajectorySampler::TrajectorySamplerParameters config,
                 ControlLimitsParams controlLimits, ControlType controlType,
                 const CollisionChecker::ShapeType robotShapeType,
                 const std::vector<float> robotDimensions,
                 const Eigen::Vector3f &sensor_position_body,
                 const Eigen::Vector4f &sensor_rotation_body,
                 CostEvaluator::TrajectoryCostsWeights costWeights,
                 const int maxNumThreads = 1);

  /**
   * @brief Reset the resolution of the robot's local map, this is equivalent to
   * the box size representing each obstacle after conversion
   *
   * @param octreeRes
   */
  void resetOctreeResolution(const double octreeRes);

  /**
   * @brief Set the robot's sensor or local map max range (meters)
   *
   * @param max_range
   */
  void setSensorMaxRange(const float max_range);

  /**
   * @brief Set the Current State of the robot
   *
   * @param position
   */
  void setCurrentState(const Path::State &position);

  /**
   * @brief Adds a new custom cost to be used in the trajectory evaluation
   *
   * @param weight
   * @param custom_cost_function
   */
  void addCustomCost(double weight,
                     CostEvaluator::CustomCostFunction custom_cost_function);

  /**
   * @brief  Given the current position, orientation, and velocity of the robot,
   * compute velocity commands to send to the base
   * @param cmd_vel Will be filled with the velocity command to be passed to the
   * robot base
   * @return True if a valid trajectory was found, false otherwise
   */

  template <typename T>
  Controller::Result computeVelocityCommand(const Velocity2D &global_vel,
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
  };

  template <typename T>
  TrajSearchResult computeVelocityCommandsSet(const Velocity2D &global_vel,
                                              const T &scan_points) {
    TrajSearchResult searchRes = findBestPath(global_vel, scan_points);
    // Update latest velocity command
    if (searchRes.isTrajFound) {
      latest_velocity_command_ = searchRes.trajectory.velocities.getFront();
    }
    return searchRes;
  };

  std::tuple<MatrixXfR, MatrixXfR> getDebuggingSamples() const;

  Control::TrajectorySamples2D getDebuggingSamplesPure() const;

  template <typename T>
  void debugVelocitySearch(const Velocity2D &global_vel, const T &scan_points,
                           const bool &drop_samples) {
    // Throw an error if the global path is not set
    if (!currentPath) {
      throw std::invalid_argument("Pointer to global path is NULL. Cannot use "
                                  "DWA local planner without "
                                  "setting a global path");
    }
    // find closest segment to use in cost computation
    determineTarget();

    // Set trajectory sampler to maintain all samples for debugging mode
    trajSampler->setSampleDroppingMode(drop_samples);

    // Generate set of valid trajectories
    debuggingSamples_ = trajSampler->generateTrajectories(
        global_vel, currentState, scan_points);
  };

protected:
  std::unique_ptr<TrajectorySampler> trajSampler;
  std::unique_ptr<CostEvaluator> trajCostEvaluator;

  /**
   * @brief Given the current position and velocity of the robot, find the
   * best trajectory to execute
   * @param global_pose The current position of the robot
   * @param global_vel The current velocity of the robot
   * @param drive_velocities The velocities to send to the robot base
   * @return The highest scoring trajectory. A cost >= 0 means the
   * trajectory is legal to execute.
   */
  template <typename T>
  TrajSearchResult findBestPath(const Velocity2D &global_vel,
                                const T &scan_points) {
    // Throw an error if the global path is not set
    if (!currentPath) {
      throw std::invalid_argument("Pointer to global path is NULL. Cannot use "
                                  "DWA local planner without "
                                  "setting a global path");
    }
    // find closest segment to use in cost computation
    determineTarget();

    if (rotate_in_place and currentTrackedTarget_->heading_error >
                                goal_orientation_tolerance * 10.0) {
      // If the robot is rotating in place and the heading error is large, we
      // do not need to sample trajectories
      LOG_DEBUG("Rotating In Place ...");
      auto trajectory = trajSampler->generateSingleSampleFromVel(
          Velocity2D(0.0, 0.0,
                     -currentTrackedTarget_->heading_error *
                         ctrlimitsParams.omegaParams.maxOmega / M_PI));
      return TrajSearchResult{trajectory, true, 0.0};
    }

    // Generate set of valid trajectories in the DW
    std::unique_ptr<TrajectorySamples2D> samples_ =
        trajSampler->generateTrajectories(global_vel, currentState,
                                          scan_points);
    if (samples_->size() == 0) {
      return TrajSearchResult{Trajectory2D(), false, 0.0};
    }

    trajCostEvaluator->setPointScan(scan_points, currentState, maxLocalRange_);

    Path::Path trackedRefPathSegment = findTrackedPathSegment();

    // Evaluate the samples and get the sample with the minimum cost
    return trajCostEvaluator->getMinTrajectoryCost(samples_, currentPath.get(),
                                                   trackedRefPathSegment);
  };

private:
  double max_forward_distance_ = 0.0;
  int maxNumThreads;
  std::unique_ptr<TrajectorySamples2D> debuggingSamples_ = nullptr;
  float maxLocalRange_ =
      10.0; // Max range of the robot sensor or local map in meters. Used to
            // calculate the cost of coming close to obstacles

  /**
   * @brief get maximum reference path length
   */
  // size_t getMaxPathLength();
  size_t getMaxPathLength();

  Path::Path findTrackedPathSegment();

  void initJitCompile();
};

}; // namespace Control
} // namespace Kompass
