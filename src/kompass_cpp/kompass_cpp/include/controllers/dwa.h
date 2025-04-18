#pragma once

#include "controllers/controller.h"
#include "datatypes/control.h"
#include "datatypes/path.h"
#include "datatypes/trajectory.h"
#include "follower.h"
#include "utils/cost_evaluator.h"
#include "utils/trajectory_sampler.h"
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
      const std::array<float, 3> &sensor_position_body,
      const std::array<float, 4> &sensor_rotation_body, const double octreeRes,
      CostEvaluator::TrajectoryCostsWeights costWeights,
      const int maxNumThreads = 1);

  DWA(TrajectorySampler::TrajectorySamplerParameters config,
      ControlLimitsParams controlLimits, ControlType controlType,
      const CollisionChecker::ShapeType robotShapeType,
      const std::vector<float> robotDimensions,
      const std::array<float, 3> &sensor_position_body,
      const std::array<float, 4> &sensor_rotation_body,
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
                   const std::array<float, 3> &sensor_position_body,
                   const std::array<float, 4> &sensor_rotation_body,
                   const double octreeRes,
                   CostEvaluator::TrajectoryCostsWeights costWeights,
                   const int maxNumThreads = 1);

  void configure(TrajectorySampler::TrajectorySamplerParameters config,
                   ControlLimitsParams controlLimits, ControlType controlType,
                   const CollisionChecker::ShapeType robotShapeType,
                   const std::vector<float> robotDimensions,
                   const std::array<float, 3> &sensor_position_body,
                   const std::array<float, 4> &sensor_rotation_body,
                   CostEvaluator::TrajectoryCostsWeights costWeights,
                   const int maxNumThreads = 1);

  void resetOctreeResolution(const double octreeRes);

  void setSensorMaxRange(const float max_range);

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
                                            const T &scan_points);

  TrajSearchResult computeVelocityCommandsSet(const Velocity2D &global_vel,
                                              const LaserScan &scan);
  TrajSearchResult
  computeVelocityCommandsSet(const Velocity2D &global_vel,
                             const std::vector<Path::Point> &cloud);

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

private:
  std::unique_ptr<TrajectorySampler> trajSampler;
  std::unique_ptr<CostEvaluator> trajCostEvaluator;
  double max_forward_distance_ = 0.0;
  int maxNumThreads;
  float maxLocalRange_ = 10.0;    // Max range of the laserscan or the robot local map, default to 10 meters. Used to calculate the cost of coming close to obstacles
  std::unique_ptr<TrajectorySamples2D> debuggingSamples_;

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
                                const T &scan_points);

  Path::Path findTrackedPathSegment();
};

}; // namespace Control
} // namespace Kompass
