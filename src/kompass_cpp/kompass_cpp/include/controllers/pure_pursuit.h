#pragma once

#include "controllers/follower.h"
#include "utils/collision_check.h"
#include "utils/logger.h"
#include <Eigen/Dense>
#include <cmath>

namespace Kompass {
namespace Control {

/**
 * @brief Basic Pure Pursuit Path Follower
 * Algorithm details from PURDUS SIGBOTS
 * https://wiki.purduesigbots.com/software/control-algorithms/basic-pure-pursuit
 *
 */
class PurePursuit : public Follower {
public:
  class PurePursuitConfig : public Follower::FollowerParameters {
  public:
    PurePursuitConfig() : Follower::FollowerParameters() {
      addParameter("wheel_base", Parameter(0.34, 0.0, 100.0));
      // Number of future steps for collision prediction
      addParameter("prediction_horizon", Parameter(10, 0, 100));
      addParameter(
          "lookahead_gain_forward",
          Parameter(0.8, 0.001, 10.0,
                    "Factor to scale lookahead distance by velocity (k * v)"));
      addParameter(
          "path_search_step",
          Parameter(0.2, 0.001, 1000.0,
                    "Offset step to search for a new path when doing obstacle avoidance"));
      addParameter("max_search_candidates",
                   Parameter(10, 2, 1000,
                             "Number of search candidates to try for obstacle avoidance"));
    }
  };

  // Destructor
  virtual ~PurePursuit() = default;

  /**
   * @brief Construct a new Pure Pursuit object
   *
   * @param robotCtrlType
   * @param ctrlLimits
   * @param cfg
   */
  PurePursuit(const ControlType &robotCtrlType,
              const ControlLimitsParams &ctrlLimits,
              const CollisionChecker::ShapeType robotShapeType,
              const std::vector<float> robotDimensions,
              const Eigen::Vector3f &sensor_position_body,
              const Eigen::Vector4f &sensor_rotation_body,
              const double octreeRes = 0.1,
              const PurePursuitConfig &cfg = PurePursuitConfig());

  /**
   * @brief Executes one Pure Pursuit control step
   *
   * @param currentPosition
   * @param deltaTime
   * @return Controller::Result
   */
  Controller::Result execute(const Path::State currentPosition, const double deltaTime);

  Controller::Result execute(const double deltaTime);

  template <typename T>
  Controller::Result execute(const double deltaTime, const T &sensor_data) {
    collision_checker_->updateState(this->currentState);
    updateCollisionCheckerData(sensor_data);

    // Compute standard pure pursuit
    auto result = execute(deltaTime);
    // If the standard command is not found -> return
    if (result.status != Result::Status::COMMAND_FOUND)
      return result;

    // Check collisions for the nominal command and return it if no collisions
    // are found
    if (!checkCommandCollisions(result.velocity_command, deltaTime)) {
      return result; // Path is clear
    }

    LOG_DEBUG("PurePursuit: Obstacle detected on nominal path. Attempting "
              "avoidance...");

    Velocity2D safe_cmd = findSafeCommand(result.velocity_command, deltaTime);

    // If safe_cmd is zero (vx=0), it implies we should stop.
    // We return COMMAND_FOUND even if 0 velocity, as stopping is a valid safety
    // action.
    return {Result::Status::COMMAND_FOUND, safe_cmd};
  };

  /**
   * @brief Executes one Pure Pursuit control step with collision avoidance
   *
   * @tparam T
   * @param currentPosition
   * @param deltaTime
   * @param sensor_data
   * @return Controller::Result
   */
  template <typename T>
  Controller::Result execute(const Path::State currentPosition, const double deltaTime,
                             const T &sensor_data) {
    // Update position for the controller and the collision checker
    setCurrentState(currentPosition);
    return execute<T>(deltaTime, sensor_data);
  };

private:
  double wheel_base{0.0};
  double lookahead_gain_forward{0.0};
  size_t last_found_index_ = 0;
  int prediction_horizon{0};
  Eigen::VectorXf search_offsets_;
  // To enable pure pursuit with simple obstacle avoidance
  std::unique_ptr<CollisionChecker> collision_checker_;

  Path::Point findLookaheadPoint(double radius);

  /**
   * @brief Collision check for a velocity command
   *
   * @param cmd
   * @param dt
   * @return true
   * @return false
   */
  bool checkCommandCollisions(const Velocity2D &cmd, double dt);

  /**
   * @brief Find a safe (collision-free) command based on the pure-pursuit command
   *
   * @param nominal
   * @param dt
   * @return Velocity2D
   */
  Velocity2D findSafeCommand(const Velocity2D &nominal, double dt);

  // Helper: Update collision checker with LaserScan data
  void updateCollisionCheckerData(const Control::LaserScan &scan) {
    collision_checker_->updateScan(scan.ranges, scan.angles);
  }

  // Helper: Update collision checker with PointCloud data
  void updateCollisionCheckerData(const std::vector<Path::Point> &cloud) {
    collision_checker_->updatePointCloud(cloud);
  }
};

} // namespace Control
} // namespace Kompass
