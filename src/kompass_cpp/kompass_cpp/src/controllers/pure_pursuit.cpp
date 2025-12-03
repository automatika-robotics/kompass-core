#include "controllers/pure_pursuit.h"
#include "controllers/follower.h"
#include "datatypes/control.h"
#include "datatypes/path.h"
#include "utils/angles.h"
#include "utils/logger.h"

namespace Kompass {
namespace Control {

PurePursuit::PurePursuit(const ControlType &robotCtrlType,
                         const ControlLimitsParams &ctrlLimits,
                         const PurePursuitConfig &cfg)
    : Follower() {
  setParams(cfg);
  setControlType(robotCtrlType);
  ctrlimitsParams = ctrlLimits;
  wheel_base = cfg.getParameter<double>("wheel_base");
  lookahead_gain_forward = cfg.getParameter<double>("lookahead_gain_forward");
}

Controller::Result PurePursuit::execute(double deltaTime) {
  if (!path_processing_) {
    return {(reached_goal_ ? Result::Status::GOAL_REACHED
                           : Result::Status::NO_COMMAND_POSSIBLE),
            {0.0, 0.0, 0.0}};
  }

  // Update a speed-based lookahead distance with the "lookahead_distance" as
  // the minimum L = max(min_lookahead, gain * current_velocity)
  double current_v_mag = std::hypot(currentVel.vx(), currentVel.vy());
  double lookahead_val = current_v_mag * lookahead_gain_forward;
  lookahead_val = std::max(lookahead_val, lookahead_distance);

  // Find Lookahead Point
  // Uses Circle-Line intersection to find the point on the path
  // 'lookahead' meters away from the robot.
  Path::Point target_point = findLookaheadPoint(lookahead_val);

  LOG_INFO("Target point", target_point.x(), ", ", target_point.y());
  // Transform Target to Robot Frame
  double dx = target_point.x() - currentState.x;
  double dy = target_point.y() - currentState.y;

  // Angle to target in world frame
  double alpha_world = std::atan2(dy, dx);
  // Angle to target in robot frame (heading error)
  double alpha_robot =
      Angle::normalizeToMinusPiPlusPi(alpha_world - currentState.yaw);

  double dist_to_target = std::hypot(dx, dy);

  double cmd_v = ctrlimitsParams.velXParams.maxVel;

  // Apply exponential speed regulation based on path curvature
  double speed_factor = calculateExponentialSpeedFactor(currentVel.omega());
  LOG_INFO("speed_factor: ", speed_factor);
  cmd_v *= speed_factor;

  Velocity2D cmd;

  if (ctrType == ControlType::OMNI) {
    LOG_INFO("Alpha robot: ", alpha_robot, " cmd_v: ", cmd_v);
    if (std::abs(alpha_robot) > (M_PI * 0.9)) {
      // Use non-omni motion
      double safe_dist =
          std::max(dist_to_target, 0.001); // To avoid division by zero
      double curvature = 2.0 * std::sin(alpha_robot) / safe_dist;

      double omega = cmd_v * curvature;
      cmd = Velocity2D(cmd_v, 0.0, omega);
    } else {
      // Velocity vector in robot frame
      double vx = cmd_v * std::cos(alpha_robot);
      double vy = cmd_v * std::sin(alpha_robot);

      LOG_INFO("vx: ", vx, " vy: ", vy);

      // Align with path heading or face target
      double omega = 2.0 * alpha_robot;

      // If error is very large (near +/- PI), the P-controller is
      // unstable/chatters

      cmd = Velocity2D(vx, vy, omega);
    }

  } else {
    // Standard Pure Pursuit for Diff Drive and ACKERMANN
    // omega = v * curvature = v * (2 * sin(alpha) / L)
    double safe_dist =
        std::max(dist_to_target, 0.001); // To avoid division by zero
    double curvature = 2.0 * std::sin(alpha_robot) / safe_dist;

    double omega = cmd_v * curvature;
    cmd = Velocity2D(cmd_v, 0.0, omega);
  }

  // Limit velocity
  double v_safe = restrictVelocityTolimits(
      currentVel.vx(), cmd.vx(), ctrlimitsParams.velXParams.maxAcceleration,
      ctrlimitsParams.velXParams.maxDeceleration,
      ctrlimitsParams.velXParams.maxVel, deltaTime);

  // Re-scale omega if linear velocity was limited to maintain curvature
  if (std::abs(cmd.vx()) > 1e-4) {
    double ratio = v_safe / cmd.vx();
    cmd.setOmega(cmd.omega() * ratio);
  }
  cmd.setVx(v_safe);

  latest_velocity_command_ = cmd;

  // Check if goal is reached
  Path::Point path_end = currentPath->getEnd();
  double dist_to_end =
      std::hypot(path_end.x() - currentState.x, path_end.y() - currentState.y);
  if (dist_to_end < goal_dist_tolerance) {
    reached_goal_ = true;
    // Return zero command if goal (end of current path) is already reached
    return {Result::Status::GOAL_REACHED, Velocity2D()};
  }

  return {Result::Status::COMMAND_FOUND, cmd};
}

Controller::Result PurePursuit::execute(Path::State currentPosition,
                                        double deltaTime) {
  setCurrentState(currentPosition);
  return execute(deltaTime);
}

Path::Point PurePursuit::findLookaheadPoint(double radius) {
  Path::Point target = currentPath->getEnd();
  bool intersection_found = false;

  // Iterate through path segments starting from the last known position
  for (size_t i = last_found_index_; i < currentPath->getSize() - 1; ++i) {
    Path::Point p1 = currentPath->getIndex(i);
    Path::Point p2 = currentPath->getIndex(i + 1);

    double d_x = p2.x() - p1.x();
    double d_y = p2.y() - p1.y();

    double f_x = p1.x() - currentState.x;
    double f_y = p1.y() - currentState.y;

    // Quadratic equation coefficients: a*t^2 + b*t + c = 0
    double a = d_x * d_x + d_y * d_y;
    double b = 2.0 * (f_x * d_x + f_y * d_y);
    double c = (f_x * f_x + f_y * f_y) - (radius * radius);

    double discriminant = b * b - 4.0 * a * c;

    if (discriminant >= 0.0) {
      // Intersection exists
      discriminant = std::sqrt(discriminant);
      double t1 = (-b - discriminant) / (2.0 * a);
      double t2 = (-b + discriminant) / (2.0 * a);

      // Check if t1 or t2 are within the segment [0, 1]
      // t2 is preferred as it will be further away
      if (t2 >= 0.0 && t2 <= 1.0) {
        target = Path::Point(p1.x() + t2 * d_x, p1.y() + t2 * d_y, 0.0);
        last_found_index_ = i;
        intersection_found = true;
      } else if (t1 >= 0.0 && t1 <= 1.0) {
        target = Path::Point(p1.x() + t1 * d_x, p1.y() + t1 * d_y, 0.0);
        last_found_index_ = i;
        intersection_found = true;
      }
    }
  }

  // If no intersection found (robot too far or path ended inside circle)
  if (!intersection_found) {
    // Check if end is reached
    double dist_to_end = std::hypot(currentPath->getEnd().x() - currentState.x,
                                    currentPath->getEnd().y() - currentState.y);
    if (dist_to_end < radius) {
      last_found_index_ = currentPath->getMaxSize() - 1;
      return currentPath->getEnd();
    } else {
      // If nothing is found increase the lookahead distance
      return findLookaheadPoint(1.1 * radius);
    }
  }

  currentTrackedTarget_->movement = Path::State(target.x(), target.y(), 0.0);

  return target;
}

} // namespace Control
} // namespace Kompass
