#include <algorithm>
#include <cmath>

#include "controllers/stanley.h"
#include "utils/angles.h"
#include "utils/logger.h"

namespace Kompass {
namespace Control {

Stanley::Stanley() : Follower() {
  cross_track_gain = config.getParameter<double>("cross_track_gain");
  heading_gain = config.getParameter<double>("heading_gain");
  min_velocity = config.getParameter<double>("cross_track_min_linear_vel");
  wheel_base = config.getParameter<double>("wheel_base");
}

Stanley::Stanley(StanleyParameters config) {
  this->config = config;
  cross_track_gain = config.getParameter<double>("cross_track_gain");
  heading_gain = config.getParameter<double>("heading_gain");
  min_velocity = config.getParameter<double>("cross_track_min_linear_vel");
  wheel_base = config.getParameter<double>("wheel_base");
}

Controller::Result Stanley::execute(Path::State currentPosition,
                                    double deltaTime) {
  setCurrentState(currentPosition);
  Result result = computeVelocityCommand(deltaTime);
  return result;
}

Controller::Result Stanley::computeVelocityCommand(double timeStep) {

  if (!path_processing_) {
    return {(reached_goal_ ? Result::Status::GOAL_REACHED
                           : Result::Status::NO_COMMAND_POSSIBLE),
            {0.0, 0.0, 0.0}};
  }

  determineTarget();
  const Target target = *currentTrackedTarget_;

  LOG_DEBUG("target: segment_index: ", target.segment_index,
            ", position_in_segment: ", target.position_in_segment,
            ", reverse: ", target.reverse, ", lookahead: ", target.lookahead,
            ", crosstrack_error: ", target.crosstrack_error,
            ", heading_error: ", target.heading_error,
            ", heading gain: ", heading_gain, "\n");

  double target_speed = target.reverse ? -ctrlimitsParams.velXParams.maxVel
                                       : ctrlimitsParams.velXParams.maxVel;

  // Stanley control law
  double control_steering_angle =
     - cross_track_gain *
          std::atan2(target.crosstrack_error,
                     std::max(std::abs(target_speed), min_velocity)) +
          heading_gain * Angle::normalizeToMinusPiPlusPi(target.heading_error);


  current_segment_index_ = target.segment_index;
  current_position_in_segment_ = target.position_in_segment;

  // Compute the new command based on the latest command and the target values
  // for Vx and steering angle
  latest_velocity_command_ = computeCommand(
      latest_velocity_command_, target_speed, control_steering_angle, timeStep);

  LOG_DEBUG("Commmand: {", latest_velocity_command_.vx, ", ",
            latest_velocity_command_.omega, "}\n");

  return {Result::Status::COMMAND_FOUND, latest_velocity_command_};
}

void Stanley::setWheelBase(double length) { robotWheelBase = length; }

Control::Velocity Stanley::computeCommand(Control::Velocity current_velocity,
                                          double linear_velocity,
                                          double steering_angle,
                                          double time_step) const {

  // Restrict the linear velocity command based on the limits
  double linearCtrl =
      restrictVelocityTolimits(current_velocity.vx, linear_velocity,
                               ctrlimitsParams.velXParams.maxAcceleration,
                               ctrlimitsParams.velXParams.maxDeceleration,
                               ctrlimitsParams.velXParams.maxVel, time_step);

  Control::Velocity velocity_command{linearCtrl, 0.0, 0.0, 0.0};
  double max_steering_angle_ = ctrlimitsParams.omegaParams.maxAngle;

  // Limit steering angle to +-max_steering_angle_:
  velocity_command.steer_ang = std::min(
      std::max(steering_angle, -max_steering_angle_), max_steering_angle_);

  // Compute angular velocity from steering angle:
  double omega = std::tan(velocity_command.steer_ang) * std::abs(linearCtrl) /
                 robotWheelBase;

  // Restrict the angular velocity
  velocity_command.omega =
      restrictVelocityTolimits(current_velocity.omega, omega,
                               ctrlimitsParams.omegaParams.maxAcceleration,
                               ctrlimitsParams.omegaParams.maxDeceleration,
                               ctrlimitsParams.omegaParams.maxOmega, time_step);

  return velocity_command;
}

} // namespace Control
} // namespace Kompass
