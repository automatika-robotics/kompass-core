#pragma once

#include "controllers/controller.h"
#include "datatypes/control.h"
#include "datatypes/parameter.h"
#include "datatypes/path.h"
#include "follower.h"

namespace Kompass {
namespace Control {

class Stanley : public Follower {
public:
  // parameters
  class StanleyParameters : public Follower::FollowerParameters {
  public:
    StanleyParameters() : Follower::FollowerParameters() {
      addParameter(
          "wheel_base",
          Parameter(0.3, 0.0001, 100.0)); // Robot wheel base length [m]
      addParameter(
          "heading_gain",
          Parameter(
              1.0, 0.0,
              10.0)); // Control gain to correct heading error; should be 1
      addParameter(
          "cross_track_min_linear_vel",
          Parameter(
              0.05, 0.0,
              10.0)); // [m/s] Lower limit for cross track error computation
      addParameter(
          "cross_track_gain",
          Parameter(10.0, 0.0,
                    50.0)); // Proportional gain for reaching target speed
    }
  };

  // Constructors
  Stanley();
  Stanley(StanleyParameters config);

  // Destructor
  ~Stanley() = default;

  /**
   * @brief Computes a new Stanley velocity control to follow the current path
   *
   * @param timeStep  Control tiem step [seconds]
   * @return Controller::Result
   */
  Controller::Result computeVelocityCommand(double timeStep);

  /**
   * @brief Perform follower action by first setting the current position then
   * computing the velocity control
   *
   * @param currentPosition
   * @param deltaTime
   * @return Controller::Result
   */
  Controller::Result execute(Path::State currentPosition, double deltaTime);

  /**
   * @brief Set the robot Wheel Base
   *
   * @param length  [meters]
   */
  void setWheelBase(double length);

protected:
  StanleyParameters config;
  double robotWheelBase{1.0};
  double cross_track_gain{0.0};
  double heading_gain{0.0};
  double min_velocity{0.0};
  double wheel_base{0.0};
  double determined_control_gain{0.0};

  /**
   * @brief Comptes the velcity control given computed Stanley control values to
   * respect robot control limits
   *
   * @param current_velocity
   * @param linear_velocity
   * @param steering_angle
   * @param time_step
   * @return Control::Velocity
   */
  Control::Velocity computeCommand(Control::Velocity current_velocity,
                                   double linear_velocity,
                                   double steering_angle,
                                   double time_step) const;
};

} // namespace Control
} // namespace Kompass
