#pragma once

#include "datatypes/control.h"
#include "datatypes/parameter.h"
#include "datatypes/path.h"
#include "utils/threadpool.h"

namespace Kompass {
namespace Control {

std::string controlTypeToString(ControlType ctrlType);

class Controller {
public:
  /**
   * @brief Struct to save control results
   *
   */
  struct Result {
    enum class Status { GOAL_REACHED, LOOSING_GOAL, COMMAND_FOUND, NO_COMMAND_POSSIBLE };

    Status status;
    Control::Velocity velocity_command;
  };

  // Nested class for controller parameters
  class ControllerParameters : public Parameters {
  public:
    ControllerParameters() : Parameters() {
      addParameter("enable_reverse_driving",
                   Parameter(true)); // Enable reverse driving if angle to goal
                                     // is larger than pi
      addParameter(
          "enable_check_blocked",
          Parameter(false)); // Enable check for blocked robot (no movement)
      addParameter(
          "max_blocked_duration",
          Parameter(1.0, 0.1,
                    360.0)); // [s] Maximum robot blocking duration until upper
                             // pipeline stages are notified
      addParameter(
          "reverse_slowdown_factor",
          Parameter(0.5, 0.01, 0.99)); // [0,100]% how much to slow down in
                                       // reverse from maximum speed
    }
  };

  // Constructor
  Controller();

  // Destructor
  virtual ~Controller();

  // Set control parameters
  void
  setLinearControlLimits(const Control::LinearVelocityControlParams &vxparams,
                         const Control::LinearVelocityControlParams &vyparams);
  void
  setAngularControlLimits(const Control::AngularVelocityControlParams &params);

  // Set control mode
  void setControlType(const Control::ControlType &controlType);

  // Update current vel
  void setCurrentVelocity(const Control::Velocity &vel);

  // Update current position
  void setCurrentState(const Path::State &position);

  void setCurrentState(double pose_x, double pose_y, double pose_yaw,
                       double speed);

  // Get the current control mode
  Control::ControlType getControlType() const;

  // Get the current control
  Control::Velocity getControl() const;

  // Compote the velocity command to respect given limits on acceleration and
  // deceleration
  double restrictVelocityTolimits(double currentVelocity, double targetVelocity,
                                  double accelerationLimit,
                                  double decelerationLimit, double maxVel,
                                  double timeStep) const;

protected:
  // Protected member variables
  Control::ControlType ctrType;
  Control::ControlLimitsParams ctrlimitsParams;
  Control::Velocity currentVel;
  Path::State currentState;
  Control::Velocity currentCtr;
  int maxNumThreads;

  ControllerParameters config;
  // num threads for parallel processing
};

} // namespace Control
} // namespace Kompass
