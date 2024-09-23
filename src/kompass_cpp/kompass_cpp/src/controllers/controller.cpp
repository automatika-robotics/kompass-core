#include "controllers/controller.h"

namespace Kompass {
namespace Control {

std::string controlTypeToString(ControlType ctrlType) {
      switch (ctrlType) {
          case ControlType::ACKERMANN:   return "ACKERMANN";
          case ControlType::DIFFERENTIAL_DRIVE: return "DIFFERENTIAL_DRIVE";
          case ControlType::OMNI:  return "OMNI";
          default:           return "Unknown";
      }
}

Controller::Controller() : ctrType(), ctrlimitsParams() {
  // default config is set in the constructor
}

Controller::~Controller() {}

void Controller::setLinearControlLimits(
    const Control::LinearVelocityControlParams &vxparams,
    const Control::LinearVelocityControlParams &vyparams) {
  this->ctrlimitsParams.velXParams = vxparams;
  this->ctrlimitsParams.velYParams = vyparams;
}

void Controller::setAngularControlLimits(
    const Control::AngularVelocityControlParams &params) {
  this->ctrlimitsParams.omegaParams = params;
}

void Controller::setControlType(const Control::ControlType &controlType) {
  this->ctrType = controlType;
}

void Controller::setCurrentVelocity(const Control::Velocity &vel) {
  this->currentVel = vel;
}

void Controller::setCurrentState(const Path::State &position) {
  this->currentState = position;
}

void Controller::setCurrentState(double pose_x, double pose_y, double pose_yaw,
                                 double speed) {
  this->currentState.x = pose_x;
  this->currentState.y = pose_y;
  this->currentState.yaw = pose_yaw;
  this->currentState.speed = speed;
}

Control::ControlType Controller::getControlType() const { return ctrType; }

Control::Velocity Controller::getControl() const { return currentCtr; }

// Function to compute the velocity command
double Controller::restrictVelocityTolimits(
    double currentVelocity, double targetVelocity, double accelerationLimit,
    double decelerationLimit, double maxVel, double timeStep) const {
  // Output command
  double velocityCommand = currentVelocity;

  if (currentVelocity < targetVelocity) {
    // Need to accelerate
    velocityCommand += accelerationLimit * timeStep;
    if (velocityCommand > targetVelocity) {
      velocityCommand = targetVelocity;
    }
  } else if (currentVelocity > targetVelocity) {
    // Need to decelerate
    velocityCommand -= decelerationLimit * timeStep;
    if (velocityCommand < targetVelocity) {
      velocityCommand = targetVelocity;
    }
  }

  // Respect the maximum absolute velocity
  if (std::abs(velocityCommand) > maxVel) {
    if (velocityCommand > 0) {
      velocityCommand = maxVel;
    } else {
      velocityCommand = -maxVel;
    }
  }

  return velocityCommand;
}

} // namespace Control
} // namespace Kompass
