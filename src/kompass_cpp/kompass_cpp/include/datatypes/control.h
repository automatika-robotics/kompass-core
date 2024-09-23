#pragma once

#include <cmath>

// Namespace for Control Types
namespace Kompass {
namespace Control {

// Enumeration for control modes
enum class ControlType { ACKERMANN = 0, DIFFERENTIAL_DRIVE = 1, OMNI = 2};

// Structure for Velocity control
struct Velocity {
  double vx; // Speed on x-asix (m/s)
  double vy;
  double omega; // angular velocity (rad/s)
  double steer_ang;

  Velocity(double velx = 0.0, double vely = 0.0, double velomega = 0.0,
           double steerVal = 0.0)
      : vx(velx), vy(vely), omega(velomega), steer_ang(steerVal) {}
};

// Structure for Forward Linear Velocity Control parameters
struct LinearVelocityControlParams {
  double maxVel;          // Maximum allowed speed
  double maxAcceleration; // Maximum acceleration in units per second squared
  double maxDeceleration; // Maximum deceleration in units per second squared

  LinearVelocityControlParams(double maxVel = 1.0, double maxAcc = 10.0,
                               double maxDec = 10.0)
      : maxVel(maxVel), maxAcceleration(maxAcc), maxDeceleration(maxDec) {}
};

// Structure for Forward Linear Velocity Control parameters
struct AngularVelocityControlParams {
  double maxAngle; // Maximum allowed steering angle
  double maxOmega;
  double maxAcceleration; // Maximum acceleration in units per second squared
  double maxDeceleration; // Maximum deceleration in units per second squared

  AngularVelocityControlParams(double maxAng = M_PI, double maxOmg = 1.0,
                               double maxAcc = 10.0, double maxDec = 10.0)
      : maxAngle(maxAng), maxOmega(maxOmg), maxAcceleration(maxAcc),
        maxDeceleration(maxDec) {}
};

// General Control parameters
struct ControlLimitsParams {
  LinearVelocityControlParams velXParams;
  LinearVelocityControlParams velYParams;
  AngularVelocityControlParams omegaParams;

  ControlLimitsParams(LinearVelocityControlParams velXCtrParams =
                          LinearVelocityControlParams(),
                      LinearVelocityControlParams velYCtrParams =
                          LinearVelocityControlParams(),
                      AngularVelocityControlParams omegaCtrParams =
                          AngularVelocityControlParams())
      : velXParams(velXCtrParams), velYParams(velYCtrParams),
        omegaParams(omegaCtrParams) {}
};

} // namespace Control
} // namespace Kompass
