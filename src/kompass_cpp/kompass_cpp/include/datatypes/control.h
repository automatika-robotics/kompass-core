#pragma once

#include <cmath>
#include <stdexcept>
#include <vector>

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

struct Velocities {
  std::vector<double> vx; // Speed on x-asix (m/s)
  std::vector<double> vy;
  std::vector<double> omega; // angular velocity (rad/s)
  int _length;

  Velocities() {
    // Initialize control vectors
    vx = {};
    vy= {};
    omega = {};
    _length = 0;
  };

  Velocities(const int length){
    // Initialize control vectors
    vx.resize(length, 0.0);
    vy.resize(length, 0.0);
    omega.resize(length, 0.0);
    _length = length;
  };
  void set(int index, double x_velocity, double y_velocity,
           double angular_velocity) {
    if (index >= 0 && index < vx.size()) {
      vx[index] = x_velocity;
      vy[index] = y_velocity;
      omega[index] = angular_velocity;
    } else {
      throw std::out_of_range("Index out of range for Velocities");
    }
  };
  void set(std::vector<double> x_velocity, std::vector<double> y_velocity,
           std::vector<double> angular_velocity) {
    if (x_velocity.size() == y_velocity.size() == angular_velocity.size() == _length) {
      vx = x_velocity;
      vy = y_velocity;
      omega = angular_velocity;
    } else {
      throw std::length_error("Incompatible vector size to the Velocities length");
    }
  }
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
