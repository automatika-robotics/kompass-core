#pragma once

#include <Eigen/Dense>
#include <cmath>
#include <stdexcept>
#include <vector>

// Namespace for Control Types
namespace Kompass {
namespace Control {

// Enumeration for control modes
enum class ControlType { ACKERMANN = 0, DIFFERENTIAL_DRIVE = 1, OMNI = 2 };

class Velocity2D : public Eigen::Vector4f {
public:
  // Default constructor
  Velocity2D() : Eigen::Vector4f(0.0, 0.0, 0.0, 0.0) {}
  Velocity2D(float vx, float vy, float omega, float steer_ang = 0.0)
      : Eigen::Vector4f(vx, vy, omega, steer_ang) {}

  Velocity2D(Eigen::Vector4f &ref) : Eigen::Vector4f(ref) {}

  // Accessors
  float vx() const { return (*this)(0); }
  float vy() const { return (*this)(1); }
  float omega() const { return (*this)(2); }
  float steer_ang() const { return (*this)(3); }

  // Setters
  void setVx(float const value) { (*this)(0) = value; }
  void setVy(float const value) { (*this)(1) = value; }
  void setOmega(float const value) { (*this)(2) = value; }
  void setSteerAng(float const value) { (*this)(3) = value; }
};

struct Velocities {
  std::vector<float> vx; // Speed on x-asix (m/s)
  std::vector<float> vy;
  std::vector<float> omega; // angular velocity (rad/s)
  unsigned int _length;

  Velocities() {
    // Initialize control vectors
    vx = {};
    vy = {};
    omega = {};
    _length = 0;
  };

  Velocities(const int length) {
    // Initialize control vectors
    vx.resize(length, 0.0);
    vy.resize(length, 0.0);
    omega.resize(length, 0.0);
    _length = length;
  };
  void set(int index, float x_velocity, float y_velocity,
           float angular_velocity) {
    if (index >= 0 && index < (int)vx.size()) {
      vx[index] = x_velocity;
      vy[index] = y_velocity;
      omega[index] = angular_velocity;
    } else {
      throw std::out_of_range("Index out of range for Velocities");
    }
  };
  void set(std::vector<float> x_velocity, std::vector<float> y_velocity,
           std::vector<float> angular_velocity) {
    if ((x_velocity.size() == y_velocity.size()) &&
        (y_velocity.size() == angular_velocity.size()) &&
        (angular_velocity.size() == _length)) {
      vx = x_velocity;
      vy = y_velocity;
      omega = angular_velocity;
    } else {
      throw std::length_error(
          "Incompatible vector size to the Velocities length");
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

  ControlLimitsParams(
      LinearVelocityControlParams velXCtrParams = LinearVelocityControlParams(),
      LinearVelocityControlParams velYCtrParams = LinearVelocityControlParams(),
      AngularVelocityControlParams omegaCtrParams =
          AngularVelocityControlParams())
      : velXParams(velXCtrParams), velYParams(velYCtrParams),
        omegaParams(omegaCtrParams) {}
};

/**
 * @brief Struct for LaserScan data
 *
 */
struct LaserScan {
  std::vector<double> ranges;
  std::vector<double> angles;

  LaserScan(std::vector<double> ranges, std::vector<double> angles)
      : ranges(ranges), angles(angles) {}
};

} // namespace Control
} // namespace Kompass
