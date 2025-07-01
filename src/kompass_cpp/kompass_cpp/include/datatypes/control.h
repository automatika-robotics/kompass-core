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

class Pose3D {

public:
  Pose3D(const Eigen::Vector3f &position, const Eigen::Vector4f &orientation)
      : position_(position), orientation_(orientation),
        rotation_matrix_(orientation_.toRotationMatrix()) {};

  Pose3D(const Eigen::Vector3f &position, const Eigen::Quaternionf &orientation)
      : position_(position), orientation_(orientation),
        rotation_matrix_(orientation.toRotationMatrix()) {};

  /**
   * @brief Construct a new Pose3D object using 2D pose information
   *
   * @param pose_x
   * @param pose_y
   * @param pose_yaw
   */
  Pose3D(const float &pose_x, const float &pose_y, const float &pose_yaw) {
    update(pose_x, pose_y, pose_yaw);
  };

  void setFrame(const std::string &frame_id) { frame_id_ = frame_id; }

  std::string getFrame() const { return frame_id_; }

  bool inFrame(const std::string &frame_id) const {
    return frame_id == frame_id_;
  }

  float norm() const { return position_.norm(); };

  /**
   * @brief Extract x coordinates
   *
   * @return float
   */
  float x() const { return position_(0); };

  /**
   * @brief Extract y coordinates
   *
   * @return float
   */
  float y() const { return position_(1); };

  /**
   * @brief Extract z coordinates
   *
   * @return float
   */
  float z() const { return position_(2); };

  /**
   * @brief Extract pitch (y-axis rotation) from the rotation matrix
   *
   * @return float
   */
  float pitch() const { return std::asin(rotation_matrix_(2, 0)); };

  /**
   * @brief Extract roll (x-axis rotation) from the rotation matrix
   *
   * @return float
   */
  float roll() const {
    return std::atan2(rotation_matrix_(2, 1), rotation_matrix_(2, 2));
  };

  /**
   * @brief Extract yaw (x-axis rotation) from the rotation matrix
   *
   * @return float
   */
  float yaw() const {
    return std::atan2(rotation_matrix_(1, 0), rotation_matrix_(0, 0));
  };

  void update(const float &pose_x, const float &pose_y, const float &pose_yaw) {
    position_ = {pose_x, pose_y, 0.0};
    setRotation(0.0, 0.0, pose_yaw);
  }

  void setRotation(const float pitch, const float roll, const float yaw) {
    Eigen::AngleAxisf rotZ(yaw, Eigen::Vector3f::UnitZ());
    Eigen::AngleAxisf rotY(pitch, Eigen::Vector3f::UnitY());
    Eigen::AngleAxisf rotX(roll, Eigen::Vector3f::UnitX());
    orientation_ = rotZ * rotY * rotX;
    rotation_matrix_ = orientation_.toRotationMatrix();
  }

protected:
  Eigen::Vector3f position_;
  Eigen::Quaternionf orientation_;
  Eigen::Matrix3f rotation_matrix_;
  std::string frame_id_;
};

class Velocity2D {
public:
  // Default constructor
  Velocity2D() = default;
  Velocity2D(double vx, double vy, double omega, double steer_ang = 0.0)
      : velocity_(vx, vy, omega, steer_ang) {}

  Velocity2D(Eigen::Vector4d &ref) : velocity_(ref) {}

  // Accessors
  double vx() const { return velocity_(0); }
  double vy() const { return velocity_(1); }
  double omega() const { return velocity_(2); }
  double steer_ang() const { return velocity_(3); }

  // Setters
  void setVx(double const value) { velocity_(0) = value; }
  void setVy(double const value) { velocity_(1) = value; }
  void setOmega(double const value) { velocity_(2) = value; }
  void setSteerAng(double const value) { velocity_(3) = value; }

  // Overload the unary minus operator
  Velocity2D operator-() const {
    return Velocity2D(-velocity_(0), -velocity_(1), -velocity_(2));
  }

private:
  Eigen::Vector4d velocity_{0.0, 0.0, 0.0, 0.0};
};

class TrackedPose2D : public Pose3D {

public:
  TrackedPose2D(const Eigen::Vector3f &position,
                const Eigen::Vector4f &orientation, const Velocity2D &vel)
      : Pose3D(position, orientation), vel_(vel) {};

  TrackedPose2D(const Eigen::Vector3f &position,
                const Eigen::Quaternionf &orientation, const Velocity2D &vel)
      : Pose3D(position, orientation), vel_(vel) {};

  TrackedPose2D(const float &pose_x, const float &pose_y, const float &pose_yaw,
                const Velocity2D &vel)
      : Pose3D(pose_x, pose_y, pose_yaw), vel_(vel) {};

  TrackedPose2D(const float &pose_x, const float &pose_y, const float &pose_yaw,
                const float &vx, const float &vy, const float &omega)
      : Pose3D(pose_x, pose_y, pose_yaw), vel_(vx, vy, omega) {};

  float v() const { return Eigen::Vector2f{vel_.vx(), vel_.vy()}.norm(); };

  float omega() const { return vel_.omega(); };

  void update(const float timeStep) {
    position_(0) +=
        (vel_.vx() * cos(this->yaw()) - vel_.vy() * sin(this->yaw())) *
        timeStep;
    position_(1) +=
        (vel_.vx() * sin(this->yaw()) + vel_.vy() * cos(this->yaw())) *
        timeStep;
    float yaw = this->yaw() + vel_.omega() * timeStep;
    setRotation(0.0, 0.0, yaw);
  }

  void update(const Velocity2D &vel, const float timeStep) {
    vel_ = vel;
    update(timeStep);
  }

  float distance(const float x, const float y, const float z = 0.0) const {
    return sqrt(pow(position_.x() - x, 2) + pow(position_.y() - y, 2) +
                pow(position_.z() - z, 2));
  }

protected:
  Velocity2D vel_;
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
  double maxVel = 1.0;
  double maxAcceleration = 10.0;
  double maxDeceleration = 10.0;

  LinearVelocityControlParams(const LinearVelocityControlParams &) = default;

  // Parameterized constructor
  LinearVelocityControlParams(double maxVel = 1.0, double maxAcc = 10.0,
                              double maxDec = 10.0)
      : maxVel(maxVel), maxAcceleration(maxAcc), maxDeceleration(maxDec) {}
};

// Structure for Angular Velocity Control parameters
struct AngularVelocityControlParams {
  double maxAngle = M_PI;
  double maxOmega = 1.0;
  double maxAcceleration = 10.0;
  double maxDeceleration = 10.0;

  AngularVelocityControlParams(const AngularVelocityControlParams &) = default;

  // Parameterized constructor
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

  ControlLimitsParams() = default;
  ControlLimitsParams(const ControlLimitsParams &) = default;

  // Parameterized constructor
  ControlLimitsParams(const LinearVelocityControlParams &velXCtrParams,
                      const LinearVelocityControlParams &velYCtrParams,
                      const AngularVelocityControlParams &omegaCtrParams)
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
