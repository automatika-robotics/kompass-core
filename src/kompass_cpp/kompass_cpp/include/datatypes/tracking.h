#pragma once

#include "datatypes/control.h"
#include "utils/logger.h"
#include <Eigen/Dense>
#include <cmath>
#include <vector>

namespace Kompass {

struct Bbox2D {
  Eigen::Vector2i top_corner = {0, 0};
  Eigen::Vector2i size = {0, 0};
  float timestamp = 0.0; // Timestamp of the detection in seconds

  Bbox2D(){};

  Bbox2D(const Bbox2D &box) : top_corner(box.top_corner), size(box.size){};

  Bbox2D(const Eigen::Vector2i top_corner, Eigen::Vector2i size)
      : top_corner(top_corner), size(size){};

  Eigen::Vector2i getXLimits() const {
    return {top_corner.x(), top_corner.x() + size.x()};
  };

  Eigen::Vector2i getYLimits() const {
    return {top_corner.y(), top_corner.y() + size.y()};
  };
};

struct Bbox3D {
  Eigen::Vector3f center = {0.0, 0.0, 0.0};
  Eigen::Vector3f size = {0.0, 0.0, 0.0};
  Eigen::Vector2i center_img_frame = {0, 0};
  Eigen::Vector2i size_img_frame = {0, 0};
  std::vector<Eigen::Vector3f> pc_points = {};
  float timestamp = 0.0; // Timestamp of the detection in seconds

  Bbox3D(){};

  Bbox3D(const Bbox3D &box)
      : center(box.center), size(box.size),
        center_img_frame(box.center_img_frame),
        size_img_frame(box.size_img_frame), pc_points(box.pc_points),
        timestamp(box.timestamp){};

  Bbox3D(const Eigen::Vector3f &center, const Eigen::Vector3f &size,
         const Eigen::Vector2i center_img_frame,
         const Eigen::Vector2i size_img_frame, const float timestamp = 0.0,
         std::vector<Eigen::Vector3f> pc_points = {})
      : center(center), size(size), center_img_frame(center_img_frame),
        size_img_frame(size_img_frame), pc_points(pc_points),
        timestamp(timestamp){};

  Bbox3D(const Eigen::Vector3f &center, const Eigen::Vector3f &size,
         const Bbox2D &box2d, std::vector<Eigen::Vector3f> pc_points = {})
      : center(center), size(size),
        center_img_frame(
            box2d.top_corner +
            Eigen::Vector2i{box2d.size.x() / 2, box2d.size.y() / 2}),
        size_img_frame(box2d.size), pc_points(pc_points),
        timestamp(box2d.timestamp){};

  Bbox3D(const Bbox2D &box2d)
      : center_img_frame(
            box2d.top_corner +
            Eigen::Vector2i{box2d.size.x() / 2, box2d.size.y() / 2}),
        size_img_frame(box2d.size), timestamp(box2d.timestamp){};

  Eigen::Vector2f getXLimitsImg() const {
    return {center_img_frame.x() - (size_img_frame.x() / 2),
            center_img_frame.x() + (size_img_frame.x() / 2)};
  };

  Eigen::Vector2f getYLimitsImg() const {
    return {center_img_frame.y() - (size_img_frame.y() / 2),
            center_img_frame.y() + (size_img_frame.y() / 2)};
  };
};

struct TrackedBbox3D {
  Bbox3D box;
  Eigen::Vector3f vel = {0.0, 0.0, 0.0};
  Eigen::Vector3f acc = {0.0, 0.0, 0.0};
  int unique_id = 0;
  Eigen::Vector3f yaw_vec = {0.0, 0.0,
                             0.0}; // To track yaw, omega and angular acc

  TrackedBbox3D(const Bbox3D &box) : box(box){};

  void setState(const Eigen::Matrix<float, 9, 1> &state_vector) {
    this->box.center = {state_vector(0), state_vector(1), 0.0f};
    this->yaw_vec = {state_vector(2), state_vector(5), state_vector(8)};
    this->vel = {state_vector(3), state_vector(4), 0.0f};
    this->acc = {state_vector(6), state_vector(7), 0.0f};
  };

  void setSize(const Eigen::Vector3f &size) { this->box.size = size; };

  void setfromBox(const Bbox3D &box) { this->box = box; };

  void updateFromNewDetection(const Bbox3D &new_box) {
    float time_step = new_box.timestamp - this->box.timestamp;
    if (time_step <= 0.0) {
      LOG_ERROR("Cannot update from new detection, invalid time step = ",
                time_step);
      return; // Invalid time step
    }
    // Compute velocity and acceleration based on location change
    Eigen::Vector3f new_vel = (new_box.center - this->box.center) / time_step;
    this->acc = (new_vel - this->vel) / time_step;
    this->vel = new_vel;
    // Update
    setfromBox(new_box);
    // New orientation (yaw) based on new velocity
    auto new_yaw = std::atan2(new_vel(1), new_vel(0));
    // Orientation difference
    auto new_omega = (new_yaw - this->yaw_vec(0)) / time_step;
    this->yaw_vec(2) = (new_omega - this->yaw_vec(1)) / time_step;
    this->yaw_vec(1) = new_omega;
    this->yaw_vec(0) = new_yaw;
  }

  TrackedBbox3D predictConstantVel(const float &dt) {
    auto predicted_tracking = *this;
    predicted_tracking.box.center += predicted_tracking.vel * dt;
    predicted_tracking.yaw_vec(0) += predicted_tracking.yaw_vec(1) * dt;
    predicted_tracking.yaw_vec(2) = 0.0; // Set angular acceleration to zero
    // Set acceleration to zero for constant velocity prediction
    predicted_tracking.acc = {0.0, 0.0, 0.0};
    predicted_tracking.box.timestamp += dt;
    return predicted_tracking;
  };

  TrackedBbox3D predictConstantAcc(const float &dt) {
    auto predicted_tracking = *this;
    predicted_tracking.vel += this->acc * dt;
    predicted_tracking.box.center += predicted_tracking.vel * dt;
    predicted_tracking.yaw_vec(1) += predicted_tracking.yaw_vec(2) * dt;
    predicted_tracking.yaw_vec(0) += predicted_tracking.yaw_vec(1) * dt;
    predicted_tracking.box.timestamp += dt;
    return predicted_tracking;
  };

  float v() const { return Eigen::Vector2f{vel.x(), vel.y()}.norm(); };

  float x() const { return box.center.x(); };

  float y() const { return box.center.y(); };

  float yaw() const { return this->yaw_vec(0); };

  float omega() const { return yaw_vec(1); };

  float timestamp() const { return box.timestamp; };

  void update(const float timeStep) {
    box.center(0) += vel.x() * timeStep;
    box.center(1) += vel.y() * timeStep;
    this->yaw_vec(0) += this->yaw_vec(1) * timeStep;
  }

  float distance(const float x, const float y, const float z = 0.0) const {
    return sqrt(pow(box.center.x() - x, 2) + pow(box.center.y() - y, 2) +
                pow(box.center.z() - z, 2));
  }

  Control::TrackedPose2D getTrackedPose() const {
    return Control::TrackedPose2D(box.center.x(), box.center.y(), yaw(),
                                  vel.x(), vel.y(), omega());
  }
};

} // namespace Kompass
