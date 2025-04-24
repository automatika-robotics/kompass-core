#pragma once

#include <Eigen/Dense>
#include <cmath>
#include <vector>
#include "datatypes/control.h"

namespace Kompass {

struct Bbox3D {
  Eigen::Vector3f center = {0.0, 0.0, 0.0};
  Eigen::Vector3f size = {0.0, 0.0, 0.0};
  Eigen::Vector2i center_img_frame = {0, 0};
  Eigen::Vector2i size_img_frame = {0, 0};
  std::vector<Eigen::Vector3f> pc_points = {};

  Bbox3D() {};

  Bbox3D(const Bbox3D &box)
      : center(box.center), size(box.size), center_img_frame(box.center_img_frame), size_img_frame(box.size_img_frame), pc_points(box.pc_points){};

  Eigen::Vector2f getXLimitsImg() const {
    return {
      center_img_frame.x() - (size_img_frame.x() / 2),
          center_img_frame.x() + (size_img_frame.x() / 2)
    };
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
  Eigen::Vector2f yaw_yaw_diff = {0.0, 0.0};

  TrackedBbox3D(const Bbox3D& box): box(box) {};

  void setState(const Eigen::Matrix<float, 6, 1>& state_vector){
    this->box.center = {state_vector(0), state_vector(1), 0.0f};
    this->vel = {state_vector(2), state_vector(3), 0.0f};
    this->acc = {state_vector(4), state_vector(5), 0.0f};
  };

  void setSize(const Eigen::Vector3f& size) {
    this->box.size = size;
  };

  void setfromBox(const Bbox3D& box){
    this->box = box;
  };

  void updateFromNewState(const Eigen::Matrix<float, 6, 1> &state_vector,
                          const float time_step) {
    this->box.center = {state_vector(0), state_vector(1), 0.0f};
    this->vel = {state_vector(2), state_vector(3), 0.0f};
    this->acc = {state_vector(4), state_vector(5), 0.0f};
    auto new_yaw = std::atan2(this->vel(1), this->vel(0));
    // Orientation difference
    this->yaw_yaw_diff(1) = (this->yaw_yaw_diff(0) - new_yaw) / time_step;
    this->yaw_yaw_diff(0) = new_yaw;
  };

  void updateFromNewDetection(const Bbox3D& new_box, const float time_step){
    // Compute velocity and acceleration based on location change
    Eigen::Vector3f new_vel = (new_box.center - this->box.center) / time_step;
    Eigen::Vector3f new_acc = (new_vel - this->vel) / time_step;
    // Update
    setfromBox(new_box);
    // New orientation (yaw) based on new velocity
    auto new_yaw = std::atan2(new_vel(1), new_vel(0));
    // Orientation difference
    this->yaw_yaw_diff(1) = (this->yaw_yaw_diff(0) - new_yaw) / time_step;
    this->yaw_yaw_diff(0) = new_yaw;
    this->vel = new_vel;
    this->acc = new_acc;
  }

  TrackedBbox3D predictConstantVel(const float &dt) {
    auto predicted_tracking = *this;
    predicted_tracking.box.center += predicted_tracking.vel * dt;
    // Set acceleration to zero for constant velocity prediction
    predicted_tracking.acc = {0.0, 0.0, 0.0};
    return predicted_tracking;
  };

  TrackedBbox3D predictConstantAcc(const float &dt) {
    auto predicted_tracking = *this;
    predicted_tracking.vel += this->acc * dt;
    predicted_tracking.box.center += predicted_tracking.vel * dt;
    return predicted_tracking;
  };

  float v() const { return Eigen::Vector2f{vel.x(), vel.y()}.norm(); };

  float x() const { return box.center.x(); };

  float y() const { return box.center.y(); };

  float yaw() const { return this->yaw_yaw_diff(0); };

  float omega() const { return yaw_yaw_diff(1); };

  void update(const float timeStep) {
    box.center(0) += vel.x() * timeStep;
    box.center(1) += vel.y() * timeStep;
    this->yaw_yaw_diff(0) += this->yaw_yaw_diff(1) * timeStep;
  }

  float distance(const float x, const float y, const float z = 0.0) const {
    return sqrt(pow(box.center.x() - x, 2) + pow(box.center.y() - y, 2) +
                pow(box.center.z() - z, 2));
  }

  Control::TrackedPose2D getTrackedPose() const {
    return Control::TrackedPose2D(box.center.x(), box.center.y(),
           yaw(), vel.x(), vel.y(), omega());
  }
};

}
