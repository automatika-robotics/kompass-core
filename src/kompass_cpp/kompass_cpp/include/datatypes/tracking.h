#pragma once

#include <Eigen/Dense>
#include <vector>

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

  void updateFromNewDetection(const Bbox3D& new_box, const float time_step){
    // Compute velocity and acceleration based on location change
    Eigen::Vector3f new_vel = (new_box.center - this->box.center) / time_step;
    Eigen::Vector3f new_acc = (new_vel - this->vel) / time_step;
    // Update
    setfromBox(new_box);
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
};

}
