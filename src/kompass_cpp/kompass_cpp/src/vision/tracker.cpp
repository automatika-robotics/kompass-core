#include "vision/tracker.h"
#include "datatypes/control.h"
#include "datatypes/tracking.h"
#include "utils/logger.h"
#include <cmath>
#include <math.h>

namespace Kompass {

FeatureBasedBboxTracker::FeatureBasedBboxTracker(const float &time_step,
                                                 const float &e_pos,
                                                 const float &e_vel,
                                                 const float &e_acc) {

  timeStep_ = time_step;
  // Setup Kalman filter matrices
  Eigen::MatrixXf A;
  A.resize(StateSize, StateSize);

  A << 1, 0, 0, time_step, 0, 0, 0.5 * pow(time_step, 2), 0, 0, 0, 1, 0, 0,
      time_step, 0, 0, 0.5 * pow(time_step, 2), 0, 0, 0, 1, 0, 0, time_step, 0,
      0, 0.5 * pow(time_step, 2), 0, 0, 0, 1, 0, 0, time_step, 0, 0, 0, 0, 0, 0,
      1, 0, 0, time_step, 0, 0, 0, 0, 0, 0, 1, 0, 0, time_step, 0, 0, 0, 0, 0,
      0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0;

  Eigen::MatrixXf B = Eigen::MatrixXf::Zero(StateSize, 1);
  Eigen::MatrixXf H = Eigen::MatrixXf::Identity(StateSize, StateSize);
  Eigen::MatrixXf Err = Eigen::MatrixXf::Identity(StateSize, StateSize);
  Err(0, 0) *= e_pos;
  Err(1, 1) *= e_pos;
  Err(2, 2) *= e_pos;
  Err(3, 3) *= e_vel;
  Err(4, 4) *= e_vel;
  Err(5, 5) *= e_vel;
  Err(6, 6) *= e_acc;
  Err(7, 7) *= e_acc;
  Err(8, 8) *= e_acc;

  stateKalmanFilter_ = std::make_unique<LinearSSKalmanFilter>(StateSize, 1);
  stateKalmanFilter_->setup(A, B, Err, H, Err);
}

bool FeatureBasedBboxTracker::setInitialTracking(const TrackedBbox3D &bBox) {
  trackedBox_ = std::make_unique<TrackedBbox3D>(bBox);
  trackedLabel_ = bBox.box.label;
  Eigen::VectorXf state_vec;
  state_vec.resize(StateSize);
  state_vec(0) = bBox.box.center[0]; // x
  state_vec(1) = bBox.box.center[1]; // y
  state_vec(2) = bBox.yaw();    // yaw
  state_vec(3) = bBox.vel[0];        // vx
  state_vec(4) = bBox.vel[1];        // vy
  state_vec(5) = bBox.omega();    // omega
  state_vec(6) = bBox.acc[0];        // ax
  state_vec(7) = bBox.acc[1];        // ay
  state_vec(8) = bBox.ang_acc();    // a_yaw
  stateKalmanFilter_->setInitialState(state_vec);
  return true;
}

bool FeatureBasedBboxTracker::setInitialTracking(const Bbox3D &bBox,
                                                 const float yaw) {
  LOG_DEBUG("Setting initial tracked box");
  trackedBox_ = std::make_unique<TrackedBbox3D>(bBox);
  trackedLabel_ = bBox.label;
  Eigen::Matrix<float, StateSize, 1> state_vec =
      Eigen::Matrix<float, StateSize, 1>::Zero();
  state_vec(0) = bBox.center.x();
  state_vec(1) = bBox.center.y();
  state_vec(2) = yaw;
  stateKalmanFilter_->setInitialState(state_vec);
  return true;
}

bool FeatureBasedBboxTracker::setInitialTracking(
    const int &pose_x_img, const int &pose_y_img,
    const std::vector<Bbox3D> &detected_boxes, const float yaw) {
  std::unique_ptr<Bbox3D> target_box;
  // Find a detected box containing the point
  for (auto box : detected_boxes) {
    auto limits_x = box.getXLimitsImg();
    if (pose_x_img >= limits_x(0) and pose_x_img <= limits_x(1)) {
      auto limits_y = box.getYLimitsImg();
      if (pose_y_img >= limits_y(0) and pose_y_img <= limits_y(1)) {
        target_box = std::make_unique<Bbox3D>(box);
        break;
      }
    }
  }
  if (!target_box) {
    // given position was not found inside any detected box
    return false;
  }
  return setInitialTracking(*target_box, yaw);
}

bool FeatureBasedBboxTracker::trackerInitialized() const {
  if (trackedBox_) {
    return true;
  }
  return false;
}

void FeatureBasedBboxTracker::updateTrackedBoxState(const int numberSteps) {
  Eigen::MatrixXf measurement;
  measurement.resize(StateSize, 1);
  measurement(0) = trackedBox_->box.center.x();
  measurement(1) = trackedBox_->box.center.y();
  measurement(2) = trackedBox_->yaw();
  measurement(3) = trackedBox_->vel.x();
  measurement(4) = trackedBox_->vel.y();
  measurement(5) = trackedBox_->omega();
  measurement(6) = trackedBox_->acc.x();
  measurement(7) = trackedBox_->acc.y();
  measurement(8) = trackedBox_->ang_acc();
  stateKalmanFilter_->estimate(measurement, numberSteps);
}

bool FeatureBasedBboxTracker::updateTracking(
    const std::vector<Bbox3D> &detected_boxes) {
  std::vector<Bbox3D> label_boxes;
  for(auto box : detected_boxes) {
    if(box.label == trackedLabel_) {
      label_boxes.push_back(box);
    }
  }
  if (label_boxes.empty()) {
    LOG_DEBUG("No boxes with label ", trackedLabel_, " found in the detected boxes!");
    return false;
  }

  float max_similarity_score = 0.0f; // Similarity score
  Bbox3D * found_box;
  float dt = label_boxes[0].timestamp - trackedBox_->box.timestamp;

  if(label_boxes.size() == 1) {
    // Only one box detected, so it is the same
    max_similarity_score = 1.0f;
    found_box = &label_boxes[0];
  }
  else{
    // Compute similarity score and find box
    // Predicted the new location of the tracked box
    auto predicted_tracked_box = trackedBox_->predictConstantAcc(dt);

    auto ref_box_features = extractFeatures(predicted_tracked_box);

    // Get the features of all the new detections
    FeaturesVector detected_boxes_feature_vec;
    size_t similar_box_idx = 0, count = 0;

    for (auto box : label_boxes) {
      detected_boxes_feature_vec = extractFeatures(box);
      FeaturesVector error_vec = detected_boxes_feature_vec - ref_box_features;
      // Error vector normalization
      for (int i = 0; i < error_vec.size(); ++i) {
        if (std::abs(ref_box_features(i)) > 0.0) {
          error_vec(i) = error_vec(i) / std::abs(ref_box_features(i));
        }
      }
      float similarity_score = std::exp(-std::pow(error_vec.norm(), 2));

      if (similarity_score > max_similarity_score) {
        max_similarity_score = similarity_score;
        similar_box_idx = count;
      }
      count++;
    }
    found_box = &label_boxes[similar_box_idx];
  }

  LOG_DEBUG("Max similarity score = ", max_similarity_score);
  if (max_similarity_score > minAcceptedSimilarityScore_) {
    // Update raw tracking
    float dt = found_box->timestamp - trackedBox_->box.timestamp;
    int number_steps = std::max(static_cast<int>(dt / timeStep_), 1);

    trackedBox_->updateFromNewDetection(*found_box);

    // Update state by applying kalman filter on the raw measurement
    updateTrackedBoxState(number_steps);
    LOG_DEBUG("Update after ", number_steps,
              " timer steps, Box Now updated to: ", trackedBox_->box.center.x(),
              ", ", trackedBox_->box.center.y(), ", v=", trackedBox_->v());
    return true;
  }
  LOG_DEBUG("Box not found in the detected boxes! Max similarity score = ",
            max_similarity_score, ", min accepted = ",
            minAcceptedSimilarityScore_);
  return false;
}

FeatureBasedBboxTracker::FeaturesVector
FeatureBasedBboxTracker::extractFeatures(
    const TrackedBbox3D &bBox_tracked) const {
  return extractFeatures(bBox_tracked.box);
}

FeatureBasedBboxTracker::FeaturesVector
FeatureBasedBboxTracker::extractFeatures(const Bbox3D &bBox) const {
  FeatureBasedBboxTracker::FeaturesVector features_vec =
      Eigen::Vector<float, 9>::Zero();
  features_vec.segment<2>(0) = bBox.center.segment<2>(0); // indices 0, 1
  features_vec.segment<3>(2) = bBox.size;                 // indices 2, 3, 4
  features_vec(5) = bBox.pc_points.size();
  if (features_vec(5) > 0.0) {
    // Compute point cloud points standard deviation
    auto std_vec = this->computePointsStdDev(bBox.pc_points);
    features_vec.segment<3>(6) = std_vec; // indices 6, 7, 8
  }
  return features_vec;
}

std::optional<TrackedBbox3D> FeatureBasedBboxTracker::getRawTracking() const {
  if (trackedBox_) {
    return *trackedBox_;
  }
  return std::nullopt;
}

std::optional<Eigen::MatrixXf>
FeatureBasedBboxTracker::getTrackedState() const {
  if (trackedBox_) {
    return stateKalmanFilter_->getState();
  }
  return std::nullopt;
}

std::optional<Control::TrackedPose2D>
FeatureBasedBboxTracker::getFilteredTrackedPose2D() const {
  if (trackedBox_) {
    auto state_vec = stateKalmanFilter_->getState().value();
    return Control::TrackedPose2D(state_vec(0), state_vec(1), state_vec(2),
                                  state_vec(3), state_vec(4), state_vec(5));
  }
  return std::nullopt;
}

Eigen::Vector3f FeatureBasedBboxTracker::computePointsStdDev(
    const std::vector<Eigen::Vector3f> &pc_points) const {
  // compute the mean in each direction
  auto size = std::max(int(pc_points.size() - 1), 1);
  Eigen::Vector3f mean = Eigen::Vector3f::Zero();
  for (auto point : pc_points) {
    mean += point;
  }
  mean /= size;
  // Compute the variance
  Eigen::Vector3f variance = Eigen::Vector3f::Zero();
  for (auto point : pc_points) {
    auto diff = point - mean;
    variance += diff.cwiseProduct(diff);
  }
  variance /= size;
  // return the standard deviation
  return variance.array().sqrt();
}

} // namespace Kompass
