#include "vision/tracker.h"
#include "datatypes/tracking.h"
#include <math.h>

namespace Kompass {

FeatureBasedBboxTracker::FeatureBasedBboxTracker(
    const float& time_step, const float& e_pos,
    const float& e_vel, const float& e_acc) {

  timeStep_ = time_step;
  // Setup Kalman filter matrices
  Eigen::MatrixXf A;
  A.resize(6, 6);

  A << 1, 0, time_step, 0, 0.5 * pow(time_step, 2), 0, 0, 1, 0, time_step, 0,
      0.5 * pow(time_step, 2), 0, 0, 1, 0, time_step, 0, 0, 0, 0, 1, 0,
      time_step, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1;

  Eigen::MatrixXf B = Eigen::MatrixXf::Zero(6, 1);
  Eigen::MatrixXf H = Eigen::MatrixXf::Identity(6, 6);
  Eigen::MatrixXf Err = Eigen::MatrixXf::Identity(6, 6);
  Err(0, 0) *= e_pos;
  Err(1, 1) *= e_pos;
  Err(2, 2) *= e_vel;
  Err(3, 3) *= e_vel;
  Err(4, 4) *= e_acc;
  Err(5, 5) *= e_acc;

  stateKalmanFilter_ = std::make_unique<LinearSSKalmanFilter>(6, 1);
  stateKalmanFilter_->setup(A, B, Err, H, Err);
}

bool FeatureBasedBboxTracker::setInitialTracking(const TrackedBbox3D &bBox) {
  trackedBox_ = std::make_unique<TrackedBbox3D>(bBox);
  Eigen::VectorXf state_vec;
  state_vec.resize(6);
  state_vec(0) = bBox.box.center[0];
  state_vec(1) = bBox.box.center[1];
  state_vec(2) = bBox.vel[0];
  state_vec(3) = bBox.vel[1];
  state_vec(4) = bBox.acc[0];
  state_vec(5) = bBox.acc[1];
  stateKalmanFilter_->setInitialState(state_vec);
  this->tracking_started_ = true;
  return true;
}

bool FeatureBasedBboxTracker::setInitialTracking(const Bbox3D &bBox) {
  trackedBox_ = std::make_unique<TrackedBbox3D>(bBox);
  Eigen::Matrix<float, 6, 1> state_vec = Eigen::Matrix<float, 6, 1>::Zero();
  state_vec(0) = bBox.center[0];
  state_vec(1) = bBox.center[1];
  stateKalmanFilter_->setInitialState(state_vec);
  this->tracking_started_ = false;
  return true;
}

bool FeatureBasedBboxTracker::setInitialTracking(const int& pose_x_img, const int& pose_y_img, const std::vector<Bbox3D>& detected_boxes){
    std::unique_ptr<Bbox3D> target_box;
    // Find a detected box containing the point
    for(auto box: detected_boxes){
      auto limits_x = box.getXLimitsImg();
      if(pose_x_img >= limits_x(0) and pose_x_img <= limits_x(1)){
        auto limits_y = box.getYLimitsImg();
        if (pose_y_img >= limits_y(0) and pose_y_img <= limits_y(1)) {
          target_box = std::make_unique<Bbox3D>(box);
          break;
        }
      }
    }
    if(!target_box){
      // given position was not found inside any detected box
      return false;
    }
    trackedBox_ = std::make_unique<TrackedBbox3D>(*target_box);
    Eigen::Vector<float, 6> state_vec = Eigen::Vector<float, 6>::Zero();
    state_vec(0) = target_box->center[0];
    state_vec(1) = target_box->center[1];
    stateKalmanFilter_->setInitialState(state_vec);
    // Target velocity is still unknown
    this->tracking_started_ = false;
    return true;
}

void FeatureBasedBboxTracker::updateTrackedBoxState(){
  Eigen::MatrixXf measurement;
  measurement.resize(6, 1);
  measurement(0) = trackedBox_->box.center.x();
  measurement(1) = trackedBox_->box.center.y();
  measurement(2) = trackedBox_->vel.x();
  measurement(3) = trackedBox_->vel.y();
  measurement(4) = trackedBox_->acc.x();
  measurement(5) = trackedBox_->acc.y();
  stateKalmanFilter_->estimate(measurement);
}

bool FeatureBasedBboxTracker::updateTracking(const std::vector<Bbox3D> &detected_boxes){

    // Predicted the new location of the tracked box
    auto predicted_tracked_box = trackedBox_->predictConstantAcc(timeStep_);

    auto ref_box_features = extractFeatures(predicted_tracked_box);

    // Get the features of all the new detections
    FeaturesVector detected_boxes_feature_vec;
    float max_similarity_score = 0.0;  // Similarity score
    size_t similar_box_idx = 0, count = 0;
    for(auto box: detected_boxes){
      detected_boxes_feature_vec = extractFeatures(box);
      auto error_vec = detected_boxes_feature_vec - ref_box_features;
      float similarity_score = std::exp(-std::pow(error_vec.norm(), 2));

      if (similarity_score > max_similarity_score){
        max_similarity_score = similarity_score;
        similar_box_idx = count;
      }
      count++;
    }
    if (max_similarity_score > minAcceptedSimilarityScore_){
      // Update raw tracking
      trackedBox_->updateFromNewDetection(detected_boxes[similar_box_idx], timeStep_);
      // Update state by applying kalman filter on the raw measurement
      updateTrackedBoxState();
      return true;
    }
    return false;
}

FeatureBasedBboxTracker::FeaturesVector
FeatureBasedBboxTracker::extractFeatures(
    const TrackedBbox3D &bBox_tracked) const {
  return extractFeatures(bBox_tracked.box);
}

FeatureBasedBboxTracker::FeaturesVector
FeatureBasedBboxTracker::extractFeatures(const Bbox3D &bBox) const {
  FeatureBasedBboxTracker::FeaturesVector features_vec = Eigen::Vector<float, 9>::Zero();
  features_vec.segment<2>(0) = bBox.center.segment<2>(0); // indices 0, 1
  features_vec.segment<3>(2) = bBox.size; // indices 2, 3, 4
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

Eigen::Vector3f FeatureBasedBboxTracker::computePointsStdDev(
    const std::vector<Eigen::Vector3f> &pc_points) const {
  // compute the mean in each direction
  auto size = std::max(int(pc_points.size() - 1), 1);
  Eigen::Vector3f mean = Eigen::Vector3f::Zero();
  for (auto point: pc_points){
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
