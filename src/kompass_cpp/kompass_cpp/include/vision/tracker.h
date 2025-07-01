#pragma once

#include "datatypes/control.h"
#include "datatypes/tracking.h"
#include "utils/kalman_filter.h"
#include <Eigen/Dense>
#include <memory>
#include <optional>
#include <vector>

namespace Kompass {

class FeatureBasedBboxTracker {
public:
  FeatureBasedBboxTracker(const float &time_step, const float &e_pos,
                          const float &e_vel, const float &e_acc);

  static constexpr int StateSize = 9;

  using FeaturesVector = Eigen::Vector<float, 9>;

  bool setInitialTracking(const TrackedBbox3D &bBox);

  bool setInitialTracking(const Bbox3D &bBox, const float yaw = 0.0);

  bool setInitialTracking(const int &pose_x_img, const int &pose_y_img,
                          const std::vector<Bbox3D> &detected_boxes,
                          const float yaw = 0.0);

  bool trackerInitialized() const;

  bool updateTracking(const std::vector<Bbox3D> &detected_boxes);

  std::optional<TrackedBbox3D> getRawTracking() const;

  std::optional<Eigen::MatrixXf> getTrackedState() const;

  std::optional<Control::TrackedPose2D> getFilteredTrackedPose2D() const;

private:
  float timeStep_, minAcceptedSimilarityScore_ = 0.0;
  std::string trackedLabel_;
  std::unique_ptr<TrackedBbox3D> trackedBox_;
  std::unique_ptr<LinearSSKalmanFilter> stateKalmanFilter_;

  FeaturesVector extractFeatures(const TrackedBbox3D &bBox) const;

  FeaturesVector extractFeatures(const Bbox3D &bBox) const;

  Eigen::Vector3f
  computePointsStdDev(const std::vector<Eigen::Vector3f> &pc_points) const;

  void updateTrackedBoxState(const int numberSteps = 1);
};

} // namespace Kompass
