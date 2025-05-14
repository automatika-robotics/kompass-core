#include "controllers/vision_dwa.h"
#include "datatypes/control.h"
#include "datatypes/path.h"
#include "datatypes/tracking.h"
#include "utils/angles.h"
#include "utils/logger.h"
#include "utils/transformation.h"
#include "vision/depth_detector.h"
#include <algorithm>
#include <cmath>
#include <memory>
#include <vector>

namespace Kompass {
namespace Control {

VisionDWA::VisionDWA(const ControlType &robotCtrlType,
                     const ControlLimitsParams &ctrlLimits,
                     const int maxLinearSamples, const int maxAngularSamples,
                     const CollisionChecker::ShapeType &robotShapeType,
                     const std::vector<float> &robotDimensions,
                     const Eigen::Vector3f &proximity_sensor_position_body,
                     const Eigen::Vector4f &proximity_sensor_rotation_body,
                     const Eigen::Vector3f &vision_sensor_position_body,
                     const Eigen::Vector4f &vision_sensor_rotation_body,
                     const double octreeRes,
                     const CostEvaluator::TrajectoryCostsWeights &costWeights,
                     const int maxNumThreads, const VisionDWAConfig &config)
    : DWA(ctrlLimits, robotCtrlType, config.control_time_step(),
          config.prediction_horizon(), config.control_horizon(),
          maxLinearSamples, maxAngularSamples, robotShapeType, robotDimensions,
          proximity_sensor_position_body, proximity_sensor_rotation_body,
          octreeRes, costWeights, maxNumThreads) {
  ctrl_limits_ = ctrlLimits;
  config_ = config;
  // Initialize the bounding box tracker
  tracker_ = std::make_unique<FeatureBasedBboxTracker>(
      config.control_time_step(), config.e_pose(), config.e_vel(),
      config.e_acc());
  vision_sensor_tf_ = getTransformation(vision_sensor_rotation_body,
                                        vision_sensor_position_body);
}

void VisionDWA::setCameraIntrinsics(const float focal_length_x,
                                    const float focal_length_y,
                                    const float principal_point_x,
                                    const float principal_point_y) {
  detector_ = std::make_unique<DepthDetector>(
      config_.depth_range(), vision_sensor_tf_,
      Eigen::Vector2f{focal_length_x, focal_length_y},
      Eigen::Vector2f{principal_point_x, principal_point_y},
      config_.depth_conversion_factor());
}

Velocity2D VisionDWA::getPureTrackingCtrl(const TrackedPose2D &tracking_pose) {
  float distance = tracking_pose.distance(currentState.x, currentState.y, 0.0);
  float gamma =
      Angle::normalizeToMinusPiPlusPi(tracking_pose.yaw() - currentState.yaw);
  float psi = Angle::normalizeToMinusPiPlusPi(
      std::atan2(tracking_pose.y() - currentState.y,
                 tracking_pose.x() - currentState.x) -
      currentState.yaw);

  float distance_error = config_.target_distance() - distance;
  float angle_error =
      Angle::normalizeToMinusPiPlusPi(config_.target_orientation() - psi);

  float distance_tolerance = config_.tolerance() * config_.target_distance();
  float angle_tolerance =
      std::max(0.001, config_.tolerance() * config_.target_orientation());

  Velocity2D followingVel;
  if (abs(distance_error) > distance_tolerance or
      abs(angle_error) > angle_tolerance) {
    double v = ((tracking_pose.v() * cos(gamma - psi)) -
                (config_.K_v() * tanh(distance_error))) /
               cos(psi);
    v = std::clamp(v, -ctrl_limits_.velXParams.maxVel,
                   ctrl_limits_.velXParams.maxVel);
    followingVel.setVx(v);
    double omega;
    if (distance > 0.0) {
      omega = -tracking_pose.omega() +
              2.0 * (v * sin(psi) / distance +
                     tracking_pose.v() * sin(gamma - psi) / distance -
                     config_.K_omega() * tanh(angle_error));
    } else {
      omega =
          -tracking_pose.omega() - 2.0 * config_.K_omega() * tanh(angle_error);
    }
    omega = std::clamp(omega, -ctrl_limits_.omegaParams.maxOmega,
                       ctrl_limits_.omegaParams.maxOmega);
    followingVel.setOmega(omega);
  }
  return followingVel;
}

bool VisionDWA::setInitialTracking(const int pose_x_img, const int pose_y_img,
                                   const std::vector<Bbox3D> &detected_boxes) {
  return tracker_->setInitialTracking(pose_x_img, pose_y_img, detected_boxes);
}

bool VisionDWA::setInitialTracking(
    const int pose_x_img, const int pose_y_img,
    const Eigen::MatrixX<unsigned short> &aligned_depth_image,
    const std::vector<Bbox2D> &detected_boxes) {
  if (!detector_) {
    throw std::runtime_error(
        "DepthDetector is not initialized with the camera intrinsics. Call "
        "'VisionDWA::setCameraIntrinsics' first");
  }
  std::unique_ptr<Bbox2D> target_box;
  for (auto box : detected_boxes) {
    auto limits_x = box.getXLimits();
    if (pose_x_img >= limits_x(0) and pose_x_img <= limits_x(1)) {
      auto limits_y = box.getYLimits();
      if (pose_y_img >= limits_y(0) and pose_y_img <= limits_y(1)) {
        target_box = std::make_unique<Bbox2D>(box);
        break;
      }
    }
  }
  if (!target_box) {
    LOG_DEBUG("Target point not found in any detected box");
    return false;
  }
  // Send current state to the detector
  detector_->updateState(currentState);
  detector_->updateBoxes(aligned_depth_image, {*target_box});
  auto boxes_3d = detector_->get3dDetections();
  if (!boxes_3d) {
    LOG_DEBUG("Failed to get 3D box from 2D target box");
    return false;
  }
  return tracker_->setInitialTracking(boxes_3d.value()[0]);
}

Trajectory2D
VisionDWA::getTrackingReferenceSegment(const TrackedPose2D &tracking_pose) {
  int step = 0;

  Trajectory2D ref_traj(config_.prediction_horizon());
  std::vector<Path::State> states;
  Path::State simulated_state = currentState;
  Path::State original_state = currentState;
  TrackedPose2D simulated_track = tracking_pose;
  Velocity2D cmd;

  // Simulate following the tracked target for the period til
  // prediction_horizon assuming the target moves with its same current
  // velocity
  while (step < config_.prediction_horizon()) {
    states.push_back(simulated_state);
    ref_traj.path.add(step,
                      Path::Point(simulated_state.x, simulated_state.y, 0.0));
    this->setCurrentState(simulated_state);
    cmd = this->getPureTrackingCtrl(simulated_track);
    simulated_state.update(cmd, config_.control_time_step());
    simulated_track.update(config_.control_time_step());
    if (step < config_.prediction_horizon() - 1) {
      ref_traj.velocities.add(step, cmd);
    }

    step++;
  }
  this->setCurrentState(original_state);

  return ref_traj;
};

} // namespace Control
} // namespace Kompass
