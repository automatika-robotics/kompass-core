#include "controllers/vision_dwa.h"
#include "datatypes/control.h"
#include "datatypes/path.h"
#include "utils/angles.h"
#include "utils/logger.h"
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
                     const std::array<float, 3> &sensor_position_body,
                     const std::array<float, 4> &sensor_rotation_body,
                     const double octreeRes,
                     const CostEvaluator::TrajectoryCostsWeights &costWeights,
                     const int maxNumThreads, const VisionDWAConfig &config,
                     const bool use_tracker)
    : DWA(ctrlLimits, robotCtrlType, config.control_time_step(),
          config.prediction_horizon(), config.control_horizon(),
          maxLinearSamples, maxAngularSamples, robotShapeType, robotDimensions,
          sensor_position_body, sensor_rotation_body, octreeRes, costWeights,
          maxNumThreads) {
  ctrl_limits_ = ctrlLimits;
  config_ = config;
  if (use_tracker) {
    // Initialize the bounding box tracker
    tracker_ = std::make_unique<FeatureBasedBboxTracker>(
        config.control_time_step(), config.e_pose(), config.e_vel(),
        config.e_acc());
  }
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
    if (distance > 0.0){
      omega = -tracking_pose.omega() +
              2.0 * (v * sin(psi) / distance +
                     tracking_pose.v() * sin(gamma - psi) / distance -
                     config_.K_omega() * tanh(angle_error));
    }
    else{
      omega = -tracking_pose.omega() -
              2.0 * config_.K_omega() * tanh(angle_error);
    }
    omega = std::clamp(omega, -ctrl_limits_.omegaParams.maxOmega,
                       ctrl_limits_.omegaParams.maxOmega);
    followingVel.setOmega(omega);
  }
  return followingVel;
}

bool VisionDWA::setInitialTracking(const int &pose_x_img, const int &pose_y_img,
                                   const std::vector<Bbox3D> &detected_boxes) {
  if (!tracker_) {
    LOG_ERROR("Tracker is not initialized. Cannot use "
              "'VisionDWA::setInitialTracking'. Initialize VisionDWA "
              "with 'use_tracker' = true");
    return false;
  }
  return tracker_->setInitialTracking(pose_x_img, pose_y_img, detected_boxes);
}

} // namespace Control
} // namespace Kompass
