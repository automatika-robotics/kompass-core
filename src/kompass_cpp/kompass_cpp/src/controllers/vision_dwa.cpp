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
  is_diff_drive_ = robotCtrlType == ControlType::DIFFERENTIAL_DRIVE;
  config_ = config;
  // Set the reaching goal distance (used in the DWA mode when vision target is
  // lost)
  track_velocity_ = config_.enable_vel_tracking();
  goal_dist_tolerance = config_.e_pose();
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
  float distance =
      tracking_pose.distance(0.0, 0.0, 0.0) - trajSampler->getRobotRadius();
  distance = std::max(distance, 0.0f);
  float psi = Angle::normalizeToMinusPiPlusPi(
      std::atan2(tracking_pose.y(), tracking_pose.x()));

  float distance_error = config_.target_distance() - distance;
  float angle_error =
      Angle::normalizeToMinusPiPlusPi(config_.target_orientation() - psi);

  float distance_tolerance =
      std::max(0.01, config_.tolerance() * config_.target_distance());
  float angle_tolerance =
      std::max(0.01, config_.tolerance() * config_.target_orientation());

  LOG_DEBUG("distance_error=", distance_error, ", angle_error=", angle_error);

  Velocity2D followingVel;
  if (abs(distance_error) > distance_tolerance or
      abs(angle_error) > angle_tolerance) {
    auto term1 =
        (track_velocity_ * tracking_pose.v() * cos(psi)) / cos(angle_error);
    auto term2 = -(config_.K_v() * ctrl_limits_.velXParams.maxVel *
                   tanh(distance_error)) /
                 cos(angle_error);
    double v = ((track_velocity_ * tracking_pose.v() * cos(psi)) -
                (config_.K_v() * ctrl_limits_.velXParams.maxVel *
                 tanh(distance_error))) /
               cos(angle_error);
    LOG_DEBUG("v=", v, ", track_velocity_term=", term1,
              ", dist_error_term=", term2);
    v = std::clamp(v, -ctrl_limits_.velXParams.maxVel,
                   ctrl_limits_.velXParams.maxVel);
    if (std::abs(v) < config_.min_vel()) {
      v = 0.0;
    }
    followingVel.setVx(v);
    double omega;
    if (distance > 0.0) {
      auto term1_o =
          track_velocity_ * (tracking_pose.omega() -
                             2.0 * sin(psi) / distance * tracking_pose.v());
      auto term2_o = -2.0 * v * sin(angle_error) / distance;
      auto term3_o = -(config_.K_omega() * tanh(angle_error));
      omega = -track_velocity_ * tracking_pose.omega() -
              2.0 * track_velocity_ * sin(psi) / distance * tracking_pose.v() -
              2.0 * v * sin(angle_error) / distance -
              2.0 * config_.K_omega() * ctrl_limits_.omegaParams.maxOmega *
                  tanh(angle_error);
      LOG_DEBUG("omega=", omega, ", term_track_velocity=", term1_o,
                ", term_v=", term2_o, ", term_angle_err=", term3_o);
    } else {
      omega = -track_velocity_ * tracking_pose.omega() -
              2.0 * config_.K_omega() * ctrl_limits_.omegaParams.maxOmega *
                  tanh(angle_error);
    }
    omega = std::clamp(omega, -ctrl_limits_.omegaParams.maxOmega,
                       ctrl_limits_.omegaParams.maxOmega);
    if (std::abs(omega) < config_.min_vel()) {
      omega = 0.0;
    }
    followingVel.setOmega(omega);
  }
  return followingVel;
}

bool VisionDWA::setInitialTracking(const int pose_x_img, const int pose_y_img,
                                   const std::vector<Bbox3D> &detected_boxes,
                                   const float yaw) {
  return tracker_->setInitialTracking(pose_x_img, pose_y_img, detected_boxes,
                                      yaw);
}

bool VisionDWA::setInitialTracking(
    const int pose_x_img, const int pose_y_img,
    const Eigen::MatrixX<unsigned short> &aligned_depth_image,
    const std::vector<Bbox2D> &detected_boxes, const float yaw) {

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

  return setInitialTracking(aligned_depth_image, *target_box, yaw);
}

bool VisionDWA::setInitialTracking(
    const Eigen::MatrixX<unsigned short> &aligned_depth_image,
    const Bbox2D &target_box_2d, const float yaw) {
  if (!detector_) {
    throw std::runtime_error(
        "DepthDetector is not initialized with the camera intrinsics. Call "
        "'VisionDWA::setCameraIntrinsics' first");
  }
  // Send current state to the detector
  detector_->updateBoxes(aligned_depth_image, {target_box_2d});
  auto boxes_3d = detector_->get3dDetections();
  if (!boxes_3d) {
    LOG_DEBUG("Failed to get 3D box from 2D target box");
    return false;
  }
  return tracker_->setInitialTracking(boxes_3d.value()[0], yaw);
}

Trajectory2D
VisionDWA::getTrackingReferenceSegment(const TrackedPose2D &tracking_pose) {
  int step = 0;

  Trajectory2D ref_traj(config_.prediction_horizon());
  Path::State simulated_state = currentState;
  Path::State original_state = currentState;
  TrackedPose2D simulated_track = tracking_pose;
  Velocity2D cmd;

  // Simulate following the tracked target for the period til
  // prediction_horizon assuming the target moves with its same current
  // velocity
  while (step < config_.prediction_horizon()) {
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

Trajectory2D VisionDWA::getTrackingReferenceSegmentDiffDrive(
    const TrackedPose2D &tracking_pose) {
  int step = 0;

  Trajectory2D ref_traj(config_.prediction_horizon());
  Path::State simulated_state = currentState;
  Path::State original_state = currentState;
  TrackedPose2D simulated_track = tracking_pose;
  Velocity2D cmd;

  // Simulate following the tracked target for the period til
  // prediction_horizon assuming the target moves with its same current
  // velocity
  while (step < config_.prediction_horizon()) {
    this->setCurrentState(simulated_state);
    cmd = this->getPureTrackingCtrl(simulated_track);
    if (std::abs(cmd.vx()) >= config_.min_vel() &&
        std::abs(cmd.omega()) >= config_.min_vel()) {
      // Rotate then Move
      ref_traj.path.add(step,
                        Path::Point(simulated_state.x, simulated_state.y, 0.0));
      auto vel_rotate = Velocity2D(0.0, 0.0, cmd.omega());
      simulated_state.update(vel_rotate, config_.control_time_step());
      if (step < config_.prediction_horizon() - 1) {
        ref_traj.velocities.add(step, vel_rotate);
      }
      step++;
      if (step < config_.prediction_horizon() - 2) {
        auto vel_move = Velocity2D(cmd.vx(), 0.0, 0.0);
        ref_traj.path.add(
            step, Path::Point(simulated_state.x, simulated_state.y, 0.0));
        simulated_state.update(vel_move, config_.control_time_step());
        ref_traj.velocities.add(step, vel_move);
        step++;
      }
    } else {
      ref_traj.path.add(step,
                        Path::Point(simulated_state.x, simulated_state.y, 0.0));
      simulated_state.update(cmd, config_.control_time_step());
      simulated_track.update(config_.control_time_step());
      if (step < config_.prediction_horizon() - 1) {
        ref_traj.velocities.add(step, cmd);
      }
      step++;
    }
  }
  this->setCurrentState(original_state);

  return ref_traj;
};

void VisionDWA::generateSearchCommands(double total_rotation,
                                       double search_radius,
                                       int max_rotation_steps,
                                       bool enable_pause) {
  // Calculate rotation direction and magnitude
  double rotation_sign = (total_rotation < 0.0) ? -1.0 : 1.0;
  int rotation_steps = max_rotation_steps;
  if (enable_pause) {
    // Modify the total number of active rotation to include the pause steps
    rotation_steps = (max_rotation_steps + config_.search_pause()) /
                     (config_.search_pause() + 1);
  }
  // Angular velocity to rotate 'total_rotation' in total time steps
  // 'rotation_steps' with dt = control_time_step
  double rotation_value =
      (total_rotation / rotation_steps) / config_.control_time_step();
  rotation_value =
      std::max(std::min(rotation_value, ctrl_limits_.omegaParams.maxOmega),
               config_.min_vel());
  // Generate velocity commands
  for (int i = 0; i <= rotation_steps; i = i + 1) {
    if (is_diff_drive_) {
      // In-place rotation
      search_commands_queue_.emplace(
          std::array<double, 3>{0.0, 0.0, rotation_sign * rotation_value});
    } else {
      // Angular velocity based on linear velocity and radius
      double omega_ackermann =
          rotation_sign * ctrl_limits_.velXParams.maxVel / search_radius;
      // Non-holonomic circular motion
      search_commands_queue_.emplace(std::array<double, 3>{
          ctrl_limits_.velXParams.maxVel, 0.0, omega_ackermann});
    }
    if (enable_pause) {
      // Add zero commands for search pause
      for (int j = 0; j <= config_.search_pause(); j++) {
        search_commands_queue_.emplace(std::array<double, 3>{0.0, 0.0, 0.0});
      }
    }
  }
  return;
}

void VisionDWA::getFindTargetCmds(const int last_direction) {
  // Generate new search commands if starting a new search or no commands are
  // available
  LOG_DEBUG("Generating new search commands in direction: ", last_direction);

  search_commands_queue_ = std::queue<std::array<double, 3>>();
  int target_search_steps_max = static_cast<int>(
      config_.target_search_timeout() / config_.control_time_step());
  // rotate pi
  generateSearchCommands(last_direction * M_PI, config_.target_search_radius(),
                         target_search_steps_max / 4, true);
  // go back
  generateSearchCommands(-last_direction * M_PI, config_.target_search_radius(),
                         target_search_steps_max / 8);
  // rotate -pi
  generateSearchCommands(-last_direction * M_PI, config_.target_search_radius(),
                         target_search_steps_max / 4, true);
  // go back
  generateSearchCommands(last_direction * M_PI, config_.target_search_radius(),
                         target_search_steps_max / 8);
}

} // namespace Control
} // namespace Kompass
