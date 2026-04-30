#include "controllers/rgbd_follower.h"
#include "controllers/follower.h"
#include "datatypes/control.h"
#include "datatypes/path.h"
#include "datatypes/tracking.h"
#include "utils/angles.h"
#include "utils/collision_check.h"
#include "utils/logger.h"
#include "utils/transformation.h"
#include "vision/depth_detector.h"
#include <algorithm>
#include <cmath>
#include <memory>
#include <vector>

namespace Kompass {
namespace Control {

RGBDFollower::RGBDFollower(const ControlType &robotCtrlType,
                           const ControlLimitsParams &ctrlLimits,
                           const CollisionChecker::ShapeType &robotShapeType,
                           const std::vector<float> &robotDimensions,
                           const Eigen::Vector3f &vision_sensor_position_body,
                           const Eigen::Vector4f &vision_sensor_rotation_body,
                           const RGBDFollowerConfig &config)
    : Follower(), RGBFollower(robotCtrlType, ctrlLimits) {
  ctrl_limits_ = ctrlLimits;
  config_ = config;
  // Set the reaching goal distance (used in the DWA mode when vision target is
  // lost)
  // TODO: Empirical testing shows that tracked velocity is not accurately
  // estimated, this term should be dropped if no accurate estimation found
  track_velocity_ = config_.enable_vel_tracking();
  goal_dist_tolerance = config_.e_pose();
  // Initialize the bounding box tracker
  tracker_ = std::make_unique<FeatureBasedBboxTracker>(
      config.control_time_step(), config.e_pose(), config.e_vel(),
      config.e_acc());
  vision_sensor_tf_ = getTransformation(vision_sensor_rotation_body,
                                        vision_sensor_position_body);

  robot_radius_ = getRobotRadius(robotShapeType, robotDimensions);
}

double
RGBDFollower::getRobotRadius(const CollisionChecker::ShapeType robot_shape_type,
                             const std::vector<float> &robot_dimensions) {
  double robot_radius;
  // Construct  a geometry object based on the robot shape
  if (robot_shape_type == CollisionChecker::ShapeType::CYLINDER) {
    robot_radius = robot_dimensions.at(0);
  } else if (robot_shape_type == CollisionChecker::ShapeType::BOX) {
    // NOTE: We are using the circumradius here for being conservative with
    // collisions
    robot_radius = std::sqrt(pow(robot_dimensions.at(0), 2) +
                             pow(robot_dimensions.at(1), 2)) /
                   2;
  } else if (robot_shape_type == CollisionChecker::ShapeType::SPHERE) {
    robot_radius = robot_dimensions.at(0);
  } else {
    throw std::invalid_argument("Invalid robot geometry type");
  }
  return robot_radius;
}

void RGBDFollower::setCameraIntrinsics(const float focal_length_x,
                                       const float focal_length_y,
                                       const float principal_point_x,
                                       const float principal_point_y) {
  detector_ = std::make_unique<DepthDetector>(
      config_.depth_range(), vision_sensor_tf_,
      Eigen::Vector2f{focal_length_x, focal_length_y},
      Eigen::Vector2f{principal_point_x, principal_point_y},
      config_.depth_conversion_factor());
}

Velocity2D RGBDFollower::getPureTrackingCtrl(const TrackedPose2D &tracking_pose,
                                             const bool update_global_error) {
  float distance, psi, gamma = 0.0f;
  if (track_velocity_) {
    // World frame: target bearing must be measured from the robot's body,
    // not from the world origin.
    distance = tracking_pose.distance(currentState.x, currentState.y, 0.0) -
               robot_radius_ - currentTargetRadius_;
    psi = Angle::normalizeToMinusPiPlusPi(
        std::atan2(tracking_pose.y() - currentState.y,
                   tracking_pose.x() - currentState.x) -
        currentState.yaw);
    gamma =
        Angle::normalizeToMinusPiPlusPi(tracking_pose.yaw() - currentState.yaw);
  } else {
    distance = tracking_pose.distance(0.0, 0.0, 0.0) - robot_radius_ -
               currentTargetRadius_;
    psi = Angle::normalizeToMinusPiPlusPi(
        std::atan2(tracking_pose.y(), tracking_pose.x()));
  }
  // Floor distance to avoid division by zero in the omega formula below
  constexpr float kMinDistance = 0.001f;
  distance = std::max(distance, kMinDistance);

  float distance_error = config_.target_distance() - distance;
  // target_orientation is the bearing-to-target to maintain in the robot
  // frame; gamma (target heading relative to robot) belongs in the motion
  // feedforward only, not in the angular error.
  float angle_error =
      Angle::normalizeToMinusPiPlusPi(config_.target_orientation() - psi);

  // Update error is enabled (to avoid update for simulated states)
  if (update_global_error) {
    dist_error_ = distance_error;
    orientation_error_ = angle_error;
  }

  float angle_diff = gamma - psi;
  float sin_diff = std::sin(angle_diff);
  float cos_diff = std::cos(angle_diff);

  Velocity2D followingVel;
  if (abs(distance_error) > config_.dist_tolerance() or
      abs(angle_error) > config_.ang_tolerance()) {
    double v =
        track_velocity_ * (tracking_pose.v() * cos_diff) -
        config_.K_v() * ctrl_limits_.velXParams.maxVel * tanh(distance_error);

    v = std::clamp(v, -ctrl_limits_.velXParams.maxVel,
                   ctrl_limits_.velXParams.maxVel);
    if (std::abs(v) < config_.min_vel()) {
      v = 0.0;
    }
    followingVel.setVx(v);
    double omega;

    omega = (track_velocity_ * tracking_pose.v() * sin_diff / distance +
             v * sin(psi) / distance -
             config_.K_omega() * ctrl_limits_.omegaParams.maxOmega *
                 tanh(angle_error));

    omega = std::clamp(omega, -ctrl_limits_.omegaParams.maxOmega,
                       ctrl_limits_.omegaParams.maxOmega);
    if (std::abs(omega) < config_.min_vel()) {
      omega = 0.0;
    }
    followingVel.setOmega(omega);
  }
  return followingVel;
}

bool RGBDFollower::setInitialTracking(const int pose_x_img,
                                      const int pose_y_img,
                                      const std::vector<Bbox3D> &detected_boxes,
                                      const float yaw) {
  const bool ok =
      tracker_->setInitialTracking(pose_x_img, pose_y_img, detected_boxes, yaw);
  if (ok) {
    refreshTargetGeometry();
  }
  return ok;
}

bool RGBDFollower::setInitialTracking(
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

bool RGBDFollower::setInitialTracking(
    const Eigen::MatrixX<unsigned short> &aligned_depth_image,
    const Bbox2D &target_box_2d, const float yaw) {
  if (!detector_) {
    throw std::runtime_error(
        "DepthDetector is not initialized with the camera intrinsics. Call "
        "'RGBDFollower::setCameraIntrinsics' first");
  }
  if (track_velocity_) {
    // Send current state to the detector
    detector_->updateBoxes(aligned_depth_image, {target_box_2d}, currentState);
  } else {
    detector_->updateBoxes(aligned_depth_image, {target_box_2d});
  }
  auto boxes_3d = detector_->get3dDetections();
  if (!boxes_3d || boxes_3d->empty()) {
    LOG_DEBUG("Failed to get 3D box from 2D target box");
    return false;
  }
  const bool ok = tracker_->setInitialTracking(boxes_3d.value()[0], yaw);
  if (ok) {
    refreshTargetGeometry();
  }
  return ok;
}

void RGBDFollower::refreshTargetGeometry() {
  if (auto raw = tracker_->getRawTracking()) {
    const auto &sz = raw->box.size;
    currentTargetRadius_ = 0.5f * std::max(sz.x(), sz.y());
  }
  // else: leave previous value untouched so a transient miss doesn't
  // erase a previously-known radius.
}

std::optional<TrajSearchResult> RGBDFollower::trySearch() {
  if (!config_.enable_search()) {
    return std::nullopt;
  }
  // Reset recorded wait time so that if we lose the target again while in
  // search, we don't immediately give up but get a fresh wait timeout to try to
  // reacquire it.
  recorded_wait_time_ = 0.0;
  if (search_commands_queue_.empty()) {
    LOG_DEBUG("Search commands queue is empty, generating new search "
              "commands");
    const int last_direction = (latest_velocity_command_.omega() < 0) ? -1 : 1;
    getFindTargetCmds(last_direction);
  }
  if (recorded_search_time_ >= config_.target_search_timeout()) {
    LOG_DEBUG("Search timeout reached. Giving up.");
    return std::nullopt;
  }
  LOG_DEBUG(
      "Number of search commands remaining: ", search_commands_queue_.size(),
      "recorded search time: ", recorded_search_time_);
  return popSearchStepResult();
}

std::optional<TrajSearchResult> RGBDFollower::tryWait() {
  if (config_.enable_search()) {
    // wait for one control step to avoid going into search immediately after
    // losing the target, which can cause the robot to move in an undesired way
    // if the target is only lost for a very short time (e.g., due to a
    // transient occlusion or detection miss)
    LOG_DEBUG("Search is enabled, waiting for one control step before starting "
              "search");
    if (recorded_wait_time_ >= config_.control_time_step()) {
      LOG_DEBUG("Waited for one control step, now falling back to search");
      return std::nullopt; // search mode handles target-lost recovery
    }
    // Reset the search command queue
    std::queue<Eigen::Vector3d> empty;
    std::swap(search_commands_queue_, empty);
    recorded_wait_time_ +=
        (config_.control_horizon() - 1) * config_.control_time_step();
    return makeHoldResult();
  }
  if (recorded_wait_time_ >= config_.target_wait_timeout()) {
    return std::nullopt;
  }
  const double timeout = config_.target_wait_timeout() - recorded_wait_time_;
  LOG_DEBUG("Target lost, waiting to get tracked target again ... timeout in ",
            timeout, " seconds");
  recorded_wait_time_ +=
      (config_.control_horizon() - 1) * config_.control_time_step();
  return makeHoldResult();
}

TrajSearchResult RGBDFollower::giveUp() {
  LOG_WARNING("Target is lost and not recovered from search or wait");
  recorded_wait_time_ = 0.0;
  recorded_search_time_ = 0.0;
  while (!search_commands_queue_.empty()) {
    search_commands_queue_.pop();
  }
  return TrajSearchResult();
}

TrajSearchResult RGBDFollower::makeHoldResult() const {
  TrajectoryVelocities2D velocities(config_.control_horizon());
  TrajectoryPath path(config_.control_horizon());
  path.add(0, 0.0, 0.0);
  for (int i = 0; i < config_.control_horizon() - 1; ++i) {
    velocities.add(i, Velocity2D(0.0, 0.0, 0.0));
    path.add(i + 1, 0.0, 0.0);
  }
  TrajSearchResult result;
  result.isTrajFound = true;
  result.trajCost = 0.0f;
  result.trajectory = Trajectory2D(velocities, path);
  return result;
}

TrajSearchResult RGBDFollower::popSearchStepResult() {
  TrajectoryVelocities2D velocities(config_.control_horizon());
  TrajectoryPath path(config_.control_horizon());
  path.add(0, 0.0, 0.0);
  for (int i = 0; i < config_.control_horizon() - 1; ++i) {
    if (search_commands_queue_.empty()) {
      LOG_DEBUG("Search commands queue is empty. Ending Search ");
      return TrajSearchResult();
    }
    const Eigen::Vector3d cmd = search_commands_queue_.front();
    search_commands_queue_.pop();
    recorded_search_time_ += config_.control_time_step();
    path.add(i + 1, 0.0, 0.0);
    velocities.add(i, Velocity2D(cmd[0], cmd[1], cmd[2]));
  }
  TrajSearchResult result;
  result.isTrajFound = true;
  result.trajCost = 0.0f;
  result.trajectory = Trajectory2D(velocities, path);
  return result;
}

TrackedPose2D
RGBDFollower::updateLocalTarget(const TrackedPose2D &current_target,
                                const Velocity2D &robot_cmd, double dt) {
  // Calculate transform of one step movement
  Path::State step_move(0.0, 0.0, 0.0);
  step_move.update(robot_cmd, dt);
  Eigen::Isometry3f step_tf = getTransformation(step_move);

  // Apply inverse transform to the target
  // i.e. The target is "pushed back" by the robot's forward movement
  auto new_pos = transformPosition(
      Eigen::Vector3f(current_target.x(), current_target.y(), 0.0),
      step_tf.inverse());

  return TrackedPose2D(new_pos.x(), new_pos.y(), current_target.z(), 0.0, 0.0,
                       0.0);
}

Trajectory2D
RGBDFollower::getTrackingReferenceSegment(const TrackedPose2D &tracking_pose) {
  Trajectory2D ref_traj(config_.prediction_horizon());

  Path::State initialState =
      track_velocity_ ? Path::State(currentState) : Path::State(0, 0, 0);
  Path::State sim_state = initialState;
  TrackedPose2D sim_target = tracking_pose;
  double dt = config_.control_time_step();

  for (int step = 0; step < config_.prediction_horizon(); ++step) {
    // Record current position in trajectory
    ref_traj.path.add(step, Path::Point(sim_state.x, sim_state.y, 0.0));

    // Nominal Control Command
    this->setCurrentState(sim_state);
    Velocity2D cmd = this->getPureTrackingCtrl(sim_target, (step == 0));

    // Update Robot State
    sim_state.update(cmd, dt);

    // Update Target Pose
    if (track_velocity_) {
      sim_target.update(dt);
    } else {
      // Using the helper function suggested previously to fix the math
      sim_target = updateLocalTarget(sim_target, cmd, dt);
    }

    // Store velocity in trajectory
    if (step < config_.prediction_horizon() - 1) {
      ref_traj.velocities.add(step, cmd);
    }
  }

  this->setCurrentState(initialState); // Restore original state
  return ref_traj;
}

} // namespace Control
} // namespace Kompass
