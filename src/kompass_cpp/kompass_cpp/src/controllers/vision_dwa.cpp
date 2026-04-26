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
#include <limits>
#include <memory>
#include <type_traits>
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
          octreeRes, costWeights, maxNumThreads),
      RGBFollower(robotCtrlType, ctrlLimits) {
  ctrl_limits_ = ctrlLimits;
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

Velocity2D VisionDWA::getPureTrackingCtrl(const TrackedPose2D &tracking_pose, const bool update_global_error) {
  float distance, psi, gamma = 0.0f;
  if (track_velocity_) {
    // World frame: target bearing must be measured from the robot's body,
    // not from the world origin.
    distance = tracking_pose.distance(currentState.x, currentState.y, 0.0) -
               trajSampler->getRobotRadius();
    psi = Angle::normalizeToMinusPiPlusPi(
        std::atan2(tracking_pose.y() - currentState.y,
                   tracking_pose.x() - currentState.x) -
        currentState.yaw);
    gamma =
        Angle::normalizeToMinusPiPlusPi(tracking_pose.yaw() - currentState.yaw);
  } else {
    distance =
        tracking_pose.distance(0.0, 0.0, 0.0) - trajSampler->getRobotRadius();
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

bool VisionDWA::setInitialTracking(const int pose_x_img, const int pose_y_img,
                                   const std::vector<Bbox3D> &detected_boxes,
                                   const float yaw) {
  const bool ok = tracker_->setInitialTracking(pose_x_img, pose_y_img,
                                               detected_boxes, yaw);
  if (ok) {
    refreshTargetGeometry();
  }
  return ok;
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
  if (track_velocity_) {
    // Send current state to the detector
    detector_->updateBoxes(aligned_depth_image, {target_box_2d}, currentState);
  } else {
    detector_->updateBoxes(aligned_depth_image, {target_box_2d});
  }
  auto boxes_3d = detector_->get3dDetections();
  if (!boxes_3d) {
    LOG_DEBUG("Failed to get 3D box from 2D target box");
    return false;
  }
  const bool ok = tracker_->setInitialTracking(boxes_3d.value()[0], yaw);
  if (ok) {
    refreshTargetGeometry();
  }
  return ok;
}

void VisionDWA::refreshTargetGeometry() {
  if (auto raw = tracker_->getRawTracking()) {
    const auto &sz = raw->box.size;
    currentTargetRadius_ = 0.5f * std::max(sz.x(), sz.y());
  }
  // else: leave previous value untouched so a transient miss doesn't
  // erase a previously-known radius.
}

std::optional<Trajectory2D>
VisionDWA::trimReferenceToStandoff(const Trajectory2D &ref,
                                   const TrackedPose2D &target,
                                   float standoff) {
  const float tx = static_cast<float>(target.x());
  const float ty = static_cast<float>(target.y());
  const size_t n = ref.path.numPointsPerTrajectory_;
  size_t trim_size = n;
  for (size_t i = 0; i < n; ++i) {
    const float dx = ref.path.x(i) - tx;
    const float dy = ref.path.y(i) - ty;
    if (std::hypot(dx, dy) < standoff) {
      trim_size = i;
      break;
    }
  }
  if (trim_size < 2) {
    return std::nullopt;
  }
  Trajectory2D trimmed(trim_size);
  for (size_t i = 0; i < trim_size; ++i) {
    trimmed.path.add(i, ref.path.x(i), ref.path.y(i), ref.path.z(i));
  }
  for (size_t i = 0; i + 1 < trim_size; ++i) {
    trimmed.velocities.add(i, ref.velocities.vx(i), ref.velocities.vy(i),
                           ref.velocities.omega(i));
  }
  return trimmed;
}

std::optional<TrajSearchResult>
VisionDWA::tryHoldAtTarget(const TrackedPose2D &target) const {
  const float robot_radius =
      static_cast<float>(trajSampler->getRobotRadius());
  // In track_velocity_ mode the tracked pose is in world frame; in
  // local-coordinate mode the tracked pose is already robot-relative,
  // so the robot sits at the origin.
  const float robot_x =
      track_velocity_ ? static_cast<float>(currentState.x) : 0.0f;
  const float robot_y =
      track_velocity_ ? static_cast<float>(currentState.y) : 0.0f;
  const float dx = static_cast<float>(target.x()) - robot_x;
  const float dy = static_cast<float>(target.y()) - robot_y;
  const float edge_gap =
      std::hypot(dx, dy) - currentTargetRadius_ - robot_radius;
  if (edge_gap <= static_cast<float>(config_.target_distance())) {
    return makeHoldResult();
  }
  return std::nullopt;
}

std::optional<TrajSearchResult> VisionDWA::trySearch() {
  if (!config_.enable_search()) {
    return std::nullopt;
  }
  if (recorded_search_time_ >= config_.target_search_timeout()) {
    return std::nullopt;
  }
  if (search_commands_queue_.empty()) {
    LOG_DEBUG("Search commands queue is empty, generating new search "
              "commands");
    const int last_direction =
        (latest_velocity_command_.omega() < 0) ? -1 : 1;
    getFindTargetCmds(last_direction);
  }
  LOG_DEBUG("Number of search commands remaining: ",
            search_commands_queue_.size(),
            "recorded search time: ", recorded_search_time_);
  return popSearchStepResult();
}

std::optional<TrajSearchResult> VisionDWA::tryWait() {
  if (config_.enable_search()) {
    return std::nullopt; // search mode handles target-lost recovery
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

TrajSearchResult VisionDWA::giveUp() {
  LOG_WARNING("Target is lost and not recovered from search or wait");
  recorded_wait_time_ = 0.0;
  recorded_search_time_ = 0.0;
  while (!search_commands_queue_.empty()) {
    search_commands_queue_.pop();
  }
  return TrajSearchResult();
}

TrajSearchResult VisionDWA::makeHoldResult() const {
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

TrajSearchResult VisionDWA::popSearchStepResult() {
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

std::vector<Path::State>
VisionDWA::trajectoryToCollisionStates(const Trajectory2D &t) {
  const size_t n = t.path.numPointsPerTrajectory_;
  std::vector<Path::State> states;
  states.reserve(n);
  for (size_t i = 0; i < n; ++i) {
    double yaw = 0.0;
    if (i + 1 < n) {
      yaw = std::atan2(t.path.y(i + 1) - t.path.y(i),
                       t.path.x(i + 1) - t.path.x(i));
    } else if (i > 0) {
      yaw = std::atan2(t.path.y(i) - t.path.y(i - 1),
                       t.path.x(i) - t.path.x(i - 1));
    }
    states.emplace_back(t.path.x(i), t.path.y(i), yaw);
  }
  return states;
}

TrackedPose2D VisionDWA::updateLocalTarget(const TrackedPose2D &current_target,
                                           const Velocity2D &robot_cmd,
                                           double dt) {
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
VisionDWA::getTrackingReferenceSegment(const TrackedPose2D &tracking_pose) {
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

// ----------------------------------------------------------------------------
// Templated helper bodies. Defined out-of-line and explicitly instantiated
// for the two T types the controller is used with (Control::LaserScan and
// std::vector<Path::Point>) so the header stays focused on declarations.
// ----------------------------------------------------------------------------

template <typename T>
T VisionDWA::filterTargetFromSensorData(const T &sensor_points,
                                        const float target_x,
                                        const float target_y,
                                        const float target_radius) {
  T filtered = sensor_points;
  if (target_radius <= 0.0f) {
    return filtered;
  }
  const float r_sq = target_radius * target_radius;
  if constexpr (std::is_same_v<T, Control::LaserScan>) {
    for (size_t i = 0; i < filtered.ranges.size(); ++i) {
      const double r = filtered.ranges[i];
      if (!std::isfinite(r)) {
        continue;
      }
      const double a = filtered.angles[i];
      const double dx = r * std::cos(a) - target_x;
      const double dy = r * std::sin(a) - target_y;
      if (dx * dx + dy * dy < r_sq) {
        filtered.ranges[i] = std::numeric_limits<double>::infinity();
      }
    }
  } else {
    filtered.clear();
    filtered.reserve(sensor_points.size());
    for (const auto &p : sensor_points) {
      const float dx = p.x() - target_x;
      const float dy = p.y() - target_y;
      if (dx * dx + dy * dy >= r_sq) {
        filtered.push_back(p);
      }
    }
  }
  return filtered;
}

template <typename T>
std::optional<TrajSearchResult>
VisionDWA::tryFollowReference(const TrackedPose2D &target,
                              const T &filtered_sensor) {
  const Trajectory2D ref_traj = getTrackingReferenceSegment(target);
  const float standoff =
      config_.target_distance() > 0.0
          ? static_cast<float>(config_.target_distance())
          : static_cast<float>(config_.dist_tolerance());
  auto trimmed_opt = trimReferenceToStandoff(ref_traj, target, standoff);

  // Publish whichever path we have (trimmed if usable, else the full
  // reference) as a single segment so the DWA fallback / search has
  // something to plan against and won't replan against a tiny stub.
  const Trajectory2D &to_publish = trimmed_opt ? *trimmed_opt : ref_traj;
  auto pub_path = Path::Path(to_publish.path.x, to_publish.path.y,
                             to_publish.path.z);
  path_segment_length_ = pub_path.totalPathLength();
  max_segment_size_ = pub_path.getSize();
  this->setCurrentPath(pub_path, true);

  if (!trimmed_opt) {
    return std::nullopt;
  }
  const bool collides = trajSampler->checkStatesFeasibility(
      trajectoryToCollisionStates(*trimmed_opt), filtered_sensor);
  if (collides) {
    return std::nullopt;
  }
  TrajSearchResult result;
  result.isTrajFound = true;
  result.trajCost = 0.0f;
  result.trajectory = *trimmed_opt;
  return result;
}

// Explicit instantiations for the two sensor types the controller
// supports. Adding a new sensor type means adding two more lines below.
template Control::LaserScan
VisionDWA::filterTargetFromSensorData<Control::LaserScan>(
    const Control::LaserScan &, const float, const float, const float);
template std::vector<Path::Point>
VisionDWA::filterTargetFromSensorData<std::vector<Path::Point>>(
    const std::vector<Path::Point> &, const float, const float, const float);

template std::optional<TrajSearchResult>
VisionDWA::tryFollowReference<Control::LaserScan>(
    const TrackedPose2D &, const Control::LaserScan &);
template std::optional<TrajSearchResult>
VisionDWA::tryFollowReference<std::vector<Path::Point>>(
    const TrackedPose2D &, const std::vector<Path::Point> &);

} // namespace Control
} // namespace Kompass
