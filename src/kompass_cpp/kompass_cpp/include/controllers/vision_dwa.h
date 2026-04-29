#pragma once

#include "controllers/rgb_follower.h"
#include "datatypes/control.h"
#include "datatypes/parameter.h"
#include "datatypes/path.h"
#include "datatypes/tracking.h"
#include "datatypes/trajectory.h"
#include "dwa.h"
#include "utils/logger.h"
#include "vision/depth_detector.h"
#include "vision/tracker.h"
#include <cmath>
#include <memory>
#include <optional>
#include <type_traits>
#include <utility>
#include <vector>

namespace Kompass {
namespace Control {

class VisionDWA : public DWA, public RGBFollower {
public:
  class VisionDWAConfig : public RGBFollower::RGBFollowerConfig {
  public:
    VisionDWAConfig() : RGBFollower::RGBFollowerConfig() {
      addParameter(
          "control_horizon",
          Parameter(2, 1, 1000, "Number of steps for applying the control"));
      addParameter(
          "prediction_horizon",
          Parameter(10, 1, 1000, "Number of steps for future prediction"));
      addParameter("distance_tolerance",
                   Parameter(0.1, 1e-6, 1e3, "Distance tolerance value (m)"));
      addParameter("angle_tolerance",
                   Parameter(0.1, 1e-6, M_PI, "Angle tolerance value (rad)"));
      addParameter(
          "target_orientation",
          Parameter(0.0, -M_PI, M_PI,
                    "Bearing angle to maintain with the target (rad)"));
      addParameter(
          "use_local_coordinates",
          Parameter(true,
                    "Track the item in the local frame of the robot. This mode "
                    "cannot track the object velocity but can operate without "
                    "knowing the robot's absolute position (world frame)"));
      addParameter("error_pose", Parameter(0.05, 1e-9, 1e9));
      addParameter("error_vel", Parameter(0.05, 1e-9, 1e9));
      addParameter("error_acc", Parameter(0.05, 1e-9, 1e9));

      // Depth parameters
      addParameter("depth_conversion_factor",
                   Parameter(1e-3, 1e-9, 1e9,
                             "Factor to convert depth image values to meters"));
      addParameter(
          "min_depth",
          Parameter(0.0, 0.0, 1e3, "Range of interest minimum depth value"));
      addParameter(
          "max_depth",
          Parameter(1e3, 1e-3, 1e9, "Range of interest minimum depth value"));
    }
    int control_horizon() const { return getParameter<int>("control_horizon"); }
    bool enable_vel_tracking() const {
      return not getParameter<bool>("use_local_coordinates");
    }
    int prediction_horizon() const {
      return getParameter<int>("prediction_horizon");
    }
    double dist_tolerance() const {
      return getParameter<double>("distance_tolerance");
    }
    double ang_tolerance() const {
      return getParameter<double>("angle_tolerance");
    }
    double target_orientation() const {
      return getParameter<double>("target_orientation");
    }
    double e_pose() const { return getParameter<double>("error_pose"); }
    double e_vel() const { return getParameter<double>("error_vel"); }
    double e_acc() const { return getParameter<double>("error_acc"); }
    double depth_conversion_factor() const {
      return getParameter<double>("depth_conversion_factor");
    }
    Eigen::Vector2f depth_range() const {
      return Eigen::Vector2f{getParameter<double>("min_depth"),
                             getParameter<double>("max_depth")};
    }
  };

  VisionDWA(const ControlType &robotCtrlType,
            const ControlLimitsParams &ctrlLimits, const int maxLinearSamples,
            const int maxAngularSamples,
            const CollisionChecker::ShapeType &robotShapeType,
            const std::vector<float> &robotDimensions,
            const Eigen::Vector3f &proximity_sensor_position_body,
            const Eigen::Vector4f &proximity_sensor_rotation_body,
            const Eigen::Vector3f &vision_sensor_position_body,
            const Eigen::Vector4f &vision_sensor_rotation_body,
            const double octreeRes,
            const CostEvaluator::TrajectoryCostsWeights &costWeights,
            const int maxNumThreads = 1,
            const VisionDWAConfig &config = VisionDWAConfig());

  // Default Destructor
  ~VisionDWA() = default;

  void setCameraIntrinsics(const float focal_length_x,
                           const float focal_length_y,
                           const float principal_point_x,
                           const float principal_point_y);

  /**
   * @brief Get the Tracking Control Result based on object tracking and DWA
   * sampling by searching a set of given detections
   *
   * @tparam T
   * @param tracking_pose
   * @param current_vel
   * @param sensor_points
   * @return Control::TrajSearchResult
   */
  template <typename T>
  Control::TrajSearchResult
  getTrackingCtrl(const std::vector<Bbox3D> &detected_boxes,
                  const Velocity2D &current_vel, const T &sensor_points) {
    std::optional<TrackedPose2D> tracked_pose = std::nullopt;
    if (!detected_boxes.empty()) {
      if (tracker_->trackerInitialized()) {
        // Update the tracker with the detected boxes
        bool tracking_updated = tracker_->updateTracking(detected_boxes);
        if (!tracking_updated) {
          LOG_WARNING(
              "Tracker failed to update target with the detected boxes");
        } else {
          tracked_pose = tracker_->getFilteredTrackedPose2D();
          refreshTargetGeometry();
        }
      } else {
        throw std::runtime_error(
            "Tracker is not initialized with an initial tracking target. Call "
            "'VisionDWA::setInitialTracking' first");
      }
    }
    return this->getTrackingCtrl<T>(tracked_pose, current_vel, sensor_points);
  };

  template <typename T>
  Control::TrajSearchResult
  getTrackingCtrl(const Eigen::MatrixX<unsigned short> &aligned_depth_img,
                  const std::vector<Bbox2D> &detected_boxes_2d,
                  const Velocity2D &current_vel, const T &sensor_points) {
    if (!detector_) {
      throw std::runtime_error(
          "DepthDetector is not initialized with the camera intrinsics. Call "
          "'VisionDWA::setCameraIntrinsics' first");
    }
    if (!tracker_->trackerInitialized()) {
      throw std::runtime_error(
          "Tracker is not initialized with an initial tracking target. Call "
          "'VisionDWA::setInitialTracking' first");
    }
    std::optional<TrackedPose2D> tracked_pose = std::nullopt;
    if (!detected_boxes_2d.empty()) {
      if (track_velocity_) {
        // Send current state to the detector
        detector_->updateBoxes(aligned_depth_img, detected_boxes_2d,
                               currentState);
      } else {
        detector_->updateBoxes(aligned_depth_img, detected_boxes_2d);
      }
      auto boxes_3d = detector_->get3dDetections();
      if (boxes_3d) {
        // Update the tracker with the detected boxes
        bool tracking_updated = tracker_->updateTracking(boxes_3d.value());
        if (!tracking_updated) {
          LOG_WARNING(
              "Tracker failed to update target with the detected boxes");
        } else {
          tracked_pose = tracker_->getFilteredTrackedPose2D();
          refreshTargetGeometry();
        }
      } else {
        LOG_WARNING("Detector failed to find 3D boxes");
      }
    }
    return this->getTrackingCtrl<T>(tracked_pose, current_vel, sensor_points);
  };

  /**
   * @brief Set the initial image position of the target to be tracked
   *
   * @param pose_x_img
   * @param pose_y_img
   * @param detected_boxes
   * @return true
   * @return false
   */
  bool setInitialTracking(const int pose_x_img, const int pose_y_img,
                          const std::vector<Bbox3D> &detected_boxes,
                          const float yaw = 0.0);

  /**
   * @brief  Set the initial image position of the target to be tracked using 2D
   * detections
   *
   * @param pose_x_img
   * @param pose_y_img
   * @param detected_boxes_2d
   * @return true
   * @return false
   */
  bool
  setInitialTracking(const int pose_x_img, const int pose_y_img,
                     const Eigen::MatrixX<unsigned short> &aligned_depth_image,
                     const std::vector<Bbox2D> &detected_boxes_2d,
                     const float yaw = 0.0);

  bool
  setInitialTracking(const Eigen::MatrixX<unsigned short> &aligned_depth_image,
                     const Bbox2D &target_box_2d, const float yaw = 0.0);

  Eigen::Vector2f getErrors() const {
    return Eigen::Vector2f(dist_error_, orientation_error_);
  }

  /**
   * @brief Inscribed-circle radius of the currently tracked target's
   * footprint: 0.5 * max(box.size.x, box.size.y). Refreshed whenever the
   * tracker accepts a new detection (initial setup or update). Returns 0
   * if the tracker has never accepted a target.
   */
  float currentTargetRadius() const { return currentTargetRadius_; }

  /**
   * @brief Convenience overload that handles frame projection and uses
   * the cached currentTargetRadius_. Preferred for any caller that
   * already has the controller in hand.
   */
  template <typename T>
  T filterTargetFromSensorData(const T &sensor_points,
                               const TrackedPose2D &target) const {
    const auto [tx, ty] = targetInSensorFrame<T>(target);
    return filterTargetFromSensorData<T>(sensor_points, tx, ty,
                                         currentTargetRadius_);
  }

  /**
   * @brief Remove tracked-target body returns from sensor data so they
   * don't trip a false collision against the target itself.
   *
   * For a LaserScan, rays whose hit point lies within `target_radius` of
   * (target_x, target_y) get their range set to +infinity (treated as
   * "no return" by the collision checker). For a point cloud (vector of
   * Path::Point), points within `target_radius` of (target_x, target_y)
   * are dropped from the returned cloud.
   *
   * The caller is responsible for expressing (target_x, target_y) in the
   * SAME frame as the sensor data: the proximity-sensor frame for a
   * LaserScan, or the cloud's frame (world or robot-local) for a point
   * cloud.
   *
   * Defined out-of-line in vision_dwa.cpp; explicitly instantiated for
   * Control::LaserScan and std::vector<Path::Point>.
   */
  template <typename T>
  static T filterTargetFromSensorData(const T &sensor_points,
                                      const float target_x,
                                      const float target_y,
                                      const float target_radius);

private:
  VisionDWAConfig config_;
  std::unique_ptr<FeatureBasedBboxTracker> tracker_;
  std::unique_ptr<DepthDetector> detector_;
  Eigen::Isometry3f vision_sensor_tf_;
  int track_velocity_;

  // Inscribed-circle radius of the tracked target's XY footprint, cached
  // from the tracker's current Bbox3D so the early-exit and target filter
  // don't recompute it each cycle. Refreshed by refreshTargetGeometry()
  // after every successful tracker update; left untouched on failures so
  // a transient miss doesn't zero out a previously-known radius.
  float currentTargetRadius_ = 0.0f;

  // Pull the latest target geometry off the tracker. No-op if the tracker
  // has nothing to report.
  void refreshTargetGeometry();

  // Trim a reference trajectory so it ends at least `standoff` away from
  // the target center. Returns nullopt if the trim leaves fewer than 2
  // points (i.e. the reference is already inside the standoff zone) — the
  // caller is expected to fall back to DWA against the untrimmed reference.
  static std::optional<Trajectory2D>
  trimReferenceToStandoff(const Trajectory2D &ref,
                          const TrackedPose2D &target, float standoff);

  // Build collision-check states from a trajectory's path. Yaw is
  // approximated from the path tangent so non-circular footprints get
  // checked at a meaningful orientation.
  static std::vector<Path::State>
  trajectoryToCollisionStates(const Trajectory2D &t);

  // Build a control-horizon-long stationary trajectory (zero velocities,
  // path pinned at origin). Pure: no side effects on timers or queues.
  TrajSearchResult makeHoldResult() const;

  // Build a result by popping commands off `search_commands_queue_`, one
  // per control-horizon step. Pops up to (control_horizon - 1) commands
  // and advances `recorded_search_time_` for each. If the queue runs dry
  // mid-build, returns a default (no-trajectory) TrajSearchResult.
  TrajSearchResult popSearchStepResult();

  // ---- Pipeline stages used by the inner getTrackingCtrl ----
  //
  // The inner control step is a chain of "try this stage; if it produced
  // a result, return it; otherwise fall through to the next stage". Each
  // stage encapsulates one logical mode of the controller (hold at
  // target, follow a clean reference, run DWA avoidance, etc.) so the
  // top-level dispatch reads top-to-bottom.

  // HoldAtTarget: emit a stationary command if the robot is already
  // inside (or at) the configured edge-to-edge gap to the target. Returns
  // nullopt if the robot is still farther away.
  std::optional<TrajSearchResult>
  tryHoldAtTarget(const TrackedPose2D &target) const;

  // FollowReference: build the reference to the target, trim it to the
  // standoff distance, publish it as currentPath (so the DWA fallback has
  // something to plan against), and return the trimmed trajectory if it
  // is collision-free against `filtered_sensor`. Returns nullopt when the
  // trim degenerates or the trimmed reference collides — the caller falls
  // through to a DWA sampling pass on the same filtered sensor.
  //
  // Defined out-of-line in vision_dwa.cpp; explicitly instantiated for
  // Control::LaserScan and std::vector<Path::Point>.
  template <typename T>
  std::optional<TrajSearchResult>
  tryFollowReference(const TrackedPose2D &target, const T &filtered_sensor);

  // DwaWithLeftoverPath: when the target is missing but a previously-set
  // path is still usable, run DWA against it. Returns nullopt if there is
  // nothing meaningful left to plan against.
  template <typename T>
  std::optional<TrajSearchResult>
  tryDWAWithLeftoverPath(const Velocity2D &current_vel,
                         const T &sensor_points) {
    if (this->hasPath() && !isGoalReached() &&
        this->currentPath->getSize() > 2) {
      auto result = this->computeVelocityCommandsSet(current_vel, sensor_points);
      if (result.isTrajFound) {
        // NOTE: If the reference path sent to DWA is too short, the DWA search and optimization can fail.
        // In case of failure return null to fallback to wait/search instead of giving up immediately.
        return result;
      }
    }
    return std::nullopt;
  }

  // Search: emit queued search commands when search is enabled and the
  // search timeout has not been reached. Returns nullopt otherwise.
  std::optional<TrajSearchResult> trySearch();

  // Wait: hold position when search is disabled and the wait timeout has
  // not been reached. Returns nullopt otherwise.
  std::optional<TrajSearchResult> tryWait();

  // GiveUp: target is gone and recovery has been exhausted. Resets the
  // wait/search book-keeping and returns an empty (no-trajectory) result.
  TrajSearchResult giveUp();

  // Project the tracked target's (x, y) into the same frame as sensor data
  // of type T:
  //   - LaserScan returns in the proximity-sensor frame (≈ robot body), so
  //     a world-frame target gets projected through currentState when
  //     track_velocity_; in local-coordinate mode the target is already
  //     robot-relative.
  //   - PointCloud shares the trajectory's frame (world if
  //     track_velocity_, robot-local otherwise) — no projection needed.
  template <typename T>
  std::pair<float, float>
  targetInSensorFrame(const TrackedPose2D &target) const {
    const float tx = static_cast<float>(target.x());
    const float ty = static_cast<float>(target.y());
    if constexpr (std::is_same_v<T, Control::LaserScan>) {
      if (track_velocity_) {
        const float c = std::cos(static_cast<float>(currentState.yaw));
        const float s = std::sin(static_cast<float>(currentState.yaw));
        const float dx = tx - static_cast<float>(currentState.x);
        const float dy = ty - static_cast<float>(currentState.y);
        return {c * dx + s * dy, -s * dx + c * dy};
      }
    }
    return {tx, ty};
  }

  // Build a reference trajectory of `prediction_horizon()` steps that
  // converges from the current robot state toward the tracked target,
  // using the pure-pursuit law in `getPureTrackingCtrl`.
  Trajectory2D getTrackingReferenceSegment(const TrackedPose2D &tracking_pose);

  // In local-coordinate mode, advance the target's robot-relative pose by
  // one step of the robot's own motion (the "target gets pushed back" by
  // the robot's forward step).
  TrackedPose2D updateLocalTarget(const TrackedPose2D &current_target,
                                  const Velocity2D &robot_cmd, double dt);

  // Pure-pursuit control law toward the tracked target. When
  // `update_global_error` is true, the dist/orientation errors are stored
  // on the controller for telemetry; off when called from simulated
  // forward-rollouts so they don't pollute the per-step error.
  Velocity2D getPureTrackingCtrl(const TrackedPose2D &tracking_pose,
                                 const bool update_global_error = false);

  // Inner control-step dispatch: pipeline of stages tried in order. The
  // public getTrackingCtrl wrappers funnel into this overload after
  // resolving detections into an optional tracked pose.
  template <typename T>
  Control::TrajSearchResult
  getTrackingCtrl(const std::optional<TrackedPose2D> &tracked_pose,
                  const Velocity2D &current_vel, const T &sensor_points) {
    // Pipeline-of-stages dispatch: each stage returns nullopt when it
    // doesn't apply, allowing the next stage to take a turn. The first
    // stage that returns a result wins.
    if (tracked_pose) {
      // Target is back in view — clear any pending search/wait state.
      recorded_wait_time_ = 0.0;
      recorded_search_time_ = 0.0;

      // TODO: decide if the forced stop should be kept or not
      // if (auto r = tryHoldAtTarget(*tracked_pose)) {
      //   LOG_WARNING("Target is at required distance, In hold state.");
      //   return *r;
      // }
      // Filter the target out of the sensor data once; both the
      // reference-collision check and the DWA fallback consume it.
      const T filtered_sensor =
          filterTargetFromSensorData<T>(sensor_points, *tracked_pose);
      if (auto r = tryFollowReference<T>(*tracked_pose, filtered_sensor)) {
        LOG_DEBUG("Following reference trajectory.");
        return *r;
      }
    }
    if (auto r = tryDWAWithLeftoverPath<T>(current_vel, sensor_points)) {
      LOG_DEBUG("Using DWA with leftover path.");
      return *r;
    }
    if (auto r = tryWait()) {
      LOG_DEBUG("Waiting for target.");
      return *r;
    }
    if (auto r = trySearch()) {
      LOG_DEBUG("Searching for target.");
      return *r;
    }
    return giveUp();
  };
};

} // namespace Control
} // namespace Kompass
