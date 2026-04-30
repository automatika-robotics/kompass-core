#pragma once

#include "controllers/follower.h"
#include "controllers/rgb_follower.h"
#include "datatypes/control.h"
#include "datatypes/parameter.h"
#include "datatypes/tracking.h"
#include "datatypes/trajectory.h"
#include "utils/collision_check.h"
#include "utils/logger.h"
#include "vision/depth_detector.h"
#include "vision/tracker.h"
#include <cmath>
#include <memory>
#include <optional>
#include <vector>

namespace Kompass {
namespace Control {

class RGBDFollower : public Follower, public RGBFollower {
public:
  class RGBDFollowerConfig : public RGBFollower::RGBFollowerConfig {
  public:
    RGBDFollowerConfig() : RGBFollower::RGBFollowerConfig() {
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

  RGBDFollower(const ControlType &robotCtrlType,
               const ControlLimitsParams &ctrlLimits,
               const CollisionChecker::ShapeType &robotShapeType,
               const std::vector<float> &robotDimensions,
               const Eigen::Vector3f &vision_sensor_position_body,
               const Eigen::Vector4f &vision_sensor_rotation_body,
               const RGBDFollowerConfig &config = RGBDFollowerConfig());

  // Default Destructor
  ~RGBDFollower() = default;

  void setCameraIntrinsics(const float focal_length_x,
                           const float focal_length_y,
                           const float principal_point_x,
                           const float principal_point_y);

  static double
  getRobotRadius(const CollisionChecker::ShapeType robot_shape_type,
                 const std::vector<float> &robot_dimensions);

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
  Control::TrajSearchResult
  getTrackingCtrl(const std::vector<Bbox3D> &detected_boxes,
                  const Velocity2D &current_vel) {
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
            "'RGBDFollower::setInitialTracking' first");
      }
    }
    return this->getTrackingCtrl(tracked_pose, current_vel);
  };

  Control::TrajSearchResult
  getTrackingCtrl(const Eigen::MatrixX<unsigned short> &aligned_depth_img,
                  const std::vector<Bbox2D> &detected_boxes_2d,
                  const Velocity2D &current_vel) {
    if (!detector_) {
      throw std::runtime_error(
          "DepthDetector is not initialized with the camera intrinsics. Call "
          "'RGBDFollower::setCameraIntrinsics' first");
    }
    if (!tracker_->trackerInitialized()) {
      throw std::runtime_error(
          "Tracker is not initialized with an initial tracking target. Call "
          "'RGBDFollower::setInitialTracking' first");
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
    return this->getTrackingCtrl(tracked_pose, current_vel);
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

private:
  RGBDFollowerConfig config_;
  std::unique_ptr<FeatureBasedBboxTracker> tracker_;
  std::unique_ptr<DepthDetector> detector_;
  Eigen::Isometry3f vision_sensor_tf_;
  int track_velocity_;
  double robot_radius_;

  float currentTargetRadius_ = 0.0f;

  // Pull the latest target geometry off the tracker. No-op if the tracker
  // has nothing to report.
  void refreshTargetGeometry();

  // Build a control-horizon-long stationary trajectory (zero velocities,
  // path pinned at origin). Pure: no side effects on timers or queues.
  TrajSearchResult makeHoldResult() const;

  // Build a result by popping commands off `search_commands_queue_`
  TrajSearchResult popSearchStepResult();

  // ---- Pipeline stages used by the inner getTrackingCtrl ----

  // Search: emit queued search commands when search is enabled and the
  // search timeout has not been reached.
  std::optional<TrajSearchResult> trySearch();

  // Wait: hold position when search is disabled and the wait timeout has
  // not been reached.
  std::optional<TrajSearchResult> tryWait();

  // GiveUp: target is gone and recovery has been exhausted.
  TrajSearchResult giveUp();

  // Build a reference trajectory of `prediction_horizon()` steps that
  // converges from the current robot state toward the tracked target,
  // using the pure-pursuit law in `getPureTrackingCtrl`.
  Trajectory2D getTrackingReferenceSegment(const TrackedPose2D &tracking_pose);

  // In local-coordinate mode, advance the target's robot-relative pose by
  // one step of the robot's own motion (the "target gets pushed back" by
  // the robot's forward step).
  TrackedPose2D updateLocalTarget(const TrackedPose2D &current_target,
                                  const Velocity2D &robot_cmd, double dt);

  // Pure-pursuit control law toward the tracked target.
  Velocity2D getPureTrackingCtrl(const TrackedPose2D &tracking_pose,
                                 const bool update_global_error = false);

  // Inner control-step dispatch
  Control::TrajSearchResult
  getTrackingCtrl(const std::optional<TrackedPose2D> &tracked_pose,
                  const Velocity2D &current_vel) {
    // Pipeline-of-stages dispatch: each stage returns nullopt when it
    // doesn't apply, allowing the next stage to take a turn. The first
    // stage that returns a result wins.
    if (tracked_pose) {
      // Target is back in view — clear any pending search/wait state.
      recorded_wait_time_ = 0.0;
      recorded_search_time_ = 0.0;
      LOG_DEBUG("Following target");

      const Trajectory2D ref_traj =
          getTrackingReferenceSegment(tracked_pose.value());
      TrajSearchResult result;
      result.isTrajFound = true;
      result.trajCost = 0.0f;
      result.trajectory = ref_traj;
      latest_velocity_command_ = result.trajectory.velocities.getFront();
      return result;
    }
    if (auto r = tryWait()) {
      LOG_DEBUG("Waiting for target.");
      return *r;
    }
    if (auto r = trySearch()) {
      LOG_DEBUG("Searching for target.");
      // Reset the latest velocity command to zero to avoid applying stale commands from before the target was lost.
      latest_velocity_command_ = Velocity2D();
      return *r;
    }
    return giveUp();
  };
};

} // namespace Control
} // namespace Kompass
