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
#include <Eigen/Dense>
#include <cmath>
#include <memory>
#include <optional>
#include <queue>
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

private:
  VisionDWAConfig config_;
  std::unique_ptr<FeatureBasedBboxTracker> tracker_;
  std::unique_ptr<DepthDetector> detector_;
  Eigen::Isometry3f vision_sensor_tf_;
  int track_velocity_;

  /**
   * @brief Get the Tracking Reference Trajectory Segment
   *
   * @tparam T
   * @param tracking_pose
   * @return std::tuple<Trajectory2D, bool>
   */
  Trajectory2D getTrackingReferenceSegment(const TrackedPose2D &tracking_pose);

  Trajectory2D
  getTrackingReferenceSegmentDiffDrive(const TrackedPose2D &tracking_pose);

  /**
   * @brief Get the Pure Tracking Control Command
   *
   * @param tracking_pose
   * @return Velocity2D
   */
  Velocity2D getPureTrackingCtrl(const TrackedPose2D &tracking_pose,
                                 const bool update_global_error = false);

  /**
   * @brief Get the Tracking Control Result based on object tracking and DWA
   * sampling by direct following of the tracked target in local frame
   *
   * @tparam T
   * @param tracking_pose
   * @param current_vel
   * @param sensor_points
   * @return Control::TrajSearchResult
   */
  template <typename T>
  Control::TrajSearchResult
  getTrackingCtrl(const std::optional<TrackedPose2D> &tracked_pose,
                  const Velocity2D &current_vel, const T &sensor_points) {
    if (tracked_pose.has_value()) {
      // Reset recorded wait and search times
      recorded_wait_time_ = 0.0;
      recorded_search_time_ = 0.0;
      LOG_INFO("Tracking target at position: ", tracked_pose->x(), ", ",
               tracked_pose->y(), " with velocity: ", tracked_pose->v(), ", ",
               tracked_pose->omega());
      LOG_INFO("Robot current position: ", currentState.x, ", ",
               currentState.y);
      // Generate reference to target
      Trajectory2D ref_traj;
      if (is_diff_drive_) {
        ref_traj = getTrackingReferenceSegmentDiffDrive(tracked_pose.value());
      } else {
        ref_traj = getTrackingReferenceSegment(tracked_pose.value());
      }

      TrajSearchResult result;
      result.isTrajFound = true;
      result.trajCost = 0.0;
      result.trajectory = ref_traj;
      latest_velocity_command_ = ref_traj.velocities.getFront();
      // ---------------------------------------------------------------
      // Update reference to use in case goal is lost
      auto referenceToTarget =
          Path::Path(ref_traj.path.x, ref_traj.path.y, ref_traj.path.z,
                     config_.prediction_horizon());
      this->setCurrentPath(referenceToTarget, false);
      // ---------------------------------------------------------------
      return result;
    }
    if (this->hasPath() and !isGoalReached() and
        this->currentPath->getSize() > 1) {
      // The tracking sample has collisions -> use DWA-like sampling and control
      return this->computeVelocityCommandsSet(current_vel, sensor_points);
    } else {
      // Start Search and/or Wait if enabled
      if (config_.enable_search()) {
        if (recorded_search_time_ < config_.target_search_timeout()) {
          if (search_commands_queue_.empty()) {
            LOG_DEBUG("Search commands queue is empty, generating new search "
                      "commands");
            int last_direction = 1;
            if (latest_velocity_command_.omega() < 0) {
              last_direction = -1;
            }
            getFindTargetCmds(last_direction);
          }
          LOG_DEBUG("Number of search commands remaining: ",
                    search_commands_queue_.size(),
                    "recorded search time: ", recorded_search_time_);
          // Create search command
          TrajectoryVelocities2D velocities(config_.control_horizon());
          TrajectoryPath path(config_.control_horizon());
          path.add(0, 0.0, 0.0);
          std::array<double, 3> search_command;
          for (int i = 0; i < config_.control_horizon() - 1; i++) {
            if (search_commands_queue_.empty()) {
              LOG_DEBUG("Search commands queue is empty. Ending Search ");
              return TrajSearchResult();
            }
            search_command = search_commands_queue_.front();
            search_commands_queue_.pop();
            recorded_search_time_ += config_.control_time_step();
            path.add(i + 1, 0.0, 0.0);
            velocities.add(i, Velocity2D(search_command[0], search_command[1],
                                         search_command[2]));
          }
          LOG_DEBUG("Sending ", config_.control_horizon(),
                    " search commands "
                    "to the controller");
          auto result = TrajSearchResult();
          result.isTrajFound = true;
          result.trajCost = 0.0;
          result.trajectory = Trajectory2D(velocities, path);
          return result;
        }
      } else {
        if (recorded_wait_time_ < config_.target_wait_timeout()) {
          auto timeout = config_.target_wait_timeout() - recorded_wait_time_;
          LOG_DEBUG("Target lost, waiting to get tracked target again ... "
                    "timeout in ",
                    timeout, " seconds");
          // Do nothing and wait
          TrajectoryVelocities2D velocities(config_.control_horizon());
          TrajectoryPath path(config_.control_horizon());
          path.add(0, 0.0, 0.0);
          for (int i = 0; i < config_.control_horizon() - 1; i++) {
            recorded_wait_time_ += config_.control_time_step();
            velocities.add(i, Velocity2D(0.0, 0.0, 0.0));
            path.add(i + 1, 0.0, 0.0);
          }
          auto result = TrajSearchResult();
          result.isTrajFound = true;
          result.trajCost = 0.0;
          result.trajectory = Trajectory2D(velocities, path);
          return result;
        }
      }
      // Target is lost and not recovered from search or wait
      LOG_WARNING("Target is lost and not recovered from search or wait");
      recorded_wait_time_ = 0.0;
      recorded_search_time_ = 0.0;
      // Empty the search commands queue
      while (!search_commands_queue_.empty()) {
        search_commands_queue_.pop();
      }
      return TrajSearchResult();
    }
  };
};

} // namespace Control
} // namespace Kompass
