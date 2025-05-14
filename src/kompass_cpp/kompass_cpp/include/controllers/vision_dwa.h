#pragma once

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
#include <boost/filesystem/path.hpp>
#include <cmath>
#include <memory>
#include <vector>

namespace Kompass {
namespace Control {

class VisionDWA : public DWA {
public:
  class VisionDWAConfig : public ControllerParameters {
  public:
    VisionDWAConfig() : ControllerParameters() {
      addParameter("control_time_step",
                   Parameter(0.1, 1e-4, 1e6, "Control time step (s)"));
      addParameter(
          "control_horizon",
          Parameter(2, 1, 1000, "Number of steps for applying the control"));
      addParameter(
          "prediction_horizon",
          Parameter(10, 1, 1000, "Number of steps for future prediction"));
      addParameter(
          "tolerance",
          Parameter(0.01, 1e-6, 1e3,
                    "Tolerance value for distance and angle following errors"));
      addParameter(
          "target_distance",
          Parameter(
              0.1, -1.0, 1e9,
              "Target distance to maintain with the target (m)")); // Use -1 for
                                                                   // None
      addParameter(
          "target_orientation",
          Parameter(0.0, -M_PI, M_PI,
                    "Bearing angle to maintain with the target (rad)"));
      // Search Parameters
      addParameter("target_wait_timeout", Parameter(30.0, 0.0, 1e3));
      addParameter("target_search_timeout", Parameter(30.0, 0.0, 1e3));
      addParameter("target_search_radius", Parameter(0.5, 1e-4, 1e4));
      addParameter("target_search_pause", Parameter(1.0, 0.0, 1e3));
      // Pure tracking control law parameters
      addParameter("rotation_gain", Parameter(0.5, 1e-2, 10.0));
      addParameter("speed_gain", Parameter(1.0, 1e-2, 10.0));
      addParameter("min_vel", Parameter(0.1, 1e-9, 1e9));
      addParameter("enable_search", Parameter(false));
      // Kalman Filter parameters
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
    bool enable_search() const { return getParameter<bool>("enable_search"); }
    double control_time_step() const {
      return getParameter<double>("control_time_step");
    }
    double target_search_timeout() const {
      return getParameter<double>("target_search_timeout");
    }
    double target_wait_timeout() const {
      return getParameter<double>("target_wait_timeout");
    }
    double target_search_radius() const {
      return getParameter<double>("target_search_radius");
    }
    double search_pause() const {
      return getParameter<double>("target_search_pause");
    }
    int control_horizon() const { return getParameter<int>("control_horizon"); }
    int prediction_horizon() const {
      return getParameter<int>("prediction_horizon");
    }
    double tolerance() const { return getParameter<double>("tolerance"); }
    double target_distance() const {
      double val = getParameter<double>("target_distance");
      return val < 0 ? -1.0 : val; // Return -1 for None
    }
    double target_orientation() const {
      return getParameter<double>("target_orientation");
    }
    void set_target_distance(double value) {
      setParameter("target_distance", value);
    }
    double K_omega() const { return getParameter<double>("rotation_gain"); }
    double K_v() const { return getParameter<double>("speed_gain"); }
    double min_vel() const { return getParameter<double>("min_vel"); }
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
   * @brief Get the Pure Tracking Control Command
   *
   * @param tracking_pose
   * @return Velocity2D
   */
  Velocity2D getPureTrackingCtrl(const TrackedPose2D &tracking_pose);

  /**
   * @brief Get the Tracking Control Result based on object tracking and DWA
   * sampling by direct following of the tracked target
   *
   * @tparam T
   * @param tracking_pose
   * @param current_vel
   * @param sensor_points
   * @return Control::TrajSearchResult
   */
  template <typename T>
  Control::TrajSearchResult getTrackingCtrl(const TrackedPose2D &tracked_pose,
                                            const Velocity2D &current_vel,
                                            const T &sensor_points) {
    LOG_DEBUG("Tracked pose: ", tracked_pose.x(), tracked_pose.y(),
              tracked_pose.yaw());
    Trajectory2D ref_traj = getTrackingReferenceSegment(tracked_pose);
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
  };

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
    if (tracker_->trackerInitialized()) {
      // Update the tracker with the detected boxes
      tracker_->updateTracking(detected_boxes);
      auto tracked_pose = tracker_->getFilteredTrackedPose2D();
      if (tracked_pose) {
        return this->getTrackingCtrl<T>(tracked_pose.value(), current_vel,
                                        sensor_points);
      } else {
        LOG_WARNING("Tracker failed to find the target");
        // Return false for trajectory found
        return TrajSearchResult();
      }
    } else {
      throw std::runtime_error(
          "Tracker is not initialized with an initial tracking target. Call "
          "'VisionDWA::setInitialTracking' first");
    }
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
    if (!detected_boxes_2d.empty()) {
      // Send current state to the detector
      detector_->updateState(currentState);
      detector_->updateBoxes(aligned_depth_img, detected_boxes_2d);
      auto boxes_3d = detector_->get3dDetections();
      if (boxes_3d) {
        LOG_DEBUG("Got 3D boxes from 2d");
        // Update the tracker with the detected boxes
        tracker_->updateTracking(boxes_3d.value());
        auto tracked_pose = tracker_->getFilteredTrackedPose2D();
        if (tracked_pose) {
          return this->getTrackingCtrl<T>(tracked_pose.value(), current_vel,
                                          sensor_points);
        }
      } else {
        LOG_WARNING("Detector failed to find 3D boxes");
      }
    }
    // IF TARGET IS LOST -> USE DWA TO LAST KNOWN LOCATION
    if (!isGoalReached()) {
      LOG_DEBUG("USING DWA SAMPLING TO GO TO LAST KNOWN LOCATION");
      // The tracking sample has collisions -> use DWA-like sampling and control
      return this->computeVelocityCommandsSet(current_vel, sensor_points);
    } else {
      LOG_WARNING("Target is Lost");
      return TrajSearchResult();
    }
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
                          const std::vector<Bbox3D> &detected_boxes);

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
                     const std::vector<Bbox2D> &detected_boxes_2d);

private:
  ControlLimitsParams ctrl_limits_;
  VisionDWAConfig config_;
  std::unique_ptr<FeatureBasedBboxTracker> tracker_;
  std::unique_ptr<DepthDetector> detector_;
  Eigen::Isometry3f vision_sensor_tf_;

  /**
   * @brief Get the Tracking Reference Trajectory Segment
   *
   * @tparam T
   * @param tracking_pose
   * @return std::tuple<Trajectory2D, bool>
   */
  Trajectory2D getTrackingReferenceSegment(const TrackedPose2D &tracking_pose);
  //
};

} // namespace Control
} // namespace Kompass
