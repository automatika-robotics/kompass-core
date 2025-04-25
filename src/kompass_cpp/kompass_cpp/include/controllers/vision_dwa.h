#pragma once

#include "datatypes/control.h"
#include "datatypes/parameter.h"
#include "datatypes/path.h"
#include "datatypes/tracking.h"
#include "datatypes/trajectory.h"
#include "dwa.h"
#include "utils/logger.h"
#include "vision/tracker.h"
#include <Eigen/Dense>
#include <cmath>
#include <memory>
#include <tuple>
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
  };

  VisionDWA(const ControlType &robotCtrlType,
            const ControlLimitsParams &ctrlLimits, const int maxLinearSamples,
            const int maxAngularSamples,
            const CollisionChecker::ShapeType &robotShapeType,
            const std::vector<float> &robotDimensions,
            const std::array<float, 3> &sensor_position_body,
            const std::array<float, 4> &sensor_rotation_body,
            const double octreeRes,
            const CostEvaluator::TrajectoryCostsWeights &costWeights,
            const int maxNumThreads = 1,
            const VisionDWAConfig &config = VisionDWAConfig(),
            const bool use_tracker = true);

  // Default Destructor
  ~VisionDWA() = default;

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
    LOG_INFO("Tracked pose: ", tracked_pose.x(),
             tracked_pose.y(), tracked_pose.yaw());
    Trajectory2D ref_traj;
    bool has_collisions;
    std::tie(ref_traj, has_collisions) =
        this->getTrackingReferenceSegment(tracked_pose, sensor_points);
    if (!has_collisions) {
      // The tracking sample is collision free -> No need to explore other
      // samples
      TrajSearchResult result;
      result.isTrajFound = true;
      result.trajCost = 0.0;
      result.trajectory = ref_traj;
      latest_velocity_command_ = ref_traj.velocities.getFront();
      return result;
    } else {
      LOG_INFO("USING DWA SAMPLING");
      // The tracking sample has collisions -> use DWA-like sampling and control
      Path::Path ref_tracking_path(ref_traj.path.x, ref_traj.path.y,
                                   ref_traj.path.z,
                                   config_.prediction_horizon());
      // Set the tracking segment as the reference path
      // Interpolation of the path is not required as the reference is already
      // created using the robot control time step
      this->setCurrentPath(ref_tracking_path, false);
      return this->computeVelocityCommandsSet(current_vel, sensor_points);
    }
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
    if (!tracker_) {
      throw std::invalid_argument(
          "Tracker is not initialized. Cannot use "
          "'VisionDWA::getTrackingCtrl' "
          "directly with "
          "'std::vector<Bbox3D> &detected_boxes' input. Initialize VisionDWA "
          "with 'use_tracker' = true");
    }
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

  /**
   * @brief Set the initial image position of the target to be tracked
   *
   * @param pose_x_img
   * @param pose_y_img
   * @param detected_boxes
   * @return true
   * @return false
   */
  bool setInitialTracking(const int &pose_x_img, const int &pose_y_img,
                          const std::vector<Bbox3D> &detected_boxes);

private:
  ControlLimitsParams ctrl_limits_;
  VisionDWAConfig config_;
  std::unique_ptr<FeatureBasedBboxTracker> tracker_;

  /**
   * @brief Get the Tracking Reference Trajectory Segment and if this segment is
   * has collision
   *
   * @tparam T
   * @param tracking_pose
   * @param sensor_points
   * @return std::tuple<Trajectory2D, bool>
   */
  template <typename T>
  std::tuple<Trajectory2D, bool>
  getTrackingReferenceSegment(const TrackedPose2D &tracking_pose,
                              const T &sensor_points){
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

    bool has_collisions =
        trajSampler->checkStatesFeasibility(states, sensor_points);

    LOG_INFO("Found reference traj with collisions: ", has_collisions);

    return std::make_tuple(ref_traj, has_collisions);
                              };
  //
};

} // namespace Control
} // namespace Kompass
