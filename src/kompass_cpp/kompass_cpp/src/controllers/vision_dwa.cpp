#include "controllers/vision_dwa.h"
#include "datatypes/control.h"
#include "datatypes/path.h"
#include "datatypes/trajectory.h"
#include "utils/angles.h"
#include "utils/logger.h"
#include <cmath>
#include <tuple>
#include <vector>

namespace Kompass {
namespace Control {

VisionDWA::VisionDWA(const ControlType robotCtrlType,
                     const ControlLimitsParams ctrlLimits, int maxLinearSamples,
                     int maxAngularSamples,
                     const CollisionChecker::ShapeType robotShapeType,
                     const std::vector<float> robotDimensions,
                     const std::array<float, 3> &sensor_position_body,
                     const std::array<float, 4> &sensor_rotation_body,
                     const double octreeRes,
                     CostEvaluator::TrajectoryCostsWeights costWeights,
                     const int maxNumThreads, const VisionDWAConfig config)
    : DWA(ctrlLimits, robotCtrlType, config.control_time_step(),
          config.prediction_horizon(), config.control_horizon(),
          maxLinearSamples, maxAngularSamples, robotShapeType, robotDimensions,
          sensor_position_body, sensor_rotation_body, octreeRes, costWeights,
          maxNumThreads) {
  _ctrlType = robotCtrlType;
  _ctrl_limits = ctrlLimits;
  _config = config;
  // Initialize time steps
  int num_steps = _config.control_horizon();
  // Initialize control vectors
  _out_vel = Velocities(num_steps);
  _rotate_in_place = _ctrlType != ControlType::ACKERMANN;
}

Velocity2D VisionDWA::getPureTrackingCtrl(const TrackedPose2D &tracking_pose) {
  float distance = tracking_pose.distance(currentState.x, currentState.y, 0.0);
  float gamma =
      Angle::normalizeToMinusPiPlusPi(tracking_pose.yaw() - currentState.yaw);
  float psi = Angle::normalizeToMinusPiPlusPi(
      std::atan2(tracking_pose.y() - currentState.y,
                 tracking_pose.x() - currentState.x) -
      currentState.yaw);

  float distance_error = _config.target_distance() - distance;
  float angle_error =
      Angle::normalizeToMinusPiPlusPi(_config.target_orientation() - psi);

  float distance_tolerance = _config.tolerance() * _config.target_distance();
  float angle_tolerance = std::max(0.001, _config.tolerance() * _config.target_orientation());

  // LOG_DEBUG("Current distance: ", distance, ", Distance_error=", distance_error,
  //           ", Angle_error=", angle_error, ", tolerance_dist=", distance_tolerance, ", an=", angle_tolerance);

  Velocity2D followingVel;
  if (abs(distance_error) > distance_tolerance or
      abs(angle_error) > angle_tolerance) {
    float v = (tracking_pose.v() * cos(gamma - psi) - _config.K_v() * tanh(distance_error)) / cos(psi);
    followingVel.setVx(v);
    float omega = -tracking_pose.omega() +
                      2 * (v * sin(psi) / distance +
                           tracking_pose.v() * sin(gamma - psi) / distance -
                           _config.K_omega() * tanh(angle_error));
    followingVel.setOmega(omega);
  }
  return followingVel;
}

template <typename T>
std::tuple<Trajectory2D, bool>
VisionDWA::getTrackingReferenceSegment(const TrackedPose2D &tracking_pose,
                                       const T &sensor_points) {
  int step = 0;

  Trajectory2D ref_traj(_config.prediction_horizon());
  std::vector<Path::State> states;
  Path::State simulated_state = currentState;
  Path::State original_state = currentState;
  TrackedPose2D simulated_track = tracking_pose;
  Velocity2D cmd;

  // Simulate following the tracked target for the period til prediction_horizon assuming the target moves with its same current velocity
  while (step < _config.prediction_horizon()) {
    states.push_back(simulated_state);
    ref_traj.path.add(step,
                      Path::Point(simulated_state.x, simulated_state.y, 0.0));
    this->setCurrentState(simulated_state);
    cmd = this->getPureTrackingCtrl(simulated_track);
    simulated_state.update(cmd, _config.control_time_step());
    simulated_track.update(_config.control_time_step());
    if(step < _config.prediction_horizon() -1){
      ref_traj.velocities.add(step,cmd);
    }

    step++;
  }
  this->setCurrentState(original_state);

  bool has_collisions = trajSampler->checkStatesFeasibility(states, sensor_points);

  LOG_INFO("Found reference traj with collisions: ", has_collisions);

  return std::make_tuple(ref_traj, has_collisions);
}

template <typename T>
TrajSearchResult VisionDWA::getTrackingCtrl(const TrackedPose2D &tracking_pose,
                                          const Velocity2D &current_vel,
                                          const T &sensor_points) {
  Trajectory2D ref_traj;
  bool has_collisions;
  std::tie(ref_traj, has_collisions) =
      this->getTrackingReferenceSegment(tracking_pose, sensor_points);
  if(!has_collisions){
    // The tracking sample is collision free -> No need to explore other samples
    TrajSearchResult result;
    result.isTrajFound = true;
    result.trajCost = 0.0;
    result.trajectory = ref_traj;
    latest_velocity_command_ = ref_traj.velocities.getFront();
    return result;
  }
  else{
    LOG_INFO("USING DWA SAMPLING");
    // The tracking sample has collisions -> use DWA-like sampling and control
    Path::Path ref_tracking_path(ref_traj.path.x, ref_traj.path.y,
                                 ref_traj.path.z);
    // Set the tracking segment as the reference path
    this->setCurrentPath(ref_tracking_path);
    return this->computeVelocityCommandsSet(current_vel, sensor_points);
  }
}

// Explicit instantiation for LaserScan
template Control::TrajSearchResult VisionDWA::getTrackingCtrl<LaserScan>(
    const TrackedPose2D &tracking_pose, const Velocity2D &current_vel,
    const LaserScan &sensor_points);

// Explicit instantiation for PointCloud
template Control::TrajSearchResult
VisionDWA::getTrackingCtrl<std::vector<Path::Point>>(
    const TrackedPose2D &tracking_pose, const Velocity2D &current_vel,
    const std::vector<Path::Point> &sensor_points);

} // namespace Control
} // namespace Kompass
