#include "controllers/rgb_follower.h"
#include "datatypes/control.h"
#include "datatypes/tracking.h"
#include "utils/logger.h"
#include <cmath>
#include <memory>
#include <algorithm>

namespace Kompass {
namespace Control {

RGBFollower::RGBFollower(const ControlType robotCtrlType,
                         const ControlLimitsParams robotCtrlLimits,
                         const RGBFollowerConfig config) {
  ctrl_limits_ = robotCtrlLimits;
  config_ = config;
  is_diff_drive_ = robotCtrlType == ControlType::DIFFERENTIAL_DRIVE;
}

void RGBFollower::resetTarget(const Bbox2D &target) {
  // empty the search command queue
  std::queue<std::array<double, 3>> empty;
  std::swap(search_commands_queue_, empty);
  // Set the target as the current tracking bounding
  // box size
  LOG_DEBUG("Got target size: ", target.size.x(), ", ", target.size.y(),
            ". Image size=", target.img_size.x(), ", ", target.img_size.y(),
            ". Center=", target.getCenter().x(), ", ", target.getCenter().y());
  float size = static_cast<float>(target.size.x() * target.size.y()) /
               static_cast<float>(target.img_size.x() * target.img_size.y());
  LOG_DEBUG("Setting vision target reference distance to size: ", size);
  config_.set_target_distance(size);
}

void RGBFollower::generateSearchCommands(float total_rotation,
                                         float search_radius,
                                         float max_rotation_time,
                                         bool enable_pause) {
  // Calculate rotation direction and magnitude
  double rotation_sign = (total_rotation < 0.0) ? -1.0 : 1.0;
  float rotation_time = max_rotation_time;
  int num_pause_steps =
      static_cast<int>(config_.search_pause() / config_.control_time_step());
  if (enable_pause) {
    // Modify the total number of active rotation to include the pause steps
    rotation_time =
        max_rotation_time * (1 - num_pause_steps / config_.control_time_step());
  }
  // Angular velocity to rotate 'total_rotation' in total time steps
  // 'rotation_steps' with dt = control_time_step
  double omega_val = total_rotation / rotation_time;

  omega_val = std::max(std::min(omega_val, ctrl_limits_.omegaParams.maxOmega),
                       config_.min_vel());
  // Generate velocity commands
  for (float t = 0.0f; t <= max_rotation_time;
       t = t + config_.control_time_step()) {
    if (is_diff_drive_) {
      // In-place rotation
      search_commands_queue_.emplace(
          std::array<double, 3>{0.0, 0.0, rotation_sign * omega_val});
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
      for (int j = 0; j <= num_pause_steps; j++) {
        search_commands_queue_.emplace(std::array<double, 3>{0.0, 0.0, 0.0});
      }
    }
  }
  return;
}

void RGBFollower::getFindTargetCmds(const int last_direction) {
  // Generate new search commands if starting a new search or no commands are
  // available
  LOG_DEBUG("Generating new search commands in direction: ", last_direction);

  search_commands_queue_ = std::queue<std::array<double, 3>>();
  const float target_searchtimeout_part = config_.target_search_timeout() / 4;
  // rotate pi
  generateSearchCommands(last_direction * M_PI, config_.target_search_radius(),
                         target_searchtimeout_part);
  // go back
  generateSearchCommands(-last_direction * M_PI, config_.target_search_radius(),
                         target_searchtimeout_part);
  // rotate -pi
  generateSearchCommands(-last_direction * M_PI, config_.target_search_radius(),
                         target_searchtimeout_part);
  // go back
  generateSearchCommands(last_direction * M_PI, config_.target_search_radius(),
                         target_searchtimeout_part);
}

bool RGBFollower::run(const std::optional<Bbox2D> target) {
  // Check if tracking has a value
  if (target.has_value()) {
    // clear all
    recorded_wait_time_ = 0.0;
    recorded_search_time_ = 0.0;
    // Access the TrackingData object
    const auto &data = target.value();
    last_tracking_ = std::make_unique<Bbox2D>(data);
    // Track the target
    trackTarget(data);
    return true;
  }
  // Tracking not available
  if (config_.enable_search()) {
    if (recorded_search_time_ < config_.target_search_timeout()) {
      if (search_commands_queue_.empty()) {
        int last_direction = 1;
        if (last_tracking_ != nullptr) {
          auto last_center = last_tracking_->getCenter();
          last_direction =
              ((last_center.x() - last_center.y() / 2.0) > 0.0) ? 1 : -1;
          last_tracking_ = nullptr;
        }
        getFindTargetCmds(last_direction);
      }
      search_command_ = search_commands_queue_.front();
      search_commands_queue_.pop();
      recorded_search_time_ += config_.control_time_step();
      return true;
    } else {
      recorded_search_time_ = 0.0;
      // Failed to find target
      return false;
    }
  } else {
    if (recorded_wait_time_ < config_.target_wait_timeout()) {
      LOG_DEBUG("Target lost, waiting to get tracked target again ...");
      last_tracking_ = nullptr;
      // Do nothing and wait
      recorded_wait_time_ += config_.control_time_step();
      return true;
    } else {
      recorded_wait_time_ = 0.0;
      // Failed to get target after waiting
      return false;
    }
  }
}

void RGBFollower::trackTarget(const Bbox2D &target) {
  float current_dist =
      static_cast<float>(target.size.x() * target.size.y()) /
      static_cast<float>(target.img_size.x() * target.img_size.y());

  dist_error_ = config_.target_distance() - current_dist;

  float distance_tolerance = config_.tolerance() * config_.target_distance();

  // Error to have the target in the center of the image (imgz_size / 2)
  float error_y = 2.0f * (static_cast<float>(target.getCenter().y()) / static_cast<float>(target.img_size.y()) -  0.5f);
  float error_x = 2.0f * (static_cast<float>(target.getCenter().x()) / static_cast<float>(target.img_size.x()) -  0.5f);

  orientation_error_ = error_x;

  LOG_DEBUG("Current size: ", current_dist,
            ", Reference size: ", config_.target_distance(),
            "size_error=", dist_error_, ", tolerance=", distance_tolerance);

  LOG_DEBUG("Img error x: ", error_x,
            ", center_x: ", static_cast<float>(target.getCenter().x()), ", img_size_x: ", static_cast<float>(target.img_size.x()));
  LOG_DEBUG("Img error y: ", error_y,
  ", center_y: ", static_cast<float>(target.getCenter().y()), ", img_size_y: ", static_cast<float>(target.img_size.y()));

  float dist_speed, omega, v;
  // If all errors are within limits -> break
  if (std::abs(dist_error_) < distance_tolerance &&
      std::abs(error_y) < config_.tolerance() &&
      std::abs(error_x) < config_.tolerance()) {
    // Set command to zero
    out_vel_ = Velocities(1);
    out_vel_.set(0, 0.0, 0.0, 0.0);
    return;
  }

  dist_speed = std::abs(dist_error_) > distance_tolerance
                    ? (dist_error_ / config_.target_distance()) * ctrl_limits_.velXParams.maxVel
                    : 0.0;

  // X center error : (2.0 * tracking.center_xy[0] / tracking.img_width - 1.0)
  // in [-1, 1] Omega in [-K_omega * omega_max, K_omega * omega_max]
  omega = -config_.K_omega() * error_x * ctrl_limits_.omegaParams.maxOmega;

  // Y center error : (2.0 * tracking.center_xy[1] / tracking.img_height
  // - 1.0) in [-1, 1] V in [-K_v * speed_max, K_v * speed_max]
  v = config_.K_v() * dist_speed;

  // Limit by the minimum allowed velocity to avoid sending meaningless low
  // commands to the robot
  omega = std::abs(omega) >= config_.min_vel() ? omega : 0.0;
  float omega_limit = static_cast<float>(ctrl_limits_.omegaParams.maxOmega);
  omega = std::clamp(omega, -omega_limit, omega_limit);

  float v_limit = static_cast<float>(ctrl_limits_.velXParams.maxVel);
  v = std::abs(v) >= config_.min_vel() ? v : 0.0;
  v = std::clamp(v, -v_limit, v_limit);

  LOG_DEBUG("dist_error ", dist_error_, ", error_x: ", error_x);
  LOG_DEBUG("v=", v, ", omega= ", omega);

  if (is_diff_drive_ and std::abs(v) >= config_.min_vel() and
      std::abs(omega) >= config_.min_vel()) {
    // Rotate then move
    out_vel_ = Velocities(2);
    out_vel_.set(0, 0.0, 0.0, omega);
    out_vel_.set(1, v, 0.0, 0.0);
  } else {
    out_vel_ = Velocities(1);
    out_vel_.set(0, v, 0.0, omega);
  }

  return;
}

const Velocities RGBFollower::getCtrl() const {
  if (recorded_search_time_ <= 0.0 && recorded_wait_time_ <= 0.0) {
    return out_vel_;
  }
  // If search is on
  else if (recorded_search_time_ > 0.0) {
    Velocities _search(1);
    LOG_DEBUG(
        "Number of search commands remaining: ", search_commands_queue_.size(),
        "recorded search time: ", recorded_search_time_);
    _search.set(0, search_command_[0], search_command_[1], search_command_[2]);
    return _search;
  }
  // If search not active -> wait till timeout
  else {
    // send 0.0 to wait
    Velocities _wait(1);
    return _wait;
  }
}

} // namespace Control
} // namespace Kompass
