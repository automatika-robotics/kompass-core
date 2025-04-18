#include "controllers/vision_follower.h"
#include "datatypes/control.h"
#include "utils/logger.h"
#include <cmath>
#include <memory>

namespace Kompass {
namespace Control {

VisionFollower::VisionFollower(const ControlType robotCtrlType,
                               const ControlLimitsParams robotCtrlLimits,
                               const VisionFollowerConfig config) {
  _ctrlType = robotCtrlType;
  _ctrl_limits = robotCtrlLimits;
  _config = config;
  // Initialize time steps
  int num_steps = _config.control_horizon();
  // Initialize control vectors
  _out_vel = Velocities(num_steps);
  _rotate_in_place = _ctrlType != ControlType::ACKERMANN;
}

void VisionFollower::resetTarget(const TrackingData tracking) {
  // empty the search command queue
  std::queue<std::array<double, 3>> empty;
  std::swap(_search_commands_queue, empty);
  // Set the reference depth as the current depth if it is not provided in the
  // config
  if (tracking.depth > 0) {
    if (_config.target_distance() < 0) {
      _config.set_target_distance(tracking.depth);
    }
  }
  // if depth is not provided set the target as the current tracking bounding
  // box size
  else {
    double size = (tracking.size_xy[0] * tracking.size_xy[1]) /
                  (tracking.img_width * tracking.img_height);
    LOG_DEBUG("Setting vision target reference distance to size: ", size);
    _config.set_target_distance(size);
  }
}

void VisionFollower::generateSearchCommands(double total_rotation,
                                            double search_radius,
                                            int max_rotation_steps,
                                            bool enable_pause) {
  // Calculate rotation direction and magnitude
  double rotation_sign = (total_rotation < 0.0) ? -1.0 : 1.0;
  int rotation_steps = max_rotation_steps;
  if (enable_pause) {
    // Modify the total number of active rotation to include the pause steps
    rotation_steps = (max_rotation_steps + _config.search_pause()) /
                     (_config.search_pause() + 1);
  }
  // Angular velocity to rotate 'total_rotation' in total time steps
  // 'rotation_steps' with dt = control_time_step
  double rotation_value =
      (total_rotation / rotation_steps) / _config.control_time_step();
  rotation_value =
      std::max(std::min(rotation_value, _ctrl_limits.omegaParams.maxOmega),
               _config.min_vel());
  // Generate velocity commands
  for (int i = 0; i <= rotation_steps; i = i + 1) {
    if (_ctrlType != ControlType::ACKERMANN) {
      // In-place rotation
      _search_commands_queue.emplace(
          std::array<double, 3>{0.0, 0.0, rotation_sign * rotation_value});
    } else {
      // Angular velocity based on linear velocity and radius
      double omega_ackermann =
          rotation_sign * _ctrl_limits.velXParams.maxVel / search_radius;
      // Non-holonomic circular motion
      _search_commands_queue.emplace(std::array<double, 3>{
          _ctrl_limits.velXParams.maxVel, 0.0, omega_ackermann});
    }
    if (enable_pause) {
      // Add zero commands for search pause
      for (int j = 0; j <= _config.search_pause(); j++) {
        _search_commands_queue.emplace(std::array<double, 3>{0.0, 0.0, 0.0});
      }
    }
  }
  return;
}

void VisionFollower::getFindTargetCmds() {
  // Generate new search commands if starting a new search or no commands are
  // available
  double last_direction = 1.0;
  if (_last_tracking != nullptr) {
    last_direction =
        ((_last_tracking->center_xy[0] - _last_tracking->img_width / 2.0) > 0.0)
            ? 1.0
            : -1.0;
    _last_tracking = nullptr;
  }
  LOG_DEBUG("Generating new search commands in direction: ", last_direction);

  _search_commands_queue = std::queue<std::array<double, 3>>();
  int target_search_steps_max = static_cast<int>(
      _config.target_search_timeout() / _config.control_time_step());
  // rotate pi
  generateSearchCommands(last_direction * M_PI, _config.target_search_radius(),
                         target_search_steps_max / 4, true);
  // go back
  generateSearchCommands(-last_direction * M_PI, _config.target_search_radius(),
                         target_search_steps_max / 8);
  // rotate -pi
  generateSearchCommands(-last_direction * M_PI, _config.target_search_radius(),
                         target_search_steps_max / 4, true);
  // go back
  generateSearchCommands(last_direction * M_PI, _config.target_search_radius(),
                         target_search_steps_max / 8);
}

bool VisionFollower::run(const std::optional<TrackingData> tracking) {
  bool tracking_available = false;
  // Check if tracking has a value
  if (tracking.has_value()) {
    // clear all
    _recorded_wait_time = 0.0;
    _recorded_search_time = 0.0;
    // Access the TrackingData object
    const auto &data = tracking.value();
    // ensure size_xy or depth have valid values
    tracking_available =
        ((data.size_xy[0] > 0 && data.size_xy[1] > 0) || data.depth > 0);
    if (tracking_available) {
      if (_last_tracking == nullptr) {
        resetTarget(data);
      }
      _last_tracking = std::make_unique<TrackingData>(data);
      // Track the target
      trackTarget(data);
      return true;
    }
  }
  // Tracking not available
  if (_config.enable_search()) {
    if (_recorded_search_time < _config.target_search_timeout()) {
      if (_search_commands_queue.empty()) {
        getFindTargetCmds();
      }
      _search_command = _search_commands_queue.front();
      _search_commands_queue.pop();
      _recorded_search_time += _config.control_time_step();
      return true;
    } else {
      _recorded_search_time = 0.0;
      // Failed to find target
      return false;
    }
  } else {
    if (_recorded_wait_time < _config.target_wait_timeout()) {
      LOG_DEBUG("Target lost, waiting to get tracked target again ...");
      _last_tracking = nullptr;
      // Do nothing and wait
      _recorded_wait_time += _config.control_time_step();
      return true;
    } else {
      _recorded_wait_time = 0.0;
      // Failed to get target after waiting
      return false;
    }
  }
}

void VisionFollower::trackTarget(const TrackingData &tracking) {
  double size = (tracking.size_xy[0] * tracking.size_xy[1]) /
                (tracking.img_width * tracking.img_height);

  double current_distance = tracking.depth > 0 ? tracking.depth : size;
  double distance_error = _config.target_distance() - current_distance;

  double distance_tolerance = _config.tolerance() * _config.target_distance();

  double error_y = (2 * tracking.center_xy[1] / tracking.img_height - 1.0);
  double error_x = (2 * tracking.center_xy[0] / tracking.img_width - 1.0);

  LOG_DEBUG("Current distance: ", size,
            ", Reference: ", _config.target_distance(),
            "Distance_error=", distance_error,
            ", Distance_tolerance=", distance_tolerance);

  // Initialize control vectors
  std::fill(_out_vel.vx.begin(), _out_vel.vx.end(), 0.0);
  std::fill(_out_vel.vy.begin(), _out_vel.vy.end(), 0.0);
  std::fill(_out_vel.omega.begin(), _out_vel.omega.end(), 0.0);

  double simulated_depth = current_distance;
  double dist_speed, omega, v;
  for (size_t i = 0; i < _out_vel._length; ++i) {
    // If all errors are within limits -> break
    if (std::abs(distance_error) < distance_tolerance &&
        std::abs(error_y) < _config.tolerance() &&
        std::abs(error_x) < _config.tolerance()) {
      break;
    }
    if (_rotate_in_place && i % 2 != 0)
      continue;

    dist_speed = std::abs(distance_error) > distance_tolerance
                     ? distance_error * _ctrl_limits.velXParams.maxVel
                     : 0.0;

    // X center error : (2.0 * tracking.center_xy[0] / tracking.img_width - 1.0)
    // in [-1, 1] Omega in [-alpha * omega_max, alpha * omega_max]
    omega = -_config.alpha() * error_x * _ctrl_limits.omegaParams.maxOmega;

    // Y center error : (2.0 * tracking.center_xy[1] / tracking.img_height
    // - 1.0) in [-1, 1] V in [-speed_max, speed_max]
    v = _config.beta() * dist_speed;

    // Limit by the minimum allowed velocity to avoid sending meaningless low
    // commands to the robot
    omega = std::abs(omega) >= _config.min_vel() ? omega : 0.0;
    v = std::abs(v) >= _config.min_vel() ? v : 0.0;

    simulated_depth += -v * _config.control_time_step();
    // Update distance error
    distance_error = _config.target_distance() - simulated_depth;

    if (_rotate_in_place) {
      if (std::abs(v) < _config.min_vel()) {
        // Set vx_ctrl[i] and vx_ctrl[i+1] to 0.0
        _out_vel.set(i, 0.0, 0.0, omega);
        if (i + 1 < _out_vel._length) {
          _out_vel.set(i + 1, 0.0, 0.0, omega);
        }
      } else if (std::abs(omega) < _config.min_vel()) {
        // Set vx_ctrl[i] and vx_ctrl[i+1] to 0.0
        _out_vel.set(i, v, 0.0, 0.0);
        if (i + 1 < _out_vel._length) {
          _out_vel.set(i + 1, v, 0.0, 0.0);
        }
      } else {
        // Set vx_ctrl[i] to 0.0
        _out_vel.set(i, v, 0.0, 0.0);
        // Set vx_ctrl[i+1] to max(v, 0.0)
        if (i + 1 < _out_vel._length) {
          _out_vel.set(i + 1, 0.0, 0.0, omega);
        }
      }
    } else {
      _out_vel.set(i, v, 0.0, omega);
    }
  }

  return;
}

const Velocities VisionFollower::getCtrl() const {
  if (_recorded_search_time <= 0.0 && _recorded_wait_time <= 0.0) {
    return _out_vel;
  }
  // If search is on
  else if (_recorded_search_time > 0.0) {
    Velocities _search(1);
    LOG_DEBUG(
        "Number of search commands remaining: ", _search_commands_queue.size(),
        "recorded search time: ", _recorded_search_time);
    _search.set(0, _search_command[0], _search_command[1], _search_command[2]);
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
