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
  double size = (tracking.size_xy[0] * tracking.size_xy[1]) /
                (tracking.img_width * tracking.img_height);
  if (_config.target_distance() < 0 && tracking.depth > 0) {
    _config.set_target_distance(tracking.depth);
  }
  _target_ref_size = size;
  _last_tracking = std::make_unique<TrackingData>(tracking);
}

void VisionFollower::generate_search_commands(double total_rotation,
                                              double search_radius,
                                              int max_rotation_steps,
                                              bool enable_pause) {
  // Calculate rotation direction and magnitude
  double rotation_sign = (total_rotation < 0.0) ? -1.0 : 1.0;
  double abs_rotation = std::abs(total_rotation);

  double rotation_time =
      std::min(max_rotation_steps * _config.control_time_step(),
               abs_rotation / _ctrl_limits.omegaParams.maxOmega);

  // For Non-holonomic circular motion
  double circumference = 2.0f * M_PI * search_radius;

  // Generate velocity commands
  for (int i = 0; i <= max_rotation_steps; i = i + 1) {
    if (_ctrlType != ControlType::ACKERMANN) {
      // In-place rotation
      _search_commands_queue.emplace(std::array<double, 3>{
          0.0, 0.0, rotation_sign * _ctrl_limits.omegaParams.maxOmega});
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

std::array<double, 3> VisionFollower::findTarget() {
  // Generate new search commands if starting a new search or no commands are
  // available
  if (_recorded_search_time == 0.0 || _search_commands_queue.empty()) {
    double last_direction = 1.0;
    if (_last_tracking != nullptr) {
      last_direction = ((_last_tracking->center_xy[0] -
                         _last_tracking->img_width / 2.0) > 0.0)
                           ? 1.0
                           : -1.0;
    }
    LOG_DEBUG("Generating new search commands for total search time ",
              last_direction);

    _search_commands_queue = std::queue<std::array<double, 3>>();
    // rotate pi/2
    generate_search_commands(last_direction * M_PI / 2,
                             _config.target_search_radius(),
                             _config.target_search_timeout() / 4, true);
    // go back
    generate_search_commands(-last_direction * M_PI / 2,
                             _config.target_search_radius(),
                             2 * _config.target_search_timeout() / 4);
    // rotate pi / 2
    generate_search_commands(-last_direction * M_PI / 2,
                             _config.target_search_radius(),
                             2 * _config.target_search_timeout() / 4, true);
    // go back
    generate_search_commands(last_direction * M_PI / 2,
                             _config.target_search_radius(),
                             2 * _config.target_search_timeout() / 4);
  }
  LOG_DEBUG("Number of search commands remaining: ",
            _search_commands_queue.size());
  std::array<double, 3> command = _search_commands_queue.front();
  _search_commands_queue.pop();
  _recorded_search_time += _config.control_time_step();
  return command;
}

bool VisionFollower::run(const std::optional<TrackingData> tracking) {
  bool tracking_available = false;
  // Check if tracking has a value
  if (tracking.has_value()) {
    // Access the TrackingData object
    const auto &data = tracking.value();
    _last_tracking = std::make_unique<TrackingData>(data);
    // ensure size_xy or depth have valid values
    tracking_available =
        ((data.size_xy[0] > 0 && data.size_xy[1] > 0) || data.depth > 0);
    if (tracking_available) {
      _recorded_search_time = 0.0;
      // Track the target
      trackTarget(data);
      return true;
    }
  }
  // Tracking not available
  if ((_recorded_search_time < _config.target_search_timeout()) &&
      _config.enable_search()) {
    _search_command = findTarget();
    return true;
  } else {
    _recorded_search_time = 0.0;
    // Failed to find target
    return false;
  }
}

void VisionFollower::trackTarget(const TrackingData &tracking) {
  double size = (tracking.size_xy[0] * tracking.size_xy[1]) /
                (tracking.img_width * tracking.img_height);

  if (_config.target_distance() < 0 && tracking.depth > 0) {
    _config.set_target_distance(tracking.depth);
  } else if (!_target_ref_size) {
    _config.set_target_distance(
        _config.target_distance() < 0 ? size : _config.target_distance());
    _target_ref_size = size;
  }

  double current_distance = tracking.depth > 0 ? tracking.depth : size;
  double distance_error = _config.target_distance() - current_distance;
  double distance_tolerance = _config.tolerance() * _config.target_distance();

  double error_y = (2 * tracking.center_xy[1] / tracking.img_height - 1.0);
  double error_x = (2 * tracking.center_xy[0] / tracking.img_width - 1.0);

  // Initialize control vectors
  std::fill(_out_vel.vx.begin(), _out_vel.vx.end(), 0.0);
  std::fill(_out_vel.vy.begin(), _out_vel.vy.end(), 0.0);
  std::fill(_out_vel.omega.begin(), _out_vel.omega.end(), 0.0);

  double simulated_depth = current_distance;
  for (size_t i = 0; i < _out_vel._length; ++i) {
    // If all errors are within limits -> break
    if (distance_error < distance_tolerance &&
        std::abs(error_y) < _config.tolerance() &&
        std::abs(error_x) < _config.tolerance()) {
      break;
    }
    if (_rotate_in_place && i % 2 != 0)
      continue;

    double dist_speed = std::abs(distance_error) > _config.tolerance()
                            ? distance_error * _ctrl_limits.velXParams.maxVel
                            : 0.0;

    // X center error : (2.0 * tracking.center_xy[0] / tracking.img_width - 1.0)
    // in [-1, 1] Omega in [-alpha * omega_max, alpha * omega_max]
    double omega =
        -_config.alpha() * error_x * _ctrl_limits.omegaParams.maxOmega;

    // Y center error : (2.0 * tracking.center_xy[1] / tracking.img_height
    // - 1.0) in [-1, 1] V in [-speed_max, speed_max]
    double v =
        -error_y * _ctrl_limits.velXParams.maxVel - _config.beta() * dist_speed;

    // Limit by the minimum allowed velocity to avoid sending meaningless low
    // commands to the robot
    omega = std::abs(omega) >= _config.min_vel() ? omega : 0.0;
    v = std::abs(v) >= _config.min_vel() ? v : 0.0;

    simulated_depth += -v * _config.control_time_step();
    // Update distance error
    distance_error = _config.target_distance() - simulated_depth;

    if (_rotate_in_place) {
      if (std::abs(v) < 1e-2) {
        // Set vx_ctrl[i] and vx_ctrl[i+1] to 0.0
        _out_vel.set(i, 0.0, 0.0, omega);
        if (i + 1 < _out_vel._length) {
          _out_vel.set(i + 1, 0.0, 0.0, omega);
        }
      } else {
        // Set vx_ctrl[i] to 0.0
        _out_vel.set(i, 0.0, 0.0, omega);
        // Set vx_ctrl[i+1] to max(v, 0.0)
        if (i + 1 < _out_vel._length) {
          _out_vel.set(i + 1, 0.0, 0.0, omega);
        }
        _out_vel.set(i, std::max(v, 0.0), 0.0, 0.0);
      }
    } else {
      _out_vel.set(i, std::max(v, 0.0), 0.0, omega);
    }
  }

  return;
}

const Velocities VisionFollower::getCtrl() const {
  if (_recorded_search_time <= 0.0) {
    return _out_vel;
  } else {
    Velocities _search(1);
    _search.set(0, _search_command[0], _search_command[1], _search_command[2]);
    return _search;
  }
}

} // namespace Control
} // namespace Kompass
