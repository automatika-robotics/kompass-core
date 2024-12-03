#include "controllers/vision_follower.h"
#include "datatypes/control.h"
#include "utils/logger.h"
#include <cmath>
#include <memory>

namespace Kompass {
namespace Control {

VisionFollower::VisionFollower(const ControlType robotCtrlType,
                               const ControlLimitsParams robotCtrlLimits,
                               const VisionFollowerConfig config){
  _ctrlType = robotCtrlType;
  _ctrl_limits = robotCtrlLimits;
  _config = config;
  // Initialize time steps
  int num_steps =
      static_cast<int>(_config.control_horizon() / _config.control_time_step());
  _time_steps.resize(num_steps);
  for (int i = 0; i < num_steps; ++i) {
    _time_steps[i] = i * _config.control_time_step();
  }
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
                                              double max_rotation_time) {
  // Calculate rotation direction and magnitude
  double rotation_sign = (total_rotation < 0.0) ? -1.0 : 1.0;
  double abs_rotation = std::abs(total_rotation);

  double rotation_time = std::min(max_rotation_time, abs_rotation / _ctrl_limits.omegaParams.maxOmega);

  // For Non-holonomic circular motion
  double circumference = 2.0f * M_PI * search_radius;
  // Generate velocity commands
  int num_steps =
      static_cast<int>(rotation_time / _config.control_time_step());

  for (int i = 0; i <= num_steps; ++i) {
    if (_ctrlType != ControlType::ACKERMANN) {
      // In-place rotation
      _search_commands_queue.emplace(
          std::array<double, 3>{0.0, 0.0, _ctrl_limits.omegaParams.maxOmega});
    } else {
      // Angular velocity based on linear velocity and radius
      double omega_ackermann =
          rotation_sign * _ctrl_limits.velXParams.maxVel / search_radius;
      // Non-holonomic circular motion
      _search_commands_queue.emplace(std::array<double, 3>{
          _ctrl_limits.velXParams.maxVel, 0.0, omega_ackermann});
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
    LOG_INFO("Generating new search commands for total search time ",
             last_direction);

    _search_commands_queue = std::queue<std::array<double, 3>>();
    generate_search_commands(last_direction * M_PI / 2,
                             _config.target_search_radius(),
                             _config.target_search_timeout() / 3);
    generate_search_commands(-last_direction * M_PI,
                             _config.target_search_radius(),
                             2 * _config.target_search_timeout() / 3);
  }
  LOG_INFO("Number of search commands remaining: ", _search_commands_queue.size());
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
    tracking_available = ((data.size_xy[0] > 0 && data.size_xy[1] > 0) ||
                       data.depth > 0);
    if(tracking_available){
      _recorded_search_time = 0.0;
      // Track the target
      trackTarget(data);
      return true;
    }
  }
  // Tracking not available
  if (_recorded_search_time < _config.target_search_timeout()) {
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
        _config.target_distance() < 0 ? 1.0 : _config.target_distance());
    _target_ref_size = size;
  }

  double target_depth =
      tracking.depth > 0 ? tracking.depth : _target_ref_size / size;
  double distance_error = target_depth - _config.target_distance();

  // Initialize control vectors
  std::fill(_out_vel.vx.begin(), _out_vel.vx.end(), 0.0);
  std::fill(_out_vel.vy.begin(), _out_vel.vy.end(), 0.0);
  std::fill(_out_vel.omega.begin(), _out_vel.omega.end(), 0.0);

  if (std::abs(distance_error) > _config.tolerance() ||
      std::abs(tracking.center_xy[0] - tracking.img_width / 2.0) >
          _config.tolerance() * tracking.img_width ||
      std::abs(tracking.center_xy[1] - tracking.img_height / 2.0) >
          _config.tolerance() * tracking.img_height) {
    double simulated_depth = target_depth;
    for (size_t i = 0; i < _time_steps.size(); ++i) {
      if (_rotate_in_place && i % 2 != 0)
        continue;
      distance_error = simulated_depth - _config.target_distance();
      double dist_speed =
          std::abs(distance_error) > 0.5
              ? std::signbit(distance_error) * _ctrl_limits.velXParams.maxVel
              : distance_error * _ctrl_limits.velXParams.maxVel +
                    std::signbit(distance_error) * _config.min_vel();

      double omega = -_config.alpha() *
                     (tracking.center_xy[0] / tracking.img_width - 0.5);
      double v = -_config.beta() *
                     (tracking.center_xy[1] / tracking.img_height - 0.5) +
                 _config.gamma() * dist_speed;

      simulated_depth += -v * _time_steps[i];
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
  }

  return;
}

const Velocities VisionFollower::getCtrl() const {
  if (_recorded_search_time <= 0.0){
    return _out_vel;
  }
  else{
    Velocities _search(1);
    _search.set(0, _search_command[0], _search_command[1], _search_command[2]);
    return _search;
  }
}


} // namespace Control
} // namespace Kompass
