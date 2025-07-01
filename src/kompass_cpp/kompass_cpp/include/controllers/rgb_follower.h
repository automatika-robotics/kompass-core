#pragma once

#include "datatypes/control.h"
#include "datatypes/parameter.h"
#include "datatypes/tracking.h"
#include <memory>
#include <optional>
#include <queue>

namespace Kompass {
namespace Control {

class RGBFollower {
public:
  class RGBFollowerConfig : public Parameters {
  public:
    RGBFollowerConfig() {
      addParameter("control_time_step",
                   Parameter(0.1, 1e-4, 1e6, "Control time step (s)"));
      addParameter("tolerance", Parameter(0.1, 0.0, 1.0, "Tolerance value"));
      addParameter(
          "target_distance",
          Parameter(
              0.1, -1.0, 1e9,
              "Target distance to maintain with the target (m)")); // Use -1 for
                                                                   // None
      // Search Parameters
      addParameter("target_wait_timeout", Parameter(30.0, 0.0, 1e3));
      addParameter("target_search_timeout", Parameter(30.0, 0.0, 1e3));
      addParameter("target_search_radius", Parameter(0.5, 1e-4, 1e4));
      addParameter("target_search_pause", Parameter(1.0, 0.0, 1e3));
      // Pure tracking control law parameters
      addParameter("rotation_gain", Parameter(1.0, 1e-2, 10.0));
      addParameter("speed_gain", Parameter(1.0, 1e-2, 10.0));
      addParameter("min_vel", Parameter(0.01, 1e-9, 1e9));
      addParameter("enable_search", Parameter(false));
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
    double tolerance() const { return getParameter<double>("tolerance"); }
    double target_distance() const {
      double val = getParameter<double>("target_distance");
      return val < 0 ? -1.0 : val; // Return -1 for None
    }
    void set_target_distance(double value) {
      setParameter("target_distance", value);
    }
    double K_omega() const { return getParameter<double>("rotation_gain"); }
    double K_v() const { return getParameter<double>("speed_gain"); }
    double min_vel() const { return getParameter<double>("min_vel"); }
  };

  RGBFollower(const ControlType robotCtrlType,
              const ControlLimitsParams ctrl_limits,
              const RGBFollowerConfig config = RGBFollowerConfig());

  // Default Destructor
  ~RGBFollower() = default;

  void resetTarget(const Bbox2D &tracking);

  bool run(const std::optional<Bbox2D> tracking);

  const Velocities getCtrl() const;

  Eigen::Vector2f getErrors() const {
    return Eigen::Vector2f(dist_error_, orientation_error_);
  }

protected:
  bool is_diff_drive_;
  ControlLimitsParams ctrl_limits_;
  double recorded_search_time_ = 0.0, recorded_wait_time_ = 0.0;
  std::queue<std::array<double, 3>> search_commands_queue_;
  std::array<double, 3> search_command_;
  std::unique_ptr<Bbox2D> last_tracking_ = nullptr;
  float dist_error_ = 0.0f, orientation_error_ = 0.0f;

  void generateSearchCommands(float total_rotation, float search_radius,
                              float max_rotation_time,
                              bool enable_pause = false);
  void getFindTargetCmds(const int last_direction = 1);

private:
  RGBFollowerConfig config_;
  Velocities out_vel_;

  void trackTarget(const Bbox2D &tracking);
  //
};

} // namespace Control
} // namespace Kompass
