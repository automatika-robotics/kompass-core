#pragma once

#include "controller.h"
#include "datatypes/control.h"
#include "datatypes/parameter.h"
#include <optional>
#include <queue>
#include <vector>

namespace Kompass {
namespace Control {

class VisionFollower : public Controller {
public:
  struct TrackingData {
    std::array<double, 2> size_xy; // width and height of the bounding box
    int img_width;
    int img_height;
    std::array<double, 2>
        center_xy; // x, y coordinates of the object center in image frame
    double depth; // -1 is equivalent to none
  };

  class VisionFollowerConfig : public ControllerParameters {
  public:
    VisionFollowerConfig() : ControllerParameters() {
      addParameter("control_time_step", Parameter(0.1, 1e-4, 1e6));
      addParameter("control_horizon", Parameter(0.2, 1e-4, 1e6));
      addParameter("tolerance", Parameter(0.1, 1e-6, 1e3));
      addParameter("target_distance",
                   Parameter(-1.0, -1.0, 1e9)); // Use -1 for None
      addParameter("target_search_timeout", Parameter(30.0, 1e-4, 1e4));
      addParameter("target_search_radius", Parameter(0.5, 1e-4, 1e4));
      addParameter("alpha", Parameter(1.0, 1e-9, 1e9));
      addParameter("beta", Parameter(1.0, 1e-9, 1e9));
      addParameter("gamma", Parameter(1.0, 1e-9, 1e9));
      addParameter("min_vel", Parameter(0.01, 1e-9, 1e9));
    }
    double control_time_step() const {
      return getParameter<double>("control_time_step");
    }
    double target_search_timeout() const {
      return getParameter<double>("target_search_timeout");
    }
    double target_search_radius() const {
      return getParameter<double>("target_search_radius");
    }
    double control_horizon() const {
      return getParameter<double>("control_horizon");
    }
    double tolerance() const { return getParameter<double>("tolerance"); }
    double target_distance() const {
      double val = getParameter<double>("target_distance");
      return val < 0 ? -1.0 : val; // Return -1 for None
    }
    void set_target_distance(double value) {
      setParameter("target_distance", value);
    }
    double alpha() const { return getParameter<double>("alpha"); }
    double beta() const { return getParameter<double>("beta"); }
    double gamma() const { return getParameter<double>("gamma"); }
    double min_vel() const { return getParameter<double>("min_vel"); }
  };

  VisionFollower(const ControlType robotCtrlType,
                 const ControlLimitsParams ctrl_limits,
                 const VisionFollowerConfig config = VisionFollowerConfig());

  ~VisionFollower();

  void resetTarget(const TrackingData tracking);

  bool run(const std::optional<TrackingData> tracking);

  const Velocities getCtrl() const;

private:
  ControlType _ctrlType;
  ControlLimitsParams _ctrl_limits;
  VisionFollowerConfig _config;

  bool _rotate_in_place;
  double _target_ref_size = 0.0;
  std::vector<double> _time_steps;
  Velocities _out_vel;
  bool _ctrl_available;
  double _recorded_search_time = 0.0;
  std::queue<std::array<double, 3>> _search_commands_queue;
  std::array<double, 3> _search_command;
  std::unique_ptr<TrackingData> _last_tracking;

  void generate_search_commands(double total_rotation, double search_radius,
                                double max_rotation_time);
  std::array<double, 3> findTarget();
  void trackTarget(const TrackingData &tracking);
  //
};

} // namespace Control
} // namespace Kompass
