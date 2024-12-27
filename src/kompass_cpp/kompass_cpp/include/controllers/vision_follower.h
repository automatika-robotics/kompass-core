#pragma once

#include "controller.h"
#include "datatypes/control.h"
#include "datatypes/parameter.h"
#include <memory>
#include <optional>
#include <queue>

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
    double depth;  // -1 is equivalent to none
  };

  class VisionFollowerConfig : public ControllerParameters {
  public:
    VisionFollowerConfig() : ControllerParameters() {
      addParameter("control_time_step", Parameter(0.1, 1e-4, 1e6));
      addParameter("control_horizon", Parameter(2, 1, 1000));
      addParameter("tolerance", Parameter(0.1, 1e-6, 1e3));
      addParameter("target_distance",
                   Parameter(-1.0, -1.0, 1e9)); // Use -1 for None
      addParameter("target_search_timeout", Parameter(30, 0, 1000));
      addParameter("target_search_radius", Parameter(0.5, 1e-4, 1e4));
      addParameter("target_search_pause", Parameter(0, 0, 1000));
      addParameter("rotation_multiple", Parameter(1.0, 1e-9, 1.0));
      addParameter("speed_depth_multiple", Parameter(0.7, 1e-9, 1.0));
      addParameter("min_vel", Parameter(0.01, 1e-9, 1e9));
      addParameter("enable_search", Parameter(false));
    }
    bool enable_search() const { return getParameter<bool>("enable_search"); }
    double control_time_step() const {
      return getParameter<double>("control_time_step");
    }
    int target_search_timeout() const {
      return getParameter<int>("target_search_timeout");
    }
    double target_search_radius() const {
      return getParameter<double>("target_search_radius");
    }
    int search_pause() const {
      return getParameter<int>("target_search_pause");
    }
    int control_horizon() const { return getParameter<int>("control_horizon"); }
    double tolerance() const { return getParameter<double>("tolerance"); }
    double target_distance() const {
      double val = getParameter<double>("target_distance");
      return val < 0 ? -1.0 : val; // Return -1 for None
    }
    void set_target_distance(double value) {
      setParameter("target_distance", value);
    }
    double alpha() const { return getParameter<double>("rotation_multiple"); }
    double beta() const { return getParameter<double>("speed_depth_multiple"); }
    double min_vel() const { return getParameter<double>("min_vel"); }
  };

  VisionFollower(const ControlType robotCtrlType,
                 const ControlLimitsParams ctrl_limits,
                 const VisionFollowerConfig config = VisionFollowerConfig());

  // Default Destructor
  ~VisionFollower() = default;

  void resetTarget(const TrackingData tracking);

  bool run(const std::optional<TrackingData> tracking);

  const Velocities getCtrl() const;

private:
  ControlType _ctrlType;
  ControlLimitsParams _ctrl_limits;
  VisionFollowerConfig _config;

  bool _rotate_in_place;
  double _target_ref_size = 0.0;
  Velocities _out_vel;
  bool _ctrl_available;
  double _recorded_search_time = 0.0;
  std::queue<std::array<double, 3>> _search_commands_queue;
  std::array<double, 3> _search_command;
  std::unique_ptr<TrackingData> _last_tracking = nullptr;

  void generate_search_commands(double total_rotation, double search_radius,
                                int max_rotation_steps,
                                bool enable_pause = false);
  std::array<double, 3> findTarget();
  void trackTarget(const TrackingData &tracking);
  //
};

} // namespace Control
} // namespace Kompass
