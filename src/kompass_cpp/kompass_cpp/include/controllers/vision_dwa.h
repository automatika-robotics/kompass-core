#pragma once

#include "datatypes/control.h"
#include "datatypes/parameter.h"
#include "dwa.h"
#include <Eigen/src/Core/Matrix.h>
#include <Eigen/src/Geometry/Quaternion.h>
#include <cmath>
#include <memory>
#include <queue>
#include <tuple>
#include <vector>

namespace Kompass {
namespace Control {

class VisionDWA : public DWA {
public:
  struct TrackingImgData {
    std::array<double, 2> size_xy; // width and height of the bounding box
    int img_width;
    int img_height;
    std::array<double, 2>
        center_xy; // x, y coordinates of the object center in image frame
    double depth;  // -1 is equivalent to none
  };

  class VisionDWAConfig : public ControllerParameters {
  public:
    VisionDWAConfig() : ControllerParameters() {
      addParameter("control_time_step", Parameter(0.1, 1e-4, 1e6));
      addParameter("control_horizon", Parameter(2, 1, 1000));
      addParameter("prediction_horizon", Parameter(10, 1, 1000));
      addParameter("tolerance", Parameter(0.01, 1e-6, 1e3));
      addParameter("target_distance",
                   Parameter(0.1, -1.0, 1e9)); // Use -1 for None
      addParameter(
          "target_orientation",
          Parameter(0.0, -M_PI, M_PI)); // target bearing angle with the target
      addParameter("target_wait_timeout", Parameter(30.0, 0.0, 1e3));
      addParameter("target_search_timeout", Parameter(30.0, 0.0, 1e3));
      addParameter("target_search_radius", Parameter(0.5, 1e-4, 1e4));
      addParameter("target_search_pause", Parameter(1.0, 0.0, 1e3));
      addParameter("rotation_gain", Parameter(0.5, 1e-2, 10.0));
      addParameter("speed_gain", Parameter(1.0, 1e-2, 10.0));
      addParameter("min_vel", Parameter(0.1, 1e-9, 1e9));
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
  };

  VisionDWA(const ControlType robotCtrlType,
            const ControlLimitsParams ctrlLimits, int maxLinearSamples,
            int maxAngularSamples,
            const CollisionChecker::ShapeType robotShapeType,
            const std::vector<float> robotDimensions,
            const std::array<float, 3> &sensor_position_body,
            const std::array<float, 4> &sensor_rotation_body,
            const double octreeRes,
            CostEvaluator::TrajectoryCostsWeights costWeights,
            const int maxNumThreads = 1,
            const VisionDWAConfig config = VisionDWAConfig());

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
   * @brief Get the Tracking Control Result based on object tracking and DWA sampling
   *
   * @tparam T
   * @param tracking_pose
   * @param current_vel
   * @param sensor_points
   * @return Control::TrajSearchResult
   */
  template <typename T>
  Control::TrajSearchResult getTrackingCtrl(const TrackedPose2D &tracking_pose,
                                            const Velocity2D &current_vel,
                                            const T &sensor_points);

private:
  ControlType _ctrlType;
  ControlLimitsParams _ctrl_limits;
  VisionDWAConfig _config;

  bool _rotate_in_place;
  Velocities _out_vel;
  std::unique_ptr<TrackingImgData> _last_tracking = nullptr;

  /**
   * @brief Get the Tracking Reference Trajectory Segment and if this segment is has collision
   *
   * @tparam T
   * @param tracking_pose
   * @param sensor_points
   * @return std::tuple<Trajectory2D, bool>
   */
  template <typename T>
  std::tuple<Trajectory2D, bool>
  getTrackingReferenceSegment(const TrackedPose2D &tracking_pose,
                              const T &sensor_points);
  //
};

} // namespace Control
} // namespace Kompass
