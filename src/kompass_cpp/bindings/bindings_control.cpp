#include "bindings.h"
#include <nanobind/stl/function.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/tuple.h>
#include <nanobind/stl/vector.h>

#include "controllers/dwa.h"
#include "controllers/follower.h"
#include "controllers/pid.h"
#include "controllers/pure_pursuit.h"
#include "controllers/rgb_follower.h"
#include "controllers/rgbd_follower.h"
#include "controllers/stanley.h"
#include "datatypes/control.h"
#include "datatypes/trajectory.h"

using namespace Kompass;

namespace {
// Private to file.

// The depth-image follower entries accept uint16 (mm) or float32 (m) arrays.
// Templated functions with per dtype bindings below.
template <typename DepthArray>
bool setInitialTrackingAtPixel(Control::RGBDFollower &self, const int pixel_x,
                               const int pixel_y, const DepthArray &depth,
                               const std::vector<Bbox2D> &boxes,
                               const float yaw) {
  const auto view = toDepthView(depth);
  py::gil_scoped_release release;
  return self.setInitialTracking(pixel_x, pixel_y, view, boxes, yaw);
}

template <typename DepthArray>
bool setInitialTrackingBox(Control::RGBDFollower &self, const DepthArray &depth,
                           const Bbox2D &target_box, const float yaw) {
  const auto view = toDepthView(depth);
  py::gil_scoped_release release;
  return self.setInitialTracking(view, target_box, yaw);
}

template <typename DepthArray>
Control::TrajSearchResult getTrackingCtrlDepth(Control::RGBDFollower &self,
                                               const DepthArray &depth,
                                               const std::vector<Bbox2D> &boxes,
                                               const Control::Velocity2D &vel) {
  const auto view = toDepthView(depth);
  // Solve without the GIL (argument conversions above still held it)
  py::gil_scoped_release release;
  return self.getTrackingCtrl(view, boxes, vel);
}

// Point-cloud variants that take the raw byte buffer (zero-copy), plus the
// layout integers
bool setInitialTrackingAtPixelCloud(Control::RGBDFollower &self,
                                    const int pixel_x, const int pixel_y,
                                    const ByteArray &data, int point_step,
                                    int row_step, int height, int width,
                                    int x_offset, int y_offset, int z_offset,
                                    const std::vector<Bbox2D> &boxes,
                                    const float yaw) {
  const PointCloudView view{toSpan(data), point_step, row_step, height,
                            width,        x_offset,   y_offset, z_offset};
  py::gil_scoped_release release;
  return self.setInitialTracking(pixel_x, pixel_y, view, boxes, yaw);
}

bool setInitialTrackingBoxCloud(Control::RGBDFollower &self,
                                const ByteArray &data, int point_step,
                                int row_step, int height, int width,
                                int x_offset, int y_offset, int z_offset,
                                const Bbox2D &target_box, const float yaw) {
  const PointCloudView view{toSpan(data), point_step, row_step, height,
                            width,        x_offset,   y_offset, z_offset};
  py::gil_scoped_release release;
  return self.setInitialTracking(view, target_box, yaw);
}

Control::TrajSearchResult
getTrackingCtrlCloud(Control::RGBDFollower &self, const ByteArray &data,
                     int point_step, int row_step, int height, int width,
                     int x_offset, int y_offset, int z_offset,
                     const std::vector<Bbox2D> &boxes,
                     const Control::Velocity2D &vel) {
  const PointCloudView view{toSpan(data), point_step, row_step, height,
                            width,        x_offset,   y_offset, z_offset};
  py::gil_scoped_release release;
  return self.getTrackingCtrl(view, boxes, vel);
}
} // namespace

// Control bindings submodule
void bindings_control(py::module_ &m) {
  auto m_control = m.def_submodule("control", "Control module");

  py::enum_<Control::ControlType>(m_control, "ControlType")
      .value("ACKERMANN", Control::ControlType::ACKERMANN)
      .value("DIFFERENTIAL_DRIVE", Control::ControlType::DIFFERENTIAL_DRIVE)
      .value("OMNI", Control::ControlType::OMNI);

  // Limits setup
  py::class_<Control::LinearVelocityControlParams>(
      m_control, "LinearVelocityControlParams")
      .def(py::init<const Control::LinearVelocityControlParams &>())
      .def(py::init<double, double, double>(), py::arg("max_vel") = 0.0,
           py::arg("max_acc") = 0.0, py::arg("max_decel") = 0.0)
      .def_rw("max_vel", &Control::LinearVelocityControlParams::maxVel)
      .def_rw("max_acc", &Control::LinearVelocityControlParams::maxAcceleration)
      .def_rw("max_decel",
              &Control::LinearVelocityControlParams::maxDeceleration);

  py::class_<Control::AngularVelocityControlParams>(
      m_control, "AngularVelocityControlParams")
      .def(py::init<const Control::AngularVelocityControlParams &>())
      .def(py::init<double, double, double, double>(),
           py::arg("max_ang") = M_PI, py::arg("max_omega") = 0.0,
           py::arg("max_acc") = 0.0, py::arg("max_decel") = 0.0)
      .def_rw("max_ang", &Control::AngularVelocityControlParams::maxAngle)
      .def_rw("max_omega", &Control::AngularVelocityControlParams::maxOmega)
      .def_rw("max_acc",
              &Control::AngularVelocityControlParams::maxAcceleration)
      .def_rw("max_decel",
              &Control::AngularVelocityControlParams::maxDeceleration);

  py::class_<Control::ControlLimitsParams>(m_control, "ControlLimitsParams")
      .def(py::init<>())
      .def(py::init<Control::LinearVelocityControlParams &,
                    Control::LinearVelocityControlParams &,
                    Control::AngularVelocityControlParams &>(),
           py::arg("vel_x_ctr_params") = Control::LinearVelocityControlParams(),
           py::arg("vel_y_ctr_params") = Control::LinearVelocityControlParams(),
           py::arg("omega_ctr_params") =
               Control::AngularVelocityControlParams())
      .def_rw("linear_x_limits", &Control::ControlLimitsParams::velXParams)
      .def_rw("linear_y_limits", &Control::ControlLimitsParams::velYParams)
      .def_rw("angular_limits", &Control::ControlLimitsParams::omegaParams);

  py::class_<Control::Controller>(m_control, "Controller")
      .def(py::init<>())
      .def("set_linear_ctr_limits",
           &Control::Controller::setLinearControlLimits)
      .def("set_angular_ctr_limits",
           &Control::Controller::setAngularControlLimits)
      .def("set_ctr_type", &Control::Controller::setControlType)
      .def("set_current_velocity", &Control::Controller::setCurrentVelocity)
      .def("set_current_state", py::overload_cast<const Path::State &>(
                                    &Control::Controller::setCurrentState))
      .def("set_current_state",
           py::overload_cast<double, double, double, double>(
               &Control::Controller::setCurrentState))
      .def("get_ctr_type", &Control::Controller::getControlType)
      .def("get_control", &Control::Controller::getControl);

  py::class_<Control::Controller::ControllerParameters, Parameters>(
      m_control, "ControllerParameters")
      .def(py::init<>());

  py::class_<Control::Follower::FollowerParameters,
             Control::Controller::ControllerParameters>(m_control,
                                                        "FollowerParameters")
      .def(py::init<>());

  py::class_<Control::Follower, Control::Controller>(m_control, "Follower")
      .def(py::init<>())
      .def(py::init<Control::Follower::FollowerParameters>())
      .def("set_interpolation_type", &Control::Follower::setInterpolationType)
      .def("set_current_path", &Control::Follower::setCurrentPath,
           py::arg("path"), py::arg("interpolate") = true)
      .def("clear_current_path", &Control::Follower::clearCurrentPath)
      .def("is_goal_reached", &Control::Follower::isGoalReached)
      .def("get_vx_cmd", &Control::Follower::getLinearVelocityCmdX)
      .def("get_vy_cmd", &Control::Follower::getLinearVelocityCmdY)
      .def("get_omega_cmd", &Control::Follower::getAngularVelocityCmd)
      .def("get_steer_cmd", &Control::Follower::getSteeringAngleCmd)
      .def("get_tracked_target", &Control::Follower::getTrackedTarget)
      // NOTE: set_current_path/clear_current_path destroy the referenced Path,
      // so return by copy to ensure no dangling pointers on python side
      .def("get_current_path", &Control::Follower::getCurrentPath,
           py::rv_policy::copy)
      .def("get_path_length", &Control::Follower::getPathLength)
      .def("has_path", &Control::Follower::hasPath);

  py::enum_<Control::Controller::Result::Status>(m_control, "FollowingStatus")
      .value("GOAL_REACHED", Control::Controller::Result::Status::GOAL_REACHED)
      .value("LOOSING_GOAL", Control::Controller::Result::Status::LOOSING_GOAL)
      .value("COMMAND_FOUND",
             Control::Controller::Result::Status::COMMAND_FOUND)
      .value("NO_COMMAND_POSSIBLE",
             Control::Controller::Result::Status::NO_COMMAND_POSSIBLE);

  py::class_<Control::Controller::Result>(m_control, "FollowingResult")
      .def(py::init<>())
      .def_rw("status", &Control::Controller::Result::status)
      .def_rw("velocity_command",
              &Control::Controller::Result::velocity_command);

  py::class_<Control::Follower::Target>(m_control, "FollowingTarget")
      .def(py::init<>())
      .def_rw("segment_index", &Control::Follower::Target::segment_index)
      .def_rw("position_in_segment",
              &Control::Follower::Target::position_in_segment)
      .def_rw("movement", &Control::Follower::Target::movement)
      .def_rw("reverse", &Control::Follower::Target::reverse)
      .def_rw("lookahead", &Control::Follower::Target::lookahead)
      .def_rw("crosstrack_error", &Control::Follower::Target::crosstrack_error)
      .def_rw("heading_error", &Control::Follower::Target::heading_error);

  // CONTROL SUBMODULES
  py::class_<Control::Stanley::StanleyParameters,
             Control::Follower::FollowerParameters>(m_control,
                                                    "StanleyParameters")
      .def(py::init<>());

  py::class_<Control::Stanley, Control::Follower>(m_control, "Stanley")
      .def(py::init<>(), "Init Stanley follower with default parameters")
      .def(py::init<Control::Stanley::StanleyParameters>(),
           "Init Stanley follower with custom config")
      .def("compute_velocity_commands",
           &Control::Stanley::computeVelocityCommand,
           py::call_guard<py::gil_scoped_release>())
      .def("execute", &Control::Stanley::execute,
           py::call_guard<py::gil_scoped_release>())
      .def("set_robot_wheelbase", &Control::Stanley::setWheelBase);

  py::class_<Control::PID, Control::Controller>(m_control, "PID")
      .def(py::init<>(), "Init PID controller with default parameters")
      .def(py::init<double, double, double>(), py::arg("kp"), py::arg("ki"),
           py::arg("kd"), "Init PID controller with parameters")
      .def("compute", &Control::PID::compute, py::arg("target"),
           py::arg("current"), py::arg("dt"));

  // PurePursuit
  // Bind PurePursuitConfig
  py::class_<Control::PurePursuit::PurePursuitConfig,
             Control::Follower::FollowerParameters>(m_control,
                                                    "PurePursuitConfig")
      .def(py::init<>());

  // Bind PurePursuit
  py::class_<Control::PurePursuit, Control::Follower>(m_control, "PurePursuit")
      .def(py::init<const Control::ControlType &,
                    const Control::ControlLimitsParams &,
                    const CollisionChecker::ShapeType,
                    const std::vector<float> &, const Eigen::Vector3f &,
                    const Eigen::Vector4f &, double,
                    const Control::PurePursuit::PurePursuitConfig &>(),
           "Init PurePursuit follower with collision avoidance configuration",
           py::arg("control_type"), py::arg("control_limits"),
           py::arg("robot_shape_type"), py::arg("robot_dimensions"),
           py::arg("sensor_position_robot"), py::arg("sensor_rotation_robot"),
           py::arg("octree_res") = 0.1,
           py::arg("config") = Control::PurePursuit::PurePursuitConfig())
      .def("execute",
           (Control::Controller::Result (Control::PurePursuit::*)(
               const Path::State, const double))&Control::PurePursuit::execute,
           "Execute Pure Pursuit control step with state update",
           py::arg("current_position"), py::arg("delta_time"),
           py::call_guard<py::gil_scoped_release>())
      .def("execute",
           (Control::Controller::Result (Control::PurePursuit::*)(
               const double))&Control::PurePursuit::execute,
           "Execute Pure Pursuit control step (uses internal state)",
           py::arg("delta_time"), py::call_guard<py::gil_scoped_release>())
      .def(
          "execute",
          [](Control::PurePursuit &self, const double dt,
             const Control::LaserScan &scan) {
            return self.execute<Control::LaserScan>(dt, scan);
          },
          "Execute Pure Pursuit with LaserScan obstacle avoidance",
          py::arg("delta_time"), py::arg("laser_scan"),
          py::call_guard<py::gil_scoped_release>())
      .def(
          "execute",
          [](Control::PurePursuit &self, const double dt,
             Eigen::Ref<const RowMatrixX3f> cloud) {
            // Zero-copy reinterpret
            const auto points = toPointSpan(cloud);
            py::gil_scoped_release release;
            return self.execute<Kompass::Span<Path::Point>>(dt, points);
          },
          "Execute Pure Pursuit with PointCloud obstacle avoidance",
          py::arg("delta_time"), py::arg("point_cloud"));

  // Trajectory sampler control result
  py::class_<Control::TrajSearchResult>(m_control, "SamplingControlResult")
      .def(py::init<>())
      .def_rw("is_found", &Control::TrajSearchResult::isTrajFound)
      .def_rw("cost", &Control::TrajSearchResult::trajCost)
      .def_rw("trajectory", &Control::TrajSearchResult::trajectory);

  // Dynamic Window Local Planner
  py::class_<Control::CostEvaluator::TrajectoryCostsWeights, Parameters>(
      m_control, "TrajectoryCostWeights")
      .def(py::init<>());

  py::class_<Control::DWA, Control::Follower>(m_control, "DWA")
      .def(py::init<Control::ControlLimitsParams, Control::ControlType, double,
                    double, double, int, int, CollisionChecker::ShapeType,
                    std::vector<float>, const Eigen::Vector3f &,
                    const Eigen::Vector4f &, double,
                    Control::CostEvaluator::TrajectoryCostsWeights, int>(),
           py::arg("control_limits"), py::arg("control_type"),
           py::arg("time_step"), py::arg("prediction_horizon"),
           py::arg("control_horizon"), py::arg("max_linear_samples"),
           py::arg("max_angular_samples"), py::arg("robot_shape_type"),
           py::arg("robot_dimensions"), py::arg("sensor_position_robot"),
           py::arg("sensor_rotation_robot"), py::arg("octree_resolution"),
           py::arg("cost_weights"), py::arg("max_num_threads") = 1)

      .def(py::init<Control::TrajectorySampler::TrajectorySamplerParameters,
                    Control::ControlLimitsParams, Control::ControlType,
                    CollisionChecker::ShapeType, std::vector<float>,
                    const Eigen::Vector3f &, const Eigen::Vector4f &,
                    Control::CostEvaluator::TrajectoryCostsWeights, int>(),
           py::arg("config"), py::arg("control_limits"),
           py::arg("control_type"), py::arg("robot_shape_type"),
           py::arg("robot_dimensions"), py::arg("sensor_position_robot"),
           py::arg("sensor_rotation_robot"), py::arg("cost_weights"),
           py::arg("max_num_threads") = 1)
      .def("compute_velocity_commands",
           py::overload_cast<const Control::Velocity2D &,
                             const Control::LaserScan &>(
               &Control::DWA::computeVelocityCommandsSet<Control::LaserScan>),
           py::call_guard<py::gil_scoped_release>())
      .def("compute_velocity_commands",
           [](Control::DWA &self, const Control::Velocity2D &vel,
              Eigen::Ref<const RowMatrixX3f> cloud)
               -> Control::TrajSearchResult {
             // Zero-copy reinterpret
             const auto points = toPointSpan(cloud);
             py::gil_scoped_release release;
             return self.computeVelocityCommandsSet<Kompass::Span<Path::Point>>(
                 vel, points);
           })
      .def(
          "compute_velocity_commands",
          // Overload for direct laserscan arrays. 1 copy of ranges/angles
          // into the LaserScan, made after the GIL release
          [](Control::DWA &self, const Control::Velocity2D &vel,
             Eigen::Ref<const Eigen::VectorXf> ranges,
             Eigen::Ref<const Eigen::VectorXf> angles)
              -> Control::TrajSearchResult {
            py::gil_scoped_release release;
            const Control::LaserScan scan{Eigen::VectorXf(ranges),
                                          Eigen::VectorXf(angles)};
            return self.computeVelocityCommandsSet<Control::LaserScan>(vel,
                                                                       scan);
          },
          py::arg("vel"), py::arg("ranges"), py::arg("angles"))
      .def("add_custom_cost",
           &Control::DWA::addCustomCost) // Custom cost function for DWA planner
                                         // of type (f(Trajectory2D, Path::Path)
                                         // -> double)
      .def("get_debugging_samples", &Control::DWA::getDebuggingSamples)
      .def("debug_velocity_search",
           // Overload for Nx3 cartesian points
           [](Control::DWA &self, const Control::Velocity2D &vel,
              Eigen::Ref<const RowMatrixX3f> cloud, const bool drop) {
             // Zero-copy reinterpret
             const auto points = toPointSpan(cloud);
             py::gil_scoped_release release;
             return self.debugVelocitySearch<Kompass::Span<Path::Point>>(
                 vel, points, drop);
           })
      .def("debug_velocity_search",
           // Overload for LaserScan
           py::overload_cast<const Control::Velocity2D &,
                             const Control::LaserScan &, const bool &>(
               &Control::DWA::debugVelocitySearch<Control::LaserScan>),
           py::call_guard<py::gil_scoped_release>())
      .def("debug_velocity_search",
           // Overload for direct laserscan arrays. 1 copy of ranges/angles
           // into the LaserScan, made after the GIL release
           [](Control::DWA &self, const Control::Velocity2D &vel,
              Eigen::Ref<const Eigen::VectorXf> ranges,
              Eigen::Ref<const Eigen::VectorXf> angles, const bool drop) {
             py::gil_scoped_release release;
             const Control::LaserScan scan{Eigen::VectorXf(ranges),
                                           Eigen::VectorXf(angles)};
             return self.debugVelocitySearch<Control::LaserScan>(vel, scan,
                                                                 drop);
           })
      .def("set_resolution", &Control::DWA::resetOctreeResolution);

  // Vision Follower
  py::class_<Control::RGBFollower::RGBFollowerConfig, Parameters>(
      m_control, "RGBFollowerParameters")
      .def(py::init<>());

  py::class_<Control::RGBFollower>(m_control, "RGBFollower")
      .def(py::init<const Control::ControlType,
                    const Control::ControlLimitsParams,
                    const Control::RGBFollower::RGBFollowerConfig>(),
           py::arg("control_type"), py::arg("control_limits"),
           py::arg("config"))
      .def("reset_target", &Control::RGBFollower::resetTarget)
      .def("get_ctrl", &Control::RGBFollower::getCtrl)
      .def("get_errors", &Control::RGBFollower::getErrors)
      .def("run", &Control::RGBFollower::run, py::arg("detection") = py::none(),
           py::call_guard<py::gil_scoped_release>());

  // Vision DWA
  py::class_<Control::RGBDFollower::RGBDFollowerConfig,
             Control::RGBFollower::RGBFollowerConfig>(m_control,
                                                      "RGBDFollowerParameters")
      .def(py::init<>());

  py::class_<Control::RGBDFollower, Control::Follower>(m_control,
                                                       "RGBDFollower")
      .def(py::init<const Control::ControlType &,
                    const Control::ControlLimitsParams &,
                    const CollisionChecker::ShapeType &,
                    const std::vector<float> &, const Eigen::Vector3f &,
                    const Eigen::Vector4f &,
                    const Control::RGBDFollower::RGBDFollowerConfig &>(),
           py::arg("control_type"), py::arg("control_limits"),
           py::arg("robot_shape_type"), py::arg("robot_dimensions"),
           py::arg("vision_sensor_position_wrt_body"),
           py::arg("vision_sensor_rotation_wrt_body"),
           py::arg("config") = Control::RGBDFollower::RGBDFollowerConfig())
      .def("set_camera_intrinsics", &Control::RGBDFollower::setCameraIntrinsics,
           py::arg("focal_length_x"), py::arg("focal_length_y"),
           py::arg("principal_point_x"), py::arg("principal_point_y"))
      .def("set_initial_tracking",
           py::overload_cast<const int, const int, const std::vector<Bbox3D> &,
                             const float>(
               &Control::RGBDFollower::setInitialTracking),
           py::arg("pixel_x"), py::arg("pixel_y"), py::arg("detected_boxes_3d"),
           py::arg("robot_orientation") = 0.0)
      // Depth image binds as a zero-copy 2-D view (uint16 mm or float32 m,
      // C-contiguous); one typed overload per dtype, sharing one template
      .def("set_initial_tracking", &setInitialTrackingAtPixel<DepthArrayU16>,
           py::arg("pixel_x"), py::arg("pixel_y"),
           py::arg("aligned_depth_image"), py::arg("detected_boxes_2d"),
           py::arg("robot_orientation") = 0.0)
      .def("set_initial_tracking", &setInitialTrackingAtPixel<DepthArrayF32>,
           py::arg("pixel_x"), py::arg("pixel_y"),
           py::arg("aligned_depth_image"), py::arg("detected_boxes_2d"),
           py::arg("robot_orientation") = 0.0)
      .def("set_initial_tracking", &setInitialTrackingBox<DepthArrayU16>,
           py::arg("aligned_depth_image"), py::arg("target_box_2d"),
           py::arg("robot_orientation") = 0.0)
      .def("set_initial_tracking", &setInitialTrackingBox<DepthArrayF32>,
           py::arg("aligned_depth_image"), py::arg("target_box_2d"),
           py::arg("robot_orientation") = 0.0)
      .def("set_point_cloud_sensor",
           &Control::RGBDFollower::setPointCloudSensor, py::arg("sensor"),
           "Describes the point-cloud sensor: mount pose in the robot body "
           "frame and encoding of its x/y/z fields, the same SensorConfig the "
           "mapper takes. Identity mount with FLOAT32 fields until called.")
      // Point cloud variants
      .def("set_initial_tracking", &setInitialTrackingAtPixelCloud,
           py::arg("pixel_x"), py::arg("pixel_y"), py::arg("data"),
           py::arg("point_step"), py::arg("row_step"), py::arg("height"),
           py::arg("width"), py::arg("x_offset"), py::arg("y_offset"),
           py::arg("z_offset"), py::arg("detected_boxes_2d"),
           py::arg("robot_orientation") = 0.0)
      .def("set_initial_tracking", &setInitialTrackingBoxCloud, py::arg("data"),
           py::arg("point_step"), py::arg("row_step"), py::arg("height"),
           py::arg("width"), py::arg("x_offset"), py::arg("y_offset"),
           py::arg("z_offset"), py::arg("target_box_2d"),
           py::arg("robot_orientation") = 0.0)
      .def("get_errors", &Control::RGBDFollower::getErrors)
      // NOTE:The C++ class also inherits RGBFollower (nanobind doesn't cover
      // multiple inheritence), so the RGBFollower interface is re-bound here
      // through lambdas as follows
      .def(
          "reset_target",
          [](Control::RGBDFollower &self, const Bbox2D &tracking) {
            self.resetTarget(tracking);
          },
          py::arg("tracking"))
      .def("get_ctrl",
           [](const Control::RGBDFollower &self)
               -> const Control::TrajectoryVelocities2D & {
             return self.getCtrl();
           })
      .def(
          "run",
          [](Control::RGBDFollower &self,
             const std::optional<Bbox2D> &detection) {
            return self.run(detection);
          },
          py::arg("detection") = py::none(),
          py::call_guard<py::gil_scoped_release>())
      .def("get_tracking_ctrl",
           py::overload_cast<const std::vector<Bbox3D> &,
                             const Control::Velocity2D &>(
               &Control::RGBDFollower::getTrackingCtrl),
           py::arg("detected_boxes_3d"), py::arg("robot_velocity"),
           py::call_guard<py::gil_scoped_release>())
      // Depth Array variants
      .def("get_tracking_ctrl", &getTrackingCtrlDepth<DepthArrayU16>,
           py::arg("aligned_depth_image"), py::arg("detected_boxes_2d"),
           py::arg("robot_velocity"))
      .def("get_tracking_ctrl", &getTrackingCtrlDepth<DepthArrayF32>,
           py::arg("aligned_depth_image"), py::arg("detected_boxes_2d"),
           py::arg("robot_velocity"))
      // Point-cloud variant
      .def("get_tracking_ctrl", &getTrackingCtrlCloud, py::arg("data"),
           py::arg("point_step"), py::arg("row_step"), py::arg("height"),
           py::arg("width"), py::arg("x_offset"), py::arg("y_offset"),
           py::arg("z_offset"), py::arg("detected_boxes_2d"),
           py::arg("robot_velocity"));
}
