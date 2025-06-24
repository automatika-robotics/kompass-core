#include <nanobind/eigen/dense.h>
#include <nanobind/nanobind.h>
#include <nanobind/stl/function.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/tuple.h>
#include <nanobind/stl/vector.h>

#include "controllers/dwa.h"
#include "controllers/follower.h"
#include "controllers/pid.h"
#include "controllers/rgb_follower.h"
#include "controllers/stanley.h"
#include "controllers/vision_dwa.h"
#include "datatypes/control.h"
#include "datatypes/trajectory.h"

namespace py = nanobind;
using namespace Kompass;

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
      .def_rw("max_steer_ang", &Control::AngularVelocityControlParams::maxAngle)
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
      .def("get_tracked_target", &Control::Follower::getTrackedTarget,
           py::rv_policy::reference_internal)
      .def("get_current_path", &Control::Follower::getCurrentPath,
           py::rv_policy::reference_internal)
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
           py::rv_policy::reference_internal)
      .def("execute", &Control::Stanley::execute)
      .def("set_robot_wheelbase", &Control::Stanley::setWheelBase);

  py::class_<Control::PID, Control::Controller>(m_control, "PID")
      .def(py::init<>(), "Init PID controller with default parameters")
      .def(py::init<double, double, double>(), py::arg("kp"), py::arg("ki"),
           py::arg("kd"), "Init PID controller with parameters")
      .def("compute", &Control::PID::compute, py::arg("target"),
           py::arg("current"), py::arg("dt"));

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
           py::rv_policy::reference_internal)
      .def("compute_velocity_commands",
           py::overload_cast<const Control::Velocity2D &,
                             const std::vector<Path::Point> &>(
               &Control::DWA::computeVelocityCommandsSet<
                   std::vector<Path::Point>>),
           py::rv_policy::reference_internal)
      .def("add_custom_cost",
           &Control::DWA::addCustomCost) // Custom cost function for DWA planner
                                         // of type (f(Trajectory2D, Path::Path)
                                         // -> double)
      .def("get_debugging_samples", &Control::DWA::getDebuggingSamples)
      .def("debug_velocity_search",
           // Overload for std::vector<Path::Point>
           py::overload_cast<const Control::Velocity2D &,
                             const std::vector<Path::Point> &, const bool &>(
               &Control::DWA::debugVelocitySearch<std::vector<Path::Point>>),
           py::call_guard<py::gil_scoped_release>())
      .def("debug_velocity_search",
           // Overload for LaserScan
           py::overload_cast<const Control::Velocity2D &,
                             const Control::LaserScan &, const bool &>(
               &Control::DWA::debugVelocitySearch<Control::LaserScan>),
           py::call_guard<py::gil_scoped_release>())
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
      .def("run", &Control::RGBFollower::run,
           py::arg("detection") = py::none());

  // Vision DWA
  py::class_<Control::VisionDWA::VisionDWAConfig,
             Control::RGBFollower::RGBFollowerConfig>(m_control,
                                                      "VisionDWAParameters")
      .def(py::init<>());

  py::class_<Control::VisionDWA, Control::DWA>(m_control, "VisionDWA")
      .def(py::init<const Control::ControlType &,
                    const Control::ControlLimitsParams &, const int, const int,
                    const CollisionChecker::ShapeType &,
                    const std::vector<float> &, const Eigen::Vector3f &,
                    const Eigen::Vector4f &, const Eigen::Vector3f &,
                    const Eigen::Vector4f &, const double,
                    const Control::CostEvaluator::TrajectoryCostsWeights &,
                    const int, const Control::VisionDWA::VisionDWAConfig &>(),
           py::arg("control_type"), py::arg("control_limits"),
           py::arg("max_linear_samples"), py::arg("max_angular_samples"),
           py::arg("robot_shape_type"), py::arg("robot_dimensions"),
           py::arg("proximity_sensor_position_wrt_body"),
           py::arg("proximity_sensor_rotation_wrt_body"),
           py::arg("vision_sensor_position_wrt_body"),
           py::arg("vision_sensor_rotation_wrt_body"), py::arg("octree_res"),
           py::arg("cost_weights"), py::arg("max_num_threads") = 1,
           py::arg("config") = Control::VisionDWA::VisionDWAConfig())
      .def("set_camera_intrinsics", &Control::VisionDWA::setCameraIntrinsics,
           py::arg("focal_length_x"), py::arg("focal_length_y"),
           py::arg("principal_point_x"), py::arg("principal_point_y"))
      .def("set_initial_tracking",
           py::overload_cast<const int, const int, const std::vector<Bbox3D> &,
                             const float>(
               &Control::VisionDWA::setInitialTracking),
           py::arg("pixel_x"), py::arg("pixel_y"), py::arg("detected_boxes_3d"),
           py::arg("robot_orientation") = 0.0)
      .def("set_initial_tracking",
           py::overload_cast<const int, const int,
                             const Eigen::MatrixX<unsigned short> &,
                             const std::vector<Bbox2D> &, const float>(
               &Control::VisionDWA::setInitialTracking),
           py::arg("pixel_x"), py::arg("pixel_y"),
           py::arg("aligned_depth_image"), py::arg("detected_boxes_2d"),
           py::arg("robot_orientation") = 0.0)
      .def("set_initial_tracking",
           py::overload_cast<const Eigen::MatrixX<unsigned short> &,
                             const Bbox2D &, const float>(
               &Control::VisionDWA::setInitialTracking),
           py::arg("aligned_depth_image"), py::arg("target_box_2d"),
           py::arg("robot_orientation") = 0.0)
      .def("get_errors", &Control::VisionDWA::getErrors)
      .def("get_tracking_ctrl",
           py::overload_cast<const Eigen::MatrixX<unsigned short> &,
                             const std::vector<Bbox2D> &,
                             const Control::Velocity2D &,
                             const std::vector<Eigen::Vector3f> &>(
               &Control::VisionDWA::getTrackingCtrl<
                   std::vector<Eigen::Vector3f>>),
           py::arg("aligned_depth_image"), py::arg("detected_boxes"),
           py::arg("robot_velocity"), py::arg("sensor_data"))
      .def("get_tracking_ctrl",
           py::overload_cast<const Eigen::MatrixX<unsigned short> &,
                             const std::vector<Bbox2D> &,
                             const Control::Velocity2D &,
                             const Control::LaserScan &>(
               &Control::VisionDWA::getTrackingCtrl<Control::LaserScan>),
           py::arg("aligned_depth_image"), py::arg("detected_boxes"),
           py::arg("robot_velocity"), py::arg("sensor_data"));
}
