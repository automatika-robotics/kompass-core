#include <pybind11/eigen.h>
#include <pybind11/functional.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "controllers/follower.h"
#include "datatypes/control.h"
#include "datatypes/trajectory.h"
#include "controllers/vision_follower.h"
#include "controllers/pid.h"
#include "controllers/stanley.h"
#include "controllers/dwa.h"

namespace py = pybind11;
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
      .def(py::init<double, double, double>(), py::arg("max_vel") = 0.0,
           py::arg("max_acc") = 0.0, py::arg("max_decel") = 0.0)
      .def_readwrite("max_vel", &Control::LinearVelocityControlParams::maxVel)
      .def_readwrite("max_acc",
                     &Control::LinearVelocityControlParams::maxAcceleration)
      .def_readwrite("max_decel",
                     &Control::LinearVelocityControlParams::maxDeceleration);

  py::class_<Control::AngularVelocityControlParams>(
      m_control, "AngularVelocityControlParams")
      .def(py::init<double, double, double, double>(),
           py::arg("max_ang") = M_PI, py::arg("max_omega") = 0.0,
           py::arg("max_acc") = 0.0, py::arg("max_decel") = 0.0)
      .def_readwrite("max_steer_ang",
                     &Control::AngularVelocityControlParams::maxAngle)
      .def_readwrite("max_omega",
                     &Control::AngularVelocityControlParams::maxOmega)
      .def_readwrite("max_acc",
                     &Control::AngularVelocityControlParams::maxAcceleration)
      .def_readwrite("max_decel",
                     &Control::AngularVelocityControlParams::maxDeceleration);

  py::class_<Control::ControlLimitsParams>(m_control, "ControlLimitsParams")
      .def(py::init<Control::LinearVelocityControlParams,
                    Control::LinearVelocityControlParams,
                    Control::AngularVelocityControlParams>(),
           py::arg("vel_x_ctr_params") = Control::LinearVelocityControlParams(),
           py::arg("vel_y_ctr_params") = Control::LinearVelocityControlParams(),
           py::arg("omega_ctr_params") =
               Control::AngularVelocityControlParams())
      .def_readwrite("linear_x_limits",
                     &Control::ControlLimitsParams::velXParams)
      .def_readwrite("linear_y_limits",
                     &Control::ControlLimitsParams::velYParams)
      .def_readwrite("angular_limits",
                     &Control::ControlLimitsParams::omegaParams);

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
      .def("set_current_path", &Control::Follower::setCurrentPath)
      .def("is_goal_reached", &Control::Follower::isGoalReached)
      .def("get_vx_cmd", &Control::Follower::getLinearVelocityCmdX)
      .def("get_vy_cmd", &Control::Follower::getLinearVelocityCmdY)
      .def("get_omega_cmd", &Control::Follower::getAngularVelocityCmd)
      .def("get_steer_cmd", &Control::Follower::getSteeringAngleCmd)
      .def("get_tracked_target", &Control::Follower::getTrackedTarget,
           py::return_value_policy::reference_internal)
      .def("get_current_path", &Control::Follower::getCurrentPath,
           py::return_value_policy::reference_internal)
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
      .def_readwrite("status", &Control::Controller::Result::status)
      .def_readwrite("velocity_command",
                     &Control::Controller::Result::velocity_command);

  py::class_<Control::Follower::Target>(m_control, "FollowingTarget")
      .def(py::init<>())
      .def_readwrite("segment_index", &Control::Follower::Target::segment_index)
      .def_readwrite("position_in_segment",
                     &Control::Follower::Target::position_in_segment)
      .def_readwrite("movement", &Control::Follower::Target::movement)
      .def_readwrite("reverse", &Control::Follower::Target::reverse)
      .def_readwrite("lookahead", &Control::Follower::Target::lookahead)
      .def_readwrite("crosstrack_error",
                     &Control::Follower::Target::crosstrack_error)
      .def_readwrite("heading_error",
                     &Control::Follower::Target::heading_error);

  // CONTROL SUBMODULES
  py::class_<Control::Stanley::StanleyParameters,
             Control::Follower::FollowerParameters,
             Control::Controller::ControllerParameters>(m_control,
                                                        "StanleyParameters")
      .def(py::init<>());

  py::class_<Control::Stanley, Control::Follower, Control::Controller>(
      m_control, "Stanley")
      .def(py::init<>(), "Init Stanley follower with default parameters")
      .def(py::init<Control::Stanley::StanleyParameters>(),
           "Init Stanley follower with custom config")
      .def("compute_velocity_commands",
           &Control::Stanley::computeVelocityCommand,
           py::return_value_policy::reference_internal)
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
      .def_readwrite("is_found", &Control::TrajSearchResult::isTrajFound)
      .def_readwrite("cost", &Control::TrajSearchResult::trajCost)
      .def_readwrite("trajectory", &Control::TrajSearchResult::trajectory);

  // Dynamic Window Local Planner
  py::class_<Control::CostEvaluator::TrajectoryCostsWeights, Parameters>(
      m_control, "TrajectoryCostWeights")
      .def(py::init<>());

  // Custom cost function for DWA planner
  py::class_<Control::CostEvaluator::CustomCostFunction>(
      m_control, "TrajectoryCustomCostFunction")
      .def(py::init<std::function<double(const Control::Trajectory &,
                                         const Path::Path &)>>());

  py::class_<Control::DWA, Control::Follower, Control::Controller>(m_control,
                                                                   "DWA")
      .def(py::init(
               [](Control::ControlLimitsParams control_limits,
                  Control::ControlType control_type, double time_step,
                  double prediction_horizon, double control_horizon,
                  int max_linear_samples, int max_angular_samples,
                  const CollisionChecker::ShapeType robot_shape_type,
                  const std::vector<float> robot_dimensions,
                  const std::array<float, 3> &sensor_position_robot,
                  const std::array<float, 4> &sensor_rotation_robot,
                  const double octree_resolution,
                  Control::CostEvaluator::TrajectoryCostsWeights cost_weights,
                  const int max_num_threads) {
                 return new Control::DWA(
                     control_limits, control_type, time_step,
                     prediction_horizon, control_horizon, max_linear_samples,
                     max_angular_samples, robot_shape_type, robot_dimensions,
                     sensor_position_robot, sensor_rotation_robot,
                     octree_resolution, cost_weights, max_num_threads);
               }),
           py::arg("control_limits"), py::arg("control_type"),
           py::arg("time_step"), py::arg("prediction_horizon"),
           py::arg("control_horizon"), py::arg("max_linear_samples"),
           py::arg("max_angular_samples"), py::arg("robot_shape_type"),
           py::arg("robot_dimensions"), py::arg("sensor_position_robot"),
           py::arg("sensor_rotation_robot"), py::arg("octree_resolution") = 0.1,
           py::arg("cost_weights"), py::arg("max_num_threads") = 1)

      .def("compute_velocity_commands",
           py::overload_cast<const Control::Velocity &,
                             const Control::LaserScan &>(
               &Control::DWA::computeVelocityCommandsSet),
           py::return_value_policy::reference_internal)
      .def("compute_velocity_commands",
           py::overload_cast<const Control::Velocity &,
                             const std::vector<Control::Point3D> &>(
               &Control::DWA::computeVelocityCommandsSet),
           py::return_value_policy::reference_internal)
      .def("add_custom_cost", &Control::DWA::addCustomCost);

  // Vision Follower
  py::class_<Control::VisionFollower::VisionFollowerConfig,
             Control::Controller::ControllerParameters, Parameters>(
      m_control, "VisionFollowerParameters")
      .def(py::init<>());

  py::class_<Control::VisionFollower, Control::Controller>(m_control,
                                                           "VisionFollower")
      .def(py::init(
               [](const Control::ControlType control_type,
                  const Control::ControlLimitsParams control_limits,
                  const Control::VisionFollower::VisionFollowerConfig config) {
                 return new Control::VisionFollower(control_type,
                                                    control_limits, config);
               }),
           py::arg("control_type"), py::arg("control_limits"),
           py::arg("config"))

      .def("reset_target", &Control::VisionFollower::resetTarget)
      .def("get_ctrl", &Control::VisionFollower::getCtrl)
      .def("run", &Control::VisionFollower::run);
}

