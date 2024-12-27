#include <pybind11/eigen.h>
#include <pybind11/functional.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <vector>

#include "controllers/controller.h"
#include "controllers/dwa.h"
#include "controllers/pid.h"
#include "controllers/stanley.h"
#include "controllers/vision_follower.h"
#include "datatypes/control.h"
#include "datatypes/parameter.h"
#include "datatypes/path.h"
#include "datatypes/trajectory.h"
#include "mapping/local_mapper.h"
#include "utils/logger.h"

namespace py = pybind11;

using namespace Kompass;

// Define a variant type to hold different parameter value types
using ParamValue = std::variant<double, int, bool, std::string>;

std::string printControlCmd(const Control::Velocity &velocity_command) {
  return "{" + std::to_string(velocity_command.vx) + ", " +
         std::to_string(velocity_command.vy) + ", " +
         std::to_string(velocity_command.omega) + "})";
}

// Method to set parameter values based on dict instance
void set_parameters_from_dict(Parameters &params,
                              const py::object &attrs_instance) {
  auto attrs_dict = attrs_instance.cast<py::dict>();
  for (const auto &item : attrs_dict) {
    const std::string &name = py::str(item.first);
    py::handle value = item.second;
    try {
      auto it = params.parameters.find(name);
      if (it != params.parameters.end()) {
        if (py::isinstance<py::bool_>(value)) {
          it->second.setValue(value.cast<bool>());
        } else if (py::isinstance<py::float_>(value)) {
          it->second.setValue(value.cast<double>());
        } else if (py::isinstance<py::str>(value)) {
          it->second.setValue(py::str(value).cast<std::string>());
        } else if (py::isinstance<py::int_>(value)) {
          it->second.setValue(value.cast<int>());
        }
      }
    } catch (const std::exception &e) {
      PyErr_SetString(PyExc_RuntimeError, e.what());
      throw py::error_already_set();
    }
  }
}

PYBIND11_MODULE(kompass_cpp, m) {
  m.doc() = "Algorithms for robot path tracking and control";

  // Types submodule
  auto m_types = m.def_submodule("types", "KOMPASS CPP data types module");

  py::enum_<Path::InterpolationType>(m_types, "PathInterpolationType")
      .value("LINEAR", Path::InterpolationType::LINEAR)
      .value("CUBIC_SPLINE", Path::InterpolationType::CUBIC_SPLINE)
      .value("HERMITE_SPLINE", Path::InterpolationType::HERMITE_SPLINE);

  py::class_<Path::State>(m_types, "State")
      .def(py::init<double, double, double, double>(), py::arg("x") = 0.0,
           py::arg("y") = 0.0, py::arg("yaw") = 0.0, py::arg("speed") = 0.0)
      .def_readwrite("x", &Path::State::x)
      .def_readwrite("y", &Path::State::y)
      .def_readwrite("yaw", &Path::State::yaw)
      .def_readwrite("speed", &Path::State::speed);

  py::class_<Path::Point>(m_types, "Point")
      .def(py::init<double, double>(), py::arg("x") = 0.0, py::arg("y") = 0.0)
      .def_readwrite("x", &Path::Point::x)
      .def_readwrite("y", &Path::Point::y);

  py::class_<Path::PathPosition>(m_types, "PathPosition")
      .def(py::init<>())
      .def_readwrite("segment_index", &Path::PathPosition::segment_index)
      .def_readwrite("segment_length", &Path::PathPosition::segment_length)
      .def_readwrite("parallel_distance",
                     &Path::PathPosition::parallel_distance)
      .def_readwrite("normal_distance", &Path::PathPosition::normal_distance);

  py::class_<Path::Path>(m_types, "Path")
      .def(py::init<const std::vector<Path::Point> &>(),
           py::arg("points") = std::vector<Path::Point>())
      .def("reached_end", &Path::Path::endReached)
      .def("get_total_length", &Path::Path::totalPathLength)
      .def_readwrite("points", &Path::Path::points);

  // Velocity control command
  py::class_<Control::Velocity>(m_types, "ControlCmd")
      .def(py::init<double, double, double, double>(), py::arg("vx") = 0.0,
           py::arg("vy") = 0.0, py::arg("omega") = 0.0,
           py::arg("steer_ang") = 0.0)
      .def_readwrite("vx", &Control::Velocity::vx)
      .def_readwrite("vy", &Control::Velocity::vy)
      .def_readwrite("omega", &Control::Velocity::omega)
      .def_readwrite("steer_ang", &Control::Velocity::steer_ang)
      .def("__str__", &printControlCmd);

  py::class_<Control::Velocities>(m_types, "ControlCmdList")
      .def(py::init<>(), "Default constructor")
      .def(py::init<int>(), "Constructor with length", py::arg("length"))
      .def_readwrite("vx", &Control::Velocities::vx, "Speed on x-axis (m/s)")
      .def_readwrite("vy", &Control::Velocities::vy, "Speed on y-axis (m/s)")
      .def_readwrite("omega", &Control::Velocities::omega,
                     "Angular velocity (rad/s)")
      .def_readwrite("length", &Control::Velocities::_length,
                     "Length of the vectors");

  py::class_<Control::Trajectory>(m_types, "Trajectory")
      .def(py::init<>())
      .def_readwrite("velocity", &Control::Trajectory::velocity)
      .def_readwrite("path", &Control::Trajectory::path);

  py::class_<Control::LaserScan>(m_types, "LaserScan")
      .def(py::init<std::vector<double>, std::vector<double>>(),
           py::arg("ranges"), py::arg("angles"))
      .def_readwrite("ranges", &Control::LaserScan::ranges)
      .def_readwrite("angles", &Control::LaserScan::angles);

  py::class_<Control::Point3D>(m_types, "Point3D")
      .def(py::init<double, double, double>(), py::arg("x") = 0.0,
           py::arg("y") = 0.0, py::arg("z") = 0.0)
      .def_readwrite("x", &Control::Point3D::x)
      .def_readwrite("y", &Control::Point3D::y)
      .def_readwrite("z", &Control::Point3D::z);

  // For collisions detection
  py::enum_<CollisionChecker::ShapeType>(m_types, "RobotGeometry")
      .value("CYLINDER", CollisionChecker::ShapeType::CYLINDER)
      .value("BOX", CollisionChecker::ShapeType::BOX)
      .def("get", [](const std::string &key) {
        if (key == "CYLINDER")
          return CollisionChecker::ShapeType::CYLINDER;
        if (key == "BOX")
          return CollisionChecker::ShapeType::BOX;
        throw std::runtime_error("Invalid key");
      });

  // Vision types
  py::class_<Control::VisionFollower::TrackingData>(m_types, "TrackingData")
      .def(py::init<std::array<double, 2>, int, int, std::array<double, 2>,
                    double>(),
           py::arg("size_xy"), py::arg("img_width"), py::arg("img_height"),
           py::arg("center_xy"), py::arg("depth") = -1.0)
      .def_readwrite("size_xy", &Control::VisionFollower::TrackingData::size_xy)
      .def_readwrite("img_width",
                     &Control::VisionFollower::TrackingData::img_width)
      .def_readwrite("img_height",
                     &Control::VisionFollower::TrackingData::img_height)
      .def_readwrite("center_xy",
                     &Control::VisionFollower::TrackingData::center_xy)
      .def_readwrite("depth", &Control::VisionFollower::TrackingData::depth);

  // --------------------------------------------------------------------------------
  // Config parameters bindings submodule
  auto m_config = m.def_submodule("configure", "Configuration classes");

  py::class_<Parameter>(m_config, "ConfigParameter")
      .def(py::init<>())
      .def(py::init<int, int, int>())
      .def(py::init<double, double, double>())
      .def(py::init<std::string>())
      .def(py::init<bool>())
      .def("set_value", (void(Parameter::*)(int)) & Parameter::setValue<int>)
      .def("set_value",
           (void(Parameter::*)(double)) & Parameter::setValue<double>)
      .def("set_value",
           (void(Parameter::*)(std::string)) & Parameter::setValue<std::string>)
      .def("get_value_int", &Parameter::getValue<int>)
      .def("get_value_double", &Parameter::getValue<double>)
      .def("get_value_string", &Parameter::getValue<std::string>);

  py::class_<Parameters>(m_config, "ConfigParameters")
      .def(py::init<>())
      .def("add_parameter",
           (void(Parameters::*)(const std::string &, const Parameter &)) &
               Parameters::addParameter)
      .def("add_parameter", (void(Parameters::*)(const std::string &, bool)) &
                                Parameters::addParameter)
      .def("add_parameter",
           (void(Parameters::*)(const std::string &, const char *)) &
               Parameters::addParameter)
      .def("add_parameter",
           (void(Parameters::*)(const std::string &, int, int, int)) &
               Parameters::addParameter)
      .def("add_parameter",
           (void(Parameters::*)(const std::string &, double, double, double)) &
               Parameters::addParameter)
      .def("from_dict", &set_parameters_from_dict);

  // ------------------------------------------------------------------------------
  // Control bindings submodule
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

  // ------------------------------------------------------------------------------
  // Mapping bindings submodule
  auto m_mapping = m.def_submodule("mapping", "Local Mapping module");
  m_mapping.def("scan_to_grid", &Mapping::scanToGrid,
                "Convert laser scan data to occupancy grid", py::arg("angles"),
                py::arg("ranges"), py::arg("grid_data"),
                py::arg("central_point"), py::arg("resolution"),
                py::arg("laser_scan_position"),
                py::arg("laser_scan_orientation"),
                py::arg("max_points_per_line"), py::arg("max_num_threads") = 1);

  m_mapping.def(
      "scan_to_grid_baysian", &Mapping::scanToGridBaysian,
      "Convert laser scan data to occupancy grid, with baysian update",
      py::arg("angles"), py::arg("ranges"), py::arg("grid_data"),
      py::arg("grid_data_prob"), py::arg("central_point"),
      py::arg("resolution"), py::arg("laser_scan_position"),
      py::arg("laser_scan_orientation"), py::arg("previous_grid_data_prob"),
      py::arg("p_prior"), py::arg("p_empty"), py::arg("p_occupied"),
      py::arg("range_sure"), py::arg("range_max"), py::arg("wall_size"),
      py::arg("max_points_per_line"), py::arg("max_num_threads") = 1);

  m_mapping.def("local_to_grid", &Mapping::localToGrid,
                py::arg("pose_target_in_central"), py::arg("central_point"),
                py::arg("resolution"),
                "Convert a point from local coordinates frame of the grid to "
                "grid indices");

  m_mapping.def("get_previous_grid_in_current_pose",
                &Mapping::getPreviousGridInCurrentPose,
                py::arg("current_position_in_previous_pose"),
                py::arg("current_orientation_in_previous_pose"),
                py::arg("previous_grid_data"), py::arg("central_point"),
                py::arg("grid_width"), py::arg("grid_height"),
                py::arg("resolution"), py::arg("unknown_value"));

  py::enum_<Mapping::OccupancyType>(m_mapping, "OCCUPANCY_TYPE")
      .value("UNEXPLORED", Mapping::OccupancyType::UNEXPLORED)
      .value("EMPTY", Mapping::OccupancyType::EMPTY)
      .value("OCCUPIED", Mapping::OccupancyType::OCCUPIED);

  // ------------------------------------------------------------------------------
  // Utils bindings submodule
  py::enum_<LogLevel>(m, "LogLevel")
      .value("DEBUG", LogLevel::DEBUG)
      .value("INFO", LogLevel::INFO)
      .value("WARNING", LogLevel::WARNING)
      .value("ERROR", LogLevel::ERROR)
      .export_values();
  m.def("set_log_level", &setLogLevel, "Set the log level");
  m.def("set_log_file", &setLogFile, "Set the log file");
}
