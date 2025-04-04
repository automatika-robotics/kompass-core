#include <nanobind/eigen/dense.h>
#include <nanobind/nanobind.h>
#include <nanobind/operators.h>
#include <nanobind/stl/array.h>
#include <nanobind/stl/vector.h>
#include <nanobind/stl/string.h>

#include "controllers/vision_follower.h"
#include "datatypes/control.h"
#include "datatypes/path.h"
#include "datatypes/trajectory.h"
#include "utils/collision_check.h"

namespace py = nanobind;
using namespace Kompass;

std::string printControlCmd(const Control::Velocity2D &velocity_command) {
  return "{" + std::to_string(velocity_command.vx()) + ", " +
         std::to_string(velocity_command.vy()) + ", " +
         std::to_string(velocity_command.omega()) + "})";
}

// Types submodule
void bindings_types(py::module_ &m) {
  auto m_types = m.def_submodule("types", "KOMPASS CPP data types module");

  py::enum_<Path::InterpolationType>(m_types, "PathInterpolationType")
      .value("LINEAR", Path::InterpolationType::LINEAR)
      .value("CUBIC_SPLINE", Path::InterpolationType::CUBIC_SPLINE)
      .value("HERMITE_SPLINE", Path::InterpolationType::HERMITE_SPLINE);

  py::class_<Path::State>(m_types, "State")
      .def(py::init<double, double, double, double>(), py::arg("x") = 0.0,
           py::arg("y") = 0.0, py::arg("yaw") = 0.0, py::arg("speed") = 0.0)
      .def_rw("x", &Path::State::x)
      .def_rw("y", &Path::State::y)
      .def_rw("yaw", &Path::State::yaw)
      .def_rw("speed", &Path::State::speed);

py::class_<Path::PathPosition>(m_types, "PathPosition")
    .def(py::init<>())
    .def_rw("segment_index", &Path::PathPosition::segment_index)
    .def_rw("segment_length", &Path::PathPosition::segment_length)
    .def_rw("parallel_distance", &Path::PathPosition::parallel_distance)
    .def_rw("normal_distance", &Path::PathPosition::normal_distance);

py::class_<Path::Path>(m_types, "Path")
    .def(py::init<const std::vector<Path::Point> &>(),
         py::arg("points") = std::vector<Path::Point>())
    .def("reached_end", &Path::Path::endReached)
    .def("get_total_length", &Path::Path::totalPathLength)
    .def_rw("points", &Path::Path::points);

// Velocity control command
py::class_<Control::Velocity2D>(m_types, "ControlCmd")
    .def(py::init<float, float, float, float>(), py::arg("vx") = 0.0,
         py::arg("vy") = 0.0, py::arg("omega") = 0.0,
         py::arg("steer_ang") = 0.0)
    .def_prop_rw("vx", &Control::Velocity2D::vx, &Control::Velocity2D::setVx)
    .def_prop_rw("vy", &Control::Velocity2D::vy, &Control::Velocity2D::setVy)
    .def_prop_rw("omega", &Control::Velocity2D::omega,
                 &Control::Velocity2D::setOmega)
    .def_prop_rw("steer_ang", &Control::Velocity2D::steer_ang,
                 &Control::Velocity2D::setSteerAng)
    .def("__str__", &printControlCmd);

py::class_<Control::Velocities>(m_types, "ControlCmdList")
    .def(py::init<>(), "Default constructor")
    .def(py::init<int>(), "Constructor with length", py::arg("length"))
    .def_rw("vx", &Control::Velocities::vx, "Speed on x-axis (m/s)")
    .def_rw("vy", &Control::Velocities::vy, "Speed on y-axis (m/s)")
    .def_rw("omega", &Control::Velocities::omega, "Angular velocity (rad/s)")
    .def_rw("length", &Control::Velocities::_length, "Length of the vectors");

py::class_<Control::TrajectoryPath>(m_types, "TrajectoryPath")
    .def(py::init<>())
    .def_ro("x", &Control::TrajectoryPath::x, py::rv_policy::reference_internal)
    .def_ro("y", &Control::TrajectoryPath::y, py::rv_policy::reference_internal)
    .def_ro("z", &Control::TrajectoryPath::z,
            py::rv_policy::reference_internal);

py::class_<Control::TrajectoryVelocities2D>(m_types, "TrajectoryVelocities2D")
    .def(py::init<>())
    .def_ro("vx", &Control::TrajectoryVelocities2D::vx,
            py::rv_policy::reference_internal)
    .def_ro("vy", &Control::TrajectoryVelocities2D::vy,
            py::rv_policy::reference_internal)
    .def_ro("omega", &Control::TrajectoryVelocities2D::omega,
            py::rv_policy::reference_internal);

py::class_<Control::Trajectory2D>(m_types, "Trajectory")
    .def(py::init<>())
    .def_ro("velocities", &Control::Trajectory2D::velocities)
    .def_ro("path", &Control::Trajectory2D::path);

py::class_<Control::LaserScan>(m_types, "LaserScan")
    .def(py::init<std::vector<double>, std::vector<double>>(),
         py::arg("ranges"), py::arg("angles"))
    .def_ro("ranges", &Control::LaserScan::ranges)
    .def_ro("angles", &Control::LaserScan::angles);

// For collisions detection
py::enum_<CollisionChecker::ShapeType>(m_types, "RobotGeometry")
    .value("CYLINDER", CollisionChecker::ShapeType::CYLINDER)
    .value("BOX", CollisionChecker::ShapeType::BOX)
    .value("SPHERE", CollisionChecker::ShapeType::SPHERE)
    .def("get", [](const std::string &key) {
      if (key == "CYLINDER")
        return CollisionChecker::ShapeType::CYLINDER;
      if (key == "BOX")
        return CollisionChecker::ShapeType::BOX;
      if (key == "SPHERE")
        return CollisionChecker::ShapeType::SPHERE;
      throw std::runtime_error("Invalid key");
    });

// Vision types
py::class_<Control::VisionFollower::TrackingData>(m_types, "TrackingData")
    .def(py::init<std::array<double, 2>, int, int, std::array<double, 2>,
                  double>(),
         py::arg("size_xy"), py::arg("img_width"), py::arg("img_height"),
         py::arg("center_xy"), py::arg("depth") = -1.0)
    .def_rw("size_xy", &Control::VisionFollower::TrackingData::size_xy)
    .def_rw("img_width", &Control::VisionFollower::TrackingData::img_width)
    .def_rw("img_height", &Control::VisionFollower::TrackingData::img_height)
    .def_rw("center_xy", &Control::VisionFollower::TrackingData::center_xy)
    .def_rw("depth", &Control::VisionFollower::TrackingData::depth);
}
