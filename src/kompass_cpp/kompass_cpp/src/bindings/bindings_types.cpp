#include <pybind11/eigen.h>
#include <pybind11/functional.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <vector>

#include "datatypes/control.h"
#include "datatypes/path.h"
#include "datatypes/trajectory.h"
#include "controllers/vision_follower.h"
#include "datatypes/path.h"
#include "datatypes/trajectory.h"
#include "utils/collision_check.h"

namespace py = pybind11;
using namespace Kompass;

std::string printControlCmd(const Control::Velocity &velocity_command) {
  return "{" + std::to_string(velocity_command.vx) + ", " +
         std::to_string(velocity_command.vy) + ", " +
         std::to_string(velocity_command.omega) + "})";
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
}
