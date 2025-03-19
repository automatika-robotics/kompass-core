#include <pybind11/eigen.h>
#include <pybind11/functional.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "controllers/vision_follower.h"
#include "datatypes/control.h"
#include "datatypes/path.h"
#include "datatypes/trajectory.h"
#include "utils/collision_check.h"

namespace py = pybind11;
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
      .def_readwrite("x", &Path::State::x)
      .def_readwrite("y", &Path::State::y)
      .def_readwrite("yaw", &Path::State::yaw)
      .def_readwrite("speed", &Path::State::speed);

  py::class_<Path::Point>(m_types, "Point")
      .def(py::init<float, float, float>(), py::arg("x") = 0.0,
           py::arg("y") = 0.0, py::arg("z") = 0.0)
      .def_property("x", &Path::Point::x, &Path::Point::setX)
      .def_property("y", &Path::Point::y, &Path::Point::setY)
      .def_property("z", &Path::Point::z, &Path::Point::setZ);

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
  py::class_<Control::Velocity2D>(m_types, "ControlCmd")
      .def(py::init<float, float, float, float>(), py::arg("vx") = 0.0,
           py::arg("vy") = 0.0, py::arg("omega") = 0.0,
           py::arg("steer_ang") = 0.0)
      .def_property("vx", &Control::Velocity2D::vx, &Control::Velocity2D::setVx)
      .def_property("vy", &Control::Velocity2D::vy, &Control::Velocity2D::setVy)
      .def_property("omega", &Control::Velocity2D::omega,
                    &Control::Velocity2D::setOmega)
      .def_property("steer_ang", &Control::Velocity2D::steer_ang,
                    &Control::Velocity2D::setSteerAng)
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

  py::class_<Control::TrajectoryPath>(m_types, "TrajectoryPath")
      .def(py::init<>())
      .def_readonly("x", &Control::TrajectoryPath::x,
                    py::return_value_policy::reference_internal)
      .def_readonly("y", &Control::TrajectoryPath::y,
                    py::return_value_policy::reference_internal)
      .def_readonly("z", &Control::TrajectoryPath::z,
                    py::return_value_policy::reference_internal);

  py::class_<Control::TrajectoryVelocities2D>(m_types, "TrajectoryVelocities2D")
      .def(py::init<>())
      .def_readonly("vx", &Control::TrajectoryVelocities2D::vx,
                    py::return_value_policy::reference_internal)
      .def_readonly("vy", &Control::TrajectoryVelocities2D::vy,
                    py::return_value_policy::reference_internal)
      .def_readonly("omega", &Control::TrajectoryVelocities2D::omega,
                    py::return_value_policy::reference_internal);

  // Now expose TrajectoryPathSamples
  py::class_<Control::TrajectoryPathSamples>(m, "TrajectoryPathSamples")
      .def(py::init<size_t, size_t>())
      .def("size", &Control::TrajectoryPathSamples::size)
      .def("__iter__", [](const Control::TrajectoryPathSamples &self) {
        return py::make_iterator(self.begin(), self.end());
      });

  py::class_<Control::Trajectory2D>(m_types, "Trajectory")
      .def(py::init<>())
      .def_readonly("velocities", &Control::Trajectory2D::velocities)
      .def_readonly("path", &Control::Trajectory2D::path);

  py::class_<Control::LaserScan>(m_types, "LaserScan")
      .def(py::init<std::vector<double>, std::vector<double>>(),
           py::arg("ranges"), py::arg("angles"))
      .def_readwrite("ranges", &Control::LaserScan::ranges)
      .def_readwrite("angles", &Control::LaserScan::angles);

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
