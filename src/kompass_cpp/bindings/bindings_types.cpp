#include <nanobind/eigen/dense.h>
#include <nanobind/nanobind.h>
#include <nanobind/operators.h>
#include <nanobind/stl/array.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>
#include <vector>

#include "datatypes/control.h"
#include "datatypes/path.h"
#include "datatypes/tracking.h"
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

  // Path types
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
      .def("size", &Path::Path::getSize)
      .def("getIndex", &Path::Path::getIndex, py::arg("index"))
      .def("x", &Path::Path::getX)
      .def("y", &Path::Path::getY);

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
      .def_ro("x", &Control::TrajectoryPath::x,
              py::rv_policy::reference_internal)
      .def_ro("y", &Control::TrajectoryPath::y,
              py::rv_policy::reference_internal)
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
      .def_static("get", [](const std::string &key) {
        if (key == "CYLINDER")
          return CollisionChecker::ShapeType::CYLINDER;
        if (key == "BOX")
          return CollisionChecker::ShapeType::BOX;
        if (key == "SPHERE")
          return CollisionChecker::ShapeType::SPHERE;
        throw std::runtime_error("Invalid key");
      });

  // Vision types
  py::class_<Bbox2D>(m_types, "Bbox2D")
      .def(py::init<>())
      .def(py::init<const Bbox2D &>())
      .def(py::init<const Eigen::Vector2i &, const Eigen::Vector2i &,
                    const float, const std::string>(),
           py::arg("top_left_corner"), py::arg("size"),
           py::arg("timestamp") = 0.0, py::arg("label") = "")
      .def_rw("top_left_corner", &Bbox2D::top_corner)
      .def_rw("size", &Bbox2D::size)
      .def_rw("timestamp", &Bbox2D::timestamp)
      .def_rw("label", &Bbox2D::label)
      .def_rw("img_size", &Bbox2D::img_size)
      .def("set_vel", &Bbox2D::setVel)
      .def("set_img_size", &Bbox2D::setImgSize);

  py::class_<Bbox3D>(m_types, "Bbox3D")
      .def(py::init<>())
      .def(py::init<const Bbox3D &>())
      .def(py::init<const Eigen::Vector3f &, const Eigen::Vector3f &,
                    const Eigen::Vector2i &, const Eigen::Vector2i &,
                    const float, const std::string,
                    const std::vector<Eigen::Vector3f> &>(),
           py::arg("center"), py::arg("size"), py::arg("center_img_frame"),
           py::arg("size_img_frame"), py::arg("timestamp") = 0.0,
           py::arg("label") = "", py ::arg("pc_points") = py::list())
      .def_rw("center", &Bbox3D::center)
      .def_rw("size", &Bbox3D::size)
      .def_rw("center_img_frame", &Bbox3D::center_img_frame)
      .def_rw("size_img_frame", &Bbox3D::size_img_frame)
      .def_rw("pc_points", &Bbox3D::pc_points)
      .def_rw("timestamp", &Bbox3D::timestamp)
      .def_rw("label", &Bbox3D::label);
}
