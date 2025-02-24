#include "utils/collision_check.h"
#include <array>
#include <pybind11/functional.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <vector>

namespace py = pybind11;
using namespace Kompass;

// Utils submodule
void bindings_utils(py::module_ &m) {
  auto m_utils = m.def_submodule("utils", "KOMPASS CPP utilities module");

  py::class_<CollisionChecker>(m_utils, "CollisionChecker")
      .def(py::init<CollisionChecker::ShapeType, const std::vector<float> &,
                    const std::array<float, 3> &, const std::array<float, 4> &,
                    double>(),
           R"pbdoc(
                Construct a new Collision Checker object

                Args:
                    robot_shape_type: Type of the robot shape geometry
                    robot_dimensions: Corresponding geometry dimensions
                    sensor_position_body: Position of the sensor w.r.t the robot body - Considered constant
                    sensor_rotation_body: Rotation of the sensor w.r.t the robot body - Considered constant
                    octree_resolution: Resolution of the constructed OctTree
            )pbdoc",
           py::arg("robot_shape"), py::arg("robot_dimensions"),
           py::arg("sensor_position_body"), py::arg("sensor_rotation_body"),
           py::arg("octree_resolution") = 0.1)
      .def("reset_octree_resolution", &CollisionChecker::resetOctreeResolution,
           py::arg("resolution"))
      .def("update_state", py::overload_cast<double, double, double>(
                               &CollisionChecker::updateState))
      .def("get_min_distance_laserscan",
           py::overload_cast<const std::vector<double> &,
                             const std::vector<double> &, double>(
               &CollisionChecker::getMinDistance),
           py::arg("ranges"), py::arg("angles"), py::arg("height") = 0.01);
}
