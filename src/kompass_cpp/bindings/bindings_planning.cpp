#include "planning/ompl.h"
#include <nanobind/eigen/dense.h>
#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/vector.h>

namespace py = nanobind;
using namespace Kompass;

// Utils submodule
void bindings_planning(py::module_ &m) {
  auto m_planning = m.def_submodule("planning", "KOMPASS CPP Planning Module");

  py::class_<Planning::OMPL2DGeometricPlanner>(m_planning,
                                               "OMPL2DGeometricPlanner")
      .def(py::init<const CollisionChecker::ShapeType &,
                    const std::vector<float> &,
                    const ompl::geometric::SimpleSetup &, const float>(),
           py::arg("robot_shape"), py::arg("robot_dimensions"),
           py::arg("ompl_setup"), py::arg("map_resolution") = 0.01)
      .def("setup_problem", &Planning::OMPL2DGeometricPlanner::setupProblem,
           py::arg("start_x"), py::arg("start_y"), py::arg("start_yaw"),
           py::arg("goal_x"), py::arg("goal_y"), py::arg("goal_yaw"),
           py::arg("map_3d"))
      .def("solve", &Planning::OMPL2DGeometricPlanner::solve,
           py::arg("planning_timeout") = 1.0)
      .def("get_solution", &Planning::OMPL2DGeometricPlanner::getPath)
      .def("set_space_bounds_from_map",
           &Planning::OMPL2DGeometricPlanner::setSpaceBoundsFromMap,
           py::arg("origin_x"), py::arg("origin_y"), py::arg("width"),
           py::arg("height"), py::arg("resolution"))
      .def("get_cost", &Planning::OMPL2DGeometricPlanner::getCost);
}
