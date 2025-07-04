#include "utils/critical_zone_check.h"
#include <nanobind/eigen/dense.h>
#include <nanobind/nanobind.h>
#include <nanobind/stl/vector.h>

namespace py = nanobind;
using namespace Kompass;

#if GPU
void bindings_utils_gpu(py::module_ &);
#endif

// Utils submodule
void bindings_utils(py::module_ &m) {
  auto m_utils = m.def_submodule("utils", "KOMPASS CPP utilities module");

  py::class_<CriticalZoneChecker>(m_utils, "CriticalZoneChecker")
      .def(py::init<CollisionChecker::ShapeType, const std::vector<float> &,
                    const Eigen::Vector3f &, const Eigen::Vector4f &, const float,
                    const float, const float>(),
           py::arg("robot_shape"), py::arg("robot_dimensions"),
           py::arg("sensor_position_body"), py::arg("sensor_rotation_body"),
           py::arg("critical_angle"), py::arg("critical_distance"),
           py::arg("slowdown_distance"))
      .def("check", &CriticalZoneChecker::check, py::arg("ranges"),
           py::arg("angles"), py::arg("forward"));

#if GPU
  bindings_utils_gpu(m_utils);
#endif
}
