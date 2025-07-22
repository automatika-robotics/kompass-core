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
                    const Eigen::Vector3f &, const Eigen::Vector4f &,
                    const float, const float, const float,
                    const std::vector<double> &, const float, const float,
                    const float>(),
           py::arg("robot_shape"), py::arg("robot_dimensions"),
           py::arg("sensor_position_body"), py::arg("sensor_rotation_body"),
           py::arg("critical_angle"), py::arg("critical_distance"),
           py::arg("slowdown_distance"), py::arg("scan_angles"),
           py::arg("max_height"), py::arg("min_height"), py::arg("range_max"))

      .def("check",
           py::overload_cast<const std::vector<double> &, bool>(
               &CriticalZoneChecker::check),
           py::arg("ranges"), py::arg("forward"))

      .def("check",
           py::overload_cast<const std::vector<int8_t> &, int, int, int, int,
                             float, float, float, bool>(
               &CriticalZoneChecker::check),
           py::arg("data"), py::arg("point_step"), py::arg("row_step"),
           py::arg("height"), py::arg("width"), py::arg("x_offset"),
           py::arg("y_offset"), py::arg("z_offset"), py::arg("forward"));

#if GPU
  bindings_utils_gpu(m_utils);
#endif
}
