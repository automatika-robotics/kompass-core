#include "mapping/local_mapper_gpu.h"
#include "utils/critical_zone_check_gpu.h"
#include <nanobind/eigen/dense.h>
#include <nanobind/nanobind.h>
#include <nanobind/stl/vector.h>

namespace py = nanobind;
using namespace Kompass;

// Mapping bindings submodule
void bindings_mapping_gpu(py::module_ &m) {
  py::class_<Mapping::LocalMapperGPU>(m, "LocalMapperGPU")
      .def(
          py::init<int, int, float, const Eigen::Vector3f &, float, int, int>(),
          py::arg("grid_height"), py::arg("grid_width"), py::arg("resolution"),
          py::arg("laserscan_position"), py::arg("laserscan_orientation"),
          py::arg("scan_size"), py::arg("max_points_per_line") = 32)

      .def("scan_to_grid", &Mapping::LocalMapperGPU::scanToGrid,
           "Convert laser scan data to occupancy grid", py::arg("angles"),
           py::arg("ranges"), py::rv_policy::reference_internal);
}

// Utils bindings submodule
void bindings_utils_gpu(py::module_ &m) {

  py::class_<CriticalZoneCheckerGPU>(m, "CriticalZoneCheckerGPU")
      .def(py::init<CollisionChecker::ShapeType, const std::vector<float> &,
                    const Eigen::Vector3f &, const Eigen::Vector4f &, float,
                    float, const std::vector<double> &>(),
           py::arg("robot_shape"), py::arg("robot_dimensions"),
           py::arg("sensor_position_body"), py::arg("sensor_rotation_body"),
           py::arg("critical_angle"), py::arg("critical_distance"),
           py::arg("scan_angles"))
      .def("check", &CriticalZoneChecker::check, py::arg("ranges"),
           py::arg("angles"), py::arg("forward"));
}
