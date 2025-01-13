#include <pybind11/eigen.h>
#include <pybind11/functional.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "mapping/local_mapper_gpu.h"

namespace py = pybind11;
using namespace Kompass;

// Mapping bindings submodule
void bindings_mapping_gpu(py::module_ &m) {
  py::class_<Mapping::LocalMapperGPU>(m, "LocalMapperGPU")
      .def(
          py::init<int, int, float, const Eigen::Vector3f &, float, int, int>(),
          py::arg("grid_height"), py::arg("grid_width"), py::arg("resolution"),
          py::arg("laserscan_position"), py::arg("laserscan_orientation"),
          py::arg("scan_size"), py::arg("max_points_per_line") = 32)

      .def("scan_to_grid", &Mapping::LocalMapper::scanToGrid,
           "Convert laser scan data to occupancy grid", py::arg("angles"),
           py::arg("ranges"), py::arg("grid_data"));
}
