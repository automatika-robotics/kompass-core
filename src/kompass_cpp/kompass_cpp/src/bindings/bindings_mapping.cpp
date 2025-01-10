#include <pybind11/eigen.h>
#include <pybind11/functional.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "mapping/local_mapper.h"

namespace py = pybind11;
using namespace Kompass;

#if GPU
void bindings_mapping_gpu(py::module_ &);
#endif

// Mapping bindings submodule
void bindings_mapping(py::module_ &m) {
  auto m_mapping = m.def_submodule("mapping", "Local Mapping module");
  py::enum_<Mapping::OccupancyType>(m_mapping, "OCCUPANCY_TYPE")
      .value("UNEXPLORED", Mapping::OccupancyType::UNEXPLORED)
      .value("EMPTY", Mapping::OccupancyType::EMPTY)
      .value("OCCUPIED", Mapping::OccupancyType::OCCUPIED);

  py::class_<Mapping::LocalMapper>(m_mapping, "LocalMapper")
      .def(
          py::init<int, int, float, const Eigen::Vector3f &, float, int, int>(),
          py::arg("grid_height"), py::arg("grid_width"), py::arg("resolution"),
          py::arg("laserscan_position"), py::arg("laserscan_orientation"),
          py::arg("max_points_per_line"), py::arg("max_num_threads") = 1)

      .def(py::init<int, int, float, const Eigen::Vector3f &, float, float,
                    float, float, float, float, float, int, int>(),
           py::arg("grid_height"), py::arg("grid_width"), py::arg("resolution"),
           py::arg("laserscan_position"), py::arg("laserscan_orientation"),
           py::arg("p_prior"), py::arg("p_empty"), py::arg("p_occupied"),
           py::arg("range_sure"), py::arg("range_max"), py::arg("wall_size"),
           py::arg("max_points_per_line"), py::arg("max_num_threads") = 1)

      .def("scan_to_grid", &Mapping::LocalMapper::scanToGrid,
           "Convert laser scan data to occupancy grid", py::arg("angles"),
           py::arg("ranges"), py::arg("grid_data"))

      .def("scan_to_grid_baysian", &Mapping::LocalMapper::scanToGridBaysian,
           "Convert laser scan data to occupancy grid, with baysian update",
           py::arg("angles"), py::arg("ranges"), py::arg("grid_data"),
           py::arg("grid_data_prob"), py::arg("previous_grid_data_prob"))

      .def("get_previous_grid_in_current_pose",
           &Mapping::LocalMapper::getPreviousGridInCurrentPose,
           py::arg("current_position_in_previous_pose"),
           py::arg("current_orientation_in_previous_pose"),
           py::arg("previous_grid_data"), py::arg("unknown_value"));

#if GPU
  bindings_mapping_gpu(m_mapping);
#endif

}
