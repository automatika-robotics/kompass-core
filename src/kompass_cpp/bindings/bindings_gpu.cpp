#include "mapping/local_mapper_gpu.h"
#include "utils/critical_zone_check_gpu.h"
#include "utils/pointcloud.h"
#include <nanobind/eigen/dense.h>
#include <nanobind/nanobind.h>
#include <nanobind/stl/vector.h>

namespace py = nanobind;
using namespace Kompass;

// Mapping bindings submodule
void bindings_mapping_gpu(py::module_ &m) {
  py::class_<Mapping::LocalMapperGPU>(m, "LocalMapperGPU")
      .def(py::init<const int, const int, float, const Eigen::Vector3f &, float,
                    bool, int, float, float, float, float, int>(),
           py::arg("grid_height"), py::arg("grid_width"), py::arg("resolution"),
           py::arg("laserscan_position"), py::arg("laserscan_orientation"),
           py::arg("is_pointcloud"), py::arg("scan_size"),
           py::arg("angle_step"), py::arg("max_height"), py::arg("min_height"),
           py::arg("range_max"), py::arg("max_points_per_line") = 32)

      .def(py::init<const int, const int, float, const Eigen::Vector3f &, float,
                    bool, int, float, float, float, float, float, float, float,
                    float, int>(),
           py::arg("grid_height"), py::arg("grid_width"), py::arg("resolution"),
           py::arg("laserscan_position"), py::arg("laserscan_orientation"),
           py::arg("is_pointcloud"), py::arg("scan_size"), py::arg("p_prior"),
           py::arg("p_occupied"), py::arg("p_empty"), py::arg("range_sure"),
           py::arg("range_max"), py::arg("angle_step"), py::arg("max_height"),
           py::arg("min_height"), py::arg("max_points_per_line") = 32)

      .def("scan_to_grid",
           py::overload_cast<const std::vector<double> &,
                             const std::vector<double> &>(
               &Mapping::LocalMapperGPU::scanToGrid),
           "Convert laser scan data to occupancy grid", py::arg("angles"),
           py::arg("ranges"), py::rv_policy::reference_internal)

      .def("scan_to_grid",
           py::overload_cast<const std::vector<int8_t> &, int, int, int, int,
                             float, float, float>(
               &Mapping::LocalMapperGPU::scanToGrid),
           "Convert raw point cloud data to occupancy grid", py::arg("data"),
           py::arg("point_step"), py::arg("row_step"), py::arg("height"),
           py::arg("width"), py::arg("x_offset"), py::arg("y_offset"),
           py::arg("z_offset"), py::rv_policy::reference_internal)

      .def("scan_to_grid_baysian",
           py::overload_cast<const std::vector<double> &,
                             const std::vector<double> &,
                             const Eigen::Vector2f &, double>(
               &Mapping::LocalMapperGPU::scanToGridBaysian),
           "Bayesian (recursive Bayes) update from a laserscan; returns a "
           "discrete occupancy grid thresholded against the prior.",
           py::arg("angles"), py::arg("ranges"),
           py::arg("position_in_previous_pose"),
           py::arg("orientation_in_previous_pose"),
           py::rv_policy::reference_internal)

      .def("scan_to_grid_baysian",
           py::overload_cast<const std::vector<int8_t> &, int, int, int, int,
                             float, float, float, const Eigen::Vector2f &,
                             double>(
               &Mapping::LocalMapperGPU::scanToGridBaysian),
           "Bayesian (recursive Bayes) update from a raw PointCloud2 buffer; "
           "runs the on-device pointcloud→laserscan conversion and the same "
           "warp/Bayesian/threshold pipeline as the laserscan overload.",
           py::arg("data"), py::arg("point_step"), py::arg("row_step"),
           py::arg("height"), py::arg("width"), py::arg("x_offset"),
           py::arg("y_offset"), py::arg("z_offset"),
           py::arg("position_in_previous_pose"),
           py::arg("orientation_in_previous_pose"),
           py::rv_policy::reference_internal)

      .def("get_probabilities", &Mapping::LocalMapperGPU::getProbabilities,
           "Debug / tuning. Copies the current log-odds buffer back to host "
           "and returns the sigmoid-transformed probability grid (values in "
           "[0, 1]). Not on the hot path.",
           py::rv_policy::reference_internal);
}

// Utils bindings submodule
void bindings_utils_gpu(py::module_ &m) {

  py::class_<CriticalZoneCheckerGPU>(m, "CriticalZoneCheckerGPU")
      .def(py::init<CriticalZoneChecker::InputType, CollisionChecker::ShapeType,
                    const std::vector<float> &, const Eigen::Vector3f &,
                    const Eigen::Vector4f &, const float, const float,
                    const float, const std::vector<double> &, const float,
                    const float, const float, const PointFieldType>(),
           py::arg("input_type"), py::arg("robot_shape"),
           py::arg("robot_dimensions"), py::arg("sensor_position_body"),
           py::arg("sensor_rotation_body"), py::arg("critical_angle"),
           py::arg("critical_distance"), py::arg("slowdown_distance"),
           py::arg("scan_angles"), py::arg("min_height"), py::arg("max_height"),
           py::arg("range_max"),
           py::arg("cloud_field_type") = PointFieldType::FLOAT32)

      .def("check",
           py::overload_cast<const std::vector<double> &, bool>(
               &CriticalZoneCheckerGPU::check),
           py::arg("ranges"), py::arg("forward"))

      .def("check",
           py::overload_cast<const std::vector<int8_t> &, int, int, int, int,
                             int, int, int, bool>(
               &CriticalZoneCheckerGPU::check),
           py::arg("data"), py::arg("point_step"), py::arg("row_step"),
           py::arg("height"), py::arg("width"), py::arg("x_offset"),
           py::arg("y_offset"), py::arg("z_offset"), py::arg("forward"));
}
