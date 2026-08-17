#include "bindings.h"
#include "mapping/local_mapper_gpu.h"
#include "utils/critical_zone_check_gpu.h"
#include <nanobind/stl/vector.h>

using namespace Kompass;

// Mapping bindings submodule
void bindings_mapping_gpu(py::module_ &m) {
  py::class_<Mapping::LocalMapperGPU>(m, "LocalMapperGPU")
      .def(py::init<const int, const int, float,
                    const std::vector<SensorConfig> &, bool, int, float, float,
                    float, int>(),
           py::arg("grid_height"), py::arg("grid_width"), py::arg("resolution"),
           py::arg("sensor_configs"), py::arg("is_pointcloud"),
           py::arg("scan_size"), py::arg("max_height"), py::arg("min_height"),
           py::arg("range_max"), py::arg("max_points_per_line") = 32,
           "One SensorConfig per sensor. Laserscan input requires exactly one "
           "sensor (mount consumed as planar); pointcloud input accepts N "
           "sensors fused into one grid, with max/min height as a BODY-frame "
           "band shared by all sensors.")

      .def("scan_to_grid",
           py::overload_cast<Eigen::Ref<const Eigen::VectorXf>,
                             Eigen::Ref<const Eigen::VectorXf>>(
               &Mapping::LocalMapperGPU::scanToGrid),
           "Convert laser scan data to occupancy grid", py::arg("angles"),
           py::arg("ranges"), py::rv_policy::reference_internal)

      .def(
          "scan_to_grid",
          [](Mapping::LocalMapperGPU &self, ByteArray data, int point_step,
             int row_step, int height, int width, int x_offset, int y_offset,
             int z_offset) -> Eigen::MatrixXi & {
            py::gil_scoped_release release;
            return self.scanToGrid(toSpan(data), point_step, row_step, height,
                                   width, x_offset, y_offset, z_offset);
          },
          "Convert raw point cloud data to occupancy grid (zero-copy input)",
          py::arg("data"), py::arg("point_step"), py::arg("row_step"),
          py::arg("height"), py::arg("width"), py::arg("x_offset"),
          py::arg("y_offset"), py::arg("z_offset"),
          py::rv_policy::reference_internal)

      .def(
          "scan_to_grid",
          [](Mapping::LocalMapperGPU &self,
             py::sequence clouds) -> Eigen::MatrixXi & {
            std::vector<ByteArray> keepalive;
            auto views = extractCloudViews(clouds, keepalive);
            py::gil_scoped_release release;
            return self.scanToGrid(views);
          },
          "Fuse N point clouds into one occupancy grid (zero-copy input). "
          "clouds[i] pairs with sensor_configs[i]; each element is a dict "
          "(e.g. PointCloudData.asdict()) carrying data/point_step/row_step/"
          "height/width/x_offset/y_offset/z_offset (extra keys ignored); "
          "None entries are skipped.",
          py::arg("clouds"), py::rv_policy::reference_internal);
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
           py::overload_cast<Eigen::Ref<const Eigen::VectorXf>, const bool>(
               &CriticalZoneCheckerGPU::check),
           py::arg("ranges"), py::arg("forward"))

      .def(
          "check",
          [](CriticalZoneCheckerGPU &self, ByteArray data, int point_step,
             int row_step, int height, int width, int x_offset, int y_offset,
             int z_offset, bool forward) {
            py::gil_scoped_release release;
            return self.check(toSpan(data), point_step, row_step, height, width,
                              x_offset, y_offset, z_offset, forward);
          },
          py::arg("data"), py::arg("point_step"), py::arg("row_step"),
          py::arg("height"), py::arg("width"), py::arg("x_offset"),
          py::arg("y_offset"), py::arg("z_offset"), py::arg("forward"));
}
