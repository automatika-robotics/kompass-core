#include "bindings.h"
#include <nanobind/stl/tuple.h>
#include <nanobind/stl/vector.h>

#include "mapping/local_mapper.h"

using namespace Kompass;

// Mapping bindings submodule
void bindings_mapping(py::module_ &m) {
  auto m_mapping = m.def_submodule("mapping", "Local Mapping module");
  py::enum_<Mapping::OccupancyType>(m_mapping, "OCCUPANCY_TYPE")
      .value("UNEXPLORED", Mapping::OccupancyType::UNEXPLORED)
      .value("EMPTY", Mapping::OccupancyType::EMPTY)
      .value("OCCUPIED", Mapping::OccupancyType::OCCUPIED);

  py::class_<Mapping::LocalMapper>(m_mapping, "LocalMapper")
      .def(py::init<const int, const int, float,
                    const std::vector<SensorConfig> &, bool, int, float, float,
                    float, int, int>(),
           py::arg("grid_height"), py::arg("grid_width"), py::arg("resolution"),
           py::arg("sensor_configs"), py::arg("is_pointcloud"),
           py::arg("scan_size"), py::arg("max_height"), py::arg("min_height"),
           py::arg("range_max"), py::arg("max_points_per_line") = 32,
           py::arg("max_num_threads") = 1,
           "One SensorConfig per sensor. Laserscan input requires exactly one "
           "sensor (mount consumed as planar); pointcloud input accepts N "
           "sensors fused into one grid, with max/min height as a BODY-frame "
           "band shared by all sensors.")

      .def(py::init<const int, const int, float,
                    const std::vector<SensorConfig> &, bool, int, float, float,
                    float, float, float, float, float, float, int, int>(),
           py::arg("grid_height"), py::arg("grid_width"), py::arg("resolution"),
           py::arg("sensor_configs"), py::arg("is_pointcloud"),
           py::arg("scan_size"), py::arg("p_prior"), py::arg("p_occupied"),
           py::arg("p_empty"), py::arg("range_sure"), py::arg("range_max"),
           py::arg("wall_size"), py::arg("max_height"), py::arg("min_height"),
           py::arg("max_points_per_line"), py::arg("max_num_threads") = 1)

      .def("scan_to_grid",
           py::overload_cast<Eigen::Ref<const Eigen::VectorXf>,
                             Eigen::Ref<const Eigen::VectorXf>>(
               &Mapping::LocalMapper::scanToGrid),
           "Convert laser scan data to occupancy grid", py::arg("angles"),
           py::arg("ranges"), py::rv_policy::reference_internal,
           py::call_guard<py::gil_scoped_release>())

      .def(
          "scan_to_grid",
          [](Mapping::LocalMapper &self, ByteArray data, int point_step,
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
          [](Mapping::LocalMapper &self,
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
          py::arg("clouds"), py::rv_policy::reference_internal)

      .def("scan_to_grid_bayesian",
           py::overload_cast<Eigen::Ref<const Eigen::VectorXf>,
                             Eigen::Ref<const Eigen::VectorXf>>(
               &Mapping::LocalMapper::scanToGridBayesian),
           "Convert laser scan data to occupancy grid, with bayesian update",
           py::arg("angles"), py::arg("ranges"),
           py::rv_policy::reference_internal,
           py::call_guard<py::gil_scoped_release>())

      .def(
          "scan_to_grid_bayesian",
          [](Mapping::LocalMapper &self, ByteArray data, int point_step,
             int row_step, int height, int width, int x_offset, int y_offset,
             int z_offset) {
            py::gil_scoped_release release;
            return self.scanToGridBayesian(toSpan(data), point_step, row_step,
                                           height, width, x_offset, y_offset,
                                           z_offset);
          },
          "Convert raw point cloud data to occupancy grid, with bayesian "
          "update (zero-copy input)",
          py::arg("data"), py::arg("point_step"), py::arg("row_step"),
          py::arg("height"), py::arg("width"), py::arg("x_offset"),
          py::arg("y_offset"), py::arg("z_offset"),
          py::rv_policy::reference_internal)

      .def("get_previous_grid_in_current_pose",
           &Mapping::LocalMapper::getPreviousGridInCurrentPose,
           py::arg("current_position_in_previous_pose"),
           py::arg("current_orientation_in_previous_pose"),
           py::call_guard<py::gil_scoped_release>());

#if GPU
  bindings_mapping_gpu(m_mapping);
#endif
}
