#include "utils/critical_zone_check.h"
#include "utils/pointcloud.h"
#include <nanobind/eigen/dense.h>
#include <nanobind/nanobind.h>
#include <nanobind/stl/array.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/pair.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/tuple.h>
#include <nanobind/stl/vector.h>

namespace py = nanobind;
using namespace Kompass;

py::ndarray<py::numpy, float, py::shape<-1, 3>, py::c_contig>
read_pcd_py(const std::string &filename) {
  auto result = readPCD(filename);

  if (!result) {
    throw std::runtime_error("Failed to read PCD file: " + filename);
  }

  std::vector<std::array<float, 3>> points = std::move(*result);
  size_t n = points.size();

  // Raw float pointer into vector storage
  float *raw_ptr = reinterpret_cast<float *>(points.data());

  // Capsule takes ownership of the vector
  auto capsule = py ::capsule(
      new std::vector<std::array<float, 3>>(std::move(points)),
      [](void *p) noexcept {
        delete reinterpret_cast<std::vector<std::array<float, 3>> *>(p);
      });

  py ::ndarray<py ::numpy, float, py ::shape<-1, 3>, py::c_contig> arr(
      raw_ptr, {n, 3}, capsule);

  return arr;
}

#if GPU
void bindings_utils_gpu(py::module_ &);
#endif

// Utils submodule
void bindings_utils(py::module_ &m) {
  auto m_utils = m.def_submodule("utils", "KOMPASS CPP utilities module");

  py::class_<CriticalZoneChecker>(m_utils, "CriticalZoneChecker")
      .def(py::init<CriticalZoneChecker::InputType, CollisionChecker::ShapeType,
                    const std::vector<float> &, const Eigen::Vector3f &,
                    const Eigen::Vector4f &, const float, const float,
                    const float, const std::vector<double> &, const float,
                    const float, const float>(),
           py::arg("input_type"), py::arg("robot_shape"),
           py::arg("robot_dimensions"), py::arg("sensor_position_body"),
           py::arg("sensor_rotation_body"), py::arg("critical_angle"),
           py::arg("critical_distance"), py::arg("slowdown_distance"),
           py::arg("scan_angles"), py::arg("max_height"), py::arg("min_height"),
           py::arg("range_max"))

      .def("check",
           py::overload_cast<const std::vector<double> &, bool>(
               &CriticalZoneChecker::check),
           py::arg("ranges"), py::arg("forward"))

      .def("check",
           py::overload_cast<const std::vector<int8_t> &, int, int, int, int,
                             int, int, int, bool>(&CriticalZoneChecker::check),
           py::arg("data"), py::arg("point_step"), py::arg("row_step"),
           py::arg("height"), py::arg("width"), py::arg("x_offset"),
           py::arg("y_offset"), py::arg("z_offset"), py::arg("forward"));

  // Overload using angle_step (Returns: tuple(ranges, angles))
  m_utils.def(
      "pointcloud_to_laserscan_from_raw",
      [](const std::vector<int8_t> &data, int point_step, int row_step,
         int height, int width, int x_offset, int y_offset, int z_offset,
         double max_range, double min_z, double max_z, double angle_step) {
        std::vector<double> ranges_out;
        std::vector<double> angles_out;

        // Call the overload function that takes angle_step
        pointCloudToLaserScanFromRaw(data, point_step, row_step, height, width,
                                     x_offset, y_offset, z_offset, max_range,
                                     min_z, max_z, angle_step, ranges_out,
                                     angles_out);

        // Return both vectors to Python
        return std::make_tuple(ranges_out, angles_out);
      },
      py::arg("data"), py::arg("point_step"), py::arg("row_step"),
      py::arg("height"), py::arg("width"), py::arg("x_offset"),
      py::arg("y_offset"), py::arg("z_offset"), py::arg("max_range"),
      py::arg("min_z"), py::arg("max_z"), py::arg("angle_step"),
      "Converts raw PointCloud2 to ranges and angles using a specific angular "
      "step.");

  // Overload using num_bins (Returns: list(ranges))
  m_utils.def(
      "pointcloud_to_laserscan_from_raw",
      [](const std::vector<int8_t> &data, int point_step, int row_step,
         int height, int width, int x_offset, int y_offset, int z_offset,
         double max_range, double min_z, double max_z, int num_bins) {
        std::vector<double> ranges_out;

        // Call the overload that takes num_bins
        pointCloudToLaserScanFromRaw(data, point_step, row_step, height, width,
                                     x_offset, y_offset, z_offset, max_range,
                                     min_z, max_z, num_bins, ranges_out);

        // Return ranges to Python
        return ranges_out;
      },
      py::arg("data"), py::arg("point_step"), py::arg("row_step"),
      py::arg("height"), py::arg("width"), py::arg("x_offset"),
      py::arg("y_offset"), py::arg("z_offset"), py::arg("max_range"),
      py::arg("min_z"), py::arg("max_z"), py::arg("num_bins"),
      "Converts raw PointCloud2 to ranges only, using a fixed number of bins.");

  m_utils.def(
      "read_pcd", &read_pcd_py, py::arg("filename"),
      "Convert PCD file to a numpy array of points (zero-copy return).");

  m_utils.def("read_pcd_to_occupancy_grid", &readPCDToOccupancyGrid,
              py::arg("filename"), py::arg("grid_resolution"),
              py::arg("z_ground_limit"), py::arg("robot_height"),
              "Convert PCD file to an occupancy grid (zero-copy return).");

#if GPU
  bindings_utils_gpu(m_utils);
#endif
}
