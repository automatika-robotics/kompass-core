#include "utils/critical_zone_check.h"
#include "utils/pointcloud.h"
#include <nanobind/eigen/dense.h>
#include <nanobind/nanobind.h>
#include <nanobind/stl/vector.h>
#include <nanobind/stl/tuple.h>

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

  m_utils.def(
      "pointcloud_to_laserscan_from_raw",
      [](const std::vector<int8_t> &data, int point_step, int row_step,
         int height, int width, int x_offset, int y_offset, int z_offset,
         double max_range, double min_z, double max_z, double angle_step) {
        std::vector<double> ranges_out;
        std::vector<double> angles_out;

        // Call the actual function
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
      "Converts raw PointCloud2 binary data to laser scan ranges and angles.");

#if GPU
  bindings_utils_gpu(m_utils);
#endif
}
