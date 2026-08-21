#include "bindings.h"
#include "utils/critical_zone_check.h"
#include "utils/pointcloud.h"
#include <nanobind/stl/array.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/pair.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/tuple.h>
#include <nanobind/stl/vector.h>

using namespace Kompass;

py::ndarray<py::numpy, float, py::shape<-1, 3>, py::c_contig>
read_pcd_py(const std::string &filename) {
  // NOTE: The capsule/ndarray construction below is Python C-API and needs the
  // GIL (a call_guard on the def would segfault)
  auto result = [&] {
    // File I/O + parse without the GIL.
    py::gil_scoped_release release;
    return readPCD(filename);
  }();

  if (!result) {
    throw std::runtime_error("Failed to read PCD file: " + filename);
  }

  std::vector<std::array<float, 3>> points = std::move(*result);
  size_t n = points.size();

  // Raw float pointer into vector storage
  float *raw_ptr = reinterpret_cast<float *>(points.data());

  // Capsule takes ownership of the vector
  auto capsule = py::capsule(
      new std::vector<std::array<float, 3>>(std::move(points)),
      [](void *p) noexcept {
        delete reinterpret_cast<std::vector<std::array<float, 3>> *>(p);
      });

  py::ndarray<py::numpy, float, py::shape<-1, 3>, py::c_contig> arr(
      raw_ptr, {n, 3}, capsule);

  return arr;
}

// Utils submodule
void bindings_utils(py::module_ &m) {
  auto m_utils = m.def_submodule("utils", "KOMPASS CPP utilities module");

  py::class_<CriticalZoneChecker>(m_utils, "CriticalZoneChecker")
      .def(py::init<CriticalZoneChecker::InputType, CollisionChecker::ShapeType,
                    const std::vector<float> &,
                    const std::vector<SensorConfig> &, const float, const float,
                    const float, const float, const float, const float,
                    const std::vector<double> &>(),
           py::arg("input_type"), py::arg("robot_shape"),
           py::arg("robot_dimensions"), py::arg("sensor_configs"),
           py::arg("critical_angle"), py::arg("critical_distance"),
           py::arg("slowdown_distance"), py::arg("min_height"),
           py::arg("max_height"), py::arg("range_max"),
           py::arg("scan_angles") = std::vector<double>(),
           "One SensorConfig per sensor. Laserscan input requires exactly one "
           "sensor and non-empty scan_angles; pointcloud input accepts N "
           "sensors (min factor wins) with min/max height as a BODY-frame "
           "band shared by all sensors.")

      .def("check",
           py::overload_cast<Eigen::Ref<const Eigen::VectorXf>, const bool>(
               &CriticalZoneChecker::check),
           py::arg("ranges"), py::arg("forward"),
           py::call_guard<py::gil_scoped_release>())

      .def(
          "check",
          [](CriticalZoneChecker &self, const ByteArray &data, int point_step,
             int row_step, int height, int width, int x_offset, int y_offset,
             int z_offset, bool forward) {
            py::gil_scoped_release release;
            return self.check(toSpan(data), point_step, row_step, height, width,
                              x_offset, y_offset, z_offset, forward);
          },
          py::arg("data"), py::arg("point_step"), py::arg("row_step"),
          py::arg("height"), py::arg("width"), py::arg("x_offset"),
          py::arg("y_offset"), py::arg("z_offset"), py::arg("forward"))

      .def(
          "check",
          [](CriticalZoneChecker &self, py::sequence clouds, bool forward) {
            std::vector<ByteArray> keepalive;
            auto views = extractCloudViews(clouds, keepalive);
            py::gil_scoped_release release;
            return self.check(views, forward);
          },
          "Check N point clouds (zero-copy input) and return the minimum "
          "safety factor. clouds[i] pairs with sensor_configs[i]; each "
          "element is a dict (e.g. PointCloudData.asdict()) carrying "
          "data/point_step/row_step/height/width/x_offset/y_offset/z_offset "
          "(extra keys ignored); None entries are skipped.",
          py::arg("clouds"), py::arg("forward"))

      .def_prop_ro("num_sensors", [](const CriticalZoneChecker &self) {
        return self.numSensors();
      });

  // Overload using angle_step (Returns: tuple(ranges, angles) as float32
  // numpy arrays)
  m_utils.def(
      "pointcloud_to_laserscan_from_raw",
      [](const ByteArray &data, int point_step, int row_step, int height, int width,
         int x_offset, int y_offset, int z_offset, double max_range,
         double min_z, double max_z, double angle_step,
         const Eigen::Vector3f &position, const Eigen::Vector4f &rotation,
         PointFieldType cloud_field_type) {
        Eigen::VectorXf ranges_out;
        Eigen::VectorXf angles_out;
        {
          py::gil_scoped_release release;
          pointCloudToLaserScanFromRaw(
              PointCloudView{toSpan(data), point_step, row_step, height, width,
                             x_offset, y_offset, z_offset},
              cloud_field_type, getTransformation(rotation, position),
              max_range, min_z, max_z, angle_step, ranges_out, angles_out);
        }
        return std::make_tuple(std::move(ranges_out), std::move(angles_out));
      },
      py::arg("data"), py::arg("point_step"), py::arg("row_step"),
      py::arg("height"), py::arg("width"), py::arg("x_offset"),
      py::arg("y_offset"), py::arg("z_offset"), py::arg("max_range"),
      py::arg("min_z"), py::arg("max_z"), py::arg("angle_step"),
      py::arg("position") = Eigen::Vector3f(0.0f, 0.0f, 0.0f),
      py::arg("rotation") = Eigen::Vector4f(0.0f, 0.0f, 0.0f, 1.0f),
      py::arg("cloud_field_type") = PointFieldType::FLOAT32,
      "Converts raw PointCloud2 to ranges and angles using a specific angular "
      "step. An optional sensor mount pose (position + quaternion x,y,z,w) "
      "rotates points into body orientation and gates min_z/max_z on the "
      "body-frame height; the default identity mount reproduces the plain "
      "sensor-frame conversion. cloud_field_type selects the x/y/z field "
      "encoding (default FLOAT32).");

  // Overload using num_bins (Returns: ranges as a float32 numpy array)
  m_utils.def(
      "pointcloud_to_laserscan_from_raw",
      [](const ByteArray &data, int point_step, int row_step, int height, int width,
         int x_offset, int y_offset, int z_offset, double max_range,
         double min_z, double max_z, int num_bins,
         const Eigen::Vector3f &position, const Eigen::Vector4f &rotation,
         PointFieldType cloud_field_type) {
        Eigen::VectorXf ranges_out;
        {
          py::gil_scoped_release release;
          pointCloudToLaserScanFromRaw(
              PointCloudView{toSpan(data), point_step, row_step, height, width,
                             x_offset, y_offset, z_offset},
              cloud_field_type, getTransformation(rotation, position),
              max_range, min_z, max_z, num_bins, ranges_out);
        }
        return ranges_out;
      },
      py::arg("data"), py::arg("point_step"), py::arg("row_step"),
      py::arg("height"), py::arg("width"), py::arg("x_offset"),
      py::arg("y_offset"), py::arg("z_offset"), py::arg("max_range"),
      py::arg("min_z"), py::arg("max_z"), py::arg("num_bins"),
      py::arg("position") = Eigen::Vector3f(0.0f, 0.0f, 0.0f),
      py::arg("rotation") = Eigen::Vector4f(0.0f, 0.0f, 0.0f, 1.0f),
      py::arg("cloud_field_type") = PointFieldType::FLOAT32,
      "Converts raw PointCloud2 to ranges only, using a fixed number of bins. "
      "An optional sensor mount pose (position + quaternion x,y,z,w) rotates "
      "points into body orientation and gates min_z/max_z on the body-frame "
      "height; the default identity mount reproduces the plain sensor-frame "
      "conversion. cloud_field_type selects the x/y/z field encoding "
      "(default FLOAT32).");

  m_utils.def(
      "read_pcd", &read_pcd_py, py::arg("filename"),
      "Convert PCD file to a numpy array of points (zero-copy return).");

  m_utils.def("read_pcd_to_occupancy_grid", &readPCDToOccupancyGrid,
              py::arg("filename"), py::arg("grid_resolution"),
              py::arg("z_ground_limit"), py::arg("robot_height"),
              py::call_guard<py::gil_scoped_release>(),
              "Convert PCD file to an occupancy grid (zero-copy return).");

#if GPU
  bindings_utils_gpu(m_utils);
#endif
}
