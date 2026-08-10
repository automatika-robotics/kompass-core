#include "test.h"
#include "utils/pointcloud.h"
#include <nlohmann/json.hpp>
#include <string>
#define BOOST_TEST_MODULE KOMPASS TESTS
#include "json_export.h"
#include <boost/dll/runtime_symbol_info.hpp> // for program_location
#include <boost/filesystem.hpp>
#include <boost/test/included/unit_test.hpp>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <random>
#include <vector>

using namespace Kompass;

using json = nlohmann::json;

/**
 * @brief Generates a point cloud representing a sphere centered at origin
 *
 * @param radius
 * @param num_points
 * @return std::vector<uint8_t>
 */
std::vector<uint8_t> generateSpherePointcloud(float radius, int num_points) {

  // Generate points on the sphere and pack them into a binary format
  std::vector<uint8_t> data;
  for (int i = 0; i < num_points; ++i) {
    float theta = acos(1 - 2.0f * i / (num_points - 1));
    float phi = sqrt(num_points * sin(theta)) * (2 * M_PI);
    float x = radius * sin(theta) * cos(phi);
    float y = radius * sin(theta) * sin(phi);
    float z = radius * cos(theta);

    // Pack the points into a binary format
    data.insert(data.end(), reinterpret_cast<const uint8_t *>(&x),
                reinterpret_cast<const uint8_t *>(&x) + sizeof(float));
    data.insert(data.end(), reinterpret_cast<const uint8_t *>(&y),
                reinterpret_cast<const uint8_t *>(&y) + sizeof(float));
    data.insert(data.end(), reinterpret_cast<const uint8_t *>(&z),
                reinterpret_cast<const uint8_t *>(&z) + sizeof(float));
  }
  return data;
}


/**
 * @brief Generates points on the surface of a cube with side length size,
 * centered at origin
 *
 * @param size
 * @param num_points
 * @return std::vector<uint8_t>
 */
std::vector<uint8_t> generateCubePointCloud(float size, int num_points) {

  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<float> dist(-size / 2.0, size / 2.0);
  std::uniform_real_distribution<float> face_choice(0.0, 1.0);

  std::vector<uint8_t> data;
  for (int i = 0; i < num_points; ++i) {
    // Randomly choose which face of the cube the point lies on
    float face = face_choice(gen);

    float x, y, z;

    if (face < 1.0 / 3.0) {
      // ±X face
      x = (face < 1.0 / 6.0) ? (size / 2.0) : (-size / 2.0);
      y = dist(gen);
      z = dist(gen);
    } else if (face < 2.0 / 3.0) {
      // ±Y face
      x = dist(gen);
      y = (face < 0.5) ? (size / 2.0) : (-size / 2.0);
      z = dist(gen);
    } else {
      // ±Z face
      x = dist(gen);
      y = dist(gen);
      z = (face < 5.0 / 6.0) ? (size / 2.0) : (-size / 2.0);
    }

    // Pack the points into a binary format
    data.insert(data.end(), reinterpret_cast<const uint8_t *>(&x),
                reinterpret_cast<const uint8_t *>(&x) + sizeof(float));
    data.insert(data.end(), reinterpret_cast<const uint8_t *>(&y),
                reinterpret_cast<const uint8_t *>(&y) + sizeof(float));
    data.insert(data.end(), reinterpret_cast<const uint8_t *>(&z),
                reinterpret_cast<const uint8_t *>(&z) + sizeof(float));
  }
  return data;
}

/**
 * @brief Runs the point cloud to laser scan conversion test and saves the results
 *
 * @param data
 * @param width
 * @param shape_name
 * @param point_step
 * @param x_offset
 * @param y_offset
 * @param z_offset
 * @param height
 */
void run_test(const std::vector<uint8_t> &data, int width,
              std::string shape_name, int point_step = 12, int x_offset = 0,
              int y_offset = 4, int z_offset = 8, int height = 1) {
  // Laserscan data
  Eigen::VectorXf ranges, angles;
  double max_range = 10.0;
  double angle_step = 0.05;
  double min_z = 1.6, max_z = 1.8;

  // PointCloud data
  int row_step = point_step * width;

  // File names
  boost::filesystem::path executablePath = boost::dll::program_location();
  std::string file_location = executablePath.parent_path().string();
  std::string pointcloud_filename =
      file_location + "/" + shape_name + "_pointcloud";
  std::string scan_out_filename =
      file_location + "/" + shape_name + "_to_scan_test";

  // Save PointCloud to file for plotting
  std::ofstream ofs(pointcloud_filename + ".bin", std::ios::binary);
  ofs.write(reinterpret_cast<const char *>(data.data()), data.size());
  ofs.close();

  pointCloudToLaserScanFromRaw(
      PointCloudView{data, point_step, row_step, height, width, x_offset,
                     y_offset, z_offset},
      Eigen::Isometry3f::Identity(), max_range, min_z, max_z, angle_step,
      ranges, angles);

  saveScanToJson(std::vector<double>(ranges.begin(), ranges.end()),
                 std::vector<double>(angles.begin(), angles.end()),
                 scan_out_filename + ".json");

  std::string command =
      "python3 " + file_location + "/pointcloud_scan_plt.py --laserscan \"" +
      scan_out_filename + "\" --pointcloud \"" + pointcloud_filename + "\"";

  // Execute the Python script
  int res = system(command.c_str());
  if (res != 0)
    throw std::system_error(res, std::generic_category(),
                            "Python script failed with error code");
}

BOOST_AUTO_TEST_CASE(test_pointcloud_conversion_sphere) {
  // Create timer
  Timer time;

  // PointCloud data
  int num_points = 10000;
  float radius = 1.0f;

  // Sphere Points
  std::vector<uint8_t> data = generateSpherePointcloud(radius, num_points);

  run_test(data, num_points, "sphere");
}

BOOST_AUTO_TEST_CASE(test_pointcloud_conversion_cube) {
  // Create timer
  Timer time;

  // PointCloud data
  int num_points = 10000;
  float size = 3.0f;

  // Sphere Points
  std::vector<uint8_t> data = generateCubePointCloud(size, num_points);

  run_test(data, num_points, "cube");
}

/**
 * @brief Packs xyz points into a raw FLOAT32 buffer (point_step 12)
 */
static std::vector<uint8_t> packPoints(const std::vector<Eigen::Vector3f> &pts) {
  std::vector<uint8_t> data;
  data.reserve(pts.size() * 3 * sizeof(float));
  for (const auto &p : pts) {
    for (int k = 0; k < 3; ++k) {
      float v = p[k];
      data.insert(data.end(), reinterpret_cast<const uint8_t *>(&v),
                  reinterpret_cast<const uint8_t *>(&v) + sizeof(float));
    }
  }
  return data;
}

/**
 * The raw-buffer conversion under a full (tilted + translated) mount isometry
 * must agree with a straightforward per-point reference: rotate, gate on
 * body-frame z, bin by planar bearing, keep the min planar range per bin.
 */
BOOST_AUTO_TEST_CASE(test_pointcloud_conversion_body_frame_reference) {
  Timer time;

  std::mt19937 gen(42); // fixed seed: deterministic
  std::uniform_real_distribution<float> coord(-3.0f, 3.0f);

  const int num_points = 2000;
  std::vector<Eigen::Vector3f> pts;
  pts.reserve(num_points);
  for (int i = 0; i < num_points; ++i) {
    pts.push_back({coord(gen), coord(gen), coord(gen)});
  }
  const std::vector<uint8_t> data = packPoints(pts);

  // Mount with translation and roll/pitch/yaw all non-zero
  Eigen::Isometry3f tf = Eigen::Isometry3f::Identity();
  tf.translate(Eigen::Vector3f(0.4f, -0.2f, 0.35f));
  tf.rotate(eulerToRotationMatrix(-10.0f * M_PI / 180.0f,
                                  20.0f * M_PI / 180.0f,
                                  130.0f * M_PI / 180.0f));

  const double max_range = 20.0;
  const double min_z = 0.0, max_z = 1.5; // body frame
  const int num_bins = 360;

  Eigen::VectorXf ranges;
  pointCloudToLaserScanFromRaw(
      PointCloudView{data, 12, 12 * num_points, 1, num_points, 0, 4, 8}, tf,
      max_range, min_z, max_z, num_bins, ranges);

  // Reference implementation (same scalar math, written independently of the
  // buffer walk)
  Eigen::VectorXf ref(num_bins);
  ref.setConstant(static_cast<float>(max_range));
  const Eigen::Matrix3f rot = tf.rotation();
  const float t_z = tf.translation().z();
  const double two_pi = 2.0 * M_PI;
  for (const auto &p : pts) {
    if (p.x() * p.x() + p.y() * p.y() + p.z() * p.z() < 1e-6f)
      continue;
    const float xr = rot(0, 0) * p.x() + rot(0, 1) * p.y() + rot(0, 2) * p.z();
    const float yr = rot(1, 0) * p.x() + rot(1, 1) * p.y() + rot(1, 2) * p.z();
    const float zb =
        rot(2, 0) * p.x() + rot(2, 1) * p.y() + rot(2, 2) * p.z() + t_z;
    if (zb < min_z || zb > max_z)
      continue;
    const float range_sq = xr * xr + yr * yr;
    if (range_sq < 1e-6f)
      continue;
    double angle = std::atan2(yr, xr);
    if (angle < 0.0)
      angle += two_pi;
    const int bin = std::clamp(static_cast<int>((angle / two_pi) * num_bins),
                               0, num_bins - 1);
    ref[bin] = std::min(ref[bin], std::sqrt(range_sq));
  }

  BOOST_REQUIRE_EQUAL(ranges.size(), num_bins);
  int populated = 0;
  for (int b = 0; b < num_bins; ++b) {
    BOOST_CHECK_SMALL(ranges[b] - ref[b], 1e-5f);
    if (ref[b] < max_range)
      ++populated;
  }
  // Sanity: the band actually kept a meaningful subset of the cloud
  BOOST_CHECK_GT(populated, 50);
}

/**
 * A yaw + height mount must be equivalent to the identity conversion with the
 * z band shifted by the mount height and the bins rotated by the yaw
 * (the documented migration formula). Points sit at bin centers so float
 * rounding cannot flip bins.
 */
BOOST_AUTO_TEST_CASE(test_pointcloud_conversion_yaw_equivalence) {
  Timer time;

  const int num_bins = 360;
  const int yaw_bins = 90; // 90 deg = exactly 90 bins
  const float yaw = M_PI / 2.0f;
  const float sensor_z = 0.3f;
  const double two_pi = 2.0 * M_PI;

  std::mt19937 gen(7);
  std::uniform_real_distribution<float> range_dist(0.5f, 8.0f);
  std::uniform_real_distribution<float> z_dist(-0.5f, 1.0f);

  std::vector<Eigen::Vector3f> pts;
  for (int b = 0; b < num_bins; b += 3) { // every third bin, at its center
    const float theta = (b + 0.5f) * static_cast<float>(two_pi) / num_bins;
    const float r = range_dist(gen);
    pts.push_back({r * std::cos(theta), r * std::sin(theta), z_dist(gen)});
  }
  const std::vector<uint8_t> data = packPoints(pts);
  const int n = static_cast<int>(pts.size());
  const PointCloudView view{data, 12, 12 * n, 1, n, 0, 4, 8};

  const double max_range = 20.0;
  const double body_min_z = 0.0, body_max_z = 1.0;

  // Mounted conversion: body-frame band
  Eigen::Isometry3f tf = Eigen::Isometry3f::Identity();
  tf.translate(Eigen::Vector3f(0.0f, 0.0f, sensor_z));
  tf.rotate(eulerToRotationMatrix(0.0f, 0.0f, yaw));
  Eigen::VectorXf ranges_mounted;
  pointCloudToLaserScanFromRaw(view, tf, max_range, body_min_z, body_max_z,
                               num_bins, ranges_mounted);

  // Identity conversion: sensor-frame band shifted down by the mount height
  Eigen::VectorXf ranges_identity;
  pointCloudToLaserScanFromRaw(view, Eigen::Isometry3f::Identity(), max_range,
                               body_min_z - sensor_z, body_max_z - sensor_z,
                               num_bins, ranges_identity);

  for (int b = 0; b < num_bins; ++b) {
    const int rotated = (b + yaw_bins) % num_bins;
    BOOST_CHECK_SMALL(ranges_mounted[rotated] - ranges_identity[b], 1e-4f);
  }
}
