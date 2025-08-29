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
 * @return std::vector<int8_t>
 */
std::vector<int8_t> generateSpherePointcloud(float radius, int num_points) {

  // Generate points on the sphere and pack them into a binary format
  std::vector<int8_t> data;
  for (int i = 0; i < num_points; ++i) {
    float theta = acos(1 - 2.0f * i / (num_points - 1));
    float phi = sqrt(num_points * sin(theta)) * (2 * M_PI);
    float x = radius * sin(theta) * cos(phi);
    float y = radius * sin(theta) * sin(phi);
    float z = radius * cos(theta);

    // Pack the points into a binary format
    data.insert(data.end(), reinterpret_cast<const int8_t *>(&x),
                reinterpret_cast<const int8_t *>(&x) + sizeof(float));
    data.insert(data.end(), reinterpret_cast<const int8_t *>(&y),
                reinterpret_cast<const int8_t *>(&y) + sizeof(float));
    data.insert(data.end(), reinterpret_cast<const int8_t *>(&z),
                reinterpret_cast<const int8_t *>(&z) + sizeof(float));
  }
  return data;
}


/**
 * @brief Generates points on the surface of a cube with side length size,
 * centered at origin
 *
 * @param size
 * @param num_points
 * @return std::vector<int8_t>
 */
std::vector<int8_t> generateCubePointCloud(float size, int num_points) {

  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<float> dist(-size / 2.0, size / 2.0);
  std::uniform_real_distribution<float> face_choice(0.0, 1.0);

  std::vector<int8_t> data;
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
    data.insert(data.end(), reinterpret_cast<const int8_t *>(&x),
                reinterpret_cast<const int8_t *>(&x) + sizeof(float));
    data.insert(data.end(), reinterpret_cast<const int8_t *>(&y),
                reinterpret_cast<const int8_t *>(&y) + sizeof(float));
    data.insert(data.end(), reinterpret_cast<const int8_t *>(&z),
                reinterpret_cast<const int8_t *>(&z) + sizeof(float));
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
void run_test(const std::vector<int8_t> &data, int width,
              std::string shape_name, int point_step = 12, int x_offset = 0,
              int y_offset = 4, int z_offset = 8, int height = 1) {
  // Laserscan data
  std::vector<double> ranges, angles;
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

  pointCloudToLaserScanFromRaw(data, point_step, row_step, height, width,
                               x_offset, y_offset, z_offset, max_range, min_z,
                               max_z, angle_step, ranges, angles);

  saveScanToJson(ranges, angles, scan_out_filename + ".json");

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
  std::vector<int8_t> data = generateSpherePointcloud(radius, num_points);

  run_test(data, num_points, "sphere");
}

BOOST_AUTO_TEST_CASE(test_pointcloud_conversion_cube) {
  // Create timer
  Timer time;

  // PointCloud data
  int num_points = 10000;
  float size = 3.0f;

  // Sphere Points
  std::vector<int8_t> data = generateCubePointCloud(size, num_points);

  run_test(data, num_points, "cube");
}
