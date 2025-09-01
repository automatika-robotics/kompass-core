#pragma once

#include "utils/logger.h"
#include <Eigen/Dense>
#include <algorithm>
#include <cstdint>
#include <cstring> // for std::memcpy
#include <vector>

/**
 * @brief Converts raw PointCloud2-style byte data to 2D LaserScan-like data.
 *
 * This function extracts 3D points (x, y, z) directly from a raw byte buffer
 * and projects them onto a 2D laser scan by computing the angle and distance
 * for each point. It divides the full 360° field of view into bins using
 * angle_step, and assigns the closest point to each bin.
 *
 * @param data         Raw point cloud data as a flattened byte array.
 * @param point_step   Number of bytes between successive points in a row.
 * @param row_step     Number of bytes between successive rows.
 * @param height       Number of rows in the point cloud.
 * @param width        Number of columns in the point cloud.
 * @param x_offset     Byte offset to the x-coordinate in a point.
 * @param y_offset     Byte offset to the y-coordinate in a point.
 * @param z_offset     Byte offset to the z-coordinate in a point.
 * @param max_range    Initial value and upper clipping range for distances.
 * @param min_z        Minimum acceptable Z value (inclusive).
 * @param max_z        Maximum acceptable Z value (inclusive). If negative,
 * disabled.
 * @param angle_step   Angular resolution (in radians) of each bin.
 * @param ranges_out   Output vector of minimum distances per bin.
 * @param angles_out   Output vector of bin angles in radians [0, 2π).
 *
 * @throws std::out_of_range If point offsets access memory out of bounds.
 */
inline void pointCloudToLaserScanFromRaw(
    const std::vector<int8_t> &data, const int point_step, const int row_step,
    const int height, const int width, const int x_offset, const int y_offset,
    const int z_offset, const double max_range, const double min_z,
    const double max_z, const double angle_step,
    std::vector<double> &ranges_out, std::vector<double> &angles_out) {

  const double two_pi = 2.0 * M_PI;
  const int num_bins = static_cast<int>(std::ceil(two_pi / angle_step));

  // Prefill angles and ranges
  angles_out.resize(num_bins);
  ranges_out.resize(num_bins);
  for (int i = 0; i < num_bins; ++i) {
    angles_out[i] = i * angle_step;
    ranges_out[i] = max_range;
  }

  // Iterate over raw points
  for (int row = 0; row < height; ++row) {
    for (int col = 0; col < row_step; col += point_step) {
      std::size_t point_start = row * row_step + col;

      std::size_t max_offset = point_start +
                               std::max({x_offset, y_offset, z_offset}) +
                               sizeof(float);
      if (max_offset > data.size()) {
        LOG_WARNING("Point offset out of bounds");
        continue;
      }

      float x, y, z;
      std::memcpy(&x, &data[point_start + x_offset], sizeof(float));
      std::memcpy(&y, &data[point_start + y_offset], sizeof(float));
      std::memcpy(&z, &data[point_start + z_offset], sizeof(float));

      // Z filtering
      if (z < min_z || (max_z >= 0.0 && z > max_z)) {
        continue;
      }

      double angle = std::atan2(y, x);
      if (angle < 0.0) {
        angle += two_pi;
      }

      int bin = static_cast<int>(angle / angle_step);
      bin = std::min(bin, num_bins - 1); // Clamp just in case

      double distance = std::sqrt(x * x + y * y);
      if (distance < ranges_out[bin]) {
        ranges_out[bin] = distance;
      }
    }
  }
}

/**
 * @brief Converts raw PointCloud2-style byte data to 2D LaserScan-like data.
 *
 * This function extracts 3D points (x, y, z) directly from a raw byte buffer
 * and projects them onto a 2D laser scan by computing the angle and distance
 * for each point. It divides the full 360° field of view into bins and assigns
 * the closest point to each bin. FOR INTERNAL USE: with pre-initialized output
 * vectors.
 *
 * @param data         Raw point cloud data as a flattened byte array.
 * @param point_step   Number of bytes between successive points in a row.
 * @param row_step     Number of bytes between successive rows.
 * @param height       Number of rows in the point cloud.
 * @param width        Number of columns in the point cloud.
 * @param x_offset     Byte offset to the x-coordinate in a point.
 * @param y_offset     Byte offset to the y-coordinate in a point.
 * @param z_offset     Byte offset to the z-coordinate in a point.
 * @param max_range    Initial value and upper clipping range for distances.
 * @param min_z        Minimum acceptable Z value (inclusive).
 * @param max_z        Maximum acceptable Z value (inclusive). If negative,
 * disabled.
 * @param num_bins     Number of rays in the laserscan.
 * @param ranges_out   Output vector of minimum distances per bin.
 *
 * @throws std::out_of_range If point offsets access memory out of bounds.
 */
inline void pointCloudToLaserScanFromRaw(
    const std::vector<int8_t> &data, const int point_step, const int row_step,
    const int height, const int width, const int x_offset, const int y_offset,
    const int z_offset, const double max_range, const double min_z,
    const double max_z, const int num_bins, std::vector<double> &ranges_out) {

  const double two_pi = 2.0 * M_PI;

  // reinitialize ranges
  ranges_out.assign(num_bins, max_range);

  // Iterate over raw points
  for (int row = 0; row < height; ++row) {
    for (int col = 0; col < row_step; col += point_step) {
      std::size_t point_start = row * row_step + col;

      std::size_t max_offset = point_start +
                               std::max({x_offset, y_offset, z_offset}) +
                               sizeof(float);
      if (max_offset > data.size()) {
        LOG_WARNING("Point offset out of bounds");
        continue;
      }

      float x, y, z;
      std::memcpy(&x, &data[point_start + x_offset], sizeof(float));
      std::memcpy(&y, &data[point_start + y_offset], sizeof(float));
      std::memcpy(&z, &data[point_start + z_offset], sizeof(float));

      // Z filtering
      if (z < min_z || (max_z >= 0.0 && z > max_z)) {
        continue;
      }

      double angle = std::atan2(y, x);
      if (angle < 0.0) {
        angle += two_pi;
      }

      int bin = static_cast<int>((angle / two_pi) * num_bins);
      bin = std::min(bin, num_bins - 1); // Clamp just in case

      double distance = std::sqrt(x * x + y * y);
      if (distance < ranges_out[bin]) {
        ranges_out[bin] = distance;
      }
    }
  }
}
