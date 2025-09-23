#pragma once

#include "utils/logger.h"
#include <Eigen/Dense>
#include <algorithm>
#include <array>
#include <charconv> // for std::from_chars
#include <cstdint>
#include <cstring> // for std::memcpy
#include <fstream>
#include <iostream>
#include <optional>
#include <string>
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

inline bool is_space(char c) {
  // ASCII whitespace: space or [\t (9) .. \r (13)]
  return c == ' ' || (c >= '\t' && c <= '\r');
}

/**
 * @brief Reads a PCD (Point Cloud Data) file and extracts 3D points.
 *
 * This function parses a PCD file header to detect the number of points,
 * field layout, and data format (`ascii` or `binary`). It extracts the `x`,
 * `y`, and `z` fields for all points and stores them in a contiguous memory
 * buffer (3 floats per point).
 *
 * @param filename Path to the PCD file to read.
 *
 * @return std::optional<std::vector<std::array<float, 3>>>
 *   - Returns std::nullopt if the file cannot be opened, is malformed,
 *     missing required fields, or uses an unsupported DATA format.
 *
 * @note
 * - Only `ascii` and `binary` PCD formats are supported.
 * - Additional fields in the PCD file are ignored.
 *
 * @throws std::runtime_error if parsing fails due to an invalid file format.
 */
inline std::optional<std::vector<std::array<float, 3>>>
readPCD(const std::string &filename) {
  std::ifstream file(filename, std::ios::binary);
  if (!file.is_open()) {
    std::cerr << "Error: Could not open file " << filename << std::endl;
    return std::nullopt;
  }

  std::string line;
  size_t num_points = 0;
  std::string data_format;

  int x_idx = -1, y_idx = -1, z_idx = -1;
  size_t point_stride = 0;
  std::vector<int> field_sizes;
  std::vector<std::string> fields;

  // --- Header parsing ---
  while (std::getline(file, line)) {
    if (line.empty() || line[0] == '#')
      continue;

    std::string_view sv(line);
    auto first_space = sv.find(' ');
    if (first_space == std::string_view::npos)
      continue;
    std::string_view keyword = sv.substr(0, first_space);
    std::string_view rest = sv.substr(first_space + 1);

    if (keyword == "FIELDS") {
      size_t pos = 0;
      while (pos < rest.size()) {
        auto next = rest.find(' ', pos);
        if (next == std::string_view::npos)
          next = rest.size();
        std::string_view field = rest.substr(pos, next - pos);
        if (field == "x")
          x_idx = fields.size();
        if (field == "y")
          y_idx = fields.size();
        if (field == "z")
          z_idx = fields.size();
        fields.emplace_back(field);
        pos = next + 1;
      }
    } else if (keyword == "SIZE") {
      size_t pos = 0;
      while (pos < rest.size()) {
        auto next = rest.find(' ', pos);
        if (next == std::string_view::npos)
          next = rest.size();
        std::string_view token = rest.substr(pos, next - pos);

        int value = 0;
        auto [ptr, ec] =
            std::from_chars(token.data(), token.data() + token.size(), value);
        if (ec == std::errc()) {
          field_sizes.push_back(value);
        }
        pos = next + 1;
      }
    } else if (keyword == "POINTS") {
      std::string_view token = rest;
      auto [ptr, ec] = std::from_chars(token.data(),
                                       token.data() + token.size(), num_points);
      if (ec != std::errc()) {
        std::cerr << "Error: Failed to parse POINTS value." << std::endl;
        return std::nullopt;
      }
    } else if (keyword == "DATA") {
      data_format = std::string(rest);
      break;
    }
  }

  if (x_idx == -1 || y_idx == -1 || z_idx == -1) {
    std::cerr << "Error: PCD file must contain 'x', 'y', and 'z' fields."
              << std::endl;
    return std::nullopt;
  }

  // Compute offsets
  size_t x_offset = 0, y_offset = 0, z_offset = 0;
  if (!data_format.empty() && data_format != "ascii") {
    if (fields.size() != field_sizes.size()) {
      std::cerr << "Error: FIELDS and SIZE do not match." << std::endl;
      return std::nullopt;
    }
    for (size_t i = 0; i < fields.size(); ++i) {
      if (i < x_idx)
        x_offset += field_sizes[i];
      if (i < y_idx)
        y_offset += field_sizes[i];
      if (i < z_idx)
        z_offset += field_sizes[i];
      point_stride += field_sizes[i];
    }
  }

  std::vector<std::array<float, 3>> points(num_points); // pre-sized vector

  // --- Data reading ---
  if (data_format == "ascii") {
    std::string ascii_block((std::istreambuf_iterator<char>(file)),
                            std::istreambuf_iterator<char>());
    const char *ptr = ascii_block.data();
    const char *end = ascii_block.data() + ascii_block.size();

    for (size_t i = 0; i < num_points; ++i) {
      float x = 0, y = 0, z = 0;

      for (float *f : {&x, &y, &z}) {
        // skip whitespace
        ptr = std::find_if_not(ptr, end, is_space);

        // find end of number token
        const char *start = ptr;
        ptr = std::find_if(ptr, end, is_space);

        std::from_chars(start, ptr, *f);
      }

      points[i] = std::array{x, y, z};
    }
  } else if (data_format == "binary") {
    std::vector<char> buffer(num_points * point_stride);
    file.read(buffer.data(), buffer.size());

    if (file.gcount() != static_cast<std::streamsize>(buffer.size())) {
      std::cerr << "Error: Failed to read expected amount of binary data."
                << std::endl;
      return std::nullopt;
    }

    for (size_t i = 0; i < num_points; ++i) {
      char *point_start = buffer.data() + i * point_stride;

      float x, y, z;
      std::memcpy(&x, point_start + x_offset, sizeof(float));
      std::memcpy(&y, point_start + y_offset, sizeof(float));
      std::memcpy(&z, point_start + z_offset, sizeof(float));

      points[i] = std::array{x, y, z};
    }
  } else {
    std::cerr << "Error: Unsupported DATA format '" << data_format << "'."
              << std::endl;
    return std::nullopt;
  }

  return points;
}
