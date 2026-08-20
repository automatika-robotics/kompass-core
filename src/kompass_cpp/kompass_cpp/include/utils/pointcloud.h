#pragma once

#include "datatypes/sensors.h"
#include "datatypes/span.h"
#include "mapping/local_mapper.h"
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
#include <stdexcept>
#include <string>
#include <vector>

// --- Compatibility shim for libstdc++ versions that lack from_chars(float) ---
#if defined(__GLIBCXX__) &&                                                    \
    (!defined(__cpp_lib_to_chars) || __cpp_lib_to_chars < 201611L)
namespace std {
inline from_chars_result from_chars(const char *first, const char *last,
                                    float &value) noexcept {
  char *end;
  value = std::strtof(first, &end);
  return {end, (end == first ? std::errc::invalid_argument : std::errc())};
}
inline from_chars_result from_chars(const char *first, const char *last,
                                    double &value) noexcept {
  char *end;
  value = std::strtod(first, &end);
  return {end, (end == first ? std::errc::invalid_argument : std::errc())};
}
} // namespace std
#endif

/**
 * @brief Converts raw PointCloud2-style byte data to a 2D LaserScan-like
 * pseudo-scan around the sensor origin, expressed in BODY orientation.
 *
 * Each point is read from the raw buffer, rotated by the mount rotation, and
 * height-gated on its BODY-frame z (rotation applied plus the mount's z
 * offset). Surviving points are binned by their bearing around the sensor
 * origin in body orientation: bin angle = atan2 of the rotated planar
 * coordinates, range = planar distance from the sensor origin. The closest
 * point per bin wins. A consumer casting rays from this pseudo-scan should
 * use the sensor's planar mount position as ray origin and orientation 0
 *
 * @param cloud          View of the raw buffer + layout metadata.
 * @param field_type     Encoding of the x/y/z fields (dispatches
 * load_and_cast_val, same as the GPU kernel).
 * @param sensor_tf_body Sensor mount pose in the body frame (full isometry:
 * roll/pitch/yaw honored).
 * @param max_range      Initial value and upper clipping range for distances.
 * @param min_z_body     Minimum acceptable BODY-frame z (inclusive).
 * @param max_z_body     Maximum acceptable BODY-frame z (inclusive). Pass
 * `std::numeric_limits<double>::infinity()` for no upper bound. Note a
 * negative bound is a legitimate one, not a request to disable the gate.
 * @param num_bins       Number of uniform bins over [0, 2π).
 * @param ranges_out     Output vector of minimum distances per bin.
 *
 * @throws std::invalid_argument on negative field offsets (corrupt metadata).
 */
inline void
pointCloudToLaserScanFromRaw(const Kompass::PointCloudView &cloud,
                             const PointFieldType field_type,
                             const Eigen::Isometry3f &sensor_tf_body,
                             const double max_range, const double min_z_body,
                             const double max_z_body, const int num_bins,
                             Eigen::VectorXf &ranges_out) {
  // Fail loudly for a negative off-set (corrupted metadata)
  if (cloud.x_offset < 0 || cloud.y_offset < 0 || cloud.z_offset < 0) {
    throw std::invalid_argument(
        "Point field offsets (x/y/z) must be non-negative: malformed point "
        "cloud metadata");
  }
  const double two_pi = 2.0 * M_PI;

  // reinitialize ranges
  ranges_out.resize(num_bins);
  ranges_out.setConstant(static_cast<float>(max_range));

  // Hoist the mount transform. Only the rotation enters x/y (bearing and
  // radius stay relative to the sensor origin); the translation's z enters
  // the body-frame height gate.
  const Eigen::Matrix3f rot = sensor_tf_body.rotation();
  const float r00 = rot(0, 0), r01 = rot(0, 1), r02 = rot(0, 2);
  const float r10 = rot(1, 0), r11 = rot(1, 1), r12 = rot(1, 2);
  const float r20 = rot(2, 0), r21 = rot(2, 1), r22 = rot(2, 2);
  const float t_z = sensor_tf_body.translation().z();

  // Points at/beyond max_range can never win a bin, calculate its sqr
  const float max_range_sq =
      static_cast<float>(max_range) * static_cast<float>(max_range);

  const int elem_size = elementSizeOf(field_type);

  // Iterate over raw points. The inner walk is bounded by the row's payload
  // (width points). Organized clouds may pad rows, and padding bytes must not
  // be decoded as points (same as GPU kernel)
  const int row_bytes = cloud.width * cloud.point_step;
  for (int row = 0; row < cloud.height; ++row) {
    for (int col = 0; col < row_bytes; col += cloud.point_step) {
      std::size_t point_start = row * cloud.row_step + col;

      std::size_t max_offset =
          point_start +
          std::max({cloud.x_offset, cloud.y_offset, cloud.z_offset}) +
          elem_size;
      if (max_offset > cloud.data.size()) {
        LOG_WARNING("Point offset out of bounds");
        continue;
      }

      const float x = load_and_cast_val(cloud.data.data(),
                                        point_start + cloud.x_offset,
                                        field_type);
      const float y = load_and_cast_val(cloud.data.data(),
                                        point_start + cloud.y_offset,
                                        field_type);
      const float z = load_and_cast_val(cloud.data.data(),
                                        point_start + cloud.z_offset,
                                        field_type);

      // Reject non-finite points (NaN padding in organized clouds)
      if (!std::isfinite(x) || !std::isfinite(y) || !std::isfinite(z)) {
        continue;
      }

      // A point at the sensor origin carries no bearing. Reject in the SENSOR
      // frame, before rotation, so it can never end up mapped to the mount
      // position (inside the robot) downstream.
      if (x * x + y * y + z * z < 1e-6f) {
        continue;
      }

      // Rotate into body orientation and gate on body-frame height
      const float xr = r00 * x + r01 * y + r02 * z;
      const float yr = r10 * x + r11 * y + r12 * z;
      const float zb = r20 * x + r21 * y + r22 * z + t_z;

      if (zb < min_z_body || zb > max_z_body) {
        continue;
      }

      // No planar extent -> no bin (point straight above/below the sensor)
      const float range_sq = xr * xr + yr * yr;
      if (range_sq < 1e-6f) {
        continue;
      }
      // Beyond max_range -> can never win the per-bin min
      if (range_sq >= max_range_sq) {
        continue;
      }

      double angle = std::atan2(yr, xr);
      if (angle < 0.0) {
        angle += two_pi;
      }

      int bin = static_cast<int>((angle / two_pi) * num_bins);
      bin = std::clamp(bin, 0, num_bins - 1); // Clamp just in case

      const float distance = static_cast<float>(std::sqrt(range_sq));
      if (distance < ranges_out[bin]) {
        ranges_out[bin] = distance;
      }
    }
  }
}

/**
 * @brief angle_step variant of the conversion above, also producing the bin
 * angle labels.
 *
 * Derives `num_bins = ceil(2π / angle_step)` and delegates to the num_bins
 * conversion, so bins are uniform at 2π/num_bins. Marginally narrower than
 * the requested step when 2π is not an exact multiple of it. Labels are the
 * requested `i * angle_step` (bin starts).
 *
 * @param angles_out   Output vector of bin angles in radians [0, 2π). Only
 * refilled when the caller's buffer doesn't already hold labels for this
 * step (size and contents are both checked).
 */
inline void pointCloudToLaserScanFromRaw(
    const Kompass::PointCloudView &cloud, const PointFieldType field_type,
    const Eigen::Isometry3f &sensor_tf_body, const double max_range,
    const double min_z_body, const double max_z_body, const double angle_step,
    Eigen::VectorXf &ranges_out, Eigen::VectorXf &angles_out) {
  const double two_pi = 2.0 * M_PI;
  const int num_bins = static_cast<int>(std::ceil(two_pi / angle_step));

  // Prefill angles only when the caller's buffer doesn't already hold them.
  // Verify both the size and the contents i.e. bin 1 holds exactly
  // static_cast<float>(angle_step) when the buffer was filled by this
  // function with the same step
  if (angles_out.size() != num_bins ||
      (num_bins > 1 && angles_out[1] != static_cast<float>(angle_step))) {
    angles_out.resize(num_bins);
    for (int i = 0; i < num_bins; ++i) {
      angles_out[i] = static_cast<float>(i * angle_step);
    }
  }

  pointCloudToLaserScanFromRaw(cloud, field_type, sensor_tf_body, max_range,
                               min_z_body, max_z_body, num_bins, ranges_out);
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
      if ((int)i < x_idx)
        x_offset += field_sizes[i];
      if ((int)i < y_idx)
        y_offset += field_sizes[i];
      if ((int)i < z_idx)
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

/**
 * @brief Converts a PCD file to a 2D occupancy grid.
 *
 * This function reads a PCD (Point Cloud Data) file containing 3D points (x, y,
 * z) and converts it into a 2D occupancy grid represented as an Eigen matrix of
 * int8_t. Each cell in the grid can have the following values:
 *   - 100: occupied (z between z_ground_limit and robot_height)
 *   - 0: free (z <= z_ground_limit)
 *   - -1: unknown (z above robot_height)
 *
 * The grid resolution defines the size of each cell in meters. The function
 * also returns the origin of the grid corresponding to the minimum x and y
 * coordinates of the point cloud (z is always 0).
 *
 * @param filename        Path to the PCD file to read.
 * @param grid_resolution Size of each grid cell in meters.
 * @param z_ground_limit  Minimum z value considered free (cells below this are
 * free).
 * @param robot_height    Maximum z value considered occupied (cells above this
 * are unknown).
 *
 * @return A pair consisting of:
 *         1. Eigen::Matrix<int8_t, Dynamic, Dynamic>: the occupancy grid
 *            with dimensions [num_cells_x, num_cells_y].
 *         2. std::array<float, 3>: the origin of the grid in world coordinates
 *            (min_x, min_y, 0.0f).
 *
 * @throws std::runtime_error If the PCD file cannot be read or parsing fails.
 */
inline std::pair<Eigen::Matrix<int8_t, Eigen::Dynamic, Eigen::Dynamic>,
                 std::array<float, 3>>
readPCDToOccupancyGrid(const std::string &filename, const float grid_resolution,
                       const float z_ground_limit, const float robot_height) {

  auto pcd_points_opt = readPCD(filename);

  if (!pcd_points_opt) {
    throw std::runtime_error("Failed to read PCD file: " + filename);
  }

  using MatrixXi8 = Eigen::Matrix<int8_t, Eigen::Dynamic, Eigen::Dynamic>;

  const auto &pcd_points = *pcd_points_opt;
  if (pcd_points.empty()) {
    return {MatrixXi8(), {0.0f, 0.0f, 0.0f}};
  }

  // Find bounding box
  float min_x = std::numeric_limits<float>::max();
  float min_y = std::numeric_limits<float>::max();
  float max_x = std::numeric_limits<float>::lowest();
  float max_y = std::numeric_limits<float>::lowest();

  for (const auto &p : pcd_points) {
    min_x = std::min(min_x, p[0]);
    min_y = std::min(min_y, p[1]);
    max_x = std::max(max_x, p[0]);
    max_y = std::max(max_y, p[1]);
  }

  // Compute grid size
  int cell_num_x =
      static_cast<int>(std::ceil((max_x - min_x) / grid_resolution));
  int cell_num_y =
      static_cast<int>(std::ceil((max_y - min_y) / grid_resolution));

  // Precompute reciprocal
  float inv_res = 1.0f / grid_resolution;

  // Initialize grid with -1 (unknown)
  MatrixXi8 grid_data = MatrixXi8::Constant(cell_num_x, cell_num_y, -1);

  // Fill grid
  for (const auto &p : pcd_points) {
    const float x = p[0];
    const float y = p[1];
    const float z = p[2];

    int cell_x = static_cast<int>((x - min_x) * inv_res);
    int cell_y = static_cast<int>((y - min_y) * inv_res);

    if (cell_x >= 0 && cell_x < cell_num_x && cell_y >= 0 &&
        cell_y < cell_num_y) {
      int8_t z_val;
      if (z > z_ground_limit && z <= robot_height) {
        z_val = static_cast<int>(
            Kompass::Mapping::OccupancyType::OCCUPIED); // occupied
      } else if (z <= z_ground_limit) {
        z_val =
            static_cast<int>(Kompass::Mapping::OccupancyType::EMPTY); // free
      } else {
        z_val = static_cast<int>(
            Kompass::Mapping::OccupancyType::UNEXPLORED); // unknown
      }

      grid_data(cell_x, cell_y) = std::max(grid_data(cell_x, cell_y), z_val);
    }
  }

  // Return Eigen matrix + origin (min_x, min_y, 0)
  return {std::move(grid_data), {min_x, min_y, 0.0f}};
}
