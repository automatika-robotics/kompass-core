#pragma once

#include "datatypes/span.h"
#include "utils/transformation.h"
#include <Eigen/Dense>
#include <cmath>
#include <stdexcept>
#include <string>

// Point field encoding of a PointCloud2-style byte buffer. Values match the
// sensor_msgs/PointField datatype codes.
enum class PointFieldType : int {
  INT8 = 1,
  UINT8 = 2,
  INT16 = 3,
  UINT16 = 4,
  INT32 = 5,
  UINT32 = 6,
  FLOAT32 = 7,
  FLOAT64 = 8
};

// Size in bytes of one element of the given field type
inline int elementSizeOf(const PointFieldType type) {
  switch (type) {
  case PointFieldType::INT8:
  case PointFieldType::UINT8:
    return 1;
  case PointFieldType::INT16:
  case PointFieldType::UINT16:
    return 2;
  case PointFieldType::INT32:
  case PointFieldType::UINT32:
  case PointFieldType::FLOAT32:
    return 4;
  case PointFieldType::FLOAT64:
    return 8;
  default:
    return 4;
  }
}

// Helper: Loads bytes safely handling potential misalignment
inline float load_and_cast_val(const uint8_t *ptr, size_t offset,
                               PointFieldType type) {
  const uint8_t *addr = ptr + offset;

  // Generic lambda to load unaligned data safely
  auto load_safe = [&](auto dummy_type) {
    using T = decltype(dummy_type);
    T val;
    // Copy byte-by-byte (compiler optimizes this to a register load)
    // use uint8_t* to match the source pointer type
    for (size_t i = 0; i < sizeof(T); ++i) {
      reinterpret_cast<uint8_t *>(&val)[i] = addr[i];
    }
    return static_cast<float>(val);
  };

  switch (type) {
  case PointFieldType::INT8:
    // INT8 is always aligned (1 byte)
    return static_cast<float>(*reinterpret_cast<const int8_t *>(addr));
  case PointFieldType::UINT8:
    // UINT8 is always aligned (1 byte)
    return static_cast<float>(*addr);
  case PointFieldType::INT16:
    return load_safe(int16_t{});
  case PointFieldType::UINT16:
    return load_safe(uint16_t{});
  case PointFieldType::INT32:
    return load_safe(int32_t{});
  case PointFieldType::UINT32:
    return load_safe(uint32_t{});
  case PointFieldType::FLOAT32:
    return load_safe(float{});
  case PointFieldType::FLOAT64:
    return load_safe(double{});

  default:
    return 0.0f;
  }
}

namespace Kompass {

/**
 * Per-sensor mount configuration, shared by the local mapper and the critical
 * zone checker. Covers pointcloud sensors.
 */
struct SensorConfig {
  /// Mount translation in the body frame
  Eigen::Vector3f position{0.0f, 0.0f, 0.0f};
  /// Mount rotation in the body frame, quaternion coefficients (x, y, z, w)
  Eigen::Vector4f rotation{0.0f, 0.0f, 0.0f, 1.0f};
  /// Encoding of the x/y/z fields in this sensor's byte buffer
  PointFieldType cloud_field_type{PointFieldType::FLOAT32};

  /// sensor -> body isometry built from position/rotation
  Eigen::Isometry3f tfBody() const {
    return getTransformation(rotation, position);
  }

  /// Convenience for planar (yaw-only) mounts
  static SensorConfig
  fromYaw(const Eigen::Vector3f &position, const float yaw,
          const PointFieldType field_type = PointFieldType::FLOAT32) {
    SensorConfig config;
    config.position = position;
    config.rotation = {0.0f, 0.0f, std::sin(yaw / 2.0f), std::cos(yaw / 2.0f)};
    config.cloud_field_type = field_type;
    return config;
  }
};

/**
 * Non-owning view of one PointCloud2-style byte buffer plus its layout
 * metadata. Zero-copy `data` refers to caller-owned memory that must stay
 * alive for the duration of the call it is passed to. In batched calls,
 * clouds pair with sensors positionally. clouds[i] belongs to sensors[i].
 */
struct PointCloudView {
  ByteSpan data{};
  int point_step{0};
  int row_step{0};
  int height{0};
  int width{0};
  int x_offset{-1};
  int y_offset{-1};
  int z_offset{-1};

  /// An empty view means this sensor contributed no data this tick
  bool empty() const { return data.empty() || width * height == 0; }

  /// Negative field offsets mean malformed PointCloud2 metadata
  bool offsetsValid() const {
    return x_offset >= 0 && y_offset >= 0 && z_offset >= 0;
  }
};

/**
 * @brief Non-owning view of an aligned depth image: a C-contiguous
 * ROW-MAJOR (rows x cols) pixel buffer plus its encoding. Expected encodings:
 * UINT16 (depth in millimetres, ROS 16UC1) and FLOAT32 (depth in metres, ROS
 * 32FC1); the consumer resolves the value scale from the encoding.
 */
struct DepthImageView {
  ByteSpan data{};
  int rows = 0;
  int cols = 0;
  PointFieldType field_type = PointFieldType::UINT16;

  DepthImageView() = default;
  DepthImageView(ByteSpan data, const int rows, const int cols,
                 const PointFieldType field_type = PointFieldType::UINT16)
      : data(data), rows(rows), cols(cols), field_type(field_type),
        elem_size_(elementSizeOf(field_type)) {}

  bool empty() const { return rows <= 0 || cols <= 0 || data.size() == 0; }

  /// The buffer must hold at least rows * cols pixels of the declared type
  bool sizeValid() const {
    return data.size() >= static_cast<std::size_t>(rows) * cols * elem_size_;
  }

  /// Raw pixel value at (row, col), cast to float. Bounds are upto the caller
  /// to set
  float at(const int row, const int col) const {
    return load_and_cast_val(
        data.data(), (static_cast<std::size_t>(row) * cols + col) * elem_size_,
        field_type);
  }

private:
  int elem_size_ = 2; // elementSizeOf(field_type), hoisted at construction
};

/**
 * Shared prologue of every batched cloud call. The batch must hold exactly one
 * view per configured sensor (positional pairing), and every non-empty view
 * must carry valid offsets.
 */
inline void validateClouds(Span<PointCloudView> clouds,
                           const size_t num_sensors) {
  if (clouds.size() != num_sensors) {
    throw std::invalid_argument("expected " + std::to_string(num_sensors) +
                                " clouds (one per configured sensor), got " +
                                std::to_string(clouds.size()));
  }
  for (size_t i = 0; i < clouds.size(); ++i) {
    if (!clouds[i].empty() && !clouds[i].offsetsValid()) {
      throw std::invalid_argument(
          "clouds[" + std::to_string(i) +
          "]: point field offsets (x/y/z) must be non-negative: malformed "
          "point cloud metadata");
    }
  }
}

} // namespace Kompass
