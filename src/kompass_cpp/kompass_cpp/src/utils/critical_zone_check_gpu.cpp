#include "utils/critical_zone_check_gpu.h"
#include "utils/logger.h"
#include <sycl/sycl.hpp>

namespace Kompass {

// Helper for unaligned loads
inline float load_float_unaligned(const int8_t *base_ptr, size_t offset) {
  float res;
  int8_t *dst = reinterpret_cast<int8_t *>(&res);
  const int8_t *src = base_ptr + offset;
#pragma unroll
  for (int i = 0; i < 4; ++i)
    dst[i] = src[i];
  return res;
}

float CriticalZoneCheckerGPU::check(const std::vector<int8_t> &data,
                                    int point_step, int row_step, int height,
                                    int width, float x_offset, float y_offset,
                                    float z_offset, const bool forward) {
  // Handle Empty Cloud
  if (data.empty() || width * height == 0) {
    return 1.0f; // No points -> Safe
  }

  try {
    // Data Transfer
    // Copy the vector to the GPU
    size_t total_bytes = data.size();
    if (m_rawCapacity < total_bytes) {
      if (m_devicePtrRawBytes)
        sycl::free(m_devicePtrRawBytes, m_q);
      m_devicePtrRawBytes = sycl::malloc_device<int8_t>(total_bytes, m_q);
      m_rawCapacity = total_bytes;
    }
    m_q.memcpy(m_devicePtrRawBytes, data.data(), total_bytes);

    // command scope
    m_q.submit([&](sycl::handler &h) {
      // Prepare Constants
      // Extract 2D transform from the 3D matrix for planar navigation
      Eigen::Matrix4f tf = sensor_tf_body_.matrix();
      float tf_00 = tf(0, 0);
      float tf_01 = tf(0, 1);
      float tf_03 = tf(0, 3); // Row 0
      float tf_10 = tf(1, 0);
      float tf_11 = tf(1, 1);
      float tf_13 = tf(1, 3); // Row 1

      // Safety Thresholds
      float r_radius = robotRadius_;
      float crit_dist = critical_distance_;
      float slow_dist = slowdown_distance_;

      // Squared thresholds for fast coarse checks
      float slow_dist_sq_limit =
          (slow_dist + r_radius) * (slow_dist + r_radius);

      // Inverse for division removal
      float dist_range = slow_dist - crit_dist;
      float inv_dist_range = (dist_range > 1e-5f) ? 1.0f / dist_range : 0.0f;

      // Angular Limits
      float forward_max_angle = angle_max_forward_;
      float forward_min_angle = angle_min_forward_;
      float backward_min_angle = angle_min_backward_;
      float backward_max_angle = angle_max_backward_;

      bool check_forward = forward;

      // Kernel Configuration
      size_t num_points = width * height;
      const size_t WG_SIZE = max_wg_size_;
      size_t global_size = ((num_points + WG_SIZE - 1) / WG_SIZE) * WG_SIZE;

      // Capture pointers
      auto raw_bytes = m_devicePtrRawBytes;
      float *result = m_result;
      *result = 1.0f; // set result default

      // Offsets cast to integers for byte arithmetic
      int x_off = static_cast<int>(x_offset);
      int y_off = static_cast<int>(y_offset);
      int z_off = static_cast<int>(z_offset);
      float min_z = min_height_; // Class member
      float max_z = max_height_; // Class member

      h.parallel_for<class CheckRawCloudSafety>(
          sycl::nd_range<1>(sycl::range<1>(global_size),
                            sycl::range<1>(WG_SIZE)),
          [=](sycl::nd_item<1> item) {
            size_t idx = item.get_global_id(0);
            float local_min_factor = 1.0f;

            // --- STRIDE LOOP ---
            for (size_t i = idx; i < num_points; i += global_size) {

              // Calculate Address
              int row = i / width;
              int col = i % width;
              size_t byte_offset = static_cast<size_t>(row) * row_step +
                                   static_cast<size_t>(col) * point_step;

              // Bounds check
              if (byte_offset + std::max({x_off, y_off, z_off}) + 4 >
                  total_bytes)
                continue;

              // Early Z-Filter (Fastest Rejection)
              float z = load_float_unaligned(raw_bytes, byte_offset + z_off);
              if (z < min_z || z > max_z)
                continue;

              // Load X, Y
              float x_sens =
                  load_float_unaligned(raw_bytes, byte_offset + x_off);
              float y_sens =
                  load_float_unaligned(raw_bytes, byte_offset + y_off);

              // Transform to Body Frame
              // x_body = x*R00 + y*R01 + Tx
              float x_body = x_sens * tf_00 + y_sens * tf_01 + tf_03;
              float y_body = x_sens * tf_10 + y_sens * tf_11 + tf_13;

              // Angular Filter
              // Check if point lies inside the critical cone
              float angle = sycl::atan2(y_body, x_body);

              bool in_zone = false;
              if (check_forward) {
                // Forward check: is angle within [-crit, +crit]?
                if (angle >= forward_max_angle || angle <= forward_min_angle)
                  in_zone = true;
              } else {
                // Backward check: is angle > (PI - crit) OR angle < (-PI +
                // crit)
                if (angle >= backward_min_angle && angle <= backward_max_angle)
                  in_zone = true;
              }

              if (!in_zone)
                continue;

              // Distance Check
              float dist_sq = x_body * x_body + y_body * y_body;

              // Skip sqrt if definitely safe
              if (dist_sq > slow_dist_sq_limit)
                continue;

              float dist = sycl::sqrt(dist_sq);
              float dist_to_robot = dist - r_radius;

              // Compute Safety Factor
              float factor = 1.0f;
              if (dist_to_robot <= crit_dist) {
                factor = 0.0f;
              } else if (dist_to_robot <= slow_dist) {
                factor = (dist_to_robot - crit_dist) * inv_dist_range;
              }

              local_min_factor = sycl::fmin(local_min_factor, factor);
            }

            // --- REDUCTION ---
            // Find min across work-group
            float group_min = sycl::reduce_over_group(
                item.get_group(), local_min_factor, sycl::minimum<float>());

            // Write to global memory
            if (item.get_local_id(0) == 0) {
              sycl::atomic_ref<float, sycl::memory_order::relaxed,
                               sycl::memory_scope::device,
                               sycl::access::address_space::global_space>
                  atomic_res(*result);
              atomic_res.fetch_min(group_min);
            }
          });
    });

    m_q.wait(); // Wait for result

  } catch (const sycl::exception &e) {
    LOG_ERROR("Exception caught: ", e.what());
    throw;
  }

  return *m_result;
}

float CriticalZoneCheckerGPU::check(const std::vector<double> &ranges,
                                    const bool forward) {
  try {
    m_q.memcpy(m_devicePtrRanges, ranges.data(), sizeof(double) * m_scanSize);

    // command scope
    m_q.submit([&](sycl::handler &h) {
      // local copies of class members to be used inside the kernel
      const double robot_radius = robotRadius_;
      auto transformation_matrix = sensor_tf_body_.matrix();
      const sycl::vec<float, 4> x_transform{
          transformation_matrix(0, 0), transformation_matrix(0, 1),
          transformation_matrix(0, 2), transformation_matrix(0, 3)};

      sycl::vec<float, 4> y_transform{
          transformation_matrix(1, 0), transformation_matrix(1, 1),
          transformation_matrix(1, 2), transformation_matrix(1, 3)};

      const auto devicePtrRanges = m_devicePtrRanges;
      const auto devicePtrCos = m_cos;
      const auto devicePtrSin = m_sin;
      const auto criticalDistance = critical_distance_;
      const auto slowdownDistance = slowdown_distance_;
      size_t *critical_indices;
      float *result = m_result;
      *result = 1.0f;
      sycl::range<1> global_size;
      if (forward) {
        critical_indices = m_devicePtrForward;
        global_size = sycl::range<1>(indicies_forward_.size());
      } else {
        critical_indices = m_devicePtrBackward;
        global_size = sycl::range<1>(indicies_backward_.size());
      }

      // kernel scope
      h.parallel_for<class checkCriticalZoneKernel>(
          global_size, [=](sycl::id<1> idx) {
            const size_t local_id = critical_indices[idx];
            double range = devicePtrRanges[local_id];

            sycl::vec<float, 4> point{range * devicePtrCos[local_id],
                                      range * devicePtrSin[local_id], 0.0, 1.0};
            sycl::vec<float, 2> transformed_point{0.0, 0.0};

            for (size_t i = 0; i < 4; ++i) {
              transformed_point[0] += x_transform[i] * point[i];
              transformed_point[1] += y_transform[i] * point[i];
            }

            float converted_range = sycl::length(transformed_point);
            float slowdownFactor;
            if (converted_range - robot_radius <= criticalDistance) {
              slowdownFactor = 0.0f;
            } else if (converted_range - robot_radius <= slowdownDistance) {
              slowdownFactor =
                  (converted_range - robot_radius - criticalDistance) /
                  (slowdownDistance - criticalDistance);
            } else {
              slowdownFactor = 1.0f;
            }

            sycl::atomic_ref<float, sycl::memory_order::relaxed,
                             sycl::memory_scope::device,
                             sycl::access::address_space::global_space>
                atomic_cost(*result);
            atomic_cost.fetch_min(slowdownFactor);
          });
    });

    m_q.wait_and_throw();

  } catch (const sycl::exception &e) {
    LOG_ERROR("Exception caught: ", e.what());
    throw;
  }

  return *m_result;
}

} // namespace Kompass
