#include "utils/critical_zone_check_gpu.h"
#include "utils/logger.h"
#include <stdexcept>
#include <sycl/sycl.hpp>

namespace Kompass {

// Mark the kernel submitter below private to this file
namespace {

/**
 * @brief Submit one safety-check kernel for one sensor's cloud.
 *
 * One thread per point (grid-stride). Each thread extracts (x, y, z) via
 * load_and_cast_val, applies the sensor's FULL mount transform, gates on
 * BODY-frame height, filters by the body-frame critical cone, and min-reduces
 * its safety factor into the shared `result`.
 */
inline void submitCloudCheckKernel(
    sycl::queue &q, const uint8_t *device_raw_bytes, const size_t total_bytes,
    float *result, const int point_step, const int row_step, const int width,
    const int height, const int x_offset, const int y_offset,
    const int z_offset, const std::array<float, 12> &tf_sensor_body,
    const float min_z_body, const float max_z_body, const float critical_angle,
    const float robot_radius, const float critical_distance,
    const float slowdown_distance, const float slow_limit_sq,
    const float inv_dist_range_in, const bool check_forward,
    const PointFieldType point_field_type, const int element_size,
    const size_t wg_size) {
  q.submit([&](sycl::handler &h) {
    // Full mount transform rows as kernel constants: rows 0/1 position the
    // point in the body plane; row 2 gives the body-frame height for the band
    // gate
    const float t00 = tf_sensor_body[0], t01 = tf_sensor_body[1],
                t02 = tf_sensor_body[2], t03 = tf_sensor_body[3];
    const float t10 = tf_sensor_body[4], t11 = tf_sensor_body[5],
                t12 = tf_sensor_body[6], t13 = tf_sensor_body[7];
    const float t20 = tf_sensor_body[8], t21 = tf_sensor_body[9],
                t22 = tf_sensor_body[10], t23 = tf_sensor_body[11];

    // Safety Thresholds
    const float crit_angle = critical_angle;
    const float r_radius = robot_radius;
    const float crit_dist = critical_distance;
    const float slow_dist = slowdown_distance;

    // Precomputed at construction
    const float slow_dist_sq_limit = slow_limit_sq;
    const float inv_dist_range = inv_dist_range_in;

    const bool k_forward = check_forward;

    // Kernel Configuration
    const size_t num_points = static_cast<size_t>(width) * height;
    const size_t WG_SIZE = wg_size;
    const size_t global_size = ((num_points + WG_SIZE - 1) / WG_SIZE) * WG_SIZE;

    // Capture params
    const int m_width = width;
    const int m_point_step = point_step;
    const int m_row_step = row_step;
    const bool is_contiguous = (row_step == width * point_step);
    const int x_off = x_offset;
    const int y_off = y_offset;
    const int z_off = z_offset;
    const float f_min_z = min_z_body;
    const float f_max_z = max_z_body;
    const PointFieldType k_type = point_field_type;
    const int k_elem_size = element_size;
    const size_t k_total_bytes = total_bytes;

    const uint8_t *raw_bytes = device_raw_bytes;

    h.parallel_for<class CheckRawCloudSafety>(
        sycl::nd_range<1>(sycl::range<1>(global_size), sycl::range<1>(WG_SIZE)),
        [=](sycl::nd_item<1> item) {
          const size_t idx = item.get_global_id(0);
          float local_min_factor = 1.0f;

          // --- STRIDE LOOP ---
          for (size_t i = idx; i < num_points; i += global_size) {

            // Calculate Address
            size_t byte_offset;
            if (is_contiguous) {
              // Fast path: direct multiplication
              byte_offset = i * m_point_step;
            } else {
              // Slow path: only used if rows have padding
              const int row = static_cast<int>(i / m_width);
              const int col = static_cast<int>(i % m_width);
              byte_offset = static_cast<size_t>(row) * m_row_step +
                            static_cast<size_t>(col) * m_point_step;
            }
            // Bounds check
            int max_offset = x_off;
            if (y_off > max_offset)
              max_offset = y_off;
            if (z_off > max_offset)
              max_offset = z_off;

            if (byte_offset + max_offset + k_elem_size > k_total_bytes)
              continue;

            // Get x,y,z
            const float x_sens =
                load_and_cast_val(raw_bytes, byte_offset + x_off, k_type);
            const float y_sens =
                load_and_cast_val(raw_bytes, byte_offset + y_off, k_type);
            const float z_sens =
                load_and_cast_val(raw_bytes, byte_offset + z_off, k_type);

            // Reject non-finite points (NaN padding in organized clouds)
            if (!sycl::isfinite(x_sens) || !sycl::isfinite(y_sens) ||
                !sycl::isfinite(z_sens))
              continue;

            // A point at the sensor origin carries no direction.
            if (x_sens * x_sens + y_sens * y_sens + z_sens * z_sens < 1e-6f) {
              continue;
            }

            // Full mount transform into the body frame
            const float x_body =
                x_sens * t00 + y_sens * t01 + z_sens * t02 + t03;
            const float y_body =
                x_sens * t10 + y_sens * t11 + z_sens * t12 + t13;
            const float z_body =
                x_sens * t20 + y_sens * t21 + z_sens * t22 + t23;

            // Body-frame height band
            if (z_body < f_min_z || z_body > f_max_z)
              continue;

            // Coarse distance rejection
            const float dist_sq = x_body * x_body + y_body * y_body;
            if (dist_sq > slow_dist_sq_limit)
              continue;

            // Angular Filter
            // Check if point lies inside the critical cone (body frame)
            const float abs_angle = sycl::fabs(sycl::atan2(y_body, x_body));

            bool in_zone = false;
            if (k_forward) {
              // Forward check: is angle within [-crit, +crit]?
              if (abs_angle <= crit_angle)
                in_zone = true;
            } else {
              // Backward check: is angle within [PI - crit, -PI + crit]
              if (abs_angle >= M_PI - crit_angle)
                in_zone = true;
            }

            if (!in_zone)
              continue;

            const float dist = sycl::sqrt(dist_sq);
            const float dist_to_robot = dist - r_radius;

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
          const float group_min = sycl::reduce_over_group(
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
}

} // namespace

float CriticalZoneCheckerGPU::check(Span<PointCloudView> clouds,
                                    const bool forward) {
  if (input_type_ != InputType::POINTCLOUD) {
    throw std::logic_error(
        "check(clouds): checker was not constructed for pointcloud input");
  }
  std::lock_guard<std::mutex> lock(m_mutex);
  validateClouds(clouds, m_sensorDev.size());

  try {
    // Reset the shared result ONCE
    m_q.fill(m_result, 1.0f, 1);

    for (size_t s = 0; s < clouds.size(); ++s) {
      const auto &cloud = clouds[s];
      if (cloud.empty()) {
        continue; // no data from this sensor this tick
      }
      auto &devState = m_sensorDev[s];
      const size_t total_bytes = cloud.data.size();

      // Grow this sensor's raw-bytes device buffer to fit the cloud
      if (devState.rawCapacity < total_bytes) {
        if (devState.rawBytes) {
          // A kernel from a previous throwing call could still be reading
          // this buffer (sycl::free is host-side and NOT queue-ordered), so
          // drain before freeing
          m_q.wait();
          sycl::free(devState.rawBytes, m_q);
        }
        devState.rawBytes = sycl::malloc_device<uint8_t>(total_bytes, m_q);
        devState.rawCapacity = total_bytes;
      }
      m_q.memcpy(devState.rawBytes, cloud.data.data(), total_bytes);

      submitCloudCheckKernel(
          m_q, devState.rawBytes, total_bytes, m_result, cloud.point_step,
          cloud.row_step, cloud.width, cloud.height, cloud.x_offset,
          cloud.y_offset, cloud.z_offset, sensors_[s].tf, min_height_,
          max_height_, critical_angle_, static_cast<float>(robotRadius_),
          critical_distance_, slowdown_distance_, slow_limit_sq_,
          inv_dist_range_, forward, sensors_[s].field_type,
          sensors_[s].elem_size, max_wg_size_);
    }

    m_q.wait(); // Wait for all sensors' kernels

  } catch (const sycl::exception &e) {
    LOG_ERROR("Exception caught: ", e.what());
    throw;
  }

  return *m_result;
}

// Single pointcloud overload
float CriticalZoneCheckerGPU::check(ByteSpan data, int point_step, int row_step,
                                    int height, int width, int x_offset,
                                    int y_offset, int z_offset,
                                    const bool forward) {
  // Allocation-free N=1 adapter
  const PointCloudView view{data,  point_step, row_step, height,
                            width, x_offset,   y_offset, z_offset};
  return check(Span<PointCloudView>(&view, 1), forward);
}

// Laserscan overload
float CriticalZoneCheckerGPU::check(Eigen::Ref<const Eigen::VectorXf> ranges,
                                    const bool forward) {
  if (input_type_ != InputType::LASERSCAN) {
    throw std::logic_error(
        "check(ranges): checker was constructed for pointcloud input");
  }
  std::lock_guard<std::mutex> lock(m_mutex);
  try {
    // Input is float32. Straight H→D copy
    m_q.memcpy(m_devicePtrRanges, ranges.data(), sizeof(float) * m_scanSize);

    // command scope
    m_q.submit([&](sycl::handler &h) {
      const double robot_radius = robotRadius_;

      // Prepare Transformation Constants
      auto tf = sensors_[0].tf_body.matrix();
      const float m00 = tf(0, 0), m01 = tf(0, 1), m03 = tf(0, 3);
      const float m10 = tf(1, 0), m11 = tf(1, 1), m13 = tf(1, 3);
      // Thresholds precomputed at construction
      const float crit_dist = critical_distance_;
      const float inv_dist_range = inv_dist_range_;
      const float safe_threshold_sq = slow_limit_sq_;

      // Select Indices
      size_t *critical_indices;
      size_t num_work_items;
      if (forward) {
        critical_indices = m_devicePtrForward;
        num_work_items = indicies_forward_.size();
      } else {
        critical_indices = m_devicePtrBackward;
        num_work_items = indicies_backward_.size();
      }

      // Reset Result
      *m_result = 1.0f;

      // Capture pointers by value for the kernel
      const auto devRanges = m_devicePtrRanges;
      const auto devCos = m_cos;
      const auto devSin = m_sin;

      // kernel scope
      h.parallel_for(
          sycl::range<1>(num_work_items),
          sycl::reduction(m_result, sycl::minimum<float>()),
          [=](sycl::id<1> idx, auto &reducer) {
            const size_t global_idx = critical_indices[idx];
            const float r = devRanges[global_idx];

            // Unrolled Matrix Multiply
            // z=0 and w=1, so we skip z-row and w-row calculations.
            // x_local = r * cos, y_local = r * sin
            const float cos_val = devCos[global_idx];
            const float sin_val = devSin[global_idx];

            // x_body = m00*x_L + m01*y_L + m03
            const float x = (m00 * r * cos_val) + (m01 * r * sin_val) + m03;
            const float y = (m10 * r * cos_val) + (m11 * r * sin_val) + m13;

            // Squared Distance Check
            const float dist_sq = x * x + y * y;

            // Only proceed if the point is unsafe
            if (dist_sq < safe_threshold_sq) {
              // Calculate distance
              const float dist = sycl::sqrt(dist_sq);
              const float dist_surface = dist - robot_radius;

              if (dist_surface <= crit_dist) {
                reducer.combine(0.0f);
              } else {
                // In the slowdown zone (between crit and slow)
                float factor = (dist_surface - crit_dist) * inv_dist_range;
                reducer.combine(factor);
              }
            }
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
