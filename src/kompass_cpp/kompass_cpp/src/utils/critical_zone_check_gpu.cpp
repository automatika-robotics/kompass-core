#include "utils/critical_zone_check_gpu.h"
#include "utils/pointcloud.h"
#include "utils/logger.h"
#include <sycl/sycl.hpp>

namespace Kompass {

float CriticalZoneCheckerGPU::check(const std::vector<int8_t> &data, int point_step, int row_step,
              int height, int width, float x_offset, float y_offset,
              float z_offset, const bool forward){

  std::vector<double> ranges;
  pointCloudToLaserScanFromRaw(
      data, point_step, row_step, height, width, x_offset, y_offset, z_offset,
      range_max_, min_height_, max_height_, sin_angles_.size(), ranges);
  return check(ranges, forward);
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
