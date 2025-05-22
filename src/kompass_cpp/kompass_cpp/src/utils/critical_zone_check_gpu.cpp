#include "utils/critical_zone_check_gpu.h"
#include "utils/logger.h"
#include <CL/sycl.hpp>
#include <hipSYCL/sycl/libkernel/marray.hpp>
#include <sycl/sycl.hpp>

namespace Kompass {

float CriticalZoneCheckerGPU::check(const std::vector<double> &ranges,
                                    const std::vector<double> &angles,
                                    const bool forward) {
  try {
    m_q.fill(m_devicePtrOutput, false, m_scan_in_zone);

    m_q.memcpy(m_devicePtrAngles, angles.data(), sizeof(double) * m_scanSize)
        .wait();
    m_q.memcpy(m_devicePtrRanges, ranges.data(), sizeof(double) * m_scanSize)
        .wait();

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
         const auto devicePtrAngles = m_devicePtrAngles;
         const auto devicePtrOutput = m_devicePtrOutput;
         const auto criticalDistance = critical_distance_;
         const auto slowdownDistance = slowdown_distance_;
         size_t *critical_indices;
         *m_result = 1.0f;
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
               double angle = devicePtrAngles[local_id];

               sycl::vec<float, 4> point{range * sycl::cos(angle),
                                         range * sycl::sin(angle), 0.0, 1.0};
               sycl::vec<float, 2> transformed_point{0.0, 0.0};

               for (size_t i = 0; i < 4; ++i) {
                 transformed_point[0] += x_transform[i] * point[i];
                 transformed_point[1] += y_transform[i] * point[i];
               }

               float converted_range = sycl::length(transformed_point);
               if (converted_range - robot_radius <= criticalDistance) {
                 // KERNEL_DEBUG("Converted Range 0 = %3f\n", converted_range);
                 // point within the zone and range is low
                 devicePtrOutput[idx] = 0.0f;
               } else if (converted_range - robot_radius <= slowdownDistance) {
                 devicePtrOutput[idx] =
                     (converted_range - robot_radius - criticalDistance) /
                     (slowdownDistance - criticalDistance);
               } else {
                 devicePtrOutput[idx] = 1.0f;
               }

               sycl::atomic_ref<float, sycl::memory_order::relaxed,
                                sycl::memory_scope::device,
                                sycl::access::address_space::global_space>
                   atomic_cost(*m_result);
               atomic_cost.fetch_min(devicePtrOutput[idx]);
             });
       })
        .wait();

    m_q.wait_and_throw();

  } catch (const sycl::exception &e) {
    LOG_ERROR("Exception caught: ", e.what());
    throw;
  }

  return *m_result;
}

} // namespace Kompass
