#include "utils/critical_zone_check_gpu.h"
#include "utils/logger.h"
#include <CL/sycl.hpp>
#include <cstddef>
#include <hipSYCL/sycl/libkernel/marray.hpp>
#include <sycl/sycl.hpp>

namespace Kompass {

bool CriticalZoneCheckerGPU::check(const std::vector<double> &ranges,
                                   const std::vector<double> &angles,
                                   const bool forward) {
  bool result;
  try {

    m_q.fill(m_devicePtrOutput, false, m_scanSize);

    m_q.memcpy(m_devicePtrAngles, angles.data(), sizeof(double) * m_scanSize);
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

      auto devicePtrRanges = m_devicePtrRanges;
      auto devicePtrAngles = m_devicePtrAngles;
      auto devicePtrOutput = m_devicePtrOutput;
      auto criticalDistance = critical_distance_;
      auto isForward = forward;
      auto angle_right_forward = angle_right_forward_;
      auto angle_left_forward = angle_left_forward_;
      auto angle_right_backward = angle_right_backward_;
      auto angle_left_backward = angle_left_backward_;

      sycl::range<1> global_size(m_scanSize); // number of laser rays

      // kernel scope
      h.parallel_for<class checkCriticalZoneKernel>(global_size, [=](sycl::id<1>
                                                                         idx) {
        const int local_id = idx[0];
        double range = devicePtrRanges[local_id];
        double angle = devicePtrAngles[local_id];

        sycl::vec<float, 4> point{range * sycl::cos(angle),
                                  range * sycl::sin(angle), 0.0, 1.0};
        sycl::vec<float, 2> transformed_point{0.0, 0.0};

        for (size_t i = 0; i < 4; ++i) {
          transformed_point[0] += x_transform[i] * point[i];
          transformed_point[1] += y_transform[i] * point[i];
        }

        double theta = sycl::atan2(transformed_point[1], transformed_point[0]);
        // Normalize the angle to [0, 2*pi]
        theta = sycl::fmod(theta, 2 * M_PI);
        if (theta < 0) {
          theta += 2 * M_PI;
        }
        // If angle is greater than pi, subtract from 2*pi to get the
        // range [0, pi]
        if (theta > 2 * M_PI) {
          theta = 2 * M_PI - theta;
        }

        if (isForward) {
          if ((theta <= sycl::max(angle_left_forward, angle_right_forward) ||
               theta >= sycl::min(angle_left_forward, angle_right_forward)) &&
              range - robot_radius <= criticalDistance) {
            // point within the zone and range is low
            devicePtrOutput[local_id] = true;
          }
        } else {
          if ((theta >= sycl::min(angle_left_backward, angle_right_backward) &&
               theta <= sycl::max(angle_left_backward, angle_right_backward)) &&
              range - robot_radius <= criticalDistance) {
            // point within the zone and range is low
            devicePtrOutput[local_id] = true;
          }
        }
      });
    });

    *m_result = 0;
    // Launch a kernel that reduces the array using a logical OR operation.
    // If any d_flags element is true, the reduction will produce true.
    m_q.submit([&](sycl::handler &h) {
      auto reduction = sycl::reduction(m_result, sycl::logical_or<bool>());
      auto devicePtrOutput = m_devicePtrOutput;
      h.parallel_for(
          sycl::range<1>(m_scanSize), reduction,
          [=](sycl::id<1> idx, auto &r) { r.combine(devicePtrOutput[idx]); });
    });

    m_q.wait_and_throw();

    result = *m_result;

  } catch (const sycl::exception &e) {
    LOG_ERROR("Exception caught: ", e.what());
  }

  return result;
}

} // namespace Kompass
