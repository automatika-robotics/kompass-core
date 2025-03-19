#pragma once

#include "utils/collision_check.h"
#include "utils/critical_zone_check.h"
#include "utils/logger.h"
#include <Eigen/Dense>
#include <algorithm>
#include <sycl/sycl.hpp>

#ifndef GPU
#define GPU 1
#endif // !GPU

namespace Kompass {

class CriticalZoneCheckerGPU : public CriticalZoneChecker {
public:
  // Constructor
  CriticalZoneCheckerGPU(const CollisionChecker::ShapeType robot_shape_type,
                         const std::vector<float> &robot_dimensions,
                         const std::array<float, 3> &sensor_position_body,
                         const std::array<float, 4> &sensor_rotation_body,
                         const float critical_angle,
                         const float critical_distance,
                         const std::vector<double> &angles)
      : CriticalZoneChecker(robot_shape_type, robot_dimensions,
                            sensor_position_body, sensor_rotation_body,
                            critical_angle, critical_distance),
        m_scanSize(angles.size()) {
    m_q = sycl::queue{sycl::default_selector_v,
                      sycl::property::queue::in_order{}};
    auto dev = m_q.get_device();
    LOG_INFO("CollisionChecker Running on :",
             dev.get_info<sycl::info::device::name>());

    m_devicePtrRanges = sycl::malloc_device<double>(m_scanSize, m_q);
    m_devicePtrAngles = sycl::malloc_device<double>(m_scanSize, m_q);
    m_result = sycl::malloc_shared<bool>(1, m_q);
    preset(angles);
    m_scan_in_zone =
        std::max(indicies_forward_.size(), indicies_backward_.size());
    m_devicePtrOutput = sycl::malloc_device<bool>(m_scan_in_zone, m_q);
  }

  // Destructor
  ~CriticalZoneCheckerGPU() {
    if (m_devicePtrOutput) {
      sycl::free(m_devicePtrOutput, m_q);
    }
    if (m_devicePtrRanges) {
      sycl::free(m_devicePtrRanges, m_q);
    }
    if (m_devicePtrAngles) {
      sycl::free(m_devicePtrAngles, m_q);
    }
    if (m_result) {
      sycl::free(m_result, m_q);
    }
  }

  /**
   * Use the GPU to processes Laserscan data (angles and ranges) for emergency
   * zone checking
   *
   * @param angles        LaserScan angles in radians
   * @param ranges         LaserScan ranges in meters
   * @param forward        If the robot is moving forward
   */
  bool check(const std::vector<double> &ranges,
             const std::vector<double> &angles, const bool forward);

private:
  const int m_scanSize;
  size_t m_scan_in_zone;
  double *m_devicePtrRanges;
  double *m_devicePtrAngles;
  bool *m_devicePtrOutput;
  bool *m_result;
  sycl::queue m_q;
};

} // namespace Kompass
