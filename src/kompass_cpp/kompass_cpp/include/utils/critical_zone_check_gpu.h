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
                         const Eigen::Vector3f &sensor_position_body,
                         const Eigen::Vector4f &sensor_rotation_body,
                         const float critical_angle,
                         const float critical_distance,
                         const float slowdown_distance,
                         const std::vector<double> &angles,
                         const float min_height, const float max_height,
                         const float range_max)
      : CriticalZoneChecker(
            robot_shape_type, robot_dimensions, sensor_position_body,
            sensor_rotation_body, critical_angle, critical_distance,
            slowdown_distance, angles, min_height, max_height, range_max),
        m_scanSize(angles.size()) {
    m_q = sycl::queue{sycl::default_selector_v,
                      sycl::property::queue::in_order{}};
    auto dev = m_q.get_device();
    LOG_INFO("CollisionChecker Running on :",
             dev.get_info<sycl::info::device::name>());

    m_devicePtrRanges = sycl::malloc_device<double>(m_scanSize, m_q);
    m_result = sycl::malloc_shared<float>(1, m_q);

    // set forward and backward indices
    m_cos = sycl::malloc_device<float>(cos_angles_.size(), m_q);
    m_q.memcpy(m_cos, cos_angles_.data(), sizeof(float) * cos_angles_.size())
        .wait();
    m_sin = sycl::malloc_device<float>(sin_angles_.size(), m_q);
    m_q.memcpy(m_sin, sin_angles_.data(), sizeof(float) * sin_angles_.size())
        .wait();

    m_devicePtrForward =
        sycl::malloc_device<size_t>(indicies_forward_.size(), m_q);
    m_devicePtrBackward =
        sycl::malloc_device<size_t>(indicies_backward_.size(), m_q);
    m_q.memcpy(m_devicePtrForward, indicies_forward_.data(),
               sizeof(size_t) * indicies_forward_.size())
        .wait();
    m_q.memcpy(m_devicePtrBackward, indicies_backward_.data(),
               sizeof(size_t) * indicies_backward_.size())
        .wait();

    m_scan_in_zone =
        std::max(indicies_forward_.size(), indicies_backward_.size());
  }

  // Destructor
  ~CriticalZoneCheckerGPU() {
    if (m_devicePtrRanges) {
      sycl::free(m_devicePtrRanges, m_q);
    }
    if (m_devicePtrForward) {
      sycl::free(m_devicePtrForward, m_q);
    }
    if (m_devicePtrBackward) {
      sycl::free(m_devicePtrBackward, m_q);
    }
    if (m_sin) {
      sycl::free(m_sin, m_q);
    }
    if (m_cos) {
      sycl::free(m_cos, m_q);
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
  float check(const std::vector<double> &ranges, const bool forward);

  /**
   * Use the GPU to process 3D point cloud data to check if robot is in slowdown
   * or critical zone.
   *
   * @param data        Flattened point cloud data (int8), typically in XYZ
   * format.
   * @param point_step  Number of bytes between each point in the data array.
   * @param row_step    Number of bytes between each row in the data array.
   * @param height      Number of rows (height of the point cloud).
   * @param width       Number of columns (width of the point cloud).
   * @param x_offset    Offset (in bytes) to the x-coordinate within a point.
   * @param y_offset    Offset (in bytes) to the y-coordinate within a point.
   * @param z_offset    Offset (in bytes) to the z-coordinate within a point.
   * @return            A 2D occupancy grid as an Eigen::MatrixXi.
   */
  float check(const std::vector<int8_t> &data, int point_step, int row_step,
              int height, int width, float x_offset, float y_offset,
              float z_offset, const bool forward);

private:
  const int m_scanSize;
  size_t m_scan_in_zone;
  double *m_devicePtrRanges;
  size_t *m_devicePtrForward;
  size_t *m_devicePtrBackward;
  float *m_cos;
  float *m_sin;
  float *m_result;
  sycl::queue m_q;
};

} // namespace Kompass
