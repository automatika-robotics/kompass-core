#pragma once

#include "utils/collision_check.h"
#include "utils/critical_zone_check.h"
#include "utils/logger.h"
#include <Eigen/Dense>
#include <sycl/sycl.hpp>
#include <vector>

#ifndef GPU
#define GPU 1
#endif // !GPU

namespace Kompass {

class CriticalZoneCheckerGPU : public CriticalZoneChecker {
public:
  /**
   * @brief Constructor for GPU-accelerated Safety Check
   * * @param input_type           Selects LASERSCAN or POINTCLOUD mode.
   * - LASERSCAN: Allocates and pre-computes angular indices.
   * - POINTCLOUD: Allocates raw byte buffer for 3D data.
   * @param robot_shape_type     Robot shape (CIRCLE, RECTANGLE, POLYGON).
   * @param robot_dimensions     Dimensions specific to the shape.
   * @param sensor_position_body Position of sensor in body frame (x, y, z).
   * @param sensor_rotation_body Rotation of sensor (Quaternion x, y, z, w).
   * @param critical_angle       Half-angle of the safety cone (radians).
   * @param critical_distance    Distance for emergency stop.
   * @param slowdown_distance    Distance for linear slowdown.
   * @param angles               (LASERSCAN ONLY) Vector of angles for the scan.
   * Pass empty {} for PointCloud.
   * @param min_height           Min Z height to consider (for PointCloud).
   * @param max_height           Max Z height to consider (for PointCloud).
   * @param range_max            Maximum valid sensor range.
   */
  CriticalZoneCheckerGPU(
      InputType input_type, const CollisionChecker::ShapeType robot_shape_type,
      const std::vector<float> &robot_dimensions,
      const Eigen::Vector3f &sensor_position_body,
      const Eigen::Vector4f &sensor_rotation_body, const float critical_angle,
      const float critical_distance, const float slowdown_distance,
      const std::vector<double> &angles, const float min_height,
      const float max_height, const float range_max)
      : CriticalZoneChecker(input_type, robot_shape_type, robot_dimensions,
                            sensor_position_body, sensor_rotation_body,
                            critical_angle, critical_distance,
                            slowdown_distance, angles, min_height, max_height,
                            range_max),
        m_scanSize(angles.size()) {

    // Initialize Queue
    m_q = sycl::queue{sycl::default_selector_v,
                      sycl::property::queue::in_order{}};
    auto dev = m_q.get_device();
    LOG_INFO("CriticalZoneCheckerGPU Running on:",
             dev.get_info<sycl::info::device::name>());
    LOG_INFO("Mode:", (input_type_ == InputType::LASERSCAN ? "LASERSCAN"
                                                           : "POINTCLOUD"));

    // Shared Result Allocation (Used by both modes)
    m_result = sycl::malloc_shared<float>(1, m_q);

    // Mode-Specific Allocation
    if (input_type_ == InputType::LASERSCAN) {
      // --- LaserScan Setup ---
      // Only allocate these if we actually have angles
      if (m_scanSize == 0) {
        LOG_ERROR(
            "InputType::LASERSCAN selected but 'angles' vector is empty!");
        return;
      }

      m_devicePtrRanges = sycl::malloc_device<double>(m_scanSize, m_q);

      // Load pre-computed Sin/Cos for fast transform
      m_cos = sycl::malloc_device<float>(cos_angles_.size(), m_q);
      m_q.memcpy(m_cos, cos_angles_.data(), sizeof(float) * cos_angles_.size());

      m_sin = sycl::malloc_device<float>(sin_angles_.size(), m_q);
      m_q.memcpy(m_sin, sin_angles_.data(), sizeof(float) * sin_angles_.size());

      // Pre-compute Forward/Backward Indices
      m_devicePtrForward =
          sycl::malloc_device<size_t>(indicies_forward_.size(), m_q);
      m_q.memcpy(m_devicePtrForward, indicies_forward_.data(),
                 sizeof(size_t) * indicies_forward_.size());

      m_devicePtrBackward =
          sycl::malloc_device<size_t>(indicies_backward_.size(), m_q);
      m_q.memcpy(m_devicePtrBackward, indicies_backward_.data(),
                 sizeof(size_t) * indicies_backward_.size());

      m_q.wait(); // Finish transfers
    } else {
      // --- PointCloud Setup ---
      // Defer large buffer allocation to the first 'check' call
      // to know the exact size.
      m_rawCapacity = 0;
      m_devicePtrRawBytes = nullptr;
      max_wg_size_ = dev.get_info<sycl::info::device::max_work_group_size>();
    }
  }

  // Destructor
  ~CriticalZoneCheckerGPU() {
    // Free Shared Result
    if (m_result)
      sycl::free(m_result, m_q);

    // Free LaserScan Resources
    if (input_type_ == InputType::LASERSCAN) {
      if (m_devicePtrRanges)
        sycl::free(m_devicePtrRanges, m_q);
      if (m_devicePtrForward)
        sycl::free(m_devicePtrForward, m_q);
      if (m_devicePtrBackward)
        sycl::free(m_devicePtrBackward, m_q);
      if (m_sin)
        sycl::free(m_sin, m_q);
      if (m_cos)
        sycl::free(m_cos, m_q);
    }

    // Free PointCloud Resources
    if (m_devicePtrRawBytes) {
      sycl::free(m_devicePtrRawBytes, m_q);
    }
  }

  /**
   * @brief Process 2D LaserScan Data
   * Only valid if initialized with InputType::LASERSCAN
   */
  float check(const std::vector<double> &ranges, const bool forward);

  /**
   * @brief Process Raw 3D PointCloud Data
   * Only valid if initialized with InputType::POINTCLOUD
   */
  float check(const std::vector<int8_t> &data, int point_step, int row_step,
              int height, int width, float x_offset, float y_offset,
              float z_offset, const bool forward);

private:
  const size_t m_scanSize;

  // -- Shared --
  float *m_result = nullptr;
  sycl::queue m_q;

  // -- LaserScan Specific --
  double *m_devicePtrRanges = nullptr;
  size_t *m_devicePtrForward = nullptr;
  size_t *m_devicePtrBackward = nullptr;
  float *m_cos = nullptr;
  float *m_sin = nullptr;

  // -- PointCloud Specific --
  size_t max_wg_size_ = 0;
  int8_t *m_devicePtrRawBytes = nullptr;
  size_t m_rawCapacity = 0;
};

} // namespace Kompass
