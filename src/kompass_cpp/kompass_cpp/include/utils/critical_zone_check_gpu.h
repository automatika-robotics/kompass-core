#pragma once

#include "utils/collision_check.h"
#include "utils/critical_zone_check.h"
#include "utils/logger.h"
#include <Eigen/Dense>
#include <mutex>
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
   *
   * @param input_type        Selects LASERSCAN or POINTCLOUD mode.
   * - LASERSCAN: requires exactly one sensor and non-empty `scan_angles`;
   * - POINTCLOUD: accepts N sensors; each check() submits one kernel per
   *   cloud, all min-reducing into one shared result.
   * @param robot_shape_type  Robot shape (CYLINDER, BOX, ...).
   * @param robot_dimensions  Dimensions specific to the shape.
   * @param sensors           One SensorConfig per sensor (mount pose in the
   * body frame + the point field encoding for pointcloud input).
   * @param critical_angle    Full angle of the safety cone (degrees).
   * @param critical_distance Distance for emergency stop (m).
   * @param slowdown_distance Distance for linear slowdown (m).
   * @param min_height        Minimum accepted point height (m); BODY-frame
   * band for pointcloud input, shared by all sensors.
   * @param max_height        Maximum accepted point height (m); body-frame
   * for pointcloud input.
   * @param range_max         Maximum valid sensor range (m).
   * @param scan_angles       (LASERSCAN only) scan angles in radians.
   */
  CriticalZoneCheckerGPU(InputType input_type,
                         const CollisionChecker::ShapeType robot_shape_type,
                         const std::vector<float> &robot_dimensions,
                         const std::vector<SensorConfig> &sensors,
                         const float critical_angle,
                         const float critical_distance,
                         const float slowdown_distance, const float min_height,
                         const float max_height, const float range_max,
                         const std::vector<double> &scan_angles = {})
      : CriticalZoneChecker(input_type, robot_shape_type, robot_dimensions,
                            sensors, critical_angle, critical_distance,
                            slowdown_distance, min_height, max_height,
                            range_max, scan_angles),
        m_scanSize(scan_angles.size()) {

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
      m_devicePtrRanges = sycl::malloc_device<float>(m_scanSize, m_q);

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
      max_wg_size_ = dev.get_info<sycl::info::device::max_work_group_size>();
      // One device-buffer slot per sensor; grown lazily on first use.
      m_sensorDev.resize(sensors_.size());
    }
  }

  // Destructor
  ~CriticalZoneCheckerGPU() {
    m_q.wait(); // wait for the queue to finish before freeing memory

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
    for (auto &devState : m_sensorDev) {
      if (devState.rawBytes) {
        sycl::free(devState.rawBytes, m_q);
      }
    }
  }

  /**
   * @brief Process 2D LaserScan Data
   * Only valid if initialized with InputType::LASERSCAN
   *
   * @throws std::logic_error when the checker was constructed for pointcloud
   * input.
   */
  float check(Eigen::Ref<const Eigen::VectorXf> ranges, const bool forward);

  /**
   * @brief Checks N point clouds (one per configured sensor, positional
   * pairing) on the GPU and returns the minimum safety factor across all of
   * them: one kernel per non-empty cloud, all atomically min-reducing into
   * one shared result, with a single wait at the end. Empty views are
   * skipped; an all-empty batch returns 1.0.
   *
   * @throws std::logic_error when the checker was not constructed for
   * pointcloud input.
   * @throws std::invalid_argument on cloud count mismatch or negative field
   * offsets (message names the offending cloud index).
   */
  float check(Span<PointCloudView> clouds, const bool forward);

  // Convenience overload for containers / braced lists
  float check(const std::vector<PointCloudView> &clouds, const bool forward) {
    return check(Span<PointCloudView>(clouds), forward);
  }

  /**
   * @brief Process Raw 3D PointCloud Data. Single-cloud adapter onto the
   * batched check above.
   */
  float check(ByteSpan data, int point_step, int row_step, int height,
              int width, int x_offset, int y_offset, int z_offset,
              const bool forward);

private:
  const size_t m_scanSize; // laserscan mode only

  // -- Shared --
  float *m_result = nullptr;
  sycl::queue m_q;

  // -- LaserScan Specific --
  float *m_devicePtrRanges = nullptr;
  size_t *m_devicePtrForward = nullptr;
  size_t *m_devicePtrBackward = nullptr;
  float *m_cos = nullptr;
  float *m_sin = nullptr;

  // -- PointCloud Specific --
  // Per-sensor DEVICE BUFFERS only; one entry per configured sensor in
  // pointcloud mode, empty in laserscan mode
  struct SensorDeviceState {
    // Raw PointCloud2 bytes. Grown lazily per call; grow-only
    uint8_t *rawBytes = nullptr;
    size_t rawCapacity = 0;
  };
  std::vector<SensorDeviceState> m_sensorDev;
  size_t max_wg_size_ = 0;
  // Mutex to make sure that two concurrent grow-and-reallocate paths dont free
  // the same buffer or copy into a buffer the other thread just freed
  std::mutex m_mutex;
};

} // namespace Kompass
