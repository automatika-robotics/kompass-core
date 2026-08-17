#pragma once

#include "local_mapper.h"
#include "utils/logger.h"
#include <Eigen/Dense>
#include <array>
#include <sycl/sycl.hpp>
#include <vector>

namespace Kompass {
namespace Mapping {

class LocalMapperGPU : public LocalMapper {
public:
  /**
   * Constructor.
   *
   * @param sensors One SensorConfig per sensor. Laserscan input
   * (`isPointCloud == false`) requires exactly one sensor and consumes its
   * mount as planar (position x/y + yaw extracted from the quaternion).
   * Pointcloud input accepts N sensors fused into one grid, each with its
   * full 3D mount applied.
   *
   * NOTE for pointcloud input `maxHeight`/`minHeight` are a BODY-frame
   * height band shared by all sensors (typically 0 .. robot_height).
   */
  LocalMapperGPU(const int gridHeight, const int gridWidth,
                 const float resolution,
                 const std::vector<SensorConfig> &sensors,
                 const bool isPointCloud, const int scanSize,
                 const float maxHeight, const float minHeight,
                 const float rangeMax, const int maxPointsPerLine = 32)
      : LocalMapper(gridHeight, gridWidth, resolution, sensors, isPointCloud,
                    scanSize, maxHeight, minHeight, rangeMax,
                    maxPointsPerLine) {
    initCommon_();
    if (isPointCloud) {
      initPointCloudMode_(sensors);
    } else {
      initLaserscanMode_();
    }
  }

  // Destructor
  ~LocalMapperGPU() {
    m_q.wait(); // wait for the queue to finish before freeing memory
    if (m_devicePtrGrid) {
      sycl::free(m_devicePtrGrid, m_q);
    }
    if (m_devicePtrRanges) {
      sycl::free(m_devicePtrRanges, m_q);
    }
    if (m_devicePtrAngles) {
      sycl::free(m_devicePtrAngles, m_q);
    }
    if (m_devicePtrDistances) {
      sycl::free(m_devicePtrDistances, m_q);
    }
    for (auto &dev : m_sensorDev) {
      if (dev.ranges) {
        sycl::free(dev.ranges, m_q);
      }
      if (dev.distances) {
        sycl::free(dev.distances, m_q);
      }
      if (dev.rawBytes) {
        sycl::free(dev.rawBytes, m_q);
      }
    }
  }

  /**
   * Use the GPU to processes Laserscan data (angles and ranges) to project on
   * a 2D grid using Bresenham line drawing for each Laserscan beam
   *
   * @param angles        LaserScan angles in radians
   * @param ranges         LaserScan ranges in meters
   * @param gridData      Current grid data
   *
   * @throws std::logic_error when the mapper was constructed for pointcloud
   * input (a laserscan call would overwrite the pre-uploaded bin angles).
   */
  Eigen::MatrixXi &scanToGrid(Eigen::Ref<const Eigen::VectorXf> angles,
                              Eigen::Ref<const Eigen::VectorXf> ranges);

  /**
   * Uses a GPU to Projects 3D point cloud data onto a 2D grid using Bresenham
   * line drawing. Single-cloud adapter onto the batched entry below (the
   * mapper must be configured with exactly one sensor).
   *
   * @param data        Flattened point cloud data (uint8), typically in XYZ
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
  Eigen::MatrixXi &scanToGrid(ByteSpan data, int point_step, int row_step,
                              int height, int width, int x_offset, int y_offset,
                              int z_offset);

  /**
   * Fuses N point clouds into one occupancy grid on the GPU. clouds[i] pairs
   * with the i-th configured sensor (positional). The device grid is reset
   * once, then each sensor's cloud runs its own conversion + ray-cast kernel
   * pair. Empty views are skipped; an all-empty batch returns an all-UNEXPLORED
   * grid.
   *
   * @throws std::logic_error when the mapper was not constructed for
   * pointcloud input.
   * @throws std::invalid_argument on cloud count mismatch or negative field
   * offsets (message names the offending cloud index).
   */
  Eigen::MatrixXi &scanToGrid(Span<PointCloudView> clouds);

  // Convenience overload for containers / braced lists
  Eigen::MatrixXi &scanToGrid(const std::vector<PointCloudView> &clouds) {
    return scanToGrid(Span<PointCloudView>(clouds));
  }

private:
  // Device-side per-sensor state; one entry per configured sensor in
  // pointcloud mode, empty in laserscan mode
  struct SensorDeviceState {
    // Raw PointCloud2 bytes. Grown lazily per call because the per-scan
    // point count isn't known at ctor time; grow-only
    uint8_t *rawBytes = nullptr;
    size_t rawCapacity = 0;
    // Conversion output: per-bin minimum planar range (scanSize floats)
    float *ranges = nullptr;
    // Per-cell planar distance from this sensor's origin; gates super-cover
    // EMPTY fills in the ray-cast kernel
    float *distances = nullptr;
    // Rows 0..2 of the sensor->body isometry [R | t], row-major
    std::array<float, 12> tf{};
    Eigen::Vector2f originXY{0.0f, 0.0f};
    Eigen::Vector2i startPoint{0, 0};
    PointFieldType fieldType = PointFieldType::FLOAT32;
    int elementSize = 4;
  };

  void initCommon_() {
    m_q = sycl::queue{sycl::default_selector_v,
                      sycl::property::queue::in_order{}};
    auto dev = m_q.get_device();
    LOG_INFO("Running on :", dev.get_info<sycl::info::device::name>());

    // Query the device's max work-group size for the pointcloud conversion
    // kernel
    m_max_wg_size = dev.get_info<sycl::info::device::max_work_group_size>();

    // NOTE: Angles stay double on device. The ray-cast kernel feeds them to
    // sin/cos, and on CPU only backend float trig measurably loses to
    // double trig (glibc). The kernel already narrows the angle to float
    // before trig, so double costs only the wider H->D copy and per-thread
    // load and the fp64 throughput penalty is negligible
    m_devicePtrAngles =
        sycl::malloc_device<double>(m_scanSize > 0 ? m_scanSize : 1, m_q);
    m_anglesWide.resize(m_scanSize);
    m_devicePtrGrid = sycl::malloc_device<int>(m_gridHeight * m_gridWidth, m_q);
  }

  /**
   * Builds a per-cell distance table from the given planar origin, in the
   * column-major layout the ray-cast kernel reads (`cell(i, j)` at flat
   * index `i + j * gridHeight`). Distances are PLANAR (xy only): the ray
   * ranges they gate against are planar too, so including a sensor's mount
   * height would inflate every cell distance and suppress EMPTY fills near
   * ray endpoints.
   */
  float *makeDistanceTable_(const Eigen::Vector2f &originXY) {
    float *table = sycl::malloc_shared<float>(m_gridHeight * m_gridWidth, m_q);
    for (int i = 0; i < m_gridHeight; ++i) {
      for (int j = 0; j < m_gridWidth; ++j) {
        const Eigen::Vector3f cell = gridToLocal({i, j});
        table[i + j * m_gridHeight] = (cell.head<2>() - originXY).norm();
      }
    }
    return table;
  }

  void initLaserscanMode_() {
    m_devicePtrRanges = sycl::malloc_device<float>(m_scanSize, m_q);
    m_devicePtrDistances = makeDistanceTable_(m_sensors[0].origin_xy);
  }

  // Builds the per-sensor device state from m_sensors (already populated by the
  // base ctor) plus each sensor's field encoding
  void initPointCloudMode_(const std::vector<SensorConfig> &sensors) {
    m_sensorDev.reserve(m_sensors.size());
    for (size_t s = 0; s < m_sensors.size(); ++s) {
      SensorDeviceState dev;
      const Eigen::Matrix4f tf = m_sensors[s].tf_body.matrix();
      dev.tf = {tf(0, 0), tf(0, 1), tf(0, 2), tf(0, 3), tf(1, 0), tf(1, 1),
                tf(1, 2), tf(1, 3), tf(2, 0), tf(2, 1), tf(2, 2), tf(2, 3)};
      dev.originXY = m_sensors[s].origin_xy;
      dev.startPoint = m_sensors[s].start_point;
      dev.fieldType = sensors[s].cloud_field_type;
      dev.elementSize = elementSizeOf(dev.fieldType);
      dev.ranges = sycl::malloc_device<float>(m_scanSize, m_q);
      dev.distances = makeDistanceTable_(m_sensors[s].origin_xy);
      m_sensorDev.push_back(dev);
    }

    // Angles are pre-populated with the `2π / scan_size` bin width the
    // conversion kernel assumes; upload them once here. All sensors share the
    // bin space (bearings are body-oriented around each sensor's own origin)
    m_anglesWide = initializedAngles.cast<double>();
    m_q.memcpy(m_devicePtrAngles, m_anglesWide.data(),
               sizeof(double) * m_scanSize);
    m_q.wait();
  }

  // Laserscan-mode buffers (null in pointcloud mode)
  float *m_devicePtrDistances = nullptr;
  float *m_devicePtrRanges = nullptr;

  // Shared buffers (both modes)
  double
      *m_devicePtrAngles; // uploaded per-call (laserscan) or once (pointcloud)
  int *m_devicePtrGrid;   // output grid

  // Pointcloud mode. One device state per configured sensor; empty in
  // laserscan mode
  std::vector<SensorDeviceState> m_sensorDev;

  // Host-side widening staging buffer for the double device-angles
  Eigen::VectorXd m_anglesWide;

  // Device-reported max work-group size. Used as the pointcloud conversion
  // kernel's block dim.
  size_t m_max_wg_size = 0;

  sycl::queue m_q;
};
} // namespace Mapping
} // namespace Kompass
