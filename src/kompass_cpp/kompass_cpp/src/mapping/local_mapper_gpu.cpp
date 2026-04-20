#include "mapping/local_mapper_gpu.h"
#include "utils/logger.h"
#include "utils/pointcloud.h"
#include <cmath>
#include <sycl/sycl.hpp>

namespace Kompass {
namespace Mapping {

namespace {

/**
 * @brief Convert a raw PointCloud2 byte buffer into a per-angle-bin
 *        laserscan on the GPU.
 *
 * One thread per input point. Each thread extracts (x, y, z) from the raw
 * buffer via load_and_cast_val, applies the Z and origin filters, computes
 * the angular bin as clamped `int((angle / 2π) * num_bins)`, and does an
 * atomic fetch_min into `device_ranges_out` for that bin.
 *
 * Caller owns every device allocation; this function only enqueues a fill
 * of `device_ranges_out` followed by the parallel_for. It does NOT wait
 * on the queue — the caller must do so before reading the result.
 *
 * @param q                 SYCL queue to dispatch on.
 * @param device_raw_bytes  Device pointer to the raw PointCloud2 buffer.
 *                          Must hold at least `total_bytes` bytes.
 * @param total_bytes       Size of the raw buffer in bytes. Used for the
 *                          per-thread out-of-bounds guard.
 * @param device_ranges_out Device pointer to the output laserscan ranges
 *                          (float, length `num_bins`). Reset to
 *                          `max_range` at the start of this call.
 * @param num_bins          Number of angular bins spanning [0, 2π).
 * @param max_range         Initial / clipping range written into every
 *                          bin before the min-reduce.
 * @param point_step        Bytes between successive points in the buffer.
 * @param row_step          Bytes between successive rows (may exceed
 *                          width * point_step if rows are padded).
 * @param width             Number of points per row.
 * @param height            Number of rows.
 * @param x_offset          Byte offset of X within a point.
 * @param y_offset          Byte offset of Y within a point.
 * @param z_offset          Byte offset of Z within a point.
 * @param min_z             Minimum acceptable Z (inclusive). There is no
 *                          disable-sentinel for the lower bound: callers
 *                          that want a one-sided filter must pass a
 *                          suitably negative value (e.g. -FLT_MAX).
 * @param max_z             Maximum acceptable Z. Negative disables the
 *                          upper bound (matches CPU behaviour).
 * @param point_field_type  Dtype of the X/Y/Z fields (dispatches
 *                          load_and_cast_val).
 * @param element_size      sizeof(field) in bytes. Used for the
 *                          per-thread bounds guard.
 * @param wg_size           Work-group size (block dim) for the kernel
 *                          launch. Should be the device's
 *                          `info::device::max_work_group_size` — the
 *                          caller queries this at ctor time
 */
inline void submitPointCloudToLaserScanKernel(
    sycl::queue &q, const int8_t *device_raw_bytes, const size_t total_bytes,
    float *device_ranges_out, const int num_bins, const float max_range,
    const int point_step, const int row_step, const int width, const int height,
    const int x_offset, const int y_offset, const int z_offset,
    const float min_z, const float max_z, const PointFieldType point_field_type,
    const int element_size, const size_t wg_size) {

  // if data is missing; return
  if (device_raw_bytes == nullptr || device_ranges_out == nullptr ||
      num_bins <= 0 || total_bytes == 0 || height * width == 0) {
    if (device_ranges_out && num_bins > 0) {
      q.fill(device_ranges_out, max_range, num_bins);
    }
    return;
  }

  q.fill(device_ranges_out, max_range, num_bins);

  q.submit([&](sycl::handler &h) {
    // Capture constants by value so they're embedded in the kernel.
    // Block dim uses the device-reported max work-group size
    const size_t num_points = static_cast<size_t>(width) * height;
    const size_t WG_SIZE = wg_size;
    const size_t global_size = ((num_points + WG_SIZE - 1) / WG_SIZE) * WG_SIZE;

    const int k_width = width;
    const int k_point_step = point_step;
    const int k_row_step = row_step;
    const bool is_contiguous = (row_step == width * point_step);
    const int x_off = x_offset;
    const int y_off = y_offset;
    const int z_off = z_offset;
    const float f_min_z = min_z;
    const float f_max_z = max_z;
    const bool max_z_enabled = (max_z >= 0.0f);
    const int k_num_bins = num_bins;
    const float k_inv_two_pi_times_bins =
        static_cast<float>(k_num_bins) / static_cast<float>(2.0 * M_PI);
    const size_t k_total_bytes = total_bytes;
    const PointFieldType k_type = point_field_type;
    const int k_elem_size = element_size;

    const int8_t *raw_bytes = device_raw_bytes;
    float *ranges_ptr = device_ranges_out;

    h.parallel_for<class pointcloudToLaserScanKernel>(
        sycl::nd_range<1>(sycl::range<1>(global_size), sycl::range<1>(WG_SIZE)),
        [=](sycl::nd_item<1> item) {
          const size_t i = item.get_global_id(0);
          if (i >= num_points)
            return;

          size_t byte_offset;
          if (is_contiguous) {
            byte_offset = i * k_point_step;
          } else {
            const int row = static_cast<int>(i / k_width);
            const int col = static_cast<int>(i % k_width);
            byte_offset = static_cast<size_t>(row) * k_row_step +
                          static_cast<size_t>(col) * k_point_step;
          }

          // Bounds check: the furthest-out field of this point must fit.
          const int max_offset = sycl::max(sycl::max(x_off, y_off), z_off);
          if (byte_offset + static_cast<size_t>(max_offset + k_elem_size) >
              k_total_bytes) {
            return;
          }

          // Early Z-filter.
          const float z =
              load_and_cast_val(raw_bytes, byte_offset + z_off, k_type);
          if (z < f_min_z)
            return;
          if (max_z_enabled && z > f_max_z)
            return;

          const float x =
              load_and_cast_val(raw_bytes, byte_offset + x_off, k_type);
          const float y =
              load_and_cast_val(raw_bytes, byte_offset + y_off, k_type);

          // Filter origin (±ε).
          const float r2 = x * x + y * y;
          if (r2 < 1e-6f)
            return;

          // Angle + bin: normalize [0, 2π), bin = clamped
          // int((angle / 2π) * num_bins).
          float angle = sycl::atan2(y, x);
          if (angle < 0.0f)
            angle += static_cast<float>(2.0 * M_PI);
          int bin = static_cast<int>(angle * k_inv_two_pi_times_bins);
          if (bin >= k_num_bins)
            bin = k_num_bins - 1;

          const float dist = sycl::sqrt(r2);
          sycl::atomic_ref<float, sycl::memory_order::relaxed,
                           sycl::memory_scope::device,
                           sycl::access::address_space::global_space>
              atomic_bin(ranges_ptr[bin]);
          atomic_bin.fetch_min(dist);
        });
  });
}

/**
 * @brief Project a laserscan (angles + ranges, already on device) onto a
 *        2D occupancy grid using super-cover Bresenham ray-casting.
 *
 * Launches `scanSize` work-groups of `maxPointsPerLine` threads each: one
 * group per ray, one thread per pixel along that ray. Thread 0 of each
 * group computes the endpoint in grid coordinates and writes the deltas
 * and step signs into shared memory; the remaining threads walk the line
 * and `atomic_fetch_max` into the grid with OccupancyType codes
 * so a later EMPTY stamp can never downgrade an earlier OCCUPIED.
 *
 * Grid memory is column-major (like Eigen): cell (x, y) at flat
 * index `x + y * rows`, `rows == gridHeight`. The caller must have
 * already filled `devicePtrGrid` with UNEXPLORED before dispatch, and
 * uploaded `devicePtrAngles` and `devicePtrRanges` for this scan.
 *
 * @param q                    SYCL queue to dispatch on.
 * @param devicePtrGrid        Output occupancy grid, `gridHeight * gridWidth`
 *                             ints, column-major. Must be pre-filled with
 *                             UNEXPLORED.
 * @param devicePtrDistances   Per-cell precomputed distance from the
 *                             laserscan origin, `gridHeight * gridWidth`
 *                             floats. Used to gate the super-cover line
 *                             fill so cells beyond the measured range
 *                             aren't wrongly marked EMPTY.
 * @param devicePtrAngles      Per-ray angle in radians, `scanSize` doubles.
 * @param devicePtrRanges      Per-ray range in metres, `scanSize` floats.
 * @param gridHeight           Grid row count (= `rows`).
 * @param gridWidth            Grid column count (= `cols`).
 * @param resolution           Cell size in metres.
 * @param laserscanOrientation Sensor yaw offset added to every ray angle.
 * @param centralPoint         Grid coordinates of the grid's central cell.
 * @param laserscanPosition    Sensor position in the local frame (metres).
 * @param startPoint           Grid coordinates of the sensor (origin of
 *                             every ray).
 * @param scanSize             Number of rays = number of work-groups to
 *                             launch.
 * @param maxPointsPerLine     Threads per work-group; caps the ray length
 *                             in cells (rays longer than this stop at the
 *                             cap without stamping an endpoint).
 */
inline void submitScanToGridKernel(
    sycl::queue &q, int *devicePtrGrid, const float *devicePtrDistances,
    const double *devicePtrAngles, const float *devicePtrRanges,
    const int gridHeight, const int gridWidth, const float resolution,
    const float laserscanOrientation, const Eigen::Vector2i &centralPoint,
    const Eigen::Vector3f &laserscanPosition, const Eigen::Vector2i &startPoint,
    const int scanSize, const int maxPointsPerLine) {
  q.submit([&](sycl::handler &h) {
    // local copies of class members to be used inside the kernel
    const int rows = gridHeight;
    const int cols = gridWidth;
    const float res = resolution;
    const float orient = laserscanOrientation;

    auto devRanges = devicePtrRanges;
    auto devAngles = devicePtrAngles;
    auto devGrid = devicePtrGrid;
    auto devDistances = devicePtrDistances;

    sycl::range global_size(scanSize);
    sycl::range work_group_size(maxPointsPerLine);

    const sycl::vec<int, 2> v_centralPoint{centralPoint(0), centralPoint(1)};
    const sycl::vec<float, 2> v_startPointLocal{laserscanPosition(0),
                                                laserscanPosition(1)};
    const sycl::vec<int, 2> v_startPoint{startPoint(0), startPoint(1)};

    auto toPoint = sycl::local_accessor<int, 1>{sycl::range{2}, h};
    auto deltas = sycl::local_accessor<int, 1>{sycl::range{2}, h};
    auto steps = sycl::local_accessor<int, 1>{sycl::range{2}, h};

    h.parallel_for<class scanToGridKernel>(
        sycl::nd_range<1>{global_size * work_group_size, work_group_size},
        [=](sycl::nd_item<1> item) {
          const size_t group_id = item.get_group().get_group_id();
          const size_t local_id = item.get_local_id();

          // Ranges is float (double casted down on host)
          float range = devRanges[group_id];
          double angle = devAngles[group_id];

          if (local_id == 0) {
            sycl::vec<float, 2> toPointLocal;
            toPointLocal[0] =
                v_startPointLocal[0] +
                (range * sycl::cos(orient + static_cast<float>(angle)));
            toPointLocal[1] =
                v_startPointLocal[1] +
                (range * sycl::sin(orient + static_cast<float>(angle)));

            toPoint[0] = v_centralPoint[0] + ceil(toPointLocal[0] / res);
            toPoint[1] = v_centralPoint[1] + ceil(toPointLocal[1] / res);
            deltas[0] = toPoint[0] - v_startPoint[0];
            deltas[1] = toPoint[1] - v_startPoint[1];
            steps[0] = (deltas[0] >= 0) ? 1 : -1;
            steps[1] = (deltas[1] >= 0) ? 1 : -1;
          }
          item.barrier(sycl::access::fence_space::local_space);

          // NOTE: Zero-range / coincident-endpoint rays produce deltas == (0, 0),
          // Bail early for every thread in the group, there's nothing to rasterise.
          // The pointcloud path already filters origin so it can't trigger this,
          // but a laserscan caller can still pass this.
          if (deltas[0] == 0 && deltas[1] == 0) {
            return;
          }

          float delta_x_f = static_cast<float>(deltas[0]);
          float delta_y_f = static_cast<float>(deltas[1]);
          float x_float, y_float;
          if (sycl::abs(deltas[0]) >= sycl::abs(deltas[1])) {
            float g = delta_y_f / delta_x_f;
            x_float = v_startPoint[0] +
                      ((delta_x_f >= 0.0) ? 1 : ((delta_x_f < 0.0) ? -1 : 0)) *
                          local_id;
            y_float = v_startPoint[1] + (g * (x_float - v_startPoint[0]));
          } else {
            float g = delta_x_f / delta_y_f;
            y_float = v_startPoint[1] +
                      ((delta_y_f > 0.0) ? 1 : ((delta_y_f < 0.0) ? -1 : 0)) *
                          local_id;
            x_float = v_startPoint[0] + (g * (y_float - v_startPoint[1]));
          }

          int x = round(x_float);
          int y = round(y_float);

          if (x >= 0 && x < rows && y >= 0 && y < cols) {
            sycl::atomic_ref<int, sycl::memory_order::relaxed,
                             sycl::memory_scope::device,
                             sycl::access::address_space::global_space>
                atomic_val(devGrid[x + y * rows]);
            sycl::atomic_ref<int, sycl::memory_order::relaxed,
                             sycl::memory_scope::device,
                             sycl::access::address_space::global_space>
                atomic_val_xstep(devGrid[(x - steps[0]) + (y * rows)]);
            sycl::atomic_ref<int, sycl::memory_order::relaxed,
                             sycl::memory_scope::device,
                             sycl::access::address_space::global_space>
                atomic_val_ystep(devGrid[x + ((y - steps[1]) * rows)]);
            if (x == toPoint[0] && y == toPoint[1]) {
              atomic_val.fetch_max(
                  static_cast<int>(Mapping::OccupancyType::OCCUPIED));
              atomic_val_xstep.fetch_max(
                  static_cast<int>(Mapping::OccupancyType::EMPTY));
              atomic_val_ystep.fetch_max(
                  static_cast<int>(Mapping::OccupancyType::EMPTY));
            } else {
              if (devDistances[x + y * rows] < range) {
                atomic_val.fetch_max(
                    static_cast<int>(Mapping::OccupancyType::EMPTY));
                atomic_val_xstep.fetch_max(
                    static_cast<int>(Mapping::OccupancyType::EMPTY));
                atomic_val_ystep.fetch_max(
                    static_cast<int>(Mapping::OccupancyType::EMPTY));
              }
            }
          }
        });
  });
}

} // namespace

Eigen::MatrixXi &LocalMapperGPU::scanToGrid(const std::vector<int8_t> &data,
                                            int point_step, int row_step,
                                            int height, int width,
                                            float x_offset, float y_offset,
                                            float z_offset) {
  try {
    // Reset output grid to UNEXPLORED before any kernel runs.
    m_q.fill(m_devicePtrGrid, static_cast<int>(OccupancyType::UNEXPLORED),
             m_gridHeight * m_gridWidth);

    // Empty cloud → nothing to project. Return the grid as all-UNEXPLORED
    const size_t total_bytes = data.size();
    if (total_bytes == 0 || height == 0 || width == 0) {
      m_q.memcpy(gridData.data(), m_devicePtrGrid,
                 sizeof(int) * m_gridWidth * m_gridHeight);
      m_q.wait_and_throw();
      return gridData;
    }

    // Grow the raw-bytes device buffer to fit this scan.
    if (m_rawCapacity < total_bytes) {
      if (m_devicePtrRawBytes) {
        sycl::free(m_devicePtrRawBytes, m_q);
      }
      m_devicePtrRawBytes = sycl::malloc_device<int8_t>(total_bytes, m_q);
      m_rawCapacity = total_bytes;
    }

    m_q.memcpy(m_devicePtrRawBytes, data.data(), total_bytes);

    // Pointcloud → per-bin laserscan ranges on device. Fills
    // m_devicePtrRanges with per-angle-bin minimum distance. Angles
    // for the subsequent ray-cast kernel were pre-uploaded
    submitPointCloudToLaserScanKernel(
        m_q, m_devicePtrRawBytes, total_bytes, m_devicePtrRanges, m_scanSize,
        static_cast<float>(m_rangeMax), point_step, row_step, width, height,
        static_cast<int>(x_offset), static_cast<int>(y_offset),
        static_cast<int>(z_offset), static_cast<float>(m_minHeight),
        static_cast<float>(m_maxHeight), PointFieldType::FLOAT32,
        /*element_size*/ 4, m_max_wg_size);

    // Ray-cast from laserscan → occupancy grid.
    submitScanToGridKernel(m_q, m_devicePtrGrid, m_devicePtrDistances,
                           m_devicePtrAngles, m_devicePtrRanges, m_gridHeight,
                           m_gridWidth, m_resolution, m_laserscanOrientation,
                           m_centralPoint, m_laserscanPosition, m_startPoint,
                           m_scanSize, m_maxPointsPerLine);

    m_q.memcpy(gridData.data(), m_devicePtrGrid,
               sizeof(int) * m_gridWidth * m_gridHeight);

    m_q.wait_and_throw();

  } catch (sycl::exception const &e) {
    LOG_ERROR("SYCL exception caught: ", e.what());
    throw; // Re-throw to Python
  } catch (std::exception const &e) {
    LOG_ERROR("Standard exception caught: ", e.what());
    throw; // Re-throw to Python
  }
  return gridData;
}

Eigen::MatrixXi &LocalMapperGPU::scanToGrid(const std::vector<double> &angles,
                                            const std::vector<double> &ranges) {

  try {
    m_q.fill(m_devicePtrGrid, static_cast<int>(OccupancyType::UNEXPLORED),
             m_gridHeight * m_gridWidth);

    // Validate host inputs before issuing H→D copies. An undersized input
    // is treated as a dropped frame return an all-UNEXPLORED grid.
    const auto required = static_cast<size_t>(m_scanSize);
    if (angles.size() < required || ranges.size() < required) {
      LOG_WARNING(
          "LocalMapperGPU::scanToGrid: angles/ranges shorter than scan_size ",
          "(got angles=", angles.size(), " ranges=", ranges.size(),
          " scan_size=", m_scanSize, "); skipping frame.");
      m_q.memcpy(gridData.data(), m_devicePtrGrid,
                 sizeof(int) * m_gridWidth * m_gridHeight);
      m_q.wait_and_throw();
      return gridData;
    }

    m_q.memcpy(m_devicePtrAngles, angles.data(), sizeof(double) * m_scanSize);

    // Ranges arrive as double but the device buffer is float
    // (to keep atomic fetch_min on the pointcloud path cheap)
    for (int i = 0; i < m_scanSize; ++i) {
      m_hostFloatRanges[i] = static_cast<float>(ranges[i]);
    }
    m_q.memcpy(m_devicePtrRanges, m_hostFloatRanges.data(),
               sizeof(float) * m_scanSize);

    submitScanToGridKernel(m_q, m_devicePtrGrid, m_devicePtrDistances,
                           m_devicePtrAngles, m_devicePtrRanges, m_gridHeight,
                           m_gridWidth, m_resolution, m_laserscanOrientation,
                           m_centralPoint, m_laserscanPosition, m_startPoint,
                           m_scanSize, m_maxPointsPerLine);

    m_q.memcpy(gridData.data(), m_devicePtrGrid,
               sizeof(int) * m_gridWidth * m_gridHeight);

    m_q.wait_and_throw();

  } catch (sycl::exception const &e) {
    LOG_ERROR("SYCL exception caught: ", e.what());
    throw; // Re-throw to Python
  } catch (std::exception const &e) {
    LOG_ERROR("Standard exception caught: ", e.what());
    throw; // Re-throw to Python
  }
  return gridData;
}
} // namespace Mapping
} // namespace Kompass
