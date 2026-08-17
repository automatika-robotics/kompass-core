#include "mapping/local_mapper_gpu.h"
#include "utils/logger.h"
#include <cmath>
#include <stdexcept>
#include <sycl/sycl.hpp>

namespace Kompass {
namespace Mapping {

// Mark the kernel submitters below private to this file
namespace {

/**
 * @brief Convert a raw PointCloud2 byte buffer into a per-angle-bin
 *        pseudo-laserscan on the GPU, around the sensor origin in BODY
 *        orientation.
 *
 * One thread per input point. Each thread extracts (x, y, z) from the raw
 * buffer via load_and_cast_val, rotates the point by the sensor mount
 * rotation, gates it on BODY-frame z (rotation plus the mount's z offset),
 * computes the angular bin from the rotated planar bearing as clamped
 * `int((angle / 2π) * num_bins)`, and does an atomic fetch_min of the
 * planar radius into `device_ranges_out` for that bin. .
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
 * @param tf_sensor_body    Rows 0..2 of the sensor→body isometry [R | t],
 *                          row-major (12 floats). Identity reproduced the
 yaw-only sensor frame conversion.
 * @param min_z_body        Minimum acceptable BODY-frame z (inclusive).
 *                          There is no disable-sentinel for the lower
 *                          bound: callers that want a one-sided filter
 *                          must pass a suitably negative value.
 * @param max_z_body        Maximum acceptable BODY-frame z (inclusive).
 *                          Applied as given. The sign of the bound carries no
                            meaning of its own. Callers wanting no upper bound
                            pass +FLT_MAX / infinity.
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
    sycl::queue &q, const uint8_t *device_raw_bytes, const size_t total_bytes,
    float *device_ranges_out, const int num_bins, const float max_range,
    const int point_step, const int row_step, const int width, const int height,
    const int x_offset, const int y_offset, const int z_offset,
    const std::array<float, 12> &tf_sensor_body, const float min_z_body,
    const float max_z_body, const PointFieldType point_field_type,
    const int element_size, const size_t wg_size) {

  // Fail loudly for a negative off-set (corrupted metadata)
  if (x_offset < 0 || y_offset < 0 || z_offset < 0) {
    throw std::invalid_argument(
        "Point field offsets (x/y/z) must be non-negative: malformed point "
        "cloud metadata");
  }

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
    const float f_min_z = min_z_body;
    const float f_max_z = max_z_body;
    const int k_num_bins = num_bins;
    const float k_inv_two_pi_times_bins =
        static_cast<float>(k_num_bins) / static_cast<float>(2.0 * M_PI);
    // Points at/beyond max_range can never win a bin, calculate its sqr
    const float k_max_range_sq = max_range * max_range;
    const size_t k_total_bytes = total_bytes;
    const PointFieldType k_type = point_field_type;
    const int k_elem_size = element_size;

    // Mount transform rows as kernel constants.
    // NOTE: Only the rotation is used in x/y (bearing and radius stay relative
    // to the sensor origin); the translation's z is used as body-frame height
    // gate
    const float r00 = tf_sensor_body[0], r01 = tf_sensor_body[1],
                r02 = tf_sensor_body[2];
    const float r10 = tf_sensor_body[4], r11 = tf_sensor_body[5],
                r12 = tf_sensor_body[6];
    const float r20 = tf_sensor_body[8], r21 = tf_sensor_body[9],
                r22 = tf_sensor_body[10];
    const float t_z = tf_sensor_body[11];

    const uint8_t *raw_bytes = device_raw_bytes;
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

          const float x =
              load_and_cast_val(raw_bytes, byte_offset + x_off, k_type);
          const float y =
              load_and_cast_val(raw_bytes, byte_offset + y_off, k_type);
          const float z =
              load_and_cast_val(raw_bytes, byte_offset + z_off, k_type);

          // Reject non-finite points (NaN padding in organized clouds)
          if (!sycl::isfinite(x) || !sycl::isfinite(y) || !sycl::isfinite(z))
            return;

          // A point at the sensor origin carries no bearing. Reject in the
          // SENSOR frame, before rotation, so it can never end up mapped to
          // the mount position (inside the robot) downstream
          if (x * x + y * y + z * z < 1e-6f)
            return;

          // Rotate into body orientation and gate on body-frame height.
          // a negative max_z is a real upper edge, not a disable-sentinel
          const float xr = r00 * x + r01 * y + r02 * z;
          const float yr = r10 * x + r11 * y + r12 * z;
          const float zb = r20 * x + r21 * y + r22 * z + t_z;
          if (zb < f_min_z || zb > f_max_z)
            return;

          // No planar extent -> no bin (point straight above/below the
          // sensor)
          const float r2 = xr * xr + yr * yr;
          if (r2 < 1e-6f)
            return;
          // Beyond max_range -> can never win the per-bin min
          if (r2 >= k_max_range_sq)
            return;

          // Angle + bin: normalize [0, 2π), bin = clamped
          // int((angle / 2π) * num_bins).
          float angle = sycl::atan2(yr, xr);
          if (angle < 0.0f)
            angle += static_cast<float>(2.0 * M_PI);
          int bin = static_cast<int>(angle * k_inv_two_pi_times_bins);
          bin = sycl::clamp(bin, 0, k_num_bins - 1);

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
 * @param devicePtrDistances   Per-cell precomputed PLANAR distance from the
 *                             ray origin, `gridHeight * gridWidth` floats,
 *                             column-major. Used to gate the super-cover
 *                             line fill so cells beyond the measured range
 *                             aren't wrongly marked EMPTY.
 * @param devicePtrAngles      Per-ray angle in radians, `scanSize` doubles.
 * @param devicePtrRanges      Per-ray range in metres, `scanSize` floats.
 * @param gridHeight           Grid row count (= `rows`).
 * @param gridWidth            Grid column count (= `cols`).
 * @param resolution           Cell size in metres.
 * @param laserscanOrientation Sensor yaw offset added to every ray angle
 *                             (0 on the pointcloud path as the conversion
 *                             already folded the mount rotation into the
 *                             bearings).
 * @param centralPoint         Grid coordinates of the grid's central cell.
 * @param laserscanPosition    Ray origin in the local frame (metres)
 * @param startPoint           Grid coordinates of the ray origin.
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

          // NOTE: Zero-range / coincident-endpoint rays produce deltas == (0,
          // 0), Bail early for every thread in the group, there's nothing to
          // rasterise. The pointcloud path already filters origin so it can't
          // trigger this, but a laserscan caller can still pass this.
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
            // Super-cover neighbor cells, bounds-checked INDIVIDUALLY at
            // grid edges (x - steps[0]) or (y - steps[1])
            const int xn = x - steps[0];
            const int yn = y - steps[1];
            const bool xn_ok = (xn >= 0 && xn < rows);
            const bool yn_ok = (yn >= 0 && yn < cols);

            sycl::atomic_ref<int, sycl::memory_order::relaxed,
                             sycl::memory_scope::device,
                             sycl::access::address_space::global_space>
                atomic_val(devGrid[x + y * rows]);

            const bool is_endpoint = (x == toPoint[0] && y == toPoint[1]);
            if (is_endpoint || devDistances[x + y * rows] < range) {
              atomic_val.fetch_max(
                  is_endpoint
                      ? static_cast<int>(Mapping::OccupancyType::OCCUPIED)
                      : static_cast<int>(Mapping::OccupancyType::EMPTY));
              if (xn_ok) {
                sycl::atomic_ref<int, sycl::memory_order::relaxed,
                                 sycl::memory_scope::device,
                                 sycl::access::address_space::global_space>
                    atomic_val_xstep(devGrid[xn + (y * rows)]);
                atomic_val_xstep.fetch_max(
                    static_cast<int>(Mapping::OccupancyType::EMPTY));
              }
              if (yn_ok) {
                sycl::atomic_ref<int, sycl::memory_order::relaxed,
                                 sycl::memory_scope::device,
                                 sycl::access::address_space::global_space>
                    atomic_val_ystep(devGrid[x + (yn * rows)]);
                atomic_val_ystep.fetch_max(
                    static_cast<int>(Mapping::OccupancyType::EMPTY));
              }
            }
          }
        });
  });
}

} // namespace

Eigen::MatrixXi &LocalMapperGPU::scanToGrid(Span<PointCloudView> clouds) {
  if (!m_isPointCloud) {
    throw std::logic_error(
        "scanToGrid(clouds): mapper was not constructed for pointcloud input");
  }
  validateClouds(clouds, m_sensorDev.size());

  try {
    // Drain anything a previous throwing call may have left in flight
    m_q.wait();

    // Reset output grid to UNEXPLORED once
    m_q.fill(m_devicePtrGrid, static_cast<int>(OccupancyType::UNEXPLORED),
             m_gridHeight * m_gridWidth);

    for (size_t i = 0; i < clouds.size(); ++i) {
      const auto &cloud = clouds[i];
      if (cloud.empty()) {
        continue; // no data from this sensor this tick
      }
      auto &dev = m_sensorDev[i];
      const size_t total_bytes = cloud.data.size();

      // Grow this sensor's raw-bytes device buffer to fit the cloud
      if (dev.rawCapacity < total_bytes) {
        if (dev.rawBytes) {
          sycl::free(dev.rawBytes, m_q);
        }
        dev.rawBytes = sycl::malloc_device<uint8_t>(total_bytes, m_q);
        dev.rawCapacity = total_bytes;
      }
      m_q.memcpy(dev.rawBytes, cloud.data.data(), total_bytes);

      // Pointcloud → per-bin pseudo-scan around this sensor's origin in
      // body orientation
      submitPointCloudToLaserScanKernel(
          m_q, dev.rawBytes, total_bytes, dev.ranges, m_scanSize,
          static_cast<float>(m_rangeMax), cloud.point_step, cloud.row_step,
          cloud.width, cloud.height, cloud.x_offset, cloud.y_offset,
          cloud.z_offset, dev.tf, static_cast<float>(m_minHeight),
          static_cast<float>(m_maxHeight), dev.fieldType, dev.elementSize,
          m_max_wg_size);

      // Ray-cast from this sensor's origin with orientation 0 (the mount
      // rotation is already folded into the bearings)
      submitScanToGridKernel(
          m_q, m_devicePtrGrid, dev.distances, m_devicePtrAngles, dev.ranges,
          m_gridHeight, m_gridWidth, m_resolution, /*orientation*/ 0.0f,
          m_centralPoint,
          Eigen::Vector3f{dev.originXY.x(), dev.originXY.y(), 0.0f},
          dev.startPoint, m_scanSize, m_maxPointsPerLine);
    }

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

// Single pointcloud overload
Eigen::MatrixXi &LocalMapperGPU::scanToGrid(ByteSpan data, int point_step,
                                            int row_step, int height, int width,
                                            int x_offset, int y_offset,
                                            int z_offset) {
  // Allocation-free N=1 adapter
  const PointCloudView view{data,  point_step, row_step, height,
                            width, x_offset,   y_offset, z_offset};
  return scanToGrid(Span<PointCloudView>(&view, 1));
}

// Laserscan overload
Eigen::MatrixXi &
LocalMapperGPU::scanToGrid(Eigen::Ref<const Eigen::VectorXf> angles,
                           Eigen::Ref<const Eigen::VectorXf> ranges) {
  if (m_isPointCloud) {
    throw std::logic_error(
        "scanToGrid(angles, ranges): mapper was constructed for pointcloud "
        "input; the laserscan overload would overwrite the pre-uploaded "
        "conversion bin angles");
  }

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

    // Ranges go straight H→D (float32 in, float32 device buffer). Angles
    // widen through the staging member buffer. The device buffer is double (see
    // the ctor note on host-backend trig performance)
    m_anglesWide = angles.cast<double>();
    m_q.memcpy(m_devicePtrAngles, m_anglesWide.data(),
               sizeof(double) * m_scanSize);
    m_q.memcpy(m_devicePtrRanges, ranges.data(), sizeof(float) * m_scanSize);

    submitScanToGridKernel(
        m_q, m_devicePtrGrid, m_devicePtrDistances, m_devicePtrAngles,
        m_devicePtrRanges, m_gridHeight, m_gridWidth, m_resolution,
        m_sensors[0].yaw, m_centralPoint,
        Eigen::Vector3f{m_sensors[0].origin_xy.x(), m_sensors[0].origin_xy.y(),
                        0.0f},
        m_sensors[0].start_point, m_scanSize, m_maxPointsPerLine);

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
