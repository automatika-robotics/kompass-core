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

/**
 * @brief Warp the last frame log-odds grid into the current robot frame
 *        using a 2D affine inverse-map with bilinear sampling and a
 *        log-odds clamp.
 *
 * One thread per output cell. Each thread applies the precomputed 2x3 inverse
 * affine `(a00..a12)` to its (col, row) coordinates to find the source position
 * in the last frame grid, rounds it to the nearest cell, and reads from
 * `in_buf` and stores the result into `out_buf`. Cells whose source coordinates
 * fall outside the input grid are filled with `h0` (the initial prior
 * log-odds), so the warped grid keeps the "no information" value.
 *
 * Caller must wait on the queue before reading `out_buf`.
 * (synchronous with submitBayesianUpdateKernel)
 *
 * Grid memory is column-major (like Eigen): cell (col, row) is at flat index
 * `row + col * gridHeight`.
 *
 * @param q          SYCL queue to dispatch on.
 * @param in_buf     Last frame log-odds grid (source of the gather),
 *                   `gridHeight * gridWidth` floats, column-major.
 * @param out_buf    Output buffer for the warped log-odds, same size and
 *                   layout.
 * @param gridHeight Grid row count.
 * @param gridWidth  Grid column count.
 * @param h0         Initial prior log-odds, written to out-of-bounds cells.
 *                   Caller computes this from `pPrior` once at init.
 * @param a00,a01,a02 Coefficients of the inverse-affine row that produces the
 *                    source column: `srcCol = a00*col + a01*row + a02`.
 * @param a10,a11,a12 Coefficients of the inverse-affine row that produces the
 *                    source row: `srcRow = a10*col + a11*row + a12`. For zero
 *                    pose delta these must reduce to identity (a00=a11=1,
 *                    a01=a10=a02=a12=0).
 */
inline void submitWarpLogOddsKernel(sycl::queue &q, const float *in_buf,
                                    float *out_buf, const int gridHeight,
                                    const int gridWidth, const float h0,
                                    const float a00, const float a01,
                                    const float a02, const float a10,
                                    const float a11, const float a12) {
  q.submit([&](sycl::handler &h) {
    const int rows = gridHeight;
    const int cols = gridWidth;
    const float h0_local = h0;
    const float c00 = a00, c01 = a01, c02 = a02;
    const float c10 = a10, c11 = a11, c12 = a12;

    sycl::range<2> global_range(
        static_cast<size_t>((gridHeight + 15) / 16) * 16,
        static_cast<size_t>((gridWidth + 15) / 16) * 16);
    sycl::range<2> local_range(16, 16);

    h.parallel_for<class warpLogOddsKernel>(
        sycl::nd_range<2>{global_range, local_range},
        [=](sycl::nd_item<2> item) {
          const int y = static_cast<int>(item.get_global_id(0));
          const int x = static_cast<int>(item.get_global_id(1));
          if (y >= rows || x >= cols) {
            return;
          }
          const float fx = static_cast<float>(x);
          const float fy = static_cast<float>(y);
          const float srcX = c00 * fx + c01 * fy + c02;
          const float srcY = c10 * fx + c11 * fy + c12;

          // 4-tap bilinear gather. Clamping each cell at MAX_LOG_ODDS bounds
          // the spread reach to ~4-5 cells even under continuous observation
          constexpr float MAX_LOG_ODDS = 5.0f;
          float value;
          if (srcX >= 0.0f && srcX < static_cast<float>(cols - 1) &&
              srcY >= 0.0f && srcY < static_cast<float>(rows - 1)) {
            const int x0 = static_cast<int>(sycl::floor(srcX));
            const int y0 = static_cast<int>(sycl::floor(srcY));
            const int x1 = x0 + 1;
            const int y1 = y0 + 1;
            const float w0 = srcX - static_cast<float>(x0);
            const float w1 = 1.0f - w0;
            const float h0w = srcY - static_cast<float>(y0);
            const float h1w = 1.0f - h0w;
            // Eigen column-major layout: linear index = row + col * rows
            const float v00 = in_buf[y0 + x0 * rows];
            const float v01 = in_buf[y0 + x1 * rows];
            const float v10 = in_buf[y1 + x0 * rows];
            const float v11 = in_buf[y1 + x1 * rows];
            value = h1w * (w1 * v00 + w0 * v01) +
                    h0w * (w1 * v10 + w0 * v11);
          } else {
            value = h0_local;
          }
          out_buf[y + x * rows] = sycl::clamp(value, -MAX_LOG_ODDS, MAX_LOG_ODDS);
        });
  });
}

/**
 * @brief Per-ray Bayesian log-odds update implementing the recursive Bayes
 *        filter from arXiv:2101.01831 (eq. 2, 8, 10).
 *
 * Launches `scanSize` work-groups of `maxPointsPerLine` threads each: one
 * group per ray, one thread per cell along the ray. The geometry (endpoint
 * computation in shared memory, parametric line walk, super-cover stepping)
 * is structurally identical to `submitScanToGridKernel`.
 *
 * Each visiting thread classifies its cell as endpoint, along ray, or past
 * the observation, then applies the inverse-observation model to compute
 * a log-odds delta and atomic adds it into the persistent
 * log-odds buffer. Past the endpoint threads early out , so cells outside the
 * swept rays retain whatever the warp kernel wrote.
 *
 * Caller must have written `devicePtrLogOdds` with the warped previous
 * posterior (or `h0` everywhere) before calling this kernel, and must wait on
 * the queue before reading back.
 *
 * @param q                    SYCL queue to dispatch on.
 * @param devicePtrLogOdds     Log-odds grid (read+written in-place via
 *                             atomic_fetch_add), `gridHeight * gridWidth`
 *                             floats, column-major.
 * @param devicePtrDistances   Per-cell precomputed Euclidean distance from
 *                             the laserscan origin in metres. Used to
 *                             classify cells as endpoint / along-ray /
 *                             past-range without re-computing the geometry
 *                             on every thread.
 * @param devicePtrAngles      Per-ray angle in radians, `scanSize` doubles.
 * @param devicePtrRanges      Per-ray range in metres, `scanSize` floats.
 * @param gridHeight           Grid row count.
 * @param gridWidth            Grid column count.
 * @param resolution           Cell size in metres.
 * @param laserscanOrientation Sensor yaw offset added to every ray angle.
 * @param centralPoint         Grid coordinates of the grid's central cell.
 * @param laserscanPosition    Sensor position in the local frame (metres).
 * @param startPoint           Grid coordinates of the sensor (origin of
 *                             every ray).
 * @param scanSize             Number of rays = number of work-groups.
 * @param maxPointsPerLine     Threads per work-group; caps the ray length
 *                             in cells.
 * @param h0                   Initial prior log-odds. Subtracted from
 *                             `log(pSensor/(1-pSensor))` to form the delta
 *                             (where `l_i(z) = h0`) yields a zero update.
 * @param pPrior,pEmpty,pOccupied  Scan-model probabilities used to build
 *                                  `pSensor` for each cell class.
 * @param rangeSure,rangeMax   Distance-graded falloff parameters: within
 *                             `rangeSure` of the sensor `pSensor` equals
 *                             the unmodified class probability; beyond,
 *                             it's linearly graded toward `pPrior` at
 *                             `rangeMax`.
 */
inline void submitBayesianUpdateKernel(
    sycl::queue &q, float *devicePtrLogOdds, const float *devicePtrDistances,
    const double *devicePtrAngles, const float *devicePtrRanges,
    const int gridHeight, const int gridWidth, const float resolution,
    const float laserscanOrientation, const Eigen::Vector2i &centralPoint,
    const Eigen::Vector3f &laserscanPosition, const Eigen::Vector2i &startPoint,
    const int scanSize, const int maxPointsPerLine, const float h0,
    const float pPrior, const float pEmpty, const float pOccupied,
    const float rangeSure, const float rangeMax) {
  q.submit([&](sycl::handler &h) {
    const int rows = gridHeight;
    const int cols = gridWidth;
    const float res = resolution;
    const float orient = laserscanOrientation;

    const float h0_local = h0;
    const float pPrior_local = pPrior;
    const float pEmpty_local = pEmpty;
    const float pOccupied_local = pOccupied;
    const float rangeSure_local = rangeSure;
    const float rangeMax_local = rangeMax;

    auto devAngles = devicePtrAngles;
    auto devRanges = devicePtrRanges;
    auto devLogOdds = devicePtrLogOdds;
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

    h.parallel_for<class bayesianUpdateKernel>(
        sycl::nd_range<1>{global_size * work_group_size, work_group_size},
        [=](sycl::nd_item<1> item) {
          const size_t group_id = item.get_group().get_group_id();
          const size_t local_id = item.get_local_id();

          const float range = devRanges[group_id];
          const double angle = devAngles[group_id];

          // No-observation guard: a range >= rangeMax means this angle bin
          // was either not hit by any point (pointcloud → laserscan
          // conversion fills empty bins with rangeMax) or genuinely
          // saturated the sensor's max range. Either way, eq. 10
          // "otherwise" → no information → no log-odds update on any cell
          // along the ray. Without this guard the empty-bin rays from a
          // pointcloud-converted laserscan accumulate large negative
          // free-class deltas at every cell they walk through, drowning
          // out the OCCUPIED endpoint stamps from the real-point rays.
          if (range >= rangeMax_local) {
            return;
          }

          // calculate start and end points in first thread of the group
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
          // sync
          item.barrier(sycl::access::fence_space::local_space);

          // early exit, see note in scanToGridKernel
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
                          static_cast<float>(local_id);
            y_float = v_startPoint[1] + (g * (x_float - v_startPoint[0]));
          } else {
            float g = delta_x_f / delta_y_f;
            y_float = v_startPoint[1] +
                      ((delta_y_f > 0.0) ? 1 : ((delta_y_f < 0.0) ? -1 : 0)) *
                          static_cast<float>(local_id);
            x_float = v_startPoint[0] + (g * (y_float - v_startPoint[1]));
          }

          const int x = sycl::round(x_float);
          const int y = sycl::round(y_float);

          if (x < 0 || x >= rows || y < 0 || y >= cols) {
            return;
          }

          // Use precomputed Euclidean cell distance (meters). This is
          // more accurate than `local_id * res` for diagonal rays
          // (parametric line walk underestimates Euclidean distance).
          const float cell_distance_m = devDistances[x + y * rows];
          const bool is_endpoint = (x == toPoint[0] && y == toPoint[1]);

          // Gate: cells past the observed range (and not the endpoint
          // itself) are unobserved, apply no update.
          if (!is_endpoint && cell_distance_m >= range) {
            return;
          }

          // Inverse observation model:
          //   - endpoint -> occupied at full pOccupied confidence (no range
          //     falloff, so a single observation flips the cell to OCCUPIED
          //     even at long range).
          //   - along-ray free cells use graded falloff toward pPrior:
          //     distant (less confident).
          float pSensor;
          if (is_endpoint) {
            pSensor = pOccupied_local;
          } else {
            const float grade =
                (cell_distance_m < rangeSure_local) ? 0.0f : 1.0f;
            pSensor =
                pEmpty_local +
                grade * ((cell_distance_m - rangeSure_local) / rangeMax_local) *
                    (pPrior_local - pEmpty_local);
          }
          const float l_i = sycl::log(pSensor / (1.0f - pSensor));
          const float delta_h = l_i - h0_local;

          // Main cell stamp
          sycl::atomic_ref<float, sycl::memory_order::relaxed,
                           sycl::memory_scope::device,
                           sycl::access::address_space::global_space>
              atomic_main(devLogOdds[x + y * rows]);
          atomic_main.fetch_add(delta_h);
        });
  });
}

/**
 * @brief Threshold the final log-odds grid into a discrete `OccupancyType`
 *
 * One thread per cell. Reads the cell's log-odds from `log_odds_buf`,
 * compares against `h0` (the initial prior log-odds), and stores one of
 * three `OccupancyType` codes into `grid_buf`:
 *   - `> h0` → OCCUPIED (cell observed and likely occupied)
 *   - `< h0` → EMPTY    (cell observed and likely free)
 *   - `== h0` → UNEXPLORED (cell never observed; log-odds still at prior)
 *
 * Threads are 1:1 with cells. The caller must wait on the queue before reading
 * `grid_buf`.
 *
 * Grid memory is column-major (like Eigen): cell (col, row) at flat index
 * `row + col * gridHeight`.
 *
 * @param q            SYCL queue to dispatch on.
 * @param log_odds_buf Input log-odds grid, `gridHeight * gridWidth` floats.
 * @param grid_buf     Output discrete grid, `gridHeight * gridWidth` ints.
 *                     Fully written by this kernel — caller does not need
 *                     to pre-fill it.
 * @param gridHeight   Grid row count.
 * @param gridWidth    Grid column count.
 * @param h0           Initial prior log-odds, used as the OCCUPIED/EMPTY/
 *                     UNEXPLORED tri-threshold.
 */
inline void submitThresholdKernel(sycl::queue &q, const float *log_odds_buf,
                                  int *grid_buf, const int gridHeight,
                                  const int gridWidth, const float h0) {
  q.submit([&](sycl::handler &h) {
    const int rows = gridHeight;
    const int cols = gridWidth;
    const float h0_local = h0;
    auto in_buf = log_odds_buf;
    auto out_buf = grid_buf;

    sycl::range<2> global_range(
        static_cast<size_t>((gridHeight + 15) / 16) * 16,
        static_cast<size_t>((gridWidth + 15) / 16) * 16);
    sycl::range<2> local_range(16, 16);

    h.parallel_for<class thresholdLogOddsKernel>(
        sycl::nd_range<2>{global_range, local_range},
        [=](sycl::nd_item<2> item) {
          const int y = static_cast<int>(item.get_global_id(0));
          const int x = static_cast<int>(item.get_global_id(1));
          if (y >= rows || x >= cols) {
            return;
          }
          const float v = in_buf[y + x * rows];
          int code;
          if (v > h0_local) {
            code = static_cast<int>(Mapping::OccupancyType::OCCUPIED);
          } else if (v < h0_local) {
            code = static_cast<int>(Mapping::OccupancyType::EMPTY);
          } else {
            code = static_cast<int>(Mapping::OccupancyType::UNEXPLORED);
          }
          out_buf[y + x * rows] = code;
        });
  });
}

} // namespace

// pointcloud variant
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

// laserscan variant
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

// Bayesian helper: Both Bayesian overloads delegate here once
// `m_devicePtrRanges` and `m_devicePtrAngles` are populated.
void LocalMapperGPU::runBayesianPipeline(
    const Eigen::Vector2f &positionInPrevPose, double orientationInPrevPose) {
  // Source = current posterior. Destination = the other buffer (warp
  // output).
  float *src = m_pingState ? m_devicePtrLogOddsB : m_devicePtrLogOddsA;
  float *dst = m_pingState ? m_devicePtrLogOddsA : m_devicePtrLogOddsB;

  if (m_frameIdx == 0) {
    // First frame: no previous posterior to warp. The src persistent buffer
    // was initialized to h0. Direct copy into dst.
    m_q.memcpy(dst, src,
               sizeof(float) * static_cast<size_t>(m_gridHeight) *
                   static_cast<size_t>(m_gridWidth));
  } else {
    // Build the inverse affine coefficients (output cell -> source cell in
    // the last frame grid).
    //
    // Derivation: output cell (col=fx, row=fy) in the CURRENT frame is
    // at local position
    //   pos_curr.x_axis = (fy - cx_row) * res    // row axis = kompass x
    //   pos_curr.y_axis = (fx - cy_col) * res    // col axis = kompass y
    // The same physical point in the LAST frame is at
    //   pos_prev = positionInPrevPose + R(θ) * pos_curr
    // Mapping back to the last frame grid:
    //   src_row = pos_prev.x_axis / res + cx_row
    //   src_col = pos_prev.y_axis / res + cy_col
    // Expanding yields the affine below. For pos=(0,0), θ=0 it reduces
    // to identity.
    const float cx_row = static_cast<float>(m_centralPoint(0));
    const float cy_col = static_cast<float>(m_centralPoint(1));
    const float dx_cells = positionInPrevPose.x() / m_resolution;
    const float dy_cells = positionInPrevPose.y() / m_resolution;
    const double angle = orientationInPrevPose;
    const float cosT = static_cast<float>(std::cos(angle));
    const float sinT = static_cast<float>(std::sin(angle));

    // srcX (source column) = cosT*fx + sinT*fy + (dy_cells + cy_col*(1-cosT)
    //                                              - sinT*cx_row)
    // srcY (source row)    = -sinT*fx + cosT*fy + (dx_cells + cx_row*(1-cosT)
    //                                              + sinT*cy_col)
    const float inv_a00 = cosT;
    const float inv_a01 = sinT;
    const float inv_a02 = dy_cells + cy_col * (1.0f - cosT) - sinT * cx_row;
    const float inv_a10 = -sinT;
    const float inv_a11 = cosT;
    const float inv_a12 = dx_cells + cx_row * (1.0f - cosT) + sinT * cy_col;

    submitWarpLogOddsKernel(m_q, src, dst, m_gridHeight, m_gridWidth, m_h0,
                            inv_a00, inv_a01, inv_a02, inv_a10, inv_a11,
                            inv_a12);
  }

  // Bayesian update on the warped posterior.
  submitBayesianUpdateKernel(m_q, dst, m_devicePtrDistances, m_devicePtrAngles,
                             m_devicePtrRanges, m_gridHeight, m_gridWidth,
                             m_resolution, m_laserscanOrientation,
                             m_centralPoint, m_laserscanPosition, m_startPoint,
                             m_scanSize, m_maxPointsPerLine, m_h0, m_pPrior,
                             m_pEmpty, m_pOccupied, m_rangeSure, m_rangeMax);

  // Threshold the final log-odds into the discrete output grid.
  submitThresholdKernel(m_q, dst, m_devicePtrGrid, m_gridHeight, m_gridWidth,
                        m_h0);

  m_q.memcpy(gridData.data(), m_devicePtrGrid,
             sizeof(int) * m_gridWidth * m_gridHeight);
  m_q.wait_and_throw();

  // Swap roles so next frame's warp source is the buffer we just wrote.
  m_pingState = !m_pingState;
  m_frameIdx++;
}

// laserscan variant
Eigen::MatrixXi &LocalMapperGPU::scanToGridBaysian(
    const std::vector<double> &angles, const std::vector<double> &ranges,
    const Eigen::Vector2f &positionInPrevPose, double orientationInPrevPose) {
  if (!m_useBayesian) {
    LOG_ERROR("scanToGridBaysian called on a LocalMapperGPU constructed "
              "without Bayesian parameters; use the Bayesian ctor.");
    throw std::runtime_error(
        "scanToGridBaysian requires the Bayesian LocalMapperGPU ctor");
  }
  try {
    const auto required = static_cast<size_t>(m_scanSize);
    if (angles.size() < required || ranges.size() < required) {
      LOG_WARNING("LocalMapperGPU::scanToGridBaysian: angles/ranges shorter "
                  "than scan_size (got angles=",
                  angles.size(), " ranges=", ranges.size(),
                  " scan_size=", m_scanSize, "); skipping frame.");
      // Threshold whatever's currently in the buffer and return.
      float *current_log_odds =
          m_pingState ? m_devicePtrLogOddsB : m_devicePtrLogOddsA;
      submitThresholdKernel(m_q, current_log_odds, m_devicePtrGrid,
                            m_gridHeight, m_gridWidth, m_h0);
      m_q.memcpy(gridData.data(), m_devicePtrGrid,
                 sizeof(int) * m_gridWidth * m_gridHeight);
      m_q.wait_and_throw();
      return gridData;
    }

    m_q.memcpy(m_devicePtrAngles, angles.data(), sizeof(double) * m_scanSize);
    for (int i = 0; i < m_scanSize; ++i) {
      m_hostFloatRanges[i] = static_cast<float>(ranges[i]);
    }
    m_q.memcpy(m_devicePtrRanges, m_hostFloatRanges.data(),
               sizeof(float) * m_scanSize);

    runBayesianPipeline(positionInPrevPose, orientationInPrevPose);
  } catch (sycl::exception const &e) {
    LOG_ERROR("SYCL exception caught: ", e.what());
    throw;
  } catch (std::exception const &e) {
    LOG_ERROR("Standard exception caught: ", e.what());
    throw;
  }
  return gridData;
}

// pointcloud variant
Eigen::MatrixXi &LocalMapperGPU::scanToGridBaysian(
    const std::vector<int8_t> &data, int point_step, int row_step, int height,
    int width, float x_offset, float y_offset, float z_offset,
    const Eigen::Vector2f &positionInPrevPose, double orientationInPrevPose) {
  if (!m_useBayesian) {
    LOG_ERROR("scanToGridBaysian called on a LocalMapperGPU constructed "
              "without Bayesian parameters; use the Bayesian ctor.");
    throw std::runtime_error(
        "scanToGridBaysian requires the Bayesian LocalMapperGPU ctor");
  }
  try {
    const size_t total_bytes = data.size();
    if (total_bytes == 0 || height == 0 || width == 0) {
      LOG_WARNING("LocalMapperGPU::scanToGridBaysian(pointcloud): empty "
                  "cloud (total_bytes=",
                  total_bytes, " height=", height, " width=", width,
                  "); skipping frame.");
      float *current_log_odds =
          m_pingState ? m_devicePtrLogOddsB : m_devicePtrLogOddsA;
      submitThresholdKernel(m_q, current_log_odds, m_devicePtrGrid,
                            m_gridHeight, m_gridWidth, m_h0);
      m_q.memcpy(gridData.data(), m_devicePtrGrid,
                 sizeof(int) * m_gridWidth * m_gridHeight);
      m_q.wait_and_throw();
      return gridData;
    }

    // Lazily grow the raw-bytes device buffer to fit this scan
    if (m_rawCapacity < total_bytes) {
      if (m_devicePtrRawBytes) {
        sycl::free(m_devicePtrRawBytes, m_q);
      }
      m_devicePtrRawBytes = sycl::malloc_device<int8_t>(total_bytes, m_q);
      m_rawCapacity = total_bytes;
    }

    m_q.memcpy(m_devicePtrRawBytes, data.data(), total_bytes);

    // Pointcloud -> per-bin laserscan ranges on-device.
    submitPointCloudToLaserScanKernel(
        m_q, m_devicePtrRawBytes, total_bytes, m_devicePtrRanges, m_scanSize,
        static_cast<float>(m_rangeMax), point_step, row_step, width, height,
        static_cast<int>(x_offset), static_cast<int>(y_offset),
        static_cast<int>(z_offset), static_cast<float>(m_minHeight),
        static_cast<float>(m_maxHeight), PointFieldType::FLOAT32,
        /*element_size*/ 4, m_max_wg_size);

    runBayesianPipeline(positionInPrevPose, orientationInPrevPose);
  } catch (sycl::exception const &e) {
    LOG_ERROR("SYCL exception caught: ", e.what());
    throw;
  } catch (std::exception const &e) {
    LOG_ERROR("Standard exception caught: ", e.what());
    throw;
  }
  return gridData;
}

/**
 * @brief Copy the current Bayesian log-odds buffer and return the
 *        sigmoid-transformed probability grid for debug.
 *
 * Selects whichever of the two log-odds buffers holds the latest
 * posterior, copies it into the host-side `gridProb`, waits on the
 * queue, then applies the sigmoid `p = 1 / (1 + exp(-l))` in place on the
 * host so the values are probabilities in [0, 1] rather than raw log-odds.
 * The sigmoid is implemented via Eigen's array expression so it vectorizes
 * on the host.
 *
 * Throws `std::runtime_error` if the LocalMapperGPU was constructed via
 * the non-Bayesian ctor (the log-odds buffers and `gridProb` matrix
 * weren't allocated). Re-throws any SYCL or standard exception that
 * escapes the memcpy / sigmoid path.
 *
 * @return Reference to the internal `gridProb` matrix
 *         (`gridHeight × gridWidth`, float, values in [0, 1]). Storage is
 *         reused across calls; copy if you need to retain it.
 */
const Eigen::MatrixXf &LocalMapperGPU::getProbabilities() {
  if (!m_useBayesian) {
    LOG_ERROR("getProbabilities called on a LocalMapperGPU constructed "
              "without Bayesian parameters; use the Bayesian ctor.");
    throw std::runtime_error(
        "getProbabilities requires the Bayesian LocalMapperGPU ctor");
  }
  try {
    float *current = m_pingState ? m_devicePtrLogOddsB : m_devicePtrLogOddsA;
    m_q.memcpy(gridProb.data(), current,
               sizeof(float) * static_cast<size_t>(m_gridHeight) *
                   static_cast<size_t>(m_gridWidth));
    m_q.wait_and_throw();

    // In-place sigmoid: p = 1 / (1 + exp(-l)). Eigen vectorizes this.
    gridProb.array() = 1.0f / (1.0f + (-gridProb.array()).exp());
  } catch (sycl::exception const &e) {
    LOG_ERROR("SYCL exception caught: ", e.what());
    throw;
  } catch (std::exception const &e) {
    LOG_ERROR("Standard exception caught: ", e.what());
    throw;
  }
  return gridProb;
}
} // namespace Mapping
} // namespace Kompass
