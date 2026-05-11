#pragma once

#include "local_mapper.h"
#include "utils/logger.h"
#include <Eigen/Dense>
#include <cmath>
#include <sycl/sycl.hpp>
#include <vector>

namespace Kompass {
namespace Mapping {

class LocalMapperGPU : public LocalMapper {
public:
  // Constructor
  LocalMapperGPU(const int gridHeight, const int gridWidth,
                 const float resolution,
                 const Eigen::Vector3f &laserscanPosition,
                 const float laserscanOrientation, const bool isPointCloud,
                 const int scanSize, const float angleStep,
                 const float maxHeight, const float minHeight,
                 const float rangeMax, const int maxPointsPerLine = 32)
      : LocalMapper(gridHeight, gridWidth, resolution, laserscanPosition,
                    laserscanOrientation, isPointCloud, scanSize, angleStep,
                    maxHeight, minHeight, rangeMax, maxPointsPerLine) {
    initializeGPU(isPointCloud, scanSize);
  }

  // Constructor with Bayesian parameters. Allocates two persistent log-odds
  // device buffers that keep the recursive Bayes state. Not reset between frames.
  LocalMapperGPU(const int gridHeight, const int gridWidth,
                 const float resolution,
                 const Eigen::Vector3f &laserscanPosition,
                 const float laserscanOrientation, const bool isPointCloud,
                 const int scanSize, const float pPrior, const float pOccupied,
                 const float pEmpty, const float rangeSure,
                 const float rangeMax, const float angleStep,
                 const float maxHeight, const float minHeight,
                 const int maxPointsPerLine = 32)
      : LocalMapper(gridHeight, gridWidth, resolution, laserscanPosition,
                    laserscanOrientation, isPointCloud, scanSize, pPrior,
                    pOccupied, pEmpty, rangeSure, rangeMax, angleStep,
                    maxHeight, minHeight, maxPointsPerLine) {
    initializeGPU(isPointCloud, scanSize);

    m_useBayesian = true;
    m_h0 = std::log(pPrior / (1.0f - pPrior));  // log prob prior

    const size_t cellCount =
        static_cast<size_t>(m_gridHeight) * static_cast<size_t>(m_gridWidth);
    m_devicePtrLogOddsA = sycl::malloc_device<float>(cellCount, m_q);
    m_devicePtrLogOddsB = sycl::malloc_device<float>(cellCount, m_q);
    m_q.fill(m_devicePtrLogOddsA, m_h0, cellCount);
    m_q.wait();
    gridProb = Eigen::MatrixXf(gridHeight, gridWidth);
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
    if (m_devicePtrRawBytes) {
      sycl::free(m_devicePtrRawBytes, m_q);
    }
    if (m_devicePtrLogOddsA) {
      sycl::free(m_devicePtrLogOddsA, m_q);
    }
    if (m_devicePtrLogOddsB) {
      sycl::free(m_devicePtrLogOddsB, m_q);
    }
  }

  /**
   * Use the GPU to processes Laserscan data (angles and ranges) to project on
   * a 2D grid using Bresenham line drawing for each Laserscan beam
   *
   * @param angles        LaserScan angles in radians
   * @param ranges         LaserScan ranges in meters
   * @param gridData      Current grid data
   */
  Eigen::MatrixXi &scanToGrid(const std::vector<double> &angles,
                              const std::vector<double> &ranges);

  /**
   * Uses a GPU to Projects 3D point cloud data onto a 2D grid using Bresenham
   * line drawing.
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
  Eigen::MatrixXi &scanToGrid(const std::vector<int8_t> &data, int point_step,
                              int row_step, int height, int width,
                              float x_offset, float y_offset, float z_offset);

  /**
   * Run the GPU Bayesian local-mapping update for one new laserscan frame.
   *
   * Implements the recursive Bayes filter, on a persistent device-resident
   * log-odds buffer. The buffer is initialized to the initial prior log-odds
   *
   * Each call executes three kernels in sequence on the SYCL queue:
   *   1. Warp (skipped on the first frame): bilinear remap of the previous
   *      posterior into the current robot frame, using the odometry delta.
   *   2. Bayesian update: for every ray, walk cells from sensor to
   *      endpoint and atomic-add a log-odds delta computed from the
   *      graded inverse sensor model.
   *   3. Threshold: compare each cell's final log-odds against the initial
   *      prior and stamp OCCUPIED / EMPTY / UNEXPLORED into the discrete
   *      output grid.
   *
   * @param angles                   Per-ray angles in radians; size must
   *                                 equal the constructor's `scanSize`.
   * @param ranges                   Per-ray ranges in metres; same size.
   * @param positionInPrevPose       Current robot position expressed in
   *                                 the PREVIOUS frame's coordinate system,
   *                                 in metres. Zero on the first frame and
   *                                 whenever the robot is stationary
   *                                 relative to the previous frame.
   * @param orientationInPrevPose    Current robot yaw expressed in the
   *                                 PREVIOUS frame, in radians. Zero on
   *                                 the first frame.
   * @return Reference to the internal discrete occupancy grid
   *         (`OccupancyType` codes) for this frame. Storage is reused
   *         across calls; copy if you need to retain it.
   */
  Eigen::MatrixXi &
  scanToGridBaysian(const std::vector<double> &angles,
                    const std::vector<double> &ranges,
                    const Eigen::Vector2f &positionInPrevPose,
                    double orientationInPrevPose);

  /**
   * Pointcloud overload of `scanToGridBaysian`. Takes a raw
   * PointCloud2 style byte buffer.
   *
   * Each call additionally runs the pointcloud -> laserscan conversion
   * kernel before the warp / Bayesian update / threshold sequence:
   *
   * @param data                   Flattened PointCloud2 byte buffer
   *                               (int8); same layout as the non-Bayesian
   *                               `scanToGrid` pointcloud overload.
   * @param point_step             Bytes between successive points.
   * @param row_step               Bytes between rows (may exceed
   *                               `width * point_step` for padded rows).
   * @param height                 Number of rows in the cloud.
   * @param width                  Number of points per row.
   * @param x_offset,y_offset,z_offset  Byte offsets of the X/Y/Z float
   *                                    fields inside one point.
   * @param positionInPrevPose,orientationInPrevPose  Odometry delta used
   *        by the warp kernel; see the laserscan overload for the
   *        coordinate-system convention. Zero on the first frame.
   * @return Reference to the internal discrete occupancy grid
   *         (`OccupancyType` codes). Storage is reused across calls;
   *         copy if you need to retain it.
   */
  Eigen::MatrixXi &
  scanToGridBaysian(const std::vector<int8_t> &data, int point_step,
                    int row_step, int height, int width, float x_offset,
                    float y_offset, float z_offset,
                    const Eigen::Vector2f &positionInPrevPose,
                    double orientationInPrevPose);

  /**
   * Debug accessor for the current frame's posterior probability grid.
   *
   * Performs a one-shot D2H copy of the current log-odds device buffer
   * into a host-side `Eigen::MatrixXf`, then applies the sigmoid
   * `p = 1 / (1 + exp(-l))` in place on the host so the returned values
   *
   * Intended for inspecting scan-model parameter effects from Python

   * @return Reference to the host-side probability grid for the most
   *         recent frame. Storage is reused across calls; copy if you
   *         need to retain it.
   */
  const Eigen::MatrixXf &getProbabilities();

private:
  // Bayesian helper for shared Bayesian pipeline.
  void runBayesianPipeline(const Eigen::Vector2f &positionInPrevPose,
                           double orientationInPrevPose);

  // Common GPU init shared by both constructors.
  void initializeGPU(bool isPointCloud, int scanSize) {
    m_q = sycl::queue{sycl::default_selector_v,
                      sycl::property::queue::in_order{}};
    auto dev = m_q.get_device();
    LOG_INFO("Running on :", dev.get_info<sycl::info::device::name>());

    m_max_wg_size = dev.get_info<sycl::info::device::max_work_group_size>();

    m_devicePtrRanges = sycl::malloc_device<float>(scanSize, m_q);
    m_devicePtrAngles = sycl::malloc_device<double>(scanSize, m_q);
    m_devicePtrGrid = sycl::malloc_device<int>(m_gridHeight * m_gridWidth, m_q);
    m_devicePtrDistances =
        sycl::malloc_shared<float>(m_gridHeight * m_gridWidth, m_q);

    m_hostFloatRanges.resize(scanSize);

    Eigen::Vector3f destPointLocal;
    for (size_t i = 0; i < m_gridHeight; ++i) {
      for (size_t j = 0; j < m_gridWidth; ++j) {
        destPointLocal = gridToLocal({i, j});
        m_devicePtrDistances[i + j * m_gridWidth] =
            (destPointLocal - m_laserscanPosition).norm();
      }
    }

    if (isPointCloud) {
      m_q.memcpy(m_devicePtrAngles, initializedAngles.data(),
                 sizeof(double) * scanSize);
      m_q.wait();
    }
  }

  // Per-cell distance from laserscan origin. Precomputed at construction;
  // read by the ray-cast kernel
  float *m_devicePtrDistances;

  // Laserscan device buffers.
  double
      *m_devicePtrAngles; // uploaded per-call (laserscan) or once (pointcloud)
  float *m_devicePtrRanges; // fed to the ray-cast kernel

  // Output grid.
  int *m_devicePtrGrid;

  // Pointcloud-only. Grown lazily on first use in `scanToGrid(bytes,...)`
  // because the per-scan point count isn't known at ctor time.
  int8_t *m_devicePtrRawBytes = nullptr;
  size_t m_rawCapacity = 0;

  // Host-side scratch buffer for the laserscan overload's double→float
  // narrowing
  std::vector<float> m_hostFloatRanges;

  // Device-reported max work-group size. Used as the pointcloud conversion
  // kernel's block dim.
  size_t m_max_wg_size = 0;

  // Bayesian recursive-Bayes state.
  // One buffer is the persistent posterior (warp source), the other is the
  // warp output. Roles swap each frame via m_pingState. Both stay on device.
  float *m_devicePtrLogOddsA = nullptr;
  float *m_devicePtrLogOddsB = nullptr;
  bool m_useBayesian = false;
  bool m_pingState = false; // false: A holds posterior; true: B holds posterior
  int m_frameIdx = 0;       // first frame skips the warp kernel
  float m_h0 = 0.0f;        // initial prior log-odds

  // Debug / tuning: host-side mirror of the log-odds grid, sigmoid-transformed
  // on the host before return. Allocated by the Bayesian ctor.
  Eigen::MatrixXf gridProb;

  sycl::queue m_q;
};
} // namespace Mapping
} // namespace Kompass
