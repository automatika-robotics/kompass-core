#pragma once

#include "datatypes/sensors.h"
#include "datatypes/span.h"
#include "utils/threadpool.h"
#include <Eigen/Dense>
#include <cmath>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <vector>

namespace Kompass {
namespace Mapping {

// Occupancy types for grid
enum class OccupancyType { UNEXPLORED = -1, EMPTY = 0, OCCUPIED = 100 };

class LocalMapper {
public:
  /**
   * Constructor with basic parameters.
   *
   * @param sensors One SensorConfig per sensor. Laserscan input
   * requires exactly one sensor and consumes its mount as planar
   * (position x/y + yaw extracted from the quaternion). Pointcloud input
   * accepts N sensors fused into one grid, each with its full 3D mount applied.
   *
   * NOTE for pointcloud input `maxHeight`/`minHeight` are a BODY-frame
   * height band shared by all sensors (typically 0 .. robot_height).
   */
  LocalMapper(const int gridHeight, const int gridWidth, const float resolution,
              const std::vector<SensorConfig> &sensors, const bool isPointCloud,
              const int scanSize, const float maxHeight, const float minHeight,
              const float rangeMax, const int maxPointsPerLine,
              const int maxNumThreads = 1)
      : m_gridHeight(gridHeight), m_gridWidth(gridWidth),
        m_resolution(resolution), m_pPrior(0.5f), m_pEmpty(0.4f),
        m_pOccupied(0.6f), m_rangeSure(1.0f), m_rangeMax(rangeMax),
        m_wallSize(0.2f), m_maxHeight(maxHeight), m_minHeight(minHeight),
        m_maxPointsPerLine(maxPointsPerLine),
        m_centralPoint(std::round(gridHeight / 2) - 1,
                       std::round(gridWidth / 2) - 1),
        m_scanSize(scanSize), m_isPointCloud(isPointCloud),
        m_maxNumThreads(maxNumThreads) {
    initMapper_(sensors);
  }

  // Constructor with additional bayesian parameters
  LocalMapper(const int gridHeight, const int gridWidth, const float resolution,
              const std::vector<SensorConfig> &sensors, const bool isPointCloud,
              const int scanSize, const float pPrior, const float pOccupied,
              const float pEmpty, const float rangeSure, const float rangeMax,
              const float wallSize, const float maxHeight,
              const float minHeight, const int maxPointsPerLine,
              const int maxNumThreads = 1)
      : m_gridHeight(gridHeight), m_gridWidth(gridWidth),
        m_resolution(resolution), m_pPrior(pPrior), m_pEmpty(pEmpty),
        m_pOccupied(pOccupied), m_rangeSure(rangeSure), m_rangeMax(rangeMax),
        m_wallSize(wallSize), m_maxHeight(maxHeight), m_minHeight(minHeight),
        m_maxPointsPerLine(maxPointsPerLine),
        m_centralPoint(std::round(gridHeight / 2) - 1,
                       std::round(gridWidth / 2) - 1),
        m_scanSize(scanSize), m_isPointCloud(isPointCloud),
        m_maxNumThreads(maxNumThreads) {
    initMapper_(sensors);
  }

  // Default destructor
  virtual ~LocalMapper() = default;

  /**
   * @brief Transform the stored previous probability grid
   * (`previousGridDataProb`) in place, re-centering it on the current
   * position given its previous position. The next Bayesian update reads
   * the shifted member directly; unknown cells are filled with the prior.
   *
   * @param current_position_in_previous_pose Current egocentric position for
   * the transformation.
   * @param current_yaw_orientation_in_previous_pose Current egocentric
   * orientation for the transformation.
   */
  void getPreviousGridInCurrentPose(
      const Eigen::Vector2f &currentPositionInPreviousPose,
      double currentOrientationInPreviousPose);

  /**
   * @brief Updates a grid cell occupancy probability using the LaserScanModel
   *
   * @param distance Hit point distance from the sensor (m)
   * @param currentRange Scan ray hit range (m)
   * @param previousProb Previous probability assigned to grid cell
   */
  float updateGridCellProbability(float distance, float currentRange,
                                  float previousProb);

  /**
   * Processes Laserscan data (angles and ranges) to project on a 2D grid
   * using Bresenham line drawing for each Laserscan beam
   *
   * @param angles        LaserScan angles in radians
   * @param ranges         LaserScan ranges in meters
   * @returns gridData      Current grid data
   */
  Eigen::MatrixXi &scanToGrid(Eigen::Ref<const Eigen::VectorXf> angles,
                              Eigen::Ref<const Eigen::VectorXf> ranges);

  /**
   * Processes Laserscan data (angles and ranges) to project on a 2D grid
   * using Bresenham line drawing for each Laserscan beam and bayesian map
   * updates
   *
   * @param angles        LaserScan angles in radians
   * @param ranges         LaserScan ranges in meters
   * @returns gridDataProb Current probabilistic grid data
   */
  std::tuple<Eigen::MatrixXi &, Eigen::MatrixXf &>
  scanToGridBayesian(Eigen::Ref<const Eigen::VectorXf> angles,
                    Eigen::Ref<const Eigen::VectorXf> ranges);

  /**
   * Projects 3D point cloud data onto a 2D grid using Bresenham line drawing.
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
   * Projects 3D point cloud data onto a 2D grid using Bresenham line drawing,
   * with Bayesian updates to build a probabilistic occupancy grid.
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
   * @return            A tuple containing:
   *                      - Discrete occupancy grid (Eigen::MatrixXi&)
   *                      - Probabilistic occupancy grid (Eigen::MatrixXf&)
   */
  std::tuple<Eigen::MatrixXi &, Eigen::MatrixXf &>
  scanToGridBayesian(ByteSpan data, int point_step, int row_step, int height,
                    int width, int x_offset, int y_offset, int z_offset);

  /**
   * Fuses N point clouds into one occupancy grid. clouds[i] pairs with the
   * i-th configured sensor (positional). The grid is reset once, then each
   * sensor's cloud is converted around its own mount pose and ray-cast from
   * its own origin; cell writes are max-monotone
   * (OCCUPIED > EMPTY > UNEXPLORED) so the per-sensor passes fuse
   * order-independently. Empty views are skipped ("no data from this sensor
   * this tick"); an all-empty batch returns an all-UNEXPLORED grid.
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

protected:
  // Transforms a point from grid coordinate (i,j) to the local coordinates
  // frame of the grid (around the central cell) (x,y,z).
  // NOTE: (INVARIANT) this must stay the exact inverse of localToGrid below
  // (cell = central + p/res  <=>  p = (cell - central)*res).
  Eigen::Vector3f gridToLocal(const Eigen::Vector2i &pointTargetInGrid,
                              float height = 0.0) {
    Eigen::Vector3f poseB;
    poseB(0) = (pointTargetInGrid(0) - m_centralPoint(0)) * m_resolution;
    poseB(1) = (pointTargetInGrid(1) - m_centralPoint(1)) * m_resolution;
    poseB(2) = height;

    return poseB;
  }

  // Function to convert a point from local coordinates frame of the grid to
  // grid indices
  Eigen::Vector2i localToGrid(const Eigen::Vector2f &poseTargetInCentral) {

    Eigen::Vector2i gridPoint;

    // Calculate grid point by rounding coordinates in the local frame to
    // nearest cell boundaries
    gridPoint(0) = m_centralPoint(0) +
                   static_cast<int>(poseTargetInCentral(0) / m_resolution);
    gridPoint(1) = m_centralPoint(1) +
                   static_cast<int>(poseTargetInCentral(1) / m_resolution);

    return gridPoint;
  }

  // Per-sensor runtime state derived from a SensorConfig. The pointcloud
  // path uses tf_body/origin_xy/start_point (rotation folded into the
  // conversion, ray-cast orientation 0); the laserscan path uses
  // origin_xy/yaw/start_point (planar consumption of the mount)
  struct SensorRuntime {
    Eigen::Isometry3f tf_body;   // sensor -> body mount
    Eigen::Vector2f origin_xy;   // planar mount position (ray-cast origin)
    float yaw;                   // planar heading extracted from the mount
    Eigen::Vector2i start_point; // ray-cast origin cell
    PointFieldType field_type;   // point field encoding (pointcloud decode)
  };

  SensorRuntime makeSensorRuntime(const SensorConfig &config) {
    SensorRuntime runtime;
    runtime.tf_body = config.tfBody();
    runtime.origin_xy = config.position.head<2>();
    runtime.field_type = config.cloud_field_type;
    const Eigen::Matrix3f rot = runtime.tf_body.rotation();
    runtime.yaw = std::atan2(rot(1, 0), rot(0, 0));
    runtime.start_point = localToGrid(runtime.origin_xy);
    return runtime;
  }

  // Shared constructor body
  void initMapper_(const std::vector<SensorConfig> &sensors) {
    gridData = Eigen::MatrixXi(m_gridHeight, m_gridWidth);
    // Only for Bayesian calls
    gridDataProb = Eigen::MatrixXf(m_gridHeight, m_gridWidth);
    previousGridDataProb = Eigen::MatrixXf(m_gridHeight, m_gridWidth);
    m_transformScratch = Eigen::MatrixXf(m_gridHeight, m_gridWidth);
    previousGridDataProb.fill(m_pPrior);  // previous prob buffer
    // initialize thread pool
    if (m_maxNumThreads > 1) {
      m_pool = std::make_unique<ThreadPool>(m_maxNumThreads);
    }

    if (sensors.empty()) {
      throw std::invalid_argument(
          "LocalMapper requires at least one sensor config");
    }
    if (!m_isPointCloud && sensors.size() != 1) {
      throw std::invalid_argument(
          "LocalMapper laserscan input supports exactly one sensor");
    }
    m_sensors.reserve(sensors.size());
    for (const auto &sensor : sensors) {
      m_sensors.push_back(makeSensorRuntime(sensor));
    }

    // initialize ranges and angles if working with pointcloud
    if (m_isPointCloud) {
      // NOTE: The pointcloud → laserscan step (CPU
      // `pointCloudToLaserScanFromRaw` and its GPU counterpart) buckets by a
      // bin width of `2π / scan_size`; the ray-cast step consumes
      // `initializedAngles[i]` and must see the exact same step or the two
      // halves drift by a fraction of a bin per ray (silent rotation)
      const double derived_step =
          (2.0 * M_PI) / static_cast<double>(m_scanSize);
      initializedAngles.resize(m_scanSize);
      initializedRanges.resize(m_scanSize);
      for (int i = 0; i < m_scanSize; ++i) {
        initializedAngles[i] = static_cast<float>(i * derived_step);
        initializedRanges[i] = m_rangeMax;
      }
    }
  }

  // Casts one ray from the given origin and writes its EMPTY/OCCUPIED cells.
  void updateGrid_(const float angle, const float range,
                   const Eigen::Vector2f originXY, const float orientation,
                   const Eigen::Vector2i startPoint);

  // Bayesian variant of updateGrid_
  void updateGridBayesian_(const float angle, const float range,
                          const Eigen::Vector2f originXY,
                          const float orientation,
                          const Eigen::Vector2i startPoint);

  // Rasterizes one whole scan from one origin into the grid. Does NOT reset the
  // grid, so several rasterize passes (one per sensor) accumulate into the same
  // grid
  void rasterizeScan_(Eigen::Ref<const Eigen::VectorXf> angles,
                      Eigen::Ref<const Eigen::VectorXf> ranges,
                      const Eigen::Vector2f originXY, const float orientation,
                      const Eigen::Vector2i startPoint);

  // Bayesian variant of rasterizeScan_
  void rasterizeScanBayesian_(Eigen::Ref<const Eigen::VectorXf> angles,
                             Eigen::Ref<const Eigen::VectorXf> ranges,
                             const Eigen::Vector2f originXY,
                             const float orientation,
                             const Eigen::Vector2i startPoint);

  /**
   * @brief Fill an area around a point on the grid with given padding.
   *
   * @param gridData Grid to be filled (2D Eigen matrix).
   * @param gridPoint Grid point indices (i,j) as a Vector2i.
   * @param gridPadding Padding to be filled (number of cells).
   * @param indicator Value to be assigned to filled cells.
   */
  void fillGridAroundPoint(Eigen::Ref<Eigen::MatrixXi> gridData,
                           const Eigen::Vector2i &gridPoint, int gridPadding,
                           int indicator);

protected:
  const int m_gridHeight;
  const int m_gridWidth;
  const float m_resolution;
  const float m_pPrior;
  const float m_pEmpty;
  const float m_pOccupied;
  const float m_rangeSure;
  const float m_rangeMax;
  const float m_wallSize;
  const float m_maxHeight;
  const float m_minHeight;
  const int m_maxPointsPerLine;
  const Eigen::Vector2i m_centralPoint;
  const int m_scanSize;
  const bool m_isPointCloud;
  Eigen::MatrixXi gridData;
  Eigen::MatrixXf gridDataProb;
  Eigen::MatrixXf previousGridDataProb;
  Eigen::VectorXf initializedRanges; // only initialized for pointcloud data
  Eigen::VectorXf initializedAngles; // only initialized for pointcloud data
  // Per-sensor runtimes, one per SensorConfig (laserscan input has exactly
  // one; pointcloud input may have several)
  std::vector<SensorRuntime> m_sensors;

  // Serializes grid writes across worker threads (one lock per ray)
  std::mutex m_gridMutex;

private:
  const int m_maxNumThreads;
  // Reusable worker pool, created when maxNumThreads > 1
  std::unique_ptr<ThreadPool> m_pool;
  // Scratch for the in-place grid transformation
  Eigen::MatrixXf m_transformScratch;
};

} // namespace Mapping
} // namespace Kompass
