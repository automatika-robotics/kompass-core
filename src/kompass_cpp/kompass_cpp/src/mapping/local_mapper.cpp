#include "mapping/local_mapper.h"
#include "mapping/line_drawing.h"
#include "utils/pointcloud.h"
#include "utils/threadpool.h"

#include <Eigen/SparseCore>
#include <cstdio>
#include <mutex>
#include <vector>

namespace Kompass {
namespace Mapping {

void LocalMapper::getPreviousGridInCurrentPose(
    const Eigen::Vector2f &currentPositionInPreviousPose,
    double currentOrientationInPreviousPose) {
  // The new center on the previous map
  Eigen::Vector2i currentCenter = localToGrid(currentPositionInPreviousPose);

  // Getting the angle from the difference in quaternion vector
  double currentOrientationAngle =
      -1 * currentOrientationInPreviousPose; // Negative for clockwise rotation

  // Create transformation matrix to translate and rotate the center of the grid
  Eigen::Matrix3f transformationMatrix;
  double cosTheta = cos(currentOrientationAngle);
  double sinTheta = sin(currentOrientationAngle);

  transformationMatrix << cosTheta, -sinTheta,
      0.5 * m_gridHeight - currentCenter(1) +
          (currentCenter(0) * sinTheta - currentCenter(1) * cosTheta),
      sinTheta, cosTheta,
      0.5 * m_gridWidth - currentCenter(0) -
          (currentCenter(0) * cosTheta + currentCenter(1) * sinTheta),
      0, 0, 1;

  // Reuse the ctor-allocated scratch as the result matrix (prior-filled)
  Eigen::MatrixXf &transformedGrid = m_transformScratch;
  transformedGrid.fill(m_pPrior);

  // The transformation is loop-invariant
  const Eigen::Matrix3f inverseTransform = transformationMatrix.inverse();

  for (int y = 0; y < m_gridHeight; ++y) {
    for (int x = 0; x < m_gridWidth; ++x) {
      Eigen::Vector3f srcPoint(x, y, 1.0);
      Eigen::Vector3f dstPoint = inverseTransform * srcPoint;

      // Bilinear interpolation coordinates
      double srcX = dstPoint(0);
      double srcY = dstPoint(1);

      if (srcX >= 0 && srcX < previousGridDataProb.cols() - 1 && srcY >= 0 &&
          srcY < previousGridDataProb.rows() - 1) {

        int x0 = static_cast<int>(floor(srcX));
        int y0 = static_cast<int>(floor(srcY));
        int x1 = x0 + 1;
        int y1 = y0 + 1;

        float w0 = srcX - x0;
        float w1 = 1.0f - w0;
        float h0 = srcY - y0;
        float h1 = 1.0f - h0;

        float value = h1 * (w1 * previousGridDataProb(y0, x0) +
                            w0 * previousGridDataProb(y0, x1)) +
                      h0 * (w1 * previousGridDataProb(y1, x0) +
                            w0 * previousGridDataProb(y1, x1));

        transformedGrid(y, x) = value;
      }
    }
  }

  previousGridDataProb.swap(m_transformScratch);
}

void LocalMapper::fillGridAroundPoint(Eigen::Ref<Eigen::MatrixXi> gridData,
                                      const Eigen::Vector2i &gridPoint,
                                      int gridPadding, int indicator) {

  int iStart = std::max(0, gridPoint.x() - gridPadding);
  int iEnd = std::min(m_gridHeight - 1, gridPoint.x() + gridPadding);
  int jStart = std::max(0, gridPoint.y() - gridPadding);
  int jEnd = std::min(m_gridWidth - 1, gridPoint.y() + gridPadding);

  // Fill the grid within the calculated bounds
  for (int i = iStart; i <= iEnd; ++i) {
    for (int j = jStart; j <= jEnd; ++j) {
      gridData(i, j) = indicator;
    }
  }

  // Ensure the central point is filled (though it should be within bounds
  // already)
  int iCentral = gridPoint.x();
  int jCentral = gridPoint.y();
  if (iCentral >= 0 && iCentral < m_gridHeight && jCentral >= 0 &&
      jCentral < m_gridWidth) {
    gridData(iCentral, jCentral) = indicator;
  }
}

float LocalMapper::updateGridCellProbability(float distance, float currentRange,
                                             float previousProb) {
  // get the current sensor probability of being occupied for an area in a given
  // distance from the scanner
  distance = distance * m_resolution;
  currentRange = currentRange - m_wallSize;

  float pF = (distance < currentRange) ? m_pEmpty : m_pOccupied;
  float delta = (distance < m_rangeSure) ? 0.0 : 1.0;

  float pSensor =
      pF + (delta * ((distance - m_rangeSure) / m_rangeMax) * (m_pPrior - pF));

  float pCurr =
      1 -
      (1 / (1 + ((previousProb / (1 - previousProb)) *
                 (pSensor / (1.0 - pSensor)) * ((1 - m_pPrior) / m_pPrior))));

  return pCurr;
}

void LocalMapper::updateGrid_(const float angle, const float range,
                              const Eigen::Vector2f originXY,
                              const float orientation,
                              const Eigen::Vector2i startPoint) {

  float x = originXY(0) + (range * cos(orientation + angle));
  float y = originXY(1) + (range * sin(orientation + angle));

  Eigen::Vector2i toPoint = localToGrid(Eigen::Vector2f(x, y));
  // Reused across rays on the same worker thread: one heap allocation per
  // thread
  static thread_local std::vector<Eigen::Vector2i> points;
  points.clear();
  points.reserve(m_maxPointsPerLine);

  bresenhamEnhanced(startPoint, toPoint, points);

  // One lock per ray. The writes below are cheap cell stores, so holding
  // the lock across the line beats a lock per cell
  std::lock_guard<std::mutex> lock(m_gridMutex);
  for (auto &pt : points) {

    if (pt(0) >= 0 && pt(0) < m_gridHeight && pt(1) >= 0 &&
        pt(1) < m_gridWidth) {

      if (pt(0) == toPoint(0) && pt(1) == toPoint(1)) {
        // fill grid for obstacles
        fillGridAroundPoint(gridData, pt, 0,
                            static_cast<int>(OccupancyType::OCCUPIED));
      } else {
        gridData(pt(0), pt(1)) = std::max(
            gridData(pt(0), pt(1)), static_cast<int>(OccupancyType::EMPTY));
      }
    }
  }
}

void LocalMapper::updateGridBayesian_(const float angle, const float range,
                                      const Eigen::Vector2f originXY,
                                      const float orientation,
                                      const Eigen::Vector2i startPoint) {

  float x = originXY(0) + (range * cos(orientation + angle));
  float y = originXY(1) + (range * sin(orientation + angle));

  Eigen::Vector2i toPoint = localToGrid(Eigen::Vector2f(x, y));
  // Reused across rays on the same worker thread
  static thread_local std::vector<Eigen::Vector2i> points;
  static thread_local std::vector<float> newValues;
  points.clear();
  points.reserve(m_maxPointsPerLine);

  bresenhamEnhanced(startPoint, toPoint, points);

  // Do probability math outside the lock
  newValues.resize(points.size());
  for (size_t i = 0; i < points.size(); ++i) {
    const auto &pt = points[i];
    if (pt(0) >= 0 && pt(0) < m_gridHeight && pt(1) >= 0 &&
        pt(1) < m_gridWidth) {
      float distance = (pt - startPoint).norm();
      newValues[i] = updateGridCellProbability(
          distance, range, previousGridDataProb(pt(0), pt(1)));
    }
  }

  // One lock per ray for all grid writes
  std::lock_guard<std::mutex> lock(m_gridMutex);
  for (size_t i = 0; i < points.size(); ++i) {
    const auto &pt = points[i];
    if (pt(0) >= 0 && pt(0) < m_gridHeight && pt(1) >= 0 &&
        pt(1) < m_gridWidth) {

      // non-bayesian update
      if (pt(0) == toPoint(0) && pt(1) == toPoint(1)) {
        // fill grid for obstacles
        fillGridAroundPoint(gridData, pt, 0,
                            static_cast<int>(OccupancyType::OCCUPIED));
      } else {
        gridData(pt(0), pt(1)) = std::max(
            gridData(pt(0), pt(1)), static_cast<int>(OccupancyType::EMPTY));
      }
      // bayesian update
      gridDataProb(pt(0), pt(1)) = newValues[i];
    }
  }
}

void LocalMapper::rasterizeScan_(Eigen::Ref<const Eigen::VectorXf> angles,
                                 Eigen::Ref<const Eigen::VectorXf> ranges,
                                 const Eigen::Vector2f originXY,
                                 const float orientation,
                                 const Eigen::Vector2i startPoint) {
  if (m_pool) {
    static thread_local std::vector<std::future<void>> futures;
    futures.clear();
    futures.reserve(angles.size());
    for (Eigen::Index i = 0; i < angles.size(); ++i) {
      futures.emplace_back(m_pool->enqueue(&LocalMapper::updateGrid_, this,
                                           angles[i], ranges[i], originXY,
                                           orientation, startPoint));
    }
    for (auto &f : futures) {
      f.wait();
    }
  } else {
    for (Eigen::Index i = 0; i < angles.size(); ++i) {
      updateGrid_(angles[i], ranges[i], originXY, orientation, startPoint);
    }
  }
}

void LocalMapper::rasterizeScanBayesian_(
    Eigen::Ref<const Eigen::VectorXf> angles,
    Eigen::Ref<const Eigen::VectorXf> ranges, const Eigen::Vector2f originXY,
    const float orientation, const Eigen::Vector2i startPoint) {
  if (m_pool) {
    static thread_local std::vector<std::future<void>> futures;
    futures.clear();
    futures.reserve(angles.size());
    for (Eigen::Index i = 0; i < angles.size(); ++i) {
      futures.emplace_back(m_pool->enqueue(&LocalMapper::updateGridBayesian_,
                                           this, angles[i], ranges[i], originXY,
                                           orientation, startPoint));
    }
    for (auto &f : futures) {
      f.wait();
    }
  } else {
    for (Eigen::Index i = 0; i < angles.size(); ++i) {
      updateGridBayesian_(angles[i], ranges[i], originXY, orientation,
                          startPoint);
    }
  }
}

// Laserscan overload
Eigen::MatrixXi &
LocalMapper::scanToGrid(Eigen::Ref<const Eigen::VectorXf> angles,
                        Eigen::Ref<const Eigen::VectorXf> ranges) {
  if (m_isPointCloud) {
    throw std::logic_error("scanToGrid(angles, ranges): mapper was "
                           "constructed for pointcloud input");
  }
  gridData.fill(static_cast<int>(Mapping::OccupancyType::UNEXPLORED));
  rasterizeScan_(angles, ranges, m_sensors[0].origin_xy, m_sensors[0].yaw,
                 m_sensors[0].start_point);
  return gridData;
}

// Laserscan Bayesian overload
std::tuple<Eigen::MatrixXi &, Eigen::MatrixXf &>
LocalMapper::scanToGridBayesian(Eigen::Ref<const Eigen::VectorXf> angles,
                                Eigen::Ref<const Eigen::VectorXf> ranges) {
  if (m_isPointCloud) {
    throw std::logic_error("scanToGridBayesian(angles, ranges): mapper was "
                           "constructed for pointcloud input");
  }
  gridData.fill(static_cast<int>(Mapping::OccupancyType::UNEXPLORED));
  gridDataProb.fill(m_pPrior);
  rasterizeScanBayesian_(angles, ranges, m_sensors[0].origin_xy,
                         m_sensors[0].yaw, m_sensors[0].start_point);
  return std::tie(gridData, gridDataProb);
}

// Multi Pointcloud sensor overload
Eigen::MatrixXi &LocalMapper::scanToGrid(Span<PointCloudView> clouds) {
  if (!m_isPointCloud) {
    throw std::logic_error(
        "scanToGrid(clouds): mapper was not constructed for pointcloud input");
  }
  validateClouds(clouds, m_sensors.size());

  // initialize grid once
  gridData.fill(static_cast<int>(Mapping::OccupancyType::UNEXPLORED));

  for (size_t i = 0; i < clouds.size(); ++i) {
    // Skip if cloud is empty
    if (clouds[i].empty()) {
      continue;
    }
    // Convert around this sensor's mount. Bearings come out in body
    // orientation around the sensor origin, so the ray cast runs with
    // orientation 0 from the sensor's own cell
    pointCloudToLaserScanFromRaw(clouds[i], m_sensors[i].field_type,
                                 m_sensors[i].tf_body, m_rangeMax, m_minHeight,
                                 m_maxHeight, m_scanSize, initializedRanges);
    rasterizeScan_(initializedAngles, initializedRanges, m_sensors[i].origin_xy,
                   0.0f, m_sensors[i].start_point);
  }
  return gridData;
}

// Single sensor pointcloud convenience overload
Eigen::MatrixXi &LocalMapper::scanToGrid(ByteSpan data, int point_step,
                                         int row_step, int height, int width,
                                         int x_offset, int y_offset,
                                         int z_offset) {
  const PointCloudView view{data,  point_step, row_step, height,
                            width, x_offset,   y_offset, z_offset};
  return scanToGrid(Span<PointCloudView>(&view, 1));
}

// Single sensor pointcloud Bayesian overload
std::tuple<Eigen::MatrixXi &, Eigen::MatrixXf &>
LocalMapper::scanToGridBayesian(ByteSpan data, int point_step, int row_step,
                                int height, int width, int x_offset,
                                int y_offset, int z_offset) {
  // Bayesian fusion stays single-sensor
  if (!m_isPointCloud || m_sensors.size() != 1) {
    throw std::logic_error(
        "scanToGridBayesian supports exactly one pointcloud sensor");
  }
  // Bin by scan_size through the member buffers. Bearings come out in body
  // orientation around the sensor origin, so the ray cast runs with orientation 0
  pointCloudToLaserScanFromRaw(PointCloudView{data, point_step, row_step,
                                              height, width, x_offset, y_offset,
                                              z_offset},
                               m_sensors[0].field_type, m_sensors[0].tf_body,
                               m_rangeMax, m_minHeight, m_maxHeight, m_scanSize,
                               initializedRanges);
  gridData.fill(static_cast<int>(Mapping::OccupancyType::UNEXPLORED));
  gridDataProb.fill(m_pPrior);
  rasterizeScanBayesian_(initializedAngles, initializedRanges,
                         m_sensors[0].origin_xy, 0.0f,
                         m_sensors[0].start_point);
  return std::tie(gridData, gridDataProb);
}
} // namespace Mapping
} // namespace Kompass
