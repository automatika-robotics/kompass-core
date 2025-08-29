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

// Mutex for grid update
static std::mutex s_gridMutex;

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

  // Initialize the result matrix with unknownValue
  Eigen::MatrixXf transformedGrid(m_gridHeight, m_gridWidth);
  transformedGrid.fill(m_pPrior);

  for (int y = 0; y < m_gridHeight; ++y) {
    for (int x = 0; x < m_gridWidth; ++x) {
      // Compute the inverse transformation
      Eigen::Vector3f srcPoint(x, y, 1.0);
      Eigen::Vector3f dstPoint = transformationMatrix.inverse() * srcPoint;

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

  previousGridDataProb = transformedGrid;
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

void LocalMapper::updateGrid_(const float angle, const float range) {

  float x =
      m_laserscanPosition(0) + (range * cos(m_laserscanOrientation + angle));
  float y =
      m_laserscanPosition(1) + (range * sin(m_laserscanOrientation + angle));

  Eigen::Vector2i toPoint = localToGrid(Eigen::Vector2f(x, y));
  std::vector<Eigen::Vector2i> points;
  points.reserve(m_maxPointsPerLine);

  bresenhamEnhanced(m_startPoint, toPoint, points);

  for (auto &pt : points) {

    if (pt(0) >= 0 && pt(0) < m_gridHeight && pt(1) >= 0 &&
        pt(1) < m_gridWidth) {

      // grid update
      {
        std::lock_guard<std::mutex> lock(s_gridMutex);
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
}

void LocalMapper::updateGridBaysian_(const float angle, const float range) {

  float x =
      m_laserscanPosition(0) + (range * cos(m_laserscanOrientation + angle));
  float y =
      m_laserscanPosition(1) + (range * sin(m_laserscanOrientation + angle));

  Eigen::Vector2i toPoint = localToGrid(Eigen::Vector2f(x, y));
  std::vector<Eigen::Vector2i> points;
  points.reserve(m_maxPointsPerLine);

  bresenhamEnhanced(m_startPoint, toPoint, points);

  for (auto &pt : points) {

    if (pt(0) >= 0 && pt(0) < m_gridHeight && pt(1) >= 0 &&
        pt(1) < m_gridWidth) {

      // calculation for baysian update
      float distance = (pt - m_startPoint).norm();

      float newValue = updateGridCellProbability(
          distance, range, previousGridDataProb(pt(0), pt(1)));

      // grid updates
      {
        std::lock_guard<std::mutex> lock(s_gridMutex);
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
        gridDataProb(pt(0), pt(1)) = newValue;
      }
    }
  }
}

Eigen::MatrixXi &LocalMapper::scanToGrid(const std::vector<double> &angles,
                                         const std::vector<double> &ranges) {

  gridData.fill(static_cast<int>(Mapping::OccupancyType::UNEXPLORED));
  // angles and ranges are handled as floats implicitly
  if (m_maxNumThreads > 1) {
    ThreadPool pool(m_maxNumThreads);
    for (std::vector<int>::size_type i = 0; i < angles.size(); ++i) {
      pool.enqueue(&LocalMapper::updateGrid_, this, angles[i], ranges[i]);
    }
  } else {
    for (std::vector<int>::size_type i = 0; i < angles.size(); ++i) {
      updateGrid_(angles[i], ranges[i]);
    }
  }
  return gridData;
}

std::tuple<Eigen::MatrixXi &, Eigen::MatrixXf &>
LocalMapper::scanToGridBaysian(const std::vector<double> &angles,
                               const std::vector<double> &ranges) {

  gridData.fill(static_cast<int>(Mapping::OccupancyType::UNEXPLORED));
  gridDataProb.fill(m_pPrior);
  // angles and ranges are handled as floats implicitly
  if (m_maxNumThreads > 1) {
    ThreadPool pool(m_maxNumThreads);
    for (std::vector<int>::size_type i = 0; i < angles.size(); ++i) {
      pool.enqueue(&LocalMapper::updateGridBaysian_, this, angles[i],
                   ranges[i]);
    }
  } else {
    for (std::vector<int>::size_type i = 0; i < angles.size(); ++i) {
      updateGridBaysian_(angles[i], ranges[i]);
    }
  }
  return std::tie(gridData, gridDataProb);
}

Eigen::MatrixXi &LocalMapper::scanToGrid(const std::vector<int8_t> &data,
                                         int point_step, int row_step,
                                         int height, int width, float x_offset,
                                         float y_offset, float z_offset) {
  pointCloudToLaserScanFromRaw(
      data, point_step, row_step, height, width, x_offset, y_offset, z_offset,
      m_rangeMax, m_minHeight, m_maxHeight, m_scanSize, initializedRanges);
  return scanToGrid(initializedAngles, initializedRanges);
}

std::tuple<Eigen::MatrixXi &, Eigen::MatrixXf &>
LocalMapper::scanToGridBaysian(const std::vector<int8_t> &data, int point_step,
                               int row_step, int height, int width,
                               float x_offset, float y_offset, float z_offset) {

  std::vector<double> angles;
  std::vector<double> ranges;
  pointCloudToLaserScanFromRaw(
      data, point_step, row_step, height, width, x_offset, y_offset, z_offset,
      m_rangeMax, m_minHeight, m_maxHeight, m_angleStep, ranges, angles);
  return scanToGridBaysian(angles, ranges);
}
} // namespace Mapping
} // namespace Kompass
