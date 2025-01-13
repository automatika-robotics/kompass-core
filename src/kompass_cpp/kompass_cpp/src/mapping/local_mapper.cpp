#include "mapping/local_mapper.h"
#include "mapping/line_drawing.h"
#include "utils/threadpool.h"

#include <Eigen/SparseCore>
#include <cstdio>
#include <mutex>

namespace Kompass {
namespace Mapping {

// Mutex for grid update
static std::mutex s_gridMutex;

Eigen::MatrixXf LocalMapper::getPreviousGridInCurrentPose(
    const Eigen::Vector2f &currentPositionInPreviousPose,
    double currentOrientationInPreviousPose,
    const Eigen::MatrixXf &previousGridData, float unknownValue) {
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
  transformedGrid.fill(unknownValue);

  for (int y = 0; y < m_gridHeight; ++y) {
    for (int x = 0; x < m_gridWidth; ++x) {
      // Compute the inverse transformation
      Eigen::Vector3f srcPoint(x, y, 1.0);
      Eigen::Vector3f dstPoint = transformationMatrix.inverse() * srcPoint;

      // Bilinear interpolation coordinates
      double srcX = dstPoint(0);
      double srcY = dstPoint(1);

      if (srcX >= 0 && srcX < previousGridData.cols() - 1 && srcY >= 0 &&
          srcY < previousGridData.rows() - 1) {

        int x0 = static_cast<int>(floor(srcX));
        int y0 = static_cast<int>(floor(srcY));
        int x1 = x0 + 1;
        int y1 = y0 + 1;

        float w0 = srcX - x0;
        float w1 = 1.0f - w0;
        float h0 = srcY - y0;
        float h1 = 1.0f - h0;

        float value = h1 * (w1 * previousGridData(y0, x0) +
                            w0 * previousGridData(y0, x1)) +
                      h0 * (w1 * previousGridData(y1, x0) +
                            w0 * previousGridData(y1, x1));

        transformedGrid(y, x) = value;
      }
    }
  }

  return transformedGrid;
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
                              Eigen::Ref<Eigen::MatrixXi> gridData) {

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

void LocalMapper::updateGridBaysian_(
    const float angle, const float range, Eigen::Ref<Eigen::MatrixXi> gridData,
    Eigen::Ref<Eigen::MatrixXf> gridDataProb,
    const Eigen::Ref<const Eigen::MatrixXf> previousGridDataProb) {

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

void LocalMapper::scanToGrid(const std::vector<double> &angles,
                             const std::vector<double> &ranges,
                             Eigen::Ref<Eigen::MatrixXi> gridData) {

  // angles and ranges are handled as floats implicitly
  if (m_maxNumThreads > 1) {
    ThreadPool pool(m_maxNumThreads);
    for (std::vector<int>::size_type i = 0; i < angles.size(); ++i) {
      pool.enqueue(&LocalMapper::updateGrid_, this, angles[i], ranges[i],
                   std::ref(gridData));
    }
  } else {
    for (std::vector<int>::size_type i = 0; i < angles.size(); ++i) {
      updateGrid_(angles[i], ranges[i], gridData);
    }
  }
}

void LocalMapper::scanToGridBaysian(
    const std::vector<double> &angles, const std::vector<double> &ranges,
    Eigen::Ref<Eigen::MatrixXi> gridData,
    Eigen::Ref<Eigen::MatrixXf> gridDataProb,
    const Eigen::Ref<const Eigen::MatrixXf> previousGridDataProb) {

  // angles and ranges are handled as floats implicitly
  if (m_maxNumThreads > 1) {
    ThreadPool pool(m_maxNumThreads);
    for (std::vector<int>::size_type i = 0; i < angles.size(); ++i) {
      pool.enqueue(&LocalMapper::updateGridBaysian_, this, angles[i], ranges[i],
                   std::ref(gridData), std::ref(gridDataProb),
                   std::ref(previousGridDataProb));
    }
  } else {
    for (std::vector<int>::size_type i = 0; i < angles.size(); ++i) {
      updateGridBaysian_(angles[i], ranges[i], gridData, gridDataProb,
                         previousGridDataProb);
    }
  }
}
} // namespace Mapping
} // namespace Kompass
