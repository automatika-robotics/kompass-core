#include "mapping/local_mapper.h"
#include "mapping/line_drawing.h"
#include "utils/threadpool.h"

#include <Eigen/SparseCore>
#include <mutex>
#include <vector>

namespace Kompass {
namespace Mapping {

// Mutex for grid update
static std::mutex s_gridMutex;

Eigen::Vector3f gridToLocal(const Eigen::Vector2i &pointTargetInGrid,
                            const Eigen::Vector2i &centralPoint,
                            double resolution, double height) {
  Eigen::Vector3f poseB;
  poseB(0) = centralPoint(0) - pointTargetInGrid(0) * resolution;
  poseB(1) = centralPoint(1) - pointTargetInGrid(1) * resolution;
  poseB(2) = height;

  return poseB;
}

Eigen::Vector2i localToGrid(const Eigen::Vector2f &poseTargetInCentral,
                            const Eigen::Vector2i &centralPoint,
                            float resolution) {

  Eigen::Vector2i gridPoint;

  // Calculate grid point by rounding coordinates in the local frame to nearest
  // cell boundaries
  gridPoint(0) =
      centralPoint(0) + static_cast<int>(poseTargetInCentral(0) / resolution);
  gridPoint(1) =
      centralPoint(1) + static_cast<int>(poseTargetInCentral(1) / resolution);

  return gridPoint;
}

Eigen::MatrixXf getPreviousGridInCurrentPose(
    const Eigen::Vector2f &currentPositionInPreviousPose,
    double currentOrientationInPreviousPose,
    const Eigen::MatrixXf &previousGridData,
    const Eigen::Vector2i &centralPoint, int gridWidth, int gridHeight,
    float resolution, float unknownValue) {
  // The new center on the previous map
  Eigen::Vector2i currentCenter =
      localToGrid(currentPositionInPreviousPose, centralPoint, resolution);

  // Getting the angle from the difference in quaternion vector
  double currentOrientationAngle =
      -1 * currentOrientationInPreviousPose; // Negative for clockwise rotation

  // Create transformation matrix to translate and rotate the center of the grid
  Eigen::Matrix3f transformationMatrix;
  double cosTheta = cos(currentOrientationAngle);
  double sinTheta = sin(currentOrientationAngle);

  transformationMatrix << cosTheta, -sinTheta,
      0.5 * gridHeight - currentCenter(1) +
          (currentCenter(0) * sinTheta - currentCenter(1) * cosTheta),
      sinTheta, cosTheta,
      0.5 * gridWidth - currentCenter(0) -
          (currentCenter(0) * cosTheta + currentCenter(1) * sinTheta),
      0, 0, 1;

  // Initialize the result matrix with unknownValue
  Eigen::MatrixXf transformedGrid(gridHeight, gridWidth);
  transformedGrid.fill(unknownValue);

  for (int y = 0; y < gridHeight; ++y) {
    for (int x = 0; x < gridWidth; ++x) {
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

void fillGridAroundPoint(Eigen::Ref<Eigen::MatrixXi> gridData,
                         const Eigen::Vector2i &gridPoint, int gridPadding,
                         int indicator) {
  // Calculate the valid range for i and j
  int rows = gridData.rows();
  int cols = gridData.cols();

  int iStart = std::max(0, gridPoint.x() - gridPadding);
  int iEnd = std::min(rows - 1, gridPoint.x() + gridPadding);
  int jStart = std::max(0, gridPoint.y() - gridPadding);
  int jEnd = std::min(cols - 1, gridPoint.y() + gridPadding);

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
  if (iCentral >= 0 && iCentral < rows && jCentral >= 0 && jCentral < cols) {
    gridData(iCentral, jCentral) = indicator;
  }
}

float updateGridCellProbability(float distance, float currentRange,
                                float previousProb, float resolution,
                                float pPrior, float pEmpty, float pOccupied,
                                float rangeSure, float rangeMax,
                                float wallSize) {
  // get the current sensor probability of being occupied for an area in a given
  // distance from the scanner
  distance = distance * resolution;
  currentRange = currentRange - wallSize;

  float pF = (distance < currentRange) ? pEmpty : pOccupied;
  float delta = (distance < rangeSure) ? 0.0 : 1.0;

  float pSensor =
      pF + (delta * ((distance - rangeSure) / rangeMax) * (pPrior - pF));

  float pCurr =
      1 - (1 / (1 + ((previousProb / (1 - previousProb)) *
                     (pSensor / (1.0 - pSensor)) * ((1 - pPrior) / pPrior))));

  return pCurr;
}

void updateGrid(const float angle, const float range,
                const Eigen::Vector2i &startPoint,
                Eigen::Ref<Eigen::MatrixXi> gridData,
                const Eigen::Vector2i &centralPoint, float resolution,
                const Eigen::Vector3f &laserscanPosition,
                float laserscanOrientation, int maxPointsPerLine) {

  float x = laserscanPosition(0) + (range * cos(laserscanOrientation + angle));
  float y = laserscanPosition(1) + (range * sin(laserscanOrientation + angle));

  Eigen::Vector2i toPoint =
      localToGrid(Eigen::Vector2f(x, y), centralPoint, resolution);
  std::vector<Eigen::Vector2i> points;
  points.reserve(maxPointsPerLine);

  bresenhamEnhanced(startPoint, toPoint, points);

  bool rayStopped = true;
  Eigen::Vector2i lastGridPoint = points[0];

  // Precompute boundary checks
  int rows = gridData.rows();
  int cols = gridData.cols();

  for (auto &pt : points) {

    if (pt(0) >= 0 && pt(0) < rows && pt(1) >= 0 && pt(1) < cols) {

      // grid update
      {
        std::lock_guard<std::mutex> lock(s_gridMutex);
        gridData(pt(0), pt(1)) = std::max(
            gridData(pt(0), pt(1)), static_cast<int>(OccupancyType::EMPTY));
      }

      // update last point drawn
      lastGridPoint(0) = pt(0);
      lastGridPoint(1) = pt(1);
    } else {
      rayStopped = false;
    }
  }
  if (rayStopped) {
    std::lock_guard<std::mutex> lock(s_gridMutex);
    /* Force non-bayesian map to set ending_point to occupied. So that
       when visualizing the map the detected obstacles can be visualized
       instead of having a map that visualize empty and unknown zones only.*/
    fillGridAroundPoint(gridData, lastGridPoint, 0,
                        static_cast<int>(OccupancyType::OCCUPIED));
  }
}

void updateGridBaysian(
    const float angle, const float range, const Eigen::Vector2i &startPoint,
    Eigen::Ref<Eigen::MatrixXi> gridData,
    Eigen::Ref<Eigen::MatrixXf> gridDataProb,
    const Eigen::Vector2i &centralPoint, float resolution,
    const Eigen::Vector3f &laserscanPosition, float laserscanOrientation,
    const Eigen::Ref<const Eigen::MatrixXf> previousGridDataProb, float pPrior,
    float pEmpty, float pOccupied, float rangeSure, float rangeMax,
    float wallSize, int maxPointsPerLine) {

  float x = laserscanPosition(0) + (range * cos(laserscanOrientation + angle));
  float y = laserscanPosition(1) + (range * sin(laserscanOrientation + angle));

  Eigen::Vector2i toPoint =
      localToGrid(Eigen::Vector2f(x, y), centralPoint, resolution);
  std::vector<Eigen::Vector2i> points;
  points.reserve(maxPointsPerLine);

  bresenhamEnhanced(startPoint, toPoint, points);

  bool rayStopped = true;
  Eigen::Vector2i lastGridPoint = points[0];

  // Precompute boundary checks
  int rows = gridData.rows();
  int cols = gridData.cols();

  for (auto &pt : points) {

    if (pt(0) >= 0 && pt(0) < rows && pt(1) >= 0 && pt(1) < cols) {

      // calculation for baysian update
      float distance = (pt - startPoint).norm();

      float newValue = updateGridCellProbability(
          distance, range, previousGridDataProb(pt(0), pt(1)), resolution,
          pPrior, pEmpty, pOccupied, rangeSure, rangeMax, wallSize);

      // grid updates
      {
        std::lock_guard<std::mutex> lock(s_gridMutex);
        // non-bayesian update
        gridData(pt(0), pt(1)) = std::max(
            gridData(pt(0), pt(1)), static_cast<int>(OccupancyType::EMPTY));
        // bayesian update
        gridDataProb(pt(0), pt(1)) = newValue;
      }

      // update last point drawn
      lastGridPoint(0) = pt(0);
      lastGridPoint(1) = pt(1);
    } else {
      rayStopped = false;
    }
  }
  if (rayStopped) {
    std::lock_guard<std::mutex> lock(s_gridMutex);
    /* Force non-bayesian map to set ending_point to occupied. So that
       when visualizing the map the detected obstacles can be visualized
       instead of having a map that visualize empty and unknown zones only.*/
    fillGridAroundPoint(gridData, lastGridPoint, 0,
                        static_cast<int>(OccupancyType::OCCUPIED));
  }
}

void scanToGrid(const std::vector<double> &angles,
                const std::vector<double> &ranges,
                Eigen::Ref<Eigen::MatrixXi> gridData,
                const Eigen::Vector2i &centralPoint, float resolution,
                const Eigen::Vector3f &laserscanPosition,
                float laserscanOrientation, int maxPointsPerLine,
                int maxNumThreads) {

  // angles and ranges are handled as floats implicitly
  Eigen::Vector2i startPoint =
      localToGrid(Eigen::Vector2f(laserscanPosition(0), laserscanPosition(1)),
                  centralPoint, resolution);

  if (maxNumThreads > 1) {
    ThreadPool pool(maxNumThreads);
    for (std::vector<int>::size_type i = 0; i < angles.size(); ++i) {
      pool.enqueue(&updateGrid, angles[i], ranges[i], startPoint,
                   std::ref(gridData), centralPoint, resolution,
                   laserscanPosition, laserscanOrientation, maxPointsPerLine);
    }
  } else {
    for (std::vector<int>::size_type i = 0; i < angles.size(); ++i) {
      updateGrid(angles[i], ranges[i], startPoint, gridData, centralPoint,
                 resolution, laserscanPosition, laserscanOrientation,
                 maxPointsPerLine);
    }
  }
}

void scanToGridBaysian(
    const std::vector<double> &angles, const std::vector<double> &ranges,
    Eigen::Ref<Eigen::MatrixXi> gridData,
    Eigen::Ref<Eigen::MatrixXf> gridDataProb,
    const Eigen::Vector2i &centralPoint, float resolution,
    const Eigen::Vector3f &laserscanPosition, float laserscanOrientation,
    const Eigen::Ref<const Eigen::MatrixXf> previousGridDataProb, float pPrior,
    float pEmpty, float pOccupied, float rangeSure, float rangeMax,
    float wallSize, int maxPointsPerLine, int maxNumThreads) {

  // angles and ranges are handled as floats implicitly
  Eigen::Vector2i startPoint =
      localToGrid(Eigen::Vector2f(laserscanPosition(0), laserscanPosition(1)),
                  centralPoint, resolution);

  if (maxNumThreads > 1) {
    ThreadPool pool(maxNumThreads);
    for (std::vector<int>::size_type i = 0; i < angles.size(); ++i) {
      pool.enqueue(&updateGridBaysian, angles[i], ranges[i], startPoint,
                   std::ref(gridData), std::ref(gridDataProb), centralPoint,
                   resolution, laserscanPosition, laserscanOrientation,
                   std::ref(previousGridDataProb), pPrior, pEmpty, pOccupied,
                   rangeSure, rangeMax, wallSize, maxPointsPerLine);
    }
  } else {
    for (std::vector<int>::size_type i = 0; i < angles.size(); ++i) {
      updateGridBaysian(angles[i], ranges[i], startPoint, gridData,
                        gridDataProb, centralPoint, resolution,
                        laserscanPosition, laserscanOrientation,
                        previousGridDataProb, pPrior, pEmpty, pOccupied,
                        rangeSure, rangeMax, wallSize, maxPointsPerLine);
    }
  }
}
} // namespace Mapping
} // namespace Kompass
