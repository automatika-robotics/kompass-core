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

Eigen::Vector2i localToGrid(const Eigen::Vector2f &poseTargetInCentral,
                            const Eigen::Vector2i &centralPoint,
                            float resolution) {

  // Initialize result vector with zeros
  Eigen::Vector2i gridPoint;

  // Calculate grid point by rounding coordinates in the local frame to nearest
  // cell boundaries
  gridPoint(0) =
      centralPoint(0) + static_cast<int>(poseTargetInCentral(0) / resolution);
  gridPoint(1) =
      centralPoint(1) + static_cast<int>(poseTargetInCentral(1) / resolution);

  return gridPoint;
}

void fillGridAroundPoint(Eigen::Ref<Eigen::MatrixXi> gridData,
                         const Eigen::Vector2i &gridPoint, int gridPadding,
                         int indicator) {
  int iStart = gridPoint.x() - gridPadding;
  int iEnd = gridPoint.x() + gridPadding;
  int jStart = gridPoint.y() - gridPadding;
  int jEnd = gridPoint.y() + gridPadding;

  for (int i = iStart; i <= iEnd; ++i) {
    if (i >= 0 && i < gridData.rows()) {
      for (int j = jStart; j <= jEnd; ++j) {
        if (j >= 0 && j < gridData.cols()) {
          gridData(i, j) = indicator;
        }
      }
    }
  }

  // Ensure the central point is filled
  int i = gridPoint(0);
  int j = gridPoint(1);
  if (i >= 0 && i < gridData.rows() && j >= 0 && j < gridData.cols()) {
    gridData(i, j) = indicator;
  }
}

double updateGridCellProbability(double distance, double currentRange,
                                 double oddLogPPrev, double resolution,
                                 double pPrior, double pEmpty, double pOccupied,
                                 double rangeSure, double rangeMax,
                                 double wallSize, double oddLogPPrior) {
  // get the current sensor probability of being occupied for an area in a given
  // distance from the scanner
  distance = distance * resolution;
  currentRange = currentRange - wallSize;

  double pF = (distance < currentRange) ? pEmpty : pOccupied;
  double delta = (distance < rangeSure) ? 0.0 : 1.0;

  double pSensor =
      pF + (delta * ((distance - rangeSure) / rangeMax) * (pPrior - pF));

  // get the current bayesian updated probability given its previous probability
  // and sensor probability
  double pCurr = oddLogPPrev + log(pSensor / (1.0 - pSensor)) - oddLogPPrior;

  return pCurr;
}

void updateGrid(const float angle, const float range,
                const Eigen::Vector2i &startPoint,
                Eigen::Ref<Eigen::MatrixXi> gridData,
                Eigen::Ref<Eigen::MatrixXi> gridDataProb,
                const Eigen::Vector2i &centralPoint, float resolution,
                const Eigen::Vector3f &laserscanPosition,
                float laserscanOrientation,
                const Eigen::Ref<const Eigen::MatrixXi> previousGridDataProb,
                float pPrior, float pEmpty, float pOccupied, float rangeSure,
                float rangeMax, float wallSize, float oddLogPPrior) {

  float x = laserscanPosition(0) + (range * cos(laserscanOrientation + angle));
  float y = laserscanPosition(1) + (range * sin(laserscanOrientation + angle));

  Eigen::Vector2i toPoint =
      localToGrid(Eigen::Vector2f(x, y), centralPoint, resolution);
  std::vector<Eigen::Vector2i> points;

  bresenhamEnhanced(startPoint, toPoint, points);

  bool rayStopped = true;
  Eigen::Vector2i lastGridPoint = points[0];

  // Precompute boundary checks
  int rows = gridData.rows();
  int cols = gridData.cols();
  int prevRows = previousGridDataProb.rows();
  int prevCols = previousGridDataProb.cols();

  // NOTE: when previous grid is transformed to align with the current
  // grid it's not perfectly aligned at the nearest grid cell.
  // Experimentally (empirically), a shift of 1 grid cell reverses the
  // mis-alignment effect. -> check test_local_grid_mapper.py for more
  // info
  int constexpr SHIFT = 1;

  for (auto &pt : points) {

    if (pt(0) >= 0 && pt(0) < rows && pt(1) >= 0 && pt(1) < cols) {

      // calculations for baysian update
      int x_prev = pt(0) + SHIFT;
      int y_prev = pt(1) + SHIFT;

      int previousValue =
          (x_prev >= 0 && x_prev < prevRows && y_prev >= 0 && y_prev < prevCols)
              ? previousGridDataProb(x_prev, y_prev)
              : oddLogPPrior;

      double distance = (pt - startPoint).norm();

      int newValue = updateGridCellProbability(
          distance, range, previousValue, resolution, pPrior, pEmpty, pOccupied,
          rangeSure, rangeMax, wallSize, oddLogPPrior);

      // grid updates
      {
        std::lock_guard<std::mutex> lock(s_gridMutex);
        // non-baysian update
        gridData(pt(0), pt(1)) = std::max(gridData(pt(0), pt(1)), 0);
        // baysian update
        gridDataProb(pt(0), pt(1)) =
            std::max(gridDataProb(pt(0), pt(1)), newValue);
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
    fillGridAroundPoint(gridData, lastGridPoint, 0, 100);
  }
}

void scanToGrid(const std::vector<double> &angles,
                const std::vector<double> &ranges,
                Eigen::Ref<Eigen::MatrixXi> gridData,
                Eigen::Ref<Eigen::MatrixXi> gridDataProb,
                const Eigen::Vector2i &centralPoint, float resolution,
                const Eigen::Vector3f &laserscanPosition,
                float laserscanOrientation,
                const Eigen::Ref<const Eigen::MatrixXi> previousGridDataProb,
                float pPrior, float pEmpty, float pOccupied, float rangeSure,
                float rangeMax, float wallSize, float oddLogPPrior,
                int maxNumThreads) {

  Eigen::Vector2i startPoint =
      localToGrid(Eigen::Vector2f(laserscanPosition(0), laserscanPosition(1)),
                  centralPoint, resolution);

  if (maxNumThreads > 1) {
    ThreadPool pool(maxNumThreads);
    for (int i = 0; i < angles.size(); ++i) {
      pool.enqueue(&updateGrid, angles[i], ranges[i], startPoint,
                   std::ref(gridData), std::ref(gridDataProb), centralPoint,
                   resolution, laserscanPosition, laserscanOrientation,
                   previousGridDataProb, pPrior, pEmpty, pOccupied, rangeSure,
                   rangeMax, wallSize, oddLogPPrior);
    }
  } else {
    for (int i = 0; i < angles.size(); ++i) {
      updateGrid(angles[i], ranges[i], startPoint, gridData, gridDataProb,
                 centralPoint, resolution, laserscanPosition,
                 laserscanOrientation, previousGridDataProb, pPrior, pEmpty,
                 pOccupied, rangeSure, rangeMax, wallSize, oddLogPPrior);
    }
  }
}
} // namespace Mapping
} // namespace Kompass
